import argparse
import logging
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
from utils import MODEL_CLASSES, get_intent_labels, get_slot_labels, init_logger, load_tokenizer


logger = logging.getLogger(__name__)


def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"


def get_args(model_dir):
    return torch.load(os.path.join(model_dir, "training_args.bin"))


def load_model(pred_config, args, device):
    # Check whether model exists
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(
            pred_config.model_dir, args=args, intent_label_lst=get_intent_labels(args), slot_label_lst=get_slot_labels(args)
        )
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except Exception:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(
    lines,
    pred_config,
    args,
    tokenizer,
    pad_token_label_id,
    cls_token_segment_id=0,
    pad_token_segment_id=0,
    sequence_a_segment_id=0,
    mask_padding_with_zero=True,
):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_label_mask.extend([pad_token_label_id + 1] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[: (args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    return dataset

def load_model_from_directory(model_dir, args, device):
    # Check whether the model directory exists
    if not os.path.exists(model_dir):
        raise Exception("Model directory doesn't exist!")

    try:
        model = MODEL_CLASSES[args.model_type][1].from_pretrained(
            model_dir, args=args, intent_label_lst=get_intent_labels(args), slot_label_lst=get_slot_labels(args)
        )
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded from: {} *****".format(model_dir))
    except Exception:
        raise Exception("Some model files might be missing...")

    return model


def predict(pred_config):
    # load model and args
    
    model_dirs = []
    with open(pred_config.model_list, 'r') as model_list_file:
        for line in model_list_file:
            model_dirs.append(line.strip())
    if not model_dirs:
        raise ValueError("No model directories found in the model_list file.")
    # model = load_model(pred_config, args, device)
    ensemble_models = []
    ensemble_args = []

    for model_dir in model_dirs:
        args = get_args(model_dir)
        device = get_device(pred_config)
        model = load_model_from_directory(model_dir, args, device)
        ensemble_models.append(model)
        ensemble_args.append(args)

    logger.info(ensemble_args[0])
    intent_label_lst = get_intent_labels(ensemble_args[0])
    slot_label_lst = get_slot_labels(ensemble_args[0])

    # Convert input file to TensorDataset
    pad_token_label_id = ensemble_args[0].ignore_index
    tokenizer = load_tokenizer(ensemble_args[0])
    lines = read_input_file(pred_config)
    dataset = convert_input_file_to_tensor_dataset(lines, pred_config, ensemble_args[0], tokenizer, pad_token_label_id)

    # Predict
    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

    all_slot_label_mask = None
    ensemble_intent_probs = []
    ensemble_slot_probs = []
    ensemble_intent_logits = []
    ensemble_slot_logits = []
    # for args, model in zip(ensemble_args, ensemble_models):
    #     # Predict using the current model
    #     intent_preds = None
    #     slot_preds = None
    print(pred_config.batch_size)
    for batch in tqdm(data_loader, desc="Predicting"):
        batch = tuple(t.to(device) for t in batch)

        intent_logits_per_batch = []
        slot_logits_per_batch = []

        for model in ensemble_models:
            with torch.no_grad():
                inputs = {
                    "input_ids": batch[0],
                    "attention_mask": batch[1],
                    "intent_label_ids": None,
                    "slot_labels_ids": None,
                }
                if args.model_type != "distilbert":
                    inputs["token_type_ids"] = batch[2]
                outputs = model(**inputs)
                _, (intent_logits, slot_logits) = outputs[:2]
                # print("slot_logit:", slot_logits)
            
                # print("slot logit:", slot_logits)
                # Intent Prediction
                # if intent_preds is None:
                #     intent_preds = intent_logits.detach().cpu().numpy()
                # else:
                #     intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                # if intent_logits_per_batch is None:
                # intent_logits_per_batch = torch.softmax(intent_logits, dim=-1).cpu().numpy()
                # else:
                    # intent_logits_per_batch = np.append(intent_logits_per_batch, torch.softmax(intent_logits, dim=-1).cpu().numpy(),axis = 0)
                # if slot_logits_per_batch is None:
                # slot_logits_per_batch = torch.softmax(slot_logits, dim=-1).cpu().numpy()

                
                # else:
                    # slot_logits_per_batch = np.append(slot_logits_per_batch, torch.softmax(slot_logits, dim=-1).cpu().numpy(), axis = 0)

                intent_logits_per_batch.append(torch.softmax(intent_logits, dim=-1).detach().cpu().numpy())
                slot_logits_per_batch.append(torch.softmax(slot_logits, dim=-1).detach().cpu().numpy())

                # Slot prediction
                # if slot_preds is None:
                #     if args.use_crf:
                #         # decode() in `torchcrf` returns list with best index directly
                #         slot_preds = np.array(model.crf.decode(slot_logits))
                #         print("use crf none")
                #     else:
                #         slot_preds = slot_logits.detach().cpu().numpy()
                #         print("not crf none")
                #     all_slot_label_mask = batch[3].detach().cpu().numpy()
                # else:
                #     if args.use_crf:
                #         print("use crf")
                #         slot_preds = np.append(slot_preds, np.array(model.crf.decode(slot_logits)), axis=0)
                #     else:
                #         print("not crf")
                #         slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                #     all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)
        
        ensemble_intent_logits.append(intent_logits_per_batch)  # Extend the list
        ensemble_slot_logits.append(slot_logits_per_batch)

        if all_slot_label_mask is None:
            all_slot_label_mask = batch[3].detach().cpu().numpy()
        else:
            all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)

    print("ensemble len:",len(ensemble_slot_logits))
    print("label:",all_slot_label_mask)

    print("intent_before_mean:", ensemble_intent_logits, "intent_before_dim_0:", ensemble_intent_logits[0])
    # print("slot_before_mean:", ensemble_slot_logits, "slot_before_mean_shape:", ensemble_slot_logits.shape)
    # print("slot_before_mean_shape:", ensemble_slot_logits.shape)
    
    ensemble_intents = np.mean(ensemble_intent_logits[0], axis=0)
    ensemble_slots = np.mean(ensemble_slot_logits[0], axis=0)

    if len(ensemble_slot_logits) > 1:
        for i in range(1, len(ensemble_slot_logits)):
            # print(np.mean(ensemble_intent_logits[i], axis = 0).shape)
            ensemble_intents = np.append(ensemble_intents, np.mean(ensemble_intent_logits[i], axis=0), axis=0)
            ensemble_slots = np.append(ensemble_slots, np.mean(ensemble_slot_logits[i], axis=0), axis=0)
        
    # ensemble_slot_logits = np.argmax(ensemble_slot_logits, axis=2)
    print("ensemble_intent_logits:", ensemble_intents)
    print("ensemble_intent_logits_shape:", ensemble_intents.shape)
    print("slot:", ensemble_slots)
    print("slot_shape:",ensemble_slots.shape)
    # intent_preds = np.argmax(intent_preds, axis=1)
    intent_preds = np.argmax(ensemble_intents, axis=1)
    # ensemble_intent_probs = np.mean(ensemble_intent_probs, axis=0)
    # ensemble_slot_probs = np.mean(ensemble_slot_probs, axis=0)

    # print("intent_means:", ensemble_intent_probs)
    # print("slot_means:", ensemble_slot_probs[1])

    # intent_preds = np.argmax(ensemble_intent_logits, axis=2)
    print("intent", intent_preds, "intentpred_shape:", intent_preds.shape)
    # if not ensemble_args[0].use_crf:
    slot_preds = np.argmax(ensemble_slots, axis=2)
    print("slot_preds:",slot_preds)
    # slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    # slot_preds_list = [[] for _ in range(slot_preds.shape[0])]
    # Slot prediction

    slot_label_map = {i: label for i, label in enumerate(slot_label_lst)}
    
    slot_preds_list = [[] for _ in range(ensemble_slots.shape[0])]
    
    print(slot_label_map)
    # slot_label_map = {tuple(k): v for k, v in slot_label_map.items()}
    # for i in range(ensemble_slot_logits.shape[0]):
    #     for j in range(ensemble_slot_logits.shape[1]):
    #         if all_slot_label_mask[i, j] != pad_token_label_id:
    #             key = tuple(ensemble_slot_logits[i][j])
    #             if key in slot_label_map:
    #                 slot_preds_list[i].append(slot_label_map[key])

    for i in range(slot_preds.shape[0]):
        for j in range(slot_preds.shape[1]):
            if all_slot_label_mask[i, j] != pad_token_label_id:
                slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
            # print("pred:", slot_preds[0][i][j])

    # Write to output file
    print("intent_preds.size:", intent_preds.size)
    with open(pred_config.output_file, "w", encoding="utf-8") as f:
        for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
            line = ""
            for word, pred in zip(words, slot_preds):
                if pred == "O":
                    line = line + word + " "
                else:
                    line = line + "[{}:{}] ".format(word, pred)
            f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

    logger.info("Prediction Done!")

    # for i in range(slot_preds.shape[0]):
    #     for j in range(slot_preds.shape[1]):
    #         if all_slot_label_mask[i, j] != pad_token_label_id:
    #             slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

    # # Write to output file
    # with open(pred_config.output_file, "w", encoding="utf-8") as f:
    #     for words, slot_preds, intent_pred in zip(lines, slot_preds_list, intent_preds):
    #         line = ""
    #         for word, pred in zip(words, slot_preds):
    #             if pred == "O":
    #                 line = line + word + " "
    #             else:
    #                 line = line + "[{}:{}] ".format(word, pred)
    #         f.write("<{}> -> {}\n".format(intent_label_lst[intent_pred], line.strip()))

    # # logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default="sample_pred_in.txt", type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default="sample_pred_out.txt", type=str, help="Output file for prediction")
    parser.add_argument("--model_list", default="model_list.txt", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    pred_config = parser.parse_args()
    predict(pred_config)