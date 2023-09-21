import json
import argparse 
import constants

def in_label_converter(train_file_path, intent_path, sentence_path): 
    # Read the JSONL file
    with open(train_file_path, 'r', encoding='utf-8') as jsonl_file:
        intents = []
        sentences = []
        for line in jsonl_file:
            entry = json.loads(line)
            
            intent = entry['intent']
            sentence = entry['sentence']
            intents.append(intent)
            sentences.append(sentence)

    # Write intents to the label file
    with open(intent_path, 'w', encoding='utf-8') as label_file:
        for intent in intents:
            label_file.write(intent + '\n')
            
    with open(sentence_path, 'w', encoding='utf-8') as in_file:
        for seq in sentences:
            modified_seq = seq.replace(',', ' ')
            modified_seq = modified_seq.replace(']', ' ] ')
            modified_seq = modified_seq.replace('\'',' ')
            modified_seq = modified_seq.replace('.',' ')
            modified_seq = modified_seq.replace('?', ' ')
            modified_seq = modified_seq.replace('!',' ')
            modified_seq = modified_seq.replace('/',' ')
            modified_seq.strip()
            in_file.write(modified_seq + '\n')
            
if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--augment_data", action="store_true", help="Enable my augmented data")
    args = args.parse_args()
    
    in_label_converter(constants.TRAIN_FILE_PATH, constants.INTENT_PATH, constants.SENTENCE_PATH)

    if args.augment_data: 
        in_label_converter(constants.AUGMENTED_TRAIN_FILE_PATH, constants.AUGMENTED_INTENT_PATH, constants.AUGMENTED_SENTENCE_PATH) 

    print("Intents extracted and saved to 'label and seq.in'")