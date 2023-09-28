import json
import random
import constants
import argparse


def read_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines

def write_file(path, lines):
    with open(path, 'w') as file:
        for line in lines:
            file.write(line)

def split_data(augmented=False):
    sentences = read_file(constants.SENTENCE_PATH)
    annotations = read_file(constants.ANNOTATION_PATH)
    intents = read_file(constants.INTENT_PATH)
    data_pairs = list(zip(sentences, annotations, intents))

    if augmented:
        augmented_sentences = read_file(constants.AUGMENTED_SENTENCE_PATH)
        augmented_annotations = read_file(constants.AUGMENTED_ANNOTATION_PATH)
        augmented_intents = read_file(constants.AUGMENTED_INTENT_PATH)
        augmented_data_pairs = list(zip(augmented_sentences, augmented_annotations, augmented_intents))
        random.shuffle(augmented_data_pairs)

        # split_point_1 = int(len(augmented_data_pairs) * 2 / 3)
        # split_point_2 = int(split_point_1 + len(augmented_data_pairs) / 6)

        # train_data = data_pairs + augmented_data_pairs[:split_point_1]
        # dev_data = augmented_data_pairs[split_point_1:split_point_2]
        # test_data = augmented_data_pairs[split_point_2:]
        train_data = data_pairs + augmented_data_pairs[:-2] 
        dev_data = augmented_data_pairs[-2:-1]
        test_data = augmented_data_pairs[-1:]
    else:
        random.shuffle(data_pairs)
        train_data = data_pairs[:-2]
        dev_data = data_pairs[-2:-1]
        test_data = data_pairs[-1:]

    print("train", len(train_data))
    print("dev", len(dev_data))
    print("test", len(test_data))

    train_sentenes, train_annotations, train_labels = zip(*train_data)
    test_sentenes, test_annotations, test_labels = zip(*test_data)
    dev_sentenes, dev_annotations, dev_labels = zip(*dev_data)

    write_file('slu_data_1/syllable-level/train/seq.in', train_sentenes)
    write_file('slu_data_1/syllable-level/train/seq.out', train_annotations)
    write_file('slu_data_1/syllable-level/train/label', train_labels)
    write_file('slu_data_1/syllable-level/test/seq.in', test_sentenes)
    write_file('slu_data_1/syllable-level/test/seq.out', test_annotations)
    write_file('slu_data_1/syllable-level/test/label', test_labels)
    write_file('slu_data_1/syllable-level/dev/seq.in', dev_sentenes)
    write_file('slu_data_1/syllable-level/dev/seq.out', dev_annotations)
    write_file('slu_data_1/syllable-level/dev/label', dev_labels)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument("--augment_data", action="store_true", help="Enable my augmented data")
    args = args.parse_args()

    split_data(args.augment_data)
    print("split successfully")
