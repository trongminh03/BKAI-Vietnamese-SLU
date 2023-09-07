import json
import random

input_file_path = './train_processed.jsonl'
output_sentence_path = './seq.in'
output_annotation_path = './seq.out'
output_label_path = './label'

input_augmented_file_path = './augmented_data.jsonl'
output_augmented_sentence_path = './augmented_seq.in'
output_augmented_annotation_path = './augmented_seq.out'
output_augmented_label_path = './augmented_label'

def read_file(path):
    with open(path, 'r') as file:
        lines = file.readlines()
    return lines

def write_file(path, lines):
    with open(path, 'w') as file:
        for line in lines:
            file.write(line)

def split_train_test():
    sentences = read_file(output_sentence_path)
    annotations = read_file(output_annotation_path)
    labels = read_file(output_label_path)
    data_pairs = list(zip(sentences, annotations, labels))
    random.shuffle(data_pairs)

    augmented_sentences = read_file(output_augmented_sentence_path)
    augmented_annotations = read_file(output_augmented_annotation_path)
    augmented_labels = read_file(output_augmented_label_path)
    augmented_data_pairs = list(zip(augmented_sentences, augmented_annotations, augmented_labels))
    random.shuffle(augmented_data_pairs)

    # split_point_1 = int(0.7 * len(data_pairs))
    # split_point_2 = split_point_1 + int(0.1 * len(data_pairs))

    # train_data = data_pairs[:split_point_1]
    # dev_data = data_pairs[split_point_1:split_point_2]
    # test_data = data_pairs[split_point_2:]
    train_data = data_pairs + augmented_data_pairs[:3120] 
    dev_data = augmented_data_pairs[3120: 5120]
    test_data = data_pairs[5120:]

    train_sentenes, train_annotations, train_labels = zip(*train_data)
    test_sentenes, test_annotations, test_labels = zip(*test_data)
    dev_sentenes, dev_annotations, dev_labels = zip(*dev_data)

    write_file('../syllable-level/train/seq.in', train_sentenes)
    write_file('../syllable-level/train/seq.out', train_annotations)
    write_file('../syllable-level/train/label', train_labels)
    write_file('../syllable-level/test/seq.in', test_sentenes)
    write_file('../syllable-level/test/seq.out', test_annotations)
    write_file('../syllable-level/test/label', test_labels)
    write_file('../syllable-level/dev/seq.in', dev_sentenes)
    write_file('../syllable-level/dev/seq.out', dev_annotations)
    write_file('../syllable-level/dev/label', dev_labels)

split_train_test()
print("split successfully")
