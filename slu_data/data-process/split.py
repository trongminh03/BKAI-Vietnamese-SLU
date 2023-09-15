import json
import random

output_sentence_path = 'slu_data/data-process/seq.in'
output_annotation_path = 'slu_data/data-process/seq.out'
output_label_path = 'slu_data/data-process/label'

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
    labels = read_file(output_label_path
                       )
    data_pairs = list(zip(sentences, annotations, labels))
    random.shuffle(data_pairs)

    train_data = data_pairs[:-2] 
    dev_data = data_pairs[-2:-1]
    test_data = data_pairs[-1:]

    train_sentenes, train_annotations, train_labels = zip(*train_data)
    test_sentenes, test_annotations, test_labels = zip(*test_data)
    dev_sentenes, dev_annotations, dev_labels = zip(*dev_data)

    write_file('slu_data/syllable-level/train/seq.in', train_sentenes)
    write_file('slu_data/syllable-level/train/seq.out', train_annotations)
    write_file('slu_data/syllable-level/train/label', train_labels)
    write_file('slu_data/syllable-level/test/seq.in', test_sentenes)
    write_file('slu_data/syllable-level/test/seq.out', test_annotations)
    write_file('slu_data/syllable-level/test/label', test_labels)
    write_file('slu_data/syllable-level/dev/seq.in', dev_sentenes)
    write_file('slu_data/syllable-level/dev/seq.out', dev_annotations)
    write_file('slu_data/syllable-level/dev/label', dev_labels)

split_train_test()
print("split successfully")
