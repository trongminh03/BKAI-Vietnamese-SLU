import json
import random

input_file_path = 'data/train.jsonl'
output_sentence_path = 'data/seq.in'
output_annotation_path = 'data/seq.out'

# Extract all input sentences
def get_sentences():
    # Extract values of the "sentence" key and write them to the output file
    with open(input_file_path, 'r') as input_file, open(output_sentence_path, 'w') as output_file:
        for line in input_file:
            try:
                data = json.loads(line)
                sentence = data.get('sentence', '')
                if sentence:
                    output_file.write(sentence + '\n')
            except json.JSONDecodeError:
                print("Error decoding JSON line:", line)

    print("Extraction and writing sentences complete.")

# Extract values of the "sentence" key and write them to the output file
def transform_annotation(input_string):
    input_string = input_string.split()
    idx = 0            
    output = ""
    while idx < len(input_string):
        if input_string[idx] == "[":
            idx += 1
            # tmp = input_string[idx]
            tmp = ""
            while idx < len(input_string) and input_string[idx] != ":":
                tmp += input_string[idx] + "_"
                idx += 1
            tmp = tmp[:-1]
            idx += 1
            id = 0
            while idx < len(input_string) and input_string[idx] != "]":
                if id == 0:
                    output += "B-" + tmp + " "
                else:
                    output += "I-" + tmp + " "
                id += 1
                idx += 1
            idx += 1
        else:
            output += "O "
            idx += 1
    return output

def get_annotations():
    with open(input_file_path, 'r') as input_file, open(output_annotation_path, 'w') as output_file:
        for line in input_file:
            try:
                data = json.loads(line)
                annotation = data.get('sentence_annotation', '')
                if annotation:
                    annotation = transform_annotation(annotation)
                    output_file.write(annotation + '\n')
            except json.JSONDecodeError:
                print("Error decoding JSON line:", line)

    print("Extraction and writing annotations complete.")

# Split the data into train and test sets
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

    split_point = int(0.7 * len(data_pairs))

    train_data = data_pairs[:split_point]
    test_data = data_pairs[split_point:]

    train_sentenes, train_annotations, train_labels = zip(*train_data)
    test_sentenes, test_annotations, test_labels = zip(*test_data)

    write_file('data/train/seq.in', train_sentenes)
    write_file('data/train/seq.out', train_annotations)
    write_file('data/train/label', train_labels)
    write_file('data/test/seq.in', test_sentenes)
    write_file('data/test/seq.out', test_annotations)
    write_file('data/test/label', test_labels)

output_label_path = 'data/label'
def extract_label():
    with open(input_file_path, 'r') as input_file, open(output_label_path, 'w') as output_file:
        for line in input_file:
            try:
                data = json.loads(line)
                label = data.get('intent', '')
                if label:
                    output_file.write(label + '\n')
            except json.JSONDecodeError:
                print("Error decoding JSON line:", line)

    print("Extraction and writing labels complete.")

if __name__ == "__main__":
    get_sentences()
    get_annotations()
    extract_label()
    split_train_test()