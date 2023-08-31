import re
import json

def label_tokens(text):
    labels = []
    
    matches = re.finditer(r'\[ ([^:]+) : ([^\]]+) \]', text)
    last_end = 0
    number = 0
    for match in matches:
        start = match.start(2)
        end = match.end(2)
        label_type = match.group(1)
        label_type = label_type.replace(' ', '_')
        num_underscores = label_type.count('_')
        label_value = match.group(2)
        # Tokelabel_typetside the brackets and on the right side of the brackets
        if (number == 0):
            labels.extend(['O'] * (len(text[last_end:start].split()) - 3 - num_underscores))
            number += 1
        else:
            labels.extend(['O'] * (len(text[last_end:start].split()) - 4 - num_underscores))
        labels.extend(['B-' + label_type] + ['I-' + label_type] * (len(label_value.split()) - 1))
        last_end = end
    # Tokenize remaining words
    labels.extend(['O'] * (len(text[last_end:].split()) - 1))
    
    return labels

# Read data from the JSONL file
data = []
with open('data-process/train_processed.jsonl', 'r', encoding='utf-8') as jsonl_file:
    for line in jsonl_file:
        entry = json.loads(line)
        data.append(entry)

with open('data-process/seq.out', 'w', encoding='utf-8') as output_file:
    for entry in data:
        sentence_annotation = entry['sentence_annotation']
        modified_seq = sentence_annotation.replace(',', ' ')
        modified_seq = modified_seq.replace(']', ' ] ')
        modified_seq = modified_seq.replace('[', ' [')
        modified_seq = modified_seq.replace('\'',' ')
        modified_seq = modified_seq.replace('.',' ')
        modified_seq = modified_seq.replace('?', ' ')
        modified_seq = modified_seq.replace('!',' ')
        modified_seq = modified_seq.replace('/',' ')
        labels = label_tokens(modified_seq)
        
        for label in labels:
            output_file.write(f"{label} ")
        output_file.write('\n')   

         
print("Label sequences generated and saved to 'seq.out'")
