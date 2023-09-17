import json
import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('--intent_predicts', required= True, help = 'Path to intent predictions')
    args.add_argument('--slot_predicts', required= True, help = 'Path to slot predictions')
    args.add_argument('--output', default = './ensemble_predict.jsonl', help = 'output predictions')
    args = args.parse_args()
    # Read the contents of both JSONL files
    with open(args.intent_predicts, 'r', encoding='utf-8') as f1, open(args.slot_predicts, 'r', encoding='utf-8') as f2:
        first_lines = f1.readlines()
        second_lines = f2.readlines()

    # Iterate through the lines and replace the intent in the second file
    for i in range(len(second_lines)):
        second_data = json.loads(second_lines[i])
        first_data = json.loads(first_lines[i])
        second_data['intent'] = first_data['intent']
        second_lines[i] = json.dumps(second_data, ensure_ascii=False) + '\n'  # Disable ASCII encoding

    # Write the updated second file with explicit UTF-8 encoding
    with open(args.output, 'w', encoding='utf-8') as f2_updated:
        f2_updated.writelines(second_lines)

    print('Ensemble done')
