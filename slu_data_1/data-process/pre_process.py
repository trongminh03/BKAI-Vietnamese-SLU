import json

import argparse


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-j', '--train_file', type=str, required = True,help='Path to train.jsonl file')
    args.add_argument('-o', '--output_file', type=str, required = True,help='Path to output file')
    args = args.parse_args()
    input_file_path = args.train_file
    output_file_path = args.output_file

    # Open the input and output files
    with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
        for line in input_file:
            # Parse the JSON object from the line
            json_data = json.loads(line.strip())
            
            # Extract sentence and entities
            sentence = json_data["sentence"].replace("phòng thời", "phòng thờ")
            entities = json_data["entities"]
            
            # Construct sentence_annotation from sentence and entities
            sentence_annotation = sentence
            for entity in entities:
                entity_type = entity["type"]
                entity_filler = entity["filler"]
                # Replace entity placeholders in sentence with [entity_type : entity_filler]
                sentence_annotation = sentence_annotation.replace(entity_filler, entity_type, 1)
            for entity in entities:
                entity_type = entity["type"]
                entity_filler = entity["filler"]
                if f"[ {entity_type} : {entity_filler} ]" not in sentence_annotation:
                    sentence_annotation = sentence_annotation.replace(entity_type, f"[ {entity_type} : {entity_filler} ]")


            # Add sentence_annotation to the JSON object
            json_data["sentence_annotation"] = sentence_annotation
            json_data["sentence"] = sentence
            # Write the updated JSON object to the output file
            output_file.write(json.dumps(json_data, ensure_ascii=False) + '\n')

    print(f'Sentence annotations have been added to {output_file_path}')
