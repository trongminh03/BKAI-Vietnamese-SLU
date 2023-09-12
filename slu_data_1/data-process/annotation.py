import json

# Input JSONL file path and output JSONL file path
input_file_path = 'train_20230909.jsonl'
output_file_path = 'aannotations.jsonl'

# Open the input and output files
with open(input_file_path, 'r', encoding='utf-8') as input_file, open(output_file_path, 'w', encoding='utf-8') as output_file:
    for line in input_file:
        # Parse the JSON object from the line
        json_data = json.loads(line.strip())
        
        # Extract sentence and entities
        sentence = json_data["sentence"]
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
            sentence_annotation = sentence_annotation.replace(entity_type, f"[{entity_type} : {entity_filler}]")


        # Add sentence_annotation to the JSON object
        json_data["sentence_annotation"] = sentence_annotation
        
        # Write the updated JSON object to the output file
        output_file.write(json.dumps(json_data, ensure_ascii=False) + '\n')

print(f'Sentence annotations have been added to {output_file_path}')
