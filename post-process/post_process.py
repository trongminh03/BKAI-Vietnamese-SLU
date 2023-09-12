import re
import json

def process_line(line, filename):
    parts = line.strip().split(" -> ")
    intent = parts[0][1:-1]  # Remove angle brackets from the intent
    entities = re.findall(r'\[(.*?)\]', parts[1])

    entity_dict = {}
    for entity in entities:
        entity_parts = entity.split(":")
        
        if len(entity_parts) >= 2:
            entity_type_parts = entity_parts[1].split("-")
            if len(entity_type_parts) >= 2:
                entity_type = entity_type_parts[1]
                entity_filler = entity_parts[0]
                
                if entity_type not in entity_dict:
                    entity_dict[entity_type] = []
                entity_dict[entity_type].append(entity_filler)
   
    processed_entities = [{"type": key.replace('_', ' '), "filler": " ".join(value)} for key, value in entity_dict.items()]
   
    return {"intent": intent, "entities": processed_entities, "file":filename}

input_filename = "uppercase.txt"
output_filename = "predictions.jsonl"
wav_urls_filename = "../transcript_new_1.txt"

filenames=[]
with open(wav_urls_filename, 'r') as wav_urls_file:
    wav_urls = wav_urls_file.readlines()

for wav in wav_urls:
    filename = re.findall(r'/(\w+\.wav)', wav)[0]
    filenames.append(filename)


with open(input_filename, "r", encoding="utf-8") as input_file, open(output_filename, "w", encoding="utf-8") as output_file:
    for line, filename in zip(input_file, filenames):
        result = process_line(line, filename)
        output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
