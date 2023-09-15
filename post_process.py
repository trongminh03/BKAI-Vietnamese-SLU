import re
import json
import argparse
import zipfile

def process_line(line, filename):
    device = None
    location = None
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

    if "kiểm tra" not in intent and "hoạt cảnh" not in intent:
        for entity in processed_entities:
            if entity["type"] == "command":
                entity["filler"] = intent.split()[0].lower()

    for entity in processed_entities:
        if entity["type"] == "device":
            device = entity["filler"]
        if entity["type"] == "location":
            location = entity["filler"]
            locations = location.split()
    if (device):
        if (device.split()[-1] == "của"):
            if (location): 
                device = device + " " +  locations[0]
                # remove first word in location
                location = " ".join(locations[1:])
                # print(device)
                for entity in processed_entities:
                    if entity["type"] == "device":
                        entity["filler"] = device
                    if entity["type"] == "location":
                        entity["filler"] = location
                        rmentity = entity
                if len(locations) == 1:
                    # remove location in entities
                    processed_entities.remove(rmentity)
        if (location): 
            if locations[0] == "của":
                device = device + " " + location
                # print(device)
                for entity in processed_entities:
                    if entity["type"] == "device":
                        entity["filler"] = device
                    if entity["type"] == "location":
                        rmentity = entity
                processed_entities.remove(rmentity)

    return {"intent": uppercase_first_character(intent), "entities": processed_entities, "file":filename}

def uppercase_first_character(input):
    first_character = input[0]
    upper_first_character = first_character.upper()
    return upper_first_character + input[1:]

def zip_file(input_file, output_zip):
    try:
        with zipfile.ZipFile(output_zip, 'w', zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(input_file, arcname=input_file.split("/")[-1])
        print(f"{input_file} has been successfully zipped to {output_zip}")
    except FileNotFoundError:
        print(f"Error: {input_file} not found.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")


zip_filename = 'Submission'

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input_file', type=str, required = True,help='Path to output.txt file')
    args.add_argument('-t', '--wav_urls', type=str, required = True,help='Path to transcript.txt file')
    args.add_argument('-o', '--output_file', type=str, required = True,help='Predictions.jsonl file') 
    args = args.parse_args()

    input_filename = args.input_file
    wav_urls_filename = args.wav_urls
    output_filename = args.output_file

    filenames=[]
    with open(wav_urls_filename, 'r', encoding="utf-8") as wav_urls_file:
        wav_urls = wav_urls_file.readlines()

    for wav in wav_urls:
        filename = re.findall(r'/(\w+\.wav)', wav)[0]
        filenames.append(filename)


    with open(input_filename, "r", encoding="utf-8") as input_file, open(output_filename, "w", encoding="utf-8") as output_file:
        for line, filename in zip(input_file, filenames):
            result = process_line(line, filename)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    zip_file(output_filename, zip_filename + ".zip")