import re
import json
import argparse
import zipfile


possible_confusion_words = ["à", "à mà thôi", "à thôi", "à nhầm", "nhầm", "à đâu", "à quên"]
def process_line(line, filename):
    device = None
    sentence = get_sentence(line)
    location = None
    parts = line.strip().split(" -> ")
    intent = parts[0][1:-1]  # Remove angle brackets from the intent
    if (len(parts) == 1):
        return {"intent": uppercase_first_character(intent), "entities": [], "file":filename}
    entities = re.findall(r'\[(.*?)\]', parts[1])
    # print(entities)
    processed_entities = []
    current_entity = {}
    entity_filler = ""
    last_entity = ""
    for entity in entities:
        loading_entity = "[" + entity + "]"
        if "device" in last_entity and "B-device" in loading_entity and  current_entity["filler"].split(" ")[0] not in entity :
            mix = last_entity + " " + loading_entity
            if mix in parts[1]:
                print(parts[1])
                entity = entity.replace("B", "I")
                print(last_entity, loading_entity)

        entity_parts = entity.split(":")
        if len(entity_parts) >= 2:
            entity_type_parts = entity_parts[1].split("-")
            if len(entity_type_parts) >= 2:

                if entity_type_parts[0] == "B":
                    if current_entity != {}:
                        # print(current_entity)       
                        processed_entities.append(current_entity)
                        current_entity = {}
                        entity_filler = ""
                
                entity_type = entity_type_parts[1].replace("_", " ")
                filler = entity_parts[0]
                # entry = {"type": entity_type, "filler": entity_filler}
                # entity_dict.append(entry)
                # if entity_type not in entity_dict:

                # entity_dict[entity_type].append(entity_filler)
                if entity_filler == "":
                    entity_filler = filler
                else:
                    entity_filler = entity_filler + " " + filler
                current_entity = {"type": entity_type, "filler": entity_filler}
        last_entity = "[" + entity + "]"
        
    if current_entity != {}:
        processed_entities.append(current_entity)
   
    # processed_entities = [{"type": key.replace('_', ' '), "filler": " ".join(value)} for key, value in entity_dict.items()]
    # processed_entities = entity_dict

    for entity in processed_entities:
        if entity["type"] == "device":
            device = entity["filler"]
        if entity["type"] == "location":
            location = entity["filler"]
            locations = location.split()
        if entity["type"] == "command" and "cho" in entity["filler"]:
            processed_entities.remove(entity)
        if entity["type"] == "command" and "làm" in entity["filler"]:
            processed_entities.remove(entity)
        if entity["type"] == "command" and entity["filler"] in ['đóng', 'sập', 'khép', 'khóa']:
            intent = "đóng thiết bị"
        if entity["type"] == "command" and entity["filler"] == "tăng":
            intent = intent.replace("giảm", "tăng")
        if entity["type"] == "command" and entity["filler"] == "mở":
            intent = "mở thiết bị"
        if entity["type"] == "command" and entity["filler"] == "dùng":
            entity["filler"] = "dừng"
            intent = "tắt thiết bị"

    if len([entity for entity in processed_entities if entity['type'] == 'command']) == 0 :
        if 'hoạt động' in sentence:
            intent = 'kiểm tra tình trạng thiết bị'

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

def format(jsonl_line):
    intent = jsonl_line['intent']
    device = ""
    pattern = r'(\d+)(\s+)độ.*'
    have_percentage = False
    have_degree = False
    for entity in jsonl_line['entities']:
        if entity['type'] == 'device':
            device = entity['filler']
        if "%" in entity['filler']:
            have_percentage = True
        if re.match(pattern, entity['filler']):
            have_degree = True
    if (have_percentage and device == ""):
        jsonl_line['intent'] = intent.replace("mức độ", "độ sáng").replace("âm lượng", "độ sáng").replace("nhiệt độ", "độ sáng")
    if (have_degree and device == ""):
        jsonl_line['intent'] = intent.replace("độ sáng", "nhiệt độ").replace("âm lượng", "nhiệt độ").replace("mức độ", "nhiệt độ")
    return jsonl_line

def get_sentence(txt):
    input_tokens = txt.split()

    # Initialize an empty list to store the generated output
    output_tokens = []

    # Loop through the input tokens
    for token in input_tokens:
        if token.startswith("[") and token.endswith("]"):
            # Extract the content inside square brackets
            content = token[1:-1]
            output_tokens.append(content)
        else:
            output_tokens.append(token)

    # Join the output tokens to form the final output text
    generated_output = " ".join(output_tokens)
    return generated_output

def cut_entity(jsonl_line):
    # return jsonl_line
    # Create a dictionary to store entities by their type
    entity_dict = {}

    # Filter and keep only the first entity of each type
    entities = []
    for entity in jsonl_line['entities']:
        entity_type = entity['type']
        # if entity_type not in entity_dict:
        entity_dict[entity_type] = entity

        print(entity_dict[entity_type])
        # else:
    for entity in entity_dict:
        entities.append(entity_dict[entity])
        # print(entity)
    jsonl_line['entities'] = entities
    return jsonl_line

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
            jsonline = process_line(line, filename)
            reformat = format(jsonline)
            result = cut_entity(reformat)
            output_file.write(json.dumps(result, ensure_ascii=False) + "\n")
    
    zip_file(output_filename, zip_filename + ".zip")