import json
import random


subjects = ["anh", "tớ", "chị", "mình", "tôi"]
names = ["minh", "quân", "my", "khang", "hân", "hải", "huy", "thịnh"]
scenes = []
locations = []
decision_words = ["ra", "vào", "ra khỏi", "đến"]
reasons = ["vì vậy", "nên"]
verbs = ["kích hoạt", "mở", "hủy", "tắt", "bỏ qua"]
def generate_random_sentence(intent_device_mapping):
    valid_intents = [intent for intent in intent_device_mapping.keys() if "hoạt cảnh" not in intent]
    if not valid_intents:
        return None  # No valid intents found
    random_intent = random.choice(valid_intents)

    subject = random.choice(subjects)

    owners = names + [subject]

    owner = random.choice(owners)
    # print(owner)
    # Extract the command from the random intent
    if "kiểm tra" in random_intent:
        command = "kiểm tra"
    else:
        command = random_intent.split()[0]
    s2 = command
    # print(s2)
    # Choose a random device fit with the intent

    devices = intent_device_mapping[random_intent] 

    device = random.choice(devices)
    
    random_device = device + random.choice(["", " của " + owner])

    if owner not in subjects:
        device = random_device
    # print(random_device)
    # Initialize the sentence
    sentence = f"{command}"
    # if random.random() < 0.5:
    partition_augment = ""
    target_number = ""
    changing_value = ""
    duration = ""
    time_at = ""
        # s5 = f" {'hộ' if random.random() < 0.5 else 'cho'} {subject}" 
    sentence_augment = random.choice( ["", random.choice(  ["hộ " + random.choice([subject, ""]), "cho " + subject, "giúp " + subject])])

    device_augment = random.choice(["cái ", ""])

    augment_pos= random.randint(1, 7)
    if "tăng" in random_intent:
        # Add "lên" after device
        
        # Randomly choose between adding changing value or target number
        if random.random() < 0.5:
            # create changing value
            partition_augment = random.choice(["lên ", ""]) + random.choice(["thêm ", ""]) + random.choice(["tầm ", "khoảng "])
            if "nhiệt độ" in random_intent:
                changing_value = random.choice(["", str(random.randint(1, 100)) + " độ c"])
            elif "mức độ" in random_intent:
                changing_value = random.choice(["", str(random.randint(1, 10)) + random.choice([" mức", ""])])
                partition_augment.replace("tầm", "").replace("khoảng","")
            else :
                changing_value = random.choice(["", str(random.randint(1, 100)) + "%"])
            if changing_value == "":
                partition_augment = ""
        else:
            # create target number
            partition_augment = random.choice(["lên ", ""]) + random.choice(["đến " , "tới "]) + random.choice(["tầm ", "khoảng "])
            if "nhiệt độ" in random_intent:
                target_number = random.choice(["", str(str(random.randint(1, 100))) + " độ c"])
            elif "mức độ" in random_intent:
                target_number = random.choice(["", random.choice(["mức ", ""]) + str(random.randint(1, 10))])
                partition_augment.replace("tầm", "").replace("khoảng","")
            else :
                target_number = random.choice(["", str(str(random.randint(1, 100))) + "%"])
            if target_number == "":
                partition_augment = ""

    elif "giảm" in random_intent:

        # Randomly choose between adding changing value or target number
        if random.random() < 0.5:
            # create changing value
            partition_augment = random.choice(["xuống ", ""]) + random.choice(["bớt ", ""]) + random.choice(["tầm ", "khoảng "])
            if "nhiệt độ" in random_intent:
                changing_value = random.choice(["", str(random.randint(1, 100)) + " độ c"])
            elif "mức độ" in random_intent:
                changing_value = random.choice(["", str(random.randint(1, 10)) + random.choice([" mức", ""])])
                partition_augment.replace("tầm", "").replace("khoảng","")
            else :
                changing_value = random.choice(["", str(random.randint(1, 100)) + "%"])
            if changing_value == "":
                partition_augment = ""
        else:
            # create target number
            partition_augment = random.choice(["xuống ", ""]) + random.choice(["đến " , "tới ", "còn "]) + random.choice(["tầm ", "khoảng "])
            if "nhiệt độ" in random_intent:
                target_number = random.choice(["", str(random.randint(1, 100)) + " độ c"])
            elif "mức độ" in random_intent:
                target_number = random.choice(["", random.choice(["mức ", ""]) + str(random.randint(1, 10))])
                partition_augment.replace("tầm", "").replace("khoảng","")
            else :
                target_number = random.choice(["", str(random.randint(1, 100)) + "%"])
            if target_number == "":
                partition_augment = ""

    time_at_prefix = ""
    duration_prefix = ""
    duration_postfix = ""
    # Add "thêm time at" or "thêm duration" randomly
    if random.random() < 0.5:
        if random.random() < 0.5:
            time_at_prefix = random.choice(["khoảng ", "vào lúc ", "vào tầm "])
            time_at = generate_random_time_at()
        else:
            duration_prefix = random.choice(["trong khoảng ", "trong vòng ", "trong "])
            duration = generate_random_duration()
            duration_postfix = random.choice([" nữa", ""])
    ending = random.choice(["với ", ""]) + random.choice(['nhé', 'nhá', 'nha', 'nhớ', ""])

    order_1 = [command, device_augment + random_device, partition_augment, target_number, changing_value, time_at_prefix + time_at,  duration_prefix + duration + duration_postfix, ending]
    if changing_value == "" and target_number =="" and time_at == "" and duration =="": 
        order_2 = [command, device_augment + random_device, ending]
    else :
        order_2 = [command, device_augment + random_device, ending, command, partition_augment, target_number, changing_value, time_at_prefix + time_at, duration_prefix + duration + duration_postfix]
    
    order_3 = [time_at_prefix + time_at , duration_prefix + duration + duration_postfix , command, device_augment + random_device, partition_augment, target_number, changing_value, ending]
    
    my_order = random.choice([order_1, order_2, order_3])


    
    possible_entities = [
        {"type": "command", "filler": f"{command}"},
        {"type": "device", "filler": f"{device}"},
        {"type": "time at", "filler": f"{time_at}"},
        {"type": "duration", "filler": f"{duration}"},
        {"type": "changing value", "filler": changing_value.replace("mức", "").strip()},
        {"type": "target number", "filler": target_number.replace("mức", "").strip()},
    ]
    entities = get_entities_order(my_order, possible_entities)
    
    generated_sentence = combineword(my_order, sentence_augment, command, ending)

    sentence_data = {
        "id": "none",  # You can generate a unique ID here if needed
        "sentence": generated_sentence,
        "intent": random_intent,
        "sentence_annotation": "sentence_annotation",
        "entities": entities,
        "file": "none.wav"  # Replace with the actual file name
    }

    return sentence_data

def get_entities_order(order, entities_set):
    entities = []
    for keyword in order:
        if keyword != "":
            for entity in entities_set:
                if entity["filler"] in keyword and entity["filler"] != "":
                    # if entity not in entities: 
                    entities.append(entity)
    return entities



    # return s1 + " " + s2 + " " + s3 + " " + s4 + " " + s5 + " " + s6 + " " + s7 + " " + s8 + " " + s9
def combineword(order, sentence_augment, command, ending):
    index = 1
    while (order[index] == command or (order[index - 1] == ending and ending != "")):
        index = random.randint(1, 5)
    sentence = ""
    order.insert(index, sentence_augment)
    for word in order:
        if word != "":
            sentence += word.strip() + " "
    return sentence.strip()

def generate_random_time_at():
    # Generate a random hour (between 0 and 23) and a random minute (between 0 and 59)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    # Format the hour and minute as a time string
    time_at = f"{hour} giờ {minute} phút"

    return time_at.strip()


def generate_random_duration():
    # Generate a random hour (between 0 and 23) and a random minute (between 0 and 59)
    hour = random.randint(0, 23)
    minute = random.randint(0, 59)
    # Format the hour and minute as a time string
    if hour == 0: 
        time_at = f"{minute} phút"
    else: 
        time_at = f"{hour} tiếng {minute} phút"
    return time_at.strip()


def mapping_data():
    intent_device_mapping = {}
    am_luong_devices = set()
    muc_do_devices = set()
    do_dang_devices = set()
    nhiet_do_devices = set()
    scene_set = set()
    location_set = set()
    input_file_path = "slu_data_1/data-process/train_processed.jsonl"
    # Open the JSONL file and iterate through its lines
    with open(input_file_path, 'r', encoding='utf-8') as input_file:
        for line in input_file:
            json_data = json.loads(line.strip())
            # Extract the intent and devices from the current line
            intent = json_data['intent']
            sentence = json_data['sentence']
            # Extract the command as a list
            command = [entity['filler'] for entity in json_data['entities'] if entity['type'] == 'command']
            devices = {entity['filler'] for entity in json_data['entities'] if entity['type'] == 'device'}  # Use set comprehension
            scene = {entity['filler'] for entity in json_data['entities'] if entity['type'] == 'scene'} 
            location = {entity['filler'] for entity in json_data['entities'] if entity['type'] == 'location'} 
            scene_set.update(scene)
            location_set.update(location)

            # Check if the intent is already in the mapping, if not, create a new set
            if intent not in intent_device_mapping:
                    intent_device_mapping[intent] = set()
            # Check if the intent contains "âm lượng"
            if "âm lượng" in intent:
                # Add the devices to the separate set for "âm lượng"
                am_luong_devices.update(devices)
            elif "mức độ" in intent:
                muc_do_devices.update(devices)
            elif "độ sáng" in intent:
                do_dang_devices.update(devices)
            elif "nhiệt độ" in intent:
                nhiet_do_devices.update(devices)
            else:
                # Add the devices to the set for the current intent
                intent_device_mapping[intent].update(devices)  # Use update to merge sets
                
    # Save the extracted text to a text file


    # Add the set of "âm lượng" devices to each intent
    for intent in intent_device_mapping:
        if "âm lượng" in intent:
            intent_device_mapping[intent].update(am_luong_devices)
        if "mức độ" in intent:
            intent_device_mapping[intent].update(muc_do_devices)
        if "độ sáng" in intent:
            intent_device_mapping[intent].update(do_dang_devices)
        if "nhiệt độ" in intent:
            intent_device_mapping[intent].update(nhiet_do_devices)

    mapping = {intent: list(devices) for intent, devices in intent_device_mapping.items()}
    scenes.extend(scene_set)
    locations.extend(location_set)
    print(scene)
    return mapping

def combinescene(order):
    sentence = ""
    for word in order:
        if word != "":
            sentence += word.strip() + " "
    return sentence.strip()

def generate_scene_sentence():
    location = random.choice(locations)
    subject = random.choice(subjects)
    decision = random.choice(decision_words)
    scene = random.choice(scenes)
    verb = random.choice(verbs)
    scene_verb = random.choice(["chuẩn bị ", " "]) + scene
    scene_config = random.choice(["hoạt cảnh ", "một chút ", "một chút sự ", "một chút không khí ", "chế độ ", ""]) + scene
    time_at = ""
    duration = ""
    time_at_prefix = ""
    duration_prefix = ""
    duration_postfix = ""
    # Add "thêm time at" or "thêm duration" randomly
    if random.random() < 0.5:
        if random.random() < 0.5:
            time_at_prefix = random.choice(["khoảng ", "vào lúc ", "vào tầm "])
            time_at = generate_random_time_at()
        else:
            duration_prefix = random.choice(["trong khoảng ", "trong vòng ", "trong "])
            duration = generate_random_duration()
            duration_postfix = random.choice([" nữa", ""])
    ending = random.choice(['nhé', 'nhá', 'nha', 'nhớ', ""])
    intent = ""

    # Generate other components based on your plan
    main_sentence = generate_main_sentence(subject, verb, scene_config, random.choice(["cảnh ", ""]) + scene, scene)
    location_word = random.choice(["ở " + location, location, ""])

    order_4 = [time_at_prefix + time_at , duration_prefix + duration + duration_postfix ,main_sentence, location_word, ending]

    order_5 = [subject, decision, location, "để", scene_verb, ending]

    order_6 = [subject, "đang ở " + location, "nhưng mà", subject,  scene_verb, ending]

    my_order = random.choice([order_4, order_5, order_6])
    if decision in my_order:
        if "ra khỏi" in decision:
            intent = "hủy hoạt cảnh"
        else:
            intent = "kích hoạt cảnh"
    elif main_sentence in my_order:
        if "không" in main_sentence or "hủy" in main_sentence or "tắt" in main_sentence or "bỏ qua" in main_sentence:
            intent = "hủy hoạt cảnh"
        else:
            intent = "kích hoạt cảnh"
    elif "nhưng mà" in my_order:
        intent = "hủy hoạt cảnh"
    else:
        intent = "kích hoạt cảnh"
    
    possible_entities = [
        {"type": "time at", "filler": f"{time_at}"},
        {"type": "duration", "filler": f"{duration}"},
        {"type": "scene", "filler": scene},
        {"type": "location", "filler": location},
    ]
    entities = get_entities_order(my_order, possible_entities)
    generated_sentence = combinescene(my_order)
    if "không cần" in generated_sentence or "không muốn" in generated_sentence:
        generated_sentence.replace("với", "nữa")
    sentence_data = {
        "id": "none",  # You can generate a unique ID here if needed
        "sentence": generated_sentence,
        "intent": intent,
        "sentence_annotation": "sentence_annotation",
        "entities": entities,
        "file": "none.wav"  # Replace with the actual file name
    }
    return sentence_data



def generate_main_sentence(subject, verb, scene_config, target_config, scene):
    # Randomly select a format
    format_choice = random.choice([1, 2, 3])

    if format_choice == 1:
        # Format: (S + sẽ/đang + muốn/cần/không muốn/ không cần + V + target_config)
        main_sentence = f"{subject} {'sẽ' if random.random() < 0.5 else 'đang'} {'muốn' if random.random() < 0.5 else 'cần' if random.random() < 0.5 else 'không muốn' if random.random() < 0.5 else 'không cần'} {target_config}"

    elif format_choice == 2:
        # Format: (V + cho + S + scene_config)
        main_sentence = f"{verb} cho {subject} {scene_config}"

    else:
        # Format: (S + chuẩn bị + scene)
        main_sentence = f"{subject} chuẩn bị {scene}"

    return main_sentence

# mapping_data = mapping_data()
# print(mapping_data)
def generate_sentences(x, y):
    intent_device_mapping = mapping_data()
    sentences_data = []
    for _ in range(x):
        sentence_data = generate_random_sentence(intent_device_mapping)
        if sentence_data:
            sentences_data.append(sentence_data)

    for _ in range(y):
        sentence_data = generate_scene_sentence()
        if sentence_data:
            sentences_data.append(sentence_data)
    # Save the sentences data to a JSON file
    output_file_path = "slu_data_1/data-process/random_sentences.jsonl"
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        for sentence_data in sentences_data:
            # Create a separate JSON string for each sentence data and write it to a separate line
            output_file.write(json.dumps(sentence_data, ensure_ascii=False) + '\n')

    print(f"Generated {x + y} random sentences data and saved to {output_file_path}")



generate_sentences(3000, 300)