vietnamese_numbers = {

'10': 'mười ',

'11': 'mười một ',

'12': 'mười hai ',

'13': 'mười ba ',

'14': 'mười bốn ',

'15': 'mười năm ',

'16': 'mười sáu ',

'17': 'mười bảy ',

'18': 'mười tám ',

'19': 'mười chín ',

'20': 'hai mươi',

'30': 'ba mươi',

'40': 'bốn mươi',

'50': 'năm mươi',

'60': 'sáu mươi',

'70': 'bảy mươi',

'80': 'tám mươi',

'90': 'chín mươi',

'0': 'không ',

'1': 'một ',

'2': 'hai ',

'3': 'ba ',

'4': 'bốn ',

'5': 'năm ',

'6': 'sáu ',

'7': 'bảy ',

'8': 'tám ',

'9': 'chín ',

'%': ' phần trăm'

}

import json

import argparse

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-j', '--jsonline_path', type=str, required = True,help='Path to jsonline train file')
    args = args.parse_args()
    
    jsonl_file_path = args.jsonline_path
    data = []
    saved = 'klm_train.txt'

    def change_to_word(transcript):

        for key,value in vietnamese_numbers.items():
            if key in transcript:
                have_number = True
            transcript = transcript.replace(key, value)

        transcript = transcript.strip()

        # Split the string into words and join them with a single space

        transcript = ' '.join(transcript.split()).lower()

        return transcript

    # Open and read the JSONL file line by line

    with open(saved,'w') as txt_file:

        with open(jsonl_file_path, 'r') as jsonl_file:

            for line in jsonl_file:

                # Parse each line as a JSON object and append it to the list

                data = json.loads(line)

                txt_file.write(change_to_word(data['sentence']) + '\n')

                # for entity in data['entities']:

                #    contains_digit = any(char.isdigit() for char in entity['filler'])

                #    if not contains_digit:

                #        txt_file.write(change_to_word(entity['filler']) + '\n')