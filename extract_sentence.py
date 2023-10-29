import argparse
import re
# # Define the input and output file paths
# input_file_path = "transcript_new_1.txt"
# output_file_path = "input.txt"

def replace_number(line: str):
    units = ['mức', 'độ', '%']
    if line.find('hạ') != -1 or line.find('giảm') != -1 or line.find('xuống') == -1:
        return line
    words = line.split()
    for i in range(len(words)):
        if words[i] == 'xuống':
            if i < len(words) - 1:
                if words[i + 1].find(units[2]) != -1:
                    return line
                else:
                    if words[i + 1].isdigit():
                        if i + 2 < len(words):
                            if words[i + 2] not in units:
                                words[i] = 'số'
                        else:
                            words[i] = 'số'
    return ' '.join(words)


if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input_path', type=str, required = True,help='Path to transcript.txt file')
    args.add_argument('-o', '--output_path', type=str, required = True,help='Path to sentence.txt file')
    args = args.parse_args()
    input_file_path = args.input_path 
    output_file_path = args.output_path


    # Create a list to store the extracted text
    extracted_text = []
    # decrease_words = ['giảm', 'hạ']
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            line = line.replace(" trăm ", "").replace("\n", "")
            line = line.replace(" chếch ", " check ")
            # line.replace(" giờ dưới ", " giờ rưỡi ")
            line = line.replace("1 chút", "một chút")
            line = line.replace("giờ dưới", "giờ rưỡi")
            line = line.replace("về nhạc", "về nhà")
            line = line.replace("máy tính đèn bàn", "máy tính để bàn")
            line = line.replace("nưng", "nâng") 
            line = line.replace("lưng", "nâng")
            line = replace_number(line)
            line = re.sub(r'(\d+)\s+dưới', r'\1 rưỡi', line)
            # Split the line based on space and take the last part as the extracted text
            # decreasing = False
            parts = line.strip().split(" ")
            text = " ".join(parts[1:])
            
            # matches = re.findall(pattern, text)
            # for word in decrease_words:
            #     if word in text:
            #         decreasing = True
            # if not decreasing and matches:
            #     text = text.replace("xuống", "số")
            extracted_text.append(text)
    
    # Open the output file for writing
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # Write the extracted text to the output file
        for text in extracted_text:
            output_file.write(text + "\n")

    print("Extraction and saving complete.")

