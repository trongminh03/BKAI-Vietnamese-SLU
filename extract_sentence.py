import argparse
import re
# # Define the input and output file paths
# input_file_path = "transcript_new_1.txt"
# output_file_path = "input.txt"

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-i', '--input_path', type=str, required = True,help='Path to transcript.txt file')
    args.add_argument('-o', '--output_path', type=str, required = True,help='Path to sentence.txt file')
    args = args.parse_args()
    input_file_path = args.input_path 
    output_file_path = args.output_path


    # Create a list to store the extracted text
    extracted_text = []
    decrease_words = ['giảm', 'hạ']

    pattern = r'xuống\s+(\d+)'

    # Open the input file for reading
    with open(input_file_path, "r", encoding="utf-8") as input_file:
        for line in input_file:
            if "khách tớ nhà" in line: 
                line = line.replace("khách tớ nhà", "khách tới nhà")

            # Split the line based on space and take the last part as the extracted text
            decreasing = False
            parts = line.strip().split(" ")
            text = " ".join(parts[1:])
            matches = re.findall(pattern, text)
            for word in decrease_words:
                if word in text:
                    decreasing = True
            if not decreasing and matches:
                text = text.replace("xuống", "số")
            extracted_text.append(text)
    
    # Open the output file for writing
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # Write the extracted text to the output file
        for text in extracted_text:
            output_file.write(text + "\n")

    print("Extraction and saving complete.")