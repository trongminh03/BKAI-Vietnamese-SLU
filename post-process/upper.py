# Function to capitalize only the first character within angle brackets
def capitalize_first_within_angle_brackets(line):
    result = []
    inside_brackets = False
    for char in line:
        if char == '<':
            inside_brackets = True
            result.append(char)
        elif inside_brackets: 
            result.append(char.upper())
            inside_brackets = False
        else:
            result.append(char)
    return ''.join(result)

# Input and output file names
input_file = 'output.txt'
output_file = 'uppercase.txt'

# Read input file, process lines, and write to output file
with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
    for line in infile:
        processed_line = capitalize_first_within_angle_brackets(line)
        outfile.write(processed_line)

print("Conversion completed. Output saved to", output_file)
