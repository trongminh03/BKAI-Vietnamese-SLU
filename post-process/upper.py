# Define the input and output file paths
input_file_path = 'output.txt'
output_file_path = 'uppercase.txt'

# Read the content of the input file
with open(input_file_path, 'r', encoding='utf-8') as input_file:
    content = input_file.read()

# Function to uppercase the first character within angle brackets
def upper_first_char(match):
    return match.group(1) + match.group(2)[0].upper() + match.group(2)[1:] + match.group(3)


# Use regular expressions to find and replace within angle brackets
import re
pattern = r'(<)([^>]*)(>)'
content = re.sub(pattern, upper_first_char, content)

# Write the modified content to the output file
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    output_file.write(content)

print("Text with first character within angle brackets converted to uppercase.")
