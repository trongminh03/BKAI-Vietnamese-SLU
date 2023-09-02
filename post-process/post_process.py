def add_space_between_number_and_char(text):
    result = ''
    for char in text:
        if char.isdigit() and result and result[-1].isalpha():
            result += ' ' + char
        elif char.isalpha() and result and result[-1].isdigit():
            result += ' ' + char
        else:
            result += char
    return result

def process_file(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as file:
        input_text = file.read()
    
    output_text = add_space_between_number_and_char(input_text)
    
    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(output_text)

if __name__ == "__main__":
    input_file = "sentence.txt"
    output_file = "fixed_sentence.txt"
    
    process_file(input_file, output_file)
    print("Processing complete. Check the output in '{}'.".format(output_file))
