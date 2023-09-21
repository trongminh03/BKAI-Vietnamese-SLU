import subprocess
import argparse 

def run_command(command): 
    subprocess.run(command, check=True)

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-j', '--train_file', type=str,required = True,help='Path to train.jsonl file')
    args.add_argument("--augment_data", action="store_true", help="Enable my augmented data")
    args = args.parse_args()
    train_file_path = args.train_file

    commands = [
        ["python", "slu_data_1/data-process/pre_process.py", "-j", args.train_file],
        ["python", "slu_data_1/data-process/in_label_convert.py"],
        ["python", "slu_data_1/data-process/out_convert.py"],
        ["python", "slu_data_1/data-process/split.py"],
        ["python", "slu_data_1/data-process/vocab_process.py"]
    ]

    if args.augment_data:
        commands[1].append("--augment_data")
        commands[2].append("--augment_data")
        commands[3].append("--augment_data")

    for command in commands:
        run_command(command)