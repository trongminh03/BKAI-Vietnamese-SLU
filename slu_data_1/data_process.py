import subprocess
import argparse 

if __name__ == '__main__':
    args = argparse.ArgumentParser()
    args.add_argument('-j', '--train_file', type=str, required = True,help='Path to train.jsonl file')
    args = args.parse_args()
    train_file_path = args.train_file

    subprocess.run(["python", "slu_data_1/data-process/pre_process.py", "-j", train_file_path])
    subprocess.run(["python", "slu_data_1/data-process/in_label_convert.py"])
    subprocess.run(["python", "slu_data_1/data-process/out_convert.py"])
    subprocess.run(["python", "slu_data_1/data-process/split.py"]) 
    subprocess.run(["python", "slu_data_1/data-process/vocab_process.py"])