import subprocess

subprocess.run(["python", "data-process/in_label_convert.py"])
subprocess.run(["python", "data-process/out_convert.py"])
subprocess.run(["python", "data-process/split.py"]) 
subprocess.run(["python", "data-process/vocab_process.py"])