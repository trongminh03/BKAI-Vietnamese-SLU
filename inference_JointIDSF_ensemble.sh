transfile=$1
model_dir=$2
model_dir_intent=$3

python3 extract_sentence.py -i $transfile -o sentence.txt

python3 predict.py --input_file sentence.txt --output_file output.txt --model_dir $model_dir

python3 predict.py --input_file sentence.txt --output_file output_intent.txt --model_dir $model_dir_intent

python3 post_process.py -i output.txt -t $transfile -o predictions.jsonl  

python3 post_process.py -i output_intent.txt -t $transfile -o intent.jsonl

python3 ensemble.py

echo "DONE"