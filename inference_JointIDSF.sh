transfile=$1
model_dir=$2

python3 extract_sentence.py -i $transfile -o sentence.txt

python3 predict.py --input_file sentence.txt --output_file output.txt --model_dir $model_dir

python3 post_process.py -i output.txt -t $transfile -o predictions.jsonl

echo "DONE"