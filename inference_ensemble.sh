transfile=$1

python3 extract_sentence.py -i $transfile -o sentence.txt

python3 predict_ensemble.py --input_file sentence.txt --output_file output.txt

python3 post_process.py -i output.txt -t $transfile -o predictions.jsonl

echo "DONE"