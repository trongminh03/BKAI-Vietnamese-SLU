# BKAI Vietnamese SLU

### Training 
1. Prepare your data
    - Run the following command to pre-process train.jsonl and prepare data for training:
        ```
        python slu_data/data_process.py -j [Path to train.jsonl file]
        ```
    - Example :
        ```cmd
        python slu_data/data_process.py -j slu_data/train.jsonl
        ```
    The processed data will be stored in `slu_data/syllable-level`
2. Run 
    - Run the following bash file to start training: 
        ```cmd
        ./run_jointIDSF_PhoBERTencoder.sh
        ```

### Inference
- Here is model checkpoints link in case you want to make inference without training the models from scratch: https://drive.google.com/drive/folders/1tZ-508QnyfQEh1_xzkoVjwkSkW38I04f?usp=drive_link


- Run command below to extract sentence in transcript file: 
```
python extract_sentence.py -i [Path to transcript.txt file] 
                           -o [Output file name]
```
- Example: 
```
python extract_sentence.py -i SLU-ASR/transcript.txt 
                           -o sentence.txt
```

- Run following command to use model checkpoint:
```
python3 predict.py  --input_file <Path to sentence.txt file> \
                    --output_file <Output file name> \
                    --model_dir <Model checkpoints>
```
where the input file is a raw text file (one utterance per line).

- Example: 
```
python3 predict.py  --input_file sentence.txt \
                    --output_file output.txt \
                    --model_dir JointIDSF_PhoBERTencoder_SLU/4e5/0.15/100
```
- To get the final output, run this the post processing file:
```
python post_process.py -i [Path to output.txt file] \
                       -t [Path to transcript.txt file] \
                       -o [predictions.jsonl file] 
```
- Example: 
```
python post_process.py -i output.txt \
                       -t SLU-ASR/transcript.txt \
                       -o predictions.jsonl 
```
Then the final output will be automatically zipped as `Submission.zip`.
