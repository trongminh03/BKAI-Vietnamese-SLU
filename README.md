# BKAI Vietnamese SLU

## Overview
- This repo contain 2 module:
     - Speech to Text module.
     - Text Intent and Slot Filling module.
- Output of `Speech to Text module` will be feed in `Text Intent and Slot Filling module` to get the final prediction.


- P/s: Your device must have Docker and GPU. You should run it on Docker.

## Docker
- Build image by this command:
```
DOCKER_BUILDKIT=1 docker build -t slu .
```

- Then run the image by this command:
```
docker run -it --name docker_slu --gpus all --rm slu
```

## Speech to Text module
### Training
- We combine the ASR model and a LM model for higher Speech to Text accuracy.
- To train the ASR model:
    1. Go to SLU-ASR folder:
        ```
        cd SLU-ASR
        ```
    2. Generate data (optional):
        - You can use our generated data by placing [this](https://drive.google.com/drive/u/1/folders/1cvYEmfY8UNJX2bXlD4cbk-8icvZCwN6k?usp=sharing&fbclid=IwAR2kjwPt1pAwNR0fsEUtkY0A73xpKMf1ZreuyQ5ET-KvX7xbchnedZ426c0&pli=1) in the same folder as origin data and use the new [train_and_aug.jsonl file](https://drive.google.com/file/d/1Zkuuc4P74sVI1wpHMUw5PlBzpVdX95Rv/view?usp=sharing) 
    3. Prepare your dataset
        - To put your dataset in correct format and process it run: 
            ```
            bash prepare_train_data.sh [Path to wav data directory] [Path to jsonline train file]
            ```
        - Example :
            ```cmd
            bash prepare_train_data.sh SLU_data/train_data/Train/  SLU_data/train.jsonl
            ```
        - The processed data will be store in `txt_data/process_train.txt`
    4. Run
        - Start training from scratch:
            ```cmd
            python train.py -c config.toml
            ```
        - Change the number of workers, epochs, batch size, vv in `config.toml`

- To train LM model: 
    1. Go to the root folder of the repo. Train the LM model by:
        ```
        bash train_lm.sh [Path to origin jsonline train file]
        ```
        - Example:
        ```
        bash train_lm.sh data/train.jsonl

        Note: Dont use the generated `train_and_aug.jsonl file` here.
        ```
        - The LM model will be stored in `your_ngram.binary`
- You can use our ASR model and a LM model checkpoints through this link:
    - ASR model : https://drive.google.com/drive/folders/1PUOZtKDbpebvtsKV-9Xo8W68yQapW3xU?usp=sharing
    - LM model : https://drive.google.com/file/d/1XdQ0O-zyKEE8Z_glH9NZuj-Sj8v3jhkg/view?usp=drive_link

### Inference
```
bash inference.sh [Path to your wav test file lists] [Path to model.tar] [Path to LM model]
```

    
- Example:
```
bash inference.sh data/public_test/ saved/ASR/checkpoints/best_model.tar your_3gram.binary
```

- Then the final transcript be in `process_trans_file.txt`

## Text Intent and Slot Filling moudule
### Training 
1. Prepare your data
    - Run the following command to pre-process train.jsonl and prepare data for training:
        ```
        python3 slu_data_1/data_process.py -j [Path to train.jsonl file]
        ```
    - Example :
        ```cmd
        python3 slu_data_1/data_process.py -j data/train.jsonl
        ```
    The processed data will be stored in `slu_data_1/syllable-level`
2. Run 
    - Run the following bash file to start training: 
        ```cmd
        ./run_jointIDSF_PhoBERTencoder.sh
        ```

### Inference
- Here is model checkpoints link in case you want to make inference without training the models from scratch: https://drive.google.com/drive/folders/1tZ-508QnyfQEh1_xzkoVjwkSkW38I04f?usp=drive_link
- Then run this command for inference:
```
 bash inference_JointIDSF.sh [Path to output transcript of ASR module] [Path to model checkpoints]
```

- Example:
```
bash inference_JointIDSF.sh SLU-ASR/process_trans_file.txt JointIDSF_PhoBERTencoder_SLU/4e5/0.15/100
```

- Then the final output will be automatically zipped as `Submission.zip`.
