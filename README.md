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
    2. Prepare your dataset
        - To put your dataset in correct format and process it run: 
            ```
            bash prepare_train_data.sh [Path to wav data directory] [Path to jsonline train file]
            ```
        - Example :
            ```cmd
            bash prepare_train_data.sh SLU_data/train_data/Train/  SLU_data/train.jsonl
            ```
        - The processed data will be store in `txt_data/process_train.txt`
    3. Run
        - Start training from scratch:
            ```cmd
            python train.py -c config.toml
            ```
        - Change the number of workers, epochs, batch size, vv in `config.toml`
        
- To train LM model:
    1. Go to kenlm folder:
        ```
        cd kenlm
        ```
    2. 

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
