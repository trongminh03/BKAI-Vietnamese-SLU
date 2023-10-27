# BKAI Vietnamese SLU
## Overview
- This repository contains 2 modules:
     - Speech to Text module.
     - Text Intent and Slot Filling module.
- Output of `Speech to Text module` will be fed into `Text Intent and Slot Filling module` to get the final prediction.
- P/s: Your device must have Docker and GPU. You should run it on Docker.
- You first need to clone the repository including its submodules:
    ```
    git clone --recurse-submodules https://github.com/trongminh03/BKAI-Vietnamese-SLU.git
    ```
## Docker
- Build image using this command:
```
DOCKER_BUILDKIT=1 docker build -t slu .
```
- Then run the image using this command:
```
docker run -it --name docker_slu --gpus all --rm slu
```
## Speech to Text module
### Training
- We combine the ASR model with a LM model for higher Speech to Text accuracy.
- To train the ASR model:
    1. Go to SLU-ASR folder:
        ```
        cd SLU-ASR
        ```
    2. Generate data and Denoise data(optional):
        - Be careful if you mount your data folder inside docker. 
        - The generated data and denoised data will be stored in the same as the origin train data folder.
        - Download our generated data and denoised data by running:
        ```
        bash download_data.sh [Path to your origin train data folder]
        ```
        - Download [train_and_aug.jsonl file](https://drive.google.com/file/d/1Zkuuc4P74sVI1wpHMUw5PlBzpVdX95Rv/view?usp=sharing) and [train_and_denoise.jsonl file](https://drive.google.com/file/d/1229wpKuDhiLa8CQkwk-PI5T920c1zKjP/view?usp=sharing)
        - You can use `gdown` to download the file.
            - `train_and_aug.jsonl` file: 
                ```
                gdown 1Zkuuc4P74sVI1wpHMUw5PlBzpVdX95Rv
                ```
            - `train_and_denoise.jsonl` file:
                ```
                gdown 1229wpKuDhiLa8CQkwk-PI5T920c1zKjP
                ```
        - Or generate wav file by yourself using this command and use the provided `train_and_aug.jsonl` file

                python3 augmented_data.py \
                --input_folder [Path to wav data directory] \
                --input_jsonlfile [Path to jsonline train file] 

        - Example:
                
                python3 augmented_data.py \
                --input_folder /data/train_data/Train \
                --input_jsonlfile /data/train.jsonl
        - denoise data by yourself using this command and use the provided `train_and_denoise.jsonl` file
        ```
        cd CleanUNet
        python3 denoise_simple.py -c configs/DNS-large-high.json \
        --ckpt_pat DNS-large-high/checkpoint/pretrained.pkl \
        -i [Path to wav data directory] \
        -o [output folder]
        ```
        - Example:
        ```
        python3 denoise_simple.py -c configs/DNS-large-high.json \
        --ckpt_pat DNS-large-high/checkpoint/pretrained.pkl \
        -i /data/train_data/Train/ \
        -o /data/train_data/Train/
        ```
        - After that the data folders should look like :
         ```bash
        data
        └──train_data
            ├── 64b420ff8e16f5f56e45a2b7.wav
            ├── 64b420118e16f55e6945a2a5.wav
            ├── ...
            ├── 648f0583bd5f017127bbb7cbBandStopFilter.wav
            ├── ...
            ├── 64b420ff8e16f5f56e45a2b7_denoised.wav
            ├── 64b420118e16f55e6945a2a5_denoised.wav
            ├── ...
        
        ```
        - This results in 2 differents dataset with 2 different jsonl files which then produces 2 different checkpoints. We will ensemble it in Inference part.
    4. Prepare your dataset
        - To put your dataset in correct format and process it run: 
            ```
            bash prepare_train_data.sh [Path to wav data directory] [Path to jsonline train file]
            ```
        - Example :
            ```cmd
            bash prepare_train_data.sh /data/train_data/Train/ /data/train.jsonl
            ```
        - The processed data will be store in `txt_data/process_train.txt`
    5. Run
        - Start training from scratch:
            ```cmd
            python3 train.py -c config.toml
            ```
        - Change the number of workers, epochs, batch size, vv in `config.toml`
        - The model will be stored in `saved/ASR/checkpoints/best_model.tar`. If you train more than one model, the newer checkpoints will replace the older. 
- To train LM model: 
    1. Go to the root folder of the repository. Train the LM model by:
        ```
        bash train_lm.sh [Path to origin jsonline train file]
        ```
        - Example:
        ```
        bash train_lm.sh /data/train.jsonl
        ```
        **Note:** Dont use the generated `train_and_aug.jsonl file` here.
        - The LM model will be stored in `your_ngram.binary`
- You can use our ASR model checkpoints and a LM model checkpoints through this link:
    - [ASR model trained on orginal and generated data](https://drive.google.com/file/d/1eUL7IgpPcofJeuLjf231cvBo2BSzRHJD/view?usp=sharing)
        ```
        gdown 1eUL7IgpPcofJeuLjf231cvBo2BSzRHJD
        ```
    - [ASR model trained on orginal and denoised data](https://drive.google.com/drive/folders/1r5Huc3dViw1XVuZbFeWJYKH_yFcydPue?usp=drive_link)
        ```
        gdown --folder 1r5Huc3dViw1XVuZbFeWJYKH_yFcydPue
        ```
    - [LM model](https://drive.google.com/file/d/1XdQ0O-zyKEE8Z_glH9NZuj-Sj8v3jhkg/view?usp=drive_link)
        ```
        gdown 1XdQ0O-zyKEE8Z_glH9NZuj-Sj8v3jhkg
        ```
### Inference
- First, go to SLU-ASR folder then run
```
bash inference.sh [Path to your wav test file lists] [Path to model.tar] [Path to LM model] [save name]
```
    
- Example:
```
bash inference.sh /data/public_test/ best_model.tar your_3gram.binary process_trans_file.txt
```
- To ensemble two ASR prediction run:
```
python3 ASR_ensemble.py -main [First txt ASR output] -sup [Second txt ASR output] -lm [Path to lm model]
```
- Output will be stored in `ensemble_trans.txt`
- To reproduce our results, you set the First txt ASR output to the output of ASR model trained on **orginal and denoised data** and Second txt ASR output to  the output of ASR model trained on **orginal and generated data**.
- For the final post-process run:
```
python3 final_process.py -j [Path to jsonline train file] -lm [Path to LM model] -p [Path to previous transcript txt file]
```
Example:
```
python3 final_process.py -j /data/train.jsonl -lm your_3gram.binary -p ensemble_trans.txt
```
- Final output of ASR will be stored in `final_trans.txt`

## Text Intent and Slot Filling module
### Training 
1. Prepare your data
    - Run the following command to pre-process default `train.jsonl` file and prepare data for training:
        ```
        python3 slu_data_1/data_process.py -j [Path to train.jsonl file]
        ```
    - Adding the `--augment_data` flag will include augmented data in the processing
    - Example :
        ```cmd
        python3 slu_data_1/data_process.py -j /data/train.jsonl --augment_data
        ```
    The processed data will be stored in `slu_data_1/syllable-level`
2. Run 
    - Run the following bash file to start training: 
        ```cmd
        ./run_jointIDSF_PhoBERTencoder.sh
        ```
### Inference
- Here is [model checkpoints link](https://drive.google.com/drive/folders/1tZ-508QnyfQEh1_xzkoVjwkSkW38I04f?usp=drive_link) in case you want to make inference without training the models from scratch
```
gdown --folder 1eCSHBAp1uD31dsgup5as-gQd2K_2rHuN
```
- Then run this command for inference:
```
 bash inference_JointIDSF.sh [Path to output transcript of ASR module] [Path to model checkpoints] [saved name]
```
- Example:
```
bash inference_JointIDSF.sh SLU-ASR/final_trans.txt JointIDSF_PhoBERTencoder_SLU/4e5/0.15/100 predictions.jsonl 
```

### Ensemble model inference
- For higher accuracy we apply confidence score ensemble method.
- Edit ``` model_list.txt ``` file which includes multiple model names that you want to ensemble. 
- Then run:
```
bash inference_ensemble.sh 
```
