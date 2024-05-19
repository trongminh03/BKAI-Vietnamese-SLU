# SoICT Hackathon 2023: Vietnamese Spoken Language Understanding Challenge
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
        - The generated data and denoised data will be stored in the same folder as the origin train data folder.
        - Download our generated data and denoised data by running:
        ```
        bash download_data.sh [Path to your origin train data folder]
        ```
        - Download [newaug_newdenoise.jsonl](https://drive.google.com/file/d/1iL-P17ULBWN58Up-AjeArjoJWhalTLZa/view?usp=drive_link) and [newaug_newdenoise2.jsonl](https://drive.google.com/file/d/12H05uTpWwqv632o6hM-Qy_wBJYsBnbPL/view?usp=drive_link):
        - You can use `gdown` to download the file.
            - `newaug_newdenoise.jsonl` file: 
                ```
                gdown 1iL-P17ULBWN58Up-AjeArjoJWhalTLZa
                ```
            - `newaug_newdenoise2.jsonl` file:
                ```
                gdown 12H05uTpWwqv632o6hM-Qy_wBJYsBnbPL
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
            ├── 64b420ff8e16f5f56e45a2b7cleantrain.wav
            ├── 64b420118e16f55e6945a2a5cleantrain.wav
            ├── ...
            ├── 64a18594883d155a21f23f651-17367-A-102.wav
            ├── ...
        
        ```
        - This results in 2 differents dataset with 2 different jsonl files which then produces 2 different checkpoints. We will ensemble it in Inference part.
        - To augment and denoise data, we use [this](https://github.com/karolpiczak/ESC-50) and [this](https://github.com/facebookresearch/denoiser) repo.
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
    - [First ASR model](https://drive.google.com/drive/folders/1eXHr0Q4RvhQTIghBY3gL3Lm2CoH3zbgf?usp=drive_link)
        ```
        gdown --folder 1eXHr0Q4RvhQTIghBY3gL3Lm2CoH3zbgf
        ``` 
    - [Second ASR model](https://drive.google.com/drive/folders/1SE3kA912bTMZohwb04iZ6dn_RHl94yoL?usp=sharing)
        ```
        gdown --folder 1SE3kA912bTMZohwb04iZ6dn_RHl94yoL
        ```
    - [LM model](https://drive.google.com/file/d/1XdQ0O-zyKEE8Z_glH9NZuj-Sj8v3jhkg/view?usp=sharing)
        ```
        gdown 1XdQ0O-zyKEE8Z_glH9NZuj-Sj8v3jhkg
        ```
### Inference
- First, go to SLU-ASR folder then run
```
bash inference_ensemble.sh  [Path to your wav test file lists] [Path to model list to ensemble] [Path to LM model] [save_path]
```
    
- Example:
```
bash inference_ensemble.sh /data/public_test/ model_list.txt add_gen_3gram.binary process_trans_file.txt
```
- Set the path to your models you want to ensemble in `SLU-ASR/model_list.txt`. To reproduce the result, keep the weight which ís a float number after the path and set the path to your actual path.

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
- Edit ``` model_list.txt ``` file which includes multiple model names and weights that you want to ensemble. 
- Then run:
```
bash inference_ensemble.sh 
```
- To reproduce the result, keep the weight and set the path to your actual path.

### Technical report
https://www.overleaf.com/project/650b05c5c9e4867104e1aedd

### Presentation slide: 
https://docs.google.com/presentation/d/177N4RaGl3ClBxHGwqgNPutTGzbPfNe-l/edit#slide=id.p1
