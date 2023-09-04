import torch_audiomentations as ta
import soundfile as sf
import torch
import os
import argparse

def read_file(file_path):
    lines = []
    dict_files = {}
    with open(file_path, 'r') as f:
        lines = f.readlines()
    for i in range(len(lines)):
        lines[i] = lines[i].strip()
        first_colon = lines[i].find(':')
        first_comma = lines[i].find(',')
        id = lines[i][first_colon + 1: first_comma]
        id = id.replace('"', '')
        id = id.strip()
        dict_files[id + '.wav'] = lines[i]
    return dict_files

def get_dict_transforms(sample_rate = 16000):
    transforms = {
        'ColouredNoise': ta.AddColoredNoise(p = 1),
        # 'ImpulseResponse': ta.ApplyImpulseResponse(p = 0.5),
        # 'BackgroundNoise': ta.AddBackgroundNoise(p = 1),
        # 'PolarityInversion': ta.PolarityInversion(p = 1),
        'BandPassFilter': ta.BandPassFilter(p = 1),
        'BandStopFilter': ta.BandStopFilter(p = 1),
        'Gain': ta.Gain(min_gain_in_db=-8, max_gain_in_db=8, p = 1),
        'LowPassFilter': ta.LowPassFilter(min_cutoff_freq=1000, max_cutoff_freq=2000),
        'PitchShift': ta.PitchShift(p = 1, sample_rate= sample_rate, min_transpose_semitones=-8, max_transpose_semitones=8),
        'Normalization': ta.PeakNormalization(p = 1),
    }
    return transforms

def transform(ta_transform, wav_path):
    wav, sr = sf.read(wav_path)
    wav = wav.squeeze()
    if wav.shape[0] == 2:
        wav = wav[0]
    wav = torch.from_numpy(wav[None, None, :]).to(torch.float32)
    # print(wav.shape)
    wav = ta_transform(wav, sample_rate = sr).to(torch.float32)
    wav = torch.flatten(wav)
    return wav, sr

def save_file(wav, output_name):
    sound, sr = wav
    sf.write(output_name, sound, samplerate= sr)

def get_wav_files(folder_path):
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    return files

def export_list_augmented(output_name, list_augmented):
    with open(output_name, 'w') as f:
        for file in list_augmented:
            f.write(file)
            f.write('\n')

def augment_data(input_folder, output_folder, input_jsonlfile, output_jsonlfile):
    lines = read_file(input_jsonlfile)
    dict_transforms = get_dict_transforms()
    list_augmented = []
    wav_files = get_wav_files(input_folder)
    # print(wav_files)

    count = 0
    total_file = len(wav_files)
    for example_file in wav_files:
        count += 1
        print(f'{count}/{total_file}')
        if example_file in lines:
            for key, value in dict_transforms.items():
                wav_file_path = os.path.join(input_folder, example_file)
                wav = transform(value, wav_file_path)
                new_wav = example_file[:-4] + key
                list_augmented.append(lines[example_file].replace(example_file[:-4], new_wav))
                new_wav = new_wav + '.wav'
                save_file(wav, os.path.join(output_folder, new_wav))
            break
    print(len(list_augmented))
    export_list_augmented(output_jsonlfile, list_augmented)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_folder', required= True, help = 'input_wav_files_folder')
    parser.add_argument('--input_jsonlfile', required= True, help = 'input_jsonlfile')
    parser.add_argument('--output_folder', default = './', help = 'output_wav_files_folder')
    parser.add_argument('--output_jsonlfile', default = './augmented_data.jsonl', help = 'output_jsonlfile')
    args = parser.parse_args()
    print(args.input_jsonlfile)

    augment_data(input_folder= args.input_folder,
                 output_folder= args.output_folder,
                 input_jsonlfile= args.input_jsonlfile,
                 output_jsonlfile= args.output_jsonlfile)