import os
import librosa
import numpy as np
from typing import Dict
from datasets import load_dataset

def convert_audio(example: Dict[str, str]) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Converts the audio file to a numpy array and adds it to the example dictionary.
    Args:
        example (Dict[str, str]): The example dictionary.
    Returns:
        example (Dict[str, Dict[str, np.ndarray, sampling_rate]]): The example dictionary with the audio array and sampling added.
    """
   
    wav_file_path = example['file']
    try:
        waveform, sampling_rate = librosa.load(wav_file_path, sr=16000, mono=True)
    except:
        print(f"Error loading file: {wav_file_path}")
        waveform = np.zeros(1601) # will be filtered out later
        sampling_rate = 16000
    # ensure waveform is of type float32
    if waveform.dtype != "float32":
        waveform = waveform.astype(np.float32)

    example['audio'] = dict()
    example['audio']['file'] = wav_file_path
    example['audio']['array'] = waveform
    example['audio']['sampling_rate'] = sampling_rate
    return example


def construct_dataset(csv_path: str, save_path: str) -> None:
    """
    Converts the audio files in the dataset to numpy arrays and saves the dataset to disk.
    Args:
        csv_path (str): The path to the CSV file.
        save_path (str): The path to save the dataset to.
    Returns:
        None
    """
    dataset_name = csv_path.split("/")[-1].split("_")[0]
    save_path = save_path + dataset_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    raw = load_dataset('csv', data_files=csv_path, cache_dir=save_path)
    dataset_processed = raw.map(convert_audio, cache_file_names={"train": f"{save_path}/cache_map.pkl"})
    dataset_processed.save_to_disk(save_path)

    # filter out examples where the audio file could not be loaded
    dataset_processed = dataset_processed.filter(lambda example: len(example['audio']['array']) != 1601, cache_file_names={"train": f"{save_path}/cache_filter.pkl"})


    dataset_processed.save_to_disk(save_path)

if __name__ == '__main__':
    csv_path = ""
    save_path = ""
    construct_dataset(csv_path, save_path)
