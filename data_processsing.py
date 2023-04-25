import os
import csv
from tqdm import tqdm

def write_csv(directory: str, output_path: str,) -> int:
    """
    Reads all the .wav files in the directory and its subdirectories and looks for a corresponding .txt file with the same name as the .wav file (except with the .txt extension instead) and reads its contents.
    The contents of the .txt file are used as the transcript for the corresponding .wav file. The function writes the contents of the mapping_dict to a CSV file.

    Args:
        directory (str): The directory to search for .wav and .txt files.

    Returns:
        count (int): The number of .wav files that did not have a corresponding .txt file.
    """
    # Create the output directory if it does not exist
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    mapping_dict = {}
    count = 0
    
    # Count the total number of files
    total_files = sum([len(files) for _, _, files in os.walk(directory)])
    
    # Use tqdm to create a progress bar
    with tqdm(total=total_files) as pbar:
        for root, dirs, files in os.walk(directory):
            for filename in files:
                if filename.endswith(".wav"):
                    wav_filepath = os.path.join(root, filename)
                    transcript_filepath = os.path.join(root, filename[:-4] + ".txt")
                    if os.path.exists(transcript_filepath):
                        with open(transcript_filepath, 'r') as transcript_file:
                            transcript = transcript_file.read().strip()
                        mapping_dict[wav_filepath] = transcript
                    else:
                        print('transcript does not exist')
                        count += 1
                # Update the progress bar
                pbar.update(1)
                
    # Write the dictionary to a CSV file
    dataset_name = directory.split("/")[-1]
    csv_filename = os.path.join(output_path, dataset_name + "_metadata.csv")
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['file', 'transcript']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for wav_filepath, transcript in mapping_dict.items():
            writer.writerow({'file': wav_filepath, 'transcript': transcript})
    return count

if __name__ == '__main__':
    directory = "/scratch/alpine/roso8920/Corpora/CuKidsSpeech/train/train-part5-ogi-1-5"
    output_path = "/projects/zhwa3087/ASR/data/"
    count = write_csv(directory, output_path)
    print(f"Number of files without a corresponding transcript: {count}")