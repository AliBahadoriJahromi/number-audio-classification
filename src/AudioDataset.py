import torch
import torchaudio
from torch.utils.data import Dataset
from torchaudio.transforms import MelSpectrogram, Resample
import os

class AudioDataset(Dataset):
    def __init__(self, root_dir, target_length=20000):
        self.root_dir = root_dir
        self.target_length = target_length  # target length in samples
        self.files = []

        # Collect all .wav files from subfolders
        for folder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, folder)
            if os.path.isdir(folder_path):
                for file in os.listdir(folder_path):
                    if file.endswith(".wav"):
                        self.files.append(os.path.join(folder_path, file))

    def __getitem__(self, idx):
        file_path = self.files[idx]

        # Load the audio file
        waveform, sample_rate = torchaudio.load(file_path)

        resample_transform = Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resample_transform(waveform)

        mel_spec_transform = MelSpectrogram(sample_rate=16000, n_mels=64)
        mel_spec = mel_spec_transform(waveform)

         # Padding or trimming to the target length
        if mel_spec.size(2) < self.target_length:
            # Padding with zeros if the length is smaller
            pad_length = self.target_length - mel_spec.size(2)
            mel_spec = torch.nn.functional.pad(mel_spec, (0, pad_length))
        elif mel_spec.size(2) > self.target_length:
            # Trimming if the length is larger
            mel_spec = mel_spec[:, :, :self.target_length]

        # Extract label from the file name (assuming it's the first digit)
        label = int(file_path.split('/')[-1].split('_')[0])

        return mel_spec, label
    
    def __len__(self):
        return len(self.files)
