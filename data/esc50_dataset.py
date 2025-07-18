import os
import torch
from torch.utils.data import Dataset
import torchaudio
import pandas as pd

class ESC50Dataset(Dataset):
    def __init__(self, files_list_path, csv_path="./data/ESC-50-master/meta/esc50.csv", sample_rate=44100, n_mels=64, n_fft=1024, hop_length=512):
        """
        files_list_path: txt 文件路径，每行一个音频绝对路径
        csv_path: esc50.csv 元数据路径
        """
        with open(files_list_path, 'r') as f:
            self.audio_files = f.read().splitlines()

        self.sample_rate = sample_rate

        # 读取 esc50.csv 文件，建立 filename -> category 的映射
        df = pd.read_csv(csv_path)
        self.filename2label = dict(zip(df['filename'], df['category']))

        # 提取所有出现的类别名，建立类别名 -> 索引映射
        self.labels = sorted(list(set(self.filename2label.values())))
        self.class2idx = {label: idx for idx, label in enumerate(self.labels)}

        # torchaudio特征提取器：Mel + Log
        self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(top_db=80)

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = self.audio_files[idx]
        assert os.path.exists(audio_path), f"[Index {idx}] {audio_path} not found"

        filename = os.path.basename(audio_path)
        label_name = self.filename2label[filename]
        label_idx = self.class2idx[label_name]

        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=self.sample_rate)
            waveform = resampler(waveform)

        mel_spec = self.mel_spectrogram(waveform)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        return log_mel_spec.squeeze(0), torch.tensor(label_idx, dtype=torch.long)
