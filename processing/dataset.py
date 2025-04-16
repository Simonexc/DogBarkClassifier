import random

import librosa
import numpy as np
from torch.utils.data import Dataset
import torch


class AudioDataset(Dataset):
    def __init__(
        self,
        file_paths,
        labels,
        sample_rate=16000,
        num_samples=16000,
        bark_file_paths=None,
        transform=None,
        p_overlap=0.0,
        is_train=False,
        preload=False,
    ):
        self.file_paths = file_paths
        self.labels = labels
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.bark_file_paths = bark_file_paths
        self.transform = transform
        self.p_overlap = p_overlap
        self.is_train = is_train

        self._data = []
        if preload:
            for file_path in file_paths:
                waveform = torch.tensor(self._load_audio(file_path), dtype=torch.float32)
                self._data.append(waveform)

    def __len__(self):
        return len(self.file_paths)

    def _load_audio(self, file_path):
        waveform, _ = librosa.load(file_path, sr=self.sample_rate, mono=True)
        return waveform

    @staticmethod
    def _combine_waveforms(waveform1, waveform2, gain1):
        if gain1 < 0.0 or gain1 > 1.0:
            raise ValueError("Gain must be between 0.0 and 1.0")
        if waveform1.shape[0] != waveform2.shape[0]:
            raise ValueError("Waveforms must be the same length")
        if waveform1.dtype != waveform2.dtype and waveform1.dtype != torch.float32:
            raise ValueError("Waveforms must be of the same type and float32")

        gain2 = 1.0 - gain1
        mixed_waveform = gain1 * waveform1 + gain2 * waveform2

        max_amp = mixed_waveform.abs().max()
        if max_amp > 1.0:
            mixed_waveform /= max_amp

        return mixed_waveform

    def __getitem__(self, idx):
        assert self._data
        label = self.labels[idx]
        waveform = self._data[idx]

        # --- Training Only Augmentations ---
        if self.is_train:
            # 1. Overlap Augmentation
            if label == 0 and self.bark_file_paths and random.random() < self.p_overlap:
                bark_file_idx = self.file_paths.index(random.choice(self.bark_file_paths))
                bark_waveform = self._data[bark_file_idx]

                bark_gain = random.uniform(0.4, 0.9)

                waveform = self._combine_waveforms(bark_waveform, waveform, bark_gain)

                label = 1  # Label becomes bark

        label_tensor = torch.tensor(label, dtype=torch.float32) # Use float for BCEWithLogitsLoss
        return waveform, label_tensor
