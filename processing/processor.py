import random

import torchaudio.transforms as T


class AudioProcessor:
    def __init__(
        self,
        transform=None,
        device="cpu",
        sample_rate=16000,
        n_fft=1024,
        hop_length=None,
        n_mels=64,
        power=2.0,
        top_db=80,
        p_time_mask=0.5,
        p_freq_mask=0.5,
        max_mask_time=16,
        max_mask_freq=8,
    ):
        self.transform = transform
        self._device = device

        # Initialize Mel Spectrogram transform
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            power=power,
        ).to(device=self._device)
        self.amplitude_to_db = T.AmplitudeToDB(stype="power", top_db=top_db).to(
            device=self._device)  # Convert power spec to dB

        # Spectrogram Augmentation parameters
        self.p_time_mask = p_time_mask
        self.p_freq_mask = p_freq_mask
        self.max_mask_time = max_mask_time
        self.max_mask_freq = max_mask_freq

    def time_masking(self, spec):
        """Apply time masking to a spectrogram.
        Args:
            spec (torch.Tensor): Spectrogram tensor (B, C, F, T).
            max_mask_time (int): Maximum number of time frames to mask.
            p (float): Probability of applying time masking.
        """
        if random.random() < self.p_time_mask:
            B, F, T = spec.shape
            mask_len = random.randint(1, self.max_mask_time)
            mask_start = random.randint(0, T - mask_len)
            spec[:, :, mask_start:mask_start + mask_len] = 0  # or torch.mean(spec) for mean masking
        return spec

    def frequency_masking(self, spec):
        """Apply frequency masking to a spectrogram.
        Args:
            spec (torch.Tensor): Spectrogram tensor (B, C, F, T).
            max_mask_freq (int): Maximum number of frequency bins to mask.
            p (float): Probability of applying frequency masking.
        """
        if random.random() < self.p_freq_mask:
            B, F, T = spec.shape
            mask_len = random.randint(1, self.max_mask_freq)
            mask_start = random.randint(0, F - mask_len)
            spec[:, mask_start:mask_start + mask_len, :] = 0  # or torch.mean(spec) for mean masking
        return spec

    def __call__(self, waveforms, augment=False):
        waveforms = waveforms.to(device=self._device)
        if self.transform:
            waveforms = self.transform(samples=waveforms.unsqueeze(1)).squeeze(1)
        return waveforms
        # --- Compute Log-Mel Spectrogram ---
        mel_spec = self.mel_spectrogram(waveforms)
        log_mel_spec = self.amplitude_to_db(mel_spec)

        # --- Normalize Spectrogram (per-instance) ---
        mean = log_mel_spec.mean()
        std = log_mel_spec.std()
        # Add epsilon to prevent division by zero
        log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)

        if augment:
            log_mel_spec = self.time_masking(log_mel_spec)
            log_mel_spec = self.frequency_masking(log_mel_spec)

        # Add channel dimension (required by Conv2d) -> [batch, 1, n_mels, time_steps]
        return log_mel_spec.unsqueeze(1)