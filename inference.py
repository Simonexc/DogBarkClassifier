from processing.processor import AudioProcessor
from models.cnn import SpectrogramCNN

import torch
import numpy as np
from transformers import AutoFeatureExtractor


class ModelInference:
    def __init__(self, model_path: str, threshold: float):
        self.threshold = threshold
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = SpectrogramCNN(num_classes=1).to(self._device)
        self._model.load_state_dict(torch.load(model_path, map_location=self._device))
        self._processor = AudioProcessor(device=self._device)
        self._wav2vec2_processor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")

    def inference(self, waveform: np.ndarray, for_wav2vec2=False) -> tuple[bool, float]:
        assert waveform.ndim == 1, "Waveform must be a 1D numpy array"

        # Preprocess the audio data
        preprocessed_data = self._processor(torch.tensor(np.expand_dims(waveform, axis=0), dtype=torch.float32))
        if for_wav2vec2:
            preprocessed_data = self._wav2vec2_processor(
                preprocessed_data, sampling_rate=16000, return_tensors="pt"
            )["input_values"].squeeze(0).to(self._device)

        # Perform inference
        with torch.no_grad():
            if for_wav2vec2:
                output = torch.sigmoid(self._model(preprocessed_data).logits).cpu().detach().squeeze(1)[0].item()
            else:
                output = torch.sigmoid(self._model(preprocessed_data)).cpu().detach().squeeze(1)[0].item()

        return output > self.threshold, output
