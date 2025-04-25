from inference import ModelInference
from processing.dataset import AudioDataset
import glob
import numpy as np

path = ""
infer = ModelInference(path, threshold=0.759)

dataset = AudioDataset(
    file_paths=[],
    labels=[],
)

print("Process NO BARK files")
files = glob.glob("data/test/no_bark/*.wav")
output = []
for file in files:
    audio = dataset._load_audio(file)
    a, b = infer.inference(audio)
    output.append(a)
    if a:
        print(file, b)

print("accuracy:", 1 - np.array(output).sum() / len(output))
print("----------------------------")
print("Process BARK files")
files = glob.glob("data/test/bark/*.wav")
output2 = []
for file in files:
    audio = dataset._load_audio(file)
    a, b = infer.inference(audio)
    output2.append(a)
    if not a:
        print(file, b)

print("accuracy:", np.array(output2).sum() / len(output2))
