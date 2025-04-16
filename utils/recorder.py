import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile


def record_audio(filename="output.wav", duration=2, fs=44100):
    """
    Records audio from the microphone for a specified duration and saves it as a WAV file.

    Args:
        filename (str, optional): The name of the WAV file to save. Defaults to "output.wav".
        duration (int, optional): The recording duration in seconds. Defaults to 2 seconds.
        fs (int, optional): The sampling rate (samples per second). Defaults to 44100 Hz.
    """
    try:
        print("Recording...")
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1, dtype='float32') # Record audio
        sd.wait()  # Wait until recording is finished
        print("Recording finished.")

        # Save as WAV file
        wavfile.write(filename, fs, recording)
        print(f"Audio saved to {filename}")

    except sd.PortAudioError as e:
        print(f"Error during recording: {e}")
        print("Please make sure you have a microphone connected and that sounddevice is properly configured.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    record_audio() # Records 2 seconds and saves to output.wav
    # To record for a different duration or filename, you can call the function with arguments:
    # record_audio(filename="my_recording.wav", duration=5) # Records 5 seconds and saves to my_recording.wav
