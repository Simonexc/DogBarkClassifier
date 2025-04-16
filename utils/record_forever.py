import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile

def record_audio_interruptible(filename="output2.wav", fs=44100):
    """
    Records audio from the microphone until a keyboard interrupt (Ctrl+C) is received,
    and saves it as a WAV file.  Prints the name of the recording device.

    Args:
        filename (str, optional): The name of the WAV file to save. Defaults to "output.wav".
        fs (int, optional): The sampling rate (samples per second). Defaults to 44100 Hz.
    """
    recording = []
    try:
        default_device_index = sd.default.device[0] # Get default input device index
        device_info = sd.query_devices(default_device_index, 'input') # Get info about the default input device
        device_name = device_info['name'] # Extract device name

        print(f"Using audio device: {device_name}") # Print the device name
        print("Recording... Press Ctrl+C to stop.")
        stream = sd.InputStream(samplerate=fs, channels=1, dtype='float32')
        stream.start()

        while True:
            try:
                data, overflowed = stream.read(1024) # Read data in chunks
                if overflowed:
                    print("Audio buffer overflowed!") # Optional: Handle overflow if it happens
                recording.append(data)
            except KeyboardInterrupt:
                print("\nStopping recording...")
                break # Exit the loop on Ctrl+C

        stream.stop()
        stream.close()
        print("Recording finished.")

        # Concatenate all recorded chunks
        recording_array = np.concatenate(recording, axis=0)

        # Save as WAV file
        wavfile.write(filename, fs, recording_array)
        print(f"Audio saved to {filename}")

    except sd.PortAudioError as e:
        print(f"Error during recording: {e}")
        print("Please make sure you have a microphone connected and that sounddevice is properly configured.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    record_audio_interruptible()
    # To record to a different filename, you can call the function with an argument:
    # record_audio_interruptible(filename="my_recording.wav")