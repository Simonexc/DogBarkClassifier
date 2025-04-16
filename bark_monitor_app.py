import flask
from flask import Flask, render_template, send_from_directory
import sounddevice as sd
import soundfile as sf
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T
from datetime import datetime, timedelta
import threading
import queue
import time
import os
from pathlib import Path
from train_bark_detector import SpectrogramCNN, SAMPLE_RATE, DURATION_SECONDS, NUM_SAMPLES, N_FFT, HOP_LENGTH, N_MELS

# --- Configuration ---
MODEL_PATH = Path("checkpoints_bark_detector_2d/best_model.pth")
AUDIO_SAVE_DIR = Path("static/barks")
AUDIO_SAVE_DIR.mkdir(parents=True, exist_ok=True) # Ensure directory exists

# Inference Params
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5 # Confidence threshold for classifying as bark (adjust based on testing)
MIC_BLOCK_DURATION_MS = 100 # How often the mic callback is called (ms)
MIC_DEVICE = None # None for default microphone

# --- Global Variables ---
audio_queue = queue.Queue()
bark_events = [] # List to store detected bark event info ({'start_time': dt, 'end_time': dt, 'filename': str})
processing_lock = threading.Lock() # To safely access bark_events list

# --- Load Model ---
def load_model(model_path, device):
    try:
        # Load the state dict
        state_dict = torch.load(model_path, map_location=device)

        # --- Recreate the model architecture ---

        model = SpectrogramCNN(num_classes=1, n_mels=N_MELS)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval() # Set model to evaluation mode
        print(f"Model loaded successfully from {model_path} onto {device}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        exit()

# --- Preprocessing (Mirrors GPU preprocessing from training) ---
# Initialize transformers once on the target device
mel_transformer = T.MelSpectrogram(
    sample_rate=SAMPLE_RATE, n_fft=N_FFT, hop_length=HOP_LENGTH, n_mels=N_MELS, power=2.0
).to(DEVICE)

db_transformer = T.AmplitudeToDB(stype='power', top_db=80).to(DEVICE)

def preprocess_inference(waveform_tensor, device):
    """Preprocesses a single waveform tensor for inference."""
    # Input waveform_tensor shape: [Num_Samples]
    # Move to device
    waveform_tensor = waveform_tensor.to(device)

    # Unsqueeze to add batch dimension: [1, Num_Samples]
    waveform_tensor = waveform_tensor.unsqueeze(0)

    # 1. Compute Mel Spectrogram
    mel_spec = mel_transformer(waveform_tensor) # Shape: [1, N_Mels, Time]

    # 2. Convert to dB scale
    log_mel_spec = db_transformer(mel_spec) # Shape: [1, N_Mels, Time]

    # 3. Normalize Spectrogram (per-instance)
    mean = log_mel_spec.mean(dim=(1, 2), keepdim=True)
    std = log_mel_spec.std(dim=(1, 2), keepdim=True)
    log_mel_spec = (log_mel_spec - mean) / (std + 1e-6)

    # 4. Add channel dimension for Conv2d
    log_mel_spec = log_mel_spec.unsqueeze(1) # Shape: [1, 1, N_Mels, Time]

    return log_mel_spec

# --- Microphone Callback ---
def audio_callback(indata, frames, time_info, status):
    """This is called (from a separate thread) for each audio block."""
    if status:
        print(status, flush=True)
    # Add the incoming audio data (numpy array) to the queue
    audio_queue.put(indata.copy())

# --- Inference Thread ---
def run_inference(model):
    print("Inference thread started...")
    accumulated_audio = np.array([], dtype=np.float32)

    while True:
        try:
            # Get audio data from the queue
            audio_chunk = audio_queue.get() # Blocks until data is available
            if audio_chunk is None: # Sentinel value to stop the thread
                break

            # Append new chunk to accumulated audio
            # Ensure audio_chunk is flattened if it has a channel dimension (it should be [frames, 1])
            accumulated_audio = np.concatenate((accumulated_audio, audio_chunk.flatten()))

            # Process in 1-second chunks
            while len(accumulated_audio) >= NUM_SAMPLES:
                # Get the 1-second chunk for processing
                process_chunk = accumulated_audio[:NUM_SAMPLES]
                # Keep the rest for the next iteration
                accumulated_audio = accumulated_audio[NUM_SAMPLES:]

                # Record timestamp *before* processing
                current_time = datetime.now()

                # Convert to tensor
                waveform_tensor = torch.tensor(process_chunk, dtype=torch.float32)

                # Preprocess
                input_tensor = preprocess_inference(waveform_tensor, DEVICE)

                # Perform inference
                with torch.no_grad():
                    output = model(input_tensor)
                    # Apply sigmoid to get probability
                    probability = torch.sigmoid(output).item()

                # print(f"Time: {current_time.strftime('%H:%M:%S.%f')[:-3]}, Prob: {probability:.4f}") # Debugging

                # Check threshold
                if probability > THRESHOLD:
                    start_time = current_time
                    end_time = start_time + timedelta(seconds=DURATION_SECONDS)
                    filename = f"bark_{start_time.strftime('%Y%m%d_%H%M%S_%f')[:-3]}.wav"
                    filepath = AUDIO_SAVE_DIR / filename

                    print(f"BARK DETECTED! Prob: {probability:.3f}, Saving to {filename}")

                    # Save the audio chunk (use the original numpy array `process_chunk`)
                    try:
                        sf.write(filepath, process_chunk, SAMPLE_RATE)
                    except Exception as e:
                        print(f"Error saving audio file {filepath}: {e}")
                        continue # Skip adding event if save fails

                    # Add event details to the shared list (thread-safe)
                    with processing_lock:
                        bark_events.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'filename': filename
                        })
                        # Optional: Limit the number of stored events to prevent memory issues
                        # MAX_EVENTS = 100
                        # if len(bark_events) > MAX_EVENTS:
                        #     # Remove oldest events (and potentially their files)
                        #     num_to_remove = len(bark_events) - MAX_EVENTS
                        #     removed_events = bark_events[:num_to_remove]
                        #     bark_events = bark_events[num_to_remove:]
                        #     # Consider deleting old files here if needed
                        #     # for event in removed_events:
                        #     #     try: os.remove(AUDIO_SAVE_DIR / event['filename'])
                        #     #     except OSError: pass


        except queue.Empty:
            # Should not happen with blocking get, but good practice
            time.sleep(0.01)
        except Exception as e:
            print(f"Error in inference loop: {e}")
            time.sleep(1) # Avoid spamming errors

    print("Inference thread finished.")


# --- Flask Web Application ---
app = Flask(__name__)

@app.route('/')
def index():
    """Renders the main page displaying bark events."""
    with processing_lock:
        # Pass a copy of the events, sorted newest first
        display_events = sorted(bark_events, key=lambda x: x['start_time'], reverse=True)
    return render_template('index.html', events=display_events)

@app.route('/audio/<filename>')
def serve_bark_audio(filename):
    """Serves the saved audio files."""
    try:
        return send_from_directory(AUDIO_SAVE_DIR, filename, as_attachment=False)
    except FileNotFoundError:
        flask.abort(404)

# --- Main Execution ---
if __name__ == '__main__':
    # Load the trained model
    model = load_model(MODEL_PATH, DEVICE)

    # Start the inference thread
    inference_thread = threading.Thread(target=run_inference, args=(model,), daemon=True)
    inference_thread.start()

    # Start the microphone stream
    try:
        print(f"Available Microphones: {sd.query_devices()}")
        print(f"Using device: {MIC_DEVICE if MIC_DEVICE is not None else 'Default'}")
        print(f"Sample Rate: {SAMPLE_RATE}, Block Duration: {MIC_BLOCK_DURATION_MS}ms")

        stream = sd.InputStream(
            device=MIC_DEVICE,       # Use default device if None
            channels=1,              # Mono input
            samplerate=SAMPLE_RATE,
            callback=audio_callback,
            dtype='float32',         # Match model input type
            blocksize=int(SAMPLE_RATE * MIC_BLOCK_DURATION_MS / 1000) # Calculate frames per callback
        )
        stream.start()
        print("Microphone stream started.")

        # Start the Flask web server
        # Use host='0.0.0.0' to make it accessible on your network
        print("\nWeb server starting. Open http://<your-ip-address>:5000 in your browser.")
        app.run(host='0.0.0.0', port=5000, debug=False) # Turn debug=False for production/stability

    except Exception as e:
        print(f"Error starting audio stream or web server: {e}")
    finally:
        # Cleanup (though daemon thread and server exit might handle this)
        print("Stopping...")
        if 'stream' in locals() and stream.active:
            stream.stop()
            stream.close()
        audio_queue.put(None) # Signal inference thread to stop
        inference_thread.join(timeout=2) # Wait briefly for thread to finish
        print("Exited.")
