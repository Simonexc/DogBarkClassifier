from pathlib import Path
import librosa
import soundfile as sf
import numpy as np
import argparse

def calculate_rms(audio_chunk):
    """Calculates the Root Mean Square (RMS) energy of an audio chunk."""
    # Ensure the chunk is not empty to avoid division by zero or NaN
    if audio_chunk.size == 0:
        return 0.0
    return np.sqrt(np.mean(audio_chunk**2))

def process_wav_files_librosa(input_dir, output_dir, target_sr, chunk_len_s, overlap_s, min_len_s, remove_silence, rms_threshold, output_subtype):
    """
    Processes WAV files using librosa and soundfile.
    Loads, resamples, converts to mono, splits into overlapping chunks,
    optionally removes silent/low-activity chunks based on RMS threshold,
    and saves them to output_dir.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)

    # Validate input directory
    if not input_path.is_dir():
        print(f"Error: Input directory '{input_path}' not found or is not a directory.")
        return

    # Create output directory if it doesn't exist
    output_path.mkdir(parents=True, exist_ok=True)
    print(f"Input directory: {input_path.resolve()}")
    print(f"Output directory: {output_path.resolve()}")

    # Calculate lengths and steps in samples
    chunk_len_samples = int(chunk_len_s * target_sr)
    overlap_samples = int(overlap_s * target_sr)
    step_samples = chunk_len_samples - overlap_samples
    min_len_samples = int(min_len_s * target_sr)

    if step_samples <= 0:
        print("Error: Overlap cannot be greater than or equal to chunk length.")
        return
    if chunk_len_samples <= 0:
        print("Error: Chunk length must be positive.")
        return

    print(f"\nParameters:")
    print(f"  Target SR: {target_sr} Hz")
    print(f"  Chunk length: {chunk_len_s}s ({chunk_len_samples} samples)")
    print(f"  Overlap: {overlap_s}s ({overlap_samples} samples)")
    print(f"  Step size: {step_samples} samples")
    print(f"  Min input length: {min_len_s}s ({min_len_samples} samples)")
    if remove_silence:
        print(f"  Silence removal: Enabled (RMS threshold = {rms_threshold:.4f})")
    else:
        print(f"  Silence removal: Disabled")
    print(f"  Output format: WAV (subtype: {output_subtype})")


    processed_count = 0
    skipped_short_count = 0
    skipped_silence_count = 0
    error_count = 0
    total_chunks_saved = 0

    # Iterate through all .wav files in the input directory
    wav_files = sorted(list(input_path.glob('*.wav'))) # Sort for consistent order
    print(f"\nFound {len(wav_files)} .wav files. Starting processing...")

    for i, file_path in enumerate(wav_files):
        print(f"\n[{i+1}/{len(wav_files)}] Processing: {file_path.name}")
        original_stem = file_path.stem # Filename without extension
        chunks_generated_this_file = 0
        chunks_skipped_silence_this_file = 0

        try:
            # Load audio, resample to target_sr, convert to mono
            audio_data, sr = librosa.load(file_path, sr=target_sr, mono=True)

            # Check duration (using samples)
            duration_samples = len(audio_data)
            if duration_samples < min_len_samples:
                print(f"  Skipping (Too Short): File duration ({duration_samples/target_sr:.2f}s) < minimum ({min_len_s}s).")
                skipped_short_count += 1
                continue

            print(f"  Loaded: duration={duration_samples/target_sr:.2f}s, original_sr={librosa.get_samplerate(file_path)}Hz -> target_sr={sr}Hz")

            # --- Splitting ---
            start_sample = 0
            chunk_counter = 0 # Simple counter for UID within this file

            while start_sample + chunk_len_samples <= duration_samples:
                end_sample = start_sample + chunk_len_samples
                chunk = audio_data[start_sample:end_sample]

                # --- Silence/Noise Filtering ---
                if remove_silence:
                    chunk_rms = calculate_rms(chunk)
                    if chunk_rms < rms_threshold:
                        # print(f"    Skipping chunk {chunk_counter:04d} (RMS: {chunk_rms:.4f} < {rms_threshold:.4f})") # Verbose
                        chunks_skipped_silence_this_file += 1
                        start_sample += step_samples # Move to the next potential chunk start
                        chunk_counter += 1 # Increment counter even if skipped to maintain sequence if needed
                        continue # Skip saving this chunk

                # --- Saving Chunk ---
                # Generate unique ID and output filename
                uid = f"{chunk_counter:04d}"
                output_filename = f"{original_stem}_{uid}.wav" # Hardcode wav extension
                output_file_path = output_path / output_filename

                # Save the chunk using soundfile
                sf.write(output_file_path, chunk, target_sr, subtype=output_subtype)
                # print(f"    Saved chunk: {output_file_path.name}") # Verbose

                chunks_generated_this_file += 1
                total_chunks_saved += 1
                chunk_counter += 1
                start_sample += step_samples # Move start sample for the next chunk

            print(f"  Finished: Saved {chunks_generated_this_file} chunks.")
            if remove_silence:
                 print(f"            Skipped {chunks_skipped_silence_this_file} silent chunks.")
            processed_count += 1
            skipped_silence_count += chunks_skipped_silence_this_file


        except Exception as e:
            print(f"  ERROR processing '{file_path.name}': {e}")
            import traceback
            traceback.print_exc() # Print detailed traceback for debugging
            error_count += 1

    print("\n--- Processing Summary ---")
    print(f"Total files processed successfully: {processed_count}")
    print(f"Total files skipped (too short): {skipped_short_count}")
    print(f"Total files skipped (errors): {error_count}")
    print(f"Total chunks saved: {total_chunks_saved}")
    if remove_silence:
        print(f"Total chunks skipped (low activity/silence): {skipped_silence_count}")
    print("--------------------------")


# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Split WAV files into overlapping chunks, normalize, and optionally remove silence.")

    parser.add_argument("-i", "--input", type=str,
                        help="Folder containing original WAV files")
    parser.add_argument("-o", "--output", type=str,
                        help="Folder to save the split WAV files")
    parser.add_argument("-sr", "--samplerate", type=int, default=16000,
                        help="Target sample rate for output files (Hz) (default: 16000)")
    parser.add_argument("-cl", "--chunklen", type=float, default=1.0,
                        help="Desired chunk length in seconds (default: 1.0)")
    parser.add_argument("-ol", "--overlap", type=float, default=0.2,
                        help="Overlap length in seconds (default: 0.2)")
    parser.add_argument("-ml", "--minlen", type=float, default=1.0,
                        help="Minimum duration in seconds for an input file to be processed (default: 1.0)")
    parser.add_argument("-rs", "--removesilence", action='store_true',
                        help="Enable removal of chunks with low RMS energy (silence/noise)")
    parser.add_argument("-rt", "--rmsthreshold", type=float, default=0.005,
                        help="RMS energy threshold below which chunks are considered silent (effective only if --removesilence is used) (default: 0.005)")
    parser.add_argument("-st", "--subtype", type=str, default='PCM_16',
                        help="Subtype for output WAV files (e.g., PCM_16, PCM_24, FLOAT) (default: PCM_16)")

    args = parser.parse_args()

    # Validate overlap < chunklen
    if args.overlap >= args.chunklen:
        parser.error("Overlap (--overlap) must be less than chunk length (--chunklen).")
    # Validate minlen >= chunklen (or adjust logic if partial chunks from short files are desired)
    if args.minlen < args.chunklen:
         print("Warning: Minimum length (--minlen) is less than chunk length (--chunklen). Files between minlen and chunklen will be skipped.")
         # Adjusting minlen to be at least chunklen to match skipping logic
         args.minlen = args.chunklen

    process_wav_files_librosa(
        input_dir=args.input,
        output_dir=args.output,
        target_sr=args.samplerate,
        chunk_len_s=args.chunklen,
        overlap_s=args.overlap,
        min_len_s=args.minlen, # Use adjusted minlen
        remove_silence=args.removesilence,
        rms_threshold=args.rmsthreshold,
        output_subtype=args.subtype
    )
