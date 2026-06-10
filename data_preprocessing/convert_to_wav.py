from pathlib import Path
import librosa
import soundfile as sf
import numpy as np



SHARPNESS_FRAME_S = 0.002


def convert_to_wav(input_folder: Path, output_folder: Path, fs=48000):
    """
    Convert all audio files in input_folder to WAV format
    with sampling rate fs and save them in output_folder.
    Split into 5s windows with 0.01s overlap (5 sharpness frames)
    to ensure full coverage after the sharpness skip.
    Skip already generated audio files.
    """

    audio_extensions = {
        ".mp3", ".flac", ".m4a", ".aac",
        ".ogg", ".wma", ".wav"
    }

    win_length = int(5 * fs)
    hop_length = win_length - int(SHARPNESS_FRAME_S * 5 * fs)

    for file_path in input_folder.iterdir():
        if not (file_path.is_file() and file_path.suffix.lower() in audio_extensions):
            print(f"Skipping (not audio or unsupported): {file_path.name}")
            continue

        try:
            audio, _ = librosa.load(file_path, sr=fs, mono=False)

            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            n_channels, n_samples = audio.shape
            base_name = file_path.stem

            if n_samples <= win_length:
                duration_ms = int(n_samples / fs * 1000)
                for ch_idx in range(n_channels):
                    output_path = output_folder / f"{base_name}_00000-{duration_ms:05d}ms.wav"
                    if n_channels > 1:
                        output_path = output_path.parent / f"{output_path.stem}_ch{ch_idx + 1}{output_path.suffix}"
                    if output_path.exists():
                        print(f"Skipping (already exists): {output_path.name}")
                        continue
                    sf.write(output_path, audio[ch_idx], fs)
                    print(f"Converted: {file_path.name} -> {output_path.name}")
            else:
                n_windows = (n_samples - win_length) // hop_length + 1

                for win_idx in range(n_windows):
                    start_sample = win_idx * hop_length
                    end_sample = start_sample + win_length
                    start_ms = int(start_sample / fs * 1000)
                    end_ms = int(end_sample / fs * 1000)

                    for ch_idx in range(n_channels):
                        ch_suffix = f"_ch{ch_idx + 1}" if n_channels > 1 else ""
                        output_path = output_folder / f"{base_name}_{start_ms:05d}-{end_ms:05d}ms{ch_suffix}.wav"
                        if output_path.exists():
                            print(f"Skipping (already exists): {output_path.name}")
                            continue

                        audio_win = audio[ch_idx, start_sample:end_sample]
                        sf.write(output_path, audio_win, fs)
                        print(f"Converted: {file_path.name} -> {output_path.name}")

        except Exception as e:
            print(f"Failed: {file_path.name} -> {e}")
