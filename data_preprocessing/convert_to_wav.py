from pathlib import Path
import librosa
import soundfile as sf
import numpy as np



SHARPNESS_FRAME_S = 0.002
SHARPNESS_OVERLAP_S = SHARPNESS_FRAME_S * 5

def convert_to_wav(input_folder: Path, output_folder: Path, fs=48000, win_length_samples=5, number_samples=None):
    """
    Convert all audio files in input_folder to WAV format
    with sampling rate fs and save them in output_folder.
    Split into 5s windows with 0.01s overlap (5 sharpness frames)
    to ensure full coverage after the sharpness skip.
    Skip already generated audio files.
    If number_samples is set, stop after that many total files exist in the output folder.
    """
    print("=" * 100)

    audio_extensions = {
        ".mp3", ".flac", ".m4a", ".aac",
        ".ogg", ".wma", ".wav"
    }

    win_length_samples = int(win_length_samples * fs)
    sharpness_length_samples = int(SHARPNESS_FRAME_S * fs)
    hop_length = win_length_samples - sharpness_length_samples

    extensions_lower = {ext.lower() for ext in audio_extensions}
    audio_files = [
        p for p in input_folder.rglob("*")
        if p.is_file() and p.suffix.lower() in extensions_lower
    ]
    if not audio_files:
        print(f"No audio files found under {input_folder}")
        return

    existing_stems: set[str] = set()
    max_idx = 0
    for f in output_folder.glob("*.wav"):
        stem = f.stem
        if "]" in stem:
            content = stem.split("]", 1)[1].lstrip("_")
            existing_stems.add(content)
            try:
                idx = int(stem.split("]")[0].lstrip("["))
                max_idx = max(max_idx, idx)
            except ValueError:
                pass
        else:
            existing_stems.add(stem)

    next_idx = max_idx + 1

    def _enumerate_path(base_stem: str, idx: int, ch_suffix: str = "") -> Path:
        return output_folder / f"[{idx:06d}]_{base_stem}{ch_suffix}.wav"

    def _check_limit(generated_count):
        if number_samples is not None and generated_count > number_samples:
            print(f"Reached limit of {number_samples} files — stopping.")
            return True
        return False

    for file_path in audio_files:

        try:
            audio, _ = librosa.load(file_path, sr=fs, mono=False)

            if audio.ndim == 1:
                audio = audio[np.newaxis, :]

            n_channels, n_samples = audio.shape
            rel_stem = f"{file_path.parent.name}_{file_path.stem}"

            if n_samples <= win_length_samples:
                duration_ms = int(n_samples / fs * 1000)
                for ch_idx in range(n_channels):
                    ch_suffix = f"_ch{ch_idx + 1}" if n_channels > 1 else ""
                    content_stem = f"{rel_stem}_00000-{duration_ms:05d}ms{ch_suffix}"
                    if content_stem in existing_stems:
                        print(f"Skipping (already generated): {content_stem}")
                        continue
                    idx = next_idx
                    if _check_limit(idx):
                        return
                    output_path = _enumerate_path(content_stem, idx)
                    sf.write(output_path, audio[ch_idx], fs)
                    existing_stems.add(content_stem)
                    next_idx = idx + 1
                    print(f"Converted ({idx}): {file_path.name} -> {output_path.name}")
            else:
                n_windows = (n_samples - win_length_samples) // hop_length + 1

                for win_idx in range(n_windows):
                    start_sample = win_idx * hop_length
                    end_sample = start_sample + win_length_samples
                    start_ms = int(start_sample / fs * 1000)
                    end_ms = int(end_sample / fs * 1000)

                    for ch_idx in range(n_channels):
                        ch_suffix = f"_ch{ch_idx + 1}" if n_channels > 1 else ""
                        content_stem = f"{rel_stem}_{start_ms:05d}-{end_ms:05d}ms{ch_suffix}"
                        if content_stem in existing_stems:
                            print(f"Skipping (already generated): {content_stem}")
                            continue
                        idx = next_idx
                        if _check_limit(idx):
                            return
                        output_path = _enumerate_path(content_stem, idx)

                        audio_win = audio[ch_idx, start_sample:end_sample]
                        sf.write(output_path, audio_win, fs)
                        existing_stems.add(content_stem)
                        next_idx = idx + 1
                        print(f"Converted ({idx}): {file_path.name} -> {output_path.name}")

        except Exception as e:
            print(f"Failed: {file_path.name} -> {e}")
