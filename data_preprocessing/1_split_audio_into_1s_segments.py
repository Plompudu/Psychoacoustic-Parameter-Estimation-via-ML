from pathlib import Path
from pydub import AudioSegment

input_path = Path(__file__).resolve().parent / ".." / "data" / "raw_sound_files"
output_path = Path(__file__).resolve().parent / ".." / "data" / "raw_sound_files_1s"
max_segment_length_ms = 1000

#create and clear output
output_path.mkdir(parents=True, exist_ok=True)
for f in output_path.glob("*.wav"):
    print(f"Deleting {f}")
    f.unlink()

audio_files = list(input_path.glob("*.wav"))
print(f"Found {len(audio_files)} audio files")

global_idx = 0

for file_idx, audio_file in enumerate(audio_files, start=1):
    print(f"[{file_idx}/{len(audio_files)}] Processing {audio_file.name}...")
    channels = AudioSegment.from_wav(audio_file).split_to_mono()
    stem = audio_file.stem

    if len(channels) == 1:
        channel = channels[0]
        for start_ms in range(0, len(channel), max_segment_length_ms):
            end_ms = min(start_ms + max_segment_length_ms, len(channel))
            channel[start_ms:end_ms].export(
                output_path / f"[{global_idx:05d}]_mono_{stem}.wav",
                format="wav",
            )
            global_idx += 1
        continue

    left_channel, right_channel = channels[:2]

    for start_ms in range(0, len(left_channel), max_segment_length_ms):
        end_ms = min(start_ms + max_segment_length_ms, len(left_channel))

        left_channel[start_ms:end_ms].export(
            output_path / f"[{global_idx:05d}]_left_{stem}.wav",
            format="wav",
        )

        right_channel[start_ms:end_ms].export(
            output_path / f"[{global_idx:05d}]_right_{stem}.wav",
            format="wav",
        )

        global_idx += 1

