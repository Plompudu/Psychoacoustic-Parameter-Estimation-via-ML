import random
import shutil
from pathlib import Path

input_path = Path(__file__).resolve().parent / ".." / "data" / "raw_sound_files_1s"
output_path_train = Path(__file__).resolve().parent / ".." / "data" / "training_set"
output_path_test = Path(__file__).resolve().parent / ".." / "data" / "test_set"
output_path_val = Path(__file__).resolve().parent / ".." / "data" / "validation_set"

for output_path in [output_path_train, output_path_test, output_path_val]:
    output_path.mkdir(parents=True, exist_ok=True)
    for f in output_path.glob("*.wav"):
        print(f"Deleting {f}")
        f.unlink()

ratios = {"train": 0.7, "test": 0.15, "validation": 0.15}

audio_files = list(input_path.glob("*.wav"))
print(f"Found {len(audio_files)} audio files")
random.shuffle(audio_files)

total_length = len(audio_files)
train_end = int(total_length * ratios["train"])
test_end = int(total_length * (ratios["train"] + ratios["test"]))

n_train = n_test = n_val = 0
for idx, audio_file in enumerate(audio_files):
    if idx < train_end:
        output_path = output_path_train
        n_train += 1
    elif idx < test_end:
        output_path = output_path_test
        n_test += 1
    else:
        output_path = output_path_val
        n_val += 1

    shutil.copy2(audio_file, output_path / audio_file.name)

print(f"Copied and moved {n_train} train / {n_test} test / {n_val} validation")