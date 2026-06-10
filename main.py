"""
Datatype Definitions:
Audio Data: soundfile, PA_Parameters

soundfile:
    48kHz
    .wav
"""
from data_preprocessing.calculate_reference_values import calculate_reference_values
from data_preprocessing.convert_to_wav import convert_to_wav
from pathlib import Path




def main():
    convert_to_wav(
        input_folder=Path("data") / "raw_audio_files",
        output_folder=Path("data") / "standardized_audio_files" / "training_set",
        fs=48000
    )

    calculate_reference_values(
        input_folder=Path("data") / "standardized_audio_files" / "training_set",
        output_folder=Path("data") / "standardized_audio_files" / "training_set" / "reference"
    )


if __name__ == "__main__":
    main()

