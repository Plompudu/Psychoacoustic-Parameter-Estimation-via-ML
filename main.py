"""
Datatype Definitions:
Audio Data: soundfile, PA_Parameters

soundfile:
    48kHz
    .wav
"""
from data_preprocessing.calculate_reference_values import calculate_reference_values
from data_preprocessing.convert_to_wav import convert_to_wav
from data_preprocessing.generate_randomized_training_set_with_applied_filters import generate_randomized_training_set_with_applied_filters
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

    generate_randomized_training_set_with_applied_filters(
        input_folder=Path("data") / "standardized_audio_files" / "training_set",
        output_folder=Path("data") / "standardized_audio_files" / "training_set" / "filtered",
        number_of_samples=3
    )


if __name__ == "__main__":
    main()

