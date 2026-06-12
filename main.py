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
from DL_model.train_model import train_model


def main():
    # convert_to_wav(
    #     input_folder=Path("data") / "raw_audio_files",
    #     output_folder=Path("data") / "standardized_audio_files" / "training_set" / "sound_files",
    #     fs=48000,
    #     win_length_samples=5
    # )
    #
    # calculate_reference_values(
    #     input_folder=Path("data") / "standardized_audio_files" / "training_set" / "sound_files",
    #     output_folder=Path("data") / "standardized_audio_files" / "training_set" / "psycho_acoustic_parameter_labels"
    # )

    train_model(
        sound_dir=Path("data") / "standardized_audio_files" / "training_set" / "sound_files",
        labels_dir=Path("data") / "standardized_audio_files" / "training_set" / "psycho_acoustic_parameter_labels",
        checkpoint_dir=Path("DL_model") / "epochs",
        losses_dir=Path("DL_model") / "losses",
        epochs=10,
        lr=1e-3,
        batch_size=16,
        n_samples_final_comparison=1
    )


if __name__ == "__main__":
    main()

