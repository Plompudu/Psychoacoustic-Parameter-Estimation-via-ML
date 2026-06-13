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
from DL_model.train_model import train_model, run_comparison


def main():
    root = Path("data") / "standardized_audio_files" / "training_set"
    raw_dir = Path("data") / "raw_audio_files"
    sound_dir = root / "sound_files"
    labels_dir = root / "psycho_acoustic_parameter_labels"
    checkpoint_dir = Path("DL_model") / "epochs"
    losses_dir = Path("DL_model") / "losses"

    convert_to_wav(
        input_folder=raw_dir,
        output_folder=sound_dir,
        fs=48000,
        win_length_samples=5,
        number_samples=32
    )

    calculate_reference_values(
        input_folder=sound_dir,
        output_folder=labels_dir
    )

    train_model(
        sound_dir=sound_dir,
        labels_dir=labels_dir,
        checkpoint_dir=checkpoint_dir,
        losses_dir=losses_dir,
        epochs=100,
        lr=1e-3,
        batch_size=16,
        device_id=0,
    )

    run_comparison(
        sound_dir=sound_dir,
        labels_dir=labels_dir,
        checkpoint_dir=checkpoint_dir,
        n_samples=1,
        device_id=0,
    )


if __name__ == "__main__":
    main()

