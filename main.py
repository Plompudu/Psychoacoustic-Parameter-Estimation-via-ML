from pathlib import Path

from data_preprocessing.calculate_reference_values import calculate_reference_values
from data_preprocessing.convert_to_wav import convert_to_wav
from DL_model.train_model import train_model, run_comparison
from DL_model.train_model import PsychoAcousticDataset


def main():
    root = Path("data") / "standardized_audio_files" / "training_set"
    raw_dir = Path("data") / "raw_audio_files"
    sound_dir = root / "sound_files"
    labels_csv_path = root / "all_psychoacoustic_labels.csv"
    checkpoint_dir = Path("DL_model") / "epochs"
    losses_dir = Path("DL_model") / "losses"

    # convert_to_wav(
    #     input_folder=raw_dir,
    #     output_folder=sound_dir,
    #     fs=48000,
    #     win_length_samples=1,
    #     number_samples=None
    # )

    # calculate_reference_values(
    #     input_folder=sound_dir,
    #     output_folder=labels_dir
    # )

    dataset = PsychoAcousticDataset(sound_dir, labels_csv_path, subset_indices=[70000])

    # train_model(
    #     sound_dir=sound_dir,
    #     labels_csv_path=labels_csv_path,
    #     checkpoint_dir=checkpoint_dir,
    #     losses_dir=losses_dir,
    #     epochs=2000,
    #     lr=1e-3,
    #     batch_size=1,
    #     device_id=0,
    #     num_workers=0,
    #     #subset_indices=[42],
    #     dataset=dataset,
    # )

    run_comparison(
        sound_dir=sound_dir,
        labels_csv_path=labels_csv_path,
        checkpoint_dir=checkpoint_dir,
        n_samples=1,
        device_id=0,
        #subset_indices=[42],
        epochs=[0, 1000, "newest"],
        dataset=dataset,
    )



if __name__ == "__main__":
    main()

"""
Input: Sound File
Output: Psychoacoustic Parameters
Reference: Psychoacoustic Parameters berechnet via MOSQITO
"""