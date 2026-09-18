from pathlib import Path

# from data_preprocessing.calculate_reference_values import calculate_reference_values
# from data_preprocessing.convert_to_wav import convert_to_wav
from DL_model.train_model import train_model, run_comparison
from DL_model.train_model import PsychoAcousticDataset
from DL_model.benchmark import run_benchmark


def main():
    train_dir = Path("data") / "training_set"
    val_dir = Path("data") / "validation_set"
    test_dir = Path("data") / "test_set"
    references_path = Path("data") / "reference_values" / "references.csv"
    checkpoint_dir = Path("DL_model") / "epochs"
    losses_dir = Path("DL_model") / "losses"


    dataset = PsychoAcousticDataset(
        train_dir,
        references_path,
        # subset_indices=list(range(len)),
        audio_workers=12
    )

    val_dataset = PsychoAcousticDataset(
        val_dir, references_path,
        # subset_indices=list(range(len_val)),
        audio_workers=12
    )

    #0-95 batch size 128, variance normalized RMSE
    #96-122 batch size 32, variance normalized RMSE
    #0-100 batch size 32, vector - "Also fixed the validation loop, which still had the old misaligned targets[n][:, :preds[n].shape[-1]] trim — it now uses _align_to_model_grid like the training step"

    train_model(
        sound_dir=train_dir,
        val_sound_dir=val_dir,
        val_dataset=val_dataset,
        references_path=references_path,
        checkpoint_dir=checkpoint_dir,
        losses_dir=losses_dir,
        epochs=200,
        lr=1e-3,
        batch_size=32,
        device_id=0,
        num_workers=0,
        use_scheduler=True,
        dataset=dataset,
        vector_loss=True
    )


    run_comparison(
        test_sound_dir=test_dir,
        references_path=references_path,
        checkpoint_dir=checkpoint_dir,
    )

    # benchmark_dataset = PsychoAcousticDataset(
    #     test_dir, references_path, audio_workers=12
    # )
    # run_benchmark(
    #     benchmark_dataset,
    #     checkpoint_dir=checkpoint_dir,
    #     output_dir=Path("DL_model") / "comparison",
    #     n_benchmark=50,
    # )



if __name__ == "__main__":
    main()

"""
Input: Sound File
Output: Psychoacoustic Parameters
Reference: Psychoacoustic Parameters berechnet via MOSQITO
"""