from pathlib import Path
import pandas as pd
import torch
from DL_model.params import PARAM_NAMES


def convert_labels_to_pt(
    input_dir: Path,
    output_dir: Path,
):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(input_dir.glob("*.csv"))
    if not csv_files:
        print(f"No CSV files found in {input_dir}")
        return

    for csv_path in csv_files:
        df = pd.read_csv(csv_path)
        targets = {}
        for name in PARAM_NAMES:
            targets[name] = torch.from_numpy(df[name].values.astype("float32"))
        pt_path = output_dir / csv_path.with_suffix(".pt").name
        torch.save(targets, pt_path)

    print(f"Converted {len(csv_files)} label files to .pt in {output_dir}")


if __name__ == "__main__":
    convert_labels_to_pt(
        input_dir=Path("data") / "standardized_audio_files" / "training_set" / "psycho_acoustic_parameter_labels",
        output_dir=Path("data") / "standardized_audio_files" / "training_set" / "psycho_acoustic_parameter_labels_pt",
    )
