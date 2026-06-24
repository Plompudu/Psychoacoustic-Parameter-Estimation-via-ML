import csv
from pathlib import Path
import torch
from DL_model.params import PARAM_NAMES


def convert_csv_to_pt(csv_path: Path, output_dir: Path):
    csv_path = Path(csv_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    current_stem: str | None = None
    current_values: dict[str, list[float]] = {n: [] for n in PARAM_NAMES}
    count = 0

    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stem = row["source_file"].replace(".csv", "")

            if stem != current_stem and current_stem is not None:
                targets = {
                    name: torch.tensor(vals, dtype=torch.float32)
                    for name, vals in current_values.items()
                }
                torch.save(targets, output_dir / f"{current_stem}.pt")
                count += 1
                current_values = {n: [] for n in PARAM_NAMES}

            current_stem = stem
            for name in PARAM_NAMES:
                val = row[name]
                current_values[name].append(float(val) if val else float("nan"))

    if current_stem is not None:
        targets = {
            name: torch.tensor(vals, dtype=torch.float32)
            for name, vals in current_values.items()
        }
        torch.save(targets, output_dir / f"{current_stem}.pt")
        count += 1

    print(f"Exported {count} label files to {output_dir}")


if __name__ == "__main__":
    convert_csv_to_pt(
        csv_path=Path("data") / "standardized_audio_files" / "training_set" / "all_psychoacoustic_labels.csv",
        output_dir=Path("data") / "standardized_audio_files" / "training_set" / "psycho_acoustic_parameter_labels_pt",
    )
