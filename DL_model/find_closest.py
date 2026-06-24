import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from DL_model.params import PARAM_NAMES


def find_closest(
    avg_csv: str | Path,
    labels_csv: str | Path,
    n: int = 3,
) -> list[tuple[str, float]]:
    """Return the N stems whose per-time-segment parameter values are
    closest (in MSE) to the population averages."""

    avg_df = pd.read_csv(avg_csv)
    avg_arrays = {name: avg_df[name].to_numpy(dtype=np.float32) for name in PARAM_NAMES}

    labels_pt = Path(labels_csv).with_suffix(".labels.pt")
    if labels_pt.exists():
        print(f"Loading pre-parsed labels from {labels_pt}...")
        cached = torch.load(labels_pt, weights_only=True)
        all_stems = cached["stems"]
        all_labels = cached["labels"]
    else:
        print("Parsing labels CSV...")
        usecols = ["source_file"] + PARAM_NAMES
        all_labels: dict[str, dict[str, list[np.ndarray]]] = {}
        for chunk in pd.read_csv(labels_csv, usecols=usecols, chunksize=100_000):
            chunk["stem"] = chunk["source_file"].str.replace(".csv", "", regex=False)
            for stem, grp in chunk.groupby("stem"):
                if stem not in all_labels:
                    all_labels[stem] = {}
                for name in PARAM_NAMES:
                    all_labels[stem].setdefault(name, []).append(grp[name].to_numpy(dtype=np.float32))
        all_stems = sorted(all_labels.keys())
        for stem in all_stems:
            for name in PARAM_NAMES:
                all_labels[stem][name] = torch.from_numpy(np.concatenate(all_labels[stem][name]))

    print(f"Computing MSE for {len(all_stems)} stems...")
    t0 = time.perf_counter()
    distances: list[tuple[str, float]] = []
    for i, stem in enumerate(all_stems):
        if i > 0 and i % 100 == 0:
            print(f"  {i}/{len(all_stems)} stems ({time.perf_counter() - t0:.1f}s)...")
        mse_sum = 0.0
        count = 0
        for name in PARAM_NAMES:
            avg_vals = avg_arrays[name]
            stem_vals = all_labels[stem][name].numpy()

            min_len = min(len(avg_vals), len(stem_vals))
            if min_len == 0:
                continue
            avg_vals = avg_vals[:min_len]
            stem_vals = stem_vals[:min_len]

            mask = ~np.isnan(avg_vals) & ~np.isnan(stem_vals)
            if mask.any():
                mse_sum += ((avg_vals[mask] - stem_vals[mask]) ** 2).sum()
                count += mask.sum()

        avg_mse = mse_sum / count if count > 0 else float("inf")
        distances.append((stem, avg_mse))

    print(f"  done ({time.perf_counter() - t0:.1f}s)")
    distances.sort(key=lambda x: x[1])
    return distances[:n]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Find N stems closest to parameter averages")
    parser.add_argument("n", type=int, nargs="?", default=10, help="Number of closest stems to return")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent.parent
    avg_csv = root / "data" / "standardized_audio_files" / "training_set" / "visualization" / "parameter_average_per_time_segment.csv"
    labels_csv = root / "data" / "standardized_audio_files" / "training_set" / "all_psychoacoustic_labels.csv"

    closest = find_closest(avg_csv, labels_csv, n=args.n)

    # Map stem names back to indices in the full stems list
    labels_pt = labels_csv.with_suffix(".labels.pt")
    cached = torch.load(labels_pt, weights_only=True)
    all_stems = cached["stems"]
    stem_index = {s: i for i, s in enumerate(all_stems)}
    indices = [stem_index[s] for s, _ in closest]

    print("\nClosest stems (name + mse):")
    for i, (stem, mse) in enumerate(closest, 1):
        print(f"  {i:>3}. {stem}  (mse={mse:.6f})")

    print(f"\nCopy-paste array:  subset_indices={indices}")
