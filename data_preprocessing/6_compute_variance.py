from pathlib import Path
import sys

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO))

from DL_model.labels import (  # noqa: E402
    build_labels_for_stems,
    collect_segment_stems,
    load_references_grouped,
    segment_local_offsets,
)
from DL_model.params import PARAM_NAMES  # noqa: E402

INPUT_DIR = _REPO / "data" / "reference_values"
REFERENCES_PATH = INPUT_DIR / "references.csv"
OUTPUT_PATH = INPUT_DIR / "variance.csv"

TRAIN_DIR = _REPO / "data" / "training_set"
SEGMENTS_DIR = _REPO / "data" / "raw_sound_files_1s"

STAT_INDEX = ["count", "mean", "std", "min", "25%", "50%", "75%", "max"]


def compute_stats(values: list[np.ndarray]) -> list[float]:
    """describe()-style statistics over concatenated non-NaN values."""
    arr = np.concatenate([v[v == v] for v in values])
    if arr.size == 0:
        return [0.0, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]
    q25, q50, q75 = np.quantile(arr, [0.25, 0.5, 0.75])
    return [
        float(arr.size),
        float(arr.mean()),
        float(arr.std(ddof=1)),
        float(arr.min()),
        float(q25),
        float(q50),
        float(q75),
        float(arr.max()),
    ]


def main():
    print(f"Reading {REFERENCES_PATH} ...")
    refs = load_references_grouped(REFERENCES_PATH)
    print(f"Loaded reference data for {len(refs)} source files")

    stems = sorted(p.stem for p in TRAIN_DIR.glob("*.wav"))
    offsets = segment_local_offsets(collect_segment_stems(SEGMENTS_DIR))
    print(f"Building labels for {len(stems)} training segments ...")
    labels = build_labels_for_stems(refs, stems, offsets=offsets)
    print(f"  built labels for {len(labels)} segments")

    values: dict[str, list[np.ndarray]] = {name: [] for name in PARAM_NAMES}
    for name in PARAM_NAMES:
        for seg_labels in labels.values():
            values[name].append(seg_labels[name])

    stats = pd.DataFrame(
        {name: compute_stats(values[name]) for name in PARAM_NAMES},
        index=STAT_INDEX,
    )
    stats.index.name = "stat"
    stats.to_csv(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")
    print(stats.loc[["std"], :].to_string())


if __name__ == "__main__":
    main()