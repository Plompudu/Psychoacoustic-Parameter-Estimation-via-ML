from pathlib import Path
import re

import numpy as np
import pandas as pd

INPUT_DIR = Path(__file__).resolve().parent / ".." / "data" / "reference_values"

SEGMENT_PATTERN = re.compile(
    r"^\[(?P<idx>\d{5})\]_(?P<fold>mono|left|right)_(?P<parent>.+)$"
)

GRID_STEP_MS = 2  # 2 ms frame grid: time_index 0 = 0 ms, 1 = 2 ms, 2 = 4 ms, ...

RESOLUTIONS_MS = {
    "loudness_zwtv": 2,
    "sharpness_din_tv": 2,
    "roughness_dw": 100,
    "tnr_ecma_perseg": 500,
}
SII_RESOLUTION_MS = 1000


def correct_partial_df1(partial_dir, output_dir):
    """Re-index the per-file df1 partials onto the shared 2 ms time grid.

    Each parameter has its own native time resolution (loudness/sharpness
    every 2 ms, roughness every 100 ms, tnr every 500 ms), but the partials
    stored them row-aligned starting at time_index 0. Rebuild the full
    frame-to-time_index mapping so every value lands on the grid position
    it actually belongs to (time_index = i * (resolution_ms / 2)), with
    NaN inserted for all frames in between.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    for csv in sorted(partial_dir.glob("*.csv")):
        df = pd.read_csv(csv)
        n_frames = len(df)

        grid = pd.DataFrame(
            {
                "source_file": csv.stem,
                "time_index": np.arange(n_frames),
            }
        )
        for col, resolution_ms in RESOLUTIONS_MS.items():
            values = df[col].dropna().to_numpy()
            step = resolution_ms // GRID_STEP_MS
            positions = np.arange(len(values)) * step
            positions = positions[positions < n_frames]

            column = np.full(n_frames, np.nan)
            column[positions] = values[: len(positions)]
            grid[col] = column

        output_file = output_dir / f"{csv.stem}.csv"
        grid.to_csv(output_file, index=False)
        print(f"Saved: {output_file} ({len(grid)} rows)")


def compute_frame_counts(partial_dir):
    return {csv.stem: len(pd.read_csv(csv)) for csv in partial_dir.glob("*.csv")}


def build_per_parent_sii_csvs(partial_dir, output_dir, frame_counts):
    """Combine the 1s-segment SII CSVs into one CSV per parent audio file.

    SII is computed at 1 Hz (one scalar per 1s segment = 1000 ms), so the
    value of segment i lands on the shared 2 ms time grid at
    time_index = i * 500. Every other frame stays undefined (NaN), and the
    grid covers the same frame count as the corresponding df1 partial.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    seg_dfs = [pd.read_csv(csv) for csv in sorted(partial_dir.glob("*.csv"))]
    seg = pd.concat(seg_dfs, ignore_index=True)
    print(f"Loaded {len(seg)} SII segments")

    parsed = seg["source_file"].str.extract(SEGMENT_PATTERN)
    seg = seg.assign(seg_idx=parsed["idx"].astype(int), parent=parsed["parent"])

    step = SII_RESOLUTION_MS // GRID_STEP_MS
    for parent, group in seg.sort_values("seg_idx").groupby("parent", sort=False):
        n_frames = frame_counts.get(
            parent,
            (group["seg_idx"].max() - group["seg_idx"].min() + 1) * step,
        )

        positions = (group["seg_idx"] - group["seg_idx"].min()).to_numpy() * step
        mask = positions < n_frames

        sii = np.full(n_frames, np.nan)
        sii[positions[mask]] = group["sii_ansi"].to_numpy()[mask]

        per_parent = pd.DataFrame(
            {
                "source_file": parent,
                "time_index": np.arange(n_frames),
                "sii_ansi": sii,
            }
        )
        output_file = output_dir / f"{parent}.csv"
        per_parent.to_csv(output_file, index=False)
        print(
            f"Saved: {output_file} "
            f"({len(per_parent)} rows, {mask.sum()} SII values)"
        )


def merge_partial_csvs(partial_dir, output_file):
    csvs = sorted(partial_dir.glob("*.csv"))
    if not csvs:
        print(f"No CSVs found in {partial_dir}")
        return pd.DataFrame()

    frames = [pd.read_csv(csv) for csv in csvs]
    df = pd.concat(frames, ignore_index=True)
    df.to_csv(output_file, index=False)
    print(f"Saved: {output_file} ({len(df)} rows)")
    return df


def merge_references(df1_path, df2_path, output_path):
    df1 = pd.read_csv(df1_path)
    df2 = pd.read_csv(df2_path)
    print(f"df1: {len(df1)} rows, df2: {len(df2)} rows")

    df = df1.merge(df2, on=["source_file", "time_index"], how="left")
    cols = ["source_file", "time_index"] + [
        c for c in df.columns if c not in ("source_file", "time_index")
    ]
    df = df[cols]
    df.to_csv(output_path, index=False)
    print(f"Saved: {output_path} ({len(df)} rows)")
    return df


if __name__ == "__main__":
    df1_partials = INPUT_DIR / "partial_df_1"
    df2_partials = INPUT_DIR / "partial_df_2"
    df1_grid = INPUT_DIR / "partial_df_1_2ms_grid"
    df2_grid = INPUT_DIR / "partial_df_2_2ms_grid"

    frame_counts = compute_frame_counts(df1_partials)
    build_per_parent_sii_csvs(df2_partials, df2_grid, frame_counts)
    correct_partial_df1(df1_partials, df1_grid)

    merge_partial_csvs(df1_grid, INPUT_DIR / "df1.csv")
    merge_partial_csvs(df2_grid, INPUT_DIR / "df2.csv")
    merge_references(
        INPUT_DIR / "df1.csv", INPUT_DIR / "df2.csv", INPUT_DIR / "references.csv"
    )