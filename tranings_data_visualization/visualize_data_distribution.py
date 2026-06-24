import os
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..")
sys.path.insert(0, _ROOT)

warnings.filterwarnings("ignore", category=FutureWarning)

CSV_PATH = os.path.join(
    _ROOT, "data", "standardized_audio_files", "training_set",
    "visualization", "all_psychoacoustic_labels.csv",
)
OUTPUT_DIR = os.path.join(_ROOT, "data", "standardized_audio_files", "training_set", "visualization")

PARAM_NAMES = [
    "loudness_zwtv",
    "sharpness_din_tv",
    "roughness_dw",
    "tnr_ecma_perseg",
    "sii_ansi",
]

PARAM_LABELS = {
    "loudness_zwtv": "Loudness (Zwicker, TV) [sone]",
    "sharpness_din_tv": "Sharpness (DIN, TV) [acum]",
    "roughness_dw": "Roughness (Daniel & Weber) [asper]",
    "tnr_ecma_perseg": "TNR (ECMA, per-segment) [dB]",
    "sii_ansi": "SII (ANSI) [0-1]",
}

CHUNK_SIZE = 50000


def _load_chunked(filepath: str) -> pd.DataFrame:
    chunks = []
    for c in pd.read_csv(
        filepath,
        usecols=["time_index"] + PARAM_NAMES,
        dtype={"time_index": "int16"} | {p: "float32" for p in PARAM_NAMES},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunks.append(c)
    return pd.concat(chunks, ignore_index=True)


def _load_with_source(filepath: str) -> pd.DataFrame:
    chunks = []
    for c in pd.read_csv(
        filepath,
        usecols=["source_file", "time_index"] + PARAM_NAMES,
        dtype={"source_file": "str", "time_index": "int16"} | {p: "float32" for p in PARAM_NAMES},
        chunksize=CHUNK_SIZE,
        low_memory=False,
    ):
        chunks.append(c)
    return pd.concat(chunks, ignore_index=True)


def plot_histograms(df: pd.DataFrame, output_dir: str):
    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    for ax, name in zip(axes, PARAM_NAMES):
        col = df[name].dropna().values
        ax.hist(col, bins=80, density=True, alpha=0.7, color="steelblue", edgecolor="white", linewidth=0.3)
        ax.set_title(PARAM_LABELS.get(name, name))
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")
    fig.tight_layout()
    path = os.path.join(output_dir, "parameter_distributions.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_average_per_time_segment(df: pd.DataFrame, output_dir: str):
    """Mean of each parameter at each time_index across all files."""
    grouped = df.groupby("time_index", observed=True)[PARAM_NAMES].mean()
    csv_path = os.path.join(output_dir, "parameter_average_per_time_segment.csv")
    grouped.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(n, 1, figsize=(10, 3 * n))
    for ax, name in zip(axes, PARAM_NAMES):
        pts = grouped[name].dropna()
        if len(pts) <= 2:
            ax.plot(pts.index.values, pts.values, color="steelblue",
                    marker="x", linestyle="", markersize=8)
        else:
            ax.plot(pts.index.values, pts.values, color="steelblue", linewidth=0.8)
        ax.set_title(PARAM_LABELS.get(name, name))
        ax.set_xlabel("Time index (frame)")
        ax.set_ylabel("Mean value")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "parameter_average_per_time_segment.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_length_distribution(df: pd.DataFrame, output_dir: str):
    """Per-file frame count for each parameter: min, max, avg + distribution."""
    csv_rows = []
    for name in PARAM_NAMES:
        col = df.groupby("source_file", observed=True)[name].count()
        csv_rows.append({"parameter": name, "min": col.min(), "max": col.max(), "mean": col.mean()})
    csv_path = os.path.join(output_dir, "parameter_length_stats.csv")
    pd.DataFrame(csv_rows).to_csv(csv_path, index=False)
    print(f"Saved: {csv_path}")

    n = len(PARAM_NAMES)
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 7),
                             gridspec_kw={"height_ratios": [1, 2]})

    for i, name in enumerate(PARAM_NAMES):
        col = df.groupby("source_file", observed=True)[name].count()
        stats = {
            "Min": col.min(),
            "Max": col.max(),
            "Mean": col.mean(),
        }
        labels_bars = list(stats.keys())
        values_bars = list(stats.values())

        axes[0, i].bar(labels_bars, values_bars, color=["cornflowerblue", "coral", "seagreen"])
        axes[0, i].set_title(PARAM_LABELS.get(name, name))
        axes[0, i].set_ylabel("Frame count")

        axes[1, i].hist(col.values, bins=min(50, col.nunique()), color="steelblue",
                        edgecolor="white", linewidth=0.3, alpha=0.7)
        axes[1, i].set_xlabel("Frame count per file")
        axes[1, i].set_ylabel("Number of files")

        for label, v in zip(labels_bars, values_bars):
            axes[0, i].text(labels_bars.index(label), v + 0.5, f"{v:.1f}",
                            ha="center", fontsize=8)

    fig.tight_layout()
    path = os.path.join(output_dir, "parameter_length_stats.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_value_stats(df: pd.DataFrame, output_dir: str):
    """Min, max, mean of actual parameter values across all data."""
    stats = df[PARAM_NAMES].describe()
    csv_path = os.path.join(output_dir, "parameter_value_stats.csv")
    stats.to_csv(csv_path)
    print(f"Saved: {csv_path}")

    x = range(len(PARAM_NAMES))
    mins = stats.loc["min"].values
    maxs = stats.loc["max"].values
    means = stats.loc["mean"].values

    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.25
    bars_min = ax.bar([i - w for i in x], mins, width=w, label="Min", color="cornflowerblue")
    bars_mean = ax.bar(x, means, width=w, label="Mean", color="seagreen")
    bars_max = ax.bar([i + w for i in x], maxs, width=w, label="Max", color="coral")

    for bars, vals in [(bars_min, mins), (bars_mean, means), (bars_max, maxs)]:
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"{v:.2g}", ha="center", va="bottom", fontsize=6)

    ax.set_xticks(x)
    ax.set_xticklabels([PARAM_LABELS.get(n, n) for n in PARAM_NAMES], rotation=20, ha="right", fontsize=8)
    ax.legend(fontsize=9)
    ax.set_ylabel("Value")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    path = os.path.join(output_dir, "parameter_value_stats.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"Reading {CSV_PATH} ...")
    df = _load_chunked(CSV_PATH)
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns (no source_file)")

    plot_histograms(df, OUTPUT_DIR)
    plot_average_per_time_segment(df, OUTPUT_DIR)
    plot_value_stats(df, OUTPUT_DIR)

    del df

    print("Re-reading with source_file for length analysis ...")
    df2 = _load_with_source(CSV_PATH)
    print(f"Loaded {len(df2):,} rows")

    plot_length_distribution(df2, OUTPUT_DIR)

    print("\nSummary statistics:")
    print(df2[PARAM_NAMES].describe().to_string())


if __name__ == "__main__":
    main()
