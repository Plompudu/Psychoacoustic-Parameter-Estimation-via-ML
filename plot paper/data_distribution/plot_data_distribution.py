import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.join(_HERE, "..", "..")
CSV_PATH = os.path.join(_ROOT, "data", "reference_values", "references.csv")
OUTPUT_DIR = _HERE

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

TIME_PER_INDEX_MS = 2  # 1 time_index = 2 ms
CHUNK_MS = 1000
CHUNK_INDICES = CHUNK_MS // TIME_PER_INDEX_MS  # 500

SCALE_CONFIG = {
    "sharpness_din_tv": {"type": "symlog", "linthresh": 1.0},
}
LOG_FLOOR = 0.01  # values <= 0 are clipped to this floor on log axes


def _load() -> pd.DataFrame:
    return pd.read_csv(
        CSV_PATH,
        usecols=["time_index"] + PARAM_NAMES,
        dtype={"time_index": "int32"} | {p: "float32" for p in PARAM_NAMES},
    )


def _chunk(df: pd.DataFrame, agg: str) -> pd.DataFrame:
    """Per-aggregation of each parameter per time_index over the first
    1000 ms, across all source files."""
    chunk = df[df["time_index"] < CHUNK_INDICES]
    if agg == "mean":
        return chunk.groupby("time_index", observed=True)[PARAM_NAMES].mean()
    if agg == "median":
        return chunk.groupby("time_index", observed=True)[PARAM_NAMES].median()
    raise ValueError(f"Unsupported aggregation: {agg}")


def _save_chunk(chunk: pd.DataFrame, agg: str):
    path = os.path.join(OUTPUT_DIR, f"{agg}_1000ms_chunk.csv")
    chunk.to_csv(path)
    print(f"Saved: {path}")


def plot_histograms(df: pd.DataFrame):
    fig, axes = plt.subplots(len(PARAM_NAMES), 1, figsize=(10, 3 * len(PARAM_NAMES)))
    fig.suptitle("Parameter distributions over all files")
    for ax, name in zip(axes, PARAM_NAMES):
        col = df[name].dropna().values
        ax.hist(col, bins=80, density=True, alpha=0.7, color="steelblue",
                edgecolor="white", linewidth=0.3)
        ax.set_title(PARAM_LABELS.get(name, name))
        ax.set_ylabel("Density")
        ax.set_xlabel("Value")
    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "histograms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def plot_chunk(chunk: pd.DataFrame, agg: str):
    """First 1000 ms chunk (mean or median), x-axis in ms at each parameter's
    native resolution (2 ms / time_index). SII has a single value per second
    and is drawn as a constant line over the chunk."""
    fig, axes = plt.subplots(len(PARAM_NAMES), 1, figsize=(10, 3 * len(PARAM_NAMES)))
    label_cap = agg.capitalize()
    fig.suptitle(f"{label_cap} of first {CHUNK_MS} ms chunk over all files")

    for ax, name in zip(axes, PARAM_NAMES):
        pts = chunk[name].dropna()
        if pts.empty:
            ax.set_title(PARAM_LABELS.get(name, name) + " (no values)")
            ax.set_ylim(0, 1)
            ax.set_ylabel(f"{label_cap} value")
            ax.set_xlabel("Time [ms]")
            ax.grid(True, alpha=0.3)
            continue
        ms = pts.index.values * TIME_PER_INDEX_MS
        if len(pts) == 1:
            v = float(pts.values[0])
            ax.plot([0, CHUNK_MS], [v, v], color="steelblue", linewidth=1.5,
                    label=f"{v:.4g}")
            ax.legend(fontsize=8, loc="best")
        else:
            ax.plot(ms, pts.values, color="steelblue", linewidth=0.8,
                    marker="o", markersize=3)
        step = TIME_PER_INDEX_MS * (pts.index[1] - pts.index[0]) if len(pts) > 1 else CHUNK_MS
        ax.set_title(f"{PARAM_LABELS.get(name, name)}  ({step:.0f} ms step)")
        ax.set_xlim(0, CHUNK_MS)
        ax.set_ylabel(f"{label_cap} value")
        ax.set_xlabel("Time [ms]")
        ax.grid(True, alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, f"{agg}_1000ms_chunk.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def _chunk_stats(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Compute median, mean, and percentile bands per time_index for the
    first 1000 ms, across all source files."""
    chunk = df[df["time_index"] < CHUNK_INDICES]
    grouped = chunk.groupby("time_index", observed=True)[PARAM_NAMES]
    return {
        "median": grouped.median(),
        "mean":   grouped.mean(),
        "p0":     grouped.min(),
        "p5":     grouped.quantile(0.05),
        "p25":    grouped.quantile(0.25),
        "p75":    grouped.quantile(0.75),
        "p95":    grouped.quantile(0.95),
        "p100":   grouped.max(),
    }


def plot_chunk_spread(stats: dict[str, pd.DataFrame]):
    """Combined spread plot: median (dark blue), mean (pink), IQR band
    (25th–75th), outlier range (5th–95th), full range (0th–100th).
    Layout: Loudness & Sharpness span the full width; Roughness, TNR and SII
    share one row at 1/3 width each."""
    fig = plt.figure(figsize=(12, 8))
    gs = fig.add_gridspec(3, 3)

    panels = {
        "loudness_zwtv":    fig.add_subplot(gs[0, :]),
        "sharpness_din_tv": fig.add_subplot(gs[1, :]),
        "roughness_dw":     fig.add_subplot(gs[2, 0]),
        "tnr_ecma_perseg":  fig.add_subplot(gs[2, 1]),
        "sii_ansi":         fig.add_subplot(gs[2, 2]),
    }
    short_names = {
        "loudness_zwtv": "Loudness",
        "sharpness_din_tv": "Sharpness",
        "roughness_dw": "Roughness",
        "tnr_ecma_perseg": "TNR",
        "sii_ansi": "SII",
    }
    bottom_row = {"roughness_dw", "tnr_ecma_perseg", "sii_ansi"}

    for name, ax in panels.items():
        idx = stats["median"][name].dropna().index.values
        ax.set_title(short_names[name])
        ax.grid(True, alpha=0.3)
        if len(idx) == 0:
            ax.set_xlim(0, CHUNK_MS)
            if name in bottom_row:
                ax.set_xlabel("Time [ms]")
            ax.set_ylabel("Value")
            continue

        if len(idx) == 1:
            ms = np.array([0.0, CHUNK_MS])
        else:
            ms = idx * TIME_PER_INDEX_MS

        def _align(key: str) -> np.ndarray:
            s = stats[key][name].reindex(idx).values.astype(float)
            if len(idx) == 1:
                return np.repeat(s, 2)
            return s

        median = _align("median")
        mean = _align("mean")
        p0 = _align("p0")
        p5 = _align("p5")
        p25 = _align("p25")
        p75 = _align("p75")
        p95 = _align("p95")
        p100 = _align("p100")

        if name in SCALE_CONFIG:
            scale = SCALE_CONFIG[name]["type"]
            if scale == "symlog":
                ax.set_yscale("symlog", linthresh=SCALE_CONFIG[name]["linthresh"])
                ax.set_yticks([0.0, 1.0, 10.0, 100.0])
                ax.set_yticklabels(["0", "1", "10", "100"])
                ax.set_ylim(bottom=0, top=max(float(np.nanmax(p100)), 100.0) * 1.05)
            elif scale == "log":
                median = np.maximum(median, LOG_FLOOR)
                mean = np.maximum(mean, LOG_FLOOR)
                p0 = np.maximum(p0, LOG_FLOOR)
                p5 = np.maximum(p5, LOG_FLOOR)
                p25 = np.maximum(p25, LOG_FLOOR)
                p75 = np.maximum(p75, LOG_FLOOR)
                p95 = np.maximum(p95, LOG_FLOOR)
                p100 = np.maximum(p100, LOG_FLOOR)
                ax.set_yscale("log")
                ax.set_ylim(bottom=LOG_FLOOR)

        ax.fill_between(ms, p0, p100, color="#cfcfcf", alpha=0.6, label="0th–100th")
        ax.fill_between(ms, p5, p95,  color="#9f9f9f", alpha=0.6, label="5th–95th")
        ax.fill_between(ms, p25, p75, color="#6f6f6f", alpha=0.7, label="25th–75th")
        ax.plot(ms, mean,   color="deeppink", linewidth=1.1, label="Mean")
        ax.plot(ms, median, color="darkblue", linewidth=1.3, label="Median")

        ax.set_xlim(0, CHUNK_MS)
        if name in bottom_row:
            ax.set_xlabel("Time [ms]")
        ax.set_ylabel("Value")
        legend_loc = "upper right" if name == "sii_ansi" else "best"
        ax.legend(fontsize=6, loc=legend_loc)

    fig.tight_layout()
    path = os.path.join(OUTPUT_DIR, "chunk_spread_1000ms.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


def main():
    print(f"Reading {CSV_PATH} ...")
    df = _load()
    print(f"Loaded {len(df):,} rows")

    plot_histograms(df)
    for agg in ["mean", "median"]:
        chunk = _chunk(df, agg)
        _save_chunk(chunk, agg)
        plot_chunk(chunk, agg)

    stats = _chunk_stats(df)
    plot_chunk_spread(stats)


if __name__ == "__main__":
    main()