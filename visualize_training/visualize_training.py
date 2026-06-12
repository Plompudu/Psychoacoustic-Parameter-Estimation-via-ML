from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

_PARAM_GRID = [
    ["loudness_zwtv", "sharpness_din_tv", "roughness_dw"],
    ["tnr_ecma_perseg", "sii_ansi", "total"],
]


def _plot_panel(df: pd.DataFrame, col: str, ax: plt.Axes):
    ax.plot(df["epoch"], df[col], marker=".", color="#00bfff")
    ax.set_title(col, color="white")
    ax.set_xlabel("Epoch", color="#cccccc")
    ax.set_ylabel("Loss", color="#cccccc")
    ax.set_yscale("log")
    ax.tick_params(colors="#cccccc")
    ax.grid(True, alpha=0.15)


def plot_losses(csv_path: Path, plot_path: Path | None = None):
    df = pd.read_csv(csv_path)
    plt.ion()
    plt.style.use("dark_background")

    fig = plt.figure(1)
    fig.clf()
    axes = fig.subplots(2, 3)
    fig.suptitle("Training Loss per Parameter", color="white")
    for row_idx, row_names in enumerate(_PARAM_GRID):
        for col_idx, name in enumerate(row_names):
            _plot_panel(df, name, axes[row_idx, col_idx])
    fig.set_size_inches(14, 7)
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
    if plot_path:
        fig.savefig(plot_path, facecolor="#1e1e1e")


def hold_plot():
    plt.ioff()
    plt.show(block=True)
