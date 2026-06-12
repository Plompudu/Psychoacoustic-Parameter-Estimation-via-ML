from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_losses(csv_path: Path, plot_path: Path | None = None):
    df = pd.read_csv(csv_path)
    plt.ion()
    fig = plt.gcf()
    fig.clf()
    ax = fig.add_subplot()
    for col in df.columns:
        if col != "epoch":
            ax.plot(df["epoch"], df[col], label=col, marker=".")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.set_title("Training Loss")
    ax.legend()
    plt.tight_layout()
    plt.draw()
    plt.pause(0.01)
    if plot_path:
        fig.savefig(plot_path)


def hold_plot():
    plt.ioff()
    plt.show(block=True)
