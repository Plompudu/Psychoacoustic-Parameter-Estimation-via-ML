import sys
import os
import time
import csv
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(__file__)

N_RUNS = 50
INPUT_SECONDS = 1.0
SR = 48000

# ============================================================
# Detect available devices
# ============================================================

cpu_device = torch.device("cpu")
cpu_name = "CPU"

gpu_device = None
gpu_name = None

if torch.cuda.is_available():
    gpu_device = torch.device("cuda")
    gpu_name = torch.cuda.get_device_name(0)
else:
    try:
        import torch_directml
        gpu_device = torch_directml.device()
        gpu_name = torch_directml.device_name(0)
    except ImportError:
        pass

devices = [("CPU", cpu_device, cpu_name)]
if gpu_device is not None:
    devices.append(("GPU", gpu_device, gpu_name))

print("Detected devices:")
for label, dev, name in devices:
    print(f"  {label}: {name} ({dev})")
print()


# ============================================================
# Model factory — fixed backbone, variable heads
# ============================================================

def make_model(n_head_stages: int) -> nn.Module:
    """
    DL_model.py backbone (5 stages, fixed) with variable
    number of 80→80 head-stages per parameter head.

    Each of the 5 parameter heads gets:
      n_head_stages × [Conv1d(80, 80, k=1) + BN + ReLU]
      then Conv1d(80, 1, k=1) + AdaptiveAvgPool1d(MAX[i])

    n_head_stages: how many 80→80 layers per head (0 = direct 80→1)
    """
    MAX = [500, 500, 10, 2, 1]

    backbone = nn.Sequential(
        nn.Conv1d(1, 10, kernel_size=512, stride=1),
        nn.BatchNorm1d(10),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(10, 20, kernel_size=256, stride=1),
        nn.BatchNorm1d(20),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(20, 40, kernel_size=128, stride=1),
        nn.BatchNorm1d(40),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(40, 60, kernel_size=64, stride=1),
        nn.BatchNorm1d(60),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),

        nn.Conv1d(60, 80, kernel_size=32, stride=1),
        nn.BatchNorm1d(80),
        nn.ReLU(),
        nn.MaxPool1d(kernel_size=2, stride=2),
    )

    heads = nn.ModuleDict()
    for i in range(5):
        layers = []
        for j in range(n_head_stages):
            layers.append(nn.Conv1d(80, 80, kernel_size=1))
            layers.append(nn.BatchNorm1d(80))
            layers.append(nn.ReLU())
        layers.append(nn.Conv1d(80, 1, kernel_size=1))
        layers.append(nn.AdaptiveAvgPool1d(MAX[i]))
        heads[str(i)] = nn.Sequential(*layers)

    class PsychoacousticModel(nn.Module):
        def __init__(self, backbone, heads):
            super().__init__()
            self.backbone = backbone
            self.heads = heads

        def forward(self, x):
            x = self.backbone(x)
            return {k: h(x).squeeze(1) for k, h in self.heads.items()}

    return PsychoacousticModel(backbone, heads)


# ============================================================
# Benchmark one device
# ============================================================

def benchmark_device(device, device_name, head_stage_counts):
    n_samples = int(SR * INPUT_SECONDS)
    results = []

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize()

    def time_fn(fn):
        t0 = time.perf_counter()
        result = fn()
        if isinstance(result, dict):
            v = next(iter(result.values()))
            if isinstance(v, torch.Tensor):
                v.reshape(-1)[0].item()
        elif isinstance(result, torch.Tensor):
            result.reshape(-1)[0].item()
        sync()
        return (time.perf_counter() - t0) * 1000

    for n_head_stages in head_stage_counts:
        print(f"  [{device_name}] head_stages={n_head_stages} ...", end=" ", flush=True)

        model = make_model(n_head_stages)
        model = model.to(device)
        model.eval()

        total_params = sum(p.numel() for p in model.parameters())
        total_bytes = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) + sum(b.numel() * b.element_size() for b in model.buffers())

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        dummy_input = torch.randn(1, 1, n_samples, device=device)

        def infer():
            return model(dummy_input)

        with torch.no_grad():
            for _ in range(5):
                time_fn(infer)

        with torch.no_grad():
            times = [time_fn(infer) for _ in range(N_RUNS)]

        times = np.array(times)

        vram_after = 0.0
        if device.type == "cuda":
            vram_after = torch.cuda.max_memory_allocated(device) / 1024**3

        stats = {
            "device": device_name,
            "n_head_stages": n_head_stages,
            "params": total_params,
            "params_M": total_params / 1e6,
            "memory_MiB": total_bytes / 1024**2,
            "vram_GiB": vram_after,
            "min": np.min(times),
            "max": np.max(times),
            "median": np.median(times),
            "mean": np.mean(times),
            "std": np.std(times),
        }
        results.append(stats)

        print(
            f"params={total_params:>12,} ({total_params/1e6:.3f} M)  "
            f"VRAM={stats['memory_MiB']:>8.1f} MiB  "
            f"median={stats['median']:>8.2f} ms  "
            f"std={stats['std']:>6.2f} ms"
        )

        del model, dummy_input
        if device.type == "cuda":
            torch.cuda.empty_cache()

    return results


# ============================================================
# Run benchmarks
# ============================================================

head_stage_counts = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
all_results = []

for label, dev, name in devices:
    print(f"Benchmarking on {name}...")
    all_results.extend(benchmark_device(dev, name, head_stage_counts))
    print()


# ============================================================
# Save CSV
# ============================================================

csv_path = Path(__file__).parent / "VRAM vs Inferencetime.csv"

with open(csv_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=[
        "device", "n_head_stages", "params", "params_M", "memory_MiB", "vram_GiB",
        "min", "max", "median", "mean", "std",
    ])
    writer.writeheader()
    writer.writerows(all_results)

print(f"CSV saved: {csv_path}")


# ============================================================
# Plotting helpers
# ============================================================

COLORS = {"CPU": "steelblue", "GPU": "pink"}

def plot_single(ax, results, label, ylim=None):
    x = list(range(len(results)))
    medians = [r["median"] for r in results]
    mins = [r["min"] for r in results]
    maxs = [r["max"] for r in results]
    stds = [r["std"] for r in results]
    tick_labels = [str(r["n_head_stages"]) for r in results]
    color = COLORS[label]

    ax.plot(x, medians, "o-", color=color, linewidth=2, markersize=8, label="Median")
    ax.fill_between(x, mins, maxs, alpha=0.15, color=color, label="Min–Max")
    ax.fill_between(
        x,
        [m - s for m, s in zip(medians, stds)],
        [m + s for m, s in zip(medians, stds)],
        alpha=0.3, color=color, label="±1 Std",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Number of Conv1D 80→80 per head", fontsize=11)
    ax.set_ylabel("Inference Time [ms]", fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    if ylim is not None:
        ax.set_ylim(ylim)

    for ref in (1000, 125, 35, 2):
        ax.axhline(y=ref, color="grey", linestyle=":", linewidth=1.2, zorder=0)
    ax.set_yticks(list(ax.get_yticks()) + [2, 35, 125, 1000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    if ylim is not None:
        ax.set_ylim(ylim)

    header = f"{'Stages':>6} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8}"
    sep = "-" * len(header)
    rows = [header, sep]
    for r in results:
        rows.append(
            f"{r['n_head_stages']:>6} {r['median']:>7.2f}ms {r['std']:>7.2f}ms"
            f" {r['min']:>7.2f}ms {r['max']:>7.2f}ms"
        )
    table_text = "\n".join(rows)
    props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
    if label == "CPU":
        tx, ty, va = 0.02, 0.05, "bottom"
    else:
        tx, ty, va = 0.02, 0.95, "top"
    ax.text(tx, ty, table_text, transform=ax.transAxes, fontsize=9,
            fontfamily="monospace", verticalalignment=va, horizontalalignment="left", bbox=props)


# ============================================================
# Shared y-axis limits
# ============================================================

all_medians = [r["median"] for r in all_results]
all_mins = [r["min"] for r in all_results]
all_maxs = [r["max"] for r in all_results]
ylim = (1, 1250)

# ============================================================
# Individual plots
# ============================================================

for label, dev, name in devices:
    dev_results = [r for r in all_results if r["device"] == name]
    fig, ax = plt.subplots(figsize=(10, 6))
    plot_single(ax, dev_results, label, ylim=ylim)
    plt.tight_layout()
    out = Path(__file__).parent / f"VRAM vs Inferencetime_{label}.png"
    fig.savefig(out, dpi=300)
    print(f"Plot saved: {out}")


# ============================================================
# Combined plot
# ============================================================

if len(devices) > 1:
    fig, ax = plt.subplots(figsize=(10, 6))
    x_positions = list(range(len(head_stage_counts)))

    for idx, (label, dev, name) in enumerate(devices):
        dev_results = [r for r in all_results if r["device"] == name]
        color = COLORS[label]
        x = x_positions
        medians = [r["median"] for r in dev_results]
        mins = [r["min"] for r in dev_results]
        maxs = [r["max"] for r in dev_results]
        stds = [r["std"] for r in dev_results]
        color = COLORS[label]

        ax.plot(x, medians, "o-", color=color, linewidth=2, markersize=8, label=f"{name} median")
        ax.fill_between(x, mins, maxs, alpha=0.15, color=color)
        ax.fill_between(
            x,
            [m - s for m, s in zip(medians, stds)],
            [m + s for m, s in zip(medians, stds)],
            alpha=0.3, color=color,
        )

    tick_labels = [str(h) for h in head_stage_counts]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(tick_labels, fontsize=10)
    ax.set_xlabel("Number of Conv1D 80→80 per head", fontsize=11)
    ax.set_ylabel("Inference Time [ms]", fontsize=11)
    ax.set_yscale("log")
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_ylim(ylim)
    for ref in (1000, 125, 35, 2):
        ax.axhline(y=ref, color="grey", linestyle=":", linewidth=1.2, zorder=0)
    ax.set_yticks(list(ax.get_yticks()) + [2, 35, 125, 1000])
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, p: f"{v:g}"))
    ax.set_ylim(ylim)
    plt.tight_layout()
    out = Path(__file__).parent / "VRAM vs Inferencetime_combined.png"
    fig.savefig(out, dpi=300)
    print(f"Plot saved: {out}")
