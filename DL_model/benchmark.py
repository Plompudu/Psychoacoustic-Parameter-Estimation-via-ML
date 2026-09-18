import gc
import re
from datetime import datetime
from pathlib import Path
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


def benchmark_model(
    model: torch.nn.Module,
    inp: torch.Tensor,
    n_warmup: int = 500,
    n_measure: int = 10000,
) -> dict[str, float]:
    """Benchmark model(inp) with warmup and return timing statistics in ms."""
    with torch.no_grad():
        for _ in range(n_warmup):
            model(inp)

        gc.collect()
        gc.disable()
        times: list[float] = []
        for _ in range(n_measure):
            t0 = time.perf_counter()
            model(inp)
            t1 = time.perf_counter()
            times.append(t1 - t0)
        gc.enable()

    times_ms = [t * 1000 for t in times]
    avg = sum(times_ms) / len(times_ms)
    mn = min(times_ms)
    mx = max(times_ms)
    std = (sum((t - avg) ** 2 for t in times_ms) / len(times_ms)) ** 0.5
    sorted_times = sorted(times_ms)
    n = len(sorted_times)
    med = (sorted_times[n // 2] if n % 2 == 1 else (sorted_times[n // 2 - 1] + sorted_times[n // 2]) / 2)
    min_idx = times_ms.index(mn) + 1
    max_idx = times_ms.index(mx) + 1
    return {"avg_ms": avg, "min_ms": mn, "max_ms": mx, "std_ms": std, "median_ms": med,
            "min_idx": min_idx, "max_idx": max_idx, "n": n, "times_ms": times_ms}


def format_benchmark(stats: dict[str, float], label: str = "model_inference", device_name: str = "") -> str:
    prefix = f"  [{device_name}] " if device_name else "  "
    return (
        f"{prefix}{label:.<30s} avg {stats['avg_ms']:.3f} ms  "
        f"med {stats['median_ms']:.3f} ms  "
        f"min {stats['min_ms']:.3f} ms (#{int(stats['min_idx'])})  "
        f"max {stats['max_ms']:.3f} ms (#{int(stats['max_idx'])})  "
        f"std {stats['std_ms']:.3f} ms  (n={int(stats['n'])})"
    )


def plot_benchmark(stats: dict[str, float], output_path: Path, epoch_tag: str = "", device_name: str = ""):
    """Violin + jittered scatter plot of inference times with percentile lines."""
    times = sorted(stats["times_ms"])
    n = len(times)
    mn = times[0]
    p50 = times[n // 2] if n % 2 == 1 else (times[n // 2 - 1] + times[n // 2]) / 2
    p95 = times[int(n * 0.95)]
    p99 = times[int(n * 0.99)]
    mx = times[-1]

    cmap = plt.cm.magma
    colors = [cmap(v) for v in np.linspace(0.15, 0.85, 5)]

    fig, ax = plt.subplots(figsize=(5, 5))
    parts = ax.violinplot(times, positions=[0], showmeans=False, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("tab:blue")
        pc.set_alpha(0.25)

    jitter = np.random.default_rng(42).uniform(-0.08, 0.08, size=n)
    ax.scatter(jitter, times, s=6, alpha=0.5, color="tab:blue", zorder=3)

    ax.set_xlim(-0.5, 0.5)
    ax.set_xticks([])
    ax.set_yscale("log")
    ax.set_ylabel("Inference time (ms)")

    tick_vals, tick_labels = [], []
    for (label, val), color in zip(
        [("Max", mx), ("P99", p99), ("P95", p95), ("P50", p50), ("Min", mn)],
        colors,
    ):
        ax.axhline(val, color=color, linewidth=1.5, zorder=4,
                    label=f"{label} = {val:.3f} ms")
        tick_vals.append(val)
        tick_labels.append(f"{label}")

    ax2 = ax.twinx()
    ax2.set_yscale("log")
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(tick_vals)
    ax2.set_yticklabels(tick_labels)
    ax2.tick_params(axis="y", which="minor", right=False)
    ax2.spines["left"].set_visible(False)
    ax.legend(loc="upper left")
    parts = [epoch_tag, device_name]
    title = f"Inference benchmark  ({' — '.join(p for p in parts if p)})" if any(parts) else "Inference benchmark"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def save_benchmark_csv(output_dir: Path, stem: str, stats: dict[str, float], epoch_tag: str = "", device_name: str = ""):
    """Save per-iteration timing results to CSV."""
    prefix = f"{epoch_tag}_" if epoch_tag else ""
    rows = []
    for i, t in enumerate(stats["times_ms"], 1):
        rows.append({"iteration": i, "time_ms": round(t, 6)})
    df = pd.DataFrame(rows)
    summary = {
        "iteration": "summary",
        "time_ms": None,
        "device": device_name,
        "avg_ms": round(stats["avg_ms"], 6),
        "median_ms": round(stats["median_ms"], 6),
        "min_ms": round(stats["min_ms"], 6),
        "max_ms": round(stats["max_ms"], 6),
        "std_ms": round(stats["std_ms"], 6),
        "min_idx": int(stats["min_idx"]),
        "max_idx": int(stats["max_idx"]),
        "n": int(stats["n"]),
    }
    df = pd.concat([df, pd.DataFrame([summary])], ignore_index=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(output_dir / f"{prefix}{ts}_{stem}_benchmark_timing.csv", index=False)


def run_benchmark(
    dataset,
    checkpoint_dir: Path,
    output_dir: Path,
    n_benchmark: int = 50,
    device: torch.device | None = None,
    epoch: int | str = "newest",
    epoch_tag: str = "",
):
    """Run inference with a specific epoch checkpoint over ``dataset`` and save
    prediction outputs plus per-iteration runtime benchmark plots/CSVs."""
    from .train_model import (
        PsychoacousticModel,
        _BIAS_STATS_PATH,
        _enumerate_devices,
        _get_device_name,
        _load_time_biases,
        _save_comparison_csv,
        _save_prediction_plots,
        _save_runtime_csv,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    biases = _load_time_biases(_BIAS_STATS_PATH)

    if epoch == "newest":
        ckpt_files = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
        if not ckpt_files:
            print("No checkpoint found — skipping benchmark")
            return
        ckpt_path = ckpt_files[-1]
    else:
        ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint {ckpt_path.name} not found — skipping")
            return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not epoch_tag or epoch_tag == "newest":
        epoch_tag = ckpt_path.stem
    print(f"Loaded {ckpt_path.name} for benchmark")

    if device is not None:
        devices = [(device, _get_device_name(device))]
    else:
        devices = _enumerate_devices()

    for dev, dev_name in devices:
        print(f"  Benchmarking on {dev_name}...")
        model = PsychoacousticModel(initial_temporal_biases=biases).to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        safe_name = re.sub(r'[^\w\-]', '_', dev_name.strip().replace('\x00', ''))
        dev_tag = f"{epoch_tag}_{safe_name}" if epoch_tag else safe_name

        with torch.no_grad():
            for idx in range(len(dataset)):
                waveform, target = dataset[idx]
                inp = waveform.unsqueeze(0).to(dev)

                preds = model(inp)
                stats = benchmark_model(model, inp, n_measure=n_benchmark)
                backbone_out = model.backbone(inp)
                stem = dataset.stems[idx]

                meta = {
                    "file": f"{stem}.wav",
                    "device": dev_name,
                    "input_samples": inp.shape[-1],
                    "input_duration_s": round(inp.shape[-1] / 48000, 2),
                    "inference_time_avg_ms": round(stats["avg_ms"], 3),
                    "inference_time_median_ms": round(stats["median_ms"], 3),
                    "inference_time_min_ms": round(stats["min_ms"], 3),
                    "inference_time_min_idx": int(stats["min_idx"]),
                    "inference_time_max_ms": round(stats["max_ms"], 3),
                    "inference_time_max_idx": int(stats["max_idx"]),
                    "inference_time_std_ms": round(stats["std_ms"], 3),
                    "inference_time_n": int(stats["n"]),
                    "backbone_frames": backbone_out.shape[-1],
                }
                _save_runtime_csv(output_dir, stem, meta, preds, dev_tag, targets=target)
                _save_prediction_plots(output_dir, stem, preds, target, dev_tag)
                _save_comparison_csv(output_dir, stem, preds, target, dev_tag)
                prefix = f"{dev_tag}_" if dev_tag else ""
                plot_benchmark(stats, output_dir / f"{prefix}{stem}_benchmark.png", epoch_tag=epoch_tag, device_name=dev_name)
                save_benchmark_csv(output_dir, stem, stats, dev_tag, device_name=dev_name)

        print(f"  {dev_name}: {format_benchmark(stats, device_name=dev_name)}")
        del model
        if dev.type != "cpu":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    print(f"Benchmark saved to {output_dir}")


def main():
    repo_root = Path(__file__).resolve().parent.parent
    sound_dir = repo_root / "data" / "raw_sound_files_1s"
    references_path = repo_root / "data" / "reference_values" / "references.csv"
    checkpoint_dir = Path(__file__).resolve().parent / "epochs"
    output_dir = Path(__file__).resolve().parent / "comparison"

    from .train_model import PsychoAcousticDataset

    dataset = PsychoAcousticDataset(sound_dir, references_path)
    run_benchmark(dataset, checkpoint_dir, output_dir)


if __name__ == "__main__":
    main()