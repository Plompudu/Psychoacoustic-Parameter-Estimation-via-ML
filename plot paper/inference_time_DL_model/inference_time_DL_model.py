import sys
import os
import re
import time
import gc
import csv
import argparse
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import torch
import torch.nn as nn

# ============================================================
# Configuration
# ============================================================

BASE_DIR = os.path.dirname(__file__)

CSV_FIELDNAMES = [
    "device", "n_head_stages", "head_width", "params", "params_M",
    "memory_MiB", "vram_GiB", "vram_limit_mib", "min", "max", "median",
    "mean", "std", "dnf_reason", "dnf_message",
]

N_RUNS = 10
INPUT_SECONDS = 1.0
SR = 48000

TIME_LIMIT_MS = 1000
VRAM_HEADROOM = 1.00

# RAM / VRAM budget (MiB), evaluated separately per device:
#   CPU         -> 32 GB system RAM
#   iGPU        -> 512 MB VRAM carve-out
#   external GPU-> 16 GB VRAM
# A device name listed here wins; unknown names fall back to a heuristic
# (integrated-looking names -> 512 MiB, everything else -> 16 GiB).
VRAM_LIMIT_MIB_BY_NAME = {
    "CPU": 32 * 1024,
    "AMD Radeon(TM) Graphics": 512,      # iGPU
    "AMD Radeon RX 7800 XT": 16 * 1024,  # external GPU
}
IGPU_NAME_MARKERS = (
    "radeon(tm) graphics", "intel", "uhd graphics", "hd graphics",
    "iris", "integrated", "vega",
)

# Hardcoded sweep: both parameters run over 2**0 ... 2**50.
# head width must stay >= 1 (Conv1d(80, 0) is invalid), the # head stages
# sweep additionally includes a standalone 0 (backbone heads only).
WIDTH_EXP_MIN, WIDTH_EXP_MAX = 1, 15
STAGES_EXP_MIN, STAGES_EXP_MAX = 0, 20

HEAD_WIDTHS = [2**i for i in range(WIDTH_EXP_MIN, WIDTH_EXP_MAX + 1)]
N_HEAD_STAGES = [0] + [2**i for i in range(STAGES_EXP_MIN, STAGES_EXP_MAX + 1)]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark inference time vs head width / # head stages."
    )
    parser.add_argument(
        "--n-runs", type=int, default=N_RUNS,
        help=f"number of timed inference runs per configuration "
             f"(default: {N_RUNS})",
    )
    return parser.parse_args()


# ============================================================
# Detect available devices (CPU + every usable GPU: iGPU + external)
# ============================================================

cpu_device = torch.device("cpu")
cpu_name = "CPU"

devices = [("CPU", cpu_device, cpu_name)]


def clean_device_name(raw):
    return raw.replace("\x00", "").strip()


def device_vram_limit_mib(device_name):
    """Memory budget (MiB) that a device may use for the model + inference."""
    name = clean_device_name(device_name)
    if name in VRAM_LIMIT_MIB_BY_NAME:
        return VRAM_LIMIT_MIB_BY_NAME[name]
    if name.upper() == "CPU":
        return 32 * 1024
    low = name.lower()
    if any(marker in low for marker in IGPU_NAME_MARKERS):
        return 512
    return 16 * 1024


def enumerate_gpu_devices():
    gpus = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append((torch.device(f"cuda:{i}"), torch.cuda.get_device_name(i)))
    else:
        try:
            import torch_directml
            for i in range(torch_directml.device_count()):
                gpus.append((torch_directml.device(i), torch_directml.device_name(i)))
        except ImportError:
            pass
    return gpus


for idx, (dev, raw_name) in enumerate(enumerate_gpu_devices(), start=1):
    name = clean_device_name(raw_name)
    low = name.lower()
    # skip software renderers / non-accelerated fallback adapters
    if not name or "microsoft" in low or "basic render" in low or "remote display" in low:
        print(f"  skipping non-accelerated render adapter: {raw_name!r}")
        continue
    devices.append((f"GPU_{idx}", dev, name))

print("Detected devices:")
for label, dev, name in devices:
    print(f"  {label}: {name} ({dev})  "
          f"limit={device_vram_limit_mib(name):.0f} MiB")
print()


def device_slug(device_name):
    name = clean_device_name(device_name)
    if name.upper() == "CPU":
        return "CPU"
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or "GPU"


# ============================================================
# Model factory — fixed backbone, variable-width heads
# ============================================================

def make_model(head_width: int, n_head_stages: int = 3) -> nn.Module:
    """
    DL_model.py backbone (5 stages, fixed) with parametric heads.

    Each of the 5 parameter heads gets:
      Conv1d(80, head_width, k=1) + BN + ReLU
      n_head_stages × [Conv1d(head_width, head_width, k=1) + BN + ReLU]
      Conv1d(head_width, 1, k=1) + AdaptiveAvgPool1d(MAX[i])

    Each head also has a preloaded 1×MAX[i] bias (like DL_model.py) that is
    added to the head output, so measurement includes the bias add.

    head_width: input/output channel count of the head Conv1d layers
    n_head_stages: number of head_width→head_width stages inside each head
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
        layers = [
            nn.Conv1d(80, head_width, kernel_size=1),
            nn.BatchNorm1d(head_width),
            nn.ReLU(),
        ]
        for _ in range(n_head_stages):
            layers.extend([
                nn.Conv1d(head_width, head_width, kernel_size=1),
                nn.BatchNorm1d(head_width),
                nn.ReLU(),
            ])
        layers.extend([
            nn.Conv1d(head_width, 1, kernel_size=1),
            nn.AdaptiveAvgPool1d(MAX[i]),
        ])
        heads[str(i)] = nn.Sequential(*layers)

    class PsychoacousticModel(nn.Module):
        def __init__(self, backbone, heads):
            super().__init__()
            self.backbone = backbone
            self.heads = heads
            for i in range(5):
                self.register_parameter(f"{i}_bias", nn.Parameter(torch.zeros(1, MAX[i])))

        def forward(self, x):
            x = self.backbone(x)
            return {k: h(x).squeeze(1) + getattr(self, f"{k}_bias")
                    for k, h in self.heads.items()}

    return PsychoacousticModel(backbone, heads)


# ============================================================
# Analytic model-size estimate (no model building needed)
# ============================================================

def estimate_model_params(head_width, n_head_stages):
    """
    Exact parameter count of make_model(head_width, n_head_stages).

    Backbone contributes 466_550 parameters:
      5 conv stages: 5,130 + 51,220 + 102,440 + 153,660 + 153,680 = 466,130
      5 BatchNorm gains/biases: 2 * (10+20+40+60+80)            =     420
    The five preloaded output biases contribute 1,013
    (MAX = [500, 500, 10, 2, 1] -> 500+500+10+2+1).

    Each of the 5 parametric heads decomposes as:
      Conv1d(80 -> w) + BN + ReLU               -> 81w + 2w     = 83w
      n_head_stages x (Conv1d(w -> w) + BN + ReLU)
                                                 -> s*(w^2 + 3w)
      Conv1d(w -> 1)                            -> w + 1
    so per head = 84w + 1 + s*(w^2 + 3w).
    """
    backbone_params = 466_550
    bias_params = 1_013
    per_head = 84 * head_width + 1 + n_head_stages * (head_width**2 + 3 * head_width)
    return backbone_params + bias_params + 5 * per_head


def estimate_model_bytes(head_width, n_head_stages):
    """
    Exact fp32 storage size of make_model(head_width, n_head_stages) in bytes.

    4 bytes per parameter plus the BatchNorm buffers that hold no gradients:
      running_mean/running_var: (420 + 10*(1+s)*w) elements
        - backbone: 2 buffers x (10+20+40+60+80) channels = 420 elements
        - heads: 5 heads x (1+s) BatchNorms x 2 buffers x w channels
      num_batches_tracked: 5 + 5*(1+s) scalar int64s, 8 bytes each
    """
    params = estimate_model_params(head_width, n_head_stages)
    bn_stat_params = 420 + 10 * (1 + n_head_stages) * head_width
    n_batches_tracked = 5 + 5 * (1 + n_head_stages)
    return 4 * params + 4 * bn_stat_params + 8 * n_batches_tracked


def estimate_model_mib(head_width, n_head_stages):
    """Exact model storage size in MiB (fp32 params + BN buffers)."""
    return estimate_model_bytes(head_width, n_head_stages) / 1024**2


def pre_run_model_size_check(head_widths, head_stage_counts, vram_limit_mib):
    """
    Pre-run: estimate the model size of every (width, stages) combination
    WITHOUT building the model. Combinations whose estimated size reaches
    the VRAM limit are skipped in the inference-time benchmark.
    """
    oom_set = set()
    for head_width in head_widths:
        for n_head_stages in head_stage_counts:
            if estimate_model_mib(head_width, n_head_stages) >= vram_limit_mib:
                oom_set.add((head_width, n_head_stages))
    return oom_set


# ============================================================
# Benchmark one device
# ============================================================

class OverLimitError(Exception):
    pass


class VramLimitError(Exception):
    pass


def benchmark_device(device, device_name, head_stage_counts, head_widths, n_runs,
                     oom_set, vram_limit_mib):
    n_samples = int(SR * INPUT_SECONDS)
    rows = []

    # Once running fails part-way through a sweep, larger stages of this width
    # and of every wider head are skipped (blocked) without running them.
    # VRAM (oom) blocks and time blocks are tracked independently so that a
    # time limit can never overwrite a VRAM limit: oom reasons always win.
    vram_block_stage = None
    time_block_stage = None

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

    def bench_one(head_width, n_head_stages):
        model = make_model(head_width, n_head_stages)
        total_params = sum(p.numel() for p in model.parameters())
        total_bytes = (
            sum(p.numel() * p.element_size() for p in model.parameters())
            + sum(b.numel() * b.element_size() for b in model.buffers())
        )

        model = model.to(device)
        model.eval()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        dummy_input = torch.randn(1, 1, n_samples, device=device)

        def infer():
            return model(dummy_input)

        with torch.no_grad():
            for _ in range(5):
                time_fn(infer)

        times = []
        with torch.no_grad():
            for _ in range(n_runs):
                t = time_fn(infer)
                if t > TIME_LIMIT_MS:
                    raise OverLimitError(f"run took {t:.2f} ms (> {TIME_LIMIT_MS} ms)")
                times.append(t)

        times = np.array(times)

        vram_after = 0.0
        if device.type == "cuda":
            vram_after = torch.cuda.max_memory_allocated(device) / 1024**3

        stats = {
            "device": device_name,
            "n_head_stages": n_head_stages,
            "head_width": head_width,
            "params": total_params,
            "params_M": total_params / 1e6,
            "memory_MiB": total_bytes / 1024**2,
            "vram_GiB": vram_after,
            "vram_limit_mib": vram_limit_mib,
            "min": np.min(times),
            "max": np.max(times),
            "median": np.median(times),
            "mean": np.mean(times),
            "std": np.std(times),
            "dnf_reason": "",
            "dnf_message": "",
        }

        del model, dummy_input
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()
        return stats

    def dnf_row(head_width, n_head_stages, reason, message=""):
        return {
            "device": device_name,
            "n_head_stages": n_head_stages,
            "head_width": head_width,
            "params": estimate_model_params(head_width, n_head_stages),
            "params_M": estimate_model_params(head_width, n_head_stages) / 1e6,
            "memory_MiB": estimate_model_mib(head_width, n_head_stages),
            "vram_GiB": "",
            "vram_limit_mib": vram_limit_mib,
            "min": "",
            "max": "",
            "median": "",
            "mean": "",
            "std": "",
            "dnf_reason": reason,
            "dnf_message": " ".join(str(message).split()),
        }

    for head_width in head_widths:
        for n_head_stages in head_stage_counts:
            reason = None
            if (head_width, n_head_stages) in oom_set:
                reason = "oom"
            elif vram_block_stage is not None and n_head_stages >= vram_block_stage:
                reason = "oom"
            elif time_block_stage is not None and n_head_stages >= time_block_stage:
                reason = "time"

            if reason is not None:
                if reason == "oom":
                    if (head_width, n_head_stages) in oom_set:
                        msg = (f"pre-run: model size >= {vram_limit_mib:.0f} MiB "
                               f"(est. {estimate_model_mib(head_width, n_head_stages):.0f} MiB)")
                        print(f"  [{device_name}] stages={n_head_stages} width={head_width} ... "
                              f"skipped ({msg})")
                    else:
                        msg = f"blocked: lower width failed at stages={vram_block_stage} (vram limit)"
                        print(f"  [{device_name}] stages={n_head_stages} width={head_width} ... "
                              f"blocked (vram limit: lower width failed at stages={vram_block_stage})")
                else:
                    msg = f"blocked: lower width failed at stages={time_block_stage} (time limit)"
                    print(f"  [{device_name}] stages={n_head_stages} width={head_width} ... "
                          f"blocked (time limit: lower width failed at stages={time_block_stage})")
                rows.append(dnf_row(head_width, n_head_stages, reason, msg))
                continue

            print(
                f"  [{device_name}] stages={n_head_stages} width={head_width} ...",
                end=" ", flush=True,
            )

            try:
                if device.type != "cpu":
                    estimated_bytes = estimate_model_bytes(
                        head_width, n_head_stages
                    )
                    mem_prev = [
                        (r["vram_GiB"] * 1024, r["memory_MiB"])
                        for r in rows
                        if r["device"] == device_name
                        and r["head_width"] == head_width
                        and r["dnf_reason"] == ""
                    ]
                    if mem_prev:
                        last_peak, last_mem = mem_prev[-1]
                        predicted_mib = max(last_peak, last_mem) * (
                            estimated_bytes / (last_mem * 1024**2)
                        )
                        if predicted_mib >= vram_limit_mib:
                            raise VramLimitError(
                                f"predicted {predicted_mib:.0f} MiB "
                                f">= {vram_limit_mib:.0f} MiB limit"
                            )

                stats = bench_one(head_width, n_head_stages)
            except VramLimitError as exc:
                print(f"\n  [{device_name}] stages={n_head_stages} width={head_width}")
                print(f"    DNF at stages={n_head_stages} "
                      f"(predicted VRAM over limit: {exc})")
                print("    stopping this width")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                rows.append(dnf_row(head_width, n_head_stages, "oom", exc))
                if vram_block_stage is None or n_head_stages < vram_block_stage:
                    vram_block_stage = n_head_stages
                continue
            except RuntimeError as exc:
                print(f"\n  [{device_name}] stages={n_head_stages} width={head_width}")
                print(f"    DNF at stages={n_head_stages} (resource exhaust: {exc})")
                print("    continuing with the next combination")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                rows.append(dnf_row(head_width, n_head_stages, "oom", exc))
                if vram_block_stage is None or n_head_stages < vram_block_stage:
                    vram_block_stage = n_head_stages
                continue
            except OverLimitError as exc:
                print(f"\n  [{device_name}] stages={n_head_stages} width={head_width}")
                print(f"    DNF at stages={n_head_stages} (first run over limit: {exc})")
                print("    stopping this width")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                rows.append(dnf_row(head_width, n_head_stages, "time", exc))
                if time_block_stage is None or n_head_stages < time_block_stage:
                    time_block_stage = n_head_stages
                continue
            except Exception as exc:
                print(f"\n  [{device_name}] stages={n_head_stages} width={head_width}")
                print(f"    DNF at stages={n_head_stages} (device error: {exc})")
                print("    continuing with the next combination")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
                rows.append(dnf_row(head_width, n_head_stages, "oom", exc))
                if vram_block_stage is None or n_head_stages < vram_block_stage:
                    vram_block_stage = n_head_stages
                continue

            rows.append(stats)

            print(
                f"params={stats['params']:>12,} ({stats['params']/1e6:.3f} M)  "
                f"VRAM={stats['memory_MiB']:>8.1f} MiB  "
                f"median={stats['median']:>8.2f} ms  "
                f"std={stats['std']:>6.2f} ms"
            )

    return rows


# ============================================================
# Main
# ============================================================

def main():
    args = parse_args()

    print(f"Head width sweep: 2**{WIDTH_EXP_MIN} .. 2**{WIDTH_EXP_MAX} "
          f"(n={len(HEAD_WIDTHS)})")
    print(f"# head stages sweep: 0, 2**{STAGES_EXP_MIN} .. 2**{STAGES_EXP_MAX} "
          f"(n={len(N_HEAD_STAGES)})")
    print()

    vram_headroom = VRAM_HEADROOM

    all_results = []

    for label, dev, name in devices:
        vram_limit_mib = device_vram_limit_mib(name) * vram_headroom
        print(f"[{name}] memory limit: {vram_limit_mib:.0f} MiB "
              f"(={vram_limit_mib / 1024:.1f} GiB)")
        print(f"Pre-run: estimating model size for "
              f"{len(HEAD_WIDTHS) * len(N_HEAD_STAGES)} combinations ...")
        oom_set = pre_run_model_size_check(HEAD_WIDTHS, N_HEAD_STAGES, vram_limit_mib)
        print(f"  {len(oom_set)} combination(s) estimate to >= "
              f"{vram_limit_mib:.0f} MiB and will be skipped in the inference benchmark")
        print()
        print(f"Benchmarking on {name}...")
        dev_results = []
        try:
            dev_results = benchmark_device(dev, name, N_HEAD_STAGES, HEAD_WIDTHS,
                                           args.n_runs, oom_set, vram_limit_mib)
        except Exception as exc:
            msg = " ".join(str(exc).split())
            print(f"\n[{name}] device aborted mid-benchmark: {msg}")
            print(f"[{name}] recording aborted status and continuing "
                  f"with the next device")
            # mark every remaining combination as dnf so the CSV still lists
            # all (device, width, stages) combinations
            for head_width in HEAD_WIDTHS:
                for n_head_stages in N_HEAD_STAGES:
                    params = estimate_model_params(head_width, n_head_stages)
                    in_pre_oom = (head_width, n_head_stages) in oom_set
                    dev_results.append({
                        "device": name,
                        "n_head_stages": n_head_stages,
                        "head_width": head_width,
                        "params": params,
                        "params_M": params / 1e6,
                        "memory_MiB": estimate_model_mib(head_width, n_head_stages),
                        "vram_GiB": "",
                        "vram_limit_mib": vram_limit_mib,
                        "min": "", "max": "", "median": "", "mean": "", "std": "",
                        "dnf_reason": "oom" if in_pre_oom else "device_error",
                        "dnf_message": ("" if in_pre_oom
                                        else f"device aborted: {msg}"),
                    })
        all_results.extend(dev_results)
        print()

    csv_path = Path(__file__).parent / "VRAM vs Inferencetime.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(all_results)

    print(f"CSV saved: {csv_path} ({len(all_results)} rows: every "
          f"(device, head_width, n_head_stages) combination)")

    # one CSV per device
    for label, dev, name in devices:
        dev_rows = [r for r in all_results if r["device"] == name]
        dev_csv = csv_path.with_name(f"{csv_path.stem} [{device_slug(name)}].csv")
        with open(dev_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            writer.writeheader()
            writer.writerows(dev_rows)
        print(f"Device CSV saved: {dev_csv} ({len(dev_rows)} rows)")

    # generate per-device + combined plots automatically
    plot_script = Path(__file__).parent / "plot results.py"
    if plot_script.exists():
        print(f"Generating plots via {plot_script.name} ...")
        import runpy
        old_argv = list(sys.argv)
        sys.argv = [str(plot_script)]
        try:
            runpy.run_path(str(plot_script), run_name="__main__")
        finally:
            sys.argv = old_argv
    else:
        print(f"Run `plot results.py` to generate plots from these CSVs.")


if __name__ == "__main__":
    main()