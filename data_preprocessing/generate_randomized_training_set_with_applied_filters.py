import numpy as np
from math import sin, cos, log10, sqrt, pi
from pathlib import Path
import soundfile as sf
import pandas as pd
import random
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"


def gain(signal, db_gain: float):
    """
    Simple gain in dB.
    """
    linear = 10 ** (db_gain / 20)
    return signal * linear




def _biquad_filter(x, b, a):
    """
    Direct Form I biquad filter.
    b = [b0, b1, b2], a = [1, a1, a2]
    """
    y = np.zeros_like(x, dtype=float)
    x1 = x2 = y1 = y2 = 0.0

    b0, b1, b2 = b
    a0, a1, a2 = a

    for i in range(len(x)):
        y[i] = (b0/a0)*x[i] + (b1/a0)*x1 + (b2/a0)*x2 \
               - (a1/a0)*y1 - (a2/a0)*y2

        x2, x1 = x1, x[i]
        y2, y1 = y1, y[i]

    return y


def peaking_eq(fs, f0, Q, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * pi * f0 / fs
    alpha = sin(w0) / (2 * Q)

    cosw0 = cos(w0)

    b0 = 1 + alpha * A
    b1 = -2 * cosw0
    b2 = 1 - alpha * A
    a0 = 1 + alpha / A
    a1 = -2 * cosw0
    a2 = 1 - alpha / A

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def _shelf_alpha(w0, A, S):
    return sin(w0) / 2 * sqrt(max(0, (A + 1/A) * (1/S - 1) + 2))


def high_shelf(fs, f0, S, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * pi * f0 / fs
    alpha = _shelf_alpha(w0, A, S)
    cosw0 = cos(w0)

    b0 = A * ((A+1) + (A-1)*cosw0 + 2*sqrt(A)*alpha)
    b1 = -2*A * ((A-1) + (A+1)*cosw0)
    b2 = A * ((A+1) + (A-1)*cosw0 - 2*sqrt(A)*alpha)
    a0 = (A+1) - (A-1)*cosw0 + 2*sqrt(A)*alpha
    a1 = 2 * ((A-1) - (A+1)*cosw0)
    a2 = (A+1) - (A-1)*cosw0 - 2*sqrt(A)*alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def low_shelf(fs, f0, S, gain_db):
    A = 10 ** (gain_db / 40)
    w0 = 2 * pi * f0 / fs
    alpha = _shelf_alpha(w0, A, S)
    cosw0 = cos(w0)

    b0 = A * ((A+1) - (A-1)*cosw0 + 2*sqrt(A)*alpha)
    b1 = 2*A * ((A-1) - (A+1)*cosw0)
    b2 = A * ((A+1) - (A-1)*cosw0 - 2*sqrt(A)*alpha)
    a0 = (A+1) + (A-1)*cosw0 + 2*sqrt(A)*alpha
    a1 = -2 * ((A-1) + (A+1)*cosw0)
    a2 = (A+1) + (A-1)*cosw0 - 2*sqrt(A)*alpha

    return np.array([b0, b1, b2]), np.array([a0, a1, a2])


def three_band_parametric_eq(signal, fs, bands):
    """
    bands: list of dicts like:
    {
        "type": "bell" | "low_shelf" | "high_shelf",
        "f0": frequency,
        "gain_db": gain,
        "Q": width
    }
    """
    filters = {
        "bell": peaking_eq,
        "low_shelf": low_shelf,
        "high_shelf": high_shelf,
    }

    out = signal.copy()

    for band in bands:
        bq, aq = filters[band["type"]](
            fs,
            band["f0"],
            band.get("Q", 1.0),
            band["gain_db"],
        )
        out = _biquad_filter(out, bq, aq)

    return out


def compressor(signal, threshold_db=-20, ratio=4, attack=0.01, release=0.1, fs=48000):
    """
    Simple feed-forward compressor using RMS detection.
    """
    signal = signal.astype(float)

    threshold = 10 ** (threshold_db / 20)

    attack_coeff = np.exp(-1 / (fs * attack))
    release_coeff = np.exp(-1 / (fs * release))

    env = 0.0
    out = np.zeros_like(signal)

    for i, x in enumerate(signal):
        x_abs = abs(x)

        # envelope follower
        if x_abs > env:
            env = attack_coeff * (env - x_abs) + x_abs
        else:
            env = release_coeff * (env - x_abs) + x_abs

        # gain computation
        if env > threshold:
            gain_db = -((20 * np.log10(env / threshold)) * (1 - 1/ratio))
            gain_lin = 10 ** (gain_db / 20)
        else:
            gain_lin = 1.0

        out[i] = x * gain_lin

    return out


def _process_sample(src_path_str, output_folder_str, config):
    t0 = time.perf_counter()
    signal, fs = sf.read(src_path_str)
    if signal.ndim > 1:
        signal = signal[:, 0]

    filtered = gain(signal, config["db_gain"])
    filtered = compressor(filtered, config["threshold_db"], config["ratio"],
                          config["attack"], config["release"], fs)

    bands = [
        {"type": config[f"type_{i}"], "f0": config[f"f0_{i}"],
         "gain_db": config[f"gain_db_{i}"], "Q": config[f"Q_{i}"]}
        for i in range(1, 4)
    ]
    filtered = three_band_parametric_eq(filtered, fs, bands)

    sf.write(Path(output_folder_str) / config["out_name"], filtered, fs)
    elapsed = time.perf_counter() - t0
    return config["out_name"], Path(src_path_str).name, elapsed


class _ColorStdout:
    def __init__(self, original):
        self.original = original

    def write(self, text):
        if text.startswith("[Warning]"):
            text = f"{BLUE}{text}{RESET}"
        self.original.write(text)

    def flush(self):
        self.original.flush()


def generate_randomized_training_set_with_applied_filters(input_folder, output_folder, number_of_samples):
    sys.stdout = _ColorStdout(sys.stdout)
    input_folder = Path(input_folder)
    output_folder = Path(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    wav_files = sorted(input_folder.glob("*.wav"))
    if not wav_files:
        print(f"{RED}No WAV files found in input folder.{RESET}")
        return

    DB_GAIN_RANGE = (-40, 00)
    THRESHOLD_DB_RANGE = (-20, 10)
    RATIO_RANGE = (1, 10)
    ATTACK_RANGE = (0.01, 0.1)
    RELEASE_RANGE = (0.1, 1)
    TYPE_RANGE = ["bell", "low_shelf", "high_shelf"]
    F0_RANGE = (20, 20000)
    GAIN_DB_RANGE = (-40, 40)
    Q_RANGE = (0.1, 10)

    # load existing metadata
    csv_path = output_folder / "filter_parameters.csv"
    existing_rows = []
    if csv_path.exists():
        existing_df = pd.read_csv(csv_path)
        existing_rows = existing_df.to_dict("records")
        print(f"Loaded {len(existing_rows)} existing entries from {csv_path}")

    existing_names = {r["filename"] for r in existing_rows}

    # determine next index per source
    next_idx_per_source = {}
    for f in output_folder.glob("*.wav"):
        stem = f.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2 and parts[1].isdigit():
            src_stem = parts[0]
            idx = int(parts[1])
            prev = next_idx_per_source.get(src_stem, 0)
            if idx >= prev:
                next_idx_per_source[src_stem] = idx + 1
    for src_stem in next_idx_per_source:
        if next_idx_per_source[src_stem] == 0:
            next_idx_per_source[src_stem] = 1

    configs = []
    total_existing = 0
    for src_path in wav_files:
        src_stem = src_path.stem
        next_idx = next_idx_per_source.get(src_stem, 1)
        existing_count = sum(1 for name in existing_names if name.startswith(f"{src_stem}_"))
        total_existing += existing_count
        needed = max(0, number_of_samples - existing_count)

        for local_idx in range(next_idx, next_idx + needed):
            out_name = f"{src_stem}_{local_idx:06d}.wav"
            configs.append({
                "source": src_path.name,
                "src_path": str(src_path),
                "out_name": out_name,
                "db_gain": random.uniform(*DB_GAIN_RANGE),
                "threshold_db": random.uniform(*THRESHOLD_DB_RANGE),
                "ratio": random.uniform(*RATIO_RANGE),
                "attack": random.uniform(*ATTACK_RANGE),
                "release": random.uniform(*RELEASE_RANGE),
                **{f"type_{i}": random.choice(TYPE_RANGE) for i in range(1, 4)},
                **{f"f0_{i}": random.uniform(*F0_RANGE) for i in range(1, 4)},
                **{f"gain_db_{i}": random.uniform(*GAIN_DB_RANGE) for i in range(1, 4)},
                **{f"Q_{i}": random.uniform(*Q_RANGE) for i in range(1, 4)},
            })

    total_configs = sum(max(0, number_of_samples) for _ in wav_files)
    print(f"Target: {total_configs} total ({number_of_samples} per source), "
          f"existing: {total_existing}, new: {len(configs)}")

    if not configs:
        print(f"{GREEN}All {total_configs} samples already exist.{RESET}")
        return

    output_folder_str = str(output_folder)

    with ProcessPoolExecutor(max_workers=10) as executor:
        fut_to_cfg = {}
        for cfg in configs:
            future = executor.submit(
                _process_sample, cfg["src_path"], output_folder_str, cfg
            )
            fut_to_cfg[future] = cfg

        new_rows = []
        for future in as_completed(fut_to_cfg):
            cfg = fut_to_cfg[future]
            try:
                out_name, src_name, elapsed = future.result()
                print(f"[{out_name}] {src_name}: {elapsed:.4f}s")
                row = {"filename": out_name}
                row.update({k: v for k, v in cfg.items() if k != "src_path"})
                new_rows.append(row)
            except Exception as e:
                print(f"{RED}[{cfg['out_name']}] Worker failed: {e}{RESET}", flush=True)

    all_rows = existing_rows + new_rows
    df = pd.DataFrame(all_rows)
    df.to_csv(csv_path, index=False)
    print(f"{GREEN}Saved: {csv_path}{RESET} ({len(all_rows)} total entries)")