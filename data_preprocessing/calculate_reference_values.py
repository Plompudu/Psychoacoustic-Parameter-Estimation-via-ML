from pathlib import Path
import numpy as np
import pandas as pd
import soundfile as sf
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

from mosqito import (
    loudness_zwtv,
    sharpness_din_tv,
    roughness_dw,
    tnr_ecma_perseg,
    sii_ansi
)

SHARPNESS_SKIP_FIRST_N_VALUES = 5
PARAM_CONFIGS = [
    ("loudness_zwtv", loudness_zwtv, (), {}),
    ("sharpness_din_tv", sharpness_din_tv, (), {"skip": 0.002 * SHARPNESS_SKIP_FIRST_N_VALUES}), #0.002 = one segment
    ("roughness_dw", roughness_dw, (), {}),
    ("tnr_ecma_perseg", tnr_ecma_perseg, (), {}),
    ("sii_ansi", sii_ansi, ("critical", "normal"), {}),
]


def _pad(arr, target_len, sharpness_prepend_SHARPNESS_SKIP_FIRST_N_VALUES=False):
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= target_len:
        return arr[:target_len]
    pad_width = target_len - len(arr)
    if sharpness_prepend_SHARPNESS_SKIP_FIRST_N_VALUES:
        return np.pad(arr, (SHARPNESS_SKIP_FIRST_N_VALUES, 0), constant_values=np.nan)
    return np.pad(arr, (0, pad_width), constant_values=np.nan)


def _compute_param(name, param_name, fn, audio_path_str, *args, **kwargs):
    audio_path = Path(audio_path_str)
    signal, sr = sf.read(audio_path)
    np.seterr(invalid='ignore', divide='ignore')

    if param_name == "sharpness_din_tv" and "_00000-" in audio_path.stem:
        kwargs = {k: v for k, v in kwargs.items() if k != "skip"}

    t0 = time.perf_counter()
    try:
        result, *_ = fn(signal, sr, *args, **kwargs)
        elapsed = time.perf_counter() - t0
        print(f"[{name}] {fn.__name__}: {elapsed:.4f}s")
        result = np.asarray(result)
        if result.ndim == 0:
            result = result.reshape(1)
        return name, param_name, result, None
    except Exception as e:
        elapsed = time.perf_counter() - t0
        duration = len(signal) / sr
        print(
            f"{RED}[{name}] {fn.__name__}: FAILED ({e}) "
            f"- signal too short ({duration:.2f}s){RESET}",
            flush=True,
        )
        return name, param_name, None, str(e)


class _ColorStdout:
    def __init__(self, original):
        self.original = original

    def write(self, text):
        if text.startswith("[Warning]"):
            text = f"{BLUE}{text}{RESET}"
        self.original.write(text)

    def flush(self):
        self.original.flush()


def calculate_reference_values(input_folder: Path, output_folder: Path):
    """
    Compute time-varying psychoacoustic features per mono audio file
    and store each result as a separate CSV file.

    Assumes input files are already mono (e.g., *_ch1.wav, *_ch2.wav).
    All parameter computations across all files are submitted to a
    10-process pool and run concurrently.
    """
    sys.stdout = _ColorStdout(sys.stdout)
    output_folder.mkdir(parents=True, exist_ok=True)

    audio_path_strs = [str(p) for p in sorted(input_folder.glob("*.wav"))]

    with ProcessPoolExecutor(max_workers=10) as executor:
        fut_to_info = {}
        submitted_count = {}
        for p_str in audio_path_strs:
            name = Path(p_str).stem
            output_file = output_folder / f"{name}.csv"
            if output_file.exists():
                print(f"Skipping (already exists): {output_file.name}")
                continue
            submitted_count[name] = 0
            for param_name, fn, args, kwargs in PARAM_CONFIGS:
                future = executor.submit(
                    _compute_param, name, param_name, fn, p_str, *args, **kwargs
                )
                fut_to_info[future] = (name, param_name)
                submitted_count[name] += 1

        file_results = {}
        remaining = {}

        for future in as_completed(fut_to_info):
            name, param_name = fut_to_info[future]
            try:
                _, _, arr, err = future.result()
                file_results.setdefault(name, {})[param_name] = arr
                if err:
                    print(f"{RED}Failed {name}/{param_name}: {err}{RESET}")
            except Exception as e:
                file_results.setdefault(name, {})[param_name] = None
                print(f"{RED}Failed {name}/{param_name}: {e}{RESET}")

            rem = remaining.get(name, submitted_count[name]) - 1
            remaining[name] = rem

            if rem == 0:
                params = file_results.pop(name)
                remaining.pop(name)
                arrays = [
                    params.get(pn) if params.get(pn) is not None else np.array([np.nan])
                    for pn, _, _, _ in PARAM_CONFIGS
                ]

                max_len = max(len(a) for a in arrays)
                prepend_sharpness = "_00000-" not in name
                N, S, R, TNR, SII = (
                    _pad(arrays[0], max_len),
                    _pad(arrays[1], max_len, sharpness_prepend_SHARPNESS_SKIP_FIRST_N_VALUES=prepend_sharpness),
                    _pad(arrays[2], max_len),
                    _pad(arrays[3], max_len),
                    _pad(arrays[4], max_len),
                )

                df = pd.DataFrame({
                    "time_index": np.arange(max_len),
                    "loudness_zwtv": N,
                    "sharpness_din_tv": S,
                    "roughness_dw": R,
                    "tnr_ecma_perseg": TNR,
                    "sii_ansi": SII,
                })

                output_file = output_folder / f"{name}.csv"
                df.to_csv(output_file, index=False)
                print(f"{GREEN}Saved: {output_file}{RESET}")