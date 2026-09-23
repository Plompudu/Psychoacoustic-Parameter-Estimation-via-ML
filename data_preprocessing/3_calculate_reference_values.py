from pathlib import Path
import time
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import soundfile as sf

from mosqito import (
    loudness_zwtv,
    sharpness_din_tv,
    roughness_dw,
    tnr_ecma_perseg,
    sii_ansi,
)

RED = "\033[91m"
GREEN = "\033[92m"
BLUE = "\033[94m"
RESET = "\033[0m"

INPUT_DIR = Path(__file__).resolve().parent / ".." / "data" / "raw_sound_files"
INPUT_DIR_1S = Path(__file__).resolve().parent / ".." / "data" / "raw_sound_files_1s"
OUTPUT_DIR = Path(__file__).resolve().parent / ".." / "data" / "reference_values"

PARAM_CONFIGS_1 = [
    ("loudness_zwtv", loudness_zwtv, (), {}),
    ("sharpness_din_tv", sharpness_din_tv, (), {}),
    ("roughness_dw", roughness_dw, (), {}),
    ("tnr_ecma_perseg", tnr_ecma_perseg, (), {}),
]

PARAM_CONFIGS_2 = [
    ("sii_ansi", sii_ansi, ("critical", "normal"), {}),
]

MAX_CONCURRENT_WORKERS = 6
RETRY_DELAY_SECONDS = 5.0


class _ColorStdout:
    def __init__(self, original):
        self.original = original

    def write(self, text):
        if text.startswith("[Warning]"):
            text = f"{BLUE}{text}{RESET}"
        self.original.write(text)

    def flush(self):
        self.original.flush()


def _pad(arr, target_len):
    arr = np.asarray(arr, dtype=float)
    if len(arr) >= target_len:
        return arr[:target_len]
    return np.pad(arr, (0, target_len - len(arr)), constant_values=np.nan)


def _compute_param(name, param_name, fn, signal, sr, *args, **kwargs):
    np.seterr(invalid="ignore", divide="ignore")

    attempt = 0
    while True:
        attempt += 1
        print(f"[{name}] {fn.__name__}: computing (attempt {attempt}) ...", flush=True)
        t0 = time.perf_counter()
        try:
            result, *_ = fn(signal, sr, *args, **kwargs)
            elapsed = time.perf_counter() - t0
            status = "OK" if attempt == 1 else f"OK (attempt {attempt})"
            return result, f"[{name}] {fn.__name__}: {status} ({elapsed:.4f}s)"
        except MemoryError:
            elapsed = time.perf_counter() - t0
            msg = (
                f"[{name}] {RED}{fn.__name__}: OUT OF MEMORY (attempt {attempt}, {elapsed:.4f}s) "
                f"- retrying in {RETRY_DELAY_SECONDS}s{RESET}"
            )
            print(msg, flush=True)
            time.sleep(RETRY_DELAY_SECONDS)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            duration = len(signal) / sr
            return None, f"[{name}] {RED}{fn.__name__}: FAILED ({e}, {duration:.2f}s signal){RESET}"


def _compute_file_param(name, audio_path_str, param_name, fn, args, kwargs):
    signal, sr = sf.read(audio_path_str)
    result, msg = _compute_param(name, param_name, fn, signal, sr, *args, **kwargs)
    return name, param_name, result, msg


def _build_block(name, params, param_configs):
    arrays = [
        (
            np.atleast_1d(np.asarray(params.get(pn), dtype=float))
            if params.get(pn) is not None
            else np.array([np.nan])
        )
        for pn, _, _, _ in param_configs
    ]
    max_len = max(len(a) for a in arrays)

    data = {"source_file": name, "time_index": np.arange(max_len)}
    for (pn, _, _, _), arr in zip(param_configs, arrays):
        data[pn] = _pad(arr, max_len)

    return pd.DataFrame(data)


def calculate_reference_values(input_dir, output_dir, param_configs):
    print("=" * 100)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_paths = sorted(input_dir.glob("*.wav"))
    total = len(audio_paths)
    print(f"Processing {total} files from {input_dir} ...")
    print(f"Using {MAX_CONCURRENT_WORKERS} concurrent workers\n")

    with ProcessPoolExecutor(max_workers=MAX_CONCURRENT_WORKERS) as executor:
        pending = {p.stem: len(param_configs) for p in audio_paths}
        file_results = {}
        fut_to_name = {}

        for p in audio_paths:
            name = p.stem
            for param_name, fn, args, kwargs in param_configs:
                fut = executor.submit(
                    _compute_file_param, name, str(p), param_name, fn, args, kwargs
                )
                fut_to_name[fut] = name

        done_files = 0
        for fut in as_completed(fut_to_name):
            name = fut_to_name[fut]
            try:
                _, param_name, result, msg = fut.result()
                print(msg, flush=True)
                file_results.setdefault(name, {})[param_name] = result
            except Exception as e:
                print(f"[{name}] {RED}FAILED: {e}{RESET}")

            pending[name] -= 1
            if pending[name] == 0:
                df = _build_block(name, file_results.pop(name), param_configs)
                df.to_csv(output_dir / f"{name}.csv", index=False)
                print(f"  -> {GREEN}saved ({len(df)} rows){RESET}\n")
                done_files += 1
                print(f"Progress: {done_files}/{total}\n")


if __name__ == "__main__":
    sys.stdout = _ColorStdout(sys.stdout)
    calculate_reference_values(INPUT_DIR, OUTPUT_DIR / "partial_df_1", PARAM_CONFIGS_1)
    calculate_reference_values(INPUT_DIR_1S, OUTPUT_DIR / "partial_df_2", PARAM_CONFIGS_2)