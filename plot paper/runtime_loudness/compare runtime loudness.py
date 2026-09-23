import time
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import matplotlib.pyplot as plt
from mosqito import loudness_zwtv

SR = 48000
DURATION = 1.0
RUNS = {"dB (20*log10(|x|))": 50, "ITU BS.1770 LUFS": 50, "mosqito loudness_zwtv": 50}

rng = np.random.default_rng(0)
signal = rng.standard_normal(int(SR * DURATION))

meter = pyln.Meter(SR)


def db_values(sig):
    return 20 * np.log10(np.abs(sig))


def lufs(sig):
    return pyln.Meter(SR).integrated_loudness(sig)


def loudness(sig):
    return loudness_zwtv(sig, SR)


def timeit(fn, repeats, *args):
    return [time_fn(fn, *args) for _ in range(repeats)]


def time_fn(fn, *args):
    t0 = time.perf_counter()
    fn(*args)
    return (time.perf_counter() - t0) * 1000


funcs = [db_values, lufs, loudness]
samples = {name: timeit(fn, RUNS[name], signal) for name, fn in zip(RUNS, funcs)}

print(f"\nSignal: {DURATION}s white noise @ {SR} Hz ({signal.shape[0]} samples)\n")
for name, times in samples.items():
    print(f"{name:<24} median {np.median(times):>8.2f} ms  (min {min(times):.2f} / max {max(times):.2f})")

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(list(samples.values()), tick_labels=list(samples.keys()),
           patch_artist=True, boxprops=dict(facecolor=plt.cm.cool(0.5)),
           medianprops=dict(color=plt.cm.cool(1.0), linewidth=2.5))
ax.set_ylabel("runtime [ms]")
# ax.set_title(f"Runtime comparison ({DURATION}s white noise @ {SR} Hz)")
ax.set_yscale("log")
ax.grid(axis="y", linestyle="--", alpha=0.5)
out = Path(__file__).parent / "runtime comparison.png"
fig.savefig(out, dpi=300)
print(f"\nSaved: {out}")