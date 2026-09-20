import numpy as np
import soundfile as sf

SAMPLE_RATE = 48000
DURATION_S = 3.0
FREQ_HZ = 1000.0
DB_START = -50.0
DB_END = -15.0

n_samples = int(SAMPLE_RATE * DURATION_S)
t = np.linspace(0, DURATION_S, n_samples, endpoint=False)

db = np.linspace(DB_START, DB_END, n_samples)
amplitude = 10.0 ** (db / 20.0)

signal = amplitude * np.sin(2 * np.pi * FREQ_HZ * t)

sf.write("example_1khz_sweep.wav", signal.astype(np.float32), SAMPLE_RATE)
print(f"Saved example_1khz_sweep.wav - {DURATION_S}s, {FREQ_HZ}Hz, {DB_START}dB to {DB_END}dB")
