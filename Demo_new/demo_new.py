import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("QtAgg")

import matplotlib.pyplot as plt
import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
import torch.nn as nn

_DIR = Path(__file__).resolve().parent
_REPO = _DIR.parent

sys.path.insert(0, str(_REPO))

from DL_model.reference_vs_prediction import param_colors  # noqa: E402

PARAM_NAMES = [
    "loudness_zwtv",
    "sharpness_din_tv",
    "roughness_dw",
    "tnr_ecma_perseg",
    "sii_ansi",
]


class PsychoacousticModel(nn.Module):
    def __init__(
            self,
            initial_temporal_biases: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            # Stage 1 — no temporal compression, preserves full resolution
            nn.Conv1d(1, 10, kernel_size=512, stride=1),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Stage 2
            nn.Conv1d(10, 20, kernel_size=256, stride=1),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Stage 3
            nn.Conv1d(20, 40, kernel_size=128, stride=1),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Stage 4
            nn.Conv1d(40, 60, kernel_size=64, stride=1),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Stage 5
            nn.Conv1d(60, 80, kernel_size=32, stride=1),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        MAX = [500, 500, 10, 2, 1]
        self.heads = nn.ModuleDict()
        for i, name in enumerate(PARAM_NAMES):
            self.heads[name] = nn.Sequential(
                nn.Conv1d(80, 1, kernel_size=1),
                nn.AdaptiveAvgPool1d(MAX[i]),
            )

            bias = torch.zeros(1, MAX[i])
            if initial_temporal_biases is not None and name in initial_temporal_biases:
                bias = initial_temporal_biases[name].view(1, -1)
            self.register_parameter(f"{name}_bias", nn.Parameter(bias))

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_output = self.backbone(waveform)
        final_outputs: dict[str, torch.Tensor] = {}
        for name in PARAM_NAMES:
            out = self.heads[name](backbone_output).squeeze(1)
            bias = getattr(self, f"{name}_bias")
            final_outputs[name] = out + bias
        return final_outputs


PARAM_YLIMS = {
    "loudness_zwtv": (-1, 70),
    "sharpness_din_tv": (-1, 4),
    "roughness_dw": (-0.2, 1.0),
    "tnr_ecma_perseg": (-2.0, 20),
    "sii_ansi": (-0.1, 1.2),
}
PARAM_UNITS = {
    "loudness_zwtv": "son1e",
    "sharpness_din_tv": "acum",
    "roughness_dw": "asper",
    "tnr_ecma_perseg": "dB",
    "sii_ansi": "",
}
PARAM_COLORS = param_colors()

SAMPLE_RATE = 48000
CHUNK_SAMPLES = SAMPLE_RATE  # 1 second
HOP_SAMPLES = SAMPLE_RATE * 20 // 1000  # 20 ms hop

# Filter selection for output display.
FILTER_ORDER = ["direct", "direct_overwrite", "impulse", "fast", "slow"]
FILTER_LABELS = {
    "direct": "direct (keep newest datapoints)",
    "direct_overwrite": "direct (overwrite with newest prediction)",
    "impulse": "impulse",
    "fast": "fast",
    "slow": "slow",
}
FILTER_TAU = {
    "direct": 0.0,            # 0 ms — no averaging, only new frames appended
    "direct_overwrite": 0.0,  # 0 ms — no averaging, past frames overwritten
    "impulse": 0.035,         # 35 ms
    "fast": 0.125,            # 125 ms
    "slow": 1.000,            # 1000 ms
}
FILTER_KEYS = {
    "1": "direct",
    "2": "direct_overwrite",
    "3": "impulse",
    "4": "fast",
    "5": "slow",
}


class _StepTimer:
    """Accumulates per-step wall-clock times and reports averages."""

    def __init__(self):
        self._data: dict[str, list[float]] = {}
        self._stamps: dict[str, list[float]] = {}
        self._t0 = time.perf_counter()

    def lap(self, name: str, start: float):
        elapsed = time.perf_counter() - start
        wall = time.perf_counter() - self._t0
        self._data.setdefault(name, []).append(elapsed)
        self._stamps.setdefault(name, []).append(wall)

    def summary(self) -> str:
        lines = ["  Timing summary:"]
        for name in self._data:
            times = self._data[name]
            stamps = self._stamps[name]
            avg = sum(times) / len(times)
            mn_i = times.index(min(times))
            mx_i = times.index(max(times))
            lines.append(
                f"    {name:.<30s} avg {avg * 1000:.6f} ms  "
                f"min {times[mn_i] * 1000:.6f} ms @ {stamps[mn_i]:.6f}s  "
                f"max {times[mx_i] * 1000:.6f} ms @ {stamps[mx_i]:.6f}s  "
                f"(n={len(times)}, total {sum(times):.6f}s)"
            )
        return "\n".join(lines)


def _load_model(device: torch.device) -> PsychoacousticModel:
    import pandas as pd
    import torch.nn.functional as F

    stats_path = (
        _REPO
        / "data"
        / "reference_values"
        / "median_1s_chunk.csv"
    )
    df = pd.read_csv(stats_path)
    counts = [500, 500, 10, 2, 1]
    biases: dict[str, torch.Tensor] = {}
    for i, name in enumerate(PARAM_NAMES):
        vals = df[name].dropna().values.astype(np.float32)
        b = torch.from_numpy(vals).float()
        T = counts[i]
        if b.numel() != T:
            b = F.interpolate(
                b.view(1, 1, -1), size=T, mode="linear", align_corners=False
            ).view(-1)
        biases[name] = b

    model = PsychoacousticModel(initial_temporal_biases=biases).to(device)

    ckpt_path = _DIR / "epoch_0483.pt"
    if not ckpt_path.exists():
        ckpt_path = _REPO / "DL_model" / "epochs" / "epoch_0483.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: {ckpt_path.name} (epoch {ckpt['epoch']})")
    return model


# ── plotting ─────────────────────────────────────────────────────────
class LivePlotter:
    WINDOW_S = 20.0

    def __init__(self, timer: _StepTimer, filter_mode: str = "fast"):
        plt.ion()
        plt.style.use("dark_background")
        plt.rcParams["keymap.save"] = []
        if filter_mode not in FILTER_ORDER:
            filter_mode = FILTER_ORDER[3]
        self.timer = timer
        self.filter_mode = filter_mode
        self.hop_s = HOP_SAMPLES / SAMPLE_RATE

        self.fig, self.axes = plt.subplots(2, 3, figsize=(14, 7), sharex=True)
        self.fig.suptitle(
            f"Psychoacoustic Parameter Estimation (Live) — filter: {FILTER_LABELS[filter_mode]}",
            color="white",
        )
        self.axes_flat = list(self.axes.flat)

        self.grid_lines: dict[str, plt.Line2D] = {}
        self.live_lines: dict[str, plt.Line2D] = {}
        self.cloud_lines: dict[str, plt.Line2D] = {}
        self.samples: dict[str, dict[float, float]] = {}
        self.cloud: dict[str, list[tuple[float, float]]] = {}
        self.current_chunk: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.window_start = 0.0
        self.window_end = self.WINDOW_S
        marker_params = {"tnr_ecma_perseg", "sii_ansi"}
        param_slots = PARAM_NAMES + [None]
        for ax, name in zip(self.axes_flat, param_slots):
            if name is None:
                ax.set_visible(False)
                continue
            color = PARAM_COLORS[name]
            cloud_kwargs = {"color": "#777777", "linewidth": 0.6, "alpha": 0.35}
            live_kwargs = {"color": color, "linewidth": 1.0, "alpha": 0.55}
            grid_kwargs = {"color": color, "linewidth": 1.4}
            if name in marker_params:
                live_kwargs.update(marker="x", markersize=6, linestyle="dotted")
                grid_kwargs.update(marker="x", markersize=6, linestyle="dotted")
            (cloud_line,) = ax.plot([], [], **cloud_kwargs)
            (live_line,) = ax.plot([], [], **live_kwargs)
            (grid_line,) = ax.plot([], [], **grid_kwargs)
            unit = PARAM_UNITS[name]
            ax.set_ylabel(f"{name}\n({unit})", color="#cccccc", fontsize=9)
            ax.set_title(name, color="white", fontsize=10)
            ax.tick_params(colors="#cccccc")
            ax.grid(True, alpha=0.15)
            self.cloud_lines[name] = cloud_line
            self.live_lines[name] = live_line
            self.grid_lines[name] = grid_line
            self.samples[name] = {}
            self.cloud[name] = []
            self.current_chunk[name] = (np.array([]), np.array([]))
            ax.set_ylim(PARAM_YLIMS[name])

        self.axes_flat[-1].set_visible(False)
        self.fig.set_size_inches(14, 7)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        self.fig.canvas.mpl_connect("key_press_event", self._on_key)

    def update(self, preds: dict[str, torch.Tensor], t_start: float):
        t_end = t_start + 1.0

        if t_end >= self.window_end:
            self.save(_DIR / "example images" / f"segment_{self.window_start:.1f}s-{self.window_end:.1f}s.png")
            for name in PARAM_NAMES:
                self.samples[name].clear()
                self.cloud[name].clear()
            self.window_start = self.window_end
            self.window_end += self.WINDOW_S

        for name in PARAM_NAMES:
            raw_vals = preds[name][0].detach().cpu().numpy()
            n_frames = len(raw_vals)
            all_t = np.linspace(t_start, t_end, n_frames)
            self.current_chunk[name] = (all_t, raw_vals)
            buf = self.samples[name]
            cloud_buf = self.cloud[name]
            overwrite = self.filter_mode == "direct_overwrite"
            last_t = max(buf) if buf else float("-inf")
            for t, v in zip(all_t, raw_vals):
                key = round(t * 1000) / 1000
                cloud_buf.append((t, float(v)))
                if t > last_t:
                    buf[key] = float(v)
                    last_t = key
                elif overwrite:
                    buf[key] = float(v)

        for name in PARAM_NAMES:
            chunk_t, chunk_v = self.current_chunk[name]
            cloud_buf = self.cloud[name]
            if cloud_buf:
                pts = np.array(cloud_buf)
                order = pts[:, 0].argsort()
                self.cloud_lines[name].set_data(pts[order, 0], pts[order, 1])
            else:
                self.cloud_lines[name].set_data([], [])
            avg_t, avg_v = self._avg_series(name)
            self.live_lines[name].set_data(chunk_t, chunk_v)
            mask = ~np.isnan(avg_v)
            if mask.any():
                self.grid_lines[name].set_data(avg_t[mask], avg_v[mask])
            else:
                self.grid_lines[name].set_data([], [])
            self.grid_lines[name].axes.set_xlim(self.window_start, self.window_end)

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def _raw_series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        buf = self.samples[name]
        if not buf:
            return np.array([]), np.array([])
        pts = np.array(sorted(buf.items()))
        return pts[:, 0], pts[:, 1]

    def _avg_series(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        mode = self.filter_mode
        if mode in ("direct", "direct_overwrite"):
            return self._raw_series(name)
        t, v = self._raw_series(name)
        if t.size == 0:
            return np.array([]), np.array([])
        tau = FILTER_TAU[mode]
        step = self.hop_s
        times = np.arange(t[0] + step, t[-1] + step, step)
        if times.size == 0:
            return np.array([]), np.array([])
        cum = np.concatenate([[0.0], np.cumsum(v)])
        left = np.searchsorted(t, times - tau, side="left")
        right = np.searchsorted(t, times, side="right")
        counts = right - left
        means = np.full(times.shape, np.nan)
        ok = counts > 0
        means[ok] = (cum[right[ok]] - cum[left[ok]]) / counts[ok]
        return times, means

    def _on_key(self, event):
        mode = FILTER_KEYS.get(event.key.lower())
        if mode is not None:
            self.filter_mode = mode
            self.fig.suptitle(
                f"Psychoacoustic Parameter Estimation (Live) — filter: {FILTER_LABELS[mode]}",
                color="white",
            )
            self.fig.canvas.draw()
            print(f"Filter: {FILTER_LABELS[mode]} ({FILTER_TAU[mode] * 1000:.0f} ms)")

    def save(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.fig.savefig(str(path), dpi=150, bbox_inches="tight")
        print(f"Saved: {path.name}")
        print(self.timer.summary())

    def close(self):
        plt.ioff()
        plt.close(self.fig)


# ── audio sources ────────────────────────────────────────────────────
def _resample_if_needed(audio: np.ndarray, sr: int) -> np.ndarray:
    if sr != SAMPLE_RATE:
        import librosa
        audio = librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE)
    return audio


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim > 1:
        return audio.mean(axis=-1)
    return audio


def run_file(
    model: PsychoacousticModel,
    device: torch.device,
    file_path: Path,
    plotter: LivePlotter,
):
    timer = plotter.timer
    t = time.perf_counter()
    audio, sr = sf.read(str(file_path))
    audio = _to_mono(audio)
    audio = _resample_if_needed(audio, sr)
    audio = audio.astype(np.float32)
    timer.lap("file_load", t)
    total_samples = len(audio)
    print(f"File: {file_path.name} — {total_samples / SAMPLE_RATE:.1f}s "
          f"@ {sr}Hz → {SAMPLE_RATE}Hz")

    if total_samples < CHUNK_SAMPLES:
        audio = np.pad(audio, (0, CHUNK_SAMPLES - total_samples))
        total_samples = len(audio)

    with torch.no_grad():
        for start in range(0, total_samples - CHUNK_SAMPLES + 1, HOP_SAMPLES):
            chunk = audio[start : start + CHUNK_SAMPLES]
            t = time.perf_counter()
            waveform = (
                torch.from_numpy(chunk).float().unsqueeze(0).unsqueeze(0).to(device)
            )
            preds = model(waveform)
            timer.lap("model_inference", t)

            t_start = start / SAMPLE_RATE
            t = time.perf_counter()
            plotter.update(preds, t_start)
            timer.lap("plot_update", t)

    print("Finished processing file.")


def run_device(
    model: PsychoacousticModel,
    device: torch.device,
    device_index: int,
    plotter: LivePlotter,
):
    dev_info = sd.query_devices(device_index)
    ch_count = min(dev_info["max_input_channels"], 2)
    print(f"Capturing from: {dev_info['name']} ({ch_count} ch)")

    ring = np.zeros(CHUNK_SAMPLES, dtype=np.float32)
    write_pos = 0
    timer = plotter.timer

    def audio_callback(indata, frames, time_info, status):
        nonlocal ring, write_pos
        if status:
            print(f"  [audio] {status}", file=sys.stderr)
        mono = indata.mean(axis=-1).astype(np.float32)
        n = min(len(mono), CHUNK_SAMPLES - write_pos)
        ring[write_pos : write_pos + n] = mono[:n]
        write_pos += n
        if write_pos >= CHUNK_SAMPLES:
            write_pos = 0

    with sd.InputStream(
        device=device_index,
        channels=ch_count,
        samplerate=SAMPLE_RATE,
        blocksize=0,
        dtype="float32",
        callback=audio_callback,
    ):
        print("Listening... (Ctrl+C to stop)")
        t0 = time.perf_counter()
        hop_s = HOP_SAMPLES / SAMPLE_RATE
        while True:
            try:
                if write_pos >= CHUNK_SAMPLES:
                    continue
                chunk = ring.copy()
                t = time.perf_counter()
                waveform = (
                    torch.from_numpy(chunk)
                    .float()
                    .unsqueeze(0)
                    .unsqueeze(0)
                    .to(device)
                )
                with torch.no_grad():
                    preds = model(waveform)
                timer.lap("model_inference", t)

                t_elapsed = time.perf_counter() - t0
                t = time.perf_counter()
                plotter.update(preds, t_elapsed - 1.0)
                timer.lap("plot_update", t)

                elapsed = time.perf_counter() - t0 - t_elapsed
                sleep_time = hop_s - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            except KeyboardInterrupt:
                break
    print("Stopped.")


# ── device listing ───────────────────────────────────────────────────
def list_devices():
    print(sd.query_devices())
    print("\nUse --device <index> to select an input device.")


DEFAULT_DEVICE = 1  # VoiceMeeter Output (VB-Audio Vo, MME


# ── main ─────────────────────────────────────────────────────────────
def main():
    print(sd.query_devices())
    raw = input(f"\nEnter device index [{DEFAULT_DEVICE}]: ").strip()
    device_index = int(raw) if raw else DEFAULT_DEVICE

    opt_txt = ", ".join(
        f"{i}={FILTER_LABELS[m]} [{FILTER_TAU[m] * 1000:.0f}ms]"
        for i, m in enumerate(FILTER_ORDER, start=1)
    )
    raw = input(f"\nFilter [fast] ({opt_txt}): ").strip().lower()
    if raw.isdigit() and 1 <= int(raw) <= len(FILTER_ORDER):
        filter_mode = FILTER_ORDER[int(raw) - 1]
    elif raw in FILTER_ORDER:
        filter_mode = raw
    else:
        filter_mode = FILTER_ORDER[3]

    if torch.cuda.is_available():
        torch_device = torch.device("cuda")
    else:
        try:
            import torch_directml
            torch_device = torch_directml.device()
        except ImportError:
            torch_device = torch.device("cpu")
    print(f"Using device: {torch_device}")
    print("Press 1-5 in the plot window to switch filter live.")

    model = _load_model(torch_device)
    timer = _StepTimer()
    plotter = LivePlotter(timer, filter_mode=filter_mode)

    try:
        run_device(model, torch_device, device_index, plotter)
    except KeyboardInterrupt:
        pass
    finally:
        plotter.close()


if __name__ == "__main__":
    main()
