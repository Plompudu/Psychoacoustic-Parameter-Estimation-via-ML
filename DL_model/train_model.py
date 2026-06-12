import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .DL_model import PsychoacousticModel
from .forward import forward
from .params import PARAM_NAMES
from .training_step import training_step
from visualize_training.visualize_training import hold_plot, plot_losses


def _load_pairs(
    sound_dir: Path, labels_dir: Path
) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """Load every (waveform, targets) pair from matching WAV/CSV files."""
    pairs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]] = []
    for csv_path in sorted(Path(labels_dir).glob("*.csv")):
        stem = csv_path.stem
        wav_path = Path(sound_dir) / f"{stem}.wav"
        if not wav_path.exists():
            print(f"  Skipping {stem} — no matching WAV")
            continue

        audio, _ = sf.read(wav_path)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        waveform = torch.from_numpy(audio).float().unsqueeze(0)  # (1, T_audio)

        df = pd.read_csv(csv_path)
        targets = {}
        for name in PARAM_NAMES:
            arr = df[name].values.astype(np.float32)
            targets[name] = torch.from_numpy(arr)  # (T_labels,)

        pairs.append((waveform, targets))
    return pairs


def _collate(
    pairs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]], device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Pad waveforms (zero) and targets (NaN) to max length in batch."""
    waveforms = [p[0] for p in pairs]
    max_wav_len = max(w.shape[-1] for w in waveforms)
    batch_waveforms = []
    for w in waveforms:
        pad = max_wav_len - w.shape[-1]
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        batch_waveforms.append(w)
    waveform_batch = torch.stack(batch_waveforms, dim=0).to(device)  # (B, 1, T_audio)

    max_frames = max(p[1][PARAM_NAMES[0]].shape[-1] for p in pairs)
    target_batch: dict[str, list[torch.Tensor]] = {n: [] for n in PARAM_NAMES}
    for p in pairs:
        for name in PARAM_NAMES:
            t = p[1][name]
            pad = max_frames - t.shape[-1]
            if pad > 0:
                t = torch.nn.functional.pad(t, (0, pad), value=float("nan"))
            target_batch[name].append(t)
    targets = {
        name: torch.stack(target_batch[name], dim=0).to(device) for name in PARAM_NAMES
    }

    return waveform_batch, targets


def _valid_frame_count(t: torch.Tensor) -> int:
    valid = ~torch.isnan(t)
    idxs = torch.where(valid.any(dim=0) if t.ndim == 2 else valid)[0]
    return idxs[-1].item() + 1 if len(idxs) > 0 else 1


def compare_predictions(
    sound_dir: Path,
    labels_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    n_samples: int = 1,
):
    """Load the newest checkpoint, run inference on up to ``n_samples`` random
    audio files, and save per-parameter overlay plots + numerical CSV."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = PsychoacousticModel().to(device)
    ckpt_files = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
    if not ckpt_files:
        print("No checkpoint found — skipping comparison")
        return
    latest = ckpt_files[-1]
    ckpt = torch.load(latest, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {latest.name} for comparison")

    pairs = _load_pairs(sound_dir, labels_dir)
    csv_paths = sorted(Path(labels_dir).glob("*.csv"))
    zipped = list(zip(pairs, csv_paths))
    random.shuffle(zipped)
    zipped = zipped[:n_samples]

    with torch.no_grad():
        for (waveform, target), csv_path in zipped:
            import time
            inp = waveform.unsqueeze(0).to(device)
            target_n_frames = {name: _valid_frame_count(target[name])
                               for name in PARAM_NAMES}
            t0 = time.perf_counter()
            preds = forward(model, inp, target_n_frames)
            elapsed = time.perf_counter() - t0
            stem = csv_path.stem

            # Runtime diagnostics CSV (metadata repeated on every row)
            meta = {
                "file": f"{stem}.wav",
                "input_samples": inp.shape[-1],
                "input_duration_s": round(inp.shape[-1] / 48000, 2),
                "inference_time_ms": round(elapsed * 1000, 2),
                "backbone_frames": model.backbone(inp).shape[-1],
            }
            runtime_rows = []
            for name in PARAM_NAMES:
                p = preds[name][0]
                row = dict(meta)
                row["parameter"] = name
                row["output_frames"] = p.shape[-1]
                row["pred_min"] = round(p.min().item(), 6)
                row["pred_max"] = round(p.max().item(), 6)
                row["pred_mean"] = round(p.mean().item(), 6)
                runtime_rows.append(row)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            pd.DataFrame(runtime_rows).to_csv(
                output_dir / f"{ts}_{stem}_runtime.csv", index=False
            )

            for name in PARAM_NAMES:
                p = preds[name][0]                # (n_frames_param,)
                t = target[name]                  # (n_frames_padded,)
                t = t[:p.shape[-1]]               # trim NaN tail

                p_np, t_np = p.cpu().numpy(), t.numpy()
                n_frames = len(p_np)

                fig, ax = plt.subplots(figsize=(10, 4))
                if n_frames == 1:
                    ax.axhline(y=t_np[0], label="target", color="tab:blue", alpha=0.8, linestyle="--")
                    ax.axhline(y=p_np[0], label="prediction", color="tab:red", alpha=0.8, linestyle="--")
                else:
                    ax.plot(t_np, label="target", color="tab:blue", alpha=0.8)
                    ax.plot(p_np, label="prediction", color="tab:red", alpha=0.8)
                ax.set_xlabel("Time frame")
                ax.set_ylabel(name)
                ax.legend()
                fig.tight_layout()
                fig.savefig(output_dir / f"{stem}_{name}.png")
                plt.close(fig)

            # Numerical comparison CSV — pad shorter arrays with NaN
            all_arrays: dict[str, np.ndarray] = {}
            for name in PARAM_NAMES:
                p = preds[name][0]
                t = target[name]
                t = t[:p.shape[-1]]
                all_arrays[f"{name}_target"] = t.numpy()
                all_arrays[f"{name}_pred"] = p.cpu().numpy()
            max_len = max(arr.shape[-1] for arr in all_arrays.values())
            df_dict: dict[str, np.ndarray] = {"time_index": np.arange(max_len)}
            for key, arr in all_arrays.items():
                pad = max_len - arr.shape[-1]
                df_dict[key] = np.pad(arr, (0, pad), constant_values=np.nan)
            pd.DataFrame(df_dict).to_csv(
                output_dir / f"{stem}_comparison.csv", index=False
            )
    print(f"Comparison saved to {output_dir}")


def train_model(
    sound_dir: Path,
    labels_dir: Path,
    checkpoint_dir: Path,
    losses_dir: Path,
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 2,
    n_samples_final_comparison: int = 1,
) -> list[dict[str, float]]:
    """Train on real audio files and psychoacoustic label CSVs.

    Saves a checkpoint after every epoch. If checkpoints exist, resumes
    from the latest one and trains up to ``epochs`` total epochs.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = PsychoacousticModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    losses_dir = Path(losses_dir)
    losses_dir.mkdir(parents=True, exist_ok=True)
    csv_path = losses_dir / "losses.csv"
    plot_path = losses_dir / "losses.png"

    # ── Load dataset ───────────────────────────────────────────────
    pairs = _load_pairs(sound_dir, labels_dir)
    if not pairs:
        print("No data found — nothing to train on.")
        return []
    print(f"Loaded {len(pairs)} audio-label pair(s)")

    # ── Resume from latest checkpoint ──────────────────────────────
    start_epoch = 0
    history: list[dict[str, float]] = []

    ckpt_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if ckpt_files:
        latest = ckpt_files[-1]
        ckpt = torch.load(latest, map_location=device, weights_only=False)
        try:
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"]
            history = ckpt["history"]
            print(f"Resumed from {latest.name} (epoch {start_epoch}/{epochs})")
        except Exception as e:
            print(f"Could not resume from {latest.name} ({e}) — starting from scratch")
            ckpt_files = []
    if not ckpt_files:
        print("No checkpoint found — starting from scratch")
        csv_path.unlink(missing_ok=True)

    # ── Rewrite CSV from loaded history (ensures consistency) ──────
    if history:
        rows = [{"epoch": i + 1, **h} for i, h in enumerate(history)]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Training loop ──────────────────────────────────────────────
    for epoch in range(start_epoch, epochs):
        random.shuffle(pairs)

        epoch_losses: list[dict[str, float]] = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            waveform, targets = _collate(batch_pairs, device)
            losses = training_step(model, waveform, targets, optimizer)
            epoch_losses.append(losses)

        avg_losses = {
            k: torch.tensor([b[k] for b in epoch_losses]).mean().item()
            for k in epoch_losses[0]
        }
        history.append(avg_losses)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_losses['total']:.4f}")

        row = {"epoch": epoch + 1, **avg_losses}
        pd.DataFrame([row]).to_csv(csv_path, mode="a", header=not csv_path.exists(), index=False)
        plot_losses(csv_path, plot_path)

        ckpt_path = checkpoint_dir / f"epoch_{epoch + 1:04d}.pt"
        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
            },
            ckpt_path,
        )

    compare_predictions(
        sound_dir,
        labels_dir,
        checkpoint_dir,
        Path(__file__).resolve().parent / "comparision_newest_epoch",
        n_samples_final_comparison
    )
    hold_plot()
    return history
