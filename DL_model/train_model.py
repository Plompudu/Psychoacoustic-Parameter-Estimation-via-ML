import random
from datetime import datetime
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch

from .DL_model import PsychoacousticModel
from .params import PARAM_NAMES
from .compute_loss import compute_loss
from visualize_training.visualize_training import hold_plot, plot_losses


def _get_device(device_id: int = 0) -> torch.device:
    """
    Get a suitable PyTorch device based on the availability of CUDA or DirectML. Fallback to CPU if neither works.

    Args:
        device_id (int): The index of the device to use. Defaults to 0.

    Returns:
        torch.device: The device to be used.
    """
    if torch.cuda.is_available():
        print(f"Available CUDA devices: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")
        idx = min(device_id, torch.cuda.device_count() - 1)
        print(f"Using CUDA device {idx}: {torch.cuda.get_device_name(idx)}")
        return torch.device("cuda", idx)
    try:
        import torch_directml
        count = torch_directml.device_count()
        print(f"Available DirectML devices: {count}")
        for i in range(count):
            print(f"  {i}: {torch_directml.device_name(i)}")
        idx = min(device_id, count - 1)
        print(f"Using DirectML device {idx}: {torch_directml.device_name(idx)}")
        return torch_directml.device(idx)
    except ImportError:
        return torch.device("cpu")


def _load_audio_label_pairs(
    sound_dir: Path, labels_dir: Path
) -> list[tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    """
    Load audio-label pairs from the given directories.

    Args:
        sound_dir (Path): The directory containing audio files.
        labels_dir (Path): The directory containing label CSV files.

    Returns:
        list[tuple[torch.Tensor, dict[str, torch.Tensor]]]: A list of tuples, where each tuple contains a waveform
            tensor and a dictionary of target tensors.
    """
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
        waveform = torch.from_numpy(audio).float().unsqueeze(0)

        df = pd.read_csv(csv_path)
        targets = {}
        for name in PARAM_NAMES:
            targets[name] = torch.from_numpy(df[name].values.astype(np.float32))

        pairs.append((waveform, targets))
    return pairs


def _collate(
    pairs: list[tuple[torch.Tensor, dict[str, torch.Tensor]]],
    device: torch.device
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    waveforms = [p[0] for p in pairs]
    max_wav_len = max(w.shape[-1] for w in waveforms)
    batch_waveforms = []
    for w in waveforms:
        pad = max_wav_len - w.shape[-1]
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        batch_waveforms.append(w)
    waveform_batch = torch.stack(batch_waveforms, dim=0).to(device)

    max_frames = {
        name: max(p[1][name].shape[-1] for p in pairs)
        for name in PARAM_NAMES
    }
    target_batch: dict[str, list[torch.Tensor]] = {n: [] for n in PARAM_NAMES}
    for p in pairs:
        for name in PARAM_NAMES:
            t = p[1][name]
            pad = max_frames[name] - t.shape[-1]
            if pad > 0:
                t = torch.nn.functional.pad(t, (0, pad), value=float("nan"))
            target_batch[name].append(t)
    targets = {
        name: torch.stack(target_batch[name], dim=0).to(device) for name in PARAM_NAMES
    }

    return waveform_batch, targets


def _valid_frame_count(t: torch.Tensor) -> int:
    """Number of non-NaN frames from the start."""
    valid = ~torch.isnan(t)
    idxs = torch.where(valid.any(dim=0) if t.ndim == 2 else valid)[0]
    return idxs[-1].item() + 1 if len(idxs) > 0 else 1


def _training_step(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    targets: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Single forward-backward-update step for one batch."""
    preds = model(waveform)
    trimmed = {name: targets[name][:, :preds[name].shape[-1]]
               for name in PARAM_NAMES}
    losses = compute_loss(model, preds, trimmed)
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    return {k: v.item() for k, v in losses.items()}


def _resume_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    checkpoint_dir: Path,
) -> tuple[int, list[dict[str, float]]]:
    """Load the latest checkpoint if available, else start from scratch."""
    ckpt_files = sorted(checkpoint_dir.glob("epoch_*.pt"))
    if not ckpt_files:
        return 0, []

    latest = ckpt_files[-1]
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    try:
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        print(f"Resumed from {latest.name} (epoch {ckpt['epoch']})")
        return ckpt["epoch"], ckpt["history"]
    except Exception as e:
        print(f"Could not resume from {latest.name} ({e}) — starting from scratch")
        return 0, []


def _save_runtime_csv(output_dir: Path, stem: str, meta: dict, preds: dict[str, torch.Tensor]):
    """Save per-parameter prediction statistics to CSV."""
    rows = []
    for name in PARAM_NAMES:
        p = preds[name][0]
        row = dict(meta)
        row["parameter"] = name
        row["output_frames"] = p.shape[-1]
        row["pred_min"] = round(p.min().item(), 6)
        row["pred_max"] = round(p.max().item(), 6)
        row["pred_mean"] = round(p.mean().item(), 6)
        rows.append(row)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(rows).to_csv(output_dir / f"{ts}_{stem}_runtime.csv", index=False)


def _save_prediction_plots(output_dir: Path, stem: str, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor]):
    """Plot prediction vs target for each parameter and save as PNG."""
    for name in PARAM_NAMES:
        p = preds[name][0]
        t = target[name][:p.shape[-1]]

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


def _save_comparison_csv(output_dir: Path, stem: str, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor]):
    """Save side-by-side target vs prediction values for all params as CSV."""
    all_arrays: dict[str, np.ndarray] = {}
    for name in PARAM_NAMES:
        p = preds[name][0]
        t = target[name][:p.shape[-1]]
        all_arrays[f"{name}_target"] = t.numpy()
        all_arrays[f"{name}_pred"] = p.cpu().numpy()
    max_len = max(arr.shape[-1] for arr in all_arrays.values())
    df_dict: dict[str, np.ndarray] = {"time_index": np.arange(max_len)}
    for key, arr in all_arrays.items():
        pad = max_len - arr.shape[-1]
        df_dict[key] = np.pad(arr, (0, pad), constant_values=np.nan)
    pd.DataFrame(df_dict).to_csv(output_dir / f"{stem}_comparison.csv", index=False)


def _compare_predictions(
    sound_dir: Path,
    labels_dir: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    n_samples: int = 1,
    device: torch.device | None = None,
):
    """Run inference on random samples and save plots + CSVs."""
    if device is None:
        device = _get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs = _load_audio_label_pairs(sound_dir, labels_dir)
    param_frame_counts = {
        name: max(_valid_frame_count(p[1][name]) for p in pairs)
        for name in PARAM_NAMES
    }
    model = PsychoacousticModel(param_frame_counts=param_frame_counts).to(device)
    ckpt_files = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
    if not ckpt_files:
        print("No checkpoint found — skipping comparison")
        return
    latest = ckpt_files[-1]
    ckpt = torch.load(latest, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded {latest.name} for comparison")

    csv_paths = sorted(Path(labels_dir).glob("*.csv"))
    zipped = list(zip(pairs, csv_paths))
    random.shuffle(zipped)
    zipped = zipped[:n_samples]

    with torch.no_grad():
        for (waveform, target), csv_path in zipped:
            inp = waveform.unsqueeze(0).to(device)
            t0 = time.perf_counter()
            preds = model(inp)
            elapsed = time.perf_counter() - t0
            stem = csv_path.stem

            meta = {
                "file": f"{stem}.wav",
                "input_samples": inp.shape[-1],
                "input_duration_s": round(inp.shape[-1] / 48000, 2),
                "inference_time_ms": round(elapsed * 1000, 2),
                "backbone_frames": model.backbone(inp).shape[-1],
            }
            _save_runtime_csv(output_dir, stem, meta, preds)
            _save_prediction_plots(output_dir, stem, preds, target)
            _save_comparison_csv(output_dir, stem, preds, target)
    print(f"Comparison saved to {output_dir}")


def run_comparison(
    sound_dir: Path,
    labels_dir: Path,
    checkpoint_dir: Path,
    n_samples: int = 1,
    device_id: int = 0,
):
    """Run inference on random samples and save plots/CSVs to comparison_newest_epoch/."""
    print("=" * 100)
    device = _get_device(device_id)
    output_dir = Path(__file__).resolve().parent / "comparison_newest_epoch"
    _compare_predictions(sound_dir, labels_dir, checkpoint_dir, output_dir, n_samples, device=device)
    hold_plot()


def _log_epoch(
    epoch: int,
    avg_losses: dict[str, float],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, float]],
    csv_path: Path,
    plot_path: Path,
    checkpoint_dir: Path,
):
    """Save loss row, update plot, and write checkpoint."""
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


def train_model(
    sound_dir: Path,
    labels_dir: Path,
    checkpoint_dir: Path,
    losses_dir: Path,
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 2,
    device_id: int = 0,
) -> list[dict[str, float]]:
    print("=" * 100)
    device = _get_device(device_id)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    losses_dir = Path(losses_dir)
    losses_dir.mkdir(parents=True, exist_ok=True)
    csv_path = losses_dir / "losses.csv"
    plot_path = losses_dir / "losses.png"

    # ── Load dataset ──
    pairs = _load_audio_label_pairs(sound_dir, labels_dir)
    if not pairs:
        print("No data found — nothing to train on.")
        return []
    print(f"Loaded {len(pairs)} audio-label pair(s)")

    # ── Compute per-parameter frame counts from data ──
    param_frame_counts = {
        name: max(_valid_frame_count(p[1][name]) for p in pairs)
        for name in PARAM_NAMES
    }
    model = PsychoacousticModel(param_frame_counts=param_frame_counts).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # ── Resume from latest checkpoint ──
    start_epoch, history = _resume_checkpoint(model, optimizer, checkpoint_dir)
    if start_epoch == 0:
        csv_path.unlink(missing_ok=True)
        print("Starting from scratch")
    elif history:
        rows = [{"epoch": i + 1, **h} for i, h in enumerate(history)]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Training loop ──

    for epoch in range(start_epoch, epochs):
        random.shuffle(pairs)

        t0 = time.perf_counter()
        epoch_losses: list[dict[str, float]] = []
        for i in range(0, len(pairs), batch_size):
            batch_pairs = pairs[i : i + batch_size]
            waveform, targets = _collate(batch_pairs, device)
            losses = _training_step(model, waveform, targets, optimizer)
            epoch_losses.append(losses)
        elapsed = time.perf_counter() - t0

        avg_losses = {
            k: torch.tensor([b[k] for b in epoch_losses]).mean().item()
            for k in epoch_losses[0]
        }
        history.append(avg_losses)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_losses['total']:.4f} — {elapsed:.1f}s")

        _log_epoch(epoch, avg_losses, model, optimizer, history, csv_path, plot_path, checkpoint_dir)

    return history
