import random
import shutil
from datetime import datetime
from pathlib import Path
import time
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .DL_model import PsychoacousticModel
from .params import PARAM_NAMES
from .compute_loss import compute_loss
from visualize_training.visualize_training import hold_plot, plot_losses

def _load_time_biases(csv_path: str | Path) -> dict[str, torch.Tensor]:
    df = pd.read_csv(csv_path)
    counts = [500, 500, 9, 2, 1]
    biases: dict[str, torch.Tensor] = {}
    for i, name in enumerate(PARAM_NAMES):
        vals = df[name].dropna().values.astype(np.float32)
        b = torch.from_numpy(vals).float()
        T = counts[i]
        if b.numel() != T:
            b = F.interpolate(b.view(1, 1, -1), size=T, mode="linear", align_corners=False).view(-1)
        biases[name] = b
    return biases


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


class PsychoAcousticDataset(Dataset):
    def __init__(self, sound_dir: Path, csv_path: Path, subset_indices: list[int] | None = None,
                 audio_workers: int = 0):
        self.sound_dir = Path(sound_dir)
        self.csv_path = Path(csv_path)

        labels_cache = self.csv_path.with_suffix(".labels.pt")
        if labels_cache.exists():
            print(f"Loading pre-parsed labels from {labels_cache}...")
            t0 = time.perf_counter()
            cached = torch.load(labels_cache, weights_only=True)
            all_stems = cached["stems"]
            all_labels = cached["labels"]
            print(f"  done in {time.perf_counter() - t0:.4f} s")
        else:
            print(f"Parsing {self.csv_path} into memory...")
            n = 0
            t0 = time.perf_counter()
            all_labels: dict[str, dict[str, torch.Tensor]] = {}
            usecols = ["source_file"] + PARAM_NAMES
            reader = pd.read_csv(self.csv_path, usecols=usecols, chunksize=100_000)
            for chunk in reader:
                chunk["stem"] = chunk["source_file"].str.replace(".csv", "", regex=False)
                for stem, grp in chunk.groupby("stem"):
                    if stem not in all_labels:
                        all_labels[stem] = {}
                    for name in PARAM_NAMES:
                        vals = grp[name].to_numpy(dtype=np.float32)
                        all_labels[stem].setdefault(name, []).append(torch.from_numpy(vals))
                n += len(chunk)
                print(f"  parsed {n / 1_000_000:.1f}M rows ({time.perf_counter() - t0:.2f}s)...")
            all_stems = sorted(all_labels.keys())
            for stem in all_stems:
                for name in PARAM_NAMES:
                    all_labels[stem][name] = torch.cat(all_labels[stem][name])
            elapsed = time.perf_counter() - t0
            print(f"  done in {elapsed:.4f} s — saving cache to {labels_cache}")
            torch.save({"stems": all_stems, "labels": all_labels}, labels_cache)

        if subset_indices is not None:
            self.stems = [all_stems[i] for i in subset_indices]
            self._labels = {s: all_labels[s] for s in self.stems}
        else:
            self.stems = all_stems
            self._labels = all_labels

        audio_cache = self.csv_path.with_suffix(".audio.pt")
        if audio_cache.exists():
            print(f"Loading pre-parsed audio from {audio_cache}...")
            t0 = time.perf_counter()
            self._waveforms = torch.load(audio_cache, weights_only=True)
            print(f"  done in {time.perf_counter() - t0:.4f} s")
        else:
            print(f"Loading {len(self.stems)} audio files into memory...")
            self._waveforms: dict[str, torch.Tensor] = {}
            t0 = time.perf_counter()
            total = len(self.stems)

            def _load_one(stem: str) -> tuple[str, torch.Tensor]:
                wav_path = self.sound_dir / f"{stem}.wav"
                audio, _ = sf.read(str(wav_path))
                if audio.ndim > 1:
                    audio = audio.mean(axis=1)
                return stem, torch.from_numpy(audio).float().unsqueeze(0)

            if audio_workers > 1:
                import concurrent.futures
                n_workers = min(audio_workers, total)
                with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
                    futures = {pool.submit(_load_one, stem): stem for stem in self.stems}
                    for i, f in enumerate(concurrent.futures.as_completed(futures), 1):
                        stem, waveform = f.result()
                        self._waveforms[stem] = waveform
                        if i % 100 == 0 or i == total:
                            print(f"  {i}/{total} done ({time.perf_counter() - t0:.4f}s)")
            else:
                for i, stem in enumerate(self.stems):
                    stem, waveform = _load_one(stem)
                    self._waveforms[stem] = waveform
                    if (i + 1) % 100 == 0 or i + 1 == total:
                        print(f"  {i + 1}/{total} done ({time.perf_counter() - t0:.4f}s)")
            load_time = time.perf_counter() - t0
            print(f"  done in {load_time:.4f} s")
            if not subset_indices:
                torch.save(self._waveforms, audio_cache)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        stem = self.stems[idx]
        return self._waveforms[stem], self._labels[stem]



def _collate(
    batch: list[tuple[torch.Tensor, dict[str, torch.Tensor]]]
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    waveforms = [b[0] for b in batch]
    max_wav_len = max(w.shape[-1] for w in waveforms)
    batch_waveforms = []
    for w in waveforms:
        pad = max_wav_len - w.shape[-1]
        if pad > 0:
            w = torch.nn.functional.pad(w, (0, pad))
        batch_waveforms.append(w)
    waveform_batch = torch.stack(batch_waveforms, dim=0)

    max_frames = {
        name: max(b[1][name].shape[-1] for b in batch)
        for name in PARAM_NAMES
    }
    target_batch: dict[str, list[torch.Tensor]] = {n: [] for n in PARAM_NAMES}
    for b in batch:
        for name in PARAM_NAMES:
            t = b[1][name]
            pad = max_frames[name] - t.shape[-1]
            if pad > 0:
                t = torch.nn.functional.pad(t, (0, pad), value=float("nan"))
            target_batch[name].append(t)
    targets = {
        name: torch.stack(target_batch[name], dim=0) for name in PARAM_NAMES
    }

    return waveform_batch, targets


def _training_step(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    targets: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    """Single forward-backward-update step for one batch."""
    waveform = waveform.to(device)
    targets = {name: t.to(device) for name, t in targets.items()}
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


def _save_runtime_csv(output_dir: Path, stem: str, meta: dict, preds: dict[str, torch.Tensor], epoch_tag: str = "", targets: dict[str, torch.Tensor] | None = None):
    """Save per-parameter prediction statistics to CSV."""
    prefix = f"{epoch_tag}_" if epoch_tag else ""
    rows = []
    for name in PARAM_NAMES:
        p = preds[name][0]
        if targets is not None:
            t = targets[name][:p.shape[-1]]
            mask = ~torch.isnan(t)
            if mask.any():
                p = p[mask]
        row = dict(meta)
        row["parameter"] = name
        row["output_frames"] = p.shape[-1]
        row["pred_min"] = round(p.min().item(), 6)
        row["pred_max"] = round(p.max().item(), 6)
        row["pred_mean"] = round(p.mean().item(), 6)
        rows.append(row)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    pd.DataFrame(rows).to_csv(output_dir / f"{prefix}{ts}_{stem}_runtime.csv", index=False)


def _save_prediction_plots(output_dir: Path, stem: str, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor], epoch_tag: str = ""):
    """Plot prediction vs target for each parameter and save as PNG."""
    prefix = f"{epoch_tag}_" if epoch_tag else ""
    for name in PARAM_NAMES:
        p = preds[name][0]
        t = target[name][:p.shape[-1]]

        p_np, t_np = p.cpu().numpy(), t.numpy()
        n_frames = len(p_np)
        p_masked = np.ma.masked_where(np.isnan(t_np), p_np)

        fig, ax = plt.subplots(figsize=(10, 4))
        if n_frames == 1:
            ax.axhline(y=t_np[0], label="target", color="tab:blue", alpha=0.8, linestyle="--")
            ax.axhline(y=p_np[0], label="prediction", color="tab:red", alpha=0.8, linestyle="--")
        else:
            ax.plot(t_np, label="target", color="tab:blue", alpha=0.8)
            ax.plot(p_masked, label="prediction", color="tab:red", alpha=0.8)
        ax.set_xlabel("Time frame")
        ax.set_ylabel(name)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}{stem}_{name}.png")
        plt.close(fig)


def _save_comparison_csv(output_dir: Path, stem: str, preds: dict[str, torch.Tensor], target: dict[str, torch.Tensor], epoch_tag: str = ""):
    """Save side-by-side target vs prediction values for all params as CSV."""
    prefix = f"{epoch_tag}_" if epoch_tag else ""
    all_arrays: dict[str, np.ndarray] = {}
    for name in PARAM_NAMES:
        p = preds[name][0]
        t = target[name][:p.shape[-1]]
        p_np = p.cpu().numpy().copy()
        p_np[np.isnan(t.numpy())] = np.nan
        all_arrays[f"{name}_target"] = t.numpy()
        all_arrays[f"{name}_pred"] = p_np
    max_len = max(arr.shape[-1] for arr in all_arrays.values())
    df_dict: dict[str, np.ndarray] = {"time_index": np.arange(max_len)}
    for key, arr in all_arrays.items():
        pad = max_len - arr.shape[-1]
        df_dict[key] = np.pad(arr, (0, pad), constant_values=np.nan)
    pd.DataFrame(df_dict).to_csv(output_dir / f"{prefix}{stem}_comparison.csv", index=False)


def _compare_epoch(
    dataset: PsychoAcousticDataset,
    checkpoint_dir: Path,
    output_dir: Path,
    n_samples: int = 1,
    device: torch.device | None = None,
    epoch: int | str = "newest",
    epoch_tag: str = "",
):
    """Run inference with a specific epoch checkpoint and save plots + CSVs."""
    if device is None:
        device = _get_device()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stats_path = Path(__file__).parent.parent / "data" / "standardized_audio_files" / "training_set" / "visualization" / "parameter_average_per_time_segment.csv"

    biases = _load_time_biases(stats_path)
    model = PsychoacousticModel(initial_temporal_biases=biases).to(device)
    if epoch == "newest":
        ckpt_files = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
        if not ckpt_files:
            print("No checkpoint found — skipping comparison")
            return
        ckpt_path = ckpt_files[-1]
    else:
        ckpt_path = Path(checkpoint_dir) / f"epoch_{epoch:04d}.pt"
        if not ckpt_path.exists():
            print(f"Checkpoint {ckpt_path.name} not found — skipping")
            return

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    if not epoch_tag or epoch_tag == "newest":
        epoch_tag = ckpt_path.stem
    print(f"Loaded {ckpt_path.name} for comparison")

    with torch.no_grad():
        for idx in range(len(dataset)):
            waveform, target = dataset[idx]
            inp = waveform.unsqueeze(0).to(device)
            t0 = time.perf_counter()
            preds = model(inp)
            elapsed = time.perf_counter() - t0
            stem = dataset.stems[idx]

            meta = {
                "file": f"{stem}.wav",
                "input_samples": inp.shape[-1],
                "input_duration_s": round(inp.shape[-1] / 48000, 2),
                "inference_time_ms": round(elapsed * 1000, 2),
                "backbone_frames": model.backbone(inp).shape[-1],
            }
            _save_runtime_csv(output_dir, stem, meta, preds, epoch_tag, targets=target)
            _save_prediction_plots(output_dir, stem, preds, target, epoch_tag)
            _save_comparison_csv(output_dir, stem, preds, target, epoch_tag)
    print(f"Comparison saved to {output_dir}")


class _DatasetView(Dataset):
    """Lightweight view over a subset of a PsychoAcousticDataset, sharing the underlying data."""
    def __init__(self, dataset: PsychoAcousticDataset, indices: list[int]):
        n = len(dataset)
        for i in indices:
            if not 0 <= i < n:
                raise IndexError(
                    f"subset index {i} is out of range for dataset of size {n}"
                )
        self._dataset = dataset
        self._indices = indices
        self.stems = [dataset.stems[i] for i in indices]

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int):
        return self._dataset[self._indices[idx]]


def run_comparison(
    sound_dir: Path,
    labels_csv_path: Path,
    checkpoint_dir: Path,
    n_samples: int = 1,
    device_id: int = 0,
    subset_indices: list[int] | None = None,
    epochs: list[int | str] | None = None,
    dataset: PsychoAcousticDataset | None = None,
    audio_workers: int = 0,
):
    """Run inference on specified epochs and save plots/CSVs.

    Parameters
    ----------
    epochs : list[int | str] | None
        Epoch numbers to compare. Use "newest" for the latest checkpoint.
        Defaults to [0, "newest"].
    """
    if epochs is None:
        epochs = [0, "newest"]
    print("=" * 100)
    device = _get_device(device_id)
    if subset_indices is not None:
        if dataset is not None:
            dataset = _DatasetView(dataset, subset_indices)
        else:
            dataset = PsychoAcousticDataset(sound_dir, labels_csv_path, subset_indices=subset_indices,
                                            audio_workers=audio_workers)
    elif dataset is None:
        dataset = PsychoAcousticDataset(sound_dir, labels_csv_path, audio_workers=audio_workers)
    output_dir = Path(__file__).resolve().parent / "comparison"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    for ep in epochs:
        tag = f"epoch_{ep:04d}" if isinstance(ep, int) else ""
        _compare_epoch(dataset, checkpoint_dir, output_dir, n_samples, device=device, epoch=ep, epoch_tag=tag)
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
    labels_csv_path: Path,
    checkpoint_dir: Path,
    losses_dir: Path,
    epochs: int = 2,
    lr: float = 1e-3,
    batch_size: int = 2,
    device_id: int = 0,
    num_workers: int = 0,
    subset_indices: list[int] | None = None,
    dataset: PsychoAcousticDataset | None = None,
    audio_workers: int = 0,
) -> list[dict[str, float]]:
    print("=" * 100)
    device = _get_device(device_id)

    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    losses_dir = Path(losses_dir)
    losses_dir.mkdir(parents=True, exist_ok=True)
    csv_path = losses_dir / "losses.csv"
    plot_path = losses_dir / "losses.png"

    # ── Model ──
    stats_path = Path(__file__).parent.parent / "data" / "standardized_audio_files" / "training_set" / "visualization" / "parameter_average_per_time_segment.csv"
    biases = _load_time_biases(stats_path)
    model = PsychoacousticModel(initial_temporal_biases=biases).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)

    # ── Resume from latest checkpoint ──
    start_epoch, history = _resume_checkpoint(model, optimizer, checkpoint_dir)
    if start_epoch == 0:
        csv_path.unlink(missing_ok=True)
        print("Starting from scratch")
        ckpt_path = checkpoint_dir / "epoch_0000.pt"
        torch.save(
            {
                "epoch": 0,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": [],
            },
            ckpt_path,
        )
        print(f"Saved initial parameters to {ckpt_path.name}")
    elif history:
        rows = [{"epoch": i + 1, **h} for i, h in enumerate(history)]
        pd.DataFrame(rows).to_csv(csv_path, index=False)

    # ── Dataset ──
    if dataset is None:
        dataset = PsychoAcousticDataset(sound_dir, labels_csv_path, subset_indices=subset_indices,
                                        audio_workers=audio_workers)
    if len(dataset) == 0:
        print("No data found — nothing to train on.")
        return []
    print(f"Loaded {len(dataset)} audio-label pair(s)")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=_collate,
        num_workers=num_workers,
    )

    t_train_start = time.perf_counter()
    n_batches = len(loader)
    print(f"Training {epochs - start_epoch} epoch(s) ({n_batches} batches each)...")

    # ── Training loop ──
    for epoch in range(start_epoch, epochs):
        t_epoch = time.perf_counter()
        t_batch_start = time.perf_counter()
        epoch_losses: list[dict[str, float]] = []
        for batch_idx, (waveform, targets) in enumerate(loader):
            losses = _training_step(model, waveform, targets, optimizer, device)
            epoch_losses.append(losses)
            if (batch_idx + 1) % 100 == 0:
                print(f"  batch {batch_idx + 1}/{n_batches} ({time.perf_counter() - t_batch_start:.4f}s)")
                t_batch_start = time.perf_counter()
        t_total = time.perf_counter() - t_epoch

        avg_losses = {
            k: torch.tensor([b[k] for b in epoch_losses]).mean().item()
            for k in epoch_losses[0]
        }
        history.append(avg_losses)
        print(f"Epoch {epoch + 1}/{epochs} — loss: {avg_losses['total']:.6f} — {t_total:.4f}s")

        _log_epoch(epoch, avg_losses, model, optimizer, history, csv_path, plot_path, checkpoint_dir)

    return history
