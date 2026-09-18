from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from .params import PARAM_NAMES

PARAM_COLORS = {
    "loudness_zwtv": "tab:blue",
    "sharpness_din_tv": "tab:green",
    "roughness_dw": "tab:orange",
    "tnr_ecma_perseg": "tab:purple",
    "sii_ansi": "tab:brown",
}


def global_reference_max(refs: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Maximum reference value per parameter over all source files."""
    maxima: dict[str, float] = {}
    for name in PARAM_NAMES:
        vmax = 0.0
        found = False
        for ref in refs.values():
            col = ref[name].to_numpy(dtype=np.float32)
            col = col[~np.isnan(col)]
            if col.size == 0:
                continue
            found = True
            c = float(col.max())
            if c > vmax:
                vmax = c
        maxima[name] = vmax if found and vmax > 0 else 1.0
    return maxima


def pred_to_target_indices(n_frames: int, n_pred: int) -> np.ndarray:
    """Map each model output index to the target-frame position it represents.

    The model produces ``MAX_i`` values evenly spaced over the segment's time
    grid (e.g. roughness: 10 values at frames 0, 50, ..., 450 of a 500-frame
    segment). For a full 500-frame segment loudness/sharpness map 1:1; sparse
    parameters align with the reference measurement grid.
    """
    if n_pred == n_frames:
        return np.arange(n_frames)
    return np.minimum(
        np.round(np.arange(n_pred) * n_frames / n_pred).astype(int),
        n_frames - 1,
    )


def plot_reference_vs_prediction(
    dataset,
    checkpoint_dir: Path,
    output_dir: Path,
    device: torch.device | None = None,
) -> dict[str, int] | None:
    """Scatter of reference-vs-prediction points per parameter for ``dataset``.

    Both axes are in % of the parameter's global reference maximum, so the
    red 1:1 diagonal ("0% Error Prediction") is comparable across parameters.
    Returns per-parameter point counts, or None if no checkpoint is found.
    """
    from .train_model import (
        PsychoacousticModel,
        _BIAS_STATS_PATH,
        _enumerate_devices,
        _get_device_name,
        _load_time_biases,
    )

    ckpt_files = sorted(Path(checkpoint_dir).glob("epoch_*.pt"))
    if not ckpt_files:
        print("No checkpoint found — skipping reference-vs-prediction plot")
        return None
    ckpt_path = ckpt_files[-1]
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if device is None:
        device, dev_name = _enumerate_devices()[0]
    else:
        dev_name = _get_device_name(device)

    biases = _load_time_biases(_BIAS_STATS_PATH)
    model = PsychoacousticModel(initial_temporal_biases=biases).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    scales = global_reference_max(dataset.refs)
    points: dict[str, tuple[list[float], list[float]]] = {
        name: ([], []) for name in PARAM_NAMES
    }

    print(f"Running {len(dataset)} segments from {dataset.sound_dir} through {ckpt_path.name}...")
    with torch.no_grad():
        for idx in range(len(dataset)):
            waveform, target = dataset[idx]
            inp = waveform.unsqueeze(0).to(device)
            preds = model(inp)
            for name in PARAM_NAMES:
                t = target[name].numpy()
                p = preds[name][0].cpu().numpy()
                n = t.shape[0]
                frame_idx = pred_to_target_indices(n, p.shape[0])
                ref = t[frame_idx]
                mask = ~np.isnan(ref)
                if not mask.any():
                    continue
                points[name][0].extend(p[mask].tolist())
                points[name][1].extend(ref[mask].tolist())

    counts = {name: len(v[1]) for name, v in points.items()}
    print(f"  reference-vs-prediction points: {counts}")

    fig, ax = plt.subplots(figsize=(8, 8))
    for name in PARAM_NAMES:
        t = np.asarray(points[name][1], dtype=float)
        p = np.asarray(points[name][0], dtype=float)
        if t.size == 0:
            continue
        scale = scales[name]
        ax.scatter(
            t / scale * 100.0,
            p / scale * 100.0,
            s=5,
            alpha=0.4,
            color=PARAM_COLORS[name],
            label=f"{name} (n={t.size})",
        )
    ax.plot([0, 100], [0, 100], color="red", linestyle="--",
            linewidth=2, label="0% Error Prediction")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Reference value [% of max]")
    ax.set_ylabel("Model prediction [% of max]")
    ax.set_title(
        f"Reference vs. prediction — {ckpt_path.stem} ({dev_name})"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="upper left")
    fig.tight_layout()

    output_path = Path(output_dir) / "reference_vs_prediction.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Saved: {output_path}")
    return counts


def plot_reference_vs_prediction_test(
    test_sound_dir: Path,
    references_path: Path,
    checkpoint_dir: Path,
    output_dir: Path,
    audio_workers: int = 0,
    device: torch.device | None = None,
) -> dict[str, int] | None:
    """Scatter over the full (unused) test set using the newest checkpoint."""
    from .train_model import PsychoAcousticDataset

    dataset = PsychoAcousticDataset(
        Path(test_sound_dir), references_path, audio_workers=audio_workers
    )
    return plot_reference_vs_prediction(dataset, checkpoint_dir, output_dir, device=device)


def main():
    repo_root = Path(__file__).resolve().parent.parent
    references_path = repo_root / "data" / "reference_values" / "references.csv"
    test_dir = repo_root / "data" / "test_set"
    checkpoint_dir = Path(__file__).resolve().parent / "epochs"
    output_dir = Path(__file__).resolve().parent / "comparison"

    counts = plot_reference_vs_prediction_test(
        test_dir, references_path, checkpoint_dir, output_dir
    )
    if counts is None:
        print("No checkpoint found in", checkpoint_dir)
        print("Run training first to produce a checkpoint.")
    else:
        print("Point counts per parameter:", counts)


if __name__ == "__main__":
    main()