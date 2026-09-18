from pathlib import Path

import pandas as pd
import torch
import torch.nn.functional as F

from .params import PARAM_NAMES

# describe()-style CSV with a "std" row per parameter, produced over the
# training set. Used by the variance-normalized mode to turn each parameter's
# raw MSE into a variance-normalized loss so no single parameter's scale
# dominates `total`.
_STATS_PATH = (
    Path(__file__).resolve().parent.parent
    / "data"
    / "reference_values"
    / "variance.csv"
)

_param_variances: dict[str, float] | None = None


def _get_param_variances(stats_path: Path = _STATS_PATH) -> dict[str, float]:
    """Load and cache per-parameter variance (std**2) from the stats CSV."""
    global _param_variances
    if _param_variances is None:
        df = pd.read_csv(stats_path, index_col=0)
        _param_variances = {
            name: float(df.loc["std", name]) ** 2 for name in PARAM_NAMES
        }
    return _param_variances


def compute_loss(
    model: torch.nn.Module,
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
    vector_wise: bool = True,
) -> dict[str, torch.Tensor]:
    """Per‑parameter MSE, combined into one of two `total` loss modes.

    NaN positions within a parameter (genuine reference failures) are masked
    out. If a parameter has no valid reference values at all in this batch,
    its loss is NaN (not 0.0) so it neither falsely looks perfect nor
    contaminates `total` — total only aggregates over parameters that had
    data. Predictions are pooled to match each target's frame count when
    they differ.

    `losses[name]` stays a raw, un-normalized MSE per parameter for
    interpretable per-parameter logging.

    `total` is controlled by `vector_wise`:
      - True  (vector-wise): every valid (prediction, target) pair across all
              frames and parameters is concatenated into one vector and the
              mean squared error over it is returned. No variance
              normalization.
      - False (variance-normalized): sums each parameter's MSE divided by
              that parameter's training-set variance, so parameters on very
              different natural scales (e.g. loudness_zwtv ~O(10) vs.
              sii_ansi ~O(0.01)) contribute comparably to the gradient and
              to whatever is driving the LR scheduler.
    """
    device = next(model.parameters()).device
    losses = {}
    vec_pred: list[torch.Tensor] = []
    vec_target: list[torch.Tensor] = []
    valid_params: list[str] = []

    for name in PARAM_NAMES:
        prediction, target = preds[name], targets[name]

        if prediction.shape[-1] != target.shape[-1]:
            prediction = F.adaptive_avg_pool1d(
                prediction.unsqueeze(1), target.shape[-1]
            ).squeeze(1)

        mask = ~torch.isnan(target)
        if not mask.any():
            # No valid reference values for this parameter in this batch
            # (e.g. tnr_ecma_perseg with no detected tones, or sii_ansi).
            # NaN signals "not evaluated" — 0.0 would falsely look perfect.
            losses[name] = torch.tensor(float("nan"), device=device)
            continue

        prediction_masked = prediction[mask]
        target_masked = target[mask]

        losses[name] = ((prediction_masked - target_masked) ** 2).mean()
        vec_pred.append(prediction_masked)
        vec_target.append(target_masked)
        valid_params.append(name)

    if vec_pred and vec_target:
        if vector_wise:
            losses["total"] = (
                (torch.cat(vec_pred) - torch.cat(vec_target)) ** 2
            ).mean()
        else:
            variances = _get_param_variances()
            losses["total"] = torch.stack(
                [losses[name] / variances[name] for name in valid_params]
            ).sum()
    else:
        losses["total"] = torch.tensor(float("nan"), device=device)

    return losses