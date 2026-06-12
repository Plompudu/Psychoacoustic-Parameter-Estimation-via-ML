import torch

from .params import PARAM_NAMES


def compute_loss(
    model: torch.nn.Module,
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per‑parameter MSE summed over all parameters.

    NaN positions (genuine reference failures) are masked out.
    Predictions are assumed to already match target time dimensions.
    """
    device = next(model.parameters()).device
    losses = {}
    total = torch.tensor(0.0, device=device)

    for name in PARAM_NAMES:
        prediction, target = preds[name], targets[name]

        mask = ~torch.isnan(target)
        if not mask.any():
            losses[name] = torch.tensor(0.0, device=device)
            continue

        prediction_masked = prediction[mask]
        target_masked = target[mask]

        loss = ((prediction_masked - target_masked) ** 2).mean()
        losses[name] = loss
        total = total + loss

    losses["total"] = total
    return losses
