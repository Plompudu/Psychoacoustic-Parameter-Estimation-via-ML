import torch

from .forward import forward
from .params import PARAM_NAMES
from .compute_loss import compute_loss


def _valid_frame_count(t: torch.Tensor) -> int:
    """Number of leading frames that are non‑NaN across the batch."""
    valid = ~torch.isnan(t)
    any_valid = valid.any(dim=0)
    idxs = torch.where(any_valid)[0]
    return idxs[-1].item() + 1 if len(idxs) > 0 else 1


def training_step(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    targets: dict[str, torch.Tensor],
    optimizer: torch.optim.Optimizer,
) -> dict[str, float]:
    """Run forward → compute_loss → backward → optimizer step."""
    target_n_frames = {name: _valid_frame_count(targets[name])
                       for name in PARAM_NAMES}
    preds = forward(model, waveform, target_n_frames)
    losses = compute_loss(model, preds, targets)
    optimizer.zero_grad()
    losses["total"].backward()
    optimizer.step()
    return {k: v.item() for k, v in losses.items()}
