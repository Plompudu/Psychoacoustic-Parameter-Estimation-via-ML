import torch

from .params import PARAM_NAMES


def _trim_or_pad(p: torch.Tensor, t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Align *p* and *t* along the trailing dimension."""
    plen, tlen = p.shape[-1], t.shape[-1]
    if plen == tlen:
        return p, t
    if plen < tlen:
        return p, t[..., :plen]
    import torch.nn.functional as F
    p = F.interpolate(p.unsqueeze(1), size=tlen, mode="linear", align_corners=False).squeeze(1)
    return p, t


def compute_loss(
    model: torch.nn.Module,
    preds: dict[str, torch.Tensor],
    targets: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per‑parameter MSE summed over all parameters.

    NaN positions (genuine reference failures) are masked out.
    The time dimension of *preds* and *targets* is aligned by trimming.
    """
    device = next(model.parameters()).device
    losses = {}
    total = torch.tensor(0.0, device=device)

    for name in PARAM_NAMES:
        p, t = preds[name], targets[name]
        p, t = _trim_or_pad(p, t)

        mask = ~torch.isnan(t)
        if not mask.any():
            losses[name] = torch.tensor(0.0, device=p.device)
            continue

        p_masked, t_masked = p[mask], t[mask]
        loss = ((p_masked - t_masked) ** 2).mean()
        losses[name] = loss
        total = total + loss

    losses["total"] = total
    return losses
