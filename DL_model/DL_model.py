import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import PARAM_NAMES


class PsychoacousticModel(nn.Module):
    """Per-parameter time-varying predictions at native target resolution.

    A shared backbone extracts features at fine temporal resolution
    (~T/64 frames).  Each parameter has a dedicated head that
    adaptively pools to the exact number of frames the reference
    algorithm produces (e.g. 2500 for loudness vs 1 for SII), so no
    interpolation or NaN‑masking is needed at loss time.

    Forward:
        Input:  (batch, 1, n_samples)
        Output: dict {param_name: (batch, n_frames_param)}
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
            nn.Conv1d(8, 8, kernel_size=15, stride=4, padding=7),
            nn.ReLU(),
        )

        # Lightweight per-parameter heads: 8 → 1 channel
        self.heads = nn.ModuleDict({
            name: nn.Conv1d(8, 1, kernel_size=1) for name in PARAM_NAMES
        })

    def forward(
        self,
        waveform: torch.Tensor,
        target_n_frames: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(waveform)           # (B, 8, T_backbone)
        outputs: dict[str, torch.Tensor] = {}
        for name in PARAM_NAMES:
            out = self.heads[name](features)         # (B, 1, T_backbone)
            if target_n_frames is not None and name in target_n_frames:
                n = target_n_frames[name]
                if out.shape[-1] != n:
                    out = F.interpolate(
                        out, size=n, mode="linear", align_corners=False
                    )
            outputs[name] = out.squeeze(1)           # (B, n_frames_param)
        return outputs
