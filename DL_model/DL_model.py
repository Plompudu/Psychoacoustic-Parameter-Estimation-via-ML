import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import PARAM_NAMES


class PsychoacousticModel(nn.Module):
    """Per-parameter time-varying predictions at native target resolution.

    A shared backbone extracts features at ~T/8 temporal resolution with
    64 channels.  Each parameter has a dedicated head that adaptively
    pools to the exact number of frames the reference algorithm produces
    (e.g. 2500 for loudness vs 1 for SII), so every output matches its
    target dimension without further alignment.
    """

    def __init__(self):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict({
            name: nn.Conv1d(64, 1, kernel_size=1) for name in PARAM_NAMES
        })

    def forward(
        self,
        waveform: torch.Tensor,
        target_n_frames: dict[str, int] | None = None,
    ) -> dict[str, torch.Tensor]:
        features = self.backbone(waveform)           # (B, 64, T_backbone)
        outputs: dict[str, torch.Tensor] = {}
        for name in PARAM_NAMES:
            out = self.heads[name](features)         # (B, 1, T_backbone)
            if target_n_frames is not None and name in target_n_frames:
                n = target_n_frames[name]
                if out.shape[-1] != n:
                    out = F.interpolate(
                        out, size=n, mode="linear"
                    )
            outputs[name] = out.squeeze(1)           # (B, n_frames_param)
        return outputs
