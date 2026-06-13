import torch
import torch.nn as nn

from .params import PARAM_NAMES


class PsychoacousticModel(nn.Module):
    """Per-parameter time-varying predictions at native target resolution.

    Uses a shared backbone + per-parameter heads instead of a single
    monolithic model.  The backbone is the heavy feature extractor (3
    Conv1D layers, run once per sample), while each head is a cheap 1x1
    Conv + AdaptiveAvgPool1d that reads out different views of those
    features at different temporal resolutions.

    Benefits of this split:
      - Efficiency: backbone runs once per waveform, not once per param.
      - Flexibility: adding a new parameter means adding a tiny head
        without retraining the backbone from scratch.
      - No alignment boilerplate: each head adaptively pools to the
        exact frame count its reference algorithm produces (e.g. ~2500
        frames for loudness vs 1 scalar for SII).
    """

    def __init__(self, param_frame_counts: dict[str, int] | None = None):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict()
        for name in PARAM_NAMES:
            self.heads[name] = nn.Sequential(
                nn.Conv1d(64, 1, kernel_size=1),
                nn.AdaptiveAvgPool1d(param_frame_counts[name])
            )

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.backbone(waveform)           # (B, 64, T_backbone)
        outputs: dict[str, torch.Tensor] = {}
        for name in PARAM_NAMES:
            outputs[name] = self.heads[name](features).squeeze(1)
        return outputs
