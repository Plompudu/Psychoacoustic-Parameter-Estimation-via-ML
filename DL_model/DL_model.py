import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import PARAM_NAMES


class PsychoacousticModel(nn.Module):
    def __init__(
        self,
        param_frame_counts: dict[str, int] | None = None,
        initial_temporal_biases: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv1d(1, 24, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(24, 48, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        self.heads = nn.ModuleDict()
        for name in PARAM_NAMES:
            T = param_frame_counts[name]
            self.heads[name] = nn.Sequential(
                nn.Conv1d(64, 16, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(16, 1, kernel_size=1),
                nn.AdaptiveAvgPool1d(T),
            )

            bias = torch.zeros(1, T)
            if initial_temporal_biases is not None and name in initial_temporal_biases:
                b = initial_temporal_biases[name]
                if b.numel() > T:
                    b = F.interpolate(b.view(1, 1, -1), size=T, mode="linear", align_corners=False).view(-1)
                elif b.numel() < T:
                    b = F.interpolate(b.view(1, 1, -1), size=T, mode="linear", align_corners=False).view(-1)
                bias = b.view(1, -1)
            self.register_parameter(f"{name}_bias", nn.Parameter(bias))

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        final_outputs: dict[str, torch.Tensor] = {}

        backbone_output = self.backbone(waveform)
        for name in PARAM_NAMES:
            out = self.heads[name](backbone_output).squeeze(1)
            bias = getattr(self, f"{name}_bias")
            final_outputs[name] = out + bias

        return final_outputs
