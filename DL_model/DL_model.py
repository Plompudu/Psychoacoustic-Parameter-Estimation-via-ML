import torch
import torch.nn as nn
import torch.nn.functional as F

from .params import PARAM_NAMES


class PsychoacousticModel(nn.Module):
    def __init__(
        self,
        initial_temporal_biases: dict[str, torch.Tensor] | None = None,
    ):
        super().__init__()
        self.backbone = nn.Sequential(
            # Stage 1 — no temporal compression, preserves full resolution
            nn.Conv1d(1, 10, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm1d(10),
            nn.ReLU(),
            # Stage 2
            nn.Conv1d(10, 20, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(20),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Stage 3
            nn.Conv1d(20, 40, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(40),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Stage 4
            nn.Conv1d(40, 60, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(60),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
            # Stage 5
            nn.Conv1d(60, 80, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(80),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2),
        )

        MAX = [500, 500, 9, 2, 1]
        self.heads = nn.ModuleDict()
        for i, name in enumerate(PARAM_NAMES):
            self.heads[name] = nn.Sequential(
                nn.Conv1d(80, 1, kernel_size=1),
                nn.AdaptiveAvgPool1d(MAX[i]),
            )

            bias = torch.zeros(1, MAX[i])
            if initial_temporal_biases is not None and name in initial_temporal_biases:
                bias = initial_temporal_biases[name].view(1, -1)
            self.register_parameter(f"{name}_bias", nn.Parameter(bias))

    def forward(self, waveform: torch.Tensor) -> dict[str, torch.Tensor]:
        backbone_output = self.backbone(waveform)
        final_outputs: dict[str, torch.Tensor] = {}
        for name in PARAM_NAMES:
            out = self.heads[name](backbone_output).squeeze(1)
            bias = getattr(self, f"{name}_bias")
            final_outputs[name] = out + bias
        return final_outputs
