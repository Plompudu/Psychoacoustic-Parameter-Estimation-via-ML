import torch


def forward(
    model: torch.nn.Module,
    waveform: torch.Tensor,
    target_n_frames: dict[str, int] | None = None,
) -> dict[str, torch.Tensor]:
    """Run the forward pass and return per-parameter predictions.

    When *target_n_frames* is provided the model adaptively pools each
    parameter head to that exact number of frames.
    """
    return model(waveform, target_n_frames=target_n_frames)
