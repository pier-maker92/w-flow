from typing import Optional
import torch
import torch.nn as nn

from ..configs import DiTBackboneConfig


class DiTBackbone(nn.Module):
    """Diffusion Transformer (DiT) velocity network for Phase 3 image experiments.

    Patch-based ViT: patchify → transformer blocks → unpatchify.
    Time conditioning via adaLN-Zero (per-block scale/shift from t embedding).
    """

    def __init__(self, config: DiTBackboneConfig, in_channels: int, image_size: int) -> None:
        super().__init__()
        self.config = config
        self.in_channels = in_channels
        self.image_size = image_size
        raise NotImplementedError(
            "DiTBackbone is a Phase 3 component. Implement patchify, transformer "
            "blocks with adaLN-Zero time conditioning, and unpatchify before using."
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, C, H, W], t: [B] → velocity: [B, C, H, W]
        raise NotImplementedError
