import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import FSQConfig
from ..output_dataclasses import QuantizerOutput


class FSQ(nn.Module):
    """Finite Scalar Quantization (Mentzer et al., 2023).

    No codebook parameters, no commitment loss — pure straight-through rounding.
    levels = [L1, ..., Ld] where d = embedding_dim.
    Effective codebook size = prod(levels).
    """

    def __init__(self, config: FSQConfig) -> None:
        super().__init__()
        self.config = config
        levels = torch.tensor(config.levels, dtype=torch.float32)
        self.register_buffer("levels", levels)                    # [d]
        self.register_buffer("half_widths", (levels - 1) / 2)    # [d]

    def _bound(self, z: torch.Tensor) -> torch.Tensor:
        # Map R → [-(L-1)/2, (L-1)/2] per dimension via tanh
        return torch.tanh(z) * self.half_widths

    def forward(self, z: torch.FloatTensor) -> QuantizerOutput:
        # z: [B, d] or [B, T, d]
        bounded = self._bound(z)
        # Straight-through rounding
        z_q = bounded + (torch.round(bounded) - bounded).detach()

        # Scalar index: shift to [0, L-1], then base-conversion
        codes = (z_q + self.half_widths).long()    # [..., d]
        batch_shape = z.shape[:-1]
        flat_codes = codes.reshape(-1, len(self.config.levels))  # [N, d]
        indices = torch.zeros(flat_codes.shape[0], dtype=torch.long, device=z.device)
        multiplier = 1
        for i, L in enumerate(self.config.levels):
            indices = indices + flat_codes[:, i] * multiplier
            multiplier *= L
        indices = indices.reshape(batch_shape)

        num_codes = int(self.levels.prod().item())
        avg_probs = F.one_hot(indices.reshape(-1), num_codes).float().mean(0)
        perplexity = (-avg_probs * (avg_probs + 1e-10).log()).sum().exp()

        return QuantizerOutput(
            z_q=z_q,
            commitment_loss=None,
            codebook_loss=None,
            indices=indices,
            perplexity=perplexity,
        )
