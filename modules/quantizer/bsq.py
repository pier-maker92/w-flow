from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..configs import BSQConfig
from ..output_dataclasses import QuantizerOutput


class BinarySphericalQuantizer(nn.Module):
    """Binary Spherical Quantization (BSQ).

    Pipeline (per sample):
      z ∈ ℝ^{in_dim}
        → Linear(in_dim → dim)   [optional projection if in_dim ≠ dim]
        → L2-normalize            → z_hat ∈ S^{dim-1}
        → sign                    → b ∈ {-1,+1}^dim
        → scale by 1/√dim         → z_q ∈ S^{dim-1}  (unit-norm binary vector)
      STE: gradient passes through sign as identity

    Codebook is implicit: 2^dim possible codes (one per orthant of the sphere).
    No learned parameters beyond the optional projection.
    """

    def __init__(self, config: BSQConfig) -> None:
        super().__init__()
        self.dim = config.dim
        self.proj: Optional[nn.Module] = None
        if config.in_dim != config.dim:
            self.proj = nn.Linear(config.in_dim, config.dim, bias=False)

    def forward(self, z: torch.Tensor) -> QuantizerOutput:
        z_f = z.float()
        if self.proj is not None:
            z_f = self.proj(z_f)

        # L2-normalize onto unit sphere
        z_hat = F.normalize(z_f, dim=-1)   # ∈ S^{dim-1}

        # Binary quantization with straight-through estimator
        b = z_hat.sign()                    # {-1, +1}^dim  (sign(0)=0, negligible)
        z_q = b * (self.dim ** -0.5)        # scale so ||z_q||_2 = 1

        # STE: forward = z_q, backward = z_hat
        z_q_ste = z_hat + (z_q - z_hat).detach()

        # Cast back to original input dtype (bf16-safe)
        z_q_out = z_q_ste.to(z.dtype)

        # Commitment loss: push z_hat toward z_q (angular alignment)
        commitment_loss = F.mse_loss(z_hat, z_q.detach())

        return QuantizerOutput(
            z_q=z_q_out,
            commitment_loss=commitment_loss,
            codebook_loss=None,
            indices=None,
            perplexity=None,
        )
