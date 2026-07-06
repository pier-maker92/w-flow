import torch
import torch.nn as nn

from ..configs import ResidualDequantizerConfig
from ..output_dataclasses import DequantizerOutput


class ResidualDequantizer(nn.Module):
    """Metodo B — bijective residual dequantizer.

    z̃ = (1-α) z_q + α x_tq
    R  = x_tq - z_q          (stored residual; must be applied when swapping codes)

    α=0 → z̃ = z_q (pure discrete)
    α=1 → z̃ = x_tq (identity; no quantization effect)

    WARNING: swapping z_q at inference without providing the matching residual
    causes severe OOD deviation. See §1 of the research plan.
    """

    def __init__(self, config: ResidualDequantizerConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, z_q: torch.FloatTensor, x_tq: torch.FloatTensor) -> DequantizerOutput:
        alpha = self.config.alpha
        residual = x_tq - z_q
        x_tq_hat = (1 - alpha) * z_q + alpha * x_tq
        return DequantizerOutput(x_tq_hat=x_tq_hat, residual=residual)
