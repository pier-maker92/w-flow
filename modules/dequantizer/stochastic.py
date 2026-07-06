import torch
import torch.nn as nn

from ..configs import StochasticDequantizerConfig
from ..output_dataclasses import DequantizerOutput


class StochasticDequantizer(nn.Module):
    """Metodo A — z̃ = z_q + σε, ε ~ N(0, I).

    At inference, σ=0 so z̃ = z_q. Stochasticity is training-only and gives
    the ODE maximum freedom to recover detail after the discrete bottleneck.
    """

    def __init__(self, config: StochasticDequantizerConfig) -> None:
        super().__init__()
        self.config = config

    def forward(self, z_q: torch.FloatTensor) -> DequantizerOutput:
        if self.training and self.config.sigma > 0:
            noise = torch.randn_like(z_q) * self.config.sigma
            x_tq_hat = z_q + noise
        else:
            x_tq_hat = z_q
        return DequantizerOutput(x_tq_hat=x_tq_hat, residual=None)
