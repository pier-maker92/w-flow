import torch
import torch.nn as nn

from ..configs import LinearDequantizerConfig
from ..output_dataclasses import DequantizerOutput


class LinearDequantizer(nn.Module):
    """Maps quantized feature vector z_q [B, in_dim] → x_tq_hat [B, out_dim]."""

    def __init__(self, config: LinearDequantizerConfig) -> None:
        super().__init__()
        self.linear = nn.Linear(config.in_dim, config.out_dim)
        nn.init.zeros_(self.linear.weight)
        nn.init.zeros_(self.linear.bias)

    def forward(self, z_q: torch.Tensor, x_tq: torch.Tensor = None) -> DequantizerOutput:
        return DequantizerOutput(x_tq_hat=self.linear(z_q))
