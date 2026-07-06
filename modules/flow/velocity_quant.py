from typing import Optional
import torch
import torch.nn as nn

from ..output_dataclasses import QuantizerOutput


class VelocityQuantizedFlow(nn.Module):
    """Section 4 variant — quantizes the velocity field v_θ(x_t, t) instead of x_t.

    Wraps any backbone. The quantizer receives the predicted velocity and returns
    a discrete approximation via straight-through. This guarantees C0 continuity
    of the trajectory (no discrete jump in x space) at the cost of the velocity
    field being restricted to a finite set of directions.

    self._last_quant_output is populated on each forward for loss logging.
    """

    def __init__(self, backbone: nn.Module, quantizer: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.quantizer = quantizer
        self._last_quant_output: Optional[QuantizerOutput] = None

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v = self.backbone(x, t)
        quant_out = self.quantizer(v)
        self._last_quant_output = quant_out
        return quant_out.z_q   # straight-through gradient flows to backbone
