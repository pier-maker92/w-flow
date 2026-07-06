from typing import Callable, Optional
import torch

from ..output_dataclasses import QuantizerOutput


class RectifiedFlow:
    """Standard Rectified Flow loss (Liu et al., 2022).

    Operates independently of the model architecture; receives a velocity_fn callable.
    Quantization bottleneck is handled at the FlowQuant level, not here.
    """

    def loss(
        self,
        x0: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: torch.FloatTensor,
        velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ) -> tuple[torch.FloatTensor, Optional[QuantizerOutput]]:
        """Standard FM loss: E[||v_θ(x_t, t) - (x1 - x0)||²].

        velocity_fn may internally call a VelocityQuantizedFlow wrapper that returns
        a quantized velocity; in that case it populates self._last_quant_output.
        Returns (fm_loss, quantizer_output_or_None).
        """
        shape_dims = tuple(1 for _ in range(x0.ndim - 1))
        t_broad = t.view(-1, *shape_dims)
        x_t = (1 - t_broad) * x0 + t_broad * x1
        v_target = x1 - x0

        # velocity_fn may be a VelocityQuantizedFlow; capture quant info if available
        quant_out: Optional[QuantizerOutput] = None
        if hasattr(velocity_fn, "__self__") and hasattr(velocity_fn.__self__, "_last_quant_output"):
            velocity_fn.__self__._last_quant_output = None

        v_pred = velocity_fn(x_t, t)

        if hasattr(velocity_fn, "__self__") and hasattr(velocity_fn.__self__, "_last_quant_output"):
            quant_out = velocity_fn.__self__._last_quant_output

        fm_loss = (v_pred - v_target).pow(2).mean()
        return fm_loss, quant_out

    @staticmethod
    def interpolate(
        x0: torch.FloatTensor,
        x1: torch.FloatTensor,
        t: float | torch.FloatTensor,
    ) -> torch.FloatTensor:
        """Linear interpolation: x_t = (1-t)*x0 + t*x1."""
        if isinstance(t, float):
            return (1 - t) * x0 + t * x1
        shape_dims = tuple(1 for _ in range(x0.ndim - 1))
        t_broad = t.view(-1, *shape_dims)
        return (1 - t_broad) * x0 + t_broad * x1
