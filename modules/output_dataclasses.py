from dataclasses import dataclass
from typing import Optional
import torch
from transformers.utils import ModelOutput


@dataclass
class QuantizerOutput(ModelOutput):
    z_q: Optional[torch.FloatTensor] = None           # quantized vector (straight-through)
    commitment_loss: Optional[torch.FloatTensor] = None
    codebook_loss: Optional[torch.FloatTensor] = None  # None for FSQ (no codebook loss)
    indices: Optional[torch.LongTensor] = None         # codebook indices
    perplexity: Optional[torch.FloatTensor] = None     # effective codebook usage


@dataclass
class DequantizerOutput(ModelOutput):
    x_tq_hat: Optional[torch.FloatTensor] = None   # dequantized state to resume ODE from
    residual: Optional[torch.FloatTensor] = None   # x_tq - z_q (Metodo B only)


@dataclass
class SolverOutput(ModelOutput):
    x1: Optional[torch.FloatTensor] = None                  # final generated sample
    trajectory: Optional[torch.FloatTensor] = None          # [B, num_steps+1, *D] if requested
    quantizer_output: Optional[QuantizerOutput] = None      # bottleneck info at t_q


@dataclass
class FlowOutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None               # total loss
    fm_loss: Optional[torch.FloatTensor] = None            # flow matching MSE
    commitment_loss: Optional[torch.FloatTensor] = None    # VQ commitment loss
    codebook_loss: Optional[torch.FloatTensor] = None      # VQ codebook loss
    quantizer_output: Optional[QuantizerOutput] = None
