"""Disentanglement evaluation for Phase 2 (dSprites).

Implements the Factor VAE / DCI score: swap a single latent factor across two samples
and verify that only the corresponding image attribute changes.
"""
from typing import Callable
import torch
import numpy as np


def factor_swap_eval(
    encode_fn: Callable[[torch.Tensor], torch.Tensor],  # image → code index [B]
    dataset,     # DSpritesDataset; must expose .latents [N, 6]
    n_pairs: int,
    factor_idx: int,  # which latent factor to swap (0=shape, 1=scale, 2=orient, 3=posX, 4=posY)
    device: torch.device,
    seed: int,
) -> dict[str, float]:
    """For n_pairs of images sharing the same factor_idx value:
    swap their codes; measure how often the output code changes.

    A good disentangled code should be invariant to intra-factor variation
    and should change when inter-factor is swapped.
    """
    raise NotImplementedError(
        "Implement factor_swap_eval once the model's encode_fn is available. "
        "Strategy: sample pairs with matching factor_idx latent, run encode_fn, "
        "compare code indices."
    )


def dci_score(
    codes: np.ndarray,     # [N, code_dim]
    latents: np.ndarray,   # [N, n_factors]
) -> dict[str, float]:
    """Disentanglement-Completeness-Informativeness (Eastwood & Williams, 2018).

    Fits a lasso regressor from codes → each latent factor; reports
    D (how unique each code dim is), C (how well each factor is captured),
    I (held-out prediction accuracy).
    """
    raise NotImplementedError(
        "Implement DCI score using sklearn LassoCV per latent factor."
    )
