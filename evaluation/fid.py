"""FID computation for Phase 2-3 image experiments.

Uses torchmetrics FrechetInceptionDistance to avoid re-implementing Inception features.
"""
from typing import Optional
import torch
from torch.utils.data import DataLoader


def compute_fid(
    real_loader: DataLoader,
    generated_samples: torch.Tensor,   # [N, C, H, W] float in [-1, 1]
    device: torch.device,
    max_samples: Optional[int] = None,
) -> float:
    try:
        from torchmetrics.image.fid import FrechetInceptionDistance
    except ImportError as e:
        raise ImportError("torchmetrics[image] required for FID: pip install torchmetrics[image]") from e

    fid = FrechetInceptionDistance(feature=2048, normalize=True).to(device)

    # Real samples
    n = 0
    for batch in real_loader:
        imgs = batch["image"].to(device)
        fid.update(_to_uint8(imgs), real=True)
        n += imgs.shape[0]
        if max_samples is not None and n >= max_samples:
            break

    # Generated samples
    fid.update(_to_uint8(generated_samples.to(device)), real=False)
    return float(fid.compute())


def _to_uint8(x: torch.Tensor) -> torch.Tensor:
    # x: float in [-1, 1] → uint8 in [0, 255], shape [B, C, H, W]
    return ((x.clamp(-1, 1) + 1) * 127.5).to(torch.uint8)
