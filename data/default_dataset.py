from dataclasses import dataclass
import torch


@dataclass(kw_only=True)
class ImageCollator:
    """Default collator for image datasets (Phase 2-3).

    Expects each feature dict to have an "image" key that is already a float
    tensor in [-1, 1] (as returned by MNISTDataset, FashionMNISTDataset, etc.).
    x0 is sampled from N(0, I) inside the model's forward, not here.
    """

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        x1 = torch.stack([f["image"] for f in features])   # [B, C, H, W]
        return {"x1": x1}
