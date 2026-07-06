from dataclasses import dataclass
import numpy as np
import torch
from torch.utils.data import Dataset


class GaussiansDataset(Dataset):
    """n_gaussians Gaussians arranged uniformly on a circle of given radius."""

    def __init__(
        self,
        n_samples: int,
        n_gaussians: int,
        radius: float,
        std: float,
        seed: int,
    ) -> None:
        rng = np.random.default_rng(seed)
        angles = np.linspace(0, 2 * np.pi, n_gaussians, endpoint=False)
        centers = radius * np.stack([np.cos(angles), np.sin(angles)], axis=-1)  # [K, 2]
        labels = rng.integers(0, n_gaussians, size=n_samples)
        noise = rng.normal(0, std, size=(n_samples, 2))
        self.data = torch.from_numpy((centers[labels] + noise).astype(np.float32))

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x1": self.data[idx]}


class SwissRollDataset(Dataset):
    """2D projection of sklearn's swiss_roll (X[:,0] and X[:,2]), z-normalized."""

    def __init__(self, n_samples: int, noise: float, seed: int) -> None:
        from sklearn.datasets import make_swiss_roll
        X, _ = make_swiss_roll(n_samples=n_samples, noise=noise, random_state=seed)
        data = np.stack([X[:, 0], X[:, 2]], axis=-1).astype(np.float32)
        data = (data - data.mean(0)) / data.std(0)
        self.data = torch.from_numpy(data)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {"x1": self.data[idx]}


@dataclass(kw_only=True)
class Toy2DCollator:
    source_std: float    # std of source p0 (typically 1.0)

    def __call__(self, features: list[dict]) -> dict[str, torch.Tensor]:
        x1 = torch.stack([f["x1"] for f in features])
        x0 = torch.randn_like(x1) * self.source_std
        return {"x1": x1, "x0_hint": x0}
        # x0_hint is informational; the model re-samples x0 independently in forward()
