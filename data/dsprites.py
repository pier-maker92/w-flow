"""dSprites dataset loader (64×64, 1 channel, 5 independent latent factors)."""
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset


class DSpritesDataset(Dataset):
    """Loads the dSprites .npz file from disk.

    Download: https://github.com/google-deepmind/dsprites-dataset
    Expected path: $SCRATCH/datasets/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz
    """

    def __init__(self, npz_path: str, split: str, seed: int, val_fraction: float = 0.1) -> None:
        data = np.load(npz_path, allow_pickle=True, encoding="latin1")
        imgs = data["imgs"]   # [737280, 64, 64] uint8 {0, 1}
        rng = np.random.default_rng(seed)
        idx = rng.permutation(len(imgs))
        n_val = int(len(idx) * val_fraction)
        if split == "train":
            idx = idx[n_val:]
        elif split == "test":
            idx = idx[:n_val]
        else:
            raise ValueError(f"unknown split '{split}'")
        # Store as float32 [0, 1] with a channel dim added in __getitem__
        self.imgs = imgs[idx]
        self.latents = data["latents_values"][idx]   # [N, 6] for disentanglement eval

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img = torch.from_numpy(self.imgs[idx]).float().unsqueeze(0)   # [1, 64, 64]
        img = img * 2.0 - 1.0   # scale to [-1, 1]
        return {"image": img, "latents": torch.from_numpy(self.latents[idx]).float()}
