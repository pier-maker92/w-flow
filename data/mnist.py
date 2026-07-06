"""MNIST dataset loader — uses HuggingFace datasets cache (load_dataset('mnist')).

Cache must exist at $HF_HOME/datasets/mnist (run scripts/download/download_mnist.py first).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


class MNISTDataset(Dataset):
    """MNIST from HF cache. Each example: image [1, 28, 28] float in [-1, 1]."""

    HF_ID = "mnist"

    def __init__(self, split: str) -> None:
        # split: "train" | "test"
        hf_home = os.environ["HF_HOME"]
        self.ds = load_dataset(
            self.HF_ID,
            split=split,
            cache_dir=os.path.join(hf_home, "datasets"),
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.ds[idx]
        img = np.array(example["image"], dtype=np.float32)   # [28, 28] uint8 → float
        img = img / 127.5 - 1.0                              # [-1, 1]
        img = torch.from_numpy(img).unsqueeze(0)             # [1, 28, 28]
        label = torch.tensor(example["label"], dtype=torch.long)
        return {"image": img, "label": label}
