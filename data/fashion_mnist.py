"""Fashion-MNIST dataset loader — uses HuggingFace datasets cache.

Cache must exist at $HF_HOME/datasets/fashion_mnist (run scripts/download/download_mnist.py first).
"""
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset

FASHION_LABELS = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
]


class FashionMNISTDataset(Dataset):
    """Fashion-MNIST from HF cache. Same image format as MNISTDataset: [1, 28, 28] in [-1, 1]."""

    HF_ID = "fashion_mnist"

    def __init__(self, split: str) -> None:
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
        img = np.array(example["image"], dtype=np.float32)
        img = img / 127.5 - 1.0
        img = torch.from_numpy(img).unsqueeze(0)   # [1, 28, 28]
        label = torch.tensor(example["label"], dtype=torch.long)
        return {"image": img, "label": label}
