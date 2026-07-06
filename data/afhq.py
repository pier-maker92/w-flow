"""AFHQ v2 64×64 dataset loader.

Expected structure on $SCRATCH:
  $SCRATCH/datasets/afhq/{train,val}/{cat,dog,wild}/*.jpg
"""
from pathlib import Path
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision.datasets import ImageFolder


class AFHQDataset(Dataset):
    def __init__(self, root: str, split: str) -> None:
        # split: "train" | "val"
        split_dir = Path(root) / split
        self.ds = ImageFolder(
            root=str(split_dir),
            transform=T.Compose([
                T.Resize(64),
                T.CenterCrop(64),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, label = self.ds[idx]
        return {"image": img, "label": torch.tensor(label, dtype=torch.long)}
