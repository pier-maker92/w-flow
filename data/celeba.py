"""CelebA 64×64 dataset loader."""
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
import torchvision.datasets as tvd


class CelebADataset(Dataset):
    def __init__(self, root: str, split: str) -> None:
        self.ds = tvd.CelebA(
            root=root,
            split=split,   # "train" | "valid" | "test"
            download=False,
            transform=T.Compose([
                T.CenterCrop(140),
                T.Resize(64),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]),
        )

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        img, attrs = self.ds[idx]
        return {"image": img, "attrs": attrs}
