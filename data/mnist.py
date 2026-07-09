"""MNIST dataset loader — uses HuggingFace datasets cache (load_dataset('mnist')).

Cache must exist at $HF_HOME/datasets/mnist (run scripts/download/download_mnist.py first).
"""

import glob
import os
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset


def _load_local_mnist_arrow(split: str):
    """Load the cached MNIST arrow shard directly, with no Hub resolution.

    On the compute nodes HF_HUB_OFFLINE is set and even the bare "mnist" repo id
    triggers a Hub lookup. The arrow shards produced by an earlier download live at
    <datasets-cache>/mnist/mnist/<version>/<hash>/mnist-<split>.arrow — we read them
    straight off disk instead.
    """
    from datasets import Dataset as HFDataset

    roots = [
        os.environ.get("HF_DATASETS_CACHE", ""),
        os.path.join(os.environ.get("HF_HOME", ""), "datasets"),
        os.path.join(os.environ.get("SCRATCH", ""), ".cache/huggingface/datasets"),
    ]
    for root in roots:
        if not root:
            continue
        matches = glob.glob(
            os.path.join(root, "mnist", "mnist", "*", "*", f"mnist-{split}.arrow")
        )
        if matches:
            return HFDataset.from_file(matches[0])
    raise FileNotFoundError(f"no local MNIST arrow shard found for split={split}")


class MNISTDataset(Dataset):
    """MNIST from HF cache. Each example: image [1, 28, 28] float in [-1, 1]."""

    HF_ID = "ylecun/mnist"

    def __init__(self, split: str) -> None:
        # split: "train" | "test"
        try:
            self.ds = load_dataset(self.HF_ID, split=split)
        except Exception:
            # Fully offline: read the cached arrow shard directly (no Hub lookup).
            self.ds = _load_local_mnist_arrow(split)

    def __len__(self) -> int:
        return len(self.ds)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        example = self.ds[idx]
        raw = example["image"]
        # Loading the arrow directly may not restore the HF Image feature, so the
        # field can be a PIL image, a {"bytes": ...} dict, an ndarray or a list.
        if isinstance(raw, dict):
            import io
            from PIL import Image
            raw = Image.open(io.BytesIO(raw["bytes"]))
        img = np.array(raw, dtype=np.float32)  # [28, 28] uint8 → float
        img = img / 127.5 - 1.0  # [-1, 1]
        img = torch.from_numpy(img).unsqueeze(0)  # [1, 28, 28]
        label = torch.tensor(example["label"], dtype=torch.long)
        return {"image": img, "label": label}
