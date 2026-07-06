import os
from typing import Any

from torch.utils.data import Dataset

from modules.configs import TrainingConfig


def build_dataset(training_cfg: TrainingConfig, split: str) -> tuple[Dataset, Any]:
    """Returns (dataset, collator) for the given dataset_name.

    Available datasets:
      "gaussians"      — 8-Gaussians (generated on-the-fly, no download needed)
      "swissroll"      — Swiss Roll 2D (generated on-the-fly)
      "mnist"          — MNIST 28x28, requires HF cache at $HF_HOME/datasets/mnist
      "fashion_mnist"  — Fashion-MNIST 28x28, requires HF cache at $HF_HOME/datasets/fashion_mnist
    """
    name = training_cfg.dataset_name if isinstance(training_cfg, TrainingConfig) else training_cfg["dataset_name"]

    if name == "gaussians":
        from data.toy_2d import GaussiansDataset, Toy2DCollator
        ds = GaussiansDataset(
            n_samples=training_cfg.gaussians_n,
            n_gaussians=8,
            radius=training_cfg.gaussians_radius,
            std=training_cfg.gaussians_std,
            seed=training_cfg.seed,
        )
        return ds, Toy2DCollator(source_std=1.0)

    if name == "swissroll":
        from data.toy_2d import SwissRollDataset, Toy2DCollator
        ds = SwissRollDataset(
            n_samples=100000,
            noise=training_cfg.swissroll_noise,
            seed=training_cfg.seed,
        )
        return ds, Toy2DCollator(source_std=1.0)

    if name == "mnist":
        from data.mnist import MNISTDataset
        from data.default_dataset import ImageCollator
        ds = MNISTDataset(split=split)
        return ds, ImageCollator()

    if name == "fashion_mnist":
        from data.fashion_mnist import FashionMNISTDataset
        from data.default_dataset import ImageCollator
        ds = FashionMNISTDataset(split=split)
        return ds, ImageCollator()

    raise ValueError(f"unknown dataset: {name!r}")


def wandb_init(training_cfg: TrainingConfig, cfg_dict: dict[str, Any]) -> None:
    if training_cfg.report_to != "wandb":
        return
    import wandb
    wandb.init(
        project=training_cfg.wandb_project,
        name=training_cfg.wandb_run_name,
        config=cfg_dict,
    )
