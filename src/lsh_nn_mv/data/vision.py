"""Vision dataset utilities (MNIST, CIFAR-10)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from .splits import TrainValTestSplit
from ..utils.seed import set_all_seeds


def _default_transform(dataset: str) -> transforms.Compose:
    if dataset.lower() == "mnist":
        return transforms.Compose([transforms.ToTensor()])
    if dataset.lower() == "cifar10":
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        )
    raise ValueError(f"Unsupported dataset: {dataset}")


def get_vision_dataloaders(
    dataset: str,
    root: str | Path,
    batch_size: int,
    val_batch_size: int | None = None,
    num_workers: int = 0,
    seed: int = 0,
    val_fraction: float = 0.1667,
    test_fraction: float = 0.1667,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Return train/validation/test dataloaders for a vision dataset."""

    set_all_seeds(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    dataset_lower = dataset.lower()
    transform = _default_transform(dataset_lower)

    if dataset_lower == "mnist":
        ds = datasets.MNIST(root=root, train=True, transform=transform, download=True)
        test_ds = datasets.MNIST(root=root, train=False, transform=transform, download=True)
    elif dataset_lower == "cifar10":
        ds = datasets.CIFAR10(root=root, train=True, transform=transform, download=True)
        test_ds = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")

    n_total = len(ds)
    split = TrainValTestSplit.from_sizes(
        n_total, 1 - val_fraction - test_fraction, val_fraction, seed
    )

    train_subset, val_subset = random_split(
        ds,
        [len(split.train_indices), len(split.val_indices)],
        generator=torch.Generator().manual_seed(seed),
    )
    test_subset = test_ds

    val_batch_size = val_batch_size or batch_size
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_subset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, test_loader
