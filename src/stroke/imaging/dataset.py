"""Datasets and transforms for the CT/MRI imaging branch.

Supports two sources:

* **ImageFolder** layout (the Kaggle Brain Stroke CT dataset), e.g.::

      data/imaging/ct/
          Normal/   *.jpg
          Stroke/   *.jpg

  Class subfolders are auto-discovered; "stroke"/"haemorrhage"/"ischemic"
  folder names map to the positive class.

* **SyntheticCTDataset** -- procedurally generated grayscale "scans" used
  by the ``--quick`` smoke test and unit tests so the pipeline runs with
  no download.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, random_split

try:
    from torchvision import transforms
    from torchvision.datasets import ImageFolder

    _HAS_TV = True
except ImportError:  # pragma: no cover
    _HAS_TV = False

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
POSITIVE_KEYWORDS = ("stroke", "haemorrhage", "hemorrhage", "ischemic", "bleed", "abnormal")


def build_transforms(img_size: int = 224, train: bool = True):
    if not _HAS_TV:
        raise ImportError("torchvision required for image transforms")
    if train:
        return transforms.Compose(
            [
                transforms.Grayscale(num_output_channels=3),
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(brightness=0.1, contrast=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class SyntheticCTDataset(Dataset):
    """Procedural grayscale 'scans'; positives carry a bright blob (lesion)."""

    def __init__(self, n: int = 200, img_size: int = 224, seed: int = 0):
        self.img_size = img_size
        rng = np.random.default_rng(seed)
        self.labels = (rng.uniform(size=n) < 0.5).astype(np.int64)
        self.seeds = rng.integers(0, 1_000_000, size=n)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rng = np.random.default_rng(int(self.seeds[idx]))
        s = self.img_size
        # Skull-like ring + brain tissue noise.
        yy, xx = np.mgrid[0:s, 0:s]
        cx = cy = s / 2
        r = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        img = np.clip(0.5 - (r / (s / 2)) * 0.3, 0, 1)
        img += rng.normal(0, 0.05, (s, s))
        label = int(self.labels[idx])
        if label == 1:  # inject a hyperdense lesion
            lx, ly = rng.integers(s // 4, 3 * s // 4, size=2)
            blob = np.exp(-(((xx - lx) ** 2 + (yy - ly) ** 2) / (2 * (s * 0.05) ** 2)))
            img += blob * 0.6
        img = np.clip(img, 0, 1).astype(np.float32)
        t = torch.from_numpy(img).unsqueeze(0).repeat(3, 1, 1)
        mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(3, 1, 1)
        return (t - mean) / std, label


def _positive_index(class_names: list[str]) -> int:
    for i, name in enumerate(class_names):
        if any(k in name.lower() for k in POSITIVE_KEYWORDS):
            return i
    return len(class_names) - 1  # fall back to last class


def build_dataloaders(
    data_dir: str | Path | None,
    img_size: int = 224,
    batch_size: int = 32,
    val_split: float = 0.2,
    synthetic: bool = False,
    synthetic_n: int = 200,
    num_workers: int = 0,
    seed: int = 42,
):
    """Return (train_loader, val_loader, n_classes, pos_index)."""
    if synthetic or data_dir is None or not Path(data_dir).exists():
        full = SyntheticCTDataset(n=synthetic_n, img_size=img_size, seed=seed)
        pos_index = 1
    else:
        if not _HAS_TV:
            raise ImportError("torchvision required for ImageFolder datasets")
        full = ImageFolder(str(data_dir), transform=build_transforms(img_size, train=True))
        pos_index = _positive_index(full.classes)

    n_val = int(len(full) * val_split)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=g)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_loader, val_loader, 2, pos_index
