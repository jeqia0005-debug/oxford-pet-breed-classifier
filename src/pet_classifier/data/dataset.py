"""Assemble train / val / test datasets and DataLoaders.

Because the train and validation sets are both carved from the same raw
``trainval`` dataset but need *different* transforms (augmented vs.
deterministic), we wrap subsets of it with :class:`SubsetWithTransform`, which
applies its own transform to the underlying (un-transformed) PIL images.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from pet_classifier.config import DATA_DIR, IMAGE_SIZE, NUM_CLASSES, SEED
from pet_classifier.data.download import get_labels, load_raw_datasets
from pet_classifier.data.splits import stratified_train_val_split
from pet_classifier.data.transforms import build_eval_transform, build_train_transform
from pet_classifier.utils.reproducibility import worker_init_fn


class SubsetWithTransform(Dataset):
    """A subset of ``base`` (selected by ``indices``) with its own transform."""

    def __init__(self, base: Dataset, indices, transform=None):
        self.base = base
        self.indices = np.asarray(indices, dtype=np.int64)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, i: int):
        image, label = self.base[int(self.indices[i])]
        if self.transform is not None:
            image = self.transform(image)
        return image, label


@dataclass
class DataBundle:
    """Everything downstream training / evaluation code needs from the data."""

    train: DataLoader
    val: DataLoader
    test: DataLoader
    class_names: list[str]
    # Raw integer labels for each split (handy for class-distribution plots).
    train_labels: np.ndarray = field(repr=False)
    val_labels: np.ndarray = field(repr=False)
    test_labels: np.ndarray = field(repr=False)


def build_data(
    root: str | Path = DATA_DIR,
    img_size: int = IMAGE_SIZE,
    val_fraction: float = 0.2,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = SEED,
    download: bool = True,
) -> DataBundle:
    """Download the data, split it, and return ready-to-use DataLoaders.

    The validation set uses the deterministic *eval* transform (no
    augmentation) even though it comes from ``trainval``.
    """
    trainval_raw, test_raw = load_raw_datasets(root=root, download=download)

    trainval_labels = get_labels(trainval_raw)
    train_idx, val_idx = stratified_train_val_split(
        trainval_labels, val_fraction=val_fraction, seed=seed
    )

    train_tf = build_train_transform(img_size)
    eval_tf = build_eval_transform(img_size)

    train_ds = SubsetWithTransform(trainval_raw, train_idx, transform=train_tf)
    val_ds = SubsetWithTransform(trainval_raw, val_idx, transform=eval_tf)
    test_ds = SubsetWithTransform(test_raw, np.arange(len(test_raw)), transform=eval_tf)

    generator = torch.Generator().manual_seed(seed)
    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=worker_init_fn,
    )
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **common)
    val_loader = DataLoader(val_ds, shuffle=False, **common)
    test_loader = DataLoader(test_ds, shuffle=False, **common)

    # torchvision exposes the authoritative label->name ordering here.
    class_names = list(getattr(trainval_raw, "classes", []))
    assert len(class_names) == NUM_CLASSES, (
        f"Expected {NUM_CLASSES} classes, dataset reported {len(class_names)}."
    )

    return DataBundle(
        train=train_loader,
        val=val_loader,
        test=test_loader,
        class_names=class_names,
        train_labels=trainval_labels[train_idx],
        val_labels=trainval_labels[val_idx],
        test_labels=get_labels(test_raw),
    )
