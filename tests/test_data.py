"""Unit tests for the data pipeline that run offline (no dataset download)."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from pet_classifier.config import NORM_MEAN, NORM_STD
from pet_classifier.data.dataset import SubsetWithTransform
from pet_classifier.data.splits import class_counts, stratified_train_val_split
from pet_classifier.data.transforms import (
    build_eval_transform,
    build_train_transform,
    denormalize,
)


def _fake_labels(num_classes=37, per_class=100):
    return np.repeat(np.arange(num_classes), per_class)


def test_stratified_split_is_disjoint_and_complete():
    labels = _fake_labels()
    train_idx, val_idx = stratified_train_val_split(labels, val_fraction=0.2, seed=0)

    assert len(set(train_idx) & set(val_idx)) == 0
    assert len(train_idx) + len(val_idx) == len(labels)
    assert sorted(np.concatenate([train_idx, val_idx])) == list(range(len(labels)))


def test_stratified_split_preserves_class_proportions():
    labels = _fake_labels(num_classes=37, per_class=100)
    train_idx, val_idx = stratified_train_val_split(labels, val_fraction=0.2, seed=0)

    val_counts = class_counts(labels[val_idx], 37)
    # 20% of 100 per class -> exactly 20 each, perfectly balanced.
    assert val_counts.min() == 20 and val_counts.max() == 20


def test_stratified_split_is_reproducible():
    labels = _fake_labels()
    a = stratified_train_val_split(labels, seed=42)
    b = stratified_train_val_split(labels, seed=42)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_split_rejects_bad_fraction():
    labels = _fake_labels()
    for bad in (0.0, 1.0, -0.1, 1.5):
        try:
            stratified_train_val_split(labels, val_fraction=bad)
        except ValueError:
            continue
        raise AssertionError(f"val_fraction={bad} should have raised")


def test_train_transform_output_shape_and_type():
    img = Image.fromarray(np.random.randint(0, 255, (300, 250, 3), dtype=np.uint8))
    out = build_train_transform(img_size=128)(img)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (3, 128, 128)


def test_eval_transform_is_deterministic():
    img = Image.fromarray(np.random.randint(0, 255, (300, 250, 3), dtype=np.uint8))
    tf = build_eval_transform(img_size=160)
    a, b = tf(img), tf(img)
    assert a.shape == (3, 160, 160)
    assert torch.allclose(a, b)


def test_denormalize_inverts_normalization():
    x01 = torch.rand(3, 16, 16)
    mean = torch.tensor(NORM_MEAN).view(-1, 1, 1)
    std = torch.tensor(NORM_STD).view(-1, 1, 1)
    normalized = (x01 - mean) / std
    assert torch.allclose(denormalize(normalized), x01, atol=1e-5)


class _TinyBase(Dataset):
    """Minimal stand-in for a torchvision dataset returning (PIL, label)."""

    def __init__(self, n=10):
        self.imgs = [
            Image.fromarray(np.full((32, 32, 3), i, dtype=np.uint8)) for i in range(n)
        ]
        self.labels = list(range(n))

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return self.imgs[i], self.labels[i]


def test_subset_with_transform_indexes_and_transforms():
    base = _TinyBase(10)
    subset = SubsetWithTransform(base, indices=[2, 5, 7], transform=build_eval_transform(24))
    assert len(subset) == 3
    img, label = subset[1]
    assert img.shape == (3, 24, 24)
    assert label == 5  # index 5 in the base dataset
