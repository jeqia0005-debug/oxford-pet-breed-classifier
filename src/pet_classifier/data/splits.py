"""Stratified train / validation splitting.

The Oxford-IIIT Pet dataset provides an official ``trainval`` set and a held-out
``test`` set. We carve a validation set out of ``trainval`` using a *stratified*
split so every breed keeps the same train:val ratio — important because each
class only has ~100 images in ``trainval``.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from pet_classifier.config import SEED


def stratified_train_val_split(
    labels: np.ndarray,
    val_fraction: float = 0.2,
    seed: int = SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Split sample indices into train/val, preserving class proportions.

    Parameters
    ----------
    labels:
        1-D array of integer class labels, one per sample.
    val_fraction:
        Fraction of samples to place in the validation set (0 < f < 1).
    seed:
        Random seed for a reproducible split.

    Returns
    -------
    (train_idx, val_idx):
        Sorted arrays of integer indices into the original dataset.
    """
    if not 0.0 < val_fraction < 1.0:
        raise ValueError(f"val_fraction must be in (0, 1), got {val_fraction}")

    labels = np.asarray(labels)
    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=val_fraction, random_state=seed
    )
    # StratifiedShuffleSplit needs an X of matching length; contents are unused.
    train_idx, val_idx = next(splitter.split(np.zeros(len(labels)), labels))
    return np.sort(train_idx), np.sort(val_idx)


def class_counts(labels: np.ndarray, num_classes: int) -> np.ndarray:
    """Return a length-``num_classes`` array of per-class sample counts."""
    return np.bincount(np.asarray(labels), minlength=num_classes)
