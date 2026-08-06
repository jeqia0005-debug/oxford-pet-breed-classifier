"""Seed all RNGs so experiments are reproducible across group members."""

from __future__ import annotations

import os
import random

import numpy as np
import torch

from pet_classifier.config import SEED


def seed_everything(seed: int = SEED) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA) for reproducible runs."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker seeding so augmentation is reproducible per worker."""
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)
