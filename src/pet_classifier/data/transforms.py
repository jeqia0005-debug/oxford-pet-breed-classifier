"""Image preprocessing and augmentation.

Two pipelines:

* **train** — random augmentation (resized crop, flips, colour jitter,
  rotation, random erasing) to combat overfitting on a small (~200
  images/breed) dataset.
* **eval**  — deterministic resize + centre crop, used for validation and test
  so metrics are stable.

Both end with ``ToTensor`` + ImageNet ``Normalize`` (shared constants in
:mod:`pet_classifier.config`) so the from-scratch CNN and the pretrained
MobileNetV2 see identically-normalized inputs.
"""

from __future__ import annotations

import torch
from torchvision import transforms

from pet_classifier.config import IMAGE_SIZE, NORM_MEAN, NORM_STD


def build_train_transform(img_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Augmentation pipeline for training images."""
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0), ratio=(0.8, 1.25)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.RandomRotation(degrees=15),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
            transforms.RandomErasing(p=0.25),
        ]
    )


def build_eval_transform(img_size: int = IMAGE_SIZE) -> transforms.Compose:
    """Deterministic pipeline for validation / test images."""
    resize = int(round(img_size * 1.14))  # resize slightly larger, then centre-crop
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ]
    )


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    """Invert ImageNet normalization for visualization; returns values in [0, 1].

    Accepts a single image ``(C, H, W)`` or a batch ``(N, C, H, W)``.
    """
    mean = torch.tensor(NORM_MEAN, device=tensor.device).view(-1, 1, 1)
    std = torch.tensor(NORM_STD, device=tensor.device).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0.0, 1.0)
