"""Lightweight metrics used during training (full evaluation lives in the
``evaluation`` package, Member 3)."""

from __future__ import annotations

import torch


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    """Top-1 accuracy for a batch."""
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


@torch.no_grad()
def topk_accuracy(logits: torch.Tensor, targets: torch.Tensor, k: int = 3) -> float:
    """Top-k accuracy: fraction of samples whose true label is in the top k."""
    k = min(k, logits.size(1))
    topk = logits.topk(k, dim=1).indices
    correct = (topk == targets.unsqueeze(1)).any(dim=1)
    return correct.float().mean().item()
