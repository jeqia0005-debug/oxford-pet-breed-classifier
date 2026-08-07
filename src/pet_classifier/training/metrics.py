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


@torch.no_grad()
def evaluate_classification(model, loader, device, topk: int = 3) -> dict:
    """Run ``model`` over ``loader`` and return headline classification metrics.

    Returns a dict with ``accuracy``, ``macro_f1``, ``top{k}_accuracy`` and
    ``num_samples``. Kept model-agnostic so the custom CNN and the transfer
    models can all be scored the same way for a fair comparison.
    """
    from sklearn.metrics import accuracy_score, f1_score

    model.eval()
    all_preds, all_targets = [], []
    topk_hits, n = 0, 0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = model(images).cpu()
        all_preds.append(logits.argmax(dim=1))
        all_targets.append(targets)
        k = min(topk, logits.size(1))
        topk_idx = logits.topk(k, dim=1).indices
        topk_hits += (topk_idx == targets.unsqueeze(1)).any(dim=1).sum().item()
        n += targets.size(0)

    preds = torch.cat(all_preds).numpy()
    tgts = torch.cat(all_targets).numpy()
    return {
        "accuracy": float(accuracy_score(tgts, preds)),
        "macro_f1": float(f1_score(tgts, preds, average="macro")),
        f"top{topk}_accuracy": topk_hits / n,
        "num_samples": int(n),
    }
