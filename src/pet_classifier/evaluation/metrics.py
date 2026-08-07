"""Full test-set evaluation for a trained checkpoint (Member 3).

Produces everything the final report needs: top-1 / macro-F1 / top-3
accuracy, a full classification report, a confusion matrix, and the list
of breed pairs the model confuses most often.

The test set is only ever touched here, at final-evaluation time — never
during model selection (see README).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm


@dataclass
class EvaluationResult:
    y_true: np.ndarray
    y_pred: np.ndarray
    y_probs: np.ndarray  # (N, num_classes) softmax probabilities
    class_names: list[str]

    top1_acc: float = field(init=False)
    top3_acc: float = field(init=False)
    macro_f1: float = field(init=False)
    confusion: np.ndarray = field(init=False)
    report: dict = field(init=False)

    def __post_init__(self) -> None:
        self.top1_acc = float((self.y_true == self.y_pred).mean())
        self.top3_acc = _topk_accuracy(self.y_true, self.y_probs, k=3)
        self.macro_f1 = float(
            f1_score(self.y_true, self.y_pred, average="macro", zero_division=0)
        )
        self.confusion = confusion_matrix(
            self.y_true, self.y_pred, labels=range(len(self.class_names))
        )
        self.report = classification_report(
            self.y_true,
            self.y_pred,
            labels=range(len(self.class_names)),
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )

    def most_confused_pairs(self, top_n: int = 10) -> list[dict]:
        """Return the ``top_n`` (true, predicted) breed pairs with the most
        misclassifications, off-diagonal only, sorted descending."""
        cm = self.confusion.copy()
        np.fill_diagonal(cm, 0)

        flat_idx = np.argsort(cm, axis=None)[::-1]
        pairs = []
        for idx in flat_idx[: top_n * 2]:  # oversample, filter zeros below
            i, j = np.unravel_index(idx, cm.shape)
            count = int(cm[i, j])
            if count == 0:
                break
            pairs.append(
                {
                    "true_breed": self.class_names[i],
                    "predicted_breed": self.class_names[j],
                    "count": count,
                }
            )
            if len(pairs) == top_n:
                break
        return pairs

    def summary(self) -> dict:
        return {
            "top1_accuracy": self.top1_acc,
            "top3_accuracy": self.top3_acc,
            "macro_f1": self.macro_f1,
            "most_confused_pairs": self.most_confused_pairs(),
        }


def _topk_accuracy(y_true: np.ndarray, y_probs: np.ndarray, k: int) -> float:
    topk_preds = np.argsort(y_probs, axis=1)[:, -k:]
    hits = np.any(topk_preds == y_true[:, None], axis=1)
    return float(hits.mean())


@torch.no_grad()
def run_full_evaluation(
    model: torch.nn.Module,
    loader: DataLoader,
    class_names: list[str],
    device: torch.device | str = "cpu",
) -> EvaluationResult:
    """Run inference over an entire DataLoader and collect predictions."""
    model.eval()
    all_true, all_pred, all_probs = [], [], []

    for images, targets in tqdm(loader, desc="test set inference", leave=False):
        images = images.to(device)
        logits = model(images)
        probs = torch.softmax(logits, dim=1)

        all_true.append(targets.numpy())
        all_pred.append(probs.argmax(dim=1).cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    return EvaluationResult(
        y_true=np.concatenate(all_true),
        y_pred=np.concatenate(all_pred),
        y_probs=np.concatenate(all_probs),
        class_names=class_names,
    )
