"""Evaluation-stage visualizations: confusion matrix and error galleries.

Mirrors the conventions in ``pet_classifier.utils.viz`` (non-interactive
Matplotlib backend, figures saved under ``reports/figures/``).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pet_classifier.data.transforms import denormalize  # noqa: E402


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_confusion_matrix(
    confusion: np.ndarray,
    class_names: list[str],
    normalize: bool = True,
    save_path: Path | str | None = None,
) -> Path | None:
    """Heatmap of the 37x37 confusion matrix.

    ``normalize=True`` divides each row by its true-class count, so the
    color scale reflects per-class recall rather than raw counts (raw
    counts are skewed by the ~50/class test split).
    """
    cm = confusion.astype(np.float64)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm = np.divide(cm, row_sums, out=np.zeros_like(cm), where=row_sums != 0)

    fig, ax = plt.subplots(figsize=(14, 12))
    im = ax.imshow(cm, cmap="viridis", vmin=0, vmax=1 if normalize else cm.max())
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=90, fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)
    ax.set_xlabel("Predicted breed")
    ax.set_ylabel("True breed")
    ax.set_title("Confusion matrix" + (" (row-normalized)" if normalize else ""))
    fig.tight_layout()

    if save_path is None:
        return None
    path = _ensure_dir(Path(save_path))
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_prediction_gallery(
    images: torch.Tensor,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    confidences: np.ndarray,
    class_names: list[str],
    n: int = 12,
    ncols: int = 4,
    save_path: Path | str | None = None,
) -> Path | None:
    """Grid of sample predictions, green title if correct, red if wrong.

    Useful for both the "correct vs. incorrect case study" section of the
    report and for sanity-checking Grad-CAM targets before running it on a
    full batch.
    """
    n = min(n, len(images))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    imgs = denormalize(images[:n]).cpu()
    for i in range(n):
        correct = y_true[i] == y_pred[i]
        axes[i].imshow(imgs[i].permute(1, 2, 0).numpy())
        axes[i].set_title(
            f"true: {class_names[y_true[i]]}\n"
            f"pred: {class_names[y_pred[i]]} ({confidences[i]:.0%})",
            fontsize=8,
            color="#2ca02c" if correct else "#d62728",
        )
        axes[i].axis("off")
    for j in range(n, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()

    if save_path is None:
        return None
    path = _ensure_dir(Path(save_path))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
