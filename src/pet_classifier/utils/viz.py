"""Visualization helpers: class distribution and sample grids.

Uses a non-interactive Matplotlib backend so the scripts run headless (e.g. in
CI or on a remote box) and save figures to ``reports/figures/``.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from pet_classifier.config import FIGURE_DIR, NUM_CLASSES  # noqa: E402
from pet_classifier.data.transforms import denormalize  # noqa: E402


def _ensure_dir(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def plot_class_distribution(
    labels: np.ndarray,
    class_names: list[str],
    title: str = "Class distribution",
    save_path: Path | str | None = None,
) -> Path | None:
    """Horizontal bar chart of samples per breed."""
    counts = np.bincount(np.asarray(labels), minlength=NUM_CLASSES)
    order = np.argsort(counts)
    fig, ax = plt.subplots(figsize=(8, 10))
    ax.barh(np.array(class_names)[order], counts[order], color="#4C72B0")
    ax.set_xlabel("Number of images")
    ax.set_title(f"{title}  (n={counts.sum()}, {len(class_names)} breeds)")
    ax.margins(y=0.01)
    fig.tight_layout()

    if save_path is None:
        return None
    path = _ensure_dir(Path(save_path))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def show_sample_grid(
    images: torch.Tensor,
    labels,
    class_names: list[str],
    n: int = 12,
    ncols: int = 4,
    save_path: Path | str | None = None,
) -> Path | None:
    """Save a grid of (denormalized) sample images with their breed labels."""
    n = min(n, len(images))
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3 * nrows))
    axes = np.atleast_1d(axes).ravel()

    imgs = denormalize(images[:n]).cpu()
    for i in range(n):
        axes[i].imshow(imgs[i].permute(1, 2, 0).numpy())
        axes[i].set_title(class_names[int(labels[i])], fontsize=9)
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


def plot_training_curves(
    history: dict,
    save_path: Path | str | None = None,
) -> Path | None:
    """Plot loss and accuracy curves from a training ``history`` dict.

    Expects keys: ``train_loss``, ``val_loss``, ``train_acc``, ``val_acc``.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 5))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title="Loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title="Accuracy")
    ax_acc.legend()

    fig.tight_layout()
    if save_path is None:
        return None
    path = _ensure_dir(Path(save_path))
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


# Default output locations, kept next to the figures dir for convenience.
DEFAULT_DISTRIBUTION_FIG = FIGURE_DIR / "class_distribution.png"
DEFAULT_SAMPLES_FIG = FIGURE_DIR / "sample_grid.png"
