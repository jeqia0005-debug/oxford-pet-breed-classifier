"""Explore the Oxford-IIIT Pet dataset.

Downloads the data (first run only), reports split sizes and per-class
counts, and saves two figures to ``reports/figures/``:

* ``class_distribution.png`` — images per breed in the trainval split
* ``sample_grid.png``        — a grid of augmented training samples

Usage
-----
    uv run python scripts/explore_data.py
    uv run python scripts/explore_data.py --img-size 160 --batch-size 16
"""

from __future__ import annotations

import argparse

import numpy as np

from pet_classifier.config import DATA_DIR, IMAGE_SIZE, NUM_CLASSES
from pet_classifier.data.dataset import build_data
from pet_classifier.data.splits import class_counts
from pet_classifier.utils.reproducibility import seed_everything
from pet_classifier.utils.viz import (
    DEFAULT_DISTRIBUTION_FIG,
    DEFAULT_SAMPLES_FIG,
    plot_class_distribution,
    show_sample_grid,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(DATA_DIR))
    p.add_argument("--img-size", type=int, default=IMAGE_SIZE)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-download", action="store_true", help="assume data is present")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    seed_everything()

    data = build_data(
        root=args.data_dir,
        img_size=args.img_size,
        val_fraction=args.val_fraction,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=not args.no_download,
    )

    print(f"\nLoaded Oxford-IIIT Pet — {NUM_CLASSES} breeds")
    print(f"  train: {len(data.train_labels):>5} images")
    print(f"  val:   {len(data.val_labels):>5} images")
    print(f"  test:  {len(data.test_labels):>5} images")

    counts = class_counts(data.train_labels, NUM_CLASSES)
    print("\nPer-class counts in the training split:")
    print(f"  min {counts.min()}  max {counts.max()}  mean {counts.mean():.1f}")
    imbalance = counts.max() / max(counts.min(), 1)
    print(f"  max/min ratio: {imbalance:.2f} (≈1.0 means well balanced)")

    dist_path = plot_class_distribution(
        data.train_labels,
        data.class_names,
        title="Oxford-IIIT Pet — training split",
        save_path=DEFAULT_DISTRIBUTION_FIG,
    )
    print(f"\nSaved class distribution figure -> {dist_path}")

    images, labels = next(iter(data.train))
    grid_path = show_sample_grid(
        images, labels, data.class_names, n=12, save_path=DEFAULT_SAMPLES_FIG
    )
    print(f"Saved augmented sample grid   -> {grid_path}")


if __name__ == "__main__":
    main()
