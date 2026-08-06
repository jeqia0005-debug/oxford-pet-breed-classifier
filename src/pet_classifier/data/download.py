"""Download and load the Oxford-IIIT Pet dataset via torchvision.

The dataset ships with an official train/val ("trainval") split and a "test"
split. We load the raw ``trainval`` and ``test`` sets here (with no image
transform); splitting ``trainval`` into train/validation and attaching
augmentation happens in :mod:`pet_classifier.data.dataset`.

Reference
---------
O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar, "Cats and Dogs",
CVPR 2012. https://www.robots.ox.ac.uk/~vgg/data/pets/
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from torchvision.datasets import OxfordIIITPet

from pet_classifier.config import DATA_DIR


def load_raw_datasets(
    root: str | Path = DATA_DIR,
    download: bool = True,
) -> tuple[OxfordIIITPet, OxfordIIITPet]:
    """Return the raw ``(trainval, test)`` Oxford-IIIT Pet datasets.

    Parameters
    ----------
    root:
        Directory to download into / read from. Created if missing.
    download:
        If ``True``, download the archive when not already present. The two
        splits share the same underlying files, so only the first call
        actually downloads (~800 MB).
    """
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)

    trainval = OxfordIIITPet(
        root=str(root),
        split="trainval",
        target_types="category",
        download=download,
    )
    test = OxfordIIITPet(
        root=str(root),
        split="test",
        target_types="category",
        # Files already fetched by the trainval call above.
        download=False,
    )
    return trainval, test


def get_labels(dataset: OxfordIIITPet) -> np.ndarray:
    """Return the integer breed label for every sample, without decoding images.

    torchvision stores labels on the private ``_labels`` attribute; we prefer
    that fast path and fall back to iterating targets if the internals change.
    """
    labels = getattr(dataset, "_labels", None)
    if labels is None:  # pragma: no cover - defensive fallback
        labels = [dataset[i][1] for i in range(len(dataset))]
    return np.asarray(labels, dtype=np.int64)
