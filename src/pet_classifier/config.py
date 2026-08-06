"""Project-wide configuration shared by all group members.

Keep constants that every part of the pipeline needs (class count, image
normalization, paths, random seed) in one place so the custom CNN, the
transfer-learning model, and the evaluation code all stay consistent.
"""

from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Repo root = three parents up from this file: src/pet_classifier/config.py
ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CHECKPOINT_DIR = ROOT_DIR / "checkpoints"
FIGURE_DIR = ROOT_DIR / "reports" / "figures"

# --- Task ------------------------------------------------------------------
NUM_CLASSES = 37

# --- Image preprocessing ---------------------------------------------------
# Default input size. MobileNetV2 (Member 2) expects 224; the custom CNN can
# override this via its own config for faster CPU training.
IMAGE_SIZE = 224

# ImageNet statistics — used for both the from-scratch CNN and the pretrained
# MobileNetV2 so that normalization is identical across all three models.
NORM_MEAN = (0.485, 0.456, 0.406)
NORM_STD = (0.229, 0.224, 0.225)

# --- Reproducibility -------------------------------------------------------
SEED = 42

# --- Class names -----------------------------------------------------------
# Canonical 37 breeds of the Oxford-IIIT Pet dataset (12 cat + 25 dog breeds).
# The authoritative label ordering at runtime comes from the torchvision
# dataset object (`dataset.classes`); this list is provided for reference,
# reporting, and offline label lookups.
CAT_BREEDS = (
    "Abyssinian",
    "Bengal",
    "Birman",
    "Bombay",
    "British Shorthair",
    "Egyptian Mau",
    "Maine Coon",
    "Persian",
    "Ragdoll",
    "Russian Blue",
    "Siamese",
    "Sphynx",
)

DOG_BREEDS = (
    "American Bulldog",
    "American Pit Bull Terrier",
    "Basset Hound",
    "Beagle",
    "Boxer",
    "Chihuahua",
    "English Cocker Spaniel",
    "English Setter",
    "German Shorthaired",
    "Great Pyrenees",
    "Havanese",
    "Japanese Chin",
    "Keeshond",
    "Leonberger",
    "Miniature Pinscher",
    "Newfoundland",
    "Pomeranian",
    "Pug",
    "Saint Bernard",
    "Samoyed",
    "Scottish Terrier",
    "Shiba Inu",
    "Staffordshire Bull Terrier",
    "Wheaten Terrier",
    "Yorkshire Terrier",
)

# Alphabetical order matches torchvision's OxfordIIITPet class ordering.
CLASS_NAMES = tuple(sorted(CAT_BREEDS + DOG_BREEDS))
assert len(CLASS_NAMES) == NUM_CLASSES, "Expected 37 breed classes."
