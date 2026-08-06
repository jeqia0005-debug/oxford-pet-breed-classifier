# Oxford-IIIT Pet Breed Classifier 🐱🐶

Fine-grained classification of cat and dog images into **37 breeds** using the
[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

This is a course final project. We build a **custom CNN from scratch** as a
baseline, then apply **transfer learning with MobileNetV2** (frozen feature
extractor vs. partial fine-tuning), compare the three models, and ship a
**web demo** with Grad-CAM explanations.

## Dataset

- **Source:** Oxford-IIIT Pet Dataset — https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Citation:** O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar.
  *Cats and Dogs.* IEEE Conference on Computer Vision and Pattern Recognition, 2012.
- ~7,349 images, 37 breeds (~200 images/breed), with variation in pose,
  lighting, scale, and background.
- Downloaded automatically on first run via `torchvision.datasets.OxfordIIITPet`
  into `data/` (git-ignored).

## Quickstart (uv)

```bash
# 1. Install uv if you don't have it: https://docs.astral.sh/uv/
# 2. Create the environment and install dependencies
uv sync

# 3. Explore the data — class distribution + sample grid (downloads on first run)
uv run python scripts/explore_data.py

# 4. Train the custom CNN baseline
uv run python scripts/train_custom_cnn.py --config configs/custom_cnn.yaml

# 5. (Member 3) Launch the web demo
uv run streamlit run src/pet_classifier/app/streamlit_app.py
```

The first data command downloads ~800 MB into `data/` and may take a few
minutes depending on your connection.

## Project structure

```
src/pet_classifier/
├── config.py            # Shared constants: 37 classes, normalization, paths, seed
├── data/                # Member 1 — download, splits, augmentation
│   ├── download.py      #   dataset download + integrity check
│   ├── dataset.py       #   OxfordPet Dataset wrapper + DataLoaders
│   ├── splits.py        #   stratified train / val split
│   └── transforms.py    #   resize / normalize / augmentation
├── models/
│   ├── custom_cnn.py    # Member 1 — from-scratch CNN with BatchNorm + Dropout
│   └── transfer.py      # Member 2 — MobileNetV2 (frozen / fine-tuned)
├── training/            # Member 1 — reusable loop, early stopping, checkpoints
│   ├── trainer.py
│   ├── early_stopping.py
│   └── metrics.py
├── evaluation/          # Member 3 — metrics, confusion matrix, Grad-CAM
├── app/                 # Member 3 — web demo
└── utils/               # seeding, visualization helpers

scripts/                 # runnable entry points
configs/                 # experiment configs (YAML)
tests/                   # lightweight unit tests
reports/                 # figures + written report
```

## Team & branches

Work is split across three members; each develops on feature branches and
merges to `main` via Pull Request.

| Member | Responsibility | Branches |
|--------|----------------|----------|
| **1** | Data pipeline + custom CNN baseline | `feature/data-pipeline`, `feature/custom-cnn` |
| **2** | Transfer learning + fine-tuning (MobileNetV2) | `feature/transfer-learning`, `feature/fine-tuning` |
| **3** | Evaluation + Grad-CAM + web app | `feature/evaluation`, `feature/gradcam`, `feature/web-app` |

## Models

1. **Custom CNN (baseline)** — convolutional blocks built from scratch with
   Batch Normalization, Dropout, and Early Stopping.
2. **MobileNetV2 frozen** — pretrained backbone as a fixed feature extractor,
   only the 37-way classifier head is trained.
3. **MobileNetV2 fine-tuned** — upper backbone layers unfrozen and trained at a
   low learning rate.

Evaluation: accuracy, macro F1, top-3 accuracy, confusion matrix, and
training/validation curves.

## Reproducibility

All entry points seed Python, NumPy and PyTorch (see
`utils/reproducibility.py`, `SEED = 42` in `config.py`).
