# Oxford-IIIT Pet Breed Classifier 🐱🐶

Fine-grained classification of cat and dog images into **37 breeds** using the
[Oxford-IIIT Pet Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).

This is a course final project implemented in **Python and PyTorch**. We build a
**custom CNN from scratch** as a baseline, then apply **transfer learning with
MobileNetV2** using both frozen feature extraction and partial fine-tuning.
Multiple fine-tuning strategies are evaluated to study whether updating
pretrained visual features improves generalization on a relatively small
fine-grained image dataset. The final system will also include a **web demo**
with Grad-CAM explanations.

## Dataset

- **Source:** Oxford-IIIT Pet Dataset — https://www.robots.ox.ac.uk/~vgg/data/pets/
- **Citation:** O. M. Parkhi, A. Vedaldi, A. Zisserman, C. V. Jawahar.
  *Cats and Dogs.* IEEE Conference on Computer Vision and Pattern Recognition, 2012.
- Approximately **7,349 images** across **37 cat and dog breeds**
  (~200 images per breed).
- Images contain substantial variation in pose, lighting, scale, and background.
- The dataset is downloaded automatically through
  `torchvision.datasets.OxfordIIITPet` and stored in `data/`, which is
  git-ignored.
- The official training split is further divided into stratified training and
  validation sets. The official test set is reserved for final evaluation and
  is not used during model selection.

## Quickstart (uv)

```bash
# 1. Install uv if needed:
# https://docs.astral.sh/uv/

# 2. Create the environment and install dependencies
uv sync

# 3. Explore the dataset
uv run python scripts/explore_data.py

# 4. Train the custom CNN baseline
uv run python scripts/train_custom_cnn.py \
  --config configs/custom_cnn.yaml

# 5. Train MobileNetV2 as a frozen feature extractor
uv run python scripts/train_mobilenet_frozen.py \
  --config configs/mobilenet_frozen.yaml

# 6. Run the first partial fine-tuning experiment
uv run python scripts/train_mobilenet_finetune.py \
  --config configs/mobilenet_finetune.yaml

# 7. Run the more conservative fine-tuning experiments
uv run python scripts/train_mobilenet_finetune.py \
  --config configs/mobilenet_finetune_v2.yaml

uv run python scripts/train_mobilenet_finetune.py \
  --config configs/mobilenet_finetune_v3.yaml

# 8. Run the test suite
uv run pytest -q

# 9. (Member 3) Launch the web demo
uv run streamlit run src/pet_classifier/app/streamlit_app.py
```

The first data command downloads approximately 800 MB into `data/` and may take
a few minutes depending on the network connection.

## Project Structure

```text
src/pet_classifier/
├── config.py            # Shared constants, paths, normalization, seed
├── data/                # Member 1 — dataset preparation
│   ├── download.py      #   dataset download + integrity checks
│   ├── dataset.py       #   Oxford Pet dataset + DataLoaders
│   ├── splits.py        #   stratified train / validation split
│   └── transforms.py    #   resize, normalization, augmentation
├── models/
│   ├── custom_cnn.py    # Member 1 — CNN built from scratch
│   └── transfer.py      # Member 2 — MobileNetV2 transfer learning
├── training/            # Reusable training infrastructure
│   ├── trainer.py       #   train / validation loop + checkpointing
│   ├── early_stopping.py
│   └── metrics.py
├── evaluation/          # Member 3 — metrics, confusion matrix, Grad-CAM
├── app/                 # Member 3 — web demo
└── utils/               # Reproducibility and visualization helpers

scripts/                 # Runnable experiment entry points
configs/                 # YAML experiment configurations
tests/                   # Lightweight unit tests
reports/                 # Experiment results, figures, and summaries
checkpoints/             # Local model checkpoints (git-ignored)
```

## Team & Branches

Development is split across three members. Each member works on dedicated
feature branches and merges changes into `main` through Pull Requests.

| Member | Responsibility | Branches |
|---|---|---|
| **1** | Data pipeline + custom CNN baseline | `feature/data-pipeline`, `feature/custom-cnn` |
| **2** | MobileNetV2 transfer learning + fine-tuning experiments | `feature/transfer-learning`, `feature/fine-tuning` |
| **3** | Final evaluation + Grad-CAM + web app | `feature/evaluation`, `feature/gradcam`, `feature/web-app` |

## Models

### 1. Custom CNN

A convolutional neural network trained from scratch using only the Oxford-IIIT
Pet training data.

The model includes:

- convolutional layers
- Batch Normalization
- ReLU activations
- pooling
- Dropout
- Global Average Pooling
- a 37-class output layer
- Early Stopping and best-checkpoint selection

This model serves as the from-scratch baseline.

**Results** (160 × 160, 60 epochs, best epoch 56):

| Split | Accuracy | Top-3 Accuracy | Macro F1 |
|---|---:|---:|---:|
| Validation | 38.32% | 64.40% | — |
| Test | 33.77% | 60.62% | 31.38% |

Training and validation curves are healthy (no overfitting; validation accuracy
was still rising at epoch 60). Full details in
`reports/member1_custom_cnn_summary.md` and `reports/custom_cnn_results.json`.

### 2. Frozen MobileNetV2

An ImageNet-pretrained MobileNetV2 is used as a fixed feature extractor.

The convolutional backbone is frozen and only a newly added 37-class
classification head is trained. Frozen feature blocks remain in evaluation
mode so that their BatchNorm running statistics are not modified during
training.

### 3. Partially Fine-Tuned MobileNetV2

The best frozen MobileNetV2 checkpoint is used as the starting point for
fine-tuning.

Several fine-tuning strategies were evaluated by changing:

- the number of unfrozen MobileNetV2 feature blocks
- the learning rate applied to the pretrained backbone
- the learning rate applied to the classification head

Smaller learning rates are used for pretrained layers to avoid excessively
changing useful ImageNet representations.

## MobileNetV2 Experiments

The frozen model was used as the transfer-learning baseline. Three partial
fine-tuning strategies were then evaluated.

| Model | Unfrozen Blocks | Validation Accuracy | Top-3 Accuracy | Validation Loss |
|---|---:|---:|---:|---:|
| **Frozen MobileNetV2** | 0 | **93.48%** | **99.05%** | **0.9676** |
| Fine-tune Trial 1 | 4 | 92.66% | 99.05% | 0.9967 |
| Fine-tune Trial 2 | 2 | 92.53% | 98.78% | 0.9855 |
| Fine-tune Trial 3 | 1 | 91.71% | 98.91% | 1.0474 |

The frozen MobileNetV2 achieved the strongest validation accuracy among the
transfer-learning experiments.

Trial 1 initially unfroze four feature blocks. Because validation accuracy
decreased, Trial 2 reduced the number of unfrozen blocks and used smaller
learning rates. Trial 3 applied an even more conservative strategy by
unfreezing only the final feature block with very small learning rates.

None of the fine-tuning strategies improved validation accuracy over the
frozen feature extractor. These results suggest that the ImageNet-pretrained
features already transfer effectively to this dataset, while updating
additional backbone parameters can reduce generalization when the available
training data is limited.

The **test set was not used for model selection** and is reserved for the
team's final evaluation stage.

Detailed experiment summaries are available in `reports/`.

## Model Comparison

Headline comparison of the three model types on the **validation** split (the
test set is reserved for final evaluation). Full transfer-learning experiments
are in the *MobileNetV2 Experiments* section above.

| Model | Pretraining | Validation Accuracy | Top-3 Accuracy |
|---|---|---:|---:|
| Custom CNN (from scratch) | None | 38.32% | 64.40% |
| Fine-tuned MobileNetV2 (best trial) | ImageNet | 92.66% | 99.05% |
| **Frozen MobileNetV2** | ImageNet | **93.48%** | **99.05%** |

Transfer learning lifts validation accuracy from **38.32%** (trained from
scratch) to **93.48%** (frozen ImageNet features) — a gain of roughly **55
percentage points**. This is the project's central finding: on a small,
fine-grained dataset, pretrained ImageNet representations transfer far more
effectively than features learned from scratch, and freezing the backbone even
outperforms fine-tuning it.

## Evaluation

The final evaluation will include:

- Top-1 accuracy
- Macro F1-score
- Top-3 accuracy
- Confusion matrix
- Per-class performance
- Training and validation curves
- Error analysis
- Grad-CAM visualizations

Grad-CAM will be used to examine which image regions influence model
predictions and whether the classifier focuses on meaningful breed-specific
features or potentially misleading background information.

## Reproducibility

All experiment entry points seed Python, NumPy, and PyTorch for reproducibility.

The default seed is:

```text
SEED = 42
```

See:

```text
src/pet_classifier/utils/reproducibility.py
src/pet_classifier/config.py
```

Experiment settings are stored in YAML files under `configs/`, while validation
results and training summaries are stored under `reports/`.

Model checkpoints and downloaded datasets are excluded from Git because of
their size.

## Development Workflow

Development is tracked through GitHub Issues, feature branches, commits, Pull
Requests, and peer review.

The project follows the general workflow:

```text
Issue
→ Feature branch
→ Incremental commits
→ Experiment / validation
→ Pull Request
→ Peer review
→ Merge into main
```

This history is intended to preserve both model development and experimental
iteration, including changes made in response to validation results.