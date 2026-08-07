# Custom CNN Baseline Experiment

## Setup

- Architecture: Custom CNN built from scratch (VGG-style)
- Pretraining: None (trained only on the Oxford-IIIT Pet trainval split)
- Blocks: 4 × (Conv → BatchNorm → ReLU → Conv → BatchNorm → ReLU → Dropout2d → MaxPool)
- Head: Global Average Pooling → Dropout(0.5) → 37-class linear layer
- Input size: 160 × 160
- Optimizer: Adam
- Initial learning rate: 0.001
- Weight decay: 0.0001
- LR schedule: Cosine annealing
- Loss: Cross-entropy with label smoothing (0.1)
- Regularization: BatchNorm, spatial Dropout (0.1), classifier Dropout (0.5),
  train-time augmentation (random resized crop, flip, colour jitter, rotation,
  random erasing)
- Early stopping patience: 10 (monitor: validation accuracy)
- Device: GPU (Google Colab, T4)

## Results

- Best epoch: 56 (of 60 run; validation accuracy was still rising, so early
  stopping did not trigger)
- Validation accuracy: 38.32%
- Validation top-3 accuracy: 64.40%
- Validation loss: 2.4981
- Test accuracy: 33.77%
- Test macro F1: 31.38%
- Test top-3 accuracy: 60.62%
- Total (trainable) parameters: 1,182,725

## Observations

Training and validation curves are healthy: both losses decrease steadily and
validation accuracy climbs to ~38% without diverging. Validation accuracy sits
slightly above training accuracy throughout — expected here, because the
training images receive strong augmentation (harder) while the validation images
use the deterministic resize/centre-crop pipeline. The model had not yet
plateaued at epoch 60, so a longer schedule could squeeze out a little more.

As a from-scratch model on a 37-way fine-grained task with only ~80 training
images per breed, this is a reasonable baseline. Its purpose is to establish a
reference point for the transfer-learning models: the frozen MobileNetV2 reaches
93.48% validation accuracy versus 38.32% here — an improvement of roughly 55
percentage points. This gap is the core finding of the project, showing that
ImageNet-pretrained features transfer very effectively to this small
fine-grained dataset, whereas learning useful representations from scratch is
data-limited.

The test set was not used for model selection; the test numbers above are
reported only as a reference and are reserved for the team's final evaluation
stage (Member 3).

## Reproduce

```bash
uv run python scripts/train_custom_cnn.py --config configs/custom_cnn.yaml
# or on a free GPU: notebooks/run_custom_cnn_colab.ipynb
```
