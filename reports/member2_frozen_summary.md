# Frozen MobileNetV2 Experiment

## Setup

- Architecture: MobileNetV2
- Pretraining: ImageNet
- Strategy: Frozen convolutional backbone
- Trainable layers: 37-class classification head only
- Input size: 224 × 224
- Optimizer: AdamW
- Initial learning rate: 0.001
- Loss: Cross-entropy with label smoothing
- Early stopping patience: 5
- Device: Apple MPS

## Results

- Best epoch: 12
- Validation accuracy: 93.48%
- Validation top-3 accuracy: 99.05%
- Validation loss: 0.9676
- Total parameters: 2,271,269
- Trainable parameters: 47,397
- Percentage of trainable parameters: 2.09%
- Test set evaluation: Not performed during model development

## Observations

The frozen MobileNetV2 model achieved a validation accuracy of 93.48% and
a top-3 accuracy of 99.05%. The model performed well even though only the
new classification head was trained. Approximately 2.09% of the model
parameters were trainable, while the pretrained convolutional backbone
remained frozen.

Validation accuracy reached its highest value at epoch 12. After that,
validation accuracy remained relatively stable but did not improve, while
training accuracy continued to increase slightly. This suggests that the
model had largely converged and had begun to show a small generalization
gap.

The high top-3 accuracy indicates that, even when the model's first
prediction was incorrect, the correct pet breed was usually among its
three most likely predictions. Remaining errors are likely to involve
visually similar breeds.

The next experiment will partially unfreeze the final MobileNetV2 feature
blocks and use a lower learning rate to determine whether fine-tuning can
improve validation performance.