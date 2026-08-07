# Member 2 MobileNetV2 Fine-Tuning Comparison

## Validation Results

| Model | Unfrozen Blocks | Validation Accuracy | Top-3 Accuracy | Validation Loss |
|---|---:|---:|---:|---:|
| Frozen MobileNetV2 | 0 | 93.48% | 99.05% | 0.9676 |
| Fine-tune Trial 1 | 4 | 92.66% | 99.05% | 0.9967 |
| Fine-tune Trial 2 | 2 | 92.53% | 98.78% | 0.9855 |

## Iteration

The frozen MobileNetV2 model achieved the strongest validation accuracy at
93.48%.

In the first fine-tuning trial, the final four feature blocks were unfrozen.
Validation accuracy decreased to 92.66%, suggesting that updating a relatively
large portion of the pretrained backbone did not improve generalization on the
small training dataset.

For the second trial, the number of unfrozen blocks was reduced from four to
two and smaller learning rates were used. Validation loss improved relative to
Trial 1, but validation accuracy remained below the frozen baseline at 92.53%.

These results suggest that the pretrained ImageNet features are already well
suited to this classification task and that additional fine-tuning can slightly
reduce generalization performance under the current dataset size and training
configuration.

The test set was not used for model selection.