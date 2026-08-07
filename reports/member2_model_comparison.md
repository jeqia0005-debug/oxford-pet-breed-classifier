# Member 2 MobileNetV2 Fine-Tuning Comparison

## Validation Results

| Model | Unfrozen Blocks | Validation Accuracy | Top-3 Accuracy | Validation Loss |
|---|---:|---:|---:|---:|
| Frozen MobileNetV2 | 0 | 93.48% | 99.05% | 0.9676 |
| Fine-tune Trial 1 | 4 | 92.66% | 99.05% | 0.9967 |
| Fine-tune Trial 2 | 2 | 92.53% | 98.78% | 0.9855 |
| Fine-tune Trial 3 | 1 | 91.71% | 98.91% | 1.0474 |

## Iteration

The frozen MobileNetV2 model achieved the strongest validation accuracy at
93.48%.

In Trial 1, the final four MobileNetV2 feature blocks were unfrozen.
Validation accuracy decreased to 92.66%, suggesting that updating a
relatively large portion of the pretrained backbone did not improve
generalization.

In Trial 2, the number of unfrozen blocks was reduced from four to two,
and smaller learning rates were used. Validation accuracy remained below
the frozen baseline at 92.53%.

In Trial 3, only the final feature block was unfrozen and even smaller
learning rates were used. Validation accuracy decreased further to 91.71%.

Overall, the experiments indicate that the ImageNet-pretrained features
already transfer well to this pet breed classification task. Fine-tuning
additional backbone parameters did not improve validation performance on
the limited training dataset and introduced a larger generalization gap.

The frozen MobileNetV2 model will therefore remain the preferred model
based on validation performance.

The test set was not used during model selection.