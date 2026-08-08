# Final Evaluation Summary (Member 3)

Held-out **test set** evaluation of the strongest model, produced by
`scripts/evaluate_model.py`. The test set was never used during model
development or selection.

## Setup

- Model: Frozen MobileNetV2 (ImageNet-pretrained backbone, trained 37-class head)
- Checkpoint: `checkpoints/mobilenet_frozen.pth`
- Input size: 224 × 224
- Test images: 3,669 (official Oxford-IIIT Pet test split)
- Metrics: top-1 accuracy, macro F1, top-3 accuracy, confusion matrix,
  most-confused pairs, Grad-CAM

## Results

| Metric | Frozen MobileNetV2 (test) | Custom CNN (test) |
|---|---:|---:|
| Top-1 accuracy | 89.86% | 33.77% |
| Macro F1 | 89.82% | 31.38% |
| Top-3 accuracy | 98.39% | 60.62% |

The frozen MobileNetV2 generalizes well to the held-out test set: 89.86% top-1
(93.48% on validation — a normal, modest generalization gap) and 98.39% top-3.
The from-scratch Custom CNN reaches 33.77%, so transfer learning improves
test-set accuracy by ~56 percentage points.

## Most-confused breed pairs

| True breed | Predicted as | Count |
|---|---|---:|
| Birman | Ragdoll | 22 |
| Egyptian Mau | Bengal | 17 |
| Ragdoll | Birman | 16 |
| Staffordshire Bull Terrier | American Pit Bull Terrier | 15 |
| American Pit Bull Terrier | American Bulldog | 14 |
| Maine Coon | Bengal | 12 |
| American Pit Bull Terrier | Staffordshire Bull Terrier | 12 |
| Staffordshire Bull Terrier | American Bulldog | 11 |

## Observations

The errors are highly interpretable and cluster around visually similar breeds:

- **Colour-point long-haired cats** — Birman and Ragdoll are confused in both
  directions (22 + 16), the single largest error source.
- **Spotted / tabby cats** — Egyptian Mau, Bengal and Maine Coon are mutually
  confused.
- **"Bully" dog breeds** — Staffordshire Bull Terrier, American Pit Bull Terrier
  and American Bulldog form a tight confusion cluster, matching how hard these
  are for people to tell apart.

Because the mistakes fall on genuine look-alikes rather than random classes, the
model appears to rely on real breed-discriminative features. Grad-CAM overlays
(`reports/figures/gradcam_mobilenet_frozen.png`) support this: activation
concentrates on the animal's face and coat rather than the background.

## Figures

Produced by `scripts/evaluate_model.py --name mobilenet_frozen` under
`reports/figures/`:

- `confusion_mobilenet_frozen.png` — row-normalized 37×37 confusion matrix
- `gradcam_mobilenet_frozen.png` — Grad-CAM overlays on test images
- `gallery_mobilenet_frozen.png` — correct/incorrect prediction gallery

## Reproduce

```bash
uv run python scripts/evaluate_model.py \
    --checkpoint checkpoints/mobilenet_frozen.pth \
    --architecture mobilenet_v2 --name mobilenet_frozen
# or on a free GPU: notebooks/run_evaluation_colab.ipynb
```
