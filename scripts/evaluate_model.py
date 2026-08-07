"""Final test-set evaluation for a trained checkpoint (Member 3).

Runs the checkpoint against the held-out test set (never used for model
selection), and saves:

* ``reports/evaluation_<name>.json``      — top-1 / macro-F1 / top-3 / most-confused pairs
* ``reports/figures/confusion_<name>.png`` — row-normalized confusion matrix
* ``reports/figures/gallery_<name>.png``   — sample predictions (correct + incorrect)
* ``reports/figures/gradcam_<name>.png``   — Grad-CAM overlays for a few test images

Usage
-----
uv run python scripts/evaluate_model.py \
    --checkpoint checkpoints/mobilenet_frozen.pth \
    --architecture mobilenet_v2 \
    --name mobilenet_frozen
"""

from __future__ import annotations

import argparse
import json

import torch

from pet_classifier.config import DATA_DIR, FIGURE_DIR, ROOT_DIR
from pet_classifier.data.dataset import build_data
from pet_classifier.evaluation.checkpoints import load_checkpoint
from pet_classifier.evaluation.gradcam import GradCAMExplainer
from pet_classifier.evaluation.metrics import run_full_evaluation
from pet_classifier.evaluation.viz import plot_confusion_matrix, plot_prediction_gallery
from pet_classifier.training.trainer import get_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pth checkpoint")
    parser.add_argument(
        "--architecture",
        choices=["custom_cnn", "mobilenet_v2"],
        default=None,
        help="Required for custom_cnn checkpoints; inferred for mobilenet_v2.",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Short label used in output filenames (default: checkpoint stem).",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--gradcam-samples", type=int, default=8)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(prefer_cuda=not args.cpu)
    print(f"Device: {device}")

    name = args.name or args.checkpoint.split("/")[-1].removesuffix(".pth")

    # ------------------------------------------------------------------
    # Load model + checkpoint metadata
    # ------------------------------------------------------------------
    model, checkpoint = load_checkpoint(
        args.checkpoint, architecture=args.architecture, device=device
    )
    class_names = checkpoint.get("class_names")
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')}")

    # ------------------------------------------------------------------
    # Data — use the SAME split as training so the test set stays untouched
    # until now.
    # ------------------------------------------------------------------
    data = build_data(
        root=DATA_DIR,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        download=False,
    )
    class_names = class_names or data.class_names

    # ------------------------------------------------------------------
    # Full test-set evaluation
    # ------------------------------------------------------------------
    result = run_full_evaluation(model, data.test, class_names, device=device)

    print(f"\n=== {name} — test set ===")
    print(f"Top-1 accuracy : {result.top1_acc:.4f}")
    print(f"Top-3 accuracy : {result.top3_acc:.4f}")
    print(f"Macro F1       : {result.macro_f1:.4f}")
    print("Most confused breed pairs:")
    for pair in result.most_confused_pairs(top_n=5):
        print(
            f"  {pair['true_breed']!r} -> {pair['predicted_breed']!r} "
            f"({pair['count']} times)"
        )

    report_path = ROOT_DIR / "reports" / f"evaluation_{name}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(
            {
                "checkpoint": args.checkpoint,
                "checkpoint_epoch": checkpoint.get("epoch"),
                **result.summary(),
                "classification_report": result.report,
            },
            file,
            indent=2,
        )
    print(f"Saved report: {report_path}")

    # ------------------------------------------------------------------
    # Confusion matrix
    # ------------------------------------------------------------------
    cm_path = plot_confusion_matrix(
        result.confusion,
        class_names,
        save_path=FIGURE_DIR / f"confusion_{name}.png",
    )
    print(f"Saved confusion matrix: {cm_path}")

    # ------------------------------------------------------------------
    # Sample prediction gallery + Grad-CAM (first batch of the test loader)
    # ------------------------------------------------------------------
    images, targets = next(iter(data.test))
    images = images[: args.gradcam_samples]
    targets = targets[: args.gradcam_samples].numpy()

    with torch.no_grad():
        probs = torch.softmax(model(images.to(device)), dim=1).cpu().numpy()
    preds = probs.argmax(axis=1)
    confidences = probs.max(axis=1)

    gallery_path = plot_prediction_gallery(
        images,
        targets,
        preds,
        confidences,
        class_names,
        n=args.gradcam_samples,
        save_path=FIGURE_DIR / f"gallery_{name}.png",
    )
    print(f"Saved prediction gallery: {gallery_path}")

    explainer = GradCAMExplainer(model, device=device)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncols = 4
    nrows = (args.gradcam_samples + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(3 * ncols, 3.3 * nrows))
    axes = axes.ravel() if hasattr(axes, "ravel") else [axes]
    for i in range(args.gradcam_samples):
        overlay, predicted, confidence = explainer.explain_tensor(images[i])
        correct = predicted == targets[i]
        axes[i].imshow(overlay)
        axes[i].set_title(
            f"true: {class_names[targets[i]]}\n"
            f"pred: {class_names[predicted]} ({confidence:.0%})",
            fontsize=8,
            color="#2ca02c" if correct else "#d62728",
        )
        axes[i].axis("off")
    for j in range(args.gradcam_samples, len(axes)):
        axes[j].axis("off")
    fig.tight_layout()
    gradcam_path = FIGURE_DIR / f"gradcam_{name}.png"
    gradcam_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(gradcam_path, dpi=120)
    plt.close(fig)
    print(f"Saved Grad-CAM gallery: {gradcam_path}")


if __name__ == "__main__":
    main()
