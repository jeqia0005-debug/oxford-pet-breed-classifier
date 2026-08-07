"""Train the from-scratch custom CNN baseline.

Downloads the data (first run), trains with Batch Normalization, Dropout and
Early Stopping, saves the best-validation checkpoint and the training curves,
then reports baseline metrics on the held-out test set.

Usage
-----
    uv run python scripts/train_custom_cnn.py --config configs/custom_cnn.yaml
    uv run python scripts/train_custom_cnn.py --config configs/custom_cnn.yaml --epochs 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from pet_classifier.config import CHECKPOINT_DIR, DATA_DIR, FIGURE_DIR, ROOT_DIR
from pet_classifier.data.dataset import build_data
from pet_classifier.models.custom_cnn import CustomCNN, count_parameters
from pet_classifier.training.early_stopping import EarlyStopping
from pet_classifier.training.metrics import evaluate_classification
from pet_classifier.training.trainer import Trainer, get_device
from pet_classifier.utils.reproducibility import seed_everything
from pet_classifier.utils.viz import plot_training_curves

import torch
from torch import nn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default=str(ROOT_DIR / "configs" / "custom_cnn.yaml"))
    p.add_argument("--epochs", type=int, default=None, help="override config epochs")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--no-download", action="store_true")
    p.add_argument("--cpu", action="store_true", help="force CPU even if CUDA is present")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size

    seed_everything()
    device = get_device(prefer_cuda=not args.cpu)
    print(f"Device: {device}")

    # --- Data --------------------------------------------------------------
    dcfg = cfg["data"]
    data = build_data(
        root=args.data_dir or DATA_DIR,
        img_size=dcfg["img_size"],
        val_fraction=dcfg["val_fraction"],
        batch_size=dcfg["batch_size"],
        num_workers=dcfg["num_workers"],
        download=not args.no_download,
    )
    print(f"train {len(data.train_labels)} | val {len(data.val_labels)} | "
          f"test {len(data.test_labels)}")

    # --- Model -------------------------------------------------------------
    mcfg = cfg["model"]
    model = CustomCNN(
        num_classes=len(data.class_names),
        base_channels=mcfg["base_channels"],
        num_blocks=mcfg["num_blocks"],
        dropout=mcfg["dropout"],
        spatial_dropout=mcfg["spatial_dropout"],
    )
    print(f"CustomCNN — {count_parameters(model):,} trainable parameters")

    # --- Optimization ------------------------------------------------------
    tcfg = cfg["training"]
    criterion = nn.CrossEntropyLoss(label_smoothing=tcfg.get("label_smoothing", 0.0))
    optimizer = torch.optim.Adam(
        model.parameters(), lr=tcfg["lr"], weight_decay=tcfg["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tcfg["epochs"]
    )
    stopper = EarlyStopping(
        patience=tcfg["early_stopping"]["patience"],
        mode=tcfg["early_stopping"]["mode"],
        restore_best=True,
    )

    trainer = Trainer(model, optimizer, criterion, device=device, scheduler=scheduler, topk=3)

    ckpt_path = ROOT_DIR / cfg.get("checkpoint", str(CHECKPOINT_DIR / "custom_cnn.pth"))
    history = trainer.fit(
        data.train,
        data.val,
        epochs=tcfg["epochs"],
        early_stopping=stopper,
        checkpoint_path=ckpt_path,
        checkpoint_meta={"class_names": data.class_names, "config": cfg},
        monitor=tcfg["monitor"],
    )

    # --- Curves ------------------------------------------------------------
    curve_path = plot_training_curves(history, save_path=FIGURE_DIR / "custom_cnn_curves.png")
    print(f"Saved training curves -> {curve_path}")

    # --- Baseline test evaluation -----------------------------------------
    # Load the best-validation checkpoint before the final test scoring so the
    # reported baseline reflects the best model, not the last epoch.
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    metrics = evaluate_classification(model, data.test, device=device, topk=3)
    metrics["best_epoch"] = ckpt.get("epoch")
    metrics["img_size"] = dcfg["img_size"]
    metrics["epochs_run"] = len(history["train_loss"])

    print("\n=== Custom CNN baseline (test set) ===")
    print(f"  accuracy       : {metrics['accuracy']:.4f}")
    print(f"  macro F1       : {metrics['macro_f1']:.4f}")
    print(f"  top-3 accuracy : {metrics['top3_accuracy']:.4f}")
    print(f"  best epoch     : {metrics['best_epoch']}  (of {metrics['epochs_run']} run)")
    print(f"  best checkpoint: {ckpt_path}")

    results_path = FIGURE_DIR.parent / "custom_cnn_results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({"model": "custom_cnn", **metrics}, f, indent=2)
    print(f"  saved metrics  -> {results_path}")


if __name__ == "__main__":
    main()
