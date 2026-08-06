"""Train MobileNetV2 as a frozen feature extractor.

Only the newly added 37-class classification head is trained. The
pretrained convolutional backbone remains frozen.

Usage
-----
uv run python scripts/train_mobilenet_frozen.py \
    --config configs/mobilenet_frozen.yaml

Quick smoke test
----------------
uv run python scripts/train_mobilenet_frozen.py \
    --config configs/mobilenet_frozen.yaml \
    --epochs 1 \
    --batch-size 8
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml
from torch import nn

from pet_classifier.config import DATA_DIR, FIGURE_DIR, ROOT_DIR
from pet_classifier.data.dataset import build_data
from pet_classifier.models.transfer import (
    MobileNetV2Classifier,
    count_all_parameters,
    count_trainable_parameters,
)
from pet_classifier.training.early_stopping import EarlyStopping
from pet_classifier.training.trainer import Trainer, get_device
from pet_classifier.utils.reproducibility import seed_everything
from pet_classifier.utils.viz import plot_training_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument(
        "--config",
        default=str(ROOT_DIR / "configs" / "mobilenet_frozen.yaml"),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--no-download", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument(
        "--evaluate-test",
        action="store_true",
        help="Evaluate on the test set after training.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, encoding="utf-8") as file:
        config = yaml.safe_load(file)

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.batch_size is not None:
        config["data"]["batch_size"] = args.batch_size

    seed_everything()
    device = get_device(prefer_cuda=not args.cpu)
    print(f"Device: {device}")

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    data_config = config["data"]

    data = build_data(
        root=args.data_dir or DATA_DIR,
        img_size=data_config["img_size"],
        val_fraction=data_config["val_fraction"],
        batch_size=data_config["batch_size"],
        num_workers=data_config["num_workers"],
        download=not args.no_download,
    )

    print(
        f"train {len(data.train_labels)} | "
        f"val {len(data.val_labels)} | "
        f"test {len(data.test_labels)}"
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model_config = config["model"]

    model = MobileNetV2Classifier(
        num_classes=len(data.class_names),
        dropout=model_config["dropout"],
        pretrained=model_config["pretrained"],
    )

    total_parameters = count_all_parameters(model)
    trainable_parameters = count_trainable_parameters(model)

    print(f"Total parameters    : {total_parameters:,}")
    print(f"Trainable parameters: {trainable_parameters:,}")
    print(
        "Trainable percentage: "
        f"{100 * trainable_parameters / total_parameters:.2f}%"
    )

    # ------------------------------------------------------------------
    # Optimization
    # ------------------------------------------------------------------
    training_config = config["training"]

    criterion = nn.CrossEntropyLoss(
        label_smoothing=training_config.get("label_smoothing", 0.0)
    )

    optimizer = torch.optim.AdamW(
        (
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        lr=training_config["lr"],
        weight_decay=training_config["weight_decay"],
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training_config["epochs"],
    )

    early_config = training_config["early_stopping"]

    early_stopping = EarlyStopping(
        patience=early_config["patience"],
        mode=early_config["mode"],
        restore_best=True,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        topk=3,
    )

    checkpoint_path = ROOT_DIR / config["checkpoint"]

    history = trainer.fit(
        train_loader=data.train,
        val_loader=data.val,
        epochs=training_config["epochs"],
        early_stopping=early_stopping,
        checkpoint_path=checkpoint_path,
        checkpoint_meta={
            "architecture": "mobilenet_v2",
            "training_stage": "frozen",
            "class_names": data.class_names,
            "config": config,
        },
        monitor=training_config["monitor"],
    )

    # Explicitly reload the best validation checkpoint before evaluation.
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])

    validation_metrics = trainer.evaluate(
        data.val,
        desc="best checkpoint [val]",
    )

    curve_path = plot_training_curves(
        history,
        save_path=FIGURE_DIR / "mobilenet_frozen_curves.png",
    )

    test_metrics = None

    if args.evaluate_test:
        test_metrics = trainer.evaluate(
            data.test,
            desc="best checkpoint [test]",
        )

    # ------------------------------------------------------------------
    # Save a lightweight JSON report. Checkpoints remain git-ignored.
    # ------------------------------------------------------------------
    report_path = ROOT_DIR / config["report"]
    report_path.parent.mkdir(parents=True, exist_ok=True)

    report = {
        "model": "MobileNetV2",
        "training_stage": "frozen",
        "device": str(device),
        "best_epoch": checkpoint["epoch"],
        "total_parameters": total_parameters,
        "trainable_parameters": trainable_parameters,
        "validation_metrics": validation_metrics,
        "test_metrics": test_metrics,
        "history": history,
    }

    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print("\n=== Frozen MobileNetV2 ===")
    print(f"Best epoch    : {checkpoint['epoch']}")
    print(f"Validation acc: {validation_metrics['acc']:.4f}")
    print(f"Validation top-3: {validation_metrics['topk']:.4f}")
    print(f"Validation loss : {validation_metrics['loss']:.4f}")

    if test_metrics is not None:
        print(f"Test accuracy   : {test_metrics['acc']:.4f}")
        print(f"Test top-3      : {test_metrics['topk']:.4f}")
        print(f"Test loss       : {test_metrics['loss']:.4f}")

    print(f"Checkpoint      : {checkpoint_path}")
    print(f"Training curves : {curve_path}")
    print(f"Results report  : {report_path}")


if __name__ == "__main__":
    main()