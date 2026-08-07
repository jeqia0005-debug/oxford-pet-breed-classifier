"""Load a trained checkpoint (custom CNN or MobileNetV2) for evaluation.

Both training scripts save a dict with at least ``model_state``, ``epoch``,
``val_metrics``, and ``class_names`` (see :class:`Trainer.save_checkpoint`).
The MobileNetV2 scripts additionally record ``architecture`` and
``training_stage``; the custom-CNN script does not, so ``architecture`` can
be passed explicitly when it can't be inferred from the checkpoint.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn

from pet_classifier.config import NUM_CLASSES
from pet_classifier.models.custom_cnn import CustomCNN
from pet_classifier.models.transfer import MobileNetV2Classifier

Architecture = Literal["custom_cnn", "mobilenet_v2"]


def load_checkpoint(
    path: str | Path,
    architecture: Architecture | None = None,
    device: torch.device | str = "cpu",
) -> tuple[nn.Module, dict]:
    """Reconstruct a model from a checkpoint file and load its weights.

    Parameters
    ----------
    path:
        Path to the ``.pth`` file saved by ``Trainer.save_checkpoint``.
    architecture:
        ``"custom_cnn"`` or ``"mobilenet_v2"``. Required for custom-CNN
        checkpoints (they don't record this themselves); inferred from the
        checkpoint for MobileNetV2 checkpoints if omitted.
    device:
        Where to place the reconstructed model.

    Returns
    -------
    (model, checkpoint) — the model in ``eval()`` mode, and the raw
    checkpoint dict (useful for ``class_names``, ``val_metrics``, etc.).
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    architecture = architecture or checkpoint.get("architecture")
    if architecture is None:
        raise ValueError(
            "Checkpoint does not record its architecture; pass "
            "architecture='custom_cnn' or 'mobilenet_v2' explicitly."
        )

    class_names = checkpoint.get("class_names")
    num_classes = len(class_names) if class_names else NUM_CLASSES

    if architecture == "custom_cnn":
        model_config = checkpoint.get("config", {}).get("model", {})
        model = CustomCNN(
            num_classes=num_classes,
            base_channels=model_config.get("base_channels", 32),
            num_blocks=model_config.get("num_blocks", 4),
            dropout=model_config.get("dropout", 0.5),
            spatial_dropout=model_config.get("spatial_dropout", 0.1),
        )
    elif architecture == "mobilenet_v2":
        model_config = checkpoint.get("config", {}).get("model", {})
        model = MobileNetV2Classifier(
            num_classes=num_classes,
            dropout=model_config.get("dropout", 0.2),
            pretrained=False,  # weights are overwritten by the checkpoint
        )
    else:
        raise ValueError(f"Unknown architecture: {architecture!r}")

    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint


def gradcam_target_layer(model: nn.Module) -> nn.Module:
    """Return the last convolutional block to target for Grad-CAM.

    Works for both ``CustomCNN`` (``model.features[-1]``) and
    ``MobileNetV2Classifier`` (``model.network.features[-1]``).
    """
    if hasattr(model, "network"):  # MobileNetV2Classifier
        return model.network.features[-1]
    return model.features[-1]  # CustomCNN
