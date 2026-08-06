"""MobileNetV2 transfer-learning model for pet breed classification.

The model supports two stages:

1. Frozen feature extraction:
   The pretrained convolutional backbone is frozen and only the new
   37-class classifier is trained.

2. Partial fine-tuning:
   The final MobileNetV2 feature blocks are unfrozen and trained with
   a lower learning rate.
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from pet_classifier.config import NUM_CLASSES


class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 classifier with controllable backbone freezing."""

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        dropout: float = 0.2,
        pretrained: bool = True,
    ) -> None:
        super().__init__()

        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.network = mobilenet_v2(weights=weights)

        in_features = self.network.classifier[1].in_features

        # Replace the original 1,000-class ImageNet head.
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes),
        )

        # Frozen feature extraction is the default first stage.
        self.freeze_backbone()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def freeze_backbone(self) -> None:
        """Freeze every convolutional feature block."""

        for parameter in self.network.features.parameters():
            parameter.requires_grad = False

        # The newly created classification head remains trainable.
        for parameter in self.network.classifier.parameters():
            parameter.requires_grad = True

    def unfreeze_last_blocks(self, num_blocks: int) -> None:
        """Unfreeze the final ``num_blocks`` MobileNetV2 feature blocks."""

        blocks = list(self.network.features.children())

        if num_blocks < 1:
            raise ValueError("num_blocks must be at least 1.")

        if num_blocks > len(blocks):
            raise ValueError(
                f"Cannot unfreeze {num_blocks} blocks; "
                f"MobileNetV2 only has {len(blocks)} feature blocks."
            )

        # Reset the backbone to fully frozen first.
        self.freeze_backbone()

        # Unfreeze only the final feature blocks.
        for block in blocks[-num_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True

    def train(self, mode: bool = True) -> "MobileNetV2Classifier":
        """Keep frozen feature blocks in evaluation mode during training.

        This prevents BatchNorm running statistics in frozen blocks from
        changing while the classification head is being trained.
        """

        super().train(mode)

        if mode:
            for block in self.network.features:
                has_trainable_parameters = any(
                    parameter.requires_grad for parameter in block.parameters()
                )

                if not has_trainable_parameters:
                    block.eval()

        return self


def count_all_parameters(model: nn.Module) -> int:
    """Return the total number of model parameters."""

    return sum(parameter.numel() for parameter in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    """Return the number of parameters updated by the optimizer."""

    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )