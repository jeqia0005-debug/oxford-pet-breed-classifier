"""From-scratch CNN baseline for 37-breed pet classification.

A compact VGG-style network built entirely from ``nn`` primitives (no
pretrained weights). Each stage is two ``Conv -> BatchNorm -> ReLU`` layers
followed by max-pooling; Batch Normalization stabilizes training and Dropout
(both spatial and in the classifier head) regularizes the small dataset.
Global average pooling keeps the classifier head small.
"""

from __future__ import annotations

import torch
from torch import nn

from pet_classifier.config import NUM_CLASSES


def _conv_block(in_ch: int, out_ch: int, drop2d: float) -> nn.Sequential:
    """Two Conv-BN-ReLU layers, spatial dropout, then 2x2 max-pool."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(drop2d),
        nn.MaxPool2d(2),
    )


class CustomCNN(nn.Module):
    """Configurable VGG-style CNN.

    Parameters
    ----------
    num_classes:
        Number of output breeds.
    base_channels:
        Channel width of the first stage; doubles each subsequent stage.
    num_blocks:
        Number of conv stages (each halves spatial resolution).
    dropout:
        Dropout probability in the classifier head.
    spatial_dropout:
        Dropout2d probability inside conv stages.
    """

    def __init__(
        self,
        num_classes: int = NUM_CLASSES,
        base_channels: int = 32,
        num_blocks: int = 4,
        dropout: float = 0.5,
        spatial_dropout: float = 0.1,
    ):
        super().__init__()
        blocks = []
        in_ch = 3
        out_ch = base_channels
        for _ in range(num_blocks):
            blocks.append(_conv_block(in_ch, out_ch, spatial_dropout))
            in_ch = out_ch
            out_ch = out_ch * 2
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(in_ch, num_classes),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)


def count_parameters(model: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
