"""Offline tests for the MobileNetV2 transfer-learning model."""

from __future__ import annotations

import torch

from pet_classifier.models.transfer import (
    MobileNetV2Classifier,
    count_all_parameters,
    count_trainable_parameters,
)


def test_mobilenet_forward_shape():
    model = MobileNetV2Classifier(
        num_classes=37,
        pretrained=False,
    )
    model.eval()

    images = torch.randn(2, 3, 96, 96)

    with torch.no_grad():
        logits = model(images)

    assert logits.shape == (2, 37)


def test_backbone_is_frozen_by_default():
    model = MobileNetV2Classifier(pretrained=False)

    assert all(
        not parameter.requires_grad
        for parameter in model.network.features.parameters()
    )

    assert all(
        parameter.requires_grad
        for parameter in model.network.classifier.parameters()
    )

    assert count_trainable_parameters(model) < count_all_parameters(model)


def test_unfreeze_last_blocks_increases_trainable_parameters():
    model = MobileNetV2Classifier(pretrained=False)

    frozen_count = count_trainable_parameters(model)
    model.unfreeze_last_blocks(num_blocks=3)
    fine_tuned_count = count_trainable_parameters(model)

    assert fine_tuned_count > frozen_count

    blocks = list(model.network.features.children())

    assert all(
        not parameter.requires_grad
        for block in blocks[:-3]
        for parameter in block.parameters()
    )

    assert all(
        parameter.requires_grad
        for block in blocks[-3:]
        for parameter in block.parameters()
    )


def test_frozen_blocks_remain_in_eval_mode():
    model = MobileNetV2Classifier(pretrained=False)
    model.train()

    # The classifier should train, while the frozen backbone stays in eval mode.
    assert model.network.classifier.training
    assert not model.network.features[0].training