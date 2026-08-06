"""Offline tests for the custom CNN, metrics, early stopping and training loop."""

from __future__ import annotations

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from pet_classifier.models.custom_cnn import CustomCNN, count_parameters
from pet_classifier.training.early_stopping import EarlyStopping
from pet_classifier.training.metrics import accuracy, topk_accuracy
from pet_classifier.training.trainer import Trainer


def test_custom_cnn_forward_shape():
    model = CustomCNN(num_classes=37, num_blocks=3, base_channels=16)
    x = torch.randn(4, 3, 96, 96)
    out = model(x)
    assert out.shape == (4, 37)


def test_custom_cnn_has_batchnorm_and_dropout():
    model = CustomCNN()
    kinds = {type(m) for m in model.modules()}
    assert nn.BatchNorm2d in kinds
    assert nn.Dropout in kinds or nn.Dropout2d in kinds
    assert count_parameters(model) > 0


def test_metrics():
    logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 1.0, 2.0]])
    targets = torch.tensor([0, 1])
    assert accuracy(logits, targets) == 0.5
    # true labels are within the top-2 for both rows
    assert topk_accuracy(logits, targets, k=2) == 1.0


def test_early_stopping_triggers_after_patience():
    stopper = EarlyStopping(patience=2, mode="min")
    stopper.step(1.0, 1)          # improvement (first)
    assert not stopper.should_stop
    stopper.step(1.1, 2)          # worse (1)
    assert not stopper.should_stop
    stopper.step(1.2, 3)          # worse (2) -> stop
    assert stopper.should_stop
    assert stopper.best_epoch == 1


def test_early_stopping_restores_best_weights():
    model = nn.Linear(4, 2)
    stopper = EarlyStopping(patience=5, mode="max", restore_best=True)
    stopper.step(0.5, 1, model)
    best = model.weight.detach().clone()
    with torch.no_grad():           # corrupt the weights after the best epoch
        model.weight.add_(1.0)
    stopper.step(0.4, 2, model)     # no improvement
    stopper.restore(model)
    assert torch.allclose(model.weight, best)


def test_trainer_fit_runs_and_reduces_loss():
    torch.manual_seed(0)
    # Tiny synthetic classification problem the model can quickly fit.
    x = torch.randn(32, 3, 32, 32)
    y = torch.randint(0, 5, (32,))
    loader = DataLoader(TensorDataset(x, y), batch_size=8)

    model = CustomCNN(num_classes=5, num_blocks=2, base_channels=8)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    trainer = Trainer(model, optim, nn.CrossEntropyLoss(), device=torch.device("cpu"), topk=3)

    history = trainer.fit(loader, loader, epochs=3, monitor="val_acc")
    assert len(history["train_loss"]) == 3
    assert all(k in history for k in ("val_acc", "val_topk"))
    # loss should not increase over the (overfittable) tiny dataset
    assert history["train_loss"][-1] <= history["train_loss"][0] + 1e-3
