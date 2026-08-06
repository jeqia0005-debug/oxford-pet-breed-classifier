"""Reusable training loop shared by the custom CNN and (later) MobileNetV2.

Handles device placement, per-epoch train/eval, top-1 and top-3 tracking,
best-checkpoint saving, optional LR scheduling, and early stopping.
"""

from __future__ import annotations

from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from pet_classifier.training.early_stopping import EarlyStopping
from pet_classifier.training.metrics import accuracy, topk_accuracy


def get_device(prefer_cuda: bool = True) -> torch.device:
    """Return the best available accelerator, or CPU when forced/unavailable."""

    if not prefer_cuda:
        return torch.device("cpu")

    if torch.cuda.is_available():
        return torch.device("cuda")

    # Apple Silicon acceleration for macOS.
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        device: torch.device | None = None,
        scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
        topk: int = 3,
    ):
        self.device = device or get_device()
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.topk = topk
        self.history: dict[str, list[float]] = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_topk": [],
        }

    def train_one_epoch(self, loader: DataLoader, desc: str = "train") -> dict:
        self.model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            batch = images.size(0)
            running_loss += loss.item() * batch
            running_acc += accuracy(logits, targets) * batch
            n += batch
        return {"loss": running_loss / n, "acc": running_acc / n}

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, desc: str = "eval") -> dict:
        self.model.eval()
        running_loss = 0.0
        running_acc = 0.0
        running_topk = 0.0
        n = 0
        for images, targets in tqdm(loader, desc=desc, leave=False):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)

            logits = self.model(images)
            loss = self.criterion(logits, targets)

            batch = images.size(0)
            running_loss += loss.item() * batch
            running_acc += accuracy(logits, targets) * batch
            running_topk += topk_accuracy(logits, targets, self.topk) * batch
            n += batch
        return {
            "loss": running_loss / n,
            "acc": running_acc / n,
            "topk": running_topk / n,
        }

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int,
        early_stopping: EarlyStopping | None = None,
        checkpoint_path: str | Path | None = None,
        checkpoint_meta: dict | None = None,
        monitor: str = "val_acc",
    ) -> dict:
        """Train for up to ``epochs``, saving the best checkpoint by ``monitor``.

        ``monitor`` is one of ``val_acc`` (maximized) or ``val_loss``
        (minimized). Returns the history dict.
        """
        maximize = monitor == "val_acc"
        best_metric = -float("inf") if maximize else float("inf")

        for epoch in range(1, epochs + 1):
            tr = self.train_one_epoch(train_loader, desc=f"epoch {epoch}/{epochs} [train]")
            va = self.evaluate(val_loader, desc=f"epoch {epoch}/{epochs} [val]")

            if self.scheduler is not None:
                self.scheduler.step()

            self.history["train_loss"].append(tr["loss"])
            self.history["train_acc"].append(tr["acc"])
            self.history["val_loss"].append(va["loss"])
            self.history["val_acc"].append(va["acc"])
            self.history["val_topk"].append(va["topk"])

            print(
                f"epoch {epoch:3d}/{epochs} | "
                f"train loss {tr['loss']:.3f} acc {tr['acc']:.3f} | "
                f"val loss {va['loss']:.3f} acc {va['acc']:.3f} "
                f"top{self.topk} {va['topk']:.3f}"
            )

            current = va["acc"] if maximize else va["loss"]
            improved = (current > best_metric) if maximize else (current < best_metric)
            if improved:
                best_metric = current
                if checkpoint_path is not None:
                    self.save_checkpoint(checkpoint_path, epoch, va, checkpoint_meta)

            if early_stopping is not None:
                score = va["acc"] if early_stopping.mode == "max" else va["loss"]
                early_stopping.step(score, epoch, self.model)
                if early_stopping.should_stop:
                    print(
                        f"Early stopping at epoch {epoch} "
                        f"(best epoch {early_stopping.best_epoch})."
                    )
                    early_stopping.restore(self.model)
                    break

        return self.history

    def save_checkpoint(
        self,
        path: str | Path,
        epoch: int,
        val_metrics: dict,
        meta: dict | None = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_state": self.model.state_dict(),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        if meta:
            payload.update(meta)
        torch.save(payload, path)
