"""Early stopping with best-weight restoration."""

from __future__ import annotations

import copy

import torch
from torch import nn


class EarlyStopping:
    """Stop training when a monitored metric stops improving.

    Parameters
    ----------
    patience:
        Number of epochs with no improvement before stopping.
    mode:
        ``"min"`` (e.g. val loss) or ``"max"`` (e.g. val accuracy).
    min_delta:
        Minimum change to qualify as an improvement.
    restore_best:
        If ``True``, keep a copy of the best model weights so the caller can
        restore them after stopping.
    """

    def __init__(
        self,
        patience: int = 7,
        mode: str = "min",
        min_delta: float = 0.0,
        restore_best: bool = True,
    ):
        if mode not in ("min", "max"):
            raise ValueError("mode must be 'min' or 'max'")
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best = restore_best

        self.best_score: float | None = None
        self.best_epoch: int = -1
        self.counter: int = 0
        self.should_stop: bool = False
        self._best_state: dict | None = None

    def _is_improvement(self, score: float) -> bool:
        if self.best_score is None:
            return True
        if self.mode == "min":
            return score < self.best_score - self.min_delta
        return score > self.best_score + self.min_delta

    def step(self, score: float, epoch: int, model: nn.Module | None = None) -> bool:
        """Update state with the latest ``score``. Returns True on improvement."""
        if self._is_improvement(score):
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0
            if self.restore_best and model is not None:
                self._best_state = copy.deepcopy(model.state_dict())
            return True

        self.counter += 1
        if self.counter >= self.patience:
            self.should_stop = True
        return False

    def restore(self, model: nn.Module) -> None:
        """Load the best-seen weights back into ``model`` (if tracked)."""
        if self._best_state is not None:
            model.load_state_dict(self._best_state)
