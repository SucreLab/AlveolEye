"""Callback system for training.

This module provides a callback interface for hooking into the training loop.
Callbacks can be used for logging, checkpointing, early stopping, and custom logic.

Classes:
    TrainingState: Container for current training state passed to callbacks
    Callback: Base class for all callbacks
    CallbackList: Manager for multiple callbacks
    EarlyStoppingCallback: Stop training when validation loss stops improving
    ModelCheckpointCallback: Save model checkpoints during training
    LambdaCallback: Simple callback using lambda functions
"""

from abc import ABC
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
import os

import torch


@dataclass
class TrainingState:
    """Container for current training state passed to callbacks.

    Attributes:
        epoch: Current epoch number (0-indexed)
        total_epochs: Total number of epochs
        model: The model being trained
        optimizer: The optimizer
        train_metrics: Metrics from training (loss, etc.)
        val_metrics: Metrics from validation
        best_val_loss: Best validation loss so far
        device: Device being used
    """
    epoch: int
    total_epochs: int
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    train_metrics: Dict[str, Any]
    val_metrics: Dict[str, Any]
    best_val_loss: float
    device: torch.device


class Callback(ABC):
    """Base class for training callbacks.

    Subclass this to create custom callbacks for training events.
    All methods receive a TrainingState object with current training info.

    Example:
        class MyCallback(Callback):
            def on_epoch_end(self, state: TrainingState) -> None:
                print(f"Epoch {state.epoch} completed with loss {state.train_metrics['loss']}")
    """

    def on_train_start(self, state: TrainingState) -> None:
        """Called at the beginning of training."""
        pass

    def on_train_end(self, state: TrainingState) -> None:
        """Called at the end of training."""
        pass

    def on_epoch_start(self, state: TrainingState) -> None:
        """Called at the beginning of each epoch."""
        pass

    def on_epoch_end(self, state: TrainingState) -> None:
        """Called at the end of each epoch."""
        pass


class CallbackList:
    """Manager for multiple callbacks."""

    def __init__(self, callbacks: Optional[List[Callback]] = None):
        self.callbacks = callbacks or []

    def add(self, callback: Callback) -> None:
        """Add a callback to the list."""
        self.callbacks.append(callback)

    def on_train_start(self, state: TrainingState) -> None:
        """Call on_train_start for all callbacks."""
        for cb in self.callbacks:
            cb.on_train_start(state)

    def on_train_end(self, state: TrainingState) -> None:
        """Call on_train_end for all callbacks."""
        for cb in self.callbacks:
            cb.on_train_end(state)

    def on_epoch_start(self, state: TrainingState) -> None:
        """Call on_epoch_start for all callbacks."""
        for cb in self.callbacks:
            cb.on_epoch_start(state)

    def on_epoch_end(self, state: TrainingState) -> None:
        """Call on_epoch_end for all callbacks."""
        for cb in self.callbacks:
            cb.on_epoch_end(state)


class EarlyStoppingCallback(Callback):
    """Stop training when validation loss stops improving.

    Args:
        patience: Number of epochs to wait before stopping
        min_delta: Minimum change to qualify as an improvement
        verbose: Whether to print messages

    Example:
        callback = EarlyStoppingCallback(patience=20, min_delta=0.001)
    """

    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        verbose: bool = True
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def on_epoch_end(self, state: TrainingState) -> None:
        val_loss = state.val_metrics.get('loss', float('inf'))

        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience} epochs without improvement")

            if self.counter >= self.patience:
                self.should_stop = True
                if self.verbose:
                    print("Early stopping triggered")


class ModelCheckpointCallback(Callback):
    """Save model checkpoints during training.

    Args:
        save_dir: Directory to save checkpoints
        save_frequency: Save every N epochs (0 to disable periodic saves)
        save_best: Save when validation improves
        filename_template: Template for filenames (supports {epoch}, {val_loss})

    Example:
        callback = ModelCheckpointCallback(
            save_dir='checkpoints',
            save_frequency=10,
            save_best=True,
        )
    """

    def __init__(
        self,
        save_dir: str = '.',
        save_frequency: int = 50,
        save_best: bool = True,
        filename_template: str = 'checkpoint_epoch_{epoch}.pth'
    ):
        self.save_dir = save_dir
        self.save_frequency = save_frequency
        self.save_best = save_best
        self.filename_template = filename_template
        self.best_loss = float('inf')

    def on_epoch_end(self, state: TrainingState) -> None:
        os.makedirs(self.save_dir, exist_ok=True)

        # Save at frequency
        if self.save_frequency > 0 and state.epoch > 0 and state.epoch % self.save_frequency == 0:
            filename = self.filename_template.format(
                epoch=state.epoch,
                val_loss=state.val_metrics.get('loss', 0)
            )
            path = os.path.join(self.save_dir, filename)
            torch.save({
                'epoch': state.epoch,
                'model_state_dict': state.model.state_dict(),
                'optimizer_state_dict': state.optimizer.state_dict(),
                'val_metrics': state.val_metrics,
                'train_metrics': state.train_metrics,
            }, path)
            print(f"[Checkpoint] Saved: {path}")

        # Save best
        val_loss = state.val_metrics.get('loss', float('inf'))
        if self.save_best and val_loss < self.best_loss:
            self.best_loss = val_loss
            path = os.path.join(self.save_dir, 'best_model.pth')
            torch.save(state.model, path)
            print(f"[Checkpoint] New best model saved: {path} (loss: {val_loss:.4f})")


class LambdaCallback(Callback):
    """Simple callback using lambda/callable functions.

    Args:
        on_epoch_end: Function called at end of each epoch
        on_train_end: Function called at end of training
        on_train_start: Function called at start of training
        on_epoch_start: Function called at start of each epoch

    Example:
        callback = LambdaCallback(
            on_epoch_end=lambda state: print(f"Epoch {state.epoch} done"),
            on_train_end=lambda state: print("Training complete!")
        )
    """

    def __init__(
        self,
        on_epoch_end: Optional[Callable[[TrainingState], None]] = None,
        on_train_end: Optional[Callable[[TrainingState], None]] = None,
        on_train_start: Optional[Callable[[TrainingState], None]] = None,
        on_epoch_start: Optional[Callable[[TrainingState], None]] = None,
    ):
        self._on_epoch_end = on_epoch_end
        self._on_train_end = on_train_end
        self._on_train_start = on_train_start
        self._on_epoch_start = on_epoch_start

    def on_epoch_end(self, state: TrainingState) -> None:
        if self._on_epoch_end:
            self._on_epoch_end(state)

    def on_train_end(self, state: TrainingState) -> None:
        if self._on_train_end:
            self._on_train_end(state)

    def on_train_start(self, state: TrainingState) -> None:
        if self._on_train_start:
            self._on_train_start(state)

    def on_epoch_start(self, state: TrainingState) -> None:
        if self._on_epoch_start:
            self._on_epoch_start(state)
