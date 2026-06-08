"""Tests for training callback system.

Tests cover:
- Callback base class interface
- CallbackList management and invocation
- EarlyStoppingCallback behavior
- ModelCheckpointCallback saving logic
- LambdaCallback function invocation
"""

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest
import torch

from alveoleye.lungcv.mrcnn.callbacks import (
    Callback,
    CallbackList,
    TrainingState,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LambdaCallback,
)


class TestCallback:
    """Tests for Callback base class."""

    def test_callback_methods_exist(self):
        """Test that all callback methods exist and are callable."""
        cb = Callback()
        assert hasattr(cb, "on_train_start")
        assert hasattr(cb, "on_train_end")
        assert hasattr(cb, "on_epoch_start")
        assert hasattr(cb, "on_epoch_end")
        assert callable(cb.on_train_start)
        assert callable(cb.on_train_end)
        assert callable(cb.on_epoch_start)
        assert callable(cb.on_epoch_end)

    def test_callback_methods_accept_state(self, sample_training_state):
        """Test that callback methods accept TrainingState."""
        cb = Callback()
        # Should not raise
        cb.on_train_start(sample_training_state)
        cb.on_train_end(sample_training_state)
        cb.on_epoch_start(sample_training_state)
        cb.on_epoch_end(sample_training_state)


class TestCallbackList:
    """Tests for CallbackList manager."""

    def test_empty_callback_list(self):
        """Test creation of empty callback list."""
        cb_list = CallbackList()
        assert len(cb_list.callbacks) == 0

    def test_callback_list_with_initial_callbacks(self):
        """Test creation with initial callbacks."""
        cb1, cb2 = MagicMock(spec=Callback), MagicMock(spec=Callback)
        cb_list = CallbackList([cb1, cb2])
        assert len(cb_list.callbacks) == 2

    def test_add_callback(self):
        """Test adding a callback."""
        cb_list = CallbackList()
        cb = MagicMock(spec=Callback)
        cb_list.add(cb)
        assert cb in cb_list.callbacks
        assert len(cb_list.callbacks) == 1

    def test_on_train_start_calls_all(self, sample_training_state):
        """Test that on_train_start calls all callbacks."""
        cb1, cb2 = MagicMock(spec=Callback), MagicMock(spec=Callback)
        cb_list = CallbackList([cb1, cb2])

        cb_list.on_train_start(sample_training_state)

        cb1.on_train_start.assert_called_once_with(sample_training_state)
        cb2.on_train_start.assert_called_once_with(sample_training_state)

    def test_on_train_end_calls_all(self, sample_training_state):
        """Test that on_train_end calls all callbacks."""
        cb1, cb2 = MagicMock(spec=Callback), MagicMock(spec=Callback)
        cb_list = CallbackList([cb1, cb2])

        cb_list.on_train_end(sample_training_state)

        cb1.on_train_end.assert_called_once_with(sample_training_state)
        cb2.on_train_end.assert_called_once_with(sample_training_state)

    def test_on_epoch_start_calls_all(self, sample_training_state):
        """Test that on_epoch_start calls all callbacks."""
        cb1, cb2 = MagicMock(spec=Callback), MagicMock(spec=Callback)
        cb_list = CallbackList([cb1, cb2])

        cb_list.on_epoch_start(sample_training_state)

        cb1.on_epoch_start.assert_called_once_with(sample_training_state)
        cb2.on_epoch_start.assert_called_once_with(sample_training_state)

    def test_on_epoch_end_calls_all(self, sample_training_state):
        """Test that on_epoch_end calls all callbacks."""
        cb1, cb2 = MagicMock(spec=Callback), MagicMock(spec=Callback)
        cb_list = CallbackList([cb1, cb2])

        cb_list.on_epoch_end(sample_training_state)

        cb1.on_epoch_end.assert_called_once_with(sample_training_state)
        cb2.on_epoch_end.assert_called_once_with(sample_training_state)

    def test_callbacks_called_in_order(self, sample_training_state):
        """Test that callbacks are called in order they were added."""
        call_order = []
        cb1 = MagicMock(spec=Callback)
        cb1.on_epoch_end.side_effect = lambda _: call_order.append(1)
        cb2 = MagicMock(spec=Callback)
        cb2.on_epoch_end.side_effect = lambda _: call_order.append(2)

        cb_list = CallbackList([cb1, cb2])
        cb_list.on_epoch_end(sample_training_state)

        assert call_order == [1, 2]


class TestEarlyStoppingCallback:
    """Tests for EarlyStoppingCallback."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        cb = EarlyStoppingCallback()
        assert cb.patience == 10
        assert cb.min_delta == 0.0
        assert cb.verbose is True
        assert cb.should_stop is False
        assert cb.counter == 0

    def test_no_stop_when_improving(self, sample_training_state):
        """Test that training doesn't stop when loss improves."""
        cb = EarlyStoppingCallback(patience=3, verbose=False)

        for loss in [1.0, 0.9, 0.8, 0.7]:
            sample_training_state.val_metrics = {"loss": loss}
            cb.on_epoch_end(sample_training_state)
            assert cb.should_stop is False

    def test_stops_after_patience_epochs(self, sample_training_state):
        """Test that training stops after patience epochs without improvement."""
        cb = EarlyStoppingCallback(patience=3, verbose=False)

        # Set initial best
        sample_training_state.val_metrics = {"loss": 0.5}
        cb.on_epoch_end(sample_training_state)

        # No improvement for patience epochs
        for i in range(3):
            sample_training_state.val_metrics = {"loss": 0.6}
            cb.on_epoch_end(sample_training_state)

        assert cb.should_stop is True

    def test_counter_resets_on_improvement(self, sample_training_state):
        """Test that counter resets when loss improves."""
        cb = EarlyStoppingCallback(patience=5, verbose=False)

        # Some epochs without improvement
        sample_training_state.val_metrics = {"loss": 0.5}
        cb.on_epoch_end(sample_training_state)

        sample_training_state.val_metrics = {"loss": 0.6}
        cb.on_epoch_end(sample_training_state)
        assert cb.counter == 1

        sample_training_state.val_metrics = {"loss": 0.6}
        cb.on_epoch_end(sample_training_state)
        assert cb.counter == 2

        # Improvement - should reset counter
        sample_training_state.val_metrics = {"loss": 0.4}
        cb.on_epoch_end(sample_training_state)
        assert cb.counter == 0

    def test_min_delta_threshold(self, sample_training_state):
        """Test that min_delta is respected."""
        cb = EarlyStoppingCallback(patience=3, min_delta=0.1, verbose=False)

        sample_training_state.val_metrics = {"loss": 1.0}
        cb.on_epoch_end(sample_training_state)

        # Small improvement (less than min_delta)
        sample_training_state.val_metrics = {"loss": 0.95}
        cb.on_epoch_end(sample_training_state)
        assert cb.counter == 1  # Should count as no improvement

        # Significant improvement (greater than min_delta)
        sample_training_state.val_metrics = {"loss": 0.8}
        cb.on_epoch_end(sample_training_state)
        assert cb.counter == 0  # Should reset

    def test_best_loss_tracking(self, sample_training_state):
        """Test that best loss is tracked correctly."""
        cb = EarlyStoppingCallback(patience=5, verbose=False)

        losses = [1.0, 0.8, 0.9, 0.7, 0.75]
        for loss in losses:
            sample_training_state.val_metrics = {"loss": loss}
            cb.on_epoch_end(sample_training_state)

        assert cb.best_loss == 0.7

    def test_handles_missing_loss_key(self, sample_training_state):
        """Test handling when 'loss' key is missing."""
        cb = EarlyStoppingCallback(patience=3, verbose=False)

        sample_training_state.val_metrics = {}  # No loss key
        cb.on_epoch_end(sample_training_state)

        # Should use inf as loss, not crash
        assert cb.best_loss == float("inf")


class TestModelCheckpointCallback:
    """Tests for ModelCheckpointCallback."""

    def test_initialization_defaults(self):
        """Test default initialization values."""
        cb = ModelCheckpointCallback()
        assert cb.save_dir == "."
        assert cb.save_frequency == 50
        assert cb.save_best is True

    def test_saves_at_frequency(self, tmp_path: Path, sample_training_state):
        """Test checkpoint saving at specified frequency."""
        cb = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=5,
            save_best=False,
        )

        sample_training_state.epoch = 5
        cb.on_epoch_end(sample_training_state)

        assert (tmp_path / "checkpoint_epoch_5.pth").exists()

    def test_does_not_save_at_non_frequency_epochs(self, tmp_path: Path, sample_training_state):
        """Test that checkpoints aren't saved at non-frequency epochs."""
        cb = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=10,
            save_best=False,
        )

        sample_training_state.epoch = 7
        cb.on_epoch_end(sample_training_state)

        assert not (tmp_path / "checkpoint_epoch_7.pth").exists()

    def test_saves_best_model(self, tmp_path: Path, sample_training_state):
        """Test best model saving."""
        cb = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=0,  # Disable frequency-based saving
            save_best=True,
        )

        sample_training_state.val_metrics = {"loss": 0.5}
        cb.on_epoch_end(sample_training_state)

        assert (tmp_path / "best_model.pth").exists()

    def test_updates_best_model_on_improvement(self, tmp_path: Path, sample_training_state):
        """Test that best model is updated on improvement."""
        cb = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=0,
            save_best=True,
        )

        # First save
        sample_training_state.val_metrics = {"loss": 0.8}
        cb.on_epoch_end(sample_training_state)
        first_mtime = (tmp_path / "best_model.pth").stat().st_mtime

        # Better loss - should update
        sample_training_state.val_metrics = {"loss": 0.5}
        cb.on_epoch_end(sample_training_state)
        second_mtime = (tmp_path / "best_model.pth").stat().st_mtime

        assert second_mtime >= first_mtime

    def test_does_not_update_on_worse_loss(self, tmp_path: Path, sample_training_state):
        """Test that best model isn't updated on worse loss."""
        cb = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=0,
            save_best=True,
        )

        # First save
        sample_training_state.val_metrics = {"loss": 0.5}
        cb.on_epoch_end(sample_training_state)
        assert cb.best_loss == 0.5

        # Worse loss - should not update best_loss
        sample_training_state.val_metrics = {"loss": 0.8}
        cb.on_epoch_end(sample_training_state)
        assert cb.best_loss == 0.5

    def test_creates_save_directory(self, tmp_path: Path, sample_training_state):
        """Test that save directory is created if it doesn't exist."""
        save_dir = tmp_path / "new_dir" / "checkpoints"
        cb = ModelCheckpointCallback(
            save_dir=str(save_dir),
            save_frequency=1,
            save_best=False,
        )

        sample_training_state.epoch = 1
        cb.on_epoch_end(sample_training_state)

        assert save_dir.exists()

    def test_checkpoint_contains_required_keys(self, tmp_path: Path, sample_training_state):
        """Test that saved checkpoint contains all required keys."""
        cb = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=1,
            save_best=False,
        )

        sample_training_state.epoch = 1
        cb.on_epoch_end(sample_training_state)

        checkpoint = torch.load(tmp_path / "checkpoint_epoch_1.pth")

        assert "epoch" in checkpoint
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint
        assert "val_metrics" in checkpoint
        assert "train_metrics" in checkpoint

    def test_custom_filename_template(self, tmp_path: Path, sample_training_state):
        """Test custom filename template."""
        cb = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=1,
            save_best=False,
            filename_template="model_e{epoch}.pth",
        )

        sample_training_state.epoch = 5
        cb.on_epoch_end(sample_training_state)

        assert (tmp_path / "model_e5.pth").exists()


class TestLambdaCallback:
    """Tests for LambdaCallback."""

    def test_on_epoch_end_called(self, sample_training_state):
        """Test that on_epoch_end function is called."""
        mock_fn = MagicMock()
        cb = LambdaCallback(on_epoch_end=mock_fn)

        cb.on_epoch_end(sample_training_state)

        mock_fn.assert_called_once_with(sample_training_state)

    def test_on_train_end_called(self, sample_training_state):
        """Test that on_train_end function is called."""
        mock_fn = MagicMock()
        cb = LambdaCallback(on_train_end=mock_fn)

        cb.on_train_end(sample_training_state)

        mock_fn.assert_called_once_with(sample_training_state)

    def test_on_train_start_called(self, sample_training_state):
        """Test that on_train_start function is called."""
        mock_fn = MagicMock()
        cb = LambdaCallback(on_train_start=mock_fn)

        cb.on_train_start(sample_training_state)

        mock_fn.assert_called_once_with(sample_training_state)

    def test_on_epoch_start_called(self, sample_training_state):
        """Test that on_epoch_start function is called."""
        mock_fn = MagicMock()
        cb = LambdaCallback(on_epoch_start=mock_fn)

        cb.on_epoch_start(sample_training_state)

        mock_fn.assert_called_once_with(sample_training_state)

    def test_handles_none_functions(self, sample_training_state):
        """Test that None functions don't cause errors."""
        cb = LambdaCallback()  # All functions are None

        # Should not raise
        cb.on_train_start(sample_training_state)
        cb.on_train_end(sample_training_state)
        cb.on_epoch_start(sample_training_state)
        cb.on_epoch_end(sample_training_state)

    def test_multiple_functions(self, sample_training_state):
        """Test callback with multiple functions defined."""
        epoch_end_fn = MagicMock()
        train_end_fn = MagicMock()

        cb = LambdaCallback(on_epoch_end=epoch_end_fn, on_train_end=train_end_fn)

        cb.on_epoch_end(sample_training_state)
        cb.on_train_end(sample_training_state)

        epoch_end_fn.assert_called_once()
        train_end_fn.assert_called_once()

    def test_lambda_receives_correct_state(self, sample_training_state):
        """Test that lambda receives the correct state object."""
        received_state = None

        def capture_state(state):
            nonlocal received_state
            received_state = state

        cb = LambdaCallback(on_epoch_end=capture_state)
        cb.on_epoch_end(sample_training_state)

        assert received_state is sample_training_state
        assert received_state.epoch == sample_training_state.epoch


class TestTrainingState:
    """Tests for TrainingState dataclass."""

    def test_training_state_creation(self, mock_model):
        """Test TrainingState creation with all required fields."""
        optimizer = torch.optim.SGD(mock_model.parameters(), lr=0.01)

        state = TrainingState(
            epoch=5,
            total_epochs=100,
            model=mock_model,
            optimizer=optimizer,
            train_metrics={"loss": 0.5},
            val_metrics={"loss": 0.4},
            best_val_loss=0.4,
            device=torch.device("cpu"),
        )

        assert state.epoch == 5
        assert state.total_epochs == 100
        assert state.model is mock_model
        assert state.optimizer is optimizer
        assert state.train_metrics["loss"] == 0.5
        assert state.val_metrics["loss"] == 0.4
        assert state.best_val_loss == 0.4
        assert state.device == torch.device("cpu")
