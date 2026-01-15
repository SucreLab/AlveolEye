"""Tests for training API.

Tests cover:
- _build_config_from_kwargs function
- _get_device function
- _apply_image_selection function
- train() validation
- TrainingResult class
- Integration tests (marked slow)
"""

from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest
import torch
from torch.utils.data import Subset

from alveoleye.lungcv.mrcnn.api import (
    train,
    TrainingResult,
    _build_config_from_kwargs,
    _get_device,
    _apply_image_selection,
)
from alveoleye.lungcv.mrcnn.config import (
    TrainingConfig,
    DataConfig,
    OptimizerConfig,
    AugmentationConfig,
    ImageSelectionConfig,
)


class TestBuildConfigFromKwargs:
    """Tests for _build_config_from_kwargs function."""

    def test_minimal_kwargs(self):
        """Test config creation with minimal kwargs."""
        config = _build_config_from_kwargs(
            dataset_path="/path/to/data",
            epochs=50,
        )
        assert config.data.dataset_path == "/path/to/data"
        assert config.epochs == 50

    def test_default_values_used(self):
        """Test that default values are used when kwargs not provided."""
        config = _build_config_from_kwargs()
        assert config.epochs == 100
        assert config.data.batch_size == 10
        assert config.optimizer.name == "sgd"
        assert config.scheduler.name == "step"

    def test_optimizer_kwargs(self):
        """Test optimizer configuration from kwargs."""
        config = _build_config_from_kwargs(
            optimizer="adamw",
            lr=0.001,
            weight_decay=0.01,
        )
        assert config.optimizer.name == "adamw"
        assert config.optimizer.lr == 0.001
        assert config.optimizer.weight_decay == 0.01

    def test_scheduler_kwargs(self):
        """Test scheduler configuration from kwargs."""
        config = _build_config_from_kwargs(
            scheduler="cosine",
            warmup_epochs=5,
        )
        assert config.scheduler.name == "cosine"
        assert config.scheduler.warmup_epochs == 5

    def test_data_kwargs(self):
        """Test data configuration from kwargs."""
        config = _build_config_from_kwargs(
            batch_size=16,
            num_workers=4,
            n_repeat_images=2,
            img_extension=".jpg",
        )
        assert config.data.batch_size == 16
        assert config.data.num_workers == 4
        assert config.data.n_repeat_images == 2
        assert config.data.img_extension == ".jpg"

    def test_image_selection_n_images(self):
        """Test image selection with n_images."""
        config = _build_config_from_kwargs(
            n_images=50,
            seed=42,
        )
        assert config.data.image_selection is not None
        assert config.data.image_selection.n_random == 50
        assert config.data.image_selection.seed == 42

    def test_image_selection_range(self):
        """Test image selection with image_range."""
        config = _build_config_from_kwargs(
            image_range=(10, 30),
            seed=123,
        )
        assert config.data.image_selection is not None
        assert config.data.image_selection.index_range == (10, 30)
        assert config.data.image_selection.seed == 123

    def test_checkpoint_kwargs(self):
        """Test checkpoint configuration from kwargs."""
        config = _build_config_from_kwargs(
            save_dir="checkpoints",
            save_frequency=10,
            save_best=False,
        )
        assert config.checkpoint.save_dir == "checkpoints"
        assert config.checkpoint.save_frequency == 10
        assert config.checkpoint.save_best is False

    def test_logging_kwargs(self):
        """Test logging configuration from kwargs."""
        config = _build_config_from_kwargs(
            use_tensorboard=False,
            log_dir="logs",
            print_freq=5,
        )
        assert config.logging.use_tensorboard is False
        assert config.logging.log_dir == "logs"
        assert config.logging.print_freq == 5

    def test_training_kwargs(self):
        """Test general training kwargs."""
        config = _build_config_from_kwargs(
            num_classes=5,
            device="cuda",
            seed=42,
            mixed_precision=True,
            gradient_clip_val=1.0,
        )
        assert config.num_classes == 5
        assert config.device == "cuda"
        assert config.seed == 42
        assert config.mixed_precision is True
        assert config.gradient_clip_val == 1.0

    def test_augmentation_kwarg(self):
        """Test augmentation config kwarg."""
        aug_config = AugmentationConfig.none()
        config = _build_config_from_kwargs(augmentation=aug_config)
        assert config.augmentation.enabled is False


class TestGetDevice:
    """Tests for _get_device function."""

    def test_explicit_cpu(self):
        """Test explicit CPU device."""
        device = _get_device("cpu")
        assert device == torch.device("cpu")

    def test_explicit_cuda(self):
        """Test explicit CUDA device (may fall back if unavailable)."""
        device = _get_device("cuda")
        assert device == torch.device("cuda")

    def test_auto_device(self):
        """Test auto device selection."""
        device = _get_device("auto")
        assert device.type in ("cpu", "cuda", "mps")

    def test_mps_device(self):
        """Test MPS device."""
        device = _get_device("mps")
        assert device == torch.device("mps")


class TestApplyImageSelection:
    """Tests for _apply_image_selection function."""

    def test_no_selection_returns_original(self):
        """Test that None config returns original dataset."""
        dataset = list(range(100))
        result = _apply_image_selection(dataset, None)
        assert result is dataset

    def test_random_selection_returns_subset(self):
        """Test random selection returns a Subset."""
        dataset = MagicMock()
        dataset.__len__ = MagicMock(return_value=100)
        config = ImageSelectionConfig(n_random=10, seed=42)

        result = _apply_image_selection(dataset, config)

        assert isinstance(result, Subset)

    def test_random_selection_correct_size(self):
        """Test random selection returns correct number of samples."""
        # Use a real list as dataset
        dataset = list(range(100))
        config = ImageSelectionConfig(n_random=25, seed=42)

        result = _apply_image_selection(dataset, config)

        assert len(result) == 25

    def test_random_selection_deterministic_with_seed(self):
        """Test that same seed produces same selection."""
        dataset = list(range(100))
        config = ImageSelectionConfig(n_random=10, seed=42)

        result1 = _apply_image_selection(dataset, config)
        result2 = _apply_image_selection(dataset, config)

        assert list(result1.indices) == list(result2.indices)

    def test_index_range_selection(self):
        """Test index range selection."""
        dataset = list(range(100))
        config = ImageSelectionConfig(index_range=(10, 20))

        result = _apply_image_selection(dataset, config)

        assert isinstance(result, Subset)
        assert len(result) == 11  # 10 to 20 inclusive

    def test_index_range_clamps_to_dataset_size(self):
        """Test that index range is clamped to dataset size."""
        dataset = list(range(50))
        config = ImageSelectionConfig(index_range=(40, 100))

        result = _apply_image_selection(dataset, config)

        # Should be clamped to 40-49
        assert len(result) == 10


class TestTrainValidation:
    """Tests for train() function validation."""

    def test_mutually_exclusive_image_selection_raises(self):
        """Test that n_images and image_range raise ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            train(n_images=50, image_range=(0, 10))


class TestTrainingResult:
    """Tests for TrainingResult class."""

    def test_training_result_creation(self, mock_model):
        """Test TrainingResult creation."""
        result = TrainingResult(
            model=mock_model,
            history={"train_loss": [1.0, 0.5]},
            best_val_loss=0.5,
            final_epoch=10,
        )
        assert result.model is mock_model
        assert result.history["train_loss"] == [1.0, 0.5]
        assert result.best_val_loss == 0.5
        assert result.final_epoch == 10

    def test_training_result_defaults(self, mock_model):
        """Test TrainingResult default values."""
        result = TrainingResult(model=mock_model)
        assert result.history == {}
        assert result.best_val_loss == float("inf")
        assert result.config is None
        assert result.final_epoch == 0

    def test_save_model(self, tmp_path: Path, mock_model):
        """Test saving model to file."""
        result = TrainingResult(model=mock_model)
        path = tmp_path / "model.pth"

        result.save(path)

        assert path.exists()
        # Verify we can load it
        loaded = torch.load(path)
        assert loaded is not None

    def test_save_checkpoint(self, tmp_path: Path, mock_model):
        """Test saving full checkpoint."""
        config = TrainingConfig(epochs=100)
        result = TrainingResult(
            model=mock_model,
            history={"train_loss": [1.0, 0.8, 0.6]},
            best_val_loss=0.6,
            final_epoch=50,
            config=config,
        )
        path = tmp_path / "checkpoint.pth"

        result.save_checkpoint(path)

        assert path.exists()

        # Verify checkpoint contents
        checkpoint = torch.load(path)
        assert "model_state_dict" in checkpoint
        assert "history" in checkpoint
        assert "best_val_loss" in checkpoint
        assert "final_epoch" in checkpoint
        assert "config" in checkpoint

        assert checkpoint["best_val_loss"] == 0.6
        assert checkpoint["final_epoch"] == 50
        assert checkpoint["history"]["train_loss"] == [1.0, 0.8, 0.6]

    def test_save_checkpoint_without_config(self, tmp_path: Path, mock_model):
        """Test saving checkpoint when config is None."""
        result = TrainingResult(
            model=mock_model,
            history={"train_loss": [1.0]},
            best_val_loss=1.0,
            final_epoch=0,
            config=None,
        )
        path = tmp_path / "checkpoint.pth"

        result.save_checkpoint(path)

        checkpoint = torch.load(path)
        assert "config" not in checkpoint


class TestTrainFunction:
    """Tests for train() function behavior."""

    def test_train_returns_training_result(self, mock_dataset):
        """Test that train returns a TrainingResult (mocked)."""
        # This is a unit test that mocks the heavy operations
        with patch("alveoleye.lungcv.mrcnn.api._run_training") as mock_run:
            mock_result = MagicMock(spec=TrainingResult)
            mock_run.return_value = mock_result

            result = train(
                dataset_path=str(mock_dataset),
                epochs=1,
            )

            assert result is mock_result
            mock_run.assert_called_once()

    def test_train_with_config_object(self, mock_dataset):
        """Test train with TrainingConfig object."""
        config = TrainingConfig(
            epochs=1,
            data=DataConfig(dataset_path=str(mock_dataset)),
        )

        with patch("alveoleye.lungcv.mrcnn.api._run_training") as mock_run:
            mock_result = MagicMock(spec=TrainingResult)
            mock_run.return_value = mock_result

            result = train(config=config)

            mock_run.assert_called_once()
            # Config should be passed to _run_training
            call_args = mock_run.call_args
            assert call_args[0][0] is config

    def test_train_creates_callbacks(self, mock_dataset):
        """Test that train creates appropriate callbacks."""
        with patch("alveoleye.lungcv.mrcnn.api._run_training") as mock_run:
            mock_result = MagicMock(spec=TrainingResult)
            mock_run.return_value = mock_result

            train(
                dataset_path=str(mock_dataset),
                epochs=1,
                save_frequency=10,
                save_best=True,
            )

            # Should have created a callback list
            call_args = mock_run.call_args
            callback_list = call_args[0][1]
            assert len(callback_list.callbacks) > 0

    def test_train_with_lambda_callbacks(self, mock_dataset):
        """Test train with lambda callbacks."""
        epoch_end_called = []

        def on_epoch_end(epoch, metrics):
            epoch_end_called.append(epoch)

        with patch("alveoleye.lungcv.mrcnn.api._run_training") as mock_run:
            mock_result = MagicMock(spec=TrainingResult)
            mock_run.return_value = mock_result

            train(
                dataset_path=str(mock_dataset),
                epochs=1,
                on_epoch_end=on_epoch_end,
            )

            # Lambda callback should be added
            call_args = mock_run.call_args
            callback_list = call_args[0][1]
            assert any(
                hasattr(cb, "_on_epoch_end")
                for cb in callback_list.callbacks
            )


@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationTraining:
    """Integration tests that run actual training with real dataset.

    These tests are slow and require downloading a dataset from Google Drive.
    Skip with: pytest -m "not slow"
    """

    def test_train_with_downloaded_dataset(self, tmp_path: Path):
        """Test training with dataset downloaded from Google Drive."""
        # Skip if gdown is not installed
        pytest.importorskip("gdown")

        from alveoleye.paper_scripts._utils import download_training_dataset

        # Download the dataset
        try:
            dataset_path = download_training_dataset(
                output_dir=str(tmp_path),
                quiet=False,
            )
        except RuntimeError as e:
            pytest.skip(f"Could not download dataset: {e}")

        # Verify dataset structure exists
        dataset = Path(dataset_path)
        assert (dataset / "images" / "train").exists(), "Missing images/train"
        assert (dataset / "classes.json").exists(), "Missing classes.json"

        # Count available images to ensure we don't request more than available
        train_images = list((dataset / "images" / "train").glob("*.png"))
        train_images.extend((dataset / "images" / "train").glob("*.jpg"))
        n_images = min(5, len(train_images))

        if n_images == 0:
            pytest.skip("Downloaded dataset has no training images")

        # Run a short training session
        result = train(
            dataset_path=dataset_path,
            epochs=1,
            n_images=n_images,
            device="auto",
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=1,
        )

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert result.final_epoch == 1
        assert result.best_val_loss < float("inf")
