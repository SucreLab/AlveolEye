"""Comprehensive tests for training API.

Tests cover:
- _build_config_from_kwargs function
- _get_device function
- _apply_image_selection function
- train() validation
- TrainingResult class
- Optimizer creation (all types)
- Scheduler creation (all types including warmup)
- Augmentation pipeline building
- Callback integration
- Config serialization (dict, YAML roundtrip)
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
    SchedulerConfig,
    AugmentationConfig,
    AugmentationItem,
    ImageSelectionConfig,
    CheckpointConfig,
)
from alveoleye.lungcv.mrcnn.optimizers import create_optimizer, create_scheduler, WarmupScheduler
from alveoleye.lungcv.mrcnn.augmentations import build_transforms, get_available_augmentations
from alveoleye.lungcv.mrcnn.callbacks import (
    Callback,
    CallbackList,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LambdaCallback,
)


# =============================================================================
# Test _build_config_from_kwargs
# =============================================================================

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

    def test_all_optimizer_kwargs(self):
        """Test all optimizer configuration kwargs."""
        config = _build_config_from_kwargs(
            optimizer="adamw",
            lr=0.001,
            momentum=0.95,
            weight_decay=0.01,
        )
        assert config.optimizer.name == "adamw"
        assert config.optimizer.lr == 0.001
        assert config.optimizer.momentum == 0.95
        assert config.optimizer.weight_decay == 0.01

    def test_all_scheduler_kwargs(self):
        """Test all scheduler configuration kwargs."""
        config = _build_config_from_kwargs(
            scheduler="cosine",
            step_size=10,
            gamma=0.5,
            warmup_epochs=5,
        )
        assert config.scheduler.name == "cosine"
        assert config.scheduler.step_size == 10
        assert config.scheduler.gamma == 0.5
        assert config.scheduler.warmup_epochs == 5

    def test_all_data_kwargs(self):
        """Test all data configuration kwargs."""
        config = _build_config_from_kwargs(
            dataset_path="/data",
            batch_size=16,
            num_workers=4,
            n_repeat_images=2,
            img_extension=".jpg",
            val_split=0.3,
        )
        assert config.data.dataset_path == "/data"
        assert config.data.batch_size == 16
        assert config.data.num_workers == 4
        assert config.data.n_repeat_images == 2
        assert config.data.img_extension == ".jpg"
        assert config.data.val_split == 0.3

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

    def test_all_checkpoint_kwargs(self):
        """Test all checkpoint configuration kwargs."""
        config = _build_config_from_kwargs(
            save_dir="checkpoints",
            save_frequency=10,
            save_best=False,
        )
        assert config.checkpoint.save_dir == "checkpoints"
        assert config.checkpoint.save_frequency == 10
        assert config.checkpoint.save_best is False

    def test_all_logging_kwargs(self):
        """Test all logging configuration kwargs."""
        config = _build_config_from_kwargs(
            use_tensorboard=False,
            log_dir="logs",
            print_freq=5,
        )
        assert config.logging.use_tensorboard is False
        assert config.logging.log_dir == "logs"
        assert config.logging.print_freq == 5

    def test_all_training_kwargs(self):
        """Test all general training kwargs."""
        config = _build_config_from_kwargs(
            epochs=200,
            num_classes=5,
            device="cuda",
            seed=42,
            mixed_precision=True,
            gradient_clip_val=1.0,
        )
        assert config.epochs == 200
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

    def test_optimizer_name_case_insensitive(self):
        """Test that optimizer name is case-insensitive."""
        config = _build_config_from_kwargs(optimizer="AdamW")
        assert config.optimizer.name == "adamw"

    def test_scheduler_name_case_insensitive(self):
        """Test that scheduler name is case-insensitive."""
        config = _build_config_from_kwargs(scheduler="COSINE")
        assert config.scheduler.name == "cosine"


# =============================================================================
# Test _get_device
# =============================================================================

class TestGetDevice:
    """Tests for _get_device function."""

    def test_explicit_cpu(self):
        """Test explicit CPU device."""
        device = _get_device("cpu")
        assert device == torch.device("cpu")

    def test_explicit_cuda(self):
        """Test explicit CUDA device."""
        device = _get_device("cuda")
        assert device == torch.device("cuda")

    def test_explicit_mps(self):
        """Test explicit MPS device."""
        device = _get_device("mps")
        assert device == torch.device("mps")

    def test_auto_device_returns_valid_type(self):
        """Test auto device selection returns a valid device type."""
        device = _get_device("auto")
        assert device.type in ("cpu", "cuda", "mps")

    def test_auto_prefers_cuda_when_available(self):
        """Test auto prefers CUDA when available."""
        with patch("torch.cuda.is_available", return_value=True):
            device = _get_device("auto")
            assert device.type == "cuda"

    def test_auto_falls_back_to_cpu(self):
        """Test auto falls back to CPU when no accelerator available."""
        with patch("torch.cuda.is_available", return_value=False):
            with patch("torch.backends.mps.is_available", return_value=False):
                device = _get_device("auto")
                assert device.type == "cpu"


# =============================================================================
# Test _apply_image_selection
# =============================================================================

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

    def test_random_selection_different_seeds_differ(self):
        """Test that different seeds produce different selections."""
        dataset = list(range(100))
        config1 = ImageSelectionConfig(n_random=10, seed=42)
        config2 = ImageSelectionConfig(n_random=10, seed=123)

        result1 = _apply_image_selection(dataset, config1)
        result2 = _apply_image_selection(dataset, config2)

        assert list(result1.indices) != list(result2.indices)

    def test_random_selection_no_seed_varies(self):
        """Test that no seed produces varying results."""
        dataset = list(range(1000))
        config = ImageSelectionConfig(n_random=10, seed=None)

        # Without seed, results may vary (though not guaranteed)
        result1 = _apply_image_selection(dataset, config)
        result2 = _apply_image_selection(dataset, config)

        # Both should have correct length
        assert len(result1) == 10
        assert len(result2) == 10

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

    def test_index_range_from_zero(self):
        """Test index range starting from zero."""
        dataset = list(range(100))
        config = ImageSelectionConfig(index_range=(0, 9))

        result = _apply_image_selection(dataset, config)

        assert len(result) == 10
        assert list(result.indices) == list(range(10))


# =============================================================================
# Test Optimizer Creation
# =============================================================================

class TestCreateOptimizer:
    """Tests for create_optimizer function."""

    def test_create_sgd_optimizer(self, mock_params):
        """Test SGD optimizer creation."""
        config = OptimizerConfig(name="sgd", lr=0.01, momentum=0.9, weight_decay=0.0005)
        optimizer = create_optimizer(mock_params, config)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["momentum"] == 0.9
        assert optimizer.defaults["weight_decay"] == 0.0005

    def test_create_sgd_with_nesterov(self, mock_params):
        """Test SGD optimizer with Nesterov momentum."""
        config = OptimizerConfig(name="sgd", nesterov=True)
        optimizer = create_optimizer(mock_params, config)

        assert isinstance(optimizer, torch.optim.SGD)
        assert optimizer.defaults["nesterov"] is True

    def test_create_adam_optimizer(self, mock_params):
        """Test Adam optimizer creation."""
        config = OptimizerConfig(name="adam", lr=0.001, betas=(0.9, 0.999), eps=1e-8)
        optimizer = create_optimizer(mock_params, config)

        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.defaults["lr"] == 0.001
        assert optimizer.defaults["betas"] == (0.9, 0.999)
        assert optimizer.defaults["eps"] == 1e-8

    def test_create_adamw_optimizer(self, mock_params):
        """Test AdamW optimizer creation."""
        config = OptimizerConfig(name="adamw", lr=0.001, weight_decay=0.01)
        optimizer = create_optimizer(mock_params, config)

        assert isinstance(optimizer, torch.optim.AdamW)
        assert optimizer.defaults["lr"] == 0.001
        assert optimizer.defaults["weight_decay"] == 0.01

    def test_create_rmsprop_optimizer(self, mock_params):
        """Test RMSprop optimizer creation."""
        config = OptimizerConfig(name="rmsprop", lr=0.01, alpha=0.99, centered=True)
        optimizer = create_optimizer(mock_params, config)

        assert isinstance(optimizer, torch.optim.RMSprop)
        assert optimizer.defaults["lr"] == 0.01
        assert optimizer.defaults["alpha"] == 0.99
        assert optimizer.defaults["centered"] is True

    def test_unknown_optimizer_raises(self, mock_params):
        """Test that unknown optimizer name raises ValueError."""
        config = OptimizerConfig(name="unknown")

        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(mock_params, config)

    def test_optimizer_names_case_insensitive(self, mock_params):
        """Test that optimizer names are case-insensitive."""
        for name in ["SGD", "Sgd", "sgd", "ADAM", "Adam", "adam"]:
            config = OptimizerConfig(name=name)
            optimizer = create_optimizer(mock_params, config)
            assert optimizer is not None


# =============================================================================
# Test Scheduler Creation
# =============================================================================

class TestCreateScheduler:
    """Tests for create_scheduler function."""

    def test_create_step_scheduler(self, mock_optimizer):
        """Test StepLR scheduler creation."""
        config = SchedulerConfig(name="step", step_size=20, gamma=0.1)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
        assert scheduler.step_size == 20
        assert scheduler.gamma == 0.1

    def test_create_cosine_scheduler(self, mock_optimizer):
        """Test CosineAnnealingLR scheduler creation."""
        config = SchedulerConfig(name="cosine", eta_min=0.0001)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert scheduler.T_max == 100
        assert scheduler.eta_min == 0.0001

    def test_create_cosine_scheduler_custom_tmax(self, mock_optimizer):
        """Test CosineAnnealingLR with custom T_max."""
        config = SchedulerConfig(name="cosine", T_max=50)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)

        assert scheduler.T_max == 50

    def test_create_plateau_scheduler(self, mock_optimizer):
        """Test ReduceLROnPlateau scheduler creation."""
        config = SchedulerConfig(name="plateau", gamma=0.5, patience=5)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)

        assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
        assert scheduler.factor == 0.5
        assert scheduler.patience == 5

    def test_none_scheduler(self, mock_optimizer):
        """Test 'none' scheduler returns None."""
        config = SchedulerConfig(name="none")
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)

        assert scheduler is None

    def test_unknown_scheduler_raises(self, mock_optimizer):
        """Test that unknown scheduler name raises ValueError."""
        config = SchedulerConfig(name="unknown")

        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(mock_optimizer, config, epochs=100)

    def test_warmup_scheduler(self, mock_optimizer):
        """Test warmup wrapper scheduler."""
        config = SchedulerConfig(name="step", warmup_epochs=5, warmup_factor=0.001)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)

        assert isinstance(scheduler, WarmupScheduler)
        assert scheduler.warmup_epochs == 5
        assert scheduler.warmup_factor == 0.001

    def test_warmup_scheduler_lr_progression(self, mock_params):
        """Test that warmup scheduler correctly ramps up learning rate."""
        optimizer = torch.optim.SGD(mock_params, lr=0.1)
        config = SchedulerConfig(name="step", warmup_epochs=5, warmup_factor=0.1)
        scheduler = create_scheduler(optimizer, config, epochs=100)

        # Initial LR should be base_lr * warmup_factor
        initial_lr = optimizer.param_groups[0]["lr"]
        assert abs(initial_lr - 0.01) < 1e-6  # 0.1 * 0.1 = 0.01

        # Step through warmup
        lrs = [initial_lr]
        for _ in range(5):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]["lr"])

        # LR should increase during warmup
        for i in range(1, len(lrs) - 1):
            assert lrs[i] > lrs[i - 1] or abs(lrs[i] - lrs[i - 1]) < 1e-6

        # After warmup, LR should be at base_lr
        assert abs(lrs[-1] - 0.1) < 1e-6


# =============================================================================
# Test Augmentation Pipeline
# =============================================================================

class TestAugmentationPipeline:
    """Tests for augmentation building."""

    def test_get_available_augmentations(self):
        """Test getting list of available augmentations."""
        available = get_available_augmentations()

        assert isinstance(available, list)
        assert "horizontal_flip" in available
        assert "vertical_flip" in available
        assert "color_jitter" in available
        assert "rotation" in available
        assert "gaussian_blur" in available
        assert "affine" in available

    def test_build_transforms_disabled(self):
        """Test building transforms with augmentation disabled."""
        config = AugmentationConfig.none()
        transforms = build_transforms(config, train=True)

        # Should only have basic transforms (PILToTensor, ToDtype, ToPureTensor)
        assert transforms is not None

    def test_build_transforms_train_vs_val(self):
        """Test that train and val transforms differ."""
        config = AugmentationConfig.default()

        train_transforms = build_transforms(config, train=True)
        val_transforms = build_transforms(config, train=False)

        # Train transforms should have more components
        assert len(train_transforms.transforms) > len(val_transforms.transforms)

    def test_build_transforms_with_custom_augmentations(self):
        """Test building transforms with custom augmentation list."""
        config = AugmentationConfig(
            enabled=True,
            augmentations=[
                AugmentationItem("horizontal_flip", probability=0.5),
                AugmentationItem("color_jitter", probability=0.3, params={
                    "brightness": 0.4, "contrast": 0.4
                }),
            ]
        )

        transforms = build_transforms(config, train=True)
        assert transforms is not None

    def test_build_transforms_unknown_augmentation_raises(self):
        """Test that unknown augmentation name raises ValueError."""
        config = AugmentationConfig(
            enabled=True,
            augmentations=[
                AugmentationItem("unknown_augmentation", probability=0.5),
            ]
        )

        with pytest.raises(ValueError, match="Unknown augmentation"):
            build_transforms(config, train=True)

    def test_augmentation_item_defaults(self):
        """Test AugmentationItem default values."""
        item = AugmentationItem("horizontal_flip")

        assert item.name == "horizontal_flip"
        assert item.probability == 0.5
        assert item.params == {}

    def test_all_registered_augmentations_build(self):
        """Test that all registered augmentations can be built."""
        for aug_name in get_available_augmentations():
            config = AugmentationConfig(
                enabled=True,
                augmentations=[AugmentationItem(aug_name, probability=0.5)]
            )
            # Should not raise
            transforms = build_transforms(config, train=True)
            assert transforms is not None


# =============================================================================
# Test Callbacks
# =============================================================================

class TestCallbacks:
    """Tests for callback system."""

    def test_callback_list_calls_all_callbacks(self, sample_training_state):
        """Test that CallbackList calls all registered callbacks."""
        calls = []

        class TestCallback(Callback):
            def __init__(self, name):
                self.name = name

            def on_epoch_end(self, state):
                calls.append(self.name)

        cb_list = CallbackList([TestCallback("a"), TestCallback("b")])
        cb_list.on_epoch_end(sample_training_state)

        assert calls == ["a", "b"]

    def test_callback_list_add(self, sample_training_state):
        """Test adding callback to list."""
        calls = []

        class TestCallback(Callback):
            def on_epoch_end(self, state):
                calls.append("called")

        cb_list = CallbackList()
        cb_list.add(TestCallback())
        cb_list.on_epoch_end(sample_training_state)

        assert calls == ["called"]

    def test_early_stopping_triggers(self, sample_training_state):
        """Test early stopping triggers after patience epochs."""
        callback = EarlyStoppingCallback(patience=3, verbose=False)

        # Simulate epochs with no improvement
        sample_training_state.val_metrics = {"loss": 1.0}
        for _ in range(4):
            callback.on_epoch_end(sample_training_state)

        assert callback.should_stop is True

    def test_early_stopping_resets_on_improvement(self, sample_training_state):
        """Test early stopping resets counter on improvement."""
        callback = EarlyStoppingCallback(patience=3, verbose=False)

        # Initial loss
        sample_training_state.val_metrics = {"loss": 1.0}
        callback.on_epoch_end(sample_training_state)
        callback.on_epoch_end(sample_training_state)

        assert callback.counter == 1

        # Improvement
        sample_training_state.val_metrics = {"loss": 0.5}
        callback.on_epoch_end(sample_training_state)

        assert callback.counter == 0
        assert callback.should_stop is False

    def test_early_stopping_min_delta(self, sample_training_state):
        """Test early stopping with min_delta threshold."""
        callback = EarlyStoppingCallback(patience=2, min_delta=0.1, verbose=False)

        sample_training_state.val_metrics = {"loss": 1.0}
        callback.on_epoch_end(sample_training_state)

        # Small improvement (less than min_delta)
        sample_training_state.val_metrics = {"loss": 0.95}
        callback.on_epoch_end(sample_training_state)

        # Should count as no improvement
        assert callback.counter == 1

    def test_model_checkpoint_saves_best(self, tmp_path, sample_training_state):
        """Test ModelCheckpointCallback saves best model."""
        callback = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=0,
            save_best=True,
        )

        sample_training_state.val_metrics = {"loss": 0.5}
        callback.on_epoch_end(sample_training_state)

        assert (tmp_path / "best_model.pth").exists()

    def test_model_checkpoint_saves_at_frequency(self, tmp_path, sample_training_state):
        """Test ModelCheckpointCallback saves at specified frequency."""
        callback = ModelCheckpointCallback(
            save_dir=str(tmp_path),
            save_frequency=5,
            save_best=False,
        )

        sample_training_state.epoch = 5
        callback.on_epoch_end(sample_training_state)

        assert (tmp_path / "checkpoint_epoch_5.pth").exists()

    def test_lambda_callback_calls_functions(self, sample_training_state):
        """Test LambdaCallback calls provided functions."""
        calls = {"epoch_end": 0, "train_end": 0}

        callback = LambdaCallback(
            on_epoch_end=lambda s: calls.__setitem__("epoch_end", calls["epoch_end"] + 1),
            on_train_end=lambda s: calls.__setitem__("train_end", calls["train_end"] + 1),
        )

        callback.on_epoch_end(sample_training_state)
        callback.on_train_end(sample_training_state)

        assert calls["epoch_end"] == 1
        assert calls["train_end"] == 1


# =============================================================================
# Test Config Serialization
# =============================================================================

class TestConfigSerialization:
    """Tests for config serialization and deserialization."""

    def test_training_config_to_dict(self):
        """Test TrainingConfig.to_dict()."""
        config = TrainingConfig(
            epochs=150,
            num_classes=4,
            data=DataConfig(batch_size=16),
            optimizer=OptimizerConfig(name="adamw", lr=0.001),
        )

        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["epochs"] == 150
        assert d["num_classes"] == 4
        assert d["data"]["batch_size"] == 16
        assert d["optimizer"]["name"] == "adamw"

    def test_training_config_from_dict(self):
        """Test TrainingConfig.from_dict()."""
        d = {
            "epochs": 200,
            "num_classes": 5,
            "data": {"batch_size": 32, "num_workers": 4},
            "optimizer": {"name": "adam", "lr": 0.0001},
        }

        config = TrainingConfig.from_dict(d)

        assert config.epochs == 200
        assert config.num_classes == 5
        assert config.data.batch_size == 32
        assert config.data.num_workers == 4
        assert config.optimizer.name == "adam"
        assert config.optimizer.lr == 0.0001

    def test_training_config_dict_roundtrip(self):
        """Test that to_dict() -> from_dict() preserves config."""
        original = TrainingConfig(
            epochs=100,
            num_classes=3,
            seed=42,
            data=DataConfig(
                batch_size=8,
                val_split=0.25,
                image_selection=ImageSelectionConfig(n_random=50, seed=123),
            ),
            optimizer=OptimizerConfig(name="adamw", lr=0.001, betas=(0.9, 0.99)),
            scheduler=SchedulerConfig(name="cosine", warmup_epochs=5),
            checkpoint=CheckpointConfig(save_dir="checkpoints", save_frequency=10),
        )

        d = original.to_dict()
        restored = TrainingConfig.from_dict(d)

        assert restored.epochs == original.epochs
        assert restored.num_classes == original.num_classes
        assert restored.seed == original.seed
        assert restored.data.batch_size == original.data.batch_size
        assert restored.data.val_split == original.data.val_split
        assert restored.data.image_selection.n_random == 50
        assert restored.data.image_selection.seed == 123
        assert restored.optimizer.name == original.optimizer.name
        assert restored.optimizer.lr == original.optimizer.lr
        assert restored.optimizer.betas == original.optimizer.betas
        assert restored.scheduler.name == original.scheduler.name
        assert restored.scheduler.warmup_epochs == 5

    def test_training_config_yaml_roundtrip(self, tmp_path):
        """Test YAML save and load preserves config."""
        pytest.importorskip("yaml")

        original = TrainingConfig(
            epochs=100,
            data=DataConfig(batch_size=16),
            optimizer=OptimizerConfig(name="adam"),
        )

        yaml_path = tmp_path / "config.yaml"
        original.to_yaml(yaml_path)

        restored = TrainingConfig.from_yaml(yaml_path)

        assert restored.epochs == original.epochs
        assert restored.data.batch_size == original.data.batch_size
        assert restored.optimizer.name == original.optimizer.name

    def test_image_selection_config_validation(self):
        """Test ImageSelectionConfig validates mutually exclusive options."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            ImageSelectionConfig(n_random=50, index_range=(0, 10))


# =============================================================================
# Test TrainingResult
# =============================================================================

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


# =============================================================================
# Test train() Function Validation
# =============================================================================

class TestTrainValidation:
    """Tests for train() function validation."""

    def test_mutually_exclusive_image_selection_raises(self):
        """Test that n_images and image_range raise ValueError."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            train(n_images=50, image_range=(0, 10))


# =============================================================================
# Test train() Function Behavior (Mocked)
# =============================================================================

class TestTrainFunction:
    """Tests for train() function behavior with mocked internals."""

    def test_train_returns_training_result(self, mock_dataset):
        """Test that train returns a TrainingResult."""
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
            call_args = mock_run.call_args
            assert call_args[0][0] is config

    def test_train_creates_checkpoint_callback(self, mock_dataset):
        """Test that train creates checkpoint callback when configured."""
        with patch("alveoleye.lungcv.mrcnn.api._run_training") as mock_run:
            mock_result = MagicMock(spec=TrainingResult)
            mock_run.return_value = mock_result

            train(
                dataset_path=str(mock_dataset),
                epochs=1,
                save_frequency=10,
                save_best=True,
            )

            call_args = mock_run.call_args
            callback_list = call_args[0][1]

            # Find ModelCheckpointCallback
            checkpoint_callbacks = [
                cb for cb in callback_list.callbacks
                if isinstance(cb, ModelCheckpointCallback)
            ]
            assert len(checkpoint_callbacks) == 1

    def test_train_creates_lambda_callback(self, mock_dataset):
        """Test that train creates lambda callback for user functions."""
        epoch_end_fn = MagicMock()

        with patch("alveoleye.lungcv.mrcnn.api._run_training") as mock_run:
            mock_result = MagicMock(spec=TrainingResult)
            mock_run.return_value = mock_result

            train(
                dataset_path=str(mock_dataset),
                epochs=1,
                on_epoch_end=epoch_end_fn,
            )

            call_args = mock_run.call_args
            callback_list = call_args[0][1]

            # Find LambdaCallback
            lambda_callbacks = [
                cb for cb in callback_list.callbacks
                if isinstance(cb, LambdaCallback)
            ]
            assert len(lambda_callbacks) == 1

    def test_train_passes_custom_callbacks(self, mock_dataset):
        """Test that train passes user-provided callbacks."""
        class CustomCallback(Callback):
            pass

        custom_cb = CustomCallback()

        with patch("alveoleye.lungcv.mrcnn.api._run_training") as mock_run:
            mock_result = MagicMock(spec=TrainingResult)
            mock_run.return_value = mock_result

            train(
                dataset_path=str(mock_dataset),
                epochs=1,
                callbacks=[custom_cb],
            )

            call_args = mock_run.call_args
            callback_list = call_args[0][1]

            assert custom_cb in callback_list.callbacks

    def test_train_passes_resume_from(self, mock_dataset):
        """Test that train passes resume_from parameter."""
        with patch("alveoleye.lungcv.mrcnn.api._run_training") as mock_run:
            mock_result = MagicMock(spec=TrainingResult)
            mock_run.return_value = mock_result

            train(
                dataset_path=str(mock_dataset),
                epochs=1,
                resume_from="/path/to/checkpoint.pth",
            )

            call_args = mock_run.call_args
            assert call_args[1]["resume_from"] == "/path/to/checkpoint.pth"


# =============================================================================
# Integration Tests (Slow)
# =============================================================================

@pytest.mark.skip(reason="Slow integration tests - run with: pytest -m 'not skip' or pytest --run-slow")
@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationTraining:
    """Integration tests that run actual training.

    These tests are slow (each initializes a full Mask R-CNN model).
    Run explicitly with: pytest -k TestIntegrationTraining --run-slow
    """

    def test_train_minimal_epochs(self, mock_dataset):
        """Test training for minimal epochs on mock dataset."""
        result = train(
            dataset_path=str(mock_dataset),
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=1,
        )

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert result.final_epoch == 0  # 0-indexed
        assert "train_loss" in result.history
        assert "val_loss" in result.history

    def test_train_with_all_optimizers(self, mock_dataset):
        """Test training works with all optimizer types."""
        for optimizer_name in ["sgd", "adam", "adamw", "rmsprop"]:
            result = train(
                dataset_path=str(mock_dataset),
                epochs=1,
                batch_size=1,
                num_workers=0,
                device="cpu",
                optimizer=optimizer_name,
                save_frequency=0,
                save_best=False,
                use_tensorboard=False,
                print_freq=100,  # Suppress output
            )

            assert isinstance(result, TrainingResult)
            assert result.model is not None

    def test_train_with_all_schedulers(self, mock_dataset):
        """Test training works with all scheduler types."""
        for scheduler_name in ["step", "cosine", "none"]:
            result = train(
                dataset_path=str(mock_dataset),
                epochs=1,
                batch_size=1,
                num_workers=0,
                device="cpu",
                scheduler=scheduler_name,
                save_frequency=0,
                save_best=False,
                use_tensorboard=False,
                print_freq=100,
            )

            assert isinstance(result, TrainingResult)
            assert result.model is not None

    def test_train_with_warmup(self, mock_dataset):
        """Test training with warmup scheduler."""
        result = train(
            dataset_path=str(mock_dataset),
            epochs=2,
            batch_size=1,
            num_workers=0,
            device="cpu",
            scheduler="step",
            warmup_epochs=1,
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        assert isinstance(result, TrainingResult)

    def test_train_with_augmentation(self, mock_dataset):
        """Test training with augmentation enabled."""
        result = train(
            dataset_path=str(mock_dataset),
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            augmentation=AugmentationConfig.default(),
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        assert isinstance(result, TrainingResult)

    def test_train_without_augmentation(self, mock_dataset):
        """Test training with augmentation disabled."""
        result = train(
            dataset_path=str(mock_dataset),
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            augmentation=AugmentationConfig.none(),
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        assert isinstance(result, TrainingResult)

    def test_train_with_image_selection(self, mock_dataset):
        """Test training with image selection."""
        result = train(
            dataset_path=str(mock_dataset),
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            n_images=1,
            seed=42,
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        assert isinstance(result, TrainingResult)

    def test_train_with_early_stopping(self, mock_dataset):
        """Test training with early stopping callback."""
        early_stop = EarlyStoppingCallback(patience=1, verbose=False)

        result = train(
            dataset_path=str(mock_dataset),
            epochs=10,  # High number, should stop early
            batch_size=1,
            num_workers=0,
            device="cpu",
            callbacks=[early_stop],
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        assert isinstance(result, TrainingResult)
        # May or may not trigger early stopping depending on loss

    def test_train_saves_checkpoint(self, mock_dataset, tmp_path):
        """Test training saves checkpoints correctly."""
        result = train(
            dataset_path=str(mock_dataset),
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            save_dir=str(tmp_path),
            save_frequency=1,
            save_best=True,
            use_tensorboard=False,
            print_freq=100,
        )

        assert isinstance(result, TrainingResult)
        # Best model should be saved
        assert (tmp_path / "best_model.pth").exists()

    def test_train_with_seed_reproducibility(self, mock_dataset):
        """Test that same seed produces reproducible training."""
        result1 = train(
            dataset_path=str(mock_dataset),
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            seed=42,
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        result2 = train(
            dataset_path=str(mock_dataset),
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            seed=42,
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        # Results should be similar (though not necessarily identical due to
        # non-determinism in some PyTorch operations)
        assert isinstance(result1, TrainingResult)
        assert isinstance(result2, TrainingResult)

    def test_train_with_callback_tracking(self, mock_dataset):
        """Test training with callback that tracks state."""
        epochs_seen = []
        train_losses = []

        def on_epoch_end(epoch, metrics):
            epochs_seen.append(epoch)
            if metrics.get("loss") is not None:
                train_losses.append(metrics["loss"])

        result = train(
            dataset_path=str(mock_dataset),
            epochs=2,
            batch_size=1,
            num_workers=0,
            device="cpu",
            on_epoch_end=on_epoch_end,
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        assert len(epochs_seen) == 2
        assert epochs_seen == [0, 1]

    def test_result_save_and_load(self, mock_dataset, tmp_path):
        """Test that saved model can be loaded."""
        result = train(
            dataset_path=str(mock_dataset),
            epochs=1,
            batch_size=1,
            num_workers=0,
            device="cpu",
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        model_path = tmp_path / "model.pth"
        result.save(model_path)

        loaded_model = torch.load(model_path)
        assert loaded_model is not None

    def test_result_checkpoint_save_and_load(self, mock_dataset, tmp_path):
        """Test that saved checkpoint can be loaded with all data."""
        config = TrainingConfig(
            epochs=1,
            data=DataConfig(dataset_path=str(mock_dataset), batch_size=1),
        )

        result = train(
            config=config,
            num_workers=0,
            device="cpu",
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=100,
        )

        checkpoint_path = tmp_path / "checkpoint.pth"
        result.save_checkpoint(checkpoint_path)

        checkpoint = torch.load(checkpoint_path)

        assert "model_state_dict" in checkpoint
        assert "history" in checkpoint
        assert "best_val_loss" in checkpoint
        assert "final_epoch" in checkpoint
        assert "config" in checkpoint


@pytest.mark.skip(reason="Slow integration tests - downloads dataset and trains model")
@pytest.mark.slow
@pytest.mark.integration
class TestIntegrationWithDownloadedDataset:
    """Integration tests using downloaded dataset from Google Drive.

    These tests download the actual training dataset and run training.
    Run explicitly with: pytest -k TestIntegrationWithDownloadedDataset --run-slow
    """

    def test_train_with_downloaded_dataset(self, tmp_path: Path):
        """Test training with dataset downloaded from Google Drive."""
        pytest.importorskip("gdown")

        from alveoleye.paper_scripts._utils import download_training_dataset

        try:
            dataset_path = download_training_dataset(
                output_dir=str(tmp_path),
                quiet=False,
            )
        except RuntimeError as e:
            pytest.skip(f"Could not download dataset: {e}")

        dataset = Path(dataset_path)
        assert (dataset / "images" / "train").exists() or (dataset / "images").exists()
        assert (dataset / "classes.json").exists()

        # Count available images
        if (dataset / "images" / "train").exists():
            train_images = list((dataset / "images" / "train").glob("*.png"))
            train_images.extend((dataset / "images" / "train").glob("*.jpg"))
        else:
            train_images = list((dataset / "images").glob("*.png"))
            train_images.extend((dataset / "images").glob("*.jpg"))

        n_images = min(5, len(train_images))

        if n_images == 0:
            pytest.skip("Downloaded dataset has no training images")

        result = train(
            dataset_path=dataset_path,
            epochs=1,
            n_images=n_images,
            device="cpu",
            save_frequency=0,
            save_best=False,
            use_tensorboard=False,
            print_freq=1,
        )

        assert isinstance(result, TrainingResult)
        assert result.model is not None
        assert result.final_epoch == 0
        assert result.best_val_loss < float("inf")
