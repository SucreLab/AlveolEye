"""Tests for training configuration dataclasses.

Tests cover:
- Default value instantiation
- Custom value instantiation
- Serialization (to_dict, from_dict, to_yaml, from_yaml)
- Validation (mutual exclusion, factory methods)
"""

from pathlib import Path

import pytest

from alveoleye.lungcv.mrcnn.config import (
    TrainingConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    AugmentationConfig,
    AugmentationItem,
    ImageSelectionConfig,
    CheckpointConfig,
    LoggingConfig,
)


class TestOptimizerConfig:
    """Tests for OptimizerConfig dataclass."""

    def test_default_values(self):
        """Test that defaults match expected values."""
        config = OptimizerConfig()
        assert config.name == "sgd"
        assert config.lr == 0.005
        assert config.momentum == 0.9
        assert config.weight_decay == 0.0005
        assert config.betas == (0.9, 0.999)
        assert config.eps == 1e-8

    def test_custom_values(self):
        """Test custom value instantiation."""
        config = OptimizerConfig(
            name="adamw",
            lr=0.001,
            weight_decay=0.01,
        )
        assert config.name == "adamw"
        assert config.lr == 0.001
        assert config.weight_decay == 0.01

    def test_sgd_specific_params(self):
        """Test SGD-specific parameters."""
        config = OptimizerConfig(
            name="sgd",
            nesterov=True,
        )
        assert config.nesterov is True

    def test_adam_specific_params(self):
        """Test Adam-specific parameters."""
        config = OptimizerConfig(
            name="adam",
            betas=(0.8, 0.99),
        )
        assert config.betas == (0.8, 0.99)


class TestSchedulerConfig:
    """Tests for SchedulerConfig dataclass."""

    def test_default_values(self):
        """Test that defaults match expected values."""
        config = SchedulerConfig()
        assert config.name == "step"
        assert config.step_size == 20
        assert config.gamma == 0.1
        assert config.warmup_epochs == 0

    def test_step_scheduler_config(self):
        """Test StepLR configuration."""
        config = SchedulerConfig(
            name="step",
            step_size=10,
            gamma=0.5,
        )
        assert config.step_size == 10
        assert config.gamma == 0.5

    def test_cosine_scheduler_config(self):
        """Test CosineAnnealingLR configuration."""
        config = SchedulerConfig(
            name="cosine",
            T_max=100,
            eta_min=1e-6,
        )
        assert config.T_max == 100
        assert config.eta_min == 1e-6

    def test_warmup_config(self):
        """Test warmup configuration."""
        config = SchedulerConfig(
            warmup_epochs=5,
            warmup_factor=0.01,
        )
        assert config.warmup_epochs == 5
        assert config.warmup_factor == 0.01


class TestAugmentationItem:
    """Tests for AugmentationItem dataclass."""

    def test_minimal_creation(self):
        """Test creation with just name."""
        item = AugmentationItem(name="horizontal_flip")
        assert item.name == "horizontal_flip"
        assert item.probability == 0.5
        assert item.params == {}

    def test_with_probability(self):
        """Test creation with custom probability."""
        item = AugmentationItem(name="color_jitter", probability=0.3)
        assert item.probability == 0.3

    def test_with_params(self):
        """Test creation with parameters."""
        item = AugmentationItem(
            name="color_jitter",
            params={"brightness": 0.2, "contrast": 0.2},
        )
        assert item.params["brightness"] == 0.2
        assert item.params["contrast"] == 0.2


class TestAugmentationConfig:
    """Tests for AugmentationConfig dataclass."""

    def test_default_values(self):
        """Test default instantiation."""
        config = AugmentationConfig()
        assert config.enabled is True
        assert config.augmentations == []

    def test_default_factory(self):
        """Test default() factory method."""
        config = AugmentationConfig.default()
        assert config.enabled is True
        assert len(config.augmentations) > 0

        names = [a.name for a in config.augmentations]
        assert "horizontal_flip" in names
        assert "vertical_flip" in names
        assert "color_jitter" in names

    def test_none_factory(self):
        """Test none() factory method."""
        config = AugmentationConfig.none()
        assert config.enabled is False
        assert len(config.augmentations) == 0

    def test_custom_augmentations(self):
        """Test custom augmentation list."""
        config = AugmentationConfig(
            augmentations=[
                AugmentationItem("horizontal_flip", probability=0.8),
                AugmentationItem("rotation", probability=0.5, params={"degrees": 15}),
            ]
        )
        assert len(config.augmentations) == 2
        assert config.augmentations[0].probability == 0.8


class TestImageSelectionConfig:
    """Tests for ImageSelectionConfig dataclass."""

    def test_default_values(self):
        """Test default instantiation (no selection)."""
        config = ImageSelectionConfig()
        assert config.n_random is None
        assert config.index_range is None
        assert config.seed is None

    def test_n_random_selection(self):
        """Test random image selection."""
        config = ImageSelectionConfig(n_random=50, seed=42)
        assert config.n_random == 50
        assert config.seed == 42

    def test_index_range_selection(self):
        """Test index range selection."""
        config = ImageSelectionConfig(index_range=(10, 30))
        assert config.index_range == (10, 30)

    def test_mutual_exclusion_raises(self):
        """Test that n_random and index_range are mutually exclusive."""
        with pytest.raises(ValueError, match="mutually exclusive"):
            ImageSelectionConfig(n_random=50, index_range=(0, 10))


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Test default instantiation."""
        config = DataConfig()
        assert config.dataset_path == "png_dataset"
        assert config.batch_size == 10
        assert config.num_workers == 0
        assert config.img_extension == ".png"
        assert config.image_selection is None

    def test_custom_values(self):
        """Test custom value instantiation."""
        config = DataConfig(
            dataset_path="/path/to/data",
            batch_size=8,
            num_workers=4,
        )
        assert config.dataset_path == "/path/to/data"
        assert config.batch_size == 8
        assert config.num_workers == 4

    def test_with_image_selection(self):
        """Test with image selection config."""
        selection = ImageSelectionConfig(n_random=25)
        config = DataConfig(image_selection=selection)
        assert config.image_selection is not None
        assert config.image_selection.n_random == 25


class TestCheckpointConfig:
    """Tests for CheckpointConfig dataclass."""

    def test_default_values(self):
        """Test default instantiation."""
        config = CheckpointConfig()
        assert config.save_dir == "."
        assert config.save_frequency == 50
        assert config.save_best is True
        assert config.save_last is True

    def test_custom_values(self):
        """Test custom value instantiation."""
        config = CheckpointConfig(
            save_dir="checkpoints",
            save_frequency=10,
            save_best=False,
        )
        assert config.save_dir == "checkpoints"
        assert config.save_frequency == 10
        assert config.save_best is False


class TestLoggingConfig:
    """Tests for LoggingConfig dataclass."""

    def test_default_values(self):
        """Test default instantiation."""
        config = LoggingConfig()
        assert config.use_tensorboard is True
        assert config.log_dir is None
        assert config.print_freq == 10
        assert config.log_images is True

    def test_custom_values(self):
        """Test custom value instantiation."""
        config = LoggingConfig(
            use_tensorboard=False,
            log_dir="logs",
            print_freq=5,
        )
        assert config.use_tensorboard is False
        assert config.log_dir == "logs"
        assert config.print_freq == 5


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_instantiation(self):
        """Test default instantiation."""
        config = TrainingConfig()
        assert config.epochs == 100
        assert config.num_classes == 3
        assert config.device == "auto"
        assert config.seed is None
        assert config.mixed_precision is False
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.optimizer, OptimizerConfig)
        assert isinstance(config.scheduler, SchedulerConfig)
        assert isinstance(config.augmentation, AugmentationConfig)
        assert isinstance(config.checkpoint, CheckpointConfig)
        assert isinstance(config.logging, LoggingConfig)

    def test_custom_instantiation(self):
        """Test custom value instantiation."""
        config = TrainingConfig(
            epochs=200,
            num_classes=5,
            device="cuda",
            seed=42,
            data=DataConfig(batch_size=16),
            optimizer=OptimizerConfig(name="adam", lr=0.001),
        )
        assert config.epochs == 200
        assert config.num_classes == 5
        assert config.device == "cuda"
        assert config.seed == 42
        assert config.data.batch_size == 16
        assert config.optimizer.name == "adam"

    def test_to_dict(self):
        """Test to_dict serialization."""
        config = TrainingConfig(epochs=50)
        d = config.to_dict()

        assert isinstance(d, dict)
        assert d["epochs"] == 50
        assert "data" in d
        assert "optimizer" in d
        assert "scheduler" in d

    def test_from_dict(self):
        """Test from_dict deserialization."""
        d = {
            "epochs": 75,
            "num_classes": 4,
            "data": {"batch_size": 8},
            "optimizer": {"name": "adamw", "lr": 0.002},
        }
        config = TrainingConfig.from_dict(d)

        assert config.epochs == 75
        assert config.num_classes == 4
        assert config.data.batch_size == 8
        assert config.optimizer.name == "adamw"
        assert config.optimizer.lr == 0.002

    def test_to_dict_from_dict_roundtrip(self):
        """Test round-trip serialization."""
        original = TrainingConfig(
            epochs=150,
            num_classes=5,
            data=DataConfig(
                dataset_path="/data",
                batch_size=16,
                image_selection=ImageSelectionConfig(n_random=50, seed=42),
            ),
            optimizer=OptimizerConfig(name="adamw", lr=0.001),
            augmentation=AugmentationConfig(
                augmentations=[
                    AugmentationItem("horizontal_flip", probability=0.5),
                ]
            ),
        )

        d = original.to_dict()
        restored = TrainingConfig.from_dict(d)

        assert restored.epochs == original.epochs
        assert restored.num_classes == original.num_classes
        assert restored.data.dataset_path == original.data.dataset_path
        assert restored.data.batch_size == original.data.batch_size
        assert restored.data.image_selection.n_random == 50
        assert restored.data.image_selection.seed == 42
        assert restored.optimizer.name == original.optimizer.name
        assert restored.optimizer.lr == original.optimizer.lr
        assert len(restored.augmentation.augmentations) == 1

    def test_yaml_roundtrip(self, tmp_path: Path):
        """Test YAML serialization round-trip."""
        original = TrainingConfig(
            epochs=200,
            data=DataConfig(batch_size=32),
            optimizer=OptimizerConfig(name="adam"),
        )

        yaml_path = tmp_path / "config.yaml"
        original.to_yaml(yaml_path)

        assert yaml_path.exists()

        restored = TrainingConfig.from_yaml(yaml_path)

        assert restored.epochs == 200
        assert restored.data.batch_size == 32
        assert restored.optimizer.name == "adam"

    def test_from_dict_with_nested_augmentations(self):
        """Test from_dict handles nested augmentation items."""
        d = {
            "epochs": 100,
            "augmentation": {
                "enabled": True,
                "augmentations": [
                    {"name": "horizontal_flip", "probability": 0.5},
                    {"name": "color_jitter", "probability": 0.3, "params": {"brightness": 0.2}},
                ],
            },
        }
        config = TrainingConfig.from_dict(d)

        assert config.augmentation.enabled is True
        assert len(config.augmentation.augmentations) == 2
        assert config.augmentation.augmentations[0].name == "horizontal_flip"
        assert config.augmentation.augmentations[1].params["brightness"] == 0.2

    def test_from_dict_with_image_selection(self):
        """Test from_dict handles image selection config."""
        d = {
            "epochs": 100,
            "data": {
                "batch_size": 8,
                "image_selection": {
                    "n_random": 50,
                    "seed": 123,
                },
            },
        }
        config = TrainingConfig.from_dict(d)

        assert config.data.image_selection is not None
        assert config.data.image_selection.n_random == 50
        assert config.data.image_selection.seed == 123


class TestConfigEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_empty_augmentation_list(self):
        """Test config with empty augmentation list."""
        config = AugmentationConfig(enabled=True, augmentations=[])
        assert config.enabled is True
        assert len(config.augmentations) == 0

    def test_path_types(self):
        """Test that Path objects work for path fields."""
        config = DataConfig(dataset_path=Path("/path/to/data"))
        assert config.dataset_path == Path("/path/to/data")

    def test_optional_fields_none(self):
        """Test optional fields with None values."""
        config = TrainingConfig(
            seed=None,
            gradient_clip_val=None,
        )
        assert config.seed is None
        assert config.gradient_clip_val is None

    def test_scheduler_none_type(self):
        """Test scheduler with 'none' type."""
        config = SchedulerConfig(name="none")
        assert config.name == "none"

    def test_checkpoint_config_with_template(self):
        """Test checkpoint config with custom filename template."""
        config = CheckpointConfig(
            filename_template="model_epoch_{epoch}_loss_{val_loss:.4f}.pth"
        )
        assert "{epoch}" in config.filename_template
        assert "{val_loss" in config.filename_template
