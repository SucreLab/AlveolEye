"""Shared pytest fixtures for training tests.

This module provides reusable fixtures for testing the training API,
callbacks, configuration, and CLI components.
"""

import json
from pathlib import Path
from typing import Generator

import numpy as np
import pytest
import torch
from PIL import Image


class MockModel(torch.nn.Module):
    """Simple mock model for testing.

    Defined at module level so it can be pickled by torch.save.
    """
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)


@pytest.fixture
def mock_dataset(tmp_path: Path) -> Path:
    """Create a minimal mock dataset for testing.

    Creates a valid dataset structure with:
    - images/train, images/val directories with 2 images each
    - masks/train, masks/val directories with 2 masks each
    - classes.json with airway and vessel classes

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path to the mock dataset directory
    """
    # Create directory structure
    (tmp_path / "images" / "train").mkdir(parents=True)
    (tmp_path / "images" / "val").mkdir(parents=True)
    (tmp_path / "masks" / "train").mkdir(parents=True)
    (tmp_path / "masks" / "val").mkdir(parents=True)

    # Create classes.json
    classes = {
        "airway": "[255 0 0]",
        "vessel": "[0 255 0]",
    }
    (tmp_path / "classes.json").write_text(json.dumps(classes))

    # Create minimal images and masks
    for split in ["train", "val"]:
        for i in range(2):
            # Create image (64x64 RGB)
            img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
            Image.fromarray(img).save(tmp_path / "images" / split / f"img_{i}.png")

            # Create mask with class regions
            mask = np.zeros((64, 64, 3), dtype=np.uint8)
            mask[10:30, 10:30] = [255, 0, 0]  # Airway region
            mask[40:55, 40:55] = [0, 255, 0]  # Vessel region
            Image.fromarray(mask).save(tmp_path / "masks" / split / f"img_{i}.png")

    return tmp_path


@pytest.fixture
def mock_model() -> torch.nn.Module:
    """Create a simple mock model for testing.

    Returns a minimal torch.nn.Module that can be used for
    optimizer/callback testing without the overhead of
    initializing a full Mask R-CNN model.

    Returns:
        A simple linear model
    """
    return MockModel()


@pytest.fixture
def mock_params(mock_model: torch.nn.Module):
    """Get trainable parameters from mock model.

    Args:
        mock_model: The mock model fixture

    Returns:
        List of parameters requiring gradients
    """
    return [p for p in mock_model.parameters() if p.requires_grad]


@pytest.fixture
def sample_training_config():
    """Create a sample TrainingConfig for testing.

    Returns:
        TrainingConfig with minimal settings
    """
    from alveoleye.lungcv.mrcnn.config import TrainingConfig, DataConfig

    return TrainingConfig(
        epochs=2,
        data=DataConfig(dataset_path="mock_path", batch_size=2),
    )


@pytest.fixture
def sample_training_state(mock_model: torch.nn.Module):
    """Create a sample TrainingState for callback testing.

    Args:
        mock_model: The mock model fixture

    Returns:
        TrainingState instance with default values
    """
    from alveoleye.lungcv.mrcnn.callbacks import TrainingState

    return TrainingState(
        epoch=0,
        total_epochs=10,
        model=mock_model,
        optimizer=torch.optim.SGD(mock_model.parameters(), lr=0.01),
        train_metrics={"loss": 1.0},
        val_metrics={"loss": 0.8},
        best_val_loss=0.8,
        device=torch.device("cpu"),
    )


@pytest.fixture
def sample_optimizer_config():
    """Create a sample OptimizerConfig for testing.

    Returns:
        OptimizerConfig with default values
    """
    from alveoleye.lungcv.mrcnn.config import OptimizerConfig

    return OptimizerConfig()


@pytest.fixture
def sample_scheduler_config():
    """Create a sample SchedulerConfig for testing.

    Returns:
        SchedulerConfig with default values
    """
    from alveoleye.lungcv.mrcnn.config import SchedulerConfig

    return SchedulerConfig()


@pytest.fixture
def sample_augmentation_config():
    """Create a sample AugmentationConfig for testing.

    Returns:
        AugmentationConfig with default augmentations
    """
    from alveoleye.lungcv.mrcnn.config import AugmentationConfig

    return AugmentationConfig.default()


@pytest.fixture
def mock_optimizer(mock_params):
    """Create a mock optimizer for scheduler testing.

    Args:
        mock_params: Parameters from mock model

    Returns:
        SGD optimizer instance
    """
    return torch.optim.SGD(mock_params, lr=0.01)


@pytest.fixture
def temp_yaml_config(tmp_path: Path) -> Path:
    """Create a temporary YAML config file for testing.

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path to the temporary config file
    """
    config_content = """
epochs: 150
num_classes: 3
device: cpu
data:
  batch_size: 16
  num_workers: 0
optimizer:
  name: adam
  lr: 0.001
scheduler:
  name: cosine
checkpoint:
  save_dir: .
  save_frequency: 50
"""
    config_path = tmp_path / "config.yaml"
    config_path.write_text(config_content)
    return config_path


@pytest.fixture
def temp_augmentation_config(tmp_path: Path) -> Path:
    """Create a temporary augmentation config file for testing.

    Args:
        tmp_path: pytest's temporary directory fixture

    Returns:
        Path to the temporary augmentation config file
    """
    config_content = """
enabled: true
augmentations:
  - name: horizontal_flip
    probability: 0.5
  - name: color_jitter
    probability: 0.3
    params:
      brightness: 0.2
      contrast: 0.2
"""
    config_path = tmp_path / "augmentation_config.yaml"
    config_path.write_text(config_content)
    return config_path


# Markers configuration
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
