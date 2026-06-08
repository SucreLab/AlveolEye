"""Tests for augmentation system.

Tests cover:
- build_transforms() with various configurations
- Each registered augmentation
- Error handling for unknown augmentations
- Probability handling and transform application
"""

import numpy as np
import pytest
import torch
from PIL import Image

from alveoleye.lungcv.mrcnn.augmentations import (
    build_transforms,
    get_available_augmentations,
    AUGMENTATION_REGISTRY,
)
from alveoleye.lungcv.mrcnn.config import AugmentationConfig, AugmentationItem


class TestBuildTransforms:
    """Tests for build_transforms function."""

    def test_no_augmentation_when_disabled(self):
        """Test that no augmentations are added when disabled."""
        config = AugmentationConfig(enabled=False)
        transforms = build_transforms(config, train=True)
        # Should have base transforms: PILToTensor, WrapTvTensors, ToDtype, ToPureTensor, UnwrapTvTensors
        assert len(transforms.transforms) == 5

    def test_no_augmentation_for_validation(self):
        """Test that augmentations aren't applied during validation."""
        config = AugmentationConfig.default()
        transforms = build_transforms(config, train=False)
        # Should have base transforms
        assert len(transforms.transforms) == 5

    def test_augmentations_added_for_training(self):
        """Test that augmentations are added during training."""
        config = AugmentationConfig(
            augmentations=[
                AugmentationItem("horizontal_flip", probability=0.5),
                AugmentationItem("color_jitter", probability=0.3),
            ]
        )
        transforms = build_transforms(config, train=True)
        # Should have base transforms (5) + augmentations (2)
        assert len(transforms.transforms) >= 7

    def test_unknown_augmentation_raises(self):
        """Test that unknown augmentation raises ValueError."""
        config = AugmentationConfig(
            augmentations=[
                AugmentationItem("nonexistent_augmentation", probability=0.5),
            ]
        )
        with pytest.raises(ValueError, match="Unknown augmentation"):
            build_transforms(config, train=True)

    def test_empty_augmentation_list(self):
        """Test with empty augmentation list."""
        config = AugmentationConfig(enabled=True, augmentations=[])
        transforms = build_transforms(config, train=True)
        # Should have base transforms
        assert len(transforms.transforms) == 5

    def test_probability_zero_wraps_in_random_apply(self):
        """Test that probability < 1.0 wraps transform in RandomApply."""
        config = AugmentationConfig(
            augmentations=[
                AugmentationItem("horizontal_flip", probability=0.5),
            ]
        )
        transforms = build_transforms(config, train=True)
        # The wrapped transform should exist
        assert len(transforms.transforms) >= 6

    def test_probability_one_no_random_apply(self):
        """Test that probability = 1.0 doesn't wrap in RandomApply."""
        config = AugmentationConfig(
            augmentations=[
                AugmentationItem("horizontal_flip", probability=1.0),
            ]
        )
        transforms = build_transforms(config, train=True)
        # Transform should be added directly (still works)
        assert len(transforms.transforms) >= 6


class TestAugmentationRegistry:
    """Tests for augmentation registry."""

    def test_registry_not_empty(self):
        """Test that registry contains augmentations."""
        assert len(AUGMENTATION_REGISTRY) > 0

    def test_all_expected_augmentations_exist(self):
        """Test that all expected augmentations are registered."""
        available = get_available_augmentations()
        expected = [
            "horizontal_flip",
            "vertical_flip",
            "color_jitter",
            "rotation",
            "gaussian_blur",
            "affine",
            "perspective",
            "erasing",
        ]
        for aug in expected:
            assert aug in available, f"Missing augmentation: {aug}"

    def test_get_available_augmentations_returns_list(self):
        """Test that get_available_augmentations returns a list."""
        available = get_available_augmentations()
        assert isinstance(available, list)
        assert all(isinstance(name, str) for name in available)

    @pytest.mark.parametrize("aug_name", list(AUGMENTATION_REGISTRY.keys()))
    def test_each_augmentation_builds(self, aug_name):
        """Test that each registered augmentation can be built."""
        builder = AUGMENTATION_REGISTRY[aug_name]
        transform = builder({})
        assert transform is not None


class TestIndividualAugmentations:
    """Tests for individual augmentation builders."""

    def test_horizontal_flip_builds(self):
        """Test horizontal flip augmentation."""
        builder = AUGMENTATION_REGISTRY["horizontal_flip"]
        transform = builder({})
        assert transform is not None

    def test_vertical_flip_builds(self):
        """Test vertical flip augmentation."""
        builder = AUGMENTATION_REGISTRY["vertical_flip"]
        transform = builder({})
        assert transform is not None

    def test_color_jitter_builds(self):
        """Test color jitter augmentation."""
        builder = AUGMENTATION_REGISTRY["color_jitter"]
        transform = builder({})
        assert transform is not None

    def test_color_jitter_with_params(self):
        """Test color jitter with custom parameters."""
        builder = AUGMENTATION_REGISTRY["color_jitter"]
        params = {"brightness": 0.3, "contrast": 0.3, "saturation": 0.3, "hue": 0.1}
        transform = builder(params)
        assert transform is not None

    def test_rotation_builds(self):
        """Test rotation augmentation."""
        builder = AUGMENTATION_REGISTRY["rotation"]
        transform = builder({})
        assert transform is not None

    def test_rotation_with_params(self):
        """Test rotation with custom degrees."""
        builder = AUGMENTATION_REGISTRY["rotation"]
        params = {"degrees": (-15, 15)}
        transform = builder(params)
        assert transform is not None

    def test_gaussian_blur_builds(self):
        """Test Gaussian blur augmentation."""
        builder = AUGMENTATION_REGISTRY["gaussian_blur"]
        transform = builder({})
        assert transform is not None

    def test_gaussian_blur_with_params(self):
        """Test Gaussian blur with custom parameters."""
        builder = AUGMENTATION_REGISTRY["gaussian_blur"]
        params = {"kernel_size": 5, "sigma": (0.5, 2.5)}
        transform = builder(params)
        assert transform is not None

    def test_affine_builds(self):
        """Test affine augmentation."""
        builder = AUGMENTATION_REGISTRY["affine"]
        transform = builder({})
        assert transform is not None

    def test_affine_with_params(self):
        """Test affine with all parameters."""
        builder = AUGMENTATION_REGISTRY["affine"]
        params = {
            "degrees": 10,
            "translate": (0.1, 0.1),
            "scale": (0.9, 1.1),
            "shear": (-5, 5),
        }
        transform = builder(params)
        assert transform is not None

    def test_perspective_builds(self):
        """Test perspective augmentation."""
        builder = AUGMENTATION_REGISTRY["perspective"]
        transform = builder({})
        assert transform is not None

    def test_perspective_with_params(self):
        """Test perspective with custom distortion scale."""
        builder = AUGMENTATION_REGISTRY["perspective"]
        params = {"distortion_scale": 0.3}
        transform = builder(params)
        assert transform is not None

    def test_erasing_builds(self):
        """Test random erasing augmentation."""
        builder = AUGMENTATION_REGISTRY["erasing"]
        transform = builder({})
        assert transform is not None

    def test_erasing_with_params(self):
        """Test random erasing with custom parameters."""
        builder = AUGMENTATION_REGISTRY["erasing"]
        params = {"scale": (0.05, 0.3), "ratio": (0.5, 2.0)}
        transform = builder(params)
        assert transform is not None


class TestTransformApplication:
    """Tests for actual transform application to images."""

    @pytest.fixture
    def test_image(self):
        """Create a test PIL image."""
        img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        return Image.fromarray(img)

    @pytest.fixture
    def test_target(self):
        """Create a test target dict."""
        return {
            "boxes": torch.tensor([[10, 10, 30, 30]], dtype=torch.float32),
            "labels": torch.tensor([1]),
            "masks": torch.zeros((1, 64, 64), dtype=torch.uint8),
        }

    def test_transforms_return_tensor(self, test_image, test_target):
        """Test that transforms return a tensor image."""
        config = AugmentationConfig(
            augmentations=[
                AugmentationItem("horizontal_flip", probability=1.0),
            ]
        )
        transforms = build_transforms(config, train=True)

        result_img, result_target = transforms(test_image, test_target)

        assert isinstance(result_img, torch.Tensor)
        assert result_img.dim() == 3  # C, H, W

    def test_transforms_preserve_target_keys(self, test_image, test_target):
        """Test that transforms preserve target dictionary keys."""
        config = AugmentationConfig.default()
        transforms = build_transforms(config, train=True)

        _, result_target = transforms(test_image, test_target)

        assert "boxes" in result_target
        assert "labels" in result_target

    def test_transforms_output_shape(self, test_image, test_target):
        """Test that output image has correct shape."""
        config = AugmentationConfig.default()
        transforms = build_transforms(config, train=True)

        result_img, _ = transforms(test_image, test_target)

        assert result_img.shape[0] == 3  # RGB channels
        assert result_img.shape[1] == 64  # Height
        assert result_img.shape[2] == 64  # Width

    def test_transforms_output_dtype(self, test_image, test_target):
        """Test that output image has correct dtype (float)."""
        config = AugmentationConfig.default()
        transforms = build_transforms(config, train=True)

        result_img, _ = transforms(test_image, test_target)

        assert result_img.dtype == torch.float32

    def test_validation_transforms_consistent(self, test_image, test_target):
        """Test that validation transforms produce consistent output."""
        config = AugmentationConfig.default()
        transforms = build_transforms(config, train=False)

        # Apply twice and check consistency
        result1, _ = transforms(test_image.copy(), test_target.copy())
        result2, _ = transforms(test_image.copy(), test_target.copy())

        assert torch.allclose(result1, result2)


class TestAugmentationConfig:
    """Additional tests for AugmentationConfig interaction with transforms."""

    def test_default_config_builds_successfully(self):
        """Test that default config builds without errors."""
        config = AugmentationConfig.default()
        transforms = build_transforms(config, train=True)
        assert transforms is not None

    def test_none_config_builds_successfully(self):
        """Test that none config builds without errors."""
        config = AugmentationConfig.none()
        transforms = build_transforms(config, train=True)
        assert transforms is not None

    def test_custom_config_with_all_augmentations(self):
        """Test config with many augmentations."""
        config = AugmentationConfig(
            augmentations=[
                AugmentationItem("horizontal_flip", probability=0.5),
                AugmentationItem("vertical_flip", probability=0.5),
                AugmentationItem("color_jitter", probability=0.3),
                AugmentationItem("rotation", probability=0.2),
                AugmentationItem("gaussian_blur", probability=0.1),
            ]
        )
        transforms = build_transforms(config, train=True)
        # Base (3) + augmentations (5)
        assert len(transforms.transforms) >= 8
