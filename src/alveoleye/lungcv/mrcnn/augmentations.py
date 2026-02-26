"""Augmentation configuration and transform builder.

This module provides a registry of available augmentations and functions
to build transform pipelines from AugmentationConfig objects.

Functions:
    build_transforms: Build a transform pipeline from AugmentationConfig
    get_available_augmentations: Get list of available augmentation names

Available Augmentations:
    - horizontal_flip: Random horizontal flip
    - vertical_flip: Random vertical flip
    - color_jitter: Random color adjustments (brightness, contrast, saturation, hue)
    - rotation: Random rotation
    - gaussian_blur: Random Gaussian blur
    - affine: Random affine transformation
    - perspective: Random perspective transformation
    - photometric_distort: Comprehensive photometric distortion
    - scale_jitter: Scale jittering (from transforms.py)
    - random_crop: IoU-based random crop (from transforms.py)
    - zoom_out: Random zoom out (from transforms.py)
"""

from typing import Dict, Callable, List, Any

import torch
from torchvision.transforms import v2 as T

from alveoleye.lungcv.mrcnn.config import AugmentationConfig


# Registry of available augmentations
# Maps augmentation name -> builder function
AUGMENTATION_REGISTRY: Dict[str, Callable[[dict], T.Transform]] = {}


def register_augmentation(name: str):
    """Decorator to register an augmentation builder function."""
    def decorator(func: Callable[[dict], T.Transform]):
        AUGMENTATION_REGISTRY[name] = func
        return func
    return decorator


# --- Standard augmentations using torchvision.transforms.v2 ---

@register_augmentation('horizontal_flip')
def _build_horizontal_flip(params: dict) -> T.Transform:
    """Build horizontal flip transform."""
    #from alveoleye.lungcv.mrcnn.transforms import RandomHorizontalFlip
    return T.RandomHorizontalFlip(p=0.5)  # Probability handled externally


@register_augmentation('vertical_flip')
def _build_vertical_flip(params: dict) -> T.Transform:
    """Build vertical flip transform."""
    #from alveoleye.lungcv.mrcnn.transforms import RandomVerticalFlip
    return T.RandomVerticalFlip(p=0.5)


@register_augmentation('color_jitter')
def _build_color_jitter(params: dict) -> T.Transform:
    """Build color jitter transform.

    Params:
        brightness (float): Brightness adjustment factor (default: 0.2)
        contrast (float): Contrast adjustment factor (default: 0.2)
        saturation (float): Saturation adjustment factor (default: 0.2)
        hue (float): Hue adjustment factor (default: 0.1)
    """
    return T.ColorJitter(
        brightness=params.get('brightness', 0.2),
        contrast=params.get('contrast', 0.2),
        saturation=params.get('saturation', 0.2),
        hue=params.get('hue', 0.1),
    )


@register_augmentation('rotation')
def _build_rotation(params: dict) -> T.Transform:
    """Build rotation transform.

    Params:
        degrees (tuple or float): Range of degrees for rotation (default: (-10, 10))
    """
    degrees = params.get('degrees', (-10, 10))
    return T.RandomRotation(degrees=degrees)


@register_augmentation('gaussian_blur')
def _build_gaussian_blur(params: dict) -> T.Transform:
    """Build Gaussian blur transform.

    Params:
        kernel_size (int): Size of blur kernel (default: 3)
        sigma (tuple): Range of sigma values (default: (0.1, 2.0))
    """
    kernel_size = params.get('kernel_size', 3)
    sigma = params.get('sigma', (0.1, 2.0))
    return T.GaussianBlur(kernel_size=kernel_size, sigma=sigma)


@register_augmentation('affine')
def _build_affine(params: dict) -> T.Transform:
    """Build random affine transform.

    Params:
        degrees (float): Range of rotation degrees (default: 0)
        translate (tuple): Range of translation (default: None)
        scale (tuple): Range of scaling (default: None)
        shear (tuple or float): Range of shear (default: None)
    """
    return T.RandomAffine(
        degrees=params.get('degrees', 0),
        translate=params.get('translate', None),
        scale=params.get('scale', None),
        shear=params.get('shear', None),
    )


@register_augmentation('perspective')
def _build_perspective(params: dict) -> T.Transform:
    """Build random perspective transform.

    Params:
        distortion_scale (float): Distortion scale (default: 0.25)
    """
    distortion_scale = params.get('distortion_scale', 0.25)
    return T.RandomPerspective(distortion_scale=distortion_scale, p=1.0)


@register_augmentation('erasing')
def _build_erasing(params: dict) -> T.Transform:
    """Build random erasing transform.

    Params:
        scale (tuple): Range of area proportion to erase (default: (0.02, 0.2))
        ratio (tuple): Range of aspect ratio (default: (0.3, 3.3))
        value (str or number): Fill value (default: 'random')
    """
    return T.RandomErasing(
        p=1.0,
        scale=params.get('scale', (0.02, 0.2)),
        ratio=params.get('ratio', (0.3, 3.3)),
        value=params.get('value', 'random'),
    )


# --- Custom augmentations from transforms.py ---

@register_augmentation('photometric_distort')
def _build_photometric_distort(params: dict) -> T.Transform:
    """Build comprehensive photometric distortion.

    Uses the custom RandomPhotometricDistort from transforms.py.

    Params:
        contrast (tuple): Contrast range (default: (0.5, 1.5))
        saturation (tuple): Saturation range (default: (0.5, 1.5))
        hue (tuple): Hue range (default: (-0.05, 0.05))
        brightness (tuple): Brightness range (default: (0.875, 1.125))
    """
    from alveoleye.lungcv.mrcnn.transforms import RandomPhotometricDistort
    return RandomPhotometricDistort(
        contrast=params.get('contrast', (0.5, 1.5)),
        saturation=params.get('saturation', (0.5, 1.5)),
        hue=params.get('hue', (-0.05, 0.05)),
        brightness=params.get('brightness', (0.875, 1.125)),
        p=1.0,  # Probability handled externally
    )


@register_augmentation('scale_jitter')
def _build_scale_jitter(params: dict) -> T.Transform:
    """Build scale jittering transform.

    Uses the custom ScaleJitter from transforms.py.

    Params:
        target_size (tuple): Target size (height, width) (default: (512, 512))
        scale_range (tuple): Scale range (default: (0.1, 2.0))
    """
    from alveoleye.lungcv.mrcnn.transforms import ScaleJitter
    return ScaleJitter(
        target_size=params.get('target_size', (512, 512)),
        scale_range=params.get('scale_range', (0.1, 2.0)),
    )


@register_augmentation('random_crop')
def _build_random_crop(params: dict) -> T.Transform:
    """Build IoU-based random crop.

    Uses the custom RandomIoUCrop from transforms.py.

    Params:
        min_scale (float): Minimum scale (default: 0.3)
        max_scale (float): Maximum scale (default: 1.0)
    """
    from alveoleye.lungcv.mrcnn.transforms import RandomIoUCrop
    return RandomIoUCrop(
        min_scale=params.get('min_scale', 0.3),
        max_scale=params.get('max_scale', 1.0),
    )


@register_augmentation('zoom_out')
def _build_zoom_out(params: dict) -> T.Transform:
    """Build random zoom out (canvas expansion).

    Uses the custom RandomZoomOut from transforms.py.

    Params:
        fill (list): Fill color for padding (default: [0.0, 0.0, 0.0])
        side_range (tuple): Range of expansion factor (default: (1.0, 4.0))
    """
    from alveoleye.lungcv.mrcnn.transforms import RandomZoomOut
    return RandomZoomOut(
        fill=params.get('fill', [0.0, 0.0, 0.0]),
        side_range=params.get('side_range', (1.0, 4.0)),
        p=1.0,
    )


def build_transforms(
    config: AugmentationConfig,
    train: bool = True,
    target_size: tuple = None,
) -> T.Compose:
    """Build a transform pipeline from augmentation configuration.

    Args:
        config: AugmentationConfig instance
        train: Whether this is for training (augmentations applied) or validation
        target_size: Optional (height, width) tuple to resize all images to.
                     Required when batch_size > 1 with variable-sized images.

    Returns:
        Composed transform pipeline

    Raises:
        ValueError: If an unknown augmentation name is specified

    Example:
        aug_config = AugmentationConfig(augmentations=[
            AugmentationItem('horizontal_flip', probability=0.5),
            AugmentationItem('color_jitter', probability=0.3, params={'brightness': 0.2}),
        ])
        transforms = build_transforms(aug_config, train=True, target_size=(1440, 1920))
    """
    transform_list = [T.PILToTensor()]

    # Wrap into tv_tensors so v2 geometric transforms also update masks/boxes
    try:
        from alveoleye.lungcv.mrcnn.transforms import WrapTvTensors, UnwrapTvTensors  # type: ignore
        wrap_supported = True
    except Exception:
        wrap_supported = False

    if wrap_supported:
        transform_list.append(WrapTvTensors())

    # Add resize transform if target_size is specified
    if target_size is not None:
        transform_list.append(T.Resize(target_size, antialias=True))

    if train and config.enabled:
        for aug_item in config.augmentations:
            if aug_item.name not in AUGMENTATION_REGISTRY:
                available = list(AUGMENTATION_REGISTRY.keys())
                raise ValueError(
                    f"Unknown augmentation: '{aug_item.name}'. "
                    f"Available: {available}"
                )

            builder = AUGMENTATION_REGISTRY[aug_item.name]
            transform = builder(aug_item.params)

            # Wrap with RandomApply for probability control
            if aug_item.probability < 1.0:
                transform = T.RandomApply([transform], p=aug_item.probability)

            transform_list.append(transform)

    # Always add final conversions
    transform_list.extend([
        T.ToDtype(torch.float, scale=True),
        T.ToPureTensor(),
    ])

    # Unwrap back to pure tensors for the model
    if wrap_supported:
        transform_list.append(UnwrapTvTensors())

    return T.Compose(transform_list)


def get_available_augmentations() -> List[str]:
    """Return list of available augmentation names.

    Returns:
        List of augmentation names that can be used in AugmentationConfig
    """
    return list(AUGMENTATION_REGISTRY.keys())
