"""Shared dataset utilities for AlveolEye.

This module provides common dataset structure detection and validation
utilities used across the codebase.
"""

import os
from pathlib import Path
from typing import Optional, Union, Literal

# =============================================================================
# Constants
# =============================================================================

# Supported image extensions for dataset detection
SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


# =============================================================================
# Dataset Structure Detection
# =============================================================================

DatasetStructure = Literal["split", "flat"]


def detect_dataset_structure(
    path: Union[str, Path],
    img_extension: Optional[str] = None,
) -> Optional[DatasetStructure]:
    """Detect whether a dataset uses split (train/val) or flat structure.

    This is the single source of truth for dataset structure detection.
    All other modules should import and use this function.

    Args:
        path: Path to the dataset directory.
        img_extension: Specific image extension to check for (e.g., ".png").
                      If None, checks for any supported extension.

    Returns:
        'split' if train/val subdirectories exist in images/ and masks/
        'flat' if images are directly in images/ and masks/
        None if structure is invalid or path doesn't exist

    Examples:
        >>> detect_dataset_structure("/path/to/dataset")
        'flat'
        >>> detect_dataset_structure("/path/to/dataset", img_extension=".png")
        'split'
    """
    path = Path(path)

    if not path.exists():
        return None

    images_dir = path / "images"
    masks_dir = path / "masks"

    if not images_dir.is_dir() or not masks_dir.is_dir():
        return None

    # Check for split structure (train/val subdirectories)
    split_dirs = [
        images_dir / "train",
        images_dir / "val",
        masks_dir / "train",
        masks_dir / "val",
    ]
    if all(d.is_dir() for d in split_dirs):
        return "split"

    # Check for flat structure (images directly in images/ and masks/)
    extensions = (img_extension,) if img_extension else SUPPORTED_IMAGE_EXTENSIONS

    def has_image_files(directory: Path) -> bool:
        """Check if directory contains image files with matching extensions."""
        if not directory.exists():
            return False
        try:
            return any(
                f.is_file() and f.suffix.lower() in extensions
                for f in directory.iterdir()
            )
        except PermissionError:
            return False

    if has_image_files(images_dir) and has_image_files(masks_dir):
        return "flat"

    return None


def is_valid_dataset_structure(
    path: Union[str, Path],
    require_classes_json: bool = True,
) -> bool:
    """Check if a path contains a valid dataset structure.

    Args:
        path: Path to the dataset directory.
        require_classes_json: Whether to require classes.json file (default: True).

    Returns:
        True if the path contains a valid dataset structure.

    Note:
        This function validates the overall structure. For training,
        classes.json is required but may not be needed for other operations.
    """
    path = Path(path)

    if require_classes_json and not (path / "classes.json").exists():
        return False

    return detect_dataset_structure(path) is not None


def count_dataset_images(
    path: Union[str, Path],
    split: Literal["train", "val", "all"] = "train",
    val_split: float = 0.2,
    img_extension: Optional[str] = None,
) -> int:
    """Count images in a dataset.

    Args:
        path: Path to the dataset directory.
        split: Which split to count - 'train', 'val', or 'all'.
        val_split: Fraction for validation when using flat structure (default: 0.2).
        img_extension: Specific image extension to count (e.g., ".png").
                      If None, counts all supported extensions.

    Returns:
        Number of images in the specified split.
    """
    path = Path(path)
    structure = detect_dataset_structure(path, img_extension)
    extensions = (img_extension,) if img_extension else SUPPORTED_IMAGE_EXTENSIONS

    def count_files(directory: Path) -> int:
        """Count image files in a directory."""
        if not directory.exists():
            return 0
        return sum(
            1 for f in directory.iterdir()
            if f.is_file() and f.suffix.lower() in extensions
        )

    if structure == "split":
        if split == "all":
            return count_files(path / "images" / "train") + count_files(path / "images" / "val")
        return count_files(path / "images" / split)

    elif structure == "flat":
        total = count_files(path / "images")

        if split == "all":
            return total

        n_val = max(1, int(total * val_split))
        n_train = total - n_val

        return n_train if split == "train" else n_val

    return 0
