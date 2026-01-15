"""LungDataset class for loading lung segmentation data.

This module provides the LungDataset class for loading training and validation
data from a structured directory format.

Supports two dataset structures:

1. Split structure (train/val subdirectories):
   dataset/
   ├── images/
   │   ├── train/
   │   └── val/
   ├── masks/
   │   ├── train/
   │   └── val/
   └── classes.json

2. Flat structure (auto-split):
   dataset/
   ├── images/
   │   ├── image1.png
   │   └── ...
   ├── masks/
   │   ├── image1.png
   │   └── ...
   └── classes.json
"""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image, ImageOps

from alveoleye._dataset_utils import detect_dataset_structure

# =============================================================================
# Constants
# =============================================================================

# Color tolerance for mask RGB matching (accounts for compression artifacts)
COLOR_TOLERANCE = 15

# Minimum blob size (in pixels) to include in masks
MIN_BLOB_SIZE = 15

# Minimum bounding box dimension (width/height) in pixels
MIN_BOX_DIMENSION = 2

# Default validation split fraction for flat datasets
DEFAULT_VAL_SPLIT = 0.2

# Default random seed for reproducible splits
DEFAULT_SEED = 42

# =============================================================================
# Logging
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# Dataset Class
# =============================================================================

class LungDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for lung segmentation training data.

    Loads image-mask pairs from a structured directory, supporting both
    pre-split (train/val subdirectories) and flat (auto-split) formats.

    Attributes:
        root: Path to the dataset root directory.
        train: Whether this is a training dataset (vs validation).
        imgs: List of image filenames.
        masks: List of mask filenames.
        class_dict: Mapping of RGB color strings to class IDs.
    """

    def __init__(
        self,
        root: str,
        transforms: Any,
        train: bool,
        n_img_load: Optional[int] = None,
        img_extension: str = ".png",
        n_repeat_images: int = 1,
        val_split: float = DEFAULT_VAL_SPLIT,
        seed: int = DEFAULT_SEED,
    ) -> None:
        """Initialize the LungDataset.

        Args:
            root: Path to dataset directory.
            transforms: Transforms to apply to images and targets.
            train: If True, load training data; if False, load validation data.
            n_img_load: Maximum number of images to load (optional).
            img_extension: File extension for images (default: ".png").
            n_repeat_images: Number of times to repeat the dataset.
            val_split: Fraction for validation when using flat structure.
            seed: Random seed for reproducible splits.

        Raises:
            ValueError: If dataset structure is invalid or image/mask counts don't match.
        """
        self.root = root
        self.transforms = transforms
        self.train = train
        self._loaded_images: Dict[int, Tuple[Image.Image, Dict[str, Any]]] = {}

        # Detect dataset structure using shared utility
        structure = detect_dataset_structure(root, img_extension)

        if structure == "split":
            self._init_split_structure(img_extension)
        elif structure == "flat":
            self._init_flat_structure(img_extension, val_split, seed)
        else:
            raise ValueError(
                f"Invalid dataset structure in {root}. Expected either:\n"
                "  1. Split structure: images/train/, images/val/, masks/train/, masks/val/\n"
                "  2. Flat structure: images/*.png, masks/*.png"
            )

        # Apply repetition
        self.imgs = self.imgs * n_repeat_images
        self.masks = self.masks * n_repeat_images

        # Limit images if requested
        if n_img_load is not None:
            self.imgs = self.imgs[:n_img_load]
            self.masks = self.masks[:n_img_load]

        # Validate counts match
        if len(self.imgs) != len(self.masks):
            raise ValueError(
                f"Image/mask count mismatch: {len(self.imgs)} images, "
                f"{len(self.masks)} masks. Check dataset structure."
            )

        # Load class definitions
        self.class_dict = self._load_classes(self.root)

        logger.info(
            f"Loaded {'training' if train else 'validation'} dataset: "
            f"{len(self.imgs)} images from {root}"
        )

    def _init_split_structure(self, img_extension: str) -> None:
        """Initialize from pre-split train/val directories."""
        self.folder = "train" if self.train else "val"
        self.use_flat = False

        images_dir = os.path.join(self.root, "images", self.folder)
        masks_dir = os.path.join(self.root, "masks", self.folder)

        self.imgs = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(img_extension)
        ])
        self.masks = sorted([
            f for f in os.listdir(masks_dir)
            if f.endswith(img_extension)
        ])

    def _init_flat_structure(
        self,
        img_extension: str,
        val_split: float,
        seed: int,
    ) -> None:
        """Initialize from flat structure with auto-split."""
        self.folder = None
        self.use_flat = True

        images_dir = os.path.join(self.root, "images")
        masks_dir = os.path.join(self.root, "masks")

        all_imgs = sorted([
            f for f in os.listdir(images_dir)
            if f.endswith(img_extension)
        ])
        all_masks = sorted([
            f for f in os.listdir(masks_dir)
            if f.endswith(img_extension)
        ])

        if len(all_imgs) != len(all_masks):
            raise ValueError(
                f"Image/mask count mismatch in flat structure: "
                f"{len(all_imgs)} images, {len(all_masks)} masks"
            )

        # Create reproducible split
        n_total = len(all_imgs)
        n_val = max(1, int(n_total * val_split))
        n_train = n_total - n_val

        # Shuffle indices with seed
        indices = list(range(n_total))
        rng = random.Random(seed)
        rng.shuffle(indices)

        train_indices = sorted(indices[:n_train])
        val_indices = sorted(indices[n_train:])

        if self.train:
            self.imgs = [all_imgs[i] for i in train_indices]
            self.masks = [all_masks[i] for i in train_indices]
        else:
            self.imgs = [all_imgs[i] for i in val_indices]
            self.masks = [all_masks[i] for i in val_indices]

    def _load_classes(self, root: str) -> Dict[str, int]:
        """Load class color definitions from classes.json.

        Args:
            root: Dataset root directory.

        Returns:
            Dictionary mapping RGB color strings to class IDs (1-indexed).
        """
        classes_path = os.path.join(root, "classes.json")

        with open(classes_path, "r") as f:
            colors = json.load(f)

        class_colors = {}
        for number, name in enumerate(colors):
            class_colors[colors[name]] = number + 1

        return class_colors

    def _get_image_paths(self, idx: int) -> Tuple[str, str]:
        """Get image and mask paths for an index."""
        if self.use_flat:
            img_path = os.path.join(self.root, "images", self.imgs[idx])
            mask_path = os.path.join(self.root, "masks", self.masks[idx])
        else:
            img_path = os.path.join(self.root, "images", self.folder, self.imgs[idx])
            mask_path = os.path.join(self.root, "masks", self.folder, self.masks[idx])

        return img_path, mask_path

    def __getitem__(self, idx: int) -> Tuple[Any, Dict[str, Any]]:
        """Get a single training example.

        Args:
            idx: Index of the example to retrieve.

        Returns:
            Tuple of (image, target) where target contains boxes, labels,
            masks, and other detection annotations.
        """
        # Check cache
        if idx in self._loaded_images:
            img, target = self._loaded_images[idx]
            if self.transforms is not None:
                img, target = self.transforms(img, target)
            return img, target

        # Load image and mask
        img_path, mask_path = self._get_image_paths(idx)

        img = Image.open(img_path).convert("RGB")
        img = ImageOps.mirror(img)

        masks, labels = self._rgb_to_class_mask_list(mask_path, self.class_dict)

        # Calculate bounding boxes and filter invalid ones
        num_objs = len(labels)
        boxes = []
        exclude_indices = []

        for i in range(num_objs):
            pos = np.nonzero(masks[i])
            if len(pos[0]) == 0 or len(pos[1]) == 0:
                exclude_indices.append(i)
                continue

            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])

            # Filter boxes with dimensions too small
            if (xmax - xmin < MIN_BOX_DIMENSION) or (ymax - ymin < MIN_BOX_DIMENSION):
                exclude_indices.append(i)

        # Apply exclusions
        boxes = np.array([
            box for i, box in enumerate(boxes)
            if i not in exclude_indices
        ])
        labels = np.array([
            label for i, label in enumerate(labels)
            if i not in exclude_indices
        ])
        masks = np.array([
            mask for i, mask in enumerate(masks)
            if i not in exclude_indices
        ])

        # Convert to tensors
        boxes = torch.as_tensor(boxes, dtype=torch.float32) if len(boxes) > 0 else torch.zeros((0, 4), dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)

        image_id = torch.tensor([idx])

        if len(boxes) > 0:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        else:
            area = torch.tensor([0], dtype=torch.float32)

        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "masks": masks,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        # Cache the loaded data
        self._loaded_images[idx] = (img, target)

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def _rgb_to_class_mask_list(
        self,
        mask_path: str,
        class_colors: Dict[str, int],
    ) -> Tuple[List[np.ndarray], List[int]]:
        """Convert RGB mask to list of binary masks per instance.

        Args:
            mask_path: Path to the RGB mask image.
            class_colors: Dictionary mapping RGB color strings to class IDs.

        Returns:
            Tuple of (masks, labels) where masks is a list of binary masks
            and labels is the corresponding class IDs.
        """
        mask_img = np.array(Image.open(mask_path).convert("RGB"))

        # Handle RGBA images
        if mask_img.shape[-1] == 4:
            mask_img = mask_img[:, :, :3]

        components = []
        num_ids = []

        logger.debug(f"Loading mask: {mask_path}")

        for color_str, class_id in class_colors.items():
            # Parse color string "[R G B]" -> [R, G, B]
            color_str = color_str.replace("[", "").replace("]", "")
            color = [int(v) for v in color_str.split()]

            # Create mask with color tolerance
            lower = np.clip(np.array(color) - COLOR_TOLERANCE, 0, 255)
            upper = np.clip(np.array(color) + COLOR_TOLERANCE, 0, 255)
            mask = cv2.inRange(mask_img, lower, upper)
            mask = np.where(mask == 255, 1, 0).astype(np.uint8)

            # Find connected components
            num_labels, labeled = cv2.connectedComponents(mask)

            # Create individual masks for each blob
            blob_masks = [
                (labeled == i).astype(np.uint8)
                for i in range(1, num_labels)
            ]

            # Filter small blobs
            blob_masks = [
                blob for blob in blob_masks
                if np.sum(blob > 0) > MIN_BLOB_SIZE
            ]

            components.extend(blob_masks)
            num_ids.extend([class_id] * len(blob_masks))

        # Stack masks or return empty array
        if components:
            final_mask = np.stack(components, axis=0)
        else:
            final_mask = np.zeros((0,) + mask_img.shape[:2], dtype=np.uint8)

        return final_mask, num_ids

    def __len__(self) -> int:
        """Return the number of examples in the dataset."""
        return len(self.imgs)
