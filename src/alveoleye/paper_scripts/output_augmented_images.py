import argparse
import os
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from PIL import Image

# Ensure package imports work when running as a script
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alveoleye.lungcv.mrcnn.config import AugmentationConfig
from alveoleye.lungcv.mrcnn.augmentations import build_transforms
from alveoleye.lungcv.mrcnn.dataset import LungDataset, DEFAULT_SEED
from alveoleye.paper_scripts._utils import find_training_dataset


def _to_uint8_image(t: torch.Tensor) -> Image.Image:
    """Convert a float tensor [C,H,W] in 0..1 to a PIL Image."""
    t = t.detach().cpu().clamp(0, 1)
    t = (t * 255.0).to(torch.uint8)
    if t.ndim == 3 and t.shape[0] in (1, 3):
        arr = t.permute(1, 2, 0).numpy()
        if arr.shape[2] == 1:
            arr = arr[:, :, 0]
        return Image.fromarray(arr)
    elif t.ndim == 2:
        return Image.fromarray(t.numpy())
    else:
        raise ValueError(f"Unexpected tensor shape for image: {t.shape}")


def _merge_instance_masks(masks: torch.Tensor) -> torch.Tensor:
    """Merge instance masks (N,H,W) into a single binary mask (H,W)."""
    if masks.numel() == 0:
        # caller should provide H,W if needed; here we return empty tensor
        return torch.zeros((0, 0), dtype=torch.uint8)
    merged = masks.any(dim=0).to(torch.uint8)
    return merged


def save_augmented_samples(
    dataset_path: str,
    output_dir: str,
    num_samples: int,
    split: str = "train",
    seed: Optional[int] = None,
    img_extension: str = ".png",
) -> None:
    """Apply training augmentations and save augmented images and masks.

    Args:
        dataset_path: Path to dataset root (with images/ and masks/ folders).
        output_dir: Directory where augmented samples will be saved.
        num_samples: Number of augmented samples to save.
        split: 'train' or 'val' subset to sample from.
        seed: Optional random seed for reproducible sampling.
        img_extension: Image extension to look for (default: .png).
    """
    # Build the same augmentation pipeline as used in training
    aug_cfg = AugmentationConfig.default()
    transforms = build_transforms(aug_cfg, train=True, target_size=None)

    # Pick subset via LungDataset which applies transforms to both image and target
    use_train = (split == "train")
    ds_seed = seed if seed is not None else DEFAULT_SEED
    dataset = LungDataset(
        root=dataset_path,
        transforms=transforms,
        train=use_train,
        img_extension=img_extension,
        val_split=0.2,
        seed=ds_seed,
    )

    # Secondary dataset without augmentations to get the original images
    dataset_no_aug = LungDataset(
        root=dataset_path,
        transforms=None,
        train=use_train,
        img_extension=img_extension,
        val_split=0.2,
        seed=ds_seed,
    )

    # Prepare output directories
    images_out = Path(output_dir) / "images"
    masks_out = Path(output_dir) / "masks"
    orig_images_out = Path(output_dir) / "original_image"
    images_out.mkdir(parents=True, exist_ok=True)
    masks_out.mkdir(parents=True, exist_ok=True)
    orig_images_out.mkdir(parents=True, exist_ok=True)

    # Determine indices to sample
    indices = list(range(len(dataset)))
    if seed is not None:
        random.Random(seed).shuffle(indices)
    # If num_samples exceeds dataset, we cap it
    k = min(num_samples, len(indices))
    indices = indices[:k]

    for i, idx in enumerate(indices, start=1):
        img, target = dataset[idx]
        # img: float tensor [C,H,W] in 0..1; target['masks']: uint8 [N,H,W]
        img_pil = _to_uint8_image(img)

        # Get the original non-transformed image
        orig_img_pil, _ = dataset_no_aug[idx]

        masks = target.get("masks")
        if masks is None:
            # Create an empty mask with same spatial size
            H, W = img.shape[-2:]
            merged = torch.zeros((H, W), dtype=torch.uint8)
        else:
            merged = _merge_instance_masks(masks)
            # If empty (0x0), create zeros matching image size
            if merged.numel() == 0:
                H, W = img.shape[-2:]
                merged = torch.zeros((H, W), dtype=torch.uint8)

        mask_img = _to_uint8_image(merged)

        img_path = images_out / f"aug_{i:04d}.png"
        mask_path = masks_out / f"aug_{i:04d}.png"
        orig_img_path = orig_images_out / f"aug_{i:04d}.png"
        img_pil.save(img_path)
        mask_img.save(mask_path)
        orig_img_pil.save(orig_img_path)

    print(f"Saved {k} augmented samples to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply the same training augmentations to dataset samples and "
            "save the augmented images and masks to 'augment_out' (configurable)."
        )
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help=(
            "Path to the dataset root containing images/ and masks/ subfolders. "
            "If not provided, the script will try to auto-detect the training dataset."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="augment_out",
        help="Directory to save augmented samples (default: augment_out)",
    )
    parser.add_argument(
        "--num",
        type=int,
        default=100,
        help="Number of augmented samples to output (default: 16)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val"],
        help="Which split to sample from (default: train)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Optional random seed for reproducible sampling",
    )
    parser.add_argument(
        "--img_extension",
        type=str,
        default=".png",
        help="Image file extension used by the dataset (default: .png)",
    )

    args = parser.parse_args()

    dataset_path = args.dataset_path
    if dataset_path is None:
        detected = find_training_dataset()
        if detected is None:
            raise SystemExit(
                "Could not auto-detect training dataset. Please provide --dataset_path."
            )
        dataset_path = detected

    save_augmented_samples(
        dataset_path=str(dataset_path),
        output_dir=args.output_dir,
        num_samples=int(args.num),
        split=args.split,
        seed=args.seed,
        img_extension=args.img_extension,
    )


if __name__ == "__main__":
    main()
