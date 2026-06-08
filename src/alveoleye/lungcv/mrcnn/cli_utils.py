"""CLI utility functions for training.

This module provides validation and helper functions for the training CLI.

Functions:
    validate_arguments: Validate CLI arguments before training
    parse_image_range: Parse 'start:end' string to tuple
    format_time: Format seconds as human-readable string
"""

import os
from typing import Tuple


def validate_arguments(args) -> None:
    """Validate CLI arguments before training.

    Args:
        args: Parsed argparse Namespace

    Raises:
        ValueError: If any argument is invalid
    """
    # Dataset validation
    if not os.path.isdir(args.dataset_path):
        raise ValueError(f"Dataset path does not exist: {args.dataset_path}")

    required_subdirs = ["images/train", "images/val", "masks/train", "masks/val"]
    for subdir in required_subdirs:
        path = os.path.join(args.dataset_path, subdir)
        if not os.path.isdir(path):
            raise ValueError(f"Dataset missing required subdirectory: {subdir}")

    classes_path = os.path.join(args.dataset_path, "classes.json")
    if not os.path.isfile(classes_path):
        raise ValueError("Dataset missing classes.json file")

    # Config file validation
    if args.config and not os.path.isfile(args.config):
        raise ValueError(f"Config file does not exist: {args.config}")

    # Resume checkpoint validation
    if args.resume_from and not os.path.isfile(args.resume_from):
        raise ValueError(f"Checkpoint file does not exist: {args.resume_from}")

    # Image range parsing and validation
    if args.image_range:
        try:
            start, end = parse_image_range(args.image_range)
            if start < 0:
                raise ValueError("Image range start must be non-negative")
            if end < start:
                raise ValueError("Image range end must be >= start")
        except ValueError as e:
            if "must be" in str(e):
                raise
            raise ValueError(
                f"Invalid image range format: '{args.image_range}'. "
                "Use 'start:end' format (e.g., '0:50')"
            )

    # Save directory validation
    if args.save_dir:
        parent = os.path.dirname(os.path.abspath(args.save_dir)) or "."
        if not os.path.isdir(parent):
            raise ValueError(f"Parent directory for save_dir does not exist: {parent}")

    # Augmentation config validation
    if hasattr(args, 'augmentation_config') and args.augmentation_config:
        if not os.path.isfile(args.augmentation_config):
            raise ValueError(
                f"Augmentation config file does not exist: {args.augmentation_config}"
            )

    # Numeric validation
    if args.epochs is not None and args.epochs < 1:
        raise ValueError("epochs must be at least 1")

    if args.batch_size is not None and args.batch_size < 1:
        raise ValueError("batch_size must be at least 1")

    if args.lr is not None and args.lr <= 0:
        raise ValueError("lr must be positive")

    if args.n_images is not None and args.n_images < 1:
        raise ValueError("n_images must be at least 1")


def parse_image_range(range_str: str) -> Tuple[int, int]:
    """Parse 'start:end' string to tuple of integers.

    Args:
        range_str: String in format 'start:end'

    Returns:
        Tuple of (start, end) integers

    Raises:
        ValueError: If format is invalid

    Examples:
        >>> parse_image_range("0:50")
        (0, 50)
        >>> parse_image_range("10:100")
        (10, 100)
    """
    if ':' not in range_str:
        raise ValueError("Invalid range format: missing ':'")

    parts = range_str.split(":")
    if len(parts) != 2:
        raise ValueError("Invalid range format: expected 'start:end'")

    try:
        start = int(parts[0].strip())
        end = int(parts[1].strip())
    except ValueError:
        raise ValueError("Invalid range format: start and end must be integers")

    return start, end


def format_time(seconds: float) -> str:
    """Format seconds as human-readable string.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted string like "1h 23m 45s" or "45.2s"

    Examples:
        >>> format_time(45.2)
        '45.2s'
        >>> format_time(3725)
        '1h 2m 5s'
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        return f"{hours}h {minutes}m {secs}s"


def print_arguments(args) -> None:
    """Print CLI arguments in a formatted way.

    Args:
        args: Parsed argparse Namespace
    """
    print("\n" + "=" * 60)
    print("Training Configuration")
    print("=" * 60)

    print(f"\nDataset:        {args.dataset_path}")

    if args.config:
        print(f"Config file:    {args.config}")

    print(f"\nTraining:")
    print(f"  Epochs:       {args.epochs}")
    print(f"  Batch size:   {args.batch_size}")
    print(f"  Device:       {args.device}")
    if args.seed is not None:
        print(f"  Seed:         {args.seed}")
    if args.mixed_precision:
        print(f"  Mixed prec:   Enabled")

    print(f"\nOptimizer:")
    print(f"  Type:         {args.optimizer}")
    print(f"  LR:           {args.lr}")
    print(f"  Momentum:     {args.momentum}")
    print(f"  Weight decay: {args.weight_decay}")

    print(f"\nScheduler:")
    print(f"  Type:         {args.scheduler}")
    if args.scheduler == "step":
        print(f"  Step size:    {args.step_size}")
    print(f"  Gamma:        {args.gamma}")
    if args.warmup_epochs > 0:
        print(f"  Warmup:       {args.warmup_epochs} epochs")

    if args.n_images or args.image_range:
        print(f"\nImage Selection:")
        if args.n_images:
            print(f"  Random N:     {args.n_images}")
        if args.image_range:
            print(f"  Range:        {args.image_range}")

    print(f"\nCheckpointing:")
    print(f"  Save dir:     {args.save_dir}")
    print(f"  Frequency:    Every {args.save_frequency} epochs")
    print(f"  Save best:    {not args.no_save_best}")

    print(f"\nLogging:")
    print(f"  TensorBoard:  {not args.no_tensorboard}")
    if args.log_dir:
        print(f"  Log dir:      {args.log_dir}")
    print(f"  Print freq:   {args.print_freq}")

    if args.resume_from:
        print(f"\nResume from:    {args.resume_from}")

    print("\n" + "=" * 60 + "\n")
