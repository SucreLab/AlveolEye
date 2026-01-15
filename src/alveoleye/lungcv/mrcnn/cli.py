"""Command-line interface for Mask R-CNN training.

This module provides a CLI for training Mask R-CNN models using the
training API. It supports both direct argument specification and
YAML configuration files.

Usage:
    alveoleye-train /path/to/dataset --epochs 100 --lr 0.001
    alveoleye-train /path/to/dataset --config training_config.yaml
    alveoleye-train /path/to/dataset --optimizer adamw --scheduler cosine

Example:
    # Simple training
    alveoleye-train my_dataset --epochs 100

    # With optimizer settings
    alveoleye-train my_dataset --optimizer adamw --lr 0.001 --weight-decay 0.01

    # Resume from checkpoint
    alveoleye-train my_dataset --resume-from checkpoints/epoch_50.pth
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

from alveoleye.lungcv.mrcnn.cli_utils import (
    validate_arguments,
    parse_image_range,
    format_time,
    print_arguments,
)
from alveoleye.lungcv.mrcnn.config import (
    TrainingConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    AugmentationConfig,
    CheckpointConfig,
    LoggingConfig,
    ImageSelectionConfig,
)
from alveoleye.lungcv.mrcnn.api import train


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser for the training CLI.

    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        description="Train a Mask R-CNN model for lung segmentation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "dataset_path",
        type=str,
        help="Path to the dataset directory",
    )

    # Configuration file support
    config_group = parser.add_argument_group("Configuration")
    config_group.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML config file (CLI args override config values)",
    )
    config_group.add_argument(
        "--save-config",
        type=str,
        default=None,
        help="Save effective configuration to YAML file before training",
    )

    # Training parameters
    train_group = parser.add_argument_group("Training")
    train_group.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    train_group.add_argument(
        "--num-classes",
        type=int,
        default=3,
        help="Number of output classes",
    )
    train_group.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to use for training",
    )
    train_group.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility",
    )
    train_group.add_argument(
        "--mixed-precision",
        action="store_true",
        help="Enable automatic mixed precision training",
    )
    train_group.add_argument(
        "--gradient-clip-val",
        type=float,
        default=None,
        help="Max gradient norm for clipping",
    )

    # Data parameters
    data_group = parser.add_argument_group("Data")
    data_group.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for training",
    )
    data_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers",
    )
    data_group.add_argument(
        "--n-repeat-images",
        type=int,
        default=1,
        help="Number of times to repeat images in dataset",
    )
    data_group.add_argument(
        "--img-extension",
        type=str,
        default=".png",
        help="Image file extension",
    )
    data_group.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation when using flat dataset structure (default: 0.2 for 80/20 train/val split)",
    )

    # Image selection (mutually exclusive)
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--n-images",
        type=int,
        default=None,
        help="Randomly select N images for training",
    )
    selection_group.add_argument(
        "--image-range",
        type=str,
        default=None,
        help="Use images in range 'start:end' (inclusive)",
    )

    # Optimizer parameters
    optim_group = parser.add_argument_group("Optimizer")
    optim_group.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam", "adamw", "rmsprop"],
        help="Optimizer type",
    )
    optim_group.add_argument(
        "--lr",
        type=float,
        default=0.005,
        help="Learning rate",
    )
    optim_group.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Momentum for SGD/RMSprop",
    )
    optim_group.add_argument(
        "--weight-decay",
        type=float,
        default=0.0005,
        help="Weight decay (L2 penalty)",
    )

    # Scheduler parameters
    sched_group = parser.add_argument_group("Scheduler")
    sched_group.add_argument(
        "--scheduler",
        type=str,
        default="step",
        choices=["step", "cosine", "plateau", "none"],
        help="Learning rate scheduler type",
    )
    sched_group.add_argument(
        "--step-size",
        type=int,
        default=20,
        help="Period of learning rate decay for StepLR",
    )
    sched_group.add_argument(
        "--gamma",
        type=float,
        default=0.1,
        help="Multiplicative factor for LR decay",
    )
    sched_group.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of warmup epochs",
    )

    # Augmentation
    aug_group = parser.add_argument_group("Augmentation")
    aug_group.add_argument(
        "--no-augmentation",
        action="store_true",
        help="Disable all data augmentations",
    )
    aug_group.add_argument(
        "--augmentation-config",
        type=str,
        default=None,
        help="Path to YAML file with augmentation configuration",
    )

    # Checkpointing
    ckpt_group = parser.add_argument_group("Checkpointing")
    ckpt_group.add_argument(
        "--save-dir",
        type=str,
        default=".",
        help="Directory to save checkpoints",
    )
    ckpt_group.add_argument(
        "--save-frequency",
        type=int,
        default=50,
        help="Save checkpoint every N epochs (0 to disable)",
    )
    ckpt_group.add_argument(
        "--no-save-best",
        action="store_true",
        help="Don't save the best model based on validation loss",
    )

    # Logging
    log_group = parser.add_argument_group("Logging")
    log_group.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging",
    )
    log_group.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="Directory for TensorBoard logs",
    )
    log_group.add_argument(
        "--print-freq",
        type=int,
        default=10,
        help="Print progress every N batches",
    )

    # Resume training
    resume_group = parser.add_argument_group("Resume")
    resume_group.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    return parser


def build_config_from_args(args) -> TrainingConfig:
    """Build a TrainingConfig from parsed CLI arguments.

    Args:
        args: Parsed argparse Namespace

    Returns:
        TrainingConfig instance
    """
    # Image selection
    image_selection = None
    if args.n_images is not None:
        image_selection = ImageSelectionConfig(
            n_random=args.n_images,
            seed=args.seed,
        )
    elif args.image_range is not None:
        start, end = parse_image_range(args.image_range)
        image_selection = ImageSelectionConfig(
            index_range=(start, end),
            seed=args.seed,
        )

    # Data config
    data_config = DataConfig(
        dataset_path=args.dataset_path,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        n_repeat_images=args.n_repeat_images,
        img_extension=args.img_extension,
        image_selection=image_selection,
        val_split=args.val_split,
    )

    # Optimizer config
    optimizer_config = OptimizerConfig(
        name=args.optimizer,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    # Scheduler config
    scheduler_config = SchedulerConfig(
        name=args.scheduler,
        step_size=args.step_size,
        gamma=args.gamma,
        warmup_epochs=args.warmup_epochs,
    )

    # Augmentation config
    if args.no_augmentation:
        augmentation_config = AugmentationConfig.none()
    elif args.augmentation_config:
        # Load from file
        import yaml
        with open(args.augmentation_config, 'r') as f:
            aug_data = yaml.safe_load(f)
        from alveoleye.lungcv.mrcnn.config import AugmentationItem
        augmentation_config = AugmentationConfig(
            enabled=aug_data.get('enabled', True),
            augmentations=[
                AugmentationItem(**a) if isinstance(a, dict) else a
                for a in aug_data.get('augmentations', [])
            ]
        )
    else:
        augmentation_config = AugmentationConfig.default()

    # Checkpoint config
    checkpoint_config = CheckpointConfig(
        save_dir=args.save_dir,
        save_frequency=args.save_frequency,
        save_best=not args.no_save_best,
    )

    # Logging config
    logging_config = LoggingConfig(
        use_tensorboard=not args.no_tensorboard,
        log_dir=args.log_dir,
        print_freq=args.print_freq,
    )

    # Main config
    return TrainingConfig(
        epochs=args.epochs,
        num_classes=args.num_classes,
        device=args.device,
        seed=args.seed,
        mixed_precision=args.mixed_precision,
        gradient_clip_val=args.gradient_clip_val,
        data=data_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        augmentation=augmentation_config,
        checkpoint=checkpoint_config,
        logging=logging_config,
    )


def load_config(args) -> TrainingConfig:
    """Load configuration from file or build from args.

    If a config file is specified, it's loaded first and then
    CLI arguments override the file values.

    Args:
        args: Parsed argparse Namespace

    Returns:
        TrainingConfig instance
    """
    if args.config:
        # Load from file
        config = TrainingConfig.from_yaml(args.config)

        # Override with CLI arguments
        config.data.dataset_path = args.dataset_path

        # Only override if explicitly provided (not default)
        # We check against parser defaults
        parser = create_parser()
        defaults = vars(parser.parse_args([args.dataset_path]))

        if args.epochs != defaults['epochs']:
            config.epochs = args.epochs
        if args.num_classes != defaults['num_classes']:
            config.num_classes = args.num_classes
        if args.device != defaults['device']:
            config.device = args.device
        if args.seed is not None:
            config.seed = args.seed
        if args.mixed_precision:
            config.mixed_precision = True
        if args.gradient_clip_val is not None:
            config.gradient_clip_val = args.gradient_clip_val

        # Data overrides
        if args.batch_size != defaults['batch_size']:
            config.data.batch_size = args.batch_size
        if args.num_workers != defaults['num_workers']:
            config.data.num_workers = args.num_workers
        if args.val_split != defaults['val_split']:
            config.data.val_split = args.val_split

        # Optimizer overrides
        if args.optimizer != defaults['optimizer']:
            config.optimizer.name = args.optimizer
        if args.lr != defaults['lr']:
            config.optimizer.lr = args.lr
        if args.momentum != defaults['momentum']:
            config.optimizer.momentum = args.momentum
        if args.weight_decay != defaults['weight_decay']:
            config.optimizer.weight_decay = args.weight_decay

        # Scheduler overrides
        if args.scheduler != defaults['scheduler']:
            config.scheduler.name = args.scheduler
        if args.step_size != defaults['step_size']:
            config.scheduler.step_size = args.step_size
        if args.gamma != defaults['gamma']:
            config.scheduler.gamma = args.gamma
        if args.warmup_epochs != defaults['warmup_epochs']:
            config.scheduler.warmup_epochs = args.warmup_epochs

        # Checkpoint overrides
        if args.save_dir != defaults['save_dir']:
            config.checkpoint.save_dir = args.save_dir
        if args.save_frequency != defaults['save_frequency']:
            config.checkpoint.save_frequency = args.save_frequency
        if args.no_save_best:
            config.checkpoint.save_best = False

        # Logging overrides
        if args.no_tensorboard:
            config.logging.use_tensorboard = False
        if args.log_dir is not None:
            config.logging.log_dir = args.log_dir
        if args.print_freq != defaults['print_freq']:
            config.logging.print_freq = args.print_freq

        # Image selection (always override if specified)
        if args.n_images is not None or args.image_range is not None:
            if args.n_images is not None:
                config.data.image_selection = ImageSelectionConfig(
                    n_random=args.n_images,
                    seed=args.seed,
                )
            else:
                start, end = parse_image_range(args.image_range)
                config.data.image_selection = ImageSelectionConfig(
                    index_range=(start, end),
                    seed=args.seed,
                )

        return config
    else:
        return build_config_from_args(args)


def run_training(args) -> None:
    """Run training with the given arguments.

    Args:
        args: Parsed argparse Namespace
    """
    # Build config
    config = load_config(args)

    # Save config if requested
    if args.save_config:
        config.to_yaml(args.save_config)
        print(f"[CLI] Configuration saved to: {args.save_config}")

    # Run training
    print("[CLI] Starting training...")
    start_time = time.time()

    result = train(
        config=config,
        resume_from=args.resume_from,
    )

    elapsed = time.time() - start_time
    print(f"\n[CLI] Training completed in {format_time(elapsed)}")
    print(f"[CLI] Best validation loss: {result.best_val_loss:.4f}")
    print(f"[CLI] Final epoch: {result.final_epoch + 1}/{config.epochs}")

    # Save final model
    final_path = Path(config.checkpoint.save_dir) / "final_model.pth"
    result.save(final_path)


def main() -> None:
    """Main entry point for the CLI."""
    parser = create_parser()
    args = parser.parse_args()

    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

    print_arguments(args)

    try:
        run_training(args)
    except KeyboardInterrupt:
        print("\n[CLI] Training interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\nError during training: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
