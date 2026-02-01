#!/usr/bin/env python
"""Determine the optimal number of training images for maximum performance.

This script trains models with increasing numbers of images (50, 100, 150, etc.)
until the validation loss meets a target threshold, helping identify the minimum
dataset size needed for effective training.

Usage:
    # Run with default dataset location (src/training_dataset/)
    python -m alveoleye.paper_scripts.optimal_training_size

    # Download dataset from Google Drive and run
    python -m alveoleye.paper_scripts.optimal_training_size --download-dataset

    # Use custom dataset location
    python -m alveoleye.paper_scripts.optimal_training_size /path/to/dataset

    # With custom parameters
    python -m alveoleye.paper_scripts.optimal_training_size /path/to/dataset \
        --threshold 0.5 \
        --step-size 50 \
        --start-images 50 \
        --max-images 500 \
        --epochs 100 \
        --output-dir ./results
"""

import argparse
import csv
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alveoleye.lungcv.mrcnn.api import train, TrainingResult
from alveoleye.lungcv.mrcnn.config import AugmentationConfig
from alveoleye.paper_scripts._utils import (
    download_training_dataset,
    find_training_dataset,
    detect_dataset_structure,
    count_dataset_images,
    DEFAULT_TRAINING_DATASET_DIR,
)


@dataclass
class TrainingRun:
    """Results from a single training run."""
    n_images: int
    best_val_loss: float
    best_val_f1: float
    best_val_precision: float
    best_val_recall: float
    best_val_f1_agnostic: float
    final_epoch: int
    training_time: float
    meets_threshold: bool


def validate_dataset(dataset_path: str, val_split: float = 0.2) -> Tuple[bool, str]:
    """Validate that the dataset has the required structure.

    Supports two structures:
    1. Split structure:
        dataset_path/
            images/train/, images/val/
            masks/train/, masks/val/
            classes.json

    2. Flat structure:
        dataset_path/
            images/*.png
            masks/*.png
            classes.json

    Args:
        dataset_path: Path to the dataset directory
        val_split: Fraction for validation split (used for flat structure)

    Returns:
        Tuple of (is_valid, error_message)
    """
    path = Path(dataset_path)

    if not path.exists():
        return False, f"Dataset path does not exist: {dataset_path}"

    if not (path / "classes.json").exists():
        return False, "Missing classes.json file"

    structure = detect_dataset_structure(path)

    if structure is None:
        return False, (
            "Invalid dataset structure. Expected either:\n"
            "  1. Split: images/train/, images/val/, masks/train/, masks/val/\n"
            "  2. Flat: images/*.png, masks/*.png"
        )

    # Count available training images
    n_images = count_dataset_images(path, split="train", val_split=val_split)

    if n_images == 0:
        return False, "No training images found"

    structure_desc = "split (train/val)" if structure == "split" else "flat (auto-split)"
    return True, f"Found {n_images} training images ({structure_desc} structure)"


def run_training_experiment(
    dataset_path: str,
    n_images: int,
    epochs: int,
    device: str,
    seed: Optional[int],
    save_dir: Optional[str],
    val_split: float = 0.2,
) -> TrainingRun:
    """Run a single training experiment with a specified number of images.

    Args:
        dataset_path: Path to the dataset
        n_images: Number of images to use for training
        epochs: Number of training epochs
        device: Device to train on
        seed: Random seed for reproducibility
        save_dir: Directory to save checkpoints (None to disable)

    Returns:
        TrainingRun with the results
    """
    print(f"\n{'='*60}")
    print(f"Training with {n_images} images")
    print(f"{'='*60}")

    start_time = time.time()

    # Disable checkpoint saving during experiments unless explicitly requested
    result = train(
        dataset_path=dataset_path,
        epochs=epochs,
        n_images=n_images,
        device=device,
        seed=seed,
        val_split=val_split,
        save_dir=save_dir if save_dir else ".",
        save_frequency=0 if not save_dir else 50,  # Disable periodic saves if no save_dir
        save_best=bool(save_dir),  # Only save best if save_dir provided
        use_tensorboard=False,  # Disable tensorboard for cleaner output
        print_freq=50,  # Less verbose output
        compute_metrics=True,  # Enable F1 metrics
    )

    training_time = time.time() - start_time

    return TrainingRun(
        n_images=n_images,
        best_val_loss=result.best_val_loss,
        best_val_f1=result.best_val_f1,
        best_val_precision=result.best_val_precision,
        best_val_recall=result.best_val_recall,
        best_val_f1_agnostic=result.best_val_f1_agnostic,
        final_epoch=result.final_epoch,
        training_time=training_time,
        meets_threshold=False,  # Will be set by caller
    )


def run_optimal_size_experiment(
    dataset_path: str,
    threshold: float,
    start_images: int,
    step_size: int,
    max_images: Optional[int],
    epochs: int,
    device: str,
    seed: Optional[int],
    save_dir: Optional[str],
    val_split: float = 0.2,
    metric: str = "f1",
) -> Tuple[List[TrainingRun], Optional[int]]:
    """Run the full experiment to find optimal training size.

    Args:
        dataset_path: Path to the dataset
        threshold: Target metric threshold (F1 score by default)
        start_images: Initial number of images to train with
        step_size: Increment for each subsequent training run
        max_images: Maximum images to try (None = use all available)
        epochs: Number of training epochs per run
        device: Device to train on
        seed: Random seed for reproducibility
        save_dir: Directory to save results and checkpoints
        val_split: Validation split fraction
        metric: Metric to use for threshold ('f1', 'f1_agnostic', 'loss')

    Returns:
        Tuple of (list of all TrainingRuns, optimal number of images or None)
    """
    available_images = count_dataset_images(dataset_path, split="train", val_split=val_split)

    if max_images is None:
        max_images = available_images
    else:
        max_images = min(max_images, available_images)

    # Determine metric description
    metric_desc = {
        'f1': 'F1 (class-aware)',
        'f1_agnostic': 'F1 (class-agnostic)',
        'loss': 'validation loss',
    }.get(metric, metric)

    print(f"\n{'#'*60}")
    print(f"# Optimal Training Size Experiment")
    print(f"{'#'*60}")
    print(f"  Dataset: {dataset_path}")
    print(f"  Available training images: {available_images}")
    print(f"  Target metric: {metric_desc} >= {threshold}")
    print(f"  Image range: {start_images} to {max_images} (step: {step_size})")
    print(f"  Epochs per run: {epochs}")
    print(f"  Device: {device}")
    print(f"  Seed: {seed}")
    print(f"{'#'*60}\n")

    results: List[TrainingRun] = []
    optimal_n_images: Optional[int] = None

    n_images = start_images
    while n_images <= max_images:
        # Run training
        run = run_training_experiment(
            dataset_path=dataset_path,
            n_images=n_images,
            epochs=epochs,
            device=device,
            seed=seed,
            save_dir=save_dir,
            val_split=val_split,
        )

        # Get the metric value for comparison
        if metric == 'f1':
            metric_value = run.best_val_f1
        elif metric == 'f1_agnostic':
            metric_value = run.best_val_f1_agnostic
        elif metric == 'loss':
            metric_value = run.best_val_loss
        else:
            metric_value = run.best_val_f1  # Default to class-aware F1

        # Check if threshold is met (for F1 metrics, higher is better; for loss, lower is better)
        if metric == 'loss':
            run.meets_threshold = metric_value <= threshold
        else:
            run.meets_threshold = metric_value >= threshold
        results.append(run)

        print(f"\n  Results for {n_images} images:")
        print(f"    Best validation loss: {run.best_val_loss:.6f}")
        print(f"    Best F1 (class-aware): {run.best_val_f1:.4f}")
        print(f"    Best F1 (class-agnostic): {run.best_val_f1_agnostic:.4f}")
        print(f"    Precision: {run.best_val_precision:.4f}, Recall: {run.best_val_recall:.4f}")
        print(f"    Training time: {run.training_time:.1f}s")
        print(f"    Meets threshold ({metric_desc} >= {threshold}): {'YES' if run.meets_threshold else 'NO'}")

        if run.meets_threshold:
            optimal_n_images = n_images
            print(f"\n  ** Threshold met with {n_images} images! **")
            break

        n_images += step_size

    return results, optimal_n_images


def export_results(
    results: List[TrainingRun],
    optimal_n_images: Optional[int],
    output_path: str,
    threshold: float,
    metric: str = "f1",
) -> str:
    """Export experiment results to a CSV file.

    Args:
        results: List of TrainingRun results
        optimal_n_images: The optimal number found (or None)
        output_path: Directory to save the results
        threshold: The target threshold used
        metric: The metric used for threshold

    Returns:
        Path to the exported file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"optimal_training_size_{timestamp}.csv"
    filepath = Path(output_path) / filename

    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header with metadata
        writer.writerow(["# Optimal Training Size Experiment Results"])
        writer.writerow([f"# Metric: {metric}"])
        writer.writerow([f"# Threshold: {threshold}"])
        writer.writerow([f"# Optimal N Images: {optimal_n_images if optimal_n_images else 'Not found'}"])
        writer.writerow([])

        # Write data header
        writer.writerow([
            "n_images",
            "best_val_loss",
            "best_val_f1",
            "best_val_precision",
            "best_val_recall",
            "best_val_f1_agnostic",
            "final_epoch",
            "training_time_seconds",
            "meets_threshold",
        ])

        # Write data rows
        for run in results:
            writer.writerow([
                run.n_images,
                f"{run.best_val_loss:.6f}",
                f"{run.best_val_f1:.4f}",
                f"{run.best_val_precision:.4f}",
                f"{run.best_val_recall:.4f}",
                f"{run.best_val_f1_agnostic:.4f}",
                run.final_epoch,
                f"{run.training_time:.1f}",
                run.meets_threshold,
            ])

    return str(filepath)


def print_summary(
    results: List[TrainingRun],
    optimal_n_images: Optional[int],
    threshold: float,
    metric: str = "f1",
):
    """Print a summary of the experiment results."""
    print(f"\n{'='*80}")
    print("EXPERIMENT SUMMARY")
    print(f"{'='*80}")

    print(f"\n{'N Images':<10} {'Val Loss':<10} {'F1':<8} {'F1 Agnos':<10} {'Prec':<8} {'Recall':<8} {'Time(s)':<8} {'Pass'}")
    print("-" * 80)
    for run in results:
        status = "YES" if run.meets_threshold else "NO"
        print(f"{run.n_images:<10} {run.best_val_loss:<10.4f} {run.best_val_f1:<8.4f} "
              f"{run.best_val_f1_agnostic:<10.4f} {run.best_val_precision:<8.4f} "
              f"{run.best_val_recall:<8.4f} {run.training_time:<8.1f} {status}")

    # Determine metric description
    metric_desc = {
        'f1': 'F1 (class-aware)',
        'f1_agnostic': 'F1 (class-agnostic)',
        'loss': 'validation loss',
    }.get(metric, metric)

    print(f"\n{'-'*80}")
    if optimal_n_images:
        print(f"OPTIMAL NUMBER OF IMAGES: {optimal_n_images}")
        print(f"  (First to achieve {metric_desc} >= {threshold})")
    else:
        print(f"Threshold ({metric_desc} >= {threshold}) was NOT met with any tested configuration.")
        if results:
            best_run = max(results, key=lambda r: r.best_val_f1)
            print(f"  Best result: {best_run.n_images} images with F1 {best_run.best_val_f1:.4f}")

    total_time = sum(r.training_time for r in results)
    print(f"\nTotal experiment time: {total_time:.1f}s ({total_time/60:.1f} minutes)")


def validate_arguments(args) -> None:
    """Validate command line arguments."""
    is_valid, message = validate_dataset(args.dataset_path)
    if not is_valid:
        raise ValueError(f"Dataset validation failed: {message}")
    print(f"[+] Dataset validation passed: {message}")

    if args.threshold <= 0:
        raise ValueError("Threshold must be positive")

    if args.start_images < 1:
        raise ValueError("start-images must be at least 1")

    if args.step_size < 1:
        raise ValueError("step-size must be at least 1")

    if args.epochs < 1:
        raise ValueError("epochs must be at least 1")

    if args.max_images is not None and args.max_images < args.start_images:
        raise ValueError("max-images must be >= start-images")


def create_parser() -> argparse.ArgumentParser:
    """Create the argument parser."""
    parser = argparse.ArgumentParser(
        description="Determine the optimal number of training images for maximum performance.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Run with default dataset (src/training_dataset/)
    python -m alveoleye.paper_scripts.optimal_training_size

    # Download dataset first, then run
    python -m alveoleye.paper_scripts.optimal_training_size --download-dataset

    # Use custom dataset location
    python -m alveoleye.paper_scripts.optimal_training_size /path/to/dataset

    # Custom threshold and step size
    python -m alveoleye.paper_scripts.optimal_training_size --threshold 0.3 --step-size 25

    # Full customization
    python -m alveoleye.paper_scripts.optimal_training_size /path/to/dataset \\
        --threshold 0.5 --start-images 25 --step-size 25 --max-images 200 \\
        --epochs 50 --seed 42 --output-dir ./results
        """,
    )

    parser.add_argument(
        "dataset_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the dataset directory (must contain images/, masks/, and classes.json). "
             "Defaults to src/training_dataset/. When used with --download-dataset, this is "
             "the parent directory where the dataset will be downloaded.",
    )

    parser.add_argument(
        "--download-dataset",
        action="store_true",
        help="Download the training dataset from Google Drive before running the experiment. "
             "Downloads to dataset_path if provided, otherwise to src/training_dataset/.",
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Target metric threshold to achieve (default: 0.7 for F1)",
    )

    parser.add_argument(
        "--metric",
        type=str,
        default="f1",
        choices=["f1", "f1_agnostic", "loss"],
        help="Metric to use for threshold comparison (default: f1 = class-aware F1 score)",
    )

    parser.add_argument(
        "--start-images",
        type=int,
        default=50,
        help="Initial number of images to train with (default: 50)",
    )

    parser.add_argument(
        "--step-size",
        type=int,
        default=50,
        help="Number of images to add for each subsequent run (default: 50)",
    )

    parser.add_argument(
        "--max-images",
        type=int,
        default=None,
        help="Maximum number of images to try (default: all available)",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs per run (default: 100)",
    )

    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu", "mps"],
        help="Device to train on (default: auto)",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Fraction of data for validation when using flat dataset structure (default: 0.2 for 80/20 train/val split)",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results CSV and checkpoints (optional)",
    )

    parser.add_argument(
        "--save-checkpoints",
        action="store_true",
        help="Save model checkpoints for each run (requires --output-dir)",
    )

    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Handle dataset download if requested
    if args.download_dataset:
        try:
            downloaded_path = download_training_dataset(
                output_dir=args.dataset_path if args.dataset_path else None,
                quiet=False,
            )
            args.dataset_path = downloaded_path
        except Exception as e:
            print(f"[-] Error downloading dataset: {e}")
            sys.exit(1)

    # Use default dataset path if not provided
    if args.dataset_path is None:
        found_path = find_training_dataset()
        if found_path:
            args.dataset_path = found_path
            print(f"[+] Found dataset at: {args.dataset_path}")
        else:
            print(f"[-] Error: No dataset found in {DEFAULT_TRAINING_DATASET_DIR}")
            print("    Run with --download-dataset to download it, or provide a path.")
            sys.exit(1)

    try:
        validate_arguments(args)
    except ValueError as e:
        print(f"[-] Error: {e}")
        sys.exit(1)

    # Determine save directory for checkpoints
    save_dir = None
    if args.save_checkpoints:
        if not args.output_dir:
            print("[-] Error: --save-checkpoints requires --output-dir")
            sys.exit(1)
        save_dir = args.output_dir

    try:
        # Run the experiment
        results, optimal_n_images = run_optimal_size_experiment(
            dataset_path=args.dataset_path,
            threshold=args.threshold,
            start_images=args.start_images,
            step_size=args.step_size,
            max_images=args.max_images,
            epochs=args.epochs,
            device=args.device,
            seed=args.seed,
            save_dir=save_dir,
            val_split=args.val_split,
            metric=args.metric,
        )

        # Print summary
        print_summary(results, optimal_n_images, args.threshold, metric=args.metric)

        # Export results if output directory specified
        if args.output_dir:
            filepath = export_results(
                results=results,
                optimal_n_images=optimal_n_images,
                output_path=args.output_dir,
                threshold=args.threshold,
                metric=args.metric,
            )
            print(f"\n[+] Results exported to: {filepath}")

    except KeyboardInterrupt:
        print("\n\n[-] Experiment interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n[-] Error during experiment: {e}")
        raise


if __name__ == "__main__":
    main()
