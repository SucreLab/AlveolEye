#!/usr/bin/env python
"""Calculate metrics for a model on a set of matched images and masks.

This script takes a trained model and evaluates its performance on a dataset,
calculating the same metrics as those output in the 'optimal_training_size.py' script.

Usage:
    python -m alveoleye.paper_scripts.calculate_metrics /path/to/dataset --model-path /path/to/model.pth
"""

import argparse
import csv
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
from torch.utils.data import DataLoader

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from alveoleye.lungcv.mrcnn.dataset import LungDataset
from alveoleye.lungcv.mrcnn.utils import eval_with_metrics, collate_fn
from alveoleye.lungcv.model_operations import init_untrained_model, get_transform, get_device, convert_syncbn_to_bn, load_checkpoint
from alveoleye.paper_scripts._utils import find_training_dataset


@dataclass
class EvaluationResult:
    """Results from a single evaluation."""
    n_images: int
    loss: float
    f1: float
    precision: float
    recall: float
    f1_agnostic: float
    evaluation_time: float


def load_model(model_path: str, device: torch.device, num_classes: int) -> torch.nn.Module:
    """Load a trained model from a checkpoint.
    
    Args:
        model_path: Path to the model checkpoint
        device: Device to load the model on
        num_classes: Number of classes the model was trained with
        
    Returns:
        Loaded and evaluated model
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")

    # Load checkpoint robustly
    checkpoint = load_checkpoint(model_path, device)
    
    model = init_untrained_model(num_classes=num_classes)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            # Assume it's a state dict directly
            state_dict = checkpoint
    elif isinstance(checkpoint, torch.nn.Module):
        # Full model pickle (can happen with DDP saved whole models)
        state_dict = checkpoint.state_dict()
    else:
        raise ValueError(f"Unknown checkpoint format at {model_path}")

    # Remove 'module.' prefix if it exists (for models saved from DDP)
    new_state_dict = {
        (k[7:] if k.startswith('module.') else k): v 
        for k, v in state_dict.items()
    }
    
    model.load_state_dict(new_state_dict)
    
    # Convert SyncBatchNorm for inference
    model = convert_syncbn_to_bn(model)
    
    model.to(device)
    model.eval()
    return model


def print_summary(res: EvaluationResult):
    """Print a summary of the evaluation results."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    print(f"Number of images: {res.n_images}")
    print(f"Evaluation time:  {res.evaluation_time:.1f}s")
    print("-" * 80)
    print(f"{'Metric':<25} {'Value':<10}")
    print("-" * 36)
    print(f"{'Loss':<25} {res.loss:<10.4f}")
    print(f"{'F1 (class-aware)':<25} {res.f1:<10.4f}")
    print(f"{'F1 (class-agnostic)':<25} {res.f1_agnostic:<10.4f}")
    print(f"{'Precision':<25} {res.precision:<10.4f}")
    print(f"{'Recall':<25} {res.recall:<10.4f}")
    print(f"{'='*80}\n")


def export_results(res: EvaluationResult, output_path: str, threshold: float) -> str:
    """Export evaluation results to a CSV file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"evaluation_metrics_{timestamp}.csv"
    filepath = Path(output_path) / filename
    
    # Ensure output directory exists
    Path(output_path).mkdir(parents=True, exist_ok=True)

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)

        # Write header with metadata
        writer.writerow(["# Evaluation Metrics Results"])
        writer.writerow([f"# Threshold: {threshold}"])
        writer.writerow([f"# Evaluated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"])
        writer.writerow([])

        # Write data header
        writer.writerow([
            "n_images",
            "loss",
            "f1",
            "precision",
            "recall",
            "f1_agnostic",
            "evaluation_time_seconds",
        ])

        # Write data row
        writer.writerow([
            res.n_images,
            f"{res.loss:.6f}",
            f"{res.f1:.4f}",
            f"{res.precision:.4f}",
            f"{res.recall:.4f}",
            f"{res.f1_agnostic:.4f}",
            f"{res.evaluation_time:.1f}",
        ])

    return str(filepath)


def create_parser():
    """Create command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Calculate metrics for a model on a set of matched images and masks."
    )
    
    parser.add_argument(
        "dataset_path",
        type=str,
        nargs="?",
        default=None,
        help="Path to the dataset directory containing images and masks",
    )
    
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to the trained model checkpoint (.pth)",
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binarizing predicted masks (default: 0.5)",
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to run on ('auto', 'cuda', 'cpu', 'mps') (default: 'auto')",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save the results CSV (optional)",
    )
    
    parser.add_argument(
        "--img-extension",
        type=str,
        default=".png",
        help="Image file extension (default: '.png')",
    )
    
    return parser


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()

    # Use default dataset path if not provided
    if args.dataset_path is None:
        args.dataset_path = find_training_dataset()
        if args.dataset_path:
            print(f"[+] Found dataset at: {args.dataset_path}")
        else:
            print("[-] Error: No dataset found. Please provide a dataset path.")
            sys.exit(1)

    device = get_device() if args.device == "auto" else torch.device(args.device)
    print(f"[+] Using device: {device}")

    # Load dataset
    print(f"[+] Loading dataset from: {args.dataset_path}")
    
    # Initialize dataset using flat layout (no split)
    dataset = LungDataset(
        args.dataset_path, 
        transforms=get_transform(train=False), 
        train=True, 
        val_split=0.0,
        img_extension=args.img_extension
    )

    num_classes = len(dataset.class_dict) + 1
    print(f"[+] Evaluating on {len(dataset)} images")

    # Load model
    print(f"[+] Loading model from: {args.model_path}")
    try:
        model = load_model(args.model_path, device, num_classes)
    except Exception as e:
        print(f"[-] Error loading model: {e}")
        sys.exit(1)

    # Create DataLoader
    data_loader = DataLoader(
        dataset,
        batch_size=1, # Use batch size 1 for evaluation
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )

    print("[+] Running evaluation...")
    start_time = time.time()
    
    try:
        losses_dict, metrics = eval_with_metrics(
            model, 
            data_loader, 
            device, 
            threshold=args.threshold
        )
    except Exception as e:
        print(f"[-] Error during evaluation: {e}")
        sys.exit(1)
        
    end_time = time.time()
    eval_time = end_time - start_time

    # Aggregate losses
    total_loss = sum(losses_dict.values()).item() if isinstance(losses_dict, dict) else 0.0

    # Create result object
    result = EvaluationResult(
        n_images=len(dataset),
        loss=total_loss,
        f1=metrics.f1_score,
        precision=metrics.precision,
        recall=metrics.recall,
        f1_agnostic=metrics.f1_agnostic,
        evaluation_time=eval_time,
    )

    # Print summary
    print_summary(result)

    # Export results
    if args.output_dir:
        filepath = export_results(result, args.output_dir, args.threshold)
        print(f"[+] Results exported to: {filepath}")


if __name__ == "__main__":
    main()
