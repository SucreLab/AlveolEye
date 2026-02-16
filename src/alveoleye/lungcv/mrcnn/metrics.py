"""Pixel-level segmentation metrics for training evaluation.

This module provides functions to compute pixel-level precision, recall, F1 score,
and IoU for instance segmentation models. Three types of metrics are computed:

1. Class-aware (primary): Pixel is a true positive only if mask AND class both match.
   This is the strictest metric and penalizes misclassification.

2. Class-agnostic: Combines all masks regardless of class. Answers "did we find
   the structures?" without considering class assignment.

3. Per-class: Separate metrics for each class (e.g., f1_class_1, f1_class_2).
   Useful for detailed per-class analysis.

Example:
    from alveoleye.lungcv.mrcnn.metrics import compute_pixel_metrics

    metrics = compute_pixel_metrics(
        pred_masks=predictions['masks'],
        pred_labels=predictions['labels'],
        gt_masks=targets['masks'],
        gt_labels=targets['labels'],
    )
    print(f"Class-aware F1: {metrics.f1_score}")
    print(f"Class-agnostic F1: {metrics.f1_agnostic}")
"""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import torch
from torch import Tensor


@dataclass
class SegmentationMetrics:
    """Container for all segmentation metrics."""
    # Class-aware (primary - strictest metric)
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    iou: float = 0.0

    # Class-agnostic (secondary)
    precision_agnostic: float = 0.0
    recall_agnostic: float = 0.0
    f1_agnostic: float = 0.0

    # Per-class: {class_id: {'precision': float, 'recall': float, 'f1': float}}
    per_class: Dict[int, Dict[str, float]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to a flat dictionary for logging."""
        result = {
            'f1': self.f1_score,
            'precision': self.precision,
            'recall': self.recall,
            'iou': self.iou,
            'f1_agnostic': self.f1_agnostic,
            'precision_agnostic': self.precision_agnostic,
            'recall_agnostic': self.recall_agnostic,
        }
        for cls_id, cls_metrics in self.per_class.items():
            for metric_name, value in cls_metrics.items():
                result[f'{metric_name}_class_{cls_id}'] = value
        return result


@dataclass
class _PixelCounts:
    """Internal container for raw pixel counts before metric computation."""
    tp_agnostic: int = 0
    fp_agnostic: int = 0
    fn_agnostic: int = 0
    tp_aware: int = 0
    fp_aware: int = 0
    fn_aware: int = 0
    per_class: Dict[int, Dict[str, int]] = field(default_factory=dict)


def _compute_precision_recall_f1(tp: int, fp: int, fn: int) -> Tuple[float, float, float]:
    """Compute precision, recall, and F1 from confusion matrix values."""
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def _compute_iou(tp: int, fp: int, fn: int) -> float:
    """Compute Intersection over Union."""
    union = tp + fp + fn
    return tp / union if union > 0 else 0.0


def _counts_to_metrics(counts: _PixelCounts) -> SegmentationMetrics:
    """Convert raw pixel counts to SegmentationMetrics."""
    prec_agnostic, rec_agnostic, f1_agnostic = _compute_precision_recall_f1(
        counts.tp_agnostic, counts.fp_agnostic, counts.fn_agnostic
    )
    prec_aware, rec_aware, f1_aware = _compute_precision_recall_f1(
        counts.tp_aware, counts.fp_aware, counts.fn_aware
    )
    iou_aware = _compute_iou(counts.tp_aware, counts.fp_aware, counts.fn_aware)

    per_class_metrics: Dict[int, Dict[str, float]] = {}
    for cls_int, cls_counts in counts.per_class.items():
        prec, rec, f1 = _compute_precision_recall_f1(
            cls_counts['tp'], cls_counts['fp'], cls_counts['fn']
        )
        per_class_metrics[cls_int] = {'precision': prec, 'recall': rec, 'f1': f1}

    return SegmentationMetrics(
        precision=prec_aware,
        recall=rec_aware,
        f1_score=f1_aware,
        iou=iou_aware,
        precision_agnostic=prec_agnostic,
        recall_agnostic=rec_agnostic,
        f1_agnostic=f1_agnostic,
        per_class=per_class_metrics,
    )


def _compute_counts_for_image(
    pred_masks: Tensor,
    pred_labels: Tensor,
    gt_masks: Tensor,
    gt_labels: Tensor,
    threshold: float,
) -> _PixelCounts:
    """Compute raw pixel counts for a single image.

    This is the core logic shared by compute_pixel_metrics and compute_batch_metrics.
    """
    counts = _PixelCounts()

    # Normalize mask dimensions: [N, 1, H, W] -> [N, H, W]
    if pred_masks.dim() == 4:
        pred_masks = pred_masks.squeeze(1)

    H, W = gt_masks.shape[-2:]

    # Resize pred_masks if needed
    if pred_masks.shape[-2:] != (H, W):
        pred_masks = torch.nn.functional.interpolate(
            pred_masks.unsqueeze(1).float(),
            size=(H, W),
            mode='bilinear',
            align_corners=False,
        ).squeeze(1)

    pred_masks_binary = pred_masks > threshold

    # Class-agnostic: combine all masks
    pred_combined = pred_masks_binary.any(dim=0)
    gt_combined = gt_masks.any(dim=0)

    counts.tp_agnostic = (pred_combined & gt_combined).sum().item()
    counts.fp_agnostic = (pred_combined & ~gt_combined).sum().item()
    counts.fn_agnostic = (~pred_combined & gt_combined).sum().item()

    # Class-aware: build class maps
    pred_class_map = torch.zeros(H, W, dtype=torch.long, device=pred_masks.device)
    for mask, label in zip(pred_masks_binary, pred_labels):
        pred_class_map[mask] = label.item()

    gt_class_map = torch.zeros(H, W, dtype=torch.long, device=gt_masks.device)
    for mask, label in zip(gt_masks, gt_labels):
        gt_class_map[mask > 0] = label.item()

    pred_has_class = pred_class_map > 0
    gt_has_class = gt_class_map > 0
    classes_match = pred_class_map == gt_class_map

    counts.tp_aware = (pred_has_class & gt_has_class & classes_match).sum().item()
    counts.fp_aware = (pred_has_class & (~gt_has_class | ~classes_match)).sum().item()
    counts.fn_aware = (gt_has_class & (~pred_has_class | ~classes_match)).sum().item()

    # Per-class counts
    all_classes = set(gt_labels.tolist()) | set(pred_labels.tolist())
    for cls in all_classes:
        cls_int = int(cls)
        pred_cls_mask = pred_class_map == cls_int
        gt_cls_mask = gt_class_map == cls_int

        counts.per_class[cls_int] = {
            'tp': (pred_cls_mask & gt_cls_mask).sum().item(),
            'fp': (pred_cls_mask & ~gt_cls_mask).sum().item(),
            'fn': (~pred_cls_mask & gt_cls_mask).sum().item(),
        }

    return counts


def compute_pixel_metrics(
    pred_masks: Tensor,
    pred_labels: Tensor,
    gt_masks: Tensor,
    gt_labels: Tensor,
    threshold: float = 0.5,
) -> SegmentationMetrics:
    """Compute pixel-level segmentation metrics for a single image.

    Args:
        pred_masks: Predicted mask logits/probabilities [N, 1, H, W] or [N, H, W]
        pred_labels: Predicted class labels [N]
        gt_masks: Ground truth binary masks [M, H, W]
        gt_labels: Ground truth class labels [M]
        threshold: Threshold for binarizing predicted masks

    Returns:
        SegmentationMetrics with class-aware, class-agnostic, and per-class metrics
    """
    # Handle empty cases
    if pred_masks.numel() == 0 and gt_masks.numel() == 0:
        return SegmentationMetrics(
            precision=1.0, recall=1.0, f1_score=1.0, iou=1.0,
            precision_agnostic=1.0, recall_agnostic=1.0, f1_agnostic=1.0,
        )
    elif pred_masks.numel() == 0:
        return SegmentationMetrics(
            per_class={int(c): {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                       for c in gt_labels.unique().tolist()},
        )
    elif gt_masks.numel() == 0:
        return SegmentationMetrics(
            per_class={int(c): {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
                       for c in pred_labels.unique().tolist()},
        )

    counts = _compute_counts_for_image(pred_masks, pred_labels, gt_masks, gt_labels, threshold)
    return _counts_to_metrics(counts)


def _accumulate_counts(total: _PixelCounts, addition: _PixelCounts) -> None:
    """Add counts from one image to accumulated totals (in-place)."""
    total.tp_agnostic += addition.tp_agnostic
    total.fp_agnostic += addition.fp_agnostic
    total.fn_agnostic += addition.fn_agnostic
    total.tp_aware += addition.tp_aware
    total.fp_aware += addition.fp_aware
    total.fn_aware += addition.fn_aware

    for cls_int, cls_counts in addition.per_class.items():
        if cls_int not in total.per_class:
            total.per_class[cls_int] = {'tp': 0, 'fp': 0, 'fn': 0}
        for key in ('tp', 'fp', 'fn'):
            total.per_class[cls_int][key] += cls_counts[key]


def compute_batch_metrics(
    batch_predictions: List[Dict[str, Tensor]],
    batch_targets: List[Dict[str, Tensor]],
    threshold: float = 0.5,
) -> SegmentationMetrics:
    """Compute aggregated pixel-level metrics over a batch of images.

    Args:
        batch_predictions: List of dicts with 'masks' and 'labels' keys
        batch_targets: List of dicts with 'masks' and 'labels' keys
        threshold: Threshold for binarizing predicted masks

    Returns:
        SegmentationMetrics aggregated over all images
    """
    total_counts = _PixelCounts()

    for pred, target in zip(batch_predictions, batch_targets):
        pred_masks = pred.get('masks', torch.empty(0))
        pred_labels = pred.get('labels', torch.empty(0, dtype=torch.long))
        gt_masks = target.get('masks', torch.empty(0))
        gt_labels = target.get('labels', torch.empty(0, dtype=torch.long))

        # Both empty - skip
        if pred_masks.numel() == 0 and gt_masks.numel() == 0:
            continue

        # Handle edge cases where only one side is empty
        if pred_masks.numel() == 0:
            # All false negatives
            fn_pixels = int(gt_masks.sum().item())
            total_counts.fn_agnostic += fn_pixels
            total_counts.fn_aware += fn_pixels
            for label in gt_labels.unique().tolist():
                cls_int = int(label)
                if cls_int not in total_counts.per_class:
                    total_counts.per_class[cls_int] = {'tp': 0, 'fp': 0, 'fn': 0}
                total_counts.per_class[cls_int]['fn'] += int(gt_masks[gt_labels == label].sum().item())
            continue

        if gt_masks.numel() == 0:
            # All false positives
            if pred_masks.dim() == 4:
                pred_masks = pred_masks.squeeze(1)
            pred_binary = pred_masks > threshold
            fp_pixels = int(pred_binary.sum().item())
            total_counts.fp_agnostic += fp_pixels
            total_counts.fp_aware += fp_pixels
            for label in pred_labels.unique().tolist():
                cls_int = int(label)
                if cls_int not in total_counts.per_class:
                    total_counts.per_class[cls_int] = {'tp': 0, 'fp': 0, 'fn': 0}
                total_counts.per_class[cls_int]['fp'] += int(pred_binary[pred_labels == label].sum().item())
            continue

        # Normal case: both have data
        img_counts = _compute_counts_for_image(pred_masks, pred_labels, gt_masks, gt_labels, threshold)
        _accumulate_counts(total_counts, img_counts)

    return _counts_to_metrics(total_counts)
