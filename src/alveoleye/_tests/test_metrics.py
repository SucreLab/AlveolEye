"""Tests for pixel-level segmentation metrics.

Tests cover:
- SegmentationMetrics dataclass
- compute_pixel_metrics function
- compute_batch_metrics function
- Edge cases (empty predictions, empty ground truth)
- Class-aware vs class-agnostic metrics
- Per-class metrics
"""

import pytest
import torch

from alveoleye.lungcv.mrcnn.metrics import (
    SegmentationMetrics,
    compute_pixel_metrics,
    compute_batch_metrics,
    _compute_precision_recall_f1,
    _compute_iou,
)


class TestComputePrecisionRecallF1:
    """Tests for _compute_precision_recall_f1 helper."""

    def test_perfect_prediction(self):
        """Test perfect prediction (all TP, no FP/FN)."""
        precision, recall, f1 = _compute_precision_recall_f1(tp=100, fp=0, fn=0)
        assert precision == 1.0
        assert recall == 1.0
        assert f1 == 1.0

    def test_no_predictions(self):
        """Test no predictions (all FN)."""
        precision, recall, f1 = _compute_precision_recall_f1(tp=0, fp=0, fn=100)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_all_false_positives(self):
        """Test all false positives (no TP, no FN)."""
        precision, recall, f1 = _compute_precision_recall_f1(tp=0, fp=100, fn=0)
        assert precision == 0.0
        assert recall == 0.0
        assert f1 == 0.0

    def test_mixed_case(self):
        """Test mixed case with TP, FP, and FN."""
        # TP=50, FP=25, FN=25
        # Precision = 50/75 = 0.667
        # Recall = 50/75 = 0.667
        # F1 = 2 * 0.667 * 0.667 / 1.334 = 0.667
        precision, recall, f1 = _compute_precision_recall_f1(tp=50, fp=25, fn=25)
        assert abs(precision - 2/3) < 0.01
        assert abs(recall - 2/3) < 0.01
        assert abs(f1 - 2/3) < 0.01

    def test_high_precision_low_recall(self):
        """Test high precision, low recall."""
        # TP=10, FP=0, FN=90
        precision, recall, f1 = _compute_precision_recall_f1(tp=10, fp=0, fn=90)
        assert precision == 1.0
        assert recall == 0.1
        assert 0.15 < f1 < 0.20  # F1 should be low


class TestComputeIoU:
    """Tests for _compute_iou helper."""

    def test_perfect_iou(self):
        """Test perfect IoU (all TP)."""
        iou = _compute_iou(tp=100, fp=0, fn=0)
        assert iou == 1.0

    def test_zero_iou(self):
        """Test zero IoU (no TP)."""
        iou = _compute_iou(tp=0, fp=50, fn=50)
        assert iou == 0.0

    def test_partial_iou(self):
        """Test partial IoU."""
        # TP=50, FP=25, FN=25 -> IoU = 50/100 = 0.5
        iou = _compute_iou(tp=50, fp=25, fn=25)
        assert iou == 0.5


class TestSegmentationMetrics:
    """Tests for SegmentationMetrics dataclass."""

    def test_default_values(self):
        """Test default values are zeros."""
        metrics = SegmentationMetrics()
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_score == 0.0
        assert metrics.iou == 0.0
        assert metrics.precision_agnostic == 0.0
        assert metrics.recall_agnostic == 0.0
        assert metrics.f1_agnostic == 0.0
        assert metrics.per_class == {}

    def test_to_dict(self):
        """Test to_dict conversion."""
        metrics = SegmentationMetrics(
            precision=0.8,
            recall=0.9,
            f1_score=0.85,
            iou=0.75,
            precision_agnostic=0.85,
            recall_agnostic=0.92,
            f1_agnostic=0.88,
            per_class={1: {'precision': 0.7, 'recall': 0.8, 'f1': 0.75}}
        )
        d = metrics.to_dict()

        assert d['precision'] == 0.8
        assert d['recall'] == 0.9
        assert d['f1'] == 0.85
        assert d['iou'] == 0.75
        assert d['precision_agnostic'] == 0.85
        assert d['recall_agnostic'] == 0.92
        assert d['f1_agnostic'] == 0.88
        assert d['precision_class_1'] == 0.7
        assert d['recall_class_1'] == 0.8
        assert d['f1_class_1'] == 0.75


class TestComputePixelMetrics:
    """Tests for compute_pixel_metrics function."""

    def test_perfect_prediction(self):
        """Test perfect prediction - masks and labels match exactly."""
        H, W = 32, 32

        # Ground truth: one mask of class 1
        gt_masks = torch.zeros(1, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0
        gt_labels = torch.tensor([1])

        # Prediction: exactly matches ground truth
        pred_masks = torch.zeros(1, 1, H, W)
        pred_masks[0, 0, 10:20, 10:20] = 1.0
        pred_labels = torch.tensor([1])

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        assert metrics.f1_score == 1.0
        assert metrics.precision == 1.0
        assert metrics.recall == 1.0
        assert metrics.f1_agnostic == 1.0

    def test_correct_mask_wrong_class(self):
        """Test correct mask location but wrong class - class-aware should be low."""
        H, W = 32, 32

        # Ground truth: mask of class 1
        gt_masks = torch.zeros(1, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0
        gt_labels = torch.tensor([1])

        # Prediction: same mask location but class 2
        pred_masks = torch.zeros(1, 1, H, W)
        pred_masks[0, 0, 10:20, 10:20] = 1.0
        pred_labels = torch.tensor([2])

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        # Class-agnostic should be perfect (we found the structure)
        assert metrics.f1_agnostic == 1.0

        # Class-aware should be zero (wrong class assignment)
        assert metrics.f1_score == 0.0
        assert metrics.precision == 0.0
        assert metrics.recall == 0.0

    def test_partial_overlap(self):
        """Test partial overlap between prediction and ground truth."""
        H, W = 32, 32

        # Ground truth: 10x10 region
        gt_masks = torch.zeros(1, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0
        gt_labels = torch.tensor([1])

        # Prediction: overlaps 50% (shifted by 5 pixels)
        pred_masks = torch.zeros(1, 1, H, W)
        pred_masks[0, 0, 15:25, 10:20] = 1.0
        pred_labels = torch.tensor([1])

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        # F1 should be between 0 and 1
        assert 0.0 < metrics.f1_score < 1.0
        assert 0.0 < metrics.f1_agnostic < 1.0

    def test_empty_predictions(self):
        """Test empty predictions with non-empty ground truth."""
        H, W = 32, 32

        gt_masks = torch.zeros(1, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0
        gt_labels = torch.tensor([1])

        pred_masks = torch.empty(0, 1, H, W)
        pred_labels = torch.empty(0, dtype=torch.long)

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        assert metrics.f1_score == 0.0
        assert metrics.recall == 0.0
        assert metrics.f1_agnostic == 0.0

    def test_empty_ground_truth(self):
        """Test non-empty predictions with empty ground truth."""
        H, W = 32, 32

        gt_masks = torch.empty(0, H, W)
        gt_labels = torch.empty(0, dtype=torch.long)

        pred_masks = torch.zeros(1, 1, H, W)
        pred_masks[0, 0, 10:20, 10:20] = 1.0
        pred_labels = torch.tensor([1])

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        assert metrics.f1_score == 0.0
        assert metrics.precision == 0.0
        assert metrics.f1_agnostic == 0.0

    def test_both_empty(self):
        """Test both predictions and ground truth empty."""
        H, W = 32, 32

        gt_masks = torch.empty(0, H, W)
        gt_labels = torch.empty(0, dtype=torch.long)

        pred_masks = torch.empty(0, 1, H, W)
        pred_labels = torch.empty(0, dtype=torch.long)

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        # Both empty = perfect match
        assert metrics.f1_score == 1.0
        assert metrics.f1_agnostic == 1.0

    def test_multiple_instances_same_class(self):
        """Test multiple instances of the same class."""
        H, W = 64, 64

        # Ground truth: two masks of class 1
        gt_masks = torch.zeros(2, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0
        gt_masks[1, 40:50, 40:50] = 1.0
        gt_labels = torch.tensor([1, 1])

        # Prediction: matches both perfectly
        pred_masks = torch.zeros(2, 1, H, W)
        pred_masks[0, 0, 10:20, 10:20] = 1.0
        pred_masks[1, 0, 40:50, 40:50] = 1.0
        pred_labels = torch.tensor([1, 1])

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        assert metrics.f1_score == 1.0
        assert metrics.f1_agnostic == 1.0

    def test_multiple_classes(self):
        """Test multiple different classes."""
        H, W = 64, 64

        # Ground truth: class 1 and class 2
        gt_masks = torch.zeros(2, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0  # class 1
        gt_masks[1, 40:50, 40:50] = 1.0  # class 2
        gt_labels = torch.tensor([1, 2])

        # Prediction: matches both perfectly
        pred_masks = torch.zeros(2, 1, H, W)
        pred_masks[0, 0, 10:20, 10:20] = 1.0
        pred_masks[1, 0, 40:50, 40:50] = 1.0
        pred_labels = torch.tensor([1, 2])

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        assert metrics.f1_score == 1.0
        assert 1 in metrics.per_class
        assert 2 in metrics.per_class
        assert metrics.per_class[1]['f1'] == 1.0
        assert metrics.per_class[2]['f1'] == 1.0

    def test_per_class_metrics(self):
        """Test per-class metrics with partial success."""
        H, W = 64, 64

        # Ground truth: class 1 and class 2
        gt_masks = torch.zeros(2, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0  # class 1
        gt_masks[1, 40:50, 40:50] = 1.0  # class 2
        gt_labels = torch.tensor([1, 2])

        # Prediction: only class 1 is correct, class 2 is missing
        pred_masks = torch.zeros(1, 1, H, W)
        pred_masks[0, 0, 10:20, 10:20] = 1.0
        pred_labels = torch.tensor([1])

        metrics = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)

        # Class 1 should be perfect
        assert metrics.per_class[1]['f1'] == 1.0

        # Class 2 should be zero (not predicted)
        assert metrics.per_class[2]['f1'] == 0.0

    def test_threshold_effect(self):
        """Test that threshold affects binarization."""
        H, W = 32, 32

        gt_masks = torch.zeros(1, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0
        gt_labels = torch.tensor([1])

        # Prediction with values below default threshold
        pred_masks = torch.zeros(1, 1, H, W)
        pred_masks[0, 0, 10:20, 10:20] = 0.3  # Below 0.5 threshold
        pred_labels = torch.tensor([1])

        # With default threshold (0.5), prediction is treated as empty
        metrics_default = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels)
        assert metrics_default.f1_score == 0.0

        # With lower threshold, prediction is detected
        metrics_low = compute_pixel_metrics(pred_masks, pred_labels, gt_masks, gt_labels, threshold=0.2)
        assert metrics_low.f1_score == 1.0

    def test_3d_vs_4d_pred_masks(self):
        """Test that both 3D [N,H,W] and 4D [N,1,H,W] pred_masks work."""
        H, W = 32, 32

        gt_masks = torch.zeros(1, H, W)
        gt_masks[0, 10:20, 10:20] = 1.0
        gt_labels = torch.tensor([1])

        # 4D prediction [N, 1, H, W]
        pred_masks_4d = torch.zeros(1, 1, H, W)
        pred_masks_4d[0, 0, 10:20, 10:20] = 1.0

        # 3D prediction [N, H, W]
        pred_masks_3d = torch.zeros(1, H, W)
        pred_masks_3d[0, 10:20, 10:20] = 1.0

        pred_labels = torch.tensor([1])

        metrics_4d = compute_pixel_metrics(pred_masks_4d, pred_labels, gt_masks, gt_labels)
        metrics_3d = compute_pixel_metrics(pred_masks_3d, pred_labels, gt_masks, gt_labels)

        assert metrics_4d.f1_score == metrics_3d.f1_score == 1.0


class TestComputeBatchMetrics:
    """Tests for compute_batch_metrics function."""

    def test_single_image_batch(self):
        """Test batch with single image."""
        H, W = 32, 32

        predictions = [{
            'masks': torch.ones(1, 1, H, W) * 0.9,
            'labels': torch.tensor([1]),
        }]
        targets = [{
            'masks': torch.ones(1, H, W),
            'labels': torch.tensor([1]),
        }]

        # Make masks match
        predictions[0]['masks'][0, 0, 10:20, 10:20] = 1.0
        predictions[0]['masks'][0, 0, :10, :] = 0.0
        predictions[0]['masks'][0, 0, 20:, :] = 0.0
        predictions[0]['masks'][0, 0, :, :10] = 0.0
        predictions[0]['masks'][0, 0, :, 20:] = 0.0

        targets[0]['masks'][:] = 0.0
        targets[0]['masks'][0, 10:20, 10:20] = 1.0

        metrics = compute_batch_metrics(predictions, targets)

        assert metrics.f1_score == 1.0
        assert metrics.f1_agnostic == 1.0

    def test_multiple_images_batch(self):
        """Test batch with multiple images."""
        H, W = 32, 32

        # Two images, both perfect
        predictions = [
            {
                'masks': torch.zeros(1, 1, H, W),
                'labels': torch.tensor([1]),
            },
            {
                'masks': torch.zeros(1, 1, H, W),
                'labels': torch.tensor([2]),
            },
        ]
        predictions[0]['masks'][0, 0, 5:15, 5:15] = 1.0
        predictions[1]['masks'][0, 0, 10:20, 10:20] = 1.0

        targets = [
            {
                'masks': torch.zeros(1, H, W),
                'labels': torch.tensor([1]),
            },
            {
                'masks': torch.zeros(1, H, W),
                'labels': torch.tensor([2]),
            },
        ]
        targets[0]['masks'][0, 5:15, 5:15] = 1.0
        targets[1]['masks'][0, 10:20, 10:20] = 1.0

        metrics = compute_batch_metrics(predictions, targets)

        assert metrics.f1_score == 1.0

    def test_empty_batch(self):
        """Test empty batch returns zeros."""
        metrics = compute_batch_metrics([], [])
        # Should return default zeros when no data
        assert isinstance(metrics, SegmentationMetrics)

    def test_mixed_empty_images(self):
        """Test batch with some empty predictions/targets."""
        H, W = 32, 32

        predictions = [
            {
                'masks': torch.zeros(1, 1, H, W),
                'labels': torch.tensor([1]),
            },
            {
                'masks': torch.empty(0, 1, H, W),  # Empty
                'labels': torch.empty(0, dtype=torch.long),
            },
        ]
        predictions[0]['masks'][0, 0, 5:15, 5:15] = 1.0

        targets = [
            {
                'masks': torch.zeros(1, H, W),
                'labels': torch.tensor([1]),
            },
            {
                'masks': torch.zeros(1, H, W),  # Has target but no prediction
                'labels': torch.tensor([1]),
            },
        ]
        targets[0]['masks'][0, 5:15, 5:15] = 1.0
        targets[1]['masks'][0, 10:20, 10:20] = 1.0

        metrics = compute_batch_metrics(predictions, targets)

        # First image is perfect, second has all FN
        # Should be between 0 and 1
        assert 0.0 < metrics.f1_score < 1.0

    def test_aggregation_across_images(self):
        """Test that metrics are correctly aggregated across images."""
        H, W = 32, 32

        # Image 1: 100 TP (10x10 mask)
        # Image 2: 100 FN (10x10 mask not predicted)
        predictions = [
            {
                'masks': torch.zeros(1, 1, H, W),
                'labels': torch.tensor([1]),
            },
            {
                'masks': torch.empty(0, 1, H, W),
                'labels': torch.empty(0, dtype=torch.long),
            },
        ]
        predictions[0]['masks'][0, 0, 0:10, 0:10] = 1.0

        targets = [
            {
                'masks': torch.zeros(1, H, W),
                'labels': torch.tensor([1]),
            },
            {
                'masks': torch.zeros(1, H, W),
                'labels': torch.tensor([1]),
            },
        ]
        targets[0]['masks'][0, 0:10, 0:10] = 1.0
        targets[1]['masks'][0, 0:10, 0:10] = 1.0

        metrics = compute_batch_metrics(predictions, targets)

        # TP=100, FP=0, FN=100
        # Precision = 100/100 = 1.0
        # Recall = 100/200 = 0.5
        # F1 = 2 * 1.0 * 0.5 / 1.5 = 0.667
        assert abs(metrics.precision - 1.0) < 0.01
        assert abs(metrics.recall - 0.5) < 0.01
        assert abs(metrics.f1_score - 2/3) < 0.01
