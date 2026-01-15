"""Model initialization and inference operations for lung segmentation.

This module provides functions for initializing, loading, and running
inference with Mask R-CNN models for lung tissue segmentation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from PIL import Image
from torchvision.models.detection import MaskRCNN, maskrcnn_resnet50_fpn, MaskRCNN_ResNet50_FPN_Weights
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
from torchvision.transforms import v2 as T

# =============================================================================
# Constants
# =============================================================================

# Google Drive URL for default model weights
DEFAULT_WEIGHTS_DRIVE_URL = "https://drive.google.com/file/d/1LjmKvnzBfVsicHCvHccWYkMP3ouOx2m6/view?usp=sharing"

# Default path for model weights relative to this file
DEFAULT_WEIGHTS_PATH = Path(__file__).resolve().parent.parent.parent / "default_weights" / "default.pth"

# Default number of output classes (background + 2 tissue types)
DEFAULT_NUM_CLASSES = 3

# Hidden layer size for mask predictor
MASK_PREDICTOR_HIDDEN_LAYER = 256

# Default augmentation probability
DEFAULT_AUGMENTATION_PROBABILITY = 0.15


# =============================================================================
# Device Utilities
# =============================================================================

def get_device() -> torch.device:
    """Get the best available compute device.

    Returns:
        torch.device for CUDA if available, otherwise CPU.
        MPS (Apple Silicon) support is commented out pending testing.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    # elif torch.backends.mps.is_available():
    #     return torch.device("mps")
    return torch.device("cpu")


# =============================================================================
# Transforms
# =============================================================================

def get_transform(train: bool = True) -> T.Compose:
    """Get image transforms for training or inference.

    Args:
        train: If True, includes data augmentation transforms.

    Returns:
        Composed transform pipeline.
    """
    transform_list = [T.PILToTensor()]

    if train:
        prob = DEFAULT_AUGMENTATION_PROBABILITY
        transform_list.extend([
            T.RandomHorizontalFlip(prob),
            T.RandomVerticalFlip(prob),
            T.RandomApply([
                T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
            ], p=prob),
            T.RandomApply([T.RandomRotation(degrees=(-5, 5))], p=prob),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=prob),
            T.RandomApply([
                T.RandomAffine(degrees=0, translate=(0.2, 0.2), scale=(0.8, 1.2), shear=(-5, 5))
            ], p=prob),
        ])

    transform_list.extend([
        T.ToDtype(torch.float, scale=True),
        T.ToPureTensor(),
    ])

    return T.Compose(transform_list)


# =============================================================================
# Model Initialization
# =============================================================================

def init_untrained_model(num_classes: int = DEFAULT_NUM_CLASSES) -> MaskRCNN:
    """Initialize a Mask R-CNN model with custom number of classes.

    Creates a model pre-trained on COCO and replaces the prediction
    heads for the specified number of classes.

    Args:
        num_classes: Number of output classes including background.

    Returns:
        Initialized MaskRCNN model (not trained on custom data).
    """
    model = maskrcnn_resnet50_fpn(weights=MaskRCNN_ResNet50_FPN_Weights.COCO_V1)

    # Replace box predictor
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # Replace mask predictor
    in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    model.roi_heads.mask_predictor = MaskRCNNPredictor(
        in_features_mask,
        MASK_PREDICTOR_HIDDEN_LAYER,
        num_classes,
    )

    return model


def init_trained_model(
    model_path: Optional[Union[str, Path]] = None,
    num_classes: int = DEFAULT_NUM_CLASSES,
) -> MaskRCNN:
    """Initialize a trained Mask R-CNN model.

    Loads model weights from the specified path, or downloads default
    weights from Google Drive if not available locally.

    Args:
        model_path: Path to model weights file. If None or doesn't exist,
                   uses/downloads default weights.
        num_classes: Number of output classes including background.

    Returns:
        Trained MaskRCNN model ready for inference.
    """
    device = get_device()
    model = init_untrained_model(num_classes)

    # Determine weights path
    weights_path = Path(model_path) if model_path else DEFAULT_WEIGHTS_PATH

    if not weights_path.exists():
        weights_path = DEFAULT_WEIGHTS_PATH

        if not weights_path.exists():
            # Create directory and download weights
            weights_path.parent.mkdir(parents=True, exist_ok=True)

            try:
                import gdown
            except ImportError as e:
                raise ImportError(
                    "gdown is required for downloading model weights. "
                    "Install it with: pip install gdown"
                ) from e

            gdown.download(
                url=DEFAULT_WEIGHTS_DRIVE_URL,
                output=str(weights_path),
                fuzzy=True,
            )

    # Load weights
    loaded_model = torch.load(weights_path, map_location=device)
    model.load_state_dict(loaded_model.state_dict())
    model.to(device)

    return model


# =============================================================================
# Inference
# =============================================================================

def run_prediction(
    image_path: Union[str, Path],
    model: MaskRCNN,
) -> Dict[str, Any]:
    """Run inference on a single image.

    Args:
        image_path: Path to the input image.
        model: Trained MaskRCNN model.

    Returns:
        Dictionary containing prediction results with keys:
        - boxes: Bounding boxes [N, 4]
        - labels: Class labels [N]
        - scores: Confidence scores [N]
        - masks: Segmentation masks [N, 1, H, W]
    """
    device = get_device()

    image = T.PILToTensor()(Image.open(image_path).convert("RGB"))
    eval_transform = get_transform(train=False)

    model.eval()

    with torch.no_grad():
        x = eval_transform(image)
        x = x.to(device)
        predictions = model([x])
        prediction = predictions[0]
        del x

    torch.cuda.empty_cache()

    return prediction
