"""Model initialization and inference operations for lung segmentation.

This module provides functions for initializing, loading, and running
inference with Mask R-CNN models for lung tissue segmentation.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from packaging.version import Version

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

    # Ensure v2 transforms also update masks/boxes by wrapping into tv_tensors
    try:
        from alveoleye.lungcv.mrcnn.transforms import WrapTvTensors, UnwrapTvTensors  # type: ignore
        wrap_supported = True
    except Exception:
        wrap_supported = False

    if wrap_supported:
        transform_list.append(WrapTvTensors())

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

    if wrap_supported:
        transform_list.append(UnwrapTvTensors())

    return T.Compose(transform_list)


# =============================================================================
# Model Initialization
# =============================================================================

def load_checkpoint(path: Union[str, Path], device: torch.device) -> Any:
    """Load a PyTorch checkpoint robustly.
    
    Handles:
    - weights_only=True (PyTorch >= 1.12)
    - Fallback to weights_only=False for full model pickles
    - DDP-wrapped models saved as full models (requires process group init)
    """
    path = str(path)
    
    # 1. Try weights_only=True (safest, works for state_dict)
    try:
        return torch.load(path, map_location=device, weights_only=True)  # type: ignore[call-arg]
    except Exception:
        pass

    # 2. Try weights_only=False (explicitly for PyTorch 2.4+ compatibility)
    last_err = None
    try:
        return torch.load(path, map_location=device, weights_only=False)  # type: ignore[call-arg]
    except TypeError:
        # Older PyTorch: weights_only not supported
        try:
            return torch.load(path, map_location=device)
        except Exception as e:
            last_err = e
    except Exception as e:
        last_err = e
    
    if last_err is not None:
        # 3. Check for process group error (DDP model)
        err_msg = str(last_err)
        if "process group" in err_msg or "Default process group" in err_msg:
            import torch.distributed as dist
            if dist.is_available() and not dist.is_initialized():
                try:
                    # Initialize dummy process group to allow unpickling DDP models
                    dist.init_process_group(
                        backend='gloo',
                        rank=0,
                        world_size=1,
                        store=dist.HashStore()
                    )
                    # Retry load (must be weights_only=False since it's a DDP model)
                    try:
                        return torch.load(path, map_location=device, weights_only=False)  # type: ignore[call-arg]
                    except TypeError:
                        return torch.load(path, map_location=device)
                except Exception:
                    # If dummy init fails, we still raise the original error
                    pass
        
        raise last_err

    return None # Should not be reached


def convert_syncbn_to_bn(model: torch.nn.Module) -> torch.nn.Module:
    """Recursively convert SyncBatchNorm layers to BatchNorm2d layers.
    
    This is important when loading models trained with DistributedDataParallel (DDP)
    that used SyncBatchNorm for inference on a single device.
    
    Args:
        model: The PyTorch model to convert.
        
    Returns:
        The model with SyncBatchNorm layers replaced by BatchNorm2d.
    """
    for name, child in model.named_children():
        if hasattr(torch.nn, 'SyncBatchNorm') and isinstance(child, torch.nn.SyncBatchNorm):
            new_bn = torch.nn.BatchNorm2d(
                child.num_features,
                child.eps,
                child.momentum,
                child.affine,
                child.track_running_stats
            )
            # Copy parameters and buffers
            new_bn.load_state_dict(child.state_dict())
            setattr(model, name, new_bn)
        else:
            convert_syncbn_to_bn(child)
    return model


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
    # Note: Do not wrap with DataParallel here. DistributedDataParallel (DDP)
    # is initialized in the training loop when running with multiple GPUs.
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

            if Version(gdown.__version__) < Version("6.0.0"):
                gdown.download(
                    url=DEFAULT_WEIGHTS_DRIVE_URL,
                    output=str(weights_path),
                    fuzzy=True,
                )
            else:
                gdown.download(
                    url=DEFAULT_WEIGHTS_DRIVE_URL,
                    output=str(weights_path)
                )
    # Load weights robustly, avoiding unpickling full DDP-wrapped models
    checkpoint = load_checkpoint(weights_path, device)
    
    # Extract state dict
    if isinstance(checkpoint, dict):
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
    elif isinstance(checkpoint, torch.nn.Module):
        # Full model pickle (can happen with DDP saved whole models)
        state_dict = checkpoint.state_dict()
    else:
        state_dict = checkpoint

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
