"""Main training API for Mask R-CNN.

This module provides the primary training interface supporting both simple
function calls with keyword arguments and configuration-based training.

Functions:
    train: Main training function

Classes:
    TrainingResult: Container for training results

Example (Simple):
    from alveoleye.lungcv.mrcnn import train

    result = train(
        dataset_path='my_dataset',
        epochs=100,
        lr=0.001,
        optimizer='adamw',
    )
    result.save('model.pth')

Example (Config-based):
    from alveoleye.lungcv.mrcnn import train, TrainingConfig, DataConfig

    config = TrainingConfig(
        epochs=200,
        data=DataConfig(dataset_path='my_dataset', batch_size=8),
    )
    result = train(config=config)
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any, Callable, Union
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision.utils import make_grid

# TensorBoard is optional
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except (ImportError, TypeError):
    SummaryWriter = None
    TENSORBOARD_AVAILABLE = False

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
from alveoleye.lungcv.mrcnn.callbacks import (
    Callback,
    CallbackList,
    TrainingState,
    ModelCheckpointCallback,
    LambdaCallback,
)
from alveoleye.lungcv.mrcnn.optimizers import create_optimizer, create_scheduler
from alveoleye.lungcv.mrcnn.augmentations import build_transforms
from alveoleye.lungcv.mrcnn.dataset import LungDataset, DEFAULT_SEED
from alveoleye.lungcv.mrcnn.utils import collate_fn, eval_forward, eval_with_metrics, SmoothedValue, _safe_torch_save, is_main_process, setup_for_distributed, get_rank
from alveoleye.lungcv.mrcnn.metrics import SegmentationMetrics
from alveoleye.lungcv.mrcnn.engine import train_one_epoch
from alveoleye.lungcv.model_operations import init_untrained_model, load_checkpoint
from PIL import Image
from collections import Counter


# ANSI formatting codes
class _Colors:
    BOLD = '\033[1m'
    RESET = '\033[0m'


def _format_number(n: int) -> str:
    """Format a number with commas for readability."""
    return f"{n:,}"


def _count_parameters(model: torch.nn.Module) -> Tuple[int, int]:
    """Count total and trainable parameters in a model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def _print_header(title: str) -> None:
    """Print a bold header."""
    print(f"\n{_Colors.BOLD}[ {title} ]{_Colors.RESET}")


def _print_kv(key: str, value: Any, indent: int = 2) -> None:
    """Print a key-value pair."""
    print(f"{' ' * indent}{key}: {value}")


@dataclass
class TrainingResult:
    """Container for training results.

    Attributes:
        model: The trained model
        history: Dictionary mapping metric names to lists of values per epoch
        best_val_loss: Best validation loss achieved during training
        best_val_f1: Best class-aware F1 score (strictest metric)
        best_val_precision: Best class-aware precision
        best_val_recall: Best class-aware recall
        best_val_f1_agnostic: Best class-agnostic F1 score
        config: The TrainingConfig used for training
        final_epoch: The final epoch number (may be less than total if early stopped)
    """
    model: torch.nn.Module
    history: Dict[str, List[float]] = field(default_factory=dict)
    best_val_loss: float = float('inf')
    best_val_f1: float = 0.0
    best_val_precision: float = 0.0
    best_val_recall: float = 0.0
    best_val_f1_agnostic: float = 0.0
    config: Optional[TrainingConfig] = None
    final_epoch: int = 0

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to a file.

        Args:
            path: Path to save the model
        """
        _safe_torch_save(self.model, path)
        print(f"  [+] Model saved: {path}")

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save a full checkpoint with model state dict and training info.

        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'best_val_f1': self.best_val_f1,
            'best_val_precision': self.best_val_precision,
            'best_val_recall': self.best_val_recall,
            'best_val_f1_agnostic': self.best_val_f1_agnostic,
            'final_epoch': self.final_epoch,
        }
        if self.config is not None:
            checkpoint['config'] = self.config.to_dict()
        _safe_torch_save(checkpoint, path)
        print(f"  [+] Checkpoint saved: {path}")


def _get_device(device_str: str) -> torch.device:
    """Determine the device to use for training.

    Args:
        device_str: Device specification ('auto', 'cuda', 'cpu', 'mps')

    Returns:
        torch.device instance
    """
    if device_str == 'auto':
        if torch.cuda.is_available():
            return torch.device('cuda')
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return torch.device('mps')
        else:
            return torch.device('cpu')
    return torch.device(device_str)


def _detect_target_size(dataset_path: str, img_extension: str = '.png') -> tuple:
    """Detect the most common image size in the dataset.

    Args:
        dataset_path: Path to the dataset directory
        img_extension: Image file extension

    Returns:
        Tuple of (height, width) for the most common size
    """
    from alveoleye._dataset_utils import detect_dataset_structure

    dataset_path = Path(dataset_path)
    structure = detect_dataset_structure(str(dataset_path), img_extension)

    # Find image directories based on structure
    if structure == "split":
        image_dirs = [
            dataset_path / "images" / "train",
            dataset_path / "images" / "val",
        ]
    else:
        image_dirs = [dataset_path / "images"]

    sizes = Counter()
    for img_dir in image_dirs:
        if img_dir.exists():
            for img_path in img_dir.glob(f"*{img_extension}"):
                with Image.open(img_path) as img:
                    # Store as (height, width) for PyTorch convention
                    sizes[(img.height, img.width)] += 1

    if not sizes:
        raise ValueError(f"No images found in dataset: {dataset_path}")

    # Get most common size
    most_common_size, count = sizes.most_common(1)[0]
    total_images = sum(sizes.values())

    if len(sizes) > 1:
        print(f"[Training] Found {len(sizes)} different image sizes:")
        for size, cnt in sizes.most_common():
            print(f"  {size[1]}x{size[0]}: {cnt} images")
        print(f"[Training] Resizing all images to {most_common_size[1]}x{most_common_size[0]} (most common)")

    return most_common_size


def _apply_image_selection(
    dataset: torch.utils.data.Dataset,
    config: Optional[ImageSelectionConfig]
) -> torch.utils.data.Dataset:
    """Apply image selection to dataset.

    Args:
        dataset: Original dataset
        config: Image selection configuration

    Returns:
        Original dataset or Subset with selected images
    """
    if config is None:
        return dataset

    total_images = len(dataset)

    if config.n_random is not None:
        # Random selection
        if config.seed is not None:
            generator = torch.Generator()
            generator.manual_seed(config.seed)
            indices = torch.randperm(total_images, generator=generator)[:config.n_random].tolist()
        else:
            indices = torch.randperm(total_images)[:config.n_random].tolist()
        return Subset(dataset, indices)

    elif config.index_range is not None:
        start, end = config.index_range
        end = min(end + 1, total_images)  # +1 to make it inclusive, clamp to dataset size
        indices = list(range(start, end))
        return Subset(dataset, indices)

    return dataset


def _build_config_from_kwargs(
    dataset_path: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_classes: Optional[int] = None,
    num_workers: Optional[int] = None,
    n_repeat_images: Optional[int] = None,
    img_extension: Optional[str] = None,
    val_split: Optional[float] = None,
    # Optimizer
    optimizer: Optional[str] = None,
    lr: Optional[float] = None,
    momentum: Optional[float] = None,
    weight_decay: Optional[float] = None,
    # Scheduler
    scheduler: Optional[str] = None,
    step_size: Optional[int] = None,
    gamma: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
    # Image selection
    n_images: Optional[int] = None,
    image_range: Optional[Tuple[int, int]] = None,
    # Augmentation
    augmentation: Optional[AugmentationConfig] = None,
    # Checkpointing
    save_dir: Optional[str] = None,
    save_frequency: Optional[int] = None,
    save_best: Optional[bool] = None,
    # Device and other
    device: Optional[str] = None,
    seed: Optional[int] = None,
    mixed_precision: Optional[bool] = None,
    gradient_clip_val: Optional[float] = None,
    # Logging
    use_tensorboard: Optional[bool] = None,
    log_dir: Optional[str] = None,
    print_freq: Optional[int] = None,
) -> TrainingConfig:
    """Build a TrainingConfig from keyword arguments.

    This allows users to specify only the parameters they want to change,
    with sensible defaults for everything else.
    """
    # Build sub-configs with defaults and overrides
    data_config = DataConfig()
    if dataset_path is not None:
        data_config.dataset_path = dataset_path
    if batch_size is not None:
        data_config.batch_size = batch_size
    if num_workers is not None:
        data_config.num_workers = num_workers
    if n_repeat_images is not None:
        data_config.n_repeat_images = n_repeat_images
    if img_extension is not None:
        data_config.img_extension = img_extension
    if val_split is not None:
        data_config.val_split = val_split

    # Image selection
    if n_images is not None or image_range is not None:
        data_config.image_selection = ImageSelectionConfig(
            n_random=n_images,
            index_range=image_range,
            seed=seed,  # Use main seed for reproducibility
        )

    # Optimizer config
    optimizer_config = OptimizerConfig()
    if optimizer is not None:
        optimizer_config.name = optimizer.lower()
    if lr is not None:
        optimizer_config.lr = lr
    if momentum is not None:
        optimizer_config.momentum = momentum
    if weight_decay is not None:
        optimizer_config.weight_decay = weight_decay

    # Scheduler config
    scheduler_config = SchedulerConfig()
    if scheduler is not None:
        scheduler_config.name = scheduler.lower()
    if step_size is not None:
        scheduler_config.step_size = step_size
    if gamma is not None:
        scheduler_config.gamma = gamma
    if warmup_epochs is not None:
        scheduler_config.warmup_epochs = warmup_epochs

    # Augmentation config
    aug_config = augmentation if augmentation is not None else AugmentationConfig.default()

    # Checkpoint config
    checkpoint_config = CheckpointConfig()
    if save_dir is not None:
        checkpoint_config.save_dir = save_dir
    if save_frequency is not None:
        checkpoint_config.save_frequency = save_frequency
    if save_best is not None:
        checkpoint_config.save_best = save_best

    # Logging config
    logging_config = LoggingConfig()
    if use_tensorboard is not None:
        logging_config.use_tensorboard = use_tensorboard
    if log_dir is not None:
        logging_config.log_dir = log_dir
    if print_freq is not None:
        logging_config.print_freq = print_freq

    # Main config
    config = TrainingConfig(
        data=data_config,
        optimizer=optimizer_config,
        scheduler=scheduler_config,
        augmentation=aug_config,
        checkpoint=checkpoint_config,
        logging=logging_config,
    )

    if epochs is not None:
        config.epochs = epochs
    if num_classes is not None:
        config.num_classes = num_classes
    if device is not None:
        config.device = device
    if seed is not None:
        config.seed = seed
    if mixed_precision is not None:
        config.mixed_precision = mixed_precision
    if gradient_clip_val is not None:
        config.gradient_clip_val = gradient_clip_val

    return config


def _run_training(
    config: TrainingConfig,
    callbacks: CallbackList,
    resume_from: Optional[str] = None,
    compute_metrics: bool = True,
) -> TrainingResult:
    """Run the training loop.

    Args:
        config: Complete training configuration
        callbacks: List of callbacks
        resume_from: Optional path to checkpoint to resume from
        compute_metrics: Whether to compute pixel-level F1 metrics during validation

    Returns:
        TrainingResult with trained model and metrics
    """
    # Set seed for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    # Determine device and (optionally) initialize distributed
    device = _get_device(config.device)
    distributed = False
    local_rank = 0
    world_size_env = os.environ.get("WORLD_SIZE")
    try:
        world_size = int(world_size_env) if world_size_env is not None else 1
    except ValueError:
        world_size = 1
    if world_size > 1 and torch.cuda.is_available():
        distributed = True
        # Initialize process group once
        if not torch.distributed.is_initialized():
            backend = "nccl"
            init_method = os.environ.get("DIST_URL", "env://")
            torch.distributed.init_process_group(backend=backend, init_method=init_method)
        # Resolve ranks
        if "LOCAL_RANK" in os.environ:
            local_rank = int(os.environ["LOCAL_RANK"])
        else:
            # Fallback: assume rank 0
            local_rank = 0
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        # Silence non-master prints
        setup_for_distributed(is_main_process())

    # Determine target size for resizing
    target_size = None
    if config.data.batch_size > 1:
        if config.data.target_size == 'auto':
            target_size = _detect_target_size(
                str(config.data.dataset_path),
                config.data.img_extension,
            )
        elif config.data.target_size is not None:
            target_size = config.data.target_size

    # Build transforms
    train_transforms = build_transforms(config.augmentation, train=True, target_size=target_size)
    val_transforms = build_transforms(config.augmentation, train=False, target_size=target_size)

    # Create datasets
    dataset_path = str(config.data.dataset_path)
    # Use default seed if not specified to ensure reproducible train/val splits
    dataset_seed = config.seed if config.seed is not None else DEFAULT_SEED
    dataset = LungDataset(
        root=dataset_path,
        transforms=train_transforms,
        train=True,
        n_repeat_images=config.data.n_repeat_images,
        img_extension=config.data.img_extension,
        val_split=config.data.val_split,
        seed=dataset_seed,
    )
    dataset_val = LungDataset(
        root=dataset_path,
        transforms=val_transforms,
        train=False,
        n_repeat_images=config.data.n_repeat_images,
        img_extension=config.data.img_extension,
        val_split=config.data.val_split,
        seed=dataset_seed,
    )

    # Apply image selection
    dataset = _apply_image_selection(dataset, config.data.image_selection)

    _print_header("DATA")
    _print_kv("Dataset", dataset_path)
    train_count = _format_number(len(dataset))
    if config.data.image_selection is not None:
        sel = config.data.image_selection
        if sel.n_random is not None:
            train_count += f" (random selection, seed={sel.seed})"
        elif sel.index_range is not None:
            train_count += f" (range {sel.index_range[0]}-{sel.index_range[1]})"
    _print_kv("Train images", train_count)
    _print_kv("Val images", _format_number(len(dataset_val)))
    _print_kv("Batch size", config.data.batch_size)
    aug_str = f"{len(config.augmentation.augmentations)} transforms" if config.augmentation.enabled else "Disabled"
    _print_kv("Augmentation", aug_str)

    # Create data loaders (use DistributedSampler if distributed)
    train_sampler = None
    val_sampler = None
    if 'torch' in globals():
        try:
            from torch.utils.data.distributed import DistributedSampler  # type: ignore
        except Exception:
            DistributedSampler = None  # type: ignore
    else:
        DistributedSampler = None  # type: ignore

    if distributed and DistributedSampler is not None:
        train_sampler = DistributedSampler(dataset, shuffle=True)
        # For validation, avoid DistributedSampler to keep evaluation logic simple and consistent.
        val_sampler = None

    data_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=(train_sampler is None),
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        sampler=train_sampler,
    )

    val_batch_size = config.data.val_batch_size or config.data.batch_size
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
        sampler=val_sampler,
    )

    # Initialize model
    model = init_untrained_model(config.num_classes)

    # Convert BatchNorm layers to SyncBatchNorm for multi-GPU distributed training
    if distributed and device.type == 'cuda':
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

    model.to(device)

    # Wrap with DistributedDataParallel if distributed
    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank] if device.type == 'cuda' else None
        )

    _, trainable_params = _count_parameters(model)

    _print_header("MODEL")
    _print_kv("Architecture", "Mask R-CNN (ResNet-50 FPN)")
    _print_kv("Classes", config.num_classes)
    _print_kv("Parameters", f"{_format_number(trainable_params)} trainable")

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer(params, config.optimizer)

    _print_header("TRAINING")
    device_str = str(device)
    if device.type == 'cuda':
        device_str = f"{device} ({torch.cuda.get_device_name(device)})"
    _print_kv("Device", device_str)
    _print_kv("Epochs", config.epochs)
    _print_kv("Optimizer", config.optimizer.name.upper())
    _print_kv("Learning rate", config.optimizer.lr)
    _print_kv("Weight decay", config.optimizer.weight_decay)
    _print_kv("Momentum", config.optimizer.momentum)

    # Create scheduler
    lr_scheduler = create_scheduler(optimizer, config.scheduler, config.epochs)

    _print_kv("Scheduler", config.scheduler.name.upper())
    if config.scheduler.name.lower() == 'step':
        _print_kv("Step size", f"{config.scheduler.step_size} epochs")
        _print_kv("Gamma", config.scheduler.gamma)
    elif config.scheduler.name.lower() == 'cosine':
        _print_kv("T_max", config.scheduler.T_max or config.epochs)
    elif config.scheduler.name.lower() == 'plateau':
        _print_kv("Patience", config.scheduler.patience)
    warmup_str = f"{config.scheduler.warmup_epochs} epochs" if config.scheduler.warmup_epochs > 0 else "None"
    _print_kv("Warmup", warmup_str)
    _print_kv("Gradient clip", config.gradient_clip_val if config.gradient_clip_val is not None else "None")
    _print_kv("Mixed precision", "Yes" if config.mixed_precision else "No")
    _print_kv("Seed", config.seed if config.seed is not None else "None")

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    best_val_f1 = 0.0
    best_val_precision = 0.0
    best_val_recall = 0.0
    best_val_f1_agnostic = 0.0
    history: Dict[str, List[float]] = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }
    # Add metrics history keys if computing metrics
    if compute_metrics:
        history.update({
            'val_f1': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1_agnostic': [],
        })

    if resume_from is not None:
        print(f"\n[+] Resuming from: {resume_from}")
        # Load checkpoint robustly
        checkpoint = load_checkpoint(resume_from, device)
        
        if isinstance(checkpoint, dict):
            model_sd = checkpoint.get('model_state_dict', checkpoint.get('state_dict'))
        elif isinstance(checkpoint, torch.nn.Module):
            model_sd = checkpoint.state_dict()
        else:
            model_sd = None

        if model_sd is None:
            raise ValueError(f"Invalid checkpoint format at {resume_from}: missing state dict.")
        
        # Remove optional 'module.' prefix
        model_sd = { (k[7:] if k.startswith('module.') else k): v for k, v in model_sd.items() }
        model.load_state_dict(model_sd)
        
        if isinstance(checkpoint, dict) and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if isinstance(checkpoint, dict) and 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        if isinstance(checkpoint, dict) and 'best_val_f1' in checkpoint:
            best_val_f1 = checkpoint['best_val_f1']
        if isinstance(checkpoint, dict) and 'best_val_precision' in checkpoint:
            best_val_precision = checkpoint['best_val_precision']
        if isinstance(checkpoint, dict) and 'best_val_recall' in checkpoint:
            best_val_recall = checkpoint['best_val_recall']
        if isinstance(checkpoint, dict) and 'best_val_f1_agnostic' in checkpoint:
            best_val_f1_agnostic = checkpoint['best_val_f1_agnostic']
        if isinstance(checkpoint, dict) and 'history' in checkpoint:
            history = checkpoint['history']
        print(f"    Epoch {start_epoch}, best val loss: {best_val_loss:.4f}")

    # Setup mixed precision if enabled
    scaler = None
    if config.mixed_precision:
        if torch.cuda.is_available():
            scaler = torch.cuda.amp.GradScaler()
        else:
            print(f"[-] Mixed precision unavailable (no CUDA)")

    # Setup TensorBoard (master process only in distributed mode)
    writer = None
    if config.logging.use_tensorboard and (not distributed or is_main_process()):
        if TENSORBOARD_AVAILABLE:
            log_dir = config.logging.log_dir
            writer = SummaryWriter(log_dir=log_dir)

    _print_header("OUTPUT")
    _print_kv("TensorBoard", writer.log_dir if writer else "Disabled")
    checkpoint_dir = os.path.abspath(config.checkpoint.save_dir)
    _print_kv("Checkpoints", checkpoint_dir)
    _print_kv("Save frequency", f"Every {config.checkpoint.save_frequency} epochs" if config.checkpoint.save_frequency > 0 else "Disabled")
    _print_kv("Save best", "Yes" if config.checkpoint.save_best else "No")

    # Log sample images to TensorBoard
    if writer is not None and config.logging.log_images and len(data_loader) > 0:
        try:
            images, _ = next(iter(data_loader))
            grid = make_grid(list(images)[:8])  # Limit to 8 images
            writer.add_image('train_images', grid, 0)

            val_images, _ = next(iter(data_loader_val))
            val_grid = make_grid(list(val_images)[:8])
            writer.add_image('val_images', val_grid, 0)
        except StopIteration:
            pass

    # Create training state for callbacks
    state = TrainingState(
        epoch=start_epoch,
        total_epochs=config.epochs,
        model=model,
        optimizer=optimizer,
        train_metrics={},
        val_metrics={},
        best_val_loss=best_val_loss,
        best_val_f1=best_val_f1,
        device=device,
    )

    # Call on_train_start
    callbacks.on_train_start(state)

    # Training loop
    print(f"\n{_Colors.BOLD}[ PROGRESS ]{_Colors.RESET}")
    print(f"  [+] Starting {config.epochs - start_epoch} epochs...")

    final_epoch = start_epoch
    early_stopped = False
    import time as time_module
    training_start_time = time_module.time()

    for epoch in range(start_epoch, config.epochs):
        epoch_start_time = time_module.time()
        state.epoch = epoch
        callbacks.on_epoch_start(state)

        # Set epoch for distributed sampler
        if distributed and train_sampler is not None:
            try:
                train_sampler.set_epoch(epoch)
            except Exception:
                pass

        # Train one epoch
        model.train()
        train_metrics = train_one_epoch(
            model, optimizer, data_loader, device, epoch,
            print_freq=config.logging.print_freq, scaler=scaler,
            gradient_clip_val=config.gradient_clip_val
        )

        # Extract training loss
        if hasattr(train_metrics, 'meters') and 'loss' in train_metrics.meters:
            if isinstance(train_metrics.meters['loss'], SmoothedValue):
                train_loss = train_metrics.meters['loss'].global_avg
            else:
                train_loss = train_metrics.meters['loss']
        else:
            train_loss = 0.0

        # Run validation
        val_seg_metrics: Optional[SegmentationMetrics] = None
        if compute_metrics:
            val_metrics_raw, val_seg_metrics = eval_with_metrics(model, data_loader_val, device)
        else:
            val_metrics_raw = eval_forward(model, data_loader_val, device)[0]

        # Extract validation loss
        if isinstance(val_metrics_raw, dict) and 'loss' in val_metrics_raw:
            if isinstance(val_metrics_raw['loss'], SmoothedValue):
                val_loss = val_metrics_raw['loss'].global_avg
            else:
                val_loss = val_metrics_raw['loss']
        else:
            # Use summed losses if available
            val_loss = sum(
                v.global_avg if isinstance(v, SmoothedValue) else v
                for k, v in val_metrics_raw.items()
                if 'loss' in k.lower()
            ) if isinstance(val_metrics_raw, dict) else 0.0

        # Extract F1 metrics
        val_f1 = val_seg_metrics.f1_score if val_seg_metrics else 0.0
        val_precision = val_seg_metrics.precision if val_seg_metrics else 0.0
        val_recall = val_seg_metrics.recall if val_seg_metrics else 0.0
        val_f1_agnostic = val_seg_metrics.f1_agnostic if val_seg_metrics else 0.0

        # Calculate epoch time
        epoch_time = time_module.time() - epoch_start_time
        epoch_mins, epoch_secs = divmod(int(epoch_time), 60)

        # Print epoch summary
        is_best_loss = val_loss < best_val_loss
        is_best_f1 = val_f1 > best_val_f1

        if compute_metrics:
            best_marker = ""
            if is_best_loss and is_best_f1:
                best_marker = " (best loss & F1)"
            elif is_best_loss:
                best_marker = " (best loss)"
            elif is_best_f1:
                best_marker = " (best F1)"

            prefix = "  [+]" if is_best_f1 else "  [ ]"
            print(f"{prefix} Epoch {epoch + 1}/{config.epochs}  "
                  f"train={train_loss:.4f}  "
                  f"val={val_loss:.4f}  "
                  f"F1={val_f1:.4f}{best_marker}  "
                  f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                  f"({epoch_mins}m {epoch_secs}s)")
        else:
            if is_best_loss:
                print(f"  [+] Epoch {epoch + 1}/{config.epochs}  "
                      f"train={train_loss:.4f}  "
                      f"val={val_loss:.4f} (best)  "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                      f"({epoch_mins}m {epoch_secs}s)")
            else:
                print(f"  [ ] Epoch {epoch + 1}/{config.epochs}  "
                      f"train={train_loss:.4f}  "
                      f"val={val_loss:.4f}  "
                      f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                      f"({epoch_mins}m {epoch_secs}s)")

        # Step scheduler (after validation so ReduceLROnPlateau has the val_loss)
        if lr_scheduler is not None:
            # Check if it's ReduceLROnPlateau (possibly wrapped in WarmupScheduler)
            is_plateau = isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
            if hasattr(lr_scheduler, 'base_scheduler'):
                is_plateau = isinstance(lr_scheduler.base_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)

            if is_plateau:
                lr_scheduler.step(val_loss)
            else:
                lr_scheduler.step()

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Update metrics history if computing metrics
        if compute_metrics:
            history['val_f1'].append(val_f1)
            history['val_precision'].append(val_precision)
            history['val_recall'].append(val_recall)
            history['val_f1_agnostic'].append(val_f1_agnostic)

            # Add per-class metrics to history
            if val_seg_metrics and val_seg_metrics.per_class:
                for cls_id, cls_metrics in val_seg_metrics.per_class.items():
                    key = f'val_f1_class_{cls_id}'
                    if key not in history:
                        history[key] = []
                    history[key].append(cls_metrics['f1'])

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

            # Log metrics if computed
            if compute_metrics and val_seg_metrics:
                writer.add_scalar('f1/val', val_f1, epoch)
                writer.add_scalar('precision/val', val_precision, epoch)
                writer.add_scalar('recall/val', val_recall, epoch)
                writer.add_scalar('f1_agnostic/val', val_f1_agnostic, epoch)
                writer.add_scalar('iou/val', val_seg_metrics.iou, epoch)

                # Log per-class metrics
                for cls_id, cls_metrics in val_seg_metrics.per_class.items():
                    writer.add_scalar(f'f1_class_{cls_id}/val', cls_metrics['f1'], epoch)
                    writer.add_scalar(f'precision_class_{cls_id}/val', cls_metrics['precision'], epoch)
                    writer.add_scalar(f'recall_class_{cls_id}/val', cls_metrics['recall'], epoch)

            # Log individual loss components
            if hasattr(train_metrics, 'meters'):
                for key, value in train_metrics.meters.items():
                    if isinstance(value, SmoothedValue):
                        writer.add_scalar(f'{key}/train', value.global_avg, epoch)
                    elif isinstance(value, (int, float)):
                        writer.add_scalar(f'{key}/train', value, epoch)

            if isinstance(val_metrics_raw, dict):
                for key, value in val_metrics_raw.items():
                    if isinstance(value, SmoothedValue):
                        writer.add_scalar(f'{key}/val', value.global_avg, epoch)
                    elif isinstance(value, (int, float)):
                        writer.add_scalar(f'{key}/val', value, epoch)

            writer.flush()

        # Update best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            state.best_val_loss = best_val_loss

        # Update best F1 metrics
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_val_precision = val_precision
            best_val_recall = val_recall
            state.best_val_f1 = best_val_f1
        if val_f1_agnostic > best_val_f1_agnostic:
            best_val_f1_agnostic = val_f1_agnostic

        # Update state for callbacks
        state.train_metrics = {'loss': train_loss}
        state.val_metrics = {'loss': val_loss, 'f1': val_f1} if compute_metrics else {'loss': val_loss}

        # Call on_epoch_end
        callbacks.on_epoch_end(state)

        # Check for early stopping
        for cb in callbacks.callbacks:
            if hasattr(cb, 'should_stop') and cb.should_stop:
                early_stopped = True
                break

        if early_stopped:
            print(f"  [-] Early stopping triggered")
            break

        final_epoch = epoch

    # Call on_train_end
    callbacks.on_train_end(state)

    # Cleanup
    if writer is not None:
        writer.close()

    # Final summary
    total_time = time_module.time() - training_start_time
    total_mins, total_secs = divmod(int(total_time), 60)
    total_hours, total_mins = divmod(total_mins, 60)

    time_str = f"{total_hours}h {total_mins}m {total_secs}s" if total_hours > 0 else f"{total_mins}m {total_secs}s"

    print(f"\n{_Colors.BOLD}[ COMPLETE ]{_Colors.RESET}")
    _print_kv("Epochs", f"{final_epoch + 1}/{config.epochs}")
    _print_kv("Time", time_str)
    _print_kv("Best val loss", f"{best_val_loss:.4f}")
    if compute_metrics:
        _print_kv("Best val F1", f"{best_val_f1:.4f}")
        _print_kv("Best val F1 (agnostic)", f"{best_val_f1_agnostic:.4f}")

    return TrainingResult(
        model=model,
        history=history,
        best_val_loss=best_val_loss,
        best_val_f1=best_val_f1,
        best_val_precision=best_val_precision,
        best_val_recall=best_val_recall,
        best_val_f1_agnostic=best_val_f1_agnostic,
        config=config,
        final_epoch=final_epoch,
    )


def train(
    config: Optional[TrainingConfig] = None,
    *,
    # Simple kwargs approach
    dataset_path: Optional[str] = None,
    epochs: Optional[int] = None,
    batch_size: Optional[int] = None,
    num_classes: Optional[int] = None,
    num_workers: Optional[int] = None,
    n_repeat_images: Optional[int] = None,
    img_extension: Optional[str] = None,
    val_split: Optional[float] = None,
    # Optimizer kwargs
    optimizer: Optional[str] = None,
    lr: Optional[float] = None,
    momentum: Optional[float] = None,
    weight_decay: Optional[float] = None,
    # Scheduler kwargs
    scheduler: Optional[str] = None,
    step_size: Optional[int] = None,
    gamma: Optional[float] = None,
    warmup_epochs: Optional[int] = None,
    # Image selection (mutually exclusive)
    n_images: Optional[int] = None,
    image_range: Optional[Tuple[int, int]] = None,
    # Augmentation
    augmentation: Optional[AugmentationConfig] = None,
    # Checkpointing
    save_dir: Optional[str] = None,
    save_frequency: Optional[int] = None,
    save_best: Optional[bool] = None,
    # Device and training options
    device: Optional[str] = None,
    seed: Optional[int] = None,
    mixed_precision: Optional[bool] = None,
    gradient_clip_val: Optional[float] = None,
    # Logging
    use_tensorboard: Optional[bool] = None,
    log_dir: Optional[str] = None,
    print_freq: Optional[int] = None,
    # Callbacks
    callbacks: Optional[List[Callback]] = None,
    on_epoch_end: Optional[Callable[[int, Dict[str, Any]], None]] = None,
    on_train_end: Optional[Callable[[Dict[str, Any]], None]] = None,
    # Resume
    resume_from: Optional[str] = None,
    # Metrics
    compute_metrics: bool = True,
) -> TrainingResult:
    """Train a Mask R-CNN model.

    This function supports two usage patterns:

    1. Simple kwargs: Pass individual parameters directly
       ```python
       result = train(
           dataset_path='my_dataset',
           epochs=100,
           lr=0.001,
           optimizer='adamw',
       )
       ```

    2. Config-based: Pass a TrainingConfig object
       ```python
       config = TrainingConfig(epochs=200, ...)
       result = train(config=config)
       ```

    Args:
        config: Complete TrainingConfig object. If provided, most kwargs are ignored.

        # Data parameters (used if config is None)
        dataset_path: Path to the dataset directory
        epochs: Number of training epochs (default: 100)
        batch_size: Batch size for training (default: 10)
        num_classes: Number of output classes (default: 3)
        num_workers: Number of data loading workers (default: 0)
        n_repeat_images: Times to repeat images in dataset (default: 1)
        img_extension: Image file extension (default: '.png')

        # Optimizer parameters
        optimizer: Optimizer type - 'sgd', 'adam', 'adamw', 'rmsprop' (default: 'sgd')
        lr: Learning rate (default: 0.005)
        momentum: Momentum for SGD/RMSprop (default: 0.9)
        weight_decay: Weight decay (default: 0.0005)

        # Scheduler parameters
        scheduler: Scheduler type - 'step', 'cosine', 'plateau', 'none' (default: 'step')
        step_size: Period for StepLR (default: 20)
        gamma: Decay factor (default: 0.1)
        warmup_epochs: Number of warmup epochs (default: 0)

        # Image selection (mutually exclusive)
        n_images: Randomly select N images for training
        image_range: Use images in range (start, end) inclusive

        # Augmentation
        augmentation: AugmentationConfig object (default: AugmentationConfig.default())

        # Checkpointing
        save_dir: Directory for checkpoints (default: '.')
        save_frequency: Save every N epochs (default: 50)
        save_best: Save best model based on val loss (default: True)

        # Training options
        device: Device to use - 'auto', 'cuda', 'cpu', 'mps' (default: 'auto')
        seed: Random seed for reproducibility
        mixed_precision: Use automatic mixed precision (default: False)
        gradient_clip_val: Max gradient norm for clipping

        # Logging
        use_tensorboard: Enable TensorBoard logging (default: True)
        log_dir: Directory for TensorBoard logs
        print_freq: Print progress every N batches (default: 10)

        # Callbacks
        callbacks: List of Callback objects
        on_epoch_end: Function called at end of each epoch with (epoch, metrics)
        on_train_end: Function called at end of training with (final_metrics)

        # Resume
        resume_from: Path to checkpoint to resume training from

        # Metrics
        compute_metrics: Compute pixel-level F1 metrics during validation (default: True)

    Returns:
        TrainingResult containing trained model, history, and configuration

    Raises:
        ValueError: If both n_images and image_range are specified

    Examples:
        # Simple training
        result = train(dataset_path='my_dataset', epochs=100)

        # With specific optimizer and scheduler
        result = train(
            dataset_path='my_dataset',
            epochs=200,
            optimizer='adamw',
            lr=0.001,
            scheduler='cosine',
        )

        # With image selection
        result = train(
            dataset_path='my_dataset',
            epochs=100,
            n_images=50,  # Train on 50 random images
        )

        # With callbacks
        from alveoleye.lungcv.mrcnn import EarlyStoppingCallback

        result = train(
            dataset_path='my_dataset',
            callbacks=[EarlyStoppingCallback(patience=20)],
            on_epoch_end=lambda e, m: print(f"Epoch {e}: loss={m['loss']:.4f}"),
        )
    """
    # Validate mutually exclusive options
    if n_images is not None and image_range is not None:
        raise ValueError("n_images and image_range are mutually exclusive")

    # Build or use config
    if config is None:
        config = _build_config_from_kwargs(
            dataset_path=dataset_path,
            epochs=epochs,
            batch_size=batch_size,
            num_classes=num_classes,
            num_workers=num_workers,
            n_repeat_images=n_repeat_images,
            img_extension=img_extension,
            val_split=val_split,
            optimizer=optimizer,
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            scheduler=scheduler,
            step_size=step_size,
            gamma=gamma,
            warmup_epochs=warmup_epochs,
            n_images=n_images,
            image_range=image_range,
            augmentation=augmentation,
            save_dir=save_dir,
            save_frequency=save_frequency,
            save_best=save_best,
            device=device,
            seed=seed,
            mixed_precision=mixed_precision,
            gradient_clip_val=gradient_clip_val,
            use_tensorboard=use_tensorboard,
            log_dir=log_dir,
            print_freq=print_freq,
        )

    # Build callback list
    callback_list = CallbackList(callbacks or [])

    # Add checkpoint callback if configured (only on main process if distributed)
    distributed_env = os.environ.get("WORLD_SIZE")
    is_distrib = False
    try:
        is_distrib = int(distributed_env) > 1 if distributed_env is not None else False
    except ValueError:
        is_distrib = False

    if config.checkpoint.save_frequency > 0 or config.checkpoint.save_best:
        if not is_distrib or is_main_process():
            checkpoint_callback = ModelCheckpointCallback(
                save_dir=str(config.checkpoint.save_dir),
                save_frequency=config.checkpoint.save_frequency,
                save_best=config.checkpoint.save_best,
                filename_template=config.checkpoint.filename_template,
            )
            callback_list.add(checkpoint_callback)

    # Add lambda callbacks for simple functions
    if on_epoch_end is not None or on_train_end is not None:
        def _on_epoch_end(state: TrainingState):
            if on_epoch_end is not None:
                metrics_dict = {
                    'loss': state.train_metrics.get('loss'),
                    'val_loss': state.val_metrics.get('loss'),
                    'best_val_loss': state.best_val_loss,
                }
                if compute_metrics:
                    metrics_dict['val_f1'] = state.val_metrics.get('f1')
                    metrics_dict['best_val_f1'] = state.best_val_f1
                on_epoch_end(state.epoch, metrics_dict)

        def _on_train_end(state: TrainingState):
            if on_train_end is not None:
                final_dict = {
                    'final_epoch': state.epoch,
                    'best_val_loss': state.best_val_loss,
                }
                if compute_metrics:
                    final_dict['best_val_f1'] = state.best_val_f1
                on_train_end(final_dict)

        lambda_callback = LambdaCallback(
            on_epoch_end=_on_epoch_end if on_epoch_end else None,
            on_train_end=_on_train_end if on_train_end else None,
        )
        callback_list.add(lambda_callback)

    # Run training
    return _run_training(config, callback_list, resume_from=resume_from, compute_metrics=compute_metrics)
