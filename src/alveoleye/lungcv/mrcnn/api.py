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
except ImportError:
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
from alveoleye.lungcv.mrcnn.utils import collate_fn, eval_forward, SmoothedValue
from alveoleye.lungcv.mrcnn.engine import train_one_epoch
from alveoleye.lungcv.model_operations import init_untrained_model


@dataclass
class TrainingResult:
    """Container for training results.

    Attributes:
        model: The trained model
        history: Dictionary mapping metric names to lists of values per epoch
        best_val_loss: Best validation loss achieved during training
        config: The TrainingConfig used for training
        final_epoch: The final epoch number (may be less than total if early stopped)
    """
    model: torch.nn.Module
    history: Dict[str, List[float]] = field(default_factory=dict)
    best_val_loss: float = float('inf')
    config: Optional[TrainingConfig] = None
    final_epoch: int = 0

    def save(self, path: Union[str, Path]) -> None:
        """Save the trained model to a file.

        Args:
            path: Path to save the model
        """
        torch.save(self.model, path)
        print(f"Model saved to: {path}")

    def save_checkpoint(self, path: Union[str, Path]) -> None:
        """Save a full checkpoint with model state dict and training info.

        Args:
            path: Path to save the checkpoint
        """
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'history': self.history,
            'best_val_loss': self.best_val_loss,
            'final_epoch': self.final_epoch,
        }
        if self.config is not None:
            checkpoint['config'] = self.config.to_dict()
        torch.save(checkpoint, path)
        print(f"Checkpoint saved to: {path}")


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
        print(f"[Image Selection] Randomly selected {len(indices)} images (seed={config.seed})")
        return Subset(dataset, indices)

    elif config.index_range is not None:
        start, end = config.index_range
        end = min(end + 1, total_images)  # +1 to make it inclusive, clamp to dataset size
        indices = list(range(start, end))
        print(f"[Image Selection] Using images {start} to {end - 1} ({len(indices)} images)")
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
) -> TrainingResult:
    """Run the training loop.

    Args:
        config: Complete training configuration
        callbacks: List of callbacks
        resume_from: Optional path to checkpoint to resume from

    Returns:
        TrainingResult with trained model and metrics
    """
    # Set seed for reproducibility
    if config.seed is not None:
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)

    # Determine device
    device = _get_device(config.device)
    print(f"[Training] Using device: {device}")

    # Build transforms
    train_transforms = build_transforms(config.augmentation, train=True)
    val_transforms = build_transforms(config.augmentation, train=False)

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

    print(f"[Training] Training images: {len(dataset)}")
    print(f"[Training] Validation images: {len(dataset_val)}")

    # Create data loaders
    data_loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    val_batch_size = config.data.val_batch_size or config.data.batch_size
    data_loader_val = DataLoader(
        dataset_val,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=config.data.num_workers,
        collate_fn=collate_fn,
        pin_memory=config.data.pin_memory and torch.cuda.is_available(),
    )

    # Initialize model
    model = init_untrained_model(config.num_classes)
    model.to(device)

    # Create optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = create_optimizer(params, config.optimizer)

    # Create scheduler
    lr_scheduler = create_scheduler(optimizer, config.scheduler, config.epochs)

    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    history: Dict[str, List[float]] = {
        'train_loss': [],
        'val_loss': [],
        'lr': [],
    }

    if resume_from is not None:
        print(f"[Training] Resuming from checkpoint: {resume_from}")
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        if 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch'] + 1
        if 'best_val_loss' in checkpoint:
            best_val_loss = checkpoint['best_val_loss']
        if 'history' in checkpoint:
            history = checkpoint['history']
        print(f"[Training] Resumed from epoch {start_epoch}")

    # Setup mixed precision if enabled
    scaler = None
    if config.mixed_precision and torch.cuda.is_available():
        scaler = torch.cuda.amp.GradScaler()
        print("[Training] Mixed precision training enabled")

    # Setup TensorBoard
    writer = None
    if config.logging.use_tensorboard:
        if not TENSORBOARD_AVAILABLE:
            print("[Training] Warning: TensorBoard not available, logging disabled")
        else:
            log_dir = config.logging.log_dir
            writer = SummaryWriter(log_dir=log_dir)
            print(f"[Training] TensorBoard logging to: {writer.log_dir}")

            # Log sample images
            if config.logging.log_images and len(data_loader) > 0:
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
        device=device,
    )

    # Call on_train_start
    callbacks.on_train_start(state)

    # Training loop
    final_epoch = start_epoch
    early_stopped = False

    for epoch in range(start_epoch, config.epochs):
        state.epoch = epoch
        callbacks.on_epoch_start(state)

        # Train one epoch
        model.train()
        train_metrics = train_one_epoch(
            model, optimizer, data_loader, device, epoch,
            print_freq=config.logging.print_freq, scaler=scaler
        )

        # Extract training loss
        if hasattr(train_metrics, 'meters') and 'loss' in train_metrics.meters:
            if isinstance(train_metrics.meters['loss'], SmoothedValue):
                train_loss = train_metrics.meters['loss'].global_avg
            else:
                train_loss = train_metrics.meters['loss']
        else:
            train_loss = 0.0

        # Step scheduler
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Run validation
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

        # Update history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['lr'].append(optimizer.param_groups[0]['lr'])

        # Log to TensorBoard
        if writer is not None:
            writer.add_scalar('loss/train', train_loss, epoch)
            writer.add_scalar('loss/val', val_loss, epoch)
            writer.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)

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

        # Update state for callbacks
        state.train_metrics = {'loss': train_loss}
        state.val_metrics = {'loss': val_loss}

        # Call on_epoch_end
        callbacks.on_epoch_end(state)

        # Check for early stopping
        for cb in callbacks.callbacks:
            if hasattr(cb, 'should_stop') and cb.should_stop:
                early_stopped = True
                break

        if early_stopped:
            print(f"[Training] Early stopping at epoch {epoch}")
            break

        final_epoch = epoch

        # Apply gradient clipping if configured
        if config.gradient_clip_val is not None:
            torch.nn.utils.clip_grad_norm_(params, config.gradient_clip_val)

    # Call on_train_end
    callbacks.on_train_end(state)

    # Cleanup
    if writer is not None:
        writer.close()

    print(f"[Training] Completed at epoch {final_epoch + 1}/{config.epochs}")

    return TrainingResult(
        model=model,
        history=history,
        best_val_loss=best_val_loss,
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

    # Add checkpoint callback if configured
    if config.checkpoint.save_frequency > 0 or config.checkpoint.save_best:
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
                on_epoch_end(state.epoch, {
                    'loss': state.train_metrics.get('loss'),
                    'val_loss': state.val_metrics.get('loss'),
                    'best_val_loss': state.best_val_loss,
                })

        def _on_train_end(state: TrainingState):
            if on_train_end is not None:
                on_train_end({
                    'final_epoch': state.epoch,
                    'best_val_loss': state.best_val_loss,
                })

        lambda_callback = LambdaCallback(
            on_epoch_end=_on_epoch_end if on_epoch_end else None,
            on_train_end=_on_train_end if on_train_end else None,
        )
        callback_list.add(lambda_callback)

    # Run training
    return _run_training(config, callback_list, resume_from=resume_from)
