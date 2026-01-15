"""Configuration dataclasses for the training API.

This module provides typed configuration objects for all aspects of training:
- OptimizerConfig: Optimizer settings (SGD, Adam, AdamW, RMSprop)
- SchedulerConfig: Learning rate scheduler settings
- AugmentationConfig: Data augmentation pipeline configuration
- ImageSelectionConfig: Image subset selection (random N or index range)
- DataConfig: Dataset and data loading settings
- CheckpointConfig: Model checkpointing settings
- LoggingConfig: TensorBoard and print logging settings
- TrainingConfig: Complete training configuration aggregating all above
"""

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Union, Literal
from pathlib import Path


@dataclass
class OptimizerConfig:
    """Configuration for the optimizer.

    Attributes:
        name: Optimizer type - 'sgd', 'adam', 'adamw', or 'rmsprop'
        lr: Learning rate (default: 0.005)
        momentum: Momentum factor for SGD/RMSprop (default: 0.9)
        weight_decay: Weight decay (L2 penalty) (default: 0.0005)
        betas: Coefficients for Adam/AdamW (default: (0.9, 0.999))
        eps: Term for numerical stability (default: 1e-8)
        alpha: Smoothing constant for RMSprop (default: 0.99)
        centered: Whether to use centered RMSprop (default: False)
        nesterov: Whether to use Nesterov momentum for SGD (default: False)
    """
    name: Literal['sgd', 'adam', 'adamw', 'rmsprop'] = 'sgd'
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    # Adam/AdamW specific
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-8
    # RMSprop specific
    alpha: float = 0.99
    centered: bool = False
    # SGD specific
    nesterov: bool = False


@dataclass
class SchedulerConfig:
    """Configuration for learning rate scheduler.

    Attributes:
        name: Scheduler type - 'step', 'cosine', 'plateau', or 'none'
        step_size: Period of learning rate decay for StepLR (default: 20)
        gamma: Multiplicative factor for StepLR/Plateau (default: 0.1)
        T_max: Max iterations for CosineAnnealingLR (default: None, uses epochs)
        eta_min: Min learning rate for CosineAnnealingLR (default: 0)
        patience: Epochs with no improvement for ReduceLROnPlateau (default: 10)
        warmup_epochs: Number of warmup epochs (default: 0)
        warmup_factor: Starting factor for warmup (default: 0.001)
    """
    name: Literal['step', 'cosine', 'plateau', 'none'] = 'step'
    step_size: int = 20
    gamma: float = 0.1
    T_max: Optional[int] = None
    eta_min: float = 0
    patience: int = 10
    warmup_epochs: int = 0
    warmup_factor: float = 0.001


@dataclass
class AugmentationItem:
    """Configuration for a single augmentation.

    Attributes:
        name: Augmentation name (e.g., 'horizontal_flip', 'color_jitter')
        probability: Probability of applying this augmentation (0.0 to 1.0)
        params: Dictionary of parameters specific to this augmentation

    Example:
        AugmentationItem('color_jitter', probability=0.3, params={
            'brightness': 0.2, 'contrast': 0.2
        })
    """
    name: str
    probability: float = 0.5
    params: dict = field(default_factory=dict)


@dataclass
class AugmentationConfig:
    """Configuration for data augmentation pipeline.

    Attributes:
        enabled: Whether to apply augmentations (default: True)
        augmentations: List of AugmentationItem configs

    Example:
        AugmentationConfig(augmentations=[
            AugmentationItem('horizontal_flip', probability=0.5),
            AugmentationItem('color_jitter', probability=0.3, params={
                'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.1
            }),
            AugmentationItem('gaussian_blur', probability=0.2, params={
                'kernel_size': 3, 'sigma': (0.1, 2.0)
            }),
        ])
    """
    enabled: bool = True
    augmentations: List[AugmentationItem] = field(default_factory=list)

    @classmethod
    def default(cls) -> 'AugmentationConfig':
        """Create default augmentation configuration matching original behavior."""
        return cls(augmentations=[
            AugmentationItem('horizontal_flip', probability=0.15),
            AugmentationItem('vertical_flip', probability=0.15),
            AugmentationItem('color_jitter', probability=0.15, params={
                'brightness': 0.2, 'contrast': 0.2, 'saturation': 0.2, 'hue': 0.2
            }),
            AugmentationItem('rotation', probability=0.15, params={'degrees': (-5, 5)}),
            AugmentationItem('gaussian_blur', probability=0.15, params={
                'kernel_size': 3, 'sigma': (0.1, 2.0)
            }),
            AugmentationItem('affine', probability=0.15, params={
                'degrees': 0, 'translate': (0.2, 0.2),
                'scale': (0.8, 1.2), 'shear': (-5, 5)
            }),
        ])

    @classmethod
    def none(cls) -> 'AugmentationConfig':
        """Create configuration with no augmentations."""
        return cls(enabled=False, augmentations=[])


@dataclass
class ImageSelectionConfig:
    """Configuration for selecting a subset of images for training.

    This allows training on a specific subset of images, either by random
    selection or by specifying an index range. The selection is made once
    at the start of training and the same images are used for every epoch.

    Attributes:
        n_random: Randomly select N images from the dataset
        index_range: Use images from index A to B (inclusive), e.g., (10, 30)
        seed: Random seed for reproducible random selection

    Note:
        n_random and index_range are mutually exclusive. If both are None,
        all images in the dataset are used.

    Examples:
        # Random 50 images with reproducible seed
        ImageSelectionConfig(n_random=50, seed=42)

        # Images at indices 10-30 (inclusive)
        ImageSelectionConfig(index_range=(10, 30))
    """
    n_random: Optional[int] = None
    index_range: Optional[Tuple[int, int]] = None
    seed: Optional[int] = None

    def __post_init__(self):
        if self.n_random is not None and self.index_range is not None:
            raise ValueError("n_random and index_range are mutually exclusive")


@dataclass
class DataConfig:
    """Configuration for dataset and data loading.

    Attributes:
        dataset_path: Path to the dataset directory
        batch_size: Batch size for training (default: 10)
        num_workers: Number of worker processes for data loading (default: 0)
        val_batch_size: Batch size for validation (default: same as batch_size)
        n_repeat_images: Number of times to repeat images in dataset (default: 1)
        img_extension: Image file extension (default: '.png')
        pin_memory: Whether to pin memory in DataLoader (default: True if CUDA)
        image_selection: Configuration for image subset selection (default: None for all)
    """
    dataset_path: Union[str, Path] = 'png_dataset'
    batch_size: int = 10
    num_workers: int = 0
    val_batch_size: Optional[int] = None
    n_repeat_images: int = 1
    img_extension: str = '.png'
    pin_memory: bool = True
    image_selection: Optional[ImageSelectionConfig] = None


@dataclass
class CheckpointConfig:
    """Configuration for model checkpointing.

    Attributes:
        save_dir: Directory to save checkpoints (default: '.')
        save_frequency: Save checkpoint every N epochs (default: 50)
        save_best: Whether to save best model based on val loss (default: True)
        save_last: Whether to always save most recent checkpoint (default: True)
        filename_template: Template for checkpoint filenames
    """
    save_dir: Union[str, Path] = '.'
    save_frequency: int = 50
    save_best: bool = True
    save_last: bool = True
    filename_template: str = 'checkpoint_epoch_{epoch}.pth'


@dataclass
class LoggingConfig:
    """Configuration for training logging.

    Attributes:
        use_tensorboard: Whether to use TensorBoard logging (default: True)
        log_dir: Directory for TensorBoard logs (default: None, auto-generated)
        print_freq: Print progress every N batches (default: 10)
        log_images: Whether to log sample images to TensorBoard (default: True)
    """
    use_tensorboard: bool = True
    log_dir: Optional[Union[str, Path]] = None
    print_freq: int = 10
    log_images: bool = True


def _default_augmentation_factory():
    """Factory for default augmentation config."""
    return AugmentationConfig.default()


@dataclass
class TrainingConfig:
    """Complete configuration for model training.

    This dataclass aggregates all configuration options for training.
    Can be passed directly to train() or saved/loaded from YAML/JSON.

    Attributes:
        epochs: Number of training epochs (default: 100)
        num_classes: Number of output classes (default: 3)
        device: Device to train on (default: 'auto' for auto-detection)
        seed: Random seed for reproducibility (default: None)
        mixed_precision: Whether to use automatic mixed precision (default: False)
        gradient_clip_val: Max gradient norm for clipping (default: None)

        data: DataConfig instance
        optimizer: OptimizerConfig instance
        scheduler: SchedulerConfig instance
        augmentation: AugmentationConfig instance
        checkpoint: CheckpointConfig instance
        logging: LoggingConfig instance

    Example:
        config = TrainingConfig(
            epochs=200,
            num_classes=3,
            data=DataConfig(dataset_path='my_dataset', batch_size=8),
            optimizer=OptimizerConfig(name='adamw', lr=0.001),
            augmentation=AugmentationConfig.default(),
        )
        train(config=config)
    """
    # Training parameters
    epochs: int = 100
    num_classes: int = 3
    device: str = 'auto'
    seed: Optional[int] = None
    mixed_precision: bool = False
    gradient_clip_val: Optional[float] = None

    # Sub-configs
    data: DataConfig = field(default_factory=DataConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    augmentation: AugmentationConfig = field(default_factory=_default_augmentation_factory)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> dict:
        """Convert config to dictionary for serialization."""
        from dataclasses import asdict
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'TrainingConfig':
        """Create config from dictionary."""
        # Handle nested dataclasses
        if 'data' in d and isinstance(d['data'], dict):
            data_dict = d['data'].copy()
            if 'image_selection' in data_dict and isinstance(data_dict['image_selection'], dict):
                img_sel = data_dict['image_selection'].copy()
                # Convert list back to tuple for index_range
                if 'index_range' in img_sel and isinstance(img_sel['index_range'], list):
                    img_sel['index_range'] = tuple(img_sel['index_range'])
                data_dict['image_selection'] = ImageSelectionConfig(**img_sel)
            d['data'] = DataConfig(**data_dict)
        if 'optimizer' in d and isinstance(d['optimizer'], dict):
            opt_dict = d['optimizer'].copy()
            # Convert list back to tuple for betas
            if 'betas' in opt_dict and isinstance(opt_dict['betas'], list):
                opt_dict['betas'] = tuple(opt_dict['betas'])
            d['optimizer'] = OptimizerConfig(**opt_dict)
        if 'scheduler' in d and isinstance(d['scheduler'], dict):
            d['scheduler'] = SchedulerConfig(**d['scheduler'])
        if 'augmentation' in d and isinstance(d['augmentation'], dict):
            augs = d['augmentation'].get('augmentations', [])
            d['augmentation'] = AugmentationConfig(
                enabled=d['augmentation'].get('enabled', True),
                augmentations=[AugmentationItem(**a) if isinstance(a, dict) else a for a in augs]
            )
        if 'checkpoint' in d and isinstance(d['checkpoint'], dict):
            d['checkpoint'] = CheckpointConfig(**d['checkpoint'])
        if 'logging' in d and isinstance(d['logging'], dict):
            d['logging'] = LoggingConfig(**d['logging'])
        return cls(**d)

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> 'TrainingConfig':
        """Load config from YAML file."""
        import yaml
        with open(path, 'r') as f:
            d = yaml.safe_load(f)
        return cls.from_dict(d)

    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save config to YAML file."""
        import yaml

        def convert_tuples_to_lists(obj):
            """Recursively convert tuples to lists for YAML compatibility."""
            if isinstance(obj, dict):
                return {k: convert_tuples_to_lists(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_tuples_to_lists(item) for item in obj]
            return obj

        with open(path, 'w') as f:
            yaml.dump(convert_tuples_to_lists(self.to_dict()), f, default_flow_style=False, sort_keys=False)
