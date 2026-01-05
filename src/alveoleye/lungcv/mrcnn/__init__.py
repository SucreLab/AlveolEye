# Training API
from alveoleye.lungcv.mrcnn.api import train, TrainingResult

# Configuration
from alveoleye.lungcv.mrcnn.config import (
    TrainingConfig,
    DataConfig,
    OptimizerConfig,
    SchedulerConfig,
    AugmentationConfig,
    AugmentationItem,
    CheckpointConfig,
    LoggingConfig,
    ImageSelectionConfig,
)

# Callbacks
from alveoleye.lungcv.mrcnn.callbacks import (
    Callback,
    CallbackList,
    TrainingState,
    EarlyStoppingCallback,
    ModelCheckpointCallback,
    LambdaCallback,
)

# Augmentation utilities
from alveoleye.lungcv.mrcnn.augmentations import (
    build_transforms,
    get_available_augmentations,
)

# Optimizer utilities
from alveoleye.lungcv.mrcnn.optimizers import (
    create_optimizer,
    create_scheduler,
)

# Training engine
from alveoleye.lungcv.mrcnn.engine import train_one_epoch, evaluate

# Dataset
from alveoleye.lungcv.mrcnn.train import LungDataset

# COCO dataset utilities
from alveoleye.lungcv.mrcnn.coco_utils import (
    CocoDetection,
    ConvertCocoPolysToMask,
    get_coco,
    get_coco_api_from_dataset,
)

# Utilities
from alveoleye.lungcv.mrcnn.utils import (
    SmoothedValue,
    MetricLogger,
    collate_fn,
    eval_forward,
    reduce_dict,
    all_gather,
    get_world_size,
    # Distributed training utilities
    get_rank,
    is_main_process,
    save_on_master,
    setup_for_distributed,
    init_distributed_mode,
)

__all__ = [
    # Training API
    "train",
    "TrainingResult",
    # Configuration
    "TrainingConfig",
    "DataConfig",
    "OptimizerConfig",
    "SchedulerConfig",
    "AugmentationConfig",
    "AugmentationItem",
    "CheckpointConfig",
    "LoggingConfig",
    "ImageSelectionConfig",
    # Callbacks
    "Callback",
    "CallbackList",
    "TrainingState",
    "EarlyStoppingCallback",
    "ModelCheckpointCallback",
    "LambdaCallback",
    # Augmentation utilities
    "build_transforms",
    "get_available_augmentations",
    # Optimizer utilities
    "create_optimizer",
    "create_scheduler",
    # Training engine
    "train_one_epoch",
    "evaluate",
    # Datasets
    "LungDataset",
    "CocoDetection",
    "ConvertCocoPolysToMask",
    "get_coco",
    "get_coco_api_from_dataset",
    # Utilities
    "SmoothedValue",
    "MetricLogger",
    "collate_fn",
    "eval_forward",
    "reduce_dict",
    "all_gather",
    "get_world_size",
    # Distributed training utilities
    "get_rank",
    "is_main_process",
    "save_on_master",
    "setup_for_distributed",
    "init_distributed_mode",
]
