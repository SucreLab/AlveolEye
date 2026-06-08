"""Optimizer and scheduler factory functions.

This module provides factory functions to create optimizers and learning rate
schedulers from configuration objects.

Functions:
    create_optimizer: Create an optimizer from OptimizerConfig
    create_scheduler: Create a scheduler from SchedulerConfig
"""

from typing import Iterator, Optional

import torch
from torch.nn.parameter import Parameter
from torch.optim.lr_scheduler import _LRScheduler

from alveoleye.lungcv.mrcnn.config import OptimizerConfig, SchedulerConfig


def create_optimizer(
    params: Iterator[Parameter],
    config: OptimizerConfig
) -> torch.optim.Optimizer:
    """Create an optimizer from configuration.

    Args:
        params: Model parameters to optimize
        config: OptimizerConfig instance

    Returns:
        Configured optimizer instance

    Raises:
        ValueError: If optimizer name is not recognized

    Example:
        params = [p for p in model.parameters() if p.requires_grad]
        optimizer = create_optimizer(params, OptimizerConfig(name='adamw', lr=0.001))
    """
    name = config.name.lower()

    if name == 'sgd':
        return torch.optim.SGD(
            params,
            lr=config.lr,
            momentum=config.momentum,
            weight_decay=config.weight_decay,
            nesterov=config.nesterov,
        )
    elif name == 'adam':
        return torch.optim.Adam(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif name == 'adamw':
        return torch.optim.AdamW(
            params,
            lr=config.lr,
            betas=config.betas,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )
    elif name == 'rmsprop':
        return torch.optim.RMSprop(
            params,
            lr=config.lr,
            alpha=config.alpha,
            eps=config.eps,
            weight_decay=config.weight_decay,
            momentum=config.momentum,
            centered=config.centered,
        )
    else:
        raise ValueError(
            f"Unknown optimizer: '{name}'. "
            f"Choose from: 'sgd', 'adam', 'adamw', 'rmsprop'"
        )


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    config: SchedulerConfig,
    epochs: int
) -> Optional[_LRScheduler]:
    """Create a learning rate scheduler from configuration.

    Args:
        optimizer: The optimizer to schedule
        config: SchedulerConfig instance
        epochs: Total number of training epochs

    Returns:
        Configured scheduler instance, or None if 'none' specified

    Raises:
        ValueError: If scheduler name is not recognized

    Example:
        scheduler = create_scheduler(optimizer, SchedulerConfig(name='cosine'), epochs=100)
    """
    name = config.name.lower()

    if name == 'none':
        return None
    elif name == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config.step_size,
            gamma=config.gamma,
        )
    elif name == 'cosine':
        T_max = config.T_max if config.T_max is not None else epochs
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=config.eta_min,
        )
    elif name == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.gamma,
            patience=config.patience,
        )
    else:
        raise ValueError(
            f"Unknown scheduler: '{name}'. "
            f"Choose from: 'step', 'cosine', 'plateau', 'none'"
        )

    # Wrap with warmup if configured
    if config.warmup_epochs > 0:
        scheduler = WarmupScheduler(
            optimizer, scheduler, config.warmup_epochs, config.warmup_factor
        )

    return scheduler


class WarmupScheduler(_LRScheduler):
    """Wrapper scheduler that adds linear warmup to any base scheduler.

    During warmup, the learning rate linearly increases from
    base_lr * warmup_factor to base_lr over warmup_epochs epochs.

    Args:
        optimizer: The optimizer
        base_scheduler: The scheduler to use after warmup
        warmup_epochs: Number of warmup epochs
        warmup_factor: Starting factor (lr = base_lr * warmup_factor at epoch 0)
    """

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        base_scheduler: _LRScheduler,
        warmup_epochs: int,
        warmup_factor: float
    ):
        self.base_scheduler = base_scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_factor = warmup_factor
        self._step_count = 0
        # Flag to skip auto-step during parent's __init__
        self._initializing = True
        # Store original base LRs (from initial_lr set by base_scheduler)
        self._warmup_base_lrs = [group['initial_lr'] for group in optimizer.param_groups]
        # Initialize parent (this will try to call step(), but we skip it)
        super().__init__(optimizer, last_epoch=-1)
        self._initializing = False
        # Override base_lrs with our stored values
        self.base_lrs = self._warmup_base_lrs
        # Set initial warmup LR (at step 0, factor = warmup_factor)
        for param_group, base_lr in zip(optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * warmup_factor

    def get_lr(self):
        if self._step_count < self.warmup_epochs:
            # Linear warmup: interpolate from warmup_factor to 1.0
            alpha = self._step_count / self.warmup_epochs
            factor = self.warmup_factor * (1 - alpha) + alpha
            return [base_lr * factor for base_lr in self.base_lrs]
        return self.base_scheduler.get_last_lr()

    def step(self, metrics=None):
        # Skip step during parent's __init__ (auto-step prevention)
        if getattr(self, '_initializing', False):
            return
        self._step_count += 1
        if self._step_count < self.warmup_epochs:
            # During warmup, use our custom LR calculation
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group['lr'] = lr
        elif self._step_count == self.warmup_epochs:
            # End of warmup - restore base LR
            for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
                param_group['lr'] = base_lr
        else:
            # After warmup, delegate to base scheduler
            # ReduceLROnPlateau requires the metric, others don't
            if isinstance(self.base_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.base_scheduler.step(metrics)
            else:
                self.base_scheduler.step()
