"""Tests for optimizer and scheduler factory functions.

Tests cover:
- create_optimizer() for all optimizer types
- create_scheduler() for all scheduler types
- WarmupScheduler behavior
- Error handling for invalid names
"""

import pytest
import torch
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR, ReduceLROnPlateau

from alveoleye.lungcv.mrcnn.optimizers import (
    create_optimizer,
    create_scheduler,
    WarmupScheduler,
)
from alveoleye.lungcv.mrcnn.config import OptimizerConfig, SchedulerConfig


class TestCreateOptimizer:
    """Tests for create_optimizer factory function."""

    @pytest.mark.parametrize(
        "name,expected_type",
        [
            ("sgd", torch.optim.SGD),
            ("adam", torch.optim.Adam),
            ("adamw", torch.optim.AdamW),
            ("rmsprop", torch.optim.RMSprop),
        ],
    )
    def test_creates_correct_optimizer_type(self, mock_params, name, expected_type):
        """Test that each optimizer name creates the correct type."""
        config = OptimizerConfig(name=name, lr=0.01)
        optimizer = create_optimizer(mock_params, config)
        assert isinstance(optimizer, expected_type)

    def test_sets_learning_rate(self, mock_params):
        """Test that learning rate is set correctly."""
        config = OptimizerConfig(name="sgd", lr=0.123)
        optimizer = create_optimizer(mock_params, config)
        assert optimizer.defaults["lr"] == 0.123

    def test_sgd_momentum(self, mock_params):
        """Test SGD momentum setting."""
        config = OptimizerConfig(name="sgd", momentum=0.95)
        optimizer = create_optimizer(mock_params, config)
        assert optimizer.defaults["momentum"] == 0.95

    def test_sgd_nesterov(self, mock_params):
        """Test SGD Nesterov momentum."""
        config = OptimizerConfig(name="sgd", momentum=0.9, nesterov=True)
        optimizer = create_optimizer(mock_params, config)
        assert optimizer.defaults["nesterov"] is True

    def test_adam_betas(self, mock_params):
        """Test Adam beta parameters."""
        config = OptimizerConfig(name="adam", betas=(0.85, 0.99))
        optimizer = create_optimizer(mock_params, config)
        assert optimizer.defaults["betas"] == (0.85, 0.99)

    def test_adamw_weight_decay(self, mock_params):
        """Test AdamW weight decay."""
        config = OptimizerConfig(name="adamw", weight_decay=0.05)
        optimizer = create_optimizer(mock_params, config)
        assert optimizer.defaults["weight_decay"] == 0.05

    def test_rmsprop_alpha(self, mock_params):
        """Test RMSprop alpha (smoothing constant)."""
        config = OptimizerConfig(name="rmsprop", alpha=0.95)
        optimizer = create_optimizer(mock_params, config)
        assert optimizer.defaults["alpha"] == 0.95

    def test_rmsprop_centered(self, mock_params):
        """Test RMSprop centered option."""
        config = OptimizerConfig(name="rmsprop", centered=True)
        optimizer = create_optimizer(mock_params, config)
        assert optimizer.defaults["centered"] is True

    def test_invalid_optimizer_raises(self, mock_params):
        """Test that invalid optimizer name raises ValueError."""
        config = OptimizerConfig(name="invalid")
        with pytest.raises(ValueError, match="Unknown optimizer"):
            create_optimizer(mock_params, config)

    def test_case_insensitive_name(self, mock_params):
        """Test that optimizer names are case insensitive."""
        config = OptimizerConfig(name="SGD", lr=0.01)
        optimizer = create_optimizer(mock_params, config)
        assert isinstance(optimizer, torch.optim.SGD)


class TestCreateScheduler:
    """Tests for create_scheduler factory function."""

    def test_none_scheduler_returns_none(self, mock_optimizer):
        """Test that 'none' scheduler returns None."""
        config = SchedulerConfig(name="none")
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert scheduler is None

    def test_step_scheduler(self, mock_optimizer):
        """Test StepLR scheduler creation."""
        config = SchedulerConfig(name="step", step_size=10, gamma=0.5)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert isinstance(scheduler, StepLR)

    def test_step_scheduler_params(self, mock_optimizer):
        """Test StepLR scheduler parameters."""
        config = SchedulerConfig(name="step", step_size=15, gamma=0.2)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert scheduler.step_size == 15
        assert scheduler.gamma == 0.2

    def test_cosine_scheduler(self, mock_optimizer):
        """Test CosineAnnealingLR scheduler creation."""
        config = SchedulerConfig(name="cosine")
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert isinstance(scheduler, CosineAnnealingLR)

    def test_cosine_scheduler_uses_epochs_as_t_max(self, mock_optimizer):
        """Test that cosine scheduler uses epochs as T_max when not specified."""
        config = SchedulerConfig(name="cosine", T_max=None)
        scheduler = create_scheduler(mock_optimizer, config, epochs=150)
        assert scheduler.T_max == 150

    def test_cosine_scheduler_custom_t_max(self, mock_optimizer):
        """Test cosine scheduler with custom T_max."""
        config = SchedulerConfig(name="cosine", T_max=200)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert scheduler.T_max == 200

    def test_cosine_scheduler_eta_min(self, mock_optimizer):
        """Test cosine scheduler eta_min parameter."""
        config = SchedulerConfig(name="cosine", eta_min=1e-6)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert scheduler.eta_min == 1e-6

    def test_plateau_scheduler(self, mock_optimizer):
        """Test ReduceLROnPlateau scheduler creation."""
        config = SchedulerConfig(name="plateau", patience=5, gamma=0.5)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert isinstance(scheduler, ReduceLROnPlateau)

    def test_invalid_scheduler_raises(self, mock_optimizer):
        """Test that invalid scheduler name raises ValueError."""
        config = SchedulerConfig(name="invalid")
        with pytest.raises(ValueError, match="Unknown scheduler"):
            create_scheduler(mock_optimizer, config, epochs=100)

    def test_warmup_wrapping(self, mock_optimizer):
        """Test that warmup > 0 wraps scheduler in WarmupScheduler."""
        config = SchedulerConfig(name="step", warmup_epochs=5)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert isinstance(scheduler, WarmupScheduler)

    def test_no_warmup_wrapping_when_zero(self, mock_optimizer):
        """Test that warmup=0 doesn't wrap scheduler."""
        config = SchedulerConfig(name="step", warmup_epochs=0)
        scheduler = create_scheduler(mock_optimizer, config, epochs=100)
        assert isinstance(scheduler, StepLR)
        assert not isinstance(scheduler, WarmupScheduler)


class TestWarmupScheduler:
    """Tests for WarmupScheduler wrapper."""

    def test_warmup_phase_lr_increases(self, mock_params):
        """Test that LR increases during warmup phase."""
        optimizer = torch.optim.SGD(mock_params, lr=0.1)
        base_scheduler = StepLR(optimizer, step_size=10)
        warmup = WarmupScheduler(
            optimizer, base_scheduler, warmup_epochs=5, warmup_factor=0.001
        )

        lrs = []
        for _ in range(5):
            lrs.append(optimizer.param_groups[0]["lr"])
            warmup.step()

        # LRs should be increasing during warmup
        for i in range(1, len(lrs)):
            assert lrs[i] >= lrs[i - 1], f"LR should increase: {lrs}"

    def test_warmup_starts_at_factor(self, mock_params):
        """Test that warmup starts at lr * warmup_factor."""
        base_lr = 0.1
        warmup_factor = 0.01
        optimizer = torch.optim.SGD(mock_params, lr=base_lr)
        base_scheduler = StepLR(optimizer, step_size=10)
        warmup = WarmupScheduler(
            optimizer, base_scheduler, warmup_epochs=5, warmup_factor=warmup_factor
        )

        # Initial LR should be approximately base_lr * warmup_factor
        initial_lr = optimizer.param_groups[0]["lr"]
        expected_initial = base_lr * warmup_factor
        assert abs(initial_lr - expected_initial) < 1e-6

    def test_warmup_reaches_base_lr(self, mock_params):
        """Test that LR reaches base LR after warmup phase."""
        base_lr = 0.1
        optimizer = torch.optim.SGD(mock_params, lr=base_lr)
        base_scheduler = StepLR(optimizer, step_size=10)
        warmup = WarmupScheduler(
            optimizer, base_scheduler, warmup_epochs=5, warmup_factor=0.001
        )

        # Step through warmup phase
        for _ in range(5):
            warmup.step()

        # After warmup, LR should be at base LR
        lr_after_warmup = optimizer.param_groups[0]["lr"]
        assert abs(lr_after_warmup - base_lr) < 1e-6

    def test_base_scheduler_takes_over_after_warmup(self, mock_params):
        """Test that base scheduler controls LR after warmup."""
        base_lr = 0.1
        step_size = 2
        gamma = 0.5
        optimizer = torch.optim.SGD(mock_params, lr=base_lr)
        base_scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma)
        warmup = WarmupScheduler(
            optimizer, base_scheduler, warmup_epochs=3, warmup_factor=0.01
        )

        # Step through warmup
        for _ in range(3):
            warmup.step()

        # Step through base scheduler period
        for _ in range(step_size):
            warmup.step()

        # LR should have decayed by gamma
        expected_lr = base_lr * gamma
        actual_lr = optimizer.param_groups[0]["lr"]
        assert abs(actual_lr - expected_lr) < 1e-6

    def test_warmup_with_different_warmup_epochs(self, mock_params):
        """Test warmup with various warmup epoch counts."""
        for warmup_epochs in [1, 3, 10]:
            optimizer = torch.optim.SGD(mock_params, lr=0.1)
            base_scheduler = StepLR(optimizer, step_size=10)
            warmup = WarmupScheduler(
                optimizer, base_scheduler, warmup_epochs=warmup_epochs, warmup_factor=0.01
            )

            # Step exactly warmup_epochs times
            for _ in range(warmup_epochs):
                warmup.step()

            # Should be at base LR
            assert abs(optimizer.param_groups[0]["lr"] - 0.1) < 1e-6


class TestOptimizerParameterGroups:
    """Tests for optimizer parameter group handling."""

    def test_optimizer_has_correct_param_count(self, mock_model):
        """Test that optimizer receives correct parameters."""
        params = list(mock_model.parameters())
        config = OptimizerConfig(name="sgd", lr=0.01)
        optimizer = create_optimizer(params, config)

        total_params = sum(p.numel() for group in optimizer.param_groups for p in group["params"])
        expected_params = sum(p.numel() for p in params)
        assert total_params == expected_params

    def test_optimizer_with_filtered_params(self, mock_model):
        """Test optimizer with only requires_grad parameters."""
        # Freeze some parameters
        for i, param in enumerate(mock_model.parameters()):
            if i == 0:
                param.requires_grad = False

        params = [p for p in mock_model.parameters() if p.requires_grad]
        config = OptimizerConfig(name="adam", lr=0.001)
        optimizer = create_optimizer(params, config)

        assert len(optimizer.param_groups) == 1
