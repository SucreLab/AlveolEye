"""Tests for training CLI.

Tests cover:
- Argument parsing
- Validation functions
- Config loading from file
- CLI override logic
- Error messages
"""

import argparse
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

from alveoleye.lungcv.mrcnn.cli import (
    create_parser,
    build_config_from_args,
    load_config,
)
from alveoleye.lungcv.mrcnn.cli_utils import (
    validate_arguments,
    parse_image_range,
    format_time,
    print_arguments,
)


class TestCreateParser:
    """Tests for argument parser creation."""

    def test_parser_returns_argument_parser(self):
        """Test that create_parser returns an ArgumentParser."""
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_dataset_path_required(self):
        """Test that dataset_path is required."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_default_values(self):
        """Test default argument values."""
        parser = create_parser()
        args = parser.parse_args(["/path/to/data"])

        assert args.dataset_path == "/path/to/data"
        assert args.epochs == 100
        assert args.batch_size == 10
        assert args.optimizer == "sgd"
        assert args.lr == 0.005
        assert args.scheduler == "step"
        assert args.device == "auto"

    def test_custom_epochs(self):
        """Test custom epochs argument."""
        parser = create_parser()
        args = parser.parse_args(["/path/to/data", "--epochs", "200"])
        assert args.epochs == 200

    def test_optimizer_choices(self):
        """Test optimizer argument choices."""
        parser = create_parser()

        for opt in ["sgd", "adam", "adamw", "rmsprop"]:
            args = parser.parse_args(["/path/to/data", "--optimizer", opt])
            assert args.optimizer == opt

    def test_invalid_optimizer_raises(self):
        """Test that invalid optimizer raises error."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["/path/to/data", "--optimizer", "invalid"])

    def test_scheduler_choices(self):
        """Test scheduler argument choices."""
        parser = create_parser()

        for sched in ["step", "cosine", "plateau", "none"]:
            args = parser.parse_args(["/path/to/data", "--scheduler", sched])
            assert args.scheduler == sched

    def test_device_choices(self):
        """Test device argument choices."""
        parser = create_parser()

        for device in ["auto", "cuda", "cpu", "mps"]:
            args = parser.parse_args(["/path/to/data", "--device", device])
            assert args.device == device

    def test_mutually_exclusive_image_selection(self):
        """Test that n_images and image_range are mutually exclusive."""
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([
                "/path/to/data",
                "--n-images", "50",
                "--image-range", "0:10",
            ])

    def test_n_images_argument(self):
        """Test n_images argument."""
        parser = create_parser()
        args = parser.parse_args(["/path/to/data", "--n-images", "25"])
        assert args.n_images == 25

    def test_image_range_argument(self):
        """Test image_range argument."""
        parser = create_parser()
        args = parser.parse_args(["/path/to/data", "--image-range", "10:50"])
        assert args.image_range == "10:50"

    def test_boolean_flags(self):
        """Test boolean flag arguments."""
        parser = create_parser()

        # Test mixed_precision
        args = parser.parse_args(["/path/to/data", "--mixed-precision"])
        assert args.mixed_precision is True

        # Test no_augmentation
        args = parser.parse_args(["/path/to/data", "--no-augmentation"])
        assert args.no_augmentation is True

        # Test no_save_best
        args = parser.parse_args(["/path/to/data", "--no-save-best"])
        assert args.no_save_best is True

        # Test no_tensorboard
        args = parser.parse_args(["/path/to/data", "--no-tensorboard"])
        assert args.no_tensorboard is True

    def test_config_file_argument(self):
        """Test config file argument."""
        parser = create_parser()
        args = parser.parse_args(["/path/to/data", "--config", "config.yaml"])
        assert args.config == "config.yaml"

    def test_resume_from_argument(self):
        """Test resume_from argument."""
        parser = create_parser()
        args = parser.parse_args(["/path/to/data", "--resume-from", "checkpoint.pth"])
        assert args.resume_from == "checkpoint.pth"


class TestValidateArguments:
    """Tests for validate_arguments function."""

    def test_missing_dataset_raises(self, tmp_path: Path):
        """Test that missing dataset raises ValueError."""
        args = argparse.Namespace(
            dataset_path=str(tmp_path / "nonexistent"),
            config=None,
            resume_from=None,
            image_range=None,
            save_dir=None,
            augmentation_config=None,
            epochs=100,
            batch_size=10,
            lr=0.005,
            n_images=None,
        )
        with pytest.raises(ValueError, match="does not exist"):
            validate_arguments(args)

    def test_missing_subdirs_raises(self, tmp_path: Path):
        """Test that missing subdirectories raise ValueError."""
        # Create empty dataset dir
        args = argparse.Namespace(
            dataset_path=str(tmp_path),
            config=None,
            resume_from=None,
            image_range=None,
            save_dir=None,
            augmentation_config=None,
            epochs=100,
            batch_size=10,
            lr=0.005,
            n_images=None,
        )
        with pytest.raises(ValueError, match="missing required subdirectory"):
            validate_arguments(args)

    def test_missing_classes_json_raises(self, tmp_path: Path):
        """Test that missing classes.json raises ValueError."""
        # Create subdirectories but no classes.json
        (tmp_path / "images" / "train").mkdir(parents=True)
        (tmp_path / "images" / "val").mkdir(parents=True)
        (tmp_path / "masks" / "train").mkdir(parents=True)
        (tmp_path / "masks" / "val").mkdir(parents=True)

        args = argparse.Namespace(
            dataset_path=str(tmp_path),
            config=None,
            resume_from=None,
            image_range=None,
            save_dir=None,
            augmentation_config=None,
            epochs=100,
            batch_size=10,
            lr=0.005,
            n_images=None,
        )
        with pytest.raises(ValueError, match="classes.json"):
            validate_arguments(args)

    def test_valid_dataset_passes(self, mock_dataset):
        """Test that valid dataset passes validation."""
        args = argparse.Namespace(
            dataset_path=str(mock_dataset),
            config=None,
            resume_from=None,
            image_range=None,
            save_dir=None,
            augmentation_config=None,
            epochs=100,
            batch_size=10,
            lr=0.005,
            n_images=None,
        )
        # Should not raise
        validate_arguments(args)

    def test_missing_config_file_raises(self, mock_dataset, tmp_path: Path):
        """Test that missing config file raises ValueError."""
        args = argparse.Namespace(
            dataset_path=str(mock_dataset),
            config=str(tmp_path / "nonexistent.yaml"),
            resume_from=None,
            image_range=None,
            save_dir=None,
            augmentation_config=None,
            epochs=100,
            batch_size=10,
            lr=0.005,
            n_images=None,
        )
        with pytest.raises(ValueError, match="Config file does not exist"):
            validate_arguments(args)

    def test_missing_checkpoint_raises(self, mock_dataset, tmp_path: Path):
        """Test that missing checkpoint file raises ValueError."""
        args = argparse.Namespace(
            dataset_path=str(mock_dataset),
            config=None,
            resume_from=str(tmp_path / "nonexistent.pth"),
            image_range=None,
            save_dir=None,
            augmentation_config=None,
            epochs=100,
            batch_size=10,
            lr=0.005,
            n_images=None,
        )
        with pytest.raises(ValueError, match="Checkpoint file does not exist"):
            validate_arguments(args)

    def test_invalid_image_range_format(self, mock_dataset):
        """Test that invalid image range format raises ValueError."""
        args = argparse.Namespace(
            dataset_path=str(mock_dataset),
            config=None,
            resume_from=None,
            image_range="10-50",  # Wrong separator
            save_dir=None,
            augmentation_config=None,
            epochs=100,
            batch_size=10,
            lr=0.005,
            n_images=None,
        )
        with pytest.raises(ValueError, match="Invalid image range"):
            validate_arguments(args)

    def test_negative_epochs_raises(self, mock_dataset):
        """Test that negative epochs raises ValueError."""
        args = argparse.Namespace(
            dataset_path=str(mock_dataset),
            config=None,
            resume_from=None,
            image_range=None,
            save_dir=None,
            augmentation_config=None,
            epochs=0,
            batch_size=10,
            lr=0.005,
            n_images=None,
        )
        with pytest.raises(ValueError, match="epochs must be at least 1"):
            validate_arguments(args)


class TestParseImageRange:
    """Tests for parse_image_range function."""

    def test_valid_range(self):
        """Test valid range parsing."""
        start, end = parse_image_range("10:50")
        assert start == 10
        assert end == 50

    def test_zero_start(self):
        """Test range starting at zero."""
        start, end = parse_image_range("0:100")
        assert start == 0
        assert end == 100

    def test_single_element_range(self):
        """Test range with same start and end."""
        start, end = parse_image_range("5:5")
        assert start == 5
        assert end == 5

    def test_missing_colon_raises(self):
        """Test that missing colon raises ValueError."""
        with pytest.raises(ValueError, match="missing ':'"):
            parse_image_range("1050")

    def test_wrong_separator_raises(self):
        """Test that wrong separator raises ValueError."""
        with pytest.raises(ValueError, match="missing ':'"):
            parse_image_range("10-50")

    def test_non_integer_raises(self):
        """Test that non-integer values raise ValueError."""
        with pytest.raises(ValueError, match="must be integers"):
            parse_image_range("abc:def")

    def test_multiple_colons_raises(self):
        """Test that multiple colons raise ValueError."""
        with pytest.raises(ValueError, match="expected 'start:end'"):
            parse_image_range("10:20:30")

    def test_whitespace_handling(self):
        """Test that whitespace is handled."""
        start, end = parse_image_range("  10  :  50  ")
        assert start == 10
        assert end == 50


class TestFormatTime:
    """Tests for format_time function."""

    def test_seconds_only(self):
        """Test formatting seconds."""
        assert format_time(45.2) == "45.2s"
        assert format_time(0.5) == "0.5s"

    def test_minutes_and_seconds(self):
        """Test formatting minutes and seconds."""
        result = format_time(125)  # 2 minutes 5 seconds
        assert "m" in result
        assert "s" in result

    def test_hours_minutes_seconds(self):
        """Test formatting hours, minutes, seconds."""
        result = format_time(3725)  # 1h 2m 5s
        assert "h" in result
        assert "m" in result
        assert "s" in result


class TestBuildConfigFromArgs:
    """Tests for build_config_from_args function."""

    def test_basic_config_creation(self, mock_dataset):
        """Test basic config creation from args."""
        parser = create_parser()
        args = parser.parse_args([str(mock_dataset), "--epochs", "50"])

        config = build_config_from_args(args)

        assert config.data.dataset_path == str(mock_dataset)
        assert config.epochs == 50

    def test_optimizer_config(self, mock_dataset):
        """Test optimizer configuration."""
        parser = create_parser()
        args = parser.parse_args([
            str(mock_dataset),
            "--optimizer", "adamw",
            "--lr", "0.001",
            "--weight-decay", "0.01",
        ])

        config = build_config_from_args(args)

        assert config.optimizer.name == "adamw"
        assert config.optimizer.lr == 0.001
        assert config.optimizer.weight_decay == 0.01

    def test_scheduler_config(self, mock_dataset):
        """Test scheduler configuration."""
        parser = create_parser()
        args = parser.parse_args([
            str(mock_dataset),
            "--scheduler", "cosine",
            "--warmup-epochs", "5",
        ])

        config = build_config_from_args(args)

        assert config.scheduler.name == "cosine"
        assert config.scheduler.warmup_epochs == 5

    def test_no_augmentation_flag(self, mock_dataset):
        """Test no_augmentation flag."""
        parser = create_parser()
        args = parser.parse_args([str(mock_dataset), "--no-augmentation"])

        config = build_config_from_args(args)

        assert config.augmentation.enabled is False

    def test_image_selection_n_images(self, mock_dataset):
        """Test image selection with n_images."""
        parser = create_parser()
        args = parser.parse_args([
            str(mock_dataset),
            "--n-images", "25",
            "--seed", "42",
        ])

        config = build_config_from_args(args)

        assert config.data.image_selection is not None
        assert config.data.image_selection.n_random == 25
        assert config.data.image_selection.seed == 42

    def test_image_selection_range(self, mock_dataset):
        """Test image selection with range."""
        parser = create_parser()
        args = parser.parse_args([
            str(mock_dataset),
            "--image-range", "10:30",
        ])

        config = build_config_from_args(args)

        assert config.data.image_selection is not None
        assert config.data.image_selection.index_range == (10, 30)


class TestLoadConfig:
    """Tests for load_config function."""

    def test_loads_from_yaml(self, mock_dataset, temp_yaml_config):
        """Test loading config from YAML file."""
        parser = create_parser()
        args = parser.parse_args([
            str(mock_dataset),
            "--config", str(temp_yaml_config),
        ])

        config = load_config(args)

        assert config.epochs == 150
        assert config.data.batch_size == 16
        assert config.optimizer.name == "adam"

    def test_cli_overrides_yaml(self, mock_dataset, temp_yaml_config):
        """Test that CLI args override YAML config."""
        parser = create_parser()
        args = parser.parse_args([
            str(mock_dataset),
            "--config", str(temp_yaml_config),
            "--epochs", "300",  # Override
        ])

        config = load_config(args)

        # Should use CLI value, not YAML value
        assert config.epochs == 300

    def test_builds_from_args_without_config(self, mock_dataset):
        """Test building config from args when no config file."""
        parser = create_parser()
        args = parser.parse_args([
            str(mock_dataset),
            "--epochs", "50",
        ])

        config = load_config(args)

        assert config.epochs == 50


class TestPrintArguments:
    """Tests for print_arguments function."""

    def test_print_arguments_no_error(self, mock_dataset, capsys):
        """Test that print_arguments runs without error."""
        parser = create_parser()
        args = parser.parse_args([str(mock_dataset)])

        # Should not raise
        print_arguments(args)

        captured = capsys.readouterr()
        assert "Training Configuration" in captured.out
        assert str(mock_dataset) in captured.out

    def test_print_arguments_shows_optimizer(self, mock_dataset, capsys):
        """Test that optimizer info is printed."""
        parser = create_parser()
        args = parser.parse_args([str(mock_dataset), "--optimizer", "adamw"])

        print_arguments(args)

        captured = capsys.readouterr()
        assert "adamw" in captured.out
