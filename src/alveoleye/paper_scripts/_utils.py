"""Utility functions for paper scripts.

This module provides shared utilities for dataset handling, file operations,
and result processing used across the paper scripts.
"""

import os
import tarfile
from pathlib import Path
from typing import Optional, Union, List, Tuple, Any, Literal

# Re-export dataset utilities from shared module for convenience
from alveoleye._dataset_utils import (
    detect_dataset_structure,
    is_valid_dataset_structure,
    count_dataset_images,
    DatasetStructure,
    SUPPORTED_IMAGE_EXTENSIONS,
)

# =============================================================================
# Constants
# =============================================================================

# Google Drive archive (tar.gz) containing the training dataset
TRAINING_DATASET_DRIVE_URL = "https://drive.google.com/file/d/1-jm7jUJJPdIfC_82ooOoWbpcIX2PBgq-/view?usp=sharing"

# Default location for training dataset
DEFAULT_TRAINING_DATASET_DIR = Path(__file__).parent.parent.parent / "training_dataset"

# Default image extensions for file listing
DEFAULT_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".gif")


# =============================================================================
# Dataset Discovery and Download
# =============================================================================

def find_training_dataset(search_dir: Optional[Union[str, Path]] = None) -> Optional[str]:
    """Find a valid training dataset directory.

    Searches for a valid dataset structure in the given directory. If the directory
    itself has the required structure, returns it. Otherwise, searches immediate
    subdirectories (useful when gdown creates a subfolder).

    Args:
        search_dir: Directory to search in. Defaults to src/training_dataset/.

    Returns:
        Path to the dataset directory as a string, or None if not found.
    """
    if search_dir is None:
        search_dir = DEFAULT_TRAINING_DATASET_DIR

    search_path = Path(search_dir)

    if not search_path.exists():
        return None

    # Check if the directory itself is a valid dataset
    if is_valid_dataset_structure(search_path):
        return str(search_path)

    # Check immediate subdirectories (handles gdown subfolder case)
    try:
        for subdir in search_path.iterdir():
            if subdir.is_dir() and is_valid_dataset_structure(subdir):
                return str(subdir)
    except PermissionError:
        pass

    return None


def download_training_dataset(
    output_dir: Optional[Union[str, Path]] = None,
    quiet: bool = False,
) -> str:
    """Download and extract the training dataset from Google Drive.

    Downloads the dataset archive to the specified output directory and extracts it.
    The extracted content will be located within output_dir.

    Args:
        output_dir: Parent directory where the dataset folder will be created.
                    Defaults to 'src/training_dataset'.
        quiet: If True, suppress download progress output.

    Returns:
        Path to the downloaded dataset directory (the subfolder containing images/,
        masks/, and classes.json).

    Raises:
        RuntimeError: If the download fails or the dataset structure is invalid.
        ImportError: If gdown is not installed.
    """
    try:
        import gdown
    except ImportError as e:
        raise ImportError(
            "gdown is required for downloading datasets. "
            "Install it with: pip install gdown"
        ) from e

    if output_dir is None:
        output_dir = DEFAULT_TRAINING_DATASET_DIR

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not quiet:
        print(f"[+] Downloading training dataset from Google Drive...")
        print(f"    Destination directory: {output_dir}")

    # Download the tar.gz file
    archive_path = output_dir / "dataset.tar.gz"
    downloaded_archive = gdown.download(
        url=TRAINING_DATASET_DRIVE_URL,
        output=str(archive_path),
        quiet=quiet,
        fuzzy=True,
    )

    if downloaded_archive is None:
        raise RuntimeError("Failed to download dataset from Google Drive")

    # Extract the archive
    if not quiet:
        print(f"[+] Extracting training dataset...")

    try:
        with tarfile.open(downloaded_archive, "r:gz") as tar:
            # Filter out members that begin with a period (hidden files/folders)
            members = [
                m for m in tar.getmembers()
                if not any(part.startswith(".") for part in Path(m.name).parts)
            ]
            tar.extractall(path=output_dir, members=members)
    except Exception as e:
        raise RuntimeError(f"Failed to extract dataset: {e}") from e
    finally:
        # Clean up the archive
        if os.path.exists(downloaded_archive):
            os.remove(downloaded_archive)

    # Find the dataset directory (could be output_dir itself or a subfolder)
    downloaded_path = find_training_dataset(output_dir)

    if downloaded_path is None:
        raise RuntimeError(
            f"Failed to find a valid dataset structure in {output_dir} after extraction. "
            "Expected either:\n"
            "  1. Split structure: images/train/, images/val/, masks/train/, masks/val/, classes.json\n"
            "  2. Flat structure: images/*.png, masks/*.png, classes.json"
        )

    dataset_path = Path(downloaded_path)
    if not quiet:
        print(f"[+] Dataset downloaded and validated: {downloaded_path}")

    return str(downloaded_path)


# =============================================================================
# File Path Utilities
# =============================================================================

def get_image_paths(
    directory_path: Union[str, Path],
    extensions: Tuple[str, ...] = DEFAULT_IMAGE_EXTENSIONS,
) -> List[str]:
    """Get paths to all image files in a directory.

    Args:
        directory_path: Path to the directory to search.
        extensions: Tuple of file extensions to include (case-insensitive).

    Returns:
        List of absolute paths to image files as strings.
    """
    directory = Path(directory_path)
    return [
        str(file.absolute())
        for file in directory.glob("*")
        if file.is_file() and file.suffix.lower() in extensions
    ]


def get_directory_paths(folder_path: Union[str, Path]) -> List[str]:
    """Get paths to all subdirectories recursively.

    Args:
        folder_path: Root directory to search.

    Returns:
        List of directory paths including the root.
    """
    return [root for root, _, _ in os.walk(folder_path)]


# =============================================================================
# Result Processing Utilities
# =============================================================================

def add_range_column(
    accumulated_results: List[List[Any]],
    metric_index: int,
) -> List[List[Any]]:
    """Add a range column to accumulated results based on a metric.

    Calculates the range (max - min) of a metric for each unique image
    and appends it to each result row.

    Args:
        accumulated_results: List of result rows, where each row is a list
                            with image name at index 0.
        metric_index: Index of the metric column to calculate range for.

    Returns:
        New list with range values appended to each row.
    """
    # Calculate min/max for each image
    image_metrics: dict[str, dict[str, float]] = {}

    for result in accumulated_results:
        image_name = result[0]
        metric = result[metric_index]

        if image_name not in image_metrics:
            image_metrics[image_name] = {"min": metric, "max": metric}
        else:
            image_metrics[image_name]["min"] = min(image_metrics[image_name]["min"], metric)
            image_metrics[image_name]["max"] = max(image_metrics[image_name]["max"], metric)

    # Calculate ranges
    image_ranges = {
        name: data["max"] - data["min"]
        for name, data in image_metrics.items()
    }

    # Append range to each result
    return [
        result + [image_ranges[result[0]]]
        for result in accumulated_results
    ]
