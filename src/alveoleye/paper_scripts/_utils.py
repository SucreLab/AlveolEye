import os
from pathlib import Path
from typing import Optional


# Google Drive folder containing the training dataset
TRAINING_DATASET_DRIVE_URL = "https://drive.google.com/drive/folders/1fH7Qm6I9udircffEEqUmef8TM7baYz2-?usp=drive_link"


def download_training_dataset(output_dir: Optional[str] = None, quiet: bool = False) -> str:
    """Download the training dataset from Google Drive.

    Downloads the dataset folder to the specified output directory. The Google Drive
    folder will be created as a subdirectory within output_dir.

    Args:
        output_dir: Parent directory where the dataset folder will be created.
                    Defaults to a 'datasets' folder in the paper_scripts directory.
        quiet: If True, suppress download progress output.

    Returns:
        Path to the downloaded dataset directory (the subfolder containing images/,
        masks/, and classes.json).

    Raises:
        RuntimeError: If the download fails or the dataset structure is invalid.
    """
    import gdown

    if output_dir is None:
        output_dir = str(Path(__file__).parent / "datasets")

    os.makedirs(output_dir, exist_ok=True)

    if not quiet:
        print(f"[+] Downloading training dataset from Google Drive...")
        print(f"    Destination directory: {output_dir}")

    # gdown.download_folder returns the path to the downloaded folder
    # (output_dir/<folder_name_from_drive>)
    downloaded_path = gdown.download_folder(
        url=TRAINING_DATASET_DRIVE_URL,
        output=output_dir,
        quiet=quiet,
    )

    if downloaded_path is None:
        raise RuntimeError("Failed to download dataset from Google Drive")

    # Validate the downloaded dataset has the expected structure
    dataset_path = Path(downloaded_path)
    if not dataset_path.exists():
        raise RuntimeError(f"Download reported success but path does not exist: {downloaded_path}")

    required_paths = [
        dataset_path / "images" / "train",
        dataset_path / "images" / "val",
        dataset_path / "masks" / "train",
        dataset_path / "masks" / "val",
        dataset_path / "classes.json",
    ]

    missing = [str(p) for p in required_paths if not p.exists()]
    if missing:
        raise RuntimeError(
            f"Downloaded dataset is missing required paths: {missing}. "
            f"Expected standard dataset structure in {downloaded_path}"
        )

    if not quiet:
        print(f"[+] Dataset downloaded and validated: {downloaded_path}")

    return str(downloaded_path)


def get_image_paths(directory_path, extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.gif')):
    return [
        str(file)
        for file in Path(directory_path).glob('*')
        if file.suffix.lower() in extensions
    ]


def get_directory_paths(folder_path):
    return [root for root, _, _ in os.walk(folder_path)]


def add_range_column(accumulated_results, metric_index):
    image_mli_ranges = {}

    for result in accumulated_results:
        image_name = result[0]
        metric = result[metric_index]

        if image_name not in image_mli_ranges:
            image_mli_ranges[image_name] = {"least": metric, "greatest": metric}
        else:
            image_mli_ranges[image_name]["least"] = min(image_mli_ranges[image_name]["least"], metric)
            image_mli_ranges[image_name]["greatest"] = max(image_mli_ranges[image_name]["greatest"], metric)

    for image_name in image_mli_ranges:
        least = image_mli_ranges[image_name]["least"]
        greatest = image_mli_ranges[image_name]["greatest"]
        image_mli_ranges[image_name]["range"] = greatest - least

    updated_accumulated_results = []
    for result in accumulated_results:
        image_name = result[0]
        range_value = image_mli_ranges[image_name]["range"]
        updated_accumulated_results.append(result + [range_value])

    return updated_accumulated_results
