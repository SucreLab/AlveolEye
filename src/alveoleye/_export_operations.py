import csv
import json
import os
import shutil
from typing import List, Optional, Dict

import numpy as np
import torch
from PIL import Image

from alveoleye._config_utils import Config
from alveoleye._models import Result


def get_unique_export_folder(base_dir: str, desired_name: str) -> str:
    root = os.path.abspath(base_dir)
    folder = desired_name
    candidate = os.path.join(root, folder)
    count = 1

    while os.path.exists(candidate):
        folder = f"{desired_name}({count})"
        candidate = os.path.join(root, folder)
        count += 1

    return candidate


def write_metrics(results: List[Result], out_path: str, fmt: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []

    for idx, r in enumerate(results, start=1):
        d = r.to_dict()
        d["case_id"] = idx
        rows.append(d)

    if fmt.lower() == "csv":
        fieldnames = ["case_id"] + [k for k in rows[0] if k != "case_id"] if rows else []
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    else:
        payload = {str(idx): {**r.to_dict(), "case_id": idx}
                   for idx, r in enumerate(results, start=1)}
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)


# export_operations.py

def write_labelmaps(results: List[Result], labelmap_dir: str, ext: str):
    """
    Write each case’s labelmaps into files named:
        {case_id}_{layer_name}.{ext}

    e.g.:
      1_initial.tif
      1_processing.tif
      etc.
    """
    import os
    import tifffile

    os.makedirs(labelmap_dir, exist_ok=True)

    for idx, r in enumerate(results, start=1):
        if not r.labelmaps:
            continue

        for layer_name, lm in r.labelmaps.items():
            fn = f"{idx}_{layer_name}.{ext}"
            outp = os.path.join(labelmap_dir, fn)
            # ensure correct dtype
            tifffile.imwrite(outp, lm.astype("uint16"))


def write_images(results: List[Result], labelmap_dir: str, ext: str):
    """
    Save each Result’s labelmaps as RGB images in {case_id}_{layer_name}.{ext}.
    - If array is (H, W), we map labels→colors.
    - If array is (1, H, W), we squeeze then map.
    - If array is (H, W, 3), we assume it’s already RGB and save it directly.
    """
    os.makedirs(labelmap_dir, exist_ok=True)
    colormap = _norm_to_rgb(Config.get_label_indexed_colormap())

    for idx, r in enumerate(results, start=1):
        if not r.labelmaps:
            continue

        for layer_name, arr in r.labelmaps.items():
            arr = np.asarray(arr)

            # Case 1: already RGB
            if arr.ndim == 3 and arr.shape[2] == 3:
                rgb_image = arr.astype(np.uint8)

            else:
                # collapse (1, H, W) → (H, W)
                if arr.ndim == 3 and arr.shape[0] == 1:
                    lm = arr[0]
                # already (H, W)
                elif arr.ndim == 2:
                    lm = arr
                else:
                    print(f"[!] Skipping {layer_name!r}: unsupported shape {arr.shape}")
                    continue

                # now lm is 2D labelmap → colorize
                h, w = lm.shape
                rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
                for label, color in colormap.items():
                    rgb_image[lm == label] = color

            fn = f"{idx}_{layer_name}.{ext}"
            outp = os.path.join(labelmap_dir, fn)
            Image.fromarray(rgb_image).save(outp)


def zip_folder(src_folder: str, zip_target: str):
    root, folder = os.path.split(src_folder.rstrip("/\\"))
    temp_archive = shutil.make_archive(os.path.join(root, folder), 'zip', root, folder)

    if os.path.abspath(temp_archive) != os.path.abspath(zip_target):
        if os.path.exists(zip_target):
            os.remove(zip_target)
        os.replace(temp_archive, zip_target)


def export_results(
    results: List[Result],
    base_dir: str,
    project_name: str,
    metrics_format: str = "csv",
    labelmap_ext: str = "tif",
    zip_it: bool = False,
    export_as_rgb: bool = False,
) -> Dict[str, Optional[str]]:
    export_folder = get_unique_export_folder(base_dir, project_name)
    os.makedirs(export_folder, exist_ok=True)

    metrics_fp = os.path.join(export_folder, f"metrics.{metrics_format}")
    write_metrics(results, metrics_fp, metrics_format)

    labelmaps_dir = None
    if any(r.labelmaps is not None for r in results):
        labelmaps_dir = os.path.join(export_folder, "labelmaps")

        if export_as_rgb:
            write_images(results, labelmaps_dir, labelmap_ext)
        else:
            write_labelmaps(results, labelmaps_dir, labelmap_ext)

    archive_fp = None
    if zip_it:
        archive_fp = os.path.join(base_dir, f"{os.path.basename(export_folder)}.zip")
        zip_folder(export_folder, archive_fp)

    return {
        "export_folder": export_folder,
        "metrics": metrics_fp,
        "labelmaps": labelmaps_dir,
        "archive": archive_fp,
    }


def is_real_writable_dir(path):
    return os.path.isdir(path) and os.access(path, os.W_OK | os.X_OK)


def _norm_to_rgb(colormap):
    rgb_map = {}
    for k, v in colormap.items():
        rgb_map[k] = [int(round(x * 255)) for x in v]

    rgb_map[0] = (255, 255, 255)
    return rgb_map


def load_image_specific_colormap(snapshot_step):
    colormap = Config.get_label_indexed_colormap()
    rgb_colormap = _norm_to_rgb(colormap)

    if snapshot_step == "GENERATE_PROCESSING_LABELMAP_AIRWAY":
        rgb_colormap[1] = rgb_colormap[2]
    elif snapshot_step == "GENERATE_PROCESSING_LABELMAP_VESSEL":
        rgb_colormap[1] = rgb_colormap[3]
    elif snapshot_step == "CONVERT_TO_GRAYSCALE":
        rgb_colormap = None

    return rgb_colormap


def save_image(data, snapshot_step, save_dir, get_colormap_function=None):
    os.makedirs(save_dir, exist_ok=True)

    name = Config.get_snapshot_names()[snapshot_step]

    base_name = name
    ext = ".png"
    candidate_name = f"{base_name}{ext}"
    counter = 1

    if get_colormap_function:
        colormap = get_colormap_function(snapshot_step)
    else:
        colormap = load_image_specific_colormap(snapshot_step)

    existing_files = set(os.listdir(save_dir))
    while candidate_name in existing_files:
        candidate_name = f"{base_name}({counter}){ext}"
        counter += 1

    save_path = os.path.join(save_dir, candidate_name)

    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    if isinstance(data, np.ndarray):
        data = np.squeeze(data)

        if data.ndim == 3 and data.shape[2] in {3, 4}:
            image = Image.fromarray(data.astype(np.uint8))
        elif data.ndim == 2:
            if colormap:
                h, w = data.shape
                rgb_image = np.zeros((h, w, 3), dtype=np.uint8)
                for label, color in colormap.items():
                    rgb_image[data == label] = color
                image = Image.fromarray(rgb_image)
            else:
                image = Image.fromarray(data.astype(np.uint8), mode='L')
        else:
            raise ValueError(f"Unsupported image shape after squeeze: {data.shape}")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    image.save(save_path)
    print(f"[+] Saved image to {save_path}")


def make_save_image_callback(save_dir, get_colormap_function=None):
    snapshots_dir = os.path.join(save_dir, "snapshots")

    def save_image_callback(data, name):
        save_image(data, name, snapshots_dir, get_colormap_function)

    return save_image_callback

