# export_operations.py
import os
import csv
import json
import shutil
from typing import List, Optional, Dict, Any
import tifffile

from alveoleye._models import Result


def get_unique_export_folder(base_dir: str, desired_name: str) -> str:
    """
    Returns a unique folder path under base_dir named desired_name.
    If base_dir/desired_name already exists, appends (1), (2), ...
    """
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
    else:  # JSON
        payload = {str(idx): {**r.to_dict(), "case_id": idx}
                   for idx, r in enumerate(results, start=1)}
        with open(out_path, "w") as fh:
            json.dump(payload, fh, indent=2)


def write_labelmaps(results: List[Result], labelmap_dir: str, ext: str):
    os.makedirs(labelmap_dir, exist_ok=True)
    for idx, r in enumerate(results, start=1):
        if r.labelmap is None:
            continue
        fn = f"{idx}_labelmap.{ext}"
        outp = os.path.join(labelmap_dir, fn)
        tifffile.imwrite(outp, r.labelmap.astype("uint16"))


def zip_folder(src_folder: str, zip_target: str):
    """
    Create a zip archive from the src_folder, saved as zip_target.
    """
    root, folder = os.path.split(src_folder.rstrip("/\\"))
    temp_archive = shutil.make_archive(os.path.join(root, folder), 'zip', root, folder)
    # If user-supplied zip_target differs, move it into place
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
) -> Dict[str, Optional[str]]:
    """
    Export results to a unique folder under base_dir with project_name.
    Returns a dict with keys 'export_folder', 'metrics', 'labelmaps', 'archive'.
    """
    export_folder = get_unique_export_folder(base_dir, project_name)
    os.makedirs(export_folder, exist_ok=True)

    # metrics
    metrics_fp = os.path.join(export_folder, f"metrics.{metrics_format}")
    write_metrics(results, metrics_fp, metrics_format)

    # labelmaps (if any)
    labelmaps_dir = None
    if any(r.labelmap is not None for r in results):
        labelmaps_dir = os.path.join(export_folder, "labelmaps")
        write_labelmaps(results, labelmaps_dir, labelmap_ext)

    # optional zip
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