# export_operations.py

import os, csv, json, shutil
from typing import List
import tifffile
from alveoleye._models import Result


def write_metrics(results: List[Result], out_path: str, fmt: str):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rows = []
    for idx, r in enumerate(results, start=1):
        d = r.to_dict()
        d["case_id"] = idx
        rows.append(d)

    if fmt == "csv":
        flds = ["case_id"] + [k for k in rows[0].keys() if k != "case_id"] if rows else []
        with open(out_path, "w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=flds)
            writer.writeheader()
            writer.writerows(rows)

    else:  # json
        payload = {}
        for idx, r in enumerate(results, start=1):
            d = r.to_dict();
            d["case_id"] = idx
            payload[str(idx)] = d
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
    base, _ = os.path.splitext(zip_target)
    root, folder = os.path.split(src_folder.rstrip("/\\"))
    archive = shutil.make_archive(os.path.join(root, folder), 'zip', root, folder)
    if archive != zip_target:
        if os.path.exists(zip_target):
            os.remove(zip_target)
        os.replace(archive, zip_target)
