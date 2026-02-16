import os
from pathlib import Path
import io

import numpy as np
from PIL import Image

from alveoleye._verifiers import verify_png_or_tiff


def save_png(path: Path, size=(4, 3), color=(10, 20, 30, 255)):
    arr = np.zeros((size[1], size[0], 4), dtype=np.uint8)
    arr[..., 0] = color[0]
    arr[..., 1] = color[1]
    arr[..., 2] = color[2]
    arr[..., 3] = color[3]
    Image.fromarray(arr, mode="RGBA").save(path)


def save_tiff(path: Path, size=(4, 3)):
    arr = (np.random.rand(size[1], size[0]) * 255).astype(np.uint8)
    Image.fromarray(arr).save(path, format="TIFF")


def test_verify_png_success(tmp_path: Path):
    p = tmp_path / "ok.png"
    save_png(p)
    ok, fmt, err, info = verify_png_or_tiff(str(p))
    assert ok is True
    assert fmt == "PNG"
    assert err is None
    assert info["width"] == 4 and info["height"] == 3


def test_verify_tiff_success(tmp_path: Path):
    p = tmp_path / "ok.tif"
    save_tiff(p)
    ok, fmt, err, info = verify_png_or_tiff(str(p))
    assert ok is True
    assert fmt == "TIFF"
    assert err is None


def test_nonexistent_file(tmp_path: Path):
    p = tmp_path / "missing.png"
    ok, fmt, err, info = verify_png_or_tiff(str(p))
    assert ok is False
    assert fmt is None
    assert "not a file" in err.lower()


def test_signature_mismatch(tmp_path: Path):
    p = tmp_path / "text.txt"
    p.write_text("hello")
    ok, fmt, err, info = verify_png_or_tiff(str(p))
    assert ok is False
    assert fmt is None
    assert "signature mismatch" in err.lower()


def test_corrupted_png_structure(tmp_path: Path):
    p = tmp_path / "bad.png"
    # Write PNG signature but invalid body
    with open(p, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(b"garbage")
    ok, fmt, err, info = verify_png_or_tiff(str(p))
    assert ok is False
    assert fmt in ("PNG", None)
    assert "failed" in err.lower()
