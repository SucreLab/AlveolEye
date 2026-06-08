# simple_image_verifier.py
from __future__ import annotations

from typing import Optional, Tuple, Dict
import os

from PIL import Image as PILImage, ImageFile
from PIL import UnidentifiedImageError

# Safety defaults: tune as needed for your app
DEFAULT_MAX_IMAGE_PIXELS = 100_000_000  # ~100 MP
ImageFile.LOAD_TRUNCATED_IMAGES = False  # fail on truncated files

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_TIFF_SIGS = {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}  # classic + BigTIFF


def _looks_like_png(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(8) == _PNG_SIG
    except Exception:
        return False


def _looks_like_tiff(path: str) -> bool:
    try:
        with open(path, "rb") as f:
            return f.read(4) in _TIFF_SIGS
    except Exception:
        return False


def _set_max_pixels(limit: Optional[int]):
    old = getattr(PILImage, "MAX_IMAGE_PIXELS", None)
    if limit is not None:
        PILImage.MAX_IMAGE_PIXELS = limit
    return old


def _restore_max_pixels(old):
    if old is not None:
        PILImage.MAX_IMAGE_PIXELS = old


def verify_png_or_tiff(
    path: str,
    *,
    decode_first_frame: bool = True,
    max_image_pixels: Optional[int] = DEFAULT_MAX_IMAGE_PIXELS,
) -> Tuple[bool, Optional[str], Optional[str], Dict[str, object]]:
    """
    Returns (ok, format, error, info)
      - format: 'PNG' or 'TIFF' when ok; otherwise best guess or None
      - error: message on failure; None on success
      - info: dict with basic metadata on success/failure context
    """
    info: Dict[str, object] = {"path": path}

    if not os.path.isfile(path):
        return False, None, "Path is not a file", info

    # Quick signature filter
    sig_png = _looks_like_png(path)
    sig_tiff = _looks_like_tiff(path)
    if not (sig_png or sig_tiff):
        return False, None, "Signature mismatch (not PNG/TIFF)", info

    old_limit = _set_max_pixels(max_image_pixels)
    try:
        # Structural verification without full decode
        try:
            with PILImage.open(path) as im:
                fmt = getattr(im, "format", None)  # Pillow's detected format
                info["detected_format"] = fmt
                if sig_png and fmt != "PNG":
                    return False, fmt, f"Signature says PNG but Pillow detected {fmt}", info
                if sig_tiff and fmt != "TIFF":
                    return False, fmt, f"Signature says TIFF but Pillow detected {fmt}", info

                if fmt not in ("PNG", "TIFF"):
                    return False, fmt, f"Unsupported format: {fmt}", info

                im.verify()  # checks structure (and PNG CRCs)
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
            # Covers truncated data, CRC errors, bad tiles, etc.
            guessed = "PNG" if sig_png else ("TIFF" if sig_tiff else None)
            return False, guessed, f"Structural verification failed: {e}", info

        if not decode_first_frame:
            return True, ("PNG" if sig_png else "TIFF"), None, info

        # Reopen and force decode of first frame/page (catches more corruption)
        try:
            with PILImage.open(path) as im:
                if getattr(im, "n_frames", 1) > 1:
                    try:
                        im.seek(0)
                    except EOFError:
                        return False, im.format, "Failed to seek first frame/page", info

                w, h = im.size
                info["width"] = w
                info["height"] = h
                im.load()  # force decompression of first frame/page

            return True, ("PNG" if sig_png else "TIFF"), None, info

        except PILImage.DecompressionBombError as e:
            return False, ("PNG" if sig_png else "TIFF"), f"Decompression bomb: {e}", info
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as e:
            return False, ("PNG" if sig_png else "TIFF"), f"Decode failed: {e}", info

    finally:
        _restore_max_pixels(old_limit)