# alveoleye/_models.py

from __future__ import annotations  # ← must be the very first non‐doc line
import numpy as np
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple, Any, Dict


@dataclass
class Result:
    """
    A single case’s input parameters + output metrics,
    plus an optional full‐segmentation labelmap array.
    All fields default to None so you can call Result() with no args.
    """

    # — input parameters —
    image_file_name: Optional[str] = None
    use_computer_vision: Optional[bool] = None
    weights_file_name: Optional[str] = None
    min_confidence: Optional[float] = None
    used_manual_threshold: Optional[bool] = None
    threshold_value: Optional[float] = None
    remove_small_particles: Optional[bool] = None
    remove_small_holes: Optional[bool] = None
    calculate_asvd: Optional[bool] = None
    calculate_mli: Optional[bool] = None
    lines: Optional[int] = None
    min_length: Optional[float] = None
    scale: Optional[float] = None

    # — computed metrics —
    asvd: Optional[float] = None
    airspace_pixels: Optional[int] = None
    non_airspace_pixels: Optional[int] = None
    mli: Optional[float] = None
    stdev: Optional[float] = None
    chords: Optional[int] = None

    # — optional segmentation map (excluded from equality/comparison) —
    labelmap: Optional[np.ndarray] = field(default=None, compare=False)

    @classmethod
    def from_raw(
            cls,
            raw: Tuple[Any, ...],
            labelmap: Optional[np.ndarray] = None
    ) -> Result:
        """
        Build a Result from the 19‐tuple of raw values
        plus an optional NumPy labelmap.
        """
        if len(raw) != 19:
            raise ValueError(f"Expected 19 items in raw tuple, got {len(raw)}")

        return cls(
            image_file_name=raw[0],
            use_computer_vision=raw[1],
            weights_file_name=raw[2],
            min_confidence=raw[3],
            used_manual_threshold=raw[4],
            threshold_value=raw[5],
            remove_small_particles=raw[6],
            remove_small_holes=raw[7],
            calculate_asvd=raw[8],
            calculate_mli=raw[9],
            lines=raw[10],
            min_length=raw[11],
            scale=raw[12],
            asvd=raw[13],
            airspace_pixels=raw[14],
            non_airspace_pixels=raw[15],
            mli=raw[16],
            stdev=raw[17],
            chords=raw[18],
            labelmap=labelmap
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to a dictionary of all scalar fields,
        dropping 'labelmap' so it’s safe for CSV/JSON.
        """
        d = asdict(self)
        d.pop("labelmap", None)
        return d
