# alveoleye/_models.py

import numpy as np
from dataclasses import dataclass, asdict, field
from typing import Optional, Any, Tuple


@dataclass
class Result:
    """
    A single case’s input parameters + output metrics,
    plus an optional full‐segmentation labelmap array.
    """
    # — input parameters —
    image_file_name: str
    use_computer_vision: bool
    weights_file_name: str
    min_confidence: float
    used_manual_threshold: bool
    threshold_value: float
    remove_small_particles: bool
    remove_small_holes: bool
    calculate_asvd: bool
    calculate_mli: bool
    lines: int
    min_length: float
    scale: float

    # — computed metrics —
    asvd: Optional[float]
    airspace_pixels: Optional[int]
    non_airspace_pixels: Optional[int]
    mli: Optional[float]
    stdev: Optional[float]
    chords: Optional[int]

    # — optional segmentation map —
    labelmap: Optional[np.ndarray] = field(default=None, compare=False)

    @classmethod
    def from_raw(
            cls,
            raw: Tuple[Any, ...],
            labelmap: Optional[np.ndarray] = None
    ) -> "Result":
        """
        Build a Result from the 19‐tuple that you currently stash in
        ActionBox.current_results plus an optional numpy labelmap.
        """
        (
            image_file_name,
            use_computer_vision,
            weights_file_name,
            min_confidence,
            used_manual_threshold,
            threshold_value,
            remove_small_particles,
            remove_small_holes,
            calculate_asvd,
            calculate_mli,
            lines,
            min_length,
            scale,
            asvd,
            airspace_pixels,
            non_airspace_pixels,
            mli,
            stdev,
            chords,
        ) = raw

        return cls(
            image_file_name=image_file_name,
            use_computer_vision=use_computer_vision,
            weights_file_name=weights_file_name,
            min_confidence=min_confidence,
            used_manual_threshold=used_manual_threshold,
            threshold_value=threshold_value,
            remove_small_particles=remove_small_particles,
            remove_small_holes=remove_small_holes,
            calculate_asvd=calculate_asvd,
            calculate_mli=calculate_mli,
            lines=lines,
            min_length=min_length,
            scale=scale,
            asvd=asvd,
            airspace_pixels=airspace_pixels,
            non_airspace_pixels=non_airspace_pixels,
            mli=mli,
            stdev=stdev,
            chords=chords,
            labelmap=labelmap,
        )

    def to_dict(self) -> dict:
        """
        Convert to a dictionary of *all* scalar fields, dropping
        the labelmap array so it’s safe for CSV/JSON serialization.
        """
        d = asdict(self)
        d.pop("labelmap", None)
        return d
