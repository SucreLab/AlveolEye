# export_operations.py

import csv
import io
import json
import os
import re
from dataclasses import dataclass, asdict, fields
from typing import Any, Dict, Iterable, List, Optional, Tuple


#
# Helpers to convert “empty string” or “NA” into None,
# otherwise to float/int.
#
def to_opt_float(value: Any) -> Optional[float]:
    if value in (None, "", "NA"):
        return None
    return float(value)


def to_opt_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    return int(value)


#
# Our one central dataclass.  All parameters + results live here.
#
@dataclass(frozen=True)
class FormatResult:
    # input parameters
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
    lines: Any  # you can tighten this type
    min_length: float
    scale: float

    # output results (optionals)
    asvd_result: Optional[float]
    airspace_pixels_result: Optional[int]
    non_airspace_pixels_result: Optional[int]
    mli_result: Optional[float]
    stdev_result: Optional[float]
    chords_result: Optional[int]


#
# Unpack a raw 19-tuple and build our FormatResult
#
def format_results(raw: Tuple[Any, ...]) -> FormatResult:
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

    return FormatResult(
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

        # convert to optional types
        asvd_result=to_opt_float(asvd),
        airspace_pixels_result=to_opt_int(airspace_pixels),
        non_airspace_pixels_result=to_opt_int(non_airspace_pixels),
        mli_result=to_opt_float(mli),
        stdev_result=to_opt_float(stdev),
        chords_result=to_opt_int(chords),
    )


#
# Turn a sequence of raw rows into a List[FormatResult]
#
def get_formatted_results(
        accumulated: Iterable[Tuple[Any, ...]]
) -> List[FormatResult]:
    return [format_results(row) for row in accumulated]


#
# CSV export
#
# We’ll pull the fieldnames in order from the dataclass definition
#
CSV_FIELDNAMES = [f.name for f in fields(FormatResult)]


def create_csv_data(
        accumulated: Iterable[Tuple[Any, ...]]
) -> str:
    """
    Return a CSV as a string, with a header row
    matching the fields of FormatResult.
    """
    fr_list = get_formatted_results(accumulated)
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDNAMES)
    writer.writeheader()
    for fr in fr_list:
        writer.writerow(asdict(fr))
    return buffer.getvalue()


def append_csv_data(
        accumulated: Iterable[Tuple[Any, ...]],
        export_file: str
) -> None:
    """
    Append to an existing CSV (or create it if missing),
    but only write the header when first creating.
    """
    data = create_csv_data(accumulated)
    exists = os.path.exists(export_file)
    mode = "a" if exists else "w"
    with open(export_file, mode, newline="") as fh:
        if exists:
            # skip the header line
            lines = data.splitlines()[1:]
            fh.write("\n".join(lines) + "\n")
        else:
            fh.write(data)


#
# JSON export
#
def create_json_data(
        accumulated: Iterable[Tuple[Any, ...]]
) -> str:
    """
    Produce a dict of
        { image_file_name: { …all fields from FormatResult… } }
    and dump it as pretty JSON.
    """
    fr_list = get_formatted_results(accumulated)
    output: Dict[str, Dict[str, Any]] = {}
    for fr in fr_list:
        output[fr.image_file_name] = asdict(fr)
    return json.dumps(output, indent=2)


#
# File‐naming helper
#
def get_unique_filename(output_dir: str, file_name: str) -> str:
    base, ext = os.path.splitext(file_name)
    pattern = re.compile(rf"^{re.escape(base)}\((\d+)\){re.escape(ext)}$")
    existing = os.listdir(output_dir)
    # collect all suffix‐numbers
    nums = [
        int(m.group(1))
        for f in existing
        if (m := pattern.match(f))
    ]
    # if the exact name exists, add or bump
    full_path = os.path.join(output_dir, file_name)
    if os.path.exists(full_path):
        if not nums:
            return f"{base}(1){ext}"
        return f"{base}({max(nums) + 1}){ext}"
    return file_name


#
# high‐level “export” that writes a brand‐new CSV file
#
def export_accumulated_results(
        accumulated: Iterable[Tuple[Any, ...]],
        output_dir: str,
        file_name: str = "test_results.csv"
) -> Optional[str]:
    """
    Write a CSV file under output_dir, picking a unique filename if needed.
    Returns the actual filename used or None if output_dir was falsy.
    """
    if not output_dir:
        return None

    data = create_csv_data(accumulated)
    os.makedirs(output_dir, exist_ok=True)
    unique_name = get_unique_filename(output_dir, file_name)
    path = os.path.join(output_dir, unique_name)

    with open(path, "w", newline="") as fh:
        fh.write(data)

    return unique_name
