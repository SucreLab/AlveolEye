import csv
import io
import json
import os
import re

import numpy as np
from PIL import Image
import torch


def format_results(result):
    (image_file_name, weights_file_name, asvd, mli, stdev, chords, airspace_pixels, non_airspace_pixels,
     lines_spin_box_value, min_length_spin_box_value, scale_spin_box_value) = result

    asvd = None if not asvd else float(asvd)
    mli = None if not mli else float(mli)
    stdev = None if stdev in ('', 'NA') else float(stdev)
    chords = None if not chords else int(chords)
    airspace_pixels = None if not airspace_pixels else int(airspace_pixels)
    non_airspace_pixels = None if not non_airspace_pixels else int(non_airspace_pixels)

    return (image_file_name, weights_file_name, asvd, mli, stdev, chords, airspace_pixels, non_airspace_pixels,
            lines_spin_box_value, min_length_spin_box_value, scale_spin_box_value)


def create_json_data(accumulated_results):
    data = {}

    for result in accumulated_results:
        (image_file_name, weights_file_name, asvd, mli, stdev, chords, airspace_pixels, non_airspace_pixels,
         lines_spin_box_value, min_length_spin_box_value, scale_spin_box_value) = format_results(result)
        data[str(image_file_name)] = {
            "Weights": weights_file_name,
            "ASVD": {
                "asvd": asvd,
                "airspace_pixels": airspace_pixels,
                "non_airspace_pixels": non_airspace_pixels
            },
            "MLI": {
                "mli": mli,
                "standard_deviation": stdev,
                "number_of_chords": chords,
                "settings": {
                    "lines": lines_spin_box_value,
                    "minimum_length": min_length_spin_box_value,
                    "scale": scale_spin_box_value
                }
            }
        }

    return json.dumps(data, indent=2)


def create_csv_data(accumulated_results, field_names=("Image", "Weights", "ASVD", "Airspace Pixels",
                                                      "Non-Airspace Pixels", "MLI", "Standard Deviation",
                                                      "Number of Chords", "Number of Lines", "Minimum Length",
                                                      "Scale")):
    csv_buffer = io.StringIO()
    writer = csv.DictWriter(csv_buffer, fieldnames=field_names)
    writer.writeheader()

    for result in accumulated_results:
        (image_file_name, weights_file_name, asvd, mli, stdev, chords, airspace_pixels, non_airspace_pixels,
         lines_spin_box_value, min_length_spin_box_value, scale_spin_box_value) = format_results(result)

        writer.writerow({
            "Image": image_file_name,
            "Weights": weights_file_name,
            "ASVD": asvd,
            "Airspace Pixels": airspace_pixels,
            "Non-Airspace Pixels": non_airspace_pixels,
            "MLI": mli,
            "Standard Deviation": stdev,
            "Number of Chords": chords,
            "Number of Lines": lines_spin_box_value,
            "Minimum Length": min_length_spin_box_value,
            "Scale": scale_spin_box_value
        })

    csv_data = csv_buffer.getvalue()
    csv_buffer.close()

    return csv_data


def append_csv_data(accumulated_results, export_file):
    csv_data = create_csv_data(accumulated_results)

    file_exists = os.path.exists(export_file)
    mode = 'a' if file_exists else 'w'

    with open(export_file, mode) as file:
        if file_exists:
            csv_lines = csv_data.splitlines()[1:]
            file.write('\n'.join(csv_lines) + '\n')
        else:
            file.write(csv_data)


def get_unique_filename(output_dir, file_name):
    base_name, ext = os.path.splitext(file_name)
    pattern = re.compile(rf"{re.escape(base_name)}\((\d+)\){re.escape(ext)}")

    existing_files = os.listdir(output_dir)
    matching_numbers = [int(match.group(1)) for f in existing_files if (match := pattern.match(f))]

    if os.path.exists(os.path.join(output_dir, file_name)):
        if not matching_numbers:
            return f"{base_name}(1){ext}"
        else:
            return f"{base_name}({max(matching_numbers) + 1}){ext}"

    return file_name


def export_accumulated_results(accumulated_results, output_dir, file_name="test_results.csv"):
    if not output_dir:
        return

    csv_data = create_csv_data(accumulated_results)

    os.makedirs(output_dir, exist_ok=True)
    unique_file_name = get_unique_filename(output_dir, file_name)
    complete_export_path = os.path.join(output_dir, unique_file_name)

    with open(complete_export_path, "w", newline="") as results_file:
        results_file.write(csv_data)

    return unique_file_name


def save_image(data, name, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    base_name, ext = os.path.splitext(name)
    candidate_name = name
    counter = 1

    existing_files = set(os.listdir(save_dir))

    while candidate_name in existing_files:
        candidate_name = f"{base_name}({counter}){ext}"
        counter += 1

    save_path = os.path.join(save_dir, candidate_name)

    # Convert tensor to numpy
    if isinstance(data, torch.Tensor):
        data = data.detach().cpu().numpy()

    # Process numpy array
    if isinstance(data, np.ndarray):
        # Squeeze out singleton dimensions
        data = np.squeeze(data)

        # Ensure it's in uint8 format
        if data.dtype != np.uint8:
            data = (255 * (data - data.min()) / (data.ptp() + 1e-8)).astype(np.uint8)

        # Create image based on number of dimensions
        if data.ndim == 2:
            image = Image.fromarray(data, mode='L')
        elif data.ndim == 3 and data.shape[2] in {1, 3, 4}:
            image = Image.fromarray(data)
        else:
            raise ValueError(f"Unsupported image shape after squeeze: {data.shape}")
    else:
        raise ValueError(f"Unsupported data type: {type(data)}")

    image.save(save_path)


def make_save_image_callback(save_dir):
    snapshots_dir = os.path.join(save_dir, "snapshots")

    def save_image_callback(data, name):
        save_image(data, name, snapshots_dir)

    return save_image_callback
