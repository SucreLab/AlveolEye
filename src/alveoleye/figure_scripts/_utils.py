import os
from pathlib import Path


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
