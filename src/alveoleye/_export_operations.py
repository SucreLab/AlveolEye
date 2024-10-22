import csv
import io
import json

def format_results(result):
    (image_file_name, weights_file_name, asvd, mli, stdev, chords, airspace_pixels, non_airspace_pixels,
     lines_spin_box_value, min_length_spin_box_value, scale_spin_box_value) = result

    asvd = float(asvd)
    mli = float(mli)
    stdev = stdev if stdev == "NA" else float(stdev)
    chords = int(chords)
    airspace_pixels = int(airspace_pixels)
    non_airspace_pixels = int(non_airspace_pixels)

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


def create_csv_data(accumulated_results):
    field_names = ["Image", "Weights", "ASVD", "Airspace Pixels", "Non-Airspace Pixels", "MLI", "Standard Deviation",
                   "Number of Chords", "Lines", "Minimum Length", "Scale"]

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
            "Lines": lines_spin_box_value,
            "Minimum Length": min_length_spin_box_value,
            "Scale": scale_spin_box_value
        })

    csv_data = csv_buffer.getvalue()
    csv_buffer.close()

    return csv_data
