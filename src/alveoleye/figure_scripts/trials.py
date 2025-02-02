import argparse
import os

import cv2

from alveoleye.figure_scripts.utils import get_image_paths
from alveoleye.figure_scripts.combined_workers import CombinedWorker
from alveoleye._export_operations import export_from_combined_worker


def validate_arguments(trial_type, input_dir, iterations, output_dir):
    if trial_type not in ["determinism", "random_line_location", "variable_line_quantity"]:
        raise ValueError(f"Error: The specified trial type '{trial_type}' is unrecognized")

    if not os.path.isdir(input_dir):
        raise ValueError(f"Error: The specified input directory '{input_dir}' does not exist or is not a directory.")

    if trial_type != "variable_line_quantity" and (not isinstance(iterations, int) or iterations < 2):
        raise ValueError("Error: The number of iterations must be an integer greater than or equal to 2.")

    if trial_type != "determinism" and not output_dir:
        raise ValueError("Error: You must enter an output directory for this trial")

    if output_dir and not os.access(os.path.dirname(output_dir) or '.', os.W_OK):
        raise ValueError(f"Error: Output directory is not writable: {output_dir}")


def run_determinism_trial(combined_worker, image_paths, iterations):
    for image_path in image_paths:
        previous_result = None
        combined_worker.set_image_path(image_path)

        for _ in range(iterations):
            combined_worker.run_complete_pipline()
            current_results = combined_worker.get_current_results()

            if previous_result is None:
                previous_result = current_results
            elif previous_result != current_results:
                print("[-] Model is non-deterministic!")
                return False

    print("[+] Model is deterministic!")
    return True


def run_randomized_line_location_trial(combined_worker, image_paths, iterations):
    combined_worker.set_randomized_distribution(True)
    for image_path in image_paths:
        combined_worker.set_image_path(image_path)
        combined_worker.run_processing()
        combined_worker.run_postprocessing()

        for _ in range(iterations):
            combined_worker.run_assessments()


def run_variable_number_of_lines_trial(combined_worker, image_paths):
    for image_path in image_paths:
        combined_worker.set_image_path(image_path)
        combined_worker.run_processing()
        combined_worker.run_postprocessing()

        image = cv2.imread(image_path, cv2.IMREAD_COLOR)[:, :, ::-1]
        height = image.shape[0]

        for lines in range(1, height - 1):
            combined_worker.set_number_of_lines(lines)
            combined_worker.run_assessments()


def main(trial_type, input_dir, iterations, output_dir):
    image_paths = get_image_paths(input_dir)
    if not image_paths:
        raise ValueError(f"Error: The specified input directory '{input_dir}' does not contain any images")

    combined_worker = CombinedWorker()

    if trial_type == "determinism":
        run_determinism_trial(combined_worker, image_paths, iterations)
    elif trial_type == "random_line_location":
        run_randomized_line_location_trial(combined_worker, image_paths, iterations)
    elif trial_type == "variable_line_quantity":
        run_variable_number_of_lines_trial(combined_worker, image_paths)
    else:
        raise ValueError("Invalid trial type specified. Choose from: determinism, random_lines, variable_lines")

    if output_dir:
        export_from_combined_worker(combined_worker, output_dir, f"{trial_type}_results.csv")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.abspath(os.path.join(script_dir, "../example_images"))

    parser = argparse.ArgumentParser(description="Run specific trials for AlveolEye")
    parser.add_argument("trial_type", type=str,
                        choices=["determinism", "random_line_location", "variable_line_quantity"],
                        help="Specify which trial to run: determinism, random_line_location, or variable_line_quantity")
    parser.add_argument("--input_dir", type=str, required=False, default=default_input_dir,
                        help="Path to the directory containing images (default: ../example_images)")
    parser.add_argument("--iterations", type=int, required=False, default=15,
                        help="Number of iterations per image for determinism and random_line_location trials (default: 15)")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="Export location for results (optional)")

    args = parser.parse_args()

    try:
        validate_arguments(args.trial_type, args.input_dir, args.iterations, args.output_dir)
    except ValueError as e:
        print(e)
        exit(1)

    print(f"Running {args.trial_type} trial with the following arguments:\n"
          f"Input Directory: {args.input_dir}\n"
          f"Iterations: {args.iterations}\n"
          f"Output Directory: {args.output_dir}\n")

    main(args.trial_type, args.input_dir, args.iterations, args.output_dir)
