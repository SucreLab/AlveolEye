import argparse
import os
import time

from PIL import Image

from alveoleye.figure_scripts._utils import get_image_paths
from alveoleye.figure_scripts._combined_workers import CombinedWorker
from alveoleye._export_operations import export_accumulated_results


def validate_arguments(args):
    if args.trial not in ["determinism_trial", "random_line_location_trial", "variable_line_quantity_trial", "1", "2", "3"]:
        raise ValueError(f"[-] Error: The specified trial type '{args.trial}' is unrecognized")

    if args.trial in ["1", "2", "3"]:
        trial_map = {"1": "determinism_trial", "2": "random_line_location_trial", "3": "variable_line_quantity_trial"}
        args.trial = trial_map.get(args.trial, "determinism_trial")  # Default to "determinism_trial" if something goes wrong

    if not os.path.isdir(args.input_dir):
        raise ValueError(
            f"[-] Error: The specified input directory '{args.input_dir}' does not exist or is not a directory.")

    if not isinstance(args.iterations, int) or args.iterations < 2:
        raise ValueError("[-] Error: The number of iterations must be an integer greater than or equal to 2.")

    if args.trial != "determinism_trial" and not args.output_dir:
        raise ValueError("[-] Error: You must enter an output directory for this trial")

    if args.output_dir and not os.access(os.path.dirname(args.output_dir) or '.', os.W_OK):
        raise ValueError(f"[-] Error: Output directory is not writable: {args.output_dir}")

    if args.weights_path and not os.path.isfile(args.weights_path):
        raise ValueError(f"[-] Error: The specified weights path '{args.weights_path}' does not exist or is not a file.")


def print_arguments(args):
    if args.weights_path:
        weights_path = args.weights_path
    else:
        weights_path = "using default weights"

    print(f"[+] Running {args.trial} trial with the following arguments:\n\n"
          f"    Input Directory: {args.input_dir}\n"
          f"    Iterations: {args.iterations}\n"
          f"    Output Directory: {args.output_dir}\n"
          f"    Weights File: {weights_path}\n")


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
                print("[+] Model is non-deterministic")
                return False

    print("[+] Model is deterministic")
    return True


def run_randomized_line_location_trial(combined_worker, image_paths, iterations):
    combined_worker.set_randomized_distribution(True)
    for image_path in image_paths:
        combined_worker.set_image_path(image_path)
        combined_worker.run_processing()
        combined_worker.run_postprocessing()

        for _ in range(iterations):
            combined_worker.run_assessments()


def run_variable_number_of_lines_trial(combined_worker, image_paths, iterations):
    for image_path in image_paths:
        combined_worker.set_image_path(image_path)
        combined_worker.run_processing()
        combined_worker.run_postprocessing()

        with Image.open(image_path) as image:
            height = image.height

        if iterations > height:
            print(f"[!] Warning: Iterations exceeds height of the image; using image height ({height}) instead")

        max_number_of_lines = min(iterations, height) + 1

        for lines in range(1, max_number_of_lines):
            combined_worker.set_number_of_lines(lines)
            combined_worker.run_assessments()


def run_trial(args):
    image_paths = get_image_paths(args.input_dir)
    if not image_paths:
        raise ValueError(f"[-] Error: The specified input directory '{args.input_dir}' does not contain any images")

    combined_worker = CombinedWorker()
    combined_worker.set_weights_path(args.weights_path if args.weights_path else None)

    if args.trial == "determinism_trial":
        run_determinism_trial(combined_worker, image_paths, args.iterations)
    elif args.trial == "random_line_location_trial":
        run_randomized_line_location_trial(combined_worker, image_paths, args.iterations)
    elif args.trial == "variable_line_quantity_trial":
        run_variable_number_of_lines_trial(combined_worker, image_paths, args.iterations)
    else:
        raise ValueError(
            "[-] Error: Invalid trial type specified. Choose from: determinism_trial, random_lines, variable_lines")

    if args.output_dir:
        accumulated_results = combined_worker.get_accumulated_results()
        base_file_name = f"{args.trial}_results.csv"
        export_file_name = export_accumulated_results(accumulated_results, args.output_dir, base_file_name)

        print(f"[+] Saved trial results as {export_file_name}")


def main(args):
    start_time = time.time()
    run_trial(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.abspath(os.path.join(script_dir, "../../example_images"))

    parser = argparse.ArgumentParser(description="Run specific trials for AlveolEye")
    parser.add_argument("trial", type=str,
                        choices=["determinism_trial", "random_line_location_trial", "variable_line_quantity_trial", "1", "2", "3"],
                        help="specify which trial to run (or use numbers 1, 2, 3 as shortcuts)")
    parser.add_argument("--input-dir", type=str, required=False, default=default_input_dir,
                        help="path to the directory containing images (default: ../../example_images")
    parser.add_argument("--iterations", type=int, required=False, default=15,
                        help="number of iterations per image as well as the range of number of lines for the variable_line_quantity trial (default: 15)")
    parser.add_argument("--output-dir", type=str, required=False,
                        help="export location for results (optional for determinism trial; required otherwise)")
    parser.add_argument("--weights-path", type=str, required=False, default=None,
                        help="path to the model weights file (optional)")

    args = parser.parse_args()

    try:
        validate_arguments(args)
    except ValueError as e:
        print(e)
        exit(1)

    print_arguments(args)

    try:
        main(args)
    except Exception as e:
        print(e)
        exit(1)
