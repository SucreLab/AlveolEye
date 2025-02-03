import argparse
import os
import time

from PIL import Image

from alveoleye.figure_scripts._utils import get_image_paths
from alveoleye.figure_scripts._combined_workers import CombinedWorker
from alveoleye._export_operations import export_accumulated_results


def validate_arguments(args):
    if args.trial not in ["determinism", "random_line_location", "variable_line_quantity"]:
        raise ValueError(f"[-] Error: The specified trial type '{args.trial}' is unrecognized")

    if not os.path.isdir(args.input_dir):
        raise ValueError(
            f"[-] Error: The specified input directory '{args.input_dir}' does not exist or is not a directory.")

    if not isinstance(args.iterations, int) or args.iterations < 2:
        raise ValueError("[-] Error: The number of iterations must be an integer greater than or equal to 2.")

    if args.trial != "determinism" and not args.output_dir:
        raise ValueError("[-] Error: You must enter an output directory for this trial")

    if args.output_dir and not os.access(os.path.dirname(args.output_dir) or '.', os.W_OK):
        raise ValueError(f"[-] Error: Output directory is not writable: {args.output_dir}")


def print_arguments(args):
    print(f"[+] Running {args.trial} trial with the following arguments:\n\n"
          f"    Input Directory: {args.input_dir}\n"
          f"    Iterations: {args.iterations}\n"
          f"    Output Directory: {args.output_dir}\n")


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
    for image_path in image_paths[:2]:
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

    if args.trial == "determinism":
        run_determinism_trial(combined_worker, image_paths, args.iterations)
    elif args.trial == "random_line_location":
        run_randomized_line_location_trial(combined_worker, image_paths, args.iterations)
    elif args.trial == "variable_line_quantity":
        run_variable_number_of_lines_trial(combined_worker, image_paths, args.iterations)
    else:
        raise ValueError(
            "[-] Error: Invalid trial type specified. Choose from: determinism, random_lines, variable_lines")

    if args.output_dir:
        accumulated_results = combined_worker.get_accumulated_results()
        complete_export_path = f"{args.trial}_results.csv"
        export_accumulated_results(accumulated_results, args.output_dir, complete_export_path)

        print(f"[+] CSV file saved to: {complete_export_path}")


def main(args):
    start_time = time.time()
    run_trial(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.abspath(os.path.join(script_dir, "../example_images"))

    parser = argparse.ArgumentParser(description="Run specific trials for AlveolEye")
    parser.add_argument("trial", type=str,
                        choices=["determinism", "random_line_location", "variable_line_quantity"],
                        help="specify which trial to run")
    parser.add_argument("--input-dir", type=str, required=False, default=default_input_dir,
                        help="path to the directory containing images (default: ../example_images)")
    parser.add_argument("--iterations", type=int, required=False, default=15,
                        help="number of iterations per image as well as the range of number of lines for the variable_line_quantity trial (default: 15)")
    parser.add_argument("--output-dir", type=str, required=False,
                        help="export location for results (optional for determinism trial; required otherwise)")

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
