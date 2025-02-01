import argparse
import os
from alveoleye.figure_scripts.utils import get_image_paths
from alveoleye.figure_scripts.combined_workers import CombinedWorker
from alveoleye._export_operations import export_from_combined_worker


def validate_arguments(input_dir, iterations, output_dir):
    if not os.path.isdir(input_dir):
        raise ValueError(f"Error: The specified input directory '{input_dir}' does not exist or is not a directory.")

    if not isinstance(iterations, int) or iterations < 2:
        raise ValueError("Error: The number of iterations must be an integer greater than or equal to 2.")

    if output_dir and not os.access(os.path.dirname(output_dir) or '.', os.W_OK):
        raise ValueError(f"Error: Output directory is not writable: {output_dir}")


def run_determinism_test(combined_worker, image_paths, iterations):
    for image_path in image_paths[:2]:
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


def main(input_dir, iterations, output_dir):
    combined_worker = CombinedWorker()
    image_paths = get_image_paths(input_dir)

    if not image_paths:
        raise ValueError(f"Error: The specified input directory '{input_dir}' does not contain any images")

    run_determinism_test(combined_worker, image_paths, iterations)

    if output_dir:
        export_from_combined_worker(output_dir, combined_worker)


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_dir = os.path.abspath(os.path.join(script_dir, "../example_images"))

    parser = argparse.ArgumentParser(description="Verify that AlveolEye is deterministic")
    parser.add_argument("--input_dir", type=str, required=False, default=default_input_dir,
                        help="Path to the directory containing images (default: ../../example_images)")
    parser.add_argument("--iterations", type=int, required=False, default=15,
                        help="The number of iterations that the program runs the complete pipeline on each image (default: 15)")
    parser.add_argument("--output_dir", type=str, required=False,
                        help="The export location if you intend to export results (optional)")

    args = parser.parse_args()

    try:
        validate_arguments(args.input_dir, args.iterations, args.output_dir)
    except ValueError as e:
        print(e)
        exit(1)

    print(f"Running with the following arguments:\n"
          f"Input Directory: {args.input_dir}\n"
          f"Iterations: {args.iterations}\n"
          f"Output Directory: {args.output_dir}\n")

    main(args.input_dir, args.iterations, args.output_dir)
