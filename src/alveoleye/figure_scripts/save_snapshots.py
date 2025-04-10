import argparse
import os
import time

from alveoleye.figure_scripts._combined_workers import CombinedWorker
from alveoleye._export_operations import make_save_image_callback


def validate_arguments(args):
    pass


def print_arguments(args):
    print(f"[+] Generating intermediate images with the following argument:\n\n"
          f"    Input Image: {args.input_image}\n"
          f"    Output Directory: {args.output_dir}\n")


def generate_intermediate_snapshots(args):
    combined_worker = CombinedWorker()
    combined_worker.set_image_path(args.input_image)
    combined_worker.set_weights_path(args.weights_path if args.weights_path else None)
    combined_worker.set_callback(make_save_image_callback(args.output_dir))

    combined_worker.run_complete_pipline()


def main(args):
    start_time = time.time()
    generate_intermediate_snapshots(args)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_image = os.path.abspath(os.path.join(script_dir, "../../example_images/3.png"))

    parser = argparse.ArgumentParser(description="Generate intermediate snapshots of image in AlveolEye pipeline")
    parser.add_argument("--input-image", type=str, required=False, default=default_input_image,
                        help="path to the image (default: ../../example_images/3.png")
    parser.add_argument("--weights-path", type=str, required=False, default=None,
                        help="path to the model weights file (optional)")
    parser.add_argument("--output-dir", type=str, required=False, help="export location for results")

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