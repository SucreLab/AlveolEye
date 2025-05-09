import argparse
import os
import time

from alveoleye.figure_scripts._combined_workers import CombinedWorker
from alveoleye._export_operations import make_save_image_callback


def validate_arguments(args):
    if not os.path.exists(args.input_image):
        raise ValueError(f"Input image does not exist: {args.input_image}")
    if args.output_dir is None:
        raise ValueError("Output directory must be specified.")


def print_arguments(args):
    print(f"[+] Generating intermediate images with the following arguments:\n\n"
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

    print(f"Elapsed time: {end_time - start_time:.2f} seconds")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_input_image = os.path.abspath(os.path.join(script_dir, "../../example_images/10.png"))

    parser = argparse.ArgumentParser(description="Generate intermediate snapshots of image in AlveolEye pipeline")
    parser.add_argument("--input-image", type=str, required=False, default=default_input_image,
                        help="Path to the image (default: ../../example_images/10.png)")
    parser.add_argument("--weights-path", type=str, required=False, default=None,
                        help="Path to the model weights file (optional)")
    parser.add_argument("--output-dir", type=str, required=False,
                        help="Export location for results")

    args = parser.parse_args()

    try:
        validate_arguments(args)
        print_arguments(args)
        main(args)
    except Exception as e:
        print(f"[!] Error: {e}")
        exit(1)
