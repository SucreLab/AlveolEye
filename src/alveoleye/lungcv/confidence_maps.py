import argparse
import os
import torch
from model_operations import init_trained_model, run_prediction
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import time


def create_heatmaps(model_path, image_path, output_dir, colorbar_orientation="vertical"):
    model = init_trained_model(model_path)
    prediction = run_prediction(image_path, model)
    masks = prediction["masks"]
    labels = prediction["labels"]
    original_image = np.array(Image.open(image_path).convert("RGB"))
    image_name = os.path.splitext(os.path.basename(image_path))[0]

    class_colormaps = {
        1: 'Blues',
        2: 'Reds',
    }

    heatmaps = {}
    for class_label, colormap in class_colormaps.items():
        combined_heatmap = create_combined_heatmap(masks, labels, class_label)
        heatmaps[class_label] = combined_heatmap
        save_heatmap_image(original_image, combined_heatmap, colormap, image_name, class_label, output_dir,
                           colorbar_orientation)

    save_combined_overlap_image(original_image, heatmaps, image_name, output_dir, colorbar_orientation)


def create_combined_heatmap(masks, labels, class_label):
    return torch.stack([mask.squeeze() for mask, label in zip(masks, labels) if label == class_label]).max(dim=0)[
        0].detach().numpy()


def save_heatmap_image(original_image, combined_heatmap, colormap, image_name, class_label, output_dir,
                       colorbar_orientation):
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image, alpha=1.0)
    heatmap = plt.imshow(combined_heatmap, cmap=colormap, alpha=0.75, interpolation='nearest', vmin=0, vmax=1)
    plt.axis('off')
    cbar = plt.colorbar(heatmap, orientation=colorbar_orientation, ticks=np.linspace(0, 1, 6))
    cbar.set_label('Confidence Level', fontsize=10)
    class_output_dir = os.path.join(output_dir, image_name)
    os.makedirs(class_output_dir, exist_ok=True)
    plt.savefig(f"{class_output_dir}/{image_name}.{class_label}.png", bbox_inches='tight')
    plt.close()


def save_combined_overlap_image(original_image, heatmaps, image_name, output_dir, colorbar_orientation):
    plt.figure(figsize=(8, 6))
    plt.imshow(original_image, alpha=1.0)
    plt.imshow(heatmaps[1], cmap='Blues', alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
    plt.imshow(heatmaps[2], cmap='Reds', alpha=0.5, interpolation='nearest', vmin=0, vmax=1)
    plt.axis('off')
    cbar1 = plt.colorbar(plt.imshow(heatmaps[1], cmap='Blues', alpha=0.5, interpolation='nearest', vmin=0, vmax=1),
                         orientation=colorbar_orientation, ticks=np.linspace(0, 1, 6))
    cbar1.set_label('Confidence Level Class 1', fontsize=10)
    cbar2 = plt.colorbar(plt.imshow(heatmaps[2], cmap='Reds', alpha=0.5, interpolation='nearest', vmin=0, vmax=1),
                         orientation=colorbar_orientation, ticks=np.linspace(0, 1, 6))
    cbar2.set_label('Confidence Level Class 2', fontsize=10)
    class_output_dir = os.path.join(output_dir, image_name)
    plt.savefig(f"{class_output_dir}/{image_name}_combined.png", bbox_inches='tight')
    plt.close()


def process_images_in_directory(directory_path, model_path, output_dir, colorbar_orientation="vertical"):
    output_dir = os.path.join(output_dir, "confidence_maps")
    image_files = [image_name for image_name in os.listdir(directory_path) if
                   image_name.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)

    print(f"[+] Producing confidence maps for {total_images}")

    for idx, image_name in enumerate(image_files):
        image_path = os.path.join(directory_path, image_name)
        create_heatmaps(model_path, image_path, output_dir, colorbar_orientation)
        print(f"[+] Processed {idx + 1}/{total_images} images", end="\r")

    print(f"[+] Produced {total_images}/{total_images} confidence maps")


def main(input_dir, model_path, output_dir, colorbar_orientation="vertical"):
    start_time = time.time()
    process_images_in_directory(input_dir, model_path, output_dir, colorbar_orientation)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Total Processing Time: {elapsed_time:.2f} seconds")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))

    default_input_dir = os.path.abspath(os.path.join(script_dir, "../data"))
    default_model_path = os.path.abspath(os.path.join(script_dir, "../../../data/default.pth"))

    parser = argparse.ArgumentParser(description="Generate heatmaps for each image in a directory")
    parser.add_argument("--input_dir", type=str, default=default_input_dir,
                        help="Path to the directory containing images (default: ../../data)")
    parser.add_argument("--model_path", type=str, default=default_model_path,
                        help="Path to the trained model file (default: ../../../../data/default.pth)")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Directory to save the output images (required)")
    parser.add_argument("--colorbar_orientation", type=str, choices=["vertical", "horizontal"],
                        default="vertical", help="Orientation of the colorbar (default: vertical)")

    args = parser.parse_args()

    print(f"Running with the following arguments:\n"
          f"Input Directory: {args.input_dir}\n"
          f"Model Path: {args.model_path}\n"
          f"Output Directory: {args.output_dir}\n"
          f"Colorbar Orientation: {args.colorbar_orientation}")

    main(args.input_dir, args.model_path, args.output_dir, args.colorbar_orientation)
