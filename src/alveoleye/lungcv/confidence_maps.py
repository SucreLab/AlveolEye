import argparse
import torch
from model_operations import init_trained_model, run_prediction
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def main(model_path, image_path, output_path=None):
    model = init_trained_model(model_path)
    prediction = run_prediction(image_path, model)
    masks = prediction["masks"]
    labels = prediction["labels"]  # Assuming labels are provided as a tensor or list

    print(prediction)

    original_image = np.array(Image.open(image_path).convert("RGB"))

    # Create confidence maps for class 1 and class 2
    combined_heatmap_class1 = torch.stack(
        [mask.squeeze() for mask, label in zip(masks, labels) if label == 1]
    ).sum(dim=0).detach().numpy()

    combined_heatmap_class2 = torch.stack(
        [mask.squeeze() for mask, label in zip(masks, labels) if label == 2]
    ).sum(dim=0).detach().numpy()

    # Create the figure and axes with a vertical colorbar on the right
    fig, axs = plt.subplots(1, 2, figsize=(8, 4), gridspec_kw={"wspace": 0.1, "right": 0.85})

    # Class 1 heatmap
    axs[0].imshow(original_image, alpha=1.0)
    axs[0].imshow(combined_heatmap_class1, cmap='hot', alpha=0.75, interpolation='nearest')
    axs[0].set_title('Confidence Map: Class 1', fontsize=10)
    axs[0].axis('off')

    # Class 2 heatmap
    axs[1].imshow(original_image, alpha=1.0)
    axs[1].imshow(combined_heatmap_class2, cmap='hot', alpha=0.75, interpolation='nearest')
    axs[1].set_title('Confidence Map: Class 2', fontsize=10)
    axs[1].axis('off')

    # Single vertical colorbar on the right
    cbar_ax = fig.add_axes([0.87, 0.15, 0.02, 0.7])  # [left, bottom, width, height]
    cbar = fig.colorbar(
        plt.cm.ScalarMappable(cmap='hot'),
        cax=cbar_ax
    )
    cbar.set_label('Confidence Level', fontsize=10)

    # Save or display the figure
    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay heatmaps for different classes on an image")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output_path", type=str, help="Path to save the output image (optional)")

    args = parser.parse_args()
    main(args.model_path, args.image_path, args.output_path)
