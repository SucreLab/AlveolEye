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

    original_image = np.array(Image.open(image_path).convert("RGB"))

    combined_heatmap = torch.stack([mask.squeeze() for mask in masks]).sum(dim=0).detach().numpy()

    plt.figure(figsize=(8, 8))
    plt.imshow(original_image, alpha=1.0)
    plt.imshow(combined_heatmap, cmap='hot', alpha=0.75, interpolation='nearest')
    plt.colorbar(label='Confidence Level')
    plt.title(f'{image_path.split("/")[-1]} Confidence Map')
    plt.axis('off')

    if output_path:
        plt.savefig(output_path, bbox_inches='tight')
    else:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Overlay heatmap on an image")
    parser.add_argument("model_path", type=str, help="Path to the trained model file")
    parser.add_argument("image_path", type=str, help="Path to the input image")
    parser.add_argument("--output_path", type=str, help="Path to save the output image (optional)")

    args = parser.parse_args()
    main(args.model_path, args.image_path, args.output_path)
