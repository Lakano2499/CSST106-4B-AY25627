"""
CSST106 Lab 2: Edge Detection Techniques (Sobel & Canny)
Applies Gaussian pre-filtering, first-order differential Sobel operators (X and Y gradients, combined magnitude),
and multi-stage Canny edge detection with hysteresis thresholding.
"""

import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def main():
    # Parse arguments from command line
    args = parse_args()
    image_path = args.get("image_path")
    output_dir = args.get("output_dir", "")
    params = args.get("params", {}) or {}

    # Validate image path
    if not image_path:
        raise ValueError("Missing required argument: 'image_path'")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Parse and sanitize parameters
    try:
        blur_kernel = int(params.get("blur_kernel", 3))
    except (TypeError, ValueError):
        blur_kernel = 3

    # Kernel size for Gaussian blur must be positive and odd
    if blur_kernel < 1:
        blur_kernel = 3
    elif blur_kernel % 2 == 0:
        blur_kernel += 1

    try:
        canny_low = int(params.get("canny_low", 100))
    except (TypeError, ValueError):
        canny_low = 100

    try:
        canny_high = int(params.get("canny_high", 200))
    except (TypeError, ValueError):
        canny_high = 200

    canny_low = max(0, min(300, canny_low))
    canny_high = max(0, min(300, canny_high))

    # 1. Load image in grayscale
    img = read_image(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Failed to load image from: {image_path}")

    height, width = img.shape[:2]

    # 2. Apply Gaussian blur with the specified kernel size
    blurred = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

    # 3. Compute Sobel X gradient: cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)
    sobelx = cv2.Sobel(blurred, cv2.CV_64F, 1, 0, ksize=3)

    # 4. Compute Sobel Y gradient: cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
    sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)

    # Convert gradients to uint8 (absolute scale) for encoding & display
    sobel_x = cv2.convertScaleAbs(sobelx)
    sobel_y = cv2.convertScaleAbs(sobely)

    # 5. Compute combined magnitude: cv2.magnitude(sobelx, sobely), normalize to uint8
    sobel_mag = cv2.magnitude(sobelx, sobely)
    sobel_combined = cv2.normalize(sobel_mag, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

    # 6. Apply Canny edge detection: cv2.Canny(blurred, canny_low, canny_high)
    canny = cv2.Canny(blurred, canny_low, canny_high)

    # 7. Create a matplotlib figure (figsize 16x10) with a 2x3 grid:
    #    Row 1: Original Grayscale | Gaussian Blurred | Sobel X
    #    Row 2: Sobel Y | Sobel Combined | Canny Edges
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))

    # Row 1
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Original Grayscale", fontsize=14, fontweight="bold", pad=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(blurred, cmap="gray")
    axes[0, 1].set_title(f"Gaussian Blurred ({blur_kernel}x{blur_kernel})", fontsize=14, fontweight="bold", pad=10)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(sobel_x, cmap="gray")
    axes[0, 2].set_title("Sobel X", fontsize=14, fontweight="bold", pad=10)
    axes[0, 2].axis("off")

    # Row 2
    axes[1, 0].imshow(sobel_y, cmap="gray")
    axes[1, 0].set_title("Sobel Y", fontsize=14, fontweight="bold", pad=10)
    axes[1, 0].axis("off")

    axes[1, 1].imshow(sobel_combined, cmap="gray")
    axes[1, 1].set_title("Sobel Combined", fontsize=14, fontweight="bold", pad=10)
    axes[1, 1].axis("off")

    axes[1, 2].imshow(canny, cmap="gray")
    axes[1, 2].set_title(f"Canny Edges ({canny_low}, {canny_high})", fontsize=14, fontweight="bold", pad=10)
    axes[1, 2].axis("off")

    fig.tight_layout()
    plot_b64 = encode_matplotlib_figure(fig)
    plt.close(fig)

    # 8. Count edge pixels for Sobel and Canny (nonzero pixels)
    sobel_edge_pixels = int(np.count_nonzero(sobel_combined))
    canny_edge_pixels = int(np.count_nonzero(canny))
    total_pixels = int(height * width)

    # Optionally save output images if output_dir is specified
    if output_dir:
        save_output_image(img, "original_gray.png", output_dir)
        save_output_image(blurred, "blurred.png", output_dir)
        save_output_image(sobel_x, "sobel_x.png", output_dir)
        save_output_image(sobel_y, "sobel_y.png", output_dir)
        save_output_image(sobel_combined, "sobel_combined.png", output_dir)
        save_output_image(canny, "canny.png", output_dir)

    # Build statistics object
    stats = {
        "sobel_edge_pixels": sobel_edge_pixels,
        "canny_edge_pixels": canny_edge_pixels,
        "image_dimensions": f"{width}x{height}",
        "sobel_edges": sobel_edge_pixels,
        "canny_edges": canny_edge_pixels,
        "total_pixels": total_pixels,
        "canny_density": round((canny_edge_pixels / total_pixels) * 100, 2) if total_pixels > 0 else 0.0,
    }

    # Encode all images to Base64 Data URIs
    images = {
        "original": encode_image(img),
        "blurred": encode_image(blurred),
        "sobel_x": encode_image(sobel_x),
        "sobel_y": encode_image(sobel_y),
        "sobel_combined": encode_image(sobel_combined),
        "canny": encode_image(canny),
    }

    # Return JSON response via respond()
    respond({
        "success": True,
        "images": images,
        "plot": plot_b64,
        "stats": stats,
        "params": {
            "blur_kernel": blur_kernel,
            "canny_low": canny_low,
            "canny_high": canny_high,
        },
    })


if __name__ == "__main__":
    run_safe(main)
