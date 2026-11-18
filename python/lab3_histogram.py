"""
CSST106 Lab 3: Histogram Equalization and Contrast Adjustment
Performs linear contrast adjustment, global histogram equalization,
CLAHE (Contrast Limited Adaptive Histogram Equalization), and color image equalization.
"""
import os
import sys
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def main():
    args = parse_args()
    image_path = args.get("image_path")
    output_dir = args.get("output_dir", "")
    params = args.get("params", {}) or {}

    if not image_path or not os.path.exists(image_path):
        raise ValueError(f"Image not found at path: {image_path}")

    # Parse and validate parameters
    alpha = float(params.get("alpha", 1.5))
    beta = int(params.get("beta", 0))
    clip_limit = float(params.get("clip_limit", 2.0))
    tile_size = int(params.get("tile_size", 8))
    if tile_size < 1:
        tile_size = 8

    # 1. Load image, convert to grayscale
    color_img = read_image(image_path)
    if color_img is None:
        raise ValueError(f"Failed to load image from: {image_path}")

    if len(color_img.shape) == 2:
        gray = color_img.copy()
        color_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif color_img.shape[2] == 4:
        color_img = cv2.cvtColor(color_img, cv2.COLOR_BGRA2BGR)
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(color_img, cv2.COLOR_BGR2GRAY)

    # 2. Apply linear contrast: cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)
    linear_contrast = cv2.convertScaleAbs(gray, alpha=alpha, beta=beta)

    # 3. Apply global histogram equalization: cv2.equalizeHist(gray)
    global_equalized = cv2.equalizeHist(gray)

    # 4. Apply CLAHE: cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size)).apply(gray)
    clahe_obj = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    clahe_img = clahe_obj.apply(gray)

    # 5. Create matplotlib figure (figsize 16x12) with 2 rows x 4 columns:
    # Row 1: Original | Linear Contrast | Global Equalized | CLAHE
    # Row 2: Histogram of Original | Histogram of Linear | Histogram of Global | Histogram of CLAHE
    # Histograms should use plt.hist with 256 bins, color='#89b4fa', alpha=0.7
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(16, 12))

    # Row 1: Images
    axes[0, 0].imshow(gray, cmap="gray")
    axes[0, 0].set_title("Original", fontsize=13, fontweight="bold", pad=8)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(linear_contrast, cmap="gray")
    axes[0, 1].set_title(f"Linear Contrast (α={alpha}, β={beta})", fontsize=13, fontweight="bold", pad=8)
    axes[0, 1].axis("off")

    axes[0, 2].imshow(global_equalized, cmap="gray")
    axes[0, 2].set_title("Global Equalized", fontsize=13, fontweight="bold", pad=8)
    axes[0, 2].axis("off")

    axes[0, 3].imshow(clahe_img, cmap="gray")
    axes[0, 3].set_title(f"CLAHE (clip={clip_limit}, tile={tile_size}x{tile_size})", fontsize=13, fontweight="bold", pad=8)
    axes[0, 3].axis("off")

    # Row 2: Histograms
    axes[1, 0].hist(gray.ravel(), bins=256, range=[0, 256], color="#89b4fa", alpha=0.7)
    axes[1, 0].set_title("Histogram of Original", fontsize=12, pad=8)
    axes[1, 0].set_xlim([0, 256])
    axes[1, 0].set_xlabel("Pixel Intensity")
    axes[1, 0].set_ylabel("Pixel Count")
    axes[1, 0].grid(True, linestyle="--", alpha=0.3)

    axes[1, 1].hist(linear_contrast.ravel(), bins=256, range=[0, 256], color="#89b4fa", alpha=0.7)
    axes[1, 1].set_title("Histogram of Linear", fontsize=12, pad=8)
    axes[1, 1].set_xlim([0, 256])
    axes[1, 1].set_xlabel("Pixel Intensity")
    axes[1, 1].set_ylabel("Pixel Count")
    axes[1, 1].grid(True, linestyle="--", alpha=0.3)

    axes[1, 2].hist(global_equalized.ravel(), bins=256, range=[0, 256], color="#89b4fa", alpha=0.7)
    axes[1, 2].set_title("Histogram of Global", fontsize=12, pad=8)
    axes[1, 2].set_xlim([0, 256])
    axes[1, 2].set_xlabel("Pixel Intensity")
    axes[1, 2].set_ylabel("Pixel Count")
    axes[1, 2].grid(True, linestyle="--", alpha=0.3)

    axes[1, 3].hist(clahe_img.ravel(), bins=256, range=[0, 256], color="#89b4fa", alpha=0.7)
    axes[1, 3].set_title("Histogram of CLAHE", fontsize=12, pad=8)
    axes[1, 3].set_xlim([0, 256])
    axes[1, 3].set_xlabel("Pixel Intensity")
    axes[1, 3].set_ylabel("Pixel Count")
    axes[1, 3].grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    plot_b64 = encode_matplotlib_figure(fig)
    plt.close(fig)

    # 6. Also process COLOR image: convert to YCrCb, equalize Y channel, convert back (for color equalization demo)
    ycrcb = cv2.cvtColor(color_img, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycrcb)
    y_eq = cv2.equalizeHist(y)
    ycrcb_eq = cv2.merge([y_eq, cr, cb])
    color_equalized = cv2.cvtColor(ycrcb_eq, cv2.COLOR_YCrCb2BGR)

    # Optionally save outputs if output directory is specified
    if output_dir:
        save_output_image(gray, "original_gray.png", output_dir)
        save_output_image(linear_contrast, "linear_contrast.png", output_dir)
        save_output_image(global_equalized, "global_equalized.png", output_dir)
        save_output_image(clahe_img, "clahe.png", output_dir)
        save_output_image(color_equalized, "color_equalized.png", output_dir)

    # Calculate statistics
    stats = {
        "original_mean": round(float(np.mean(gray)), 2),
        "original_std": round(float(np.std(gray)), 2),
        "equalized_mean": round(float(np.mean(global_equalized)), 2),
        "equalized_std": round(float(np.std(global_equalized)), 2),
        "clahe_mean": round(float(np.mean(clahe_img)), 2),
        "clahe_std": round(float(np.std(clahe_img)), 2),
    }

    # Base64 encoded images
    images = {
        "original": encode_image(gray),
        "linear_contrast": encode_image(linear_contrast),
        "global_equalized": encode_image(global_equalized),
        "clahe": encode_image(clahe_img),
        "color_equalized": encode_image(color_equalized),
    }

    # Respond with JSON output
    respond({
        "success": True,
        "images": images,
        "plot": plot_b64,
        "stats": stats,
    })


if __name__ == "__main__":
    run_safe(main)
