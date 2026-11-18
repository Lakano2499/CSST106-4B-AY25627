"""
CSST106 Lab 1: Introduction to Computer Vision - Image Manipulation
Performs image loading, property inspection, pixel reading/editing, cropping, resizing, and rotation.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(__file__))
from utils import *
import cv2
import numpy as np


def handle_full_analysis(img, output_dir):
    """
    Perform full image analysis:
    - Dimensions, dtype, total pixels, min/max values
    - Convert to grayscale
    - Render matplotlib comparison figure
    - Return JSON response
    """
    # 1. Properties
    shape = img.shape
    height = int(shape[0])
    width = int(shape[1])
    channels = int(shape[2]) if len(shape) > 2 else 1
    total_pixels = height * width
    min_pixel = int(np.min(img))
    max_pixel = int(np.max(img))

    properties = {
        "height": height,
        "width": width,
        "channels": channels,
        "dtype": str(img.dtype),
        "total_pixels": total_pixels,
        "color_space": "BGR",
        "min_pixel": min_pixel,
        "max_pixel": max_pixel,
    }

    # 2. Grayscale conversion
    if len(shape) == 2 or channels == 1:
        gray = img.copy()
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    else:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 3. Matplotlib figure with 2 subplots (Original BGR->RGB and Grayscale)
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(img_rgb)
    axes[0].set_title(f"Original (RGB) - {width}x{height}")
    axes[0].set_xlabel("X (Width)")
    axes[0].set_ylabel("Y (Height)")

    axes[1].imshow(gray, cmap="gray")
    axes[1].set_title(f"Grayscale - {width}x{height}")
    axes[1].set_xlabel("X (Width)")
    axes[1].set_ylabel("Y (Height)")

    fig.tight_layout()
    plot_b64 = encode_matplotlib_figure(fig)
    plt.close(fig)

    # 4. Encode images
    original_b64 = encode_image(img)
    grayscale_b64 = encode_image(gray)

    # Optional saving to output_dir
    if output_dir:
        save_output_image(img, "original.png", output_dir)
        save_output_image(gray, "grayscale.png", output_dir)

    respond({
        "success": True,
        "properties": properties,
        "images": {
            "original": original_b64,
            "grayscale": grayscale_b64,
        },
        "plot": plot_b64,
    })


def handle_pixel_inspect(img, params):
    """
    Read pixel value at (x, y) and return {success, pixel: {b, g, r}, position: {x, y}}.
    """
    x = int(params.get("x", 0))
    y = int(params.get("y", 0))

    height, width = img.shape[:2]
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"Coordinates ({x}, {y}) out of bounds for image dimensions ({width}x{height})")

    if len(img.shape) == 2 or img.shape[2] == 1:
        val = int(img[y, x])
        b, g, r = val, val, val
    else:
        b = int(img[y, x, 0])
        g = int(img[y, x, 1])
        r = int(img[y, x, 2])

    respond({
        "success": True,
        "pixel": {"b": b, "g": g, "r": r},
        "position": {"x": x, "y": y},
    })


def handle_modify_pixel(img, params, output_dir):
    """
    Set pixel at (x, y) to (b, g, r) and return modified image base64.
    """
    x = int(params.get("x", 0))
    y = int(params.get("y", 0))
    r = int(params.get("r", 0))
    g = int(params.get("g", 0))
    b = int(params.get("b", 0))

    height, width = img.shape[:2]
    if x < 0 or x >= width or y < 0 or y >= height:
        raise ValueError(f"Coordinates ({x}, {y}) out of bounds for image dimensions ({width}x{height})")

    # Clamp color values to valid uint8 range 0-255
    r = max(0, min(255, r))
    g = max(0, min(255, g))
    b = max(0, min(255, b))

    modified = img.copy()
    if len(modified.shape) == 2:
        modified = cv2.cvtColor(modified, cv2.COLOR_GRAY2BGR)

    modified[y, x] = [b, g, r]

    encoded = encode_image(modified)
    if output_dir:
        save_output_image(modified, "modified.png", output_dir)

    respond({
        "success": True,
        "image": encoded,
        "images": {"modified": encoded},
        "position": {"x": x, "y": y},
        "pixel": {"b": b, "g": g, "r": r},
    })


def handle_crop(img, params, output_dir):
    """
    Crop ROI img[y1:y2, x1:x2] and return cropped image base64.
    """
    height, width = img.shape[:2]

    x1 = int(params.get("x1", 0))
    y1 = int(params.get("y1", 0))
    x2 = int(params.get("x2", width))
    y2 = int(params.get("y2", height))

    # Normalize bounds
    x_start = max(0, min(x1, x2))
    y_start = max(0, min(y1, y2))
    x_end = min(width, max(x1, x2))
    y_end = min(height, max(y1, y2))

    if x_start >= x_end or y_start >= y_end:
        raise ValueError(
            f"Invalid crop ROI [{y1}:{y2}, {x1}:{x2}] for image of dimensions ({width}x{height})"
        )

    cropped = img[y_start:y_end, x_start:x_end]
    encoded = encode_image(cropped)
    if output_dir:
        save_output_image(cropped, "cropped.png", output_dir)

    respond({
        "success": True,
        "image": encoded,
        "images": {"cropped": encoded},
        "roi": {"x1": x_start, "y1": y_start, "x2": x_end, "y2": y_end},
        "dimensions": {
            "width": int(x_end - x_start),
            "height": int(y_end - y_start),
        },
    })


def handle_resize(img, params, output_dir):
    """
    Resize image to (width, height) and return resized image base64.
    """
    new_w = int(params.get("width", 0))
    new_h = int(params.get("height", 0))

    if new_w <= 0 or new_h <= 0:
        raise ValueError(
            f"Invalid resize dimensions ({new_w}, {new_h}). Width and height must be positive integers."
        )

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    encoded = encode_image(resized)
    if output_dir:
        save_output_image(resized, "resized.png", output_dir)

    respond({
        "success": True,
        "image": encoded,
        "images": {"resized": encoded},
        "dimensions": {"width": new_w, "height": new_h},
    })


def handle_rotate(img, params, output_dir):
    """
    Rotate image by angle degrees using cv2.rotate() and return rotated image base64.
    Angle must be one of 0, 90, 180, 270.
    """
    angle = int(params.get("angle", 0))
    norm_angle = angle % 360

    if norm_angle == 0:
        rotated = img.copy()
    elif norm_angle == 90:
        rotated = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    elif norm_angle == 180:
        rotated = cv2.rotate(img, cv2.ROTATE_180)
    elif norm_angle == 270:
        rotated = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError(
            f"Unsupported rotation angle: {angle}. Allowed angles are 0, 90, 180, 270 degrees."
        )

    encoded = encode_image(rotated)
    if output_dir:
        save_output_image(rotated, "rotated.png", output_dir)

    rot_h, rot_w = rotated.shape[:2]
    respond({
        "success": True,
        "image": encoded,
        "images": {"rotated": encoded},
        "angle": angle,
        "dimensions": {"width": int(rot_w), "height": int(rot_h)},
    })


def main():
    args = parse_args()
    if not args:
        raise ValueError("No input arguments provided or invalid JSON format.")

    image_path = args.get("image_path")
    if not image_path:
        raise ValueError("Missing 'image_path' in input arguments.")
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file does not exist: {image_path}")

    output_dir = args.get("output_dir", "")
    action = args.get("action")
    if not action:
        raise ValueError("Missing 'action' in input arguments.")

    params = args.get("params") or {}

    # Load image using cv2.imread
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"cv2.imread failed to load image from: {image_path}")

    if action == "full_analysis":
        handle_full_analysis(img, output_dir)
    elif action == "pixel_inspect":
        handle_pixel_inspect(img, params)
    elif action == "modify_pixel":
        handle_modify_pixel(img, params, output_dir)
    elif action == "crop":
        handle_crop(img, params, output_dir)
    elif action == "resize":
        handle_resize(img, params, output_dir)
    elif action == "rotate":
        handle_rotate(img, params, output_dir)
    else:
        raise ValueError(
            f"Unknown action '{action}'. Valid actions: full_analysis, pixel_inspect, modify_pixel, crop, resize, rotate."
        )


if __name__ == "__main__":
    run_safe(main)
