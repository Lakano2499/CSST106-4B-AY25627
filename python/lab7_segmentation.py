"""
CSST106 Lab 7: Segmentation Techniques
Performs thresholding (Otsu & Adaptive), region-based watershed, edge-based contour detection,
and optional deep learning semantic segmentation (DeepLabV3).
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(__file__))
from utils import *

import cv2
import numpy as np

# Optional skimage import
try:
    import skimage
    from skimage import segmentation, color, filters, morphology
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Pascal VOC class definitions (21 classes)
VOC_CLASSES = [
    "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
    "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant",
    "sheep", "sofa", "train", "tvmonitor"
]

# Standard Pascal VOC Colormap (RGB)
VOC_COLORMAP = [
    (0, 0, 0),         # 0 = background
    (128, 0, 0),       # 1 = aeroplane
    (0, 128, 0),       # 2 = bicycle
    (128, 128, 0),     # 3 = bird
    (0, 0, 128),       # 4 = boat
    (128, 0, 128),     # 5 = bottle
    (0, 128, 128),     # 6 = bus
    (128, 128, 128),   # 7 = car
    (64, 0, 0),        # 8 = cat
    (192, 0, 0),       # 9 = chair
    (64, 128, 0),      # 10 = cow
    (192, 128, 0),     # 11 = diningtable
    (64, 0, 128),      # 12 = dog
    (192, 0, 128),     # 13 = horse
    (64, 128, 128),    # 14 = motorbike
    (192, 128, 128),   # 15 = person
    (0, 64, 0),        # 16 = pottedplant
    (128, 64, 0),      # 17 = sheep
    (0, 192, 0),       # 18 = sofa
    (128, 192, 0),     # 19 = train
    (0, 64, 128)       # 20 = tvmonitor
]


def process_segmentation():
    # 0. Parse input arguments
    args = parse_args()
    image_path = args.get("image_path")
    output_dir = args.get("output_dir", "outputs")
    params = args.get("params", {})

    if not image_path:
        raise ValueError("Missing 'image_path' in arguments.")

    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Read image
    img = read_image(image_path)
    if img is None:
        raise ValueError(f"Failed to read image from path: {image_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Extract and sanitize parameters
    threshold_method = params.get("threshold_method", "otsu").lower()

    # block_size: int (3-31, odd)
    block_size = int(params.get("block_size", 11))
    if block_size < 3:
        block_size = 3
    elif block_size > 31:
        block_size = 31
    if block_size % 2 == 0:
        block_size += 1

    # watershed_threshold: float (0.1-0.9)
    watershed_threshold = float(params.get("watershed_threshold", 0.5))
    watershed_threshold = max(0.1, min(0.9, watershed_threshold))

    # contour_min_area: int
    contour_min_area = int(params.get("contour_min_area", 50))
    contour_min_area = max(0, contour_min_area)

    # use_deep_learning: bool
    use_deep_learning = bool(params.get("use_deep_learning", False))

    # -------------------------------------------------------------
    # 1. THRESHOLDING
    # -------------------------------------------------------------
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu thresholding (always computed for comparison)
    otsu_val, thresh_otsu = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # Adaptive thresholding (mean or gaussian)
    if threshold_method == "adaptive_mean":
        adaptive_type = cv2.ADAPTIVE_THRESH_MEAN_C
        adaptive_title = "Adaptive Mean"
    else:
        # Default or adaptive_gaussian
        adaptive_type = cv2.ADAPTIVE_THRESH_GAUSSIAN_C
        adaptive_title = "Adaptive Gaussian"

    thresh_adaptive = cv2.adaptiveThreshold(
        blurred, 255, adaptive_type, cv2.THRESH_BINARY, block_size, 2
    )

    # -------------------------------------------------------------
    # 2. REGION-BASED (Watershed)
    # -------------------------------------------------------------
    # Convert to grayscale, threshold with Otsu
    _, thresh_w = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Ensure foreground objects are white (255) and background is black (0) for distance transform
    h, w = gray.shape[:2]
    corners = [thresh_w[0, 0], thresh_w[0, w - 1], thresh_w[h - 1, 0], thresh_w[h - 1, w - 1]]
    if sum(1 for c in corners if c == 255) >= 3:
        thresh_w = cv2.bitwise_not(thresh_w)

    # Morphological opening to remove noise
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh_w, cv2.MORPH_OPEN, kernel, iterations=2)

    # Sure background area via dilation
    sure_bg = cv2.dilate(opening, kernel, iterations=3)

    # Distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)

    # Threshold distance transform at watershed_threshold * max
    max_dist = dist_transform.max()
    if max_dist > 0:
        _, sure_fg = cv2.threshold(
            dist_transform, watershed_threshold * max_dist, 255, cv2.THRESH_BINARY
        )
        sure_fg = np.uint8(sure_fg)
    else:
        sure_fg = np.zeros_like(opening)

    # Unknown region between sure background and sure foreground
    unknown = cv2.subtract(sure_bg, sure_fg)

    # Marker labelling
    _, markers = cv2.connectedComponents(sure_fg)
    # Add one to all labels so sure background is 1, not 0
    markers = markers + 1
    # Mark unknown region as 0
    markers[unknown == 255] = 0

    # Apply cv2.watershed, mark boundaries in red on original image
    watershed_img = img.copy()
    markers = cv2.watershed(watershed_img, markers)
    # Boundaries are marked with -1 -> set to Red [0, 0, 255] in BGR
    watershed_img[markers == -1] = [0, 0, 255]

    # Count segmented regions (labels > 1)
    unique_markers = np.unique(markers)
    segmented_regions = [m for m in unique_markers if m > 1]
    num_watershed_regions = int(len(segmented_regions))

    # -------------------------------------------------------------
    # 3. EDGE-BASED
    # -------------------------------------------------------------
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)

    # Find contours
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by min area
    filtered_contours = [c for c in contours if cv2.contourArea(c) >= contour_min_area]

    # Draw contours on original image in green
    contours_img = img.copy()
    cv2.drawContours(contours_img, filtered_contours, -1, (0, 255, 0), 2)
    num_contours = int(len(filtered_contours))

    # -------------------------------------------------------------
    # 4. SEMANTIC SEGMENTATION (if use_deep_learning=true)
    # -------------------------------------------------------------
    semantic_overlay = None
    semantic_classes = None
    semantic_message = None

    if use_deep_learning:
        try:
            import torch
            from torchvision import transforms
            from PIL import Image

            # Try to load torchvision.models.segmentation.deeplabv3_resnet101
            try:
                from torchvision.models.segmentation import deeplabv3_resnet101, DeepLabV3_ResNet101_Weights
                weights = DeepLabV3_ResNet101_Weights.DEFAULT
                model = deeplabv3_resnet101(weights=weights)
            except Exception:
                from torchvision.models.segmentation import deeplabv3_resnet101
                model = deeplabv3_resnet101(pretrained=True)

            model.eval()

            # Preprocess image (resize to 520, normalize with ImageNet stats)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb_img)

            preprocess = transforms.Compose([
                transforms.Resize(520),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                ),
            ])

            input_tensor = preprocess(pil_img).unsqueeze(0)

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():
                output = model(input_tensor)["out"][0]
            output_predictions = output.argmax(0).byte().cpu().numpy()

            # Resize predicted mask back to original image size
            orig_h, orig_w = img.shape[:2]
            pred_mask = cv2.resize(
                output_predictions, (orig_w, orig_h), interpolation=cv2.INTER_NEAREST
            )

            # Create color-coded segmentation map (PASCAL VOC colormap)
            color_mask_bgr = np.zeros((orig_h, orig_w, 3), dtype=np.uint8)
            detected_classes = []

            for class_id in np.unique(pred_mask):
                class_id = int(class_id)
                if class_id == 0:
                    continue  # background
                if class_id < len(VOC_CLASSES):
                    detected_classes.append(VOC_CLASSES[class_id])
                rgb_color = VOC_COLORMAP[class_id % len(VOC_COLORMAP)]
                bgr_color = (int(rgb_color[2]), int(rgb_color[1]), int(rgb_color[0]))
                color_mask_bgr[pred_mask == class_id] = bgr_color

            # Overlay on original image with alpha blending
            mask_active = (pred_mask > 0)
            semantic_overlay = img.copy()
            if np.any(mask_active):
                blended = cv2.addWeighted(img, 0.65, color_mask_bgr, 0.35, 0)
                semantic_overlay[mask_active] = blended[mask_active]

            semantic_classes = sorted(list(set(detected_classes)))
            semantic_message = "Semantic segmentation inference successful."

        except ImportError:
            semantic_message = "PyTorch or torchvision is not available in the Python environment."
        except Exception as e:
            semantic_message = f"Semantic segmentation error: {str(e)}"
    else:
        semantic_message = "Semantic segmentation was not requested (use_deep_learning is false)."

    # -------------------------------------------------------------
    # 5. MATPLOTLIB COMPARISON FIGURE (figsize 16x12, 2x3 grid)
    # -------------------------------------------------------------
    plt = setup_matplotlib()
    fig, axes = plt.subplots(2, 3, figsize=(16, 12))

    # Row 1: Original | Otsu Threshold | Adaptive Threshold
    axes[0, 0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0, 0].set_title("Original Image", fontsize=14, fontweight="bold", pad=10)
    axes[0, 0].axis("off")

    axes[0, 1].imshow(thresh_otsu, cmap="gray")
    axes[0, 1].set_title(
        f"Otsu Threshold (T = {otsu_val:.1f})",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    axes[0, 1].axis("off")

    axes[0, 2].imshow(thresh_adaptive, cmap="gray")
    axes[0, 2].set_title(
        f"{adaptive_title} (Block Size = {block_size})",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    axes[0, 2].axis("off")

    # Row 2: Watershed | Edge Contours | Semantic Segmentation (or placeholder)
    axes[1, 0].imshow(cv2.cvtColor(watershed_img, cv2.COLOR_BGR2RGB))
    axes[1, 0].set_title(
        f"Watershed Segmentation ({num_watershed_regions} regions)",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    axes[1, 0].axis("off")

    axes[1, 1].imshow(cv2.cvtColor(contours_img, cv2.COLOR_BGR2RGB))
    axes[1, 1].set_title(
        f"Edge Contours ({num_contours} found, area >= {contour_min_area})",
        fontsize=14,
        fontweight="bold",
        pad=10
    )
    axes[1, 1].axis("off")

    if semantic_overlay is not None:
        axes[1, 2].imshow(cv2.cvtColor(semantic_overlay, cv2.COLOR_BGR2RGB))
        cls_summary = ", ".join(semantic_classes[:4]) if semantic_classes else "None detected"
        if semantic_classes and len(semantic_classes) > 4:
            cls_summary += f" (+{len(semantic_classes) - 4} more)"
        axes[1, 2].set_title(
            f"Semantic Segmentation (DeepLabV3)\nClasses: {cls_summary}",
            fontsize=13,
            fontweight="bold",
            pad=10
        )
        axes[1, 2].axis("off")
    else:
        axes[1, 2].set_facecolor("#181825")
        status_box = (
            "Semantic Segmentation\n\n"
            + (
                "(Deep Learning Disabled)\nSet 'use_deep_learning': true to enable"
                if not use_deep_learning
                else f"(Model Unavailable)\n{semantic_message}"
            )
        )
        axes[1, 2].text(
            0.5,
            0.5,
            status_box,
            ha="center",
            va="center",
            fontsize=12,
            color="#bac2de",
            multialignment="center",
            bbox=dict(
                boxstyle="round,pad=1.2",
                facecolor="#11111b",
                edgecolor="#45475a",
                alpha=0.9
            ),
        )
        axes[1, 2].set_title(
            "Semantic Segmentation (Placeholder)",
            fontsize=14,
            fontweight="bold",
            pad=10
        )
        axes[1, 2].axis("off")

    fig.suptitle(
        "CSST106 Lab 7: Segmentation Techniques Comparison",
        fontsize=18,
        fontweight="bold",
        color="#cdd6f4",
        y=0.98
    )
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])

    plot_b64 = encode_matplotlib_figure(fig)
    plt.close(fig)

    # Save output images to output directory
    save_output_image(img, "original.png", output_dir)
    save_output_image(thresh_otsu, "threshold_otsu.png", output_dir)
    save_output_image(thresh_adaptive, "threshold_adaptive.png", output_dir)
    save_output_image(watershed_img, "watershed.png", output_dir)
    save_output_image(contours_img, "contours.png", output_dir)
    if semantic_overlay is not None:
        save_output_image(semantic_overlay, "semantic.png", output_dir)

    # Prepare response payload
    images = {
        "original": encode_image(img),
        "threshold_otsu": encode_image(thresh_otsu),
        "threshold_adaptive": encode_image(thresh_adaptive),
        "watershed": encode_image(watershed_img),
        "contours": encode_image(contours_img),
        "semantic": encode_image(semantic_overlay) if semantic_overlay is not None else None,
    }

    stats = {
        "otsu_threshold_value": round(float(otsu_val), 2),
        "num_contours": num_contours,
        "num_watershed_regions": num_watershed_regions,
        "semantic_classes": semantic_classes,
    }

    response = {
        "success": True,
        "images": images,
        "plot": plot_b64,
        "stats": stats,
        "message": semantic_message,
    }

    respond(response)


if __name__ == "__main__":
    run_safe(process_segmentation)
