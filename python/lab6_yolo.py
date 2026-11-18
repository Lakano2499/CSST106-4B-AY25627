"""
CSST106 Lab 6: YOLOv11 Object Detection
Performs real-time object detection using pre-trained Ultralytics YOLOv11 models.
Extracts bounding boxes, confidence scores, class labels, and generates comparison plots.
"""

import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
from utils import *

# Attempt to import ultralytics YOLO gracefully
try:
    from ultralytics import YOLO, time
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    try:
        from ultralytics import YOLO
        ULTRALYTICS_AVAILABLE = True
    except ImportError:
        YOLO = None
        ULTRALYTICS_AVAILABLE = False


def main():
    # Graceful check for ultralytics dependency
    if not ULTRALYTICS_AVAILABLE or YOLO is None:
        respond_error(
            "The 'ultralytics' package is not installed. "
            "Please install it using 'pip install ultralytics' to run YOLOv11 object detection."
        )
        return

    # Parse arguments passed from Node.js server
    args = parse_args()
    image_path = args.get("image_path")
    output_dir = args.get("output_dir", "outputs")
    params = args.get("params", {}) or {}

    # Validate image path
    if not image_path:
        respond_error("Missing required parameter: 'image_path'")
        return

    if not os.path.exists(image_path):
        respond_error(f"Image file not found: {image_path}")
        return

    # Read original image using utils.read_image
    orig_img = read_image(image_path)
    if orig_img is None:
        respond_error(f"Failed to read image at: {image_path}")
        return

    # Extract model parameters
    model_name = params.get("model_name", "yolo11n.pt")
    if not model_name or not isinstance(model_name, str):
        model_name = "yolo11n.pt"

    try:
        confidence = float(params.get("confidence", 0.25))
    except (TypeError, ValueError):
        confidence = 0.25

    # Clamp confidence to a reasonable range [0.01, 1.0]
    confidence = max(0.01, min(1.0, confidence))

    # 1. Load YOLO model (auto-downloads weights if needed)
    try:
        model = YOLO(model_name)
    except Exception as e:
        respond_error(f"Failed to load YOLO model '{model_name}': {str(e)}")
        return

    # 2. Run inference
    try:
        results = model(image_path, conf=confidence)
    except Exception as e:
        respond_error(f"Inference failed: {str(e)}")
        return

    if not results or len(results) == 0:
        respond_error("Inference produced no results.")
        return

    result = results[0]

    # 3. For each detection, extract: class name, confidence score, bounding box [x1, y1, x2, y2]
    detections = []
    boxes = result.boxes
    if boxes is not None and len(boxes) > 0:
        for box in boxes:
            # Class ID and class name
            cls_id = int(box.cls[0].item()) if hasattr(box.cls, "__len__") else int(box.cls.item())
            class_name = (
                result.names.get(cls_id, str(cls_id))
                if hasattr(result, "names") and result.names
                else str(cls_id)
            )

            # Confidence score
            conf_val = float(box.conf[0].item()) if hasattr(box.conf, "__len__") else float(box.conf.item())

            # Bounding box coordinates [x1, y1, x2, y2]
            xyxy_raw = box.xyxy[0].tolist() if hasattr(box.xyxy, "__len__") else box.xyxy.tolist()
            bbox = [round(float(coord), 2) for coord in xyxy_raw]

            detections.append({
                "class_name": class_name,
                "confidence": round(conf_val, 4),
                "bbox": bbox
            })

    # 4. Get the annotated image from results[0].plot()
    annotated_img = result.plot()

    # Save to outputs directory if specified
    if output_dir:
        try:
            filename = f"yolo11_{os.path.basename(image_path)}"
            save_output_image(annotated_img, filename, output_dir)
        except Exception:
            pass

    # 5. Create a summary
    total_objects = len(detections)

    objects_by_class = {}
    for det in detections:
        c = det["class_name"]
        objects_by_class[c] = objects_by_class.get(c, 0) + 1

    speed_dict = getattr(result, "speed", {}) or {}
    speed = {
        "preprocess_ms": round(float(speed_dict.get("preprocess", 0.0)), 2),
        "inference_ms": round(float(speed_dict.get("inference", 0.0)), 2),
        "postprocess_ms": round(float(speed_dict.get("postprocess", 0.0)), 2)
    }

    summary = {
        "total_objects": total_objects,
        "objects_by_class": objects_by_class,
        "speed": speed
    }

    # Extract model metadata
    task = getattr(model, "task", None)
    if not task and hasattr(model, "overrides") and isinstance(model.overrides, dict):
        task = model.overrides.get("task", "detect")
    if not task:
        task = "detect"

    names_dict = getattr(model, "names", None) or getattr(result, "names", None) or {}
    num_classes = len(names_dict) if names_dict else 80

    model_info = {
        "name": model_name,
        "task": str(task),
        "num_classes": num_classes
    }

    # 6. Create matplotlib figure with 2 subplots (Left: Original, Right: Annotated)
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    # Left: Original image (BGR -> RGB)
    orig_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    axes[0].imshow(orig_rgb)
    axes[0].set_title("Original Image", fontsize=14, fontweight="bold", pad=12)
    axes[0].axis("off")

    # Right: Annotated detection image (BGR -> RGB)
    annotated_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    axes[1].imshow(annotated_rgb)
    axes[1].set_title(
        f"YOLOv11 Detections ({total_objects} object{'s' if total_objects != 1 else ''})",
        fontsize=14,
        fontweight="bold",
        pad=12
    )
    axes[1].axis("off")

    fig.tight_layout()
    plot_b64 = encode_matplotlib_figure(fig)
    plt.close(fig)

    # Encode original and annotated images as Base64 Data URIs
    orig_b64 = encode_image(orig_img)
    annotated_b64 = encode_image(annotated_img)

    # Return final JSON payload
    respond({
        "success": True,
        "images": {
            "original": orig_b64,
            "annotated": annotated_b64
        },
        "plot": plot_b64,
        "detections": detections,
        "summary": summary,
        "model_info": model_info
    })


if __name__ == "__main__":
    run_safe(main)
