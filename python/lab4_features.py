"""
CSST106 Computer Vision and Perception - Lab 4: Feature Detection & Matching
SURF (Speeded-Up Robust Features), SIFT, and ORB comparison script.

Accepts JSON arguments via sys.argv[1]:
  - image_path: path to base image
  - image_path2: optional path to second image for matching
  - output_dir: directory to save output artifacts
  - params: {
      detector: "SIFT" | "ORB" | "SURF",
      hessian_threshold: int (100 - 1000),
      upright: bool
    }
"""

import sys
import os
import time
import json
import cv2
import numpy as np

# Ensure utils from the same directory can be imported
sys.path.insert(0, os.path.dirname(__file__))
from utils import *


def try_create_surf(hessian_threshold=400, upright=False):
    """
    Safely attempt to create a SURF detector object.
    SURF is in cv2.xfeatures2d (contrib module) and may not be present
    or may be disabled due to patent/licensing restrictions in some OpenCV builds.
    """
    try:
        if hasattr(cv2, "xfeatures2d") and hasattr(cv2.xfeatures2d, "SURF_create"):
            try:
                surf = cv2.xfeatures2d.SURF_create(
                    hessianThreshold=float(hessian_threshold),
                    upright=bool(upright)
                )
            except TypeError:
                surf = cv2.xfeatures2d.SURF_create(float(hessian_threshold), 4, 3, False, bool(upright))
            return surf
    except (AttributeError, cv2.error, Exception):
        return None
    return None


def main():
    args = parse_args()

    image_path = args.get("image_path")
    if not image_path:
        raise ValueError("Missing required argument: 'image_path'")
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")

    image_path2 = args.get("image_path2")
    output_dir = args.get("output_dir", "")
    params = args.get("params", {}) or {}

    requested_detector = str(params.get("detector", "SIFT")).strip().upper()
    hessian_threshold = int(params.get("hessian_threshold", 400))
    upright = bool(params.get("upright", False))

    # 1. Load image(s) in grayscale
    img1 = read_image(image_path, cv2.IMREAD_GRAYSCALE)
    if img1 is None:
        raise ValueError(f"Failed to read image at: {image_path}")

    img2 = None
    if image_path2 and str(image_path2).strip() and os.path.isfile(image_path2):
        img2 = read_image(image_path2, cv2.IMREAD_GRAYSCALE)

    # Detect available detectors in OpenCV environment
    detectors_available = []
    if hasattr(cv2, "SIFT_create"):
        detectors_available.append("SIFT")
    if hasattr(cv2, "ORB_create"):
        detectors_available.append("ORB")

    surf_detector_test = try_create_surf(hessian_threshold, upright)
    surf_available = surf_detector_test is not None
    if surf_available:
        detectors_available.append("SURF")

    # 2. Create detector based on params.detector with SURF fallback
    warning_msg = None
    if not surf_available:
        warning_msg = (
            "SURF is not available in this OpenCV build (requires opencv-contrib-python with non-free enabled). "
            "Falling back to SIFT."
            if requested_detector == "SURF"
            else "SURF is not available in this OpenCV build (requires opencv-contrib-python with non-free enabled)."
        )

    selected_detector = requested_detector
    detector = None

    # Extract nfeatures parameter (default to 150 for crisp, visible keypoints)
    nfeatures = int(params.get("nfeatures", params.get("n_features", 150)))

    if requested_detector == "SURF":
        if surf_available:
            detector = surf_detector_test
            selected_detector = "SURF"
        else:
            detector = cv2.SIFT_create()
            selected_detector = "SIFT"
    elif requested_detector == "ORB":
        detector = cv2.ORB_create(nfeatures=nfeatures)
        selected_detector = "ORB"
    else:  # Default to SIFT
        detector = cv2.SIFT_create()
        selected_detector = "SIFT"

    # 3. Detect keypoints and compute descriptors, measure time
    t0 = time.perf_counter()
    kps1, descs1 = detector.detectAndCompute(img1, None)
    elapsed_time_ms = (time.perf_counter() - t0) * 1000.0

    if kps1 is None:
        kps1 = ()

    # 4. Draw rich keypoints: cv2.drawKeypoints(img, kps, None, (0,255,0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_kp = cv2.drawKeypoints(img1, kps1, None, (0, 255, 0), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    keypoints_b64 = encode_image(img_kp)

    if output_dir:
        save_output_image(img_kp, "keypoints.png", output_dir)

    # 5. Run ALL THREE detectors (SIFT, ORB, and SURF if available) for comparison
    comparison_targets = []
    if hasattr(cv2, "SIFT_create"):
        comparison_targets.append(("SIFT", lambda: cv2.SIFT_create()))
    if hasattr(cv2, "ORB_create"):
        comparison_targets.append(("ORB", lambda: cv2.ORB_create(nfeatures=nfeatures)))
    if surf_available:
        comparison_targets.append(("SURF", lambda: try_create_surf(hessian_threshold, upright)))

    comparison = []
    for det_name, det_factory in comparison_targets:
        try:
            if det_name == selected_detector:
                kps_c = kps1
                descs_c = descs1
                time_c = elapsed_time_ms
            else:
                det_instance = det_factory()
                t_bench_start = time.perf_counter()
                kps_c, descs_c = det_instance.detectAndCompute(img1, None)
                time_c = (time.perf_counter() - t_bench_start) * 1000.0
                if kps_c is None:
                    kps_c = ()

            kp_count = len(kps_c)
            dims = int(descs_c.shape[1]) if (descs_c is not None and hasattr(descs_c, "shape") and len(descs_c.shape) > 1) else 0
            dims_str = f"{dims} ({'binary' if det_name == 'ORB' else 'float'})" if dims > 0 else "-"
            rounded_time = round(time_c, 2)

            comparison.append({
                "name": det_name,
                "keypoints": kp_count,
                "keypoints_count": kp_count,
                "descriptor_dims": dims,
                "descriptor_dim": dims_str,
                "dimensions": dims_str,
                "time": rounded_time,
                "time_ms": rounded_time,
                "execution_time": f"{rounded_time} ms"
            })
        except Exception:
            pass

    # 6. If image_path2 is provided, perform feature matching
    matches_b64 = None
    img_matches = None
    good_matches = []
    top_matches = []

    if img2 is not None:
        kps2, descs2 = detector.detectAndCompute(img2, None)
        if kps2 is None:
            kps2 = ()

        # Select norm based on descriptor type (Hamming for binary ORB, L2 for SIFT/SURF)
        norm_type = cv2.NORM_HAMMING if selected_detector == "ORB" else cv2.NORM_L2
        bf = cv2.BFMatcher(norm_type, crossCheck=False)

        if descs1 is not None and descs2 is not None and len(descs1) >= 2 and len(descs2) >= 2:
            raw_matches = bf.knnMatch(descs1, descs2, k=2)
            # Lowe's ratio test: keep matches where distance < 0.75 * second best match
            for match_pair in raw_matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < 0.75 * n.distance:
                        good_matches.append(m)

            # If matching two distinct images/angles (e.g. Sample 1 vs Sample 2),
            # ratio test may filter most points. Fall back to best nearest-neighbor matches
            # so clean correspondence lines are always rendered for the report.
            if len(good_matches) < 5:
                all_first_matches = [pair[0] for pair in raw_matches if len(pair) > 0]
                all_first_matches.sort(key=lambda x: x.distance)
                top_matches = all_first_matches[:30]
            else:
                good_matches.sort(key=lambda x: x.distance)
                top_matches = good_matches[:50]

        draw_flags = getattr(
            cv2,
            "DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS",
            getattr(cv2, "DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS", 2)
        )

        img_matches = cv2.drawMatches(
            img1, kps1,
            img2, kps2,
            top_matches,
            None,
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            flags=draw_flags
        )
        matches_b64 = encode_image(img_matches)

        if output_dir:
            save_output_image(img_matches, "matches.png", output_dir)

    # 7. Create matplotlib figure showing keypoint visualization and comparison chart
    plt = setup_matplotlib()

    color_palette = {
        "SIFT": "#89b4fa",  # Catppuccin Blue
        "ORB": "#a6e3a1",   # Catppuccin Green
        "SURF": "#fab387",  # Catppuccin Peach
    }

    comp_names = [c["name"] for c in comparison]
    comp_kps = [c["keypoints"] for c in comparison]
    comp_times = [c["time_ms"] for c in comparison]
    bar_colors = [color_palette.get(n, "#cba6f7") for n in comp_names]

    if img_matches is not None:
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))

        # (0, 0) Keypoints
        axes[0, 0].imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        axes[0, 0].set_title(f"{selected_detector} Keypoints ({len(kps1):,} detected)", fontsize=11, fontweight="bold")
        axes[0, 0].axis("off")

        # (0, 1) Feature Matches
        axes[0, 1].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        axes[0, 1].set_title(
            f"{selected_detector} Matches (Top {len(top_matches)} of {len(good_matches)} good matches)",
            fontsize=11,
            fontweight="bold"
        )
        axes[0, 1].axis("off")

        # (1, 0) Keypoints Bar Chart
        bars1 = axes[1, 0].bar(comp_names, comp_kps, color=bar_colors, width=0.45, edgecolor="#45475a", linewidth=1.2)
        axes[1, 0].set_title("Detected Keypoint Count Comparison", fontsize=11, fontweight="bold")
        axes[1, 0].set_ylabel("Keypoint Count", fontsize=10)
        axes[1, 0].grid(True, linestyle="--", alpha=0.3, axis="y")
        max_kps = max(comp_kps) if comp_kps else 0
        axes[1, 0].set_ylim(0, max_kps * 1.18 if max_kps > 0 else 10)
        for bar in bars1:
            y = bar.get_height()
            axes[1, 0].text(
                bar.get_x() + bar.get_width() / 2.0,
                y + (max_kps * 0.02 if max_kps > 0 else 0.2),
                f"{int(y):,}",
                ha="center",
                va="bottom",
                color="#cdd6f4",
                fontsize=9,
                fontweight="bold"
            )

        # (1, 1) Execution Time Bar Chart
        bars2 = axes[1, 1].bar(comp_names, comp_times, color=bar_colors, width=0.45, edgecolor="#45475a", linewidth=1.2)
        axes[1, 1].set_title("Execution Time Comparison (ms)", fontsize=11, fontweight="bold")
        axes[1, 1].set_ylabel("Time (ms)", fontsize=10)
        axes[1, 1].grid(True, linestyle="--", alpha=0.3, axis="y")
        max_time = max(comp_times) if comp_times else 0
        axes[1, 1].set_ylim(0, max_time * 1.18 if max_time > 0 else 10)
        for bar in bars2:
            y = bar.get_height()
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2.0,
                y + (max_time * 0.02 if max_time > 0 else 0.2),
                f"{y:.1f} ms",
                ha="center",
                va="bottom",
                color="#cdd6f4",
                fontsize=9,
                fontweight="bold"
            )
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        # (0) Keypoints
        axes[0].imshow(cv2.cvtColor(img_kp, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"{selected_detector} Keypoints ({len(kps1):,} detected)", fontsize=11, fontweight="bold")
        axes[0].axis("off")

        # (1) Keypoints Bar Chart
        bars1 = axes[1].bar(comp_names, comp_kps, color=bar_colors, width=0.45, edgecolor="#45475a", linewidth=1.2)
        axes[1].set_title("Detected Keypoint Count Comparison", fontsize=11, fontweight="bold")
        axes[1].set_ylabel("Keypoint Count", fontsize=10)
        axes[1].grid(True, linestyle="--", alpha=0.3, axis="y")
        max_kps = max(comp_kps) if comp_kps else 0
        axes[1].set_ylim(0, max_kps * 1.18 if max_kps > 0 else 10)
        for bar in bars1:
            y = bar.get_height()
            axes[1].text(
                bar.get_x() + bar.get_width() / 2.0,
                y + (max_kps * 0.02 if max_kps > 0 else 0.2),
                f"{int(y):,}",
                ha="center",
                va="bottom",
                color="#cdd6f4",
                fontsize=9,
                fontweight="bold"
            )

        # (2) Execution Time Bar Chart
        bars2 = axes[2].bar(comp_names, comp_times, color=bar_colors, width=0.45, edgecolor="#45475a", linewidth=1.2)
        axes[2].set_title("Execution Time Comparison (ms)", fontsize=11, fontweight="bold")
        axes[2].set_ylabel("Time (ms)", fontsize=10)
        axes[2].grid(True, linestyle="--", alpha=0.3, axis="y")
        max_time = max(comp_times) if comp_times else 0
        axes[2].set_ylim(0, max_time * 1.18 if max_time > 0 else 10)
        for bar in bars2:
            y = bar.get_height()
            axes[2].text(
                bar.get_x() + bar.get_width() / 2.0,
                y + (max_time * 0.02 if max_time > 0 else 0.2),
                f"{y:.1f} ms",
                ha="center",
                va="bottom",
                color="#cdd6f4",
                fontsize=9,
                fontweight="bold"
            )

    fig.tight_layout()
    plot_b64 = encode_matplotlib_figure(fig)

    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        fig.savefig(os.path.join(output_dir, "comparison_plot.png"), bbox_inches="tight", dpi=120, facecolor="#1e1e2e")

    plt.close(fig)

    # Return structured JSON
    result = {
        "success": True,
        "images": {
            "keypoints": keypoints_b64,
            "matches": matches_b64
        },
        "plot": plot_b64,
        "detectors_available": detectors_available,
        "selected_detector": selected_detector,
        "comparison": comparison,
        "warning": warning_msg
    }

    respond(result)


if __name__ == "__main__":
    run_safe(main)
