"""
CSST106 Computer Vision and Perception
Lab Exercise 5: Facial Recognition using Principal Component Analysis (Eigenfaces)

This script loads the LFW (Labeled Faces in the Wild) benchmark dataset,
performs PCA dimensionality reduction to extract orthogonal Eigenface basis vectors,
analyzes the cumulative explained variance trade-off, and trains a Support Vector
Machine (SVM) classifier for facial recognition in the reduced eigenspace.
"""

import os
import sys
import time
import numpy as np

# Ensure current script directory is in sys.path to import local utils
sys.path.insert(0, os.path.dirname(__file__))
from utils import (
    setup_matplotlib,
    encode_matplotlib_figure,
    encode_image,
    save_output_image,
    parse_args,
    respond,
    respond_error,
    run_safe
)

from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score


def create_eigenfaces_plot(plt, eigenfaces, num_components, h, w, output_dir=None):
    """
    Generate a 3x4 gallery grid of the top 12 orthogonal eigenfaces.
    """
    num_to_show = min(12, num_components)
    n_rows, n_cols = 3, 4
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.5, 7.5))
    fig.suptitle(
        "Top Orthogonal Eigenfaces (Ghostly Basis Vectors)",
        fontsize=13,
        fontweight="bold",
        color="#cdd6f4",
        y=0.98
    )

    for i in range(n_rows * n_cols):
        r = i // n_cols
        c = i % n_cols
        ax = axes[r, c]

        if i < num_to_show:
            ax.imshow(eigenfaces[i], cmap=plt.cm.gray)
            ax.set_title(f"Eigenface #{i + 1}", fontsize=9.5, color="#89b4fa", pad=4)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color("#45475a")

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, "lab5_eigenfaces.png"), bbox_inches="tight", dpi=120, facecolor="#1e1e2e")
        except Exception:
            pass

    encoded = encode_matplotlib_figure(fig)
    plt.close(fig)
    return encoded


def create_variance_plot(plt, cumsum_var, n_components, variance_90_components, total_variance_retained, output_dir=None):
    """
    Generate the Cumulative Explained Variance vs. Number of Principal Components plot.
    Draws a horizontal 90% threshold line and annotates the component cutoff point.
    """
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    components_range = np.arange(1, len(cumsum_var) + 1)

    # Plot cumulative explained variance curve
    ax.plot(
        components_range,
        cumsum_var,
        color="#89b4fa",
        linewidth=2.5,
        label="Cumulative Explained Variance"
    )

    # 90% threshold line
    ax.axhline(
        y=0.90,
        color="#f38ba8",
        linestyle="--",
        linewidth=1.6,
        label="90% Variance Threshold"
    )

    # Annotate 90% threshold component if reached
    if variance_90_components is not None and variance_90_components <= len(cumsum_var):
        val_at_90 = cumsum_var[variance_90_components - 1]
        ax.axvline(
            x=variance_90_components,
            color="#a6e3a1",
            linestyle=":",
            linewidth=1.5,
            label=f"90% Cutoff (k={variance_90_components})"
        )
        ax.plot(
            variance_90_components,
            val_at_90,
            marker="o",
            markersize=7,
            color="#f9e2af",
            zorder=5
        )

        # Annotation callout
        offset_x = max(3, int(len(cumsum_var) * 0.06))
        ax.annotate(
            f"k = {variance_90_components}\n({val_at_90:.1%} var)",
            xy=(variance_90_components, val_at_90),
            xytext=(variance_90_components + offset_x, val_at_90 - 0.08),
            arrowprops=dict(
                facecolor="#f9e2af",
                edgecolor="#f9e2af",
                arrowstyle="->",
                lw=1.5
            ),
            fontsize=9.5,
            color="#f9e2af",
            fontweight="bold"
        )
    else:
        ax.text(
            0.5,
            0.5,
            f"90% threshold requires > {n_components} components\n(Current retention: {total_variance_retained:.1%})",
            transform=ax.transAxes,
            ha="center",
            va="center",
            color="#f9e2af",
            fontsize=10
        )

    ax.set_title(
        "Cumulative Explained Variance vs. Number of Principal Components",
        fontsize=12,
        fontweight="bold",
        pad=10
    )
    ax.set_xlabel("Number of Principal Components (k)", fontsize=10.5)
    ax.set_ylabel("Cumulative Explained Variance Ratio", fontsize=10.5)
    ax.set_xlim(1, len(cumsum_var))
    ax.set_ylim(0, 1.05)
    ax.grid(True, linestyle="--", alpha=0.3, color="#45475a")
    ax.legend(loc="lower right", facecolor="#181825", edgecolor="#45475a", fontsize=9.5)

    plt.tight_layout()

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, "lab5_variance_curve.png"), bbox_inches="tight", dpi=120, facecolor="#1e1e2e")
        except Exception:
            pass

    encoded = encode_matplotlib_figure(fig)
    plt.close(fig)
    return encoded


def create_failures_plot(plt, X_test, y_test, y_pred, target_names, h, w, misclassified_indices, output_dir=None):
    """
    Generate a failure analysis gallery showing up to 8 misclassified faces
    with predicted vs actual ground-truth identity labels.
    """
    if len(misclassified_indices) == 0:
        return None

    num_failures = min(8, len(misclassified_indices))
    n_cols = min(4, num_failures)
    n_rows = int(np.ceil(num_failures / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(2.7 * n_cols, 3.2 * n_rows))
    fig.suptitle(
        "Failure Analysis: Misclassified Test Faces",
        fontsize=12.5,
        fontweight="bold",
        color="#f38ba8",
        y=0.98
    )

    if n_rows == 1 and n_cols == 1:
        axes_list = [axes]
    elif n_rows == 1 or n_cols == 1:
        axes_list = list(axes)
    else:
        axes_list = list(axes.flatten())

    for i, ax in enumerate(axes_list):
        if i < num_failures:
            idx = misclassified_indices[i]
            img = X_test[idx].reshape((h, w))
            true_label = target_names[y_test[idx]]
            pred_label = target_names[y_pred[idx]]

            ax.imshow(img, cmap=plt.cm.gray)
            ax.set_title(
                f"Pred: {pred_label}\nTrue: {true_label}",
                fontsize=8.5,
                color="#f38ba8",
                pad=4
            )
            for spine in ax.spines.values():
                spine.set_color("#f38ba8")
                spine.set_linewidth(1.5)
        else:
            ax.axis("off")

        ax.set_xticks([])
        ax.set_yticks([])

    plt.tight_layout(rect=[0, 0, 1, 0.94])

    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
            fig.savefig(os.path.join(output_dir, "lab5_failure_analysis.png"), bbox_inches="tight", dpi=120, facecolor="#1e1e2e")
        except Exception:
            pass

    encoded = encode_matplotlib_figure(fig)
    plt.close(fig)
    return encoded


def run_lab5():
    """Main execution function for CSST106 Lab 5."""
    # 0. Parse input JSON arguments
    args = parse_args()
    output_dir = args.get("output_dir") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "outputs")
    )
    cache_dir = args.get("cache_dir") or os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "cache")
    )
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(cache_dir, exist_ok=True)

    params = args.get("params", {})
    if not isinstance(params, dict):
        params = {}

    # Extract parameters with defaults
    n_components = int(params.get("n_components", 150))
    n_components = max(10, min(n_components, 150))

    whiten_val = params.get("whiten", True)
    if isinstance(whiten_val, str):
        whiten = whiten_val.lower() not in ("false", "0", "no")
    else:
        whiten = bool(whiten_val)

    action = str(params.get("action", "train")).lower()

    # 1. Load LFW benchmark dataset
    print("[Progress] Checking LFW dataset cache...", flush=True)
    lfw_home_dir = os.path.join(cache_dir, "lfw_home")
    if not os.path.exists(lfw_home_dir):
        print(
            "[Progress] Downloading LFW dataset (~200MB, this may take a few minutes on first run)...",
            flush=True
        )

    t0_data = time.time()
    lfw_people = fetch_lfw_people(
        min_faces_per_person=70,
        resize=0.4,
        data_home=cache_dir
    )
    print(f"[Progress] LFW dataset loaded in {time.time() - t0_data:.2f}s", flush=True)

    # 2. Extract dimensions and array data
    n_samples, h, w = lfw_people.images.shape
    X = lfw_people.data
    y = lfw_people.target
    target_names = list(lfw_people.target_names)
    n_classes = len(target_names)
    n_features = int(X.shape[1])

    # 3. Train/Test split: 80% train, 20% test (random_state=42)
    print("[Progress] Splitting data into 80% train and 20% test sets...", flush=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Ensure n_components does not exceed training sample count
    n_components = min(n_components, X_train.shape[0], X_train.shape[1])

    # 4 & 5. Apply PCA with randomized SVD solver and whitening
    print(f"[Progress] Fitting PCA with k={n_components} components (whiten={whiten})...", flush=True)
    t0_pca = time.time()
    pca = PCA(
        n_components=n_components,
        svd_solver="randomized",
        whiten=whiten,
        random_state=42
    )
    pca.fit(X_train)
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    pca_time_seconds = round(float(time.time() - t0_pca), 4)
    print(f"[Progress] PCA transformation completed in {pca_time_seconds:.4f}s", flush=True)

    # 6. Extract and reshape eigenfaces
    eigenfaces = pca.components_.reshape((n_components, h, w))

    # Setup matplotlib with dark theme
    plt = setup_matplotlib()

    # 7. Create eigenfaces gallery plot (3x4 grid)
    print("[Progress] Rendering eigenfaces gallery...", flush=True)
    eigenfaces_b64 = create_eigenfaces_plot(plt, eigenfaces, n_components, h, w, output_dir)

    # 8. Create cumulative explained variance plot
    print("[Progress] Calculating cumulative explained variance curve...", flush=True)
    cumsum_var = pca.explained_variance_ratio_.cumsum()
    total_variance_retained = round(float(cumsum_var[-1]), 4)

    idx_90 = np.where(cumsum_var >= 0.90)[0]
    variance_90_components = int(idx_90[0] + 1) if len(idx_90) > 0 else None

    variance_curve_b64 = create_variance_plot(
        plt,
        cumsum_var,
        n_components,
        variance_90_components,
        total_variance_retained,
        output_dir
    )

    # 9. Train SVM Classifier: SVC(kernel='rbf', class_weight='balanced', C=1000, gamma='scale')
    print("[Progress] Training SVM classifier in reduced eigenspace...", flush=True)
    t0_svm = time.time()
    clf = SVC(
        kernel="rbf",
        class_weight="balanced",
        C=1000,
        gamma="scale",
        random_state=42
    )
    clf.fit(X_train_pca, y_train)
    svm_time_seconds = round(float(time.time() - t0_svm), 4)
    print(f"[Progress] SVM fitting completed in {svm_time_seconds:.4f}s", flush=True)

    # 10. Predict on test set and generate classification report
    print("[Progress] Predicting test samples and computing classification metrics...", flush=True)
    y_pred = clf.predict(X_test_pca)
    accuracy = round(float(accuracy_score(y_test, y_pred)), 4)
    report_str = classification_report(
        y_test,
        y_pred,
        target_names=target_names,
        zero_division=0
    )

    # 11. Find misclassified samples & create failure analysis gallery
    print("[Progress] Extracting failure analysis samples...", flush=True)
    misclassified_indices = np.where(y_test != y_pred)[0]
    failures_b64 = create_failures_plot(
        plt,
        X_test,
        y_test,
        y_pred,
        target_names,
        h,
        w,
        misclassified_indices,
        output_dir
    )

    # Prepare return payload
    response_data = {
        "success": True,
        "dataset": {
            "n_samples": int(n_samples),
            "n_features": int(n_features),
            "n_classes": int(n_classes),
            "image_size": [int(h), int(w)],
            "target_names": target_names
        },
        "pca_time_seconds": pca_time_seconds,
        "svm_time_seconds": svm_time_seconds,
        "variance_90_components": variance_90_components,
        "total_variance_retained": total_variance_retained,
        "classification_report": report_str,
        "accuracy": accuracy,
        "plots": {
            "eigenfaces": eigenfaces_b64,
            "variance_curve": variance_curve_b64,
            "failures": failures_b64
        }
    }

    # Handle action == "test_random" (or attach test sample details)
    # Pick a random test sample, encode with encode_image, and include in response
    rand_idx = int(np.random.choice(len(X_test)))
    rand_face = X_test[rand_idx]
    pred_class = int(clf.predict(X_test_pca[rand_idx:rand_idx + 1])[0])
    true_class = int(y_test[rand_idx])
    is_correct = bool(pred_class == true_class)

    # Convert normalized image array to uint8 (0-255) for encode_image
    face_uint8 = np.clip(rand_face.reshape((h, w)) * 255.0, 0, 255).astype(np.uint8)
    sample_image_b64 = encode_image(face_uint8)

    response_data["test_sample"] = {
        "index": rand_idx,
        "true_label": target_names[true_class],
        "predicted_label": target_names[pred_class],
        "is_correct": is_correct,
        "image": sample_image_b64
    }

    if action == "test_random":
        response_data["action"] = "test_random"

    # Send JSON response via standard stdout utility
    respond(response_data)


if __name__ == "__main__":
    run_safe(run_lab5)
