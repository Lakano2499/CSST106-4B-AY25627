# CSST106: Perception & Computer Vision — Interactive Portfolio

A unified, full-stack interactive web application compiling all 7 laboratory experiments for **CSST106: Perception and Computer Vision** (College of Computer Studies).

Built with **Node.js (Express)** for the server/UI and **Python 3.12** for real-time computer vision processing, featuring a **minimalist, flat, pure-white academic theme** (`box-shadow: none`, `1px solid #E2E8F0` hairline borders) optimized for taking sharp screenshots for printed A4 laboratory reports.

---

## 🔬 Compilation of Laboratory Experiments

| Lab | Topic | Technologies & Core Algorithms |
| :---: | :--- | :--- |
| **01** | **Image Manipulation & Pixel Representation** | OpenCV `cv2.imread`, BGR vs RGB inspection, pixel modification, ROI cropping, dimensional scaling, and orthogonal rotation. |
| **02** | **Edge Detection Techniques** | Gaussian smoothing (3×3, 5×5, 7×7), Sobel gradient vector magnitude (X and Y), and Canny hysteresis edge detection (dual thresholding). |
| **03** | **Histogram Equalization & Contrast Adjustment** | Linear contrast stretching ($\alpha f(x) + \beta$), global equalization (`cv2.equalizeHist`), and Contrast Limited Adaptive Histogram Equalization (`CLAHE`) with distribution plots. |
| **04** | **SURF / SIFT / ORB Feature Detection** | Hessian determinant thresholding, Upright SURF (U-SURF), SIFT, ORB binary descriptors, rich keypoint visualization, and dual-image `BFMatcher`. |
| **05** | **Facial Recognition using PCA (Eigenfaces)** | Labeled Faces in the Wild (LFW) dataset, top 12 ghostly Eigenface portraits, 90% cumulative explained variance curve, and RBF-kernel SVM classifier with classification report. |
| **06** | **YOLOv11 Real-Time Object Detection** | Ultralytics YOLOv11 (`yolo11n.pt`, `yolo11s.pt`), bounding box coordinates, confidence percentages, multi-class recognition, and inference latency (ms). |
| **07** | **Image Segmentation Techniques (4-Way Comparison)** | 1) Global Otsu & Adaptive Thresholding, 2) Watershed touching object separation, 3) Edge-based closed contours, and 4) DeepLabV3 semantic segmentation mask overlay. |

---

## 🚀 Quick Start Guide

### Prerequisites
- **Node.js**: v18.0.0 or higher (v22+ recommended)
- **Python**: Python 3.10+ (Python 3.12 recommended)

### 1. Installation
Clone or navigate to the repository directory:
```bash
cd "c:\Users\aeron jhed\Downloads\web"
```

Install Node.js dependencies:
```bash
npm install
```

Install Python computer vision dependencies:
```bash
py -3.12 -m pip install -r requirements.txt
```

### 2. Start the Application
Launch the Express server:
```bash
npm start
```
The server will start at:
👉 **`http://localhost:3000`**

Open your web browser and navigate to `http://localhost:3000` to interact with all 7 laboratory modules.

---

## 📁 Repository Structure
```text
csst106-cv-web-ui/
├── package.json              # Node.js dependencies (express, multer, cors)
├── server.js                 # Express server & Python spawn execution engine
├── requirements.txt          # Python computer vision libraries
├── README.md                 # Project documentation
├── uploads/                  # Temporary uploaded images
├── outputs/                  # Processed images and plots
├── cache/                    # Cached machine learning datasets (e.g. LFW people)
├── docs/                     # Static frontend root (Served by GitHub Pages & Express)
│   ├── index.html            # Single-Page academic portfolio dashboard
│   ├── assets/               # Sample images and demonstration result assets
│   ├── css/
│   │   └── styles.css        # Minimalist white & dark theme styles (box-shadow: none, #E2E8F0 borders)
│   └── js/
│       ├── app.js            # Full-screen tabs, theme toggles, dropzones, toasts & static fallback engine
│       └── labs.js           # Lab parameter handlers and dynamic output canvas renderers
└── python/
    ├── utils.py              # Shared Base64 image encoders, white Matplotlib styling, JSON I/O
    ├── lab1_images.py        # Lab 1 worker: manipulation, cropping, resizing, rotation
    ├── lab2_edges.py         # Lab 2 worker: Sobel combined magnitude & Canny edges
    ├── lab3_histogram.py     # Lab 3 worker: linear contrast, equalizeHist, CLAHE & histograms
    ├── lab4_features.py      # Lab 4 worker: SURF/SIFT/ORB keypoints & BFMatcher
    ├── lab5_pca.py           # Lab 5 worker: LFW Eigenfaces, variance curve, SVM metrics
    ├── lab6_yolo.py          # Lab 6 worker: YOLOv11 multi-class object detection
    └── lab7_segmentation.py  # Lab 7 worker: Otsu, Watershed, Contours & DeepLabV3
```

---

## 🌐 GitHub Pages Deployment Guide
This repository is pre-configured to host directly via **GitHub Pages** using the `/docs` directory:
1. Navigate to your repository on GitHub: **`https://github.com/Lakano2499/CSST106-4B-AY25627`**
2. Click **Settings** (⚙️) > **Pages** (under the "Code and automation" section).
3. Under **Build and deployment** > **Source**, select **`Deploy from a branch`**.
4. Set **Branch** to **`main`** and select the **`/docs`** folder from the dropdown.
5. Click **Save**. Your portfolio will be live at:
   👉 **`https://lakano2499.github.io/CSST106-4B-AY25627/`**

### 🔄 Dual-Mode Architecture:
- **Local Mode (`http://localhost:3000`):** Real-time OpenCV, Scikit-Learn, and YOLOv11 execution with live Python worker processes.
- **GitHub Pages Demo Mode (`github.io`):** Automatically detected static fallback mode providing interactive precomputed sample outputs and download capabilities with zero backend requirements.

---

## 🎓 Academic Lab Report Reference
Each laboratory card in the web interface includes an expandable **Lab Theory & Report Answers** panel with pre-formatted, comprehensive answers addressing all questions required by the course laboratory manuals. You can click **Download Result** on any processed module to save high-resolution screenshots formatted for printed A4 laboratory report submission.
