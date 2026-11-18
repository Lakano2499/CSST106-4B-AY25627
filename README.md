# CSST106: Perception & Computer Vision — Interactive Portfolio

A unified, full-stack interactive web application compiling all 7 laboratory experiments for **CSST106: Perception and Computer Vision** (College of Computer Studies).

Built with **Node.js (Express)** for the server/UI and **Python 3.12** for real-time computer vision processing, featuring a **minimalist, flat, pure-white academic theme** (`box-shadow: none`, `1px solid #E2E8F0` hairline borders) optimized for taking sharp screenshots for printed A4 laboratory reports.

---

## 🎨 UI/UX Architecture (Clean White Academic Theme)
- **Top Sticky Navigation Bar (`64px`):** Fixed header with logo, course code pill (`CSST106`), quick-jump dropdown (`01` to `07`), student badge (`Aeron Jhed | BSCS-4B`), and GitHub link.
- **Hero & Curriculum Metric Split Section (65% / 35%):**
  - **Left:** Course compilation header, high-level description, and technical stack badges.
  - **Right:** 2×2 metric grid displaying **07** Total Experiments, **07** Interactive Modules, **100%** UI Integration, and pipeline status.
- **Search & Category Filter Bar:** Real-time search by keyword/algorithm + category pills (`All`, `Image Processing 01–03`, `Feature & Dimensionality 04–05`, `Detection & Segmentation 06–07`), plus `Expand All` and `Collapse All` triggers.
- **Accordion Lab Cards (`01` to `07`):** Wide flat white cards with pastel numbered badges (`01`–`07`), status pills, and smooth expansion into a **2-Pane Lab Workspace**:
  - **Left Control Panel (32%):** File dropzone, interactive sliders with live numeric badges, and primary `Run Algorithm` buttons.
  - **Right Output Canvas (68%):** Side-by-side output grids, Matplotlib distribution histograms, quantitative metric cards, a `Download Result` button, and a built-in **Lab Theory & Report Answers** reference panel for hard-copy report submission.

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
├── public/
│   ├── index.html            # Single-Page white academic portfolio dashboard
│   ├── css/
│   │   └── styles.css        # Minimalist white theme styles (box-shadow: none, #E2E8F0 borders)
│   └── js/
│       ├── app.js            # Accordion, search, category filters, dropzones & toasts
│       └── labs.js           # API request handlers and dynamic result canvas renderers
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

## 🎓 Academic Lab Report Reference
Each laboratory card in the web interface includes an expandable **Lab Theory & Report Answers** panel with pre-formatted, comprehensive answers addressing all questions required by the course laboratory manuals. You can click **Download Result** on any processed module to save high-resolution screenshots formatted for printed A4 laboratory report submission.
