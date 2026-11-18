/**
 * CSST106 Computer Vision Interactive Portfolio
 * app.js: Clean Full-Screen Lab Navigation, DropZones, Toasts & APIs
 */

(function () {
  'use strict';

  // =========================================================================
  // 0. Automatic GitHub Pages Detection & Static Fallback Configuration
  // =========================================================================
  const IS_GITHUB_PAGES = window.location.hostname.includes('github.io') ||
                          window.location.protocol === 'file:' ||
                          window.location.search.includes('demo=true');

  function getStaticFallbackResponse(endpoint, data) {
    let action = 'all';
    if (data instanceof FormData) {
      action = data.get('action') || 'all';
    } else if (data && typeof data === 'object') {
      action = data.action || 'all';
    }

    if (endpoint.includes('/api/lab1')) {
      return {
        success: true,
        action: action,
        properties: {
          height: 384,
          width: 512,
          channels: 3,
          dtype: 'uint8',
          total_pixels: 196608,
          color_space: 'BGR'
        },
        bgr_inspection: [
          { coord: [50, 50], bgr: [124, 152, 189], rgb: [189, 152, 124] },
          { coord: [100, 100], bgr: [45, 88, 210], rgb: [210, 88, 45] },
          { coord: [150, 150], bgr: [82, 134, 95], rgb: [95, 134, 82] }
        ],
        outputs: {
          original: './assets/sample_results/original.png',
          bgr_channels: './assets/sample_results/original.png',
          grayscale: './assets/sample_results/grayscale.png',
          cropped: './assets/sample_results/cropped.png',
          resized: './assets/sample_results/resized.png',
          rotated: './assets/sample_results/rotated.png'
        }
      };
    }

    if (endpoint.includes('/api/lab2')) {
      return {
        success: true,
        kernel_size: 5,
        low_threshold: 50,
        high_threshold: 150,
        metrics: {
          input_shape: "384x512",
          gaussian_kernel: "5x5",
          canny_thresholds: "50 / 150",
          sobel_operator: "Sobel-Feldman",
          edges_detected_pct: "8.42%"
        },
        outputs: {
          original: './assets/sample_results/original.png',
          smoothed: './assets/sample_results/blurred.png',
          sobel_x: './assets/sample_results/sobel_x.png',
          sobel_y: './assets/sample_results/sobel_y.png',
          sobel_magnitude: './assets/sample_results/sobel_combined.png',
          canny: './assets/sample_results/canny.png'
        }
      };
    }

    if (endpoint.includes('/api/lab3')) {
      return {
        success: true,
        alpha: 1.5,
        beta: 20,
        clip_limit: 2.0,
        grid_size: [8, 8],
        metrics: {
          contrast_alpha: 1.5,
          brightness_beta: 20,
          clahe_clip_limit: 2.0,
          clahe_tile_grid: "8x8",
          dynamic_range: "0-255"
        },
        outputs: {
          original: './assets/sample_results/original.png',
          linear: './assets/sample_results/linear_contrast.png',
          global_equalization: './assets/sample_results/global_equalized.png',
          clahe: './assets/sample_results/clahe.png'
        },
        plot: './assets/sample_results/global_equalized.png'
      };
    }

    if (endpoint.includes('/api/lab4')) {
      return {
        success: true,
        method: 'ORB',
        keypoints_count: 500,
        descriptor_dim: 32,
        matches_count: 248,
        execution_time_ms: 18.4,
        metrics: {
          algorithm: "ORB (Oriented FAST and Rotated BRIEF)",
          keypoints_detected: 500,
          descriptor_dimension: 32,
          good_matches: 248,
          matching_distance: "Hamming"
        },
        outputs: {
          keypoints: './assets/sample_results/keypoints.png',
          matches: './assets/sample_results/matches.png'
        }
      };
    }

    if (endpoint.includes('/api/lab5')) {
      return {
        success: true,
        n_components: 150,
        whiten: true,
        variance_explained: 0.9084,
        accuracy: 0.842,
        eigenfaces_gallery: './assets/sample_results/lab5_eigenfaces.png',
        variance_curve: './assets/sample_results/lab5_variance_curve.png',
        failure_gallery: './assets/sample_results/lab5_failure_analysis.png',
        metrics: {
          dataset: "Labeled Faces in the Wild (LFW)",
          classes: "7 Target Individuals",
          principal_components: 150,
          cumulative_variance: "90.84%",
          svm_test_accuracy: "84.2%"
        },
        classification_report: `                   precision    recall  f1-score   support

     Ariel Sharon       0.83      0.77      0.80        13
     Colin Powell       0.82      0.88      0.85        60
  Donald Rumsfeld       0.85      0.78      0.81        27
    George W Bush       0.88      0.91      0.89       133
Gerhard Schroeder       0.81      0.79      0.80        28
      Hugo Chavez       0.85      0.73      0.79        15
       Tony Blair       0.83      0.81      0.82        36

         accuracy                           0.84       312
        macro avg       0.84      0.81      0.82       312
     weighted avg       0.85      0.84      0.84       312`
      };
    }

    if (endpoint.includes('/api/lab6')) {
      return {
        success: true,
        model: 'yolo11n',
        confidence_threshold: 0.25,
        inference_time_ms: 14.8,
        total_detections: 3,
        annotated_image: './assets/sample_results/yolo11_1794999357701-images.jpg',
        detections: [
          { class_name: 'person', confidence: 0.89, bbox: [120, 45, 290, 360] },
          { class_name: 'laptop', confidence: 0.84, bbox: [210, 200, 340, 310] },
          { class_name: 'cup', confidence: 0.72, bbox: [350, 240, 390, 300] }
        ],
        metrics: {
          model_architecture: "YOLOv11 Nano (PyTorch)",
          total_detected_objects: 3,
          classes_detected: "person, laptop, cup",
          latency_preprocess: "1.2 ms",
          latency_inference: "14.8 ms",
          latency_postprocess: "1.5 ms"
        }
      };
    }

    if (endpoint.includes('/api/lab7')) {
      return {
        success: true,
        metrics: {
          input_dimensions: "384x512",
          otsu_threshold_value: 128,
          watershed_markers_count: 14,
          contours_detected_count: 22,
          semantic_model: "DeepLabV3-ResNet50"
        },
        outputs: {
          original: './assets/sample_results/original.png',
          threshold_otsu: './assets/sample_results/threshold_otsu.png',
          threshold_adaptive: './assets/sample_results/threshold_adaptive.png',
          watershed: './assets/sample_results/watershed.png',
          contours: './assets/sample_results/contours.png',
          semantic: './assets/sample_results/semantic.png'
        },
        comparison_plot: './assets/sample_results/comparison_plot.png'
      };
    }

    return { success: true };
  }

  // =========================================================================
  // 1. Toast Notification System
  // =========================================================================
  function showToast(message, type = 'info', duration = 4000) {
    const container = document.getElementById('toast-container');
    if (!container) return;

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;

    let iconClass = 'fa-info-circle';
    if (type === 'success') iconClass = 'fa-circle-check';
    if (type === 'error') iconClass = 'fa-circle-exclamation';
    if (type === 'warning') iconClass = 'fa-triangle-exclamation';

    toast.innerHTML = `
      <i class="fa-solid ${iconClass}" style="margin-top: 2px;"></i>
      <div style="flex: 1;">${message}</div>
    `;

    container.appendChild(toast);

    setTimeout(() => {
      toast.style.opacity = '0';
      toast.style.transform = 'translateY(-6px)';
      toast.style.transition = 'all 0.2s ease';
      setTimeout(() => toast.remove(), 220);
    }, duration);
  }

  // =========================================================================
  // 2. Loading Overlay & Cancel Task Management
  // =========================================================================
  let activeAbortController = null;

  function showLoading(text = 'Processing Computer Vision Algorithm...') {
    const overlay = document.getElementById('loading-overlay');
    const label = document.getElementById('loading-text');
    if (label) label.textContent = text;
    if (overlay) overlay.classList.add('is-active');
  }

  function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) overlay.classList.remove('is-active');
  }

  function promptCancelAnalysis() {
    const modal = document.getElementById('cancel-confirm-modal');
    if (modal) modal.style.display = 'flex';
  }

  function dismissCancelModal() {
    const modal = document.getElementById('cancel-confirm-modal');
    if (modal) modal.style.display = 'none';
  }

  function confirmCancelAnalysis() {
    dismissCancelModal();
    if (activeAbortController) {
      activeAbortController.abort();
      activeAbortController = null;
    }
    hideLoading();
    showToast('Python analysis was cancelled by user.', 'warning');
  }

  // =========================================================================
  // 3. DropZone Component (Reusable File Uploader)
  // =========================================================================
  class DropZone {
    constructor(elementOrSelector, options = {}) {
      this.el = typeof elementOrSelector === 'string'
        ? document.querySelector(elementOrSelector)
        : elementOrSelector;
      this.options = Object.assign({
        accept: 'image/*',
        maxSizeMB: 50,
        onFile: null
      }, options);

      this.currentFile = null;
      if (this.el) {
        this.init();
      }
    }

    init() {
      let input = this.el.querySelector('input[type="file"]');
      if (!input) {
        input = document.createElement('input');
        input.type = 'file';
        input.accept = this.options.accept;
        input.style.display = 'none';
        this.el.appendChild(input);
      }
      this.fileInput = input;
      this.previewEl = this.el.querySelector('.dropzone-preview');

      this.el.addEventListener('click', (e) => {
        if (e.target !== this.fileInput) {
          this.fileInput.click();
        }
      });

      ['dragenter', 'dragover'].forEach(evt => {
        this.el.addEventListener(evt, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.el.classList.add('dragover');
        });
      });

      ['dragleave', 'drop'].forEach(evt => {
        this.el.addEventListener(evt, (e) => {
          e.preventDefault();
          e.stopPropagation();
          this.el.classList.remove('dragover');
        });
      });

      this.el.addEventListener('drop', (e) => {
        const files = e.dataTransfer?.files;
        if (files && files.length > 0) {
          this.handleFile(files[0]);
        }
      });

      this.fileInput.addEventListener('change', (e) => {
        const files = e.target.files;
        if (files && files.length > 0) {
          this.handleFile(files[0]);
        }
      });
    }

    handleFile(file) {
      if (!file) return;

      const maxBytes = this.options.maxSizeMB * 1024 * 1024;
      if (file.size > maxBytes) {
        showToast(`File is too large (${(file.size / 1024 / 1024).toFixed(1)}MB). Max limit is ${this.options.maxSizeMB}MB.`, 'error');
        return;
      }

      this.currentFile = file;
      this.el.classList.add('has-file');

      if (file.type.startsWith('image/')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          if (this.previewEl) {
            this.previewEl.src = e.target.result;
            this.previewEl.style.display = 'block';
          }
        };
        reader.readAsDataURL(file);
      }

      const hintEl = this.el.querySelector('.dropzone-hint');
      if (hintEl) {
        hintEl.textContent = `Selected: ${file.name} (${(file.size / 1024).toFixed(1)} KB)`;
      }

      if (typeof this.options.onFile === 'function') {
        this.options.onFile(file);
      }
    }

    getFile() {
      if (this.currentFile) return this.currentFile;
      if (this.fileInput && this.fileInput.files && this.fileInput.files.length > 0) {
        return this.fileInput.files[0];
      }
      return null;
    }

    hasFile() {
      return Boolean(this.getFile());
    }

    reset() {
      this.currentFile = null;
      if (this.fileInput) this.fileInput.value = '';
      this.el.classList.remove('has-file');
      if (this.previewEl) {
        this.previewEl.src = '';
        this.previewEl.style.display = 'none';
      }
    }
  }

  // =========================================================================
  // 4. API Communication Helper (with AbortController & Static Fallback)
  // =========================================================================
  async function apiCall(endpoint, data) {
    if (IS_GITHUB_PAGES) {
      // Simulate realistic processing time and return static precomputed response
      await new Promise(r => setTimeout(r, 450));
      return getStaticFallbackResponse(endpoint, data);
    }

    activeAbortController = new AbortController();

    const isFormData = data instanceof FormData;
    const fetchOpts = {
      method: 'POST',
      body: isFormData ? data : JSON.stringify(data),
      signal: activeAbortController.signal
    };

    if (!isFormData) {
      fetchOpts.headers = { 'Content-Type': 'application/json' };
    }

    try {
      const resp = await fetch(endpoint, fetchOpts);
      if (!resp.ok) {
        let msg = `HTTP ${resp.status}: ${resp.statusText}`;
        try {
          const errJson = await resp.json();
          if (errJson && errJson.error) msg = errJson.error;
        } catch (_) {}
        throw new Error(msg);
      }
      return await resp.json();
    } catch (err) {
      if (err.name === 'AbortError' || (err.message && err.message.includes('aborted'))) {
        throw new Error('Analysis was aborted by user.');
      }
      throw err;
    } finally {
      activeAbortController = null;
    }
  }

  // =========================================================================
  // 5. Clean Lab Activity Navigation (Full-Screen Switching)
  // =========================================================================
  function switchLab(labId) {
    if (!labId) return;

    // 1. Update Navigation Tabs
    const tabButtons = document.querySelectorAll('.lab-tab-btn');
    tabButtons.forEach(btn => {
      if (btn.getAttribute('data-lab') === labId) {
        btn.classList.add('active');
      } else {
        btn.classList.remove('active');
      }
    });

    // 2. Switch Active Lab View
    const labViews = document.querySelectorAll('.lab-view');
    labViews.forEach(view => {
      if (view.id === labId || view.getAttribute('data-lab') === labId) {
        view.classList.add('active');
      } else {
        view.classList.remove('active');
      }
    });

    // 3. Remember selection & update URL hash smoothly
    try {
      localStorage.setItem('active_lab', labId);
      history.replaceState(null, '', `#${labId}`);
    } catch (_) {}

    // Scroll smoothly to top of view
    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // =========================================================================
  // 6. Download Result Helper (Optimized for A4 Reports)
  // =========================================================================
  function downloadImage(dataUriOrUrl, filename = 'lab_result.png') {
    const a = document.createElement('a');
    a.href = dataUriOrUrl;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
  }

  function downloadLabResults(labId) {
    const resultsContainer = document.getElementById(`${labId}-results`);
    if (!resultsContainer) {
      showToast('No results container found.', 'warning');
      return;
    }

    const img = resultsContainer.querySelector('.result-plot-img') || 
                resultsContainer.querySelector('img.result-img') ||
                resultsContainer.querySelector('img');

    if (img && img.src && (img.src.startsWith('data:') || img.src.startsWith('http') || img.src.startsWith('/'))) {
      const filename = `CSST106_${labId.toUpperCase()}_Result.png`;
      downloadImage(img.src, filename);
      showToast(`Saved ${filename} for lab documentation!`, 'success');
    } else {
      showToast('Please execute the algorithm first to generate outputs before downloading.', 'warning');
    }
  }

  // =========================================================================
  // 7. General Utility Functions
  // =========================================================================
  function formatNumber(num) {
    if (typeof num !== 'number') return num;
    return num.toLocaleString();
  }

  function createImageElement(base64Src, labelText) {
    const card = document.createElement('div');
    card.className = 'result-card';

    const header = document.createElement('div');
    header.className = 'result-card-header';
    header.innerHTML = `
      <span class="result-card-label">${labelText}</span>
      <span class="result-card-badge">Output</span>
    `;

    const img = document.createElement('img');
    img.className = 'result-img';
    img.src = base64Src;
    img.alt = labelText;

    card.appendChild(header);
    card.appendChild(img);
    return card;
  }

  // =========================================================================
  // 8. Dark / Light Theme System (Light Mode is Default)
  // =========================================================================
  function initTheme() {
    const saved = localStorage.getItem('vision_labs_theme');
    // Light mode is strictly default unless user explicitly chose dark
    const theme = (saved === 'dark') ? 'dark' : 'light';
    applyTheme(theme, false);
  }

  function applyTheme(theme, notify = true) {
    document.documentElement.setAttribute('data-theme', theme);
    const metaTheme = document.getElementById('meta-theme-color');
    if (metaTheme) {
      metaTheme.setAttribute('content', theme === 'dark' ? '#111827' : '#FFFFFF');
    }
    const icon = document.getElementById('theme-toggle-icon');
    if (icon) {
      icon.className = theme === 'dark' ? 'fa-solid fa-sun' : 'fa-solid fa-moon';
    }
    localStorage.setItem('vision_labs_theme', theme);
    if (notify) {
      showToast(`Switched to ${theme === 'dark' ? 'Dark' : 'Light'} Mode`, 'info');
    }
  }

  function toggleTheme() {
    const current = document.documentElement.getAttribute('data-theme') || 'light';
    const next = current === 'dark' ? 'light' : 'dark';
    applyTheme(next, true);
  }

  // =========================================================================
  // 9. Comprehensive Q&A Database & Multi-Format Downloader
  // =========================================================================
  const QA_DATABASE = [
    {
      labNum: "01",
      labTitle: "Lab 01: Image Manipulation & Pixel Representation",
      items: [
        {
          q: "1. How are digital images digitally represented and manipulated in Computer Vision?",
          a: "In computer vision, digital images are modeled as multi-dimensional numerical matrices (tensors) of pixel values. Grayscale images are 2D arrays of shape (Height, Width) with values from 0 (black) to 255 (white). Color images are 3D arrays of shape (Height, Width, Channels). Pixel manipulation involves mathematical matrix slicing, spatial transformations, and matrix operations."
        },
        {
          q: "2. Why does OpenCV use BGR color channel ordering instead of RGB?",
          a: "Historically, when OpenCV was first developed in 1999 by Intel, camera sensor manufacturers and video software systems stored pixel bytes in Blue-Green-Red (BGR) order in memory. While modern web browsers and display monitors use Red-Green-Blue (RGB), OpenCV retains BGR for backward compatibility."
        },
        {
          q: "3. What are real-life applications of basic image operations?",
          a: "ROI cropping is used in automated license plate recognition (ALPR) to isolate plates; resizing is vital for neural network preprocessing where inputs must match fixed dimensions (e.g., 640×640 in YOLO); and rotation is foundational in dataset augmentation for computer vision training."
        }
      ]
    },
    {
      labNum: "02",
      labTitle: "Lab 02: Edge Detection Techniques (Sobel & Canny)",
      items: [
        {
          q: "Explain the differences between the edge maps produced by Sobel and Canny:",
          a: "1. Sobel Edge Detection: A 1st-order derivative gradient operator that computes horizontal (Sobel X) and vertical (Sobel Y) intensity differences. The resulting magnitude map displays continuous, gradient-shaded edges with varying thickness. Because it lacks thresholding and non-maximum suppression, edges can appear blurry and sensitive to background noise.\n\n2. Canny Edge Detection: A comprehensive 4-stage edge detection pipeline: (1) Gaussian smoothing, (2) Sobel gradient computation, (3) Non-Maximum Suppression (thinning candidate edges to 1-pixel width), and (4) Hysteresis Thresholding using dual thresholds to link strong and weak edge pixels. Consequently, Canny yields clean, 1-pixel thin, closed, and noise-free binary edge contours."
        }
      ]
    },
    {
      labNum: "03",
      labTitle: "Lab 03: Histogram Equalization & Contrast Enhancement",
      items: [
        {
          q: "1. What is histogram equalization and why do we need it?",
          a: "Histogram equalization is a non-linear contrast enhancement technique that redistributes pixel intensities evenly across the dynamic range (0–255). It is essential in computer vision because underexposed or overexposed images bunch pixel values into narrow intensity clusters, obscuring edges and features required for downstream feature extraction."
        },
        {
          q: "2. How does CLAHE differ from Global Histogram Equalization?",
          a: "Global histogram equalization computes a single Cumulative Distribution Function (CDF) for the entire image, which often over-amplifies background noise in homogeneous regions. In contrast, CLAHE divides the image into contextual local tiles (e.g., 8×8), applies equalization locally, and strictly caps histogram bins at a 'clipLimit' before using bilinear interpolation to blend tile borders seamlessly."
        }
      ]
    },
    {
      labNum: "04",
      labTitle: "Lab 04: Feature Detection & Matching (SURF, SIFT, ORB)",
      items: [
        {
          q: "1. What effect does changing the Hessian threshold have on detected keypoints?",
          a: "The Hessian threshold acts as a determinant blob-response filter. Increasing the threshold retains only the most prominent, high-contrast corner and blob points (yielding fewer keypoints with higher repeatability). Decreasing the threshold admits weaker responses, increasing the total keypoint count but introducing noisy, less stable points."
        },
        {
          q: "2. How does SURF compare with SIFT and ORB?",
          a: "SIFT uses Difference-of-Gaussians (DoG) and produces 128-dimensional floating-point descriptors; it is highly accurate and scale/rotation invariant but computationally intensive. SURF uses integral images and Haar wavelets with 64-dimensional descriptors, running significantly faster than SIFT. ORB is an unpatented, open-source binary descriptor (256-bit Hamming distance) that executes orders of magnitude faster than both SIFT and SURF."
        }
      ]
    },
    {
      labNum: "05",
      labTitle: "Lab 05: Facial Recognition using PCA (Eigenfaces & SVM)",
      items: [
        {
          q: "1. What does mathematical whitening do, and why is it beneficial before feeding PCA data into an SVM?",
          a: "When whiten=True is specified in PCA, each projected principal component coordinate is divided by its singular value. Mathematically, this projects the transformed features into an isotropic uncorrelated Gaussian distribution with unit variance. For an RBF-kernel SVM, whitening ensures all principal components exert equal geometric weight in distance computations, preventing dominant components from overpowering subtle diagnostic features."
        },
        {
          q: "2. Visual Interpretation of Eigenfaces:",
          a: "The initial Eigenfaces capture coarse, global illumination and low-frequency structures (e.g., forehead width, overall lighting angle, jawline boundaries). In contrast, higher-order Eigenfaces capture high-frequency localized features (e.g., mouth shape, nose bridge curvature, eye corner details, and hairline transitions)."
        }
      ]
    },
    {
      labNum: "06",
      labTitle: "Lab 06: Object Detection using YOLOv11",
      items: [
        {
          q: "What is object detection in computer vision, and how does it differ from image classification?",
          a: "Image Classification takes an entire image as input and outputs a single categorical label indicating what the dominant subject is (e.g., 'dog'), without providing any spatial location.\n\nObject Detection solves both classification and spatial localization concurrently. It predicts both 'what' objects are present and 'where' they are located by generating bounding box coordinates [x1, y1, x2, y2] for multiple co-occurring objects across diverse classes in a single forward pass."
        }
      ]
    },
    {
      labNum: "07",
      labTitle: "Lab 07: Image Segmentation (Thresholding, Region, Edge, Semantic)",
      items: [
        {
          q: "Explain the differences between thresholding, region-based, edge-based, and semantic segmentation with advantages and limitations:",
          a: "1. Thresholding: Splits pixels based on intensity cutoff (Global Otsu / Local Adaptive).\n- Advantage: Fastest, minimal computational overhead.\n- Limitation: Fails under uneven illumination or gradients.\n\n2. Region-Based (Watershed): Treats gradients as topographic relief, flooding from marker minima.\n- Advantage: Separates touching, overlapping objects cleanly.\n- Limitation: Prone to over-segmentation without exact markers.\n\n3. Edge-Based: Identifies boundaries via gradient discontinuities (Canny) and groups contours.\n- Advantage: Preserves exact boundary shapes and perimeters.\n- Limitation: Sensitive to noise; broken edge gaps prevent closed regions.\n\n4. Semantic (DeepLabV3): Assigns class labels to every pixel using deep convolutional feature maps.\n- Advantage: Understands complex real-world multi-class scenes.\n- Limitation: Computationally heavy, requires pre-trained weights."
        }
      ]
    }
  ];

  function generateQAText() {
    let out = "CSST106: PERCEPTION & COMPUTER VISION — LABORATORY Q&A COMPILATION\n";
    out += "Student: Aeron Jhed | BSCS-4B\n";
    out += "Course: CSST106 | College of Computer Studies\n";
    out += "=".repeat(76) + "\n\n";

    QA_DATABASE.forEach(lab => {
      out += `[${lab.labTitle.toUpperCase()}]\n`;
      out += "-".repeat(60) + "\n";
      lab.items.forEach(it => {
        out += `Q: ${it.q}\n\n`;
        out += `A: ${it.a}\n\n`;
      });
      out += "\n";
    });
    return out;
  }

  function generateQAMarkdown() {
    let md = "# CSST106: Perception & Computer Vision — Laboratory Q&A Compilation\n\n";
    md += "**Student:** Aeron Jhed | BSCS-4B  \n";
    md += "**Course:** CSST106 — Perception & Computer Vision  \n";
    md += "**Institution:** College of Computer Studies  \n\n";
    md += "---\n\n";

    QA_DATABASE.forEach(lab => {
      md += `## ${lab.labTitle}\n\n`;
      lab.items.forEach(it => {
        md += `### ${it.q}\n\n`;
        md += `${it.a}\n\n`;
      });
      md += "---\n\n";
    });

    md += "### Segmentation Comparison Reference (Lab 07)\n\n";
    md += "| Technique | Core Principle | Key Advantage | Key Limitation |\n";
    md += "|---|---|---|---|\n";
    md += "| **Thresholding** | Splits pixels based on intensity cutoff (Global Otsu / Local Adaptive) | Fastest, minimal computational overhead | Fails under uneven illumination or gradients |\n";
    md += "| **Region-Based (Watershed)** | Treats gradients as topographic relief, flooding from marker minima | Separates touching, overlapping objects cleanly | Prone to over-segmentation without exact markers |\n";
    md += "| **Edge-Based** | Identifies boundaries via gradient discontinuities (Canny) and groups contours | Preserves exact boundary shapes and perimeters | Sensitive to noise; broken edge gaps prevent closed regions |\n";
    md += "| **Semantic (DeepLabV3)** | Assigns class labels to every pixel using deep convolutional feature maps | Understands complex real-world multi-class scenes | Computationally heavy, requires pre-trained weights |\n";

    return md;
  }

  function generateQADoc() {
    return `<html xmlns:o="urn:schemas-microsoft-com:office:office" xmlns:w="urn:schemas-microsoft-com:office:word" xmlns="http://www.w3.org/TR/REC-html40">
<head>
  <meta charset="utf-8">
  <title>CSST106 Laboratory Theory &amp; Report Q&amp;A Compilation - Aeron Jhed</title>
  <!--[if gte mso 9]>
  <xml>
    <w:WordDocument>
      <w:View>Print</w:View>
      <w:Zoom>100</w:Zoom>
      <w:DoNotOptimizeForBrowser/>
    </w:WordDocument>
  </xml>
  <![endif]-->
  <style>
    @page Section1 {
      size: 8.5in 11.0in;
      margin: 1.0in 1.0in 1.0in 1.0in;
      mso-header-margin: 0.5in;
      mso-footer-margin: 0.5in;
      mso-paper-source: 0;
    }
    div.Section1 { page: Section1; }
    * {
      color: #000000 !important;
      box-sizing: border-box;
    }
    body {
      font-family: 'Calibri', 'Times New Roman', 'Arial', sans-serif;
      font-size: 11pt;
      line-height: 1.5;
      color: #000000 !important;
      background-color: #FFFFFF !important;
      margin: 0;
      padding: 0;
    }
    h1 {
      font-size: 18pt;
      font-weight: bold;
      color: #000000 !important;
      margin-top: 0;
      margin-bottom: 8pt;
      border-bottom: 2pt solid #000000;
      padding-bottom: 4pt;
    }
    .doc-meta {
      font-size: 10.5pt;
      color: #000000 !important;
      margin-bottom: 20pt;
      padding-bottom: 6pt;
      border-bottom: 1pt solid #000000;
    }
    h2 {
      font-size: 13.5pt;
      font-weight: bold;
      color: #000000 !important;
      margin-top: 18pt;
      margin-bottom: 6pt;
      border-bottom: 1pt solid #000000;
      padding-bottom: 2pt;
    }
    h3 {
      font-size: 11pt;
      font-weight: bold;
      color: #000000 !important;
      margin-top: 10pt;
      margin-bottom: 4pt;
    }
    p {
      font-size: 11pt;
      color: #000000 !important;
      margin-top: 0;
      margin-bottom: 6pt;
      text-align: justify;
    }
    strong, b {
      font-weight: bold;
      color: #000000 !important;
    }
    code {
      font-family: 'Consolas', 'Courier New', monospace;
      font-size: 10pt;
      color: #000000 !important;
      background-color: #F1F1F1;
      padding: 1pt 3pt;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      margin-top: 10pt;
      margin-bottom: 12pt;
      border: 1pt solid #000000;
    }
    th, td {
      border: 1pt solid #000000;
      padding: 6pt 8pt;
      font-size: 10pt;
      color: #000000 !important;
      vertical-align: top;
    }
    th {
      font-weight: bold;
      background-color: #F2F2F2;
      color: #000000 !important;
      text-align: left;
    }
  </style>
</head>
<body>
  <div class="Section1">
    <h1>CSST106: Perception &amp; Computer Vision — Laboratory Q&amp;A Compilation</h1>
    <div class="doc-meta">
      <strong>Student Name:</strong> Aeron Jhed &nbsp;|&nbsp; <strong>Section:</strong> BSCS-4B &nbsp;|&nbsp; <strong>Course Code:</strong> CSST106 &nbsp;|&nbsp; <strong>College:</strong> College of Computer Studies
    </div>
    ${QA_DATABASE.map(lab => `
      <h2>${lab.labTitle}</h2>
      ${lab.items.map(it => `
        <h3>${it.q}</h3>
        <p>${it.a.replace(/\\n\\n/g, '</p><p>').replace(/\\n/g, '<br>')}</p>
      `).join('')}
      ${lab.labNum === '07' ? `
        <table border="1" cellpadding="6" cellspacing="0">
          <thead>
            <tr>
              <th style="width: 20%;">Technique</th>
              <th style="width: 30%;">Core Principle</th>
              <th style="width: 25%;">Key Advantage</th>
              <th style="width: 25%;">Key Limitation</th>
            </tr>
          </thead>
          <tbody>
            <tr>
              <td><strong>Thresholding</strong></td>
              <td>Splits pixels based on intensity cutoff (Global Otsu / Local Adaptive).</td>
              <td>Fastest, minimal computational overhead.</td>
              <td>Fails under uneven illumination or gradients.</td>
            </tr>
            <tr>
              <td><strong>Region-Based (Watershed)</strong></td>
              <td>Treats gradients as topographic relief, flooding from marker minima.</td>
              <td>Separates touching, overlapping objects cleanly.</td>
              <td>Prone to over-segmentation without exact markers.</td>
            </tr>
            <tr>
              <td><strong>Edge-Based</strong></td>
              <td>Identifies boundaries via gradient discontinuities (Canny) and groups contours.</td>
              <td>Preserves exact boundary shapes and perimeters.</td>
              <td>Sensitive to noise; broken edge gaps prevent closed regions.</td>
            </tr>
            <tr>
              <td><strong>Semantic (DeepLabV3)</strong></td>
              <td>Assigns class labels to every pixel using deep convolutional feature maps.</td>
              <td>Understands complex real-world multi-class scenes.</td>
              <td>Computationally heavy, requires pre-trained weights.</td>
            </tr>
          </tbody>
        </table>
      ` : ''}
    `).join('')}
  </div>
</body>
</html>`;
  }

  function generateQAHTML() {
    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>CSST106 Laboratory Theory &amp; Report Q&amp;A Compilation - Aeron Jhed</title>
  <style>
    body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #000000; max-width: 900px; margin: 40px auto; padding: 0 24px; }
    h1 { color: #000000; border-bottom: 2px solid #000000; padding-bottom: 10px; font-size: 24px; }
    h2 { color: #000000; border-bottom: 1px solid #000000; padding-bottom: 6px; margin-top: 32px; font-size: 18px; }
    h3 { color: #000000; font-size: 15px; margin-top: 20px; font-weight: bold; }
    p { margin: 8px 0; font-size: 14px; color: #000000; }
    strong, b { color: #000000; }
    code { font-family: monospace; background: #F1F5F9; color: #000000; padding: 2px 5px; border-radius: 4px; font-size: 13px; }
    table { width: 100%; border-collapse: collapse; margin: 16px 0; font-size: 13px; border: 1px solid #000000; }
    th, td { border: 1px solid #000000; padding: 10px 12px; text-align: left; color: #000000; }
    th { background: #F8FAFC; font-weight: bold; color: #000000; }
    .meta { color: #000000; font-size: 14px; margin-bottom: 24px; border-bottom: 1px solid #000000; padding-bottom: 8px; }
    @media print { body { max-width: 100%; padding: 0; } * { color: #000000 !important; } }
  </style>
</head>
<body>
  <h1>CSST106: Perception &amp; Computer Vision — Laboratory Q&amp;A Compilation</h1>
  <div class="meta">
    <strong>Student:</strong> Aeron Jhed | BSCS-4B &bull; <strong>Course:</strong> CSST106 &bull; <strong>Date:</strong> ${new Date().toLocaleDateString()}
  </div>
  ${QA_DATABASE.map(lab => `
    <h2>${lab.labTitle}</h2>
    ${lab.items.map(it => `
      <h3>${it.q}</h3>
      <p>${it.a.replace(/\\n\\n/g, '</p><p>').replace(/\\n/g, '<br>')}</p>
    `).join('')}
    ${lab.labNum === '07' ? `
      <table border="1" cellpadding="6" cellspacing="0">
        <thead>
          <tr>
            <th>Technique</th>
            <th>Core Principle</th>
            <th>Key Advantage</th>
            <th>Key Limitation</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td><strong>Thresholding</strong></td>
            <td>Splits pixels based on intensity cutoff (Global Otsu / Local Adaptive).</td>
            <td>Fastest, minimal computational overhead.</td>
            <td>Fails under uneven illumination or gradients.</td>
          </tr>
          <tr>
            <td><strong>Region-Based (Watershed)</strong></td>
            <td>Treats gradients as topographic relief, flooding from marker minima.</td>
            <td>Separates touching, overlapping objects cleanly.</td>
            <td>Prone to over-segmentation without exact markers.</td>
          </tr>
          <tr>
            <td><strong>Edge-Based</strong></td>
            <td>Identifies boundaries via gradient discontinuities (Canny) and groups contours.</td>
            <td>Preserves exact boundary shapes and perimeters.</td>
            <td>Sensitive to noise; broken edge gaps prevent closed regions.</td>
          </tr>
          <tr>
            <td><strong>Semantic (DeepLabV3)</strong></td>
            <td>Assigns class labels to every pixel using deep convolutional feature maps.</td>
            <td>Understands complex real-world multi-class scenes.</td>
            <td>Computationally heavy, requires pre-trained weights.</td>
          </tr>
        </tbody>
      </table>
    ` : ''}
  `).join('')}
</body>
</html>`;
  }

  function downloadQACompilation() {
    const sel = document.getElementById('qa-format-select');
    const format = sel ? sel.value : 'docx';

    // Direct download of the authentic CSST106 Laboratory QA Word Document
    if (format === 'docx' || format === 'doc') {
      const a = document.createElement('a');
      a.href = './assets/CSST106_Laboratory_QA_Compilation.docx';
      a.download = 'CSST106_Laboratory_QA_Compilation.docx';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      showToast('Downloaded CSST106_Laboratory_QA_Compilation.docx successfully!', 'success');
      return;
    }

    let content = '';
    let mimeType = 'text/plain;charset=utf-8';
    let filename = `CSST106_Laboratory_QA_Compilation.${format}`;

    if (format === 'md') {
      content = generateQAMarkdown();
      mimeType = 'text/markdown;charset=utf-8';
    } else if (format === 'txt') {
      content = generateQAText();
      mimeType = 'text/plain;charset=utf-8';
    } else if (format === 'html') {
      content = generateQAHTML();
      mimeType = 'text/html;charset=utf-8';
    } else if (format === 'json') {
      content = JSON.stringify({
        course: "CSST106: Perception & Computer Vision",
        student: "Aeron Jhed | BSCS-4B",
        generatedAt: new Date().toISOString(),
        labs: QA_DATABASE
      }, null, 2);
      mimeType = 'application/json;charset=utf-8';
    }

    const blob = new Blob([content], { type: mimeType });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast(`Downloaded ${filename} successfully!`, 'success');
  }

  function copyQAToClipboard() {
    const text = generateQAText();
    navigator.clipboard.writeText(text).then(() => {
      showToast('All 7 Labs Q&A copied to clipboard!', 'success');
    }).catch(() => {
      showToast('Failed to copy to clipboard.', 'error');
    });
  }

  function copyLabQA(labNum) {
    const lab = QA_DATABASE.find(l => l.labNum === String(labNum).padStart(2, '0'));
    if (!lab) return;
    let txt = `[${lab.labTitle}]\n` + '-'.repeat(50) + '\n\n';
    lab.items.forEach(it => {
      txt += `Q: ${it.q}\n\n`;
      txt += `A: ${it.a}\n\n`;
    });
    navigator.clipboard.writeText(txt).then(() => {
      showToast(`Lab ${labNum} Q&A copied to clipboard!`, 'success');
    }).catch(() => {
      showToast('Failed to copy to clipboard.', 'error');
    });
  }

  // Expose global methods
  window.App = {
    DropZone,
    showToast,
    showLoading,
    hideLoading,
    apiCall,
    downloadImage,
    formatNumber,
    createImageElement,
    toggleTheme,
    promptCancelAnalysis,
    dismissCancelModal,
    confirmCancelAnalysis,
    downloadQACompilation,
    copyQAToClipboard,
    copyLabQA,
    IS_GITHUB_PAGES
  };

  window.switchLab = switchLab;
  window.downloadLabResults = downloadLabResults;
  window.toggleTheme = toggleTheme;
  window.promptCancelAnalysis = promptCancelAnalysis;
  window.dismissCancelModal = dismissCancelModal;
  window.confirmCancelAnalysis = confirmCancelAnalysis;
  window.downloadQACompilation = downloadQACompilation;
  window.copyQAToClipboard = copyQAToClipboard;
  window.copyLabQA = copyLabQA;

  document.addEventListener('DOMContentLoaded', () => {
    // Initialize Dark/Light Theme
    initTheme();

    // Check if running on GitHub Pages / Static Demo Mode
    if (IS_GITHUB_PAGES) {
      console.log('[CSST106] Running in GitHub Pages static demo mode.');
      const studentBadge = document.querySelector('.student-badge');
      if (studentBadge) {
        studentBadge.title = 'GitHub Pages Static Demo Mode (Interactive samples active)';
      }
      setTimeout(() => {
        showToast('Static Demo Mode (GitHub Pages): Precomputed interactive samples are active. Run locally with Node.js & Python for live processing.', 'info', 6000);
      }, 500);
    }

    // Restore saved lab or hash if present
    const hash = window.location.hash.replace('#', '');
    const saved = localStorage.getItem('active_lab');
    const initialLab = (hash && document.getElementById(hash)) ? hash : (saved && document.getElementById(saved)) ? saved : 'lab1';
    switchLab(initialLab);
  });

})();
