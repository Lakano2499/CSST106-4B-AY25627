/**
 * CSST106 Perception & Computer Vision — Lab Dashboard
 * labs.js: Lab-specific JavaScript initializing controls and form submissions for Labs 1 to 7.
 * 
 * Interacts with window.App:
 *   - DropZone
 *   - showToast
 *   - showLoading
 *   - hideLoading
 *   - apiCall
 *   - downloadImage
 *   - createImageElement
 */

(function () {
  'use strict';

  // =========================================================================
  // Core Utilities & Wrappers
  // =========================================================================

  /**
   * Safely access window.App methods with graceful fallbacks
   */
  const AppBridge = {
    showLoading: (msg) => {
      if (window.App && typeof window.App.showLoading === 'function') {
        window.App.showLoading(msg);
      }
    },
    hideLoading: () => {
      if (window.App && typeof window.App.hideLoading === 'function') {
        window.App.hideLoading();
      }
    },
    showToast: (msg, type = 'info') => {
      if (typeof msg === 'string' && (msg.includes('aborted by user') || msg.includes('cancelled by user'))) {
        return; // Silent on user cancellation; warning toast is handled by confirmCancelAnalysis
      }
      if (window.App && typeof window.App.showToast === 'function') {
        window.App.showToast(msg, type);
      } else {
        console.log(`[Toast: ${type}] ${msg}`);
      }
    },
    apiCall: async (endpoint, data) => {
      if (window.App && typeof window.App.apiCall === 'function') {
        return await window.App.apiCall(endpoint, data);
      }
      // Native fallback
      const isFormData = data instanceof FormData;
      const opts = {
        method: 'POST',
        body: isFormData ? data : JSON.stringify(data)
      };
      if (!isFormData) {
        opts.headers = { 'Content-Type': 'application/json' };
      }
      const resp = await fetch(endpoint, opts);
      if (!resp.ok) {
        let errMessage = `HTTP error ${resp.status}: ${resp.statusText}`;
        try {
          const errData = await resp.json();
          if (errData && errData.error) errMessage = errData.error;
        } catch (_) {}
        throw new Error(errMessage);
      }
      return await resp.json();
    },
    createDropZone: (selector, options = {}) => {
      const el = document.querySelector(selector);
      if (!el) return null;
      if (window.App && typeof window.App.DropZone === 'function') {
        try {
          return new window.App.DropZone(selector, options);
        } catch (e) {
          try {
            return window.App.DropZone(selector, options);
          } catch (err) {
            console.warn(`Failed to instantiate DropZone on ${selector}:`, err);
          }
        }
      }
      return null;
    },
    createImageElement: (src, label) => {
      if (window.App && typeof window.App.createImageElement === 'function') {
        try {
          return window.App.createImageElement(src, label);
        } catch (_) {}
      }
      return null;
    },
    downloadImage: (dataUrl, filename) => {
      if (window.App && typeof window.App.downloadImage === 'function') {
        window.App.downloadImage(dataUrl, filename);
      } else {
        const link = document.createElement('a');
        link.href = dataUrl;
        link.download = filename || 'download.png';
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
      }
    }
  };

  /**
   * Helper to retrieve selected file from DropZone instance or fallback input
   */
  function getDropzoneFile(dzInstance, selector) {
    if (dzInstance) {
      if (typeof dzInstance.getFile === 'function') {
        const file = dzInstance.getFile();
        if (file) return file;
      }
      if (dzInstance.file) return dzInstance.file;
      if (dzInstance.files && dzInstance.files.length > 0) return dzInstance.files[0];
    }
    const container = typeof selector === 'string' ? document.querySelector(selector) : selector;
    if (container) {
      const input = container.querySelector('input[type="file"]');
      if (input && input.files && input.files.length > 0) {
        return input.files[0];
      }
    }
    if (window.App && window.App.IS_GITHUB_PAGES) {
      // In GitHub Pages demo mode, provide a sample fallback File if user hasn't uploaded one
      return new File(["sample_placeholder"], "sample.jpg", { type: "image/jpeg" });
    }
    return null;
  }

  /**
   * Safe HTML string escaping
   */
  function escapeHtml(str) {
    if (str === null || str === undefined) return '';
    return String(str)
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  /**
   * Format image data string into valid data URI
   */
  function formatImageSrc(src) {
    if (!src) return '';
    if (typeof src !== 'string') return '';
    if (src.startsWith('data:') || src.startsWith('http://') || src.startsWith('https://') || src.startsWith('/') || src.startsWith('./') || src.startsWith('assets/')) {
      return src;
    }
    return `data:image/png;base64,${src}`;
  }

  /**
   * Convert snake_case or camelCase key into readable Title Case label
   */
  function formatLabelName(str) {
    if (!str) return '';
    return str
      .replace(/_/g, ' ')
      .replace(/([a-z])([A-Z])/g, '$1 $2')
      .replace(/\b\w/g, char => char.toUpperCase());
  }

  /**
   * Retrieve element value from multiple fallback selectors
   */
  function getInputValue(...selectors) {
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el && el.value !== undefined) {
        return el.value;
      }
    }
    return '';
  }

  /**
   * Retrieve checkbox / toggle checked status from multiple selectors
   */
  function getInputChecked(...selectors) {
    for (const sel of selectors) {
      const el = document.querySelector(sel);
      if (el && el.type === 'checkbox') {
        return el.checked;
      }
    }
    return false;
  }

  /**
   * Retrieve or create a lab's result container
   */
  function getResultsContainer(labNum) {
    return document.querySelector(`#lab${labNum}-results`) ||
           document.querySelector(`#lab${labNum} .results-container`) ||
           document.querySelector(`#lab-${labNum}-results`) ||
           document.querySelector(`#lab-${labNum} .results-container`);
  }

  // =========================================================================
  // Required Helper Functions
  // =========================================================================

  /**
   * Helper function renderResultImages(container, images, labels):
   * Clear container, for each image create a labeled img element and append
   * 
   * @param {HTMLElement|string} container 
   * @param {Array|Object} images 
   * @param {Array|Object} labels 
   */
  function renderResultImages(container, images, labels = []) {
    const el = typeof container === 'string' ? document.querySelector(container) : container;
    if (!el) return;
    el.innerHTML = '';

    const imageList = [];
    if (Array.isArray(images)) {
      images.forEach((img, idx) => {
        if (!img) return;
        const src = typeof img === 'string' ? img : (img.src || img.data || img.url);
        const label = (typeof img === 'object' && img.label) ? img.label : (labels[idx] || `Image ${idx + 1}`);
        imageList.push({ src, label });
      });
    } else if (images && typeof images === 'object') {
      Object.entries(images).forEach(([key, val], idx) => {
        if (!val) return;
        const src = typeof val === 'string' ? val : (val.src || val.data || val.url);
        const label = (Array.isArray(labels) ? labels[idx] : labels?.[key]) || formatLabelName(key);
        imageList.push({ src, label });
      });
    }

    const grid = document.createElement('div');
    grid.className = 'results-grid';

    imageList.forEach(item => {
      let customImgEl = AppBridge.createImageElement(formatImageSrc(item.src), item.label);

      if (customImgEl instanceof HTMLElement) {
        grid.appendChild(customImgEl);
      } else if (typeof customImgEl === 'string') {
        const wrap = document.createElement('div');
        wrap.innerHTML = customImgEl.trim();
        grid.appendChild(wrap.firstElementChild || wrap);
      } else {
        const card = document.createElement('div');
        card.className = 'result-image-card';

        const wrapper = document.createElement('div');
        wrapper.className = 'image-wrapper';

        const imgTag = document.createElement('img');
        imgTag.src = formatImageSrc(item.src);
        imgTag.alt = item.label;
        imgTag.loading = 'lazy';
        imgTag.className = 'result-img';

        wrapper.appendChild(imgTag);

        const labelTag = document.createElement('div');
        labelTag.className = 'image-label';
        labelTag.textContent = item.label;

        card.appendChild(wrapper);
        card.appendChild(labelTag);
        grid.appendChild(card);
      }
    });

    el.appendChild(grid);
  }

  /**
   * Helper function renderStatsGrid(container, stats):
   * Clear container, for each stat create stat-item with value and label
   * 
   * @param {HTMLElement|string} container 
   * @param {Array|Object} stats 
   */
  function renderStatsGrid(container, stats) {
    const el = typeof container === 'string' ? document.querySelector(container) : container;
    if (!el) return;
    el.innerHTML = '';

    const statList = [];
    if (Array.isArray(stats)) {
      stats.forEach(st => {
        if (!st) return;
        statList.push({
          label: st.label || st.name || '',
          value: st.value !== undefined ? st.value : ''
        });
      });
    } else if (stats && typeof stats === 'object') {
      Object.entries(stats).forEach(([k, v]) => {
        if (v === null || v === undefined) return;
        let displayVal = v;
        if (typeof v === 'number') {
          displayVal = Number.isInteger(v) ? v.toLocaleString() : v.toFixed(3);
        } else if (typeof v === 'boolean') {
          displayVal = v ? 'True' : 'False';
        } else if (typeof v === 'object') {
          displayVal = JSON.stringify(v);
        }
        statList.push({
          label: formatLabelName(k),
          value: displayVal
        });
      });
    }

    const grid = document.createElement('div');
    grid.className = 'stats-grid';

    statList.forEach(item => {
      const itemEl = document.createElement('div');
      itemEl.className = 'stat-item';

      const valEl = document.createElement('div');
      valEl.className = 'stat-value';
      valEl.textContent = item.value;

      const lblEl = document.createElement('div');
      lblEl.className = 'stat-label';
      lblEl.textContent = item.label;

      itemEl.appendChild(valEl);
      itemEl.appendChild(lblEl);
      grid.appendChild(itemEl);
    });

    el.appendChild(grid);
  }

  /**
   * Helper function renderTable(container, headers, rows):
   * Build table HTML string and set innerHTML
   * 
   * @param {HTMLElement|string} container 
   * @param {Array<string>} headers 
   * @param {Array<Array|Object>} rows 
   */
  function renderTable(container, headers = [], rows = []) {
    const el = typeof container === 'string' ? document.querySelector(container) : container;
    if (!el) return;

    let html = '<div class="table-responsive"><table class="data-table">';

    if (headers && headers.length > 0) {
      html += '<thead><tr>';
      headers.forEach(h => {
        html += `<th>${escapeHtml(h)}</th>`;
      });
      html += '</tr></thead>';
    }

    html += '<tbody>';
    if (!rows || rows.length === 0) {
      const colSpan = headers.length || 1;
      html += `<tr><td colspan="${colSpan}" class="text-center text-muted" style="text-align:center; padding:1.5rem; color:#a6adc8;">No data available</td></tr>`;
    } else {
      rows.forEach(row => {
        html += '<tr>';
        if (Array.isArray(row)) {
          row.forEach(cell => {
            html += `<td>${cell !== null && cell !== undefined ? escapeHtml(String(cell)) : '-'}</td>`;
          });
        } else if (typeof row === 'object' && row !== null) {
          headers.forEach(h => {
            const keyCandidates = [
              h.toLowerCase().replace(/[^a-z0-9]/g, '_'),
              h.toLowerCase(),
              h
            ];
            let cellVal = undefined;
            for (const key of keyCandidates) {
              if (row[key] !== undefined) {
                cellVal = row[key];
                break;
              }
            }
            if (cellVal === undefined) {
              const vals = Object.values(row);
              cellVal = vals.length > 0 ? vals[0] : '-';
            }
            html += `<td>${cellVal !== null && cellVal !== undefined ? escapeHtml(String(cellVal)) : '-'}</td>`;
          });
        }
        html += '</tr>';
      });
    }
    html += '</tbody></table></div>';

    el.innerHTML = html;
  }

  /**
   * Automatically synchronize range inputs with badge/span indicators
   */
  function setupRangeIndicators() {
    document.querySelectorAll('input[type="range"]').forEach(range => {
      const updateIndicator = () => {
        const id = range.id;
        const candidates = [
          document.getElementById(`${id}-val`),
          document.getElementById(`${id}-value`),
          range.parentElement?.querySelector('.range-value'),
          range.parentElement?.querySelector('output'),
          range.nextElementSibling
        ];
        const indicator = candidates.find(el => el && (el.classList.contains('range-value') || el.id.includes('val')));
        if (indicator) {
          indicator.textContent = range.value;
        }
      };
      range.addEventListener('input', updateIndicator);
      range.addEventListener('change', updateIndicator);
      updateIndicator();
    });
  }

  // =========================================================================
  // LAB 1: Image Manipulation
  // =========================================================================

  function initLab1() {
    const dropzone = AppBridge.createDropZone('#lab1-dropzone');
    const resultsContainer = getResultsContainer(1);

    /**
     * Send Lab 1 action request
     */
    async function sendLab1Action(action, params = {}, loadingMessage = 'Processing Lab 1...') {
      const file = getDropzoneFile(dropzone, '#lab1-dropzone');
      if (!file) {
        AppBridge.showToast('Please upload an image or click Sample 1 / Sample 2 to test.', 'warning');
        return;
      }

      const formData = new FormData();
      formData.append('image', file);
      formData.append('action', action);
      formData.append('params', JSON.stringify({ action, ...params }));

      // Also append individual param keys for server flexibility
      Object.entries(params).forEach(([k, v]) => {
        formData.append(k, v);
      });

      AppBridge.showLoading(loadingMessage);
      try {
        const data = await AppBridge.apiCall('/api/lab1', formData);
        if (!data || data.success === false) {
          throw new Error(data?.error || `Failed to perform ${action}`);
        }

        AppBridge.showToast(`Lab 1 ${formatLabelName(action)} completed successfully!`, 'success');
        renderLab1Output(data, action);
      } catch (err) {
        AppBridge.showToast(err.message || 'Lab 1 processing failed.', 'error');
      } finally {
        AppBridge.hideLoading();
      }
    }

    /**
     * Render results for Lab 1
     */
    function renderLab1Output(data, action) {
      if (!resultsContainer) return;
      resultsContainer.innerHTML = '';

      const sectionTitle = document.createElement('h3');
      sectionTitle.className = 'results-title';
      sectionTitle.textContent = `Lab 1: ${formatLabelName(action)} Results`;
      resultsContainer.appendChild(sectionTitle);

      // 1. Properties Card / Stats
      const props = data.properties || {
        height: data.height,
        width: data.width,
        channels: data.channels,
        dtype: data.dtype,
        total_pixels: data.total_pixels,
        color_space: data.color_space
      };

      if (props && (props.height !== undefined || props.total_pixels !== undefined)) {
        const statsBox = document.createElement('div');
        statsBox.className = 'result-section';
        statsBox.innerHTML = '<h4 class="section-subtitle">Image Properties</h4>';
        const statsGridWrap = document.createElement('div');
        renderStatsGrid(statsGridWrap, [
          { label: 'Height', value: `${props.height ?? '-'} px` },
          { label: 'Width', value: `${props.width ?? '-'} px` },
          { label: 'Channels', value: props.channels ?? '-' },
          { label: 'Data Type', value: props.dtype ?? '-' },
          { label: 'Total Pixels', value: props.total_pixels ? props.total_pixels.toLocaleString() : '-' },
          { label: 'Color Space', value: props.color_space ?? 'BGR / RGB' }
        ]);
        statsBox.appendChild(statsGridWrap);
        resultsContainer.appendChild(statsBox);
      }

      // 2. Images Grid
      const imagesToRender = [];
      const labelsToRender = [];

      if (data.original_image || data.original) {
        imagesToRender.push(data.original_image || data.original);
        labelsToRender.push('Original Image');
      }

      if (action === 'full_analysis') {
        if (data.grayscale_image || data.grayscale) {
          imagesToRender.push(data.grayscale_image || data.grayscale);
          labelsToRender.push('Grayscale Image');
        }
      } else if (action === 'crop') {
        const cropped = data.cropped_image || data.result_image || data.image;
        if (cropped) {
          imagesToRender.push(cropped);
          labelsToRender.push('Cropped Region (ROI)');
        }
      } else if (action === 'resize') {
        const resized = data.resized_image || data.result_image || data.image;
        if (resized) {
          imagesToRender.push(resized);
          labelsToRender.push(`Resized Image (${data.params?.width || ''}x${data.params?.height || ''})`);
        }
      } else if (action === 'rotate') {
        const rotated = data.rotated_image || data.result_image || data.image;
        if (rotated) {
          imagesToRender.push(rotated);
          labelsToRender.push(`Rotated Image (${data.params?.angle || ''}°)`);
        }
      } else if (data.result_image || data.image) {
        imagesToRender.push(data.result_image || data.image);
        labelsToRender.push('Processed Image');
      }

      if (imagesToRender.length > 0) {
        const imgSection = document.createElement('div');
        imgSection.className = 'result-section';
        imgSection.innerHTML = '<h4 class="section-subtitle">Image Comparison</h4>';
        const imgGridWrap = document.createElement('div');
        renderResultImages(imgGridWrap, imagesToRender, labelsToRender);
        imgSection.appendChild(imgGridWrap);
        resultsContainer.appendChild(imgSection);
      }

      // 3. Matplotlib Plot
      const plotData = data.plot || data.figure || data.plot_url;
      if (plotData) {
        const plotSection = document.createElement('div');
        plotSection.className = 'result-section';
        plotSection.innerHTML = '<h4 class="section-subtitle">Color Channels & Pixel Intensity Distribution</h4>';

        const plotImg = document.createElement('img');
        plotImg.src = formatImageSrc(plotData);
        plotImg.alt = 'Lab 1 Analysis Plot';
        plotImg.className = 'result-plot';

        plotSection.appendChild(plotImg);
        resultsContainer.appendChild(plotSection);
      }

      resultsContainer.classList.add('visible');
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    // Button: #lab1-full-analysis
    const fullAnalysisBtn = document.getElementById('lab1-full-analysis');
    if (fullAnalysisBtn) {
      fullAnalysisBtn.addEventListener('click', (e) => {
        e.preventDefault();
        sendLab1Action('full_analysis', {}, 'Performing full image analysis & calculating statistics...');
      });
    }

    // Button: #lab1-crop
    const cropBtn = document.getElementById('lab1-crop');
    if (cropBtn) {
      cropBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const x1 = parseInt(getInputValue('#lab1-crop-x1', '#lab1-x1'), 10) || 0;
        const y1 = parseInt(getInputValue('#lab1-crop-y1', '#lab1-y1'), 10) || 0;
        const x2 = parseInt(getInputValue('#lab1-crop-x2', '#lab1-x2'), 10) || 100;
        const y2 = parseInt(getInputValue('#lab1-crop-y2', '#lab1-y2'), 10) || 100;
        sendLab1Action('crop', { x1, y1, x2, y2 }, `Cropping ROI [(${x1}, ${y1}) to (${x2}, ${y2})]...`);
      });
    }

    // Button: #lab1-resize
    const resizeBtn = document.getElementById('lab1-resize');
    if (resizeBtn) {
      resizeBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const width = parseInt(getInputValue('#lab1-resize-width', '#lab1-width'), 10) || 300;
        const height = parseInt(getInputValue('#lab1-resize-height', '#lab1-height'), 10) || 300;
        sendLab1Action('resize', { width, height }, `Resizing image to ${width}x${height} pixels...`);
      });
    }

    // Button: #lab1-rotate
    const rotateBtn = document.getElementById('lab1-rotate');
    if (rotateBtn) {
      rotateBtn.addEventListener('click', (e) => {
        e.preventDefault();
        const angle = parseInt(getInputValue('#lab1-rotate-angle', '#lab1-angle', '#lab1-rotate-select'), 10) || 90;
        sendLab1Action('rotate', { angle }, `Rotating image by ${angle}°...`);
      });
    }
  }

  // =========================================================================
  // LAB 2: Edge Detection
  // =========================================================================

  function initLab2() {
    const dropzone = AppBridge.createDropZone('#lab2-dropzone');
    const processBtn = document.getElementById('lab2-process');
    const resultsContainer = getResultsContainer(2);

    if (processBtn) {
      processBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        const file = getDropzoneFile(dropzone, '#lab2-dropzone');
        if (!file) {
          AppBridge.showToast('Please upload an image or click Sample 1 / Sample 2 to test.', 'warning');
          return;
        }

        const blurKernel = parseInt(getInputValue('#lab2-blur-kernel', '#lab2-blur', '#lab2-kernel'), 10) || 3;
        const cannyLow = parseInt(getInputValue('#lab2-canny-low', '#lab2-low'), 10) || 50;
        const cannyHigh = parseInt(getInputValue('#lab2-canny-high', '#lab2-high'), 10) || 150;

        const params = {
          blur_kernel: blurKernel,
          canny_low: cannyLow,
          canny_high: cannyHigh
        };

        const formData = new FormData();
        formData.append('image', file);
        formData.append('params', JSON.stringify(params));
        formData.append('blur_kernel', blurKernel);
        formData.append('canny_low', cannyLow);
        formData.append('canny_high', cannyHigh);

        AppBridge.showLoading('Executing Sobel & Canny edge detection algorithms...');
        try {
          const data = await AppBridge.apiCall('/api/lab2', formData);
          if (!data || data.success === false) {
            throw new Error(data?.error || 'Edge detection process failed.');
          }

          AppBridge.showToast('Edge detection completed successfully!', 'success');
          renderLab2Output(data);
        } catch (err) {
          AppBridge.showToast(err.message || 'Lab 2 edge detection failed.', 'error');
        } finally {
          AppBridge.hideLoading();
        }
      });
    }

    function renderLab2Output(data) {
      if (!resultsContainer) return;
      resultsContainer.innerHTML = '';

      const title = document.createElement('h3');
      title.className = 'results-title';
      title.textContent = 'Lab 2: Edge Detection Analysis';
      resultsContainer.appendChild(title);

      // 1. Edge Pixel Stats Card
      const statsObj = data.edge_pixel_stats || data.stats || {};
      const statsList = [
        { label: 'Gaussian Kernel', value: `${data.params?.blur_kernel || 3}x${data.params?.blur_kernel || 3}` },
        { label: 'Canny Low Threshold', value: data.params?.canny_low ?? 50 },
        { label: 'Canny High Threshold', value: data.params?.canny_high ?? 150 }
      ];

      if (statsObj.sobel_edges !== undefined || statsObj.sobel_pixel_count !== undefined) {
        statsList.push({
          label: 'Sobel Edge Pixels',
          value: (statsObj.sobel_edges ?? statsObj.sobel_pixel_count ?? 0).toLocaleString()
        });
      }
      if (statsObj.canny_edges !== undefined || statsObj.canny_pixel_count !== undefined) {
        statsList.push({
          label: 'Canny Edge Pixels',
          value: (statsObj.canny_edges ?? statsObj.canny_pixel_count ?? 0).toLocaleString()
        });
      }
      if (statsObj.canny_density !== undefined || statsObj.edge_percentage !== undefined) {
        const pct = statsObj.canny_density ?? statsObj.edge_percentage ?? 0;
        statsList.push({
          label: 'Canny Edge Density',
          value: typeof pct === 'number' ? `${pct.toFixed(2)}%` : `${pct}%`
        });
      }

      const statsSection = document.createElement('div');
      statsSection.className = 'result-section';
      statsSection.innerHTML = '<h4 class="section-subtitle">Edge Detection Statistics</h4>';
      const statsWrap = document.createElement('div');
      renderStatsGrid(statsWrap, statsList);
      statsSection.appendChild(statsWrap);
      resultsContainer.appendChild(statsSection);

      // 2. 6-Panel Comparison Plot
      const plot = data.plot || data.comparison_plot || data.figure;
      if (plot) {
        const plotSection = document.createElement('div');
        plotSection.className = 'result-section';
        plotSection.innerHTML = '<h4 class="section-subtitle">6-Panel Comparison Plot</h4>';

        const plotImg = document.createElement('img');
        plotImg.src = formatImageSrc(plot);
        plotImg.alt = '6-Panel Edge Detection Comparison';
        plotImg.className = 'result-plot';

        plotSection.appendChild(plotImg);
        resultsContainer.appendChild(plotSection);
      }

      // 3. Individual Images in Results Grid
      const images = data.images || {
        original: data.original_image || data.original,
        blurred: data.blurred_image || data.blurred,
        sobel_x: data.sobel_x_image || data.sobel_x,
        sobel_y: data.sobel_y_image || data.sobel_y,
        sobel_combined: data.sobel_image || data.sobel_combined,
        canny: data.canny_image || data.canny
      };

      const validImages = {};
      const labels = {
        original: 'Original Grayscale',
        blurred: 'Gaussian Blurred',
        sobel_x: 'Sobel Gradient X',
        sobel_y: 'Sobel Gradient Y',
        sobel_combined: 'Sobel Magnitude',
        canny: 'Canny Edges'
      };

      Object.entries(images).forEach(([key, src]) => {
        if (src) validImages[key] = src;
      });

      if (Object.keys(validImages).length > 0) {
        const imgSection = document.createElement('div');
        imgSection.className = 'result-section';
        imgSection.innerHTML = '<h4 class="section-subtitle">Individual Edge Maps</h4>';
        const imgWrap = document.createElement('div');
        renderResultImages(imgWrap, validImages, labels);
        imgSection.appendChild(imgWrap);
        resultsContainer.appendChild(imgSection);
      }

      resultsContainer.classList.add('visible');
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // =========================================================================
  // LAB 3: Histogram Equalization
  // =========================================================================

  function initLab3() {
    const dropzone = AppBridge.createDropZone('#lab3-dropzone');
    const processBtn = document.getElementById('lab3-process');
    const resultsContainer = getResultsContainer(3);

    if (processBtn) {
      processBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        const file = getDropzoneFile(dropzone, '#lab3-dropzone');
        if (!file) {
          AppBridge.showToast('Please upload an image or click Sample 1 / Sample 2 to test.', 'warning');
          return;
        }

        const alpha = parseFloat(getInputValue('#lab3-alpha')) || 1.5;
        const beta = parseInt(getInputValue('#lab3-beta'), 10) || 20;
        const clipLimit = parseFloat(getInputValue('#lab3-clip-limit', '#lab3-clip_limit')) || 2.0;
        const tileSize = parseInt(getInputValue('#lab3-tile-size', '#lab3-tile_size'), 10) || 8;

        const params = {
          alpha,
          beta,
          clip_limit: clipLimit,
          tile_size: tileSize
        };

        const formData = new FormData();
        formData.append('image', file);
        formData.append('params', JSON.stringify(params));
        formData.append('alpha', alpha);
        formData.append('beta', beta);
        formData.append('clip_limit', clipLimit);
        formData.append('tile_size', tileSize);

        AppBridge.showLoading('Computing histograms & applying global, CLAHE, and linear adjustments...');
        try {
          const data = await AppBridge.apiCall('/api/lab3', formData);
          if (!data || data.success === false) {
            throw new Error(data?.error || 'Histogram equalization failed.');
          }

          AppBridge.showToast('Histogram Equalization completed successfully!', 'success');
          renderLab3Output(data);
        } catch (err) {
          AppBridge.showToast(err.message || 'Lab 3 processing failed.', 'error');
        } finally {
          AppBridge.hideLoading();
        }
      });
    }

    function renderLab3Output(data) {
      if (!resultsContainer) return;
      resultsContainer.innerHTML = '';

      const title = document.createElement('h3');
      title.className = 'results-title';
      title.textContent = 'Lab 3: Histogram Equalization & Contrast Enhancement';
      resultsContainer.appendChild(title);

      // 1. Stats Table / Grid (Mean & Std for each method)
      const stats = data.stats || {};
      const rows = [];
      const headers = ['Method / Pipeline', 'Mean Intensity', 'Std Deviation', 'Min / Max'];

      const methods = [
        { key: 'original', name: 'Original (Low Contrast)' },
        { key: 'linear_contrast', name: 'Linear Contrast Stretching (α, β)' },
        { key: 'global_equalization', name: 'Global Equalization (equalizeHist)' },
        { key: 'clahe', name: 'Adaptive Equalization (CLAHE)' }
      ];

      methods.forEach(m => {
        const st = stats[m.key] || stats[m.key.replace(/_/g, '')];
        if (st) {
          rows.push([
            m.name,
            typeof st.mean === 'number' ? st.mean.toFixed(2) : (st.mean ?? '-'),
            typeof st.std === 'number' ? st.std.toFixed(2) : (st.std ?? '-'),
            `${st.min ?? 0} / ${st.max ?? 255}`
          ]);
        }
      });

      if (rows.length > 0) {
        const statsSection = document.createElement('div');
        statsSection.className = 'result-section';
        statsSection.innerHTML = '<h4 class="section-subtitle">Image Contrast Statistics Comparison</h4>';
        const tableWrap = document.createElement('div');
        renderTable(tableWrap, headers, rows);
        statsSection.appendChild(tableWrap);
        resultsContainer.appendChild(statsSection);
      }

      // 2. 8-Panel Plot (4 Images + 4 Histograms)
      const plot = data.plot || data.histogram_plot || data.figure;
      if (plot) {
        const plotSection = document.createElement('div');
        plotSection.className = 'result-section';
        plotSection.innerHTML = '<h4 class="section-subtitle">8-Panel Comparison (Images & Histograms)</h4>';

        const plotImg = document.createElement('img');
        plotImg.src = formatImageSrc(plot);
        plotImg.alt = '8-Panel Histogram Comparison Plot';
        plotImg.className = 'result-plot';

        plotSection.appendChild(plotImg);
        resultsContainer.appendChild(plotSection);
      }

      // 3. Color Equalized Result & Processed Images Grid
      const images = [];
      const labels = [];

      if (data.original_image || data.images?.original) {
        images.push(data.original_image || data.images?.original);
        labels.push('Original Image');
      }
      if (data.color_equalized || data.color_result || data.images?.color_equalized) {
        images.push(data.color_equalized || data.color_result || data.images?.color_equalized);
        labels.push('Color Equalized (YUV / HSV)');
      }
      if (data.clahe_image || data.images?.clahe) {
        images.push(data.clahe_image || data.images?.clahe);
        labels.push('CLAHE Result');
      }
      if (data.equalized_image || data.images?.global_equalized) {
        images.push(data.equalized_image || data.images?.global_equalized);
        labels.push('Global Equalization');
      }

      if (images.length > 0) {
        const imgSection = document.createElement('div');
        imgSection.className = 'result-section';
        imgSection.innerHTML = '<h4 class="section-subtitle">Processed & Equalized Images</h4>';
        const imgWrap = document.createElement('div');
        renderResultImages(imgWrap, images, labels);
        imgSection.appendChild(imgWrap);
        resultsContainer.appendChild(imgSection);
      }

      resultsContainer.classList.add('visible');
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // =========================================================================
  // LAB 4: Feature Detection
  // =========================================================================

  function initLab4() {
    const dropzone1 = AppBridge.createDropZone('#lab4-dropzone1') || AppBridge.createDropZone('#lab4-dropzone');
    const dropzone2 = AppBridge.createDropZone('#lab4-dropzone2');
    const processBtn = document.getElementById('lab4-process');
    const resultsContainer = getResultsContainer(4);

    if (processBtn) {
      processBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        const file1 = getDropzoneFile(dropzone1, '#lab4-dropzone1') || getDropzoneFile(dropzone1, '#lab4-dropzone');
        if (!file1) {
          AppBridge.showToast('Please upload Image 1 or click Sample 1 / Sample 2 to test.', 'warning');
          return;
        }

        const file2 = getDropzoneFile(dropzone2, '#lab4-dropzone2');

        const detector = getInputValue('#lab4-detector') || 'all';
        const hessianThreshold = parseInt(getInputValue('#lab4-hessian-threshold', '#lab4-hessian'), 10) || 400;
        const upright = getInputChecked('#lab4-upright');

        const params = {
          detector,
          hessian_threshold: hessianThreshold,
          upright
        };

        const formData = new FormData();
        formData.append('image', file1);
        if (file2) {
          formData.append('image2', file2);
        }
        formData.append('params', JSON.stringify(params));
        formData.append('detector', detector);
        formData.append('hessian_threshold', hessianThreshold);
        formData.append('upright', upright);

        AppBridge.showLoading(`Detecting features using ${detector.toUpperCase()} (Hessian: ${hessianThreshold})...`);
        try {
          const data = await AppBridge.apiCall('/api/lab4', formData);
          if (!data || data.success === false) {
            throw new Error(data?.error || 'Feature detection processing failed.');
          }

          AppBridge.showToast('Feature detection completed successfully!', 'success');
          renderLab4Output(data);
        } catch (err) {
          AppBridge.showToast(err.message || 'Lab 4 processing failed.', 'error');
        } finally {
          AppBridge.hideLoading();
        }
      });
    }

    function renderLab4Output(data) {
      if (!resultsContainer) return;
      resultsContainer.innerHTML = '';

      const title = document.createElement('h3');
      title.className = 'results-title';
      title.textContent = 'Lab 4: Local Feature Detection & Matching';
      resultsContainer.appendChild(title);

      // 1. SURF Patent / Contrib Warning if unavailable
      if (data.surf_available === false || data.warning) {
        const warnBox = document.createElement('div');
        warnBox.className = 'alert alert-warning';
        warnBox.style.padding = '1rem 1.25rem';
        warnBox.style.marginBottom = '1.5rem';
        warnBox.style.borderRadius = '8px';
        warnBox.style.backgroundColor = 'rgba(249, 226, 175, 0.15)';
        warnBox.style.border = '1px solid #f9e2af';
        warnBox.style.color = '#f9e2af';
        warnBox.innerHTML = `<strong>Notice:</strong> ${escapeHtml(data.warning || 'SURF is patented and unavailable in standard OpenCV builds. The system has automatically fallen back to SIFT and ORB.')}`;
        resultsContainer.appendChild(warnBox);
        AppBridge.showToast(data.warning || 'SURF unavailable. Using alternative detectors.', 'warning');
      }

      // 2. Comparison Table (Detector Name, Keypoints, Descriptor Dims, Time)
      const comparisonData = data.comparison || data.detectors || data.table || [];
      const headers = ['Detector Name', 'Keypoints Count', 'Descriptor Dimensions', 'Execution Time'];
      const rows = [];

      if (Array.isArray(comparisonData)) {
        comparisonData.forEach(item => {
          rows.push([
            item.name || item.detector || 'Unknown',
            (item.keypoints || item.keypoints_count || 0).toLocaleString(),
            item.descriptor_dim || item.dimensions || (item.name?.toLowerCase().includes('orb') ? '32 (binary)' : '128 (float)'),
            typeof item.time === 'number' ? `${item.time.toFixed(2)} ms` : `${item.time || '-'}`
          ]);
        });
      } else if (typeof comparisonData === 'object' && comparisonData !== null) {
        Object.entries(comparisonData).forEach(([det, info]) => {
          rows.push([
            det.toUpperCase(),
            (info.keypoints || info.count || 0).toLocaleString(),
            info.descriptor_dim || info.dimension || '-',
            typeof info.time === 'number' ? `${info.time.toFixed(2)} ms` : `${info.time || '-'}`
          ]);
        });
      }

      if (rows.length > 0) {
        const tableSection = document.createElement('div');
        tableSection.className = 'result-section';
        tableSection.innerHTML = '<h4 class="section-subtitle">Detector Performance Comparison</h4>';
        const tableWrap = document.createElement('div');
        renderTable(tableWrap, headers, rows);
        tableSection.appendChild(tableWrap);
        resultsContainer.appendChild(tableSection);
      }

      // 3. Images Grid (Keypoints & Matches)
      const images = [];
      const labels = [];

      if (data.keypoints_image || data.images?.keypoints) {
        images.push(data.keypoints_image || data.images?.keypoints);
        labels.push('Detected Keypoints (Rich Markers)');
      }

      if (data.matches_image || data.images?.matches) {
        images.push(data.matches_image || data.images?.matches);
        labels.push('Feature Matching (FLANN / BFMatcher)');
      }

      if (images.length > 0) {
        const imgSection = document.createElement('div');
        imgSection.className = 'result-section';
        imgSection.innerHTML = '<h4 class="section-subtitle">Visualized Features & Matching</h4>';
        const imgWrap = document.createElement('div');
        renderResultImages(imgWrap, images, labels);
        imgSection.appendChild(imgWrap);
        resultsContainer.appendChild(imgSection);
      }

      resultsContainer.classList.add('visible');
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // =========================================================================
  // LAB 5: PCA Eigenfaces
  // =========================================================================

  function initLab5() {
    const trainBtn = document.getElementById('lab5-train');
    const testBtn = document.getElementById('lab5-test');
    const resultsContainer = getResultsContainer(5);

    if (trainBtn) {
      trainBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        const nComponents = parseInt(getInputValue('#lab5-components', '#lab5-n-components', '#lab5-n_components'), 10) || 150;
        const whiten = getInputChecked('#lab5-whiten');

        const payload = {
          action: 'train',
          n_components: nComponents,
          whiten: whiten,
          params: {
            action: 'train',
            n_components: nComponents,
            whiten: whiten
          }
        };

        AppBridge.showLoading('Extracting Eigenfaces & training SVM (1-2 mins on first run if downloading LFW dataset)...');
        try {
          const data = await AppBridge.apiCall('/api/lab5', payload);
          if (!data || data.success === false) {
            throw new Error(data?.error || 'PCA Eigenfaces model training failed.');
          }

          AppBridge.showToast('PCA Eigenfaces training completed successfully!', 'success');
          renderLab5Output(data);
        } catch (err) {
          AppBridge.showToast(err.message || 'Lab 5 PCA execution failed.', 'error');
        } finally {
          AppBridge.hideLoading();
        }
      });
    }

    if (testBtn) {
      testBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        AppBridge.showLoading('Evaluating face recognition on test split...');
        try {
          const data = await AppBridge.apiCall('/api/lab5', {
            action: 'test',
            params: { action: 'test' }
          });
          if (!data || data.success === false) {
            throw new Error(data?.error || 'Evaluation on test set failed.');
          }
          AppBridge.showToast('Face recognition evaluation finished!', 'success');
          renderLab5Output(data);
        } catch (err) {
          AppBridge.showToast(err.message || 'Lab 5 test action failed.', 'error');
        } finally {
          AppBridge.hideLoading();
        }
      });
    }

    function renderLab5Output(data) {
      if (!resultsContainer) return;
      resultsContainer.innerHTML = '';

      const title = document.createElement('h3');
      title.className = 'results-title';
      title.textContent = 'Lab 5: Dimensionality Reduction via PCA (Eigenfaces)';
      resultsContainer.appendChild(title);

      // 1. Dataset Info Card & Stats
      const dInfo = data.dataset_info || {};
      const accuracyVal = data.accuracy ?? data.metrics?.accuracy;
      const accText = typeof accuracyVal === 'number' ? `${(accuracyVal * 100).toFixed(1)}%` : (accuracyVal || 'N/A');

      const stats = [
        { label: 'SVM Test Accuracy', value: accText },
        { label: '90% Variance Components', value: data.variance_90_components ?? data.components_for_90_percent ?? 'N/A' },
        { label: 'Total Samples', value: (dInfo.n_samples ?? data.n_samples ?? '-').toLocaleString() },
        { label: 'Image Dimensions', value: `${dInfo.h ?? data.h ?? 50}x${dInfo.w ?? data.w ?? 37} (${dInfo.n_features ?? data.n_features ?? 1850} px)` },
        { label: 'Classes (People)', value: dInfo.n_classes ?? data.n_classes ?? '-' },
        { label: 'Top Components', value: data.n_components ?? data.params?.n_components ?? 150 }
      ];

      const statsSection = document.createElement('div');
      statsSection.className = 'result-section';
      statsSection.innerHTML = '<h4 class="section-subtitle">Dataset & Model Metrics</h4>';
      const statsWrap = document.createElement('div');
      renderStatsGrid(statsWrap, stats);
      statsSection.appendChild(statsWrap);
      resultsContainer.appendChild(statsSection);

      // 2. Eigenfaces Gallery Plot
      const eigenPlot = data.eigenfaces_plot || data.gallery_plot || data.eigenfaces;
      if (eigenPlot) {
        const efSection = document.createElement('div');
        efSection.className = 'result-section';
        efSection.innerHTML = '<h4 class="section-subtitle">Extracted Eigenfaces Gallery (Principal Components)</h4>';

        const plotImg = document.createElement('img');
        plotImg.src = formatImageSrc(eigenPlot);
        plotImg.alt = 'Eigenfaces Gallery';
        plotImg.className = 'result-plot';

        efSection.appendChild(plotImg);
        resultsContainer.appendChild(efSection);
      }

      // 3. Cumulative Explained Variance Curve Plot
      const varPlot = data.variance_plot || data.cumulative_variance_plot || data.variance_curve;
      if (varPlot) {
        const varSection = document.createElement('div');
        varSection.className = 'result-section';
        varSection.innerHTML = '<h4 class="section-subtitle">Cumulative Explained Variance Curve</h4>';

        const varImg = document.createElement('img');
        varImg.src = formatImageSrc(varPlot);
        varImg.alt = 'Cumulative Explained Variance Curve';
        varImg.className = 'result-plot';

        varSection.appendChild(varImg);
        resultsContainer.appendChild(varSection);
      }

      // 4. Failure Gallery Plot
      const failPlot = data.failure_plot || data.failures_plot || data.misclassified_plot;
      if (failPlot) {
        const failSection = document.createElement('div');
        failSection.className = 'result-section';
        failSection.innerHTML = '<h4 class="section-subtitle">Classification Failure Analysis (Misclassified Faces)</h4>';

        const failImg = document.createElement('img');
        failImg.src = formatImageSrc(failPlot);
        failImg.alt = 'Misclassified Faces Analysis';
        failImg.className = 'result-plot';

        failSection.appendChild(failImg);
        resultsContainer.appendChild(failSection);
      }

      // 5. Classification Report in <pre class="report-pre">
      const report = data.classification_report || data.report;
      if (report) {
        const repSection = document.createElement('div');
        repSection.className = 'result-section';
        repSection.innerHTML = '<h4 class="section-subtitle">Detailed Classification Report (Precision / Recall / F1)</h4>';

        const pre = document.createElement('pre');
        pre.className = 'report-pre';
        pre.style.backgroundColor = '#181825';
        pre.style.color = '#cdd6f4';
        pre.style.padding = '1.25rem';
        pre.style.borderRadius = '8px';
        pre.style.overflowX = 'auto';
        pre.style.fontFamily = 'monospace';
        pre.textContent = report;

        repSection.appendChild(pre);
        resultsContainer.appendChild(repSection);
      }

      resultsContainer.classList.add('visible');
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // =========================================================================
  // LAB 6: YOLO Object Detection
  // =========================================================================

  function initLab6() {
    const dropzone = AppBridge.createDropZone('#lab6-dropzone');
    const processBtn = document.getElementById('lab6-process');
    const resultsContainer = getResultsContainer(6);

    if (processBtn) {
      processBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        const file = getDropzoneFile(dropzone, '#lab6-dropzone');
        if (!file) {
          AppBridge.showToast('Please upload an image or click Sample 1 / Sample 2 to test.', 'warning');
          return;
        }

        const modelName = getInputValue('#lab6-model-name', '#lab6-model') || 'yolov8n.pt';
        const confidence = parseFloat(getInputValue('#lab6-confidence', '#lab6-conf')) || 0.25;

        const params = {
          model_name: modelName,
          confidence: confidence
        };

        const formData = new FormData();
        formData.append('image', file);
        formData.append('params', JSON.stringify(params));
        formData.append('model_name', modelName);
        formData.append('confidence', confidence);

        AppBridge.showLoading(`Running ${modelName} inference (confidence: ${(confidence * 100).toFixed(0)}%)...`);
        try {
          const data = await AppBridge.apiCall('/api/lab6', formData);
          if (!data || data.success === false) {
            throw new Error(data?.error || 'YOLO object detection failed.');
          }

          AppBridge.showToast('Object detection completed successfully!', 'success');
          renderLab6Output(data);
        } catch (err) {
          AppBridge.showToast(err.message || 'Lab 6 YOLO execution failed.', 'error');
        } finally {
          AppBridge.hideLoading();
        }
      });
    }

    function renderLab6Output(data) {
      if (!resultsContainer) return;
      resultsContainer.innerHTML = '';

      const title = document.createElement('h3');
      title.className = 'results-title';
      title.textContent = 'Lab 6: YOLO Deep Learning Object Detection';
      resultsContainer.appendChild(title);

      // 1. Original + Annotated Images Side by Side
      const images = [];
      const labels = [];

      if (data.original_image || data.original) {
        images.push(data.original_image || data.original);
        labels.push('Original Input Image');
      }

      if (data.annotated_image || data.annotated || data.result_image) {
        images.push(data.annotated_image || data.annotated || data.result_image);
        labels.push('YOLO Detections (Bounding Boxes & Confidences)');
      }

      if (images.length > 0) {
        const imgSection = document.createElement('div');
        imgSection.className = 'result-section';
        imgSection.innerHTML = '<h4 class="section-subtitle">Visual Detection Results</h4>';
        const imgWrap = document.createElement('div');
        renderResultImages(imgWrap, images, labels);
        imgSection.appendChild(imgWrap);
        resultsContainer.appendChild(imgSection);
      }

      // 2. Speed Metrics & Summary Stats
      const speed = data.speed || {};
      const detections = data.detections || [];
      const totalObjects = data.total_objects ?? detections.length;

      const prepMs = speed.preprocess !== undefined ? Number(speed.preprocess).toFixed(1) : '-';
      const inferMs = speed.inference !== undefined ? Number(speed.inference).toFixed(1) : '-';
      const postMs = speed.postprocess !== undefined ? Number(speed.postprocess).toFixed(1) : '-';
      const totalSpeedMs = (speed.preprocess || speed.inference || speed.postprocess)
        ? ((Number(speed.preprocess) || 0) + (Number(speed.inference) || 0) + (Number(speed.postprocess) || 0)).toFixed(1)
        : '-';

      const stats = [
        { label: 'Total Objects Found', value: totalObjects },
        { label: 'Pre-process Speed', value: `${prepMs} ms` },
        { label: 'Inference Speed', value: `${inferMs} ms` },
        { label: 'Post-process Speed', value: `${postMs} ms` },
        { label: 'Total Inference Time', value: `${totalSpeedMs} ms` },
        { label: 'Active Model', value: data.params?.model_name || data.model_name || 'YOLOv8' }
      ];

      const statsSection = document.createElement('div');
      statsSection.className = 'result-section';
      statsSection.innerHTML = '<h4 class="section-subtitle">Performance & Detection Summary</h4>';
      const statsWrap = document.createElement('div');
      renderStatsGrid(statsWrap, stats);
      statsSection.appendChild(statsWrap);

      // Objects by Class Summary Badges
      const classCounts = data.objects_by_class || {};
      if (Object.keys(classCounts).length === 0 && detections.length > 0) {
        detections.forEach(d => {
          const cName = d.class || d.label || 'object';
          classCounts[cName] = (classCounts[cName] || 0) + 1;
        });
      }

      if (Object.keys(classCounts).length > 0) {
        const badgeWrap = document.createElement('div');
        badgeWrap.style.marginTop = '1rem';
        badgeWrap.style.display = 'flex';
        badgeWrap.style.flexWrap = 'wrap';
        badgeWrap.style.gap = '0.5rem';

        Object.entries(classCounts).forEach(([cName, count]) => {
          const badge = document.createElement('span');
          badge.className = 'badge badge-class';
          badge.style.padding = '0.35rem 0.75rem';
          badge.style.borderRadius = '9999px';
          badge.style.backgroundColor = '#313244';
          badge.style.color = '#cdd6f4';
          badge.style.fontSize = '0.875rem';
          badge.textContent = `${formatLabelName(cName)}: ${count}`;
          badgeWrap.appendChild(badge);
        });

        statsSection.appendChild(badgeWrap);
      }

      resultsContainer.appendChild(statsSection);

      // 3. Detections Table
      const headers = ['#', 'Class Name', 'Confidence', 'Bounding Box (x1, y1, x2, y2)'];
      const rows = [];

      detections.forEach((d, idx) => {
        const confNum = typeof d.confidence === 'number' ? (d.confidence <= 1 ? d.confidence * 100 : d.confidence) : null;
        const confStr = confNum !== null ? `${confNum.toFixed(1)}%` : (d.confidence || '-');

        let bboxStr = '-';
        if (Array.isArray(d.bbox)) {
          bboxStr = `[${d.bbox.map(n => Math.round(n)).join(', ')}]`;
        } else if (d.box) {
          bboxStr = typeof d.box === 'string' ? d.box : JSON.stringify(d.box);
        }

        rows.push([
          idx + 1,
          formatLabelName(d.class || d.label || 'Unknown'),
          confStr,
          bboxStr
        ]);
      });

      const tableSection = document.createElement('div');
      tableSection.className = 'result-section';
      tableSection.innerHTML = '<h4 class="section-subtitle">Detected Objects Registry</h4>';
      const tableWrap = document.createElement('div');
      renderTable(tableWrap, headers, rows);
      tableSection.appendChild(tableWrap);
      resultsContainer.appendChild(tableSection);

      resultsContainer.classList.add('visible');
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // =========================================================================
  // LAB 7: Segmentation
  // =========================================================================

  function initLab7() {
    const dropzone = AppBridge.createDropZone('#lab7-dropzone');
    const processBtn = document.getElementById('lab7-process');
    const resultsContainer = getResultsContainer(7);

    if (processBtn) {
      processBtn.addEventListener('click', async (e) => {
        e.preventDefault();
        const file = getDropzoneFile(dropzone, '#lab7-dropzone');
        if (!file) {
          AppBridge.showToast('Please upload an image or click Sample 1 / Sample 2 to test.', 'warning');
          return;
        }

        const thresholdMethod = getInputValue('#lab7-threshold-method', '#lab7-method') || 'otsu';
        const blockSize = parseInt(getInputValue('#lab7-block-size', '#lab7-block_size'), 10) || 11;
        const watershedThreshold = parseFloat(getInputValue('#lab7-watershed-threshold', '#lab7-watershed')) || 0.5;
        const contourMinArea = parseInt(getInputValue('#lab7-contour-min-area', '#lab7-contour_min_area'), 10) || 100;
        const useDeepLearning = getInputChecked('#lab7-use-deep-learning', '#lab7-deep-learning');

        const params = {
          threshold_method: thresholdMethod,
          block_size: blockSize,
          watershed_threshold: watershedThreshold,
          contour_min_area: contourMinArea,
          use_deep_learning: useDeepLearning
        };

        const formData = new FormData();
        formData.append('image', file);
        formData.append('params', JSON.stringify(params));
        formData.append('threshold_method', thresholdMethod);
        formData.append('block_size', blockSize);
        formData.append('watershed_threshold', watershedThreshold);
        formData.append('contour_min_area', contourMinArea);
        formData.append('use_deep_learning', useDeepLearning);

        AppBridge.showLoading('Executing thresholding, watershed, contour, and semantic segmentation...');
        try {
          const data = await AppBridge.apiCall('/api/lab7', formData);
          if (!data || data.success === false) {
            throw new Error(data?.error || 'Segmentation process failed.');
          }

          AppBridge.showToast('Segmentation analysis completed successfully!', 'success');
          renderLab7Output(data);
        } catch (err) {
          AppBridge.showToast(err.message || 'Lab 7 segmentation execution failed.', 'error');
        } finally {
          AppBridge.hideLoading();
        }
      });
    }

    function renderLab7Output(data) {
      if (!resultsContainer) return;
      resultsContainer.innerHTML = '';

      const title = document.createElement('h3');
      title.className = 'results-title';
      title.textContent = 'Lab 7: Multi-Technique Image Segmentation';
      resultsContainer.appendChild(title);

      // 1. Stats Card (Otsu Threshold, Contour Count, Watershed Regions, Semantic Classes)
      const statsObj = data.stats || {};
      const otsuVal = data.otsu_threshold ?? statsObj.otsu_threshold ?? 'N/A';
      const contourVal = data.contour_count ?? statsObj.contour_count ?? 0;
      const watershedVal = data.watershed_regions ?? statsObj.watershed_regions ?? 0;

      let semanticClasses = data.semantic_classes ?? statsObj.semantic_classes;
      if (Array.isArray(semanticClasses)) {
        semanticClasses = semanticClasses.length > 0 ? semanticClasses.join(', ') : 'None detected';
      } else if (!semanticClasses) {
        semanticClasses = data.params?.use_deep_learning ? 'None detected' : 'N/A (DL disabled)';
      }

      const stats = [
        { label: 'Otsu Optimal Threshold', value: otsuVal },
        { label: 'Detected Contours', value: contourVal.toLocaleString() },
        { label: 'Watershed Segments', value: watershedVal.toLocaleString() },
        { label: 'Semantic Classes', value: semanticClasses }
      ];

      const statsSection = document.createElement('div');
      statsSection.className = 'result-section';
      statsSection.innerHTML = '<h4 class="section-subtitle">Segmentation Metrics</h4>';
      const statsWrap = document.createElement('div');
      renderStatsGrid(statsWrap, stats);
      statsSection.appendChild(statsWrap);
      resultsContainer.appendChild(statsSection);

      // 2. 6-Panel Comparison Plot
      const plot = data.plot || data.comparison_plot || data.figure;
      if (plot) {
        const plotSection = document.createElement('div');
        plotSection.className = 'result-section';
        plotSection.innerHTML = '<h4 class="section-subtitle">6-Panel Segmentation Comparison</h4>';

        const plotImg = document.createElement('img');
        plotImg.src = formatImageSrc(plot);
        plotImg.alt = '6-Panel Segmentation Comparison Plot';
        plotImg.className = 'result-plot';

        plotSection.appendChild(plotImg);
        resultsContainer.appendChild(plotSection);
      }

      // 3. Individual Segmented Maps (if available)
      const images = data.images || {
        original: data.original_image || data.original,
        threshold: data.threshold_image || data.threshold,
        adaptive: data.adaptive_image || data.adaptive,
        watershed: data.watershed_image || data.watershed,
        contours: data.contours_image || data.contours,
        semantic: data.semantic_image || data.semantic
      };

      const validImages = {};
      const labels = {
        original: 'Original Image',
        threshold: 'Otsu Global Thresholding',
        adaptive: 'Adaptive Thresholding',
        watershed: 'Watershed Markers & Basins',
        contours: 'Extracted Contours',
        semantic: 'Deep Learning Semantic Mask'
      };

      Object.entries(images).forEach(([k, v]) => {
        if (v) validImages[k] = v;
      });

      if (Object.keys(validImages).length > 0) {
        const imgSection = document.createElement('div');
        imgSection.className = 'result-section';
        imgSection.innerHTML = '<h4 class="section-subtitle">Individual Segmentation Layers</h4>';
        const imgWrap = document.createElement('div');
        renderResultImages(imgWrap, validImages, labels);
        imgSection.appendChild(imgWrap);
        resultsContainer.appendChild(imgSection);
      }

      resultsContainer.classList.add('visible');
      resultsContainer.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }

  // =========================================================================
  // Initialization on DOMContentLoaded
  // =========================================================================

  function initializeDashboard() {
    setupRangeIndicators();
    initLab1();
    initLab2();
    initLab3();
    initLab4();
    initLab5();
    initLab6();
    initLab7();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeDashboard);
  } else {
    initializeDashboard();
  }

  // Export helpers and initialization API to window.Labs for extensibility
  window.Labs = {
    renderResultImages,
    renderStatsGrid,
    renderTable,
    initLab1,
    initLab2,
    initLab3,
    initLab4,
    initLab5,
    initLab6,
    initLab7,
    initializeDashboard
  };

})();
