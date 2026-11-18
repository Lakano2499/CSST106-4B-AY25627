/**
 * CSST106 Computer Vision Lab Dashboard - Express.js Backend Server
 * 
 * Serves static files and provides API endpoints that execute Python scripts
 * for computer vision laboratory activities.
 */

const express = require('express');
const multer = require('multer');
const cors = require('cors');
const path = require('path');
const fs = require('fs');
const { spawn, spawnSync } = require('child_process');

const app = express();
const PORT = process.env.PORT || 3000;

// Directory configuration
const DOCS_DIR = path.join(__dirname, 'docs');
const UPLOAD_DIR = path.join(__dirname, 'uploads');
const OUTPUT_DIR = path.join(__dirname, 'outputs');
const CACHE_DIR = path.join(__dirname, 'cache');
const PYTHON_DIR = path.join(__dirname, 'python');

// Ensure required directories exist
[DOCS_DIR, UPLOAD_DIR, OUTPUT_DIR, CACHE_DIR, PYTHON_DIR].forEach((dir) => {
  if (!fs.existsSync(dir)) {
    fs.mkdirSync(dir, { recursive: true });
  }
});

// Detect Python executable (Windows "py -3.12" first, fallback to "python")
let pythonConfig = {
  command: 'python',
  args: [],
  version: 'unknown'
};

function detectPython() {
  // 1. Try Windows py launcher with -3.12
  try {
    const res = spawnSync('py', ['-3.12', '--version'], { encoding: 'utf8', timeout: 5000 });
    if (res.status === 0 || (res.stdout && res.stdout.includes('Python')) || (res.stderr && res.stderr.includes('Python'))) {
      const ver = (res.stdout || res.stderr).trim();
      pythonConfig = {
        command: 'py',
        args: ['-3.12'],
        version: ver || 'Python 3.12'
      };
      console.log(`[Python] Detected Windows launcher: ${pythonConfig.command} ${pythonConfig.args.join(' ')} (${pythonConfig.version})`);
      return;
    }
  } catch (err) {
    // 'py' launcher not available
  }

  // 2. Fallback to standard "python"
  try {
    const res = spawnSync('python', ['--version'], { encoding: 'utf8', timeout: 5000 });
    if (res.status === 0 || (res.stdout && res.stdout.includes('Python')) || (res.stderr && res.stderr.includes('Python'))) {
      const ver = (res.stdout || res.stderr).trim();
      pythonConfig = {
        command: 'python',
        args: [],
        version: ver || 'Python'
      };
      console.log(`[Python] Detected standard python: ${pythonConfig.command} (${pythonConfig.version})`);
      return;
    }
  } catch (err) {
    // 'python' executable not found
  }

  console.warn('[Python] Warning: Neither "py -3.12" nor "python" was found. Falling back to default "python".');
  pythonConfig = {
    command: 'python',
    args: [],
    version: 'not detected'
  };
}

detectPython();

// Middleware
app.use(cors());
app.use(express.json({ limit: '50mb' }));
app.use(express.urlencoded({ extended: true, limit: '50mb' }));

// Static file serving
app.use(express.static(DOCS_DIR));
app.use('/outputs', express.static(OUTPUT_DIR));
app.use('/uploads', express.static(UPLOAD_DIR));

// Multer Storage & Configuration
const storage = multer.diskStorage({
  destination: (req, file, cb) => {
    cb(null, UPLOAD_DIR);
  },
  filename: (req, file, cb) => {
    const timestamp = Date.now();
    const sanitized = file.originalname.replace(/[^a-zA-Z0-9._-]/g, '_');
    cb(null, `${timestamp}-${sanitized}`);
  }
});

const fileFilter = (req, file, cb) => {
  const allowedExtensions = /\.(jpe?g|png|bmp|tiff?|webp)$/i;
  const allowedMimeTypes = /^image\/(jpeg|png|bmp|x-ms-bmp|tiff|webp)$/i;

  const extMatch = allowedExtensions.test(path.extname(file.originalname).toLowerCase());
  const mimeMatch = allowedMimeTypes.test(file.mimetype);

  if (extMatch || mimeMatch) {
    cb(null, true);
  } else {
    cb(new Error('Only image files (jpeg, jpg, png, bmp, tiff, webp) are allowed!'));
  }
};

const upload = multer({
  storage,
  fileFilter,
  limits: {
    fileSize: 50 * 1024 * 1024 // 50MB max file size
  }
});

// Parameter parser helper
function parseParams(rawParams, fallbackBody = {}) {
  let parsed = {};
  if (rawParams) {
    if (typeof rawParams === 'object') {
      parsed = rawParams;
    } else if (typeof rawParams === 'string') {
      try {
        parsed = JSON.parse(rawParams);
      } catch (e) {
        parsed = {};
      }
    }
  }

  // Merge any top-level parameters from the body if not already present
  if (fallbackBody && typeof fallbackBody === 'object') {
    const extra = { ...fallbackBody };
    delete extra.params;
    delete extra.action;
    parsed = { ...extra, ...parsed };
  }

  return parsed;
}

// Timeout for Python scripts: 300 seconds (5 minutes)
const SCRIPT_TIMEOUT_MS = 300 * 1000;

/**
 * Spawns a Python script, passes args as a JSON string argument,
 * captures output, and resolves with parsed JSON or rejects with error.
 */
function runPython(scriptName, args) {
  return new Promise((resolve, reject) => {
    const scriptPath = path.join(PYTHON_DIR, scriptName);

    if (!fs.existsSync(scriptPath)) {
      return reject(new Error(`Python script not found: ${scriptPath}`));
    }

    const jsonArgs = JSON.stringify(args);
    const spawnArgs = [...pythonConfig.args, scriptPath, jsonArgs];

    const child = spawn(pythonConfig.command, spawnArgs, {
      cwd: __dirname,
      windowsHide: true
    });

    let stdoutData = '';
    let stderrData = '';
    let isSettled = false;

    const timer = setTimeout(() => {
      if (!isSettled) {
        isSettled = true;
        child.kill();
        reject(new Error(`Python script execution timed out after ${SCRIPT_TIMEOUT_MS / 1000} seconds`));
      }
    }, SCRIPT_TIMEOUT_MS);

    child.stdout.on('data', (chunk) => {
      stdoutData += chunk.toString('utf8');
    });

    child.stderr.on('data', (chunk) => {
      stderrData += chunk.toString('utf8');
    });

    child.on('error', (err) => {
      if (!isSettled) {
        isSettled = true;
        clearTimeout(timer);
        reject(new Error(`Failed to start Python process: ${err.message}`));
      }
    });

    child.on('close', (code) => {
      if (isSettled) return;
      isSettled = true;
      clearTimeout(timer);

      if (code !== 0) {
        const errMsg = stderrData.trim() || stdoutData.trim() || `Python process exited with code ${code}`;
        return reject(new Error(errMsg));
      }

      const trimmed = stdoutData.trim();
      if (!trimmed) {
        return reject(new Error('Python script produced empty output'));
      }

      // Parse JSON from stdout (search from end to handle any earlier stdout prints/logs)
      try {
        const parsed = JSON.parse(trimmed);
        return resolve(parsed);
      } catch (parseErr) {
        const lines = trimmed.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
        for (let i = lines.length - 1; i >= 0; i--) {
          try {
            const parsed = JSON.parse(lines[i]);
            return resolve(parsed);
          } catch (_) {}
        }
        return reject(new Error(`Failed to parse Python JSON output: ${parseErr.message}\nRaw output:\n${stdoutData}`));
      }
    });
  });
}

// -------------------------------------------------------------
// API Endpoints
// -------------------------------------------------------------

/**
 * GET /api/health
 * Returns server health and detected Python version.
 */
app.get('/api/health', (req, res) => {
  res.json({
    status: 'ok',
    python: pythonConfig.version
  });
});

/**
 * POST /api/lab1 - Lab 1: Image Manipulation
 * Multipart form: image file + action (string) + params (JSON string)
 * Runs: python/lab1_images.py with { image_path, output_dir, action, params }
 */
app.post('/api/lab1', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'Image file is required (field: "image")' });
    }

    const params = parseParams(req.body.params, req.body);
    const action = req.body.action || params.action || '';
    const args = {
      image_path: path.resolve(req.file.path),
      output_dir: path.resolve(OUTPUT_DIR),
      action,
      params
    };

    const result = await runPython('lab1_images.py', args);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message || 'Lab 1 execution failed'
    });
  }
});

/**
 * POST /api/lab2 - Lab 2: Edge Detection
 * Multipart form: image file + params (JSON string with blur_kernel, canny_low, canny_high)
 * Runs: python/lab2_edges.py with { image_path, output_dir, params }
 */
app.post('/api/lab2', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'Image file is required (field: "image")' });
    }

    const params = parseParams(req.body.params, req.body);
    const args = {
      image_path: path.resolve(req.file.path),
      output_dir: path.resolve(OUTPUT_DIR),
      params
    };

    const result = await runPython('lab2_edges.py', args);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message || 'Lab 2 execution failed'
    });
  }
});

/**
 * POST /api/lab3 - Lab 3: Histogram Equalization
 * Multipart form: image file + params (JSON string with alpha, beta, clip_limit, tile_size)
 * Runs: python/lab3_histogram.py with { image_path, output_dir, params }
 */
app.post('/api/lab3', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'Image file is required (field: "image")' });
    }

    const params = parseParams(req.body.params, req.body);
    const args = {
      image_path: path.resolve(req.file.path),
      output_dir: path.resolve(OUTPUT_DIR),
      params
    };

    const result = await runPython('lab3_histogram.py', args);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message || 'Lab 3 execution failed'
    });
  }
});

/**
 * POST /api/lab4 - Lab 4: Feature Detection
 * Multipart form: image file + optional image2 file + params (JSON string)
 * Runs: python/lab4_features.py with { image_path, image2_path, output_dir, params }
 */
app.post(
  '/api/lab4',
  upload.fields([{ name: 'image' }, { name: 'image2' }]),
  async (req, res) => {
    try {
      const files = req.files || {};
      const imageFile = files.image && files.image[0] ? files.image[0] : null;
      const image2File = files.image2 && files.image2[0] ? files.image2[0] : null;

      if (!imageFile) {
        return res.status(400).json({ success: false, error: 'Primary image file is required (field: "image")' });
      }

      const params = parseParams(req.body.params, req.body);
      const args = {
        image_path: path.resolve(imageFile.path),
        image2_path: image2File ? path.resolve(image2File.path) : null,
        image_path2: image2File ? path.resolve(image2File.path) : null,
        output_dir: path.resolve(OUTPUT_DIR),
        params
      };

      const result = await runPython('lab4_features.py', args);
      res.json(result);
    } catch (err) {
      res.status(500).json({
        success: false,
        error: err.message || 'Lab 4 execution failed'
      });
    }
  }
);

/**
 * POST /api/lab5 - Lab 5: PCA Eigenfaces
 * JSON body (no file upload): params (JSON with n_components, whiten, action)
 * Runs: python/lab5_pca.py with { output_dir, cache_dir, params }
 */
app.post('/api/lab5', async (req, res) => {
  try {
    let params = {};
    if (req.body && req.body.params) {
      params = parseParams(req.body.params, req.body);
    } else if (req.body && typeof req.body === 'object') {
      params = req.body;
    }

    const args = {
      output_dir: path.resolve(OUTPUT_DIR),
      cache_dir: path.resolve(CACHE_DIR),
      params
    };

    const result = await runPython('lab5_pca.py', args);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message || 'Lab 5 execution failed'
    });
  }
});

/**
 * POST /api/lab6 - Lab 6: YOLO Object Detection
 * Multipart form: image file + params (JSON string)
 * Runs: python/lab6_yolo.py with { image_path, output_dir, params }
 */
app.post('/api/lab6', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'Image file is required (field: "image")' });
    }

    const params = parseParams(req.body.params, req.body);
    const args = {
      image_path: path.resolve(req.file.path),
      output_dir: path.resolve(OUTPUT_DIR),
      params
    };

    const result = await runPython('lab6_yolo.py', args);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message || 'Lab 6 execution failed'
    });
  }
});

/**
 * POST /api/lab7 - Lab 7: Segmentation
 * Multipart form: image file + params (JSON string)
 * Runs: python/lab7_segmentation.py with { image_path, output_dir, params }
 */
app.post('/api/lab7', upload.single('image'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ success: false, error: 'Image file is required (field: "image")' });
    }

    const params = parseParams(req.body.params, req.body);
    const args = {
      image_path: path.resolve(req.file.path),
      output_dir: path.resolve(OUTPUT_DIR),
      params
    };

    const result = await runPython('lab7_segmentation.py', args);
    res.json(result);
  } catch (err) {
    res.status(500).json({
      success: false,
      error: err.message || 'Lab 7 execution failed'
    });
  }
});

// -------------------------------------------------------------
// Upload Cleanup Task
// Deletes uploaded files older than 1 hour on startup & every 30 mins
// -------------------------------------------------------------
const CLEANUP_MAX_AGE_MS = 60 * 60 * 1000; // 1 hour
const CLEANUP_INTERVAL_MS = 30 * 60 * 1000; // 30 minutes

function cleanupUploadedFiles() {
  try {
    if (!fs.existsSync(UPLOAD_DIR)) return;
    const now = Date.now();
    const files = fs.readdirSync(UPLOAD_DIR);
    let deletedCount = 0;

    for (const file of files) {
      if (file === '.gitkeep') continue;
      const filePath = path.join(UPLOAD_DIR, file);
      try {
        const stats = fs.statSync(filePath);
        if (stats.isFile() && (now - stats.mtimeMs > CLEANUP_MAX_AGE_MS)) {
          fs.unlinkSync(filePath);
          deletedCount++;
        }
      } catch (err) {
        console.error(`[Cleanup] Error processing file "${file}":`, err.message);
      }
    }

    if (deletedCount > 0) {
      console.log(`[Cleanup] Removed ${deletedCount} uploaded file(s) older than 1 hour.`);
    }
  } catch (err) {
    console.error('[Cleanup] Failed to run directory cleanup:', err.message);
  }
}

// Run cleanup on startup and schedule periodic execution
cleanupUploadedFiles();
setInterval(cleanupUploadedFiles, CLEANUP_INTERVAL_MS);

// -------------------------------------------------------------
// Error Handling Middleware
// -------------------------------------------------------------
app.use((err, req, res, next) => {
  console.error('[Error Middleware]', err);

  if (err instanceof multer.MulterError) {
    if (err.code === 'LIMIT_FILE_SIZE') {
      return res.status(400).json({
        success: false,
        error: 'File size exceeds 50MB limit'
      });
    }
    return res.status(400).json({
      success: false,
      error: `Upload error: ${err.message}`
    });
  }

  if (err) {
    return res.status(err.status || 400).json({
      success: false,
      error: err.message || 'Invalid request'
    });
  }

  next();
});

// -------------------------------------------------------------
// Server Start
// -------------------------------------------------------------
const server = app.listen(PORT, () => {
  console.log('====================================================');
  console.log(' CSST106 Computer Vision Lab Dashboard Server');
  console.log(` Running at: http://localhost:${PORT}`);
  console.log(` Static files: ${DOCS_DIR}`);
  console.log(` Upload directory: ${UPLOAD_DIR}`);
  console.log(` Output directory: ${OUTPUT_DIR}`);
  console.log(` Cache directory:  ${CACHE_DIR}`);
  console.log(` Python launcher:  ${pythonConfig.command} ${pythonConfig.args.join(' ')} (${pythonConfig.version})`);
  console.log('====================================================');
});

server.on('error', (err) => {
  if (err.code === 'EADDRINUSE') {
    console.error(`\n[Server Error] Port ${PORT} is already in use by another process.`);
    console.error(`To free port ${PORT} in PowerShell, run:`);
    console.error(`  Stop-Process -Id (Get-NetTCPConnection -LocalPort ${PORT}).OwningProcess -Force`);
    console.error(`Or run on another port:`);
    console.error(`  $env:PORT=3001; npm run dev\n`);
    process.exit(1);
  } else {
    console.error('[Server Error]', err);
  }
});

module.exports = { app, server, runPython, detectPython, cleanupUploadedFiles };
