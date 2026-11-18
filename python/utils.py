"""
Shared utilities for all CSST106 lab Python scripts.
Handles image I/O, Base64 encoding, JSON output, and matplotlib rendering.
"""
import sys, os, json, base64, io, traceback
import cv2
import numpy as np

def read_image(path, mode=cv2.IMREAD_COLOR):
    """Read an image, return numpy array or None."""
    img = cv2.imread(path, mode)
    return img

def encode_image(img, fmt=".png"):
    """Encode a numpy image to base64 data URI string."""
    ok, buf = cv2.imencode(fmt, img)
    if not ok:
        return None
    b64 = base64.b64encode(buf).decode("utf-8")
    mime = "image/png" if fmt == ".png" else "image/jpeg"
    return f"data:{mime};base64,{b64}"

def encode_matplotlib_figure(fig):
    """Encode a matplotlib figure to base64 PNG data URI with crisp white background for A4 lab reports."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight", dpi=140, facecolor="#FFFFFF")
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return f"data:image/png;base64,{b64}"

def save_output_image(img, name, output_dir):
    """Save an image to the outputs directory, return the filename."""
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, name)
    cv2.imwrite(path, img)
    return name

def respond(data):
    """Print JSON response to stdout for the Node server to capture."""
    print(json.dumps(data, default=str))
    sys.stdout.flush()

def respond_error(msg):
    """Print error JSON response."""
    respond({"success": False, "error": str(msg)})

def run_safe(func):
    """Decorator-style wrapper: catches exceptions and returns error JSON."""
    try:
        func()
    except Exception as e:
        traceback.print_exc(file=sys.stderr)
        respond_error(str(e))

def parse_args():
    """Parse JSON args from command line argument."""
    if len(sys.argv) < 2:
        return {}
    try:
        return json.loads(sys.argv[1])
    except json.JSONDecodeError:
        return {}

def setup_matplotlib():
    """Configure matplotlib for non-interactive backend with clean white academic theme."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "#FFFFFF",
        "axes.facecolor": "#FFFFFF",
        "text.color": "#0F172A",
        "axes.labelcolor": "#0F172A",
        "xtick.color": "#475569",
        "ytick.color": "#475569",
        "axes.edgecolor": "#CBD5E1",
        "grid.color": "#E2E8F0",
        "font.family": "sans-serif",
    })
    return plt
