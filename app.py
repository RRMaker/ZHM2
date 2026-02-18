from flask import Flask, render_template, request
import os
import uuid

import cv2
import numpy as np
from PIL import Image


# Always resolve paths relative to this file (so running from any directory works)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

# Where traced outputs are saved (matches url_for('static', filename='outputs/...'))
OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional safety: reject huge uploads (e.g., 10 MB)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def compute_fish_contour_and_mask(img_rgb: np.ndarray):
    """
    Uses your existing HSV threshold + morphology + 'largest non-circular contour' logic.
    Returns:
      best_contour (or None),
      fish_mask_filled (uint8 0/255),
      hsv (uint8),
      base_mask (uint8)  # the original threshold mask (pre-contour selection)
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Mask non-background pixels (your original thresholds)
    lower = np.array([0, 40, 40], dtype=np.uint8)
    upper = np.array([180, 255, 220], dtype=np.uint8)
    base_mask = cv2.inRange(hsv, lower, upper)

    # Morphology clean-up
    kernel = np.ones((5, 5), np.uint8)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(base_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0

    # Pick the largest non-circular contour (your existing criteria)
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        if circularity < 0.6 and area > best_area:
            best_area = area
            best = c

    # Build a filled fish mask from the chosen contour
    fish_mask = np.zeros((img_rgb.shape[0], img_rgb.shape[1]), dtype=np.uint8)
    if best is not None:
        cv2.drawContours(fish_mask, [best], -1, 255, thickness=cv2.FILLED)

    return best, fish_mask, hsv, base_mask


def compute_bleeding_metrics(img_rgb: np.ndarray, fish_mask: np.ndarray, hsv: np.ndarray):
    """
      Total_Fish_Pixel_Area = area of whole-fish mask (pixels)
      Red_Pixel_Area = area of red pixels inside fish (pixels)
      Mean_Red_Pixel_Intensity = mean brightness (V channel) of those red pixels
      Integrated_Density = Red_Pixel_Area * Mean_Red_Pixel_Intensity
      Bleeding_Index = Integrated_Density / Total_Fish_Pixel_Area
    """
    total_fish_pixel_area = int(cv2.countNonZero(fish_mask))

    # HSV V channel for brightness
    _, _, v = cv2.split(hsv)

    # Red hue wraps around, so use two ranges
    lower1 = np.array([0, 80, 50], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)

    lower2 = np.array([170, 80, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    red_mask1 = cv2.inRange(hsv, lower1, upper1)
    red_mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # Restrict red detection to within the fish outline
    red_mask_in_fish = cv2.bitwise_and(red_mask, red_mask, mask=fish_mask)

    red_pixel_area = int(cv2.countNonZero(red_mask_in_fish))

    if red_pixel_area > 0:
        mean_red_pixel_intensity = float(np.mean(v[red_mask_in_fish > 0]))
    else:
        mean_red_pixel_intensity = 0.0

    integrated_density = float(red_pixel_area) * mean_red_pixel_intensity

    if total_fish_pixel_area > 0:
        bleeding_index = integrated_density / float(total_fish_pixel_area)
    else:
        bleeding_index = 0.0

    return {
        "total_fish_pixel_area": total_fish_pixel_area,
        "red_pixel_area": red_pixel_area,
        "mean_red_pixel_intensity": mean_red_pixel_intensity,
        "integrated_density": integrated_density,
        "bleeding_index": bleeding_index,
        "red_mask_in_fish": red_mask_in_fish,  # filled mask for measurement only
    }


def clean_red_mask(red_mask_in_fish: np.ndarray) -> np.ndarray:
    """
    Light cleanup so the edge trace is smoother and less speckly.
    """
    kernel = np.ones((3, 3), np.uint8)
    red_clean = cv2.morphologyEx(red_mask_in_fish, cv2.MORPH_OPEN, kernel, iterations=1)
    red_clean = cv2.morphologyEx(red_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    return red_clean


def make_edge_outline_mask(binary_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """
    Returns an EDGE-ONLY mask (255 on boundary pixels, 0 elsewhere).
    This cannot fill the region because it only contains boundary pixels.
    """
    # Canny expects 8-bit; binary_mask already is 0/255
    edges = cv2.Canny(binary_mask, 50, 150)

    if thickness > 1:
        k = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)

    return edges


def draw_bleed_edge_on_image(traced_rgb: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    """
    Overlay ONLY the edge pixels in RED. No fill, ever.
    traced_rgb is RGB => red is (255, 0, 0).
    """
    traced_rgb[edge_mask > 0] = (255, 0, 0)
    return traced_rgb


def make_bleed_outline_debug_image(edge_mask: np.ndarray) -> np.ndarray:
    """
    Black background with ONLY the red outline drawn.
    """
    h, w = edge_mask.shape
    out = np.zeros((h, w, 3), dtype=np.uint8)
    out[edge_mask > 0] = (255, 0, 0)
    return out


@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None

    metrics = None
    fish_mask_image = None
    red_mask_image = None  # will now be OUTLINE-ONLY debug image

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            return "No image uploaded", 400

        if not allowed_file(file.filename):
            return "Unsupported file type", 400

        # Load -> RGB numpy array
        try:
            img_rgb = np.array(Image.open(file.stream).convert("RGB"))
        except Exception:
            return "Invalid image file", 400

        # --- Fish outline (kept): black outline around fish ---
        best_contour, fish_mask, hsv, _base_mask = compute_fish_contour_and_mask(img_rgb)

        traced = img_rgb.copy()
        if best_contour is not None:
            cv2.drawContours(traced, [best_contour], -1, (0, 0, 0), thickness=4)

        # --- Bleeding metrics (kept) ---
        metrics_full = compute_bleeding_metrics(img_rgb=img_rgb, fish_mask=fish_mask, hsv=hsv)

        # --- NEW: EDGE-ONLY trace of bleed region (NO FILL) ---
        red_clean = clean_red_mask(metrics_full["red_mask_in_fish"])
        bleed_edge = make_edge_outline_mask(red_clean, thickness=2)  # adjust thickness if you want
        traced = draw_bleed_edge_on_image(traced, bleed_edge)

        # Save output outline image (to static/outputs/)
        uid = uuid.uuid4().hex
        filename = f"{uid}.png"
        out_path = os.path.join(OUTPUT_DIR, filename)
        Image.fromarray(traced).save(out_path)
        output_image = filename

        # OPTIONAL: save fish mask (filled, for debugging)
        fish_mask_filename = f"{uid}_fish_mask.png"
        Image.fromarray(fish_mask).save(os.path.join(OUTPUT_DIR, fish_mask_filename))
        fish_mask_image = fish_mask_filename

        # Save bleed outline debug image (outline ONLY, no blob)
        bleed_outline_img = make_bleed_outline_debug_image(bleed_edge)
        red_mask_filename = f"{uid}_bleed_outline.png"
        Image.fromarray(bleed_outline_img).save(os.path.join(OUTPUT_DIR, red_mask_filename))
        red_mask_image = red_mask_filename

        # Template-friendly metrics dict
        metrics = {
            "total_fish_pixel_area": metrics_full["total_fish_pixel_area"],
            "red_pixel_area": metrics_full["red_pixel_area"],
            "mean_red_pixel_intensity": metrics_full["mean_red_pixel_intensity"],
            "integrated_density": metrics_full["integrated_density"],
            "bleeding_index": metrics_full["bleeding_index"],
        }

    return render_template(
        "index.html",
        output_image=output_image,
        metrics=metrics,
        fish_mask_image=fish_mask_image,
        red_mask_image=red_mask_image,
    )


if __name__ == "__main__":
    app.run(debug=True)
