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

# --- easy knobs ---
FISH_OUTLINE_THICKNESS = 10    # thicker black fish outline
BLEED_EDGE_THICKNESS = 2       # red bleed outline thickness

# Fish-mask stability knobs (these address your 176k vs 12k issue)
BORDER_TOUCH_MARGIN = 4        # reject contours that touch image border within this margin
MIN_FISH_AREA_FRAC = 0.03      # if fish mask < 3% of image, HSV likely failed -> fallback
MAX_FISH_AREA_FRAC = 0.75      # if fish mask > 75% of image, likely grabbed background/dish

# Edge-fallback knobs (addresses transparency/opacity problems)
EDGE_BLUR_KSIZE = 5
CANNY1, CANNY2 = 40, 120
EDGE_DILATE = 3                # thicker edges -> easier fill


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    """
    Keeps only the largest connected component of a 0/255 mask.
    This prevents dish/background blobs from merging in.
    """
    if binary_mask is None or binary_mask.size == 0:
        return binary_mask

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats((binary_mask > 0).astype(np.uint8), connectivity=8)
    if num_labels <= 1:
        return binary_mask

    # label 0 is background
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    out[labels == largest_idx] = 255
    return out


def contour_touches_border(contour: np.ndarray, h: int, w: int, margin: int = 0) -> bool:
    x, y, cw, ch = cv2.boundingRect(contour)
    if x <= margin or y <= margin:
        return True
    if (x + cw) >= (w - margin) or (y + ch) >= (h - margin):
        return True
    return False


def compute_fish_mask_hsv(img_rgb: np.ndarray):
    """
    Your original HSV non-background threshold + morphology.
    PLUS: keep largest connected component (stabilizes fish area a lot).
    Returns: hsv, base_mask_clean
    """
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

    # Original thresholds (non-background)
    lower = np.array([0, 40, 40], dtype=np.uint8)
    upper = np.array([180, 255, 220], dtype=np.uint8)
    base_mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=2)

    # NEW: keep only largest component to avoid grabbing dish edge/background regions
    base_mask = keep_largest_connected_component(base_mask)

    return hsv, base_mask


def compute_fish_mask_edge_fallback(img_rgb: np.ndarray) -> np.ndarray:
    """
    Edge-based segmentation for transparent/low-saturation fish:
    gray -> blur -> canny -> dilate -> find contour -> fill.
    """
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (EDGE_BLUR_KSIZE, EDGE_BLUR_KSIZE), 0)

    edges = cv2.Canny(gray, CANNY1, CANNY2)
    if EDGE_DILATE > 0:
        k = np.ones((EDGE_DILATE, EDGE_DILATE), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)

    # Find contours on edges and fill the best one
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    h, w = gray.shape[:2]

    best = None
    best_area = 0.0
    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        if contour_touches_border(c, h, w, margin=BORDER_TOUCH_MARGIN):
            continue
        if area > best_area:
            best_area = area
            best = c

    mask = np.zeros((h, w), dtype=np.uint8)
    if best is not None:
        cv2.drawContours(mask, [best], -1, 255, thickness=cv2.FILLED)

    # Keep largest component (just in case)
    mask = keep_largest_connected_component(mask)
    return mask


def compute_fish_contour_and_mask(img_rgb: np.ndarray):
    """
    Now does:
    1) HSV mask (your original method) + largest component
    2) Sanity-check fish area fraction
    3) If too small/too big, fallback to edge-based mask
    4) From final mask, pick best contour (largest non-circular, not touching border)
    Returns:
      best_contour (or None),
      fish_mask_filled (uint8 0/255),
      hsv (uint8),
      base_mask (uint8)  # the HSV mask (after cleanup), even if fallback used
    """
    h, w = img_rgb.shape[:2]
    img_area = float(h * w)

    hsv, base_mask = compute_fish_mask_hsv(img_rgb)
    fish_area = float(cv2.countNonZero(base_mask))
    frac = fish_area / img_area if img_area > 0 else 0.0

    # NEW: fallback if HSV is clearly wrong (solves opacity / dish-grab cases)
    final_mask = base_mask
    if frac < MIN_FISH_AREA_FRAC or frac > MAX_FISH_AREA_FRAC:
        fallback = compute_fish_mask_edge_fallback(img_rgb)
        fb_area = float(cv2.countNonZero(fallback))
        fb_frac = fb_area / img_area if img_area > 0 else 0.0

        # If fallback looks more reasonable, use it
        if MIN_FISH_AREA_FRAC <= fb_frac <= MAX_FISH_AREA_FRAC and fb_area > 0:
            final_mask = fallback

    # Find contours from final_mask
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    best = None
    best_area = 0.0

    for c in contours:
        area = cv2.contourArea(c)
        if area < 500:
            continue
        if contour_touches_border(c, h, w, margin=BORDER_TOUCH_MARGIN):
            continue

        perimeter = cv2.arcLength(c, True)
        if perimeter <= 0:
            continue

        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # Keep your "largest non-circular contour" behavior
        if circularity < 0.6 and area > best_area:
            best_area = area
            best = c

    # Build filled fish mask from chosen contour (this keeps your old behavior)
    fish_mask = np.zeros((h, w), dtype=np.uint8)
    if best is not None:
        cv2.drawContours(fish_mask, [best], -1, 255, thickness=cv2.FILLED)

    # If contour selection somehow fails, fall back to final_mask directly
    if best is None and cv2.countNonZero(final_mask) > 0:
        fish_mask = final_mask.copy()

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
        "red_mask_in_fish": red_mask_in_fish,  # FILLED mask for measurement + saving (B/W)
    }


def clean_red_mask(red_mask_in_fish: np.ndarray) -> np.ndarray:
    """
    Light cleanup so the edge trace is smoother and less speckly.
    Keeps it as a FILLED B/W mask (0/255).
    """
    kernel = np.ones((3, 3), np.uint8)
    red_clean = cv2.morphologyEx(red_mask_in_fish, cv2.MORPH_OPEN, kernel, iterations=1)
    red_clean = cv2.morphologyEx(red_clean, cv2.MORPH_CLOSE, kernel, iterations=1)
    red_clean = keep_largest_connected_component(red_clean)  # NEW: stabilize bleed blobs
    return red_clean


def make_edge_outline_mask(binary_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    """
    Returns an EDGE-ONLY mask (255 on boundary pixels, 0 elsewhere).
    """
    edges = cv2.Canny(binary_mask, 50, 150)
    if thickness > 1:
        k = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)
    return edges


def draw_bleed_edge_on_image(traced_rgb: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    """
    Overlay ONLY the edge pixels in RED. No fill.
    traced_rgb is RGB => red is (255, 0, 0).
    """
    traced_rgb[edge_mask > 0] = (255, 0, 0)
    return traced_rgb


@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None
    metrics = None
    fish_mask_image = None
    red_mask_image = None  # FILLED black/white bleed mask

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

        # --- Fish outline (black) ---
        best_contour, fish_mask, hsv, _base_mask = compute_fish_contour_and_mask(img_rgb)

        traced = img_rgb.copy()
        if best_contour is not None:
            cv2.drawContours(
                traced, [best_contour], -1,
                (0, 0, 0),
                thickness=FISH_OUTLINE_THICKNESS
            )
        else:
            # If we couldn't get a contour, outline fish_mask edges instead (rare)
            fish_edges = make_edge_outline_mask(fish_mask, thickness=max(2, FISH_OUTLINE_THICKNESS // 3))
            traced[fish_edges > 0] = (0, 0, 0)

        # --- Bleeding metrics ---
        metrics_full = compute_bleeding_metrics(img_rgb=img_rgb, fish_mask=fish_mask, hsv=hsv)

        # Clean bleed mask (still FILLED B/W)
        red_clean = clean_red_mask(metrics_full["red_mask_in_fish"])

        # --- Red OUTLINE overlay (no fill) ---
        bleed_edge = make_edge_outline_mask(red_clean, thickness=BLEED_EDGE_THICKNESS)
        traced = draw_bleed_edge_on_image(traced, bleed_edge)

        # Save output outlined image
        uid = uuid.uuid4().hex
        filename = f"{uid}.png"
        Image.fromarray(traced).save(os.path.join(OUTPUT_DIR, filename))
        output_image = filename

        # Save fish mask (filled B/W)
        fish_mask_filename = f"{uid}_fish_mask.png"
        Image.fromarray(fish_mask).save(os.path.join(OUTPUT_DIR, fish_mask_filename))
        fish_mask_image = fish_mask_filename

        # Save bleed mask (FILLED B/W)
        red_mask_filename = f"{uid}_bleed_mask.png"
        Image.fromarray(red_clean).save(os.path.join(OUTPUT_DIR, red_mask_filename))
        red_mask_image = red_mask_filename

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