from flask import Flask, render_template, request
import os
import uuid
import argparse

import cv2
import numpy as np
from PIL import Image

# Always resolve paths relative to this file (so running from any directory works)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

OUTPUT_DIR = os.path.join(STATIC_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024
ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

# --- display knobs ---
FISH_OUTLINE_THICKNESS = 5
BLEED_EDGE_THICKNESS = 2

# Fish-mask stability knobs
BORDER_TOUCH_MARGIN = 4
MIN_FISH_AREA_FRAC = 0.03
MAX_FISH_AREA_FRAC = 0.75

# Processing resolution
WORK_MAX_DIM = 1600

# -----------------------------
# Strong fin/tail enhancement
# -----------------------------
CLAHE_CLIP = 3.0
UNSHARP_AMOUNT = 2.0
UNSHARP_BLUR = 3

LINE_MULTISCALE_KS = (9, 15, 25)
LINE_BLACKHAT_WEIGHT = 1.2

# Fin/tail constrained growth
ENABLE_FIN_GROW = True
FIN_BAND_FRAC_1 = 0.020
FIN_BAND_FRAC_2 = 0.040
NEAR_EDGE_DILATE = 17

# Tail ROI (extra-permissive)
TAIL_ROI_ENABLE = True
TAIL_ROI_FRAC = 0.21
TAIL_CANNY1, TAIL_CANNY2 = 4, 20
TAIL_GROW_EXTRA_ITERS = 2

# --- protect fin/tail wisps while tightening/peeling ---
PROTECT_FIN_TEXTURE_ENABLE = True
FIN_TEXTURE_BLACKHAT_K = 25
FIN_TEXTURE_THR = 14
FIN_TEXTURE_DILATE_PX = 14

PROTECT_ROI_ENABLE = True
PEEL_GRAD_THR_ROI_DELTA = 6
TIGHTEN_ERODE_PX_ROI = 0

# Halo/padding removal (edge-guided peel)
PEEL_ENABLE = True
PEEL_ITERS = 7
PEEL_RING_PX = 3
PEEL_GRAD_THR = 24

# Edge-guided peel
EDGE_GUIDED_ENABLE = True
EDGE_GUIDED_DILATE_PX = 6
EDGE_GUIDED_MAX_DIST = 7
EDGE_GUIDED_GRAD_BONUS = 10

# Tighten (small) then restore fins near edges
TIGHTEN_ENABLE = True
TIGHTEN_ERODE_PX = 0
TIGHTEN_RESTORE_BAND_PX = 22
TIGHTEN_EDGE_THR = 18

# -----------------------------
# OUTLINE smoothing / snapping
# -----------------------------
OUTLINE_SMOOTH_ENABLE = True

SMOOTH_TARGET_STEP_PX = 1.15
SMOOTH_SIGMA = 5.2
SMOOTH_PASSES = 7

OUTLINE_SNAP_ENABLE = False

SAFE_SNAP_MIN_AREA_FRAC = 0.985
SNAP_SEARCH_PX = 12
SNAP_SAMPLES = 13
SNAP_GRAD_MIN = 12
SNAP_BLEND_ALPHA = 0.15
SNAP_MAX_MOVE_PX = 5

# -----------------------------
# BIG SMOOTHING: Signed-distance smoothing
# -----------------------------
SDF_SMOOTH_ENABLE = True
SDF_SIGMA = 3.8
SDF_ITERS = 1

# Keep smoothing mild so boundary stays close
SDF_LEVEL_STAGE_A = -0.06
FISH_TIGHTEN_ENABLE = True
FISH_TIGHTEN_ERODE_PX = 0
SDF_LEVEL_STAGE_B = 0.03

# -----------------------------
# Full-res tail recovery pass
# -----------------------------
TAIL_RECOVER_ENABLE = True
TAIL_RECOVER_FRAC = 0.28
TAIL_RECOVER_DILATE_ITERS = 4
TAIL_RECOVER_CANNY1 = 2
TAIL_RECOVER_CANNY2 = 18
TAIL_RECOVER_S_MAX = 140
TAIL_RECOVER_V_MIN = 20
TAIL_RECOVER_EDGE_DILATE = 3

# -----------------------------
# Tail-corner recovery
# -----------------------------
TAIL_CORNER_RECOVER_ENABLE = True
TAIL_CORNER_RECOVER_ROI_DILATE_PX = 36
TAIL_CORNER_RECOVER_NEAR_PX = 32
TAIL_CORNER_RECOVER_MAX_ADD_FRAC = 0.03
TAIL_CORNER_RECOVER_BG_MIN = 4.0
TAIL_CORNER_RECOVER_GRAD_MIN = 4.0
TAIL_CORNER_RECOVER_MIN_DROP_PX = 4
TAIL_CORNER_RECOVER_ANCHOR_SPAN_PX = 34
TAIL_CORNER_RECOVER_ANCHOR_GAP_PX = 10
TAIL_CORNER_RECOVER_ANCHOR_BAND_PX = 80
TAIL_CORNER_RECOVER_LOCAL_HALF_W_PX = 46
TAIL_CORNER_RECOVER_LOCAL_UP_PX = 40
TAIL_CORNER_RECOVER_LOCAL_DOWN_PX = 8
TAIL_CORNER_RECOVER_TOUCH_DILATE_PX = 4
TAIL_CORNER_RECOVER_LOCAL_MIN_AREA = 3
TAIL_CORNER_RECOVER_NEAR_COMBINED_PX = 3
TAIL_CORNER_RECOVER_LOCAL_Y_PAD_PX = 1

# -----------------------------
# Small boundary smoothing on mask
# -----------------------------
BOUNDARY_SMOOTH_ENABLE = True
BOUNDARY_SMOOTH_K = 3
BOUNDARY_SMOOTH_ITERS = 1

# -----------------------------
# NEW: local mask refinement
# -----------------------------
LOCAL_MASK_REFINE_ENABLE = True
LOCAL_MASK_REFINE_ITERS = 3
LOCAL_MASK_INNER_ERODE_PX = 10
LOCAL_MASK_OUTER_DILATE_PX = 12
LOCAL_MASK_OUTER_FIN_DILATE_PX = 24
LOCAL_MASK_OUTER_TAIL_DILATE_PX = 30
LOCAL_MASK_BG_THRESH = 30.0
LOCAL_MASK_EDGE_THRESH = 12.0
LOCAL_MASK_GRAD_THRESH = 10.0
LOCAL_MASK_POST_CLOSE_K = 3
LOCAL_MASK_POST_OPEN_K = 3

# -----------------------------
# contour edge refinement
# -----------------------------
# Turned off because it is the main source of small bumps/jagged snapping
CONTOUR_EDGE_REFINE_ENABLE = False
CONTOUR_REFINE_STEP_PX = 1.0
CONTOUR_REFINE_SMOOTH_SIGMA = 2.5
CONTOUR_REFINE_SMOOTH_PASSES = 3

CONTOUR_SEARCH_IN_PX = 4
CONTOUR_SEARCH_OUT_PX = 10
CONTOUR_SEARCH_OUT_FIN_PX = 26
CONTOUR_SEARCH_OUT_TAIL_PX = 34
CONTOUR_SEARCH_SAMPLES = 25

CONTOUR_EDGE_MIN_SCORE = 7.0
CONTOUR_DIST_PENALTY = 0.16
CONTOUR_MAX_MOVE_PX = 12.0
CONTOUR_MASK_PULL = 0.02
CONTOUR_FIN_BONUS = 20.0
CONTOUR_TAIL_BONUS = 26.0
CONTOUR_TAIL_CORNER_BONUS = 22.0

EDGE_SCORE_GRAD_WEIGHT = 1.00
EDGE_SCORE_CANNY_WEIGHT = 0.60
EDGE_SCORE_BG_WEIGHT = 0.65

DISPLAY_FIN_ROI_DILATE_PX = 22
DISPLAY_TAIL_ROI_DILATE_PX = 22
DISPLAY_POST_SMOOTH_SIGMA = 1.7
DISPLAY_POST_SMOOTH_PASSES = 2

# -----------------------------
# Tail-preserving display fix
# -----------------------------
TAIL_PRESERVE_ENABLE = True
TAIL_PRESERVE_ROI_FRAC = 0.34
TAIL_PRESERVE_DILATE_PX = 10
TAIL_PRESERVE_CLOSE_K = 5
TAIL_PRESERVE_FINAL_SMOOTH_SIGMA = 1.8
TAIL_PRESERVE_FINAL_SMOOTH_PASSES = 2

# Tail-corner fill after smoothing/preservation
TAIL_CORNER_FILL_ENABLE = True
TAIL_CORNER_FILL_HALF_W_PX = 56
TAIL_CORNER_FILL_UP_PX = 56
TAIL_CORNER_FILL_BOTTOM_BAND_PX = 16
TAIL_CORNER_FILL_ROI_DILATE_PX = 12
TAIL_CORNER_FILL_MAX_ADD_PX = 160

# -----------------------------
# Red mask smoothing / better fit
# -----------------------------
RED_MIN_COMPONENT_AREA = 25
RED_SDF_SMOOTH_ENABLE = True
RED_SDF_SIGMA = 2.2
RED_SDF_ITERS = 1
RED_TIGHTEN_ENABLE = True
RED_TIGHTEN_ERODE_PX = 1
RED_SDF_LEVEL_STAGE_A = 0.0
RED_SDF_LEVEL_STAGE_B = 0.15


def allowed_file(filename: str) -> bool:
    _, ext = os.path.splitext(filename.lower())
    return ext in ALLOWED_EXTENSIONS


def keep_largest_connected_component(binary_mask: np.ndarray) -> np.ndarray:
    if binary_mask is None or binary_mask.size == 0:
        return binary_mask
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        (binary_mask > 0).astype(np.uint8), connectivity=8
    )
    if num_labels <= 1:
        return binary_mask
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_idx = 1 + int(np.argmax(areas))
    out = np.zeros_like(binary_mask, dtype=np.uint8)
    out[labels == largest_idx] = 255
    return out


def remove_small_components(binary_mask: np.ndarray, min_area: int) -> np.ndarray:
    if binary_mask is None or binary_mask.size == 0:
        return binary_mask
    m = (binary_mask > 0).astype(np.uint8)
    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return (m * 255).astype(np.uint8)
    out = np.zeros_like(m, dtype=np.uint8)
    for lab in range(1, num):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area >= int(min_area):
            out[labels == lab] = 1
    return (out * 255).astype(np.uint8)


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    if mask is None or mask.size == 0:
        return mask
    h, w = mask.shape[:2]
    flood = mask.copy()
    floodfill_mask = np.zeros((h + 2, w + 2), np.uint8)
    cv2.floodFill(flood, floodfill_mask, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    filled = cv2.bitwise_or(mask, holes)
    return filled


def smooth_mask_boundary(mask: np.ndarray, k: int = 3, iters: int = 1) -> np.ndarray:
    if mask is None or mask.size == 0 or cv2.countNonZero(mask) == 0:
        return mask
    kk = int(k)
    kk = kk + 1 if kk % 2 == 0 else kk
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))

    out = mask.copy()
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, ker, iterations=int(iters))
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, ker, iterations=int(iters))
    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out


def sdf_smooth_mask(mask: np.ndarray, sigma: float = 6.0, iters: int = 1, level: float = 0.0) -> np.ndarray:
    if mask is None or mask.size == 0 or cv2.countNonZero(mask) == 0:
        return mask

    m = (mask > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3).astype(np.float32)
    sdf = dist_in - dist_out

    out_sdf = sdf.copy()
    for _ in range(int(max(1, iters))):
        out_sdf = cv2.GaussianBlur(out_sdf, (0, 0), float(sigma))

    out = (out_sdf > float(level)).astype(np.uint8) * 255
    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out


def resize_for_processing(img: np.ndarray, max_dim: int = WORK_MAX_DIM):
    h, w = img.shape[:2]
    if max(h, w) <= max_dim:
        return img, 1.0
    scale = max_dim / float(max(h, w))
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, scale


def super_boost_lines_gray(img_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP), tileGridSize=(8, 8))
    g = clahe.apply(gray)

    bh_sum = np.zeros_like(g, dtype=np.float32)
    for k in LINE_MULTISCALE_KS:
        kk = int(k) + (int(k) % 2 == 0)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
        bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel).astype(np.float32)
        bh_sum += bh
    bh_sum /= max(1.0, float(len(LINE_MULTISCALE_KS)))

    boosted = cv2.addWeighted(g.astype(np.float32), 1.0, bh_sum, float(LINE_BLACKHAT_WEIGHT), 0)
    boosted = np.clip(boosted, 0, 255).astype(np.uint8)

    blur_ksz = int(UNSHARP_BLUR)
    blur_ksz = blur_ksz + 1 if blur_ksz % 2 == 0 else blur_ksz
    b = cv2.GaussianBlur(boosted, (blur_ksz, blur_ksz), 0)
    boosted = cv2.addWeighted(boosted, 1.0 + float(UNSHARP_AMOUNT), b, -float(UNSHARP_AMOUNT), 0)
    return boosted


def fin_texture_roi(img_rgb: np.ndarray, base_mask: np.ndarray | None = None) -> np.ndarray | None:
    if (not PROTECT_FIN_TEXTURE_ENABLE) or img_rgb is None or img_rgb.size == 0:
        return None

    g = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=float(CLAHE_CLIP), tileGridSize=(8, 8))
    g = clahe.apply(g)

    kk = int(FIN_TEXTURE_BLACKHAT_K)
    kk = kk + 1 if kk % 2 == 0 else kk
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    bh_n = cv2.normalize(bh, None, 0, 255, cv2.NORM_MINMAX)
    _, tex = cv2.threshold(bh_n, int(FIN_TEXTURE_THR), 255, cv2.THRESH_BINARY)

    if base_mask is not None and base_mask.size > 0 and cv2.countNonZero(base_mask) > 0:
        h, w = base_mask.shape[:2]
        band_px = int(max(18, 0.05 * min(h, w)))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
        band = cv2.bitwise_and(cv2.dilate(base_mask, k, iterations=1), cv2.bitwise_not(base_mask))
        tex = cv2.bitwise_and(tex, tex, mask=band)

    d = int(max(1, FIN_TEXTURE_DILATE_PX))
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
    tex = cv2.dilate(tex, kd, iterations=1)
    return tex


def gradient_u8(gray_or_rgb: np.ndarray) -> np.ndarray:
    if gray_or_rgb.ndim == 3:
        gray = cv2.cvtColor(gray_or_rgb, cv2.COLOR_RGB2GRAY)
    else:
        gray = gray_or_rgb.copy()
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    return cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def compute_background_distance(img_rgb: np.ndarray, border_px: int = 12):
    h, w = img_rgb.shape[:2]
    b = int(max(2, min(border_px, h // 4, w // 4)))

    lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    top = lab[:b, :, :].reshape(-1, 3)
    bottom = lab[h - b:h, :, :].reshape(-1, 3)
    left = lab[:, :b, :].reshape(-1, 3)
    right = lab[:, w - b:w, :].reshape(-1, 3)

    border_pixels = np.concatenate([top, bottom, left, right], axis=0)
    bg = np.median(border_pixels, axis=0)

    dist = np.sqrt(np.sum((lab - bg) ** 2, axis=2)).astype(np.float32)
    p98 = float(np.percentile(dist, 98))
    if p98 > 1e-6:
        dist_norm = np.clip(dist / p98, 0.0, 1.0) * 255.0
    else:
        dist_norm = np.zeros_like(dist, dtype=np.float32)
    return dist_norm.astype(np.float32)


def compute_fish_mask_alpha(img_rgba: np.ndarray) -> np.ndarray | None:
    if img_rgba is None or img_rgba.ndim != 3 or img_rgba.shape[2] < 4:
        return None
    alpha = img_rgba[:, :, 3]
    mask = (alpha > 15).astype(np.uint8) * 255

    k = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k, iterations=1)
    mask = keep_largest_connected_component(mask)
    mask = fill_mask_holes(mask)
    return mask


def compute_fish_mask_hsv(img_rgb: np.ndarray):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    lower = np.array([0, 40, 40], dtype=np.uint8)
    upper = np.array([180, 255, 220], dtype=np.uint8)
    base_mask = cv2.inRange(hsv, lower, upper)

    kernel = np.ones((5, 5), np.uint8)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    base_mask = cv2.morphologyEx(base_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    base_mask = keep_largest_connected_component(base_mask)
    base_mask = fill_mask_holes(base_mask)
    return hsv, base_mask


def compute_fish_mask_edge_fallback(img_rgb: np.ndarray, alpha_mask: np.ndarray | None = None) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 40, 120)

    if alpha_mask is not None and alpha_mask.size > 0:
        a_edges = cv2.Canny(alpha_mask, 20, 60)
        edges = cv2.bitwise_or(edges, a_edges)

    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

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

    mask = keep_largest_connected_component(mask)
    mask = fill_mask_holes(mask)
    return mask


def contour_touches_border(contour: np.ndarray, h: int, w: int, margin: int = 0) -> bool:
    x, y, cw, ch = cv2.boundingRect(contour)
    if x <= margin or y <= margin:
        return True
    if (x + cw) >= (w - margin) or (y + ch) >= (h - margin):
        return True
    return False


def tail_roi_mask(fish_mask: np.ndarray, frac: float) -> np.ndarray | None:
    if fish_mask is None or cv2.countNonZero(fish_mask) == 0:
        return None

    h, w = fish_mask.shape[:2]
    cnts, _ = cv2.findContours(fish_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)

    M = cv2.moments(fish_mask, binaryImage=True)
    if M["m00"] == 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    pts = c[:, 0, :].astype(np.float32)
    d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
    tx, ty = pts[int(np.argmax(d2))]
    tx, ty = int(round(tx)), int(round(ty))

    r = int(max(40, frac * min(h, w)))
    x1, x2 = max(0, tx - r), min(w, tx + r)
    y1, y2 = max(0, ty - r), min(h, ty + r)

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[y1:y2, x1:x2] = 255
    return roi


def _tail_corner_roi(fish_mask: np.ndarray) -> np.ndarray | None:
    if fish_mask is None or cv2.countNonZero(fish_mask) == 0:
        return None

    cnt = contour_from_mask(fish_mask)
    if cnt is None or len(cnt) < 10:
        return None

    h, w = fish_mask.shape[:2]
    pts = cnt[:, 0, :].astype(np.float32)

    M = cv2.moments(fish_mask, binaryImage=True)
    if M["m00"] <= 0:
        return None
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    d2 = (pts[:, 0] - cx) ** 2 + (pts[:, 1] - cy) ** 2
    tail_pt = pts[int(np.argmax(d2))]

    x_band = np.abs(pts[:, 0] - tail_pt[0]) < 34
    y_band = np.abs(pts[:, 1] - tail_pt[1]) < 34
    near = pts[x_band | y_band]
    if len(near) == 0:
        near = np.array([tail_pt], dtype=np.float32)

    x1 = int(max(0, np.min(near[:, 0]) - 34))
    x2 = int(min(w, np.max(near[:, 0]) + 34))
    y1 = int(max(0, np.min(near[:, 1]) - 34))
    y2 = int(min(h, np.max(near[:, 1]) + 34))

    roi = np.zeros((h, w), dtype=np.uint8)
    roi[y1:y2, x1:x2] = 255
    return roi


def tail_recover_fullres(img_rgb: np.ndarray, fish_mask: np.ndarray) -> np.ndarray:
    if (not TAIL_RECOVER_ENABLE) or fish_mask is None or cv2.countNonZero(fish_mask) == 0:
        return fish_mask

    roi = tail_roi_mask(fish_mask, TAIL_RECOVER_FRAC)
    if roi is None:
        return fish_mask

    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    _, s, v = cv2.split(hsv)

    cand_color = ((s <= int(TAIL_RECOVER_S_MAX)) & (v >= int(TAIL_RECOVER_V_MIN))).astype(np.uint8) * 255
    cand_color = cv2.bitwise_and(cand_color, cand_color, mask=roi)

    boosted = super_boost_lines_gray(img_rgb)
    edges = cv2.Canny(boosted, int(TAIL_RECOVER_CANNY1), int(TAIL_RECOVER_CANNY2))
    if int(TAIL_RECOVER_EDGE_DILATE) > 0:
        kd = np.ones((int(TAIL_RECOVER_EDGE_DILATE), int(TAIL_RECOVER_EDGE_DILATE)), np.uint8)
        edges = cv2.dilate(edges, kd, iterations=1)
    edges = cv2.bitwise_and(edges, edges, mask=roi)

    candidate = cv2.bitwise_or(cand_color, edges)

    out = fish_mask.copy()
    k3 = np.ones((3, 3), np.uint8)
    for _ in range(int(max(1, TAIL_RECOVER_DILATE_ITERS))):
        grow = cv2.dilate(out, k3, iterations=1)
        add = cv2.bitwise_and(grow, candidate)
        add = cv2.bitwise_and(add, add, mask=roi)
        out = cv2.bitwise_or(out, add)

    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out


def contour_from_mask(mask: np.ndarray):
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    return max(cnts, key=cv2.contourArea)


def contour_to_filled_mask(h: int, w: int, contour: np.ndarray) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    if contour is not None:
        cv2.drawContours(m, [contour], -1, 255, thickness=cv2.FILLED)
    return m


def _gaussian_kernel1d(sigma: float, radius: int):
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    k = np.exp(-(x * x) / (2.0 * sigma * sigma))
    k /= np.sum(k)
    return k


def _circular_convolve_1d(arr: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    r = len(kernel) // 2
    pad = np.concatenate([arr[-r:], arr, arr[:r]], axis=0)
    out = np.convolve(pad, kernel, mode="valid")
    return out


def smooth_contour_highres(contour: np.ndarray, step_px: float, sigma: float, passes: int) -> np.ndarray:
    if contour is None or len(contour) < 50:
        return contour

    pts = contour[:, 0, :].astype(np.float32)
    d = np.sqrt(np.sum((np.roll(pts, -1, axis=0) - pts) ** 2, axis=1))
    per = float(np.sum(d))
    if per < 10:
        return contour

    step = max(0.6, float(step_px))
    n = int(max(900, per / step))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    s = np.linspace(0.0, per, n, endpoint=False)

    idx = np.searchsorted(cum, s, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 1)

    seg_len = d[idx]
    seg_len = np.where(seg_len < 1e-6, 1e-6, seg_len)
    t = (s - cum[idx]) / seg_len

    p0 = pts[idx]
    p1 = pts[(idx + 1) % len(pts)]
    res = (1.0 - t[:, None]) * p0 + t[:, None] * p1

    radius = int(max(2, 3 * sigma))
    k = _gaussian_kernel1d(float(sigma), radius)

    x = res[:, 0]
    y = res[:, 1]
    for _ in range(int(max(1, passes))):
        x = _circular_convolve_1d(x, k)
        y = _circular_convolve_1d(y, k)

    sm = np.stack([x, y], axis=1)
    sm = np.round(sm).astype(np.int32)
    return sm.reshape(-1, 1, 2)


def _resample_closed_contour(contour: np.ndarray, step_px: float) -> np.ndarray:
    if contour is None or len(contour) < 5:
        return None

    pts = contour[:, 0, :].astype(np.float32)
    seg = np.roll(pts, -1, axis=0) - pts
    d = np.sqrt(np.sum(seg * seg, axis=1))
    per = float(np.sum(d))
    if per < 5:
        return None

    n = int(max(320, round(per / max(0.75, float(step_px)))))
    cum = np.concatenate([[0.0], np.cumsum(d)])
    s = np.linspace(0.0, per, n, endpoint=False)

    idx = np.searchsorted(cum, s, side="right") - 1
    idx = np.clip(idx, 0, len(pts) - 1)

    seg_len = d[idx]
    seg_len = np.where(seg_len < 1e-6, 1e-6, seg_len)
    t = (s - cum[idx]) / seg_len

    p0 = pts[idx]
    p1 = pts[(idx + 1) % len(pts)]
    return (1.0 - t[:, None]) * p0 + t[:, None] * p1


def _smooth_closed_polyline(points: np.ndarray, sigma: float, passes: int) -> np.ndarray:
    if points is None or len(points) < 5:
        return points

    radius = int(max(2, 3 * sigma))
    k = _gaussian_kernel1d(float(sigma), radius)

    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)
    for _ in range(int(max(1, passes))):
        x = _circular_convolve_1d(x, k)
        y = _circular_convolve_1d(y, k)

    return np.stack([x, y], axis=1)


def _edge_score_map(img_rgb: np.ndarray) -> np.ndarray:
    boosted = super_boost_lines_gray(img_rgb)
    grad = gradient_u8(boosted).astype(np.float32)
    canny = cv2.Canny(boosted, 5, 20).astype(np.float32)
    canny = cv2.dilate(canny.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.float32)
    bg = compute_background_distance(img_rgb, border_px=12).astype(np.float32)

    score = (
        float(EDGE_SCORE_GRAD_WEIGHT) * grad
        + float(EDGE_SCORE_CANNY_WEIGHT) * canny
        + float(EDGE_SCORE_BG_WEIGHT) * bg
    )
    return score.astype(np.float32)


def _dilate_mask(mask: np.ndarray | None, px: int) -> np.ndarray | None:
    if mask is None:
        return None
    p = int(max(1, px))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * p + 1, 2 * p + 1))
    return cv2.dilate(mask, k, iterations=1)


def recover_tail_corner_hull(img_rgb: np.ndarray, fish_mask: np.ndarray) -> np.ndarray:
    """
    Recover a faint tail-tip corner by finding the deepest supported point
    below the current tail edge, then bridging to it with a narrow wedge.
    """
    if (not TAIL_CORNER_RECOVER_ENABLE) or fish_mask is None or cv2.countNonZero(fish_mask) == 0:
        return fish_mask

    roi = _tail_corner_roi(fish_mask)
    if roi is None:
        return fish_mask

    roi = _dilate_mask(roi, TAIL_CORNER_RECOVER_ROI_DILATE_PX)
    if roi is None or cv2.countNonZero(roi) == 0:
        return fish_mask

    near_tail = _dilate_mask(fish_mask, TAIL_CORNER_RECOVER_NEAR_PX)
    if near_tail is None or cv2.countNonZero(near_tail) == 0:
        return fish_mask

    boosted = super_boost_lines_gray(img_rgb)
    grad = gradient_u8(boosted).astype(np.float32)
    bg_dist = compute_background_distance(img_rgb, border_px=12).astype(np.float32)

    supported = np.zeros_like(fish_mask, dtype=np.uint8)
    supported[
        (roi > 0)
        & (near_tail > 0)
        & (fish_mask == 0)
        & (
            (bg_dist >= float(TAIL_CORNER_RECOVER_BG_MIN))
            | (grad >= float(TAIL_CORNER_RECOVER_GRAD_MIN))
        )
    ] = 255

    ys_mask, xs_mask = np.where((fish_mask > 0) & (roi > 0))
    if len(xs_mask) == 0:
        return fish_mask

    bottom_by_x: dict[int, int] = {}
    for x, y in zip(xs_mask, ys_mask):
        x = int(x)
        y = int(y)
        prev = bottom_by_x.get(x)
        if prev is None or y > prev:
            bottom_by_x[x] = y

    below_edge = np.zeros_like(fish_mask, dtype=np.uint8)
    min_drop = int(max(1, TAIL_CORNER_RECOVER_MIN_DROP_PX))
    for x, y0 in bottom_by_x.items():
        ys = np.where(supported[:, x] > 0)[0]
        if len(ys) == 0:
            continue
        ys = ys[ys >= (int(y0) + min_drop)]
        below_edge[ys, x] = 255

    below_edge = remove_small_components(below_edge, min_area=6)
    if cv2.countNonZero(below_edge) == 0:
        return fish_mask

    cand_pts = np.column_stack(np.where(below_edge > 0))
    anchor_y, anchor_x = cand_pts[int(np.argmax(cand_pts[:, 0]))]
    anchor = np.array([int(anchor_x), int(anchor_y)], dtype=np.int32)

    contour = contour_from_mask(fish_mask)
    if contour is None or len(contour) < 20:
        return fish_mask
    contour_pts = contour[:, 0, :].astype(np.int32)
    band = contour_pts[contour_pts[:, 1] >= (anchor[1] - int(TAIL_CORNER_RECOVER_ANCHOR_BAND_PX))]
    if len(band) < 2:
        return fish_mask

    span = int(max(8, TAIL_CORNER_RECOVER_ANCHOR_SPAN_PX))
    gap = int(max(0, TAIL_CORNER_RECOVER_ANCHOR_GAP_PX))

    left_pts = band[band[:, 0] <= (anchor[0] - gap)]
    right_pts = band[band[:, 0] >= (anchor[0] + gap)]
    if len(left_pts) == 0 or len(right_pts) == 0:
        return fish_mask

    left_target = anchor[0] - span
    right_target = anchor[0] + span

    left_score = left_pts[:, 1].astype(np.float32) - 0.8 * np.abs(left_pts[:, 0] - left_target)
    right_score = right_pts[:, 1].astype(np.float32) - 0.8 * np.abs(right_pts[:, 0] - right_target)

    left_anchor = left_pts[int(np.argmax(left_score))]
    right_anchor = right_pts[int(np.argmax(right_score))]

    if left_anchor[0] >= right_anchor[0]:
        return fish_mask

    poly = np.array([left_anchor, anchor, right_anchor], dtype=np.int32).reshape(-1, 1, 2)
    addition = np.zeros_like(fish_mask, dtype=np.uint8)
    cv2.fillPoly(addition, [poly], 255)
    addition = cv2.bitwise_and(addition, addition, mask=roi)
    addition = cv2.bitwise_and(addition, near_tail)
    addition = cv2.bitwise_and(addition, cv2.bitwise_not(fish_mask))

    if cv2.countNonZero(addition) == 0:
        return fish_mask

    fish_area = int(cv2.countNonZero(fish_mask))
    max_add = max(180, int(round(fish_area * float(TAIL_CORNER_RECOVER_MAX_ADD_FRAC))))
    if cv2.countNonZero(addition) > max_add:
        return fish_mask

    combined = cv2.bitwise_or(fish_mask, addition)

    # Use only nearby, touching support around the recovered tip so we pick up
    # a missing tail corner without stretching the whole tip farther down.
    local_roi = np.zeros_like(fish_mask, dtype=np.uint8)
    half_w = int(max(12, TAIL_CORNER_RECOVER_LOCAL_HALF_W_PX))
    up_px = int(max(12, TAIL_CORNER_RECOVER_LOCAL_UP_PX))
    down_px = int(max(2, TAIL_CORNER_RECOVER_LOCAL_DOWN_PX))
    cv2.rectangle(
        local_roi,
        (max(0, int(anchor[0]) - half_w), max(0, int(anchor[1]) - up_px)),
        (min(fish_mask.shape[1] - 1, int(anchor[0]) + half_w), min(fish_mask.shape[0] - 1, int(anchor[1]) + down_px)),
        255,
        thickness=cv2.FILLED,
    )

    touch_px = int(max(1, TAIL_CORNER_RECOVER_TOUCH_DILATE_PX))
    k_touch = np.ones((2 * touch_px + 1, 2 * touch_px + 1), np.uint8)
    touch_seed = cv2.dilate(combined, k_touch, iterations=1)
    touch_seed = cv2.bitwise_and(touch_seed, local_roi)

    corner_support = cv2.bitwise_and(supported, touch_seed)
    corner_support = remove_small_components(corner_support, min_area=int(max(1, TAIL_CORNER_RECOVER_LOCAL_MIN_AREA)))

    if cv2.countNonZero(corner_support) > 0:
        local_mask = cv2.bitwise_and(combined, local_roi)
        hull_pts = []
        for src in (local_mask, corner_support):
            cnts, _ = cv2.findContours(src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            for c in cnts:
                if len(c) > 0:
                    hull_pts.append(c)

        if hull_pts:
            local_hull = cv2.convexHull(np.concatenate(hull_pts, axis=0))
            local_hull_mask = contour_to_filled_mask(fish_mask.shape[0], fish_mask.shape[1], local_hull)
            local_hull_mask = cv2.bitwise_and(local_hull_mask, local_hull_mask, mask=local_roi)
            local_hull_mask = cv2.bitwise_and(local_hull_mask, near_tail)
            near_combined_px = int(max(1, TAIL_CORNER_RECOVER_NEAR_COMBINED_PX))
            k_near = np.ones((2 * near_combined_px + 1, 2 * near_combined_px + 1), np.uint8)
            near_combined = cv2.dilate(combined, k_near, iterations=1)
            local_hull_mask = cv2.bitwise_and(local_hull_mask, near_combined)

            current_contour = contour_from_mask(combined)
            if current_contour is not None and len(current_contour) > 0:
                current_tip_y = int(np.max(current_contour[:, 0, 1]))
                clip = np.zeros_like(local_hull_mask, dtype=np.uint8)
                cv2.rectangle(
                    clip,
                    (0, 0),
                    (fish_mask.shape[1] - 1, min(fish_mask.shape[0] - 1, current_tip_y + int(max(0, TAIL_CORNER_RECOVER_LOCAL_Y_PAD_PX)))),
                    255,
                    thickness=cv2.FILLED,
                )
                local_hull_mask = cv2.bitwise_and(local_hull_mask, clip)

            addition = cv2.bitwise_or(addition, cv2.bitwise_and(local_hull_mask, cv2.bitwise_not(fish_mask)))
            combined = cv2.bitwise_or(fish_mask, addition)

    out = combined
    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out


def local_band_refine_mask(img_rgb: np.ndarray, fish_mask: np.ndarray) -> np.ndarray:
    """
    Bigger change: refine the mask itself in a narrow band around the fish.
    This is localized, so it should help pick up missing fins/tail without
    creating a huge white halo.
    """
    if (not LOCAL_MASK_REFINE_ENABLE) or fish_mask is None or cv2.countNonZero(fish_mask) == 0:
        return fish_mask

    h, w = fish_mask.shape[:2]
    boosted = super_boost_lines_gray(img_rgb)
    grad = gradient_u8(boosted).astype(np.float32)
    canny = cv2.Canny(boosted, 5, 20)
    canny = cv2.dilate(canny, np.ones((3, 3), np.uint8), iterations=1)
    bg_dist = compute_background_distance(img_rgb, border_px=12)

    roi_fin = fin_texture_roi(img_rgb, base_mask=fish_mask)
    roi_fin = _dilate_mask(roi_fin, DISPLAY_FIN_ROI_DILATE_PX)

    roi_tail = tail_roi_mask(fish_mask, max(TAIL_ROI_FRAC, TAIL_RECOVER_FRAC))
    roi_tail = _dilate_mask(roi_tail, DISPLAY_TAIL_ROI_DILATE_PX)

    roi_tail_corner = _tail_corner_roi(fish_mask)
    roi_tail_corner = _dilate_mask(roi_tail_corner, 18)

    inner_px = int(max(1, LOCAL_MASK_INNER_ERODE_PX))
    k_inner = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * inner_px + 1, 2 * inner_px + 1))
    sure_fg = cv2.erode(fish_mask, k_inner, iterations=1)

    outer_px = int(max(1, LOCAL_MASK_OUTER_DILATE_PX))
    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * outer_px + 1, 2 * outer_px + 1))
    outer = cv2.dilate(fish_mask, k_outer, iterations=1)

    if roi_fin is not None:
        k_fin = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(LOCAL_MASK_OUTER_FIN_DILATE_PX) + 1, 2 * int(LOCAL_MASK_OUTER_FIN_DILATE_PX) + 1)
        )
        outer_fin = cv2.dilate(fish_mask, k_fin, iterations=1)
        outer = cv2.bitwise_or(outer, cv2.bitwise_and(outer_fin, roi_fin))

    if roi_tail is not None:
        k_tail = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(LOCAL_MASK_OUTER_TAIL_DILATE_PX) + 1, 2 * int(LOCAL_MASK_OUTER_TAIL_DILATE_PX) + 1)
        )
        outer_tail = cv2.dilate(fish_mask, k_tail, iterations=1)
        outer = cv2.bitwise_or(outer, cv2.bitwise_and(outer_tail, roi_tail))

    if roi_tail_corner is not None:
        k_tc = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (2 * int(LOCAL_MASK_OUTER_TAIL_DILATE_PX) + 1, 2 * int(LOCAL_MASK_OUTER_TAIL_DILATE_PX) + 1)
        )
        outer_tc = cv2.dilate(fish_mask, k_tc, iterations=1)
        outer = cv2.bitwise_or(outer, cv2.bitwise_and(outer_tc, roi_tail_corner))

    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)
    gc_mask[outer == 0] = cv2.GC_BGD
    gc_mask[sure_fg > 0] = cv2.GC_FGD
    gc_mask[fish_mask > 0] = cv2.GC_PR_FGD

    strong_bg = ((bg_dist < float(LOCAL_MASK_BG_THRESH)) & (grad < float(LOCAL_MASK_GRAD_THRESH)) & (canny == 0))
    gc_mask[strong_bg & (outer == 0)] = cv2.GC_BGD
    gc_mask[strong_bg & (outer > 0) & (sure_fg == 0)] = cv2.GC_PR_BGD

    edge_fg = ((bg_dist > float(LOCAL_MASK_BG_THRESH)) | (grad > float(LOCAL_MASK_EDGE_THRESH)) | (canny > 0))
    gc_mask[edge_fg & (outer > 0)] = np.where(
        gc_mask[edge_fg & (outer > 0)] == cv2.GC_BGD,
        cv2.GC_BGD,
        cv2.GC_PR_FGD,
    )

    if roi_fin is not None:
        fin_keep = (roi_fin > 0) & edge_fg
        gc_mask[fin_keep] = np.where(gc_mask[fin_keep] == cv2.GC_BGD, cv2.GC_BGD, cv2.GC_PR_FGD)

    if roi_tail is not None:
        tail_keep = (roi_tail > 0) & edge_fg
        gc_mask[tail_keep] = np.where(gc_mask[tail_keep] == cv2.GC_BGD, cv2.GC_BGD, cv2.GC_PR_FGD)

    if roi_tail_corner is not None:
        tail_corner_keep = (roi_tail_corner > 0) & edge_fg
        gc_mask[tail_corner_keep] = np.where(gc_mask[tail_corner_keep] == cv2.GC_BGD, cv2.GC_BGD, cv2.GC_PR_FGD)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    try:
        cv2.grabCut(
            img_rgb,
            gc_mask,
            None,
            bgd_model,
            fgd_model,
            int(max(1, LOCAL_MASK_REFINE_ITERS)),
            mode=cv2.GC_INIT_WITH_MASK,
        )
    except cv2.error:
        return fish_mask

    out = np.where(
        (gc_mask == cv2.GC_FGD) | (gc_mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    # keep anything already in the old mask to avoid regressions
    out = cv2.bitwise_or(out, fish_mask)

    kk_close = int(max(1, LOCAL_MASK_POST_CLOSE_K))
    kk_open = int(max(1, LOCAL_MASK_POST_OPEN_K))
    kk_close = kk_close + 1 if kk_close % 2 == 0 else kk_close
    kk_open = kk_open + 1 if kk_open % 2 == 0 else kk_open

    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk_close, kk_close))
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk_open, kk_open))

    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, k_close, iterations=1)
    out = cv2.morphologyEx(out, cv2.MORPH_OPEN, k_open, iterations=1)
    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out


def refine_contour_to_fish_edges(
    contour: np.ndarray,
    img_rgb: np.ndarray,
    fish_mask: np.ndarray,
) -> np.ndarray:
    if contour is None or len(contour) < 20:
        return contour

    pts = _resample_closed_contour(contour, CONTOUR_REFINE_STEP_PX)
    if pts is None or len(pts) < 20:
        return contour

    pts = _smooth_closed_polyline(
        pts,
        sigma=float(CONTOUR_REFINE_SMOOTH_SIGMA),
        passes=int(CONTOUR_REFINE_SMOOTH_PASSES),
    )

    score_map = _edge_score_map(img_rgb)
    h, w = score_map.shape[:2]

    m = (fish_mask > 0).astype(np.uint8)
    dist_in = cv2.distanceTransform(m, cv2.DIST_L2, 3).astype(np.float32)
    dist_out = cv2.distanceTransform(1 - m, cv2.DIST_L2, 3).astype(np.float32)
    sdf = dist_in - dist_out

    roi_tail = tail_roi_mask(fish_mask, max(TAIL_ROI_FRAC, TAIL_RECOVER_FRAC))
    roi_tail = _dilate_mask(roi_tail, DISPLAY_TAIL_ROI_DILATE_PX)

    roi_fin = fin_texture_roi(img_rgb, base_mask=fish_mask)
    roi_fin = _dilate_mask(roi_fin, DISPLAY_FIN_ROI_DILATE_PX)

    roi_tail_corner = _tail_corner_roi(fish_mask)
    roi_tail_corner = _dilate_mask(roi_tail_corner, 18)

    M = cv2.moments(fish_mask, binaryImage=True)
    if M["m00"] > 0:
        centroid = np.array([M["m10"] / M["m00"], M["m01"] / M["m00"]], dtype=np.float32)
    else:
        centroid = np.mean(pts, axis=0).astype(np.float32)

    n_pts = len(pts)
    snapped = pts.copy()

    for i in range(n_pts):
        p_prev = pts[(i - 1) % n_pts]
        p = pts[i]
        p_next = pts[(i + 1) % n_pts]

        tangent = p_next - p_prev
        tnorm = float(np.linalg.norm(tangent))
        if tnorm < 1e-6:
            continue
        tangent /= tnorm

        normal = np.array([-tangent[1], tangent[0]], dtype=np.float32)
        if float(np.dot(normal, p - centroid)) < 0:
            normal = -normal

        px = int(round(p[0]))
        py = int(round(p[1]))
        if px < 0 or px >= w or py < 0 or py >= h:
            continue

        out_px = int(CONTOUR_SEARCH_OUT_PX)
        bonus = 0.0

        if roi_fin is not None and roi_fin[py, px] > 0:
            out_px = max(out_px, int(CONTOUR_SEARCH_OUT_FIN_PX))
            bonus += float(CONTOUR_FIN_BONUS)

        if roi_tail is not None and roi_tail[py, px] > 0:
            out_px = max(out_px, int(CONTOUR_SEARCH_OUT_TAIL_PX))
            bonus += float(CONTOUR_TAIL_BONUS)

        if roi_tail_corner is not None and roi_tail_corner[py, px] > 0:
            out_px = max(out_px, int(CONTOUR_SEARCH_OUT_TAIL_PX))
            bonus += float(CONTOUR_TAIL_CORNER_BONUS)

        best_pt = p.copy()
        best_score = -1e18

        for t in np.linspace(-float(CONTOUR_SEARCH_IN_PX), float(out_px), int(CONTOUR_SEARCH_SAMPLES)):
            cand = p + normal * float(t)
            x = int(round(cand[0]))
            y = int(round(cand[1]))

            if x < 0 or x >= w or y < 0 or y >= h:
                continue

            local_score = float(score_map[y, x])
            local_bonus = bonus

            if roi_fin is not None and roi_fin[y, x] > 0:
                local_bonus += float(CONTOUR_FIN_BONUS) * 0.6
            if roi_tail is not None and roi_tail[y, x] > 0:
                local_bonus += float(CONTOUR_TAIL_BONUS) * 0.6
            if roi_tail_corner is not None and roi_tail_corner[y, x] > 0:
                local_bonus += float(CONTOUR_TAIL_CORNER_BONUS) * 0.8

            if local_score + local_bonus < float(CONTOUR_EDGE_MIN_SCORE):
                continue

            total = (
                local_score
                + local_bonus
                - float(CONTOUR_DIST_PENALTY) * abs(float(t))
                - float(CONTOUR_MASK_PULL) * abs(float(sdf[y, x]))
            )

            if total > best_score:
                best_score = total
                best_pt = cand

        move = best_pt - p
        move_len = float(np.linalg.norm(move))
        if move_len > float(CONTOUR_MAX_MOVE_PX):
            best_pt = p + move * (float(CONTOUR_MAX_MOVE_PX) / move_len)

        snapped[i] = best_pt

    snapped = _smooth_closed_polyline(
        snapped,
        sigma=float(DISPLAY_POST_SMOOTH_SIGMA),
        passes=int(DISPLAY_POST_SMOOTH_PASSES),
    )
    snapped = np.round(snapped).astype(np.int32).reshape(-1, 1, 2)
    return snapped


def preserve_tail_from_mask(
    display_mask: np.ndarray,
    source_mask: np.ndarray,
    img_rgb: np.ndarray | None = None,
) -> np.ndarray:
    """
    Prevent the final smoothed display contour from shaving off the tail tip/corner.
    We keep the smoothed outline globally, but locally union back reliable pixels
    from the original mask inside an expanded tail ROI, then rebuild the contour.
    """
    if (not TAIL_PRESERVE_ENABLE) or display_mask is None or source_mask is None:
        return display_mask
    if cv2.countNonZero(display_mask) == 0 or cv2.countNonZero(source_mask) == 0:
        return display_mask

    roi = tail_roi_mask(source_mask, TAIL_PRESERVE_ROI_FRAC)
    if roi is None:
        return display_mask

    corner_roi = _tail_corner_roi(source_mask)
    if corner_roi is not None:
        roi = cv2.bitwise_or(roi, corner_roi)

    roi = _dilate_mask(roi, TAIL_PRESERVE_DILATE_PX)

    src_tail = cv2.bitwise_and(source_mask, source_mask, mask=roi)
    disp_tail = cv2.bitwise_and(display_mask, display_mask, mask=roi)

    missing_tail = cv2.bitwise_and(src_tail, cv2.bitwise_not(disp_tail))
    if cv2.countNonZero(missing_tail) == 0:
        return display_mask

    if img_rgb is not None:
        boosted = super_boost_lines_gray(img_rgb)
        grad = gradient_u8(boosted)
        edge = cv2.Canny(boosted, 4, 18)
        edge = cv2.dilate(edge, np.ones((3, 3), np.uint8), iterations=1)

        guided = np.zeros_like(missing_tail)
        guided[(missing_tail > 0) & ((grad > 6) | (edge > 0))] = 255

        # Fall back to the original missing pixels if the edge cue is too weak.
        if cv2.countNonZero(guided) > 0:
            missing_tail = cv2.bitwise_or(missing_tail, guided)

    merged = cv2.bitwise_or(display_mask, missing_tail)

    kk = int(TAIL_PRESERVE_CLOSE_K)
    kk = kk + 1 if kk % 2 == 0 else kk
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk))
    merged_roi = cv2.bitwise_and(merged, merged, mask=roi)
    merged_roi = cv2.morphologyEx(merged_roi, cv2.MORPH_CLOSE, k, iterations=1)
    merged[roi > 0] = merged_roi[roi > 0]

    merged = keep_largest_connected_component(merged)
    merged = fill_mask_holes(merged)
    return merged


def _open_profile_1d(values: np.ndarray, kernel: int = 5) -> np.ndarray:
    """
    Remove short positive spikes from a 1D boundary profile.
    """
    if values is None or values.size == 0:
        return values

    kk = int(max(3, kernel))
    kk = kk + 1 if kk % 2 == 0 else kk
    half = kk // 2

    eroded = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        eroded[i] = int(np.min(values[lo:hi]))

    opened = np.empty_like(values)
    for i in range(len(values)):
        lo = max(0, i - half)
        hi = min(len(values), i + half + 1)
        opened[i] = int(np.max(eroded[lo:hi]))

    return opened


def _cap_profile_local_peaks(
    values: np.ndarray,
    shoulder_slack: int = 1,
    peak_tol: int = 1,
    passes: int = 2,
) -> np.ndarray:
    """
    Clip narrow one- and two-row outward spikes from a 1D right-edge profile.
    """
    if values is None or values.size < 3:
        return values

    capped = values.copy()
    for _ in range(int(max(1, passes))):
        prev = capped.copy()

        for i in range(1, len(capped) - 1):
            shoulder = int(max(capped[i - 1], capped[i + 1]))
            if capped[i] > shoulder + int(peak_tol):
                capped[i] = min(capped[i], shoulder + int(shoulder_slack))

        for i in range(1, len(capped) - 2):
            shoulder = int(max(capped[i - 1], capped[i + 2]))
            plateau = int(max(capped[i], capped[i + 1]))
            if plateau > shoulder + int(peak_tol):
                cap = shoulder + int(shoulder_slack)
                capped[i] = min(capped[i], cap)
                capped[i + 1] = min(capped[i + 1], cap)

        if np.array_equal(capped, prev):
            break

    return capped


def _cap_upper_connector_spur(
    values: np.ndarray,
    trim_rows: int = 30,
    lookahead_offset: int = 4,
    lookahead_span: int = 8,
    ref_percentile: float = 35.0,
    spur_tol: int = 3,
    slack: int = 1,
    passes: int = 2,
) -> np.ndarray:
    """
    Clip the upper recovered connector rows if they stick farther right than
    the extension profile immediately below them.
    """
    if values is None or values.size < (lookahead_offset + 2):
        return values

    capped = values.copy()
    last_i = min(int(trim_rows), len(capped) - (int(lookahead_offset) + 1))
    if last_i <= 0:
        return capped

    for _ in range(int(max(1, passes))):
        prev = capped.copy()
        for i in range(last_i):
            j0 = i + int(lookahead_offset)
            j1 = min(len(capped), j0 + int(lookahead_span))
            future = capped[j0:j1]
            if future.size == 0:
                continue

            ref = int(np.percentile(future, float(ref_percentile)))
            if capped[i] > ref + int(spur_tol):
                capped[i] = ref + int(slack)

        if np.array_equal(capped, prev):
            break

    return capped


def trim_tail_corner_right_bumps(
    base_mask: np.ndarray,
    recovered_mask: np.ndarray,
    tip_x: int,
    tip_y: int,
) -> np.ndarray:
    """
    Shave short outward bumps from the recovered lower-right tail corner
    without undoing the recovered corner itself.
    """
    if (
        base_mask is None
        or recovered_mask is None
        or cv2.countNonZero(base_mask) == 0
        or cv2.countNonZero(recovered_mask) == 0
    ):
        return recovered_mask

    added = cv2.bitwise_and(recovered_mask, cv2.bitwise_not(base_mask))
    if cv2.countNonZero(added) == 0:
        return recovered_mask

    band_px = 74
    head_rows = 8
    foot_rows = 6
    right_roi_left_pad = 8
    slack_px = 1

    y0 = max(0, int(tip_y) - band_px)
    ys: list[int] = []
    base_profile: list[int] = []
    right_profile: list[int] = []

    for y in range(y0, int(tip_y) + 1):
        rec_row = np.where(recovered_mask[y] > 0)[0]
        if len(rec_row) == 0:
            continue

        base_row = np.where(base_mask[y] > 0)[0]
        rec_right = int(rec_row.max())
        base_right = int(base_row.max()) if len(base_row) > 0 else -1

        if rec_right <= base_right:
            continue

        ys.append(y)
        base_profile.append(base_right)
        right_profile.append(rec_right)

    if len(right_profile) < 6:
        return recovered_mask

    base_arr = np.array(base_profile, dtype=np.int32)
    profile = np.array(right_profile, dtype=np.int32)
    ext_profile = np.maximum(0, profile - base_arr)

    ext_opened = _open_profile_1d(ext_profile, kernel=9)
    allowed_ext = np.minimum(ext_profile, ext_opened + 1)
    allowed_ext = np.minimum(
        allowed_ext,
        _cap_upper_connector_spur(
            allowed_ext,
            trim_rows=30,
            lookahead_offset=4,
            lookahead_span=8,
            ref_percentile=35.0,
            spur_tol=3,
            slack=1,
            passes=2,
        ),
    )

    lower_rows = min(28, len(profile))
    lower_start = len(profile) - lower_rows
    lower_profile = profile[lower_start:]
    opened = _open_profile_1d(lower_profile, kernel=5)

    head_n = min(head_rows, len(opened))
    foot_n = min(foot_rows, len(opened))
    head_x = int(np.percentile(opened[:head_n], 25))
    foot_x = int(np.percentile(opened[-foot_n:], 50))
    linear_cap = np.rint(np.linspace(head_x, foot_x, len(lower_profile))).astype(np.int32)

    lower_allowed = np.minimum(lower_profile, np.minimum(opened + slack_px, linear_cap + slack_px))
    lower_allowed = np.minimum(
        lower_allowed,
        _cap_profile_local_peaks(
            lower_allowed,
            shoulder_slack=1,
            peak_tol=1,
            passes=2,
        ),
    )
    allowed_ext[lower_start:] = np.minimum(
        allowed_ext[lower_start:],
        np.maximum(0, lower_allowed - base_arr[lower_start:]),
    )

    trimmed_added = added.copy()
    cut_left = max(0, int(tip_x) - right_roi_left_pad)
    for idx, y in enumerate(ys):
        allowed_right = int(base_arr[idx] + allowed_ext[idx])
        cut_x = max(cut_left, allowed_right + 1)
        trimmed_added[y, cut_x:] = 0

    if cv2.countNonZero(trimmed_added) == cv2.countNonZero(added):
        return recovered_mask

    trimmed_added = remove_small_components(trimmed_added, min_area=3)
    trimmed = cv2.bitwise_or(base_mask, trimmed_added)
    trimmed = keep_largest_connected_component(trimmed)
    trimmed = fill_mask_holes(trimmed)
    return trimmed


def recover_tail_corner_right_flank_from_image(
    img_rgb: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """
    Recover the clipped upper-right tail flank by fitting the visible right
    flank near the tail tip, then extending the current edge toward the
    strongest nearby image support on that fitted side only.
    """
    if img_rgb is None or mask is None or cv2.countNonZero(mask) == 0:
        return mask

    contour = contour_from_mask(mask)
    if contour is None or len(contour) < 40:
        return mask

    pts = contour[:, 0, :].astype(np.int32)
    tip_idx = int(np.argmax(pts[:, 1]))
    tip = pts[tip_idx]
    tip_y = int(tip[1])

    def extract_arc(step: int, max_points: int = 100, max_y_drop: int = 64) -> np.ndarray:
        n = len(pts)
        arc = []
        i = tip_idx
        for _ in range(max_points):
            p = pts[i % n]
            arc.append(p)
            if tip_y - int(p[1]) >= max_y_drop:
                break
            i += step
        return np.array(arc, dtype=np.int32)

    forward_arc = extract_arc(+1)
    backward_arc = extract_arc(-1)
    right_arc = forward_arc if float(np.mean(forward_arc[:, 0])) > float(np.mean(backward_arc[:, 0])) else backward_arc
    fit_pts = right_arc[8:min(len(right_arc), 52)].astype(np.float32)
    if len(fit_pts) < 8:
        return mask

    vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(
        fit_pts.reshape(-1, 1, 2),
        cv2.DIST_L2,
        0,
        0.01,
        0.01
    ).reshape(-1)]
    if abs(vy) < 1e-4:
        return mask

    boost = super_boost_lines_gray(img_rgb)
    grad = gradient_u8(boost).astype(np.float32)
    bg = compute_background_distance(img_rgb, border_px=12).astype(np.float32)
    edge = cv2.Canny(boost, 4, 18).astype(np.float32)
    score = grad + 0.8 * bg + 0.5 * edge

    add = np.zeros_like(mask, dtype=np.uint8)
    outer_pts: list[tuple[int, int]] = []
    y_top = max(0, tip_y - 60)
    for y in range(y_top, tip_y + 1):
        row = np.where(mask[y] > 0)[0]
        if len(row) == 0:
            continue

        edge_x = int(row.max())
        pred_x = int(round(x0 + (y - y0) * (vx / vy)))
        x_from = max(edge_x + 1, pred_x - 22)
        x_to = min(mask.shape[1] - 1, pred_x + 22)
        if x_to <= edge_x:
            continue

        row_score = score[y, x_from:x_to + 1]
        if row_score.size == 0:
            continue

        best_rel = int(np.argmax(row_score))
        if float(row_score[best_rel]) < 12.0:
            continue

        best_x = x_from + best_rel
        target_x = min(edge_x + 36, best_x)
        if target_x > edge_x:
            add[y, edge_x:target_x + 1] = 255
            outer_pts.append((target_x, y))

    roi = np.zeros_like(mask, dtype=np.uint8)
    cv2.rectangle(
        roi,
        (max(0, int(tip[0]) - 92), max(0, tip_y - 74)),
        (min(mask.shape[1] - 1, int(tip[0]) + 100), tip_y),
        255,
        thickness=cv2.FILLED,
    )
    add = cv2.bitwise_and(add, roi)
    add = cv2.bitwise_and(add, cv2.bitwise_not(mask))
    add = remove_small_components(add, min_area=4)

    # Keep only the added patch that remains attached to the existing flank.
    touch = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    add = cv2.bitwise_and(add, touch)

    # After we recover sparse row-wise support, explicitly fill the wedge
    # between those recovered outer points and the visible right flank so the
    # whole clipped corner comes back, not just a few scanlines.
    if len(outer_pts) >= 8:
        right_seg = right_arc[(right_arc[:, 1] >= y_top) & (right_arc[:, 1] <= tip_y)]
        right_seg = right_seg[np.argsort(right_seg[:, 1])]

        outer = np.array(outer_pts, dtype=np.int32)
        ys_unique = np.unique(outer[:, 1])
        outer_reduced = []
        for yy in ys_unique:
            xs = outer[outer[:, 1] == yy, 0]
            outer_reduced.append([int(xs.max()), int(yy)])
        outer = np.array(outer_reduced, dtype=np.int32)
        outer = outer[np.argsort(outer[:, 1])[::-1]]

        if len(right_seg) >= 6 and len(outer) >= 6:
            poly = np.vstack([right_seg, outer]).reshape(-1, 1, 2)
            wedge = np.zeros_like(mask, dtype=np.uint8)
            cv2.fillPoly(wedge, [poly], 255)
            wedge = cv2.bitwise_and(wedge, roi)
            wedge = cv2.bitwise_and(wedge, cv2.bitwise_not(mask))
            add = cv2.bitwise_or(add, wedge)

    if cv2.countNonZero(add) == 0:
        return mask

    out = cv2.bitwise_or(mask, add)
    out = trim_tail_corner_right_bumps(
        base_mask=mask,
        recovered_mask=out,
        tip_x=int(tip[0]),
        tip_y=tip_y,
    )
    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out


def recover_tail_corner_fitline_gap(
    mask: np.ndarray,
    contour_pts: np.ndarray,
    tip_y: int,
    anchor_x: int,
    roi: np.ndarray,
) -> np.ndarray:
    """
    Recover a clipped tail-corner flank by fitting short lines to the contour
    on each side of the bottom tip, then filling the gap between the current
    edge and the stronger fitted flank. This targets the missing side only.
    """
    if mask is None or contour_pts is None or len(contour_pts) < 20 or roi is None:
        return np.zeros_like(mask, dtype=np.uint8)

    n = len(contour_pts)
    tip_candidates = np.where(contour_pts[:, 1] == int(tip_y))[0]
    if len(tip_candidates) > 0:
        tip_idx = int(tip_candidates[np.argmin(np.abs(contour_pts[tip_candidates, 0] - int(anchor_x)))])
    else:
        tip_idx = int(np.argmax(contour_pts[:, 1]))

    def circular_segment(start_idx: int, end_idx: int) -> np.ndarray:
        seg = []
        i = int(start_idx) % n
        end_idx = int(end_idx) % n
        for _ in range(n + 1):
            seg.append(contour_pts[i])
            if i == end_idx:
                break
            i = (i + 1) % n
        return np.array(seg, dtype=np.int32)

    forward_arc = circular_segment(tip_idx, tip_idx + 56)
    backward_arc = circular_segment(tip_idx - 56, tip_idx)

    # Label the two arcs by side, then build one candidate gap per side.
    arcs = []
    if len(forward_arc) >= 6:
        arcs.append((forward_arc, +1 if float(np.mean(forward_arc[:, 0])) >= float(anchor_x) else -1))
    if len(backward_arc) >= 6:
        arcs.append((backward_arc, +1 if float(np.mean(backward_arc[:, 0])) >= float(anchor_x) else -1))

    best_gap = np.zeros_like(mask, dtype=np.uint8)
    best_area = 0
    fit_skip = 3
    fit_count = 22
    y_band = 30
    max_offset_px = 12

    for arc, sign in arcs:
        fit_pts = arc[fit_skip:min(len(arc), fit_skip + fit_count)].astype(np.float32)
        if len(fit_pts) < 6:
            continue

        vx, vy, x0, y0 = [float(v) for v in cv2.fitLine(
            fit_pts.reshape(-1, 1, 2),
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        ).reshape(-1)]
        if abs(vy) < 1e-4:
            continue

        candidate = np.zeros_like(mask, dtype=np.uint8)
        y_top = max(0, int(tip_y) - y_band)
        for y in range(y_top, int(tip_y) + 1):
            row_mask = (mask[y] > 0) & (roi[y] > 0)
            if not np.any(row_mask):
                continue

            xs_row = np.where(row_mask)[0]
            edge_x = int(xs_row.max()) if sign > 0 else int(xs_row.min())
            fit_x = int(round(x0 + (y - y0) * (vx / vy)))

            if sign > 0:
                target = min(edge_x + max_offset_px, fit_x)
                if target > edge_x:
                    candidate[y, edge_x:target + 1] = 255
            else:
                target = max(edge_x - max_offset_px, fit_x)
                if target < edge_x:
                    candidate[y, target:edge_x + 1] = 255

        candidate = cv2.bitwise_and(candidate, roi)
        candidate = cv2.bitwise_and(candidate, cv2.bitwise_not(mask))
        candidate = remove_small_components(candidate, min_area=4)
        area = int(cv2.countNonZero(candidate))
        if area > best_area:
            best_gap = candidate
            best_area = area

    return best_gap


def fill_tail_corner_local_hull(mask: np.ndarray) -> np.ndarray:
    """
    Fill a tiny missing tail corner using only the local tail-tip geometry.
    The fill is clipped to the current tip height so it broadens the corner
    instead of lengthening the tail.
    """
    if (not TAIL_CORNER_FILL_ENABLE) or mask is None or cv2.countNonZero(mask) == 0:
        return mask

    contour = contour_from_mask(mask)
    if contour is None or len(contour) < 20:
        return mask

    pts = contour[:, 0, :].astype(np.int32)
    tip_y = int(np.max(pts[:, 1]))
    band_px = int(max(6, TAIL_CORNER_FILL_BOTTOM_BAND_PX))
    bottom_pts = pts[pts[:, 1] >= (tip_y - band_px)]
    if len(bottom_pts) == 0:
        return mask

    anchor_x = int(np.median(bottom_pts[:, 0]))

    roi = np.zeros_like(mask, dtype=np.uint8)
    half_w = int(max(18, TAIL_CORNER_FILL_HALF_W_PX))
    up_px = int(max(18, TAIL_CORNER_FILL_UP_PX))
    cv2.rectangle(
        roi,
        (max(0, anchor_x - half_w), max(0, tip_y - up_px)),
        (min(mask.shape[1] - 1, anchor_x + half_w), tip_y),
        255,
        thickness=cv2.FILLED,
    )

    corner_roi = _tail_corner_roi(mask)
    if corner_roi is not None:
        corner_roi = _dilate_mask(corner_roi, TAIL_CORNER_FILL_ROI_DILATE_PX)
        if corner_roi is not None:
            roi = cv2.bitwise_and(roi, corner_roi)

    if cv2.countNonZero(roi) == 0:
        return mask

    local = cv2.bitwise_and(mask, roi)
    cnts, _ = cv2.findContours(local, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return mask

    hull = cv2.convexHull(np.concatenate(cnts, axis=0))
    hull_mask = contour_to_filled_mask(mask.shape[0], mask.shape[1], hull)
    hull_mask = cv2.bitwise_and(hull_mask, roi)

    fitline_gap = recover_tail_corner_fitline_gap(
        mask=mask,
        contour_pts=pts,
        tip_y=tip_y,
        anchor_x=anchor_x,
        roi=roi,
    )
    hull_mask = cv2.bitwise_or(hull_mask, fitline_gap)

    addition = cv2.bitwise_and(hull_mask, cv2.bitwise_not(mask))
    if cv2.countNonZero(addition) == 0:
        return mask
    if cv2.countNonZero(addition) > int(max(20, TAIL_CORNER_FILL_MAX_ADD_PX)):
        return mask

    out = cv2.bitwise_or(mask, addition)
    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out

def compute_final_fish_mask(img_rgb: np.ndarray, img_rgba: np.ndarray | None) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    img_area = float(h * w)

    _, base_hsv = compute_fish_mask_hsv(img_rgb)
    base = base_hsv

    alpha = compute_fish_mask_alpha(img_rgba) if img_rgba is not None else None
    if alpha is not None:
        a_area = float(cv2.countNonZero(alpha))
        a_frac = a_area / img_area if img_area > 0 else 0.0
        if MIN_FISH_AREA_FRAC <= a_frac <= MAX_FISH_AREA_FRAC:
            base = alpha

    fish_area = float(cv2.countNonZero(base))
    frac = fish_area / img_area if img_area > 0 else 0.0
    if frac < MIN_FISH_AREA_FRAC or frac > MAX_FISH_AREA_FRAC:
        base = compute_fish_mask_edge_fallback(img_rgb, alpha_mask=alpha)

    base = keep_largest_connected_component(base)
    base = fill_mask_holes(base)

    if ENABLE_FIN_GROW:
        base = fin_grow_pass(img_rgb, base, FIN_BAND_FRAC_1, pass_no=1)
        base = fin_grow_pass(img_rgb, base, FIN_BAND_FRAC_2, pass_no=2)

    protect_roi = None
    if PROTECT_ROI_ENABLE:
        rois = []
        if TAIL_ROI_ENABLE:
            roi_tail = tail_roi_mask(base, TAIL_ROI_FRAC)
            if roi_tail is not None:
                rois.append(roi_tail)
        roi_fin = fin_texture_roi(img_rgb, base_mask=base)
        if roi_fin is not None:
            rois.append(roi_fin)
        if rois:
            protect_roi = rois[0]
            for r in rois[1:]:
                protect_roi = cv2.bitwise_or(protect_roi, r)

    base = peel_padding_by_gradient(img_rgb, base, protect_roi=protect_roi)
    base = tighten_then_restore_fins(img_rgb, base, protect_roi=protect_roi)

    base = keep_largest_connected_component(base)
    base = fill_mask_holes(base)
    return base


def fin_grow_pass(img_rgb: np.ndarray, fish_mask: np.ndarray, band_frac: float, pass_no: int = 1) -> np.ndarray:
    if fish_mask is None or cv2.countNonZero(fish_mask) == 0:
        return fish_mask

    h, w = fish_mask.shape[:2]
    band_px = int(max(14, band_frac * min(h, w)))
    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    dil = cv2.dilate(fish_mask, k_outer, iterations=1)
    band = cv2.bitwise_and(dil, cv2.bitwise_not(fish_mask))

    near_k = NEAR_EDGE_DILATE if pass_no == 1 else min(NEAR_EDGE_DILATE + 6, 27)
    near_edge = cv2.dilate(fish_mask, np.ones((near_k, near_k), np.uint8), iterations=1)
    band = cv2.bitwise_and(band, band, mask=near_edge)

    boosted = super_boost_lines_gray(img_rgb)

    e1 = 10 if pass_no == 1 else 6
    e2 = 35 if pass_no == 1 else 24
    edges = cv2.Canny(boosted, e1, e2)
    edges = cv2.bitwise_and(edges, edges, mask=band)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)

    candidate = edges.copy()

    if TAIL_ROI_ENABLE:
        roi = tail_roi_mask(fish_mask, TAIL_ROI_FRAC)
        if roi is not None:
            band_tail = cv2.bitwise_and(band, band, mask=roi)
            tail_edges = cv2.Canny(boosted, int(TAIL_CANNY1), int(TAIL_CANNY2))
            tail_edges = cv2.bitwise_and(tail_edges, tail_edges, mask=band_tail)
            tail_edges = cv2.dilate(
                tail_edges, np.ones((3, 3), np.uint8),
                iterations=1 + int(TAIL_GROW_EXTRA_ITERS)
            )
            candidate = cv2.bitwise_or(candidate, tail_edges)

    union = cv2.bitwise_or(fish_mask, candidate)
    _, labels = cv2.connectedComponents((union > 0).astype(np.uint8), connectivity=8)

    keep = np.zeros_like(union)
    fish_labels = np.unique(labels[fish_mask > 0])
    for lab in fish_labels:
        if lab == 0:
            continue
        keep[labels == lab] = 255

    keep = keep_largest_connected_component(keep)
    keep = fill_mask_holes(keep)
    return keep


def peel_padding_by_gradient(img_rgb: np.ndarray, fish_mask: np.ndarray, protect_roi: np.ndarray | None = None) -> np.ndarray:
    if not PEEL_ENABLE:
        return fish_mask
    if fish_mask is None or cv2.countNonZero(fish_mask) == 0:
        return fish_mask

    boosted = super_boost_lines_gray(img_rgb)
    g = gradient_u8(boosted)
    out = fish_mask.copy()

    edge_dil = None
    if EDGE_GUIDED_ENABLE:
        edge_map = cv2.Canny(boosted, 8, 28)
        d = int(max(1, EDGE_GUIDED_DILATE_PX))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * d + 1, 2 * d + 1))
        edge_dil = cv2.dilate(edge_map, k, iterations=1)

    for _ in range(int(PEEL_ITERS)):
        ring_px = int(max(1, PEEL_RING_PX))
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * ring_px + 1, 2 * ring_px + 1))
        er = cv2.erode(out, k, iterations=1)
        ring = cv2.bitwise_and(out, cv2.bitwise_not(er))

        base_thr = int(PEEL_GRAD_THR)
        if PROTECT_ROI_ENABLE and protect_roi is not None:
            roi_thr = base_thr + int(PEEL_GRAD_THR_ROI_DELTA)
            pr = (protect_roi > 0)
            remove_low = ((g < base_thr) & (~pr)) | ((g < roi_thr) & pr)
        else:
            remove_low = (g < base_thr)

        remove_low = remove_low.astype(np.uint8) * 255
        remove_low = cv2.bitwise_and(remove_low, remove_low, mask=ring)

        if edge_dil is not None:
            near_edge = (edge_dil > 0)
            dist = cv2.distanceTransform((~near_edge).astype(np.uint8), cv2.DIST_L2, 3)
            far = (dist > float(EDGE_GUIDED_MAX_DIST))

            remove_far = (far & (g < int(PEEL_GRAD_THR + EDGE_GUIDED_GRAD_BONUS))).astype(np.uint8) * 255
            remove_far = cv2.bitwise_and(remove_far, remove_far, mask=ring)
            remove = cv2.bitwise_or(remove_low, remove_far)
        else:
            remove = remove_low

        candidate = out.copy()
        candidate[remove > 0] = 0

        if cv2.countNonZero(candidate) >= 0.90 * cv2.countNonZero(out):
            out = keep_largest_connected_component(candidate)
            out = fill_mask_holes(out)
        else:
            break

    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out


def tighten_then_restore_fins(img_rgb: np.ndarray, fish_mask: np.ndarray, protect_roi: np.ndarray | None = None) -> np.ndarray:
    if not TIGHTEN_ENABLE:
        return fish_mask
    if fish_mask is None or cv2.countNonZero(fish_mask) == 0:
        return fish_mask

    e = int(max(0, TIGHTEN_ERODE_PX))
    tight = fish_mask.copy()
    if e > 0:
        k_e = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
        tight = cv2.erode(tight, k_e, iterations=1)
    if PROTECT_ROI_ENABLE and protect_roi is not None and int(TIGHTEN_ERODE_PX_ROI) <= 0:
        tight[protect_roi > 0] = fish_mask[protect_roi > 0]

    band_px = int(max(8, TIGHTEN_RESTORE_BAND_PX))
    k_outer = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * band_px + 1, 2 * band_px + 1))
    dil = cv2.dilate(tight, k_outer, iterations=1)
    band = cv2.bitwise_and(dil, cv2.bitwise_not(tight))

    boosted = super_boost_lines_gray(img_rgb)

    edge_map = cv2.Canny(boosted, 8, 28)
    edge_map = cv2.dilate(edge_map, np.ones((3, 3), np.uint8), iterations=1)

    grad = gradient_u8(boosted)
    _, edge_ok_g = cv2.threshold(grad, int(TIGHTEN_EDGE_THR), 255, cv2.THRESH_BINARY)
    edge_ok = cv2.bitwise_or(edge_ok_g, edge_map)
    edge_ok = cv2.bitwise_and(edge_ok, edge_ok, mask=band)

    out = tight.copy()
    for _ in range(3):
        grow = cv2.dilate(out, np.ones((3, 3), np.uint8), iterations=1)
        add = cv2.bitwise_and(grow, edge_ok)
        out = cv2.bitwise_or(out, add)

    if PROTECT_ROI_ENABLE and protect_roi is not None:
        out[protect_roi > 0] = fish_mask[protect_roi > 0]

    out = keep_largest_connected_component(out)
    out = fill_mask_holes(out)
    return out


def compute_bleeding_metrics(img_rgb: np.ndarray, fish_mask: np.ndarray, hsv: np.ndarray):
    total_fish_pixel_area = int(cv2.countNonZero(fish_mask))

    _, _, v = cv2.split(hsv)

    lower1 = np.array([0, 80, 50], dtype=np.uint8)
    upper1 = np.array([10, 255, 255], dtype=np.uint8)
    lower2 = np.array([170, 80, 50], dtype=np.uint8)
    upper2 = np.array([180, 255, 255], dtype=np.uint8)

    red_mask1 = cv2.inRange(hsv, lower1, upper1)
    red_mask2 = cv2.inRange(hsv, lower2, upper2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    red_mask_in_fish = cv2.bitwise_and(red_mask, red_mask, mask=fish_mask)
    red_pixel_area = int(cv2.countNonZero(red_mask_in_fish))

    if red_pixel_area > 0:
        mean_red_pixel_intensity = float(np.mean(v[red_mask_in_fish > 0]))
    else:
        mean_red_pixel_intensity = 0.0

    integrated_density = float(red_pixel_area) * mean_red_pixel_intensity
    bleeding_index = integrated_density / float(total_fish_pixel_area) if total_fish_pixel_area > 0 else 0.0

    return {
        "total_fish_pixel_area": total_fish_pixel_area,
        "red_pixel_area": red_pixel_area,
        "mean_red_pixel_intensity": mean_red_pixel_intensity,
        "integrated_density": integrated_density,
        "bleeding_index": bleeding_index,
        "red_mask_in_fish": red_mask_in_fish,
    }


def smooth_red_mask(red_mask_in_fish: np.ndarray, img_rgb: np.ndarray | None = None) -> np.ndarray:
    if red_mask_in_fish is None or red_mask_in_fish.size == 0 or cv2.countNonZero(red_mask_in_fish) == 0:
        return red_mask_in_fish

    mask = (red_mask_in_fish > 0).astype(np.uint8) * 255
    mask = remove_small_components(mask, min_area=40)

    if img_rgb is not None:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)

        lower1 = np.array([0, 110, 60], dtype=np.uint8)
        upper1 = np.array([12, 255, 255], dtype=np.uint8)
        lower2 = np.array([168, 110, 60], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        strong_red = cv2.inRange(hsv, lower1, upper1)
        strong_red2 = cv2.inRange(hsv, lower2, upper2)
        strong_red = cv2.bitwise_or(strong_red, strong_red2)

        mask = cv2.bitwise_and(mask, strong_red)

    mask = fill_mask_holes(mask)

    if img_rgb is not None:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        grad = gradient_u8(gray)
        edge_keep = (grad > 25).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, cv2.bitwise_or(mask, edge_keep))

    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, ke, iterations=1)
    mask = sdf_smooth_mask(mask, sigma=2.0, iters=1, level=0.05)

    if img_rgb is not None:
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        lower1 = np.array([0, 100, 50], dtype=np.uint8)
        upper1 = np.array([12, 255, 255], dtype=np.uint8)
        lower2 = np.array([168, 100, 50], dtype=np.uint8)
        upper2 = np.array([180, 255, 255], dtype=np.uint8)

        strong_red = cv2.inRange(hsv, lower1, upper1)
        strong_red2 = cv2.inRange(hsv, lower2, upper2)
        strong_red = cv2.bitwise_or(strong_red, strong_red2)

        mask = cv2.bitwise_and(mask, strong_red)

    mask = fill_mask_holes(mask)
    mask = keep_largest_connected_component(mask)
    return mask


def make_edge_outline_mask(binary_mask: np.ndarray, thickness: int = 2) -> np.ndarray:
    edges = cv2.Canny(binary_mask, 50, 150)
    if thickness > 1:
        k = np.ones((thickness, thickness), np.uint8)
        edges = cv2.dilate(edges, k, iterations=1)
    return edges


def draw_bleed_edge_on_image(traced_rgb: np.ndarray, edge_mask: np.ndarray) -> np.ndarray:
    traced_rgb[edge_mask > 0] = (255, 0, 0)
    return traced_rgb


@app.route("/", methods=["GET", "POST"])
def index():
    output_image = None
    metrics = None
    fish_mask_image = None
    red_mask_image = None

    if request.method == "POST":
        file = request.files.get("image")
        if file is None or file.filename == "":
            return "No image uploaded", 400
        if not allowed_file(file.filename):
            return "Unsupported file type", 400

        try:
            pil_img = Image.open(file.stream)
            pil_rgba = pil_img.convert("RGBA")
            img_rgba_full = np.array(pil_rgba)
            img_rgb_full = img_rgba_full[:, :, :3].copy()
        except Exception:
            return "Invalid image file", 400

        img_rgb_proc, scale = resize_for_processing(img_rgb_full, WORK_MAX_DIM)
        if scale != 1.0:
            new_w = img_rgb_proc.shape[1]
            new_h = img_rgb_proc.shape[0]
            img_rgba_proc = cv2.resize(img_rgba_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_rgba_proc = img_rgba_full

        fish_mask_proc = compute_final_fish_mask(img_rgb_proc, img_rgba_proc)

        if scale != 1.0:
            fish_mask_full = cv2.resize(
                fish_mask_proc,
                (img_rgb_full.shape[1], img_rgb_full.shape[0]),
                interpolation=cv2.INTER_NEAREST
            )
        else:
            fish_mask_full = fish_mask_proc

        fish_mask_full = keep_largest_connected_component(fish_mask_full)
        fish_mask_full = fill_mask_holes(fish_mask_full)

        fish_mask_full = tail_recover_fullres(img_rgb_full, fish_mask_full)

        if LOCAL_MASK_REFINE_ENABLE:
            fish_mask_full = local_band_refine_mask(img_rgb_full, fish_mask_full)

        fish_mask_full = recover_tail_corner_hull(img_rgb_full, fish_mask_full)

        fish_mask_full = keep_largest_connected_component(fish_mask_full)
        fish_mask_full = fill_mask_holes(fish_mask_full)

        if BOUNDARY_SMOOTH_ENABLE:
            fish_mask_full = smooth_mask_boundary(
                fish_mask_full, k=BOUNDARY_SMOOTH_K, iters=BOUNDARY_SMOOTH_ITERS
            )

        if SDF_SMOOTH_ENABLE:
            fish_mask_full = sdf_smooth_mask(
                fish_mask_full, sigma=SDF_SIGMA, iters=SDF_ITERS, level=SDF_LEVEL_STAGE_A
            )

            if FISH_TIGHTEN_ENABLE and int(FISH_TIGHTEN_ERODE_PX) > 0:
                e = int(FISH_TIGHTEN_ERODE_PX)
                ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
                fish_mask_full = cv2.erode(fish_mask_full, ke, iterations=1)
                fish_mask_full = sdf_smooth_mask(
                    fish_mask_full, sigma=SDF_SIGMA, iters=1, level=SDF_LEVEL_STAGE_B
                )

        fish_mask_full = keep_largest_connected_component(fish_mask_full)
        fish_mask_full = fill_mask_holes(fish_mask_full)

        fish_mask_full = recover_tail_corner_right_flank_from_image(
            img_rgb=img_rgb_full,
            mask=fish_mask_full,
        )
        fish_mask_full = keep_largest_connected_component(fish_mask_full)
        fish_mask_full = fill_mask_holes(fish_mask_full)

        best_contour = contour_from_mask(fish_mask_full)

        # Smooth the final contour so the displayed outline is continuous.
        if best_contour is not None and OUTLINE_SMOOTH_ENABLE:
            best_contour = smooth_contour_highres(
                best_contour,
                step_px=SMOOTH_TARGET_STEP_PX,
                sigma=SMOOTH_SIGMA,
                passes=SMOOTH_PASSES
            )

        if best_contour is not None:
            fish_mask_display = contour_to_filled_mask(
                img_rgb_full.shape[0], img_rgb_full.shape[1], best_contour
            )
            fish_mask_display = keep_largest_connected_component(fish_mask_display)
            fish_mask_display = fill_mask_holes(fish_mask_display)
            fish_mask_display = recover_tail_corner_right_flank_from_image(
                img_rgb=img_rgb_full,
                mask=fish_mask_display,
            )
            best_contour = contour_from_mask(fish_mask_display)
        else:
            fish_mask_display = fish_mask_full

        hsv_full = cv2.cvtColor(img_rgb_full, cv2.COLOR_RGB2HSV)

        traced = img_rgb_full.copy()
        if best_contour is not None:
            cv2.drawContours(
                traced, [best_contour], -1, (0, 0, 0),
                thickness=FISH_OUTLINE_THICKNESS,
                lineType=cv2.LINE_AA
            )
        else:
            fish_edges = make_edge_outline_mask(
                fish_mask_display,
                thickness=max(2, FISH_OUTLINE_THICKNESS // 3)
            )
            traced[fish_edges > 0] = (0, 0, 0)

        metrics_full = compute_bleeding_metrics(
            img_rgb=img_rgb_full,
            fish_mask=fish_mask_display,
            hsv=hsv_full
        )

        red_clean = smooth_red_mask(metrics_full["red_mask_in_fish"], img_rgb_full)

        bleed_edge = make_edge_outline_mask(red_clean, thickness=BLEED_EDGE_THICKNESS)
        traced = draw_bleed_edge_on_image(traced, bleed_edge)

        uid = uuid.uuid4().hex
        filename = f"{uid}.png"
        Image.fromarray(traced).save(os.path.join(OUTPUT_DIR, filename))
        output_image = filename

        fish_mask_filename = f"{uid}_fish_mask.png"
        Image.fromarray(fish_mask_display).save(os.path.join(OUTPUT_DIR, fish_mask_filename))
        fish_mask_image = fish_mask_filename

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "5000")))
    args = parser.parse_args()
    app.run(host="0.0.0.0", port=args.port, debug=True)
