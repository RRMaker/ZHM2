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
FISH_OUTLINE_THICKNESS = 10
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
TIGHTEN_ERODE_PX = 1
TIGHTEN_RESTORE_BAND_PX = 22
TIGHTEN_EDGE_THR = 18

# -----------------------------
# OUTLINE smoothing / snapping
# -----------------------------
OUTLINE_SMOOTH_ENABLE = True

SMOOTH_TARGET_STEP_PX = 0.85
SMOOTH_SIGMA = 3.6
SMOOTH_PASSES = 5

# Snapping can re-introduce bumps; keep OFF for smooth curve.
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
SDF_SIGMA = 6.0
SDF_ITERS = 1

# Stage A (slight expand helps tail) then Stage B tighten without losing smoothness
SDF_LEVEL_STAGE_A = -0.35   # was looser; less expansion now
FISH_TIGHTEN_ENABLE = True
FISH_TIGHTEN_ERODE_PX = 1   # tighten fit
SDF_LEVEL_STAGE_B = 0.25    # shrink a bit after erosion, keeps smoothness via re-SDF

# -----------------------------
# Full-res tail recovery pass
# -----------------------------
TAIL_RECOVER_ENABLE = True
TAIL_RECOVER_FRAC = 0.28
TAIL_RECOVER_DILATE_ITERS = 5
TAIL_RECOVER_CANNY1 = 2
TAIL_RECOVER_CANNY2 = 18
TAIL_RECOVER_S_MAX = 140
TAIL_RECOVER_V_MIN = 20
TAIL_RECOVER_EDGE_DILATE = 3

# -----------------------------
# Small boundary smoothing on mask
# -----------------------------
BOUNDARY_SMOOTH_ENABLE = True
BOUNDARY_SMOOTH_K = 3
BOUNDARY_SMOOTH_ITERS = 1

# -----------------------------
# NEW: Red mask smoothing / better fit
# -----------------------------
RED_MIN_COMPONENT_AREA = 25       # remove tiny red specks
RED_SDF_SMOOTH_ENABLE = True
RED_SDF_SIGMA = 2.2               # small so it hugs bleed region
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
    """Fill interior holes."""
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
    """Signed distance smoothing -> continuous rounded outline."""
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
        if x := cv2.boundingRect(c):
            pass
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


# --- ONLY SHOWING THE MODIFIED FUNCTION ---
# Replace your existing smooth_red_mask function with THIS one

def smooth_red_mask(red_mask_in_fish: np.ndarray, img_rgb: np.ndarray | None = None) -> np.ndarray:
    """
    Stronger red mask fitting:
    - Remove specks
    - Re-project to strong HSV red only
    - Edge-aware tightening
    - Signed distance smoothing (keeps smooth curve)
    - Final clamp to real red pixels
    """

    if red_mask_in_fish is None or red_mask_in_fish.size == 0 or cv2.countNonZero(red_mask_in_fish) == 0:
        return red_mask_in_fish

    mask = (red_mask_in_fish > 0).astype(np.uint8) * 255

    # Remove tiny specks
    mask = remove_small_components(mask, min_area=40)

    # ---- STRICT RED RE-PROJECTION (important fix) ----
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

    # Fill holes before smoothing
    mask = fill_mask_holes(mask)

    # ---- EDGE-AWARE TIGHTENING ----
    if img_rgb is not None:
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        grad = gradient_u8(gray)

        # Only keep red where gradient is reasonably strong
        edge_keep = (grad > 25).astype(np.uint8) * 255
        mask = cv2.bitwise_and(mask, cv2.bitwise_or(mask, edge_keep))

    # Slight erosion to tighten boundary
    ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.erode(mask, ke, iterations=1)

    # ---- SDF SMOOTH (keeps curve continuous) ----
    mask = sdf_smooth_mask(mask, sigma=2.0, iters=1, level=0.05)

    # Final clamp to real red again (prevents orange bleed)
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

        # process at working size
        img_rgb_proc, scale = resize_for_processing(img_rgb_full, WORK_MAX_DIM)
        if scale != 1.0:
            new_w = img_rgb_proc.shape[1]
            new_h = img_rgb_proc.shape[0]
            img_rgba_proc = cv2.resize(img_rgba_full, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
            img_rgba_proc = img_rgba_full

        fish_mask_proc = compute_final_fish_mask(img_rgb_proc, img_rgba_proc)

        # scale mask back to full-res
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

        # Recover missing tail tip first
        fish_mask_full = tail_recover_fullres(img_rgb_full, fish_mask_full)

        # gentle boundary cleanup
        if BOUNDARY_SMOOTH_ENABLE:
            fish_mask_full = smooth_mask_boundary(
                fish_mask_full, k=BOUNDARY_SMOOTH_K, iters=BOUNDARY_SMOOTH_ITERS
            )

        # Stage A: SDF smooth (keeps curve), mild expansion (less loose than before)
        if SDF_SMOOTH_ENABLE:
            fish_mask_full = sdf_smooth_mask(
                fish_mask_full, sigma=SDF_SIGMA, iters=SDF_ITERS, level=SDF_LEVEL_STAGE_A
            )

            # Tighten fit WITHOUT losing smoothness:
            # erode a touch, then run SDF again (this keeps the same smooth curve quality)
            if FISH_TIGHTEN_ENABLE and int(FISH_TIGHTEN_ERODE_PX) > 0:
                e = int(FISH_TIGHTEN_ERODE_PX)
                ke = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * e + 1, 2 * e + 1))
                fish_mask_full = cv2.erode(fish_mask_full, ke, iterations=1)
                fish_mask_full = sdf_smooth_mask(
                    fish_mask_full, sigma=SDF_SIGMA, iters=1, level=SDF_LEVEL_STAGE_B
                )

        # Contour from smoothed/tightened mask
        best_contour = contour_from_mask(fish_mask_full)

        if best_contour is not None and OUTLINE_SMOOTH_ENABLE:
            best_contour = smooth_contour_highres(
                best_contour,
                step_px=SMOOTH_TARGET_STEP_PX,
                sigma=SMOOTH_SIGMA,
                passes=SMOOTH_PASSES
            )
            # keep snap OFF by default

        # Display mask from contour so outline matches mask
        if best_contour is not None:
            fish_mask_display = contour_to_filled_mask(
                img_rgb_full.shape[0], img_rgb_full.shape[1], best_contour
            )
            fish_mask_display = keep_largest_connected_component(fish_mask_display)
            fish_mask_display = fill_mask_holes(fish_mask_display)
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
            fish_edges = make_edge_outline_mask(fish_mask_display, thickness=max(2, FISH_OUTLINE_THICKNESS // 3))
            traced[fish_edges > 0] = (0, 0, 0)

        metrics_full = compute_bleeding_metrics(img_rgb=img_rgb_full, fish_mask=fish_mask_display, hsv=hsv_full)

        # NEW: smoother red mask that hugs the bleed region better
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