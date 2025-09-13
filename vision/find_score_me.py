# vision/find_score_me.py
import os
import sys
import re
import glob
import cv2 as cv
import numpy as np
import pytesseract

from .config import ROI, THRESH
from .detect import map_roi


# ---------------- Utilities ----------------

def _ensure_dir(p):
    os.makedirs(p, exist_ok=True)
    return p

def _read_gray(path):
    img = cv.imread(path, cv.IMREAD_UNCHANGED)
    if img is None:
        return None
    if img.ndim == 2:
        return img
    if img.shape[2] == 4:
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def _binarize_clean(gray):
    # Otsu -> binary (white glyph on black), light denoise to drop borders
    _, bin0 = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    er = cv.erode(bin0, cv.getStructuringElement(cv.MORPH_RECT, (2, 2)), iterations=1)
    clean = cv.morphologyEx(er, cv.MORPH_CLOSE, cv.getStructuringElement(cv.MORPH_RECT, (3, 3)), iterations=1)
    return clean  # white glyph on black

def _largest_blob(binary_w_on_b):
    cnts, _ = cv.findContours(binary_w_on_b, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None, None
    H, W = binary_w_on_b.shape[:2]
    best, best_area = None, -1
    for c in cnts:
        x, y, w, h = cv.boundingRect(c)
        area = w * h
        if area > best_area:
            best_area = area
            best = (x, y, w, h)
    if best is None:
        return None, None
    x, y, w, h = best
    m = 2
    x = max(0, x - m); y = max(0, y - m)
    w = min(W - x, w + 2 * m); h = min(H - y, h + 2 * m)
    crop = binary_w_on_b[y:y+h, x:x+w]
    return crop, best

def _prep_for_tess(digit_white_on_black):
    # Tesseract prefers black text on white
    digit_black_on_white = cv.bitwise_not(digit_white_on_black)
    k = cv.getStructuringElement(cv.MORPH_RECT, (2, 2))
    digit_black_on_white = cv.morphologyEx(digit_black_on_white, cv.MORPH_CLOSE, k, iterations=1)
    digit_black_on_white = cv.dilate(digit_black_on_white, k, iterations=1)
    digit_black_on_white = cv.copyMakeBorder(digit_black_on_white, 10, 10, 10, 10,
                                             borderType=cv.BORDER_CONSTANT, value=255)
    digit_black_on_white = cv.resize(digit_black_on_white, None, fx=3.0, fy=3.0, interpolation=cv.INTER_CUBIC)
    return digit_black_on_white

def _ocr_digit(digit_black_on_white):
    for psm in (10, 8, 7):  # single char → word → line
        cfg = f"--oem 3 --psm {psm} -c tessedit_char_whitelist=0123456789"
        try:
            txt = pytesseract.image_to_string(digit_black_on_white, config=cfg)
        except pytesseract.TesseractNotFoundError:
            return None
        m = re.search(r"\d", (txt or "").strip())
        if m:
            return int(m.group())
    return None

# ---------------- Template loading ----------------

def _load_templates(base_dir):
    tmpl_dir = os.path.join(base_dir, "templates", "score")
    paths = sorted(glob.glob(os.path.join(tmpl_dir, "*.png")))
    tmpls = []
    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        m = re.match(r"^\s*(\d+)", name)
        if not m:
            continue
        val = int(m.group(1))
        g = _read_gray(p)
        if g is None:
            continue
        tmpls.append({"val": val, "gray": g, "path": p})
    return tmpls, tmpl_dir

# ---------------- Fallback B: sliding template match ----------------

def _sliding_template_fallback(roi_gray, templates, debug_dir=None):
    """
    Multi-scale sliding correlation on roi_gray (binary variants).
    Returns (val or None, best_score).
    """
    # Build ROI variants (both polarities)
    _, bin_otsu = cv.threshold(roi_gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    variants = [
        ("wb", bin_otsu),  # white glyph on black
        ("bw", cv.bitwise_not(bin_otsu)),
    ]

    best_val, best_score = None, -1.0
    for t in templates:
        # Prepare template in both polarities and a few scales
        _, tbin = cv.threshold(t["gray"], 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        tpairs = [("wb", tbin), ("bw", cv.bitwise_not(tbin))]
        for vname, vr in variants:
            hR, wR = vr.shape[:2]
            if hR < 12 or wR < 8:
                continue
            for tname, timg in tpairs:
                for scale in (0.6, 0.75, 0.9, 1.0, 1.15, 1.3):
                    h, w = timg.shape[:2]
                    nh, nw = max(1, int(round(h * scale))), max(1, int(round(w * scale)))
                    ts = cv.resize(timg, (nw, nh), interpolation=cv.INTER_CUBIC)
                    if hR < nh or wR < nw:
                        continue
                    res = cv.matchTemplate(vr, ts, cv.TM_CCOEFF_NORMED)
                    _, mx, _, _ = cv.minMaxLoc(res)
                    if mx > best_score:
                        best_score = float(mx)
                        best_val = t["val"]
    if debug_dir:
        with open(os.path.join(debug_dir, "template_sliding_debug.txt"), "w", encoding="utf-8") as f:
            f.write(f"best_score={best_score:.4f}, best_val={best_val}\n")
    min_ok = float(THRESH.get("scoreMatchMin", 0.78))
    return (best_val if best_score >= min_ok else None, best_score)

# ---------------- Fallback C: contour (shape) matching ----------------

def _contour_from_binary(binary_w_on_b):
    # Find the largest contour as the digit outline
    cnts, _ = cv.findContours(binary_w_on_b, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    # choose by contour area
    c = max(cnts, key=cv.contourArea)
    return c

def _contour_match_digit(digit_white_on_black, templates, debug_dir=None):
    """
    Scale/rotation/polarity-invariant shape comparison using Hu moments.
    Uses cv.matchShapes on largest contours.
    Returns (val or None, best_dist).
    """
    dc = _contour_from_binary(digit_white_on_black)
    if dc is None or cv.contourArea(dc) < 20:
        return None, 1e9

    # Build template contours
    entries = []
    for t in templates:
        g = t["gray"]
        _, tb = cv.threshold(g, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        # Ensure glyph is white on black for contour extraction
        if np.count_nonzero(tb) < np.count_nonzero(255 - tb):
            tb = 255 - tb
        tc = _contour_from_binary(tb)
        if tc is None or cv.contourArea(tc) < 20:
            continue
        entries.append((t["val"], tc))

    if not entries:
        return None, 1e9

    # Compare shapes (lower distance is better)
    scored = []
    for val, tc in entries:
        d = cv.matchShapes(dc, tc, cv.CONTOURS_MATCH_I1, 0.0)
        scored.append((val, float(d)))
    scored.sort(key=lambda x: x[1])  # ascending distance

    if debug_dir:
        with open(os.path.join(debug_dir, "template_contour_debug.txt"), "w", encoding="utf-8") as f:
            for v, d in scored:
                f.write(f"{v}: dist={d:.6f}\n")

    best_val, best_dist = scored[0]
    # A loose acceptance threshold; typical good matches are < 0.2
    max_ok = float(THRESH.get("scoreShapeMax", 0.35))
    return (best_val if best_dist <= max_ok else None, best_dist)

# ---------------- Main matching pipeline ----------------

def match_score(bgr_roi_mat, base_dir, debug_dir=None):
    """
    Returns int digit on success, or None on failure.
    Order:
      1) OCR (Tesseract on black-on-white)
      2) Contour (shape) matching (size/polarity independent)
      3) Sliding template match (multi-scale, both polarities)
    """
    if bgr_roi_mat is None or bgr_roi_mat.size == 0:
        return None

    gray = cv.cvtColor(bgr_roi_mat, cv.COLOR_BGR2GRAY)
    bin_clean = _binarize_clean(gray)
    digit_wb, bbox = _largest_blob(bin_clean)
    if digit_wb is None:
        return None

    if debug_dir:
        os.makedirs(debug_dir, exist_ok=True)
        cv.imwrite(os.path.join(debug_dir, "roi_gray.png"), gray)
        cv.imwrite(os.path.join(debug_dir, "roi_bin_clean.png"), bin_clean)
        dbg = bgr_roi_mat.copy()
        x, y, w, h = bbox
        cv.rectangle(dbg, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv.imwrite(os.path.join(debug_dir, "roi_digit_box.png"), dbg)
        cv.imwrite(os.path.join(debug_dir, "digit_white_on_black.png"), digit_wb)

    # 1) OCR
    tess_img = _prep_for_tess(digit_wb)
    if debug_dir:
        cv.imwrite(os.path.join(debug_dir, "digit_for_tess.png"), tess_img)
    v = _ocr_digit(tess_img)
    if v is not None:
        return v

    # Load templates (and log)
    templates, tmpl_dir = _load_templates(os.path.dirname(__file__))
    if debug_dir:
        with open(os.path.join(debug_dir, "templates_loaded.txt"), "w", encoding="utf-8") as f:
            f.write(f"template_dir={tmpl_dir}\n")
            for t in templates:
                f.write(f"{t['val']} -> {t['path']}\n")

    # 2) Contour (shape) matching
    if templates:
        v2, d2 = _contour_match_digit(digit_wb, templates, debug_dir=debug_dir)
        if v2 is not None:
            return v2

        # 3) Sliding template fallback (multi-scale)
        v3, s3 = _sliding_template_fallback(gray, templates, debug_dir=debug_dir)
        if v3 is not None:
            return v3

    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_score_me <screenshot.png>")
        return

    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return

    shot_h, shot_w = img.shape[:2]
    x, y, w, h = map_roi(ROI["scoreMe"], shot_w, shot_h, shot_w, shot_h)

    pad = int(THRESH.get("ocrPad", 4))
    x = max(0, x - pad); y = max(0, y - pad)
    w = min(w + 2 * pad, shot_w - x); h = min(h + 2 * pad, shot_h - y)

    roi = img[y:y+h, x:x+w]

    debug_dir = _ensure_dir("score_me_debug")
    cv.imwrite(os.path.join(debug_dir, "roi_raw.png"), roi)

    base_dir = os.path.dirname(__file__)
    val = match_score(roi, base_dir, debug_dir=debug_dir)

    count = 0 if val is None else val
    print(f"scoreMe count: {count}")

    out = img.copy()
    cv.rectangle(out, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(out, str(count), (x, max(0, y - 10)),
               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Save both (compat + folder)
    cv.imwrite("score_me_debug.png", out)
    cv.imwrite(os.path.join(debug_dir, "annotated.png"), out)
    print("Annotated screenshot saved to score_me_debug.png and score_me_debug/annotated.png")


if __name__ == "__main__":
    main()
