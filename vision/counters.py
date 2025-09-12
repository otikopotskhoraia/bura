# vision/counters.py
import cv2 as cv
import numpy as np
import pytesseract
import re
from .config import THRESH
from .templates import counter_templates, score_templates


def read_green_number(bgr_roi_mat):
    hsv = cv.cvtColor(bgr_roi_mat, cv.COLOR_BGR2HSV)
    low = np.array(THRESH["greenHSV"]["low"], dtype=np.uint8)
    high = np.array(THRESH["greenHSV"]["high"], dtype=np.uint8)
    mask = cv.inRange(hsv, low, high)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    inv = cv.bitwise_not(clean)
    up = cv.resize(inv, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    try:
        text = pytesseract.image_to_string(up, config=config)
    except pytesseract.TesseractNotFoundError:
        text = ""
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0


def match_counter(bgr_roi_mat):
    """Match the counter ROI against known templates.

    Args:
        bgr_roi_mat: cropped counter image.

    Returns:
        int: detected counter value or 0 if no match meets the threshold.
    """
    gray = cv.cvtColor(bgr_roi_mat, cv.COLOR_BGR2GRAY)
    best = {"name": None, "score": 0}
    for tmpl in counter_templates:
        t_gray = cv.cvtColor(tmpl["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(res)
        if max_val > best["score"]:
            best = {"name": tmpl["name"], "score": float(max_val)}
    if best["score"] >= THRESH["matchMinScore"] and best["name"] is not None:
        try:
            return int(best["name"])
        except ValueError:
            return 0
    return 0


def match_score(bgr_roi_mat):
    """Match the score ROI against known digit templates.

    Args:
        bgr_roi_mat: cropped score image.

    Returns:
        int | None: detected score value or None if no match meets the threshold.
    """
    gray = cv.cvtColor(bgr_roi_mat, cv.COLOR_BGR2GRAY)
    best = {"name": None, "score": 0}
    for tmpl in score_templates:
        t_gray = cv.cvtColor(tmpl["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(res)
        if max_val > best["score"]:
            best = {"name": tmpl["name"], "score": float(max_val)}
    if best["score"] >= THRESH["matchMinScore"] and best["name"] is not None:
        try:
            return int(best["name"])
        except ValueError:
            return None
    return None
