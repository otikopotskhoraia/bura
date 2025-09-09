import cv2 as cv
import numpy as np
from .templates import suit_templates
from .trump import detect_trump


def find_trump_card(img):
    """Search the screenshot for the trump card.

    Args:
        img: BGR screenshot image.

    Returns:
        dict with keys 'rank', 'suit', 'conf', and 'bbox'. 'bbox' is a
        (x, y, w, h) tuple in pixels or None if no candidate region was found.
    """
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    candidates = []
    for tmpl in suit_templates:
        t_gray = cv.cvtColor(tmpl["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.8)
        for (x, y) in zip(xs, ys):
            candidates.append({"x": int(x), "y": int(y)})
    if not candidates:
        return {"rank": None, "suit": None, "conf": 0, "bbox": None}
    h, w = img.shape[:2]
    cand = max(candidates, key=lambda c: c["x"])
    size = 140
    x0 = max(0, min(w - size, cand["x"] - 40))
    y0 = max(0, min(h - size, cand["y"] - 40))
    roi = img[y0 : y0 + size, x0 : x0 + size]
    best = {"rank": None, "suit": None, "conf": 0}
    rotations = [None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
    for rot in rotations:
        rimg = cv.rotate(roi, rot) if rot is not None else roi
        res = detect_trump(rimg)
        if res["conf"] > best["conf"]:
            best = res
    return {**best, "bbox": (x0, y0, size, size)}
