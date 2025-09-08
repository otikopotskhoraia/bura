# vision/cards.py
import cv2 as cv
import numpy as np
from math import hypot
from .config import THRESH


def detect_card_in_slot(bgr_slot_mat, rank_tmps, suit_tmps):
    """Identify the card present in a slot image.

    Rather than returning the best rank and suit independently, this version
    searches for pairs of rank and suit glyphs that appear close together.  A
    whole card is recognised only if a rank and suit candidate are within
    ``THRESH["glyphMaxDist"]`` pixels of one another.
    """

    gray = cv.cvtColor(bgr_slot_mat, cv.COLOR_BGR2GRAY)
    h, w = gray.shape
    annotated = bgr_slot_mat.copy()
    cv.rectangle(annotated, (0, 0), (w, h), (0, 255, 0), 1)

    rank_candidates = []
    for r in rank_tmps:
        t_gray = cv.cvtColor(r["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= THRESH["matchMinScore"])
        for (x, y) in zip(xs, ys):
            rank_candidates.append({
                "name": r["name"],
                "score": float(res[y, x]),
                "loc": (int(x), int(y)),
                "shape": t_gray.shape[::-1],
            })

    suit_candidates = []
    for s in suit_tmps:
        t_gray = cv.cvtColor(s["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= THRESH["matchMinScore"])
        for (x, y) in zip(xs, ys):
            suit_candidates.append({
                "name": s["name"],
                "score": float(res[y, x]),
                "loc": (int(x), int(y)),
                "shape": t_gray.shape[::-1],
            })

    best = None
    for r in rank_candidates:
        rx, ry = r["loc"]
        rw, rh = r["shape"]
        r_center = (rx + rw / 2, ry + rh / 2)
        for s in suit_candidates:
            sx, sy = s["loc"]
            sw, sh = s["shape"]
            s_center = (sx + sw / 2, sy + sh / 2)
            if hypot(r_center[0] - s_center[0], r_center[1] - s_center[1]) <= THRESH.get("glyphMaxDist", float("inf")):
                conf = min(r["score"], s["score"])
                if not best or conf > best["conf"]:
                    best = {"rank": r, "suit": s, "conf": conf}

    result = {"card": None, "conf": 0, "debug_img": annotated}
    if best and best["conf"] >= THRESH["matchMinScore"]:
        r = best["rank"]
        s = best["suit"]
        rx, ry = r["loc"]
        rw, rh = r["shape"]
        sx, sy = s["loc"]
        sw, sh = s["shape"]
        cv.rectangle(annotated, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 1)
        cv.rectangle(annotated, (sx, sy), (sx + sw, sy + sh), (0, 0, 255), 1)
        result.update({
            "card": f"{r['name']}-{s['name']}",
            "conf": best["conf"],
        })

    return result
