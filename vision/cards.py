# vision/cards.py
import cv2 as cv
from math import hypot
from .config import THRESH


def detect_card_in_slot(bgr_slot_mat, rank_tmps, suit_tmps):
    """Identify the card present in a slot image.

    Previously the template matching was restricted to a small glyph region in
    the upper-left corner of the slot.  To allow matching on the entire slot,
    use the full grayscale image instead of a cropped rectangle.
    """
    gray = cv.cvtColor(bgr_slot_mat, cv.COLOR_BGR2GRAY)
    x, y = 0, 0
    h, w = gray.shape
    glyph = gray

    best_rank = {"name": None, "score": 0, "loc": None, "shape": None}
    for r in rank_tmps:
        t_gray = cv.cvtColor(r["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(glyph, t_gray, cv.TM_CCOEFF_NORMED)
        _, score, _, loc = cv.minMaxLoc(res)
        if score > best_rank["score"]:
            best_rank = {
                "name": r["name"],
                "score": score,
                "loc": loc,
                "shape": t_gray.shape[::-1],
            }

    best_suit = {"name": None, "score": 0, "loc": None, "shape": None}
    for s in suit_tmps:
        t_gray = cv.cvtColor(s["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(glyph, t_gray, cv.TM_CCOEFF_NORMED)
        _, score, _, loc = cv.minMaxLoc(res)
        if score > best_suit["score"]:
            best_suit = {
                "name": s["name"],
                "score": score,
                "loc": loc,
                "shape": t_gray.shape[::-1],
            }

    annotated = bgr_slot_mat.copy()
    cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 1)
    rank_center = None
    if best_rank["loc"] and best_rank["shape"]:
        rx, ry = best_rank["loc"]
        rw, rh = best_rank["shape"]
        rank_center = (rx + rw / 2, ry + rh / 2)
        cv.rectangle(
            annotated,
            (x + rx, y + ry),
            (x + rx + rw, y + ry + rh),
            (255, 0, 0),
            1,
        )
    suit_center = None
    if best_suit["loc"] and best_suit["shape"]:
        sx, sy = best_suit["loc"]
        sw, sh = best_suit["shape"]
        suit_center = (sx + sw / 2, sy + sh / 2)
        cv.rectangle(
            annotated,
            (x + sx, y + sy),
            (x + sx + sw, y + sy + sh),
            (0, 0, 255),
            1,
        )

    result = {"card": None, "conf": 0, "debug_img": annotated}
    if (
        best_rank["score"] >= THRESH["matchMinScore"]
        and best_suit["score"] >= THRESH["matchMinScore"]
    ):
        too_far = False
        if rank_center and suit_center:
            dist = hypot(rank_center[0] - suit_center[0], rank_center[1] - suit_center[1])
            if dist > THRESH.get("glyphMaxDist", float("inf")):
                too_far = True
        if not too_far:
            result.update(
                {
                    "card": f"{best_rank['name']}-{best_suit['name']}",
                    "conf": min(best_rank["score"], best_suit["score"]),
                }
            )
    return result
