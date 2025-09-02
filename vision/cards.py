# vision/cards.py
import cv2 as cv
from .config import THRESH


def detect_card_in_slot(bgr_slot_mat, rank_tmps, suit_tmps):
    gray = cv.cvtColor(bgr_slot_mat, cv.COLOR_BGR2GRAY)
    glyph_rect = (
        5,
        5,
        max(20, int(gray.shape[1] * 0.45)),
        max(25, int(gray.shape[0] * 0.35)),
    )
    x, y, w, h = glyph_rect
    glyph = gray[y:y+h, x:x+w]

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
    if best_rank["loc"] and best_rank["shape"]:
        rx, ry = best_rank["loc"]
        rw, rh = best_rank["shape"]
        cv.rectangle(
            annotated,
            (x + rx, y + ry),
            (x + rx + rw, y + ry + rh),
            (255, 0, 0),
            1,
        )
    if best_suit["loc"] and best_suit["shape"]:
        sx, sy = best_suit["loc"]
        sw, sh = best_suit["shape"]
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
        result.update(
            {
                "card": f"{best_rank['name']}-{best_suit['name']}",
                "conf": min(best_rank["score"], best_suit["score"]),
            }
        )
    return result
