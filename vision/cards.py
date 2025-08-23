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

    best_rank = {"name": None, "score": 0}
    for r in rank_tmps:
        t_gray = cv.cvtColor(r["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(glyph, t_gray, cv.TM_CCOEFF_NORMED)
        score = res.max()
        if score > best_rank["score"]:
            best_rank = {"name": r["name"], "score": score}

    best_suit = {"name": None, "score": 0}
    for s in suit_tmps:
        t_gray = cv.cvtColor(s["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(glyph, t_gray, cv.TM_CCOEFF_NORMED)
        score = res.max()
        if score > best_suit["score"]:
            best_suit = {"name": s["name"], "score": score}

    if (
        best_rank["score"] >= THRESH["matchMinScore"]
        and best_suit["score"] >= THRESH["matchMinScore"]
    ):
        return {
            "card": f"{best_rank['name']}-{best_suit['name']}",
            "conf": min(best_rank["score"], best_suit["score"]),
        }
    return {"card": None, "conf": 0}
