# vision/trump.py
import cv2 as cv
from .config import THRESH
from .templates import rank_templates, suit_templates


def _best_match(gray_src, tmpl):
    res = cv.matchTemplate(gray_src, tmpl, cv.TM_CCOEFF_NORMED)
    return res.max()


def detect_trump(bgr_roi_mat):
    gray = cv.cvtColor(bgr_roi_mat, cv.COLOR_BGR2GRAY)
    best_rank = {"name": None, "score": 0}
    for r in rank_templates:
        t_gray = cv.cvtColor(r["mat"], cv.COLOR_BGR2GRAY)
        score = _best_match(gray, t_gray)
        if score > best_rank["score"]:
            best_rank = {"name": r["name"], "score": score}
    best_suit = {"name": None, "score": 0}
    for s in suit_templates:
        t_gray = cv.cvtColor(s["mat"], cv.COLOR_BGR2GRAY)
        score = _best_match(gray, t_gray)
        if score > best_suit["score"]:
            best_suit = {"name": s["name"], "score": score}
    if (
        best_rank["score"] >= THRESH["matchMinScore"]
        and best_suit["score"] >= THRESH["matchMinScore"]
    ):
        return {
            "rank": best_rank["name"],
            "suit": best_suit["name"],
            "conf": min(best_rank["score"], best_suit["score"]),
        }
    return {"rank": None, "suit": None, "conf": 0}
