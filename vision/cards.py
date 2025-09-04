# vision/cards.py
import cv2 as cv
from math import hypot
from .config import THRESH


def detect_card_in_slot(bgr_slot_mat, rank_tmps, suit_tmps):
    """Identify the card present in a slot image.

    Template matching works best on the small rank/suit glyph typically found
    in the upper-left corner of a card.  Restrict the search area to this
    region instead of using the entire slot image, which can contain noise and
    lead to poor matches.
    """
    gray = cv.cvtColor(bgr_slot_mat, cv.COLOR_BGR2GRAY)

    # Focus on the glyph region near the top-left corner of the card.  Use a
    # rectangle relative to the overall slot size to accommodate different card
    # resolutions while still ignoring the card artwork.
    glyph_rect = (
        5,
        5,
        max(20, int(gray.shape[1] * 0.45)),
        max(25, int(gray.shape[0] * 0.35)),
    )
    x, y, w, h = glyph_rect
    glyph = gray[y : y + h, x : x + w]

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
