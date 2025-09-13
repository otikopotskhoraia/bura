import os
from .config import ROI, THRESH
from .detect import map_roi
from .find_score_me import match_score as _match_score_me
from .find_score_opp import match_score as _match_score_opp


def _extract_score(img, roi_def, matcher, pad, base_dir):
    """Crop ROI from image and run the provided matcher."""
    h, w = img.shape[:2]
    x, y, rw, rh = map_roi(roi_def, w, h, w, h)
    x = max(0, x - pad)
    y = max(0, y - pad)
    rw = min(rw + 2 * pad, w - x)
    rh = min(rh + 2 * pad, h - y)
    roi = img[y : y + rh, x : x + rw]
    return matcher(roi, base_dir) or 0


def find_scores(img, base_dir=None):
    """Return detected scores for the player and opponent.

    Args:
        img: BGR screenshot image.
        base_dir: Optional base directory for loading templates.

    Returns:
        dict with integer scores for keys 'me' and 'opp'.  If detection fails,
        the score defaults to 0.
    """
    base_dir = base_dir or os.path.dirname(__file__)
    pad = int(THRESH.get("ocrPad", 4))
    score_me = _extract_score(img, ROI["scoreMe"], _match_score_me, pad, base_dir)
    score_opp = _extract_score(img, ROI["scoreOpp"], _match_score_opp, pad, base_dir)
    return {"me": score_me, "opp": score_opp}
