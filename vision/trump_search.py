import cv2 as cv
from .config import ROI
from .detect import map_roi
from .trump import detect_trump

# Minimum confidence required to accept a detected trump card
MIN_TRUMP_CONF = 0.9


def find_trump_card(img):
    """Detect the trump card from a fixed slot on the screenshot.

    Args:
        img: BGR screenshot image.

    Returns:
        dict with keys 'rank', 'suit', 'conf', and 'bbox'.  'bbox' is a
        (x, y, w, h) tuple in pixels of the region that was inspected.  If the
        ``trumpSlot`` ROI is not defined, ``bbox`` will be ``None`` and the
        rank and suit will also be ``None``.
    """
    h, w = img.shape[:2]
    if "trumpSlot" not in ROI:
        return {"rank": None, "suit": None, "conf": 0, "bbox": None}

    x, y, rw, rh = map_roi(ROI["trumpSlot"], w, h, 1920, 1080)
    roi = img[y : y + rh, x : x + rw]

    best = {"rank": None, "suit": None, "conf": 0}
    rotations = [None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]
    for rot in rotations:
        rimg = cv.rotate(roi, rot) if rot is not None else roi
        res = detect_trump(rimg)
        if res["conf"] > best["conf"]:
            best = res

    result = {**best, "bbox": (x, y, rw, rh)}
    if (
        result["conf"] < MIN_TRUMP_CONF
        or not result["rank"]
        or not result["suit"]
    ):
        result["rank"] = None
        result["suit"] = None
    return result
