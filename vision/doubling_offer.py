import cv2 as cv
from .config import ROI
from .detect import map_roi


def detect_doubling_popup(img):
    """Return True if a black popup with text is present near the table center.

    Args:
        img: BGR screenshot image.

    Returns:
        bool indicating whether the doubling popup is detected.
    """
    shot_h, shot_w = img.shape[:2]
    if "tableCenter" not in ROI:
        return False

    x, y, w, h = map_roi(ROI["tableCenter"], shot_w, shot_h, shot_w, shot_h)
    roi = img[y : y + h, x : x + w]

    gray = cv.cvtColor(roi, cv.COLOR_BGR2GRAY)
    # Portion of dark pixels (background)
    _, dark = cv.threshold(gray, 50, 255, cv.THRESH_BINARY_INV)
    dark_ratio = cv.countNonZero(dark) / float(w * h)
    # Portion of very bright pixels (text)
    _, light = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    light_ratio = cv.countNonZero(light) / float(w * h)

    return dark_ratio > 0.6 and light_ratio > 0.001
