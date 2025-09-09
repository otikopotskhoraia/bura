# vision/counters.py
import cv2 as cv
import numpy as np
import pytesseract
import re
from .config import THRESH


def read_green_number(bgr_roi_mat):
    hsv = cv.cvtColor(bgr_roi_mat, cv.COLOR_BGR2HSV)
    low = np.array(THRESH["greenHSV"]["low"], dtype=np.uint8)
    high = np.array(THRESH["greenHSV"]["high"], dtype=np.uint8)
    mask = cv.inRange(hsv, low, high)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    clean = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=1)
    inv = cv.bitwise_not(clean)
    up = cv.resize(inv, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    try:
        text = pytesseract.image_to_string(up, config=config)
    except pytesseract.TesseractNotFoundError:
        text = ""
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0
