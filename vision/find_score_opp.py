import os
import sys
import re
import cv2 as cv
import pytesseract
from .config import ROI, THRESH
from .detect import map_roi

def match_score(bgr_roi_mat):
    """OCR the white score numbers from the provided ROI."""
    gray = cv.cvtColor(bgr_roi_mat, cv.COLOR_BGR2GRAY)
    _, thresh = cv.threshold(gray, 200, 255, cv.THRESH_BINARY)
    inv = cv.bitwise_not(thresh)
    up = cv.resize(inv, None, fx=2, fy=2, interpolation=cv.INTER_CUBIC)
    config = "--psm 7 -c tessedit_char_whitelist=0123456789"
    try:
        text = pytesseract.image_to_string(up, config=config)
    except pytesseract.TesseractNotFoundError:
        text = ""
    m = re.search(r"\d+", text)
    return int(m.group()) if m else 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_score_opp <screenshot.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return
    shot_h, shot_w = img.shape[:2]
    x, y, w, h = map_roi(ROI["scoreOpp"], shot_w, shot_h, shot_w, shot_h)
    pad = THRESH.get("ocrPad", 0)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(w + pad * 2, shot_w - x)
    h = min(h + pad * 2, shot_h - y)
    count = match_score(img[y:y + h, x:x + w])
    print(f"scoreOpp count: {count}")
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(
        img,
        str(count),
        (x, max(0, y - 10)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv.imwrite("score_opp_debug.png", img)
    print("Annotated screenshot saved to score_opp_debug.png")


if __name__ == "__main__":
    main()

