import sys
import cv2 as cv
from .config import ROI
from .detect import map_roi
from .counters import match_score


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_score_opp_1 <screenshot.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return
    shot_h, shot_w = img.shape[:2]
    x, y, w, h = map_roi(ROI['scoreOpp_1'], shot_w, shot_h, shot_w, shot_h)
    pad = round(max(w, h) * 0.1)
    x = max(0, x - pad)
    y = max(0, y - pad)
    w = min(shot_w - x, w + 2 * pad)
    h = min(shot_h - y, h + 2 * pad)
    score = match_score(img[y:y+h, x:x+w])
    print(f"scoreOpp_1: {score}")
    cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv.putText(
        img,
        str(score),
        (x, max(0, y - 10)),
        cv.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
    )
    cv.imwrite("score_opp_1_debug.png", img)
    print("Annotated screenshot saved to score_opp_1_debug.png")


if __name__ == "__main__":
    main()

