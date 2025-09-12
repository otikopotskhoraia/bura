import os
import sys
import cv2 as cv
from .config import ROI, THRESH
from .detect import map_roi
from .templates import load_templates

score_templates = load_templates(
    os.path.join(os.path.dirname(__file__), "templates", "score")
)


def match_score(bgr_roi_mat):
    gray = cv.cvtColor(bgr_roi_mat, cv.COLOR_BGR2GRAY)
    best = {"name": None, "score": 0}
    for tmpl in score_templates:
        t_gray = cv.cvtColor(tmpl["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
        _, max_val, _, _ = cv.minMaxLoc(res)
        if max_val > best["score"]:
            best = {"name": tmpl["name"], "score": float(max_val)}
    if best["score"] >= THRESH["matchMinScore"] and best["name"] is not None:
        try:
            return int(best["name"])
        except ValueError:
            return 0
    return 0


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_score_me <screenshot.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return
    shot_h, shot_w = img.shape[:2]
    x, y, w, h = map_roi(ROI['scoreMe'], shot_w, shot_h, shot_w, shot_h)
    score = match_score(img[y:y+h, x:x+w])
    print(f"scoreMe: {score}")
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
    cv.imwrite("score_me_debug.png", img)
    print("Annotated screenshot saved to score_me_debug.png")


if __name__ == "__main__":
    main()
