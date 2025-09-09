import sys
import cv2 as cv
from .config import ROI
from .detect import map_roi
from .counters import read_green_number


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_taken_me <screenshot.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return
    shot_h, shot_w = img.shape[:2]
    x, y, w, h = map_roi(ROI['takenMe'], shot_w, shot_h, shot_w, shot_h)
    count = read_green_number(img[y:y+h, x:x+w])
    print(f"takenMe count: {count}")
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
    cv.imwrite("taken_me_debug.png", img)
    print("Annotated screenshot saved to taken_me_debug.png")


if __name__ == "__main__":
    main()
