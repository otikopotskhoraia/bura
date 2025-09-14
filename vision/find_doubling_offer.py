import sys
import cv2 as cv
from .doubling_offer import detect_doubling_popup


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_doubling_offer <image.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return
    res = detect_doubling_popup(img)
    print(f"hasOfferedDoubling: {res}")


if __name__ == "__main__":
    main()
