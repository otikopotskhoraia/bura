import sys
import cv2 as cv
from .trump import detect_trump


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_trump <image.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return
    result = detect_trump(img)
    if result["rank"] and result["suit"]:
        print(f"Detected {result['rank']}-{result['suit']} (confidence: {result['conf']:.2f})")
    else:
        print("No trump card match found")


if __name__ == "__main__":
    main()
