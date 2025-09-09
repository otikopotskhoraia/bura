import sys
import cv2 as cv
from .trump_search import find_trump_card


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_trump <image.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return

    res = find_trump_card(img)
    if res["rank"] and res["suit"]:
        print(
            f"Detected {res['rank']}-{res['suit']} (confidence: {res['conf']:.2f})"
        )
        if res["bbox"]:
            x0, y0, w, h = res["bbox"]
            top_left = (x0, y0)
            bottom_right = (x0 + w, y0 + h)
            cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
            cv.putText(
                img,
                f"{res['rank']}-{res['suit']}",
                (x0, max(0, y0 - 10)),
                cv.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
            )
            cv.imwrite("trump_match_debug.png", img)
            print("Trump card highlighted in trump_match_debug.png")
    else:
        print("No trump card match found")


if __name__ == "__main__":
    main()
