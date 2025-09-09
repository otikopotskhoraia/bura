import sys
import cv2 as cv
import numpy as np
from .trump import detect_trump
from .templates import suit_templates


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_trump <image.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    candidates = []
    # Locate potential suit glyphs across the whole screenshot.
    for tmpl in suit_templates:
        t_gray = cv.cvtColor(tmpl["mat"], cv.COLOR_BGR2GRAY)
        res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
        ys, xs = np.where(res >= 0.8)
        for (x, y) in zip(xs, ys):
            candidates.append({"x": int(x), "y": int(y)})

    if not candidates:
        print("No trump card match found")
        return

    # The trump card is expected on the right edge; pick the rightmost suit match
    # and examine a square region around it.
    h, w = img.shape[:2]
    cand = max(candidates, key=lambda c: c["x"])
    size = 140
    x0 = max(0, min(w - size, cand["x"] - 40))
    y0 = max(0, min(h - size, cand["y"] - 40))
    roi = img[y0 : y0 + size, x0 : x0 + size]

    best = {"rank": None, "suit": None, "conf": 0}
    for rot in [None, cv.ROTATE_90_CLOCKWISE, cv.ROTATE_180, cv.ROTATE_90_COUNTERCLOCKWISE]:
        rimg = cv.rotate(roi, rot) if rot is not None else roi
        res = detect_trump(rimg)
        if res["conf"] > best["conf"]:
            best = res

    if best["rank"] and best["suit"]:
        print(
            f"Detected {best['rank']}-{best['suit']} (confidence: {best['conf']:.2f})"
        )
        top_left = (x0, y0)
        bottom_right = (x0 + size, y0 + size)
        cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv.putText(
            img,
            f"{best['rank']}-{best['suit']}",
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
