import sys
import cv2 as cv
from .templates import rank_templates


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_k <image.png>")
        return
    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return
    template = next((t for t in rank_templates if t["name"] == "K"), None)
    if template is None:
        print("K template not found")
        return
    gray_img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    gray_tpl = cv.cvtColor(template["mat"], cv.COLOR_BGR2GRAY)
    res = cv.matchTemplate(gray_img, gray_tpl, cv.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv.minMaxLoc(res)
    print(f"Best match score for K: {max_val:.2f}")
    threshold = 0.5
    if max_val >= threshold:
        h, w = gray_tpl.shape
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        cv.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
        cv.imwrite("k_match_debug.png", img)
        print("K found and annotated in k_match_debug.png")
    else:
        print("K not found")


if __name__ == "__main__":
    main()
