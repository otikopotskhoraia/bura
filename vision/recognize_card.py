# vision/recognize_card.py
import sys
import cv2 as cv
from .templates import rank_templates, suit_templates
from .cards import detect_card_in_slot


def main():
    if len(sys.argv) < 2:
        print("Usage: python vision/recognize_card.py <image.png>")
        return
    img_path = sys.argv[1]
    img = cv.imread(img_path)
    result = detect_card_in_slot(img, rank_templates, suit_templates)
    if result["debug_img"] is not None:
        cv.imwrite("match_debug.png", result["debug_img"])
    if result["card"]:
        print(f"Detected {result['card']} (confidence: {result['conf']:.2f})")
    else:
        print("No card match found")


if __name__ == "__main__":
    main()
