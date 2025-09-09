# server.py
from flask import Flask
import cv2 as cv
from vision.templates import rank_templates, suit_templates
from vision.cards import detect_card_in_slot
from vision.config import ROI
from vision.detect import map_roi
from vision.trump_search import find_trump_card

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Dummy server running"

if __name__ == "__main__":
    img_path = "./sample.png"
    try:
        img = cv.imread(img_path)
        if img is None:
            raise ValueError("Image data is empty")
        shot_h, shot_w = img.shape[:2]

        res = find_trump_card(img)
        if res["rank"] and res["suit"]:
            print(
                f"Trump: {res['rank']}-{res['suit']} (confidence: {res['conf']:.2f})"
            )
        else:
            print("Trump: No trump card match found")

        for idx, slot in enumerate(ROI.get("handSlots", [])):
            x, y, w, h = map_roi(slot, shot_w, shot_h, 1920, 1080)
            slot_img = img[y:y + h, x:x + w]
            result = detect_card_in_slot(slot_img, rank_templates, suit_templates)
            if result["debug_img"] is not None:
                cv.imwrite(f"match_debug_{idx}.png", result["debug_img"])
            if result["card"]:
                print(
                    f"Slot {idx}: Detected {result['card']} (confidence: {result['conf']:.2f})"
                )
            else:
                print(f"Slot {idx}: No card match found")
    except Exception as e:
        print("Failed to process image:", img_path)
        print(e)
    app.run(port=3000)
