# server.py
from flask import Flask
import cv2 as cv
from vision.templates import rank_templates, suit_templates
from vision.cards import detect_card_in_slot
from vision.config import ROI
from vision.detect import map_roi
from vision.trump import detect_trump

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

        # Detect trump card
        trump_slot = ROI.get("trumpSlot")
        if trump_slot:
            x, y, w, h = map_roi(trump_slot, shot_w, shot_h, 1920, 1080)
            trump_img = img[y:y + h, x:x + w]
            t_res = detect_trump(trump_img)
            if t_res["rank"] and t_res["suit"]:
                print(
                    f"Trump: {t_res['rank']}-{t_res['suit']} "
                    f"(confidence: {t_res['conf']:.2f})"
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
                    f"Slot {idx}: Detected {result['card']} "
                    f"(confidence: {result['conf']:.2f})"
                )
            else:
                print(f"Slot {idx}: No card match found")
    except Exception as e:
        print("Failed to process image:", img_path)
        print(e)
    app.run(port=3000)
