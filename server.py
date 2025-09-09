# server.py
from flask import Flask
import cv2 as cv
from vision.templates import rank_templates, suit_templates
import numpy as np
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

        # Detect trump card using full-frame search similar to vision.find_trump
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        candidates = []
        for tmpl in suit_templates:
            t_gray = cv.cvtColor(tmpl["mat"], cv.COLOR_BGR2GRAY)
            res = cv.matchTemplate(gray, t_gray, cv.TM_CCOEFF_NORMED)
            ys, xs = np.where(res >= 0.8)
            for (x, y) in zip(xs, ys):
                candidates.append({"x": int(x), "y": int(y)})

        if candidates:
            h, w = img.shape[:2]
            cand = max(candidates, key=lambda c: c["x"])
            size = 140
            x0 = max(0, min(w - size, cand["x"] - 40))
            y0 = max(0, min(h - size, cand["y"] - 40))
            roi = img[y0 : y0 + size, x0 : x0 + size]

            best = {"rank": None, "suit": None, "conf": 0}
            for rot in [
                None,
                cv.ROTATE_90_CLOCKWISE,
                cv.ROTATE_180,
                cv.ROTATE_90_COUNTERCLOCKWISE,
            ]:
                r_img = cv.rotate(roi, rot) if rot is not None else roi
                t_res = detect_trump(r_img)
                if t_res["conf"] > best["conf"]:
                    best = t_res

            if best["rank"] and best["suit"]:
                print(
                    f"Trump: {best['rank']}-{best['suit']} "
                    f"(confidence: {best['conf']:.2f})"
                )
            else:
                print("Trump: No trump card match found")
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
