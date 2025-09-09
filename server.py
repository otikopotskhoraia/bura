# server.py
from flask import Flask
import cv2 as cv
from vision.scan import analyze_image

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
        state = analyze_image(img)
        res = state["trump"]
        if res["rank"] and res["suit"]:
            print(
                f"Trump: {res['rank']}-{res['suit']} (confidence: {res['conf']:.2f})",
            )
        else:
            print("Trump: No trump card match found")

        for slot in state["slots"]:
            idx = slot["slot"]
            if slot["debug_img"] is not None:
                cv.imwrite(f"match_debug_{idx}.png", slot["debug_img"])
            if slot["card"]:
                print(
                    f"Slot {idx}: Detected {slot['card']} (confidence: {slot['conf']:.2f})",
                )
            else:
                print(f"Slot {idx}: No card match found")
    except Exception as e:
        print("Failed to process image:", img_path)
        print(e)
    app.run(port=3000)
