# server.py
from flask import Flask
import cv2 as cv
from vision.templates import rank_templates, suit_templates
from vision.cards import detect_card_in_slot

app = Flask(__name__)

@app.route("/")
def index():
    return "✅ Dummy server running"

if __name__ == "__main__":
    import sys
    img_path = sys.argv[1] if len(sys.argv) > 1 else "vision/sample.png"
    try:
        img = cv.imread(img_path)
        result = detect_card_in_slot(img, rank_templates, suit_templates)
        if result["card"]:
            print(f"Detected {result['card']} (confidence: {result['conf']:.2f})")
        else:
            print("No card match found")
    except Exception as e:
        print("Failed to read image:", img_path)
        print(e)
    app.run(port=3000)
