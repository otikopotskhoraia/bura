# server.py
from flask import Flask
import cv2 as cv
from vision.trump_search import find_trump_card
from vision.scan import analyze_image
from vision.score_search import find_scores

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

        trump = find_trump_card(img)
        if trump["rank"] and trump["suit"]:
            print(
                f"Trump: {trump['rank']}-{trump['suit']} (confidence: {trump['conf']:.2f})",
            )
        else:
            print("Trump: No trump card match found")

        state = analyze_image(img, trump=trump)
        print(f"Taken by me: {state.get('takenMe', 0)}")
        print(f"Taken by opponent: {state.get('takenOpp', 0)}")
        print(f"Offered doubling: {state.get('hasOfferedDoubling', False)}")

        scores = find_scores(img)
        print(f"Score me: {scores['me']}")
        print(f"Score opponent: {scores['opp']}")

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
