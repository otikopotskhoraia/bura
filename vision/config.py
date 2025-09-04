# vision/config.py

ROI = {
    "deckCount": {"x": 1220, "y": 180, "w": 60,  "h": 40},
    "takenOpp":  {"x": 1560, "y": 110, "w": 80,  "h": 55},
    "takenMe":   {"x": 1560, "y": 860, "w": 80,  "h": 55},
    "trumpSlot": {"x": 1265, "y": 235, "w": 55,  "h": 85},
    "tableCenter": {"x": 900,  "y": 330, "w": 400, "h": 280},
    "handSlots": [
        {"x": 740, "y": 780, "w": 105, "h": 140},
        {"x": 870, "y": 780, "w": 105, "h": 140},
        {"x": 1000,"y": 780, "w": 105, "h": 140},
    ],
}

THRESH = {
    "greenHSV": {"low": [45, 80, 60], "high": [85, 255, 255]},
    "ocrPad": 4,
    "matchMinScore": 0.5,
    "confirmFrames": 2,
    # Maximum allowed distance between the centers of the detected
    # rank and suit glyphs in pixels.  A larger separation implies the
    # glyphs likely come from different cards.
    "glyphMaxDist": 50,
}
