# vision/config.py

ROI = {
    "deckCount": {"x": 1220, "y": 180, "w": 60, "h": 40},
    "takenOpp":  {"x": 1560, "y": 110, "w": 80, "h": 55},
    "takenMe":   {"x": 1560, "y": 860, "w": 80, "h": 55},
    "trumpSlot": {"x": 1265, "y": 235, "w": 55, "h": 85},
    "tableCenter": {"x": 900, "y": 330, "w": 400, "h": 280},
    "handSlots": [
        {"x": 855, "y": 700, "w": 113, "h": 145},
        {"x": 973, "y": 700, "w": 113, "h": 145},
        {"x": 1091,"y": 700, "w": 113, "h": 145},
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
