# vision/config.py

ROI = {
    "deckCount": {"x": 1220, "y": 180, "w": 60, "h": 40},
    # Shift taken counters inward so the OCR crops are not tight against the
    # screen edges.  This provides a little extra margin around the green
    # numbers which helps recognition.
    # Increase vertical margin to keep counters away from screen edges
    "takenOpp":  {"x": 1510, "y": 180, "w": 80, "h": 55},
    "takenMe":   {"x": 1510, "y": 790, "w": 80, "h": 55},
    "trumpSlot": {"x": 1265, "y": 235, "w": 55, "h": 85},
    "tableCenter": {"x": 900, "y": 330, "w": 400, "h": 280},
    "handSlots": [
        # Original ``y`` coordinate placed the crop slightly above the actual
        # cards, which meant the rank/suit glyphs were cut off and template
        # matching struggled.  Shift the slot windows downward so each card is
        # fully captured.
        {"x": 855, "y": 791, "w": 113, "h": 145},
        {"x": 973, "y": 791, "w": 113, "h": 145},
        {"x": 1091,"y": 791, "w": 113, "h": 145},
    ],
}

THRESH = {
    "greenHSV": {"low": [45, 80, 60], "high": [85, 255, 255]},
    "ocrPad": 4,
    # Minimum template-matching score for a glyph to be considered.
    # Lowering this slightly helps pick up weaker suit symbols.
    "matchMinScore": 0.4,
    "confirmFrames": 2,
    # Maximum allowed distance between the centers of the detected
    # rank and suit glyphs in pixels.  The previous limit was too strict for
    # the captured resolution, causing genuine rank/suit pairs to be rejected
    # (e.g. the "Q" and heart symbol were ~30px apart).  Relaxing this value
    # allows legitimate pairs to be recognised while still filtering out
    # distant false positives.
    "glyphMaxDist": 40,
}
