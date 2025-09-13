# vision/config.py

ROI = {
    "deckCount": {"x": 1220, "y": 180, "w": 60, "h": 40},
    # Shift taken counters inward so the OCR crops are not tight against the
    # screen edges.  This provides a little extra margin around the green
    # numbers which helps recognition.
    "takenOpp":  {"x": 1510, "y": 110, "w": 80, "h": 55},
    "takenMe":   {"x": 1510, "y": 760, "w": 80, "h": 55},
    # New: Score counters (white numbers on left side)
    # Slightly enlarged score regions for more reliable detection.
    "scoreOpp": {"x": 120, "y": 295, "w": 90, "h": 80},   # top-left score
    "scoreMe":  {"x": 90, "y": 560, "w": 110, "h": 60},  # bottom-left score
    # Fixed location of the trump card: right edge near the vertical center.
    # Use a larger window so the rank and suit glyphs are fully captured.
    "trumpSlot": {"x": 1430, "y": 470, "w": 140, "h": 140},
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
    # Extra padding (in pixels) applied around score ROIs before running
    # template matching.  Enlarging this region helps when the configured ROI
    # is slightly too tight and was clipping the digits.
    "ocrPad": 8,
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
