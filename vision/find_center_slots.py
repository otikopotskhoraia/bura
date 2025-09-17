import sys
import cv2 as cv

from .cards import detect_card_in_slot
from .config import ROI
from .detect import map_roi
from .templates import rank_templates, suit_templates

CONF_THRESHOLD = 0.9


def main():
    if len(sys.argv) < 2:
        print("Usage: python -m vision.find_center_slots <screenshot.png>")
        return

    if not ROI.get("centerSlots"):
        print("No centerSlots configured in vision.config. Nothing to do.")
        return

    img = cv.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image {sys.argv[1]}")
        return

    shot_h, shot_w = img.shape[:2]
    annotated = img.copy()

    for idx, slot in enumerate(ROI["centerSlots"]):
        x, y, w, h = map_roi(slot, shot_w, shot_h, 1920, 1080)
        crop = img[y:y + h, x:x + w]
        result = detect_card_in_slot(crop, rank_templates, suit_templates)

        conf = result["conf"]
        card = result["card"]
        is_confident = bool(card) and conf >= CONF_THRESHOLD

        if not is_confident:
            # Treat low-confidence identifications as misses so downstream
            # consumers don't mistake them for genuine matches.
            result["card"] = None

        label = card if is_confident else "no match"
        print(f"Slot {idx}: {label} (conf={conf:.2f})")

        cv.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv.putText(
            annotated,
            f"{label} {conf:.2f}",
            (x, max(0, y - 10)),
            cv.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )

    out_path = "center_slots_debug.png"
    cv.imwrite(out_path, annotated)
    print(f"Annotated screenshot saved to {out_path}")


if __name__ == "__main__":
    main()
