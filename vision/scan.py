from .templates import rank_templates, suit_templates
from .cards import detect_card_in_slot
from .config import ROI
from .detect import map_roi
from .trump_search import find_trump_card


def analyze_image(img):
    """Detect trump card and cards in hand slots from a screenshot image.

    Args:
        img: BGR screenshot image.

    Returns:
        dict with keys 'trump' and 'slots'.
        'trump' is the result from ``find_trump_card``.
        'slots' is a list of dictionaries with keys 'slot', 'card', 'conf',
        and 'debug_img'.
    """
    shot_h, shot_w = img.shape[:2]

    trump = find_trump_card(img)

    slots = []
    for idx, slot in enumerate(ROI.get("handSlots", [])):
        x, y, w, h = map_roi(slot, shot_w, shot_h, 1920, 1080)
        slot_img = img[y:y + h, x:x + w]
        result = detect_card_in_slot(slot_img, rank_templates, suit_templates)
        slots.append({
            "slot": idx,
            "card": result["card"],
            "conf": result["conf"],
            "debug_img": result["debug_img"],
        })

    return {"trump": trump, "slots": slots}
