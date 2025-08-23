# vision/smooth.py
import cv2 as cv
import numpy as np
from .config import ROI
from .detect import map_roi
from .counters import read_green_number
from .trump import detect_trump
from .cards import detect_card_in_slot
from .templates import rank_templates, suit_templates


def detect_from_screenshot(png_buffer, viewport_w, viewport_h):
    shot = cv.imdecode(np.frombuffer(png_buffer, np.uint8), cv.IMREAD_COLOR)
    shot_h, shot_w = shot.shape[:2]

    deck_rect = map_roi(ROI['deckCount'], shot_w, shot_h, viewport_w, viewport_h)
    taken_opp_rect = map_roi(ROI['takenOpp'], shot_w, shot_h, viewport_w, viewport_h)
    taken_me_rect = map_roi(ROI['takenMe'], shot_w, shot_h, viewport_w, viewport_h)

    x, y, w, h = deck_rect
    deck_count = read_green_number(shot[y:y+h, x:x+w])
    x, y, w, h = taken_opp_rect
    taken_opp = read_green_number(shot[y:y+h, x:x+w])
    x, y, w, h = taken_me_rect
    taken_me = read_green_number(shot[y:y+h, x:x+w])

    trump_rect = map_roi(ROI['trumpSlot'], shot_w, shot_h, viewport_w, viewport_h)
    x, y, w, h = trump_rect
    trump_info = detect_trump(shot[y:y+h, x:x+w])

    hand = []
    for slot in ROI['handSlots']:
        r = map_roi(slot, shot_w, shot_h, viewport_w, viewport_h)
        x, y, w, h = r
        card_info = detect_card_in_slot(shot[y:y+h, x:x+w], rank_templates, suit_templates)
        hand.append(card_info['card'])

    trump_card = None
    if trump_info['rank'] and trump_info['suit']:
        trump_card = f"{trump_info['rank']}-{trump_info['suit']}"

    return {
        'deckCount': deck_count,
        'takenOpp': taken_opp,
        'takenMe': taken_me,
        'trump': trump_card,
        'hand': hand,
    }
