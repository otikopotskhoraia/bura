# vision/templates.py
import cv2 as cv
import os


def load_templates(directory):
    files = [f for f in os.listdir(directory) if f.endswith('.png')]
    templates = []
    for f in files:
        path = os.path.join(directory, f)
        mat = cv.imread(path)
        name = f
        while name.lower().endswith('.png'):
            name = name[: -4]
        templates.append({"name": name, "mat": mat})
    return templates

base_dir = os.path.dirname(__file__)
rank_templates = load_templates(os.path.join(base_dir, 'templates', 'ranks'))
suit_templates = load_templates(os.path.join(base_dir, 'templates', 'suits'))
