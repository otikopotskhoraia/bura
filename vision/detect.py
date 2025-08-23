# vision/detect.py

def map_roi(roi, shot_w, shot_h, viewport_w, viewport_h):
    sx = round(roi['x'] * (shot_w / max(1, viewport_w)))
    sy = round(roi['y'] * (shot_h / max(1, viewport_h)))
    sw = round(roi['w'] * (shot_w / max(1, viewport_w)))
    sh = round(roi['h'] * (shot_h / max(1, viewport_h)))
    x = max(0, min(sx, shot_w - 1))
    y = max(0, min(sy, shot_h - 1))
    w = max(1, min(sw, shot_w - x))
    h = max(1, min(sh, shot_h - y))
    return (x, y, w, h)
