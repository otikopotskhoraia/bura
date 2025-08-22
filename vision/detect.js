// vision/detect.js
import cv from 'opencv4nodejs';
export function mapRoi(roi, shotW, shotH, viewportW, viewportH) {
  const sx = Math.round(roi.x * (shotW / Math.max(1, viewportW)));
  const sy = Math.round(roi.y * (shotH / Math.max(1, viewportH)));
  const sw = Math.round(roi.w * (shotW / Math.max(1, viewportW)));
  const sh = Math.round(roi.h * (shotH / Math.max(1, viewportH)));
  // clamp
  return new cv.Rect(
    Math.max(0, Math.min(sx, shotW - 1)),
    Math.max(0, Math.min(sy, shotH - 1)),
    Math.max(1, Math.min(sw, shotW - Math.max(0, Math.min(sx, shotW - 1)))),
    Math.max(1, Math.min(sh, shotH - Math.max(0, Math.min(sy, shotH - 1))))
  );
}
