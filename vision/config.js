// vision/config.js
export const ROI = {
  // all values in CSS px; you’ll map to bitmap using (shotW/viewportW, shotH/viewportH)
  deckCount:      { x: 1220, y: 180, w: 60,  h: 40 },   // green "12"
  takenOpp:       { x: 1560, y: 110, w: 80,  h: 55 },   // top-right pile green
  takenMe:        { x: 1560, y: 860, w: 80,  h: 55 },   // bottom-right pile green
  trumpSlot:      { x: 1265, y: 235, w: 55,  h: 85 },   // face-up trump card next to deck
  tableCenter:    { x: 900,  y: 330, w: 400, h: 280 },  // table play area (for context)
  handSlots: [ // 3-card Bura; adjust x per slot
    { x: 740, y: 780, w: 105, h: 140 },
    { x: 870, y: 780, w: 105, h: 140 },
    { x: 1000,y: 780, w: 105, h: 140 }
  ]
};

export const THRESH = {
  greenHSV: { low: [45, 80, 60], high: [85, 255, 255] }, // approximate; tune
  ocrPad: 4,   // px padding before OCR
  matchMinScore: 0.72, // template matching threshold
  confirmFrames: 2     // #consecutive frames to accept change
};
