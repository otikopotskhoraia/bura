// vision/cards.js
import cv from 'opencv4nodejs';
import { THRESH } from './config.js';
// reuse templates loader from trump.js

export function detectCardInSlot(bgrSlotMat, rankTmps, suitTmps) {
  const gray = bgrSlotMat.cvtColor(cv.COLOR_BGR2GRAY);

  // Crop the corner where glyphs appear (tune for your UI)
  const glyphRect = new cv.Rect(5, 5, Math.max(20, Math.floor(gray.cols*0.45)), Math.max(25, Math.floor(gray.rows*0.35)));
  const glyph = gray.getRegion(glyphRect);

  let bestRank = { name: null, score: 0 };
  for (const r of rankTmps) {
    const tGray = r.mat.cvtColor(cv.COLOR_BGR2GRAY);
    const score = glyph.matchTemplate(tGray, cv.TM_CCOEFF_NORMED).minMaxLoc().maxVal;
    if (score > bestRank.score) bestRank = { name: r.name, score };
  }

  let bestSuit = { name: null, score: 0 };
  for (const s of suitTmps) {
    const tGray = s.mat.cvtColor(cv.COLOR_BGR2GRAY);
    const score = glyph.matchTemplate(tGray, cv.TM_CCOEFF_NORMED).minMaxLoc().maxVal;
    if (score > bestSuit.score) bestSuit = { name: s.name, score };
  }

  if (bestRank.score >= THRESH.matchMinScore && bestSuit.score >= THRESH.matchMinScore) {
    return { card: `${bestRank.name}-${bestSuit.name}`, conf: Math.min(bestRank.score, bestSuit.score) };
  }
  return { card: null, conf: 0 };
}
