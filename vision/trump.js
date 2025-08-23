// vision/trump.js
import cv from 'opencv4nodejs';
import { THRESH } from './config.js';
import { rankTemplates, suitTemplates } from './templates.js';

function bestMatch(graySrc, tmpl) {
  const res = graySrc.matchTemplate(tmpl, cv.TM_CCOEFF_NORMED);
  const minMax = res.minMaxLoc();
  return minMax.maxVal;
}

export function detectTrump(bgrRoiMat) {
  const gray = bgrRoiMat.cvtColor(cv.COLOR_BGR2GRAY);
  let bestRank = { name: null, score: 0 };
  for (const r of rankTemplates) {
    const tGray = r.mat.cvtColor(cv.COLOR_BGR2GRAY);
    const score = bestMatch(gray, tGray);
    if (score > bestRank.score) bestRank = { name: r.name, score };
  }
  let bestSuit = { name: null, score: 0 };
  for (const s of suitTemplates) {
    const tGray = s.mat.cvtColor(cv.COLOR_BGR2GRAY);
    const score = bestMatch(gray, tGray);
    if (score > bestSuit.score) bestSuit = { name: s.name, score };
  }
  if (bestRank.score >= THRESH.matchMinScore && bestSuit.score >= THRESH.matchMinScore) {
    return { rank: bestRank.name, suit: bestSuit.name, conf: Math.min(bestRank.score, bestSuit.score) };
  }
  return { rank: null, suit: null, conf: 0 };
}
