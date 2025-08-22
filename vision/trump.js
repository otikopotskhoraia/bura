// vision/trump.js
import cv from 'opencv4nodejs';
import { THRESH } from './config.js';
import fs from 'node:fs';
import path from 'node:path';

function loadTemplates(dir) {
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.png'));
  return files.map(f => ({
    name: path.basename(f, '.png'),
    mat: cv.imdecode(fs.readFileSync(path.join(dir, f)))
  }));
}

const rankTemplates = loadTemplates('vision/templates/ranks'); // A,K,Q,J,10,...
const suitTemplates = loadTemplates('vision/templates/suits'); // spade, heart, diamond, club

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
