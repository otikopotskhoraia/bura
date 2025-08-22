// vision/detect.js
import cv from 'opencv4nodejs';
import { ROI, THRESH } from './config.js';
import { mapRoi } from './detect.js';
import { readGreenNumber } from './counters.js';
import { detectTrump } from './trump.js';
import { detectCardInSlot } from './cards.js';
import fs from 'node:fs';
import path from 'node:path';

function loadTmps(dir) {
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.png'));
  return files.map(f => ({ name: path.basename(f, '.png'), mat: cv.imdecode(fs.readFileSync(path.join(dir, f))) }));
}
const rankTmps = loadTmps('vision/templates/ranks');
const suitTmps = loadTmps('vision/templates/suits');

export async function detectFromScreenshot(pngBuffer, viewportW, viewportH) {
  const shot = cv.imdecode(pngBuffer); // BGR Mat
  const shotW = shot.cols, shotH = shot.rows;

  // Counters
  const deckRect = mapRoi(ROI.deckCount, shotW, shotH, viewportW, viewportH);
  const takenOppRect = mapRoi(ROI.takenOpp, shotW, shotH, viewportW, viewportH);
  const takenMeRect = mapRoi(ROI.takenMe, shotW, shotH, viewportW, viewportH);

  const deckCount = await readGreenNumber(shot.getRegion(deckRect));
  const takenOpp  = await readGreenNumber(shot.getRegion(takenOppRect));
  const takenMe   = await readGreenNumber(shot.getRegion(takenMeRect));

  // Trump
  const trumpRect = mapRoi(ROI.trumpSlot, shotW, shotH, viewportW, viewportH);
  const trump = detectTrump(shot.getRegion(trumpRect));

  // Hand slots
  const hand = [];
  for (const slot of ROI.handSlots) {
    const r = mapRoi(slot, shotW, shotH, viewportW, viewportH);
    const info = detectCardInSlot(shot.getRegion(r), rankTmps, suitTmps);
    hand.push(info.card); // can be null if uncertain this frame
  }

  return {
    deckCount, takenOpp, takenMe,
    trump: trump.rank && trump.suit ? `${trump.rank}-${trump.suit}` : null,
    hand
  };
}
