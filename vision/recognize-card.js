// vision/recognize-card.js
import cv from 'opencv4nodejs';
import { rankTemplates, suitTemplates } from './templates.js';
import { detectCardInSlot } from './cards.js';

const imgPath = process.argv[2];
if (!imgPath) {
  console.error('Usage: node vision/recognize-card.js <image.png>');
  process.exit(1);
}

const img = cv.imread(imgPath);
const result = detectCardInSlot(img, rankTemplates, suitTemplates);

if (result.card) {
  console.log(`Detected ${result.card} (confidence: ${result.conf.toFixed(2)})`);
} else {
  console.log('No card match found');
}
