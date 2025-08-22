// vision/counters.js
import cv from 'opencv4nodejs';
import { createWorker } from 'tesseract.js';
import { THRESH } from './config.js';

const worker = await createWorker({ logger: () => {} });
await worker.loadLanguage('eng');
await worker.initialize('eng');
await worker.setParameters({ tessedit_char_whitelist: '0123456789' });

export async function readGreenNumber(bgrRoiMat) {
  const hsv = bgrRoiMat.cvtColor(cv.COLOR_BGR2HSV);
  const low = new cv.Vec(...THRESH.greenHSV.low);
  const high = new cv.Vec(...THRESH.greenHSV.high);
  const mask = hsv.inRange(low, high);
  // Clean up noise
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3,3));
  const clean = mask.morphologyEx(kernel, cv.MORPH_OPEN, { iterations: 1 });

  // Invert to black text on white for OCR
  const inv = clean.bitwiseNot();

  // Upscale for OCR
  const up = inv.resize(new cv.Size(0,0), 2, 2, cv.INTER_CUBIC);

  // Tesseract expects PNG Buffer
  const png = cv.imencode('.png', up);
  const { data: { text } } = await worker.recognize(Buffer.from(png));
  const m = text.match(/\d+/);
  return m ? parseInt(m[0], 10) : 0;
}
