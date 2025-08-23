// vision/templates.js
import cv from 'opencv4nodejs';
import fs from 'node:fs';
import path from 'node:path';

export function loadTemplates(dir) {
  const files = fs.readdirSync(dir).filter(f => f.endsWith('.png'));
  return files.map(f => ({
    name: f.replace(/(\.png)+$/i, ''),
    mat: cv.imdecode(fs.readFileSync(path.join(dir, f)))
  }));
}

export const rankTemplates = loadTemplates('vision/templates/ranks');
export const suitTemplates = loadTemplates('vision/templates/suits');
