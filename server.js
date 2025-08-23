// server.js
import express from "express";
import bodyParser from "body-parser";
import cv from "opencv4nodejs";
import { rankTemplates, suitTemplates } from "./vision/templates.js";
import { detectCardInSlot } from "./vision/cards.js";

const app = express();
const PORT = 3000;

// Allow large payloads for image data
app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ extended: true, limit: "50mb" }));

// Perform static card recognition on startup
const imgPath = process.argv[2] || "vision/sample.png";
try {
  const img = cv.imread(imgPath);
  const { card, conf } = detectCardInSlot(img, rankTemplates, suitTemplates);
  if (card) {
    console.log(`Detected ${card} (confidence: ${conf.toFixed(2)})`);
  } else {
    console.log("No card match found");
  }
} catch (err) {
  console.error("Failed to read image:", imgPath);
  console.error(err.message);
}

// app.post("/upload", (req, res) => {
//   console.log("---- New Request ----");
//   console.log("Headers:", req.headers);
//   console.log("Body keys:", Object.keys(req.body));
//   console.log("Body preview:", JSON.stringify(req.body).slice(0, 300) + "...");
//   console.log("----------------------");

//   res.json({ status: "ok", message: "Data received" });
// });

app.listen(PORT, () => {
  console.log(`✅ Dummy server running on http://localhost:${PORT}`);
});
