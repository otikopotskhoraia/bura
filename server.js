// server.js
import express from "express";
import bodyParser from "body-parser";

const app = express();
const PORT = 3000;

// Allow large payloads for image data
app.use(bodyParser.json({ limit: "50mb" }));
app.use(bodyParser.urlencoded({ extended: true, limit: "50mb" }));

// Dummy endpoint
app.post("/upload", (req, res) => {
  console.log("---- New Request ----");
  console.log("Headers:", req.headers);
  console.log("Body keys:", Object.keys(req.body));
  console.log("Body preview:", JSON.stringify(req.body).slice(0, 300) + "...");
  console.log("----------------------");

  res.json({ status: "ok", message: "Data received" });
});

app.listen(PORT, () => {
  console.log(`✅ Dummy server running on http://localhost:${PORT}`);
});
