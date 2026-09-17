const canvas = document.getElementById("draw-canvas");
const ctx = canvas.getContext("2d");
const predictBtn = document.getElementById("predict-btn");
const clearBtn = document.getElementById("clear-canvas-btn");
const digitEl = document.getElementById("prediction-digit");
const statusEl = document.getElementById("prediction-status");
const networkViz = document.getElementById("network-viz");
const hiddenStrip = document.getElementById("hidden-strip");
const logitsChart = document.getElementById("logits-chart");

const DISPLAY = canvas.width;
const MODEL = 28;
const BRUSH = 9;

let drawing = false;
let worker = null;
let hiddenCells = [];
let logitBars = [];

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL("./mnist_worker.js", import.meta.url));
  }
  return worker;
}

function ensureVizDom() {
  if (hiddenCells.length === 0) {
    hiddenStrip.replaceChildren();
    for (let i = 0; i < 128; i++) {
      const cell = document.createElement("div");
      cell.className = "hidden-cell";
      cell.title = `Neuron ${i}`;
      hiddenStrip.appendChild(cell);
      hiddenCells.push(cell);
    }
  }
  if (logitBars.length === 0) {
    logitsChart.replaceChildren();
    for (let i = 0; i < 10; i++) {
      const col = document.createElement("div");
      col.className = "logit-col";
      const bar = document.createElement("div");
      bar.className = "logit-bar";
      const fill = document.createElement("div");
      fill.className = "logit-fill";
      bar.appendChild(fill);
      const label = document.createElement("span");
      label.className = "logit-label";
      label.textContent = String(i);
      col.appendChild(bar);
      col.appendChild(label);
      logitsChart.appendChild(col);
      logitBars.push({ col, fill, label });
    }
  }
}

function softmax(logits) {
  const max = Math.max(...logits);
  const exps = logits.map((x) => Math.exp(x - max));
  const sum = exps.reduce((a, b) => a + b, 0);
  return exps.map((e) => e / sum);
}

function renderNetworkViz(digit, logits, hidden) {
  ensureVizDom();
  networkViz.hidden = false;

  const maxHidden = Math.max(...hidden, 1e-6);
  hidden.forEach((value, i) => {
    const t = Math.min(1, Math.max(0, value / maxHidden));
    hiddenCells[i].style.opacity = String(0.08 + t * 0.92);
    hiddenCells[i].title = `Neuron ${i}: ${value.toFixed(3)}`;
  });

  const probs = softmax(logits);
  const maxLogit = Math.max(...logits);
  const minLogit = Math.min(...logits);
  const span = Math.max(maxLogit - minLogit, 1e-6);

  logitBars.forEach((bar, i) => {
    const height = ((logits[i] - minLogit) / span) * 100;
    bar.fill.style.height = `${height}%`;
    bar.col.classList.toggle("winner", i === digit);
    bar.label.title = `logit ${logits[i].toFixed(2)} · ${(probs[i] * 100).toFixed(1)}%`;
  });
}

function clearViz() {
  networkViz.hidden = true;
  hiddenCells.forEach((cell) => {
    cell.style.opacity = "0.08";
  });
  logitBars.forEach((bar) => {
    bar.fill.style.height = "0%";
    bar.col.classList.remove("winner");
    bar.label.title = "";
  });
}

function clearCanvas() {
  ctx.fillStyle = "#d4ecff";
  ctx.fillRect(0, 0, DISPLAY, DISPLAY);
  digitEl.textContent = "—";
  statusEl.textContent = "ready";
  statusEl.className = "prediction-status";
  clearViz();
}

function canvasPos(event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  return {
    x: (clientX - rect.left) * (DISPLAY / rect.width),
    y: (clientY - rect.top) * (DISPLAY / rect.height),
  };
}

function drawDot(x, y) {
  ctx.fillStyle = "#000000";
  ctx.beginPath();
  ctx.arc(x, y, BRUSH, 0, Math.PI * 2);
  ctx.fill();
}

function startDraw(event) {
  drawing = true;
  const { x, y } = canvasPos(event);
  drawDot(x, y);
  event.preventDefault();
}

function moveDraw(event) {
  if (!drawing) return;
  const { x, y } = canvasPos(event);
  drawDot(x, y);
  event.preventDefault();
}

function endDraw() {
  drawing = false;
}

function samplePixels() {
  const off = document.createElement("canvas");
  off.width = MODEL;
  off.height = MODEL;
  const offCtx = off.getContext("2d");
  offCtx.drawImage(canvas, 0, 0, MODEL, MODEL);
  const { data } = offCtx.getImageData(0, 0, MODEL, MODEL);
  const pixels = new Float32Array(MODEL * MODEL);
  for (let i = 0; i < MODEL * MODEL; i++) {
    const gray = data[i * 4];
    pixels[i] = (gray / 255 - 0.5) / 0.5;
  }
  return pixels;
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "prediction-status " + (cls || "");
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", moveDraw);
window.addEventListener("mouseup", endDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", moveDraw, { passive: false });
window.addEventListener("touchend", endDraw);

clearBtn.addEventListener("click", clearCanvas);

predictBtn.addEventListener("click", () => {
  predictBtn.disabled = true;
  setStatus("loading...", "generating");

  const w = getWorker();
  const onMessage = (event) => {
    const msg = event.data;
    if (msg.type === "status") {
      setStatus(msg.text, "generating");
      return;
    }
    if (msg.type === "error") {
      w.removeEventListener("message", onMessage);
      digitEl.textContent = "!";
      setStatus(msg.message, "error");
      predictBtn.disabled = false;
      return;
    }
    if (msg.type === "result") {
      w.removeEventListener("message", onMessage);
      digitEl.textContent = String(msg.digit);
      renderNetworkViz(msg.digit, msg.logits, msg.hidden);
      setStatus("done", "done");
      predictBtn.disabled = false;
    }
  };

  w.addEventListener("message", onMessage);
  w.postMessage({ type: "predict", pixels: samplePixels() });
});

clearCanvas();
