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
const BRUSH = 12;
// MNIST digits were scaled into a 20px box inside the 28px frame and centred by
// their centre of mass; strokes in that frame are about 2.5px wide.
const BOX = 20;
const STROKE = 2.5;
const SUPERSAMPLE = 8;

let drawing = false;
let strokes = [];
const modelCanvas = document.createElement("canvas");
modelCanvas.width = MODEL * SUPERSAMPLE;
modelCanvas.height = MODEL * SUPERSAMPLE;
const modelCtx = modelCanvas.getContext("2d", { willReadFrequently: true });
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
      col.appendChild(label);
      col.appendChild(bar);
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
  logitBars.forEach((bar, i) => {
    bar.fill.style.transform = `scaleX(${probs[i]})`;
    bar.col.classList.toggle("winner", i === digit);
    bar.label.title = `logit ${logits[i].toFixed(2)}, ${(probs[i] * 100).toFixed(1)}%`;
  });
}

function clearViz() {
  ensureVizDom();
  hiddenCells.forEach((cell) => {
    cell.style.opacity = "0.08";
  });
  logitBars.forEach((bar) => {
    bar.fill.style.transform = "scaleX(0)";
    bar.col.classList.remove("winner");
    bar.label.title = "";
  });
}

function clearCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, DISPLAY, DISPLAY);
  strokes = [];
  digitEl.textContent = "?";
  digitEl.classList.add("is-empty");
  statusEl.textContent = "Ready";
  statusEl.className = "prediction-status";
  clearViz();
}

function canvasPos(event) {
  const rect = canvas.getBoundingClientRect();
  const clientX = event.touches ? event.touches[0].clientX : event.clientX;
  const clientY = event.touches ? event.touches[0].clientY : event.clientY;
  const clamp = (v) => Math.min(DISPLAY, Math.max(0, v));
  return {
    x: clamp((clientX - rect.left) * (DISPLAY / rect.width)),
    y: clamp((clientY - rect.top) * (DISPLAY / rect.height)),
  };
}

function startDraw(event) {
  drawing = true;
  const p = canvasPos(event);
  strokes.push([p]);
  ctx.fillStyle = "#000000";
  ctx.beginPath();
  ctx.arc(p.x, p.y, BRUSH, 0, Math.PI * 2);
  ctx.fill();
  event.preventDefault();
}

function moveDraw(event) {
  if (!drawing) return;
  const p = canvasPos(event);
  const stroke = strokes[strokes.length - 1];
  const prev = stroke[stroke.length - 1];
  stroke.push(p);
  // Join to the previous point so fast strokes don't break into dots.
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = BRUSH * 2;
  ctx.lineCap = "round";
  ctx.beginPath();
  ctx.moveTo(prev.x, prev.y);
  ctx.lineTo(p.x, p.y);
  ctx.stroke();
  event.preventDefault();
}

function endDraw() {
  drawing = false;
}

function resumeDraw(event) {
  if (event.buttons & 1) startDraw(event);
}

// Draw the recorded strokes into the 28x28 frame: white ink on black, rendered
// at 8x and box-averaged down. Returns ink in [0, 1] per pixel.
function renderStrokes(scale, offsetX, offsetY) {
  const size = MODEL * SUPERSAMPLE;
  const at = (p) => [(p.x * scale + offsetX) * SUPERSAMPLE, (p.y * scale + offsetY) * SUPERSAMPLE];
  modelCtx.fillStyle = "#000000";
  modelCtx.fillRect(0, 0, size, size);
  modelCtx.fillStyle = "#ffffff";
  modelCtx.strokeStyle = "#ffffff";
  modelCtx.lineWidth = STROKE * SUPERSAMPLE;
  modelCtx.lineCap = "round";
  modelCtx.lineJoin = "round";
  for (const stroke of strokes) {
    modelCtx.beginPath();
    if (stroke.length === 1) {
      const [x, y] = at(stroke[0]);
      modelCtx.arc(x, y, (STROKE * SUPERSAMPLE) / 2, 0, Math.PI * 2);
      modelCtx.fill();
      continue;
    }
    modelCtx.moveTo(...at(stroke[0]));
    for (let i = 1; i < stroke.length; i++) modelCtx.lineTo(...at(stroke[i]));
    modelCtx.stroke();
  }
  const { data } = modelCtx.getImageData(0, 0, size, size);
  const ink = new Float32Array(MODEL * MODEL);
  for (let y = 0; y < size; y++) {
    const row = Math.floor(y / SUPERSAMPLE) * MODEL;
    for (let x = 0; x < size; x++) {
      ink[row + Math.floor(x / SUPERSAMPLE)] += data[(y * size + x) * 4];
    }
  }
  for (let i = 0; i < ink.length; i++) ink[i] /= 255 * SUPERSAMPLE * SUPERSAMPLE;
  return ink;
}

// Prepare the drawing the way MNIST prepared its digits: fit the longer side
// into a 20px box, then shift so the centre of mass sits in the middle.
function modelInput() {
  const points = strokes.flat();
  if (points.length === 0) return null;
  const xs = points.map((p) => p.x);
  const ys = points.map((p) => p.y);
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const scale = (BOX - STROKE) / Math.max(maxX - minX, maxY - minY, 1);
  const offsetX = MODEL / 2 - ((minX + maxX) / 2) * scale;
  const offsetY = MODEL / 2 - ((minY + maxY) / 2) * scale;

  let ink = renderStrokes(scale, offsetX, offsetY);
  let mass = 0;
  let massX = 0;
  let massY = 0;
  for (let i = 0; i < ink.length; i++) {
    mass += ink[i];
    massX += ink[i] * ((i % MODEL) + 0.5);
    massY += ink[i] * (Math.floor(i / MODEL) + 0.5);
  }
  if (mass > 0) {
    ink = renderStrokes(scale, offsetX + MODEL / 2 - massX / mass, offsetY + MODEL / 2 - massY / mass);
  }
  // The model was trained on pixels scaled to [-1, 1]: background -1, ink +1.
  return ink.map((v) => v * 2 - 1);
}

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "prediction-status " + (cls || "");
}

canvas.addEventListener("mousedown", startDraw);
canvas.addEventListener("mousemove", moveDraw);
canvas.addEventListener("mouseleave", endDraw);
canvas.addEventListener("mouseenter", resumeDraw);
window.addEventListener("mouseup", endDraw);
canvas.addEventListener("touchstart", startDraw, { passive: false });
canvas.addEventListener("touchmove", moveDraw, { passive: false });
window.addEventListener("touchend", endDraw);

clearBtn.addEventListener("click", clearCanvas);

predictBtn.addEventListener("click", () => {
  const pixels = modelInput();
  if (!pixels) {
    setStatus("Draw a digit first", "");
    return;
  }
  predictBtn.disabled = true;
  setStatus("Loading...", "generating");

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
      digitEl.classList.remove("is-empty");
      renderNetworkViz(msg.digit, msg.logits, msg.hidden);
      setStatus("Done", "done");
      predictBtn.disabled = false;
    }
  };

  w.addEventListener("message", onMessage);
  w.postMessage({ type: "predict", pixels });
});

clearCanvas();
