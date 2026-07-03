const PROMPTS = [
  "ROMEO\n",
  "HAMLET\n",
  "To be, or not to be,\n",
  "KING LEAR:\n",
  "All the world's a stage,\n",
  "JULIET:\n",
];

const USE_SERVER = new URLSearchParams(window.location.search).get("server") === "1";

const chipsEl = document.getElementById("prompt-chips");
const promptEl = document.getElementById("prompt");
const tempEl = document.getElementById("temperature");
const tempVal = document.getElementById("temp-val");
const charsEl = document.getElementById("max-chars");
const charsVal = document.getElementById("chars-val");
const timeHintEl = document.getElementById("time-hint");
const genBtn = document.getElementById("generate-btn");
const clearBtn = document.getElementById("clear-btn");
const outputEl = document.getElementById("output-text");
const statusEl = document.getElementById("status");

let worker = null;
let typeTimer = null;
let streamCursor = null;

PROMPTS.forEach(p => {
  const chip = document.createElement("button");
  chip.className = "prompt-chip";
  chip.textContent = p.replace(/\n/g, "↵");
  chip.addEventListener("click", () => { promptEl.value = p; });
  chipsEl.appendChild(chip);
});

function estimateSeconds(chars) {
  const base = Math.ceil(chars / 50);
  if (chars <= 128) {
    return Math.max(1, base);
  }
  return Math.ceil(base * 1.5);
}

function updateTimeHint() {
  const chars = parseInt(charsEl.value, 10);
  const secs = estimateSeconds(chars);
  timeHintEl.textContent = `~${secs}s in browser (~1s per 50 chars; longer runs slow down)`;
}

tempEl.addEventListener("input", () => { tempVal.textContent = (+tempEl.value).toFixed(2); });
charsEl.addEventListener("input", () => {
  charsVal.textContent = charsEl.value;
  updateTimeHint();
});

updateTimeHint();

function setStatus(text, cls) {
  statusEl.textContent = text;
  statusEl.className = "output-status " + (cls || "");
}

function beginStreamOutput() {
  clearInterval(typeTimer);
  outputEl.classList.remove("empty");
  outputEl.textContent = "";
  streamCursor = document.createElement("span");
  streamCursor.className = "cursor";
  outputEl.appendChild(streamCursor);
}

function appendStreamChar(ch) {
  if (!streamCursor) {
    beginStreamOutput();
  }
  streamCursor.insertAdjacentText("beforebegin", ch);
}

function finishStreamOutput() {
  if (streamCursor) {
    streamCursor.remove();
    streamCursor = null;
  }
}

function typewrite(el, text, onDone) {
  clearInterval(typeTimer);
  el.classList.remove("empty");
  el.textContent = "";

  const cursor = document.createElement("span");
  cursor.className = "cursor";
  el.appendChild(cursor);

  let i = 0;
  const speed = Math.max(8, Math.min(30, Math.round(6000 / text.length)));

  typeTimer = setInterval(() => {
    if (i >= text.length) {
      clearInterval(typeTimer);
      cursor.remove();
      if (onDone) onDone();
      return;
    }

    cursor.insertAdjacentText("beforebegin", text[i]);
    i++;
  }, speed);
}

function getWorker() {
  if (!worker) {
    worker = new Worker(new URL("./infer_worker.js", import.meta.url));
  }
  return worker;
}

async function generateViaServer(prompt, temperature, max_chars) {
  const resp = await fetch("/generate", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, temperature, max_chars }),
  });

  if (!resp.ok) {
    let msg = `Server error (${resp.status})`;
    const body = await resp.text();
    try {
      const err = JSON.parse(body);
      if (err.detail) msg = typeof err.detail === "string" ? err.detail : JSON.stringify(err.detail);
    } catch {
      if (body) msg = body.slice(0, 300);
    }
    throw new Error(msg);
  }

  const data = await resp.json();
  return data.text;
}

function generateViaWasm(prompt, temperature, max_chars) {
  return new Promise((resolve, reject) => {
    const w = getWorker();
    const onMessage = (event) => {
      const msg = event.data;
      if (msg.type === "status") {
        setStatus(msg.text, "generating");
        return;
      }
      if (msg.type === "token") {
        setStatus("generating...", "generating");
        appendStreamChar(msg.ch);
        return;
      }
      if (msg.type === "error") {
        w.removeEventListener("message", onMessage);
        finishStreamOutput();
        reject(new Error(msg.message));
        return;
      }
      if (msg.type === "result") {
        w.removeEventListener("message", onMessage);
        finishStreamOutput();
        resolve(msg.text);
      }
    };
    w.addEventListener("message", onMessage);
    w.postMessage({
      type: "generate",
      prompt,
      temperature,
      max_chars,
      seed: Math.floor(Math.random() * 0xffffffff),
    });
  });
}

genBtn.addEventListener("click", async () => {
  const prompt = promptEl.value.trim() || "First Citizen:\n";
  const temperature = parseFloat(tempEl.value);
  const max_chars = parseInt(charsEl.value, 10);

  genBtn.disabled = true;
  clearInterval(typeTimer);
  finishStreamOutput();
  outputEl.classList.remove("empty");
  outputEl.textContent = "";

  if (USE_SERVER) {
    setStatus("generating...", "generating");
  } else {
    beginStreamOutput();
    setStatus("loading model...", "generating");
  }

  try {
    if (USE_SERVER) {
      const text = await generateViaServer(prompt, temperature, max_chars);
      setStatus("typing...", "generating");
      typewrite(outputEl, text, () => setStatus("done", "done"));
    } else {
      await generateViaWasm(prompt, temperature, max_chars);
      setStatus("done", "done");
    }
  } catch (e) {
    outputEl.classList.add("empty");
    outputEl.textContent = "Error: " + e.message;
    setStatus("error", "error");
  } finally {
    genBtn.disabled = false;
  }
});

clearBtn.addEventListener("click", () => {
  clearInterval(typeTimer);
  finishStreamOutput();
  outputEl.classList.add("empty");
  outputEl.textContent = "Press generate to speak with the Bard.";
  setStatus("ready", "");
});
