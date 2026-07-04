const BASE = (() => {
  const path = new URL(self.location.href).pathname;
  const idx = path.lastIndexOf("/");
  return path.slice(0, idx + 1);
})();

let modulePromise = null;
let moduleRef = null;

function wasmUrl(file) {
  const clean = file.replace(/^\//, "");
  return `${BASE}${clean}`;
}

async function loadModule() {
  if (moduleRef) {
    return moduleRef;
  }
  if (!modulePromise) {
    modulePromise = (async () => {
      importScripts(wasmUrl("tiramisu_infer.js"));
      const factory = self.createTiramisuModule;
      if (typeof factory !== "function") {
        throw new Error("createTiramisuModule not found");
      }
      moduleRef = await factory({
        locateFile: (path) => wasmUrl(path),
      });
      return moduleRef;
    })();
  }
  return modulePromise;
}

async function ensureModel() {
  const Module = await loadModule();
  if (Module._tiramisu_model_ready) {
    return Module;
  }

  const resp = await fetch(wasmUrl("assets/shakespeare_10m_int8.ckpt"));
  if (!resp.ok) {
    throw new Error(`Failed to fetch checkpoint (${resp.status})`);
  }
  const bytes = new Uint8Array(await resp.arrayBuffer());
  const ptr = Module._malloc(bytes.length);
  Module.HEAPU8.set(bytes, ptr);
  const rc = Module._tiramisu_init(ptr, bytes.length);
  Module._free(ptr);
  if (rc !== 0) {
    throw new Error("Failed to load checkpoint into WASM model");
  }
  Module._tiramisu_model_ready = true;
  return Module;
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function streamGenerate(Module, msg) {
  const prompt = msg.prompt || "First Citizen:\n";
  const maxChars = msg.max_chars || 150;
  const temperature = msg.temperature || 0.85;
  const seed = msg.seed || 42;

  const rc = Module.ccall(
    "tiramisu_generate_begin",
    "number",
    ["string", "number", "number", "number"],
    [prompt, maxChars, temperature, seed],
  );
  if (rc !== 0) {
    throw new Error("Failed to begin generation");
  }

  let text = "";
  try {
    for (let i = 0; i < maxChars; i++) {
      const code = Module._tiramisu_generate_step();
      if (code < 0) {
        break;
      }
      const ch = String.fromCharCode(code);
      text += ch;
      self.postMessage({ type: "token", ch, text });
      if (i % 4 === 3) {
        await sleep(0);
      }
    }
    self.postMessage({ type: "result", text, prompt });
  } finally {
    Module._tiramisu_generate_end();
  }
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (!msg || msg.type !== "generate") {
    return;
  }

  try {
    self.postMessage({ type: "status", text: "loading model..." });
    const Module = await ensureModel();
    self.postMessage({ type: "status", text: "generating..." });
    await streamGenerate(Module, msg);
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
