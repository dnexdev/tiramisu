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
      importScripts(wasmUrl("mnist_infer.js"));
      const factory = self.createMnistModule;
      if (typeof factory !== "function") {
        throw new Error("createMnistModule not found");
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
  if (Module._mnist_model_ready) {
    return Module;
  }

  const resp = await fetch(wasmUrl("assets/mnist_mlp.bin"));
  if (!resp.ok) {
    throw new Error(`Failed to fetch weights (${resp.status})`);
  }
  const bytes = new Uint8Array(await resp.arrayBuffer());
  const ptr = Module._malloc(bytes.length);
  Module.HEAPU8.set(bytes, ptr);
  const rc = Module._mnist_init(ptr, bytes.length);
  Module._free(ptr);
  if (rc !== 0) {
    throw new Error("Failed to load MNIST weights into WASM model");
  }
  Module._mnist_model_ready = true;
  return Module;
}

self.onmessage = async (event) => {
  const msg = event.data;
  if (!msg || msg.type !== "predict") {
    return;
  }

  try {
    self.postMessage({ type: "status", text: "loading model..." });
    const Module = await ensureModel();
    self.postMessage({ type: "status", text: "predicting..." });

    const pixels = new Float32Array(msg.pixels);
    const pixelPtr = Module._malloc(pixels.length * pixels.BYTES_PER_ELEMENT);
    const logitsPtr = Module._malloc(10 * 4);
    const hiddenPtr = Module._malloc(128 * 4);

    Module.HEAPF32.set(pixels, pixelPtr >> 2);
    const digit = Module._mnist_predict_ex(pixelPtr, logitsPtr, hiddenPtr);

    Module._free(pixelPtr);

    if (digit < 0) {
      Module._free(logitsPtr);
      Module._free(hiddenPtr);
      throw new Error("Prediction failed");
    }

    const logits = Array.from(Module.HEAPF32.subarray(logitsPtr >> 2, (logitsPtr >> 2) + 10));
    const hidden = Array.from(Module.HEAPF32.subarray(hiddenPtr >> 2, (hiddenPtr >> 2) + 128));
    Module._free(logitsPtr);
    Module._free(hiddenPtr);

    self.postMessage({ type: "result", digit, logits, hidden });
  } catch (error) {
    self.postMessage({
      type: "error",
      message: error instanceof Error ? error.message : String(error),
    });
  }
};
