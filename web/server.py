"""
tiramisu examples hub
FastAPI backend for Shakespeare native fallback + static demos
"""

import os
import re
import subprocess
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).parent.parent
STATIC = Path(__file__).parent / "static"
BINARY = ROOT / "build" / "examples" / "train_shakespeare"
WASM_DIR = ROOT / "build-wasm" / "examples"
SHAKESPEARE_CKPT = ROOT / "checkpoints" / "shakespeare_2m.ckpt"
MNIST_WEIGHTS = ROOT / "checkpoints" / "mnist_mlp.bin"
DATA = ROOT / "data" / "tiny_shakespeare.txt"
PRESET = "2m"
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "180"))

app = FastAPI(title="tiramisu examples")


class GenerateRequest(BaseModel):
  prompt: str = Field(default="First Citizen:\n", max_length=200)
  temperature: float = Field(default=0.8, ge=0.1, le=1.5)
  max_chars: int = Field(default=150, ge=50, le=800)


class GenerateResponse(BaseModel):
  text: str
  prompt: str


def call_binary(prompt: str, temperature: float, max_chars: int) -> str:
  if not BINARY.exists():
    raise RuntimeError(f"Binary not found at {BINARY}. Did you run cmake --build?")
  if not SHAKESPEARE_CKPT.exists():
    raise RuntimeError(f"Checkpoint not found at {SHAKESPEARE_CKPT}")
  if not DATA.exists():
    raise RuntimeError(f"Corpus not found at {DATA}")

  cmd = [
    str(BINARY),
    "--preset", PRESET,
    "--data", str(DATA),
    "--checkpoint", str(SHAKESPEARE_CKPT),
    "--generate-only",
    "--prompt", prompt,
    "--sample-chars", str(max_chars),
    "--temperature", str(temperature),
  ]
  if os.environ.get("USE_CUDA", "").lower() in ("1", "true", "yes"):
    cmd.insert(1, "--cuda")

  try:
    result = subprocess.run(
      cmd,
      capture_output=True,
      text=True,
      timeout=GENERATION_TIMEOUT,
    )
  except subprocess.TimeoutExpired as e:
    raise RuntimeError(
      f"Generation timed out after {GENERATION_TIMEOUT}s "
      f"(try fewer characters or set GENERATION_TIMEOUT)"
    ) from e

  if result.returncode != 0:
    raise RuntimeError(f"Binary failed: {result.stderr[:500]}")

  match = re.search(r"---\s*\n(.*)", result.stdout, re.DOTALL)
  if not match:
    raise RuntimeError(f"Unexpected binary output: {result.stdout[:200]}")
  return match.group(1).strip()


@app.post("/generate", response_model=GenerateResponse)
async def generate(req: GenerateRequest):
  try:
    text = call_binary(req.prompt, req.temperature, req.max_chars)
  except (RuntimeError, subprocess.SubprocessError) as e:
    raise HTTPException(status_code=500, detail=str(e)) from e
  return GenerateResponse(text=text, prompt=req.prompt)


@app.get("/health")
async def health():
  return {
    "binary": BINARY.exists(),
    "shakespeare_checkpoint": SHAKESPEARE_CKPT.exists(),
    "mnist_weights": MNIST_WEIGHTS.exists(),
    "data": DATA.exists(),
    "shakespeare_wasm_js": (WASM_DIR / "tiramisu_infer.js").exists(),
    "shakespeare_wasm_bin": (WASM_DIR / "tiramisu_infer.wasm").exists(),
    "mnist_wasm_js": (WASM_DIR / "mnist_infer.js").exists(),
    "mnist_wasm_bin": (WASM_DIR / "mnist_infer.wasm").exists(),
  }


def wasm_file(name: str, mime: str):
  path = WASM_DIR / name
  if not path.exists():
    raise HTTPException(
      status_code=404,
      detail=f"WASM not built. Run: cmake --build build-wasm --target {path.stem}",
    )
  return FileResponse(path, media_type=mime)


@app.get("/shakespeare/tiramisu_infer.js")
async def shakespeare_wasm_js():
  return wasm_file("tiramisu_infer.js", "application/javascript")


@app.get("/shakespeare/tiramisu_infer.wasm")
async def shakespeare_wasm_bin():
  return wasm_file("tiramisu_infer.wasm", "application/wasm")


@app.get("/shakespeare/assets/shakespeare_2m.ckpt")
async def shakespeare_checkpoint():
  if not SHAKESPEARE_CKPT.exists():
    raise HTTPException(status_code=404, detail=f"Checkpoint not found at {SHAKESPEARE_CKPT}")
  return FileResponse(SHAKESPEARE_CKPT, media_type="application/octet-stream")


@app.get("/mnist/mnist_infer.js")
async def mnist_wasm_js():
  return wasm_file("mnist_infer.js", "application/javascript")


@app.get("/mnist/mnist_infer.wasm")
async def mnist_wasm_bin():
  return wasm_file("mnist_infer.wasm", "application/wasm")


@app.get("/mnist/assets/mnist_mlp.bin")
async def mnist_weights():
  if not MNIST_WEIGHTS.exists():
    raise HTTPException(status_code=404, detail=f"Weights not found at {MNIST_WEIGHTS}")
  return FileResponse(MNIST_WEIGHTS, media_type="application/octet-stream")


app.mount("/", StaticFiles(directory=STATIC, html=True), name="static")


if __name__ == "__main__":
  import uvicorn
  port = int(os.environ.get("PORT", 8000))
  uvicorn.run(app, host="0.0.0.0", port=port)
