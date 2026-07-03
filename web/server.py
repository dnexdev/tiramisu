"""
tiramisu Shakespeare demo
FastAPI backend
Calls the compiled train_shakespeare binary for generation
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
BINARY = ROOT / "build" / "examples" / "train_shakespeare"
WASM_DIR = ROOT / "build-wasm" / "examples"
CHECKPOINT = ROOT / "checkpoints" / "shakespeare_2m.ckpt"
DATA = ROOT / "data" / "tiny_shakespeare.txt"
PRESET = "2m"
# CPU inference for 2m ~0.15s/char; allow headroom for load + 800-char max.
GENERATION_TIMEOUT = int(os.environ.get("GENERATION_TIMEOUT", "180"))

app = FastAPI(title="tiramisu Shakespeare demo")

class GenerateRequest(BaseModel):
  prompt: str = Field(default="First Citizen:\n", max_length=200)
  temperature: float = Field(default=0.8, ge=0.1,  le=1.5)
  max_chars: int = Field(default=150, ge=50, le=800)

class GenerateResponse(BaseModel):
  text: str
  prompt: str

def call_binary(prompt: str, temperature: float, max_chars: int) -> str:
  if not BINARY.exists():
    raise RuntimeError(f"Binary not found at {BINARY}. Did you run cmake --build?")
  if not CHECKPOINT.exists():
    raise RuntimeError(f"Checkpoint not found at {CHECKPOINT}")
  if not DATA.exists():
    raise RuntimeError(f"Corpus not found at {DATA}")

  cmd = [
    str(BINARY),
    "--preset", PRESET,
    "--data", str(DATA),
    "--checkpoint", str(CHECKPOINT),
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
  
  # Output format: \n-- sample (prompt: ...) ---\n{text}\n
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
  wasm_js = WASM_DIR / "tiramisu_infer.js"
  wasm_bin = WASM_DIR / "tiramisu_infer.wasm"
  return {
    "binary": BINARY.exists(),
    "checkpoint": CHECKPOINT.exists(),
    "data": DATA.exists(),
    "preset": PRESET,
    "wasm_js": wasm_js.exists(),
    "wasm_bin": wasm_bin.exists(),
  }

@app.get("/tiramisu_infer.js")
async def wasm_js():
  path = WASM_DIR / "tiramisu_infer.js"
  if not path.exists():
    raise HTTPException(
      status_code=404,
      detail="WASM not built. Run: cmake --build build-wasm --target tiramisu_infer",
    )
  return FileResponse(path, media_type="application/javascript")

@app.get("/tiramisu_infer.wasm")
async def wasm_binary():
  path = WASM_DIR / "tiramisu_infer.wasm"
  if not path.exists():
    raise HTTPException(
      status_code=404,
      detail="WASM not built. Run: cmake --build build-wasm --target tiramisu_infer",
    )
  return FileResponse(path, media_type="application/wasm")

@app.get("/assets/shakespeare_2m.ckpt")
async def wasm_checkpoint():
  if not CHECKPOINT.exists():
    raise HTTPException(status_code=404, detail=f"Checkpoint not found at {CHECKPOINT}")
  return FileResponse(CHECKPOINT, media_type="application/octet-stream")

app.mount("/", StaticFiles(directory=Path(__file__).parent / "static", html=True))

if __name__ == "__main__":
  import uvicorn
  port = int(os.environ.get("PORT", 8000))
  uvicorn.run(app, host="0.0.0.0", port=port)
