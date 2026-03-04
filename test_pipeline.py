"""
SAM3 Full Pipeline Test
-----------------------
Tests the complete path:
  test image → HTTP POST → SAM3 server → response validation → output PNGs saved

Run with:
    C:/Users/Zach/.conda/envs/sam3/python.exe test_pipeline.py

If the server is already running on port 8765 it will be used directly.
Otherwise the script starts it as a subprocess, waits for it to be ready,
runs the tests, then shuts it down.

Output images are saved to:
    C:/Users/Zach/Desktop/Meta_AI/Sam-TD/test_pipeline_output/
"""

import sys
import os
import io
import base64
import json
import time
import subprocess
import urllib.request
import urllib.error

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):      print(f"  {GREEN}[OK]{RESET}   {msg}")
def fail(msg):    print(f"  {RED}[FAIL]{RESET} {msg}"); sys.exit(1)
def info(msg):    print(f"  {CYAN}-->{RESET}    {msg}")
def warn(msg):    print(f"  {YELLOW}[!]{RESET}    {msg}")
def section(t):   print(f"\n{CYAN}{'-'*52}{RESET}\n  {t}\n{CYAN}{'-'*52}{RESET}")

# ── Config ────────────────────────────────────────────────────────────────────
SERVER_URL  = "http://localhost:8765"
PYTHON_EXE  = sys.executable
SERVER_DIR  = os.path.dirname(os.path.abspath(__file__))
TEST_IMAGE  = os.path.join(SERVER_DIR, "sam3", "assets", "images", "truck.jpg")
PROMPT      = "truck"
OUT_DIR     = os.path.join(SERVER_DIR, "test_pipeline_output")
os.makedirs(OUT_DIR, exist_ok=True)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _health() -> dict | None:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=3) as r:
            return json.loads(r.read())
    except Exception:
        return None


def _encode_multipart(fields: dict, files: dict) -> tuple:
    boundary = "----PipelineTestBoundary"
    body = b""
    for name, value in fields.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        body += f"{value}\r\n".encode()
    for name, (filename, data, mime) in files.items():
        body += f"--{boundary}\r\n".encode()
        body += f'Content-Disposition: form-data; name="{name}"; filename="{filename}"\r\n'.encode()
        body += f"Content-Type: {mime}\r\n\r\n".encode()
        body += data + b"\r\n"
    body += f"--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


def _decode_png(b64: str, name: str) -> "Image":
    from PIL import Image
    data = base64.b64decode(b64)
    img  = Image.open(io.BytesIO(data))
    path = os.path.join(OUT_DIR, name)
    img.save(path)
    return img


# ── 1. Server startup ─────────────────────────────────────────────────────────
section("1 / 5  Server")

server_proc = None
health = _health()

if health and health.get("ready"):
    ok("Server already running and model ready")
else:
    info("Starting server…")
    server_proc = subprocess.Popen(
        [PYTHON_EXE, "-m", "uvicorn", "td_server:app",
         "--host", "0.0.0.0", "--port", "8765",
         "--app-dir", SERVER_DIR],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    deadline = time.time() + 120
    while time.time() < deadline:
        h = _health()
        if h and h.get("ready"):
            break
        if server_proc.poll() is not None:
            out = server_proc.stdout.read()
            fail(f"Server process exited early:\n{out}")
        time.sleep(2)
        print("    waiting for model to load…", flush=True)
    else:
        server_proc.terminate()
        fail("Server did not become ready within 120 s")

    ok("Server ready")

health = _health()
info(f"CUDA available: {health.get('cuda')}")


# ── 2. POST /segment ──────────────────────────────────────────────────────────
section("2 / 5  POST /segment")

if not os.path.exists(TEST_IMAGE):
    fail(f"Test image not found: {TEST_IMAGE}")

with open(TEST_IMAGE, "rb") as f:
    img_bytes = f.read()

info(f"Image:  {TEST_IMAGE}")
info(f"Prompt: '{PROMPT}'")

body, ct = _encode_multipart(
    fields={
        "prompt":            PROMPT,
        "include_composite": "true",
        "include_textured":  "true",
        "include_merged":    "true",
    },
    files={"image": ("truck.jpg", img_bytes, "image/jpeg")},
)

req = urllib.request.Request(
    f"{SERVER_URL}/segment",
    data=body,
    headers={"Content-Type": ct},
    method="POST",
)

t0 = time.time()
try:
    with urllib.request.urlopen(req, timeout=60) as resp:
        result = json.loads(resp.read())
except urllib.error.HTTPError as e:
    fail(f"HTTP {e.code}: {e.read().decode()}")
except Exception as e:
    fail(f"Request failed: {e}")

elapsed = time.time() - t0
ok(f"Response received in {elapsed:.2f} s")


# ── 3. Validate response structure ────────────────────────────────────────────
section("3 / 5  Response Validation")

# Top-level fields
for field in ("num_masks", "instances", "composite_b64",
              "textured_composite_b64", "merged_mask_b64"):
    if field not in result:
        fail(f"Missing top-level field: '{field}'")
ok("All top-level fields present")

n = result["num_masks"]
info(f"num_masks = {n}")
if n == 0:
    warn("No instances detected — try a different prompt or check the model")
else:
    ok(f"{n} instance(s) detected")

# Per-instance fields
for inst in result["instances"]:
    for field in ("id", "score", "box", "color", "mask_b64", "textured_b64"):
        if field not in inst:
            fail(f"Instance {inst.get('id')} missing field: '{field}'")
    score = inst["score"]
    if not (0.0 <= score <= 1.0):
        fail(f"Instance {inst['id']} score out of range: {score}")
    if len(inst["color"]) != 3:
        fail(f"Instance {inst['id']} color is not RGB: {inst['color']}")
    if inst["box"] is not None and len(inst["box"]) != 4:
        fail(f"Instance {inst['id']} box is not [x1,y1,x2,y2]: {inst['box']}")

ok("All instance fields valid")

for inst in result["instances"]:
    box = inst.get("box")
    box_str = f"  box=[{', '.join(str(round(v)) for v in box)}]" if box else ""
    print(f"    [{inst['id']}] score={inst['score']:.3f}  color={inst['color']}{box_str}")


# ── 4. Decode and save all images ─────────────────────────────────────────────
section("4 / 5  Decode & Save Output Images")

from PIL import Image

def check_image(b64: str, name: str, expected_mode: str) -> Image.Image:
    img = _decode_png(b64, name)
    if img.mode != expected_mode:
        fail(f"{name}: expected mode {expected_mode}, got {img.mode}")
    ok(f"{name}  ({img.width}x{img.height}  {img.mode})")
    return img

check_image(result["composite_b64"],          "composite.png",          "RGBA")
check_image(result["textured_composite_b64"],  "textured_composite.png", "RGBA")
check_image(result["merged_mask_b64"],         "merged_mask.png",        "RGBA")

for inst in result["instances"]:
    i = inst["id"]
    check_image(inst["mask_b64"],     f"instance_{i}_mask.png",     "L")
    check_image(inst["textured_b64"], f"instance_{i}_textured.png", "RGBA")

# Verify textured composite has transparent bg where no mask exists
import numpy as np
tc_np = np.array(Image.open(os.path.join(OUT_DIR, "textured_composite.png")))
if n > 0:
    # At least some pixels should be fully transparent
    transparent_pixels = (tc_np[..., 3] == 0).sum()
    if transparent_pixels == 0:
        warn("textured_composite has no transparent pixels — unexpected")
    else:
        ok(f"textured_composite has correct transparency ({transparent_pixels} transparent px)")


# ── 5. POST /segment_raw modes ────────────────────────────────────────────────
section("5 / 5  POST /segment_raw  (all modes)")

for mode in ["textured", "composite", "merged"] + [f"textured:{i}" for i in range(n)] + [f"instance:{i}" for i in range(n)]:
    body, ct = _encode_multipart(
        fields={"prompt": PROMPT, "mode": mode},
        files={"image": ("truck.jpg", img_bytes, "image/jpeg")},
    )
    req = urllib.request.Request(
        f"{SERVER_URL}/segment_raw",
        data=body, headers={"Content-Type": ct}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            raw_png = resp.read()
            ct_resp = resp.headers.get("content-type", "")
    except urllib.error.HTTPError as e:
        fail(f"segment_raw mode='{mode}' HTTP {e.code}: {e.read().decode()}")

    if "image/png" not in ct_resp:
        fail(f"segment_raw mode='{mode}' wrong content-type: {ct_resp}")

    img = Image.open(io.BytesIO(raw_png))
    fname = f"raw_{mode.replace(':', '_')}.png"
    img.save(os.path.join(OUT_DIR, fname))
    ok(f"mode={mode:<16}  {img.width}x{img.height}  {img.mode}")


# ── Cleanup & summary ─────────────────────────────────────────────────────────
if server_proc is not None:
    server_proc.terminate()
    server_proc.wait()
    info("Server stopped")

print(f"\n{GREEN}All pipeline tests passed!{RESET}")
print(f"  Output images saved to: {OUT_DIR}\n")
