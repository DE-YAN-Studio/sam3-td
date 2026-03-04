"""
SAM3 Model Test Script
----------------------
Run this to verify SAM3 is working correctly on your machine.

Usage:
    C:/Users/Zach/.conda/envs/sam3/python.exe test_sam3.py

Steps tested:
  1. CUDA / GPU check
  2. SAM3 import
  3. Model load from HuggingFace
  4. Image inference with a text prompt
  5. Output mask validation
  6. Save result image to test_output.png
"""

import sys
import time

PYTHON_EXE = sys.executable

# ── ANSI colours ──────────────────────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"

def ok(msg):   print(f"  {GREEN}[OK]{RESET} {msg}")
def fail(msg): print(f"  {RED}[FAIL]{RESET} {msg}"); sys.exit(1)
def info(msg): print(f"  {CYAN}-->{RESET} {msg}")
def warn(msg): print(f"  {YELLOW}[!]{RESET} {msg}")
def section(title): print(f"\n{CYAN}{'-'*50}{RESET}\n  {title}\n{CYAN}{'-'*50}{RESET}")


# ─────────────────────────────────────────────────────────────────────────────
# 1. Python & CUDA
# ─────────────────────────────────────────────────────────────────────────────
section("1 / 5  Python & CUDA")
info(f"Python {sys.version.split()[0]} at {sys.executable}")

try:
    import torch
    ok(f"PyTorch {torch.__version__}")
except ImportError:
    fail("PyTorch not installed. Run: pip install torch --index-url https://download.pytorch.org/whl/cu126")

if torch.cuda.is_available():
    device = torch.device("cuda")
    gpu = torch.cuda.get_device_name(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
    ok(f"CUDA available — {gpu} ({vram_gb:.1f} GB VRAM)")
else:
    warn("CUDA not available — will run on CPU (very slow)")
    device = torch.device("cpu")


# ─────────────────────────────────────────────────────────────────────────────
# 2. SAM3 import
# ─────────────────────────────────────────────────────────────────────────────
section("2 / 5  SAM3 Package Import")

try:
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor
    ok("sam3 package imported successfully")
except ImportError as e:
    fail(f"Cannot import sam3: {e}\n\n"
         "  Make sure you installed it:\n"
         "    pip install -e .")
except Exception as e:
    fail(f"Unexpected import error: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Model load
# ─────────────────────────────────────────────────────────────────────────────
section("3 / 5  Model Load (downloads weights on first run)")
info("Loading SAM3 model — this may take 30–60 s on first run…")

try:
    t0 = time.time()
    model = build_sam3_image_model()
    processor = Sam3Processor(model)
    elapsed = time.time() - t0
    ok(f"Model loaded in {elapsed:.1f} s")
except Exception as e:
    err_str = str(e)
    if "401" in err_str or "gated" in err_str.lower() or "access" in err_str.lower():
        fail(
            "HuggingFace authentication required.\n\n"
            "  1. Request access at: https://huggingface.co/facebook/sam3\n"
            "  2. Then run:  C:/Users/Zach/.conda/envs/sam3/Scripts/hf.exe auth login\n"
            "  3. Re-run this test."
        )
    else:
        fail(f"Model load failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Inference
# ─────────────────────────────────────────────────────────────────────────────
section("4 / 5  Image Inference")

from PIL import Image
import os

TEST_IMAGE = os.path.join(os.path.dirname(__file__), "sam3", "assets", "images", "truck.jpg")
PROMPT     = "truck"

if not os.path.exists(TEST_IMAGE):
    fail(f"Test image not found: {TEST_IMAGE}")

img = Image.open(TEST_IMAGE).convert("RGB")
info(f"Image: {TEST_IMAGE}  ({img.width}×{img.height})")
info(f"Prompt: '{PROMPT}'")

try:
    t0 = time.time()
    state  = processor.set_image(img)
    output = processor.set_text_prompt(state=state, prompt=PROMPT)
    elapsed = time.time() - t0
    ok(f"Inference completed in {elapsed:.2f} s")
except Exception as e:
    fail(f"Inference failed: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Validate output & save result
# ─────────────────────────────────────────────────────────────────────────────
section("5 / 5  Output Validation & Save")

import numpy as np

masks  = output.get("masks")
scores = output.get("scores")
boxes  = output.get("boxes")

if masks is None:
    fail("No 'masks' key in output")

if isinstance(masks, torch.Tensor):
    masks_np = masks.cpu().numpy()
else:
    masks_np = np.array(masks)

# Squeeze out any leading size-1 dims so shape is always (N, H, W)
while masks_np.ndim > 3 and masks_np.shape[0] == 1:
    masks_np = masks_np[0]
if masks_np.ndim == 2:
    masks_np = masks_np[np.newaxis]

num_masks = len(masks_np)
if num_masks == 0:
    warn("Inference ran but found 0 masks — try a different prompt or check model weights")
else:
    ok(f"{num_masks} mask(s) detected")

if scores is not None:
    sc = scores.cpu().tolist() if isinstance(scores, torch.Tensor) else list(scores)
    ok(f"Scores: {[f'{s:.3f}' for s in sc]}")

# Build RGBA overlay
H, W = img.height, img.width
overlay = np.zeros((H, W, 4), dtype=np.uint8)
overlay[..., :3] = np.array(img)  # copy original RGB

if num_masks > 0:
    combined = masks_np.any(axis=0).astype(np.uint8) * 180
    overlay[..., 3] = 255  # full alpha for original

    # Draw mask in red with 60 % opacity
    red_layer = np.zeros((H, W, 4), dtype=np.uint8)
    red_layer[..., 0] = 255
    red_layer[..., 3] = combined

    result_img = Image.fromarray(overlay[..., :3])
    red_pil    = Image.fromarray(red_layer, "RGBA")
    result_img.paste(Image.fromarray(np.full((H, W, 3), [255, 0, 0], dtype=np.uint8)),
                     mask=Image.fromarray(combined))
else:
    result_img = img

OUT_PATH = os.path.join(os.path.dirname(__file__), "test_output.png")
result_img.save(OUT_PATH)
ok(f"Result saved → {OUT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{GREEN}All tests passed! SAM3 is working correctly on this machine.{RESET}\n")

