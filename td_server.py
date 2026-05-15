"""
SAM3 TouchDesigner Server
-------------------------
FastAPI server that loads SAM3 and exposes it over HTTP so TouchDesigner
can send frames and receive segmentation masks.

Start with:
    start_server.bat
or:
    C:/Users/Zach/.conda/envs/sam3/python.exe -m uvicorn td_server:app --host 0.0.0.0 --port 8765 --reload
"""

import io
import base64
import logging

import numpy as np
import torch
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sam3-td-server")

app = FastAPI(title="SAM3 TouchDesigner Server")

# ---------------------------------------------------------------------------
# Model state – loaded once at startup
# ---------------------------------------------------------------------------

model_state = {
    "processor": None,
    "ready": False,
}


@app.on_event("startup")
async def load_model():
    logger.info("Loading SAM3 model...")
    try:
        from sam3.model_builder import build_sam3_image_model
        from sam3.model.sam3_image_processor import Sam3Processor

        model = build_sam3_image_model()
        model_state["processor"] = Sam3Processor(model)
        model_state["ready"] = True
        logger.info("SAM3 ready on %s", "cuda" if torch.cuda.is_available() else "cpu")
    except Exception as e:
        logger.error("Failed to load SAM3: %s", e)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Distinct colors for up to 20 instances (RGB)
_INSTANCE_COLORS = [
    (255,  59,  48), (255, 149,   0), ( 52, 199,  89), ( 90, 200, 250),
    (  0, 122, 255), (175,  82, 222), (255,  45,  85), (255, 204,   0),
    ( 48, 176, 199), (162, 132, 234), ( 50, 173, 230), ( 99, 230,  19),
    (255, 159,  10), (191,  90, 242), ( 30, 212, 145), (255,  69,  58),
    ( 10, 132, 255), ( 48, 209,  88), (100, 210, 255), (205, 193, 255),
]


def _to_tensor(x):
    if isinstance(x, torch.Tensor):
        return x.cpu().numpy()
    return np.array(x) if not isinstance(x, np.ndarray) else x


def normalize_masks(masks) -> np.ndarray:
    """
    Normalize SAM3 mask output to a clean (N, H, W) bool array.
    Handles extra size-1 dimensions that SAM3 sometimes adds anywhere.
    """
    arr = _to_tensor(masks)
    # Squeeze ALL size-1 dims, then reshape to (N, H, W)
    arr = arr.squeeze()
    if arr.ndim == 2:
        arr = arr[np.newaxis]   # (H, W) → (1, H, W)
    return arr.astype(bool)


def pil_to_b64(img: Image.Image, fmt: str = "PNG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def mask_to_grayscale_b64(mask_2d: np.ndarray) -> str:
    """Single (H, W) bool mask → base64 grayscale PNG."""
    img = Image.fromarray((mask_2d.astype(np.uint8) * 255), mode="L")
    return pil_to_b64(img)


def build_composite(masks: np.ndarray) -> Image.Image:
    """
    Color-coded RGBA composite: each instance gets a unique color, alpha = presence.
    Instances are drawn back-to-front so lower-index masks sit on top.
    """
    H, W = masks.shape[1], masks.shape[2]
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    for i in range(len(masks) - 1, -1, -1):  # draw back-to-front
        color = _INSTANCE_COLORS[i % len(_INSTANCE_COLORS)]
        m = masks[i]  # (H, W) bool
        canvas[m, 0] = color[0]
        canvas[m, 1] = color[1]
        canvas[m, 2] = color[2]
        canvas[m, 3] = 200  # semi-transparent

    return Image.fromarray(canvas, mode="RGBA")


def build_merged(masks: np.ndarray) -> Image.Image:
    """All instances collapsed to a single white-on-transparent RGBA mask."""
    combined = masks.any(axis=0).astype(np.uint8) * 255
    rgba = np.zeros((*combined.shape, 4), dtype=np.uint8)
    rgba[..., 0] = 255
    rgba[..., 3] = combined
    return Image.fromarray(rgba, mode="RGBA")


def build_textured_instance(mask: np.ndarray, source: np.ndarray) -> Image.Image:
    """
    Single instance textured cutout: masked pixels show the original source
    image colors, unmasked pixels are fully transparent.

    mask   – (H, W) bool
    source – (H, W, 3) uint8 RGB array
    """
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    rgba[..., :3] = source
    rgba[..., 3]  = mask.astype(np.uint8) * 255
    return Image.fromarray(rgba, mode="RGBA")


def build_textured_composite(masks: np.ndarray, source: np.ndarray) -> Image.Image:
    """
    All instances combined as a textured cutout: every masked pixel shows
    its original source color at full opacity, unmasked pixels are transparent.
    Lower-index instances are drawn on top when masks overlap.
    """
    H, W = masks.shape[1], masks.shape[2]
    canvas = np.zeros((H, W, 4), dtype=np.uint8)

    for i in range(len(masks) - 1, -1, -1):  # back-to-front; idx 0 wins overlaps
        m = masks[i]
        canvas[m, :3] = source[m]
        canvas[m,  3] = 255

    return Image.fromarray(canvas, mode="RGBA")


def run_inference(pil_img: Image.Image, prompt: str):
    """Run SAM3 and return (masks_nhw, scores_list, boxes_list)."""
    processor = model_state["processor"]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        state = processor.set_image(pil_img)
        output = processor.set_text_prompt(state=state, prompt=prompt)

    masks = normalize_masks(output["masks"])

    raw_scores = output.get("scores", [])
    scores = _to_tensor(raw_scores).flatten().tolist() if len(raw_scores) else []

    raw_boxes = output.get("boxes", [])
    if len(raw_boxes):
        boxes_arr = _to_tensor(raw_boxes)
        # Squeeze same leading dims as masks
        while boxes_arr.ndim > 2 and boxes_arr.shape[0] == 1:
            boxes_arr = boxes_arr[0]
        boxes = boxes_arr.tolist()
    else:
        boxes = []

    # Align lengths (pad with None if needed)
    n = len(masks)
    scores = (scores + [None] * n)[:n]
    boxes  = (boxes  + [None] * n)[:n]

    return masks, scores, boxes


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {"ready": model_state["ready"], "cuda": torch.cuda.is_available()}


@app.post("/segment")
async def segment(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    include_composite: bool = Form(True,  description="Color-coded flat composite"),
    include_textured:  bool = Form(True,  description="Source-color textured composite + per-instance cutouts"),
    include_merged:    bool = Form(False, description="Legacy single merged mask"),
):
    """
    Segment objects in an image using a text prompt.

    Returns JSON with per-instance data:

        num_masks   – total number of detected instances
        instances   – list of per-instance objects:
            id              – 0-based index
            score           – confidence (float or null)
            box             – [x1, y1, x2, y2] in pixels (or null)
            color           – [R, G, B] used in the composite
            mask_b64        – grayscale PNG; white = masked
            textured_b64    – RGBA PNG; masked pixels = original source colors (if include_textured)
        composite_b64           – flat color-coded RGBA (if include_composite)
        textured_composite_b64  – all instances with real source colors (if include_textured)
        merged_mask_b64         – single merged mask (if include_merged)
    """
    if not model_state["ready"]:
        raise HTTPException(503, "Model not ready yet")

    raw = await image.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    source_np = np.array(pil_img)  # (H, W, 3) uint8, kept for textured outputs

    try:
        masks, scores, boxes = run_inference(pil_img, prompt)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    instances = []
    for i, mask in enumerate(masks):
        color = _INSTANCE_COLORS[i % len(_INSTANCE_COLORS)]
        inst = {
            "id":       i,
            "score":    scores[i],
            "box":      boxes[i],
            "color":    list(color),
            "mask_b64": mask_to_grayscale_b64(mask),
        }
        if include_textured:
            inst["textured_b64"] = pil_to_b64(build_textured_instance(mask, source_np))
        instances.append(inst)

    response = {
        "num_masks": len(masks),
        "instances": instances,
    }

    if include_composite:
        response["composite_b64"] = pil_to_b64(build_composite(masks)) if len(masks) else None

    if include_textured:
        response["textured_composite_b64"] = (
            pil_to_b64(build_textured_composite(masks, source_np)) if len(masks) else None
        )

    if include_merged:
        response["merged_mask_b64"] = pil_to_b64(build_merged(masks)) if len(masks) else None

    return JSONResponse(response)


@app.post("/segment_raw")
async def segment_raw(
    image: UploadFile = File(...),
    prompt: str = Form(...),
    mode: str = Form(
        "textured",
        description="'composite' | 'textured' | 'merged' | 'instance:N' | 'textured:N'",
    ),
):
    """
    Returns a single PNG directly (no JSON wrapper).

    mode options:
        textured    – all instances with original source colors, transparent bg (default)
        composite   – flat color-coded RGBA of all instances
        merged      – single white-on-transparent RGBA mask
        instance:N  – grayscale mask for instance N (0-based)
        textured:N  – source-color cutout for instance N only
    """
    if not model_state["ready"]:
        raise HTTPException(503, "Model not ready yet")

    raw = await image.read()
    pil_img = Image.open(io.BytesIO(raw)).convert("RGB")
    source_np = np.array(pil_img)

    try:
        masks, scores, boxes = run_inference(pil_img, prompt)
    except Exception as e:
        raise HTTPException(500, f"Inference error: {e}")

    if len(masks) == 0:
        blank = Image.new("RGBA", (pil_img.width, pil_img.height), (0, 0, 0, 0))
        buf = io.BytesIO()
        blank.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")

    if mode == "textured":
        out_img = build_textured_composite(masks, source_np)
    elif mode == "composite":
        out_img = build_composite(masks)
    elif mode == "merged":
        out_img = build_merged(masks)
    elif mode.startswith("instance:"):
        idx = int(mode.split(":")[1])
        if idx >= len(masks):
            raise HTTPException(400, f"Instance {idx} not found (only {len(masks)} detected)")
        out_img = Image.fromarray((masks[idx].astype(np.uint8) * 255), mode="L")
    elif mode.startswith("textured:"):
        idx = int(mode.split(":")[1])
        if idx >= len(masks):
            raise HTTPException(400, f"Instance {idx} not found (only {len(masks)} detected)")
        out_img = build_textured_instance(masks[idx], source_np)
    else:
        raise HTTPException(
            400,
            f"Unknown mode '{mode}'. Use 'textured', 'composite', 'merged', 'instance:N', or 'textured:N'",
        )

    buf = io.BytesIO()
    out_img.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
