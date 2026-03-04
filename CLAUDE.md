# CLAUDE.md

## Project

SAM3 (Segment Anything with Concepts) from Meta, integrated with TouchDesigner via a local HTTP server. SAM3 runs in a conda environment on the GPU; TouchDesigner communicates with it over HTTP.

## Environment

- Conda env: `sam3`
- Python: 3.12
- PyTorch: 2.7 + CUDA 12.6
- Conda executable: found dynamically by the `.bat` scripts (searches common install paths)
- To run anything: `conda run -n sam3 python <script>`

## Key Files

| File | Purpose |
|---|---|
| `td_server.py` | FastAPI server — loads SAM3 at startup, exposes `/segment`, `/segment_raw`, `/health` |
| `start_server.bat` | One-click server launcher (portable, finds conda automatically) |
| `setup.bat` | Full setup for a fresh machine — creates env, installs deps, prompts HF login |
| `test.bat` | Runs `test_sam3.py` via the conda env |
| `test_sam3.py` | 5-step model test: CUDA → import → load → inference → output validation |
| `test_pipeline.py` | End-to-end HTTP pipeline test — starts server, fires real requests, validates all output modes |
| `td_scripts/sam3_td.py` | TouchDesigner client — paste into a Text DAT named `sam3` |
| `sam3/model/edt.py` | **Modified from upstream** — triton import wrapped in try/except with scipy CPU fallback |

## Architecture

```
TouchDesigner
  Text DAT (sam3_td.py)
    └─ background thread ──► POST /segment  (localhost:8765)
                                  │
                             FastAPI server (td_server.py)
                                  │
                             SAM3 model (GPU)
                                  │
                             JSON response { instances[], composite_b64, textured_composite_b64 }
                                  │
                        write PNGs to %TEMP%
                                  │
                        td.run() → manage_tops()
                                  │
                        Movie File In TOPs (auto-created as siblings of the Text DAT)
```

## Server API

**POST /segment** — main endpoint
- Fields: `image` (file), `prompt` (str), `include_composite` (bool), `include_textured` (bool), `include_merged` (bool)
- Returns JSON with `instances[]` (per-instance `mask_b64`, `textured_b64`, `score`, `box`, `color`), `composite_b64`, `textured_composite_b64`

**POST /segment_raw** — returns a single PNG directly
- Fields: `image` (file), `prompt` (str), `mode` (str)
- Modes: `textured`, `composite`, `merged`, `instance:N`, `textured:N`

**GET /health** — returns `{ ready, cuda }`

## TouchDesigner Usage

```python
# In Textport or any Execute/Panel DAT:
mod('sam3').sam3_segment(
    top_name  = 'videodevicein1',
    prompt    = 'person',
    script_op = op('sam3'),           # enables auto TOP creation
    modes     = {'textured_composite'}  # optional — controls which TOPs are created
)
```

The call returns immediately (non-blocking). Inference runs on a background thread. If a request is already in flight, subsequent calls are skipped.

## Auto-created TOPs

After each inference, these Movie File In TOPs are created/updated as siblings of the Text DAT:

| Name | Content |
|---|---|
| `sam3_composite` | Flat color-coded RGBA, all instances |
| `sam3_textured_composite` | All instances with real source colors, transparent bg |
| `sam3_instance_N` | Grayscale mask for instance N |
| `sam3_textured_N` | Source-color cutout for instance N |

All TOPs use `reload = 2` (Always Reload).

## Known Issues / Patches

### triton not available on Windows
`sam3/model/edt.py` has a hard `import triton` which fails on Windows. Patched to wrap in `try/except` with a `scipy.ndimage.distance_transform_edt` CPU fallback. The triton path is only used during video tracking (`fill_holes_in_mask_scores`), not image inference.

### Mask shape normalization
SAM3 sometimes returns masks with extra leading size-1 dimensions (e.g. `(1, 1, H, W)`). The `normalize_masks()` function in `td_server.py` calls `.squeeze()` before reshaping to `(N, H, W)`.

## Model Weights

Weights are gated on HuggingFace. Requires:
1. Access request at https://huggingface.co/facebook/sam3
2. Auth: `conda run -n sam3 hf auth login`

Weights are downloaded automatically by `build_sam3_image_model()` on first run and cached by HuggingFace hub.
