# SAM3 × TouchDesigner

Real-time open-vocabulary segmentation inside TouchDesigner, powered by [SAM3](https://ai.meta.com/sam3) (Segment Anything with Concepts) from Meta.

Send a text prompt like `"person"` or `"red ball"` — get back per-instance segmentation masks and textured cutouts as live TOPs in your network.

## How it works

A FastAPI server loads SAM3 onto your GPU and listens on `localhost:8765`. A Text DAT in TouchDesigner captures frames, posts them to the server on a background thread, and automatically creates Movie File In TOPs for each detected instance — all without blocking the TD main thread.

```
TD source TOP  →  Text DAT (sam3_td.py)  →  POST /segment
                                                    │
                                              SAM3 on GPU
                                                    │
                                         per-instance masks + cutouts
                                                    │
                                        auto-created Movie File In TOPs
```

## Requirements

- Windows (tested on Windows 11)
- NVIDIA GPU with CUDA 12.6+ (tested on RTX 3090)
- [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or Anaconda
- TouchDesigner 2022 or later
- HuggingFace account with access to [facebook/sam3](https://huggingface.co/facebook/sam3)

## Setup

**1. Request model access**

Go to https://huggingface.co/facebook/sam3 and request access. Approval is usually quick.

**2. Clone this repo with submodules**

```bash
git clone --recurse-submodules https://github.com/DE-YAN-Studio/sam3-td.git
cd sam3-td
```

Or if you already cloned without `--recurse-submodules`:

```bash
git submodule update --init
```

**3. Run setup**

Double-click `setup.bat`. It will:
- Create a `sam3` conda environment with Python 3.12
- Install PyTorch 2.7 + CUDA 12.6
- Install SAM3 and all server dependencies
- Prompt you to log in to HuggingFace

**4. Verify**

Double-click `test.bat` to confirm the model loads and runs inference correctly on your machine.

## Running the server

Double-click `start_server.bat`. The first startup takes ~30 seconds while SAM3 loads onto the GPU. You'll see:

```
INFO: SAM3 ready on cuda
INFO: Uvicorn running on http://0.0.0.0:8765
```

Check http://localhost:8765/health to confirm it's ready.

## TouchDesigner setup

1. Create a **Text DAT** in your network, name it `sam3`
2. Paste the contents of `td_scripts/sam3_td.py` into it
3. Call it from a Panel Execute DAT, CHOP Execute DAT, or the Textport:

```python
mod('sam3').sam3_segment(
    top_name  = 'videodevicein1',   # your source TOP
    prompt    = 'person',
    script_op = op('sam3'),         # enables auto TOP creation
)
```

After the first inference, these TOPs appear automatically as siblings of the Text DAT:

| TOP | Content |
|---|---|
| `sam3_textured_composite` | All instances with original source colors, transparent background |
| `sam3_composite` | Flat color-coded overlay, all instances |
| `sam3_instance_0`, `_1`, ... | Grayscale mask per instance |
| `sam3_textured_0`, `_1`, ... | Source-color cutout per instance |

TOPs are created and removed automatically as the instance count changes between frames.

## Controlling outputs

Pass a `modes` set to request only what you need — fewer outputs means less data over HTTP:

```python
# Textured composite only (fastest)
mod('sam3').sam3_segment('cam1', 'person', op('sam3'),
    modes={'textured_composite'})

# Grayscale masks only
mod('sam3').sam3_segment('cam1', 'person', op('sam3'),
    modes={'masks'})

# Everything (default)
mod('sam3').sam3_segment('cam1', 'person', op('sam3'))
```

Available modes: `composite`, `textured_composite`, `masks`, `textured`

## Triggering inference

The call is non-blocking — it returns immediately and runs in the background. If a request is already in flight, the call is skipped.

```python
# Every 60 frames (~2 sec at 30fps) — CHOP Execute DAT
def onFrameStart(frame):
    if frame % 60 == 0:
        mod('sam3').sam3_segment('videodevicein1', 'person', op('sam3'))

# On button press — Panel Execute DAT
def onValueChange(panelValue):
    if panelValue == 1:
        mod('sam3').sam3_segment('videodevicein1', 'person', op('sam3'))

# Check if inference is in flight
if mod('sam3').is_running():
    print('still processing...')
```

## Server API

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | `{ ready, cuda }` |
| `/segment` | POST | JSON response with per-instance data |
| `/segment_raw` | POST | Single PNG returned directly |

### /segment fields

| Field | Type | Default | Description |
|---|---|---|---|
| `image` | file | required | JPEG or PNG |
| `prompt` | string | required | Text prompt e.g. `"red ball"` |
| `include_composite` | bool | true | Flat color-coded RGBA composite |
| `include_textured` | bool | true | Source-color cutouts |
| `include_merged` | bool | false | Single merged mask |

### /segment response

```json
{
  "num_masks": 2,
  "instances": [
    {
      "id": 0,
      "score": 0.91,
      "box": [120, 340, 480, 720],
      "color": [255, 59, 48],
      "mask_b64": "...",
      "textured_b64": "..."
    }
  ],
  "composite_b64": "...",
  "textured_composite_b64": "..."
}
```

### /segment_raw modes

`textured` · `composite` · `merged` · `instance:N` · `textured:N`

## Performance

Tested on RTX 3090 at 1080p:

| Phase | Time |
|---|---|
| First inference (CUDA warmup) | ~2–3 s |
| Subsequent inferences | ~0.8–1.5 s |

SAM3 is not designed for real-time use at every frame. Trigger every 30–60 frames for a responsive feel without overloading the GPU.

## Credits

SAM3 model and original codebase by Meta — [paper](https://ai.meta.com/research/publications/sam-3-segment-anything-with-concepts/) · [project page](https://ai.meta.com/sam3)

TouchDesigner integration by [Zachary Kampf](https://github.com/kampfz).
