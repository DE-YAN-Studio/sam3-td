"""
SAM3 TouchDesigner Integration Script
--------------------------------------
Paste this into a Text DAT in TouchDesigner (name it e.g. "sam3").

After each inference this script will:
  - Write PNG files to the temp folder
  - Automatically create/update/remove Movie File In TOPs in the same
    network as the Script DAT

TOP naming convention (created as siblings of this Script DAT):
  sam3_composite           – flat color-coded RGBA, all instances
  sam3_textured_composite  – all instances with real source colors, transparent bg
  sam3_instance_0          – grayscale mask, instance 0
  sam3_instance_1          – grayscale mask, instance 1
  sam3_textured_0          – source-color cutout, instance 0
  sam3_textured_1          – source-color cutout, instance 1
  ...

Each TOP is set to Always Reload so it updates every inference.
"""

import urllib.request
import urllib.error
import base64
import json
import os
import tempfile
import threading

# ── Config ─────────────────────────────────────────────────────────────────
SERVER_URL = "http://localhost:8765"
TEMP_DIR   = tempfile.gettempdir()

# Prefix for all TOPs this script manages (so it never touches unrelated ops)
TOP_PREFIX = "sam3_"

# ── Async state ─────────────────────────────────────────────────────────────
_inference_thread = None   # currently running background thread
_is_running       = False  # True while a request is in flight


# ── Server helpers ──────────────────────────────────────────────────────────

def _check_server() -> bool:
    try:
        with urllib.request.urlopen(f"{SERVER_URL}/health", timeout=2) as resp:
            return json.loads(resp.read()).get("ready", False)
    except Exception:
        return False


def _encode_multipart(fields: dict, files: dict) -> tuple:
    boundary = "----SAM3TDBoundary7MA4YWxkTrZu0gW"
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


def _grab_frame(top_name: str) -> bytes:
    top_op = op(top_name)
    try:
        return top_op.saveByteArray(".jpg")
    except AttributeError:
        import tempfile as _tmp
        tmp = _tmp.NamedTemporaryFile(suffix=".jpg", delete=False)
        tmp.close()
        top_op.save(tmp.name)
        with open(tmp.name, "rb") as f:
            data = f.read()
        os.unlink(tmp.name)
        return data


# ── TOP management ──────────────────────────────────────────────────────────

def _make_top(parent_comp, name: str, file_path: str, x: int, y: int):
    """Create a Movie File In TOP and configure it, or update an existing one."""
    existing = parent_comp.op(name)
    if existing is None:
        top = parent_comp.create(moviefileinTOP, name)
        top.nodeX = x
        top.nodeY = y
    else:
        top = existing

    top.par.file   = file_path
    top.par.reload = 2        # 0=Don't reload  1=Pulse  2=Always
    return top


def _remove_top(parent_comp, name: str):
    """Delete a managed TOP if it exists."""
    existing = parent_comp.op(name)
    if existing is not None:
        existing.destroy()


def manage_tops(script_op, instances: list, composite_path: str | None,
                textured_composite_path: str | None, modes: set):
    """
    Create/update Movie File In TOPs for each detected instance.
    Only creates TOPs for output types listed in *modes*.
    Removes TOPs for instances that are no longer present.

    Column layout:
      col 0  sam3_composite / sam3_textured_composite
      col 1  sam3_instance_N   (grayscale masks)
      col 2  sam3_textured_N   (source-color cutouts)
    """
    parent_comp = script_op.parent()
    base_x = script_op.nodeX + 200
    base_y = script_op.nodeY
    col    = 220
    row    = 150

    # ── Composite TOPs ────────────────────────────────────────────────────
    for name, path, mode_key in [
        (f"{TOP_PREFIX}composite",          composite_path,          "composite"),
        (f"{TOP_PREFIX}textured_composite", textured_composite_path, "textured_composite"),
    ]:
        offset = 0 if mode_key == "composite" else row
        if mode_key in modes and path and os.path.exists(path):
            _make_top(parent_comp, name, path, base_x, base_y - offset)
        else:
            _remove_top(parent_comp, name)

    # ── Per-instance TOPs ─────────────────────────────────────────────────
    new_ids = {inst["id"] for inst in instances}

    # Remove stale TOPs
    idx = 0
    while True:
        found_any = False
        for prefix in ("instance_", "textured_"):
            existing = parent_comp.op(f"{TOP_PREFIX}{prefix}{idx}")
            if existing is not None:
                found_any = True
                if idx not in new_ids:
                    existing.destroy()
        if not found_any and idx > max(new_ids, default=-1):
            break
        idx += 1

    # Create/update TOPs for current instances
    for inst in instances:
        i = inst["id"]
        y = base_y - i * row

        if "masks" in modes:
            _make_top(
                parent_comp,
                f"{TOP_PREFIX}instance_{i}",
                os.path.join(TEMP_DIR, f"sam3_instance_{i}.png"),
                base_x + col, y,
            )
        else:
            _remove_top(parent_comp, f"{TOP_PREFIX}instance_{i}")

        if "textured" in modes and inst.get("textured_b64") is not None:
            _make_top(
                parent_comp,
                f"{TOP_PREFIX}textured_{i}",
                os.path.join(TEMP_DIR, f"sam3_textured_{i}.png"),
                base_x + col * 2, y,
            )
        else:
            _remove_top(parent_comp, f"{TOP_PREFIX}textured_{i}")


# ── Main inference call ─────────────────────────────────────────────────────

def _do_segment(img_bytes: bytes, prompt: str, script_op_path: str | None,
                modes: set):
    """Blocking worker — runs on a background thread."""
    global _is_running

    if not _check_server():
        print("SAM3: server not ready – run start_server.bat")
        _is_running = False
        return

    need_composite = "composite"          in modes
    need_textured  = "textured"           in modes or "textured_composite" in modes

    body, content_type = _encode_multipart(
        fields={
            "prompt":            prompt,
            "include_composite": "true" if need_composite else "false",
            "include_textured":  "true" if need_textured  else "false",
            "include_merged":    "false",
        },
        files={"image": ("frame.jpg", img_bytes, "image/jpeg")},
    )

    req = urllib.request.Request(
        f"{SERVER_URL}/segment",
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
    except urllib.error.HTTPError as e:
        print(f"SAM3: HTTP {e.code} – {e.read().decode()}")
        _is_running = False
        return
    except Exception as e:
        print(f"SAM3: request error – {e}")
        _is_running = False
        return

    # ── Write PNGs to disk ────────────────────────────────────────────────
    def _save(b64: str, filename: str) -> str:
        path = os.path.join(TEMP_DIR, filename)
        with open(path, "wb") as f:
            f.write(base64.b64decode(b64))
        return path

    composite_path          = _save(result["composite_b64"],          "sam3_composite.png")          if result.get("composite_b64")          else None
    textured_composite_path = _save(result["textured_composite_b64"], "sam3_textured_composite.png") if result.get("textured_composite_b64") else None

    for inst in result.get("instances", []):
        _save(inst["mask_b64"], f"sam3_instance_{inst['id']}.png")
        if inst.get("textured_b64"):
            _save(inst["textured_b64"], f"sam3_textured_{inst['id']}.png")

    # ── Manage TOPs (must happen on TD's main thread via run()) ───────────
    if script_op_path is not None:
        instances               = result.get("instances", [])
        n                       = result["num_masks"]

        def _update_tops():
            script_op = op(script_op_path)
            if script_op:
                manage_tops(script_op, instances,
                            composite_path, textured_composite_path, modes)

        run(_update_tops, delayFrames=1)

    # ── Log ───────────────────────────────────────────────────────────────
    n = result["num_masks"]
    print(f"SAM3: {n} instance(s) detected")
    for inst in result.get("instances", []):
        box = inst.get("box")
        box_str = f"  box={[round(v) for v in box]}" if box else ""
        print(f"  [{inst['id']}] score={inst['score']:.3f}  color={inst['color']}{box_str}")

    _is_running = False


def sam3_segment(top_name: str = "videodevicein1", prompt: str = "person",
                 script_op=None,
                 modes: set = {"composite", "textured_composite", "masks", "textured"}) -> bool:
    """
    Non-blocking: captures a frame immediately, then fires the server request
    on a background thread so TouchDesigner stays responsive.

    Returns True if the request was dispatched, False if skipped (server not
    ready, or a previous request is still in flight).

    Args:
        top_name   – name of the source TOP operator
        prompt     – text prompt, e.g. "person" or "red ball"
        script_op  – pass op('sam3') to enable automatic TOP creation
        modes      – which outputs to request/create TOPs for:
                       "composite"          flat color-coded RGBA
                       "textured_composite" all instances with real source colors
                       "masks"              per-instance grayscale masks
                       "textured"           per-instance source-color cutouts

    Examples:
        # Textured composite only (fastest)
        sam3_segment('cam1', 'person', op('sam3'), modes={'textured_composite'})

        # Everything (default)
        sam3_segment('cam1', 'person', op('sam3'))
    """
    global _inference_thread, _is_running

    if _is_running:
        return False  # previous request still in flight — skip this frame

    try:
        img_bytes = _grab_frame(top_name)
    except Exception as e:
        print(f"SAM3: cannot grab frame from '{top_name}': {e}")
        return False

    # Resolve the op path now (on the main thread) so the background thread
    # can look it up safely via op() later
    script_op_path = script_op.path if script_op is not None else None

    _is_running = True
    _inference_thread = threading.Thread(
        target=_do_segment,
        args=(img_bytes, prompt, script_op_path, modes),
        daemon=True,
    )
    _inference_thread.start()
    return True


def is_running() -> bool:
    """Returns True while a SAM3 request is in flight."""
    return _is_running


