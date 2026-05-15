# SAM 3.1 Realtime Streaming — Implementation Plan

Branch: `sam3p1`. Work paused 2026-05-15.

## Goal

Replace the current per-frame `/segment` HTTP path with a streaming session
backed by the SAM 3.1 multiplex video predictor, so TouchDesigner can do
realtime segmentation on a live camera feed with persistent instance IDs.

## Status

### Done (committed on `sam3p1`, `d332cae`)

- Submodule `vendor/sam3` bumped to upstream `main` (29 commits, includes
  SAM 3.1 release + post-release stability fixes). Triton fallback patch
  re-applied on top. Pushed to `DE-YAN-Studio/sam3` as branch `sam3p1`.
- Submodule clone moved from `sam3/` to `vendor/sam3/` — fixes a namespace-
  package collision where Python loaded `<repo>/sam3/` (no `__init__.py`) as
  a namespace package, shadowing the editable install and breaking
  `pkg_resources.resource_filename("sam3", ...)`.
- `td_server.py` and `test_sam3.py` now wrap inference in
  `torch.autocast("cuda", dtype=torch.bfloat16)`. Upstream notebooks now
  require this — without it, matmul fails because some checkpoint weights
  are stored as bfloat16. See `vendor/sam3/examples/sam3_image_predictor_example.ipynb:98-99`.
- `setup.bat`, `test_pipeline.py` paths updated to `vendor/sam3/...`.
- HF access to `facebook/sam3.1` requested and granted.
- Smoke test passes: model loads 7.3s, image inference 0.67s.

### Not done

The streaming work itself. Existing per-frame `/segment` path still works.

## The real blocker

True unbounded live streaming is **not feasible** with the multiplex predictor
as-shipped. The predictor preallocates `(num_frames, C, H, W)` as a single
dense tensor at session start (`vendor/sam3/sam3/model/sam3_multiplex_tracking.py:207`,
storage via `vendor/sam3/sam3/model/io_utils.py:31`). At default
`image_size=1008` fp16, that's ~330 MB per second of 30fps video — preallocating
10 min eats ~200 GB, doesn't fit on a 3090 even with CPU offload.

The tracker itself only needs a small window (`num_maskmem=7`,
`max_cond_frames_in_attn=4`, `max_trk_keep_alive=30` — see `build_tracker` call
in `model_builder.py:434`). The dense-tensor preallocation is an artifact of
the offline batch-processing design, not a tracker requirement.

## Three paths forward

### Option 1 — Periodic-restart sessions (recommended start)

Run e.g. 30-second sessions back-to-back. ~10 GB preallocation each. Tracker
memory resets at each boundary; re-detection from the text prompt re-grounds
objects, so visually it's a brief flicker, not a catastrophe.

- Effort: ~1–2 days
- LoC: ~200–300 (subclass + dispatch + TD client)
- Trade-off: instance IDs don't persist across boundaries — instance #3 at
  second 29 may become instance #5 at second 31 if objects re-detect in a
  different order. May or may not matter depending on TD use.

### Option 2 — Patch the predictor to use ring storage

Decouple frame storage (small ring, ~64–128 slots — sized to >
`max_trk_keep_alive=30`) from frame indexing (unbounded counter). Matches the
tracker's actual access pattern. Requires changing:

- `_construct_initial_input_batch` (`sam3_multiplex_tracking.py:106`) to
  allocate a ring tensor, not a dense one.
- `init_state` (`sam3_multiplex_tracking.py:207`) to accept streaming mode
  and skip resource loading.
- All per-frame metadata lists (`previous_stages_out`, `per_frame_raw_point_input`,
  `per_frame_cur_step` etc, `sam3_multiplex_tracking.py:172-178`) — these
  also need to become rings or dict-keyed-on-frame-idx structures.
- `_run_single_frame_inference` (`sam3_multiplex_base.py:430`) — verify it
  doesn't index past the ring window.

Effort: ~4–6 days. LoC: ~500–800 across 3–4 files. "Right" fix; persistent
instance IDs forever. Largest commitment.

### Option 3 — Skip video predictor, image model + custom tracker

Keep SAM3 image model loaded (what we do today, no per-call rebuild). Add a
small IoU-based or Hungarian-matching tracker in `td_server.py` to assign
stable IDs across frames.

- Effort: ~1–2 days
- LoC: ~150
- Trade-off: loses multiplex's joint reasoning and shared-memory speedup
  (back to per-frame image cost). Decent ID stability in slow scenes; poor
  under occlusion. Unbounded by design, no upstream patches.

## Recommendation

Start with Option 1. Get realtime streaming working end-to-end with multiplex
quality. If the boundary flicker is unacceptable in practice, escalate to
Option 2. Option 3 is the fallback if Option 2 turns out worse than expected.

## Task list (preserved in TaskList; reproduced here)

- [x] **#1** Bump sam3 submodule, re-apply triton patch
- [x] **#2** Spike push_frame against `Sam3VideoPredictorMultiGPU`
- [x] **#3** HF access to `facebook/sam3.1` — access granted (code-side wire-up
  trivial: `download_ckpt_from_hf(version="sam3.1")` already exists at
  `model_builder.py:657`)
- [ ] **#4** Add streaming endpoints to `td_server.py` (`/stream/start`,
  `/stream/frame`, `/stream/end`). Shape depends on chosen option.
- [ ] **#5** Rewrite TD client (`td_scripts/sam3_td.py`) for streaming session.
  Replace per-call POST with session opened on first call; push frames over
  HTTP keepalive (or websocket). Switch `manage_tops` keying from positional
  index to tracker `obj_id` so TOPs stop reshuffling across frames.
- [ ] **#6** Profile `fill_holes_in_mask_scores` scipy fallback. Video path
  hits triton on Linux; on Windows we fall back to scipy CPU
  (`vendor/sam3/sam3/model/edt.py`). May bottleneck at TD frame rates. If so:
  set `fill_hole_area=0` in the builder, or get a Windows triton build
  working.

## Resume steps

### 0. Verify env is intact (~5 min)

```powershell
git checkout sam3p1
git submodule update --init --recursive
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n sam3 hf auth whoami
& "C:\ProgramData\anaconda3\Scripts\conda.exe" run -n sam3 python test_sam3.py
```

Expect: `Not logged in` → re-run `hf auth login` (interactive, paste token).
Then `test_sam3.py` should print "All tests passed" with model load ~7s,
inference ~0.7s. If it fails: `pip install -e ./vendor/sam3 --no-deps` in
the sam3 conda env, then re-run.

### 1. Pick an option (5 min, just decide)

See "Three paths forward" above. Default to Option 1 unless you have a new
reason to skip it. The rest of this checklist assumes Option 1.

### 2. Build SessionManager in `td_server.py` (~4 hours)

New class in `td_server.py`, loaded lazily on first `/stream/start`:

- Constructor: build multiplex predictor via
  `build_sam3_multiplex_video_predictor()` (see `model_builder.py:1070`).
  Pass `version="sam3.1"` to `download_ckpt_from_hf` via `checkpoint_path`
  arg if needed.
- `start(prompt, max_frames=900)` (= 30s @ 30fps): call
  `predictor.handle_request({"type": "start_session", "resource_path": [PIL.Image.new("RGB",(W,H))] * max_frames})`,
  then `add_prompt` with the text. Returns `session_id` and current `frame_idx=0`.
- `push_frame(session_id, pil_img)`: write `pil_img` into the slot at the
  current frame_idx in `inference_state["input_batch"].img_batch.tensors`
  (after applying the normalize transform from `io_utils.py:54-65`), then
  call `_run_single_frame_inference(state, frame_idx, reverse=False, ...)`
  directly (skipping the resource-file iteration in `propagate_in_video`).
  Increment frame_idx. Return masks + per-instance obj_id + scores + boxes.
- If frame_idx hits max_frames: close session and start a new one with the
  same prompt. Boundary flicker is expected here.
- `end(session_id)`: `predictor.handle_request({"type": "close_session", ...})`.

Reuse the existing mask serialization helpers (`build_composite`,
`build_textured_composite` etc. in `td_server.py:98-154`).

### 3. Add HTTP endpoints (~1 hour)

In `td_server.py`:

- `POST /stream/start` — body: `{prompt}`. Returns `{session_id, frame_idx}`.
- `POST /stream/frame` — multipart: `session_id`, `image` (JPEG). Returns
  the same JSON shape as `/segment` today, plus `frame_idx` and a per-
  instance `obj_id` field. Keep `/segment` working for one-shots.
- `POST /stream/end` — body: `{session_id}`. Returns `{closed: true}`.

### 4. Rewrite TD client (~3 hours)

In `td_scripts/sam3_td.py`:

- Replace per-call POST flow with a session opened on first `sam3_segment()`
  call. Cache `session_id` on the module.
- Push frames over HTTP keepalive (use `http.client.HTTPConnection`,
  reuse one connection across calls — much faster than current per-call
  `urllib`).
- Switch `manage_tops` (`td_scripts/sam3_td.py:108-177`) keying from
  positional index to tracker `obj_id` so the `sam3_instance_N` and
  `sam3_textured_N` TOPs stop reshuffling when instance counts change.
- Drop the `_is_running` guard for frame submission — keepalive makes
  serial pushes cheap.

### 5. End-to-end smoke (~1 hour)

- Restart server: `start_server.bat`.
- Run `test_pipeline.py` against the old `/segment` to confirm nothing
  regressed there.
- Write a small new test (`test_stream.py` modelled on `test_pipeline.py`)
  that opens a stream, pushes 200 frames from a webcam capture or a video
  file decoded with cv2, and validates that `obj_id`s persist across frames.
- Open TD, load `sam3` Text DAT, fire `sam3_segment()` against a Video
  Device In TOP. Watch for: stable obj_id across frames, brief flicker at
  the 30s boundary.

### 6. If Option 2 instead

Start by reading `sam3_multiplex_tracking.py:106-280` end-to-end before
touching anything. The metadata lists are the real obstacle, not the image
tensor. Then plan a ring-storage rewrite as a separate commit series on
the `DE-YAN-Studio/sam3` fork's `sam3p1` branch.

## Key file refs (from spike)

- Multiplex builder: `vendor/sam3/sam3/model_builder.py:1070`
- Request dispatch: `vendor/sam3/sam3/model/sam3_base_predictor.py:50-97`
  (simple if-elif chain, easy to extend)
- Per-frame compute entry: `vendor/sam3/sam3/model/sam3_multiplex_base.py:430`
  (`_run_single_frame_inference`)
- Session init: `vendor/sam3/sam3/model/sam3_multiplex_tracking.py:207`
- Frame loader (accepts PIL list!): `vendor/sam3/sam3/model/io_utils.py:31-69`
- Tracker memory params: see `build_tracker` call args in `model_builder.py:434+`

## Gotchas (learned the hard way)

- Don't put the submodule at `<repo>/sam3/` — namespace shadowing. Keep it
  at `vendor/sam3/`.
- All inference must run inside `torch.autocast("cuda", dtype=torch.bfloat16)`
  context. Otherwise: "mat1 and mat2 must have the same dtype, but got
  BFloat16 and Float."
- Conda isn't on PATH in this user's shells. Use full path
  `C:\ProgramData\anaconda3\Scripts\conda.exe` or activate first.
- The `pkg_resources` deprecation warning is noise; upstream still uses it.
