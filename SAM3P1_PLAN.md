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

## Pick-up checklist

When resuming:

1. `git checkout sam3p1`
2. `git submodule update --init --recursive` (since path moved to `vendor/sam3`)
3. Run `test_sam3.py` to confirm env still works
4. Pick an option above
5. If Option 1: start with `td_server.py` — add a `SessionManager` class that
   holds the predictor, opens 30s sessions with a list of dummy PIL frames,
   auto-restarts at frame N. New endpoints dispatch through it.
6. If Option 2: start by reading `sam3_multiplex_tracking.py:106-280` end-to-end
   before touching anything. The metadata lists are the real obstacle, not
   the image tensor.

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
