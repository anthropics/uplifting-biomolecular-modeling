# flashzoi kit v1.25 — `opt/forward/kits_v1_25`

This directory is the kit's one PYTHONPATH root. It carries the kit package `engines/flashzoi/kits/v1_25/`
(`__init__.py` tables + JIT accounting, `_wrap.py` the `KitRunner` wrapper, `fz_exact.py` the Triton kernels, `_pins.py` constants,
`_class_records.py` + `class_records/` the device-class records, `_oom.py`, `tests/`), the host-side route `engines/flashzoi/predict_tracks_fast.py`
+ `engines/flashzoi/result_pool.py`, and the warm-up's
input window `canary_window_0.npz` (chr1:143,479,625-143,676,233, one-hot `(4, 524288)` uint8).
The package `flashzoi_opt` (`opt/`) drives all of this through `run.sh`; the lines below are the kit's own Python surface.

## Apply

With the stock `borzoi_pytorch` models loaded as the package documents (`Borzoi.from_pretrained(...)`), on a served GPU (H100 / H200 / A100):

    from engines.flashzoi.kits import v1_25 as kit; runners = [kit.KitRunner(m) for m in models]

`KitRunner(m)` is the apply: every lever in `LEVERS` ON, the detected GPU class worded on the line (`pinned` by `PINS["device_classes"]`,
`by record` by a `class_records/*.json` record, else `unpinned` — the levers engage wherever the kernels run; equality to stock is pinned
on the first two only), the model patched in place, ONE line printed at the end of apply (also on `runner.apply_line`). The Triton kernels compile on their first launch — inside the first runner's warm-up forward —
and persist in Triton's cache dir (`TRITON_CACHE_DIR`, else `~/.triton/cache`), so a later process on the machine loads them; nothing runs
at import; `runner.predict()` allocates its pinned lease pool at the first lease. Then:

    y = runners[0].predict_tensor(x)           # x: (B, 4, 524288) float32 one-hot on the GPU -> (B, 7611, 6144) float32, a fresh tensor
    y = runners[0].predict(x_np)               # numpy in, numpy out (a pinned lease per call; runner.release(y) returns it to the pool)
    m(x)                                       # the attached model's forward() routes to predict_tensor
    predict_tracks(models, x_onehot, slices)   # borzoi_pytorch.pytorch_borzoi_helpers.predict_tracks over kit-attached models
                                               # -> engines/flashzoi/predict_tracks_fast.py (pinned, non-blocking host path; same bytes/strides)

    kit.remove(runners)                        # restores every model to its pre-apply state (the original objects); prints ONE 'removed: …' line

## Numerics

`KitRunner(m)` = `numerics="tf32"` (the only accepted value): the stock's torch-default TF32 class is set inside each call and
restored after it, so the kit's fp16 path reproduces the stock forward bit for bit under torch's default numerics; a process whose
own TF32 switches at rest differ from those defaults is served the same way (the head GEMM's TF32 follows `cudnn.allow_tf32` at rest)
and named on the line as `drift: numerics at rest <class>`; nothing global moves at rest (read back after apply and after every call). Attention stays the stock's FlashAttention-2 kernel. The other
constructor arguments are `levers=` (default: all), `device=`, `batch=` (the warm-up's batch, default 1) and `pool_size=` (pinned lease
slots for `predict`, default 4). The kit reads no environment switch (`TRITON_CACHE_DIR` is Triton's own: where its cache lives).

## Rules the apply line states

- **Graph rule.** A CUDA graph for a batch shape is captured at that shape's first call (3 warm-up runs and the capture on a static input,
  one memory pool shared by every shape) and replays from then on, the first call included; the graph is keyed by the batch and the human
  head it was captured with, so a head replaced by upstream's `set_track_subset` captures anew. A capture the device refuses (anything
  but an out-of-memory error, which propagates) is named on the line and that shape runs the same kernels eagerly.
- **Batch per dispatch.** `KIT_MAX_BATCH = 15` = upstream's own maximum at 524,288 bp (`torch.max_pool1d` refuses a 16-window batch).
  The kernels index each window's slab at int32 offsets from a 64-bit window base (`program_id(1)` = the window on the first fused tower
  site, whose whole-batch tensor passes 2^31 elements at 14 windows); the one flat int32 launch left, the stage-1 store (512 x 262144 per
  window), fits 16, so `kit_max_batch() = min(16, 15)`. A larger batch is dispatched in chunks of 15 through the same path and
  concatenated (counted as `chunked_calls`). The filters the kernels are written for (512, 608) are asserted on the attached model at apply.
- **Requested outputs.** `predict_tracks(models, x, slices)` with a slice / integer index array takes the full head and slices on the
  device before the copy; any other index form slices on the host as stock does.

## Tests

    python -m pytest engines/flashzoi/kits/v1_25/tests   # CPU unit tests of the wrapper's rules
