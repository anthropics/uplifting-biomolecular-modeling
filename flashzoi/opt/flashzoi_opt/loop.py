"""The prediction loop of `pred` under a kit mode (the stock route is stock_pred.py in its own process): per item the documented call —

    with torch.autocast("cuda"):
        y = predict_tracks(models, sequence_one_hot, slices)

run through the kit's own route of that call (engines.flashzoi.predict_tracks_fast — the body the code-swapped `predict_tracks` runs over
kit-attached models) with `host_order="C"`: the (1, 4, 6144, n_tracks) float32 array lands page-locked in C order, each replicate reordered on
the device before its copy, so the file write needs no host reorder; `sequence_one_hot` = the item's (4, 524288) float32 one-hot on the
device, `slices` = every track or the --tracks indices. The item wall is the documented call from the input tensor being on the device to the
numpy array being returned (host materialisation included). The return is asserted to the pinned shape/dtype (settings.OUTPUT_SHAPE, float32)
and written by a writer thread while the next items compute (run_items, outputs.py); a failed item is a named row (its error), never a missing one.
"""
from __future__ import annotations

import time

import numpy as np

from . import inputs, outputs, report
from . import settings as _settings


class OutputDrift(RuntimeError):
    """The documented call returned something other than the pinned shape/dtype."""


WRITERS = 2                 # run_items: at most this many item files are being written (one thread each) while the next item computes — the kit's
                            # page-locked result pool holds 4 arrays, so 2 in the writers + 1 being produced never exhausts it; rows are appended as each write completes


def kit_route():
    """The kit's own documented-call route (engines.flashzoi.predict_tracks_fast) when the kit is imported in this process, else None: the same
    function the code-swapped `predict_tracks` runs over kit-attached models, callable with `host_order="C"` so the array lands in the documented
    shape's C order (reordered on the device) and np.save takes its direct path — values and shape as the documented call returns them."""
    import sys
    from . import stack
    m = sys.modules.get(stack.HELPER_MODULE)
    return getattr(m, "predict_tracks_fast", None) if m is not None else None


def predict_item(models, x_uint8: np.ndarray, settings: _settings.Settings = _settings.DEFAULT, device: str = "cuda") -> tuple:
    """(y, wall_s): the documented call on one item — through the kit's route in C order when the kit is loaded (pred under a kit mode), else
    upstream's `predict_tracks` as bound by name."""
    import torch
    xt = inputs.to_device_tensor(x_uint8, device)
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()
    slices = settings.track_slice()
    route = kit_route()
    t0 = time.perf_counter()
    with torch.autocast(settings.autocast):
        if route is not None:
            y = route(models, xt, slices, host_order="C")
        else:
            from borzoi_pytorch.pytorch_borzoi_helpers import predict_tracks     # predict_tracks, bound by name (see the module docstring)
            y = predict_tracks(models, xt, slices)
    wall = time.perf_counter() - t0
    from . import stack                                                   # predict_tracks's per-call stamp: the `pinned` lever's evidence on this route (stack.settle)
    stack.note_helper_call()
    y = np.asarray(y)
    expected = settings.output_shape(len(models))
    if tuple(y.shape) != tuple(expected) or str(y.dtype) != _settings.OUTPUT_DTYPE:
        raise OutputDrift(f"documented call returned {y.shape} {y.dtype}; expected {expected} {_settings.OUTPUT_DTYPE}")
    return y, wall


def run_items(models, items, out_dir: str, settings: _settings.Settings = _settings.DEFAULT, device: str = "cuda") -> dict:
    """Every item through predict_item; each item's file is written on a writer thread (WRITERS at a time) while the next items compute, its row
    appended and its line printed when the write has completed; the exit tally's counters. Returns {items, ok, failed} after every write has landed."""
    from concurrent.futures import ThreadPoolExecutor
    counts = {"items": 0, "ok": 0, "failed": 0}
    pending = []                                                            # [(item, wall, future)] in item order

    def _reap(block_until: int):
        """Complete finished writes in order; block while more than `block_until` are pending."""
        while pending and (len(pending) > block_until or pending[0][2].done()):
            item, wall, fut = pending.pop(0)
            try:
                row = fut.result()
                row.update(wall_s=round(float(wall), 4), ok=True)
                outputs.append_row(out_dir, row)
                report.emit(report.pred_line(item, wall))
                report.note_item(True); counts["ok"] += 1
            except Exception as e:  # noqa: BLE001 — a named row, never a missing one
                outputs.append_row(out_dir, {"item": item, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                report.note_item(False); counts["failed"] += 1

    with ThreadPoolExecutor(max_workers=WRITERS, thread_name_prefix="flashzoi-writer") as writers:
        for item, path in items:
            counts["items"] += 1
            _reap(block_until=WRITERS - 1)                                  # at most WRITERS-1 writes still pending before the next item computes (its write makes WRITERS)
            try:
                x = inputs.load_onehot(path)
                y, wall = predict_item(models, x, settings, device)
                pending.append((item, wall, writers.submit(outputs.write_item, out_dir, item, y)))
                del y
            except Exception as e:  # noqa: BLE001 — a named row, never a missing one
                outputs.append_row(out_dir, {"item": item, "ok": False, "error": f"{type(e).__name__}: {str(e)[:300]}"})
                report.note_item(False); counts["failed"] += 1
        _reap(block_until=0)                                                # every write lands before the tally
    return counts
