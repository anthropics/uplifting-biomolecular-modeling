"""warm — imports + the kit's apply line on one replicate + one forward on the kit's shipped canary window
(`<kit root>/canary_window_0.npz`, key `x`) through the documented call, to fill the Triton cache of this machine (the kernels compile
on their first launch and persist in the effective cache dir — TRITON_CACHE_DIR when set, else Triton's default; a later process loads
them). Counts the files under the effective cache dir before and after. After the
forward the kit's evidence is read back (stack.settle): a lever without a true flag is components_fallback and the status is PARTIAL
(the CLI exits 3 unless --allow-partial, recorded here as allow_partial). The stock route has no warm.
"""
from __future__ import annotations

import os
import time

from . import report, stack

CANARY = "canary_window_0.npz"


def cache_dir() -> str:
    return os.environ.get("TRITON_CACHE_DIR") or os.path.join(os.path.expanduser("~"), ".triton", "cache")


def _cache_count() -> int:
    d = cache_dir()
    if not os.path.isdir(d):
        return 0
    return sum(len(files) for _, _, files in os.walk(d))


def run(mode: str = "exact", allow_partial: bool = False) -> dict:
    device = "cuda"
    t0 = time.perf_counter()
    before = _cache_count()
    res = {"status": "FAIL", "mode": mode, "forwards": 0, "triton_cache_files": {"before": before, "after": before}, "wall_s": 0.0, "cache_dir": cache_dir(),
           "partial": False, "components_fallback": [], "allow_partial": bool(allow_partial)}
    rep = stack.activate(mode, strict=False, trigger="warm")
    if not rep.get("active"):
        res.update(status="NOT ACTIVE", reason=rep.get("reason"), wall_s=round(time.perf_counter() - t0, 1))
        return res
    from . import inputs, loop, weights
    from . import settings as _settings
    try:
        pins = stack.read_pins()
        _settings.apply_numerics(_settings.DEFAULT)
        m, wrec = weights.load_replicate(pins, _settings.DEFAULT.replicates[0], device)
        stack.apply_to(m, trigger="warm")
        x = inputs.load_onehot(os.path.join(stack.kit_root(), CANARY))
        y, wall = loop.predict_item([m], x, _settings.DEFAULT, device)
        res.update(forwards=1, status="PASS", item_wall_s=round(wall, 3), output_sha256=__import__("flashzoi_opt.outputs", fromlist=["sha256_array"]).sha256_array(y),
                   weights=wrec, replicate=_settings.DEFAULT.replicates[0])
        settled = stack.settle()                                                  # the kit's evidence after the forward: the running verb's partial rule
        res.update(partial=bool(settled.get("partial")), components_fallback=list(settled.get("components_fallback") or []), partial_detection=settled.get("partial_detection"))
        if res["partial"]:
            res.update(status="PARTIAL", reason=f"components_fallback={','.join(res['components_fallback'])}")
    except Exception as e:  # noqa: BLE001
        res.update(status="FAIL", reason=f"{type(e).__name__}: {str(e)[:300]}")
    res["triton_cache_files"]["after"] = _cache_count()
    res["wall_s"] = round(time.perf_counter() - t0, 1)
    return res
