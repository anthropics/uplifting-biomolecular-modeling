"""boltz2_opt.pairfuse — the engine adapter of the PAIRFUSE layer driver (``opt/forward/pairfuse/bz_pairfuse.py``): the pair stacks of the
trunk (PairformerModule, 64 blocks x 4 passes), the confidence module (8 blocks) and the MSA module (one PairformerNoSeqLayer per MSALayer)
driven on ONE resident copy of the pair tensor z per stack call — every sub-layer reads z once (LayerNorm inside the core's prologue) and
writes it once (the residual added inside the core's epilogue, in place), the ending node addressed by strides (no transposes), the
sequence-attention pair bias projected from the resident z in one pass.  Attached by ``boltz2_opt.worker_launch --attach pairfuse`` right
after ``boltz.model.modules.trunkv2`` imports (both hooked modules exist by then), i.e. BEFORE the worker applies boltz_trunk_levels
(bz_worker_lev.py:63), whose mask2 scope wrappers then wrap this driver's module forwards — the designed composition.

Switch ``BOLTZ_PAIRFUSE`` (ONE row word, R1; the mode table sets it): ``bf16`` = z resident in bf16 (fast tier: the sites' bf16-autocast
arithmetic unchanged, plus bf16 rounding of the residual stream once per site); ``fp32`` = z resident in fp32 (kit-fast's rounding points
exactly; the attention/transition residuals added by the driver until a provider takes fp32 res/out).  Absent = nothing applied.
The attention core is the core triangle-attention provider's row BY TIER WORD (``triatt=core.fast`` on the fast row, ``core.big`` on the
memory row: opt_core.kernels.triattn names the row per card and token count), inside the fused surround (opt_core.attn.pair_fused prologue /
epilogue / transition cells, ln='fused'); the TriMul site likewise (``trimul=core.<tier>``) — all READ ONLY; this adapter changes no core byte.

Stacks the driver hands back take the module's ORIGINAL forward, counted by reason (``report()['stats']['fallback']``); ``EXPECTED`` lists the
reasons a healthy run may show (kernels off; the template stack's C=64; inputs below the driver's 128-token floor); anything else, a displaced hook, or a
kernel error refuses the fail-closed gate (``verdict()``).  Off = the switch absent (no class is touched, the stock statements run) or the
row's ablation word ``BOLTZ_PAIRFUSE=off`` (the lever named off BY THE ROW — nothing is applied, ``dispositions()`` names it
``off:ablation`` so the attach hook accepts the empty apply, and the report's line reads ``LEVER name=… state=off reason=ablation``).
"""
import os
import sys
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
NAME = "pairfuse"                                   # registry name (boltz2_opt.registry)
SWITCH = "BOLTZ_PAIRFUSE"
VARIANTS = ("bf16", "fp32")
OFF_WORD = "off"                                    # the row's ablation entry for this lever — applied nothing, reported state=off reason=ablation
LEVERS = (NAME,)                                    # worker_launch: an adapter with levers installed (the driver; its sites' provider picks are words of its own row word)
KIT_DIR = "forward/pairfuse"                        # opt/-relative: the driver's directory (put on sys.path at apply)
EXPECTED = ("kernels_off", "below_min_tokens", "c:64")   # fallback words a healthy run may show (the template Pairformer's dim; kernels-off rows; tiny inputs)

_STATE: Dict[str, Any] = {"driver": None, "applied": [], "variant": None, "disposed": {}}
LINE_OFF = f"[{TAG}] LEVER name=LOCAL.boltz2.pairfuse state=off reason=ablation"


def variant(environ=None) -> Optional[str]:
    """The row word when it names a residency this adapter knows ('bf16' | 'fp32' | '<residency>,<site>=<impl>,...'), else None."""
    env = os.environ if environ is None else environ
    v = env.get(SWITCH, "").strip().lower()
    return v if v.split(",")[0].strip() in VARIANTS else None


def requested(environ=None) -> bool:
    return variant(environ) is not None


def named_off(environ=None) -> bool:
    env = os.environ if environ is None else environ
    return env.get(SWITCH, "").strip().lower() == OFF_WORD


def dispositions() -> Dict[str, str]:
    """Levers this adapter names as disposed of rather than installed (worker_launch accepts an empty apply() by these words)."""
    return dict(_STATE["disposed"])


def _driver():
    d = _STATE["driver"]
    if d is None:
        from . import stack
        p = stack.kit_path(KIT_DIR)
        if not os.path.isdir(p):
            raise RuntimeError(f"{NAME}: driver directory {p} absent from this kit tree")
        if p not in sys.path:
            sys.path.insert(0, p)
        import bz_pairfuse as d                      # noqa: E402 — resolved from the kit tree only (the path inserted above)
        if os.path.realpath(os.path.dirname(d.__file__)) != os.path.realpath(p):
            raise RuntimeError(f"{NAME}: bz_pairfuse imported from {d.__file__}, not the kit tree {p}")
        _STATE["driver"] = d
    return d


def apply(spec: Optional[str] = None) -> List[str]:
    """Install the driver's three class-level hooks (idempotent). ``spec`` overrides the env switch."""
    if _STATE["applied"]:
        return list(_STATE["applied"])
    if spec is None and named_off():                                      # the row names the lever off (ablation): nothing is hooked, the stock statements run
        _STATE["disposed"] = {NAME: "off:ablation"}; _STATE["variant"] = OFF_WORD
        print(LINE_OFF, file=sys.stderr, flush=True)
        return []
    v = variant() if spec is None else (str(spec).strip().lower() if str(spec).strip().lower().split(",")[0] in VARIANTS else None)
    if v is None:
        return []
    d = _driver()
    d.apply(v)                                                             # raises by name on an unknown provider pick or the unadmitfied fp32 rows (worker_launch: REFUSED, exit 3)
    _STATE["variant"] = v
    _STATE["applied"] = [NAME]
    return list(_STATE["applied"])


def remove() -> None:
    if _STATE["driver"] is not None:
        _STATE["driver"].remove()
    _STATE["applied"] = []


def verdict() -> Dict[str, Any]:
    d = _STATE["driver"]
    if d is None or not _STATE["applied"]:
        return {"ok": False, "idle": False, "reason": "not applied"}
    g = d.gate()
    unexpected = [k for k in d.STATS["fallback"] if k not in EXPECTED and not k.startswith("c:") and not (hasattr(d, "declared") and d.declared(k))]   # a site's no_cell:<site>:<detail> hand-off (no provider cell on this card: the per-layer path serves the stack, by name) is declared
    if g["ok"] and unexpected:
        return {"ok": False, "idle": False, "reason": "unexpected fallback: " + ",".join(sorted(unexpected))}
    return g


def line() -> Optional[str]:
    d = _STATE["driver"]
    return None if d is None else d.line()


def census() -> Optional[Dict[str, Any]]:
    d = _STATE["driver"]
    if d is None:
        return None
    st = d.STATS
    return {"served": dict(st["served_by"]), "served_total": st["stack_served"], "calls": st["stack_calls"], "layers": st["layers"], "noseq_layers": st["noseq_layers"],
            "fallback": dict(st["fallback"]), "sites": dict(st["sites"]), "errors": dict(st["errors"]), "shapes": dict(st["shapes"]),
            "mask_trivial": st["mask_trivial"], "mask_real": st["mask_real"], "first_served": st["first_served"], "alloc": dict(st["alloc"]),
            "core_cells": dict(st.get("core_cells") or {}), "handed": dict(st.get("handed") or {}),
            "core_routes": (d._core_routes() if hasattr(d, "_core_routes") else {}), "picks": dict(d._STATE.get("picks") or {})}


def report() -> Dict[str, Any]:
    d = _STATE["driver"]
    if _STATE["disposed"]:
        return {"applied": [], "disabled": dict(_STATE["disposed"]), "line": LINE_OFF, "census": None,
                "gate": {"ok": True, "idle": True, "reason": "off:ablation"}, "patched": [], "variant": OFF_WORD, "expected": list(EXPECTED)}
    if d is None or not _STATE["applied"]:
        return {"applied": [], "disabled": {}, "line": None, "census": None, "gate": None, "patched": []}
    r = d.report()
    return {"applied": list(_STATE["applied"]), "disabled": {}, "line": d.line(), "census": census(), "gate": verdict(), "patched": r.get("patched", []),
            "variant": _STATE["variant"], "residency": r.get("residency"), "providers": r.get("providers"), "picks": r.get("picks"), "core_gate": r.get("core_gate"),
            "expected": list(EXPECTED), "fp32_rows_admitted": r.get("fp32_rows_admitted"), "torch": r.get("torch")}
