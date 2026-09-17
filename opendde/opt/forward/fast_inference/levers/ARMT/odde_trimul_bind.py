"""odde_trimul_bind -- OpenDDE's triangle-multiplication sites served by the core's TriMul provider (opt_core.kernels.trimul via opt_core.trimul.by_word).

Two kit routes reach ``TriangleMultiplicativeUpdate.forward``: the exact line's FPF adapter (``fpf_engines`` binds the two stock forwards to
the callables ``trimul_out`` / ``trimul_in`` below) and ARM U's TriMul route (``odde_arm_t.make_trimul_forward``: design gate, scope, then
``serve()``).  Both hand EVERY admitted call -- at every row count -- to ``serve()``: the provider selects the row per (compute capability,
precision, c_z, c_hidden, N bucket, direction) cell from the core's measured table (``TRIMUL_CELLS.json``) by the TIER WORD and serves it.  A
cell whose measured winner is the stock op (``cell_stock``), a refusal by name, or a row that cannot run on this stack raise ``Aside`` -- the
caller runs the engine's STOCK TriMul (the cuEquivariance module statement) BY NAME, counted; the kit carries no TriMul construction of its
own and re-routes nothing the provider decided.

One switch, read once at import:

  ODDE_TRIMUL=exact    the provider's exact tier: rows bit-identical to the engine's stock cuEquivariance TriMul on this stack only (vouched per
                       stack in the core's table; e.g. ``native_exact`` / ``tmk3_exact``); an unvouched cell is the stock op by name.
  ODDE_TRIMUL=fast     the provider's fast tier: the measured winner of the cell.
  ODDE_TRIMUL=big    the provider's big tier: the cell's winner measured for peak memory as well as time (a cell without a big row falls
                       to the tier's fast column by the core's rule).
  ODDE_TRIMUL=<row>    pin one row by name (native | native_exact | tmk3_exact | tmk3_fast | cueq | ...: the provider's row names); a refusal is an Aside by name.
  unset / 0 / off      inert: the exact line's adapter callables raise FPFFallback (the stock forward by name); ARM U's route runs the stock forward.

The exact line reaches this unit through ``FPF_IMPL=trimul_out=odde_trimul_bind:trimul_out,trimul_in=odde_trimul_bind:trimul_in`` (the two
callables keep the adapter's contract: ``FPFFallback`` = the stock forward by name); ARM U's route calls ``serve()`` itself.

Census: ``COUNTS`` (calls per row / cell line / precision / row-count bucket, asides by reason, refusals, excluded rows) -- ``describe()`` is
the LEVER line's evidence; ``opendde_opt.report`` prints it.
"""
from __future__ import annotations

import os

__version__ = "0.2"

_RAW = os.environ.get("ODDE_TRIMUL", "").strip()
WORD = None if _RAW.lower() in ("", "0", "off", "none", "stock") else _RAW
TIER_WORDS = ("fast", "exact", "big")                       # the provider's tier words (opt_core.kernels.trimul TIER_WORDS): exact = S1, fast = LSTAR2A, big = the memory mode's lines
N_MIN = 101                 # upstream's own regime: at and below 100 rows the stock module runs cuEquivariance's small-N torch algorithm (a different statement than the
                            # library kernel the exact rows are bit-identical to) -- those calls ARE the stock forward, by name (asides n_le_100_stock_small_n_path)
BUCKETS = (256, 400, 512, 800, 1024, 1200, 1556, 2048)         # census only: served calls counted per row-count bucket (N<=b; 'gt2048' above) on the LEVER line

COUNTS: dict = {"version": __version__, "word": WORD, "active": WORD is not None, "calls": 0, "rows": {}, "cells": {}, "precisions": {},
                "stock_calls": 0, "asides": {}, "refusals": {}, "excluded": [], "errors": {}, "channels": {}, "sites": {}, "buckets": {},
                "resolved_from": None, "memo_pops": 0, "memo_api": None}
_EXCLUDED: set = set()      # rows that raised at call time in this process: never selected again (COUNTS['excluded'] / ['errors'] name them)
_PROV: dict = {}            # word -> opt_core.trimul Provider (by_word)
_STACK: dict = {}           # the provider's stack word for this process (torch / triton / cuequivariance versions), read once


class Aside(Exception):
    """The call is the engine's stock TriMul BY NAME (measured stock winner, refusal by name, excluded row, small N); .reason names it."""
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


def active() -> bool:
    return WORD is not None


def _bump(key, sub=None, n=1):
    if sub is None:
        COUNTS[key] = int(COUNTS.get(key) or 0) + n
    else:
        d = COUNTS.setdefault(key, {})
        d[sub] = int(d.get(sub) or 0) + n


def bucket_word(N: int) -> str:
    for b in BUCKETS:
        if N <= b:
            return f"le{b}"
    return f"gt{BUCKETS[-1]}"


def weights_of(m) -> dict:
    """OpenDDE 1.x TriangleMultiplicativeUpdate attributes -> the provider's ten canonical tensors (no biases)."""
    return {"ln_in_w": m.layer_norm_in.weight, "ln_in_b": m.layer_norm_in.bias,
            "w_ag": m.linear_a_g.weight, "w_ap": m.linear_a_p.weight, "w_bg": m.linear_b_g.weight, "w_bp": m.linear_b_p.weight,
            "ln_out_w": m.layer_norm_out.weight, "ln_out_b": m.layer_norm_out.bias, "w_o": m.linear_z.weight, "w_og": m.linear_g.weight}


def _provider(word: str):
    p = _PROV.get(word)
    if p is None:
        from opt_core import trimul as T
        p = _PROV[word] = T.by_word(weights_of, word, name=f"odde_trimul:{word}", cache_key=f"odde.{word}")
    return p


CAST_MEMO_KEY = "_z_cast"       # the provider's per-module cast memo (opt_core.kernels.trimul.compute_input: the fp32 z object + its autocast-dtype copy, kept in the
                                # module's provider cache so admission and serving cast once) -- released after every call below: this engine updates z IN PLACE on one
                                # tensor object through the stack and builds a fresh z per confidence sample, so a memo keyed on tensor identity must not outlive the call
                                # (no fp32/bf16 pair copies pinned across calls or samples; a stale copy is never served)


def release_cast_memo(T, module) -> int:
    """Release the provider's per-module z-cast memo for ``module`` after a served call; returns the number of memo entries released (0 when the
    row made none: native / exact rows, a core without the memo). Goes through the provider face: its public release helper when the core
    carries one, else the module's provider cache dict (``T._CACHE_ATTR``) -- every name guarded, a harmless no-op when absent."""
    for name in ("release_cast_memo", "forget_cast", "clear_cast_memo"):              # a public helper of the face, when the core has one
        fn = getattr(T, name, None)
        if callable(fn):
            COUNTS["memo_api"] = f"opt_core.trimul.{name}"
            try:
                r = fn(module)
                return int(r) if isinstance(r, (bool, int)) else 0
            except Exception:  # noqa: BLE001
                return 0
    attr = getattr(T, "_CACHE_ATTR", None)                                              # else: the per-module cache dict the face keeps on the module
    cache = getattr(module, attr, None) if isinstance(attr, str) else None
    if not isinstance(cache, dict):
        return 0
    COUNTS["memo_api"] = f"module.{attr}[*].{CAST_MEMO_KEY}"
    n = 0
    if cache.pop(CAST_MEMO_KEY, None) is not None:
        n += 1
    for sub in list(cache.values()):
        if isinstance(sub, dict) and sub.pop(CAST_MEMO_KEY, None) is not None:
            n += 1
    return n


def _facts(KT, module, z):
    zs = z if z.dim() == 4 else z[None]
    prec, _cdt = KT.call_precision(zs)
    res = "fp32" if prec.startswith("f32z_") else None
    dt = prec.split("_")[-1] if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
    return zs, KT._device_cc(zs), prec, dt, res, int(module.linear_a_p.weight.shape[0])


def serve(module, z, mask, *, residual: bool, site: str = "arm"):
    """Serve one admitted TriMul call: ``module`` the TriangleMultiplicativeUpdate instance, ``z`` the PRE-LayerNorm pair tensor [N,N,c] | [B,N,N,c],
    ``mask`` [N,N] | [B,N,N] | None, ``residual`` = the stock forward returns z + update for this call (inplace_safe and _add_with_inplace).
    Returns what the row returns (the update, or z + update, in the stock op's dtype for the call) or raises ``Aside(reason)``."""
    from opt_core import trimul as T
    from opt_core.kernels import trimul as KT
    from opt_core.oom import is_oom
    if WORD is None:
        raise Aside("no_word")
    if not getattr(z, "is_cuda", False) or z.dim() not in (3, 4):
        raise Aside("not_cuda_or_rank")
    N, C = int(z.shape[-2]), int(z.shape[-1])
    if N < N_MIN:
        raise Aside("n_le_100_stock_small_n_path")
    direction = "outgoing" if getattr(module, "_outgoing", True) else "incoming"
    zs, cc, prec, dt, res, D = _facts(KT, module, z)
    _bump("sites", site); _bump("channels", f"C{C}")
    try:                                                        # the pure selection first: the tier word's measured stock winner is the stock op BY NAME (Aside cell_stock)
        if "word" not in _STACK:
            _STACK["word"] = KT.stack_word(zs); COUNTS["stack"] = _STACK["word"]
        sel = KT.select(cc, dt, C, D, N, direction, word=WORD, residency=res, tf32=(prec == "tf32"), has_cueq=True, stack=_STACK["word"])
    except KT.Refusal as r:
        _bump("refusals", f"{r.row}:{r.kind}"); raise Aside(f"refused:{r.row}:{r.kind}"[:120])
    if sel.row in KT.STOCK_ROWS:
        raise Aside("cell_stock")
    if sel.row in _EXCLUDED:
        raise Aside(f"excluded:{sel.row}")
    prov = _provider(WORD)
    call = T.Call(module, z, mask, direction, residual, orig=None)
    try:
        prov.eligible(call)
    except T.Refused as r:
        _bump("refusals", str(r.reason)[:80]); raise Aside(f"refused:{r.reason}"[:120])
    row = getattr(call.extra.get("trimul_selection"), "row", sel.row)
    if row in KT.STOCK_ROWS:
        raise Aside("cell_stock")
    if row in _EXCLUDED:
        raise Aside(f"excluded:{row}")
    try:
        out = prov.fn(call)
    except T.Refused as r:                                      # a refusal with the tensors in hand (the kernel's own supported()): by name
        _bump("refusals", str(r.reason)[:80]); raise Aside(f"refused:{r.reason}"[:120])
    except Exception as e:  # noqa: BLE001 -- a row that cannot run on this stack is excluded BY NAME for the process; OOM is the caller's
        if is_oom(e):
            raise
        _EXCLUDED.add(row); COUNTS["excluded"] = sorted(_EXCLUDED); COUNTS["errors"][row] = repr(e)[:300]
        kind = "unavailable" if isinstance(e, (T.CannotRun, ImportError, OSError)) or T.cannot_run(e) else "error"
        _bump("refusals", f"{row}:{kind}:{type(e).__name__}")
        raise Aside(f"{kind}:{row}:{type(e).__name__}")
    finally:                                                    # the provider's per-module cast memo never outlives the call (z is updated in place / rebuilt per sample here)
        n_memo = release_cast_memo(T, module)
        if n_memo:
            _bump("memo_pops", n=n_memo)
    _bump("calls"); _bump("rows", row); _bump("precisions", prec); _bump("buckets", f"{bucket_word(N)}:{row}")
    s2 = call.extra.get("trimul_selection")
    _bump("cells", KT.describe(s2) if s2 is not None else f"{row}|{prec}|C{C}|no-cell")
    if COUNTS["resolved_from"] is None:
        COUNTS["resolved_from"] = getattr(prov, "resolved_from", None) or "core"
    return out


def note_aside(reason: str):
    """A caller records a call it ran on the stock forward on this unit's account (reason from ``Aside``)."""
    _bump("stock_calls"); _bump("asides", reason)


# ---------------------------------------------------------------------------------------------------- the exact line's FPF_IMPL callables
def _adapter(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative, direction):
    """fpf_engines' contract for one bound stock forward: return the stock-shaped output or raise FPFFallback (the stock forward by name).
    The provider by this unit's word; an Aside is FPFFallback(<reason>): the adapter runs the stock forward, counted under its reason."""
    from fpf_engines import FPFFallback
    if triangle_multiplicative != "cuequivariance" or module.c_z != module.c_hidden:
        raise FPFFallback("torch_path_class")
    if module.training:
        raise FPFFallback("training")
    add = bool(inplace_safe is True and _add_with_inplace)
    try:
        out = serve(module, z, mask, residual=add, site="fpf_exact")
    except Aside as a:
        note_aside(a.reason)
        raise FPFFallback(f"provider:{a.reason}"[:120])
    return out if add else out.to(z.dtype)


def trimul_out(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    return _adapter(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative, "outgoing")


def trimul_in(module, z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch"):
    return _adapter(module, z, mask, inplace_safe, _add_with_inplace, _inplace_chunk_size, triangle_multiplicative, "incoming")


def describe() -> dict:
    """The LEVER line's evidence: word, calls per row, precisions, row-count buckets, asides, refusals, excluded rows, the distinct cell lines."""
    return {"word": WORD, "calls": int(COUNTS.get("calls") or 0), "rows": dict(COUNTS.get("rows") or {}), "stock_calls": int(COUNTS.get("stock_calls") or 0),
            "asides": dict(COUNTS.get("asides") or {}), "refusals": dict(COUNTS.get("refusals") or {}), "excluded": list(COUNTS.get("excluded") or []),
            "cells": dict(COUNTS.get("cells") or {}), "precisions": dict(COUNTS.get("precisions") or {}), "buckets": dict(COUNTS.get("buckets") or {}),
            "channels": dict(COUNTS.get("channels") or {}), "sites": dict(COUNTS.get("sites") or {}), "resolved_from": COUNTS.get("resolved_from"),
            "errors": dict(COUNTS.get("errors") or {}), "memo_pops": int(COUNTS.get("memo_pops") or 0), "memo_api": COUNTS.get("memo_api")}


def reset_counts():
    for k in ("calls", "stock_calls", "memo_pops"):
        COUNTS[k] = 0
    for k in ("rows", "cells", "asides", "refusals", "precisions", "buckets", "channels", "sites"):
        COUNTS[k] = {}
