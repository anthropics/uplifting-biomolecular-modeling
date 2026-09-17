"""fpf_mkpf — cross-statement fusion levers (LayerNorm in registers + the statement that consumes it) over the FlashPairformer prologue / transition kernels
(TIER-2: in-register two-pass LayerNorm; EXACT where the 'welford' fast_layernorm emulation serves; see kernels.py for the numerics class).

  PTX_MK_PF=F1        ending-node tri-attention prologue reads z^T by strides with LN in registers (drops fast_layernorm + its [N,N,C] transposed copy)      [default node set: END]
  PTX_MK_PF=F1,F1S    ... also the starting node (drops fast_layernorm(z));  F1S alone is allowed
  PTX_MK_PF=...,F3    pair transition with LN in registers (drops layernorm1 + the y round trip)
  FPF_MKPF_CELLS_JSON=<path> overrides cells; FPF_MKPF_ALLOW_TRITON=1 to run on an unchecked (cc, triton).
Install contract (the kit's installer hook): FPF_MKPF_INSTALL=fpf_mkpf:install -> install() patches, in place and idempotently:
  ptx_trunk2_levers._blk_ln_mode            -> returns ('mkpf', None) for selected nodes (eligible modules: c_in == 256, H*D == 256, bf16 CUDA z), else the original
  fpf_triatt_pro.prologue.triatt_prologue / .triatt_prologue_padded (and glue wrappers above them) -> ln_mode == 'mkpf' routes to kernels.prologue_ln (strided z read + LN + projections)
  fpf_transition.transition._forward        -> F3: eligible residual calls (fn_residual from the BLK2 block path) run kernels.transition_ln_v2 per stock chunk
install() RAISES if it cannot engage (no levers module, PTX_BLK != 2, no cell for this (cc, triton), kernels not importable) so a launcher never reports a vacuous PASS.
Everything not selected passes through to the tested path unchanged; counters in stats()."""
import os, sys, json
__version__ = "0.2-mkpf-20260823"
_STATE = {"installed": False, "patched": set(), "levers": [], "why": "", "cells": {}, "calls": {"f1_end": 0, "f1_start": 0, "f1_padded": 0, "f1_passthru": 0, "f3": 0, "f3_passthru": 0}}

def _say(msg):
    sys.stderr.write(f"[fpf_mkpf] {msg}\n"); sys.stderr.flush()

def _triton_mm():
    try:
        import triton; v = triton.__version__.split("."); return f"{v[0]}.{v[1]}"
    except Exception:
        return "?"

def levers_requested():
    raw = os.environ.get("PTX_MK_PF", "")
    lv = [x.strip().upper() for x in raw.split(",") if x.strip()]
    out = set()
    for x in lv:
        if x == "F1": out.add("F1")
        elif x in ("F1E", "F1END"): out.add("F1E")
        elif x in ("F1S", "F1START"): out.add("F1S")
        elif x in ("F1B", "F1BOTH"): out.update(("F1E", "F1S"))
        elif x == "F3": out.add("F3")
        elif x in ("F2", "F4"): _say(f"{x}: subsumed by consumer-side LN fusion (no kernel; see README) -> ignored")
        else: _say(f"unknown lever {x!r} ignored")
    return out

MEASURED_OFF_CC = {"10.3": "B300: the certified compositions run without MK-PF (no cell measured on cc 10.3)",
                   "8.0": "A100: the certified compositions run without MK-PF (no cell measured on cc 8.0)"}    # capabilities the measured compositions run with these levers OFF, by name


def resolve_cells(device=None, *, cc=None, triton=None):
    """-> the cells entry for this (cc, triton), or {} with ``_STATE['why']`` naming the case.  Resolution: the exact ``<cc>|<triton M.m>`` entry; else the capability's
    ``by_cc`` entry, applied and NAMED as unverified on this triton (``FPF_MKPF_ALLOW_TRITON=0`` refuses it instead, by name); else none — a capability the measured compositions
    run with these levers off (MEASURED_OFF_CC) or an UNKNOWN capability (these kernels carry per-capability cells only: ``install()`` refuses by name and the composed path runs
    without them).  ``cc`` / ``triton`` name the environment for a decision off the device."""
    path = os.environ.get("FPF_MKPF_CELLS_JSON") or os.path.join(os.path.dirname(os.path.abspath(__file__)), "MKPF_CELLS.json")
    with open(path) as _fh:
        doc = json.load(_fh)
    if cc is None:
        import torch
        dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        cap = torch.cuda.get_device_capability(dev); cc = f"{cap[0]}.{cap[1]}"
    tmm = triton if triton is not None else _triton_mm()
    key = f"{cc}|{tmm}"
    ent = doc.get("by_cc_triton", {}).get(key)
    note = ""
    if ent is None:
        fb = doc.get("by_cc", {}).get(str(cc))
        if fb is not None and os.environ.get("FPF_MKPF_ALLOW_TRITON", "") != "0":
            _say(f"WARNING: no pinned cells for {key}: the capability's cells '{fb}' APPLY, unverified on triton {tmm} (re-check with selftest; FPF_MKPF_ALLOW_TRITON=0 refuses instead)"); ent = fb
            note = f" (triton {tmm} unverified)"
        elif fb is not None:
            _STATE["why"] = f"no pinned MKPF cells for (cc|triton) = {key} (pinned: {sorted(doc.get('by_cc_triton', {}))}); FPF_MKPF_ALLOW_TRITON=0 refuses the by_cc cells '{fb}'"
            return {}
        elif str(cc) in MEASURED_OFF_CC:
            _STATE["why"] = f"no MKPF cells for cc {cc} (off: not-measured — {MEASURED_OFF_CC[str(cc)]})"
            return {}
        else:
            _STATE["why"] = f"no MKPF cells for cc {cc} (unknown capability: these kernels carry per-capability cells only; cells exist for cc {sorted(doc.get('by_cc', {}))})"
            return {}
    cells = doc["entries"][ent]; _STATE["why"] = f"{key} -> '{ent}'{note}"; _STATE["cells"] = cells
    return cells

def _write_report(rep):
    with open(rep, "a") as _fh:
        _fh.write(json.dumps({"fpf_mkpf": stats()}) + "\n")

def stats():
    return {"version": __version__, "installed": _STATE["installed"], "patched": sorted(_STATE["patched"]), "levers": sorted(_STATE["levers"]), "why": _STATE["why"], "calls": dict(_STATE["calls"]),
            "cells": {k: (v.get("cfg") if isinstance(v, dict) else v) for k, v in _STATE["cells"].items()} if _STATE["cells"] else {}}

def _eligible_triatt_shape(module, x):
    import torch
    try:
        mha = module.mha
        return (x.is_cuda and x.dtype == torch.bfloat16 and x.dim() == 3 and x.shape[-1] == 256 and int(module.c_in) == 256 and mha.no_heads * mha.c_hidden == 256 and x.stride(-1) == 1)
    except Exception:
        return False

def _n_gate_ok(x):
    """The ARM T size gate for a TIER-2 cell (ln_arith != 'welford', e.g. the 9.0|3.3 two-pass cell): serve only N_token >= PTX_T_MIN_TOKENS
    (fallback FPF_SMALLN_K2B_MIN_TOKENS, default 300); below it the statement stays on the tested path. EXACT (welford) cells are ungated."""
    cells = _STATE.get("cells") or {}
    if cells.get("ln_arith") == "welford" or str(cells.get("class", "")).upper().startswith("EXACT"):
        return True
    thr = int(os.environ.get("PTX_T_MIN_TOKENS", os.environ.get("FPF_SMALLN_K2B_MIN_TOKENS", "300")) or 0)
    n = int(x.shape[-2]) if x.dim() >= 2 else 0
    if n < thr:
        _STATE["calls"]["f1_gated_small"] = _STATE["calls"].get("f1_gated_small", 0) + 1
        return False
    return True

def _eligible_triatt(module, x):
    return _eligible_triatt_shape(module, x) and _n_gate_ok(x)

def install(verbose=True, levers=None):
    """Apply the requested levers (PTX_MK_PF or `levers`) on top of the composed tested path. Idempotent. Raises RuntimeError if nothing can engage."""
    import torch
    lv = set(levers) if levers is not None else levers_requested()
    if not lv:
        raise RuntimeError("fpf_mkpf.install(): no levers requested (set PTX_MK_PF=F1[,F1S][,F3])")
    if os.environ.get("PTX_BLK", "0") != "2":
        raise RuntimeError("fpf_mkpf.install(): requires the BLK2 block path (PTX_BLK=2, i.e. the FPF env.sh)")
    cells = _STATE["cells"] or resolve_cells()
    if not cells:
        raise RuntimeError("fpf_mkpf.install(): " + _STATE["why"])
    from . import kernels as MK
    import ptx_trunk2_levers as LEV
    import fpf_triatt_pro.prologue as PRO
    if "F1" in lv:                     # generic F1 -> the cell's default node set (the 9.0|3.3 TIER-2 cell: END only; the triton 3.7 EXACT cells: both)
        lv.discard("F1"); lv.update({"END": "F1E", "START": "F1S"}[n] for n in cells.get("nodes_default", ["END"]))
    want_f1 = bool(lv & {"F1E", "F1S"}); want_f3 = "F3" in lv
    f1cfg = dict(cells["f1"]["cfg"]) if "f1" in cells else None
    f3cfg = dict(cells["f3"]["cfg"]) if "f3" in cells else None
    if want_f1 and f1cfg is None: raise RuntimeError("fpf_mkpf.install(): F1 requested but no f1 cell for this (cc, triton)")
    if want_f3 and f3cfg is None: raise RuntimeError("fpf_mkpf.install(): F3 requested but no f3 cell for this (cc, triton) (F3 may be absent by design: see README)")
    arith = os.environ.get("FPF_MKPF_LN", cells.get("ln_arith", "fused")); fma = tuple(bool(int(f)) for f in cells.get("fma_flags", [1, 1, 1]))
    if arith == "welford" and not MK.HAS_WELFORD: raise RuntimeError("fpf_mkpf.install(): cell arith=welford but fpf_triatt_pro welford helpers not importable")

    # ---- F1: (a) node selection in the levers' LN-mode hook
    if want_f1 and "blk_ln_mode" not in _STATE["patched"]:
        _orig_lnm = LEV._blk_ln_mode
        sel_end, sel_start = "F1E" in lv, "F1S" in lv
        def _blk_ln_mode(module, x, ending, _orig=_orig_lnm):
            if ((ending and sel_end) or ((not ending) and sel_start)) and _eligible_triatt(module, x):
                LEV._STATS["blk_ln_mkpf"] = LEV._STATS.get("blk_ln_mkpf", 0) + 1
                return "mkpf", None
            return _orig(module, x, ending)
        _blk_ln_mode.__wrapped__ = _orig_lnm
        LEV._blk_ln_mode = _blk_ln_mode; _STATE["patched"].add("blk_ln_mode")
    # ---- F1: (b) the prologue entry points understand ln_mode='mkpf' (x = the x-frame VIEW of z: contiguous for start, transposed view for end -> strides carry the frame)
    if want_f1 and "triatt_prologue" not in _STATE["patched"]:
        _orig_pro = PRO.triatt_prologue
        def triatt_prologue(module, z, ending=False, ln_mode="fused", x_ln=None, **kw):
            if ln_mode == "mkpf":
                cch = PRO.get_cache(module, z.device)
                is_T = (z.dim() == 3 and z.stride(0) < z.stride(1))           # transposed view => this is the ending node's x-frame
                _STATE["calls"]["f1_end" if is_T else "f1_start"] += 1
                return MK.prologue_ln(module, z, cch, f1cfg, ending=bool(ending), ln_arith=arith, fma_flags=fma)
            _STATE["calls"]["f1_passthru"] += 1
            return _orig_pro(module, z, ending=ending, ln_mode=ln_mode, x_ln=x_ln, **kw)
        triatt_prologue.__wrapped__ = _orig_pro
        PRO.triatt_prologue = triatt_prologue; _STATE["patched"].add("triatt_prologue")
        # the levers bound `_pro` by from-import at apply time: if BLK2 is already applied, rebind the closure cell they use (blk_forward references module-level _pro via closure of _apply_blk2)
        _rebind_levers_prologue(LEV, _orig_pro, triatt_prologue)
        _orig_pad = getattr(PRO, "triatt_prologue_padded", None)
        if _orig_pad is not None:
            def triatt_prologue_padded(module, z, out, ending=False, ln_mode="stock", x_ln=None, **kw):
                if ln_mode == "mkpf":
                    cch = PRO.get_cache(module, z.device)
                    _STATE["calls"]["f1_padded"] += 1
                    return MK.prologue_ln(module, z, cch, f1cfg, ending=bool(ending), ln_arith=arith, fma_flags=fma, out=out)
                return _orig_pad(module, z, out, ending=ending, ln_mode=ln_mode, x_ln=x_ln, **kw)
            triatt_prologue_padded.__wrapped__ = _orig_pad
            PRO.triatt_prologue_padded = triatt_prologue_padded; _STATE["patched"].add("triatt_prologue_padded")
    # ---- F3: transition
    if want_f3 and "transition" not in _STATE["patched"]:
        import fpf_transition.transition as TR
        _orig_fwd = TR._forward
        def _forward(module, x, residual, ln_mode, _orig=_orig_fwd):
            if residual and ln_mode == 0 and x.is_cuda and x.dtype == torch.bfloat16 and x.shape[-1] == 256 and int(module.c_in) == 256:
                ok, why = TR._eligible(module, x)
                if ok and module.linear_no_bias.weight.shape[1] % int(f3cfg["BH"]) == 0:
                    x2 = x.reshape(-1, 256)
                    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) != 256): x2 = x2.contiguous()
                    cache = TR._weights(module, x2.device)
                    out = torch.empty_like(x2)
                    {1: MK.transition_ln, 2: MK.transition_ln_v2, 3: MK.transition_ln_v3}[int(f3cfg.get("V", 2))](x2, cache, out, {k: v for k, v in f3cfg.items() if k != "V"}, ln_arith=arith, fma_flags=fma)
                    TR.STATS["fused_calls"] += 1; _STATE["calls"]["f3"] += 1
                    return out.reshape(*x.shape[:-1], module.c_in)
            _STATE["calls"]["f3_passthru"] += 1
            return _orig(module, x, residual, ln_mode)
        _forward.__wrapped__ = _orig_fwd
        TR._forward = _forward; _STATE["patched"].add("transition")
    _STATE["levers"] = sorted(lv); _STATE["installed"] = True
    if verbose:
        _say(f"installed levers={sorted(lv)} arith={arith} cells: {_STATE['why']} f1={f1cfg} f3={f3cfg} patched={sorted(_STATE['patched'])}")
    try:
        import atexit
        rep = os.environ.get("PTX_LEVER_REPORT")
        if rep and "atexit" not in _STATE["patched"]:
            atexit.register(_write_report, rep); _STATE["patched"].add("atexit")
    except Exception:
        pass
    return stats()


def _rebind_levers_prologue(LEV, orig, new):
    """Repoint the block lever's closure cells from the original prologue ``orig`` to the wrapper ``new`` (body in ``_engine_adapter``, imported on
    first call: it walks the engine's PairformerBlock.forward, importing the engine's upstream package lazily)."""
    from ._engine_adapter import _rebind_levers_prologue as _f
    return _f(LEV, orig, new)
