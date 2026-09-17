"""fpf_glue_v2 — restructured-loop kernel overlay over the shipped fpf_triatt_pro / fpf_triatt_epi / fpf_transition kernels.  Default OFF; enable with PTX_GLUE_V2=1.

What it does (install() — call BEFORE ptx_trunk2_levers.apply_from_env() binds the kernel callables, i.e. from an interpreter-startup hook
(sitecustomize) that runs ahead of the levers):
  * fpf_triatt_pro.prologue.triatt_prologue   -> wrapper: ln_mode=='stock' and a GLUE cell for (C, H*D, arch) -> prologue v4 kernel (pipelined W-chunk loop); else the shipped v3.
  * fpf_triatt_pro.prologue.triatt_prologue_padded (if present: the padded / PAD8 provider path) -> same condition -> prologue_v4_padded (padded output pitches); else shipped.
  * fpf_triatt_epi.epilogue.triatt_epilogue   -> wrapper: o_layout=='ihjd' and a GLUE cell for (c, H, D, arch)  -> epilogue v3 kernel (K-chunk loop);           else the shipped v2/v1.
  * fpf_transition.transition._launch         -> wrapper: ln_mode==0, C==256 and a GLUE cell for (C, NH, arch)   -> transition kernel with warp_specialize;     else the shipped kernel.
Everything else (caches, layouts, block-mode residual semantics, fallbacks, counters) is the shipped code; the wrappers only swap the Triton kernel that runs a VERIFIED cell.
Cells: GLUE_CELLS.json next to this file (schema in the file); resolved by compute capability (torch.cuda.get_device_capability) and triton major.minor (resolve_cells: exact (cc, triton), else the
capability's cells named 'unverified on this triton', else no cell BY NAME — measured-off capability / unknown capability -> the shipped kernels serve, said once).
Exactness: every cell shipped here is torch.equal to the shipped kernels' output on the real stock activation dumps 356/546/705/813/1493 (both regimes, start+end nodes, r2r x3) on the
named (cc, triton).  The K-accumulation chains are identical to the shipped kernels by construction (same operands, rounding points and ascending K order; only loop structure and
tiling differ), so the overlay is eligible for a kit's exact mode; the kit's composed determinism check decides.
"""
import os, json, sys
import torch

__version__ = "2.2-glue-sm100-20260823-rc2"
_HERE = os.path.dirname(os.path.abspath(__file__))
_STATE = {"installed": False, "resolved": False, "patched": set(), "cells": {}, "why": "", "calls": {"prologue_v4": 0, "prologue_v4_padded": 0, "epilogue_v3": 0, "transition_ws": 0, "prologue_passthru": 0, "prologue_padded_passthru": 0, "epilogue_passthru": 0, "transition_passthru": 0}}


def _say(msg):
    sys.stderr.write(f"[fpf_glue_v2] {msg}\n")


def _triton_mm():
    try:
        import triton
        p = triton.__version__.split(".")
        return f"{p[0]}.{p[1]}"
    except Exception:
        return "?"


MEASURED_OFF_CC = {"10.3": "B300: the certified compositions run the shipped kernels (no GLUE cell measured on cc 10.3)",
                   "8.0": "A100: the certified compositions run the shipped kernels (no GLUE cell measured on cc 8.0)"}   # capabilities the measured compositions run with this overlay OFF, by name


def resolve_cells(device=None, path=None, *, cc=None, triton=None):
    """-> dict family -> {cfg, shape, ...} for this (cc, triton) or {} (the shipped kernels serve every family; ``_STATE['why']`` names which case).
    Resolution: the exact ``<cc>|<triton M.m>`` entry; else the capability's ``by_cc`` entry — applied and NAMED when the running triton is not one the cells were
    measured on (``triton <mm> unverified``; ``FPF_GLUE_V2_ALLOW_TRITON=0`` keeps the shipped kernels there instead, by name); else no cell: a capability the measured
    compositions run with this overlay off (MEASURED_OFF_CC: ``off: not-measured``) or an UNKNOWN capability — this overlay carries per-capability kernel variants only, so
    the shipped kernels ARE its capability-free settings (``unknown capability``).  ``cc`` / ``triton`` name the environment for a decision off the device."""
    path = path or os.environ.get("FPF_GLUE_CELLS_JSON") or os.path.join(_HERE, "GLUE_CELLS.json")
    try:
        with open(path) as _fh:
            doc = json.load(_fh)
    except Exception as e:
        _STATE["why"] = f"cells file unreadable ({e!r})"; return {}
    if cc is None:
        if not torch.cuda.is_available():
            _STATE["why"] = "no cuda"; return {}
        dev = device if device is not None else torch.device("cuda", torch.cuda.current_device())
        cap = torch.cuda.get_device_capability(dev)
        key_cc = f"{cap[0]}.{cap[1]}"
    else:
        key_cc = str(cc)
    tmm = triton if triton is not None else _triton_mm()
    ent = doc.get("by_cc_triton", {}).get(f"{key_cc}|{tmm}")
    note = ""
    if ent is None:
        ent = doc.get("by_cc", {}).get(key_cc)
        if ent is not None and tmm not in doc.get("triton_tested", []):
            if os.environ.get("FPF_GLUE_V2_ALLOW_TRITON", "") == "0":                  # the explicit engineering opt-out: the shipped kernels, by name
                _say(f"cc {key_cc} cells were measured on triton {doc.get('triton_tested')}, running triton {tmm}: FPF_GLUE_V2_ALLOW_TRITON=0 -> cells NOT applied (the shipped kernels serve)")
                _STATE["why"] = f"triton {tmm} unverified, opted out (FPF_GLUE_V2_ALLOW_TRITON=0)"; return {}
            _say(f"WARNING: cc {key_cc} cells were measured on triton {doc.get('triton_tested')}, running triton {tmm}: the capability's cells '{ent}' APPLY, unverified on this triton "
                 f"(re-check with selftest.py; FPF_GLUE_V2_ALLOW_TRITON=0 keeps the shipped kernels)")
            note = f" (triton {tmm} unverified)"
    if ent is None:
        if key_cc in MEASURED_OFF_CC:
            _STATE["why"] = f"no cells for cc {key_cc} (off: not-measured — {MEASURED_OFF_CC[key_cc]})"
        else:
            _STATE["why"] = f"no cells for cc {key_cc} (unknown capability: the shipped kernels serve — this overlay carries per-capability kernel variants only; cells exist for cc {sorted(doc.get('by_cc', {}))})"
        return {}
    cells = doc["entries"][ent]
    _STATE["why"] = f"cc {key_cc} triton {tmm} -> entry '{ent}'{note}"
    return {k: v for k, v in cells.items() if isinstance(v, dict) and "cfg" in v}


def _fpf_append(path, txt):
    with open(path, "a") as _fh:
        _fh.write(txt)


def install(verbose=True, modules=None):
    """Idempotent, per module. modules: iterable of already-imported module names to patch (default: import and patch all three).
    Returns the dict of active cells (family -> cell); empty dict = everything stays on the shipped kernels."""
    if not _STATE["cells"] and not _STATE.get("resolved"):
        cells = resolve_cells()
        fams = os.environ.get("PTX_GLUE_V2_FAMILIES", "prologue,epilogue,transition").split(",")     # e.g. PTX_GLUE_V2_FAMILIES=epilogue,transition keeps the shipped prologue
        _STATE["cells"] = {k: v for k, v in cells.items() if k in fams}; _STATE["resolved"] = True
        if verbose:
            _say((f"cells ({_STATE['why']}): " + ", ".join(f"{k}={v['cfg']}" for k, v in _STATE["cells"].items())) if _STATE["cells"] else f"no-op ({_STATE['why']})")
        try:
            import atexit
            rep = os.environ.get("PTX_LEVER_REPORT")
            if rep:
                def _dump():
                    try:
                        _fpf_append(rep, json.dumps({"fpf_glue_v2": {"version": __version__, "why": _STATE["why"], "cells": {k: v["cfg"] for k, v in _STATE["cells"].items()}, "patched": sorted(_STATE["patched"]), "calls": _STATE["calls"]}}) + "\n")
                    except Exception:
                        pass
                atexit.register(_dump)
        except Exception:
            pass
    cells = _STATE["cells"]
    if not cells:
        _STATE["installed"] = True; return cells
    from . import kernels as GK
    want = {"prologue": "fpf_triatt_pro.prologue", "epilogue": "fpf_triatt_epi.epilogue", "transition": "fpf_transition.transition"}
    if modules is None:
        import importlib
        modules = [importlib.import_module(m).__name__ for m in want.values()]
    modules = set(modules)

    if "prologue" in cells and want["prologue"] in modules and "prologue" not in _STATE["patched"]:
        import fpf_triatt_pro.prologue as PRO
        pc = dict(cells["prologue"]["cfg"]); pshape = tuple(cells["prologue"]["shape"])          # (C, HD)
        _orig_pro = PRO.triatt_prologue
        def triatt_prologue(module, z, ending=False, ln_mode="fused", x_ln=None, write_x=False, cfg=None, **kw):
            if ln_mode == "stock" and not write_x and cfg is None and not kw and x_ln is not None and x_ln.is_cuda and not ending:
                cch = PRO.get_cache(module, x_ln.device)
                if (cch["C"], cch["H"] * cch["D"]) == pshape and x_ln.dim() == 3 and x_ln.is_contiguous() and x_ln.dtype == torch.bfloat16:
                    _STATE["calls"]["prologue_v4"] += 1
                    return GK.prologue_v4(module, x_ln, cch, pc)
            _STATE["calls"]["prologue_passthru"] += 1
            return _orig_pro(module, z, ending=ending, ln_mode=ln_mode, x_ln=x_ln, write_x=write_x, cfg=cfg, **kw)
        triatt_prologue.__wrapped__ = _orig_pro
        PRO.triatt_prologue = triatt_prologue; _STATE["patched"].add("prologue")
        # the PADDED-layout entry (the PAD8-EXACT / padded provider path calls triatt_prologue_padded) -> same v4 kernel with padded output pitches
        _orig_pad = getattr(PRO, "triatt_prologue_padded", None)
        if _orig_pad is not None:
            def triatt_prologue_padded(module, z, out, ending=False, ln_mode="stock", x_ln=None, fma_flags=(True, True, True), cfg=None, **kw):
                if ln_mode == "stock" and cfg is None and not kw and not ending and x_ln is not None and x_ln.is_cuda and isinstance(out, dict):
                    cch = PRO.get_cache(module, x_ln.device)
                    if (cch["C"], cch["H"] * cch["D"]) == pshape and x_ln.dim() == 3 and x_ln.is_contiguous() and x_ln.dtype == torch.bfloat16 and all(k_ in out for k_ in ("q", "k", "v", "bias")):
                        _STATE["calls"]["prologue_v4_padded"] = _STATE["calls"].get("prologue_v4_padded", 0) + 1
                        return GK.prologue_v4_padded(module, x_ln, cch, pc, out)
                _STATE["calls"]["prologue_padded_passthru"] = _STATE["calls"].get("prologue_padded_passthru", 0) + 1
                return _orig_pad(module, z, out, ending=ending, ln_mode=ln_mode, x_ln=x_ln, fma_flags=fma_flags, cfg=cfg, **kw)
            triatt_prologue_padded.__wrapped__ = _orig_pad
            PRO.triatt_prologue_padded = triatt_prologue_padded; _STATE["patched"].add("prologue_padded")

    if "epilogue" in cells and want["epilogue"] in modules and "epilogue" not in _STATE["patched"]:
        import fpf_triatt_epi.epilogue as EPI
        ec = dict(cells["epilogue"]["cfg"]); eshape = tuple(cells["epilogue"]["shape"])          # (c, H, D)
        _orig_epi = EPI.triatt_epilogue
        def triatt_epilogue(o, g, wo16, z=None, *, ending=False, residual=False, out=None, cfg=None, woT16=None, o_layout="ihjd", **kw):
            if cfg is None and o_layout == "ihjd" and not kw and o.is_cuda and o.dim() in (4, 5) and o.dtype == torch.bfloat16:
                H, D = o.shape[-3], o.shape[-1]
                if (wo16.shape[0], H, D) == eshape and (H * D) % ec.get("KC", 64) == 0:
                    _STATE["calls"]["epilogue_v3"] += 1
                    return GK.epilogue_v3(o, g, wo16, z, ending=ending, residual=residual, out=out, cfg=ec, woT16=woT16)
            _STATE["calls"]["epilogue_passthru"] += 1
            return _orig_epi(o, g, wo16, z, ending=ending, residual=residual, out=out, cfg=cfg, woT16=woT16, o_layout=o_layout, **kw)
        triatt_epilogue.__wrapped__ = _orig_epi
        EPI.triatt_epilogue = triatt_epilogue; _STATE["patched"].add("epilogue")

    if "transition" in cells and want["transition"] in modules and "transition" not in _STATE["patched"]:
        import fpf_transition.transition as TR
        tc = dict(cells["transition"]["cfg"]); tshape = tuple(cells["transition"]["shape"])       # (C, NH)
        _orig_launch = TR._launch
        def _launch(y2d, cache, out2d, res2d=None, ln_mode=0, **kw):
            if ln_mode == 0 and not kw and y2d.is_cuda and y2d.dim() == 2 and y2d.shape[1] == tshape[0] and cache["wo16"].shape[1] == tshape[1] and y2d.stride(1) == 1 and out2d.stride(1) == 1:
                _STATE["calls"]["transition_ws"] += 1
                return GK.transition_v3(y2d, cache, out2d, res2d=res2d, cfg=tc)
            _STATE["calls"]["transition_passthru"] += 1
            return _orig_launch(y2d, cache, out2d, res2d=res2d, ln_mode=ln_mode, **kw)
        _launch.__wrapped__ = _orig_launch
        TR._launch = _launch; _STATE["patched"].add("transition")

    _STATE["installed"] = all(f in _STATE["patched"] for f in cells)
    return cells


def stats():
    return dict(_STATE)
