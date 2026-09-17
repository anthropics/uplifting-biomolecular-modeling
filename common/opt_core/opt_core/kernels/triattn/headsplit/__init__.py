"""fpf_triatt_headsplit — kit lever `triatt_headsplit_exact`.  EXACT-BY-CONSTRUCTION.

WHAT: in the exact tier's BLK2 tri-attention statement (prologue -> cuEquivariance triangle_attention -> epilogue), above N_SPLIT tokens the
attention core is called ONCE PER HEAD instead of once for all 8 heads.  cuDNN's sm80 fmha (what cuequivariance 0.11 runs on sm_90) re-reads
the fp32 pair bias [8, N, N] per row tile; at N > ~1024 that bias (>= 34 MB) no longer fits H100's L2 and the kernel falls from ~90 to ~63
TF/s.  One head's bias slice (N^2*4 B) stays L2-resident; every CTA computes exactly what it computed before, so outputs are BITWISE equal
(torch.equal per-head vs full: True at every size measured; the lever also shape-checks the first call of every new shape (bitwise) and REFUSES BY
NAME on a mismatch — never a silent fallback).  Measured per statement core (H100, cuequivariance 0.11.1, stock row chunks): 1200 tok 27.0 -> 17.4
ms (x1.55), 1400 42.9 -> 25.2 ms (x1.70), 2000 131.6 -> 79.9 ms (x1.65); <= 1024 tok: x1.01 (not engaged).

HOW (zero-copy): cuequivariance's public wrapper copies any non-contiguous q/k/v/bias.  A head slice of the prologue's [I,H,J,D] q is not
contiguous, so the lever also swaps in a HEAD-MAJOR variant of the MK-PF F1 prologue (prologue_hm.py: the opt_core LN/projection jit helpers
themselves, only the q/k/v store address differs -> [H,I,J,D] storage returned as [I,H,J,D]-shaped views), after which each head's
(row-chunk of) q/k/v and bias[:, :, h] are contiguous and go to cuequivariance untouched.  Per-head outputs are concatenated into the
[R,H,J,D] tensor the epilogue reads (one extra pass over o: +3 % of the new statement time; a cat-free epilogue is follow-up work).
Row chunks (blk2_chunked_exact), PAD8 (padded buffers re-interpreted head-major, pads kept zero) and the XL lean 'prologue' mode compose.

ENVELOPE: the MECHANISM is a call pattern through cuequivariance's public API (no arch-specific code), so the lever engages on every device the
exact block path runs on: split iff H > 1 and key length N > n_split(device) = max(1024, floor(sqrt(0.6 * L2_bytes / 32))) — i.e. when the
8-head fp32 bias (8*N^2*4 B) exceeds ~0.6x the device's L2 (H100 50 MB -> 991 -> floored at stock's chunking threshold 1024; A100 40 MB ->
1024; a 126 MB-L2 part -> 1577).  Zero-copy ('zc' route) when q/k/v arrive head-major (the F1 head-major prologue below: wherever MK-PF F1
serves the node); otherwise the per-head slices are copied by cuequivariance itself ('copy' route: same bytes as the one copy a stock
transposed-view caller already pays; +3 slab copies per head for a contiguous kit prologue output — still a large net win above n_split).
TESTED cell: cc 9.0 (H100, cuequivariance 0.11.1, torch 2.13/cu130).  Any other device is NAMED 'untested' on the activation line and
engages; the per-shape bitwise shape check (first call of every new (site, rows, H, N, D, mask) shape: torch.equal(split, full) or REFUSE BY
NAME) is the guard everywhere.  Routes outside the envelope are declared and counted ('pro_passthru', '<site>_single').
SWITCH: PTX_TRIATT_HEADSPLIT=1 (exported by the kit's modes/env plumbing; inert when unset).  Ablation: MODEL_OPT_LEVERS_OFF=triatt_headsplit_exact.
COUNTERS: ptx_trunk2_levers._STATS["headsplit"] = compact live tally {zc, copy, single, pro_hm, pro_pass, heads, chk_eq, chk_ne} for the kit's
LEVER/EXIT line (rendered `headsplit=zc:..,copy:..,single:..,pro_hm:..,pro_pass:..,heads:..,chk_eq:..,chk_ne:..`; an envelope-inert item reads
zc:0,copy:0,single:n = engaged with 0 eligible calls, never a fallback); report() / the
PTX_LEVER_REPORT json line {"fpf_triatt_headsplit": {...}, "unit", "pid"} carries the full per-site counters (pro_hm / pro_hm_padded /
pro_passthru / pro_padded_passthru, tl_split_zc / tl_split_copy / tl_single, pad8_split_zc / pad8_split_copy / pad8_single, head_calls,
masked_split, min/max_n_split) and shapecheck {shapes, eq, ne}.
"""
from __future__ import annotations
import os, sys, json, atexit
import torch

__version__ = "1.0.2"
LEVER = "triatt_headsplit_exact"
SWITCH = "PTX_TRIATT_HEADSPLIT"
N_SPLIT_FLOOR = 1024           # never split at or below stock's chunking threshold (<= 1024 the measured gain is x1.01: not worth a layout switch); the engaged regime is always row-chunked
L2_FRACTION = 0.6              # split when the 8-head fp32 bias 8*N^2*4 B exceeds this fraction of the device L2.  H100 policy curve (per-head vs full, cuequivariance 0.11.1, sustained
                               # clocks): 896 x1.01, 960 x1.01, 1024 x1.01, 1088 x1.13, 1152 x1.24, 1200 x1.55-1.61,
                               # 1400 x1.70-1.82, 2000 x1.65; the cliff sits where 8*N^2*4 B passes ~30 MB of H100's 50 MB (2 x 25 MB partitions).
N_SPLIT = N_SPLIT_FLOOR        # per-device value fixed at install() from torch.cuda.get_device_properties(dev).L2_cache_size
TESTED_CC = ((9, 0),)          # measured cells (bitwise + end-to-end byte-identical outputs); every other device engages and is NAMED untested (lever policy: an untested environment is named, never disengaged)

_STATE = {"installed": False, "why": "", "patched": [], "cells": None, "rebound": None,
          "device": {}, "untested": None,
          "calls": {"pro_hm": 0, "pro_hm_padded": 0, "pro_passthru": 0, "pro_padded_passthru": 0,
                    "tl_split_zc": 0, "tl_split_copy": 0, "tl_single": 0, "pad8_split_zc": 0, "pad8_split_copy": 0, "pad8_single": 0,
                    "head_calls": 0, "masked_split": 0, "max_n_split": 0, "min_n_split": 0},
          "shapecheck": {"shapes": [], "eq": 0, "ne": 0}}
_C = _STATE["calls"]
# Compact per-route tally for the kit's LEVER / EXIT line (ptx_trunk2_levers._STATS["headsplit"], report.TALLY_KEYS): an envelope-inert item reads
# 'engaged with 0 eligible calls' (zc=0 copy=0 single=n), never a fallback.  zc/copy/single = attention calls (both sites), pro_hm/pro_pass = prologue
# calls (unpadded + padded), heads = per-head kernel calls issued, chk_eq/chk_ne = per-shape bitwise shape checks passed/failed (ne > 0 never survives: it raises).
_T = {"zc": 0, "copy": 0, "single": 0, "pro_hm": 0, "pro_pass": 0, "heads": 0, "chk_eq": 0, "chk_ne": 0}


class Refused(RuntimeError):
    """Raised BY NAME when the lever cannot serve (install-time envelope) or when a bitwise shape check fails (run time). Never caught by the lever."""


def _say(msg):
    print(f"[FPF] fpf_triatt_headsplit {msg}", file=sys.stderr, flush=True)


def enabled_by_env() -> bool:
    return (os.environ.get(SWITCH, "0") or "0") not in ("", "0")


def stats() -> dict:
    return {"version": __version__, "lever": LEVER, "installed": _STATE["installed"], "why": _STATE["why"], "patched": list(_STATE["patched"]),
            "n_split": N_SPLIT, "device": dict(_STATE["device"]), "untested": _STATE["untested"], "tested_cc": [list(c) for c in TESTED_CC],
            "cells": _STATE["cells"], "rebound_closure_cells": _STATE["rebound"],
            "tally": dict(_T), "calls": dict(_C), "shapecheck": {"shapes": [list(s) for s in _STATE["shapecheck"]["shapes"]], "eq": _STATE["shapecheck"]["eq"], "ne": _STATE["shapecheck"]["ne"]}}


report = stats


def marker() -> str:
    """The word appended to ptx_trunk2_levers._STATS['applied'] (ACTIVE/LEVER census)."""
    if not _STATE["installed"]:
        return f"HEADSPLIT:refused({_STATE['why']})"
    pro = "f1hm" if "prologue" in _STATE["patched"] else "none(copy-route)"
    cell = f"UNTESTED({_STATE['untested']})" if _STATE["untested"] else "tested"
    return f"HEADSPLIT:on(n>{N_SPLIT},L2={_STATE['device'].get('l2_mb')}MB,g=1,prologue={pro},sites={'+'.join(p for p in _STATE['patched'] if p in ('tl', 'pad8'))},cell={cell})"


def _write_report(path):
    try:
        with open(path, "a") as fh:
            fh.write(json.dumps({"fpf_triatt_headsplit": stats(), "unit": "fpf_triatt_headsplit", "pid": os.getpid()}) + "\n")
    except Exception as e:  # pragma: no cover
        _say(f"report write failed: {e!r}")


# ----------------------------------------------------------------------------------------------------------------------------- layout predicates
def _is_head_major(t) -> bool:
    """True for a [.., R, H, S, D] tensor whose per-head [R, S, D] slabs are each contiguous and disjoint (head-major storage seen through a
    permuted view): stride(D)=1, stride(S)=D, stride(R)=S*D, stride(H) >= R*S*D; leading dims (if any) of size 1."""
    if t.dim() == 5:
        if t.shape[0] != 1:
            return False
    elif t.dim() != 4:
        return False
    R, H, S, D = (int(x) for x in t.shape[-4:])
    if H < 2:
        return False
    sr, sh, ss, sd = t.stride(-4), t.stride(-3), t.stride(-2), t.stride(-1)
    return sd == 1 and ss == D and sr == S * D and sh >= R * S * D


def _narrow_head(t, h):
    v = t.narrow(-3, h, 1)
    if getattr(t, "_fpf_padfilled", False):          # pad8 core contract tag (bias pad columns already hold BIAS_PAD_FILL) must survive the slicing
        try:
            v._fpf_padfilled = True
        except Exception:
            pass
    return v


def _as_tensor(o):
    return o[0] if isinstance(o, (tuple, list)) else o


# ----------------------------------------------------------------------------------------------------------------------------- the split (attention call sites)
def _make_split(orig, site):
    """Wrap a triangle_attention-like callable f(q, k, v, bias, mask=None, scale=None, ...): above N_SPLIT keys (and H > 1) call `orig` once per
    head and concatenate the outputs along the head dim — zero-copy when q/k/v arrive head-major (the lever's own prologue produced them; route
    'zc'), through cuequivariance's own per-head copies otherwise (route 'copy'); anything else passes through untouched (counted 'single')."""
    sc = _STATE["shapecheck"]

    def split_attn(q, k, v, bias, mask=None, scale=None, *a, **kw):
        if not (torch.is_tensor(q) and torch.is_tensor(k) and torch.is_tensor(v) and q.dim() in (4, 5) and k.dim() == q.dim() and v.dim() == q.dim()
                and int(q.shape[-3]) > 1 and int(k.shape[-2]) > N_SPLIT and (q.dim() == 4 or int(q.shape[0]) == 1)
                and torch.is_tensor(bias) and bias.dim() >= 3 and int(bias.shape[-3]) == int(q.shape[-3])):
            _C[site + "_single"] += 1; _T["single"] += 1
            return orig(q, k, v, bias, mask, scale, *a, **kw)
        H = int(q.shape[-3]); S = int(k.shape[-2]); R = int(q.shape[-4])
        zc = _is_head_major(q) and _is_head_major(k) and _is_head_major(v)          # zero-copy (head-major storage) vs cuequivariance's own per-head .contiguous() copies
        outs = []
        for h in range(H):
            outs.append(_as_tensor(orig(_narrow_head(q, h), _narrow_head(k, h), _narrow_head(v, h), _narrow_head(bias, h), mask, scale, *a, **kw)))
        o = torch.cat(outs, dim=-3)
        _C[site + ("_split_zc" if zc else "_split_copy")] += 1; _C["head_calls"] += H; _T["zc" if zc else "copy"] += 1; _T["heads"] += H
        if mask is not None: _C["masked_split"] += 1
        _C["max_n_split"] = max(_C["max_n_split"], S); _C["min_n_split"] = S if not _C["min_n_split"] else min(_C["min_n_split"], S)
        key = (site, R, H, S, int(q.shape[-1]), mask is not None, zc)
        if key not in sc["shapes"]:
            # bitwise shape check (first call of every new shape): the full call on row-major copies (what the statement did before this lever) must equal the per-head result.
            ref = _as_tensor(orig(q.contiguous(), k.contiguous(), v.contiguous(), bias, mask, scale, *a, **kw))
            eq = (ref.shape == o.shape) and bool(torch.equal(ref, o))
            del ref
            sc["shapes"].append(key); sc["eq" if eq else "ne"] += 1; _T["chk_eq" if eq else "chk_ne"] += 1
            if not eq:
                raise Refused(f"fpf_triatt_headsplit ({LEVER}): REFUSED — per-head split != full call (torch.equal False) at site={site} rows={R} H={H} N={S} "
                              f"mask={'yes' if mask is not None else 'no'} route={'zc' if zc else 'copy'} device={_STATE['device']}; run with MODEL_OPT_LEVERS_OFF={LEVER} and report this line")
        return o

    split_attn.__wrapped__ = orig
    split_attn._fpf_headsplit = site
    return split_attn


# ----------------------------------------------------------------------------------------------------------------------------- install
def _cells_from_mkpf():
    """The F1 cell fpf_mkpf serves with (None when MK-PF F1 is not installed on this row: then no head-major prologue -> the split runs the 'copy' route)."""
    try:
        import fpf_mkpf as MKP
    except ImportError:
        return None
    st = getattr(MKP, "_STATE", {})
    if not st.get("installed") or "triatt_prologue" not in st.get("patched", ()):
        return None
    cells = st.get("cells") or MKP.resolve_cells()
    if not cells or "f1" not in cells:
        return None
    arith = os.environ.get("FPF_MKPF_LN", cells.get("ln_arith", "fused"))            # == fpf_mkpf.install()'s own expression
    fma = tuple(bool(int(f)) for f in cells.get("fma_flags", [1, 1, 1]))
    return dict(cfg=dict(cells["f1"]["cfg"]), ln_arith=arith, fma_flags=list(fma), mkpf_class=str(cells.get("class", "")), mkpf_levers=sorted(st.get("levers") or []))


def _device_policy():
    """n_split for this device: max(N_SPLIT_FLOOR, floor(sqrt(L2_FRACTION * L2_bytes / (8 heads * 4 B))))."""
    import math
    dev = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(dev)
    l2 = int(getattr(props, "L2_cache_size", 0) or 0)
    src = "device"
    if l2 <= 0:
        l2 = 50 * 2**20; src = "assumed(50MB: torch exposes no L2_cache_size)"
    n_l2 = int(math.floor(math.sqrt(L2_FRACTION * l2 / 32.0)))
    cc = (int(props.major), int(props.minor))
    return dict(name=str(props.name), cc=list(cc), l2_bytes=l2, l2_mb=round(l2 / 2**20), l2_source=src, n_l2=n_l2, n_split=max(N_SPLIT_FLOOR, n_l2)), cc


def install(verbose=True):
    """Engage the lever (idempotent): the per-head split at the cuequivariance call sites on every device (n_split keyed on its L2), plus the
    head-major F1 prologue wherever MK-PF F1 serves the node (zero-copy route).  Raises Refused (by name) only when nothing can engage
    (no CUDA / the stock attention module is not importable)."""
    global N_SPLIT
    if _STATE["installed"]:
        return stats()
    if not torch.cuda.is_available():
        raise Refused("fpf_triatt_headsplit: CUDA not available")
    pol, cc = _device_policy()
    N_SPLIT = int(pol["n_split"]); _STATE["device"] = pol
    _STATE["untested"] = None if cc in TESTED_CC else f"cc{cc[0]}.{cc[1]}"
    cells = _cells_from_mkpf()
    _STATE["cells"] = cells

    # (1) prologue: head-major F1 above N_SPLIT (wraps fpf_mkpf's wrapper; everything it does not take passes through to it).  No F1 on this row -> copy route only.
    if cells is not None:
        _install_prologue(cells)

    # (2) attention call sites: the stock wrapper name the BLK2 core (and the stock statement) calls, and the pad8 provider's captured cuequivariance entry
    import protenix.model.triangular.layers as TL
    if not getattr(TL.cuequivariance_triangular_attn, "_fpf_headsplit", None):
        TL.cuequivariance_triangular_attn = _make_split(TL.cuequivariance_triangular_attn, "tl"); _STATE["patched"].append("tl")
    try:
        import fpf_cueq_pad8exact as P8
        if getattr(P8, "_cue_tri", None) is not None and not getattr(P8._cue_tri, "_fpf_headsplit", None):
            P8._cue_tri = _make_split(P8._cue_tri, "pad8"); _STATE["patched"].append("pad8")
    except ImportError:
        pass
    _STATE["installed"] = True
    _STATE["why"] = (f"{pol['name']} cc {cc[0]}.{cc[1]} L2={pol['l2_mb']}MB({pol['l2_source']}) -> n_split=max({N_SPLIT_FLOOR},{pol['n_l2']})={N_SPLIT}; "
                     + (f"f1 cell {cells['cfg']} arith={cells['ln_arith']}" if cells else "no MK-PF F1 on this row: prologue unchanged, split via cuequivariance's per-head copies")
                     + ("" if not _STATE["untested"] else f"; UNTESTED device {_STATE['untested']} (tested: {['%d.%d' % c for c in TESTED_CC]}) — engaged, guarded by the per-shape bitwise shape check"))
    try:
        import ptx_trunk2_levers as LEV
        LEV._STATS["applied"].append(marker()); LEV._STATS["headsplit"] = _T          # live compact tally -> trunk record / EXIT line (TALLY_KEYS += ("headsplit",)); full counters in our own json line
    except Exception:
        pass
    rep = os.environ.get("PTX_LEVER_REPORT")
    if rep:
        atexit.register(_write_report, rep)
    if verbose:
        _say(f"v{__version__} installed: {marker()} :: {_STATE['why']} patched={_STATE['patched']} rebound={_STATE['rebound']}")
    return stats()


def _install_prologue(cells):
    import ptx_trunk2_levers as LEV
    import fpf_triatt_pro.prologue as PRO
    import fpf_mkpf as MKP
    from . import prologue_hm as HM
    f1cfg, arith, fma = cells["cfg"], cells["ln_arith"], tuple(cells["fma_flags"])
    inner = PRO.triatt_prologue
    def triatt_prologue(module, z, ending=False, ln_mode="fused", x_ln=None, **kw):
        if ln_mode == "mkpf" and z.dim() == 3 and int(z.shape[0] if ending else z.shape[1]) > N_SPLIT:
            _C["pro_hm"] += 1; _T["pro_hm"] += 1
            out = HM.prologue_ln_hm(module, z, PRO.get_cache(module, z.device), f1cfg, ending=bool(ending), ln_arith=arith, fma_flags=fma)
            return out
        _C["pro_passthru"] += 1; _T["pro_pass"] += 1
        return inner(module, z, ending=ending, ln_mode=ln_mode, x_ln=x_ln, **kw)
    triatt_prologue.__wrapped__ = inner
    PRO.triatt_prologue = triatt_prologue; _STATE["patched"].append("prologue")
    _STATE["rebound"] = None
    try:                                     # the block lever closed over the prologue it from-imported at apply time: repoint that cell (fpf_mkpf's own adapter)
        MKP._rebind_levers_prologue(LEV, inner, triatt_prologue)
        _STATE["rebound"] = MKP._STATE.get("rebound_closure_cells")
    except Exception as e:
        _STATE["rebound"] = f"ERR {e!r}"[:120]
    inner_pad = getattr(PRO, "triatt_prologue_padded", None)
    if inner_pad is not None:
        def triatt_prologue_padded(module, z, out, ending=False, ln_mode="stock", x_ln=None, **kw):
            if ln_mode == "mkpf" and z.dim() == 3 and int(z.shape[0] if ending else z.shape[1]) > N_SPLIT:
                _C["pro_hm_padded"] += 1; _T["pro_hm"] += 1
                res = HM.prologue_ln_hm(module, z, PRO.get_cache(module, z.device), f1cfg, ending=bool(ending), ln_arith=arith, fma_flags=fma, out=out)
                return res
            HM.mark_padded_layout(out, "rm")
            _C["pro_padded_passthru"] += 1; _T["pro_pass"] += 1
            return inner_pad(module, z, out, ending=ending, ln_mode=ln_mode, x_ln=x_ln, **kw)
        triatt_prologue_padded.__wrapped__ = inner_pad
        PRO.triatt_prologue_padded = triatt_prologue_padded; _STATE["patched"].append("prologue_padded")
