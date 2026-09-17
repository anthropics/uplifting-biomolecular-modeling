"""lnl_fused.py — Triton fusion kernels for the pairformer's non-triangle pair ops, LayerNorm fused into the consuming linear (H100, bf16 pair stream under bf16 autocast).

1. fused_transition(x, ln_w, ln_b, W1, W2, W3, eps)  ==  linear_3( silu(linear_1(LN(x))) * linear_2(LN(x)) )   (the pair Transition, n*c hidden)
   One kernel: LN in fp32 on the register tile -> bf16 -> for each hidden chunk: two bf16 MMAs (fp32 acc) -> round to bf16 exactly where the stock
   graph rounds (GEMM outputs, silu output, product) -> third MMA accumulated in fp32 -> bf16 out.  The n*c-wide intermediate never touches HBM.
   Numerics: same rounding points as stock under autocast (bf16 operands, fp32 accumulation); accumulation ORDER differs from cuBLAS -> not bit-exact (tier 2, same class).
2. ln_linear(x, ln_w, ln_b, W[NOUT,C], eps, write_y, transpose)  ->  (y16 = bf16(LN(x)) written in [b,i,j] or transposed [b,j,i] row order | None,  out = bf16(y16 @ W^T) [.., NOUT])
   Used for AttentionPairBias  (B = to_b(ln_0(Z)); y not written)  and the TriangleAttention prologue (y = norm(pair) as bf16, transposed for the ending node; bias = to_b(y)).
All kernels: last dim C in {64, 128} (constexpr), rows flattened, CUDA only. Fallback decisions are made by the caller.
"""
import torch, triton, triton.language as tl

_TCFG_ALL = [triton.Config({"BM": 128, "BH": 64}, num_warps=8, num_stages=2), triton.Config({"BM": 64, "BH": 64}, num_warps=4, num_stages=2),
             triton.Config({"BM": 64, "BH": 128}, num_warps=4, num_stages=2), triton.Config({"BM": 128, "BH": 128}, num_warps=8, num_stages=1),
             # added for the Blackwell pass (sm_100 / sm_103): deeper pipelines and larger row tiles
             triton.Config({"BM": 128, "BH": 64}, num_warps=8, num_stages=3), triton.Config({"BM": 128, "BH": 128}, num_warps=8, num_stages=2),
             triton.Config({"BM": 256, "BH": 64}, num_warps=8, num_stages=2), triton.Config({"BM": 256, "BH": 128}, num_warps=16, num_stages=2),
             triton.Config({"BM": 64, "BH": 64}, num_warps=4, num_stages=3)]
import json as _json, os as _os

from opt_core.kernels import safe_settings as _safe     # the core's ONE cc-keyed resolution + safe-settings mechanism (stdlib at import)


def _cc_key():
    try:
        return "%d.%d" % torch.cuda.get_device_capability()
    except Exception:
        return "unknown"


class BuildFailed(RuntimeError):
    """This lever cannot run in the process: its SAFE single-stage tile failed to build too (or a kernel without an alternative tile failed to
    build). A kit turns it into its hard error naming the one-flag escape (`--mode off`, or its explicit opt-out for this lever)."""
    kind = "build_failed"


_NET = _safe.SafeNet("lnl_fused", refused=BuildFailed)          # this lever's safety net (safe_state / settings_word / the ONE line)
_ROW_STATE: dict = {}                                           # kernel name -> "row" (this cc's tiles row narrows the autotuner) | "none" (no row: the safe tile) | "all" (PF_LNL_AUTOTUNE_ALL)
_SAFE_LEVER = {"_fused_transition_kernel": "lnl:transition", "_ln_linear_kernel": "lnl:ln_linear"}


def _safe_config(kernel_name):
    """The kernel's SAFE tile (opt_core.kernels.safe_settings.SAFE_ROWS: the smallest tile of its configuration space, one stage) as a triton.Config."""
    row = _safe.safe_row(_SAFE_LEVER[kernel_name], None if _cc_key() == "unknown" else _cc_key())
    s = dict(row["settings"])
    return triton.Config({k: v for k, v in s.items() if k not in ("num_warps", "num_stages")}, num_warps=int(s.get("num_warps", 4)), num_stages=int(s.get("num_stages", 1)))


def safe_state():
    """``{"on", "reason", "where", "note"}``: whether this process serves the SAFE tiles (a build failure of the tuned tiles) / the no-row note."""
    return _NET.snapshot()


def settings_word():
    """The LEVER line's ``settings=`` fact: ``"safe:build_failed:<exception class>"`` once the safe tiles serve, else None."""
    return _NET.word()


def cells_note():
    """The LEVER line's ``cells_note=`` fact: ``"default:no_row"`` when no tiles row names this capability (the default space serves), else None."""
    return _NET.note()


def _tile_table():
    """per-arch tile table carried beside this module as lnl_fused.tiles_by_arch.json (PF_LNL_TILES overrides the path) {cc: {kernel_name: {autotune_key_str: {kwargs..., num_warps, num_stages}}}}.
    When a row exists for (cc, kernel) every launch takes the row's tile for its autotune key -- the row's principal tile for a key the row
    does not name -- WITHOUT a search (the autotuner's cache is pinned: _PinnedTiles); no row for this capability = the module's default tile,
    pinned likewise; the full list is searched only under PF_LNL_AUTOTUNE_ALL=1 (the sweep that writes this table)."""
    p = _os.environ.get("PF_LNL_TILES") or _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "lnl_fused.tiles_by_arch.json")
    try:
        with open(p) as fh:
            return _json.load(fh)
    except Exception as e:                                       # an unreadable table is NAMED once (never a silent default): the default tile space serves
        if _TILES_STATE.get("error") is None:
            _TILES_STATE["error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
            import sys
            print("[opt_core/lnl_fused] tiles table unreadable (%s: %s): the default tile serves every kernel (pinned, no search; tiles=unreadable)"
                  % (p, _TILES_STATE["error"]), file=sys.stderr, flush=True)
        return {}


_TILES_STATE = {"error": None}                                   # tiles_state(): 'unreadable:<why>' once the table failed to load, else None


def tiles_state():
    """``unreadable:<Exc>: <msg>`` when the tiles table could not be read in this process (the default tile space serves), else None — a fact for the lever line."""
    return None if _TILES_STATE.get("error") is None else "unreadable:" + _TILES_STATE["error"]


class _PinnedTiles(dict):
    """The autotuner's per-key cache in serving mode: EVERY key answers without a benchmark -- the tiles row's config for a key the row names
    (keys are the autotuner's own key tuples, recorded by autotune_report() on the stack that measured them), else the row's principal config
    (its most frequent one).  ``untabled`` records the keys that met the principal config (a fact for the sweep: shapes the table lacks)."""

    def __init__(self, pins, principal):
        super().__init__(pins)
        self.principal = principal
        self.untabled = []

    def __contains__(self, key):                  # the autotuner asks `key in cache` before benchmarking: always answered here
        return True

    def __missing__(self, key):
        k = str(key)
        if k not in self.untabled:
            self.untabled.append(k)
        return self.principal


_PINS: dict = {}                                                # kernel name -> _PinnedTiles (serving mode) once its configs were resolved


def _row_pins(row, configs):
    """({autotune key tuple: triton.Config}, principal Config) of a tiles row, the Configs taken BY VALUE from `configs` (the row's distinct set)."""
    import ast as _ast
    from collections import Counter
    def sig_of(cfg):
        return (tuple(sorted((k, int(v)) for k, v in cfg.items() if k not in ("num_warps", "num_stages"))), int(cfg.get("num_warps", 4)), int(cfg.get("num_stages", 2)))
    by_sig = {(tuple(sorted((k, int(v)) for k, v in dict(c.kwargs).items())), int(c.num_warps), int(c.num_stages)): c for c in configs}
    pins, count = {}, Counter()
    for key_str, cfg in row.items():
        c = by_sig.get(sig_of(cfg))
        if c is None:
            continue
        count[sig_of(cfg)] += 1
        try:
            key = _ast.literal_eval(key_str)
        except (ValueError, SyntaxError):
            continue
        pins[key if isinstance(key, tuple) else (key,)] = c
    principal = by_sig[count.most_common(1)[0][0]] if count else configs[0]
    return pins, principal


def _configs_for(kernel_name, full):
    """The autotuner's configuration list for this compute capability: the tiles row's distinct configs (each launch takes ITS key's config
    from the pinned cache, no search: _PinnedTiles), the full space under PF_LNL_AUTOTUNE_ALL=1 (the sweep), else -- no row for this
    capability -- the module's default tile alone (full[0]; ONE info line at the first launch; a BUILD failure there drops to the SAFE tile)."""
    if _os.environ.get("PF_LNL_AUTOTUNE_ALL", "0") == "1":
        _ROW_STATE[kernel_name] = "all"
        return full
    row = _tile_table().get(_cc_key(), {}).get(kernel_name)
    if not row:
        _ROW_STATE[kernel_name] = "none"          # no row for this capability: the default tile, pinned (no search), said ONCE at the first launch
        _PINS[kernel_name] = _PinnedTiles({}, full[0])
        return full[:1]
    _ROW_STATE[kernel_name] = "row"
    seen, out = set(), []
    for cfg in row.values():
        kw = {k: v for k, v in cfg.items() if k not in ("num_warps", "num_stages")}
        sig = (tuple(sorted(kw.items())), cfg.get("num_warps"), cfg.get("num_stages"))
        if sig in seen: continue
        seen.add(sig); out.append(triton.Config(kw, num_warps=int(cfg.get("num_warps", 4)), num_stages=int(cfg.get("num_stages", 2))))
    if not out:
        _PINS[kernel_name] = _PinnedTiles({}, full[0])
        return full[:1]
    pins, principal = _row_pins(row, out)
    _PINS[kernel_name] = _PinnedTiles(pins, principal)
    return out


def _run(kernel_name, autotuner, launch):
    """``launch()`` under the lever's safety net (opt_core.kernels.safe_settings): the autotuner keeps this capability's tiles; a BUILD failure of them
    switches the process to the kernel's SAFE tile (ONE line, settings=safe:build_failed:<exc>); no tiles row for this capability = the default space
    with ONE info line (cells_note=default:no_row); the SAFE tile failing to build raises BuildFailed. Every other exception propagates."""
    where = _safe.where_word(None if _cc_key() == "unknown" else _cc_key(), _safe.triton_mm())
    safe = [_safe_config(kernel_name)]

    def sig(cfgs):                               # triton.Config by value: tile kwargs, warps, stages (+ ctas / maxnreg where the triton has them)
        return [(sorted(dict(c.kwargs).items()), c.num_warps, c.num_stages, getattr(c, "num_ctas", None), getattr(c, "maxnreg", None)) for c in cfgs]

    def build(cfgs):
        if sig(autotuner.configs) != sig(cfgs):      # switching the tile set resets the autotuner's per-key picks
            autotuner.configs = list(cfgs)
            cache = getattr(autotuner, "cache", None)
            if isinstance(cache, dict) and not isinstance(cache, _PinnedTiles):
                cache.clear()
            if sig(cfgs) == sig(safe):               # the SAFE tile serves from here: every key answers it, no search
                autotuner.cache = _PinnedTiles({}, cfgs[0])
        elif kernel_name in _PINS and not isinstance(getattr(autotuner, "cache", None), _PinnedTiles) and _ROW_STATE.get(kernel_name) != "all":
            autotuner.cache = _PINS[kernel_name]     # serving mode: the row's per-key tiles, pinned (the autotuner never benchmarks)
        return launch()

    if _ROW_STATE.get(kernel_name) == "none" and not _NET.on:      # no tiles row names this capability: the default space serves (ONE info line); a build failure below -> the safe tile
        _NET.inform("no row for cc %s" % _cc_key(), where, tuned=_safe.table_ccs(_tile_table()))
    return _NET.run(build, list(autotuner.configs), safe, where=where)


def pinned_state():
    """{kernel: {"mode": row|none|all|-, "pinned_keys": n, "untabled": [key str, ...]}}: serving-mode facts (no benchmark ever ran unless mode == all)."""
    out = {}
    for name in ("_fused_transition_kernel", "_ln_linear_kernel"):
        pc = _PINS.get(name)
        out[name] = {"mode": _ROW_STATE.get(name, "-"), "pinned_keys": (len(dict.keys(pc)) if pc is not None else 0), "untabled": (list(pc.untabled) if pc is not None else [])}
    return out


def autotune_report():
    """{kernel: {key: {BM.., num_warps, num_stages}}} of what the autotuner picked in this process (for lnl_fused.tiles_by_arch.json; under
    PF_LNL_AUTOTUNE_ALL=1 these are measured picks, otherwise the pinned tiles that served)."""
    rep = {}
    for name, obj in (("_fused_transition_kernel", globals().get("_fused_transition_kernel")), ("_ln_linear_kernel", globals().get("_ln_linear_kernel"))):
        cache = getattr(obj, "cache", None) or {}
        r = {}
        for key, cfg in cache.items():
            try:
                r[str(key)] = {**{k: int(v) for k, v in cfg.kwargs.items()}, "num_warps": int(cfg.num_warps), "num_stages": int(cfg.num_stages)}
            except Exception as e:
                r[str(key)] = {"error": repr(e)[:80]}
        rep[name] = r
    return {"cc": _cc_key(), "triton": triton.__version__, "kernels": rep}


_TCFG = _configs_for("_fused_transition_kernel", _TCFG_ALL)


@triton.autotune(configs=_TCFG, key=["MB", "C", "HID", "RESIDUAL"])
@triton.jit
def _fused_transition_kernel(X, Y, LNW, LNB, W1, W2, W3, M, MB, eps, C: tl.constexpr, HID: tl.constexpr, RESIDUAL: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = tl.arange(0, C)
    x = tl.load(X + rows[:, None].to(tl.int64) * C + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, 1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(LNW + cols).to(tl.float32)
    b = tl.load(LNB + cols).to(tl.float32)
    y16 = (xc * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    acc = tl.zeros([BM, C], dtype=tl.float32)
    for h0 in range(0, HID, BH):
        hcols = h0 + tl.arange(0, BH)
        w1t = tl.load(W1 + hcols[None, :] * C + cols[:, None])      # [C, BH]  = W1[hchunk,:]^T
        w2t = tl.load(W2 + hcols[None, :] * C + cols[:, None])
        a = tl.dot(y16, w1t)                                        # fp32 [BM, BH]
        g = tl.dot(y16, w2t)
        a16 = a.to(tl.bfloat16).to(tl.float32)                      # stock: bf16 GEMM outputs
        g16 = g.to(tl.bfloat16).to(tl.float32)
        s16 = (a16 / (1.0 + tl.exp(-a16))).to(tl.bfloat16).to(tl.float32)   # silu computed in fp32 on bf16 value, rounded to bf16 (torch opmath)
        h16 = (s16 * g16).to(tl.bfloat16)                           # bf16 * bf16 -> fp32 product rounded to bf16
        w3t = tl.load(W3 + cols[None, :] * HID + hcols[:, None])    # [BH, C]  = W3[:, hchunk]^T
        acc += tl.dot(h16, w3t)
    if RESIDUAL:   # stock: Z (bf16) + update (bf16) -> fp32 add -> bf16 ; x tile is already in registers
        outv = (x + acc.to(tl.bfloat16).to(tl.float32)).to(tl.bfloat16)
    else:
        outv = acc.to(tl.bfloat16)
    tl.store(Y + rows[:, None].to(tl.int64) * C + cols[None, :], outv, mask=rmask[:, None])


def fused_transition(x, ln_w, ln_b, W1, W2, W3, eps=1e-5, residual=False):
    """x [..., C] (bf16 or fp32, contiguous), W1/W2 [HID, C] bf16, W3 [C, HID] bf16 -> bf16 [..., C] (= x + transition(x) if residual; residual requires bf16 x)."""
    C = x.shape[-1]; HID = W1.shape[0]
    assert W1.shape == (HID, C) and W2.shape == (HID, C) and W3.shape == (C, HID)
    xs = x.contiguous().view(-1, C); M = xs.shape[0]
    y = torch.empty((M, C), device=x.device, dtype=torch.bfloat16)
    MB = 0 if M < 65536 else (1 if M < 400000 else 2)   # autotune key bucket
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)
    _run("_fused_transition_kernel", _fused_transition_kernel, lambda: _fused_transition_kernel[grid](xs, y, ln_w, ln_b, W1, W2, W3, M, MB, eps, C=C, HID=HID, RESIDUAL=bool(residual)))
    return y.view(x.shape)


_LCFG_ALL = [triton.Config({"BM": 64}, num_warps=4, num_stages=2), triton.Config({"BM": 128}, num_warps=4, num_stages=2), triton.Config({"BM": 128}, num_warps=8, num_stages=2),
             triton.Config({"BM": 256}, num_warps=8, num_stages=2), triton.Config({"BM": 128}, num_warps=4, num_stages=3), triton.Config({"BM": 64}, num_warps=2, num_stages=3)]
_LCFG = _configs_for("_ln_linear_kernel", _LCFG_ALL)


@triton.autotune(configs=_LCFG, key=["MB", "C", "NOUT", "WRITE_Y", "TRANSPOSE"])
@triton.jit
def _ln_linear_kernel(X, Y, OUT, LNW, LNB, W, M, MB, I, J, eps, C: tl.constexpr, NOUT: tl.constexpr, WRITE_Y: tl.constexpr, TRANSPOSE: tl.constexpr, BM: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    cols = tl.arange(0, C)
    r64 = rows.to(tl.int64)
    x = tl.load(X + r64[:, None] * C + cols[None, :], mask=rmask[:, None], other=0.0).to(tl.float32)
    mean = tl.sum(x, 1) / C
    xc = x - mean[:, None]
    var = tl.sum(xc * xc, 1) / C
    rstd = 1.0 / tl.sqrt(var + eps)
    w = tl.load(LNW + cols).to(tl.float32)
    b = tl.load(LNB + cols).to(tl.float32)
    y16 = (xc * rstd[:, None] * w[None, :] + b[None, :]).to(tl.bfloat16)
    if WRITE_Y:
        if TRANSPOSE:
            bidx = r64 // (I * J); rem = r64 - bidx * (I * J); ii = rem // J; jj = rem - ii * J
            dst = bidx * (I * J) + jj * I + ii
        else:
            dst = r64
        tl.store(Y + dst[:, None] * C + cols[None, :], y16, mask=rmask[:, None])
    ncols = tl.arange(0, NOUT)
    wt = tl.load(W + ncols[None, :] * C + cols[:, None])            # [C, NOUT] = W^T
    o = tl.dot(y16, wt)                                             # fp32 [BM, NOUT]
    tl.store(OUT + r64[:, None] * NOUT + ncols[None, :], o.to(tl.bfloat16), mask=rmask[:, None])


def ln_linear(x, ln_w, ln_b, W, eps=1e-5, write_y=False, transpose=False):
    """x [B, I, J, C] or [I, J, C] (contiguous, bf16/fp32); W [NOUT, C] bf16 with NOUT a power of two >= 16 (pad by the caller).
    Returns (y16 | None, out): y16 = bf16(LN(x)) as [B, I, J, C] (or [B, J, I, C] holding LN(x)[b,i,j] at [b,j,i] when transpose), out = bf16(y16 @ W^T) [B, I, J, NOUT] (never transposed)."""
    squeeze = x.dim() == 3
    x4 = x.unsqueeze(0) if squeeze else x
    B, I, J, C = x4.shape; NOUT = W.shape[0]
    xs = x4.contiguous().view(-1, C); M = xs.shape[0]
    out = torch.empty((B, I, J, NOUT), device=x.device, dtype=torch.bfloat16)
    y = torch.empty((B, J, I, C) if transpose else (B, I, J, C), device=x.device, dtype=torch.bfloat16) if write_y else out
    MB = 0 if M < 65536 else (1 if M < 400000 else 2)
    grid = lambda meta: (triton.cdiv(M, meta["BM"]),)
    _run("_ln_linear_kernel", _ln_linear_kernel, lambda: _ln_linear_kernel[grid](xs, y, out, ln_w, ln_b, W, M, MB, I, J, eps, C=C, NOUT=NOUT, WRITE_Y=bool(write_y), TRANSPOSE=bool(transpose)))
    if squeeze:
        out = out[0]; y = y[0] if write_y else None
    return (y if write_y else None), out


# 3. gate_transpose(O[B,I,H,J,D] (attention output, internal layout), G = gate logits living at columns [goff, goff+H*D) of a [B,I,J,ldg] tensor)
#    -> Y[B,I,J,H*D] = O.permute(0,1,3,2,4) * sigmoid(G)   (start node)   or   Y[B,J,I,H*D] (TRANSPOSE: ending node, ready for a contiguous to_out GEMM)
#    replaces: sigmoid kernel + strided mul + (end node) transpose copy.  Rounding as torch: s = bf16(sigmoid(g)), y = bf16(o * s).
@triton.jit
def _gate_transpose_kernel(O, G, Y, I, J, ldg, goff, H: tl.constexpr, D: tl.constexpr, BR: tl.constexpr, TRANSPOSE: tl.constexpr):
    pid = tl.program_id(0)
    nj = tl.cdiv(J, BR)
    bi = pid // nj; jb = pid - bi * nj            # bi = b*I + i
    b = bi // I; i = bi - b * I
    js = jb * BR + tl.arange(0, BR); jm = js < J
    cs = tl.arange(0, H * D); h = cs // D; d = cs - h * D
    bi64 = bi.to(tl.int64)
    o_ptr = O + ((bi64 * H + h[None, :]) * J + js[:, None]) * D + d[None, :]
    o = tl.load(o_ptr, mask=jm[:, None], other=0.0).to(tl.float32)
    g_ptr = G + (bi64 * J + js[:, None]) * ldg + goff + cs[None, :]
    g = tl.load(g_ptr, mask=jm[:, None], other=0.0).to(tl.float32)
    s16 = (1.0 / (1.0 + tl.exp(-g))).to(tl.bfloat16).to(tl.float32)
    y = (o * s16).to(tl.bfloat16)
    if TRANSPOSE:
        y_ptr = Y + ((b.to(tl.int64) * J + js[:, None]) * I + i) * (H * D) + cs[None, :]
    else:
        y_ptr = Y + (bi64 * J + js[:, None]) * (H * D) + cs[None, :]
    tl.store(y_ptr, y, mask=jm[:, None])


def gate_transpose(o, g_src, goff, transpose):
    """o [B,I,H,J,D] contiguous bf16; g_src [B,I,J,ldg] contiguous (gate logits at columns goff:goff+H*D) -> bf16 [B,I,J,H*D] or [B,J,I,H*D] (transpose)."""
    B, I, H, J, D = o.shape; ldg = g_src.shape[-1]
    assert o.is_contiguous() and g_src.is_contiguous() and g_src.shape[:3] == (B, I, J)
    y = torch.empty((B, J, I, H * D) if transpose else (B, I, J, H * D), device=o.device, dtype=torch.bfloat16)
    BR = 64
    grid = (B * I * triton.cdiv(J, BR),)
    try:
        _gate_transpose_kernel[grid](o, g_src, y, I, J, ldg, goff, H=H, D=D, BR=BR, TRANSPOSE=bool(transpose), num_warps=4)
    except _safe.catchable() as e:                      # this kernel has one tile: a BUILD failure means the lever cannot run here (safe_settings case iii)
        if not _safe.is_build_failure(e):
            raise
        raise BuildFailed("lnl_fused: the gate/transpose kernel failed to build (%s): %s: %s — this lever cannot run in this process; run the kit with `--mode off` or its explicit opt-out for this lever"
                          % (_safe.where_word(None if _cc_key() == "unknown" else _cc_key(), _safe.triton_mm()), type(e).__name__, str(e)[:200])) from e
    return y
