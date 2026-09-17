"""fpf_trimul_v4.generic — MODULE-AGNOSTIC Tier-2 TriangleMultiplication.  No dependence on any model's attribute names: you pass tensors.

    from fpf_trimul_v4 import generic as G
    w = G.pack_weights(ln_in_w=..., ln_in_b=..., w_ag=..., w_ap=..., w_bg=..., w_bp=..., ln_out_w=..., ln_out_b=..., w_o=..., w_og=...,
                       b_ag=None, b_ap=None, b_bg=None, b_bp=None, b_o=None, b_og=None, cache_owner=my_module)      # once; cached on cache_owner if given
    out = G.trimul(z, mask, direction="outgoing", weights=w, residual=False)                                           # every call
    # or fused spec: G.pack_weights(..., w_proj=W[?,C], proj_split=("ag","ap","bg","bp")  (row-block order of W), b_proj=b or None, ...)

Math (x = LayerNorm_in(z); per pair (i,j); 'outgoing' / 'incoming' = the update over outgoing / incoming edges):
    a_ij = sigmoid(x_ij W_ag^T + b_ag) * (x_ij W_ap^T + b_ap) * mask_ij      b_ij likewise            (D channels each)
    X_ij = sum_k a_ik * b_jk  (outgoing)   |   sum_k a_ki * b_kj  (incoming)
    out_ij = sigmoid(x_ij W_og^T + b_og) * (LayerNorm_out(X_ij) W_o^T + b_o)   [+ z_ij if residual]
Served cell: CUDA; z [N,N,C] or [B,N,N,C], C = c_z in {128,256}; D in {128,256}; z dtype bf16 (fast path; fp32 z is served by the pointer-load K1 with fp32 LN input and fp32 output);
mask None / [N,N] / [1,N,N] / [B,N,N] (float or bool; multiplies a and b); N >= N_MIN (no upper size limit: the workspace, workspace_bytes(), is allocated like any torch operation's);
a table.json row for (cc|triton).  Anything else raises TrimulUnsupported (callers keep their own stock path; `supported(...)` answers without raising).  Precision class: bf16
tensor-core GEMMs with fp32 accumulation, fp32 LayerNorm statistics, fp32 gating, rounding points of the cuEq fused TriMul (Tier-2 vs an fp32 torch module; same class as
torch-under-bf16-autocast).  Deterministic: fixed tiles, no atomics, no split-K; CUDA-graph capturable.
Batch: a leading batch B is served as ONE launch set (kernels.trimul_v4_forward on the [B,N,N,C] tensor: K1 / K3 carry b on grid axis 2, one cuBLAS GEMM over the B*D planes) —
serving word BATCH_MODE = "native" (env FPF_TRIMUL_V4_BATCH, default).  BATCH_MODE = "loop" serves the batch as B single-plane launch sets (b = 0 .. B-1 in a Python
loop); both words are recorded in COUNTS["batch_mode"].  Every batch element gets the single-plane arithmetic under either word; K1 / K3 planes are bit-exact the loop's by
construction, the contraction is the same cuBLAS problem per plane at batch count B*D instead of D (bit-exact-equal on the tested rows of the batch equality test).
A batch beyond the limits of ONE launch set (kernels.batch_launch_limit: 'tma_rows>int32' = the descriptor K1's / K3's int32 row coordinate over the batch's B*N*N token rows,
i.e. z of >= 512 GiB at C = 128; 'grid_batch>65535' = CUDA grid axis 2) is served under "native" as B single-plane launch sets — a named, counted event
(COUNTS["batch_loop_fallback"][name] and one printed line), never a launch with a truncated coordinate; kernels.trimul_v4_forward itself refuses such a batch by name
(kernels.BatchLimit) before allocating anything.
The module-signature provider `fpf_trimul_v4.trimul:fn` is a thin wrapper over `trimul_packed` (byte-identical outputs to calling it directly)."""
import atexit, os, sys, json, torch
from . import __version__
from . import kernels as K
from . import cells as CELLS
from . import kdesc as KD

__all__ = ["trimul", "trimul_packed", "pack_weights", "supported", "TrimulUnsupported", "COUNTS", "N_MIN", "BATCH_MODE", "BATCH_MODES", "workspace_bytes", "probe", "SAME_CLASS_RMS", "SAME_CLASS_MAX"]
N_MIN = int(os.environ.get("FPF_TRIMUL_V4_NMIN", "101"))       # below it the vendor TriMul runs a different (small-N torch) algorithm: that regime is the stock path's, not this cell's
_STOCK_ROUND = os.environ.get("FPF_TRIMUL_V4_STOCK_ROUND", "1") != "0"
BATCH_MODES = ("native", "loop")                                # how a leading batch is served: ONE launch set over [B,N,N,C] | B single-plane launch sets (see the module docstring)
BATCH_MODE = os.environ.get("FPF_TRIMUL_V4_BATCH", "native")
if BATCH_MODE not in BATCH_MODES:
    raise ImportError("FPF_TRIMUL_V4_BATCH=%r: must be one of %s" % (BATCH_MODE, BATCH_MODES))
COUNTS = {"version": __version__, "generic_calls": 0, "generic_served": 0, "unsupported": {}, "first_call": None, "batch_mode": BATCH_MODE, "batched_calls": 0, "batch_loop_fallback": {}, "probe": {},
          "kernels": K.LAUNCHES}          # per-kernel launch tally (kernels.LAUNCHES: k1d / k1t / k1c, k3d / k3c — which cell implementation served, decided per call)
SAME_CLASS_RMS = 1.25        # the numerics class this kernel's cells are admitted in (TIER2 same class): the served kernels' error vs an fp64 statement is at most 1.25 x (rms) and
SAME_CLASS_MAX = 2.5         # 2.5 x (max-abs) the bf16-autocast torch statement's own error — the op-level admission bar every row of table.json passed; the warm probe holds each served
                             # shape class to it once per process
PROBE_CHUNK_BYTES = 128 << 20                                   # the warm probe's fp64 statement and bf16-class yardstick are evaluated per OUTPUT BLOCK (rows I x cols J) whose
_PROBE_MEM = {"blocked_refs": 0, "max_block_rows": 0, "released": 0, "stamp_hits": 0, "stamp_writes": 0, "runs": 0}   # transients stay under this budget (was: whole [n, n, C|D] fp64 tensors)
PROBE_STAMP_SCHEMA = "opt_core.fpf_trimul_v4.probe_stamp/v1"   # a PASSED probe verdict is stamped per (package digest + version, shape class, sizes, cell, bars, device cc + name,
PROBE_STAMP_LEAF = "fpf_trimul_v4"                              # driver, CUDA, torch, triton, python) through kernels.trimul.native's stamp directory chain (leaf fpf_trimul_v4)
PROBE_N = 128                # the warm probe's pair size (>= N_MIN; one [N, N, C] tensor per shape class, once per process and device)
_PROBES = {}                 # (device, C, D, has_bias, fp32 input) -> {"ok", "ratio_max", "ratio_rms", "cell"}
_WARM = set()                # (device, C, D, has_bias, fp32 input, residual, masked): classes whose first call (probe + first launches = Triton compiling the cell's config) has run


TrimulUnsupported = K.TrimulUnsupported     # raised (before any kernel launch) when the inputs are outside the served cell; .reason is a short machine-readable tag; ONE class for the
                                             # kernel-level and the module-agnostic entry (kernels.BatchLimit, the launch-set limit refusal of kernels.trimul_v4_forward, is a subclass)


def workspace_bytes(N, D, C=None, elem_out=2, pad=16, B=1):
    """The per-call transient device bytes of the served path — a documented formula (HAZARDS), not a gate: the a|b planes [2, B, D, Np, Np] and the x planes
    [B, D, Np, Np] in bf16 that kernels.trimul_v4_forward allocates (Np = kernels.ceil_to(N, pad)), plus the [B, N, N, C] output it returns (elem_out bytes per
    element; C defaults to D) = B * (3*D*Np*Np*2 + N*N*C*elem_out); for C = D and bf16 z that is 4 x the bytes of z.  N=705, D=C=128, bf16, pad 16, B=1: Np=720,
    3*128*720*720*2 + 705*705*128*2 = 525,369,600.  Under BATCH_MODE "native" the whole batch's workspace is live at once (B x the single-plane figure); under
    "loop" one plane set at a time plus the [B, N, N, C] output.  Nothing consults device memory before a call: the workspace is allocated like any torch operation's
    and an out-of-memory propagates to the caller as torch's own OutOfMemoryError."""
    N, D, pad, B = int(N), int(D), int(pad), int(B)
    Np = K.ceil_to(N, pad)
    return B * (3 * D * Np * Np * 2 + N * N * int(D if C is None else C) * int(elem_out))


def _log(msg):
    sys.stderr.write("[fpf_trimul_v4.generic] %s\n" % msg); sys.stderr.flush()


_DIRS = {"outgoing": True, "out": True, "incoming": False, "in": False}


def pack_weights(*, ln_in_w, ln_in_b, ln_out_w, ln_out_b, w_o, w_og,
                 w_ag=None, w_ap=None, w_bg=None, w_bp=None, b_ag=None, b_ap=None, b_bg=None, b_bp=None,
                 w_proj=None, b_proj=None, proj_split=("ag", "ap", "bg", "bp"),
                 b_o=None, b_og=None, cache_owner=None, cache_key="default"):
    """Pack (and optionally cache on `cache_owner._fpf_cache`) the weights for `trimul(..., weights=w)`.
    Either give the four [D, C] matrices w_ag (a gate), w_ap (a projection), w_bg, w_bp (+ optional [D] biases), or ONE fused w_proj [4D, C] (+ optional b_proj [4D]) with
    proj_split naming the row-block order, e.g. ("ap","bp","ag","bg") for a module whose fused Linear emits [a_proj | b_proj | a_gate | b_gate].
    ln_in_* [C]; ln_out_* [D]; w_o [C, D] (+ b_o [C]); w_og [C, C] (+ b_og [C]).  GEMM weights are used in bf16 (cast as autocast would), LN params and biases in fp32."""
    if cache_owner is not None:
        cache = getattr(cache_owner, "_fpf_cache", None)
        if cache is None:
            cache = cache_owner._fpf_cache = {}
        k = ("trimul_v4_generic", cache_key)
        if k in cache:
            return cache[k]
    if w_proj is not None:
        assert w_proj.dim() == 2 and w_proj.shape[0] % 4 == 0, "w_proj must be [4D, C]"
        D = w_proj.shape[0] // 4
        blocks = dict(zip(proj_split, torch.split(w_proj.detach(), D, 0)))
        bblocks = dict(zip(proj_split, torch.split(b_proj.detach(), D, 0))) if b_proj is not None else {}
        assert set(blocks) == {"ag", "ap", "bg", "bp"}, "proj_split must be a permutation of ('ag','ap','bg','bp')"
        w_ag, w_ap, w_bg, w_bp = blocks["ag"], blocks["ap"], blocks["bg"], blocks["bp"]
        if bblocks:
            b_ag, b_ap, b_bg, b_bp = bblocks["ag"], bblocks["ap"], bblocks["bg"], bblocks["bp"]
    assert all(t is not None for t in (w_ag, w_ap, w_bg, w_bp)), "give w_ag/w_ap/w_bg/w_bp or w_proj"
    w = K.pack_generic(ln_in_w=ln_in_w, ln_in_b=ln_in_b, w_ag=w_ag, w_ap=w_ap, w_bg=w_bg, w_bp=w_bp, b_ag=b_ag, b_ap=b_ap, b_bg=b_bg, b_bp=b_bp,
                       ln_out_w=ln_out_w, ln_out_b=ln_out_b, w_o=w_o, w_og=w_og, b_o=b_o, b_og=b_og)
    if cache_owner is not None:
        cache_owner._fpf_cache[("trimul_v4_generic", cache_key)] = w
    return w


def supported(z, mask=None, *, weights=None, c_z=None, d_hidden=None, n_min=None):      # n_min: the caller's token floor for THIS call class (the core table's small-N cert, kernels.trimul v4_n_min); None = N_MIN
    """(ok: bool, reason: str) without raising or launching anything."""
    try:
        _check(z, mask, weights, c_z, d_hidden, n_min=n_min); return True, "ok"
    except TrimulUnsupported as e:
        return False, e.reason


def _check(z, mask, w, c_z, d_hidden, *, n_min=None):            # n_min keyword-only (4.4.6): the positional face (z, mask, w, c_z, d_hidden) is the 4.4.4 one
    if not torch.is_tensor(z) or not z.is_cuda:
        raise TrimulUnsupported("device", "z must be a CUDA tensor")
    if z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3]:
        raise TrimulUnsupported("shape", "z must be [N,N,C] or [B,N,N,C], got %s" % (tuple(z.shape),))
    C = int(z.shape[-1]); N = int(z.shape[-2])
    if c_z is not None and C != int(c_z):
        raise TrimulUnsupported("c_z", "z has C=%d, expected %d" % (C, c_z))
    if C not in K.SUPPORTED_C:
        raise TrimulUnsupported("c_z", "C=%d not in %s" % (C, K.SUPPORTED_C))
    if w is not None:
        if w["C"] != C:
            raise TrimulUnsupported("c_z", "weights packed for C=%d, z has %d" % (w["C"], C))
        if w["D"] not in K.SUPPORTED_D:
            raise TrimulUnsupported("d_hidden", "D=%d not in %s" % (w["D"], K.SUPPORTED_D))
    if d_hidden is not None and int(d_hidden) not in K.SUPPORTED_D:
        raise TrimulUnsupported("d_hidden", "D=%d not in %s" % (d_hidden, K.SUPPORTED_D))
    if z.dtype not in (torch.bfloat16, torch.float32):
        raise TrimulUnsupported("dtype", "z dtype %s (bf16 fast path; fp32 accepted)" % z.dtype)
    nmin = N_MIN if n_min is None else int(n_min)                   # a per-call floor below N_MIN is the core table's admitted floor for the class (kernels.trimul v4_n_min); the
    if N < nmin:                                                    # kernels, their launch cells and every call at N >= N_MIN are unchanged (the binaries that serve 101..511 serve 2..100)
        raise TrimulUnsupported("N<%d" % nmin, "N=%d" % N)
    if mask is not None:
        if not torch.is_tensor(mask) or tuple(mask.shape[-2:]) != (N, N) or mask.dim() not in (2, 3) or (mask.dim() == 3 and z.dim() == 4 and mask.shape[0] not in (1, z.shape[0])) or (mask.dim() == 3 and z.dim() == 3 and mask.shape[0] != 1):
            raise TrimulUnsupported("mask-shape", "mask %s for z %s" % (tuple(mask.shape), tuple(z.shape)))
    cfg = CELLS.cell_for(z.device)
    if cfg is None:                                                       # no row and no SAFE cell for this capability (no-cell), or the lever went off by name (none:<why>)
        raise TrimulUnsupported(CELLS.off_word() or "no-cell", CELLS.INFO.get(str(z.device), {}).get("source", ""))
    D = int(w["D"]) if w is not None else (int(d_hidden) if d_hidden is not None else C)
    pr = _PROBES.get((str(z.device), C, D, bool(w.get("has_bias")) if w is not None else False, z.dtype == torch.float32))
    if pr is not None and not pr["ok"]:                                   # the warm probe refused this shape class in this process: the caller's stock TriMul, by that name
        raise TrimulUnsupported("probe-failed", pr["line"])
    k1, k3 = K.resolve_cfg(cfg, C, D, bool(w.get("has_bias")) if w is not None else False)
    lim = K.batch_launch_limit(1, N, k1, z.dtype == torch.float32, k3)  # ONE pair representation beyond a launch-set limit (more than 2^31 - 1 token rows, N*N, under the TMA K1: from
    if lim is not None:                                                  # N = 46341, z of >= 512 GiB at C = 128) cannot be served plane by plane either: refused by that name, before any launch
        raise TrimulUnsupported(lim, "N=%d: N*N token rows per pair representation" % N)
    return cfg


def trimul_packed(z, mask, *, outgoing, weights, residual=False, eps=1e-5, stock_round=None, pad=16, n_min=None):
    """Hot path with pre-packed weights (see pack_weights). Returns a NEW tensor (z's dtype): update, or z + update if residual.
    The FIRST call of a (device, C, D, bias, input dtype, residual, masked) class -- the warm probe and the launches that compile the cell's one
    configuration -- runs through ``opt_core.kernels.trimul._pystack.padded_call`` (the interpreter's stack-chunk boundary never sits under a
    Triton front end or a library import made from here); every later call of the class dispatches directly."""
    wk = (str(z.device), int(weights["C"]), int(weights["D"]), bool(weights.get("has_bias")), z.dtype == torch.float32, bool(residual), mask is not None)
    if wk not in _WARM:
        _WARM.add(wk)
        try:
            from opt_core.kernels.trimul import _pystack as _PS, stock_preload_once as _spo
        except ImportError:                                                                # pragma: no cover - the package served without the core's provider face: direct
            return _trimul_packed(z, mask, outgoing=outgoing, weights=weights, residual=residual, eps=eps, stock_round=stock_round, pad=pad, n_min=n_min)
        _spo()                                                                             # the stack's cuequivariance import happens here (padded, once per process), not mid-item at a model's lazy import
        return _PS.padded_call(_trimul_packed, z, mask, outgoing=outgoing, weights=weights, residual=residual, eps=eps, stock_round=stock_round, pad=pad, n_min=n_min)   # 4.4.6: the FIRST (padded) call of a class carries the caller's floor too (4.4.5 dropped it here: one N<101 refusal on the first small item)
    return _trimul_packed(z, mask, outgoing=outgoing, weights=weights, residual=residual, eps=eps, stock_round=stock_round, pad=pad, n_min=n_min)


def _trimul_packed(z, mask, *, outgoing, weights, residual=False, eps=1e-5, stock_round=None, pad=16, n_min=None):
    COUNTS["generic_calls"] += 1
    try:
        cfg = _check(z, mask, weights, None, None, n_min=n_min)
    except TrimulUnsupported as e:
        COUNTS["unsupported"][e.reason] = COUNTS["unsupported"].get(e.reason, 0) + 1
        raise
    N = int(z.shape[-2])
    pk = (str(z.device), int(weights["C"]), int(weights["D"]), bool(weights.get("has_bias")), z.dtype == torch.float32)
    if pk not in _PROBES:                                                                  # the warm probe: once per (device, shape class), before the first served call of the class
        try:
            if probe(z.device, pk[1], pk[2], pk[3], pk[4]) is not None:
                cfg = _check(z, mask, weights, None, None, n_min=n_min)                                 # the probe may have switched the process to the SAFE cell, turned the lever off, or refused the class
        except TrimulUnsupported as e:
            COUNTS["unsupported"][e.reason] = COUNTS["unsupported"].get(e.reason, 0) + 1
            raise
    zs = z if z.dim() == 4 else z.unsqueeze(0)
    B = int(zs.shape[0])
    sr = _STOCK_ROUND if stock_round is None else bool(stock_round)
    def fwd(zz, mm, c):                                        # every launch set runs under the lever's safety net (cells.run: build failure -> SAFE cell; SAFE failing -> lever off by name)
        try:
            return CELLS.run(lambda cc_: K.trimul_v4_forward(zz, bool(outgoing), mm, weights, cc_, eps=eps, residual=bool(residual), stock_round=sr, pad=pad), c, z.device)
        except TrimulUnsupported as e:
            COUNTS["unsupported"][e.reason] = COUNTS["unsupported"].get(e.reason, 0) + 1
            raise
    lim = None
    if BATCH_MODE == "native" and B > 1:                       # the two limits of ONE launch set (kernels.batch_launch_limit: the TMA K1's int32 row coordinate over B*N*N rows, CUDA grid
        k1, k3 = K.resolve_cfg(cfg, weights["C"], weights["D"], bool(weights.get("has_bias")))  # axis 2 over B): a batch beyond them is served plane by plane — a NAMED, COUNTED event
        lim = K.batch_launch_limit(B, N, k1, z.dtype == torch.float32, k3)                # (COUNTS["batch_loop_fallback"][name], one printed line per name), never a truncated launch
        if lim is not None:
            if lim not in COUNTS["batch_loop_fallback"]:
                print("[fpf_trimul_v4.generic] BATCH LOOP FALLBACK by name: %s (B = %d, N = %d): this call and every later call over that limit is served as B single-plane launch sets "
                      "(bitwise the same output; the loop's speed)" % (lim, B, N), file=sys.stderr, flush=True)
            COUNTS["batch_loop_fallback"][lim] = COUNTS["batch_loop_fallback"].get(lim, 0) + 1
    if (BATCH_MODE == "native" and lim is None) or B == 1:    # ONE launch set over the whole batch (a [N,N,C] / B = 1 call is the single-plane launch set under either word)
        zc = zs if zs.is_contiguous() else zs.contiguous()
        out = fwd(zc, mask, cfg)
        if B > 1:
            COUNTS["batched_calls"] += 1
    else:                                                      # "loop" (or a native call over a launch-set limit): B single-plane launch sets
        ms = None
        if mask is not None:
            ms = mask if mask.dim() == 3 else mask.unsqueeze(0)
            if ms.shape[0] != B:
                ms = ms.expand(B, N, N)
        outs = []
        for bi in range(B):
            zb = zs[bi] if zs[bi].is_contiguous() else zs[bi].contiguous()
            mb = None if ms is None else ms[bi]
            outs.append(fwd(zb, mb, cfg))
        out = torch.stack(outs, 0)
    out = out[0] if z.dim() == 3 else out
    COUNTS["generic_served"] += 1
    if COUNTS["first_call"] is None:
        COUNTS["first_call"] = {"N": N, "B": B, "batch_mode": BATCH_MODE, "pad": pad, "C": weights["C"], "D": weights["D"], "dir": "out" if outgoing else "in", "residual": bool(residual),
                                "mask": mask is not None, "bias": bool(weights.get("has_bias")), "dtype": str(z.dtype), "device": torch.cuda.get_device_name(z.device)}
        _log("FIRST CALL served: %s %s cfg=%s cells_sha=%s" % (COUNTS["first_call"], CELLS.cell_word(z.device), json.dumps(CELLS.cell_for(z.device)), CELLS.cells_sha()))
    return out


def probe_sizes(cfg, C, D, has_bias, in_f32):
    """The pair sizes the warm probe runs for one shape class: PROBE_N always; plus kdesc.N_MIN_DESC when the class's resolved cells name a host-descriptor
    implementation this stack can run (bf16 input, kdesc.names_desc, kdesc.HAS_TD) — those kernels serve only plane extents >= that gate, so a probe at PROBE_N
    alone would hold the sub-gate kernels to the bar and never the ones serving every larger call."""
    sizes = [PROBE_N]
    k1, k3 = K.resolve_cfg(cfg, int(C), int(D), bool(has_bias))
    if not in_f32 and KD.HAS_TD and any(KD.names_desc(k1, k3)) and KD.N_MIN_DESC > PROBE_N:
        sizes.append(int(KD.N_MIN_DESC))
    return sizes


def _report():
    """One exit line with the generic entry's COUNTS (incl. the per-kernel launch tally) when this process served through it."""
    if COUNTS["generic_calls"]:
        _log("COUNTS " + json.dumps(COUNTS, default=str))


atexit.register(_report)


def probe(device, C, D, has_bias, in_f32):
    """The warm numerics probe of one shape class (C, D, bias, input dtype) on ``device``, ONCE per process: the served kernels on a random [n, n, C] pair tensor for
    every n of :func:`probe_sizes` (outgoing and incoming) against an fp64 statement of the same math (:func:`reference_torch`), beside the bf16-autocast torch statement's own error; the
    class is served when its error is at most SAME_CLASS_RMS x (rms) and SAME_CLASS_MAX x (max-abs) the bf16-autocast statement's, refused BY NAME (``probe-failed``: the lever
    cannot run for this class) otherwise. Returns the
    record ``{"ok", "ratio_max", "ratio_rms", "cell", "line"}`` (also ``COUNTS["probe"]``), or None while the probe cannot run yet (a CUDA graph is being captured on
    this stream: the class is served meanwhile and probed on its next call outside capture)."""
    key = (str(device), int(C), int(D), bool(has_bias), bool(in_f32))
    if key in _PROBES:
        return _PROBES[key]
    if torch.cuda.is_current_stream_capturing():
        if not COUNTS["probe"].get("deferred"):
            COUNTS["probe"]["deferred"] = True; _log("warm probe deferred: a CUDA graph capture is in progress (the class is served; probed on its next call outside capture)")
        return None
    cfg = CELLS.cell_for(device)
    if cfg is None:
        return None
    C, D = int(C), int(D)
    sizes = probe_sizes(cfg, C, D, has_bias, in_f32)
    try:
        facts = probe_stamp_facts(device, C, D, has_bias, in_f32, sizes)
        st = probe_stamp_read(facts)
    except _STAMP_ERRORS + (AssertionError, LookupError):                                  # the stamp machinery can never fail a probe (its own functions already return None)
        facts, st = None, None
    if st is not None:                                                                     # this exact (package, class, sizes, cell, bars, device, stack) PASSED the probe before:
        rec = dict(st.get("record") or {})                                                # the verdict is taken from the stamp, nothing is allocated
        line = "warm probe (C=%d, D=%d, %s, %s in; N=%s) %s: verdict stamped earlier on this stack / device (%s) -> served" % (
            C, D, "bias" if has_bias else "no bias", "fp32" if in_f32 else "bf16", "+".join(str(n) for n in sizes), CELLS.cell_word(device), st.get("_path", "-"))
        _PROBES[key] = {"ok": True, "ratio_max": float(rec.get("ratio_max", 0.0)), "ratio_rms": float(rec.get("ratio_rms", 0.0)), "cell": CELLS.cell_word(device), "line": line, "stamp": "hit"}
        COUNTS["probe"]["%s C%d D%d %s %s" % (key[0], C, D, "bias" if has_bias else "nobias", "f32" if in_f32 else "bf16")] = {"ok": True, "stamp": "hit"}
        _PROBE_MEM["stamp_hits"] += 1
        _log(line)
        return _PROBES[key]
    g = torch.Generator(device="cpu").manual_seed(5489)
    rnd = lambda *shape: torch.randn(*shape, generator=g, dtype=torch.float32)
    raw = dict(ln_in_w=1.0 + 0.1 * rnd(C), ln_in_b=0.1 * rnd(C), w_ag=rnd(D, C) * C ** -0.5, w_ap=rnd(D, C) * C ** -0.5, w_bg=rnd(D, C) * C ** -0.5, w_bp=rnd(D, C) * C ** -0.5,
               ln_out_w=1.0 + 0.1 * rnd(D), ln_out_b=0.1 * rnd(D), w_o=rnd(C, D) * D ** -0.5, w_og=rnd(C, C) * C ** -0.5)
    if has_bias:
        raw.update(b_ag=0.1 * rnd(D), b_ap=0.1 * rnd(D), b_bg=0.1 * rnd(D), b_bp=0.1 * rnd(D), b_o=0.1 * rnd(C), b_og=0.1 * rnd(C))
    raw = {k: v.to(device) for k, v in raw.items()}
    w = pack_weights(**raw)
    worst = {"max": 0.0, "rms": 0.0}
    _PROBE_MEM["runs"] += 1
    try:
        with torch.no_grad():
            for n in sizes:
                z = rnd(n, n, C).to(device=device, dtype=torch.float32 if in_f32 else torch.bfloat16)
                for outgoing in (True, False):
                    served = CELLS.run(lambda c: K.trimul_v4_forward(z, outgoing, None, w, c, eps=1e-5, residual=False, stock_round=_STOCK_ROUND, pad=16), CELLS.cell_for(device) or cfg, device)
                    with torch.autocast("cuda", torch.bfloat16):                         # the bf16-autocast statement = the class yardstick, whole and unchanged (its rounding
                        cls = reference_torch(z, None, direction="outgoing" if outgoing else "incoming", dtype=torch.float32, **raw)   # realization is what the bar is relative to)
                    stt = probe_error_stats(served, cls, z, None, direction="outgoing" if outgoing else "incoming", raw=raw)
                    del cls
                    worst["max"] = max(worst["max"], stt["max_s"] / max(stt["max_c"], 1e-30))
                    worst["rms"] = max(worst["rms"], (stt["ss_s"] / stt["count"]) ** 0.5 / max((stt["ss_c"] / stt["count"]) ** 0.5, 1e-30))
                    del served
                del z
    finally:
        del w, raw
        release_probe_memory()                                                             # nothing the probe allocated is referenced or reserved afterwards
    ok = worst["rms"] <= SAME_CLASS_RMS and worst["max"] <= SAME_CLASS_MAX
    shape = "C=%d, D=%d, %s, %s in" % (C, D, "bias" if has_bias else "no bias", "fp32" if in_f32 else "bf16")
    line = ("warm probe (%s; N=%s out+in) %s: err vs fp64 = %.2fx rms / %.2fx max of the bf16-autocast statement's (same class: rms <= %.2fx, max <= %.1fx) -> %s"
            % (shape, "+".join(str(n) for n in sizes), CELLS.cell_word(device), worst["rms"], worst["max"], SAME_CLASS_RMS, SAME_CLASS_MAX, "served" if ok else "REFUSED: the lever cannot run for this shape class (probe-failed)"))
    _PROBES[key] = {"ok": ok, "ratio_max": worst["max"], "ratio_rms": worst["rms"], "cell": CELLS.cell_word(device), "line": line, "stamp": "-"}
    if ok and facts is not None:                                                           # a PASSED verdict is stamped for later processes on this stack / device (a refusal never is)
        pth = probe_stamp_write(facts, {"ok": True, "ratio_max": round(worst["max"], 6), "ratio_rms": round(worst["rms"], 6), "sizes": sizes, "cell": CELLS.cell_word(device)})
        _PROBES[key]["stamp"] = "written:%s" % pth if pth else "unwritable"
    COUNTS["probe"]["%s C%d D%d %s %s" % (key[0], C, D, "bias" if has_bias else "nobias", "f32" if in_f32 else "bf16")] = {"ok": ok, "ratio_max": round(worst["max"], 3), "ratio_rms": round(worst["rms"], 3), "cell": CELLS.cell_word(device)}
    _log(line)
    return _PROBES[key]



def probe_memory_stats():
    """Bookkeeping of the warm probe's memory scope and verdict stamps: {'blocked_refs', 'max_block_rows', 'released', 'stamp_hits', 'stamp_writes', 'runs'}."""
    return dict(_PROBE_MEM)


def release_probe_memory():
    """gc + hand the caching allocator's unused blocks back to the driver once the probe's transients are dropped."""
    import gc
    gc.collect()
    if torch.cuda.is_available() and torch.cuda.is_initialized():
        torch.cuda.empty_cache()
    _PROBE_MEM["released"] += 1


def probe_block_rows(n, C, D, esize, chunk_bytes=PROBE_CHUNK_BYTES):
    """Rows per output block so that one block's statement transients (~6 [rows, n, max(C, D)] tensors of esize bytes) stay under chunk_bytes."""
    per_row = 6 * int(n) * max(int(C), int(D)) * int(esize)
    return max(1, min(int(n), int(chunk_bytes) // max(1, per_row)))


def reference_torch_blocks(z, mask, *, direction, dtype=torch.float32, block_rows=None, chunk_bytes=PROBE_CHUNK_BYTES, residual=False, eps=1e-5, **w):
    """reference_torch evaluated per OUTPUT BLOCK: yields (rows slice I, cols slice J, o[I, J]) with exactly reference_torch's per-element operations
    (LayerNorm over C per position, the two gated projections, the full-k contraction for every (i, j), LayerNorm over D, the gated out-projection),
    materialising only [rows, n, C|D]-sized operands per block instead of whole [n, n, C|D] tensors.  3-D z ([n, n, C]); mask [n, n] or None."""
    import torch.nn.functional as F
    t = lambda x: None if x is None else x.detach().to(dtype)
    n, C = int(z.shape[0]), int(z.shape[-1]); D = int(w["w_ap"].shape[0])
    rows = int(block_rows) if block_rows else probe_block_rows(n, C, D, torch.empty((), dtype=dtype).element_size(), chunk_bytes)
    _PROBE_MEM["blocked_refs"] += 1; _PROBE_MEM["max_block_rows"] = max(_PROBE_MEM["max_block_rows"], rows)
    out_dir = _DIRS[direction]
    lw, lb, low, lob = t(w["ln_in_w"]), t(w["ln_in_b"]), t(w["ln_out_w"]), t(w["ln_out_b"])
    wag, bag, wap, bap = t(w["w_ag"]), t(w.get("b_ag")), t(w["w_ap"]), t(w.get("b_ap"))
    wbg, bbg, wbp, bbp = t(w["w_bg"]), t(w.get("b_bg")), t(w["w_bp"]), t(w.get("b_bp"))
    wo, bo, wog, bog = t(w["w_o"]), t(w.get("b_o")), t(w["w_og"]), t(w.get("b_og"))

    def ln_in(zpart):
        return F.layer_norm(zpart.detach().to(dtype), (C,), lw, lb, eps)

    def proj(x, wg, bg, wp, bp, mpart):
        v = torch.sigmoid(F.linear(x, wg, bg)) * F.linear(x, wp, bp)
        return v if mpart is None else v * mpart

    def mrows(sl, cols=None):                                    # mask rows (outgoing planes) / mask columns (incoming planes) as [., ., 1]
        if mask is None:
            return None
        m = mask.to(dtype)
        return (m[sl] if cols is None else m[:, cols]).unsqueeze(-1)
    for i0 in range(0, n, rows):
        I = slice(i0, min(n, i0 + rows))
        if out_dir:                                              # outgoing: X[i, j, d] = sum_k a[i, k, d] b[j, k, d] -> rows I of a
            a_I = proj(ln_in(z[I]), wag, bag, wap, bap, mrows(I))
        else:                                                    # incoming: X[i, j, d] = sum_k a[k, i, d] b[k, j, d] -> columns I of a
            a_I = proj(ln_in(z[:, I]), wag, bag, wap, bap, mrows(None, I))
        for j0 in range(0, n, rows):
            J = slice(j0, min(n, j0 + rows))
            if out_dir:
                b_J = proj(ln_in(z[J]), wbg, bbg, wbp, bbp, mrows(J))
                X = torch.einsum("ikd,jkd->ijd", a_I, b_J)
            else:
                b_J = proj(ln_in(z[:, J]), wbg, bbg, wbp, bbp, mrows(None, J))
                X = torch.einsum("kid,kjd->ijd", a_I, b_J)
            del b_J
            zIJ = z[I][:, J]
            x_IJ = ln_in(zIJ)
            o = torch.sigmoid(F.linear(x_IJ, wog, bog)) * F.linear(F.layer_norm(X, (D,), low, lob, eps), wo, bo)
            if residual:
                o = o + zIJ.detach().to(dtype)
            del X, x_IJ
            yield I, J, o
        del a_I


def probe_error_stats(served, cls, z, mask, *, direction, raw, chunk_bytes=PROBE_CHUNK_BYTES, block_rows=None):
    """The probe's error statistics with the fp64 statement streamed over output blocks (never materialised whole): e_s = served - fp64 statement,
    e_c = cls (the bf16-autocast class statement, whole, as before) - fp64 statement; -> {'max_s', 'max_c' (max-abs: exact), 'ss_s', 'ss_c' (sums of
    squares accumulated in fp64 per block), 'count'}."""
    n, C = int(z.shape[0]), int(z.shape[-1]); D = int(raw["w_ap"].shape[0])
    rows = int(block_rows) if block_rows else probe_block_rows(n, C, D, 8, chunk_bytes)
    out = {"max_s": 0.0, "max_c": 0.0, "ss_s": 0.0, "ss_c": 0.0, "count": 0}
    for I, J, r64 in reference_torch_blocks(z, mask, direction=direction, dtype=torch.float64, block_rows=rows, **raw):
        e_s = served[I][:, J].double() - r64
        e_c = cls[I][:, J].double() - r64
        out["max_s"] = max(out["max_s"], float(e_s.abs().max())); out["max_c"] = max(out["max_c"], float(e_c.abs().max()))
        out["ss_s"] += float(e_s.pow(2).sum(dtype=torch.float64)); out["ss_c"] += float(e_c.pow(2).sum(dtype=torch.float64))
        out["count"] += int(e_s.numel())
        del e_s, e_c, r64
    return out


_PKG_DIGEST = {}


def package_digest():
    """sha256 over this package's source and table files (what a stamped probe verdict is keyed by, beside the version): any edit is another key."""
    if "d" not in _PKG_DIGEST:
        import hashlib
        h = hashlib.sha256()
        here = os.path.dirname(os.path.abspath(__file__))
        for fn in sorted(os.listdir(here)):
            if fn.endswith((".py", ".json")):
                with open(os.path.join(here, fn), "rb") as fh:
                    h.update(fn.encode()); h.update(b"\0"); h.update(fh.read()); h.update(b"\0")
        _PKG_DIGEST["d"] = h.hexdigest()
    return _PKG_DIGEST["d"]


def probe_stamp_facts(device, C, D, has_bias, in_f32, sizes):
    """The identity a probe verdict stamp is keyed by (None when the device facts are unavailable: then nothing is read or written and the probe runs):
    {schema, package version + digest, C, D, bias, input dtype, sizes, cell word, bars, device cc + name, driver, CUDA, torch, triton, python}."""
    import collections
    try:
        dev = torch.device(device)
        if dev.type != "cuda" or not torch.cuda.is_available():
            return None
        idx = dev.index if dev.index is not None else torch.cuda.current_device()
        cc = "%d.%d" % tuple(torch.cuda.get_device_capability(idx))
        name = torch.cuda.get_device_name(idx)
        try:
            drv = int(torch.cuda.driver_version()) if hasattr(torch.cuda, "driver_version") else None
        except (RuntimeError, AttributeError, TypeError):            # an older torch without the query
            drv = None
        try:
            import triton
            tv = str(triton.__version__)
        except ImportError:
            tv = None
        return collections.OrderedDict([
            ("schema", PROBE_STAMP_SCHEMA), ("package", "fpf_trimul_v4"), ("version", str(__version__)), ("digest", package_digest()),
            ("C", int(C)), ("D", int(D)), ("bias", bool(has_bias)), ("in_f32", bool(in_f32)), ("sizes", [int(n) for n in sizes]), ("cell", str(CELLS.cell_word(device))),
            ("bars", [SAME_CLASS_RMS, SAME_CLASS_MAX]), ("cc", cc), ("device_name", name), ("driver_version", drv),
            ("cuda_version", str(getattr(torch.version, "cuda", None))), ("torch", str(torch.__version__)), ("triton", tv), ("python", sys.version.split()[0])])
    except _STAMP_ERRORS + (AssertionError,):                           # no device facts / anything the fact gathering trips on: no stamp, the probe runs
        return None


_STAMP_ERRORS = (ImportError, OSError, ValueError, TypeError, AttributeError, KeyError, RuntimeError)   # the stamp machinery never raises into a probe: no stamp = the probe runs


def _stamp_api():
    """kernels.trimul.native (the byte gate's stamp directory chain -- $OPT_CORE_VERDICT_DIR | JIT root | XDG | HOME | tmp -- and file format), imported by
    its ABSOLUTE name: this module also runs as the top-level package ``fpf_trimul_v4`` when a kit routes that bare name to the core copy, where a relative
    import beyond the package would raise.  None when the shared core is not importable (a standalone copy): no stamps."""
    import importlib
    try:
        return importlib.import_module("opt_core.kernels.trimul.native")
    except ImportError:
        return None


def probe_stamp_read(facts):
    """The stamp of a PASSED probe for exactly these facts along the stamp directory chain, or None (never raises)."""
    if facts is None:
        return None
    try:
        NV = _stamp_api()
        if NV is None or not hasattr(NV, "stamp_dirs"):
            return None
        for kind, d in NV.stamp_dirs(leaf=PROBE_STAMP_LEAF):
            st = NV.stamp_read(d, facts)
            if st is not None:
                st["_kind"] = kind
                return st
    except _STAMP_ERRORS:
        return None
    return None


def probe_stamp_write(facts, record):
    """Record a PASSED probe verdict in the first writable stamp directory; returns the path or None (never raises)."""
    if facts is None:
        return None
    try:
        NV = _stamp_api()
        if NV is None or not hasattr(NV, "stamp_dirs"):
            return None
        for _kind, d in NV.stamp_dirs(leaf=PROBE_STAMP_LEAF):
            if os.access(d, os.W_OK | os.X_OK):
                pth = NV.stamp_write(d, facts, {"record": record, "gate_note": "fpf_trimul_v4 warm probe PASSED"})
                if pth:
                    _PROBE_MEM["stamp_writes"] += 1
                return pth
    except _STAMP_ERRORS:
        return None
    return None


def probe_all_classes(device=None, log=print):
    """Run (or find stamped) the warm probe for every shape class this package serves on ``device`` -- what an image bake calls once so processes on the
    image find the verdicts stamped: C x D in SUPPORTED x {no bias, bias} x {bf16, fp32 input}.  -> list of (class, record or None)."""
    dev = torch.device(device if device is not None else "cuda")
    out = []
    for C in K.SUPPORTED_C:
        for D in K.SUPPORTED_D:
            for has_bias in (False, True):
                for in_f32 in (False, True):
                    try:
                        rec = probe(dev, C, D, has_bias, in_f32)
                    except TrimulUnsupported as e:                 # a class this device's cell does not serve: reported, not stamped
                        rec = {"ok": False, "line": "unsupported: %s" % getattr(e, "reason", e)}
                    out.append(((C, D, has_bias, in_f32), rec))
                    if log:
                        log("[fpf_trimul_v4 probe] C=%d D=%d %s %s: %s" % (C, D, "bias" if has_bias else "nobias", "f32" if in_f32 else "bf16",
                            "-" if rec is None else ("%s (stamp %s)" % ("ok" if rec.get("ok") else "REFUSED", rec.get("stamp", "-")))))
    return out


def trimul(z, mask=None, *, direction, weights=None, residual=False, eps=1e-5, pad=16,
           ln_in_w=None, ln_in_b=None, w_ag=None, w_ap=None, w_bg=None, w_bp=None, b_ag=None, b_ap=None, b_bg=None, b_bp=None,
           w_proj=None, b_proj=None, proj_split=("ag", "ap", "bg", "bp"), ln_out_w=None, ln_out_b=None, w_o=None, w_og=None, b_o=None, b_og=None,
           cache_owner=None, cache_key="default", n_min=None):
    """Module-agnostic TriMul. direction in {'outgoing','incoming'} ('out'/'in' accepted). Pass either weights=pack_weights(...) (preferred: pack once) or the raw tensors
    (packed per call unless cache_owner is given, in which case the pack is cached on it under cache_key — use distinct keys for distinct weight sets on one owner)."""
    if direction not in _DIRS:
        raise TrimulUnsupported("direction", repr(direction))
    if weights is None:
        weights = pack_weights(ln_in_w=ln_in_w, ln_in_b=ln_in_b, w_ag=w_ag, w_ap=w_ap, w_bg=w_bg, w_bp=w_bp, b_ag=b_ag, b_ap=b_ap, b_bg=b_bg, b_bp=b_bp,
                               w_proj=w_proj, b_proj=b_proj, proj_split=proj_split, ln_out_w=ln_out_w, ln_out_b=ln_out_b, w_o=w_o, w_og=w_og, b_o=b_o, b_og=b_og,
                               cache_owner=cache_owner, cache_key=cache_key)
    return trimul_packed(z, mask, outgoing=_DIRS[direction], weights=weights, residual=residual, eps=eps, pad=pad, n_min=n_min)


def reference_torch(z, mask, *, direction, ln_in_w, ln_in_b, w_ag, w_ap, w_bg, w_bp, ln_out_w, ln_out_b, w_o, w_og,
                    b_ag=None, b_ap=None, b_bg=None, b_bp=None, b_o=None, b_og=None, residual=False, eps=1e-5, dtype=torch.float32):
    """Pure-torch reference of exactly the math above in `dtype` (float64 for the error reference, float32 for the 'stock fp32 module' class). No autocast inside."""
    import torch.nn.functional as F
    t = lambda x: None if x is None else x.detach().to(dtype)
    zz = z.detach().to(dtype); C = zz.shape[-1]; D = w_ap.shape[0]
    x = F.layer_norm(zz, (C,), t(ln_in_w), t(ln_in_b), eps)
    m = None if mask is None else mask.to(dtype).unsqueeze(-1)
    def proj(wg, bg, wp, bp):
        v = torch.sigmoid(F.linear(x, t(wg), t(bg))) * F.linear(x, t(wp), t(bp))
        return v if m is None else v * m
    a = proj(w_ag, b_ag, w_ap, b_ap); b = proj(w_bg, b_bg, w_bp, b_bp)
    if _DIRS[direction]:
        X = torch.einsum("...ikd,...jkd->...ijd", a, b)
    else:
        X = torch.einsum("...kid,...kjd->...ijd", a, b)
    o = torch.sigmoid(F.linear(x, t(w_og), t(b_og))) * F.linear(F.layer_norm(X, (D,), t(ln_out_w), t(ln_out_b), eps), t(w_o), t(b_o))
    return o + zz if residual else o
