"""ptx_trunk2_levers.py — env-gated, monkeypatch-only trunk levers for Protenix v2 (2.0.0).  All OFF by default.
Import + apply_from_env() BEFORE the model runs.  Every lever keeps the stock module objects/weights; only the execution changes.

PTX_T1_TRANS=fuse1   Transition (eval path, primitives.Transition.forward, not training): stock does
                         x = LN(x); a = linear_a(x); b = linear_b(x); [chunked or not] a = F.silu(a, inplace); a *= b   (bf16 under autocast)
                         out = linear_no_bias(a)
                     eager rounding points: a,b are bf16 GEMM outputs; F.silu on bf16 computes in fp32 internally (x * sigmoid(x)) and rounds
                     to bf16; a*b multiplies two bf16 -> computed in fp32, rounded to bf16.  fuse1 = ONE Triton kernel doing
                         h = bf16( float(bf16(silu_fp32(a))) * float(b) )
                     i.e. identical rounding points; only exp implementation may differ (libdevice expf vs ATen's expf -> both CUDA expf).
                     The GEMMs are ALSO kept untouched (same nn.Linear calls). Expected: bitwise vs stock.
PTX_TRANSITION=exact|fast|big  (levers transition_core_exact | transition_core; src/ptx_transition_core.py) every eval Transition call and the block
                     statement's in-block pair transition through opt_core.kernels.transition by the mode's TIER WORD; a call the provider hands back
                     (stock cell, refusal by name, no cell word for the shape) runs the forward found at install (T1's when PTX_T1_TRANS is set, else stock).
PTX_T2_NOCOPY=fused  triangular.layers.Attention._prep_qkv for the cuEq triangle-attention path (apply_scale=False):
                     stock: q=linear_q(x) k=linear_k(x) v=linear_v(x) -> view heads -> transpose(-2,-3) (strided) -> cuEq .contiguous() x3
                     fused: y = x @ cat(Wq,Wk,Wv)^T  (ONE GEMM, N = 3*H*D) -> ONE copy into [.., 3H, Q, D] contiguous -> q,k,v contiguous views.
                     GEMM shape differs from stock => accumulation order may differ => Tier 2 a priori.
PTX_T2_NOCOPY=samegemm  keep the 3 stock GEMMs (bitwise operands) but do the head-major copy ourselves with one fused
                     torch.stack-free strided copy per tensor (this is what cuEq does anyway -> pure control, expected bitwise & same speed).
"""
from opt_core.oom import is_oom   # a broad handler that reroutes around a lever re-raises device out-of-memory first (opt_core.oom.is_oom)
import os, sys, ast, math
import torch
import torch.nn.functional as F
import ptx_transition_core as _TRC       # every trunk Transition call through opt_core.kernels.transition by the mode's tier word (levers transition_core_exact | transition_core)
try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice as _ld
    except Exception:
        try:
            from triton.language.extra.cuda import libdevice as _ld
        except Exception:
            _ld = None
    _HAS_TRITON = True
except Exception:                       # pragma: no cover
    triton = None; tl = None; _ld = None; _HAS_TRITON = False

if _HAS_TRITON:
    _HAS_LD = tl.constexpr(_ld is not None)

    @triton.jit
    def _silu_mul_kernel(A, B, OUT, n_elem, BLOCK: tl.constexpr, ROUND_SILU: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n_elem
        a = tl.load(A + offs, mask=m, other=0.0)
        b = tl.load(B + offs, mask=m, other=0.0)
        af = a.to(tl.float32)
        # ATen silu (CUDA, reduced-precision input): opmath fp32:  x / (1 + exp(-x))   [ActivationSiluKernel.cu]
        if _HAS_LD:
            e = _ld.exp(-af)
        else:
            e = tl.exp(-af)
        sv = af / (1.0 + e)
        if ROUND_SILU:
            sv = sv.to(a.dtype).to(tl.float32)        # stock: F.silu(a, True) writes bf16, then b *= a reads that bf16
        h = b.to(tl.float32) * sv                     # ATen mul on bf16: fp32 opmath multiply, one rounding
        tl.store(OUT + offs, h.to(a.dtype), mask=m)

    @triton.jit
    def _silu_mul_2d_kernel(A, B, OUT, NC, sa, sb, so, BLOCK_N: tl.constexpr):
        row = tl.program_id(0).to(tl.int64); cb = tl.program_id(1)
        cols = cb * BLOCK_N + tl.arange(0, BLOCK_N)
        m = cols < NC
        a = tl.load(A + row * sa + cols, mask=m, other=0.0)
        b = tl.load(B + row * sb + cols, mask=m, other=0.0)
        af = a.to(tl.float32)
        if _HAS_LD:
            e = _ld.exp(-af)
        else:
            e = tl.exp(-af)
        sv = af / (1.0 + e)
        sv = sv.to(a.dtype).to(tl.float32)
        h = b.to(tl.float32) * sv
        tl.store(OUT + row * so + cols, h.to(a.dtype), mask=m)

    @triton.jit
    def _gate_out_kernel(O, G, OUT, J, H, D: tl.constexpr, so_i, so_h, so_j, sg_i, sg_j, BLOCK_J: tl.constexpr, HD_P2: tl.constexpr):
        # O: [I, H, J, D] contiguous-last (strides so_i, so_h, so_j, 1) = cuEq output; G: [I, J, H*D] (strides sg_i, sg_j, 1) = linear_g output (pre-sigmoid)
        # OUT[i, j, h*D + d] = bf16( fp32(O[i,h,j,d]) * fp32( bf16(sigmoid(fp32(G[i,j,hD+d]))) ) )   -- stock: g = sigmoid(linear_g(x)) (bf16), o = o * g
        i = tl.program_id(0).to(tl.int64); jb = tl.program_id(1)
        js = jb * BLOCK_J + tl.arange(0, BLOCK_J).to(tl.int64)          # [BJ]
        c = tl.arange(0, HD_P2)                                          # [HD]
        h = c // D; d = c - h * D
        jm = js < J; cm = c < H * D
        m2 = jm[:, None] & cm[None, :]
        o_off = i * so_i + h[None, :].to(tl.int64) * so_h + js[:, None] * so_j + d[None, :]
        g_off = i * sg_i + js[:, None] * sg_j + c[None, :]
        o = tl.load(O + o_off, mask=m2, other=0.0)
        g = tl.load(G + g_off, mask=m2, other=0.0)
        gf = g.to(tl.float32)
        if _HAS_LD:
            e = _ld.exp(-gf)
        else:
            e = tl.exp(-gf)
        sg = (1.0 / (1.0 + e)).to(g.dtype).to(tl.float32)               # ATen sigmoid (opmath fp32: one / (one + exp(-x))) rounded to bf16
        r = o.to(tl.float32) * sg
        out_off = (i * J + js[:, None]) * (H * D) + c[None, :]
        tl.store(OUT + out_off, r.to(o.dtype), mask=m2)

    @triton.jit
    def _gate_out_kernel2(O, G, OUT, J, so_i, so_h, so_j, sg_i, sg_j, H: tl.constexpr, D: tl.constexpr, BLOCK_J: tl.constexpr):
        # v2 (same math as _gate_out_kernel, coalesced): per program (i, j-block): for each head h load the CONTIGUOUS [BLOCK_J, D] slab O[i,h,j0:j1,:]
        # (so_j == D -> BLOCK_J*D*2 contiguous bytes), the [BLOCK_J, D] gate tile (rows at pitch sg_j), write OUT[i, j, h*D:(h+1)*D].
        i = tl.program_id(0).to(tl.int64); jb = tl.program_id(1)
        js = jb * BLOCK_J + tl.arange(0, BLOCK_J).to(tl.int64)          # [BJ]
        d = tl.arange(0, D)                                              # [D]
        jm = js < J
        for h in tl.static_range(H):
            o = tl.load(O + i * so_i + h * so_h + js[:, None] * so_j + d[None, :], mask=jm[:, None], other=0.0)
            g = tl.load(G + i * sg_i + js[:, None] * sg_j + h * D + d[None, :], mask=jm[:, None], other=0.0)
            gf = g.to(tl.float32)
            if _HAS_LD:
                e = _ld.exp(-gf)
            else:
                e = tl.exp(-gf)
            sg = (1.0 / (1.0 + e)).to(g.dtype).to(tl.float32)
            r = o.to(tl.float32) * sg
            tl.store(OUT + (i * J + js[:, None]) * (H * D) + h * D + d[None, :], r.to(o.dtype), mask=jm[:, None])

    @triton.jit
    def _bgemm_kernel(A, B, C, M, N, K, s_ab, s_am, s_ak, s_bb, s_bk, s_bn, s_cb, s_cm, s_cn,
                      LAYOUT: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, GROUP: tl.constexpr):
        # C[b, m, n] = sum_k A[b, m, k] * B[b, k, n]  (A, B given by strides, so transposed views are free); fp32 accumulate, K ascending
        pid = tl.program_id(0); bid = tl.program_id(1).to(tl.int64)
        num_m = tl.cdiv(M, BM); num_n = tl.cdiv(N, BN)
        num_in_group = GROUP * num_n
        group_id = pid // num_in_group
        first_m = group_id * GROUP
        gsz = min(num_m - first_m, GROUP)
        pid_m = first_m + ((pid % num_in_group) % gsz)
        pid_n = (pid % num_in_group) // gsz
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        rk = tl.arange(0, BK)
        A += bid * s_ab; B += bid * s_bb; C += bid * s_cb
        a_ptrs = A + (rm[:, None] * s_am + rk[None, :] * s_ak)
        b_ptrs = B + (rk[:, None] * s_bk + rn[None, :] * s_bn)
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        mm = rm < M; nn = rn < N
        for k0 in range(0, K, BK):
            km = (k0 + rk) < K
            a = tl.load(a_ptrs, mask=mm[:, None] & km[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=km[:, None] & nn[None, :], other=0.0)
            acc = tl.dot(a, b, acc)
            a_ptrs += BK * s_ak; b_ptrs += BK * s_bk
        c_ptrs = C + rm[:, None] * s_cm + rn[None, :] * s_cn
        tl.store(c_ptrs, acc.to(C.dtype.element_ty), mask=mm[:, None] & nn[None, :])

    @triton.jit
    def _swap_mid_copy(X, Y, I, J, C, sxb, sxi, sxj, syb, syi, syj, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
        # X logical [B, I, J, C] (C contiguous); Y logical [B, J, I, C] contiguous; strides passed per (b, i, j) for both.
        pid = tl.program_id(0); b = tl.program_id(1)
        r = pid.to(tl.int64) * BLOCK_R + tl.arange(0, BLOCK_R).to(tl.int64)      # linear index over output rows (j-major, then i)
        n_rows = I.to(tl.int64) * J
        rm = r < n_rows
        j = r // I
        i = r - j * I
        xoff = b.to(tl.int64) * sxb + i * sxi + j * sxj
        yoff = b.to(tl.int64) * syb + i * syi + j * syj
        for c0 in range(0, C, BLOCK_C):
            cc = c0 + tl.arange(0, BLOCK_C)
            cm = cc < C
            m2 = rm[:, None] & cm[None, :]
            v = tl.load(X + xoff[:, None] + cc[None, :], mask=m2)
            tl.store(Y + yoff[:, None] + cc[None, :], v, mask=m2)

_STATS = {"t1_calls": 0, "t1_fallback": 0,
          "t2_calls": 0, "t2_mode": None, "t1_mode": None, "applied": []}

# =====================================================================================  portability: explicit GPU arch classes; checked-cell tables are per class
# Only sm90 (H100/H200/H800) has checked Triton cells today. Every other class has EMPTY checked tables => the block path runs the exact STOCK sub-path for that op
# (never raises inside a prediction) and prints one line per (op, arch) the first time.  Classes: sm80 (A100), sm89 (L40S/RTX 6000 Ada), sm90, sm100 (B200), sm120 (RTX PRO 6000
# Blackwell), other.
_ARCH = {}
def gpu_arch(device=None) -> str:
    idx = torch.device(device).index if device is not None and torch.device(device).index is not None else torch.cuda.current_device()
    a = _ARCH.get(idx)
    if a is None:
        mj, mn = torch.cuda.get_device_capability(idx)
        a = {(8, 0): "sm80", (8, 6): "sm86", (8, 9): "sm89", (9, 0): "sm90", (10, 0): "sm100", (12, 0): "sm120"}.get((mj, mn), f"sm{mj}{mn}")
        _ARCH[idx] = a
    return a
_ARCH_VERIFIED_BLK2 = {"sm90"}          # arch classes with checked (256,256) prologue/epilogue + (256,1024) transition cells
# Per-arch cell tables are DATA, not code: $FPF_HOME/CELLS.json (optional) = {"blk2_arch": ["sm90", "sm100"], "t1_fused_min_smem_kb": 160,
#   ...} (prologue / epilogue cells: opt_core.attn.pair_fused rows; the transitions: opt_core.kernels.transition)  -> another arch drops checked cells in without code changes.
# CELLS.json "blk2_triatt_min_tokens": {gpu_arch: N_floor}: below N_floor tokens the block statement's FUSED tri-attention statement (prologue -> attention core ->
# epilogue: this unit's (256, 256) / (256, 8, 32) cells, the GLUE_V2 / MK-PF kernels bound over them, the K2B core under ARM=T) STEPS ASIDE BY NAME to the stock
# tri-attention statement on that arch (the statement every arch without prologue / epilogue cells runs): on sm_80 the fused statement is byte-identical to the stock
# one end to end from the floor up under the deterministic recipe and is not known to be bitwise below it on
# that arch, so the stock statement runs there by name. The same kind of data row as small_m_stock_rows (the in-block transition keeps that row: both floors hold independently): the '[FPF] CELLS.json loaded'
# line prints the table, the first below-floor statement of the arch prints one '[FPF] BLK: fused tri-attention statement -> stock ... below <N_floor> tokens' line,
# _STATS carries blk2_triatt_min_tokens=<N_floor> (this process's arch) and the tally blk_triatt_below_tokens=<statements>. An arch without a row is byte-unchanged.
_TRIATT_MIN_TOKENS = {}
try:
    _cells_path = os.environ.get("FPF_CELLS_JSON") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "CELLS.json")
    if os.path.exists(_cells_path):
        import json as _json
        _CELLS = _json.load(open(_cells_path))
        # OPTIONAL per-triton overlay (the best cells can differ per Triton release). Format: "by_triton": {"3.6": {"prologue": {...}, "epilogue": {...},
        # "transition": {...}, "t1_fused": {...}, "blk2_arch": [...]}} — the overlay for the running triton MAJOR.MINOR is merged over the base tables (base = the pinned stack's Triton).
        try:
            import triton as _tr
            _tv = ".".join(_tr.__version__.split(".")[:2])
            _ov = (_CELLS.get("by_triton") or {}).get(_tv)
            if _ov:
                if "blk2_arch" in _ov: _CELLS["blk2_arch"] = list(_ov["blk2_arch"])
                print(f"[FPF] CELLS.json: triton {_tv} overlay applied ({sorted(k for k in _ov)})", file=sys.stderr, flush=True)
        except Exception as _e:
            print(f"[FPF] CELLS.json: triton overlay skipped: {_e!r}", file=sys.stderr, flush=True)
        _ARCH_VERIFIED_BLK2 |= set(_CELLS.get("blk2_arch", []))
        _TRIATT_MIN_TOKENS.update({str(_a): int(_v) for _a, _v in (_CELLS.get("blk2_triatt_min_tokens") or {}).items() if int(_v) > 0})
        # the v3 prologue / v2 epilogue cells of this unit's tri-attention shape (c_z 256, 8 heads x 32) on every configured card are the shared core's
        # pair_fused cell rows (opt_core.attn.pair_fused: impl fpf, variant v3 / v2, key [256, 8, 32]; an exact (cc, triton) row before the (cc, '*') row = the former
        # by_triton overlay) — read through the provider's own lookup and merged into the carried kernels' PINNED_CONFIG;
        # only TABLE rows count (the provider's safe / default settings for a card
        # without a row are not a cell of this statement: such a card keeps the kernels' built-in cells, as before).
        try:
            from opt_core.attn import pair_fused as _PFC
            try:
                import triton as _trc; _tvc = ".".join(_trc.__version__.split(".")[:2])
            except Exception:
                _tvc = "*"
            _ARCHW = {"8.0": "sm80", "9.0": "sm90", "10.0": "sm100", "10.3": "sm103", "12.0": "sm120"}
            for _sec, _var, _kf in (("prologue", "v3", lambda a: (256, 256, a)), ("epilogue", "v2", lambda a: (256, 8, 32, a))):
                _rows = {}
                for _ccw, _aw in _ARCHW.items():
                    _dc = _PFC.lookup_cell("fpf", _sec, (256, 8, 32), variant=_var, stack=(_ccw, _tvc))
                    if _dc.row is not None and str(_dc.served_by).startswith(_ccw + "|") and isinstance(_dc.row.get("cfg"), dict):
                        _rows[str(_kf(_aw))] = dict(_dc.row["cfg"])
                if _rows:
                    _CELLS[_sec] = {**_rows, **(_CELLS.get(_sec) or {})}          # a caller's FPF_CELLS_JSON section still wins per key (maintainer override)
            print(f"[FPF] CELLS: prologue/epilogue cells from opt_core.attn.pair_fused rows (triton {_tvc}): prologue={sorted(_CELLS.get('prologue', {}))} epilogue={sorted(_CELLS.get('epilogue', {}))}", file=sys.stderr, flush=True)
        except Exception as _e:
            print(f"[FPF] CELLS: provider prologue/epilogue rows not read ({_e!r}); the carried kernels keep their built-in cells", file=sys.stderr, flush=True)
        for _mod, _attr in (("fpf_triatt_pro.prologue", "PINNED_CONFIG"), ("fpf_triatt_epi.epilogue", "PINNED_CONFIG")):   # the pair transition is the provider's (opt_core.kernels.transition by tier word: ptx_transition_core); no kit cell pins it
            _tab = _CELLS.get(_mod.split(".")[-1].replace("prologue", "prologue").split("_")[-1], None) or _CELLS.get(_mod, None)
            if _tab:
                try:
                    import importlib as _il
                    _m = _il.import_module(_mod); _T = getattr(_m, _attr)
                    for _k, _cfg in _tab.items():
                        _key = tuple(ast.literal_eval(_k)) if isinstance(_k, str) else tuple(_k)
                        _T[_key] = _cfg
                except Exception as _e:
                    print(f"[FPF] CELLS.json: could not merge cells for {_mod}: {_e!r}", file=sys.stderr, flush=True)
        print(f"[FPF] CELLS.json loaded: blk2_arch={sorted(_ARCH_VERIFIED_BLK2)}" + (f" blk2_triatt_min_tokens={dict(sorted(_TRIATT_MIN_TOKENS.items()))}" if _TRIATT_MIN_TOKENS else ""), file=sys.stderr, flush=True)
except Exception as _e:
    print(f"[FPF] CELLS.json ignored: {_e!r}", file=sys.stderr, flush=True)
_ONCE = set()
def _say_once(key, line):
    if key not in _ONCE:
        _ONCE.add(key); print(line, flush=True)
        _STATS.setdefault("portability_lines", []).append(line)
def _triatt_below_floor(z):
    """CELLS.json blk2_triatt_min_tokens: True when this block statement's N_token is below the arch's floor -> the caller runs the STOCK tri-attention
    statement for this node (counted as blk_triatt_below_tokens, one per statement; one '[FPF] ...' line per arch the first time). False (and nothing
    written) on an arch without a row."""
    if not _TRIATT_MIN_TOKENS:
        return False
    arch = gpu_arch(z.device)
    floor = _TRIATT_MIN_TOKENS.get(arch, 0)
    if not floor:
        return False
    _STATS["blk2_triatt_min_tokens"] = floor                  # the floor of THIS process's arch
    _STATS.setdefault("blk_triatt_below_tokens", 0)
    n = int(z.shape[-2])
    if n >= floor:
        return False
    _STATS["blk_triatt_below_tokens"] += 1
    if ("triatt_floor", arch) not in _ONCE:                   # one line per arch; kept under its own report key (portability_lines stays the arch-level list)
        _ONCE.add(("triatt_floor", arch))
        line = (f"[FPF] BLK: fused tri-attention statement -> stock tri-attention statement below {floor} tokens on gpu_arch={arch} (first such call N_token={n}; "
                f"the fused statement's bitwise measurement on this arch starts at the floor: CELLS.json blk2_triatt_min_tokens; exact by construction, counted as blk_triatt_below_tokens)")
        print(line, flush=True); _STATS.setdefault("triatt_floor_lines", []).append(line)
    return True

def report():
    try:                                                    # lever ln_core: the LayerNorm provider binding's live counters + one summary line (src/protenix_ptx_ln_core.py)
        import sys as _s
        _lnc = _s.modules.get("protenix_ptx_ln_core")
        if _lnc is not None and _lnc.report()["installed"]:
            _STATS["ln_core"] = _lnc.report(); print(_lnc.summary_line(), file=sys.stderr, flush=True)
    except Exception as _e:
        _STATS["ln_core_err"] = repr(_e)
    try:                                                    # XL ledger (what each lever holds / saves in memory at the largest N_token seen) — one stderr line + report()["xl"]
        _STATS["xl"] = _xl_ledger()
        print("[FPF] XLMEM " + " ".join(f"{k}={v}" for k, v in _STATS["xl"].items()), file=sys.stderr, flush=True)
    except Exception as _e:
        _STATS["xl_err"] = repr(_e)
    try:
        import sys as _s
        m = _s.modules.get("ptx_trimul_routes")
        if m is not None: _STATS["trimul_routes"] = m.report()   # the pair-stack TriMul lever's census (opt_core.kernels.trimul by tier word)
        e = _s.modules.get("fpf_triatt_epi.epilogue")
        if e is not None: _STATS["fpf_epi_stats"] = dict(getattr(e, "_STATS", {}))
        p = _s.modules.get("fpf_triatt_pro.prologue")
        if p is not None: _STATS["fpf_pro_stats"] = dict(getattr(p, "STATS", getattr(p, "_STATS", {})) or {})
    except Exception as _e:
        _STATS["report_err"] = repr(_e)
    try:                                                    # ROUTED vs SERVED attention counters side by side (a call routed to the Tier-2 branch can still be served by the library op below the package's own gate)
        _SN = sys.modules.get("fpf_smalln")                  # only if the arm actually loaded it (ARM T); never import at exit
        if _SN is None: raise LookupError("fpf_smalln not loaded")
        _c = getattr(_SN, "COUNTS", {}) or {}
        _STATS["blk_att_served"] = {"k2b": _c.get("att_k2b_calls"), "cueq": _c.get("att_cueq_calls"), "routed_to_tier2_branch": _STATS.get("blk_att_k2b_routed", 0)}
        print(f"[FPF] attention core: routed_to_tier2_branch={_STATS.get('blk_att_k2b_routed', 0)} served k2b={_c.get('att_k2b_calls')} cueq={_c.get('att_cueq_calls')} (routed = entered the Tier-2 branch of the block core; served = kernel actually executed after fpf_smalln's size/cell gate)", file=sys.stderr, flush=True)
    except Exception:
        if _STATS.get("blk_att_k2b_routed"): _STATS["blk_att_served"] = {"k2b": _STATS.get("blk_att_k2b_routed"), "cueq": None, "routed_to_tier2_branch": _STATS.get("blk_att_k2b_routed"), "note": "fpf_smalln absent: routed == served"}
    try:                                                    # dit_attn_exact: the unit's live record (calls / routes / installed / loadcheck) into the trunk record
        if _DXA_REPORT is not None: _STATS["dit_attn_exact"] = _DXA_REPORT()
    except Exception as _e:
        _STATS["dit_attn_exact"] = {"report_error": repr(_e)}
    try:                                                    # triatt_exact: the binding's live record (word / sites / calls / provider / refused / member counts) into the trunk record
        if _TXA_REPORT is not None: _STATS["triatt_exact"] = _TXA_REPORT()
    except Exception as _e:  # noqa: BLE001
        _STATS["triatt_exact"] = {"report_error": repr(_e)}
    try:                                                    # transition_core_exact | transition_core: the provider binding's live record (word / calls / served rows / module reasons)
        _STATS["transition_core"] = _TRC.report()
    except Exception as _e:  # noqa: BLE001
        _STATS["transition_core"] = {"report_error": repr(_e)}
    return dict(_STATS)


# =====================================================================================  kernel-selection guard (cat-GEMM exactness is stack dependent)
# T1-dual and T2-fused replace k separate GEMMs x@W_i^T by ONE GEMM x@cat(W_i)^T and slice.  Each output element is the same dot product, but
# whether cuBLAS evaluates it with the same kernel / same K-split (hence bitwise-identical fp32 accumulation) for N=sum(N_i) as for N=N_i is a
# property of the cuBLAS heuristics on THIS GPU / library version.  Guard: the first time a (module-class, M, K, N_parts, dtype) shape is seen,
# compute both ways and compare with torch.equal; if not identical, that shape is pinned to the stock k-GEMM path for the rest of the process
# (counted in the report).  PTX_CATGEMM_GUARD=0 disables the check (not recommended); the check costs one extra set of GEMMs per new shape.
_CAT_OK = {}
def catgemm_exact(key, fused_parts, stock_fn):
    """key: hashable shape key; fused_parts: tuple of tensors (slices of the fused GEMM output); stock_fn(): tuple of stock GEMM outputs.
    Returns True if the fused result may be used (bitwise-equal for this key), False -> caller must use the stock outputs (returned via _CAT_LAST)."""
    ok = _CAT_OK.get(key)
    if ok is None:
        if os.environ.get("PTX_CATGEMM_GUARD", "1") == "0":
            ok = True
        else:
            ref = stock_fn()
            ok = all(torch.equal(f, r) for f, r in zip(fused_parts, ref))
            _CAT_LAST[key] = ref
            _STATS.setdefault("catgemm_guard", {})[str(key)] = bool(ok)
        _CAT_OK[key] = ok
    return ok
_CAT_LAST = {}
# =====================================================================================  T1: fused silu*mul (Triton), stock rounding points
_T1_KERNEL = None
def _build_t1_kernel():
    global _T1_KERNEL
    assert _HAS_TRITON, "triton not available"
    _T1_KERNEL = _silu_mul_kernel

def silu_mul_(a, b):
    """in-place on a: a = bf16(silu(a)) * b with stock rounding; returns a."""
    assert a.is_contiguous() and b.is_contiguous() and a.shape == b.shape and a.dtype == b.dtype
    n = a.numel()
    BLOCK = 2048
    grid = ((n + BLOCK - 1) // BLOCK,)
    _T1_KERNEL[grid](a, b, a, n, BLOCK=BLOCK, ROUND_SILU=True, num_warps=8)
    return a

def _apply_t1(mode):
    import protenix.model.modules.primitives as PR
    _build_t1_kernel()
    _orig_forward = PR.Transition.forward

    def forward(self, x):
        if self.training:
            return _orig_forward(self, x)
        # ---- replicate the stock eval path verbatim except the silu/mul pair ----
        if not (x.is_cuda and x.dtype in (torch.bfloat16, torch.float16, torch.float32)):
            _STATS["t1_fallback"] += 1
            return _orig_forward(self, x)
        _STATS["t1_calls"] += 1
        return _t1_forward_body(self, x)
    PR.Transition.forward = forward
    _STATS["t1_mode"] = mode
    _STATS["applied"].append(f"T1:{mode}")

def _t1_forward_body(self, x):
    """Copy of primitives.Transition.forward eval branch (v2.0.0) with F.silu(a, inplace=True); a *= b  ->  silu_mul_(a, b).
    Kept literally identical otherwise (chunking threshold, preallocated out, reshape) so GEMM calls/shapes match stock."""
    other_dims = x.shape[:-1]
    dim_size = x.shape[-1]
    size = x.shape[-2]
    x = x.reshape(-1, dim_size)
    chunk_num = 1 if size < 3200 else 8
    chunks = torch.chunk(x, chunk_num, dim=-2)
    outputs = torch.empty((x.shape[0], self.c_in), dtype=x.dtype, device=x.device)
    start = 0
    mode = _STATS.get("t1_mode")
    for chunk in chunks:
        y = self.layernorm1(chunk)
        if mode == "fused" and _HAS_TRITON:
            # T1 by name: the stock two-GEMM body with the fused silu*mul below (the one-kernel transition is the shared core's row, served through
            # opt_core.kernels.transition by the mode's tier word — levers transition_core_exact | transition_core; this forward answers the calls it hands back)
            # not eligible (fp32 path / other widths): stock two-GEMM body below with the fused silu*mul
            a = self.linear_no_bias_a(y); b = self.linear_no_bias_b(y); del y
            if a.dtype not in (torch.bfloat16, torch.float16) or not a.is_contiguous() or not b.is_contiguous():
                a = F.silu(a, True); b *= a; h = b
            else:
                h = silu_mul_(a, b)
            del b
            h = self.linear_no_bias(h)
            outputs[start: start + h.shape[0]] = h
            start += h.shape[0]
            del a, h
            continue
        if mode == "dual":
            # ONE GEMM y @ cat(Wa, Wb)^T (N = 2*n*c) instead of two; a, b are column views of the [M, 2nc] result -> silu_mul_strided
            W = _t1_cat_weight(self)
            ab = F.linear(y, W)                     # [M, 2*nc] (autocast bf16)
            nc = W.shape[0] // 2
            a = ab[:, :nc]; b = ab[:, nc:]
            key = ("T1", y.shape[0], y.shape[1], nc, str(ab.dtype))
            if not catgemm_exact(key, (a, b), lambda: (self.linear_no_bias_a(y), self.linear_no_bias_b(y))):
                _STATS["t1_dual_fallback"] = _STATS.get("t1_dual_fallback", 0) + 1
                a = self.linear_no_bias_a(y) if key not in _CAT_LAST else _CAT_LAST[key][0]
                b = self.linear_no_bias_b(y) if key not in _CAT_LAST else _CAT_LAST[key][1]
                _CAT_LAST.pop(key, None)
                del y, ab
                if a.dtype not in (torch.bfloat16, torch.float16):
                    a = F.silu(a, True); b *= a; h = b
                else:
                    h = silu_mul_(a, b)
            else:
                _CAT_LAST.pop(key, None)
                del y
                if ab.dtype in (torch.bfloat16, torch.float16):
                    h = torch.empty((ab.shape[0], nc), dtype=ab.dtype, device=ab.device)
                    silu_mul_strided(a, b, h)
                else:
                    h = F.silu(a) * b
                del ab
            del a, b
        else:
            a = self.linear_no_bias_a(y)
            b = self.linear_no_bias_b(y)          # stock computes silu(a) between the two GEMMs; GEMM results do not depend on that order
            del y
            if a.dtype not in (torch.bfloat16, torch.float16) or not a.is_contiguous() or not b.is_contiguous():
                a = F.silu(a, True); b *= a          # stock path verbatim (fp32: keep ATen)
                h = b
            else:
                h = silu_mul_(a, b)                  # a <- bf16(silu(a)) * b  (== stock's b *= silu(a): fp32 multiply is commutative)
            del a, b
        h = self.linear_no_bias(h)
        outputs[start: start + h.shape[0]] = h
        start += h.shape[0]
        del h
    outputs = outputs.reshape(*other_dims, self.c_in)
    return outputs

_T1_WCACHE = {}
def _t1_cat_weight(self):
    key = id(self)
    W = _T1_WCACHE.get(key)
    wa = self.linear_no_bias_a.weight
    if W is None or W.device != wa.device or W.dtype != wa.dtype:
        W = torch.cat([wa, self.linear_no_bias_b.weight], 0).detach().contiguous()
        _T1_WCACHE[key] = W
    return W

def silu_mul_strided(a, b, out):
    """out[m, j] = bf16(silu(a[m, j])) * b[m, j] for 2-D row-strided a, b (column slices of one [M, 2nc] tensor), out contiguous [M, nc]."""
    M, NC = out.shape
    assert a.stride(1) == 1 and b.stride(1) == 1 and out.is_contiguous()
    BLOCK_N = triton_next_pow2(NC) if NC <= 2048 else 1024
    grid = (M, (NC + BLOCK_N - 1) // BLOCK_N)
    _silu_mul_2d_kernel[grid](a, b, out, NC, a.stride(0), b.stride(0), out.stride(0), BLOCK_N=BLOCK_N, num_warps=4)
    return out

# =====================================================================================  T2: q/k/v projections without cuEq's 3 hidden copies
def _apply_t2(mode):
    import protenix.model.triangular.layers as TL
    _orig_prep = TL.Attention._prep_qkv
    _cache = {}

    def _prep_qkv(self, q_x, kv_x, apply_scale=True):
        if apply_scale or (q_x is not kv_x) or mode == "view":
            if mode == "view":
                _STATS["t2_calls"] += 1
            return _orig_prep(self, q_x, kv_x, apply_scale=apply_scale)
        H, D = self.no_heads, self.c_hidden
        _STATS["t2_calls"] += 1
        if mode == "samegemm":
            q = self.linear_q(q_x); k = self.linear_k(kv_x); v = self.linear_v(kv_x)          # identical GEMMs to stock
            q = q.view(q.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
            k = k.view(k.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
            v = v.view(v.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
            return q, k, v
        if mode in ("fused", "fusedg"):
            gmode = (mode == "fusedg" and self.linear_g is not None)
            key = id(self)
            W = _cache.get(key)
            if W is None or W.device != q_x.device:
                ws = [self.linear_q.weight, self.linear_k.weight, self.linear_v.weight] + ([self.linear_g.weight] if gmode else [])
                W = torch.cat(ws, 0).detach()   # [3*H*D (+H*D for fusedg), C]
                _cache[key] = W
            y = F.linear(q_x, W)                                       # autocast -> bf16 GEMM, N = 3*H*D (fused) or 4*H*D (fusedg: + linear_g)
            HD = H * D
            if gmode:
                key = ("T2g", tuple(q_x.shape), HD, str(y.dtype))
                if not catgemm_exact(key, (y[..., :HD], y[..., HD:2 * HD], y[..., 2 * HD:3 * HD], y[..., 3 * HD:]),
                                     lambda: (self.linear_q(q_x), self.linear_k(kv_x), self.linear_v(kv_x), self.linear_g(q_x))):
                    _STATS["t2_fused_fallback"] = _STATS.get("t2_fused_fallback", 0) + 1
                    _CAT_LAST.pop(key, None); del y
                    q = self.linear_q(q_x); k = self.linear_k(kv_x); v = self.linear_v(kv_x)
                    q = q.view(q.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
                    k = k.view(k.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
                    v = v.view(v.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
                    return q, k, v
                _CAT_LAST.pop(key, None)
                self._t2g_gate = y[..., 3 * HD:]                            # [*, Q, HD] strided view (row pitch 4*HD) -> consumed by OG (_wrap_up) instead of linear_g(q_x)
                y3 = y[..., :3 * HD].unflatten(-1, (3, H, D))               # view [*, Q, 3, H, D] (row pitch 4HD)
                nd = y3.dim()
                perm = [nd - 3] + list(range(nd - 4)) + [nd - 2, nd - 4, nd - 1]
                y3 = y3.permute(*perm).contiguous()                         # ONE strided copy -> [3, *, H, Q, D]
                return y3[0], y3[1], y3[2]
            key = ("T2", tuple(q_x.shape), HD, str(y.dtype))
            if not catgemm_exact(key, (y[..., :HD], y[..., HD:2 * HD], y[..., 2 * HD:]),
                                 lambda: (self.linear_q(q_x), self.linear_k(kv_x), self.linear_v(kv_x))):
                _STATS["t2_fused_fallback"] = _STATS.get("t2_fused_fallback", 0) + 1
                _CAT_LAST.pop(key, None); del y
                q = self.linear_q(q_x); k = self.linear_k(kv_x); v = self.linear_v(kv_x)      # stock GEMMs + explicit head-major copies (== samegemm, bitwise)
                q = q.view(q.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
                k = k.view(k.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
                v = v.view(v.shape[:-1] + (H, -1)).transpose(-2, -3).contiguous()
                return q, k, v
            _CAT_LAST.pop(key, None)
            # [*, Q, 3, H, D] -> [3, *, H, Q, D] contiguous in ONE copy (3 outermost so q, k, v are each contiguous)
            y = y.view(y.shape[:-1] + (3, H, D))
            nd = y.dim()                                               # dims: lead..., Q, 3, H, D  (indices nd-4 = Q, nd-3 = 3, nd-2 = H, nd-1 = D)
            perm = [nd - 3] + list(range(nd - 4)) + [nd - 2, nd - 4, nd - 1]
            y = y.permute(*perm).contiguous()
            q, k, v = y[0], y[1], y[2]                                 # each [*, H, Q, D] contiguous
            return q, k, v
        raise ValueError(mode)
    TL.Attention._prep_qkv = _prep_qkv
    _STATS["t2_mode"] = mode
    _STATS["applied"].append(f"T2:{mode}")

# =====================================================================================  T2b: fast EXACT layout copies (tiled transpose)
# Stock PairformerBlock (inplace path) does  z = z.transpose(-2,-3).contiguous()  twice per block ([N,N,256] bf16, swapping the two token axes);
# ATen lowers this to the generic strided elementwise copy kernel (elementwise_kernel<128,2|4> direct_copy), which runs far below HBM bandwidth
# for this access pattern.  A copy is a copy: any kernel that moves the same elements is bitwise identical.  We route  [.., I, J, C] -> [.., J, I, C]
# copies (C contiguous) through a Triton kernel that reads/writes full C-rows (coalesced both sides since C*2B = 512 B rows).
# Also used for cuEq's q/k/v head-major copies ([I, J, H, D] -> [I, H, J, D], D=32 -> 64 B rows) when PTX_T2_NOCOPY=fastcopy.
_T2B_KERNEL = None
def _build_t2b_kernel():
    global _T2B_KERNEL
    assert _HAS_TRITON, "triton not available"
    _T2B_KERNEL = _swap_mid_copy

def swap_mid_contiguous(x):
    """return x.transpose(-2,-3).contiguous() for x [..., I, J, C] with stride(-1)==1, via the tiled kernel (bitwise identical values)."""
    if _T2B_KERNEL is None or not x.is_cuda or x.dim() < 3 or x.stride(-1) != 1:
        return x.transpose(-2, -3).contiguous()
    lead = x.shape[:-3]; I, J, C = x.shape[-3], x.shape[-2], x.shape[-1]
    xb = x.reshape((-1, I, J, C)) if len(lead) != 1 else x
    if xb.dim() == 3: xb = xb.unsqueeze(0)
    if xb.data_ptr() != x.data_ptr() and len(lead) > 1:
        return x.transpose(-2, -3).contiguous()      # reshape copied (non-viewable) -> give up, stock path
    B = xb.shape[0]
    y = torch.empty((B, J, I, C), dtype=x.dtype, device=x.device)
    BLOCK_R = 16; BLOCK_C = min(256, triton_next_pow2(C))
    grid = ((I * J + BLOCK_R - 1) // BLOCK_R, B)
    _T2B_KERNEL[grid](xb, y, I, J, C, xb.stride(0), xb.stride(1), xb.stride(2), y.stride(0), y.stride(2), y.stride(1), BLOCK_R=BLOCK_R, BLOCK_C=BLOCK_C, num_warps=4)
    return y.reshape(tuple(lead) + (J, I, C))

def triton_next_pow2(n):
    p = 1
    while p < n: p *= 2
    return p

def _apply_t2b(mode):
    """PTX_ZT=fast : PairformerBlock.forward's two z.transpose(-2,-3).contiguous() -> swap_mid_contiguous (exact)."""
    import protenix.model.modules.pairformer as PF
    _build_t2b_kernel()
    _orig = PF.PairformerBlock.forward
    import types
    # Only the inplace branch (the inference path) is re-implemented; non-inplace falls back to stock.
    from protenix.model.modules.fused_ops import dropout_add_rowwise
    def forward(self, s, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        if not inplace_safe:
            return _orig(self, s, z, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        _STATS["zt_calls"] = _STATS.get("zt_calls", 0) + 1
        z = self.tri_mul_out(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
        z = self.tri_mul_in(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
        z += self.tri_att_start(z, mask=pair_mask, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        z = swap_mid_contiguous(z)
        z += self.tri_att_end(z, mask=pair_mask.transpose(-1, -2) if pair_mask is not None else None, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        z = swap_mid_contiguous(z)
        z += self.pair_transition(z)
        if self.c_s > 0:
            s = s + self.attention_pair_bias(a=s, s=None, z=z)
            s = s + self.single_transition(s)
        return s, z
    PF.PairformerBlock.forward = forward
    _STATS["zt_mode"] = mode
    _STATS["applied"].append(f"ZT:{mode}")

# =====================================================================================  T5: trimul projection einsum kernel selection
# cuEq 0.8.0 _tri_mul_update issues  torch.einsum("dbik,dbjk->dbij", a, b)  (outgoing) / ("dbki,dbkj->dbij") (incoming) with a, b = chunk(ab, 2, 0),
# ab = [2*c, B, N, N] bf16 (transpose_out=True output of the gated dual GEMM).  torch lowers this to at::bmm on [c*B, N, N] operands; on this stack
# cuBLAS serves it with cutlass_75_tensorop_bf16_s1688gemm_bf16_256x128 (an sm75-class WMMA kernel) at 705/813 tokens.  PTX_T5_EINSUM=<mode> swaps
# ONLY this einsum for an equivalent formulation; candidates are tested op-level (optest) and in-model (det config):
#   bmm      : torch.bmm(A, B^T) / torch.bmm(A^T, B) on reshaped [c*B, N, N] views (may or may not pick another kernel)
def bgemm_tri(A, Bm):
    """C = A @ Bm for 3-D strided views A [b, M, K], Bm [b, K, N] (bf16/fp16), fp32 accumulation, output contiguous [b, M, N] in A.dtype."""
    b, M, K = A.shape; N = Bm.shape[2]
    C = torch.empty((b, M, N), dtype=A.dtype, device=A.device)
    grid = lambda META: (triton.cdiv(M, META["BM"]) * triton.cdiv(N, META["BN"]), b)
    LAYOUT = int(A.stride(2) == 1) * 2 + int(Bm.stride(2) == 1)
    _bgemm_kernel[grid](A, Bm, C, M, N, K, A.stride(0), A.stride(1), A.stride(2), Bm.stride(0), Bm.stride(1), Bm.stride(2), C.stride(0), C.stride(1), C.stride(2), LAYOUT=LAYOUT)
    return C

def _apply_t5(mode):
    # NOTE: `import cuequivariance_ops_torch.triangle_multiplicative_update as TMU` binds the *function* of that name re-exported by the package
    # __init__ (IMPORT_FROM prefers the package attribute over the submodule), so patching TMU.torch never reaches the call
    # site.  Bind the real submodule from sys.modules instead.
    import importlib, sys as _sys
    importlib.import_module("cuequivariance_ops_torch.triangle_multiplicative_update")
    TMU = _sys.modules["cuequivariance_ops_torch.triangle_multiplicative_update"]
    assert hasattr(TMU, "_tri_mul_update") and hasattr(TMU, "torch"), "unexpected cuequivariance_ops_torch layout"
    _orig_einsum = torch.einsum
    def _einsum_hook(eq, *ops):
        if len(ops) == 2 and eq in ("dbik,dbjk->dbij", "dbki,dbkj->dbij") and ops[0].dim() == 4:
            a, b = ops
            _STATS["t5_calls"] = _STATS.get("t5_calls", 0) + 1
            c, B, N, K = a.shape if eq == "dbik,dbjk->dbij" else (a.shape[0], a.shape[1], a.shape[3], a.shape[2])
            A = a.reshape(c * B, a.shape[2], a.shape[3]); Bm = b.reshape(c * B, b.shape[2], b.shape[3])
            if mode == "bmm":
                if eq == "dbik,dbjk->dbij":
                    x = torch.bmm(A, Bm.transpose(1, 2))
                else:
                    x = torch.bmm(A.transpose(1, 2), Bm)
                return x.view(c, B, x.shape[1], x.shape[2])
            elif mode in ("pad", "padk"):
                # T5-pad: cuBLAS picks an sm75 'align1' kernel because the row pitch (N*2 bytes) of the [cB, N, N] bf16 operands is not 16-B aligned at
                # N = 356/546/705/813.  Zero-pad to Np = ceil8(N): products against zero pads contribute exact zeros; only the kernel (hence fp32
                # accumulation order) changes -> Tier 2 'reordered accumulation, same precision class'.  mode 'padk': pad the contraction dim only
                # (output needs no fix-up); mode 'pad': pad rows+cols too (fully aligned operands and output), then slice + contiguous.
                if a.dtype in (torch.bfloat16, torch.float16):
                    N_i = A.shape[1] if eq == "dbik,dbjk->dbij" else A.shape[2]
                    K = A.shape[2] if eq == "dbik,dbjk->dbij" else A.shape[1]
                    if K % 8 == 0 and (mode == "padk" or N_i % 8 == 0):
                        return _orig_einsum(eq, *ops)                 # already aligned -> stock
                    PM = int(os.environ.get("PTX_T5_PADMULT", "8"))
                    Kp = (K + PM - 1) // PM * PM
                    cB = A.shape[0]
                    if mode == "padk":
                        if eq == "dbik,dbjk->dbij":                   # A [cB, I, K], Bm [cB, J, K] -> pad K (last dim)
                            Ap = F.pad(A, (0, Kp - K)); Bp = F.pad(Bm, (0, Kp - K))
                            x = torch.bmm(Ap, Bp.transpose(1, 2))
                        else:                                          # A [cB, K, I], Bm [cB, K, J] -> pad K (middle dim)
                            Ap = F.pad(A, (0, 0, 0, Kp - K)); Bp = F.pad(Bm, (0, 0, 0, Kp - K))
                            x = torch.bmm(Ap.transpose(1, 2), Bp)
                        _STATS["t5_pad_calls"] = _STATS.get("t5_pad_calls", 0) + 1
                        return x.view(c, B, x.shape[1], x.shape[2])
                    else:
                        Np = (N_i + PM - 1) // PM * PM
                        Ap = F.pad(A, (0, Kp - K if eq == "dbik,dbjk->dbij" else Np - N_i, 0, Np - N_i if eq == "dbik,dbjk->dbij" else Kp - K))
                        Bp = F.pad(Bm, (0, Kp - K if eq == "dbik,dbjk->dbij" else Np - N_i, 0, Np - N_i if eq == "dbik,dbjk->dbij" else Kp - K))
                        if eq == "dbik,dbjk->dbij":
                            x = torch.bmm(Ap, Bp.transpose(1, 2))      # [cB, Np, Np]
                        else:
                            x = torch.bmm(Ap.transpose(1, 2), Bp)
                        x = x[:, :N_i, :N_i].contiguous()
                        _STATS["t5_pad_calls"] = _STATS.get("t5_pad_calls", 0) + 1
                        return x.view(c, B, N_i, N_i)
            elif mode == "tri":
                # Triton batched GEMM, fp32 accumulation over K in BK chunks (Tier 2 a priori: accumulation order differs from cuBLAS's kernel)
                if a.dtype in (torch.bfloat16, torch.float16):
                    if eq == "dbik,dbjk->dbij":
                        x = bgemm_tri(A, Bm.transpose(1, 2))          # [cB, I, K] @ [cB, K, J]
                    else:
                        x = bgemm_tri(A.transpose(1, 2), Bm)          # a:[cB, K, I] -> A^T [cB, I, K] @ b [cB, K, J]
                    return x.view(c, B, x.shape[1], x.shape[2])
        return _orig_einsum(eq, *ops)
    # patch only the name used inside the cuEq module (not torch.einsum globally)
    TMU.torch = _TorchProxy(torch, einsum=_einsum_hook)
    _STATS["t5_mode"] = mode; _STATS.setdefault("t5_calls", 0); _STATS.setdefault("t5_pad_calls", 0)
    _STATS["applied"].append(f"T5:{mode}")

class _TorchProxy:
    """module proxy: attribute lookups go to torch except the overridden names (keeps the patch local to one importing module)."""
    def __init__(self, mod, **over):
        object.__setattr__(self, "_mod", mod); object.__setattr__(self, "_over", over)
    def __getattr__(self, name):
        over = object.__getattribute__(self, "_over")
        if name in over:
            return over[name]
        return getattr(object.__getattribute__(self, "_mod"), name)

# =====================================================================================  OG: fused output gating of triangle attention
# stock (layers.Attention.forward/_wrap_up): o = cueq(...)[0] -> [I,H,J,D] contiguous; o = o.transpose(-2,-3) (strided [I,J,H,D]);
#   g = sigmoid(linear_g(q_x)) [I,J,H*D] bf16; g = g.view(.., H, D); o = o * g (strided read -> ATen mul, output takes o's (strided) layout);
#   o = flatten_final_dims(o, 2) -> reshape forces a copy; o = linear_o(o).
# OG=fuse: g_lin = linear_g(q_x) (same GEMM); ONE Triton kernel reads o (head-major) + g_lin, applies sigmoid with stock rounding (bf16), multiplies in
#   fp32, writes [I,J,H*D] contiguous; then linear_o.  Removes: sigmoid kernel, strided mul, reshape copy.  Expected bitwise.
def _apply_og(mode):
    import protenix.model.triangular.layers as TL
    from protenix.model.utils import flatten_final_dims
    _orig_wrap = TL.Attention._wrap_up
    def _wrap_up(self, o, q_x):
        # o: [*, Q, H, D] view; fast path when it is the transpose of a contiguous [*, H, Q, D] tensor and gating is on, 2 leading dims (I, J) case
        if (self.linear_g is None or o.dim() != 4 or o.dtype not in (torch.bfloat16, torch.float16)
                or not o.transpose(-2, -3).is_contiguous() or q_x.dim() != 3):
            _STATS["og_fallback"] = _STATS.get("og_fallback", 0) + 1
            return _orig_wrap(self, o, q_x)
        _STATS["og_calls"] = _STATS.get("og_calls", 0) + 1
        I, J, H, D = o.shape
        g_pre = getattr(self, "_t2g_gate", None)
        if g_pre is not None and g_pre.shape[-1] == H * D and tuple(g_pre.shape[:-1]) == tuple(q_x.shape[:-1]) and g_pre.dtype == o.dtype:
            g_lin = g_pre; self._t2g_gate = None; _STATS["og_gate_from_t2g"] = _STATS.get("og_gate_from_t2g", 0) + 1
        else:
            g_lin = self.linear_g(q_x)                                  # [I, J, H*D] (autocast bf16), same GEMM as stock
        if g_lin.dtype != o.dtype or g_lin.stride(-1) != 1:
            _STATS["og_fallback"] = _STATS.get("og_fallback", 0) + 1
            return _orig_wrap(self, o, q_x)
        ob = o.transpose(-2, -3)                                    # [I, H, J, D] contiguous base
        out = torch.empty((I, J, H * D), dtype=o.dtype, device=o.device)
        BLOCK_J = 8
        grid = (I, (J + BLOCK_J - 1) // BLOCK_J)
        if mode == "fuse2":
            BJ2 = 64 if J >= 512 else 32
            _gate_out_kernel2[(I, (J + BJ2 - 1) // BJ2)](ob, g_lin, out, J, ob.stride(0), ob.stride(1), ob.stride(2), g_lin.stride(0), g_lin.stride(1), H=H, D=D, BLOCK_J=BJ2, num_warps=4)
        else:
            _gate_out_kernel[grid](ob, g_lin, out, J, H, D, ob.stride(0), ob.stride(1), ob.stride(2), g_lin.stride(0), g_lin.stride(1),
                                   BLOCK_J=BLOCK_J, HD_P2=triton_next_pow2(H * D), num_warps=4)
        return self.linear_o(out)
    TL.Attention._wrap_up = _wrap_up
    _STATS["og_mode"] = mode
    _STATS["applied"].append(f"OG:{mode}")

# =====================================================================================  NM: triangle attention without the all-ones mask (exact)
# ---- EXACTNESS GUARD: on cu13 cuEquivariance builds, compute capability 10.x, a triangle_attention call with
# mask=None at N % 8 == 0 is dispatched by cuEq itself to its Blackwell 'sm100f' kernel (results differ element-wise from the masked sm80 path the STOCK Protenix
# call always takes, because stock passes a dense bool mask). All ARM-E code paths that pass mask=None (NOMASK wrapper, BLK2 core, chunked core, template/v02 core,
# fpf_smalln exact branch) consult ONE predicate and pass the stock all-True bool mask when exposed. ARM T providers (K2B etc.) are unaffected (Tier-2 by label).
_SM100F_STATE = {"decided": False, "build_exposed": False, "why": "", "printed": False}
def _cueq_build_has_sm100f() -> bool:
    """True iff this process COULD hit the sm100f path: cc major == 10 AND the cuequivariance build is a cu13 build (or exposes an sm100f symbol).
    Conservative: any detection error on cc 10.x counts as exposed (fail towards the stock mask = exact). PTX_SM100F_GUARD=0 disables, =1 forces."""
    st = _SM100F_STATE
    if st["decided"]:
        return st["build_exposed"]
    st["decided"] = True
    ovr = os.environ.get("PTX_SM100F_GUARD", "")
    try:
        cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
    except Exception:
        cc = (0, 0)
    if ovr == "0":
        st.update(build_exposed=False, why="PTX_SM100F_GUARD=0 (guard disabled by user)"); return False
    if ovr == "1":
        st.update(build_exposed=True, why="PTX_SM100F_GUARD=1 (forced)"); return True
    if cc[0] != 10:
        st.update(build_exposed=False, why=f"cc={tuple(cc)} (not 10.x)"); return False
    why = []
    exposed = None
    try:
        import importlib.metadata as _md
        names = [d.metadata["Name"] for d in _md.distributions() if (d.metadata["Name"] or "").lower().startswith("cuequivariance")]
        why.append("dists=" + ",".join(sorted(set(names)))[:120])
        if any("cu13" in n.lower() for n in names): exposed = True
        elif any("cu12" in n.lower() or "cu11" in n.lower() for n in names): exposed = False
    except Exception as e:
        why.append(f"dist-probe:{e!r}"[:80])
    if exposed is None:
        try:
            import cuequivariance_ops_torch as _cot  # noqa
            syms = [a for a in dir(_cot) if "sm100" in a.lower()]
            try:
                import cuequivariance_ops as _co; syms += [a for a in dir(_co) if "sm100" in a.lower()]
            except Exception:
                pass
            if syms: exposed = True; why.append("symbols=" + ",".join(syms)[:80])
        except Exception as e:
            why.append(f"ops-probe:{e!r}"[:60])
    if exposed is None:
        exposed = True; why.append("undetermined on cc 10.x -> treated as exposed (fail towards stock mask)")
    st.update(build_exposed=bool(exposed), why="; ".join(why))
    return st["build_exposed"]
def _cueq_sm100f_exposed(n_kv: int) -> bool:
    """Per call: True => pass the stock bool mask instead of None (exactness guard). Prints the decision once per process."""
    ex = _cueq_build_has_sm100f() and (int(n_kv) % 8 == 0)
    st = _SM100F_STATE
    if not st["printed"] and (ex or (st["build_exposed"] and not st["printed"])):
        st["printed"] = True
        print(f"[FPF] NOMASK exactness guard: cuEq sm100f-capable build on cc 10.x ({st['why']}) -> stock bool mask passed at N%8==0 (first such N={int(n_kv)}: {'MASK' if ex else 'None'})", flush=True)
        _STATS["sm100f_guard"] = dict(st)
    if ex: _STATS["sm100f_guard_masked_calls"] = _STATS.get("sm100f_guard_masked_calls", 0) + 1
    return ex
def _stock_true_mask(q, n_q_rows: int, n_kv: int):
    """The mask stock hands cuEq when Protenix passes mask=None: (inf*(ones-1) == 0) -> all-True bool of shape [*, I, 1, 1, J] broadcast; cuEq accepts [B?, I, 1, 1, J].
    Built to match stock exactly: mask_bias = (inf * (mask - 1))[..., :, None, None, :] with mask = ones[I, J] -> (mask_bias == 0) -> True everywhere."""
    lead = q.shape[:-4]                                  # q [.., I, H, J, D]
    return torch.ones(tuple(lead) + (int(n_q_rows), 1, 1, int(n_kv)), dtype=torch.bool, device=q.device)

def _apply_nomask():
    """PTX_NOMASK=1 : Protenix calls TriangleAttention with mask=None (pair_mask=None in the trunk); stock then builds mask = ones, mask_bias = inf*(mask-1) = 0,
    and hands cuEquivariance `(mask_bias == 0).bool()` = all True. cuEq triangle_attention(mask=None) is bitwise identical to mask=all-True and
    cheaper per call; the mask/mask_bias construction disappears too. Only for the cuequivariance backend without chunking and when no tri-attention
    provider (PF_TRIATTN) is set (that kernel's mask=None behaviour is not ours to test) -> otherwise stock path."""
    import math as _math
    import protenix.model.triangular.triangular as TT
    import protenix.model.triangular.layers as TL
    from protenix.model.utils import permute_final_dims
    if os.environ.get("PF_TRIATTN", "").strip():
        _STATS["applied"].append("NM:skipped(flash tri-att patch set)"); return
    _orig_tt = TT.TriangleAttention.forward; _orig_att = TL.Attention.forward
    def tt_forward(self, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
        if mask is not None or triangle_attention != "cuequivariance" or chunk_size is not None or x.shape[-2] <= 16:
            return _orig_tt(self, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
        _STATS["nm_calls"] = _STATS.get("nm_calls", 0) + 1
        if not self.starting:
            x = x.transpose(-2, -3)
        x = self.layer_norm(x)
        triangle_bias = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)
        x = self.mha(q_x=x, kv_x=x, biases=[None, triangle_bias], triangle_attention=triangle_attention)
        if not self.starting:
            x = x.transpose(-2, -3)
        return x
    def att_forward(self, q_x, kv_x, biases=None, triangle_attention="torch"):
        if triangle_attention == "cuequivariance" and biases is not None and len(biases) == 2 and biases[0] is None:
            q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=False)
            scale = 1.0 / _math.sqrt(self.c_hidden)
            o = TL.cuequivariance_triangular_attn(q, k, v, biases[1].float(), (_stock_true_mask(q, q.shape[-4], k.shape[-2]) if _cueq_sm100f_exposed(k.shape[-2]) else None), scale)[0]   # exactness guard (sm100f)
            o = o.transpose(-2, -3)
            return self._wrap_up(o, q_x)
        return _orig_att(self, q_x, kv_x, biases=biases, triangle_attention=triangle_attention)
    TT.TriangleAttention.forward = tt_forward
    TL.Attention.forward = att_forward
    _STATS.setdefault("nm_calls", 0)
    _STATS["applied"].append("NM:1")

# =====================================================================================  BLK: FPF block-mode composition of PairformerBlock (inference, inplace path)
# PTX_BLK=1 : replaces PairformerBlock.forward (c_s>=0, incl. MSA/template pair stacks) for the inplace_safe & cuequivariance path by:
#   z = tri_mul_out(z) ; z = tri_mul_in(z)                          (stock module calls => cuEq pipeline, or PF pad if that add-on patched the module class)
#   z += tri_att_start(z)      via Fusion epilogue block-mode (stock LN/bias/qkv[g] GEMMs [+ our T2/T2g], cuEq/flash kernel, ONE fused epilogue: sigmoid(g)*o @ Wo^T + residual in place)
#   z[j,i] += tri_att_end(z^T)[i,j]  via the same kernel with ending=True: NO z.transpose(-2,-3).contiguous() passes at all (stock does two per block)
#   z += pair_transition(z)   via Fusion fn_residual if importable (fused LN-call + MLP + residual), else our T1 path + add
#   single track (c_s>0): stock attention_pair_bias + single_transition (unchanged statements)
# Exactness: every statement is a kernel that is bitwise-equal to the stock statement it replaces => the block is expected EXACT.
def _apply_blk():
    import protenix.model.modules.pairformer as PF
    try:
        from fpf_triatt_epi.epilogue import fn_block_residual as _epi_block
    except Exception as e:
        _STATS["applied"].append(f"BLK:unavailable({e!r})"); return
    _orig = PF.PairformerBlock.forward
    def blk_forward(self, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        if (not inplace_safe) or triangle_attention != "cuequivariance" or pair_mask is not None or chunk_size is not None or z.dtype != torch.bfloat16 or z.shape[-2] <= 16 or self.training:
            _STATS["blk_fallback"] = _STATS.get("blk_fallback", 0) + 1
            return _orig(self, s, z, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        if int(self.tri_att_start.c_in) != 256 or int(self.tri_att_start.mha.no_heads) != 8:      # only the checked (256,8,32) cell; template c=64 stack -> stock block
            _STATS["blk_fallback_c"] = _STATS.get("blk_fallback_c", 0) + 1
            return _orig(self, s, z, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        z = self.tri_mul_out(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
        z = self.tri_mul_in(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
        _epi_block(self.tri_att_start, z, ending=False, mask=None, triangle_attention=triangle_attention)
        _epi_block(self.tri_att_end, z, ending=True, mask=None, triangle_attention=triangle_attention)
        if os.environ.get("PTX_BLK_TRANS", "fusion") == "fusion":
            z = _TRC.block_pair_transition(self.pair_transition, z)      # z += pair_transition(z): opt_core.kernels.transition by the mode's tier word (residual folded), else the module statement in place, by name
        else:
            z += self.pair_transition(z)
        _STATS["blk_calls"] = _STATS.get("blk_calls", 0) + 1
        if self.c_s > 0:
            s = s + self.attention_pair_bias(a=s, s=None, z=z)
            s = s + self.single_transition(s)
            return s, z
        return s, z      # stock returns (s, z) also when c_s == 0 (s passes through untouched)
    PF.PairformerBlock.forward = blk_forward
    _STATS["applied"].append("BLK:1" + "+transition_core")

# =====================================================================================  BLK2: level-3 exact tri-att path = Fusion PROLOGUE (stock LN call + one producer kernel: q,k,v [I,H,J,D], g, fp32 bias;
# ending node reads z^T through strides) -> cuEq attention (mask=None) -> Fusion EPILOGUE block-mode (sigmoid(g)*o @ Wo^T + residual into z / z^T in place). PTX_BLK=2.
_BLK_ATT = {}
_T_MIN_TOK = 0 if os.environ.get("FPF_SMALLN", "0") not in ("", "0") else int(os.environ.get("PTX_T_MIN_TOKENS", "0") or 0)   # when fpf_smalln routes ARM=T (env.sh default) IT owns both thresholds (one mechanism); the in-levers gate is only the fallback when fpf_smalln is absent
_BLK_ATT_MIN_TOK = int(os.environ.get("PTX_BLK_ATT_MIN_TOKENS", "0") or 0) or _T_MIN_TOK
_T_CERT_MAX = 1775                                                            # largest N_token with op-level numerics coverage for BOTH Tier-2 kernels (K2B, the Triton TriMul); above it each prints one notice line per process
_T_MAX_TOK = int(os.environ.get("PTX_T_MAX_TOKENS", "0") or 0)                # 0 (default) = no ceiling: one loud line per process above _T_CERT_MAX; N = gate: above N both Tier-2 kernels -> exact E sub-paths
def _t_ceiling(n, what):
    """ARM=T only. Returns True if the Tier-2 kernel `what` must NOT serve this N (gate set and n > gate). Prints one notice per (what) when serving above _T_CERT_MAX."""
    if _T_MAX_TOK and n > _T_MAX_TOK:
        _STATS["t_gated_large_" + what] = _STATS.get("t_gated_large_" + what, 0) + 1
        _say_once("tmax_" + what, f"[FPF] ARM=T: N={n} > PTX_T_MAX_TOKENS={_T_MAX_TOK} -> {what} routed to the exact E sub-path")
        return True
    if n > _T_CERT_MAX:
        _STATS["t_unverified_at_n_" + what] = _STATS.get("t_unverified_at_n_" + what, 0) + 1
        _say_once("tcert_" + what, f"[FPF] ARM=T: N={n} above op-tested max {_T_CERT_MAX} tok — Tier-2 kernel {what} serving UNTESTED-AT-N (set PTX_T_MAX_TOKENS={_T_CERT_MAX} to route it to the exact path above that)")
    return False
def _blk_core_att(n):
    """The block core's attention callable for a tri-attention statement over N=n keys on THIS arm, or None = the stock cuEq call.
    ARM E never binds one (None: exact). ARM T: the routed K2B slot (`_BLK_ATT['fn']`, fpf_smalln-gated) unless below PTX_BLK_ATT_MIN_TOKENS / above
    PTX_T_MAX_TOKENS (`_t_ceiling`, counted + printed) or the slot is the PAD8-only marker. ONE rule for the full-N statement and the XL lean statement
    (bundle branch below and third_party/blockfuse_addon/blockfuse_xl), so the lean statement can never silently drop the arm's kernel again."""
    att = _BLK_ATT.get("fn")
    if att is not None and (n < _BLK_ATT_MIN_TOK or _t_ceiling(int(n), "K2B")):
        _STATS["blk_att_gated_small"] = _STATS.get("blk_att_gated_small", 0) + 1; return None
    if getattr(att, "_fpf_pad8_marker", False):
        return None
    return att
   # K2B size gate inside the block core (ARM=T); 0 = off. Printed at apply time.
_BLK_CHUNKED = os.environ.get("PTX_BLK_CHUNKED", "0")            # LARGE-N: '1'|'mirror' = chunk-aware BLK2 core for chunk_size != None (N > 1024 on the stock CLI): per-stock-row-chunk cuEq call
                                                                # (batch = chunk rows, full key axis, full fp32 bias == TriangleAttention._chunk/chunk_layer semantics) + per-chunk epilogue in place;
                                                                # 'unchunked' = one cuEq call over all rows (bitwise with the per-chunk form); 'k2b' = with PTX_BLK_ATT=k2b, ARM T's K2B core over all rows (TIER-2, NOT exact)
# ---- XL TOKEN POLICY (graceful degradation at large N_token; every degradation below is an EXACT stock statement, so ARM E stays exact by construction;
#      the composition is DET-tested at the size where it engages).  Knobs (read at import):
#        PTX_FPF_CHUNK_TOK   = 0 | <N> | auto : N_token >  value -> the BLK2 tri-attention statement (fused prologue = full-N q/k/v/g, ~4*N*N*c_z*2 B transient) is replaced by the
#                              LEAN statement: projections per stock row-chunk (bounded workspace); the attention CORE follows the arm:
#                              ARM E -> the stock cuEq kernel per chunk (exact, as before); ARM T -> the block core's K2B per chunk (`_blk_core_att`; bitwise == the
#                              full-N K2B statement: K2B's grid batches pair rows, stock chunk sizes are multiples of its row group).
#                              TriMul above FPF_TRIMUL_EXACT_NMAX is already the stock kernel.
#        PTX_XL_PAD8_MAX_TOK = 0 | <N> | auto : N_token >  value -> PAD8-exact provider not used (unpadded stock cuEq call; exact either way) -> no padded Q/K/V/bias copies.
#        auto = largest N at which the FULL E* block path stays under 85 % of device memory by the audited quadratic (4.0 GB + 45.0 GB x (N/1981)^2): 80 GB -> 2304, 141 GB -> 3072, 178 GB -> 3584.
#        PTX_FPF_CHUNK_MODE  = stock | prologue : HOW the lean tri-attention runs above the threshold: 'stock' = exact stock chunked statement (DET-proven 9/9 @1,981/2,109; ~x1.10 vs stock);
#                              'prologue' = tested BLK2 prologue/cuEq/epilogue per stock row-chunk (bias rows concatenated; keeps the block-path speed; DET at 2,559 pending -> opt-in).
#      Ledger: report()["xl"] + one '[FPF] XLMEM ...' stderr line at exit = what each lever holds/saves in memory at the largest N_token seen (estimates are analytic: 4*N^2*256 B fused-prologue
#      transient, PAD8 copies from the provider's own byte counters when present) + torch peak allocated/reserved.
_XL_MODEL = {"base_gb": 4.0, "full_gb_at_ref": 45.0, "ref_ntok": 1981, "headroom": 0.85}   # memory model of the FULL statement (H100 80 GB reference point at ref_ntok tokens): E* full peak 49.0 GB = 4.0 + 45.0; stock 41.1
def _xl_auto(name, default="0"):
    """0/off -> disabled; <N> -> explicit; auto -> largest N (multiple of 128) at which the FULL E* block path is predicted to stay under headroom*device memory:
    peak_full(N) ~= base + full_gb_at_ref * (N/ref)^2  (quadratic in N_token; calibration in _XL_MODEL, from the XL audit)."""
    v = (os.environ.get(name, default) or "0").strip().lower()
    if v in ("0", "off", ""): return 0
    if v != "auto": return int(v)
    try:
        tot = torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory / 2**30
        n = _XL_MODEL["ref_ntok"] * math.sqrt(max(0.0, _XL_MODEL["headroom"] * tot - _XL_MODEL["base_gb"]) / _XL_MODEL["full_gb_at_ref"])
        n = max(1024, int(n // 128) * 128)             # 80 GB -> 2304, 141 GB -> 3072, 178 GB -> 3584, 48 GB -> 1664
        if tot <= 90.0: n = min(n, 2299)                # the E* FULL path runs out of memory near 2,559 tokens on an 80 GB card (trunk + fused prologue workspace) -> on <= 80 GB-class cards
                                                        # every N_token >= 2,300 runs LEAN (stock chunked tri-attention / no PAD8); >= 141 GB cards keep the formula (full path)
        return n
    except Exception:
        return 0
_XL_CHUNK_TOK = _xl_auto("PTX_FPF_CHUNK_TOK")
_XL_PAD8_MAX = _xl_auto("PTX_XL_PAD8_MAX_TOK")
_XL_CHUNK_MODE = (os.environ.get("PTX_FPF_CHUNK_MODE", "stock") or "stock").strip().lower()   # 'stock' (the stock chunked statements: exact) | 'prologue' (tested BLK2
                                                                                                # prologue/cuEq/epilogue run PER STOCK ROW-CHUNK -> q/k/v/g never full-N; bias rows concatenated; DET pending -> not default)
_XL_MEM = {"lean_k2b_chunks": 0, "lean_cueq_chunks": 0, "max_ntok": 0, "lean_triatt_stmts": 0, "lean_prologue_calls": 0, "pad8_skipped_gt_max": 0, "chunk_mode": _XL_CHUNK_MODE}
def _xl_lean(z):
    """True -> this block's tri-attention runs the STOCK chunked statement (mode 'stock' and N_token > PTX_FPF_CHUNK_TOK). Records the largest N seen for the ledger."""
    n = int(z.shape[-2])
    if n > _XL_MEM["max_ntok"]: _XL_MEM["max_ntok"] = n
    return _XL_CHUNK_TOK > 0 and n > _XL_CHUNK_TOK and _XL_CHUNK_MODE != "prologue"
def _xl_lean_prologue(n):
    """True -> mode 'prologue': inside the BLK2 core, run the prologue per stock row-chunk (bounded transient) instead of once over all rows."""
    return _XL_CHUNK_TOK > 0 and int(n) > _XL_CHUNK_TOK and _XL_CHUNK_MODE == "prologue"
def _xl_ledger():
    n = _XL_MEM["max_ntok"]; gb = 2**30
    d = dict(_XL_MEM, chunk_tok=_XL_CHUNK_TOK, pad8_max_tok=_XL_PAD8_MAX, alloc_conf=os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
             est_fused_prologue_transient_gb=round(4 * n * n * 256 * 2 / gb, 2),                  # q,k,v,g bf16 [N,N,4x64] each = one [N,N,1024] bf16 buffer (measured 12.49 GiB @2,559), held only inside the call; mode 'prologue' bounds it to chunk/N of this
             est_pad8_copies_gb=round((3 * ((n + 7) // 8 * 8) ** 2 * 128 * 2 + 4 * ((n + 7) // 8 * 8) ** 2 * 4) / gb, 2) if n % 8 else 0.0,   # shared padded Q/K/V bf16 + fp32 bias set (one set per process)
             lean_active=bool(_XL_CHUNK_TOK and n > _XL_CHUNK_TOK), pad8_off_by_max=bool(_XL_PAD8_MAX and n > _XL_PAD8_MAX))
    try:
        d["torch_max_allocated_gb"] = round(torch.cuda.max_memory_allocated() / gb, 2); d["torch_max_reserved_gb"] = round(torch.cuda.max_memory_reserved() / gb, 2)
        fr, tot = torch.cuda.mem_get_info(); d["device_total_gb"] = round(tot / gb, 1)
    except Exception:
        pass
    return d

# ---- PADDED-LAYOUT provider core (ARM T only; default-off): PTX_BLK_PADDED=8 + PTX_BLK_ATT=<module>:<attr> provider that declares kv_pad=8.
# ONE SHARED buffer set per (P, H, D, HD, device) — NOT per module (per-module sets would multiply the buffer memory by the module count). Block calls are
# sequential on one stream and every module at a given N overwrites the whole valid region, so one static set per P is sufficient (and still address-static for stackgraph).
# Set = Qp/Kp/Vp zeros[P,H,P,D] bf16, Bp zeros[1,H,P,P] fp32 with columns [N,P) := bias_pad_fill (tagged _fpf_padfilled/_fpf_zeroed),
# g [P,P,HD]; the padded prologue writes only the valid region; provider gets full padded views + kv_len=q_len=N; epilogue reads o[..., :N, :, :N, :] as a strided view.
_BLK_LN = os.environ.get("PTX_BLK_LN", "stock")                      # LayerNorm feeding the BLK2 tri-att prologue: 'stock' (module.layer_norm call: exact by construction; on the ENDING node
#   fast_layernorm makes a [N,N,C] .contiguous() copy of the z^T view) | 'welford' (in-prologue emulation of fast_layernorm's Welford/butterfly for C==256, reads z^T BY STRIDES: no LN kernel,
#   no copy; EXACT only if tested bitwise) | 'fused' (in-prologue two-pass fp32 LN: TIER-2).  FPF_BLK_LN_FMA="1,1,1" = (m2, merge, affine) FMA flags for 'welford'.
if _BLK_LN not in ("stock", "welford", "fused"): _BLK_LN = "stock"
_BLK_LN_FMA = tuple(x.strip() == "1" for x in (os.environ.get("FPF_BLK_LN_FMA", "1,1,1") + ",1,1").split(",")[:3])
_BLK_LN_SCOPE = os.environ.get("PTX_BLK_LN_SCOPE", "both")               # 'ending' = only the ending node (where the copy is) | 'both'
def _blk_ln_mode(module, x, ending):
    """-> (ln_mode, x_ln or None). Non-stock modes only for C==256 (welford) and when in scope; anything else -> stock call."""
    if _BLK_LN == "stock" or (_BLK_LN_SCOPE == "ending" and not ending) or (_BLK_LN == "welford" and x.shape[-1] != 256):
        return "stock", module.layer_norm(x)
    _STATS["blk_ln_" + _BLK_LN] = _STATS.get("blk_ln_" + _BLK_LN, 0) + 1
    return _BLK_LN, None
_BLK_PADDED = int(os.environ.get("PTX_BLK_PADDED", "0") or 0)          # 0 = off; 8 = pad token axes to a multiple of 8 for providers with kv_pad == 8
_BLK_PADDED_ASSERT = os.environ.get("PTX_BLK_PADDED_ASSERT", "0") == "1"   # debug: isfinite + zero check of the K/V pad slices before EVERY provider call (host sync; S13b: NaN in pad K/V rows is served and poisons all valid outputs)
_BLK_ATT_META = {"kv_pad": 1, "bias_pad_fill": None, "wants_kv_len": False, "name": "cueq", "label": "EXACT", "refused_exc": (), "raw": None, "min_tokens": 512}
_PAD_BUFS = {}
_PAD_STATS_KEYS = ("blk_att_padded_calls", "blk_att_padded_provider_refused", "blk_att_padded_bufs")
def _pad8_marker(*a, **k):                      # registered in _BLK_ATT['fn'] under PTX_E_PAD8 only so the core evaluates the provider branch; never called
    raise RuntimeError("pad8 marker called")
_pad8_marker._fpf_pad8_marker = True

def _padded_bufs(module, N: int, H: int, D: int, HD: int, dev, fill):
    P = ((N + _BLK_PADDED - 1) // _BLK_PADDED) * _BLK_PADDED
    key = (P, int(H), int(D), int(HD), str(dev))                       # shared across modules
    b = _PAD_BUFS.get(key)
    _capturing = bool(torch.cuda.is_available() and torch.cuda.is_current_stream_capturing())   # CUDA-graph capture (stackgraph): host-conditional re-zero would bake stale pads into the graph ->
    if b is None or b["N"] != N or b["fill"] != fill or _capturing:  # while capturing ALWAYS re-zero + re-fill inside the captured region (replay-correct for any interleaving of N within a P class)
        if b is None:
            if _capturing: _STATS["blk_att_padded_alloc_during_capture"] = _STATS.get("blk_att_padded_alloc_during_capture", 0) + 1   # allocation inside capture comes from the graph pool (stackgraph check2 guards correctness)
            b = {"q": torch.zeros((P, H, P, D), dtype=torch.bfloat16, device=dev), "k": torch.zeros((P, H, P, D), dtype=torch.bfloat16, device=dev),
                 "v": torch.zeros((P, H, P, D), dtype=torch.bfloat16, device=dev), "bias": torch.zeros((1, H, P, P), dtype=torch.float32, device=dev),
                 "g": torch.zeros((P, P, HD), dtype=torch.bfloat16, device=dev), "P": P, "N": None, "fill": None}
            b["bytes"] = sum(t.numel() * t.element_size() for t in (b["q"], b["k"], b["v"], b["bias"], b["g"]))
            _STATS["blk_att_padded_bufs"] = _STATS.get("blk_att_padded_bufs", 0) + 1
            _STATS["blk_att_padded_buf_bytes"] = _STATS.get("blk_att_padded_buf_bytes", 0) + b["bytes"]
            nP = len({k[0] for k in _PAD_BUFS} | {P})
            assert _STATS["blk_att_padded_bufs"] <= nP * 2, f"padded buffer sets {_STATS['blk_att_padded_bufs']} exceed 2 x distinct P ({nP})"   # invariant: <= 1 set per distinct (P, dims); x2 slack only for a second (H,D) cell
            if _STATS["blk_att_padded_bufs"] <= 3 or os.environ.get("FPF_VERBOSE"):
                print(f"[FPF] padded buffers: new shared set P={P} (H={H},D={D}) {b['bytes']/2**20:.0f} MiB; sets={_STATS['blk_att_padded_bufs']} total={_STATS['blk_att_padded_buf_bytes']/2**30:.2f} GiB", flush=True)
        else:                                        # stale valid data of the previous N now sits in pad rows/cols -> re-zero everything (pads must be zero AND NaN-free)
            for t in (b["q"], b["k"], b["v"], b["g"]): t.zero_()
            b["bias"].zero_()
            _STATS["blk_att_padded_rezero"] = _STATS.get("blk_att_padded_rezero", 0) + 1
        if fill is not None and N < P:
            b["bias"][..., N:] = float(fill)         # pad COLUMNS (all rows); rows >= N are don't-care (zero)
        b["bias"]._fpf_padfilled = True; b["bias"]._fpf_zeroed = True
        for t in (b["q"], b["k"], b["v"]): t._fpf_zeroed = True
        b["N"] = N; b["fill"] = fill
        _PAD_BUFS[key] = b
    return b
def _triatt_block_pro_epi(module, z, ending, epi_block_kernel, prologue, get_cache, chunk_size=None):
    import protenix.model.triangular.layers as TL
    x = z.transpose(-2, -3) if ending else z                     # x-frame view (no copy)
    _lnm, x_ln = _blk_ln_mode(module, x, ending)                 # PTX_BLK_LN: stock (default) = module.layer_norm(x) = stock fast_layernorm (copies the strided view internally exactly like stock's LN on transposed-contiguous z? NO:
    # stock runs LN on the contiguous transposed z; LN is row-wise so LN(z^T view) == LN(zT contiguous) elementwise & bitwise (same kernel, same rows) -> x_ln contiguous [I,J,C] in x-frame
    att = _BLK_ATT.get("fn")
    if att is not None and (x.shape[-2] < _BLK_ATT_MIN_TOK or _t_ceiling(int(x.shape[-2]), "K2B")):   # size gate + ceiling notice (default-off: PTX_BLK_ATT_MIN_TOKENS=0): below the gate the block core uses the stock cuEq kernel
        att = None; _STATS["blk_att_gated_small"] = _STATS.get("blk_att_gated_small", 0) + 1
    _rawp = _BLK_ATT_META.get("raw")
    _att_is_marker = getattr(att, "_fpf_pad8_marker", False)
    if _rawp is not None and _XL_PAD8_MAX > 0 and _BLK_ATT_META.get("only_unaligned") and int(x.shape[-2]) > _XL_PAD8_MAX:   # XL: PAD8 off above PTX_XL_PAD8_MAX_TOK -> unpadded stock cuEq path (exact)
        _rawp = None; _XL_MEM["pad8_skipped_gt_max"] += 1
    if (att is not None and _rawp is not None and _BLK_PADDED > 1 and _BLK_ATT_META.get("kv_pad", 1) > 1 and x.dim() == 3 and not (_BLK_ATT_META.get("only_unaligned") and int(x.shape[-2]) % 8 == 0)
            and int(x.shape[-2]) >= max(int(_BLK_ATT_META.get("min_tokens", 0) or 0), int(_BLK_ATT_MIN_TOK or 0), int(os.environ.get("FPF_SMALLN_K2B_MIN_TOKENS", "0") or 0))):   # PADDED provider path (ARM T; Tier-2): RAW provider at N >= its threshold and >= every exact gate; below -> unpadded path (K2B / cuEq via the routed slot)
        from fpf_triatt_pro.prologue import triatt_prologue_padded as _pro_pad
        if torch.cuda.is_current_stream_capturing() and _BLK_ATT_META.get("only_unaligned"):      # PAD8 provider probes (extra stock call + sync) on the first call per class/N: never inside a capture.
            _pw = getattr(_rawp, "__self__", None)                                                  # stackgraph captures only from the 2nd call of a signature, so the eager 1st call has probed already;
            _STATS["blk_att_padded_calls_in_capture"] = _STATS.get("blk_att_padded_calls_in_capture", 0) + 1   # counted so a census can confirm probes never coincide with capture
        N = int(x.shape[-2]); Hh = module.mha.num_heads if hasattr(module.mha, "num_heads") else module.mha.linear_q.weight.shape[0] // module.mha.c_hidden; Dd = module.mha.c_hidden
        bufs = _padded_bufs(module, N, Hh, Dd, Hh * Dd, z.device, _BLK_ATT_META.get("bias_pad_fill"))
        qP, kP, vP, gv, bP = _pro_pad(module, (x if x_ln is None else x_ln), bufs, ending=False, ln_mode=_lnm, x_ln=x_ln, fma_flags=_BLK_LN_FMA)
        scale = 1.0 / math.sqrt(module.mha.c_hidden)
        cch = getattr(module, "_fpf_cache", None)
        if cch is None:
            cch = {}; module._fpf_cache = cch
        ns = cch.setdefault("trunk2", {})
        wo16 = ns.get("wo16")
        if wo16 is None or wo16.device != z.device:
            wo16 = module.mha.linear_o.weight.detach().to(torch.bfloat16).contiguous(); ns["wo16"] = wo16
        cch.setdefault("wo16", wo16)
        P = bufs["P"]
        rngs = [(0, N)] if chunk_size is None or _BLK_CHUNKED == "unchunked" else [(i0, min(N, i0 + int(chunk_size))) for i0 in range(0, N, int(chunk_size))]
        try:
            for (i0, i1) in rngs:                                                     # provider chunk rule: slice Qp/Kp/Vp ROWS together, shared Bp, kv_len=N
                full = (i0 == 0 and i1 == N and len(rngs) == 1)
                q5 = (qP if full else qP[i0:i1]).unsqueeze(0); k5 = (kP if full else kP[i0:i1]).unsqueeze(0); v5 = (vP if full else vP[i0:i1]).unsqueeze(0)
                if _BLK_PADDED_ASSERT and P > N:                                      # S13b contract: NaN anywhere in K/V pad rows poisons EVERY valid output; the provider probe runs once per class and does NOT protect later calls
                    assert bool(torch.isfinite(kP[:, :, N:]).all()) and bool(torch.isfinite(vP[:, :, N:]).all()) and bool(torch.isfinite(kP[N:]).all()) and bool(torch.isfinite(vP[N:]).all()), "padded K/V pad slice not finite"
                    assert float(kP[:, :, N:].abs().max()) == 0.0 and float(vP[:, :, N:].abs().max()) == 0.0, "padded K/V pad columns not zero"
                    _STATS["blk_att_padded_assert_checks"] = _STATS.get("blk_att_padded_assert_checks", 0) + 1
                o = _rawp(q5, k5, v5, bP, mask=None, scale=scale, kv_len=N, q_len=(N if full else (i1 - i0)))
                o = o[0] if isinstance(o, (tuple, list)) else o
                if o.dim() == 5:
                    assert o.shape[0] == 1, f"provider output leading dim {tuple(o.shape)}"; o = o[0]
                ov = o[: (N if full else (i1 - i0)), :, :N, :]                          # STRIDED VIEW crop (no copy); epilogue addresses o by strides
                assert ov.stride(-1) == 1, "provider output last-dim stride must be 1"
                zr = (z[:, i0:i1] if ending else z[i0:i1]) if not full else z
                epi_block_kernel(ov, gv[i0:i1] if not full else gv, wo16, zr, ending=bool(ending), residual=True)
                del o, ov
            _STATS["blk_att_padded_calls"] = _STATS.get("blk_att_padded_calls", 0) + 1
            return z
        except _BLK_ATT_META.get("refused_exc", ()) as _e:                             # provider Refused -> fall through to the regular (unpadded) T/E path below; counted
            _STATS["blk_att_padded_provider_refused"] = _STATS.get("blk_att_padded_provider_refused", 0) + 1
            rs = _STATS.setdefault("blk_att_padded_refusals", {}); k_ = repr(_e)[:60]; rs[k_] = rs.get(k_, 0) + 1
            pass                                                                       # fall through to the unpadded path below
    if _att_is_marker: att = None                                                          # PTX_E_PAD8 alone: the unpadded path is the stock cuEq call (exact)
    if chunk_size is not None and _xl_lean_prologue(x.shape[-2]):                        # XL mode 'prologue': lean statement; the core follows the arm (cuEq under E, K2B under T)
        att_lean = None if _att_is_marker else att                                         # == _blk_core_att(N): `att` was gated above exactly as the helper gates it
        _kw = {"fma_flags": _BLK_LN_FMA} if _lnm == "welford" else {}
        xin = (x if x_ln is None else x_ln); NI = int(x.shape[-3]) if x.dim() == 3 else int(x.shape[0])
        NI = int(xin.shape[0])
        rngs = [(i0, min(NI, i0 + int(chunk_size))) for i0 in range(0, NI, int(chunk_size))]
        parts = []
        for (i0, i1) in rngs:                                                                  # pass 1: bias rows (q/k/v/g of the chunk discarded) — bias[h, j, k] needs every pair row j
            _o = prologue(module, xin[i0:i1], ending=False, ln_mode=_lnm, x_ln=(x_ln[i0:i1] if x_ln is not None else None), **_kw); parts.append(_o[4]); del _o
        bias = torch.cat(parts, dim=1).contiguous(); del parts
        b4 = bias.unsqueeze(0); scale = 1.0 / math.sqrt(module.mha.c_hidden)
        cch = getattr(module, "_fpf_cache", None)
        if cch is None:
            cch = {}; module._fpf_cache = cch
        ns = cch.setdefault("trunk2", {})
        wo16 = ns.get("wo16")
        if wo16 is None or wo16.device != z.device:
            wo16 = module.mha.linear_o.weight.detach().to(torch.bfloat16).contiguous(); ns["wo16"] = wo16
        cch.setdefault("wo16", wo16)
        for (i0, i1) in rngs:                                                                  # pass 2: per chunk prologue -> cuEq (== stock chunk_layer call) -> epilogue in place
            q, k, v, g, _b = prologue(module, xin[i0:i1], ending=False, ln_mode=_lnm, x_ln=(x_ln[i0:i1] if x_ln is not None else None), **_kw); del _b
            if att_lean is None:
                o_r = TL.cuequivariance_triangular_attn(q, k, v, b4, (_stock_true_mask(q, i1 - i0, k.shape[-2]) if _cueq_sm100f_exposed(k.shape[-2]) else None), scale)
                _XL_MEM["lean_cueq_chunks"] += 1
            else:                                                                          # ARM T: the block core's K2B on this row-chunk (rows = batch in K2B's grid: bitwise == the full-N call), full fp32 bias
                o_r = att_lean(*(t if t.dim() == 5 else t.unsqueeze(0) for t in (q, k, v)), b4, mask=None, scale=scale)
                _XL_MEM["lean_k2b_chunks"] += 1; _STATS["blk_att_k2b_routed"] = _STATS.get("blk_att_k2b_routed", 0) + 1; _STATS["blk_att_k2b_calls"] = _STATS["blk_att_k2b_routed"]
            o_r = o_r[0] if isinstance(o_r, (tuple, list)) else o_r
            if o_r.dim() == 5:
                assert o_r.shape[0] == 1, f"attention output leading dim {o_r.shape}"; o_r = o_r[0]
            assert o_r.dim() == 4 and o_r.shape[0] == (i1 - i0) and o_r.shape[1] == q.shape[-3], f"lean-prologue attention output layout {tuple(o_r.shape)} rows {i0}:{i1}"
            zr = z[:, i0:i1] if ending else z[i0:i1]
            epi_block_kernel(o_r, g, wo16, zr, ending=bool(ending), residual=True)
            del o_r, q, k, v, g
        del bias, b4
        _XL_MEM["lean_prologue_calls"] += 1; _STATS["blk2_tri_chunked_calls"] = _STATS.get("blk2_tri_chunked_calls", 0) + 1; _STATS["blk2_tri_chunks"] = _STATS.get("blk2_tri_chunks", 0) + len(rngs)
        return z
    q, k, v, g, bias = prologue(module, (x if x_ln is None else x_ln), ending=False, ln_mode=_lnm, x_ln=x_ln, **({"fma_flags": _BLK_LN_FMA} if _lnm == "welford" else {}))
    scale = 1.0 / math.sqrt(module.mha.c_hidden)
    if chunk_size is not None and not (_BLK_CHUNKED == "k2b" and att is not None):       # LARGE-N chunked regime: the core is cuEq per stock row-chunk (K2B is not called per chunk: label-safe, counted) unless PTX_BLK_CHUNKED=k2b (TIER-2 evidence arm: K2B core, one unchunked call)
        if att is not None: _STATS["blk_att_chunked_cueq_instead_of_k2b"] = _STATS.get("blk_att_chunked_cueq_instead_of_k2b", 0) + 1
        cch = getattr(module, "_fpf_cache", None)
        if cch is None:
            cch = {}; module._fpf_cache = cch
        ns = cch.setdefault("trunk2", {})
        wo16 = ns.get("wo16")
        if wo16 is None or wo16.device != z.device:
            wo16 = module.mha.linear_o.weight.detach().to(torch.bfloat16).contiguous(); ns["wo16"] = wo16
        cch.setdefault("wo16", wo16)
        NI = q.shape[0]
        b4 = bias.unsqueeze(0)
        if _BLK_CHUNKED == "unchunked":
            rngs = [(0, NI)]
        else:
            rngs = [(i0, min(NI, i0 + int(chunk_size))) for i0 in range(0, NI, int(chunk_size))]      # chunk_layer boundaries over the leading (row) axis of x, ragged last chunk
        for (i0, i1) in rngs:
            o_r = TL.cuequivariance_triangular_attn(q[i0:i1], k[i0:i1], v[i0:i1], b4, (_stock_true_mask(q[i0:i1], i1 - i0, k.shape[-2]) if _cueq_sm100f_exposed(k.shape[-2]) else None), scale)   # == stock Attention.forward per chunk: cuEq(q,k,v, biases[1].float(), all-True mask, scale) (NOMASK lever: None == all-True, tested)
            o_r = o_r[0] if isinstance(o_r, (tuple, list)) else o_r
            if o_r.dim() == 5:
                assert o_r.shape[0] == 1, f"attention output leading dim {o_r.shape}"; o_r = o_r[0]
            assert o_r.dim() == 4 and o_r.shape[0] == (i1 - i0) and o_r.shape[1] == q.shape[-3], f"chunked attention output layout {tuple(o_r.shape)} rows {i0}:{i1}"
            zr = z[:, i0:i1] if ending else z[i0:i1]                                                  # start: z rows r ; ending: x = z^T so x-rows r are z COLUMNS r (strided view; epilogue addresses z by strides)
            epi_block_kernel(o_r, g[i0:i1], wo16, zr, ending=bool(ending), residual=True)
            del o_r
        _STATS["blk2_tri_chunked_calls"] = _STATS.get("blk2_tri_chunked_calls", 0) + 1; _STATS["blk2_tri_chunks"] = _STATS.get("blk2_tri_chunks", 0) + len(rngs)
        return z
    if att is None:
        o = TL.cuequivariance_triangular_attn(q, k, v, bias.unsqueeze(0), (_stock_true_mask(q, q.shape[-4], k.shape[-2]) if _cueq_sm100f_exposed(k.shape[-2]) else None), scale)      # stock cuEq kernel (EXACT arm; sm100f guard)
    else:
        q5, k5, v5 = (t if t.dim() == 5 else t.unsqueeze(0) for t in (q, k, v))
        o = att(q5, k5, v5, bias.unsqueeze(0) if bias.dim() == 3 else bias, mask=None, scale=scale)   # K2B flash (TIER-2 arm), cuEq signature/layout
        _STATS["blk_att_k2b_routed"] = _STATS.get("blk_att_k2b_routed", 0) + 1; _STATS["blk_att_k2b_calls"] = _STATS["blk_att_k2b_routed"]   # ROUTED to the Tier-2 branch (blk_att_k2b_calls is an alias of the same count); SERVED split = fpf_smalln COUNTS att_k2b_calls / att_cueq_calls, printed beside at exit
    o = o[0] if isinstance(o, (tuple, list)) else o
    if o.dim() == 5:
        assert o.shape[0] == 1, f"attention output leading dim {o.shape}"; o = o[0]
    assert o.dim() == 4 and o.shape[0] == q.shape[-4] and o.shape[1] == q.shape[-3], f"attention output layout {tuple(o.shape)} vs q {tuple(q.shape)}"   # [I,H,J,D]
    # cache rule: never replace another package's module._fpf_cache dict — setdefault into it (namespaced key)
    cch = getattr(module, "_fpf_cache", None)
    if cch is None:
        cch = {}; module._fpf_cache = cch
    ns = cch.setdefault("trunk2", {})
    wo16 = ns.get("wo16")
    if wo16 is None or wo16.device != z.device:
        wo16 = module.mha.linear_o.weight.detach().to(torch.bfloat16).contiguous(); ns["wo16"] = wo16
    cch.setdefault("wo16", wo16)                                  # also satisfy fpf_triatt_epi's flat schema if its own fn runs on this module later
    epi_block_kernel(o, g, wo16, z, ending=bool(ending), residual=True)
    return z

_BLK2_TRI_OK = {"v": True}
def _blk2_triatt_or_stock(module, z, ending, epi_k, pro, gc, triangle_attention, inplace_safe, chunk_size):
    """Portability: run the tested prologue/cuEq/epilogue block statement; if the kernels are unavailable on this card/stack BEFORE anything was written into z
    (config lookup KeyError/NotImplementedError, triton compile/OutOfResources), fall back ONCE AND FOR ALL to the exact stock statement (z += tri_att(z) in the right frame,
    strided for the ending node) and print one line.  The prologue/epilogue write z only in the final epilogue kernel launch, so an exception raised by pick_config / JIT happens
    before z is touched (checked: prologue allocates q,k,v,g,bias then launches; epilogue asserts then launches)."""
    try:
        _triatt_block_pro_epi(module, z, ending, epi_k, pro, gc, chunk_size=(chunk_size if _BLK_CHUNKED not in ("", "0") else None))
        return
    except (KeyError, NotImplementedError, AssertionError, RuntimeError) as e:   # RuntimeError covers triton OutOfResources/CompilationError subclasses
        if _STATS.get("blk2_tri_calls", 0) > 0 and not isinstance(e, (KeyError, NotImplementedError)):
            raise                                                                  # kernels already ran fine in this process -> a later failure is a real error: fail loud
        _BLK2_TRI_OK["v"] = False
        _STATS["blk2_tri_stock_fallback"] = repr(e)[:200]
        _say_once(("blk2tri", gpu_arch(z.device)), f"BLK2: tri-att prologue/epilogue -> stock on gpu_arch={gpu_arch(z.device)} ({type(e).__name__}: {str(e)[:90]})")
    x = z.transpose(-2, -3) if ending else z
    u = module(x, mask=None, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
    z += (u.transpose(-2, -3) if ending else u)


def _apply_blk2():
    # GLUE V2 restructured kernels (bitwise-equal to the kernels they stand in for; default OFF). HAZARD 1 (install order): must patch
    # fpf_triatt_pro.prologue.triatt_prologue / fpf_triatt_epi.epilogue.triatt_epilogue / fpf_transition.transition._launch BEFORE this function binds them by from-import below.
    if os.environ.get("PTX_GLUE_V2", "0") not in ("", "0"):
        try:
            import fpf_glue_v2 as _GL2
            _GL2.install(verbose=True); _st = _GL2.stats() if hasattr(_GL2, "stats") else {}
            _STATS["applied"].append("GLUE_V2:" + ("on(" + ",".join(sorted(_st.get("patched", []) or [])) + ")" if (_st.get("installed") if isinstance(_st, dict) else True) else f"refused({_st.get('why','')})"))
            _STATS["glue_v2"] = _GL2._STATE.get("calls") if hasattr(_GL2, "_STATE") else _st      # live counters dict (prologue_v4 / epilogue_v3 / transition_ws / *_passthru)
        except Exception as _e:
            _STATS["applied"].append(f"GLUE_V2:unavailable({_e!r})"[:160]); print(f"[FPF] PTX_GLUE_V2 unavailable -> shipped kernels: {_e!r}", file=sys.stderr, flush=True)
    # MK-PF F1 = tri-att prologue with LayerNorm in registers (start: z rows; end: z^T VIEW by strides -> drops fast_layernorm + its transposed copy),
    # F1-padded for PAD8 statements, F3 = transition LN-in-registers (10.0 only). Cells keyed (cc|triton): 10.0|3.7 F1 both+F3 welford EXACT; 9.0|3.7 F1 both welford EXACT;
    # 9.0|3.3 F1 END two-pass TIER-2 (OPT-IN only; honours PTX_T_MIN_TOKENS via fpf_mkpf._n_gate_ok). Installed HERE = after fpf_glue_v2.install() and BEFORE the from-imports
    # below bind triatt_prologue into this function's closure, so fpf_mkpf's closure-rebinding fallback is never needed (stats()['rebound_closure_cells'] must stay 0/None).
    _mk_tokens = [x.strip() for x in os.environ.get("PTX_MK_PF", "").split(",") if x.strip()]
    _mk_for_mkpf = list(_mk_tokens)
    if _mk_for_mkpf and ",".join(_mk_for_mkpf) not in ("", "0"):
        try:
            import fpf_mkpf as _MK
            _mk_env_saved = os.environ.get("PTX_MK_PF"); os.environ["PTX_MK_PF"] = ",".join(_mk_for_mkpf)
            try:
                _mkst = _MK.install(verbose=True) or {}
            finally:
                os.environ["PTX_MK_PF"] = _mk_env_saved
            _mkst = _MK.stats() if hasattr(_MK, "stats") else _mkst
            _mkS = getattr(_MK, "_STATE", {}) or {}; _mkc = _mkS.get("cells") or {}
            _STATS["applied"].append("MK_PF:on(" + ",".join(sorted((_mkst.get("levers") or _mkst.get("patched") or []))) + f",arith={os.environ.get('FPF_MKPF_LN') or _mkc.get('ln_arith')},class={str(_mkc.get('class', '?')).split(' ')[0]},cells={_mkS.get('why')},rebound={_mkS.get('rebound_closure_cells')})")
            _STATS["mkpf"] = getattr(_MK, "_STATE", {}).get("calls")          # live counters: f1_end / f1_start / f1_padded / f1_passthru / f3 / f3_passthru / f1_gated_small
            print(f"[FPF] PTX_MK_PF={os.environ.get('PTX_MK_PF')} -> fpf_mkpf installed: {_mkst.get('levers') or _mkst.get('patched')} arith={_mkc.get('ln_arith')} class={_mkc.get('class')} cells={_mkS.get('why')} (EXACT only with welford cells; two-pass cells are TIER-2, opt-in, gated at PTX_T_MIN_TOKENS)", file=sys.stderr, flush=True)
        except Exception as _e:
            _STATS["applied"].append(f"MK_PF:refused({_e!r})"[:200]); print(f"[FPF] PTX_MK_PF refused -> default path unchanged: {_e!r}", file=sys.stderr, flush=True)
    # lever triatt_prologue_cuda (PTX_TRIATT_PROCUDA=1; third_party/protenix_fpf_triatt_procuda, sm_90a prebuilt; the kit README rows of cc 9.0 export the switch):
    # the MK-PF F1 tri-attention prologue (LayerNorm -> q|k|v|g + pair bias) as one CUDA kernel, patched onto fpf_mkpf's row-major entry and, when
    # triatt_headsplit_exact is installed, its head-major entry — after both (module attributes looked up at call time). install() appends exactly one
    # PROCUDA:on(...) marker and publishes its live tally as _STATS["procuda"]; it raises Refused by name (-> PROCUDA:unavailable: the mode refuses).
    if os.environ.get("PTX_TRIATT_PROCUDA", "0") not in ("", "0"):
        try:
            import protenix_fpf_triatt_procuda as _PC
            from protenix_fpf_triatt_procuda import binding as _PCB
            from protenix_opt.binary_sums import binary_refusal as _binary_refusal   # the shipped sm_90a library is held to the SHA256SUMS beside it before install() maps it; refused by name -> unavailable, as an absent one is
            _so = os.path.join(_PCB.PREBUILT_DIR, _PCB.SO_NAME)
            _held = _binary_refusal(_so) if os.path.isfile(_so) else None                  # an absent binary is install()'s own by-name refusal
            if _held is not None:
                raise RuntimeError(f"protenix_fpf_triatt_procuda: prebuilt binary {_PCB.SO_NAME} refused: {_held}")
            _PC.install(verbose=True)
        except Exception as _e:
            _STATS["applied"].append(f"PROCUDA:unavailable({_e!r})"[:200]); print(f"[FPF] PTX_TRIATT_PROCUDA unavailable: {_e!r}", file=sys.stderr, flush=True)
    import protenix.model.modules.pairformer as PF
    try:
        from fpf_triatt_epi.epilogue import triatt_epilogue as _epi_k, fn_block_residual as _epi_block
        from fpf_triatt_pro.prologue import triatt_prologue as _pro, get_cache as _gc, PINNED_CONFIG as _PC
    except Exception as e:
        _STATS["applied"].append(f"BLK2:unavailable({e!r})"); return
    _orig = PF.PairformerBlock.forward
    _VERIFIED = {(256, 256)}                                     # (C, H*D) cells known bitwise-equal on sm_90; the 'default' config must NOT run other cells
    def supported(module, z):
        try:
            if gpu_arch(z.device) not in _ARCH_VERIFIED_BLK2:
                _say_once(("blk2arch", gpu_arch(z.device)), f"BLK2: no cells for gpu_arch={gpu_arch(z.device)}: tri-att prologue/epilogue + fused transition -> stock path (exact); block path stays stock on this card")
                return False
            return (int(module.c_in), int(module.mha.no_heads * module.mha.c_hidden)) in _VERIFIED and z.shape[-1] == 256
        except Exception:
            return False
    def blk_forward(self, s, z, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        if (not inplace_safe) or triangle_attention != "cuequivariance" or pair_mask is not None or (chunk_size is not None and _BLK_CHUNKED in ("", "0")) or z.dtype != torch.bfloat16 or z.shape[-2] <= 16 or self.training or z.dim() != 3:
            _STATS["blk_fallback"] = _STATS.get("blk_fallback", 0) + 1
            return _orig(self, s, z, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        if not supported(self.tri_att_start, z):                 # template stack c=64 etc. -> whole stock block
            _STATS["blk_fallback_c"] = _STATS.get("blk_fallback_c", 0) + 1
            return _orig(self, s, z, pair_mask=pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)
        shape0 = tuple(z.shape)
        if getattr(self.tri_mul_out, "_deadskip", False): _STATS["blk_deadskip_stmts"] = _STATS.get("blk_deadskip_stmts", 0) + 1
        else: z = self.tri_mul_out(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
        if getattr(self.tri_mul_in, "_deadskip", False): _STATS["blk_deadskip_stmts"] = _STATS.get("blk_deadskip_stmts", 0) + 1
        else: z = self.tri_mul_in(z, mask=pair_mask, inplace_safe=inplace_safe, _add_with_inplace=True, triangle_multiplicative=triangle_multiplicative)
        ptr0 = z.data_ptr()
        if getattr(self.tri_att_start, "_deadskip", False): _STATS["blk_deadskip_stmts"] = _STATS.get("blk_deadskip_stmts", 0) + 1     # exact: stock adds an all-zero update
        elif _xl_lean(z): z += self.tri_att_start(z, mask=None, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size); _XL_MEM["lean_triatt_stmts"] += 1   # XL: stock chunked statement above PTX_FPF_CHUNK_TOK (exact; bounded workspace)
        elif _BLK2_TRI_OK.get("v", True) and not _triatt_below_floor(z): _blk2_triatt_or_stock(self.tri_att_start, z, False, _epi_k, _pro, _gc, triangle_attention, inplace_safe, chunk_size)
        else: z += self.tri_att_start(z, mask=None, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size)   # the stock tri-attention statement: no kernels for this arch in this process, or below the arch's CELLS.json blk2_triatt_min_tokens floor (named once, counted)
        if getattr(self.tri_att_end, "_deadskip", False): _STATS["blk_deadskip_stmts"] = _STATS.get("blk_deadskip_stmts", 0) + 1
        elif _xl_lean(z): z += self.tri_att_end(z.transpose(-2, -3), mask=None, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size).transpose(-2, -3); _XL_MEM["lean_triatt_stmts"] += 1
        elif _BLK2_TRI_OK.get("v", True) and not _triatt_below_floor(z): _blk2_triatt_or_stock(self.tri_att_end, z, True, _epi_k, _pro, _gc, triangle_attention, inplace_safe, chunk_size)
        else: z += self.tri_att_end(z.transpose(-2, -3), mask=None, triangle_attention=triangle_attention, inplace_safe=inplace_safe, chunk_size=chunk_size).transpose(-2, -3)
        assert tuple(z.shape) == shape0 and z.data_ptr() == ptr0, f"BLK2: z shape/aliasing changed across tri-att ({shape0} -> {tuple(z.shape)})"   # K2B shape contract: fail loud
        _STATS["blk2_tri_calls"] = _STATS.get("blk2_tri_calls", 0) + 2
        if getattr(self.pair_transition, "_deadskip", False):
            _STATS["blk_deadskip_stmts"] = _STATS.get("blk_deadskip_stmts", 0) + 1
        elif os.environ.get("PTX_BLK_TRANS", "fusion") == "fusion":
            z = _TRC.block_pair_transition(self.pair_transition, z)      # z += pair_transition(z): opt_core.kernels.transition by the mode's tier word (residual folded), else the module statement in place, by name
        else:
            z += self.pair_transition(z)
        _STATS["blk_calls"] = _STATS.get("blk_calls", 0) + 1
        if chunk_size is not None: _STATS["blk_calls_chunked"] = _STATS.get("blk_calls_chunked", 0) + 1
        assert tuple(z.shape) == shape0, f"BLK2: z shape changed across block ({shape0} -> {tuple(z.shape)})"
        if self.c_s > 0:
            s = s + self.attention_pair_bias(a=s, s=None, z=z)
            s = s + self.single_transition(s)
        return s, z
    attname = "cueq"
    sel = os.environ.get("PTX_BLK_ATT", "")
    if ":" in sel:                                                                 # generic TIER-2 provider "module:attr" (ARM T only; env.sh never sets this under ARM=E)
        # fn(q5, k5, v5, bias4, mask=None, scale=...) in the cuEq layout; served at N >= PTX_BLK_ATT_PROVIDER_MIN_TOKENS (default 512), K2B (if importable) or cuEq below;
        # printed once; label TIER2 unless the provider is separately tested EXACT. This is the hook NX-Blackwell's sm100f pad-copy arm plugs into today.
        try:
            import importlib
            _modp, _attr = sel.split(":", 1)
            _prov = getattr(importlib.import_module(_modp.strip()), _attr.strip())
            _pmin = int(os.environ.get("PTX_BLK_ATT_PROVIDER_MIN_TOKENS", "512") or 0)
            _below = None
            try:
                if not os.environ.get("PF_TRIATTN_TABLE"):
                    _kc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "fpf_triatt_k2b", "K2B_CELLS.json")
                    if os.path.exists(_kc): os.environ["PF_TRIATTN_TABLE"] = _kc
                import fpf_triatt_k2b as _K
                _below = _K.attn_k2b
            except Exception as _e:
                _below = None
            def _routed(q5, k5, v5, bias, mask=None, scale=None, _p=_prov, _b=_below, _pmin=_pmin):
                n = int(q5.shape[-2])
                if n >= _pmin:
                    _STATS["blk_att_provider_calls"] = _STATS.get("blk_att_provider_calls", 0) + 1
                    return _p(q5, k5, v5, bias, mask=mask, scale=scale)
                if _b is not None:
                    _STATS["blk_att_k2b_calls_below_provider_min"] = _STATS.get("blk_att_k2b_calls_below_provider_min", 0) + 1
                    return _b(q5, k5, v5, bias, mask=mask, scale=scale)
                import protenix.model.triangular.layers as TL
                q, k, v = (t[0] if t.dim() == 5 and t.shape[0] == 1 else t for t in (q5, k5, v5))
                return TL.cuequivariance_triangular_attn(q, k, v, bias if bias.dim() == 4 else bias.unsqueeze(0), None, scale)
            _routed.__name__ = f"provider[{sel}]"; _routed.__module__ = "ptx_trunk2_levers"
            _BLK_ATT["fn"] = _routed
            try:                                                                     # provider registry meta (padded layout) from <module>.PROVIDER / .Refused
                _pm = importlib.import_module(_modp.strip())
                _meta = dict(getattr(_pm, "PROVIDER", {}) or {})
                _BLK_ATT_META.update(kv_pad=int(_meta.get("kv_pad", 1) or 1), bias_pad_fill=_meta.get("bias_pad_fill"), wants_kv_len=bool(_meta.get("wants_kv_len", False)),
                                     name=str(_meta.get("name", sel)), label=str(_meta.get("label", "TIER2")), refused_exc=((getattr(_pm, "Refused"),) if hasattr(_pm, "Refused") else ()),
                                     raw=_prov, min_tokens=_pmin)
                if _BLK_PADDED > 1 and _BLK_ATT_META["kv_pad"] > 1:
                    # padded path calls the RAW provider (not _routed): min-token routing below the provider threshold is handled by giving _routed to the unpadded path only
                    print(f"[FPF] ARM=T padded layout ACTIVE: PTX_BLK_PADDED={_BLK_PADDED}, provider {_BLK_ATT_META['name']} kv_pad={_BLK_ATT_META['kv_pad']} bias_pad_fill={_BLK_ATT_META['bias_pad_fill']} (zeroed per-(module,P) buffers; strided-view epilogue; Tier-2)", flush=True)
                    attname += f",padded={_BLK_PADDED}"
                elif _BLK_PADDED > 1:
                    print(f"[FPF] ARM=T padded layout requested (PTX_BLK_PADDED={_BLK_PADDED}) but provider kv_pad={_BLK_ATT_META['kv_pad']} -> unpadded path", flush=True)
            except Exception as _e:
                _STATS["applied"].append(f"BLK_ATT:provider meta unavailable({_e!r})")
            attname = f"PROVIDER({sel},tier2,min_tokens={_pmin},below={'k2b' if _below is not None else 'cueq'})"
            print(f"[FPF] ARM=T attention provider: {sel} at N>={_pmin} (below: {'K2B' if _below is not None else 'cuEq'}); label TIER-2 unless separately tested", flush=True)
        except Exception as e:
            _STATS["applied"].append(f"BLK_ATT:provider {sel} unavailable({e!r}) -> cueq"); attname = "cueq"
    elif sel in ("k2b", "k2"):
        try:
            if not os.environ.get("PF_TRIATTN_TABLE"):                                   # bundled cc-keyed K2B cells (read by the package loader at import)
                _kc = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "third_party", "fpf_triatt_k2b", "K2B_CELLS.json")
                if os.path.exists(os.environ.get("FPF_K2B_CELLS_JSON", _kc)): os.environ["PF_TRIATTN_TABLE"] = os.environ.get("FPF_K2B_CELLS_JSON", _kc)
            import fpf_triatt_k2b as _K
            _BLK_ATT["fn"] = _K.attn_k2b if sel == "k2b" else _K.attn_k2; attname = sel.upper() + "(tier2)" + (f",min_tokens={_BLK_ATT_MIN_TOK}" if _BLK_ATT_MIN_TOK else ",min_tokens=0(off)")
            # K2B launch cells = ONE mechanism, the K2B package loader (keyed on compute capability first) reading the bundled
            # third_party/fpf_triatt_k2b/K2B_CELLS.json via PF_TRIATTN_TABLE (set below BEFORE the package import unless the user set one).
            # Every entry is output-bitwise-identical to the tested default (cc 10.0: MAXNREG=128, 80/80; cc 12.0: uncapped, 16/16). Entries whose status starts with
            # UNTESTED (cc 10.3 = B300 copy of the sm100 entry) are used and NAMED (an untested cell is environment uncertainty, printed once, never a reason to disengage).
            try:
                import triatt_k2b as _TK
                _eff = dict((_TK._CONFIG_TABLE.get(32) or [(None, {})])[-1][1]); _key = getattr(_TK, "_PF_TABLE_KEY", None)
                _src = f"K2B_CELLS[{_key}]" if _key else "package default"
                if _key and str(_key).upper().find("UNTESTED") >= 0:
                    _src += " (an UNTESTED entry: in use, named — its outputs are not checked on this compute capability)"
                    print(f"[FPF] ARM=T K2B launch cells: {_src}", flush=True)
                if os.environ.get("PTX_K2B_MAXNREG", ""):
                    _mx = int(os.environ["PTX_K2B_MAXNREG"])
                    for _D in (16, 32):
                        _TK._CONFIG_TABLE[_D] = [(ms, dict(c, MAXNREG=_mx)) for (ms, c) in _TK._CONFIG_TABLE[_D]]
                    _eff = dict(_TK._CONFIG_TABLE[32][-1][1]); _src += " + PTX_K2B_MAXNREG"
                _STATS["k2b_cells"] = {"key": _key, "source": _src, "effective_D32": _eff, "table": os.environ.get("PF_TRIATTN_TABLE", "")}
                print(f"[FPF] ARM=T K2B launch cells (via package loader v1.1.1): {_src}: D32 {_eff}", flush=True)
                attname += f",cells={'package' if _src.startswith('package') else _key},maxnreg={_eff.get('MAXNREG')}"
            except Exception as _e:
                _STATS["k2b_cells"] = {"error": repr(_e)[:160]}
            print((f"[FPF] ARM=T size gate owned by fpf_smalln: exact TriMul + cuEq attention below {os.environ.get('FPF_SMALLN_K2B_MIN_TOKENS')} tokens, K2B + triton TriMul at/above (PTX_T_MIN_TOKENS={os.environ.get('PTX_T_MIN_TOKENS')})" if os.environ.get("FPF_SMALLN", "0") not in ("", "0") else f"[FPF] ARM=T size gates (in-levers fallback, fpf_smalln absent): K2B min_tokens={_BLK_ATT_MIN_TOK}, partner-TriMul min_tokens={_T_MIN_TOK} (0=off; below the gate the block path is the exact E path)"), flush=True)
        except Exception as e:
            _STATS["applied"].append(f"BLK_ATT:{sel} unavailable({e!r}) -> cueq")
    PF.PairformerBlock.forward = blk_forward
    _STATS["applied"].append(f"BLK:2(pro+{attname}+epi" + "+transition_core" + (f",chunked={_BLK_CHUNKED}" if _BLK_CHUNKED not in ("", "0") else "") + ")")

_DXA_REPORT = None                                          # protenix_opt.apb_core.report once dit_attn_exact is applied (report() reads it live)
_TXA_REPORT = None                                          # fpf.triatt_exact.report once triatt_exact is applied (report() reads it live)

def apply_from_env():
    t1 = os.environ.get("PTX_T1_TRANS", "")
    if t1:
        _apply_t1(t1)
    trw = os.environ.get("PTX_TRANSITION", "")                     # levers transition_core_exact | transition_core (env.sh L8: the mode's tier word): every Transition call through
    if trw:                                                         # opt_core.kernels.transition; installed AFTER T1 so a call the provider hands back runs the forward found here (T1's, by name)
        _STATS["applied"].append(_TRC.apply(trw))
    t2 = os.environ.get("PTX_T2_NOCOPY", "")
    if t2:
        _apply_t2(t2)
    zt = os.environ.get("PTX_ZT", "")
    if zt:
        _apply_t2b(zt)
    t5 = os.environ.get("PTX_T5_EINSUM", "")
    if t5:
        _apply_t5(t5)
    og = os.environ.get("PTX_OG", "")
    if og:
        _apply_og(og)
    if os.environ.get("PTX_BLK", "") == "1":
        _apply_blk()
    elif os.environ.get("PTX_BLK", "") == "2":
        _apply_blk2()
    if os.environ.get("PTX_DEADSKIP", ""):
        _apply_deadskip(os.environ["PTX_DEADSKIP"])
    if os.environ.get("PTX_PWA_ZCACHE", "") == "1":
        try:
            from ptx_msa_adapt import pwa_zcache as _PZ
            _STATS["applied"].append("PWAZ:" + str(_PZ.install()))
        except Exception as e:
            _STATS["applied"].append(f"PWAZ:unavailable({e!r})")
    if os.environ.get("PTX_NOMASK", "") == "1":
        _apply_nomask()
    # lever ln_core: the model's standalone LayerNorm calls bound to opt_core.kernels.ln by TIER WORD ($PTX_LN_TIER = fast | big; exact = the
    # library op fast_layernorm by name, not installed). A carried row the word resolves to is executed by the provider; the stock row / uncovered widths /
    # non-bf16o calls run the module's stream-correct fast_layernorm BY NAME (src/protenix_ptx_ln_core.py). Cannot engage -> LNCORE:unavailable(...) (refused by name).
    if os.environ.get("PTX_LN_TIER", ""):
        try:
            import protenix_ptx_ln_core as _lnc
            _st = _lnc.install()
            _STATS["applied"].append(f"LNCORE:on(word={_st['word']},core={_st['opt_core']})")
        except Exception as _e:
            _STATS["applied"].append(f"LNCORE:unavailable({_e})"[:200]); print(f"[FPF] LNCORE unavailable -> every LayerNorm on the module's statement by name: {_e}", file=sys.stderr, flush=True)
    # PAD8-EXACT: stock cuEq tri-attention on ceil8-padded q/k/v views + -1e9 bias columns runs the SAME sm80 kernel and is BITWISE == the
    # stock call (probed at first use), at lower cost per call. Wired through the padded core (shared zeroed per-P buffers); ARM E or T;
    # gates: cc major in (8, 9) only (cc 10.x excluded: aligned unmasked calls reach cudnn_sm100 there), N % 8 != 0 only (else inert = unpadded path), N >= PTX_E_PAD8_MIN_TOKENS (default 384).
    # DEFAULT OFF until CLI DET all-files E+pad8 == stock; label printed as 'EXACT-candidate' until then.
    if os.environ.get("PTX_E_PAD8", "") not in ("", "0"):
        try:
            _cc = torch.cuda.get_device_capability() if torch.cuda.is_available() else (0, 0)
            if _cc[0] not in (8, 9):
                _STATS["applied"].append(f"E_PAD8:refused(cc={_cc})"); print(f"[FPF] PTX_E_PAD8 refused: cc={_cc} not in (8.x, 9.x) -> unpadded exact path", file=sys.stderr, flush=True)
            elif _BLK_ATT_META.get("raw") is not None or _BLK_ATT.get("fn") is not None or os.environ.get("PTX_BLK_ATT", ""):
                _STATS["applied"].append("E_PAD8:refused(PTX_BLK_ATT set: Tier-2 attention owns the slot)"); print("[FPF] PTX_E_PAD8 refused: PTX_BLK_ATT (K2B/provider) is configured — PAD8-EXACT is an exact-attention (cuEq) lever", file=sys.stderr, flush=True)
            else:
                try:
                    import fpf_cueq_pad8exact as _P8                 # provider (cc gate {8,9}, kv_len>=104, first-call bitwise probe per class / per N below 256)
                except ImportError:
                    import fpf_pad8exact as _P8                      # fallback module without the first-call probe
                global _BLK_PADDED
                _BLK_PADDED = 8
                _pmin = int(os.environ.get("PTX_E_PAD8_MIN_TOKENS", "256") or 0)
                _BLK_ATT["fn"] = _pad8_marker                            # non-None so the core evaluates the provider branch; the unpadded path maps the marker back to the stock cuEq call
                _BLK_ATT_META.update(kv_pad=8, bias_pad_fill=-1e9, wants_kv_len=True, name="pad8exact(stock cuEq on padded views)", label="EXACT-candidate", refused_exc=((_P8.Refused,)), raw=_P8.attn, min_tokens=max(_pmin, int((getattr(_P8, "PROVIDER", {}) or {}).get("min_tokens", 0) or 0)), only_unaligned=True, provider_module=_P8.__name__)
                _STATS["applied"].append(f"E_PAD8:on({_P8.__name__},min_tokens={_BLK_ATT_META['min_tokens']},cc={_cc[0]}.{_cc[1]})"); _STATS["pad8"] = getattr(_P8, "STATS", None) or getattr(_P8, "_STATE", None) or {}
                print(f"[FPF] PTX_E_PAD8 ON via {_P8.__name__}: stock cuEq on ceil8-padded views for N%8!=0, N>={_BLK_ATT_META['min_tokens']} (cc {_cc[0]}.{_cc[1]}); EXACT-candidate until CLI DET certifies; counters _STATS['pad8'] + provider report()", file=sys.stderr, flush=True)
        except Exception as _e:
            _STATS["applied"].append(f"E_PAD8:unavailable({_e!r})"[:160]); print(f"[FPF] PTX_E_PAD8 unavailable -> unpadded path: {_e!r}", file=sys.stderr, flush=True)
    # levers keep_pool / summary_hostidx (src/protenix_ptx_keep_pool.py, src/protenix_ptx_summary_host.py): pure python, no card envelope, EXACT-BITWISE.
    # Installed after every FPF lever above and before the sampler hook (sitecustomize imports fpf_clisampler after apply_from_env); inert when the switch is absent/0;
    # any other switch value raises by name (enabled()); a summary_hostidx install that finds the pinned stock surface drifted is SUMHOST:unavailable(...) -> the mode refuses by name.
    import protenix_ptx_keep_pool as _kp, protenix_ptx_summary_host as _sh
    if _kp.enabled():
        _STATS["applied"].append(_kp.MARK + ":" + _kp.install())            # -> "KEEP_POOL:on(skip=3of5,over=empty_cache)"
    if _sh.enabled():
        try:
            _STATS["applied"].append(_sh.MARK + ":" + _sh.install())        # -> "SUMHOST:on(inference-call;d2h=2,h2d=2/item)"
        except Exception as _e:                                              # pinned stock surface drifted: named, the mode refuses
            _STATS["applied"].append(f"{_sh.MARK}:unavailable({_e!r})"[:200]); print(f"[FPF] PTX_SUMMARY_HOST unavailable: {_e!r}", file=sys.stderr, flush=True)
    # lever dit_attn_exact (EXACT-BITWISE; protenix_opt.apb_core -> the shared core's opt_core.kernels.apb by the tier word `exact`):
    # protenix.model.modules.primitives._attention routed per call class; the provider's dit_exact row (prebuilt sm_90 kernel, bit-identical to the
    # memory-efficient SDPA kernel, vouched per stack in the core's cell table) serves the classes vouched on this stack, every other class keeps the statement
    # by name. The exact composition of the cc-9.0 rows exports PTX_DIT_ATTN_EXACT=1 (modes.README_ROWS); inert when the switch is absent/0 (no .so load, no
    # CUDA work). apply() = the row's manifest / sha / version checks + the per-process 3-case load-time bit check against F.scaled_dot_product_attention
    # (mandatory, never cached across processes) and raises RuntimeError("dit_attn_exact: <reason>") by name -> DITATTN:unavailable(...) -> the kit refuses the mode by name.
    if os.environ.get("PTX_DIT_ATTN_EXACT", "0") not in ("", "0"):
        try:
            from protenix_opt import apb_core as _dx
            _STATS["applied"].append(_dx.apply())                            # -> "DITATTN:on(opt_core.kernels.apb <core> word=exact: dit_exact v4 torch2.13.0-cu130-sm90 serves ...; loadcheck=3cases-bit-equal; ...)"
            global _DXA_REPORT
            _DXA_REPORT = _dx.report                                         # live report() for the trunk record at exit (calls / routes / installed / loadcheck)
            print(f"[FPF] dit_attn_exact: {_STATS['applied'][-1]}", file=sys.stderr, flush=True)
        except Exception as _e:
            _STATS["applied"].append(f"DITATTN:unavailable({_e!r})"[:200]); print(f"[FPF] PTX_DIT_ATTN_EXACT unavailable: {_e!r}", file=sys.stderr, flush=True)

    # lever triatt_exact (EXACT-BITWISE; src/fpf/triatt_exact.py -> the shared core's opt_core.kernels.triattn by the tier word `exact` alone): the pair stacks'
    # triangle attention at the library sites (protenix.model.triangular.layers.cuequivariance_triangular_attn; fpf_cueq_pad8exact's own handle when that provider is
    # loaded above) asks the provider per call with the site's callable as the stock op: its row triattn_exact (bit-identical to the library op on its proven cells)
    # serves where the core's cell table vouches for this stack, the library op BY NAME everywhere else; a refusal by name takes the site's callable for that call,
    # counted. The exact composition of the cc-9.0 and cc-8.0 rows exports PTX_TRIATT_EXACT=1 (modes.README_ROWS); inert when the switch is absent/0. A bind
    # failure (opt_core below the lever's floor, the engine module absent, an unknown switch value) raises by name -> TRIATT_EXACT:unavailable(...) -> the kit
    # refuses the mode by name.
    if os.environ.get("PTX_TRIATT_EXACT", "0") not in ("", "0"):
        try:
            from fpf import triatt_exact as _tx
            _STATS["applied"].append(_tx.apply())                            # -> "TRIATT_EXACT:on(opt_core.kernels.triattn <core> word=exact bind=tier:exact sites=tl[+pad8] cc=.. select=..; ...)"
            global _TXA_REPORT
            _TXA_REPORT = _tx.report                                         # live report() for the trunk record at exit (sites / calls / provider / refused / member counts)
            print(f"[FPF] triatt_exact: {_STATS['applied'][-1]}", file=sys.stderr, flush=True)
        except Exception as _e:  # noqa: BLE001
            _STATS["applied"].append(f"TRIATT_EXACT:unavailable({_e!r})"[:200]); print(f"[FPF] PTX_TRIATT_EXACT unavailable: {_e!r}", file=sys.stderr, flush=True)

    return report()


def triatt_pro_exactln_c256(module, x, mask=None, chunk_size=None, triangle_attention="torch", inplace_safe=False):
    """FPF_OPS-safe wrapper around Fusion prologue v0 fn_exactln: only (c_in=256, H*D=256) modules; everything else (template c=64 stack) -> stock forward."""
    import fpf
    if int(module.c_in) == 256 and int(module.mha.no_heads * module.mha.c_hidden) == 256 and x.shape[-1] == 256:
        from fpf_triatt_pro.triatt import fn_exactln
        _STATS["pro_calls"] = _STATS.get("pro_calls", 0) + 1
        return fn_exactln(module, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)
    _STATS["pro_fallback_c"] = _STATS.get("pro_fallback_c", 0) + 1
    return fpf.original("triatt")(module, x, mask=mask, chunk_size=chunk_size, triangle_attention=triangle_attention, inplace_safe=inplace_safe)


# =====================================================================================  DEADSKIP (Numerics lead; EXACT by construction): modules whose parameters are ~1e-37 output exactly 0.
# PTX_DEADSKIP=<manifest.json> : after InferenceRunner.init_model loads the checkpoint, deadskip.install(model, manifest, ckpt) wraps the checked-dead instances (sha + live re-probe); each is marked
# with ._deadskip=True so BLK2 skips the statement, and TriangleMultiplication instances get an inplace-safe forward (stock inplace path does `z = tri_mul(z, _add_with_inplace=True)`,
# which must return z itself, not zeros).
def _apply_deadskip(manifest_path):
    import json
    try:
        import deadskip as _DS
        import runner.inference as RI
    except Exception as e:
        _STATS["applied"].append(f"DEADSKIP:unavailable({e!r})"); return
    man = json.load(open(manifest_path))
    _orig_load = RI.InferenceRunner.load_checkpoint          # hook AFTER the weights are loaded (init_model builds a randomly initialised model; the absmax guard refuses everything there)
    def init_model(self):
        _orig_load(self)
        ckpt = os.path.join(self.configs.load_checkpoint_dir, f"{self.configs.model_name}.pt")
        if not os.path.exists(ckpt):
            ckpt = man.get("checkpoint_path")
        skipped = _DS.install(self.model, man, ckpt, reprobe=True)
        mods = dict(self.model.named_modules())
        n_tm = 0
        for pth in skipped:
            mod = mods[pth]; mod._deadskip = True
            if type(mod).__name__.startswith("TriangleMultiplication") or "tri_mul" in pth.split(".")[-1]:   # TriangleMultiplicationOutgoing/Incoming: stock inplace path returns z itself (+0), NOT the update
                def _tm_fwd(z, mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative="torch", **kw):
                    if inplace_safe and _add_with_inplace:
                        return z                                            # stock: z updated in place by +0 and returned
                    dt = torch.get_autocast_gpu_dtype() if torch.is_autocast_enabled() else z.dtype
                    return torch.zeros(z.shape, device=z.device, dtype=dt)  # stock non-inplace: returns the update (exactly 0)
                mod.forward = _tm_fwd; n_tm += 1
        expected = [m["path"] for m in man["modules"] if m.get("probe_pass")]
        on_stock = sorted(set(expected) - set(skipped))
        _rep = getattr(_DS.install, "last_report", None) or []
        reasons = {r["path"]: r.get("reason", "?").split(" ")[0] for r in _rep if r.get("action") == "STOCK"} or {p: "?" for p in on_stock}
        msg = (f"deadskip: {len(skipped)}/{len(expected)} modules skipped (ckpt sha ok; per-module live re-probe exactly 0 on this process{'; trimul inplace-safe: %d' % n_tm}); "
               f"on stock: {reasons if on_stock else 'none'}")
        _STATS["deadskip"] = {"n": len(skipped), "n_expected": len(expected), "on_stock": on_stock, "n_trimul_inplace_safe": n_tm, "manifest_n": len(man["modules"]), "msg": msg,
                              "gpu_arch": gpu_arch()}
        print(msg, flush=True)                                  # per-module gate (deadskip.install's design); a module whose live probe is not exactly 0 on this card stays on stock, the rest remain individually exact
    RI.InferenceRunner.load_checkpoint = init_model
    _STATS["applied"].append("DEADSKIP:hooked(load_checkpoint)")


# ---- BlockFuse XL lean-prologue add-on (vendored third_party/blockfuse_addon). Engages ONLY inside the XL lean-prologue
# branch (PTX_FPF_CHUNK_MODE=prologue and N_token > PTX_FPF_CHUNK_TOK) and ONLY when fpf_mkpf resolves an EXACT welford F1 cell for this (cc, triton);
# otherwise it refuses (logged, non-fatal) and the bundle's own lean-prologue statement runs. PTX_BLOCKFUSE_XL=0 disables. Counters: report()["blockfuse"].
_BLOCKFUSE_STATE = {"requested": os.environ.get("PTX_BLOCKFUSE_XL", "0") == "1", "installed": False, "why": "not requested"}
def _blockfuse_hook():
    if not _BLOCKFUSE_STATE["requested"]:
        return
    if _XL_CHUNK_MODE != "prologue" or not _XL_CHUNK_TOK:
        _BLOCKFUSE_STATE["why"] = f"inert: chunk_mode={_XL_CHUNK_MODE} chunk_tok={_XL_CHUNK_TOK}"; return
    try:
        import blockfuse_xl as _bf
        _bf.install(L=sys.modules[__name__], verbose=True)
        _BLOCKFUSE_STATE.update(installed=True, why=_bf._STATE.get("why", "installed"), stats=_bf.stats)
    except Exception as e:   # refusal (no EXACT welford F1 cell on this stack) or import error -> bundle path, loudly
        _BLOCKFUSE_STATE["why"] = f"refused: {e!r}"[:400]
        print(f"[FPF] blockfuse_xl NOT installed ({_BLOCKFUSE_STATE['why']}); XL lean-prologue branch uses the bundle statement", file=sys.stderr, flush=True)
_blockfuse_hook()

def _blockfuse_report():
    st = dict(_BLOCKFUSE_STATE); f = st.pop("stats", None)
    if f is not None:
        try: st["stats"] = f()
        except Exception as e: st["stats"] = repr(e)
    return st
import atexit as _bf_atexit
_bf_atexit.register(lambda: print(f"[FPF] BLOCKFUSE {_blockfuse_report()}", file=sys.stderr, flush=True))
