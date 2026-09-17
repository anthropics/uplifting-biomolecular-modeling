"""fpf_transition/transition.py — registry op `transition`: the fused pair transition (the stock primitives.Transition.forward, eval path).

Measured on torch 2.7.1 + triton 3.3.1, H100 (sm_90); nothing Hopper-specific is used (portable to sm_80).

Exports (all with the stock signature  fn(module, x) ; module = the stock Transition instance, never mutated):
  fn(module, x)            EXACT path.  y = module.layernorm1(x2d)  [the STOCK fast_layernorm CUDA kernel call, unchanged -> bit-exact LN]
                           then ONE Triton kernel for everything after the LN:  a = y@Wa^T, b = y@Wb^T (bf16 MMA, fp32 accumulate over K=c in ONE
                           chain -> rounded to bf16 == cuBLAS bf16 GEMM output bit pattern for K<=256 on H100), h = bf16(fp32(b16) * fp32(bf16(silu_fp32(a16))))
                           (stock rounding points: F.silu(a, inplace=True) rounds to bf16, then b *= a rounds to bf16), out = sum over ascending hidden chunks of
                           h_chunk @ Wo_chunk^T accumulated in fp32 (== cuBLAS K-order for the out GEMM on this stack), stored bf16.  Returns the UPDATE (caller adds).
                           Dispatch on module.c_in / hidden: shape classes listed in PINNED_CONFIGS (and enabled in FUSED_C_IN) are fused; anything else (fp32 inputs,
                           CPU, other widths, training mode) runs the stock body verbatim.
  fn_residual(module, x)   returns x + update with the residual folded into the kernel epilogue exactly like the caller's `z += transition(z)` under autocast:
                           out = bf16( fp32(x) + fp32( bf16(acc) ) )   (update rounded to bf16 FIRST, then fp32 add, one final rounding).  For the pairformer_block composition.
  fn_lnfused(module, x)    OPTIONAL Tier-2 variant: LayerNorm computed in the kernel prologue (fp32 statistics; two-pass mean / sum of squared deviations on the register tile,
                           IEEE sqrt+div instead of fast-math rsqrt) -> removes the LN write + re-read.  NOT bit-exact vs stock's fast_layernorm (different reduction tree and
                           rsqrt implementation) => label TIER2 a priori; measured.  fn stays the path.
  fn_stockbody(module, x)  the stock eval body re-typed here (used as fallback and as an in-file reference).

Fixed-config policy: tile configs come from the literal PINNED_CONFIGS table below keyed by (c_in, hidden, gpu_class); selection is a pure function of
(shape, dtype, device class).  No autotune, no atomics, no split-K: every output tile is produced by exactly one program with a fixed, ascending K / hidden-chunk order, so the
result is independent of grid scheduling, co-tenancy, batch size (rows are independent; BM tiles start at row 0 of the flattened [M, c] view) and CUDA-graph replay.
CANDIDATE_CONFIGS_UNVERIFIED is never read by dispatch (pick_config consults PINNED_CONFIGS only); cells listed there run the stock body.
Env overrides exist ONLY for offline sweeps (FPF_TRANSITION_CFG="BM,BH,WARPS,STAGES[,IL]"); a run must not set them (describe() reports it; timing run driver asserts).
"""
from __future__ import annotations
import hashlib, json, os
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    try:
        from triton.language.extra import libdevice as _ld
    except Exception:  # pragma: no cover
        try:
            from triton.language.extra.cuda import libdevice as _ld
        except Exception:
            _ld = None
    _HAS_TRITON = True
except Exception:  # pragma: no cover
    triton = None; tl = None; _ld = None; _HAS_TRITON = False
if _HAS_TRITON and _ld is None:     # the SiLU's exp: libdevice expf (== ATen's) when importable; else tl.exp — said ONCE, never silent (describe()['libdevice_exp'] records it too)
    import sys as _sys
    print("[fpf_transition] triton libdevice not importable: the fused SiLU uses tl.exp (libdevice_exp=False; tolerance-class rounding, not the ATen expf bits)", file=_sys.stderr, flush=True)

__version__ = "0.2.1"

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
# PINNED CONFIG TABLE (the table).  key: (c_in, hidden = n*c_in, gpu_class) -> dict(BM, BH, num_warps, num_stages, IL)
#   IL = 1: Wa/Wb rows interleaved so that ONE MMA of width 2*BH yields (a_j, b_j) column pairs (split in registers); IL = 0: two MMAs of width BH.
#   gpu_class: "sm90" (H100/H200), "sm80" (A100 class; less smem -> smaller tiles).  sm90 c=256 was re-swept from a 2-stage IL=0 seed cell (bit-exact at all 4 sizes).
#   smem per CTA (bf16): y tile BM*C*2 + weight tiles (2*C*BH*2 + BH*C*2) * num_stages + h tile BM*BH*2 (+ layout-conversion scratch).
PINNED_CONFIGS = {
    # ONLY cells checked bit-exact vs stock on the REAL stock activation dumps (H100 sm90, torch 2.7.1/cu126, triton 3.3.1) are listed; every other (c_in, hidden, gpu) -> stock body fallback.
    (256, 1024, "sm90"): {"BM": 128, "BH": 32, "num_warps": 8, "num_stages": 3, "IL": 0},   # pair transition (pairformer + MSA-module pair stack): bit-exact 8/8 cells N in {356,546,705,813} x {pf_c1_b0, pf_c10_b47} (r2); -7% vs seed cfg (stages 2)
    (128, 512, "sm90"):  {"BM": 128, "BH": 64, "num_warps": 8, "num_stages": 2, "IL": 0},   # MSA transition_m (c_m=128, n=4): bit-exact 4/4 on the real msa_b0 'm' dumps [n_msa, N, 128], N in {356,546,705,813} (r2)
    (384, 1536, "sm90"): {"BM": 16, "BH": 64, "num_warps": 4, "num_stages": 2, "IL": 0},    # single transition (c_s=384, K zero-padded to 512 in-register, M = N rows): bit-exact 8/8 cells (r2); speed parity with stock (0.19 vs 0.20 ms)
    (128, 512, "sm80"):  {"BM": 64, "BH": 32, "num_warps": 4, "num_stages": 2, "IL": 1},    # A100 (sm_80, 163 KB smem; A100-SXM4/PCIe-80GB, torch 2.7.1/cu126, triton 3.3.1): bit-exact vs the cuBLAS post-LN chain at M = N^2 rows, N in {200,384,705} (+ residual form), all 24 W=4 tiles bit-exact; x2.1-2.3 vs the stock chain (synthetic LN-output rows)
}
# NOT in the table (=> stock fallback, by rule 'checked cells only'): template-embedder pair transition (c=64, n=2, hidden 128; 80 block calls/trunk, tiny), c=128 n=2, sm80 c=256 / c=384 (below),
# fp32 inputs (OpenFold3-class TF32 trunk), training mode.  Candidate configs for those cells (from the seed tile sweep) are kept in CANDIDATE_CONFIGS_UNVERIFIED, unchecked.
# NOTE: CANDIDATE_CONFIGS_UNVERIFIED is never read by dispatch (pick_config consults PINNED_CONFIGS only); cells listed there run the stock body.
CANDIDATE_CONFIGS_UNVERIFIED = {
    (64, 128, "sm90"):   {"BM": 128, "BH": 128, "num_warps": 8, "num_stages": 1, "IL": 0},
    (64, 256, "sm90"):   {"BM": 128, "BH": 128, "num_warps": 8, "num_stages": 1, "IL": 0},
    (128, 256, "sm90"):  {"BM": 128, "BH": 64, "num_warps": 8, "num_stages": 2, "IL": 0},
    (256, 1024, "sm80"): {"BM": 64, "BH": 32, "num_warps": 4, "num_stages": 1, "IL": 1},   # A100: bit-exact vs the cuBLAS chain (14/14 fitting tiles, N in {200,384,546}) but x0.81-1.09 = parity-to-slower -> not pinned (the kit decides per card)
    (384, 1536, "sm80"): {"BM": 16, "BH": 32, "num_warps": 4, "num_stages": 2, "IL": 0},    # A100: NO tile is bit-exact vs the cuBLAS chain at M = N rows (max|d| 1 bf16 ulp, 0/8 tiles) -> a tier-2 cell only (x1.19); the exact line keeps the stock body on sm_80
}
# Cell decision (cell_verdict, the ONE statement): a PINNED cell (incl. cells a kit merges into PINNED_CONFIGS at run time from its cells file) is fused; a MEASURED_OFF cell — the candidate cells
# and the cells the measured compositions meet with this kernel OFF (c_in=128 n=4 and the c_s=384 single transition on sm100 / sm103) — runs the stock body BY NAME
# ('no-config-<c>-<hidden>+off(not-measured)'); any OTHER (c_in, hidden, gpu) cell is UNKNOWN and is fused with the SAFE transition settings of its capability
# (opt_core.kernels.safe_settings 'pair_fused:transition': 16-row single-stage tiles), engaged and named ONCE ('[opt_core/fpf_transition] safe settings served (no_cell:<cell>, ..)').
MEASURED_OFF = set(CANDIDATE_CONFIGS_UNVERIFIED) | {(128, 512, "sm100"), (128, 512, "sm103"), (384, 1536, "sm100"), (384, 1536, "sm103")}
SAFE_LEVER = "pair_fused:transition"                    # the safe-settings rows whose settings this kernel's cfg reads (BM, BH, num_warps, num_stages, IL)
from opt_core.kernels import safe_settings as _safe, cell_words as _cw      # noqa: E402 — the core's one cell vocabulary + safe-settings rows (stdlib at import)
_NET = _safe.SafeNet("fpf_transition")                  # names the SAFE engagement once per process (describe()['settings'])
# shape classes for which the fused kernel is the path: c_in -> bool.
FUSED_C_IN = {128: True, 256: True, 384: os.environ.get("FPF_TRANSITION_C384", "1") == "1"}   # c_in=384: padded-512 single chain bit-exact vs cuBLAS 8/8 cells, speed parity -> ON; FPF_TRANSITION_C384=0 pins it to the stock body


def config_table_json() -> str:
    return json.dumps({f"{k[0]},{k[1]},{k[2]}": v for k, v in sorted(PINNED_CONFIGS.items())}, sort_keys=True)


def config_table_sha256() -> str:
    return hashlib.sha256(config_table_json().encode()).hexdigest()


def gpu_class(device) -> str:
    major, _minor = torch.cuda.get_device_capability(device)
    if (major, _minor) == (9, 0):
        return "sm90"
    if major == 8:
        return "sm80"
    return f"sm{major}{_minor}"                                        # exact arch key; sm100 / sm103 / sm120 cells come from a kit's cells file merged into PINNED_CONFIGS


def _cc(device) -> str:
    return "%d.%d" % tuple(torch.cuda.get_device_capability(device))


def cell_verdict(c_in: int, hidden: int, device=None, *, gpu: str = None, cc: str = None):
    """The ONE cell decision -> (kind, cfg, word) (opt_core.kernels.cell_words.decide): 'pinned' (PINNED_CONFIGS), 'off' (MEASURED_OFF: 'no-config-<c>-<hidden>+off(not-measured)'),
    'safe' (an UNKNOWN cell: the capability's SAFE settings, word 'no_cell:<c>-<hidden>-<gpu>'), 'none' (nothing admits it: 'no-config-<c>-<hidden>[+no_safe(<why>)]').
    ``gpu`` / ``cc`` name the card for a decision off the device (tests)."""
    gpu = gpu if gpu is not None else gpu_class(device)
    cc = cc if cc is not None else _cc(device)
    key = (int(c_in), int(hidden), gpu)
    return _cw.decide(pinned=PINNED_CONFIGS.get(key), measured_off=key in MEASURED_OFF, miss_word=f"no-config-{int(c_in)}-{int(hidden)}", safe_lever=SAFE_LEVER, cc=cc,
                      dims={"c_z": int(c_in), "n_hidden": int(hidden)}, shape_word="%d-%d-%s" % key)


def pick_config(c_in: int, hidden: int, device, **kw):
    """Pure function of (shape class, device class): the PINNED cell, or the capability's SAFE settings for an UNKNOWN cell (engaged and named ONCE), or None (a MEASURED_OFF
    cell / nothing admits it: the stock body, named by _eligible).  Env FPF_TRANSITION_CFG is for OFFLINE sweeps only (never set in a run)."""
    ov = os.environ.get("FPF_TRANSITION_CFG", "").strip()
    if ov:
        vals = [int(v) for v in ov.split(",")]
        BM, BH, W, S = vals[:4]; IL = vals[4] if len(vals) > 4 else 0
        return {"BM": BM, "BH": BH, "num_warps": W, "num_stages": S, "IL": IL, "override": True}
    kind, cfg, word = cell_verdict(c_in, hidden, device, **kw)
    if kind == "safe":
        _NET.engage(word, _safe.where_word(kw.get("cc") or _cc(device), _safe.triton_mm()))
    return cfg


STATS = {"fused_calls": 0, "fallback_calls": 0, "fallback_reasons": {}}


def _note_fallback(why: str):
    STATS["fallback_calls"] += 1
    STATS["fallback_reasons"][why] = STATS["fallback_reasons"].get(why, 0) + 1


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
if _HAS_TRITON:
    _USE_LD = tl.constexpr(_ld is not None)

    @triton.jit
    def _silu_mul_bf16(a16, b16):
        # stock: a = F.silu(a, inplace=True) on a bf16 tensor -> fp32 opmath x/(1+exp(-x)) rounded to bf16 ;  b *= a -> fp32 multiply of two bf16 values rounded to bf16
        af = a16.to(tl.float32)
        if _USE_LD:
            e = _ld.exp(-af)
        else:
            e = tl.exp(-af)
        sv = (af / (1.0 + e)).to(tl.bfloat16).to(tl.float32)
        return (b16.to(tl.float32) * sv).to(tl.bfloat16)

    @triton.jit
    def _silu_mul_liger(a16, b16):
        # the Liger-Kernel SiLU*b form (LigerSiLUMulFunction on the two bf16 Linear outputs): c = silu(fp32(a)) * b with silu(x) = x * tl.sigmoid(x) in fp32,
        # the bf16 b promoted to fp32 by the product, ONE rounding to bf16 at the store -- the same Triton statements as that kernel, so the same lowering
        af = a16.to(tl.float32)
        return ((af * tl.sigmoid(af)) * b16.to(tl.float32)).to(tl.bfloat16)

    @triton.jit
    def _fused_transition_kernel(Y, WAB, WO, OUT, RES, LNW, LNB, M, s_ym, s_wab, s_wo, s_om, s_rm, eps,
                                 C: tl.constexpr, CP: tl.constexpr, NH: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr,
                                 IL: tl.constexpr, HAS_RES: tl.constexpr, LN_MODE: tl.constexpr, SILU: tl.constexpr = 0):
        """One program = BM rows of the flattened [M, C] input; everything after (or, LN_MODE=1, including) the LayerNorm stays on-chip.
        Y   [M, C]  bf16 : LN output (LN_MODE=0) or raw x (LN_MODE=1)
        WAB [2*NH, C] bf16 : IL=0 -> rows 0..NH-1 = Wa, rows NH.. = Wb ; IL=1 -> row 2i = Wa[i], row 2i+1 = Wb[i]
        WO  [C, NH] bf16 ; OUT [M, C] bf16 ; RES [M, C] bf16 (read only if HAS_RES) ; LNW/LNB [C] bf16 (read only if LN_MODE=1)
        CP = C padded to a power of two (== C except c_in=384 -> 512; padded K columns are exact zeros on both operands and contribute +0.0).
        SILU = 0: h = bf16(bf16(silu(a)) * b) (F.silu then the product, two roundings: the plain statement); 1: h = bf16(silu(fp32 a) * fp32 b) (the Liger SiLU*b form)."""
        PADK: tl.constexpr = CP != C
        pid = tl.program_id(0)
        rm = pid * BM + tl.arange(0, BM)
        mrow = rm < M
        rc = tl.arange(0, CP)
        mcol = rc < C
        rm64 = rm.to(tl.int64)
        if PADK:
            ymask = mrow[:, None] & mcol[None, :]
        else:
            ymask = mrow[:, None]
        y = tl.load(Y + rm64[:, None] * s_ym + rc[None, :], mask=ymask, other=0.0)
        if LN_MODE == 1:
            # fp32 statistics: two-pass on the register tile (mean; then mean of squared deviations = biased variance), eps inside the sqrt, IEEE sqrt + div.
            xf = y.to(tl.float32)
            mean = tl.sum(xf, axis=1) / C
            d = xf - mean[:, None]
            if PADK:
                d = tl.where(mcol[None, :], d, 0.0)
            var = tl.sum(d * d, axis=1) / C
            inv = 1.0 / tl.sqrt_rn(var + eps)
            w = tl.load(LNW + rc, mask=mcol, other=0.0).to(tl.float32)
            bb = tl.load(LNB + rc, mask=mcol, other=0.0).to(tl.float32)
            y = (d * inv[:, None] * w[None, :] + bb[None, :]).to(tl.bfloat16)
            if PADK:
                y = tl.where(mcol[None, :], y, 0.0).to(tl.bfloat16)
        acc = tl.zeros((BM, CP), dtype=tl.float32)
        for j in range(0, NH, BH):
            rh = j + tl.arange(0, BH)
            if IL == 1:
                r2 = 2 * j + tl.arange(0, 2 * BH)
                if PADK:
                    wT = tl.load(WAB + r2[None, :] * s_wab + rc[:, None], mask=mcol[:, None], other=0.0)      # [CP, 2BH] columns = (a_j, b_j, a_j+1, b_j+1, ...)
                else:
                    wT = tl.load(WAB + r2[None, :] * s_wab + rc[:, None])
                ab = tl.dot(y, wT)                                                                            # [BM, 2BH] fp32, one chain over K
                a32, b32 = tl.split(tl.reshape(ab, (BM, BH, 2)))
                a16 = a32.to(tl.bfloat16)
                b16 = b32.to(tl.bfloat16)
            else:
                if PADK:
                    waT = tl.load(WAB + rh[None, :] * s_wab + rc[:, None], mask=mcol[:, None], other=0.0)     # [CP, BH] = (Wa rows j..j+BH)^T
                    wbT = tl.load(WAB + (rh + NH)[None, :] * s_wab + rc[:, None], mask=mcol[:, None], other=0.0)
                else:
                    waT = tl.load(WAB + rh[None, :] * s_wab + rc[:, None])
                    wbT = tl.load(WAB + (rh + NH)[None, :] * s_wab + rc[:, None])
                a16 = tl.dot(y, waT).to(tl.bfloat16)                                                          # fp32 acc over the full K in one chain -> bf16 (== cuBLAS out)
                b16 = tl.dot(y, wbT).to(tl.bfloat16)
            if SILU == 1:
                h = _silu_mul_liger(a16, b16)                                                                 # [BM, BH] bf16, the Liger SiLU*b form (one rounding)
            else:
                h = _silu_mul_bf16(a16, b16)                                                                  # [BM, BH] bf16, stock rounding points
            if PADK:
                woT = tl.load(WO + rc[None, :] * s_wo + rh[:, None], mask=mcol[None, :], other=0.0)           # [BH, CP] = rows j..j+BH of Wo^T
            else:
                woT = tl.load(WO + rc[None, :] * s_wo + rh[:, None])
            acc = tl.dot(h, woT, acc)                                                                         # ascending hidden chunks == cuBLAS K order (checked bit-exact)
        o16 = acc.to(tl.bfloat16)
        if PADK:
            omask = mrow[:, None] & mcol[None, :]
        else:
            omask = mrow[:, None]
        if HAS_RES:
            r = tl.load(RES + rm64[:, None] * s_rm + rc[None, :], mask=omask, other=0.0)
            o16 = (r.to(tl.float32) + o16.to(tl.float32)).to(tl.bfloat16)                                    # z += u under autocast: fp32 add of two bf16, one rounding
        tl.store(OUT + rm64[:, None] * s_om + rc[None, :], o16, mask=omask)


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------
def _weights(module, device):
    """Packed bf16 weights cached on the module instance (module._fpf_cache); parameters are never mutated.  Re-packed if the parameters move/are replaced."""
    wa, wb, wo = module.linear_no_bias_a.weight, module.linear_no_bias_b.weight, module.linear_no_bias.weight
    key = (wa.data_ptr(), wb.data_ptr(), wo.data_ptr(), wa._version, wb._version, wo._version, str(device))
    cache = getattr(module, "_fpf_cache", None)
    if not isinstance(cache, dict) or cache.get("transition_key") != key:
        with torch.no_grad():
            wa16 = wa.detach().to(device=device, dtype=torch.bfloat16); wb16 = wb.detach().to(device=device, dtype=torch.bfloat16)
            wab16 = torch.cat([wa16, wb16], 0).contiguous()                                        # [2*NH, C]  (IL=0 layout)
            wabI16 = torch.stack([wa16, wb16], 1).reshape(2 * wa16.shape[0], wa16.shape[1]).contiguous()   # row 2i = Wa[i], 2i+1 = Wb[i]  (IL=1 layout)
            wo16 = wo.detach().to(device=device, dtype=torch.bfloat16).contiguous()                # [C, NH]
            ln = module.layernorm1
            lnw = (ln.weight.detach() if getattr(ln, "weight", None) is not None else torch.ones(module.c_in, device=device)).to(device=device, dtype=torch.bfloat16).contiguous()
            lnb = (ln.bias.detach() if getattr(ln, "bias", None) is not None else torch.zeros(module.c_in, device=device)).to(device=device, dtype=torch.bfloat16).contiguous()
        new = dict(cache) if isinstance(cache, dict) else {}
        new.update({"transition_key": key, "wab16": wab16, "wabI16": wabI16, "wo16": wo16, "lnw16": lnw, "lnb16": lnb, "eps": float(getattr(module.layernorm1, "eps", 1e-5))})
        cache = new
        try:
            module._fpf_cache = cache
        except Exception:
            pass
    return cache


def _launch(y2d, cache, out2d, res2d=None, ln_mode=0):
    """y2d [M, C] bf16 (row stride arbitrary, unit column stride); out2d [M, C] bf16 view to write (may be a row-slice of a bigger tensor); res2d optional [M, C] bf16."""
    M, C = y2d.shape
    wo16 = cache["wo16"]
    NH = wo16.shape[1]
    cfg = pick_config(C, NH, y2d.device)
    if cfg is None:
        raise KeyError(f"no pinned config for (c_in={C}, hidden={NH}, {gpu_class(y2d.device)})")
    BM, BH, IL = cfg["BM"], cfg["BH"], cfg.get("IL", 0)
    assert NH % BH == 0, (NH, BH)
    assert y2d.stride(1) == 1 and out2d.stride(1) == 1 and (res2d is None or res2d.stride(1) == 1)
    wab = cache["wabI16"] if IL == 1 else cache["wab16"]
    CP = triton.next_power_of_2(C)
    if M == 0:
        return out2d
    grid = (triton.cdiv(M, BM),)
    res = res2d if res2d is not None else out2d
    _fused_transition_kernel[grid](y2d, wab, wo16, out2d, res, cache["lnw16"], cache["lnb16"], M,
                                   y2d.stride(0), wab.stride(0), wo16.stride(0), out2d.stride(0), res.stride(0), cache["eps"],
                                   C=C, CP=CP, NH=NH, BM=BM, BH=BH, IL=IL, HAS_RES=(res2d is not None), LN_MODE=ln_mode,
                                   num_warps=cfg["num_warps"], num_stages=cfg["num_stages"])
    return out2d


def _eligible(module, x):
    """Fused path preconditions (else: stock body).  Pure function of shapes/dtypes/flags."""
    if not _HAS_TRITON:
        return False, "no-triton"
    if not (torch.is_tensor(x) and x.is_cuda):
        return False, "not-cuda"
    if x.dtype != torch.bfloat16:
        return False, f"dtype-{x.dtype}"            # fp32 inputs -> stock cuBLAS path; only the bf16 class is tested
    if module.training:
        return False, "training"
    if x.dim() < 2 or x.numel() == 0:
        return False, "shape"
    c = module.c_in
    if x.shape[-1] != c or not FUSED_C_IN.get(c, False):
        return False, f"c_in-{c}"
    NH = module.linear_no_bias.weight.shape[1]
    kind, _cfg, word = cell_verdict(c, NH, x.device)
    if kind in ("off", "none") and not os.environ.get("FPF_TRANSITION_CFG", "").strip():
        return False, word                          # 'no-config-<c>-<NH>+off(not-measured)' (measured off) / 'no-config-<c>-<NH>[+no_safe(..)]' (nothing admits the cell)
    for lin in (module.linear_no_bias_a, module.linear_no_bias_b, module.linear_no_bias):
        if getattr(lin, "precision", None) is not None or getattr(lin, "bias", None) is not None:
            return False, "linear-variant"
    return True, ""


def fn_stockbody(module, x):
    """The stock primitives.Transition.forward eval branch (v2.0.0), verbatim semantics."""
    other_dims = x.shape[:-1]
    dim_size = x.shape[-1]
    size = x.shape[-2]
    x = x.reshape(-1, dim_size)
    chunk_num = 1 if size < 3200 else 8
    chunks = torch.chunk(x, chunk_num, dim=-2)
    outputs = torch.empty((x.shape[0], module.c_in), dtype=x.dtype, device=x.device)
    start = 0
    for chunk in chunks:
        y = module.layernorm1(chunk)
        a = module.linear_no_bias_a(y)
        a = F.silu(a, True)
        b = module.linear_no_bias_b(y)
        del y
        b *= a
        del a
        b = module.linear_no_bias(b)
        outputs[start: start + b.shape[0]] = b
        start += b.shape[0]
        del b
    return outputs.reshape(*other_dims, module.c_in)


def _forward(module, x, residual: bool, ln_mode: int):
    ok, why = _eligible(module, x)
    if not ok:
        _note_fallback(why)
        out = fn_stockbody(module, x)
        if residual:
            xr = x.clone(); xr += out                 # same op as the caller's in-place `z += update`
            return xr
        return out
    other_dims = x.shape[:-1]
    c = x.shape[-1]
    size = x.shape[-2]
    x2 = x.reshape(-1, c)
    if x2.stride(-1) != 1 or (x2.shape[0] > 1 and x2.stride(0) != c):
        x2 = x2.contiguous()
    cache = _weights(module, x2.device)
    chunk_num = 1 if size < 3200 else 8                      # same chunking as stock (rows are independent in the fused kernel -> numerically irrelevant; kept for memory parity)
    chunks = torch.chunk(x2, chunk_num, dim=-2)
    outputs = torch.empty((x2.shape[0], module.c_in), dtype=x2.dtype, device=x2.device)
    start = 0
    for chunk in chunks:
        n = chunk.shape[0]
        out_view = outputs[start: start + n]
        if ln_mode == 0:
            y = module.layernorm1(chunk)                      # STOCK fast_layernorm kernel (bit-exact by construction); FusedLayerNorm does .contiguous() itself
            if y.dtype != torch.bfloat16:
                y = y.to(torch.bfloat16)
            if y.stride(-1) != 1:
                y = y.contiguous()
            _launch(y, cache, out_view, res2d=(chunk if residual else None), ln_mode=0)
            del y
        else:
            _launch(chunk, cache, out_view, res2d=(chunk if residual else None), ln_mode=1)
        STATS["fused_calls"] += 1
        start += n
    return outputs.reshape(*other_dims, module.c_in)


def fn(module, x):
    """registry `transition`: returns the update (caller adds).  EXACT class (stock LN kernel + fused post-LN kernel)."""
    return _forward(module, x, residual=False, ln_mode=0)


def fn_residual(module, x):
    """returns x + transition(x) with the residual add inside the kernel epilogue (bf16(fp32(x) + fp32(bf16(update)))) == stock `z += transition(z)`."""
    return _forward(module, x, residual=True, ln_mode=0)


def fn_lnfused(module, x):
    """Tier-2 variant: LayerNorm in the kernel prologue (fp32 stats, IEEE sqrt/div).  Returns the update."""
    return _forward(module, x, residual=False, ln_mode=1)


def fn_lnfused_residual(module, x):
    return _forward(module, x, residual=True, ln_mode=1)


def post_ln(module, y2d, residual_src=None, il=None):
    """Timing run helper: the fused post-LN kernel alone on an LN output y2d [M, C] bf16 (the post-LN chain without the LayerNorm kernel)."""
    cache = _weights(module, y2d.device)
    out = torch.empty_like(y2d)
    return _launch(y2d, cache, out, res2d=residual_src, ln_mode=0)


def describe() -> dict:
    return {"version": __version__, "pinned_configs": json.loads(config_table_json()), "config_table_sha256": config_table_sha256(),
            "fused_c_in": {str(k): v for k, v in FUSED_C_IN.items()}, "stats": json.loads(json.dumps(STATS, default=str)),
            "env_override": os.environ.get("FPF_TRANSITION_CFG", ""), "libdevice_exp": _ld is not None, "settings": _NET.word()}
