"""Row vjp: the DIFFERENTIABLE triangle attention (jax.custom_vjp), selected only by word (triangle_attention(..., vjp="auto" | <bwd>)).

  forward   per compute capability (CELLS.json "vjp"."fwd_by_cc", first servable): `cuda_lse` = the sm_90a CUDA kernel of row cuda_sm90a built WITH a
            per-row log-sum-exp store (csrc/cuda_lse.patch -> bin/cuda/sm_90a/libtriattn_mw_cuda_lse.so; cc 9.0, bf16, S >= knobs.cuda_lse.min_S; out
            bit-identical to row cuda_sm90a's) else `k2bl` = the K2B kernel compiled ahead of time with the store (csrc/k2b_lse_kernel.py ->
            bin/k2b/sm_*/k2bl_fwd_*.cubin; out bit-identical to row k2b_aot's).  Residuals kept for the backward: (q, k, v, bias, mask, out, lse) --
            out + lse = B*N*H*SQ*(D*itemsize + 4) bytes on top of the inputs (no [N,H,S,S] logits, no re-run of the forward).
  backward  one of BWDS, Pallas kernels lowered through Triton by the running jax, chosen per (compute capability, dtype) by CELLS.json "vjp":
              attbwd     kernels/pallas_triatt attbwd_dkdv._bwd: dK/dV over key blocks + dQ and the d(bias) partials summed over groups of rows over
                         query blocks (partials [ceil(N/G),H,SQ,SKp], reduced by XLA); no atomics, run-to-run bitwise
              flash      kernels/pallas_attn af2_flash_pallas._bwd with d(bias) summed over the rows inside a third kernel (no [N,H,SQ,SK] partials)
              flash_xla  the same with per-row d(bias) partials [N,H,SQ,SK] reduced by XLA (fewer FLOPs, N*H*SQ*SK*itemsize bytes of partials)
            d(bias) is the sum over the N rows (and is returned in the bias' dtype and shape [B,H,SQ,SK]); the mask takes no gradient.
  serves    layout BNHSD, B == 1 (one pair bias per call), head_dim 32, bf16 (log2-domain softmax statistics) and fp32 (TF32 products, forward and
            backward), keys masked per row (uint8/bool [B,N,SK]) or no mask; compute capability 9.0 (sm_90 cubins) and 8.x (sm_80 cubins).
  refuses   BY NAME, naming the fallback (VJP_FALLBACK: XLA autodiff of the program's own einsum/softmax attention), when the shape/dtype/card is
            not served, when the backward's Pallas modules do not import or lower on the running jax (the error is quoted), or when the word
            `triattn_xla:vjp` (or `triattn_xla`) is in MODEL_OPT_LEVERS_OFF.
Fully-masked rows: the forward serves them as the forward-only rows do (uniform average of v); the backward gives such a row dV from the uniform
weights (sum of dO over the queries / SK on every key) and no dQ / dK / d(bias) -- the XLA autodiff of the -1e9-masked reference -- with
either backward (`attbwd`: the kernels' own zero-logit convention; `flash`: the row is taken out of the kernel and its dV term added by XLA)."""
import math
import os
from typing import Dict, List, Optional

from . import PKG_DIR, FALLBACK, Refused, binaries, cells
from . import _k2b

LOG2E = 1.4426950408889634
LN2 = 0.6931471805599453
BWDS = ("attbwd", "flash", "flash_xla")
VJP_FALLBACK = "XLA autodiff of the program's own einsum/softmax attention (this package's reference())"
_CALLS: Dict[str, object] = {}
_IMPORT_ERR: Dict[str, Optional[str]] = {}
SERVED: Dict[str, int] = {}


def vjp_cells() -> dict:
    return cells().get("vjp", {})


def dims_built(arch: str, dtype: str) -> List[int]:
    return sorted({b["head_dim"] for b in _k2b.cubin_index().values() if b.get("role") == "fwd_lse" and b["arch"] == arch and b.get("dtype") == dtype})


def variant(dtype: str, D: int, SQ: int, SK: int, N: int, has_mask: bool):
    b16, div, kmain, kprep = _k2b.variant(dtype, D, SQ, SK, N, has_mask)
    return b16, div, "k2bl" + kmain[len("k2b"):], kprep


def bwd_order(cc: str, dtype: str) -> List[str]:
    by = vjp_cells().get("by_cc", {})
    ent = by.get(cc) or by.get(cc.split(".")[0] + ".x") or {}
    order = ent.get(dtype) or ent.get("order") or list(BWDS)
    return [b for b in order if b in BWDS]


def bwd_import_error(bwd: str) -> Optional[str]:
    """None when the backward's modules import on this interpreter, else the error text (cached)."""
    fam = "attbwd" if bwd == "attbwd" else "flash"
    if fam not in _IMPORT_ERR:
        try:
            if fam == "attbwd":
                from opt_core.kernels.pallas_triatt import triatt_attn, attbwd_dkdv   # noqa: F401
            else:
                from opt_core.kernels.pallas_attn import af2_flash_pallas             # noqa: F401
            _IMPORT_ERR[fam] = None
        except Exception as e:   # noqa: BLE001 -- an old jax without these Pallas APIs, no triton lowering, ...: quoted in the refusal
            _IMPORT_ERR[fam] = "%s: %s" % (type(e).__name__, str(e)[:300])
    return _IMPORT_ERR[fam]


def cannot_serve(cc: str, dtype: str, D: int, SQ: int, SK: int, N: int, H: int, B: int, has_mask: bool, layout: int, bwd: str) -> Optional[str]:
    from . import arch_for_cc
    if layout != 0:
        return "layout BNSHD (the differentiable row reads BNHSD: rows, heads, queries, head_dim -- the layout both backward kernels take)"
    if B != 1:
        return "batch B=%d (one pair bias per call: B == 1; call per batch element)" % B
    arch = arch_for_cc(cc)
    if arch is None:
        return "no cubin architecture for compute capability %s (built: sm_90 for 9.0, sm_80 for 8.x)" % cc
    if dtype not in _k2b.DTYPES:
        return "dtype %s (served: bf16, fp32)" % dtype
    built = dims_built(arch, dtype)
    if not built:
        return "no %s %s forward-with-lse cubins (k2bl_fwd_*) in this package" % (arch, dtype)
    if D not in built:
        return "head_dim %d (forward-with-lse cubins built for %s %s: %s)" % (D, arch, dtype, built)
    if SQ < 16 or SK < 16:
        return "S below 16 (SQ=%d SK=%d): not kernel work" % (SQ, SK)
    b16, div, kmain, kprep = variant(dtype, D, SQ, SK, N, has_mask)
    idx = _k2b.cubin_index()
    for key in (kmain, kprep):
        if arch + "/" + key not in idx:
            return "cubin %s/%s not in this package" % (arch, key)
        if not os.path.isfile(os.path.join(PKG_DIR, idx[arch + "/" + key]["file"])):
            return "cubin file %s missing on disk" % idx[arch + "/" + key]["file"]
    kc = _k2b._k2b_consts()
    nqb = -(-SQ // int(kc["prep"]["PB"]))
    n_flags = B * H * nqb if b16 else 1
    if n_flags > int(kc["nflag"]):
        return "B*H*ceil(SQ/32) = %d bias-preparation flags exceed the compiled NFLAG %d" % (n_flags, kc["nflag"])
    SKp = -(-SK // 16) * 16
    cell = idx[arch + "/" + kmain].get("cell", {})
    bn = int(cell.get("BLOCK_N", 32))
    if bn * max(H * SK * D, SK * D, D) + D >= 2 ** 31 or SQ * SKp >= 2 ** 31 or B * N * H * SQ >= 2 ** 31:
        return "tensor too large for the kernel's int32 tile offsets"
    if B * H > 65535 or -(-N // int(cell.get("ROWS", 2))) > 65535:
        return "grid limits (B*H <= 65535, row groups <= 65535)"
    if bwd not in BWDS:
        return "unknown backward %r (backwards: %s)" % (bwd, ", ".join(BWDS))
    if bwd == "flash_xla":                       # per-row d(bias) partials [N,H,SQ,SK] in the bias dtype: a memory cost, bounded by the table
        cap = int(vjp_cells().get("knobs", {}).get("flash", {}).get("dbias_xla_max_bytes", 512 << 20))
        nbytes = N * H * SQ * SK * (4 if dtype == "fp32" else 2)
        if nbytes > cap:
            return "d(bias) partials N*H*SQ*SK = %d MiB exceed %d MiB (CELLS.json vjp.knobs.flash.dbias_xla_max_bytes); `flash` sums them in-kernel" % (nbytes >> 20, cap >> 20)
    err = bwd_import_error(bwd)
    if err:
        return "backward %s does not import on this jax (%s)" % (bwd, err)
    return None


def _fwd_cannot_serve(arch: str, dtype: str, D: int, SQ: int, SK: int, N: int, H: int, B: int, has_mask: bool) -> Optional[str]:
    """The forward-with-lse launch envelope for a (possibly folded) batch size B."""
    built = dims_built(arch, dtype)
    if D not in built:
        return "head_dim %d (built: %s)" % (D, built)
    b16, div, kmain, kprep = variant(dtype, D, SQ, SK, N, has_mask)
    kc = _k2b._k2b_consts()
    n_flags = B * H * -(-SQ // int(kc["prep"]["PB"])) if b16 else 1
    if n_flags > int(kc["nflag"]):
        return "B*H*ceil(SQ/32) = %d bias-preparation flags exceed the compiled NFLAG %d" % (n_flags, kc["nflag"])
    cell = _k2b.cubin_index()[arch + "/" + kmain].get("cell", {})
    if B * H > 65535 or -(-N // int(cell.get("ROWS", 2))) > 65535 or B * N * H * SQ >= 2 ** 31:
        return "grid / offset limits"
    return None


FWDS = ("cuda_lse", "k2bl")


def fwd_order(cc: str) -> List[str]:
    """Forward launches of the differentiable row for this compute capability, first servable wins: CELLS.json "vjp"."fwd_by_cc" (cuda_lse = the sm_90a
    CUDA kernel + its log-sum-exp store, cc 9.0; k2bl = the K2B cubin with the store, every card).  TRIATTN_XLA_VJP_FWD=<name> forces one (measurement)."""
    forced = os.environ.get("TRIATTN_XLA_VJP_FWD")
    if forced:
        return [forced]
    tab = vjp_cells().get("fwd_by_cc", {})
    return list(tab.get(cc) or tab.get(cc.split(".")[0] + ".x") or ["k2bl"])


def _cuda_lse_cannot_serve(cc: str, dtype: str, D: int, SQ: int, SK: int, N: int, H: int, B: int, has_mask: bool, explicit: bool) -> Optional[str]:
    from . import _cuda
    why = _cuda.cannot_serve(cc, dtype, D, SQ, SK, N, H, B, has_mask)
    if why:
        return why
    if _cuda.lib_entry("fwd_lse") is None:
        return "no lse CUDA library (role fwd_lse) in manifest.json"
    min_s = int(vjp_cells().get("knobs", {}).get("cuda_lse", {}).get("min_S", 0))
    if not explicit and SQ < min_s:
        return "S %d < %d (CELLS.json vjp.knobs.cuda_lse.min_S: below it the K2B cubin is the faster forward)" % (SQ, min_s)
    return None


def select_fwd(cc: str, arch: str, dtype: str, D: int, SQ: int, SK: int, N: int, H: int, B: int, has_mask: bool):
    """(name, reasons): the first forward of fwd_order(cc) whose envelope holds."""
    reasons = []
    explicit = bool(os.environ.get("TRIATTN_XLA_VJP_FWD"))
    for name in fwd_order(cc):
        if name == "cuda_lse":
            why = _cuda_lse_cannot_serve(cc, dtype, D, SQ, SK, N, H, B, has_mask, explicit)
        elif name == "k2bl":
            why = _fwd_cannot_serve(arch, dtype, D, SQ, SK, N, H, B, has_mask)
        else:
            why = "unknown forward %r (forwards: %s)" % (name, ", ".join(FWDS))
        if why is None:
            return name, reasons
        reasons.append("%s: %s" % (name, why))
    return None, reasons


def _fwd_call(name: str, B, N, H, SQ, SK, D, dtype: str, has_mask: bool, scale: float, arch: str, bias_dtype: str):
    if name == "cuda_lse":
        from . import _cuda
        return _cuda.forward_lse_call(B, N, H, SQ, D, has_mask, float(scale), bias_dtype)
    return _forward_lse_call(B, N, H, SQ, SK, D, dtype, has_mask, float(scale), arch)


def select_bwd(cc: str, dtype: str, D: int, SQ: int, SK: int, N: int, H: int, B: int, has_mask: bool, layout: int, word: str):
    """(bwd, reasons) for vjp=<word>: 'auto' walks CELLS.json vjp.by_cc order; a named backward is tried alone."""
    cands = bwd_order(cc, dtype) if word == "auto" else [word]
    reasons = []
    for b in cands:
        why = cannot_serve(cc, dtype, D, SQ, SK, N, H, B, has_mask, layout, b)
        if why is None:
            return b, reasons
        reasons.append("%s: %s" % (b, why))
    if not cands:
        reasons.append("no backward listed for cc=%s dtype=%s" % (cc, dtype))
    return None, reasons


# ----------------------------------------------------------------------------------------------------------------- forward (+lse)
def _forward_lse_call(B, N, H, SQ, SK, D, dtype: str, has_mask: bool, scale: float, arch: str):
    """raw(q, k, v, bias, mask_u8) -> (out [B,N,H,SQ,D], lse2 [B,N,H,SQ] fp32): the k2bl cubin launch (bias preparation as row k2b_aot's)."""
    import jax
    import jax.numpy as jnp
    from . import _launch
    b16, div, kmain, kprep = variant(dtype, D, SQ, SK, N, has_mask)
    kc = _k2b._k2b_consts()
    PB = int(kc["prep"]["PB"])
    SKp = -(-SK // 16) * 16
    nqb = -(-SQ // PB)
    n_flags = B * H * nqb if b16 else 1
    key = "%s|%s|%s|B%d N%d H%d SQ%d SK%d D%d|%.10g" % (arch, kmain, kprep, B, N, H, SQ, SK, D, scale)
    hit = _CALLS.get(key)
    if hit is not None:
        return hit
    mp, mm = _k2b._load_cubin(arch, kprep), _k2b._load_cubin(arch, kmain)
    f32 = jnp.float32
    el = jnp.bfloat16 if dtype == "bf16" else jnp.float32
    pvals = {"sbb": H * SQ * SK, "sbh": SQ * SK, "sbq": SK, "sbk": 1, "H": H, "SQ": SQ, "SK": SK, "SKp": SKp, "NQB": nqb}
    pptr = {"Bias": (0, 0), "Out32": (1, 0), "Out16": (1, 1 if b16 else 0), "Flags": (1, 2 if b16 else 0)}
    kinds, ivals, fvals = [], [], []
    for n in mp["params"]:
        if n in pptr:
            kinds.append(pptr[n][0]); ivals.append(pptr[n][1]); fvals.append(0.0)
        else:
            kinds.append(2); ivals.append(int(pvals[n])); fvals.append(0.0)
    prep_out = [jax.ShapeDtypeStruct((B, 1, H, SQ, SKp), f32)]
    if b16:
        prep_out += [jax.ShapeDtypeStruct((B, 1, H, SQ, SKp), jnp.bfloat16), jax.ShapeDtypeStruct((max(n_flags, 1),), jnp.int32)]
    sid_p = _launch.register_spec("prep|" + key, mp, mp["kernel"]["name"], mp["kernel"]["shared"], mp["kernel"]["num_warps"], (nqb, B * H),
                                  kinds, ivals, fvals, mp["n_trailing_null"])
    prep_call = _launch.launch_call(sid_p, prep_out)
    cell = mm["cell"]
    bm, rows, order = int(cell["BLOCK_M"]), int(cell["ROWS"]), int(cell.get("ORDER", 0))
    sq_ = _k2b._strides(0, N, H, SQ, D); sk_ = _k2b._strides(0, N, H, SK, D)
    mvals = dict(zip(["sqb", "sqi", "sqh", "sqq"], sq_)); mvals.update(zip(["skb", "ski", "skh", "skk"], sk_)); mvals.update(zip(["svb", "svi", "svh", "svk"], sk_))
    mvals.update({"sbb": H * SQ * SKp, "sbh": SQ * SKp, "sbq": SKp})
    mvals.update({"smb": N * SK, "smi": SK, "smk": 1} if has_mask else {"smb": 0, "smi": 0, "smk": 0})
    mvals.update(zip(["sob", "soi", "soh", "soq"], sq_))
    mvals.update({"N_ROWS": N, "SEQ_Q": SQ, "SEQ_K": SK, "H": H, "n_flags": n_flags})
    idx = {"Q": 0, "K": 1, "V": 2, "Bias32": 3}
    nin = 4
    if b16:
        idx["Bias16"] = nin; idx["Flags"] = nin + 1; nin += 2
    else:
        idx["Bias16"] = 3; idx["Flags"] = 3
    if has_mask:
        idx["Mask"] = nin; nin += 1
    else:
        idx["Mask"] = 0
    kinds, ivals, fvals = [], [], []
    for n in mm["params"]:
        if n == "Out":
            kinds.append(1); ivals.append(0); fvals.append(0.0)
        elif n == "Lse":
            kinds.append(1); ivals.append(1); fvals.append(0.0)
        elif n in idx:
            kinds.append(0); ivals.append(idx[n]); fvals.append(0.0)
        elif n == "qk_scale":
            kinds.append(4); ivals.append(0); fvals.append(float(scale))
        else:
            kinds.append(2); ivals.append(int(mvals[n])); fvals.append(0.0)
    grid = (-(-SQ // bm), -(-N // rows), B * H) if order == 0 else (-(-N // rows), -(-SQ // bm), B * H)
    sid_m = _launch.register_spec("fwdlse|" + key, mm, mm["kernel"]["name"], mm["kernel"]["shared"], mm["kernel"]["num_warps"], grid,
                                  kinds, ivals, fvals, mm["n_trailing_null"])
    main_call = _launch.launch_call(sid_m, [jax.ShapeDtypeStruct((B, N, H, SQ, D), el), jax.ShapeDtypeStruct((B, N, H, SQ), f32)])

    def raw(qq, kk, vv, bb, mm_):
        pouts = prep_call(bb.astype(f32))
        ins = [qq, kk, vv, pouts[0]]
        if b16:
            ins += [pouts[1], pouts[2]]
        if has_mask:
            ins.append(mm_)
        out = main_call(*ins)
        return out[0], out[1]
    raw.cubin = kmain
    _CALLS[key] = raw
    return raw


# ----------------------------------------------------------------------------------------------------------------- backwards
def _knobs(bwd: str) -> dict:
    return dict(vjp_cells().get("knobs", {}).get("attbwd" if bwd == "attbwd" else "flash", {}))


def _bwd_attbwd(q, k, v, bias, mask_u8, out, lse2, do, scale: float, has_mask: bool):
    """kernels/pallas_triatt: rows = the kernels' batch, bias [H,SQ,SKp] in the compute dtype, key mask -> additive row + dead flag (their _prep)."""
    import jax.numpy as jnp
    from opt_core.kernels.pallas_triatt import triatt_attn as T, attbwd_dkdv as A
    kn = _knobs("attbwd")
    N, SK = int(q.shape[1]), int(k.shape[3])
    q0, k0, v0, o0, do0 = q[0], k[0], v[0], out[0], do[0]
    key_mask = (mask_u8[0] != 0) if has_mask else jnp.ones((N, SK), dtype=bool)
    bias2, madd, dead = T._prep(bias[0], key_mask, SK, q.dtype)
    lse = lse2[0]
    if has_mask:                                     # a fully-masked row: the kernels' zero-logit convention (uniform weights over the SK keys)
        lse = jnp.where((dead != 0)[:, None, None], jnp.float32(math.log2(SK)), lse)
    pdt = jnp.float32 if (q.dtype == jnp.float32 or kn.get("dbias_partials", "input") == "f32") else q.dtype
    k1 = dict(A.DEFAULTS_BWD["k1"]); k1.update(kn.get("k1", {}))
    k2 = dict(A.DEFAULTS_BWD["k2"]); k2.update(kn.get("k2", {}))
    dq, dk, dv, dbias = A._bwd(q0, k0, v0, bias2, madd, dead, o0, lse, do0, scale=float(scale), k1=k1, k2=k2, order=kn.get("grid_order", T.DEFAULTS["grid_order"]),
                               f32p=kn.get("f32_precision", T.DEFAULTS["f32_precision"]), has_mask=has_mask, bias_mma=True, pdt=pdt,
                               resident=int(kn.get("resident", A.DEFAULTS_BWD.get("resident", 1))))
    return dq[None], dk[None], dv[None], dbias.astype(bias.dtype)[None]


def _bwd_flash(q, k, v, bias, mask_u8, out, lse2, do, scale: float, has_mask: bool, dbias_mode: str):
    """kernels/pallas_attn: rows = batch, bias [H,SQ,SK] in q's dtype, boolean key mask, natural-log lse."""
    import jax.numpy as jnp
    from opt_core.kernels.pallas_attn import af2_flash_pallas as F
    kn = _knobs("flash")
    N, SK = int(q.shape[1]), int(k.shape[3])
    q0, k0, v0, o0, do0 = q[0], k[0], v[0], out[0], do[0]
    kmask = (mask_u8[0] != 0) if has_mask else jnp.ones((N, SK), dtype=bool)
    biasq = bias[0].astype(q.dtype)
    lse = lse2[0] * jnp.float32(LN2)
    if has_mask:     # a fully-masked row attends uniformly (out = mean of v): its only gradient is dV = sum_q(dO)/SK on every key; dQ = dK = d(bias) = 0.
        dead = ~jnp.any(kmask, axis=-1)                                                    # [N]  The kernel's exp(s - lse) cannot represent that row
        do0 = jnp.where(dead[:, None, None, None], jnp.zeros((), do0.dtype), do0)            # (lse = -1e9 + ln SK is not an fp32 number): take the row out
    dbias_dtype = jnp.float32 if (q.dtype == jnp.float32 or kn.get("dbias_partials", "input") == "f32") else biasq.dtype
    dq, dk, dv, dbias = F._bwd(q0, k0, v0, biasq, kmask, o0, lse, do0, sm_scale=float(scale), bq=int(kn.get("bq", 64)), bk=int(kn.get("bk", 64)),
                               num_warps=int(kn.get("num_warps", 4)), num_stages=int(kn.get("num_stages", 3)), dbias_dtype=dbias_dtype, precise=False,
                               f32p=kn.get("f32_precision", "tf32"), dq_mode="kernel", dbias_mode=dbias_mode)
    if has_mask:     # ... and add its uniform-average term here (XLA; N*H*D numbers)
        dv_dead = (jnp.sum(do[0].astype(jnp.float32), axis=2) / SK).astype(dv.dtype)       # [N,H,D]
        dv = dv + jnp.where(dead[:, None, None, None], dv_dead[:, :, None, :], jnp.zeros((), dv.dtype))
    return dq[None], dk[None], dv[None], dbias.astype(bias.dtype)[None]


def _dbias_mode(bwd: str, N: int, H: int, SQ: int, SK: int) -> str:
    if bwd == "flash_xla":
        return "xla"
    return "kernel"


# ----------------------------------------------------------------------------------------------------------------- the differentiable call
def attention(q, k, v, bias, mask_u8, scale: float, layout: int, cc: str, arch: str, word: str):
    """q/k/v rank 5 in layout 0 (BNHSD), bias [B,H,SQ,SK], mask u8 [B,N,SK] | None -> out (q's shape); differentiable in q, k, v, bias."""
    import numpy as np
    import jax
    import jax.numpy as jnp
    B, N, H, SQ, D = (int(x) for x in q.shape); SK = int(k.shape[3])
    dtype = "bf16" if q.dtype == jnp.bfloat16 else ("fp32" if q.dtype == jnp.float32 else str(q.dtype))
    has_mask = mask_u8 is not None
    bwd, reasons = select_bwd(cc, dtype, D, SQ, SK, N, H, B, has_mask, layout, word)
    if bwd is None:
        raise Refused("triattn_xla: the differentiable row (vjp=%r) does not serve cc=%s dtype=%s D=%d S=%d/%d N=%d H=%d B=%d mask=%s -- %s; fallback: %s"
                      % (word, cc, dtype, D, SQ, SK, N, H, B, has_mask, "; ".join(reasons), VJP_FALLBACK))
    fwd_name, freasons = select_fwd(cc, arch, dtype, D, SQ, SK, N, H, B, has_mask)
    if fwd_name is None:
        raise Refused("triattn_xla: the differentiable row (vjp=%r): no forward serves cc=%s dtype=%s D=%d S=%d/%d N=%d H=%d B=%d mask=%s -- %s; fallback: %s"
                      % (word, cc, dtype, D, SQ, SK, N, H, B, has_mask, "; ".join(freasons), VJP_FALLBACK))
    bias_dt = str(bias.dtype)
    raw = _fwd_call(fwd_name, B, N, H, SQ, SK, D, dtype, has_mask, float(scale), arch, bias_dt)
    key = "vjp|%s|%s|%s|%s|B%d N%d H%d SQ%d SK%d D%d|mask%d|%s|%.10g" % (arch, fwd_name, bwd, dtype, B, N, H, SQ, SK, D, int(has_mask), bias_dt, scale)
    fn = _CALLS.get(key)
    if fn is None:
        from . import _batching
        dbm = _dbias_mode(bwd, N, H, SQ, SK)

        def fwd_generic(*arrays):                    # any leading batch size (the vmap rule folds the vmapped axis into B): envelope by name, then launch
            qq, kk, vv, bb = arrays[:4]; mm_ = arrays[4] if has_mask else None
            Bq, Nq, Hq, SQq, Dq = (int(x) for x in qq.shape); SKq = int(kk.shape[3])
            if fwd_name == "cuda_lse":
                why = _cuda_lse_cannot_serve(cc, dtype, Dq, SQq, SKq, Nq, Hq, Bq, has_mask, True)
            else:
                why = _fwd_cannot_serve(arch, dtype, Dq, SQq, SKq, Nq, Hq, Bq, has_mask)
            if why:
                raise Refused("triattn_xla: differentiable row, forward %s for B=%d: %s" % (fwd_name, Bq, why))
            return _fwd_call(fwd_name, Bq, Nq, Hq, SQq, SKq, Dq, dtype, has_mask, float(scale), arch, bias_dt)(qq, kk, vv, bb, mm_)

        def fwd_fixed(*arrays):
            return raw(arrays[0], arrays[1], arrays[2], arrays[3], arrays[4] if has_mask else None)
        rawb = _batching.fold_into_batch(fwd_fixed, fwd_generic, n_out=2)

        def bwd_impl(qq, kk, vv, bb, mm_, oo, ll, do):
            if bwd == "attbwd":
                return _bwd_attbwd(qq, kk, vv, bb, mm_ if has_mask else None, oo, ll, do, scale, has_mask)
            return _bwd_flash(qq, kk, vv, bb, mm_ if has_mask else None, oo, ll, do, scale, has_mask, dbm)
        bwdb = _batching.per_sample(bwd_impl, n_out=4)       # a vmapped gradient: the backward kernels share one bias per call -> per sample

        def backward(res, do):
            qq, kk, vv, bb, mm_, oo, ll = res
            dq, dk, dv, db = bwdb(qq, kk, vv, bb, mm_, oo, ll, do)
            return dq, dk, dv, db, np.zeros(mm_.shape, dtype=jax.dtypes.float0)      # the mask (uint8) takes no gradient

        @jax.custom_vjp
        def core(qq, kk, vv, bb, mm_):
            return rawb(*((qq, kk, vv, bb, mm_) if has_mask else (qq, kk, vv, bb)))[0]

        def core_fwd(qq, kk, vv, bb, mm_):
            out, lse2 = rawb(*((qq, kk, vv, bb, mm_) if has_mask else (qq, kk, vv, bb)))
            return out, (qq, kk, vv, bb, mm_, out, lse2)

        core.defvjp(core_fwd, backward)

        def fn(qq, kk, vv, bb, mm_):
            dummy = mm_ if has_mask else jnp.zeros((1,), jnp.uint8)
            return core(qq, kk, vv, bb, dummy)
        fn.row = "vjp[fwd %s (%s) + bwd %s]" % (fwd_name, getattr(raw, "cubin", arch), bwd)
        _CALLS[key] = fn
    SERVED[bwd] = SERVED.get(bwd, 0) + 1; SERVED["fwd:" + fwd_name] = SERVED.get("fwd:" + fwd_name, 0) + 1
    return fn(q, k, v, bias, mask_u8)


def forward_with_lse(q, k, v, bias, mask_u8, scale: float, arch: str, fwd: str = "k2bl"):
    """(out, lse2) of the k2bl cubin (fwd="k2bl": out bit-identical to row k2b_aot's) or of the lse CUDA library (fwd="cuda_lse": out bit-identical to row
    cuda_sm90a's) -- comparison entry point; lse2 fp32 [B,N,H,S] in log2 units either way."""
    import jax.numpy as jnp
    B, N, H, SQ, D = (int(x) for x in q.shape); SK = int(k.shape[3])
    dtype = "bf16" if q.dtype == jnp.bfloat16 else "fp32"
    raw = _fwd_call(fwd, B, N, H, SQ, SK, D, dtype, mask_u8 is not None, float(scale), arch, str(bias.dtype))
    return raw(q, k, v, bias, mask_u8)


def status() -> dict:
    per = {}
    for b in _k2b.cubin_index().values():
        if b.get("role") == "fwd_lse":
            per.setdefault(b["arch"], []).append(b["key"])
    from . import _cuda
    return {"lse_cubins": {a: len(v) for a, v in per.items()}, "cuda_lse": _cuda.lib_entry("fwd_lse") is not None, "forwards": list(FWDS), "fwd_by_cc": vjp_cells().get("fwd_by_cc", {}),
            "backwards": list(BWDS), "import_errors": {k: e for k, e in _IMPORT_ERR.items() if e},
            "served": dict(SERVED), "by_cc": vjp_cells().get("by_cc", {})}
