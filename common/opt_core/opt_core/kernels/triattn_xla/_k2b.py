"""Row k2b_aot: the K2B flash triangle-attention Triton kernel (kernels/fpf_triatt_k2b/triatt_k2b.py: _bias_prep + _flash_triattn_fwd)
launched from pre-compiled cubins.  The launch arithmetic below restates flash_triangle_attention()'s host code: bias preparation pass
(fp32 copy with a 16-element row pitch + optional lossless 16-bit copy and flags), tile grid, the int32 offset bound, the constexpr
set of the cell; the cubin is chosen by (arch, dtype, D, mask, BIAS16, divisibility class)."""
import json
import math
import os
from typing import Dict, List, Optional

from . import PKG_DIR, FALLBACK, Refused, binaries, cells
from ...gates import binary_refusal

_CUBINS: Dict[str, dict] = {}          # "<arch>/<key>" -> sidecar meta (+ "bytes")
_CALLS: Dict[str, object] = {}
SERVED: Dict[str, int] = {}
DTYPES = ("bf16", "fp32")


def _k2b_consts() -> dict:
    return cells().get("k2b", {"bias16_min_sk": 256, "nflag": 1024, "prep": {"PB": 32, "PK": 128}})


def cubin_index() -> Dict[str, dict]:
    if not _CUBINS:
        for b in binaries("k2b_cubin"):
            _CUBINS[b["arch"] + "/" + b["key"]] = b
    return _CUBINS


def dims_built(arch: str, dtype: str) -> List[int]:
    return sorted({b["head_dim"] for b in cubin_index().values() if b.get("role") == "fwd" and b["arch"] == arch and b.get("dtype") == dtype})


def variant(dtype: str, D: int, SQ: int, SK: int, N: int, has_mask: bool):
    b16 = 1 if (dtype == "bf16" and SK >= int(_k2b_consts()["bias16_min_sk"])) else 0
    div = "s16" if (SQ % 16 == 0 and SK % 16 == 0 and N % 16 == 0) else "any"
    if os.environ.get("TRIATTN_XLA_FORCE_DIV") == "any":          # measurement switch: the non-specialised cubin on a specialisable shape
        div = "any"
    return b16, div, "k2b_fwd_%s_d%d_mask%d_b16%d_%s" % (dtype, D, int(bool(has_mask)), b16, div), "k2b_prep_mk%d_%s" % (b16, div)


def cannot_serve(cc: str, dtype: str, D: int, SQ: int, SK: int, N: int, H: int, B: int, has_mask: bool) -> Optional[str]:
    from . import arch_for_cc
    arch = arch_for_cc(cc)
    if arch is None:
        return "no cubin architecture for compute capability %s (built: sm_90 for 9.0, sm_80 for 8.x)" % cc
    if dtype not in DTYPES:
        return "dtype %s (served: bf16, fp32)" % dtype
    built = dims_built(arch, dtype)
    if not built:
        return "no %s %s cubins in this package" % (arch, dtype)
    if D not in built:
        return "head_dim %d (cubins built for %s %s: %s)" % (D, arch, dtype, built)
    if SQ < 1 or SK < 1:
        return "empty sequence"
    b16, div, kmain, kprep = variant(dtype, D, SQ, SK, N, has_mask)
    for key in (kmain, kprep):
        if arch + "/" + key not in cubin_index():
            return "cubin %s/%s not in this package" % (arch, key)
        if not os.path.isfile(os.path.join(PKG_DIR, cubin_index()[arch + "/" + key]["file"])):
            return "cubin file %s missing on disk" % cubin_index()[arch + "/" + key]["file"]
        why = binary_refusal(os.path.join(PKG_DIR, cubin_index()[arch + "/" + key]["file"]))
        if why:
            return "cubin file %s refused: %s" % (cubin_index()[arch + "/" + key]["file"], why)
    kc = _k2b_consts()
    nqb = -(-SQ // int(kc["prep"]["PB"]))
    n_flags = B * H * nqb if b16 else 1
    if n_flags > int(kc["nflag"]):
        return "B*H*ceil(SQ/32) = %d bias-preparation flags exceed the compiled NFLAG %d (B*H*SQ too large for the 16-bit bias path)" % (n_flags, kc["nflag"])
    SKp = -(-SK // 16) * 16
    cell = cubin_index()[arch + "/" + kmain].get("cell", {})
    bn = int(cell.get("BLOCK_N", 32))
    if bn * max(H * SK * D, SK * D, D) + D >= 2 ** 31 or SQ * SKp >= 2 ** 31:
        return "tensor too large for the kernel's int32 tile offsets"
    if B * H > 65535 or -(-N // int(cell.get("ROWS", 2))) > 65535:
        return "grid limits (B*H <= 65535, row groups <= 65535)"
    return None


def _load_cubin(arch: str, key: str) -> dict:
    m = cubin_index()[arch + "/" + key]
    if "bytes_" not in m:
        with open(os.path.join(PKG_DIR, m["file"]), "rb") as fh:
            data = fh.read()
        why = binary_refusal(os.path.join(PKG_DIR, m["file"]), data)
        if why:
            raise Refused("triattn_xla: k2b: cubin %s refused: %s; fallback: %s" % (m["file"], why, FALLBACK))
        m["bytes_"] = data
    return m


def _strides(layout: int, N: int, H: int, S: int, D: int):
    """Element strides (sB, sN, sH, sS) of a dense [B,N,H,S,D] (layout 0) / [B,N,S,H,D] (layout 1) array indexed as [b, n, h, s, d]."""
    if layout == 0:
        return (N * H * S * D, H * S * D, S * D, D)
    return (N * S * H * D, S * H * D, D, H * D)


def forward(q, k, v, bias, mask_u8, scale: float, layout: int, arch: str):
    import jax
    import jax.numpy as jnp
    from . import _launch
    B, N = int(q.shape[0]), int(q.shape[1]); D = int(q.shape[4])
    if layout == 0:
        H, SQ = int(q.shape[2]), int(q.shape[3]); SK = int(k.shape[3])
    else:
        SQ, H = int(q.shape[2]), int(q.shape[3]); SK = int(k.shape[2])
    dtype = "bf16" if q.dtype == jnp.bfloat16 else ("fp32" if q.dtype == jnp.float32 else str(q.dtype))
    has_mask = mask_u8 is not None
    b16, div, kmain, kprep = variant(dtype, D, SQ, SK, N, has_mask)
    kc = _k2b_consts()
    PB = int(kc["prep"]["PB"])
    SKp = -(-SK // 16) * 16
    nqb = -(-SQ // PB)
    n_flags = B * H * nqb if b16 else 1
    key = "%s|%s|%s|B%d N%d H%d SQ%d SK%d D%d|L%d|%.10g" % (arch, kmain, kprep, B, N, H, SQ, SK, D, layout, scale)
    fn = _CALLS.get(key)
    if fn is None:
        mp, mm = _load_cubin(arch, kprep), _load_cubin(arch, kmain)
        f32 = jnp.float32
        el = jnp.bfloat16 if dtype == "bf16" else jnp.float32
        # ---- bias preparation: Bias fp32 [B,H,SQ,SK] (dense) -> Out32 [B,1,H,SQ,SKp] (+ Out16, Flags when BIAS16)
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
        # ---- the forward
        cell = mm["cell"]
        bm, bn, rows, order = int(cell["BLOCK_M"]), int(cell["BLOCK_N"]), int(cell["ROWS"]), int(cell.get("ORDER", 0))
        sq_ = _strides(layout, N, H, SQ, D); sk_ = _strides(layout, N, H, SK, D)
        mvals = dict(zip(["sqb", "sqi", "sqh", "sqq"], sq_)); mvals.update(zip(["skb", "ski", "skh", "skk"], sk_)); mvals.update(zip(["svb", "svi", "svh", "svk"], sk_))
        mvals.update({"sbb": H * SQ * SKp, "sbh": SQ * SKp, "sbq": SKp})
        mvals.update({"smb": N * SK, "smi": SK, "smk": 1} if has_mask else {"smb": 0, "smi": 0, "smk": 0})
        mvals.update(zip(["sob", "soi", "soh", "soq"], sq_))
        mvals.update({"N_ROWS": N, "SEQ_Q": SQ, "SEQ_K": SK, "H": H, "n_flags": n_flags})
        # inputs of the main call: q, k, v, bias32[, bias16, flags][, mask]
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
            elif n in idx:
                kinds.append(0); ivals.append(idx[n]); fvals.append(0.0)
            elif n == "qk_scale":
                kinds.append(4); ivals.append(0); fvals.append(float(scale))
            else:
                kinds.append(2); ivals.append(int(mvals[n])); fvals.append(0.0)
        grid = (-(-SQ // bm), -(-N // rows), B * H) if order == 0 else (-(-N // rows), -(-SQ // bm), B * H)
        sid_m = _launch.register_spec("fwd|" + key, mm, mm["kernel"]["name"], mm["kernel"]["shared"], mm["kernel"]["num_warps"], grid,
                                      kinds, ivals, fvals, mm["n_trailing_null"])
        main_call = _launch.launch_call(sid_m, [jax.ShapeDtypeStruct(tuple(int(x) for x in q.shape), el)])
        row_label = "k2b_aot[%s %s]" % (arch, kmain)

        def raw(qq, kk, vv, bb, mm_):
            b32 = bb.astype(f32)
            pouts = prep_call(b32)
            ins = [qq, kk, vv, pouts[0]]
            if b16:
                ins += [pouts[1], pouts[2]]
            if has_mask:
                ins.append(mm_)
            return main_call(*ins)[0]

        fn = _forward_only(raw, row_label, has_mask)
        _CALLS[key] = fn
    SERVED[kmain] = SERVED.get(kmain, 0) + 1
    return fn(q, k, v, bias, mask_u8)


def _forward_only(raw, row_label: str, has_mask: bool):
    """raw wrapped so that any differentiation (jvp, and therefore grad / vjp) raises Refused by name; the mask is a non-differentiable operand."""
    import jax

    @jax.custom_jvp
    def core(qq, kk, vv, bb, mm_):
        return raw(qq, kk, vv, bb, mm_ if has_mask else None)

    @core.defjvp
    def _jvp(primals, tangents):
        raise Refused("triattn_xla: row %s is forward-only (no JVP/VJP is defined); a program that differentiates triangle attention keeps %s "
                      "(or a forward+backward Pallas kernel) for those calls" % (row_label, FALLBACK))

    def f(qq, kk, vv, bb, mm_):
        import jax.numpy as jnp
        dummy = mm_ if has_mask else jnp.zeros((1,), jnp.uint8)
        return core(qq, kk, vv, bb, dummy)
    return f


def status() -> dict:
    idx = cubin_index()
    per = {}
    for b in idx.values():
        per.setdefault(b["arch"], []).append(b["key"])
    return {"cubins": {a: sorted(v) for a, v in per.items()}, "triton": sorted({b.get("triton", "?") for b in idx.values()}), "served": dict(SERVED),
            "bias16_min_sk": _k2b_consts().get("bias16_min_sk")}
