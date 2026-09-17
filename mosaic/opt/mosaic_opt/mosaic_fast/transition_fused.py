"""F9 `transition_fused` — the trunk pair TRANSITION (LayerNorm → fc1|fc2 → silu(a)·b → fc3, bias-free; joltz.Transition) served by two fused
Pallas (Triton lowering) kernels, FORWARD and BACKWARD, on P7's bfloat16 operands: the [rows, 4c] intermediates (a, b, silu(a)·b, their
cotangents) never round-trip HBM inside the forward, and are written once (bf16) by the backward kernel for the three weight-gradient GEMMs
XLA runs over them. XLA's own chain for this sub-layer is memory-bound; what the kernels save is the intermediates' HBM traffic.

Routes (the spec IS the route word; WORDS):
  `zres`  — the default setting: P7's transition LINE. `mosaic.fast.halfpair` (P7) hands this lever the block's float32 pair activation z and
            the bf16-cast module at its one extension point (`halfpair.set_tz_body`): the kernels read z (f32), cast to bf16 in registers, and
            write z + f32(bf16(transition)) — P7's cast-in, cast-out AND the residual add inside the kernel, with P7's own rounding points
            (x = bf16(z); bf16 GEMM outputs; the sub-layer output rounded to bf16 before the float32 add; in the backward the transition's
            cotangent is bf16(g) and dz = g + f32(bf16(dx))). Served only where P7 runs `tz` (its `pf` word does): P7 off, or `tz` outside its
            spec → this lever steps aside BY NAME (`aside=p7_tz_off`, nothing rebound is reached in bf16).
  `sub`   — the named alternative: `joltz.Transition.__call__` rebound; a call whose activation arrives in bfloat16 (P7 cast it) runs ONE forward
            kernel (bf16 in / bf16 out) with the fused backward; P7 keeps its casts and the block its residual add (XLA).
  `stock` | `off` — nothing served (the rebound call is joltz's own arithmetic by identity routing; the P7 slot released).
A float32 call (the sequence transition, the MSA module's transitions unless P7 `msa`, P7 off) is joltz's own body, COUNTED (`f32_stock`).
A bf16 call outside the served envelope (fc bias present, hidden != fc2's, C % 16, hidden % tile) runs joltz's body / P7's line, COUNTED by
reason (`unserved`, `unserved_reason`) — never silent.

Arithmetic (both routes). LayerNorm statistics in float32 (two-pass variance, equinox's), scale/offset the module's bf16 values; fc1/fc2/fc3 as
bf16 tensor-core dots with float32 accumulation, GEMM outputs rounded to bf16 where XLA materialises them (a, b, dh; knob `rnd`); silu and
the gating product in float32, h rounded to bf16 once; the backward recomputes a, b per row tile (nothing of the forward is saved: the op is
its own rematerialisation — under P5 `sub` it needs no checkpoint wrapper), does the LayerNorm backward in registers, and leaves dWa / dWb /
dW3 to three XLA GEMMs over the bf16 planes it writes. Numerics class `fast`: the same rounding points as P7's XLA chain with fewer bf16
roundings of intermediates, summation order differs — not bitwise with stock.

Refusals (by name, before anything is traced): `unknown_spec`; `not_installed`; `no_gpu` (jax's default backend is not a GPU: the kernels
are Triton lowerings); `probe_failed:<kind>` (the kernels do not lower / compile / agree with the XLA arithmetic on a small probe problem on this device);
`joltz_missing`.

    from mosaic.fast import transition_fused as F9
    F9.install(); F9.configure(None)          # None = SETTING ('zres'); 'sub' = the named alternative; 'stock' = off
    F9.describe()                             # flat single-token record: on, spec, route, probe, cfg, served_zres, served_sub, f32_stock, unserved, …
    F9.emit_line(tag); F9.gate()              # the census line; the fail-closed gate

Uniform lever API (mosaic_opt.levers): install / uninstall / configure(None = SETTING) / describe / gate / emit_line / ENV_REQUIRED.
"""
import sys
import collections
import functools
from typing import Any, Dict, Optional

LEVER = "F9"
NAME = "transition_fused"
MODULE = "mosaic.fast.transition_fused"
KLASS = "fast"
VERSION = "1.0"
KERNEL = f"transition_fused_pallas@{VERSION}"
SETTING = "zres"                                                  # the default setting: configure(None) applies it
WORDS = ("zres", "sub")                                           # the route words (module docstring)
OFF_WORDS = ("stock", "off", "none", "xla")
SPEC_WORDS = "'zres' (default: P7's transition line, casts + residual folded) | 'sub' (Transition.__call__ on bf16 calls) | 'stock'; optional `:k=v` kernel settings (t tf num_warps num_stages sb tb tfb num_warps_b num_stages_b t2 rnd)"
HALFPAIR_MODULE = "mosaic.fast.halfpair"                          # P7: its extension point set_tz_body / its STATE (regions)
ENV_REQUIRED: Dict[str, str] = {}
CFG: Dict[str, int] = dict(t=64, tf=64, num_warps=4, num_stages=2, sb=0, tb=32, tfb=64, num_warps_b=4, num_stages_b=2, t2=32, rnd=1)    # defaults tuned on sm_90, used on every cc unless CFG_CC overrides: ONE fused backward kernel (sb=0) with 32-row tiles; sb=1 selects the split backward (k_b1 / k_b2)
CFG_CC: Dict[str, Dict[str, int]] = {}                                                     # per-compute-capability overrides of CFG (empty: the defaults fit sm_80 and sm_90 alike)
CFG_KEYS = tuple(CFG)

STATE: Dict[str, Any] = {"installed": False, "on": False, "spec": None, "route": "none", "probe": "unprobed", "cfg": dict(CFG), "slot": "none"}
CENSUS: "collections.Counter[str]" = collections.Counter()
SHAPES: Dict[str, int] = {}
_ORIG: Dict[str, Any] = {}
_NS: Dict[str, Any] = {}


class Refusal(RuntimeError):
    """A named refusal: `.reason` is the single word the arm fails by."""
    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        super().__init__(f"{LEVER} {NAME}: {reason}" + (f" — {detail}" if detail else ""))


def _token(v) -> str:
    s = str(v)
    for a, b in ((" ", ""), ("\t", ""), ("\n", ""), ("'", ""), ('"', "")):
        s = s.replace(a, b)
    return s or "none"


# --------------------------------------------------------------------------------------------------------------------------- the grammar
def parse(spec: Optional[str]) -> Dict[str, Any]:
    """None → SETTING; 'zres' | 'sub' [':k=v' ...]; 'stock'/'off' → off. Anything else → Refusal('unknown_spec')."""
    text = SETTING if spec is None else str(spec).strip().lower()
    if text in OFF_WORDS or text == "":
        return {"on": False, "spec": "stock", "route": "none", "cfg": dict(CFG)}
    head, *kvs = text.split(":")
    if head not in WORDS:
        raise Refusal("unknown_spec", f"{spec!r}: expected {SPEC_WORDS}")
    cfg = dict(CFG)
    for kv in kvs:
        k, _, v = kv.partition("=")
        if k not in CFG_KEYS or not v.lstrip("-").isdigit():
            raise Refusal("unknown_spec", f"{spec!r}: setting {kv!r} is not one of {CFG_KEYS} with an integer value")
        cfg[k] = int(v)
    canon = head + "".join(f":{k}={cfg[k]}" for k in CFG_KEYS if cfg[k] != CFG[k])
    return {"on": True, "spec": canon, "route": head, "cfg": cfg}


# --------------------------------------------------------------------------------------------------------------------------- the kernels
def _ns():
    """Build (once) the Pallas kernels and the two custom_vjp ops over jax; imported on first use, never at import."""
    if _NS:
        return _NS
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as plgpu
    CP = getattr(plgpu, "CompilerParams", None) or getattr(plgpu, "TritonCompilerParams")
    F32, BF16 = jnp.float32, jnp.bfloat16

    def dot(a, b):
        return jnp.dot(a, b, preferred_element_type=F32)

    def dot_t(a, bt):                                            # [M,K] @ [N,K]^T
        return jax.lax.dot_general(a, bt, (((1,), (1,)), ((), ())), preferred_element_type=F32)

    def dot_tn(at, b):                                           # [K,M]^T @ [K,N]  (contract the leading axes: the long row reduction of a weight gradient)
        return jax.lax.dot_general(at, b, (((0,), (0,)), ((), ())), preferred_element_type=F32)

    def rt(v, rnd):                                              # a GEMM output as XLA materialises it (bf16) or kept f32
        return v.astype(BF16).astype(F32) if rnd else v

    def valid_rows(t, m_rows, ragged):                           # [t, 1] bool row mask of this program's tile, or None when the rows divide by the tile (no mask traced)
        if not ragged:
            return None
        rows = pl.program_id(0) * t + jnp.arange(t)
        return (rows < m_rows)[:, None]

    def ld(ref, valid):
        return ref[...] if valid is None else plgpu.load(ref, mask=jnp.broadcast_to(valid, ref.shape), other=0.0)

    def st(ref, val, valid):
        if valid is None:
            ref[...] = val
        else:
            plgpu.store(ref, val.astype(ref.dtype), mask=jnp.broadcast_to(valid, ref.shape))

    def ln_fwd(x, s, o, eps):
        mean = jnp.mean(x, axis=1, keepdims=True)
        d = x - mean
        var = jnp.maximum(jnp.mean(d * d, axis=1, keepdims=True), 0.0)
        rstd = jax.lax.rsqrt(var + eps)
        xhat = d * rstd
        return xhat, rstd, (xhat * s[None, :] + o[None, :]).astype(BF16)

    def swiglu_fwd(xn, wa_ref, wb_ref, w2_ref, nf, tf, rnd, T, C):
        def body(f, acc):
            fs = pl.ds(f * tf, tf)
            a = rt(dot(xn, wa_ref[:, fs]), rnd)
            b = rt(dot(xn, wb_ref[:, fs]), rnd)
            c = (a * jax.nn.sigmoid(a) * b).astype(BF16)
            return acc + dot(c, w2_ref[fs, :])
        return jax.lax.fori_loop(0, nf, body, jnp.zeros((T, C), F32))

    def swiglu_bwd(xn, g, wa_ref, wb_ref, w2_ref, da_ref, db_ref, c_ref, nf, tf, rnd, T, C, valid):
        def body(f, dxn):
            fs = pl.ds(f * tf, tf)
            a = rt(dot(xn, wa_ref[:, fs]), rnd)
            b = rt(dot(xn, wb_ref[:, fs]), rnd)
            sig = jax.nn.sigmoid(a)
            sa = a * sig
            st(c_ref.at[:, fs], (sa * b).astype(BF16), valid)
            dc = rt(dot_t(g, w2_ref[fs, :]), rnd)
            db = (dc * sa).astype(BF16)
            da = (dc * b * (sig * (1.0 + a * (1.0 - sig)))).astype(BF16)
            st(da_ref.at[:, fs], da, valid)
            st(db_ref.at[:, fs], db, valid)
            return dxn + dot_t(da, wa_ref[:, fs]) + dot_t(db, wb_ref[:, fs])
        return jax.lax.fori_loop(0, nf, body, jnp.zeros((T, C), F32))

    def ln_bwd(dxn, s, xhat, rstd):
        dv = dxn.astype(BF16).astype(F32)                        # the materialised bf16 cotangent of the LayerNorm output
        gx = dv * s[None, :]
        mg = jnp.mean(gx, axis=1, keepdims=True)
        mgx = jnp.mean(gx * xhat, axis=1, keepdims=True)
        return rstd * (gx - mg - xhat * mgx), dv

    # ---- route `sub`: bf16 in, bf16 out
    def k_fwd(x_ref, s_ref, o_ref, wa_ref, wb_ref, w2_ref, out_ref, *, nf, tf, eps, rnd, m_rows, ragged):
        T, C = x_ref.shape
        valid = valid_rows(T, m_rows, ragged)
        x = ld(x_ref, valid).astype(F32)
        _, _, xn = ln_fwd(x, s_ref[...], o_ref[...], eps)
        st(out_ref, swiglu_fwd(xn, wa_ref, wb_ref, w2_ref, nf, tf, rnd, T, C), valid)

    def k_bwd(x_ref, s_ref, o_ref, wa_ref, wb_ref, w2_ref, g_ref, dx_ref, xn_ref, da_ref, db_ref, c_ref, dso_ref, *, nf, tf, eps, rnd, m_rows, ragged):
        T, C = x_ref.shape
        valid = valid_rows(T, m_rows, ragged)
        x = ld(x_ref, valid).astype(F32)
        s = s_ref[...]
        xhat, rstd, xn = ln_fwd(x, s, o_ref[...], eps)
        st(xn_ref, xn, valid)
        dxn = swiglu_bwd(xn, ld(g_ref, valid), wa_ref, wb_ref, w2_ref, da_ref, db_ref, c_ref, nf, tf, rnd, T, C, valid)
        dx, dv = ln_bwd(dxn, s, xhat, rstd)
        st(dx_ref, dx, valid)
        dso_ref[0:1, :] = jnp.sum(dv * xhat, axis=0, keepdims=True)
        dso_ref[1:2, :] = jnp.sum(dv, axis=0, keepdims=True)

    # ---- route `zres`: f32 z in, z + f32(bf16(transition)) out
    def k_fwdz(z_ref, s_ref, o_ref, wa_ref, wb_ref, w2_ref, out_ref, *, nf, tf, eps, rnd, m_rows, ragged):
        T, C = z_ref.shape
        valid = valid_rows(T, m_rows, ragged)
        z = ld(z_ref, valid)
        x = z.astype(BF16).astype(F32)
        _, _, xn = ln_fwd(x, s_ref[...], o_ref[...], eps)
        st(out_ref, z + swiglu_fwd(xn, wa_ref, wb_ref, w2_ref, nf, tf, rnd, T, C).astype(BF16).astype(F32), valid)

    def k_bwdz(z_ref, s_ref, o_ref, wa_ref, wb_ref, w2_ref, g_ref, dz_ref, xn_ref, da_ref, db_ref, c_ref, dso_ref, *, nf, tf, eps, rnd, m_rows, ragged):
        T, C = z_ref.shape
        valid = valid_rows(T, m_rows, ragged)
        x = ld(z_ref, valid).astype(BF16).astype(F32)
        s = s_ref[...]
        xhat, rstd, xn = ln_fwd(x, s, o_ref[...], eps)
        st(xn_ref, xn, valid)
        g32 = ld(g_ref, valid)
        dxn = swiglu_bwd(xn, g32.astype(BF16), wa_ref, wb_ref, w2_ref, da_ref, db_ref, c_ref, nf, tf, rnd, T, C, valid)
        dx, dv = ln_bwd(dxn, s, xhat, rstd)
        st(dz_ref, g32 + dx.astype(BF16).astype(F32), valid)
        dso_ref[0:1, :] = jnp.sum(dv * xhat, axis=0, keepdims=True)
        dso_ref[1:2, :] = jnp.sum(dv, axis=0, keepdims=True)

    # ---- the SPLIT backward (setting sb=1): B1 recomputes a|b per row tile and writes the bf16 planes h, da, db (no [T,C] accumulator);
    #      dv = da @ Wa^T + db @ Wb^T is left to XLA's GEMMs (bf16, at roofline); B2 = the LayerNorm backward (+ P7's cast VJPs and the residual
    #      under `zres`) per row tile.  sb=0 (the default) = the single fused backward kernel above (dv accumulated in registers, small row tiles).
    def k_b1(x_ref, s_ref, o_ref, wa_ref, wb_ref, w2_ref, g_ref, xn_ref, da_ref, db_ref, c_ref, *, nf, tf, eps, rnd, m_rows, ragged, zin):
        T, C = x_ref.shape
        valid = valid_rows(T, m_rows, ragged)
        x = ld(x_ref, valid)
        x = (x.astype(BF16) if zin else x).astype(F32)
        _, _, xn = ln_fwd(x, s_ref[...], o_ref[...], eps)
        st(xn_ref, xn, valid)
        g = ld(g_ref, valid).astype(BF16)

        def body(f, carry):
            fs = pl.ds(f * tf, tf)
            a = rt(dot(xn, wa_ref[:, fs]), rnd)
            b = rt(dot(xn, wb_ref[:, fs]), rnd)
            sig = jax.nn.sigmoid(a)
            sa = a * sig
            st(c_ref.at[:, fs], (sa * b).astype(BF16), valid)
            dc = rt(dot_t(g, w2_ref[fs, :]), rnd)
            st(db_ref.at[:, fs], (dc * sa).astype(BF16), valid)
            st(da_ref.at[:, fs], (dc * b * (sig * (1.0 + a * (1.0 - sig)))).astype(BF16), valid)
            return carry
        jax.lax.fori_loop(0, nf, body, 0)

    def k_b2(x_ref, s_ref, dv_ref, g_ref, dx_ref, dso_ref, *, eps, m_rows, ragged, zin):
        T, C = x_ref.shape
        valid = valid_rows(T, m_rows, ragged)
        x = ld(x_ref, valid)
        x = (x.astype(BF16) if zin else x).astype(F32)
        s = s_ref[...]
        xhat, rstd, _ = ln_fwd(x, s, jnp.zeros_like(s), eps)
        dx, dv = ln_bwd(ld(dv_ref, valid).astype(F32), s, xhat, rstd)
        if zin:
            st(dx_ref, ld(g_ref, valid) + dx.astype(BF16).astype(F32), valid)     # dz = g + f32(bf16(dx)): P7's cast-in VJP + the residual's identity
        else:
            st(dx_ref, dx, valid)
        dso_ref[0:1, :] = jnp.sum(dv * xhat, axis=0, keepdims=True)
        dso_ref[1:2, :] = jnp.sum(dv, axis=0, keepdims=True)

    def bwd_split_call(zin, name, x, s, o, wa, wb, w2, g, cfg):
        t, tf, t2 = cfg["tb"], cfg["tfb"], cfg["t2"]
        M, C = x.shape; F = wa.shape[1]
        G = -(-M // t); G2 = -(-M // t2)
        row = lambda i: (i, 0)
        full2 = lambda i: (0, 0)
        vec = lambda i: (0,)
        xn, da, db, c = pl.pallas_call(
            functools.partial(k_b1, nf=F // tf, tf=tf, eps=cfg["eps"], rnd=int(cfg["rnd"]), m_rows=M, ragged=bool(M % t), zin=zin), grid=(G,),
            in_specs=[pl.BlockSpec((t, C), row), pl.BlockSpec((C,), vec), pl.BlockSpec((C,), vec),
                      pl.BlockSpec((C, F), full2), pl.BlockSpec((C, F), full2), pl.BlockSpec((F, C), full2), pl.BlockSpec((t, C), row)],
            out_specs=[pl.BlockSpec((t, C), row), pl.BlockSpec((t, F), row), pl.BlockSpec((t, F), row), pl.BlockSpec((t, F), row)],
            out_shape=[jax.ShapeDtypeStruct((M, C), BF16), jax.ShapeDtypeStruct((M, F), BF16), jax.ShapeDtypeStruct((M, F), BF16), jax.ShapeDtypeStruct((M, F), BF16)],
            compiler_params=CP(num_warps=cfg["num_warps_b"], num_stages=cfg["num_stages_b"]), name=name + "_b1",
        )(x, s, o, wa, wb, w2, g)
        dv = (dot_t(da, wa) + dot_t(db, wb)).astype(BF16)                     # [M, C]: XLA GEMMs (the LayerNorm output's bf16 cotangent, as P7's chain materialises it)
        gin = g if zin else dv                                                # B2 reads g only under zres (the residual path); a placeholder otherwise
        dx, dso = pl.pallas_call(
            functools.partial(k_b2, eps=cfg["eps"], m_rows=M, ragged=bool(M % t2), zin=zin), grid=(G2,),
            in_specs=[pl.BlockSpec((t2, C), row), pl.BlockSpec((C,), vec), pl.BlockSpec((t2, C), row), pl.BlockSpec((t2, C), row)],
            out_specs=[pl.BlockSpec((t2, C), row), pl.BlockSpec((None, 2, C), lambda i: (i, 0, 0))],
            out_shape=[jax.ShapeDtypeStruct((M, C), F32 if zin else BF16), jax.ShapeDtypeStruct((G2, 2, C), F32)],
            compiler_params=CP(num_warps=4, num_stages=2), name=name + "_b2",
        )(x, s, dv, gin)
        dwa = dot_tn(xn, da).astype(wa.dtype)
        dwb = dot_tn(xn, db).astype(wb.dtype)
        dw2 = dot_tn(c, g.astype(BF16)).astype(w2.dtype)
        dso = jnp.sum(dso, axis=0)
        return dx, dso[0].astype(s.dtype), dso[1].astype(o.dtype), dwa, dwb, dw2

    def fwd_call(kern, out_dtype, name, x, s, o, wa, wb, w2, cfg):
        t, tf = cfg["t"], cfg["tf"]
        M, C = x.shape; F = wa.shape[1]
        G = -(-M // t)                                            # a ragged last tile is masked in the kernel (no pad copy of the [rows, C] planes)
        full2 = lambda i: (0, 0)
        return pl.pallas_call(
            functools.partial(kern, nf=F // tf, tf=tf, eps=cfg["eps"], rnd=int(cfg["rnd"]), m_rows=M, ragged=bool(M % t)), grid=(G,),
            in_specs=[pl.BlockSpec((t, C), lambda i: (i, 0)), pl.BlockSpec((C,), lambda i: (0,)), pl.BlockSpec((C,), lambda i: (0,)),
                      pl.BlockSpec((C, F), full2), pl.BlockSpec((C, F), full2), pl.BlockSpec((F, C), full2)],
            out_specs=pl.BlockSpec((t, C), lambda i: (i, 0)), out_shape=jax.ShapeDtypeStruct((M, C), out_dtype),
            compiler_params=CP(num_warps=cfg["num_warps"], num_stages=cfg["num_stages"]), name=name,
        )(x, s, o, wa, wb, w2)

    def bwd_call(kern, dx_dtype, name, x, s, o, wa, wb, w2, g, cfg):
        t, tf = cfg["tb"], cfg["tfb"]
        M, C = x.shape; F = wa.shape[1]
        G = -(-M // t)
        row = lambda i: (i, 0)
        full2 = lambda i: (0, 0)
        dx, xn, da, db, c, dso = pl.pallas_call(
            functools.partial(kern, nf=F // tf, tf=tf, eps=cfg["eps"], rnd=int(cfg["rnd"]), m_rows=M, ragged=bool(M % t)), grid=(G,),
            in_specs=[pl.BlockSpec((t, C), row), pl.BlockSpec((C,), lambda i: (0,)), pl.BlockSpec((C,), lambda i: (0,)),
                      pl.BlockSpec((C, F), full2), pl.BlockSpec((C, F), full2), pl.BlockSpec((F, C), full2), pl.BlockSpec((t, C), row)],
            out_specs=[pl.BlockSpec((t, C), row), pl.BlockSpec((t, C), row), pl.BlockSpec((t, F), row), pl.BlockSpec((t, F), row),
                       pl.BlockSpec((t, F), row), pl.BlockSpec((None, 2, C), lambda i: (i, 0, 0))],
            out_shape=[jax.ShapeDtypeStruct((M, C), dx_dtype), jax.ShapeDtypeStruct((M, C), BF16), jax.ShapeDtypeStruct((M, F), BF16),
                       jax.ShapeDtypeStruct((M, F), BF16), jax.ShapeDtypeStruct((M, F), BF16), jax.ShapeDtypeStruct((G, 2, C), F32)],
            compiler_params=CP(num_warps=cfg["num_warps_b"], num_stages=cfg["num_stages_b"]), name=name,
        )(x, s, o, wa, wb, w2, g)
        dwa = dot_tn(xn, da).astype(wa.dtype)                   # the weight cotangents: XLA GEMMs over the bf16 planes
        dwb = dot_tn(xn, db).astype(wb.dtype)
        dw2 = dot_tn(c, g.astype(BF16)).astype(w2.dtype)
        dso = jnp.sum(dso, axis=0)
        return dx, dso[0].astype(s.dtype), dso[1].astype(o.dtype), dwa, dwb, dw2

    def make(route, cfg):
        cfg = {**cfg, "eps": cfg.get("eps", 1e-5)}
        if route == "sub":
            fk, bk, fdt, name = k_fwd, k_bwd, BF16, "transition_fused"
        else:
            fk, bk, fdt, name = k_fwdz, k_bwdz, F32, "transition_fused_zres"

        @jax.custom_vjp
        def op(x, s, o, wa, wb, w2):
            return fwd_call(fk, fdt, name + "_fwd", x, s, o, wa, wb, w2, cfg)

        def op_fwd(x, s, o, wa, wb, w2):
            return fwd_call(fk, fdt, name + "_fwd", x, s, o, wa, wb, w2, cfg), (x, s, o, wa, wb, w2)

        def op_bwd(res, g):
            x, s, o, wa, wb, w2 = res
            if int(cfg.get("sb", 1)):
                return bwd_split_call(route != "sub", name, x, s, o, wa, wb, w2, g.astype(fdt), cfg)
            return bwd_call(bk, fdt, name + "_bwd", x, s, o, wa, wb, w2, g.astype(fdt), cfg)

        op.defvjp(op_fwd, op_bwd)
        return op

    _NS.update(jax=jax, jnp=jnp, make=make, ops={})
    return _NS


def _op(route: str):
    ns = _ns()
    key = (route, tuple(sorted(STATE["cfg"].items())))
    if key not in ns["ops"]:
        ns["ops"][key] = ns["make"](route, dict(STATE["cfg"]))
    return ns["ops"][key]


def set_eps(op_eps):                                              # (unused hook: joltz's LayerNorm eps rides the module; see _operands)
    return op_eps


# --------------------------------------------------------------------------------------------------------------------------- the adapter
def _operands(module, act_dtype_bf16=True):
    """joltz.Transition `module` → (s, o, wa, wb, w2, eps) or (None, reason). Served envelope: bias-free fc1/fc2/fc3, fc1/fc2 the same hidden
    size, C % 16 == 0, hidden % forward AND backward tile, weights bf16 (P7 cast the module)."""
    jnp = _ns()["jnp"]
    n, fc1, fc2, fc3 = module.norm, module.fc1, module.fc2, module.fc3
    if any(getattr(l, "bias", None) is not None for l in (fc1, fc2, fc3)):
        return None, "fc_bias"
    F, C = fc1.weight.shape
    if fc2.weight.shape != (F, C) or fc3.weight.shape != (C, F):
        return None, "shape"
    cfg = STATE["cfg"]
    if C % 16 or F % cfg["tf"] or F % cfg["tfb"] or C > 256:
        return None, "tile"
    if fc1.weight.dtype != jnp.bfloat16:
        return None, f"w_{jnp.dtype(fc1.weight.dtype).name}"
    s = jnp.ones((C,), jnp.float32) if n.weight is None else n.weight.astype(jnp.float32)
    o = jnp.zeros((C,), jnp.float32) if n.bias is None else n.bias.astype(jnp.float32)
    return (s, o, fc1.weight.T, fc2.weight.T, fc3.weight.T, float(n.eps)), None


def _shape_note(kind: str, x):
    key = f"{kind}:M{int(x.size // x.shape[-1])}:C{int(x.shape[-1])}"
    if key in SHAPES or len(SHAPES) < 12:
        SHAPES[key] = SHAPES.get(key, 0) + 1


def _apply(route: str, module, x):
    """The fused op on x [..., C] (bf16 for `sub`, f32 for `zres`); None when the call is outside the served envelope (reason counted)."""
    ops, reason = _operands(module)
    if ops is None:
        CENSUS["unserved"] += 1
        CENSUS[f"unserved:{reason}"] += 1
        STATE["unserved_reason"] = reason
        return None
    s, o, wa, wb, w2, eps = ops
    if abs(eps - 1e-5) > 1e-12:
        STATE["cfg"] = {**STATE["cfg"], "eps": eps}
    C = x.shape[-1]
    lead = x.shape[:-1]
    out = _op(route)(x.reshape(-1, C), s, o, wa, wb, w2)
    CENSUS[f"served_{route}"] += 1
    _shape_note(route, x)
    return out.reshape(*lead, C)


def _transition_call(self, x):
    """`joltz.Transition.__call__` with the lever: a bfloat16 activation (P7's cast) → the fused op (route `sub`, or under `zres` too when a bf16
    call reaches the class outside P7's line); float32 → joltz's own body, counted."""
    orig = _ORIG["transition"]
    if not STATE["on"] or STATE["route"] == "aside":                     # off, or stood aside on this device (probe_failed): joltz's own body
        return orig(self, x)
    jnp = _ns()["jnp"]
    if x.dtype != jnp.bfloat16:
        CENSUS["f32_stock"] += 1
        return orig(self, x)
    out = _apply("sub", self, x)
    return orig(self, x) if out is None else out


def _tz_body(module16, z32):
    """P7's transition line under `zres`: z32 [.., N, N, C] float32 → z32 + f32(bf16(transition(bf16(z32)))) by the kernels; NotImplemented
    (P7 runs its own line, counted here) outside the envelope."""
    jnp = _ns()["jnp"]
    if not (STATE["on"] and STATE["route"] == "zres") or z32.dtype != jnp.float32:
        CENSUS["slot_declined"] += 1
        return NotImplemented
    out = _apply("zres", module16, z32)
    return NotImplemented if out is None else out


def _cc() -> str:
    """The default device's compute capability ('9.0', '8.0', …) or 'none' (no GPU / unknown)."""
    try:
        jax = _ns()["jax"]
        d = jax.devices()[0]
        return str(getattr(d, "compute_capability", "none") or "none")
    except Exception:  # noqa: BLE001
        return "none"


# --------------------------------------------------------------------------------------------------------------------------- probe
def _probe(route: str) -> str:
    """Lower + compile + run both kernels of `route` on a 200-row problem (a padded tile) and compare with the XLA arithmetic; 'ok' or a reason token."""
    try:
        ns = _ns(); jax, jnp = ns["jax"], ns["jnp"]
        if jax.default_backend() != "gpu":
            return "no_gpu"
        C, F, M = 128, 512, 200
        k = jax.random.split(jax.random.PRNGKey(7), 6)
        z = jax.random.normal(k[0], (M, C), jnp.float32)
        s = 1.0 + 0.1 * jax.random.normal(k[1], (C,)); o = 0.1 * jax.random.normal(k[2], (C,))
        wa = (jax.random.normal(k[3], (C, F)) / C ** 0.5).astype(jnp.bfloat16); wb = (jax.random.normal(k[4], (C, F)) / C ** 0.5).astype(jnp.bfloat16)
        w2 = (jax.random.normal(k[5], (F, C)) / F ** 0.5).astype(jnp.bfloat16)
        sb, ob = s.astype(jnp.bfloat16).astype(jnp.float32), o.astype(jnp.bfloat16).astype(jnp.float32)

        def ref(zz, wa, wb, w2):                                  # P7's XLA chain + residual
            x = zz.astype(jnp.bfloat16).astype(jnp.float32)
            m = x.mean(-1, keepdims=True); v = jnp.maximum(((x - m) ** 2).mean(-1, keepdims=True), 0.0)
            xn = ((x - m) * jax.lax.rsqrt(v + 1e-5) * sb + ob).astype(jnp.bfloat16)
            a = jnp.dot(xn, wa); b = jnp.dot(xn, wb)
            out = jnp.dot((jax.nn.silu(a) * b), w2)
            return zz + out.astype(jnp.float32) if route == "zres" else out.astype(jnp.float32)

        op = _op(route)

        def fused(zz, wa, wb, w2):
            xin = zz if route == "zres" else zz.astype(jnp.bfloat16)
            return op(xin, sb, ob, wa, wb, w2).astype(jnp.float32)

        g = jax.random.normal(k[0], (M, C), jnp.float32)
        lr = lambda f: (lambda zz, wa, wb, w2: jnp.sum(f(zz, wa, wb, w2) * g))
        o1 = jax.jit(fused)(z, wa, wb, w2); o0 = jax.jit(ref)(z, wa, wb, w2)
        g1 = jax.jit(jax.grad(lr(fused), argnums=(0, 1, 2, 3)))(z, wa, wb, w2); g0 = jax.jit(jax.grad(lr(ref), argnums=(0, 1, 2, 3)))(z, wa, wb, w2)

        def rel(a, b):
            a = a.astype(jnp.float32); b = b.astype(jnp.float32)
            return float(jnp.linalg.norm(a - b) / (jnp.linalg.norm(b) + 1e-30))
        errs = [rel(o1, o0)] + [rel(a, b) for a, b in zip(g1, g0)]
        if not all(e == e and e < 0.05 for e in errs):
            return _token(f"mismatch_{max(errs):.3g}")
        return "ok"
    except Exception as e:  # noqa: BLE001
        return _token(f"{type(e).__name__}")[:60]


# --------------------------------------------------------------------------------------------------------------------------- lever API
def _joltz():
    import importlib
    try:
        return importlib.import_module("joltz")
    except ImportError as e:
        raise Refusal("joltz_missing", f"{type(e).__name__}: {e}") from None


def _halfpair():
    import importlib
    try:
        return importlib.import_module(HALFPAIR_MODULE)
    except ImportError:
        return None


def install():
    """Idempotent: rebinds `joltz.Transition.__call__` (identity-routing to joltz's own until configured on). The P7 slot is taken at configure."""
    if STATE["installed"]:
        return
    J = _joltz()
    _ORIG["transition"] = J.Transition.__call__
    J.Transition.__call__ = _transition_call
    STATE["installed"] = True


def installed() -> bool:
    return bool(STATE["installed"])


def uninstall():
    H = _halfpair()
    if H is not None and hasattr(H, "set_tz_body") and getattr(H, "TZ_SLOT", {}).get("fn") is _tz_body:
        H.set_tz_body(None)
    if STATE["installed"]:
        J = _joltz()
        if "transition" in _ORIG:
            J.Transition.__call__ = _ORIG.pop("transition")
    STATE.update(installed=False, on=False, spec=None, route="none", probe="unprobed", cfg=dict(CFG), slot="none")
    STATE.pop("unserved_reason", None); STATE.pop("cc", None); STATE.pop("aside_probe", None)
    CENSUS.clear(); SHAPES.clear()
    if _NS:
        _NS.get("ops", {}).clear()


def configure(spec=None) -> Dict[str, Any]:
    """On → probe the route's kernels by name (`no_gpu`, `probe_failed:<kind>` BEFORE any model trace), take P7's slot under `zres`; off → release.
    Starts a fresh census; returns describe()."""
    cfg = parse(spec)
    if not installed():
        raise Refusal("not_installed", f"configure({spec!r}) before install()")
    CENSUS.clear(); SHAPES.clear(); STATE.pop("unserved_reason", None)
    H = _halfpair()
    if H is not None and hasattr(H, "set_tz_body") and getattr(H, "TZ_SLOT", {}).get("fn") is _tz_body:
        H.set_tz_body(None)
    STATE["cfg"] = dict(cfg["cfg"])
    if _NS:
        _NS.get("ops", {}).clear()
    if cfg["on"]:
        cc = _cc()
        STATE["cc"] = cc
        for k, v in CFG_CC.get(cc, {}).items():                        # the card's defaults for settings the spec leaves at the module default
            if f":{k}=" not in cfg["spec"]:
                STATE["cfg"][k] = v
        verdict = _probe(cfg["route"])
        STATE["probe"] = verdict
        if verdict == "no_gpu":
            raise Refusal("no_gpu", "the fused transition kernels are Pallas-Triton GPU lowerings; jax's default backend is not a GPU")
        if verdict != "ok":
            STATE["probe"] = f"failed:{verdict}"; STATE["aside_probe"] = f"probe_failed:{verdict}"   # the kernels do not lower / compile / agree on this device (e.g. another
            STATE.update(on=True, spec=cfg["spec"], route="aside", slot="none")                 # compute capability): the lever steps aside BY NAME — no slot taken, no call re-routed: the transition runs
            sys.stderr.write(f"[mosaic-opt] {LEVER}: probe {verdict} on this device — stepping aside by name (aside=probe_failed:{verdict}); the pair transition runs its XLA body\n")
            return describe()                                                                # joltz's / P7's XLA body; describe()/the TRANSITION line say so; the gate passes on the aside
        slot = "none"
        if cfg["route"] == "zres":
            if H is not None and hasattr(H, "set_tz_body"):
                H.set_tz_body(_tz_body, f"{LEVER}:{cfg['route']}")
                slot = "halfpair"
            else:
                slot = "absent"                                   # an older P7 without the extension point: `zres` cannot engage — bf16 calls still reach `sub` through the class
        STATE.update(on=True, spec=cfg["spec"], route=cfg["route"], slot=slot)
    else:
        STATE.update(on=False, spec="stock", route="none", slot="none", probe=STATE.get("probe", "unprobed"))
    return describe()


def _p7_tz() -> str:
    """What P7 says about the transition region right now: 'on' | 'off' | 'p7_off' | 'absent'."""
    H = _halfpair()
    if H is None:
        return "absent"
    st = getattr(H, "STATE", {})
    if not st.get("on"):
        return "p7_off"
    return "on" if "tz" in tuple(st.get("regions", ())) else "off"


def _served() -> int:
    return int(CENSUS.get("served_zres", 0) + CENSUS.get("served_sub", 0))


def describe() -> Dict[str, Any]:
    """FLAT single-token record in every state."""
    c = STATE["cfg"]
    p7 = _p7_tz() if STATE["installed"] else "unknown"
    aside = STATE.get("aside_probe") or "none"                                          # probe_failed:<kind> = stood aside on this device (the XLA body runs), else p7_tz_off / none below
    seen = _served() + int(CENSUS.get("f32_stock", 0)) + int(CENSUS.get("unserved", 0)) + int(CENSUS.get("slot_declined", 0))
    if aside == "none" and STATE["on"] and _served() == 0 and seen > 0 and p7 in ("off", "p7_off", "absent"):   # transition calls traced, none in bf16 through the kernels, P7's tz off: stood aside by name
        aside = "p7_tz_off"
    out: Dict[str, Any] = {
        "lever": LEVER, "lever_name": NAME, "module": MODULE, "klass": KLASS, "installed": int(installed()), "on": int(bool(STATE["on"])),
        "spec": STATE["spec"] or ("stock" if installed() else "none"), "route": STATE["route"], "slot": STATE["slot"], "p7_tz": p7, "aside": aside,
        "probe": _token(STATE["probe"]), "impl": KERNEL if STATE["on"] else MODULE, "origin": "kit", "backend": "pallas_triton",
        "fwd_tile": f"{c['t']}x{c['tf']}w{c['num_warps']}s{c['num_stages']}", "bwd_tile": f"{c['tb']}x{c['tfb']}w{c['num_warps_b']}s{c['num_stages_b']}",
        "bwd_form": ("split" if int(c.get("sb", 1)) else "fused"), "t2": int(c.get("t2", 32)),
        "rnd": int(c["rnd"]), "cc": _token(STATE.get("cc", "unprobed")), "ln_stats": "f32", "operands": "bf16", "accumulate": "f32", "residuals": "inputs_only",
        "served": _served(), "served_zres": int(CENSUS.get("served_zres", 0)), "served_sub": int(CENSUS.get("served_sub", 0)),
        "f32_stock": int(CENSUS.get("f32_stock", 0)), "unserved": int(CENSUS.get("unserved", 0)),
        "unserved_reason": _token(STATE.get("unserved_reason", "none")), "slot_declined": int(CENSUS.get("slot_declined", 0)),
        "shapes": _token("+".join(f"{k}={v}" for k, v in sorted(SHAPES.items())) or "none"),
    }
    return out


def census() -> Dict[str, int]:
    return dict(CENSUS)


def emit_line(tag: str = "") -> str:
    """`[mosaic-opt] TRANSITION tag=<t> lever=F9 spec=<s> route=<r> served_zres=<n> served_sub=<n> f32_stock=<n> unserved=<n>(<reason>) aside=<w> kernel=<k> tiles=<f>/<b> shapes=<…>`"""
    d = describe()
    line = (f"[mosaic-opt] TRANSITION tag={_token(tag) if tag else 'none'} lever={LEVER} spec={d['spec']} route={d['route']} slot={d['slot']} "
            f"served_zres={d['served_zres']} served_sub={d['served_sub']} f32_stock={d['f32_stock']} unserved={d['unserved']}({d['unserved_reason']}) "
            f"aside={d['aside']} kernel={d['impl']} tiles={d['fwd_tile']}/{d['bwd_tile']} rnd={d['rnd']} probe={d['probe']} shapes={d['shapes']}")
    print(line, flush=True)
    return line


def gate():
    """Fail closed: installed but never configured on; configured on, P7 served `tz` blocks, and nothing came through the kernels (a defect: the
    run would carry the lever's name on XLA's transition); any unserved bf16 call (an envelope miss is named, and on this model it is a defect).
    P7 off / `tz` outside P7's spec / no P7 → the lever stood aside BY NAME (describe: aside=p7_tz_off) and the gate passes."""
    if not STATE["on"]:
        if installed() and STATE["spec"] is None:
            raise Refusal("not_configured", f"{LEVER} is installed but configure() was never called")
        return
    d = describe()
    if d["unserved"]:
        raise Refusal("unserved_calls", f"{d['unserved']} bf16 transition call(s) outside the served envelope ({d['unserved_reason']})")
    if d["served"] == 0 and d["aside"] == "none":
        raise Refusal("nothing_served", f"{LEVER} configured on ({d['spec']}), P7 tz={d['p7_tz']}, and no pair transition was traced through the kernels")


PATCHED = ("joltz.Transition.__call__",)
