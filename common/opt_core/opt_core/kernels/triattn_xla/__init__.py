"""triattn_xla -- triangle attention (pair bias + key mask, forward) for JAX / XLA programs, served by pre-compiled GPU kernels
through one XLA-FFI launcher.  No compiler runs at import or at prediction time: the package ships the binaries (bin/, listed with
their sha256 in manifest.json) and refuses BY NAME whatever they do not cover.

    out = triangle_attention(q, k, v, bias, mask=None, scale=None, *, impl="auto", layout="BNHSD", vjp=None)

  q, k, v : [B, N, H, S, D] ("BNHSD", the cuEquivariance / K2B convention) or [B, N, S, H, D] ("BNSHD", the AlphaFold-family
            module layout); rank 4 = no batch.  N = pair rows, S = queries == keys of one row.  Dense (row-major) device arrays.
  bias    : [B, H, S, S] or [B, 1, H, S, S] or [H, S, S]; fp32 / bf16 / fp16; shared by the N rows of a batch element.
  mask    : [B, N, S] (or [B, N, 1, 1, S] / [N, S]) bool or uint8, True/1 = attend; None = no mask.  A row with no attended key
            attends uniformly (mean of v) -- the cuEquivariance / K2B semantics.
  scale   : softmax scale, default D ** -0.5.
  returns : an array of q's shape, dtype and layout.

Rows (implementations), each individually switchable off through MODEL_OPT_LEVERS_OFF (words `triattn_xla` = all,
`triattn_xla:cuda_sm90a`, `triattn_xla:k2b_aot`):
  cuda_sm90a  the hand-written sm_90a CUDA kernel (bin/cuda/sm_90a/; sources kernels/triattn/cuda_sm90a/csrc/, compiled unmodified): compute capability 9.0 only, bf16, D = 32,
              any S / N / H, mask or none.  Numerics class: bf16 tensor-core products, fp32 softmax and bias -- the class of the K2B
              Triton kernel and of cuEquivariance; deterministic.
  k2b_aot     the K2B Triton kernel of kernels/fpf_triatt_k2b compiled ahead of time (bin/k2b/sm_90, bin/k2b/sm_80): compute capability
              9.0 (sm_90 cubins) and 8.x (sm_80 cubins), bf16 (base-2 softmax domain) and fp32 (TF32 products, natural domain), D in
              {16, 32}; identical arithmetic to the torch-launched kernel (same compiler, same launch cells, same constexpr set).
Selection (impl="auto"): CELLS.json `by_cc[<cc>].order`, first row that serves the call; a row that cannot serve says why; when none can,
`Refused` is raised naming every row's reason and the fallback (`stock: xla` -- the program's own einsum/softmax), never a silent
substitute.  `select()` answers the same question without arrays; `report()` lists what loaded and what was served or refused.

UNDER jax.vmap (module _batching): every launch carries a batching rule -- the vmapped axis is folded into the kernel batch axis B and
launched once (operands that are not vmapped are broadcast; a folded call outside a row's envelope is launched per sample instead), through
jax.custom_batching where the running jax has it; every FFI call is also built with vmap_method="sequential" so a vmapped trace the rule
does not cover still runs one launch per sample.  Either way each sample gets the bytes an unbatched call gives.  report()["vmap"].

FORWARD ONLY BY DEFAULT (vjp=None).  The rows above define no VJP/JVP: differentiating through them (jax.grad / jax.vjp / jax.jvp)
raises `Refused` naming the row.  No vmap rule either (batch through B / N).

THE DIFFERENTIABLE ROW is reached only by word: vjp="auto" (or the name of one backward: "attbwd", "flash", "flash_xla"; module _vjp).
Forward = on compute capability 9.0 the sm_90a CUDA kernel built WITH a per-row log-sum-exp store (bin/cuda/sm_90a/
libtriattn_mw_cuda_lse.so = the sealed sources + csrc/cuda_lse.patch; output bit-identical to row cuda_sm90a's), elsewhere the K2B
kernel compiled with the store (bin/k2b/sm_*/k2bl_fwd_*: output bit-identical to row k2b_aot's); residuals out + lse, no logits kept; backward = Pallas kernels lowered by the running jax, chosen per (compute
capability, dtype) from CELLS.json "vjp": `attbwd` (kernels/pallas_triatt: dK/dV over key blocks, dQ + row-summed d(bias) partials over
query blocks) or `flash` / `flash_xla` (kernels/pallas_attn's backward, d(bias) summed in-kernel / per-row partials reduced by XLA).
Gradients: dq, dk, dv in the inputs' dtype, d(bias) = the sum over the N rows in the bias' dtype and shape; the mask takes none.  Serves
layout BNHSD, B == 1, head_dim 32, bf16 and fp32 (TF32 products), compute capability 9.0 / 8.x; anything else -- or a jax on which the
backward's Pallas modules do not import -- is refused BY NAME with the fallback (XLA autodiff of the program's own einsum/softmax
attention) and the reason; MODEL_OPT_LEVERS_OFF word `triattn_xla:vjp` switches the differentiable row off.

Launcher: bin/launcher/ffi-<major>.<minor>/libtriattn_xla_launch.so, one build per XLA-FFI API version of the jaxlib headers (the
version a jaxlib carries is read from its own include directory at load); registered with jax.ffi (typed FFI, api_version=1).  A jax
without `jax.ffi`, or a jaxlib whose FFI API version has no build here, is refused by name with the exact error.
"""
import json
import math
import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ... import cell_census as _CENSUS                   # stdlib-only: the coverage census (one record per decided call class; pure observation)

__all__ = ["triangle_attention", "select", "report", "reference", "cells", "manifest", "Refused", "ROWS", "VJP_WORDS", "FALLBACK", "levers_off",
           "compute_capability", "vector_cases", "vector_inputs", "__version__"]
__version__ = "1.6.1"

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROWS = ("triattn_native", "cuda_sm90a", "cuda_80", "k2b_aot")
VJP_WORDS = ("auto", "attbwd", "flash", "flash_xla")          # vjp=<word>: the differentiable row (module _vjp); None = the forward-only rows
FALLBACK = "stock: xla"
LEVER = "triattn_xla"
_CELLS: Optional[dict] = None
_MANIFEST: Optional[dict] = None
COUNTS: Dict[str, int] = {"triattn_native": 0, "cuda_sm90a": 0, "cuda_80": 0, "k2b_aot": 0, "vjp": 0, "refused": 0}
LAST_REFUSAL: Dict[str, str] = {}


class Refused(NotImplementedError):
    """The package cannot serve this call (card, dtype, shape, missing binary, switched off, differentiation).  `.fallback` names what the
    caller keeps using; nothing was substituted."""

    def __init__(self, msg: str, fallback: str = FALLBACK, reasons: Optional[Dict[str, str]] = None):
        NotImplementedError.__init__(self, msg)
        self.fallback = fallback
        self.reasons = dict(reasons or {})


# --------------------------------------------------------------------------------------------------------------------- tables
def cells() -> dict:
    global _CELLS
    if _CELLS is None:
        with open(os.path.join(PKG_DIR, "CELLS.json"), encoding="utf-8") as fh:
            _CELLS = json.load(fh)
    return _CELLS


def manifest() -> dict:
    global _MANIFEST
    if _MANIFEST is None:
        with open(os.path.join(PKG_DIR, "manifest.json"), encoding="utf-8") as fh:
            _MANIFEST = json.load(fh)
    return _MANIFEST


def binaries(kind: Optional[str] = None) -> List[dict]:
    return [b for b in manifest().get("binaries", []) if kind is None or b.get("kind") == kind]


def levers_off() -> List[str]:
    """Words of MODEL_OPT_LEVERS_OFF (comma / space / colon-free separated) that address this package."""
    raw = os.environ.get("MODEL_OPT_LEVERS_OFF", "")
    words = [w.strip() for w in raw.replace(";", ",").replace(" ", ",").split(",") if w.strip()]
    return [w for w in words if w == LEVER or w.startswith(LEVER + ":")]


def row_switched_off(row: str) -> Optional[str]:
    off = levers_off()
    if LEVER in off:
        return "switched off by MODEL_OPT_LEVERS_OFF=" + LEVER
    if (LEVER + ":" + row) in off:
        return "switched off by MODEL_OPT_LEVERS_OFF=" + LEVER + ":" + row
    return None


def compute_capability(device: Any = None) -> str:
    """'<major>.<minor>' of a jax device (default: the first device of the default backend); TRIATTN_XLA_CC overrides (tests, tracing
    on a host whose default device is not the target)."""
    env = os.environ.get("TRIATTN_XLA_CC")
    if env:
        return env.strip()
    if device is None:
        import jax
        device = jax.devices()[0]
    cc = getattr(device, "compute_capability", None)
    if cc is None:
        return "none:" + str(getattr(device, "platform", "?"))
    return str(cc)


def arch_for_cc(cc: str) -> Optional[str]:
    """The cubin architecture that serves a compute capability: by_cc entry, else the same-major rule for 8.x (sm_80 cubins run on 8.6 / 8.9)."""
    ent = cells().get("by_cc", {}).get(cc)
    if ent and ent.get("arch"):
        return ent["arch"]
    if cc.startswith("8."):
        return "sm_80"
    return None


def row_order(cc: str) -> List[str]:
    ent = cells().get("by_cc", {}).get(cc)
    if ent:
        return list(ent.get("order", []))
    if cc.startswith("8."):
        return list(cells().get("by_cc", {}).get("8.0", {}).get("order", ["k2b_aot"]))
    return []


# --------------------------------------------------------------------------------------------------------------------- selection
def rule_refusal(cc: str, row: str, dtype: str, head_dim: int, S: int, has_mask: Optional[bool] = None) -> Optional[str]:
    """A measured preference among rows that CAN serve (CELLS.json by_cc[cc].rules): e.g. below some S another row is faster.  Applies to
    impl="auto" only; naming a row explicitly bypasses it.  A row's rules may carry a `masked` sub-table (min_S / max_S_excl / why) that
    replaces the row-level edges when the call carries a key mask (the form whose cells set it), and `min_S_by_head_dim`."""
    rules = cells().get("by_cc", {}).get(cc, {}).get("rules", {}).get(row, {})
    if has_mask and isinstance(rules.get("masked"), dict):
        rules = dict(rules, **rules["masked"])
    min_s = rules.get("min_S_by_head_dim", {}).get(str(int(head_dim)), rules.get("min_S"))      # a per-head-dim lower edge overrides the row's
    if min_s is not None and S < int(min_s):
        return "S %d < %d: %s (CELLS.json by_cc %s rules)" % (S, int(min_s), rules.get("why", "another row measured faster below this size"), cc)
    max_x = rules.get("max_S_excl")
    if max_x is not None and S >= int(max_x):
        return "S %d >= %d: %s (CELLS.json by_cc %s rules)" % (S, int(max_x), rules.get("why_above", rules.get("why", "another row measured at parity or faster from this size")), cc)
    hds = rules.get("auto_head_dims")
    if hds and int(head_dim) not in [int(h) for h in hds]:
        return "head_dim %d not in %s for this row under auto on cc %s (CELLS.json rules: %s)" % (int(head_dim), hds, cc, rules.get("why_head_dims", "no cell measured"))
    only = rules.get("dtypes")
    if only and dtype not in only:
        return "dtype %s not in %s for this row on cc %s (CELLS.json rules)" % (dtype, only, cc)
    return None


def _dtype_name(dt) -> str:
    s = str(getattr(dt, "name", dt)).lower().replace("jnp.", "").replace("numpy.", "")
    return {"bfloat16": "bf16", "float32": "fp32", "float16": "fp16"}.get(s, s)


def select(cc: str, dtype, head_dim: int, S: int, N: int = 1, H: int = 1, B: int = 1, has_mask: bool = True, impl: str = "auto",
           bias_dtype=None, SK: Optional[int] = None) -> Tuple[str, dict]:
    """(row, cell) that serves the call on a card of compute capability `cc`, or raise Refused (reasons per row, fallback named).
    No arrays, no jax import: the launch-free part of the decision (binaries present, envelope, switches).  The decision is recorded ONCE
    per call class in :mod:`opt_core.cell_census` (pure observation: decided first, returned unchanged)."""
    try:
        row, cell = _select(cc, dtype, head_dim, S, N=N, H=H, B=B, has_mask=has_mask, impl=impl, bias_dtype=bias_dtype, SK=SK)
    except Refused as r:
        _census(cc, dtype, head_dim, S, H, impl, None, dict(LAST_REFUSAL), r)
        raise
    _census(cc, dtype, head_dim, S, H, impl, row, dict(LAST_PASSED), None)
    return row, cell


LAST_PASSED: Dict[str, str] = {}                          # row -> why, for the rows passed over ahead of the row the last select() returned


def _census(cc, dtype, head_dim, S, H, impl, row, reasons, refusal) -> None:
    """Classify the decision from the by_cc entry that decided and the rows passed over: an envelope / switch refusal ahead of the served
    row = named_fallback; a CELLS.json size rule ahead of it = the measured preference (cell_hit); a cc served through the same-major 8.x
    rule = inherited; Refused = named_fallback served by the caller's stock path; impl named = opt_in."""
    try:
        ccs = str(cc)
        ent = cells().get("by_cc", {}).get(ccs)
        ckey = ("by_cc:%s" % ccs) if ent else None
        near = "by_cc:8.0" if (ent is None and ccs.startswith("8.")) else None
        key = dict(cc=ccs, stack="jax", dtype=_dtype_name(dtype), shape="D%sH%s" % (int(head_dim), int(H)), bucket="S=%d" % int(S), form="fwd", word=impl)
        named = [(r, w) for r, w in (reasons or {}).items() if r != "*" and "CELLS.json" not in str(w)]      # envelope / switch refusals (size rules are the cell's own preference)
        if refusal is not None:
            first = named[0] if named else next(iter((reasons or {"-": "-"}).items()))
            _CENSUS.record("triattn_xla", key, "named_fallback", "caller:xla", cell_id=ckey or near, refused="%s:%s" % (first[0], first[1]), note="raised")
        elif impl != "auto":
            _CENSUS.record("triattn_xla", key, "opt_in", row, cell_id=ckey or near)
        elif named:
            _CENSUS.record("triattn_xla", key, "named_fallback", row, cell_id=ckey or near, refused="%s:%s" % named[0])
        elif ckey is None:
            _CENSUS.record("triattn_xla", key, "inherited", row, cell_id=near, note="guard:cc(same_major_rule:sm_80_cubin)" if near else "family:none(no_by_cc_entry)")
        else:
            _CENSUS.record("triattn_xla", key, "cell_hit", row, cell_id=ckey, note=("rule:" + ",".join(sorted(reasons))) if reasons else "")
    except Exception:                                      # the census counts; it never gates or breaks a selection
        return


def _select(cc: str, dtype, head_dim: int, S: int, N: int = 1, H: int = 1, B: int = 1, has_mask: bool = True, impl: str = "auto",
            bias_dtype=None, SK: Optional[int] = None) -> Tuple[str, dict]:
    from . import _k2b, _cuda, _native, _sm80
    dt = _dtype_name(dtype)
    SK = S if SK is None else SK
    order = [impl] if impl != "auto" else row_order(cc)
    if impl != "auto" and impl not in ROWS:
        raise Refused("triattn_xla: unknown impl %r (rows: %s); fallback: %s" % (impl, ", ".join(ROWS), FALLBACK))
    reasons: Dict[str, str] = {}
    if not order:
        reasons["*"] = "no row for compute capability %s (triattn_native / cuda_sm90a: 9.0; cuda_80: 8.0; k2b_aot: sm_90 / sm_80 cubins)" % cc
    for row in order:
        off = row_switched_off(row)
        if off:
            reasons[row] = off
            continue
        if row == "triattn_native":
            why = _native.cannot_serve(cc, dt, head_dim, S, SK, N, H, B, has_mask, bias_dtype)
        elif row == "cuda_sm90a":
            why = _cuda.cannot_serve(cc, dt, head_dim, S, SK, N, H, B, has_mask, bias_dtype)
        elif row == "cuda_80":
            why = _sm80.cannot_serve(cc, dt, head_dim, S, SK, N, H, B, has_mask, bias_dtype)
        elif row == "k2b_aot":
            why = _k2b.cannot_serve(cc, dt, head_dim, S, SK, N, H, B, has_mask)
        else:
            why = "not a row of this package"
        if why is None and impl == "auto":
            why = rule_refusal(cc, row, dt, head_dim, S, has_mask=has_mask)
        if why is None:
            LAST_PASSED.clear(); LAST_PASSED.update(reasons)
            return row, {"cc": cc, "dtype": dt, "head_dim": head_dim, "S": S, "SK": SK, "N": N, "H": H, "B": B, "has_mask": bool(has_mask)}
        reasons[row] = why
    COUNTS["refused"] += 1
    LAST_REFUSAL.clear(); LAST_REFUSAL.update(reasons)
    raise Refused("triattn_xla: no row serves cc=%s dtype=%s D=%d S=%d N=%d H=%d B=%d mask=%s -- %s; fallback: %s"
                  % (cc, dt, head_dim, S, N, H, B, bool(has_mask), "; ".join("%s: %s" % kv for kv in reasons.items()), FALLBACK), FALLBACK, reasons)


# --------------------------------------------------------------------------------------------------------------------- the face
_LAYOUTS = {"BNHSD": 0, "BNSHD": 1}


def _canon(q, k, v, bias, mask, layout: str):
    """Rank-5 q/k/v in `layout`, bias [B,H,SQ,SK], mask u8 [B,N,SK] | None; plus (B, N, H, SQ, SK, D)."""
    import jax.numpy as jnp
    if layout not in _LAYOUTS:
        raise Refused("triattn_xla: layout %r (layouts: BNHSD, BNSHD); fallback: %s" % (layout, FALLBACK))
    squeeze = False
    if q.ndim == 4:
        q, k, v = q[None], k[None], v[None]; squeeze = True
    if q.ndim != 5 or k.ndim != 5 or v.ndim != 5:
        raise Refused("triattn_xla: q/k/v must be rank 4 or 5 (got %d); fallback: %s" % (q.ndim, FALLBACK))
    B, N = int(q.shape[0]), int(q.shape[1]); D = int(q.shape[4])
    if layout == "BNHSD":
        H, SQ = int(q.shape[2]), int(q.shape[3]); SK = int(k.shape[3])
    else:
        SQ, H = int(q.shape[2]), int(q.shape[3]); SK = int(k.shape[2])
    if bias.ndim == 3:
        bias = bias[None]
    if bias.ndim == 5:
        if int(bias.shape[1]) != 1:
            raise Refused("triattn_xla: rank-5 bias must be [B,1,H,SQ,SK]; fallback: %s" % FALLBACK)
        bias = bias[:, 0]
    if bias.ndim != 4 or tuple(int(x) for x in bias.shape[1:]) != (H, SQ, SK):
        raise Refused("triattn_xla: bias shape %s does not match [B,H,SQ,SK]=[*,%d,%d,%d]; fallback: %s" % (tuple(bias.shape), H, SQ, SK, FALLBACK))
    if int(bias.shape[0]) != B:
        if int(bias.shape[0]) == 1:
            bias = jnp.broadcast_to(bias, (B, H, SQ, SK))
        else:
            raise Refused("triattn_xla: bias batch %d vs B=%d; fallback: %s" % (int(bias.shape[0]), B, FALLBACK))
    if mask is not None:
        if mask.ndim == 5:
            mask = mask[:, :, 0, 0, :]
        elif mask.ndim == 2:
            mask = mask[None]
        if mask.ndim != 3 or tuple(int(x) for x in mask.shape) != (B, N, SK):
            raise Refused("triattn_xla: mask shape %s does not match [B,N,SK]=[%d,%d,%d]; fallback: %s" % (tuple(mask.shape), B, N, SK, FALLBACK))
        if mask.dtype != jnp.uint8:
            mask = mask.astype(jnp.uint8)
    return q, k, v, bias, mask, squeeze, (B, N, H, SQ, SK, D)


def triangle_attention(q, k, v, bias, mask=None, scale: Optional[float] = None, *, impl: str = "auto", layout: str = "BNHSD",
                       vjp: Optional[str] = None, cc: Optional[str] = None, return_row: bool = False):
    """The face (module docstring).  Under jax.jit the row is chosen at trace time from static shapes / dtypes and the card's compute
    capability (`cc`, default: the default backend's first device).  vjp=None: the forward-only rows (impl); vjp="auto" | "attbwd" |
    "flash" | "flash_xla": the differentiable row (its forward is the K2B cubin with the log-sum-exp store whatever `impl` says)."""
    from . import _k2b, _cuda
    q, k, v, bias, mask_u8, squeeze, (B, N, H, SQ, SK, D) = _canon(q, k, v, bias, mask, layout)
    if k.dtype != q.dtype or v.dtype != q.dtype:
        k = k.astype(q.dtype); v = v.astype(q.dtype)
    cc = cc or compute_capability()
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(D)
    if vjp is not None:
        from . import _vjp
        if vjp not in VJP_WORDS:
            raise Refused("triattn_xla: unknown vjp word %r (words: %s); fallback: %s" % (vjp, ", ".join(VJP_WORDS), _vjp.VJP_FALLBACK))
        off = row_switched_off("vjp")
        if off:
            COUNTS["refused"] += 1; LAST_REFUSAL["vjp"] = off
            raise Refused("triattn_xla: the differentiable row is %s; fallback: %s" % (off, _vjp.VJP_FALLBACK))
        try:
            out = _vjp.attention(q, k, v, bias, mask_u8, sc, _LAYOUTS[layout], cc, arch_for_cc(cc), vjp)
        except Refused as e:
            COUNTS["refused"] += 1; LAST_REFUSAL["vjp"] = str(e)
            raise
        COUNTS["vjp"] += 1
        if squeeze:
            out = out[0]
        return (out, "vjp") if return_row else out
    row, _cell = select(cc, q.dtype, D, SQ, N=N, H=H, B=B, has_mask=mask_u8 is not None, impl=impl, bias_dtype=bias.dtype, SK=SK)
    out = _rows_call(sc, _LAYOUTS[layout], cc, impl, mask_u8 is not None)(*((q, k, v, bias, mask_u8) if mask_u8 is not None else (q, k, v, bias)))
    COUNTS[row] += 1
    if squeeze:
        out = out[0]
    return (out, row) if return_row else out


_ROW_CALLS: Dict[tuple, object] = {}


def _rows_forward(q, k, v, bias, mask_u8, sc: float, layout: int, cc: str, impl: str):
    """The forward-only rows for canonical rank-5 operands of ANY leading batch size: select (envelope, by name) + launch.  Shape-generic: the
    vmap rule re-enters it with the vmapped axis folded into B."""
    from . import _k2b, _cuda, _native, _sm80
    B, N = int(q.shape[0]), int(q.shape[1]); D = int(q.shape[4])
    H, SQ, SK = (int(q.shape[2]), int(q.shape[3]), int(k.shape[3])) if layout == 0 else (int(q.shape[3]), int(q.shape[2]), int(k.shape[2]))
    row, _cell = select(cc, q.dtype, D, SQ, N=N, H=H, B=B, has_mask=mask_u8 is not None, impl=impl, bias_dtype=bias.dtype, SK=SK)
    if row == "triattn_native":
        return _native.forward(q, k, v, bias, mask_u8, sc, layout)
    if row == "cuda_sm90a":
        return _cuda.forward(q, k, v, bias, mask_u8, sc, layout)
    if row == "cuda_80":
        return _sm80.forward(q, k, v, bias, mask_u8, sc, layout)
    return _k2b.forward(q, k, v, bias, mask_u8, sc, layout, arch_for_cc(cc))


def _rows_call(sc: float, layout: int, cc: str, impl: str, has_mask: bool):
    """The forward-only dispatch wrapped with the vmap rule (module _batching): under jax.vmap the vmapped axis is folded into B (one launch)."""
    key = (sc, layout, cc, impl, has_mask)
    fn = _ROW_CALLS.get(key)
    if fn is None:
        from . import _batching

        def generic(*arrays):
            q, k, v, bias = arrays[:4]
            return _rows_forward(q, k, v, bias, arrays[4] if has_mask else None, sc, layout, cc, impl)
        fn = _batching.fold_into_batch(generic, generic)
        _ROW_CALLS[key] = fn
    return fn


def reference(q, k, v, bias, mask=None, scale: Optional[float] = None, *, layout: str = "BNHSD", precision: str = "highest"):
    """out = softmax_k(scale q.k + bias, masked keys excluded) @ v in fp32 XLA ops (einsum / where / softmax); a fully-masked row = mean of v.
    The numerics reference the rows are measured against (and the arithmetic of the fallback the rows name)."""
    import jax
    import jax.numpy as jnp
    q, k, v, bias, mask_u8, squeeze, (B, N, H, SQ, SK, D) = _canon(q, k, v, bias, mask, layout)
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(D)
    f32 = jnp.float32
    qs, ks, vs = q.astype(f32), k.astype(f32), v.astype(f32)
    eq_s = "bnhqd,bnhkd->bnhqk" if layout == "BNHSD" else "bnqhd,bnkhd->bnhqk"
    s = jnp.einsum(eq_s, qs, ks, precision=precision) * sc + bias.astype(f32)[:, None]
    if mask_u8 is not None:
        keep = (mask_u8 != 0)[:, :, None, None, :]
        s = jnp.where(keep, s, -1.0e9)
    p = jax.nn.softmax(s, axis=-1)
    eq_o = "bnhqk,bnhkd->bnhqd" if layout == "BNHSD" else "bnhqk,bnkhd->bnqhd"
    out = jnp.einsum(eq_o, p, vs, precision=precision).astype(q.dtype)
    return out[0] if squeeze else out


def report() -> dict:
    """What is loaded and what was served: launcher (file, XLA-FFI API version), CUDA library, cubins per arch, counts, last refusal."""
    from . import _launch, _k2b, _cuda, _native, _vjp, _batching, _sm80
    return {"version": __version__, "rows": list(ROWS), "vjp_words": list(VJP_WORDS), "fallback": FALLBACK, "vjp_fallback": _vjp.VJP_FALLBACK,
            "levers_off": levers_off(), "launcher": _launch.status(), "triattn_native": _native.status(), "cuda_sm90a": _cuda.status(), "cuda_80": _sm80.status(), "k2b_aot": _k2b.status(), "vjp": _vjp.status(),
            "vmap": _batching.status(), "counts": dict(COUNTS), "last_refusal": dict(LAST_REFUSAL)}


def vector_cases() -> List[str]:
    from . import _vectors
    return list(_vectors.CASES)


def vector_inputs(name: str) -> dict:
    from . import _vectors
    return _vectors.inputs(name)
