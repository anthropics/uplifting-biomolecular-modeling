"""Attention with a pair bias shared by a batch of samples — the entry the kits call (AlphaFold3 Algorithm 24 core: the token diffusion
transformer's AttentionPairBias over S samples, the pairformer single track's over one):

    out = pair_bias_attention(q, k, v, pair_bias, key_mask=None, *, gate=None, num_samples=None, num_heads=None, layout="rows",
                              out=None, scale=None, inf=1e9)

    out[s, i, h*D:(h+1)*D] = sigmoid(gate[s, i, hD..]) * softmax_j( scale * q.k + pair_bias[s|0, h, i, j] - inf * (key_mask[s|0, j] == 0) ) @ v

One launch of the carried kernel ``apb_attn`` for all samples: the pair bias is read once per (query tile, key tile, head) and shared by
the samples through L2, the key mask is applied by name (stock's additive -inf semantics: all-masked rows give the uniform average, never
NaN), the sigmoid gate is fused in the epilogue, and q/k/v/gate/out are served IN PLACE whatever their strides — no transposes, no copies.

Layouts (S = num_samples, H = num_heads, N tokens, D head dim):
  "rows"  q, k, v, gate are 2-D [S*N, H*D] row blocks with unit column stride — typically the four column slices of ONE [S*N, 4*H*D]
          projection GEMM output (row stride 4*H*D; sample-major rows: row = s*N + i); ``num_samples`` and ``num_heads`` are required;
          returns [S*N, H*D] (contiguous unless ``out=`` is given; ``out`` may be a strided 2-D view with unit column stride).
  "shnd"  q, k, v [S, H, N, D] with any strides whose innermost stride is 1 (e.g. the transposed views of [S, N, H*D] GEMM outputs);
          gate [S, N, H*D] (or [S, N, H, D]) or None; returns [S, N, H*D] (contiguous unless ``out=`` is given).
pair_bias  [H, N, N], or [SB, H, N, N] with SB in {1, S} (per-sample planes), with any number of leading singleton dims; bf16 / fp16 /
           fp32; head-major with unit key stride and 16-byte aligned rows (row stride a multiple of 8 elements: allocate [.., N, ceil8(N)],
           view [..., :N]) is the fast path at every token count; a view whose key stride is not 1 (the stock heads-last permute of an
           [N, N, H] projection output) is copied head-major ONCE per call, counted (``events.relayout_copy``); unaligned rows are served in
           place and counted (``events.rows_unaligned``).
key_mask   [N], [1, N] or [S, N] (leading singleton dims allowed), any dtype, nonzero = attend; None = no mask.
scale      None = D ** -0.5; pass ``scale=1.0`` when q arrives pre-scaled.
Refusals raise ``Unsupported`` (``.event`` = one census word: rank / layout:<..> / shape:<..> / dtype:<..> / stride:<..> / size:<..> /
device:<..> / head_dim:<n> / import:<..>) BEFORE any work (the relayout copy included): the caller runs its stock statement and books the word. ``census()`` reports this
process's served calls (by shape, capped at 32 keys), refusals by event and the layout events; ``kernel()`` resolves the carried kernel
module through the core's route (``opt_core.kernels.route("apb_attn")``) once per process.
Numerics class: tolerance (flash arithmetic — fp32 online softmax over 16-bit tensor-core products; on trained-model tensors the error to
an fp64 statement is at or below the stock bf16 einsum path's).
"""
import importlib
import math
import threading
from typing import Any, Dict, Optional

KERNEL = "apb_attn"
LAYOUTS = ("rows", "shnd")
CENSUS: Dict[str, Any] = {"served": 0, "refused": {}, "events": {}, "by_shape": {}}
_LOCK = threading.Lock()
_MOD: Dict[str, Any] = {"kernel": None, "error": None}


class Unsupported(Exception):
    """An input outside the served domain, refused by name before any work (``.event``, ``.detail``)."""

    def __init__(self, event: str, detail: str = ""):
        super().__init__(f"{event}: {detail}" if detail else event)
        self.event, self.detail = event, detail


def _book(event: Optional[str], key: Optional[str] = None, counter: str = "refused") -> None:
    with _LOCK:
        if event is None:
            CENSUS["served"] += 1
            if key is not None and (key in CENSUS["by_shape"] or len(CENSUS["by_shape"]) < 32):
                CENSUS["by_shape"][key] = CENSUS["by_shape"].get(key, 0) + 1
        else:
            CENSUS[counter][event] = CENSUS[counter].get(event, 0) + 1


def census() -> Dict[str, Any]:
    with _LOCK:
        out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in CENSUS.items()}
    k = _MOD["kernel"]
    out["kernel"] = getattr(k, "__version__", None) if k is not None else None
    if k is not None and hasattr(k, "launch_census"):
        out["launch"] = k.launch_census()                      # direct launches of the compiled handle vs JITFunction launches (kernels/apb_attn.py)
    return out


def kernel():
    """The carried kernel module, resolved through the core's route once per process; raises Unsupported("import:apb_attn") by name."""
    if _MOD["kernel"] is not None:
        return _MOD["kernel"]
    if _MOD["error"] is not None:
        raise Unsupported("import:" + KERNEL, _MOD["error"])
    try:
        from opt_core.kernels import route
        route(KERNEL)
        _MOD["kernel"] = importlib.import_module(KERNEL)
    except Exception as e:  # noqa: BLE001
        _MOD["error"] = "%s: %s" % (type(e).__name__, str(e)[:160])
        raise Unsupported("import:" + KERNEL, _MOD["error"])
    return _MOD["kernel"]


def pair_bias_attention(q, k, v, pair_bias, key_mask=None, *, gate=None, num_samples: Optional[int] = None, num_heads: Optional[int] = None,
                        layout: str = "rows", out=None, scale: Optional[float] = None, inf: float = 1e9, config: Optional[dict] = None):
    """See the module docstring."""
    import torch
    try:
        K = kernel()
        if not math.isfinite(float(inf)) or (scale is not None and not math.isfinite(float(scale))):
            raise Unsupported("nonfinite:inf_or_scale", f"inf and scale must be finite (a masked key adds -inf as a large finite negative, as the engines do), got inf={inf} scale={scale}")
        if layout == "shnd":
            if q.dim() != 4:
                raise Unsupported("rank", f"layout shnd wants q/k/v [S,H,N,D], got {tuple(q.shape)}")
            S, H, N, D = (int(x) for x in q.shape)
            if num_samples is not None and int(num_samples) != S:
                raise Unsupported("shape:num_samples", f"num_samples={num_samples} but q has S={S}")
            q4, k4, v4 = q, k, v
            g3 = gate
            if g3 is not None and g3.dim() == 4:
                if g3.stride(-1) != 1 or g3.stride(-2) != g3.shape[-1]:
                    raise Unsupported("stride:gate", "gate [S,N,H,D] must merge to [S,N,H*D] without a copy")
                g3 = g3.reshape(g3.shape[0], g3.shape[1], -1)
            out3 = out
        elif layout == "rows":
            if num_samples is None or num_heads is None:
                raise Unsupported("layout:rows_needs_S_H", "layout rows needs num_samples and num_heads")
            S, H = int(num_samples), int(num_heads)
            for name, t in (("q", q), ("k", k), ("v", v)) + ((("gate", gate),) if gate is not None else ()):
                if t.dim() != 2 or t.shape[0] % S or t.shape[1] % H or t.stride(1) != 1:
                    raise Unsupported(f"shape:{name}", f"layout rows wants {name} 2-D [S*N, H*D] with unit column stride, got {tuple(t.shape)} strides {tuple(t.stride())}")
            N, D = int(q.shape[0]) // S, int(q.shape[1]) // H
            NK = int(k.shape[0]) // S
            if gate is not None and tuple(gate.shape) != (S * N, H * D):
                raise Unsupported("shape:gate", f"layout rows wants gate [S*N, H*D] = {(S * N, H * D)}, got {tuple(gate.shape)}")

            def as4(t, n):      # [S*n, H*D] rows -> [S, H, n, D] view (no copy)
                rs = t.stride(0)
                return t.as_strided((S, H, n, D), (n * rs, D, rs, 1), t.storage_offset())
            q4, k4, v4 = as4(q, N), as4(k, NK), as4(v, NK)
            g3 = None if gate is None else gate.as_strided((S, N, H * D), (N * gate.stride(0), gate.stride(0), 1), gate.storage_offset())
            out3 = None
            if out is not None:
                if out.dim() != 2 or tuple(out.shape) != (S * N, H * D) or out.stride(1) != 1:
                    raise Unsupported("shape:out", f"layout rows wants out [S*N, H*D] with unit column stride, got {tuple(out.shape)}")
                out3 = out.as_strided((S, N, H * D), (N * out.stride(0), out.stride(0), 1), out.storage_offset())
        else:
            raise Unsupported(f"layout:{layout}", "known layouts: " + ", ".join(LAYOUTS))
        pb = pair_bias
        while pb.dim() > 4 and int(pb.shape[0]) == 1:
            pb = pb[0]
        if pb.dim() == 4 and int(pb.shape[0]) == 1:
            pb = pb[0]
        if pb.dim() not in (3, 4):
            raise Unsupported("shape:pair_bias", f"pair_bias must be [H,N,N] or [S,H,N,N] after leading singleton dims, got {tuple(pair_bias.shape)}")
        km = None
        if key_mask is not None:
            km = key_mask
            while km.dim() > 2 and int(km.shape[0]) == 1:
                km = km[0]
            if km.dim() == 1:
                km = km[None, :]
            if km.dim() != 2 or int(km.shape[0]) not in (1, S):
                raise Unsupported("shape:mask", f"key_mask must be [N], [1,N] or [S,N] (leading singleton dims allowed), got {tuple(key_mask.shape)}")
        try:
            pb, km = K.check(q4, k4, v4, pb, km, g3, out3)         # every refusal by name BEFORE any work (the relayout below included)
        except K.Unsupported as e:
            raise Unsupported(e.event, e.detail)
        if int(pb.stride(-1)) != 1 and int(pb.shape[-1]) > 1:
            pb = pb.contiguous()                                  # a heads-last view: one head-major copy per call, counted
            _book("relayout_copy", counter="events")
        info: Dict[str, Any] = {}
        res = K.apb_attention(q4, k4, v4, pb, mask=km, gate=g3, out=out3, scale=scale, mask_neg=-float(inf), config=config, info=info, checked=True)
        if not info.get("bias_rows_aligned", True):
            _book("rows_unaligned", counter="events")
    except Unsupported as e:
        _book(e.event)
        raise
    _book(None, f"S{S}H{H}N{N}D{D}")
    if layout == "rows":
        return out if out is not None else res.view(S * N, H * D)
    return out if out is not None else res
