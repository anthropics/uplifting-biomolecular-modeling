"""ONE fused scaled-dot-product attention call with an additive pair bias — the shared body of every 'attention-with-pair-bias
via SDPA' lever (pairformer single attention, diffusion-transformer token attention, any softmax(QK^T*scale + b [+ mask]) V site).

Contract. ``sdpa_bias(q, k, v, bias, key_mask, ...)`` returns ``(out, event)``: ``out`` in the caller's layout and dtype policy,
``event`` a short machine word for the activation line (``served:sdpa`` plus ``:<backend>`` when one was pinned and ``:<dtype>``
when a compute dtype was applied, e.g. ``served:sdpa:efficient:bf16``). Every precondition it cannot meet RAISES ``Refused`` with
``.event`` (a word: ``torch_missing``, ``rank``, ``heads``, ``dtype:<t>``, ``bias_shape``, ``mask_shape``, ``backend_unknown:<b>``,
``backend_unavailable:<b>``, ``no_kernel:<b>``) and ``.reason`` (a sentence) — the kit adapter catches it, books
``gate.fallback(e.event)`` (``opt_core.attn.size_gate``) and runs the stock math. There is no silent stock path in here.

    from opt_core.attn.sdpa_bias import sdpa_bias, Refused
    try:
        o, ev = sdpa_bias(q, k, v, bias=b, key_mask=m, layout="blhd", bias_layout="bqkh",
                          mask_value=STOCK_MASK_VALUE, compute_dtype="bf16", backend="auto", scale=scaling)
    except Refused as e:
        GATE.fallback(e.event); o = stock_attention(...)

Arguments (all keyword but q, k, v):
    layout        'bhld' (batch…, heads, length, head_dim — SDPA's own) or 'blhd' (length before heads, the common module layout);
                  the transposes in and out are views. Leading batch dims are free ('…').
    bias          additive, broadcastable to […, H, Lq, Lk] after ``bias_layout`` ('bhqk' as given · 'bqkh' heads-last, permuted
                  as a view); None = no bias. The folded additive term (bias + key mask) is handed to the kernel in q's compute
                  dtype, as the fused call requires (an fp32 bias meets bf16 q as bf16 — the stock-under-autocast class).
    key_mask      […, Lk], 1/True = attend; folded into the bias as ``+ mask_value`` on masked keys (``mask_value`` is the
                  adapter's — pass the stock module's own constant, e.g. -1e9 or the dtype's finite min, so an all-masked row
                  behaves as in stock; -inf gives NaN rows exactly where stock would).
    scale         softmax scale; None = 1/sqrt(head_dim) (SDPA's default).
    compute_dtype None (as given) · 'bf16' · 'fp16' · 'fp32' or a torch dtype: q, k, v AND the bias are cast before the call and the
                  output is cast back to ``out_dtype`` (default: q's dtype as given). A recast is a numerics decision of the MODE, named
                  in the event word; it is never applied implicitly.
    backend       'auto' (the library's own dispatch) or a pin: 'efficient' · 'flash' · 'cudnn' · 'math'. A pin the running torch
                  cannot honour is ``Refused('backend_unavailable:<b>')``; a pinned kernel that rejects the inputs (flash takes no
                  dense bias) is ``Refused('no_kernel:<b>')`` — a pin never degrades to another kernel unannounced.

Numerics. A fused attention kernel re-associates the softmax normalisation and the P·V accumulation (fp32 statistics, tiles):
it is NOT bit-exact with an eager softmax(QK^T + b) V on any stack, so a lever built on this call is fast-class (the engine's tier-2
band decides), whatever the dtypes. Between fused backends the results differ in the last bits too — pin the backend in a mode
that promises run-to-run equality. Nothing here syncs the device or allocates beyond the kernel's own workspace and the folded bias.

Python floor: this file parses under 3.9; torch (>= 2.0 for ``scaled_dot_product_attention``; >= 2.1 for ``scale=``; the backend
pin uses ``torch.nn.attention.sdpa_kernel`` when present, else ``torch.backends.cuda.sdp_kernel``) is imported inside the calls.
"""
from __future__ import annotations

import contextlib
from typing import Any, Optional, Tuple

LAYOUTS = ("bhld", "blhd")
BIAS_LAYOUTS = ("bhqk", "bqkh")
BACKENDS = ("auto", "efficient", "flash", "cudnn", "math")
DTYPE_WORDS = {"bf16": "bfloat16", "bfloat16": "bfloat16", "fp16": "float16", "float16": "float16", "half": "float16",
               "fp32": "float32", "float32": "float32", "float": "float32"}


class Refused(RuntimeError):
    """A named refusal: ``event`` is the census word (no spaces), ``reason`` the sentence. The caller runs its stock path
    and books the event; nothing in this module falls back on its own."""

    def __init__(self, event: str, reason: str):
        super().__init__(f"{event}: {reason}")
        self.event = event
        self.reason = reason


def _torch():
    try:
        import torch                      # noqa: WPS433 — lazy by contract: a stack without torch imports opt_core.attn fine
    except ImportError as e:             # pragma: no cover - exercised on torch-less stacks
        raise Refused("torch_missing", f"torch is not importable ({e}); the SDPA lever cannot serve") from None
    if not hasattr(torch.nn.functional, "scaled_dot_product_attention"):
        raise Refused("torch_too_old", f"torch {torch.__version__} has no scaled_dot_product_attention (needs >= 2.0)")
    return torch


def resolve_dtype(word: Any):
    """'bf16' | 'fp16' | 'fp32' | a torch dtype | None -> torch dtype or None; an unknown word is Refused('dtype_word:<w>')."""
    if word is None:
        return None
    torch = _torch()
    if isinstance(word, torch.dtype):
        return word
    name = DTYPE_WORDS.get(str(word).lower())
    if name is None:
        raise Refused(f"dtype_word:{word}", f"unknown compute dtype word {word!r} (known: {sorted(DTYPE_WORDS)})")
    return getattr(torch, name)


def _dtype_word(dt) -> str:
    s = str(dt).replace("torch.", "")
    return {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}.get(s, s)


def available_backends() -> dict:
    """``{'sdpa_kernel_api': 'nn.attention'|'backends.cuda'|'none', 'flash': bool, 'efficient': bool, 'math': bool, 'cudnn': bool|None,
    'torch': <version>}`` — what the running torch reports as enabled (a process-wide setting, not a per-shape promise)."""
    torch = _torch()
    cuda_b = torch.backends.cuda
    out = {"torch": torch.__version__,
           "flash": bool(getattr(cuda_b, "flash_sdp_enabled", lambda: False)()),
           "efficient": bool(getattr(cuda_b, "mem_efficient_sdp_enabled", lambda: False)()),
           "math": bool(getattr(cuda_b, "math_sdp_enabled", lambda: True)()),
           "cudnn": (bool(cuda_b.cudnn_sdp_enabled()) if hasattr(cuda_b, "cudnn_sdp_enabled") else None)}
    try:
        import torch.nn.attention as _na  # noqa: F401
        out["sdpa_kernel_api"] = "nn.attention"
    except Exception:                    # noqa: BLE001
        out["sdpa_kernel_api"] = "backends.cuda" if hasattr(cuda_b, "sdp_kernel") else "none"
    return out


def describe() -> str:
    """One ``key=value`` fragment for the activation line: ``sdpa=torch<ver> api=<...> flash=0|1 efficient=0|1 cudnn=0|1|na math=0|1``."""
    b = available_backends()
    cud = "na" if b["cudnn"] is None else str(int(b["cudnn"]))
    return (f"sdpa=torch{b['torch']} api={b['sdpa_kernel_api']} flash={int(b['flash'])} efficient={int(b['efficient'])} "
            f"cudnn={cud} math={int(b['math'])}")


def _backend_context(torch, backend: str):
    """The context that pins ONE backend, or a null context for 'auto'. Refuses by name when the pin cannot be expressed."""
    if backend == "auto":
        return contextlib.nullcontext()
    if backend not in BACKENDS:
        raise Refused(f"backend_unknown:{backend}", f"backend must be one of {BACKENDS}")
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend          # torch >= 2.3
        table = {"efficient": "EFFICIENT_ATTENTION", "flash": "FLASH_ATTENTION", "math": "MATH", "cudnn": "CUDNN_ATTENTION"}
        chosen = getattr(SDPBackend, table[backend], None)
        if chosen is None:
            raise Refused(f"backend_unavailable:{backend}", f"torch {torch.__version__} SDPBackend has no {table[backend]}")
        return sdpa_kernel(chosen)
    except ImportError:
        pass
    sdp_kernel = getattr(torch.backends.cuda, "sdp_kernel", None)          # torch 2.0 - 2.2
    if sdp_kernel is None:
        raise Refused(f"backend_unavailable:{backend}", f"torch {torch.__version__} exposes no backend pin API")
    if backend == "cudnn":
        raise Refused("backend_unavailable:cudnn", f"torch {torch.__version__} has no cuDNN SDPA backend switch")
    return sdp_kernel(enable_flash=(backend == "flash"), enable_math=(backend == "math"), enable_mem_efficient=(backend == "efficient"))


def fold_key_mask(bias, key_mask, mask_value: float, dtype, torch=None):
    """The additive term SDPA takes: ``bias`` (may be None) plus ``mask_value`` on keys where ``key_mask`` is 0/False.
    ``key_mask`` […, Lk] broadcasts as […, 1, 1, Lk]. Returns None when both are None."""
    torch = torch or _torch()
    if key_mask is None:
        return bias
    km = key_mask
    if km.dtype != torch.bool:
        km = km != 0
    add = torch.zeros(km.shape, dtype=dtype, device=km.device).masked_fill(~km, mask_value)   # […, Lk]
    add = add[..., None, None, :]                                                               # […, 1, 1, Lk]
    return add if bias is None else bias + add


def sdpa_bias(q, k, v, bias=None, key_mask=None, *, layout: str = "bhld", bias_layout: str = "bhqk", scale: Optional[float] = None,
              mask_value: float = -1e9, compute_dtype=None, out_dtype=None, backend: str = "auto",
              dropout_p: float = 0.0) -> Tuple[Any, str]:
    """softmax(q k^T * scale + bias [+ mask_value on masked keys]) v in ONE fused call. Returns ``(out, event)``; raises ``Refused``."""
    torch = _torch()
    F = torch.nn.functional
    if layout not in LAYOUTS:
        raise Refused(f"layout:{layout}", f"layout must be one of {LAYOUTS}")
    if bias_layout not in BIAS_LAYOUTS:
        raise Refused(f"bias_layout:{bias_layout}", f"bias_layout must be one of {BIAS_LAYOUTS}")
    if q.dim() < 3 or k.dim() != q.dim() or v.dim() != q.dim():
        raise Refused("rank", f"q, k, v need the same rank >= 3 (got {q.dim()}, {k.dim()}, {v.dim()})")
    for name, t in (("q", q), ("k", k), ("v", v)):
        if t.dtype not in (torch.float32, torch.bfloat16, torch.float16):
            raise Refused(f"dtype:{_dtype_word(t.dtype)}", f"{name} has dtype {t.dtype}; fused SDPA serves fp32, bf16, fp16")
    if layout == "blhd":                                   # […, L, H, D] -> […, H, L, D] (views)
        q_, k_, v_ = q.transpose(-3, -2), k.transpose(-3, -2), v.transpose(-3, -2)
    else:
        q_, k_, v_ = q, k, v
    H, Lq, D = q_.shape[-3], q_.shape[-2], q_.shape[-1]
    Lk = k_.shape[-2]
    if k_.shape[-3] != H or v_.shape[-3] != H or k_.shape[-1] != D or v_.shape[-2] != Lk:
        raise Refused("heads", f"q {tuple(q_.shape)} / k {tuple(k_.shape)} / v {tuple(v_.shape)} disagree on heads, keys or head_dim")
    b_ = bias
    if b_ is not None:
        if b_.dim() < 2:
            raise Refused("bias_shape", f"bias rank {b_.dim()} < 2")
        if bias_layout == "bqkh":
            if b_.dim() < 3:
                raise Refused("bias_shape", "a heads-last bias needs rank >= 3 ([…, Lq, Lk, H])")
            b_ = b_.movedim(-1, -3)                        # […, Lq, Lk, H] -> […, H, Lq, Lk] (view)
        if b_.shape[-1] != Lk or b_.shape[-2] not in (1, Lq) or (b_.dim() >= 3 and b_.shape[-3] not in (1, H)):
            raise Refused("bias_shape", f"bias {tuple(b_.shape)} does not broadcast to [..., {H}, {Lq}, {Lk}]")
    if key_mask is not None and key_mask.shape[-1] != Lk:
        raise Refused("mask_shape", f"key_mask {tuple(key_mask.shape)} last dim != Lk {Lk}")
    cd = resolve_dtype(compute_dtype)
    odt = resolve_dtype(out_dtype) or q.dtype
    event = "served:sdpa"
    if backend != "auto":
        event += ":" + backend
    if cd is not None and (cd != q_.dtype or (b_ is not None and b_.dtype != cd)):
        q_, k_, v_ = q_.to(cd), k_.to(cd), v_.to(cd)
        if b_ is not None:
            b_ = b_.to(cd)
        event += ":" + _dtype_word(cd)
    elif k_.dtype != q_.dtype or v_.dtype != q_.dtype:
        raise Refused("dtype:mixed", f"q/k/v dtypes differ ({q_.dtype}, {k_.dtype}, {v_.dtype}) and no compute_dtype was given")
    if b_ is not None and b_.dtype != q_.dtype:
        b_ = b_.to(q_.dtype)                               # the fused call takes the additive term in q's dtype
    attn_mask = fold_key_mask(b_, key_mask, mask_value, dtype=q_.dtype, torch=torch)
    kwargs = {"attn_mask": attn_mask, "dropout_p": dropout_p}
    if scale is not None:
        kwargs["scale"] = float(scale)
    ctx = _backend_context(torch, backend)
    try:
        with ctx:
            o = F.scaled_dot_product_attention(q_, k_, v_, **kwargs)
    except TypeError as e:                                  # torch 2.0: no scale= keyword
        if "scale" in kwargs and "scale" in str(e):
            raise Refused("torch_too_old", f"torch {torch.__version__} scaled_dot_product_attention takes no scale= (needs >= 2.1)") from None
        raise
    except RuntimeError as e:
        msg = str(e)
        if "No available kernel" in msg or "no kernel found" in msg.lower():
            raise Refused(f"no_kernel:{backend}", msg.splitlines()[0][:200]) from None
        raise
    if layout == "blhd":
        o = o.transpose(-3, -2)
    if o.dtype != odt:
        o = o.to(odt)
    return o, event
