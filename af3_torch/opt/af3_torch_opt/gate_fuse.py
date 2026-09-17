"""Fused gating glue (lever ``gate_fuse``): the pair stack's ``x *= mask``, ``torch.sigmoid(gate)``, ``x *= sig`` statements as one kernel
with the stock rounding points.

Stock (under ``exact``: bf16 activations, ATen elementwise kernels with fp32 opmath): in the triangle multiplication
``projection *= mask[..., None]`` (bf16 × fp32 → rounded to bf16), ``projection *= torch.sigmoid(gate)`` (ATen's sigmoid:
``1 / (1 + expf(-x))`` in fp32, rounded to bf16; then a bf16 × bf16 product in fp32, rounded to bf16) and ``pair *= torch.sigmoid(gate_out)``;
in the grid self-attention ``weighted_avg *= torch.sigmoid(gate_values)``. Each statement is a full pass over an ``[N, N, 128…256]``
tensor: at 800 tokens ≈ 2.3 GB of traffic per triangle-multiplication call where 1 GB carries the information.

Here: ``gated_(x, gate, mask)`` — ONE Triton kernel per statement group that loads ``x``, ``gate`` (and ``mask``), computes in fp32 with the
SAME intermediate roundings as the stock sequence (the masked product rounded to bf16 before the gate product; the sigmoid rounded to bf16
before it multiplies) and stores into ``x`` in place. Two forms, chosen once for the kit by measurement on live activations
(``FORM``): ``full`` — the sigmoid inside the kernel on libdevice's ``expf`` (the function ATen's kernel calls) — and ``muls`` — the
sigmoid left to ATen (``torch.sigmoid``), only the products fused (pure IEEE products: bitwise by construction). The lever composes on
the restated forwards of ``tri_layout`` (triangle multiplication) and ``attn_layout`` (grid self-attention), which call ``gated_`` where
the stock statements stand; with those levers absent it has nothing to bind and says so. Census ``COUNTS`` rides forward.json and the
LEVER line.
"""
from __future__ import annotations

COUNTS = {"calls": 0, "fused": 0, "generic": 0}
FORM = "full"                     # "full" | "muls" — set by the kit from the live-activation equality measurement (see CHANGES)
BLOCK = 2048
_STATE = {"full": None, "muls": None, "on": False}
_BOUND = {"torch": None}


def take() -> dict:
    out = dict(COUNTS)
    for k in COUNTS:
        COUNTS[k] = 0
    return out


def _kernels():
    if _STATE["full"] is not None:
        return _STATE
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice

    @triton.jit
    def _gate_full(X, G, M, n, C, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr):
        # x[i] = bf16( f32(bf16(f32(x[i]) * m[i // C])) * f32(bf16(1 / (1 + expf(-f32(g[i]))))) )  — the stock statements' arithmetic and roundings
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        ok = offs < n
        x = tl.load(X + offs, mask=ok, other=0.0).to(tl.float32)
        if HAS_MASK:
            m = tl.load(M + offs // C, mask=ok, other=0.0).to(tl.float32)
            x = (x * m).to(X.dtype.element_ty).to(tl.float32)
        g = tl.load(G + offs, mask=ok, other=0.0).to(tl.float32)
        one = 1.0
        s = one / (one + libdevice.exp(-g))
        s = s.to(G.dtype.element_ty).to(tl.float32)
        y = (x * s).to(X.dtype.element_ty)
        tl.store(X + offs, y, mask=ok)

    @triton.jit
    def _gate_muls(X, S, M, n, C, HAS_MASK: tl.constexpr, BLOCK: tl.constexpr):
        # x[i] = bf16( f32(bf16(f32(x[i]) * m[i // C])) * f32(s[i]) )  with s = torch.sigmoid(gate) computed by ATen (bf16)
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK).to(tl.int64)
        ok = offs < n
        x = tl.load(X + offs, mask=ok, other=0.0).to(tl.float32)
        if HAS_MASK:
            m = tl.load(M + offs // C, mask=ok, other=0.0).to(tl.float32)
            x = (x * m).to(X.dtype.element_ty).to(tl.float32)
        s = tl.load(S + offs, mask=ok, other=0.0).to(tl.float32)
        y = (x * s).to(X.dtype.element_ty)
        tl.store(X + offs, y, mask=ok)

    _STATE["full"], _STATE["muls"] = _gate_full, _gate_muls
    return _STATE


def eligible(x, gate, mask) -> bool:
    """CUDA bf16/fp16 contiguous x and gate of one shape; mask None or a tensor over x's leading dims (x.shape[:-1])."""
    torch = _BOUND["torch"]
    if not (_STATE["on"] and x.is_cuda and x.is_contiguous() and gate.is_contiguous() and x.shape == gate.shape
            and x.dtype in (torch.bfloat16, torch.float16) and gate.dtype == x.dtype):
        return False
    if mask is not None and (tuple(mask.shape) != tuple(x.shape[:-1]) or not mask.is_contiguous() or not mask.is_cuda
                             or mask.dtype not in (torch.float32, x.dtype)):
        return False
    return True


def gated_(x, gate, mask=None, form=None):
    """In place: ``x *= mask[..., None]`` (when given) then ``x *= torch.sigmoid(gate)`` — fused. Returns x. The caller checked ``eligible``."""
    import triton
    torch = _BOUND["torch"]
    k = _kernels()
    n = x.numel(); C = x.shape[-1]
    grid = (triton.cdiv(n, BLOCK),)
    form = form or FORM
    if form == "full":
        k["full"][grid](x, gate, mask if mask is not None else x, n, C, HAS_MASK=mask is not None, BLOCK=BLOCK, num_warps=4)
    else:
        s = torch.sigmoid(gate)
        k["muls"][grid](x, s, mask if mask is not None else x, n, C, HAS_MASK=mask is not None, BLOCK=BLOCK, num_warps=4)
    return x


def mask_gate_(x, gate, mask=None):
    """What the restated forwards call where the stock statements stand: fused when the lever is on and the tensors are its case, else the
    stock statements. Counted."""
    torch = _BOUND["torch"] or _bind()["torch"]
    COUNTS["calls"] += 1
    if _STATE["on"] and eligible(x, gate, mask):
        COUNTS["fused"] += 1
        return gated_(x, gate, mask)
    COUNTS["generic"] += 1
    if mask is not None:
        x *= mask[..., None]
    x *= torch.sigmoid(gate)
    return x


def _bind():
    if _BOUND["torch"] is None:
        import torch
        _BOUND["torch"] = torch
    return _BOUND


def install(model) -> dict:
    """Switch the fused path on for the restated forwards that call ``mask_gate_`` (tri_layout, attn_layout). {'installed', 'already', 'form',
    'hosts', 'reason'} — ``hosts``: which restated forwards are bound in this process (the lever binds nothing itself)."""
    _bind()
    import importlib
    hosts = []
    try:
        tm = importlib.import_module("xfold.nn.triangle_multiplication").TriangleMultiplication
        if getattr(tm, "_tri_layout", None):
            hosts.append("tri_layout")
    except Exception:                      # noqa: BLE001 — a missing module is reported as no host, never a failure of this lever
        pass
    try:
        ga = importlib.import_module("xfold.nn.attention").GridSelfAttention
        if getattr(ga, "_attn_layout", None) is True:
            hosts.append("attn_layout")
    except Exception:                      # noqa: BLE001
        pass
    already = _STATE["on"]
    _STATE["on"] = bool(hosts)
    return {"installed": bool(hosts), "already": already, "form": FORM, "hosts": hosts,
            "reason": None if hosts else "no_host_forward(tri_layout_and_attn_layout_absent)"}
