"""Grid self-attention operand layout (lever ``attn_layout``): the head-major q / k / v operands and the token-major output written by a
row-regroup kernel instead of torch's generic strided copy.

Stock (``xfold/nn/attention.py`` ``GridSelfAttention._attention``, the triangle attention ``off`` / ``exact`` run on the pair track, 104
calls per trunk pass): the three projections ``[b, n, h·d]`` are viewed head-major (``einops 'b n (h d) -> b h n d'``: a strided view) and
``fastnn.dot_product_attention`` → ``dot_product_attention_triton`` copies each ``.contiguous()`` — torch's generic strided-copy kernel,
2-byte elements through a 4-D offset calculator, far below the card's copy bandwidth, 312 copies per pass; the kernel's ``[b, h, n, d]``
output goes back token-major (``'b h n d -> b n (h d)'``: another generic copy per call). The attention
kernel itself (``_attention_core``) addresses its operands as ``(z·H + h)·stride_h + m·stride_m + d`` — it cannot read the projection's
memory in place, so the copies are real; only their kernel is slow.

Here (``_attention`` below, the stock statements restated): each operand is regrouped by ``regroup`` — a Triton kernel that moves whole
``d``-rows (64 bytes, unit stride on both sides) between the two row orders, loads and stores only — into exactly the contiguous
``[b, h, n, d]`` tensor stock's ``.contiguous()`` makes (so the wrapper's own ``.contiguous()`` returns it as it is and ``_attention_core``
runs on the same bytes at the same strides), and the output is regrouped into the contiguous ``[b, n, h·d]`` tensor the stock rearrange
makes. Data movement only: bitwise by construction (tests/test_attn_layout.py holds the restatement to the kit's source; torch.equal on the
module output on the box). Bound where the class's ``_attention`` is still the kit's own (``exact``; under ``fast`` / ``big`` the triangle
attention kernel levers own the class and this lever steps aside, named). Census ``COUNTS`` rides forward.json (``attn_layout``) and the
LEVER line.
"""
from __future__ import annotations

COUNTS = {"calls": 0, "regrouped": 0, "small": 0, "generic": 0}
MIN_TOKENS = 512                  # pair tracks shorter than this keep the stock copies: there the trunk pass is launch-bound and the four regroup launches cost more host time than the device time they save (measured: +0.15 s per item at 400 tokens, −0.57 / −1.20 s at 800 / 1200; H100)
BLOCK_R = 64                      # rows (of d elements) per program
_STATE = {"kernel": None}
_BOUND = {"torch": None, "fastnn": None, "einops": None}


def take() -> dict:
    out = dict(COUNTS)
    for k in COUNTS:
        COUNTS[k] = 0
    return out


def _kernel():
    if _STATE["kernel"] is not None:
        return _STATE["kernel"]
    import triton
    import triton.language as tl

    @triton.jit
    def _regroup(SRC, DST, G, R, s_sb, s_sg, s_sr, s_db, s_dg, s_dr, D: tl.constexpr, BLOCK_R: tl.constexpr):
        # DST[b, g, r, :] = SRC[b, g, r, :] over explicit element strides for (b, g, r); the D elements of a row are unit-stride on both
        # sides and move as they are (a [BLOCK_R, D] tile per program: loads and stores only).
        pid_bg = tl.program_id(1).to(tl.int64)
        b = pid_bg // G
        g = pid_bg % G
        rs = (tl.program_id(0) * BLOCK_R + tl.arange(0, BLOCK_R)).to(tl.int64)
        ds = tl.arange(0, D).to(tl.int64)
        m = (rs[:, None] < R) & (ds[None, :] < D)
        v = tl.load(SRC + b * s_sb + g * s_sg + rs[:, None] * s_sr + ds[None, :], mask=m)
        tl.store(DST + b * s_db + g * s_dg + rs[:, None] * s_dr + ds[None, :], v, mask=m)

    _STATE["kernel"] = _regroup
    return _regroup


def regroup(src, dst, B, G, R, D, s_src, s_dst):
    """dst[b, g, r, 0:D] = src[b, g, r, 0:D] for b < B, g < G, r < R; s_src / s_dst = element strides of (b, g, r); the row's D elements
    are unit-stride in both."""
    import triton
    grid = (triton.cdiv(R, BLOCK_R), B * G)
    _kernel()[grid](src, dst, G, R, s_src[0], s_src[1], s_src[2], s_dst[0], s_dst[1], s_dst[2], D=D, BLOCK_R=BLOCK_R, num_warps=4)
    return dst


def head_major(t, h):
    """[b, n, h·d] contiguous -> [b, h, n, d] contiguous: the tensor stock's ``rearrange(t, 'b n (h d) -> b h n d').contiguous()`` is."""
    torch = _BOUND["torch"]
    b, n, hd = t.shape
    d = hd // h
    out = torch.empty((b, h, n, d), dtype=t.dtype, device=t.device)
    return regroup(t, out, b, h, n, d, (n * hd, d, hd), (h * n * d, n * d, d))


def token_major(o):
    """[b, h, n, d] contiguous -> [b, n, h·d] contiguous: the tensor stock's ``rearrange(o, 'b h n d -> b n (h d)')`` is."""
    torch = _BOUND["torch"]
    b, h, n, d = o.shape
    out = torch.empty((b, n, h * d), dtype=o.dtype, device=o.device)
    regroup(o, out, b, h, n, d, (h * n * d, n * d, d), (n * h * d, d, h * d))
    return out


def _eligible(q, h) -> bool:
    """CUDA, 3-D [b, n, h·d] contiguous, a head dim the Triton attention serves (else the stock statements run: the torch path)."""
    fastnn = _BOUND["fastnn"]
    d = q.shape[-1] // h if q.dim() == 3 else 0
    return (q.is_cuda and q.dim() == 3 and q.is_contiguous() and d in (16, 32, 64, 128) and d & (d - 1) == 0
            and fastnn.config.dot_product_attention_implementation == "triton" and q.dtype in (_BOUND["torch"].float16, _BOUND["torch"].bfloat16))


def _attention(self, pair, mask, bias):
    """``GridSelfAttention._attention`` restated: the operand / output regroupings on the row-regroup kernel."""
    torch, fastnn, einops = _BOUND["torch"], _BOUND["fastnn"], _BOUND["einops"]
    COUNTS["calls"] += 1
    q = self.q_projection(pair)
    k = self.k_projection(pair)
    v = self.v_projection(pair)

    small = q.dim() == 3 and q.shape[1] < MIN_TOKENS
    if small:
        COUNTS["small"] += 1
    if not small and _eligible(q, self.num_head) and k.is_contiguous() and v.is_contiguous():
        COUNTS["regrouped"] += 1
        q, k, v = map(lambda t: head_major(t, self.num_head), [q, k, v])   # stock: einops 'b n (h d) -> b h n d' views, copied contiguous by the wrapper

        weighted_avg = fastnn.dot_product_attention(q, k, v,
                                                    mask=mask,
                                                    bias=bias)

        weighted_avg = token_major(weighted_avg)                               # stock: einops 'b h n d -> b n (h d)' (a copy of the kernel's output)

        gate_values = self.gating_query(pair)
        weighted_avg = _gate().mask_gate_(weighted_avg, gate_values)           # stock: weighted_avg *= torch.sigmoid(gate_values); one fused kernel under lever gate_fuse
        return self.output_projection(weighted_avg)
    else:                                                                      # not this lever's case (short pair tracks — counted `small` — or CPU / the torch attention path / odd shapes — counted `generic`): the stock statements
        if not small:
            COUNTS["generic"] += 1
        q, k, v = map(lambda t: einops.rearrange(
            t, 'b n (h d) -> b h n d', h=self.num_head), [q, k, v])

        weighted_avg = fastnn.dot_product_attention(q, k, v,
                                                    mask=mask,
                                                    bias=bias)

        weighted_avg = einops.rearrange(weighted_avg, 'b h n d -> b n (h d)')

    gate_values = self.gating_query(pair)

    weighted_avg *= torch.sigmoid(gate_values)
    return self.output_projection(weighted_avg)


def _gate():
    """The package's gate_fuse module — the SAME module object forward.py installed (it imports the package levers by their bare names from
    its own directory; a second import as ``af3_torch_opt.gate_fuse`` would be another module with its own switch and census)."""
    import sys
    m = sys.modules.get("gate_fuse") or sys.modules.get("af3_torch_opt.gate_fuse")
    if m is not None:
        return m
    import importlib
    try:
        return importlib.import_module("gate_fuse")
    except ImportError:
        return importlib.import_module("af3_torch_opt.gate_fuse")


def _bind_modules():
    if _BOUND["torch"] is None:
        import importlib
        import torch
        _BOUND["torch"] = torch
        _BOUND["fastnn"] = importlib.import_module("xfold.fastnn")
        _BOUND["einops"] = importlib.import_module("einops")
    return _BOUND


def _is_stock(f, mod_name, qualname) -> bool:
    return getattr(f, "__module__", None) == mod_name and getattr(f, "__qualname__", "") == qualname


def install(model) -> dict:
    """Bind ``_attention`` on the GridSelfAttention CLASS when both its ``forward`` and ``_attention`` are still the kit's own (no kernel
    lever owns the class: ``exact``); else step aside, named. Idempotent. {'installed', 'already', 'modules', 'reason'}."""
    import importlib
    _bind_modules()
    am = importlib.import_module("xfold.nn.attention")
    GA = am.GridSelfAttention
    n = sum(1 for m in model.modules() if isinstance(m, GA))
    if getattr(GA, "_attn_layout", None):
        return {"installed": GA._attn_layout is True, "already": True, "modules": n, "reason": getattr(GA, "_attn_layout_reason", None)}
    probe = next((m for m in model.modules() if isinstance(m, GA)), None)
    if probe is None:
        raise RuntimeError("attn_layout: the model has no GridSelfAttention module (the kit's pair stack changed shape)")
    for name in ("q_projection", "k_projection", "v_projection", "gating_query", "output_projection", "num_head"):
        if not hasattr(probe, name):
            raise RuntimeError(f"attn_layout: GridSelfAttention has no {name!r} (the kit's module changed shape)")
    reason = None
    if _is_stock(GA._attention, am.__name__, "GridSelfAttention._attention") and _is_stock(GA.forward, am.__name__, "GridSelfAttention.forward"):
        GA._stock_attention = GA._attention
        GA._attention = _attention
        installed = True
    else:
        installed = False; reason = "class_rebound_by_a_kernel_lever"
    GA._attn_layout = True if installed else "aside"; GA._attn_layout_reason = reason
    return {"installed": installed, "already": False, "modules": n, "reason": reason}
