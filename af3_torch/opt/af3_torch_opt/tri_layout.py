"""Triangle-multiplication operand layout (lever ``tri_layout``): the stock triangle multiplication's three big layout copies are made by
one tiled transpose kernel each instead of torch's generic strided copy — the same bytes land in the same layouts, so the GEMM and the
layer norm that consume them run exactly as stock runs them (bitwise by construction: the lever moves data, it computes nothing).

Stock (``xfold/nn/triangle_multiplication.py`` ``TriangleMultiplication.forward``, the path ``off`` / ``exact`` execute, and the template
pair stack's c=64 rows under every mode): the dual projection ``[N, N, 2c]`` is viewed channel-major (``permute(2, 0, 1)``), masked and
gated as that strided view, split into the interleaved halves ``a`` = channels 0::2, ``b`` = 1::2 (``reshape(c, 2, N, N)``), and handed to
``torch.einsum`` — whose batched GEMM cannot address either half (no unit stride) and first CLONES each into a contiguous ``[c, N, N]``
operand (torch ``bmm``: ``prepare_batch_matrix_for_cublas`` → ``clone(contiguous)``; for the left operand of ``'ckj,cki->cij'`` and the
right of ``'cik,cjk->cij'`` that clone also swaps the two token axes); the ``[c, N, N]`` product is viewed ``permute(1, 2, 0)`` and made
contiguous ``[N, N, c]`` inside ``center_norm`` (fastnn LayerNorm's ``x.contiguous()`` / torch layer_norm). Those three copies read with
a stride of N·N (or 2c) elements between neighbouring lanes: at 800 tokens they are the largest kernels of the ``exact`` trunk.

Here (``forward`` below, the stock statements restated): the mask and gate products are taken on the projection as the Linear wrote it
(``[N, N, 2c]``, both factors contiguous: the same products of the same elements), the two GEMM operands are written directly in the
layouts stock's clones have (``operands``: contiguous ``[c, N, N]``, token axes in the order the equation's GEMM reads them) and the
product is written ``[N, N, c]`` contiguous (``channels_last``) — each by ``_tile_copy``, a Triton kernel that loads a tile coalesced along
the source's unit-stride axis and stores it coalesced along the destination's (loads and stores only). ``torch.einsum`` then receives
operands its GEMM addresses as they are (the same cuBLAS call: same transposition flags, sizes and leading dimensions as after stock's
clones) and ``center_norm`` a contiguous input (its ``.contiguous()`` returns it as is). Under ``fast`` / ``big`` the ``trimul`` kernel
lever owns this module's class forward and falls back by name to the stock forward for what it does not serve (the template pair stack's
c=64 rows); ``install`` makes that fallback this forward (``scope=fallback``), so every row the stock statements serve gets the tiled copies. The census (``COUNTS``)
rides forward.json (``tri_layout``) and the LEVER line; tests/test_tri_layout.py holds the restatement to the kit's source.
"""
from __future__ import annotations

import types

COUNTS = {"calls": 0, "tiled": 0, "generic": 0}      # forward calls; calls whose copies ran on the tiled kernel; calls that took the stock strided path (CPU / unsupported)
_STATE = {"kernel": None}


def take() -> dict:
    out = dict(COUNTS)
    for k in COUNTS:
        COUNTS[k] = 0
    return out


def _kernel():
    """The tiled copy kernel, built on first use (Triton imported in the model process only)."""
    if _STATE["kernel"] is not None:
        return _STATE["kernel"]
    import triton
    import triton.language as tl

    @triton.jit
    def _tile_copy(SRC, DST, X, Y, s_sb, s_sx, s_sy, s_db, s_dx, s_dy, BX: tl.constexpr, BY: tl.constexpr):
        # DST[b, x, y] = SRC[b, x, y] over explicit element strides: a pure copy between two layouts of one [B, X, Y] index space. The
        # source is (near-)unit-stride along x — the tile's fast axis, so a warp's loads cover whole lines — and the destination is
        # unit-stride along y (s_dy == 1, which Triton specialises): the store is laid along y and the tile crosses from one register
        # layout to the other through shared memory. Loads and stores only: the destination holds the source's bytes.
        b = tl.program_id(2).to(tl.int64)
        xs = (tl.program_id(0) * BX + tl.arange(0, BX)).to(tl.int64)
        ys = (tl.program_id(1) * BY + tl.arange(0, BY)).to(tl.int64)
        m = (ys[:, None] < Y) & (xs[None, :] < X)
        v = tl.load(SRC + b * s_sb + ys[:, None] * s_sy + xs[None, :] * s_sx, mask=m)
        tl.store(DST + b * s_db + ys[:, None] * s_dy + xs[None, :] * s_dx, v, mask=m)

    _STATE["kernel"] = _tile_copy
    return _tile_copy


BX, BY = 64, 64


def tile_copy(src, dst, B, X, Y, s_src, s_dst):
    """dst[b, x, y] = src[b, x, y] for b < B, x < X, y < Y with element strides s_src / s_dst = (batch, x, y) into the two tensors' storage."""
    import triton
    grid = (triton.cdiv(X, BX), triton.cdiv(Y, BY), B)
    _kernel()[grid](src, dst, X, Y, s_src[0], s_src[1], s_src[2], s_dst[0], s_dst[1], s_dst[2], BX=BX, BY=BY, num_warps=4)
    return dst


def operands(projection, c, equation):
    """The two einsum operands as stock's GEMM receives them, from the gated projection ``[N, N, 2c]`` (contiguous; channel 2k = a_k,
    2k+1 = b_k): contiguous ``[c, N, N]`` tensors — for ``'cik,cjk->cij'`` a[c, i, k] = P[i, k, 2c] and b laid (c, k, j): P[j, k, 2c+1];
    for ``'ckj,cki->cij'`` a laid (c, j, k): P[k, j, 2c] and b[c, k, i] = P[k, i, 2c+1]. Returned as the views einsum's subscripts name
    (``a``: 'cik' / 'ckj', ``b``: 'cjk' / 'cki'), i.e. exactly the tensors stock passes, over storage its clones would have produced."""
    import torch
    N = projection.shape[0]
    assert projection.shape == (N, N, 2 * c) and projection.is_contiguous(), (tuple(projection.shape), projection.stride())
    kept = torch.empty((c, N, N), dtype=projection.dtype, device=projection.device)      # [c, p, q] = P[p, q, 2c + par]
    swapped = torch.empty((c, N, N), dtype=projection.dtype, device=projection.device)   # [c, q, p] = P[p, q, 2c + par]
    outgoing = equation == 'cik,cjk->cij'
    par_kept, par_swapped = (0, 1) if outgoing else (1, 0)
    sP = projection.stride()                                                              # (N*2c, 2c, 1)
    # kept: index space (b=p, x=c, y=q): source P[p, q, 2c+par] -> strides (sP[0], 2, sP[1]) (x near-unit: every other channel); destination kept[c, p, q] -> (N, N*N, 1)
    tile_copy(projection[:, :, par_kept:], kept, N, c, N, (sP[0], 2 * sP[2], sP[1]), (N, N * N, 1))
    # swapped: index space (b=q, x=c, y=p): source -> (sP[1], 2, sP[0]); destination swapped[c, q, p] -> (N, N*N, 1)
    tile_copy(projection[:, :, par_swapped:], swapped, N, c, N, (sP[1], 2 * sP[2], sP[0]), (N, N * N, 1))
    if outgoing:                                      # a = 'cik' = kept (even channels) as is; b = 'cjk' over storage laid (c, k, j) = swapped (odd) viewed with its token axes exchanged
        return kept, swapped.transpose(1, 2)
    return swapped.transpose(1, 2), kept              # a = 'ckj' over storage laid (c, j, k) = swapped (even channels); b = 'cki' = P[k, i, 2c+1] = kept (odd) as is


def channels_last(pair):
    """``[c, N, N]`` (any strides: einsum's product, a permuted view for the incoming equation) -> contiguous ``[N, N, c]``: the tensor
    stock's ``pair.permute(1, 2, 0)`` becomes inside ``center_norm``."""
    import torch
    c, N, _ = pair.shape
    out = torch.empty((N, N, c), dtype=pair.dtype, device=pair.device)
    sc, si, sj = pair.stride()
    if sj <= si:                                      # product laid (c, i, j): index space (b=i, x=j, y=c) — source unit-stride along j, destination along c
        tile_copy(pair, out, N, N, c, (si, sj, sc), (N * c, c, 1))
    else:                                             # product laid (c, j, i) (the incoming equation's permuted result): (b=j, x=i, y=c)
        tile_copy(pair, out, N, N, c, (sj, si, sc), (c, N * c, 1))
    return out


def _generic(projection, mask):
    """The stock statements from the channel-major view on (CPU tensors, or a projection the tiled path does not take)."""
    import torch
    projection = projection.permute(2, 0, 1)
    if mask is not None:
        projection *= mask[None, ...]
    return projection


def forward(self, pair, mask):
    """
    Args:
        pair (torch.Tensor): [N_token, N_token, c_pair]
        mask (torch.Tensor): [N_token]
    Returns:
        torch.Tensor: [N_token, N_token, c_pair]
    """
    import torch
    COUNTS["calls"] += 1

    pair = self.left_norm_input(pair)
    input_pair = pair

    projection = self.projection(pair)
    tiled = projection.is_cuda and projection.dim() == 3 and projection.is_contiguous() and (mask is None or mask.dim() == 2)
    if not tiled:                                     # not this lever's case (CPU, batched leading dims, a 1-D mask): the stock statements, counted
        COUNTS["generic"] += 1
        return _stock_tail(self, projection, pair, input_pair, mask)
    COUNTS["tiled"] += 1
    gate = self.gate(pair)
    projection = _gate().mask_gate_(projection, gate, mask)   # stock: `projection *= mask[None, ...]` on the (ch, i, j) view, then `projection *= torch.sigmoid(gate)` —
                                                              # the same factors on the same elements where the Linear wrote them; one fused kernel under lever gate_fuse

    a, b = operands(projection, self.c_pair, self.equation)   # stock: reshape(c, 2, N, N) + chunk -> strided halves einsum's GEMM clones contiguous
    pair = torch.einsum(self.equation, a, b)

    pair = channels_last(pair)                        # stock: pair.permute(1, 2, 0), made contiguous by center_norm's first statement
    pair = self.center_norm(pair)
    pair = self.output_projection(pair)

    gate_out = self.gating_linear(input_pair)
    pair = _gate().mask_gate_(pair, gate_out)         # stock: pair *= torch.sigmoid(gate_out); one fused kernel under lever gate_fuse

    return pair


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


def _stock_tail(self, projection, pair, input_pair, mask):
    """TriangleMultiplication.forward from its third statement on, verbatim (the generic path)."""
    import torch
    projection = projection.permute(2, 0, 1)
    if mask is not None:
        projection *= mask[None, ...]

    gate = self.gate(pair)
    gate = gate.permute(2, 0, 1)
    projection *= torch.sigmoid(gate)

    projection = projection.reshape(self.c_pair, 2, *projection.shape[1:])

    a, b = torch.chunk(projection, 2, dim=1)
    a, b = torch.squeeze(a, dim=1), torch.squeeze(b, dim=1)
    pair = torch.einsum(self.equation, a, b)

    pair = pair.permute(1, 2, 0)
    pair = self.center_norm(pair)
    pair = self.output_projection(pair)

    gate_out = self.gating_linear(input_pair)
    pair *= torch.sigmoid(gate_out)

    return pair


def _is_stock_forward(f, tm) -> bool:
    """True when ``f`` is the kit's own TriangleMultiplication.forward (no lever's function)."""
    return getattr(f, "__module__", None) == tm.__name__ and getattr(f, "__qualname__", "") == "TriangleMultiplication.forward"


def _kernel_adapters():
    """The kit's kernel-lever adapter module (af3_kernels) — every module object it is imported as in this process — that carries its
    table of the stock forwards it rebound (``_ORIG``)."""
    import sys
    return [m for m in list(sys.modules.values())
            if (getattr(m, "__file__", None) or "").endswith("af3_kernels.py") and isinstance(getattr(m, "_ORIG", None), dict)]


def install(model) -> dict:
    """Bind ``forward`` where the stock forward serves. ``scope``: ``class`` — the class forward is still the kit's own function (no kernel
    lever rebound it: ``exact``, or ``trimul`` ablated) and is replaced; ``fallback`` — the ``trimul`` kernel lever owns the class forward
    (``fast`` / ``big``) and falls back BY NAME to the stock forward it saved (``af3_kernels._ORIG['trimul']``: the template pair stack's
    c=64 rows, a lever gone dead), which now is this forward (that lever keeps its residual statement and its FALLBACK census); ``none`` —
    neither holds (named, nothing bound). Idempotent. {'installed', 'already', 'scope', 'reason'}."""
    import importlib
    tm = importlib.import_module("xfold.nn.triangle_multiplication")
    TM = tm.TriangleMultiplication
    if getattr(TM, "_tri_layout", None):
        return {"installed": True, "already": True, "scope": TM._tri_layout, "reason": getattr(TM, "_tri_layout_reason", None)}
    found = [m for m in model.modules() if isinstance(m, TM)]
    if not found:
        raise RuntimeError("tri_layout: the model has no TriangleMultiplication module (the kit's pair stack changed shape)")
    for name in ("projection", "gate", "center_norm", "output_projection", "gating_linear", "left_norm_input", "equation", "c_pair"):
        if not hasattr(found[0], name):
            raise RuntimeError(f"tri_layout: TriangleMultiplication has no {name!r} (the kit's module changed shape)")
    reason = None
    if _is_stock_forward(TM.forward, tm):
        TM._stock_forward = TM.forward
        TM.forward = forward
        scope = "class"
    else:
        aks = [ak for ak in _kernel_adapters() if _is_stock_forward(ak._ORIG.get("trimul"), tm)]
        if aks:
            TM._stock_forward = aks[0]._ORIG["trimul"]
            for ak in aks:
                ak._ORIG["trimul"] = forward
            scope = "fallback"
        else:
            scope = "none"; reason = "class_forward_rebound_and_no_stock_fallback_table" if not _kernel_adapters() else "adapter_fallback_is_not_the_stock_forward"
    TM._tri_layout = scope; TM._tri_layout_reason = reason
    return {"installed": True, "already": False, "scope": scope, "reason": reason}
