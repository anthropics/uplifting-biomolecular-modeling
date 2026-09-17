"""fused / _hs_write.py — ONE scatter-with-cast launch per collected hidden state on the PADDED fast route.
out[idx[t], :] = float32(h[t, :]) for t < T  (h bf16 or fp32 [T, D]; idx int64 [T] = the forward's own unpad `indices`; out fp32 [B*L, D],
zero-filled once per forward for the pads) = the stock's pad_input(h) followed by torch.stack's promotion — a copy with an exact widening,
no arithmetic: bitwise by construction. Triton (JIT on first use per source dtype); without Triton the torch form (h.float() + index_put_).
"""
import torch; from esmc_opt._oom import is_oom

_KERNEL = None
_COUNTERS = {"triton_calls": 0, "torch_calls": 0, "jit_compiles": 0}


def _build():
    global _KERNEL
    if _KERNEL is not None:
        return _KERNEL
    import triton
    import triton.language as tl

    @triton.jit
    def _hs_scatter_cast(h_ptr, idx_ptr, out_ptr, D, BLOCK_D: tl.constexpr):
        t = tl.program_id(0)
        dst = tl.load(idx_ptr + t).to(tl.int64)
        offs = tl.arange(0, BLOCK_D)
        for d0 in range(0, D, BLOCK_D):
            m = (d0 + offs) < D
            v = tl.load(h_ptr + t.to(tl.int64) * D + d0 + offs, mask=m, other=0.0)
            tl.store(out_ptr + dst * D + d0 + offs, v.to(tl.float32), mask=m)

    _KERNEL = _hs_scatter_cast
    return _KERNEL


def scatter_cast(out_row: torch.Tensor, idx: torch.Tensor, h: torch.Tensor) -> None:
    """out_row: fp32 [B*L, D] contiguous (pads already zero); idx: int64 [T] on device; h: [T, D] contiguous bf16/fp32."""
    T, D = int(h.shape[0]), int(h.shape[1])
    assert out_row.dtype == torch.float32 and out_row.is_contiguous() and h.is_contiguous() and idx.dtype == torch.int64 and idx.numel() == T
    if T == 0:
        return
    try:
        k = _build()
    except Exception as _e:  # noqa: BLE001  (no Triton: the torch form, the same bytes)
        if is_oom(_e): raise                                  # an out-of-memory propagates: no fallback applied
        k = None
    if k is None:
        out_row[idx] = h.float(); _COUNTERS["torch_calls"] += 1
        return
    k[(T,)](h, idx, out_row, D, BLOCK_D=1024)
    _COUNTERS["triton_calls"] += 1
