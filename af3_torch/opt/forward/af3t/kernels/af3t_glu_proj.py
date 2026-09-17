# af3t_glu_proj.py -- xfold's gated-linear-unit statement and the transition's output projection as ONE Triton kernel, inference only.
#
# The statement it restates (xfold @ 22bdeed, `--fastnn`: xfold/fastnn/gated_linear_unit.py::gated_linear_unit_triton + nn/primitives.py::Transition):
#   y   = input_layer_norm(x)                                  [given: the module's own layer norm serves it — this kernel never restates the norm]
#   a|b = y @ W1^T | y @ W2^T   (tl.dot, fp32 accumulators; W1 = transition1.weight[:HID] (silu branch), W2 = transition1.weight[HID:])
#   h   = (a * sigmoid(a) * b) -> y.dtype                      [ONE rounding: the fastnn kernel's own `acc0 * tl.sigmoid(acc0) * acc1` stored in y's dtype]
#   out = h(bf16) @ W3^T (fp32 accumulate over HID in ascending order) -> bf16   [stock: transition2 under bf16 autocast = cuBLAS bf16 GEMM, fp32 accumulate]
#   (+ x: `pair += out`, torch's add on the pair stream: fp32 opmath, one rounding to x's dtype)          [residual=True]
# The HID-wide h never touches HBM. Rounding points AND accumulation order are the stock kernels': the dual GEMM is the same tl.dot chain the fastnn
# kernel issues (k ascending into one fp32 accumulator), the projection accumulates HID ascending into one fp32 accumulator from zero exactly as a
# one-pass GEMM does. Whether the cuBLAS kernel torch selects for `[M, HID] x [HID, C]` on this device / library is such a one-pass kernel is a FACT
# the caller checks once per (M-bucket, C, HID, dtype) in process (af3_kernels: torch.equal against the stock statement on the first call of a shape
# class; a class that differs is served by stock BY NAME) — this module makes no claim it cannot check. Deterministic (no atomics, no split-K); the
# launch tile (BM / BH / warps / stages: ONE fixed tile, _TILE) changes neither the rounding points nor the accumulation sequence (every candidate in _CFGS keeps BM >= 64: one MMA class).
# int64 row offsets before `* C` / `* HID` (rows * 512 passes 2**31 - 1 past 2048 tokens; upstream issue XFOLD-003).
import torch
import triton
import triton.language as tl


@triton.jit
def _glu_proj_kernel(Y, X, OUT, W1, W2, W3, M, MB,
                     C: tl.constexpr, HID: tl.constexpr, RESIDUAL: tl.constexpr, BM: tl.constexpr, BH: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BM + tl.arange(0, BM)
    rmask = rows < M
    r64 = rows.to(tl.int64)
    cols = tl.arange(0, C)
    y = tl.load(Y + r64[:, None] * C + cols[None, :], mask=rmask[:, None], other=0.0)          # [BM, C] in y's dtype (bf16: bf16 MMA; fp32: the fastnn kernel's own fp32 tl.dot)
    acc = tl.zeros([BM, C], dtype=tl.float32)
    for h0 in range(0, HID, BH):
        hcols = h0 + tl.arange(0, BH)
        w1t = tl.load(W1 + hcols[None, :] * C + cols[:, None])                                  # [C, BH] = W1[hchunk, :]^T  (W1 is [HID, C] row-major, y's dtype)
        w2t = tl.load(W2 + hcols[None, :] * C + cols[:, None])
        a = tl.zeros([BM, BH], dtype=tl.float32)
        b = tl.zeros([BM, BH], dtype=tl.float32)
        a = tl.dot(y, w1t, a)                                                                    # the fastnn kernel's statement: acc0 = tl.dot(x, w, acc0) over K = C
        b = tl.dot(y, w2t, b)
        h = (a * tl.sigmoid(a) * b).to(Y.dtype.element_ty).to(tl.bfloat16)                      # its `out = acc0 * tl.sigmoid(acc0) * acc1` stored in y's dtype; then autocast's bf16 operand cast (a no-op on bf16)
        w3t = tl.load(W3 + cols[None, :] * HID + hcols[:, None])                                # [BH, C] = W3[:, hchunk]^T  (W3 is [C, HID] row-major, bf16)
        acc = tl.dot(h, w3t, acc)                                                                # HID ascending into one fp32 accumulator from zero: a one-pass GEMM's sequence
    out16 = acc.to(tl.bfloat16)                                                                  # the bf16 GEMM's output rounding
    optr = OUT + r64[:, None] * C + cols[None, :]
    if RESIDUAL:
        x = tl.load(X + r64[:, None] * C + cols[None, :], mask=rmask[:, None], other=0.0)
        tl.store(optr, (x.to(tl.float32) + out16.to(tl.float32)).to(X.dtype.element_ty), mask=rmask[:, None])   # torch's `x += out`: fp32 opmath, one rounding to x's dtype
    else:
        tl.store(optr, out16, mask=rmask[:, None])


# Every config keeps BM >= 64 and 4 | 8 warps (one MMA instruction class on sm_80 / sm_90); BH divides every HID this model has (128 / 256 / 512).
_CFGS = [triton.Config({"BM": 128, "BH": 64}, num_warps=8, num_stages=2), triton.Config({"BM": 64, "BH": 64}, num_warps=4, num_stages=2),
         triton.Config({"BM": 128, "BH": 128}, num_warps=8, num_stages=2), triton.Config({"BM": 64, "BH": 128}, num_warps=4, num_stages=2),
         triton.Config({"BM": 128, "BH": 64}, num_warps=4, num_stages=2), triton.Config({"BM": 64, "BH": 64}, num_warps=4, num_stages=3),
         triton.Config({"BM": 128, "BH": 64}, num_warps=8, num_stages=3), triton.Config({"BM": 64, "BH": 32}, num_warps=4, num_stages=2)]
_TILE = {"BM": 64, "BH": 64, "num_warps": 4, "num_stages": 2}   # the ONE launch tile of this kernel (the tile it has run with on every measured card, cc 9.0 / 8.0; a member of _CFGS' MMA class); nothing is timed

SUPPORTED_C = (64, 128)            # the y tile is [BM, C] whole (tl.arange: a power of two); c = 384 (the single transition) is the stock statement's (K padding 384 -> 512 loses to cuBLAS)
SUPPORTED_DTYPES = (torch.bfloat16, torch.float32)


def row_bucket(M):
    """The launch key's row bucket (a kernel argument kept for the launch's key form; ONE tile serves every bucket) — also the grain of the caller's once-per-shape-class check."""
    return 0 if M < 16384 else (1 if M < 131072 else (2 if M < 1048576 else 3))


def glu_proj(y, W1, W2, W3, residual=None, out=None):
    """y [..., C] (bf16 | fp32, CUDA, the layer norm's output), W1 / W2 [HID, C] contiguous in y's dtype, W3 [C, HID] bf16 contiguous ->
    [..., C]: bf16 (residual None) or residual's dtype (residual: x [..., C], the block's `x + update` folded into the epilogue).
    out = (silu-gate(y W1^T) * (y W2^T)).bf16 @ W3^T (+ x)."""
    C = y.shape[-1]; HID = W1.shape[0]
    assert C in SUPPORTED_C and y.dtype in SUPPORTED_DTYPES, (C, y.dtype)
    assert W1.shape == (HID, C) and W2.shape == (HID, C) and W3.shape == (C, HID), (W1.shape, W2.shape, W3.shape)
    assert W1.dtype == y.dtype and W2.dtype == y.dtype and W3.dtype == torch.bfloat16, (W1.dtype, W2.dtype, W3.dtype, y.dtype)
    assert HID % 32 == 0 and W1.is_contiguous() and W2.is_contiguous() and W3.is_contiguous()
    ys = y.contiguous().view(-1, C); M = ys.shape[0]
    if residual is not None:
        assert residual.shape == y.shape, (residual.shape, y.shape)
        xs = residual.contiguous().view(-1, C)
        o = torch.empty((M, C), device=y.device, dtype=residual.dtype) if out is None else out.view(-1, C)
    else:
        xs = ys
        o = torch.empty((M, C), device=y.device, dtype=torch.bfloat16) if out is None else out.view(-1, C)
    grid = (triton.cdiv(M, _TILE["BM"]),)
    _glu_proj_kernel[grid](ys, xs, o, W1, W2, W3, M, row_bucket(M), C=C, HID=HID, RESIDUAL=residual is not None, **_TILE)
    return o.view(y.shape)


def stock_statement(y, w_t1, w_t2, residual=None):
    """The statement this kernel restates, through the stock modules' own code paths: xfold's fastnn gated-linear-unit Triton kernel on y with
    transition1's weight, transition2 under the ambient autocast (bf16: cuBLAS), then the block's in-place add on a copy of x. w_t1 = transition1.weight
    [2 HID, C], w_t2 = transition2.weight [C, HID] (the module's own parameters, fp32 or bf16 as the model holds them)."""
    from xfold.fastnn.gated_linear_unit import gated_linear_unit_triton
    h = gated_linear_unit_triton(y, w_t1.T)
    upd = torch.nn.functional.linear(h, w_t2)
    if residual is None:
        return upd
    x = residual.clone()
    x += upd
    return x
