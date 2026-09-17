# Row LayerNorm forward over the last dimension: one Triton kernel per row class, fp32 statistics, output in the input dtype.
import torch
import triton
import triton.language as tl


SURFACE = ("LNPolicy", "patch_flat", "shim_of")   # the names this package's callers reach (a kit's lever-surface test checks they exist, statically)

@triton.jit
def _ln_rows_kernel(X, W, B, Y, M, eps,
                    N: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_M: tl.constexpr,
                    HAS_W: tl.constexpr, HAS_B: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    rmask = rows < M
    cmask = cols < N
    mask = rmask[:, None] & cmask[None, :]
    offs = rows.to(tl.int64)[:, None] * N + cols[None, :]
    x = tl.load(X + offs, mask=mask, other=0.0)
    mean = tl.sum(x, 1) / N
    d = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(d * d, 1) / N
    rstd = tl.div_rn(1.0, tl.sqrt_rn(var + eps))
    y = d * rstd[:, None]
    if HAS_W:
        w = tl.load(W + cols, mask=cmask, other=0.0)
        y = y * w[None, :]
    if HAS_B:
        b = tl.load(B + cols, mask=cmask, other=0.0)
        y = y + b[None, :]
    tl.store(Y + offs, y, mask=mask)


def _cfg(N):
    BLOCK_N = max(16, triton.next_power_of_2(N))
    if BLOCK_N <= 64:
        BLOCK_M = 4096 // BLOCK_N; nw = 8
    elif BLOCK_N <= 1024:
        BLOCK_M = max(1, 4096 // BLOCK_N); nw = 4
    elif BLOCK_N <= 4096:
        BLOCK_M = 1; nw = 8
    else:
        BLOCK_M = 1; nw = 16
    return BLOCK_N, BLOCK_M, nw


def fast_layer_norm_triton(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    N = 1
    for s in normalized_shape:
        N *= int(s)
    xc = x.contiguous()
    M = xc.numel() // N
    y = torch.empty_like(xc)
    w = weight.contiguous() if weight is not None else xc
    b = bias.contiguous() if bias is not None else xc
    BLOCK_N, BLOCK_M, nw = _cfg(N)
    grid = (triton.cdiv(M, BLOCK_M),)
    _ln_rows_kernel[grid](xc, w, b, y, M, float(eps), N=N, BLOCK_N=BLOCK_N, BLOCK_M=BLOCK_M,
                          HAS_W=weight is not None, HAS_B=bias is not None, num_warps=nw)
    return y.view(x.shape)


class LNPolicy:
    """which torch.layer_norm calls go to the Triton kernel.
    mode='big' : fp32 CUDA, single normalized dim N <= 64 and M >= 16384 rows (PyTorch's one-block-per-row kernel is pathological there)
    mode='off' : never"""
    def __init__(self, mode="big"):
        self.mode = mode
        self.n_fast = 0; self.n_slow = 0

    def want(self, x, normalized_shape):
        if self.mode == "off" or not torch.is_tensor(x) or not x.is_cuda or x.dtype != torch.float32 or len(normalized_shape) != 1:
            return False
        N = int(normalized_shape[0]); M = x.numel() // max(N, 1)
        if self.mode == "big":
            return N <= 64 and M >= 16384
        raise ValueError(self.mode)

    def __call__(self, x, normalized_shape, weight=None, bias=None, eps=1e-5, cudnn_enable=True):
        if self.want(x, normalized_shape):
            self.n_fast += 1
            return fast_layer_norm_triton(x, normalized_shape, weight, bias, eps)
        self.n_slow += 1
        return torch.layer_norm(x, normalized_shape, weight, bias, eps, cudnn_enable)


# ---- binding into the ts2eager flat module ---------------------------------------------------------------------------------
def shim_of(flat):
    """the TorchShim instance bound to the name `torch` inside the transpiled methods of this flat component."""
    rt = object.__getattribute__(flat, "_rt")
    return rt.globals["torch"]


def patch_flat(flat, mode="big"):
    """rebind torch.layer_norm inside `flat`'s transpiled namespace (instance attribute on its TorchShim; other components untouched)."""
    shim = shim_of(flat)
    pol = LNPolicy(mode)
    shim.__dict__["layer_norm"] = pol
    return pol
