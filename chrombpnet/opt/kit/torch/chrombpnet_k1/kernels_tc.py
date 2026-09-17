"""K1 tensor-core variant: the same im2col dot as kernels.py with tl.dot(input_precision='tf32') — the operands rounded to TF32 and multiplied on the
tensor cores, fp32 accumulation: the precision class stock itself convolves in by default on Ampere / Hopper (TF32). Not bitwise to any stock output;
the bitwise route is the fp32 FFMA chains of kernels.py, and this file is never on that path. kernels.py stays byte-identical (its sha256 is the
shipped caches' pin); these kernels carry their own tile (_TILE / set_tile) and their own cache entries. Epilogue as kernels.py: + bias -> ReLU ->
+ residual. The profile head and the counts head stay on kernels.py's fp32 kernels (K=75 conv / GAP + dense).
"""
import torch, triton, triton.language as tl
from .kernels import K1BPNetV3, head_serial, conv_dot, exact_counts, COMPILED

_TILE = "128x128x32x8x3"          # BLOCK_M x BLOCK_N x BLOCK_K x num_warps x num_stages of the tensor-core dot; apply(tc_tile=...) sets the arch's value from arch_tiles.json tc_tile, this default serves otherwise
_BM, _BN, _BK, _NW, _NS = (int(v) for v in _TILE.split("x"))


def set_tile(tile):
    global _BM, _BN, _BK, _NW, _NS
    _BM, _BN, _BK, _NW, _NS = (int(v) for v in tile.split("x"))


@triton.jit
def _im2col_dot_tc_kernel(X, WT, B, Y, L, Lout, dil, R, C: tl.constexpr, K: tl.constexpr, Co: tl.constexpr, RELU: tl.constexpr, RESID: tl.constexpr,
                          BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    n = tl.program_id(2); pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    l = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); co = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    lm = l < Lout; cm = co < Co
    xbase = X + n * C * L
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    j = tl.arange(0, BLOCK_K)
    for r0 in range(0, R, BLOCK_K):
        r = r0 + j
        rm = r < R
        c = r // K
        k = r % K
        a = tl.load(xbase + c[None, :] * L + l[:, None] + k[None, :] * dil, mask=lm[:, None] & rm[None, :], other=0.0)   # [BM, BK]
        w = tl.load(WT + r[:, None] * Co + co[None, :], mask=rm[:, None] & cm[None, :], other=0.0)                        # [BK, BN]
        acc = tl.dot(a, w, acc, input_precision="tf32")
    b = tl.load(B + co, mask=cm, other=0.0)
    y = acc + b[None, :]
    if RELU:
        y = tl.maximum(y, 0.0)
    if RESID:
        xr = tl.load(xbase + co[None, :] * L + l[:, None] + dil * (K // 2), mask=lm[:, None] & cm[None, :], other=0.0)
        y = xr + y
    tl.store(Y + n * Co * Lout + co[None, :] * Lout + l[:, None], y, mask=lm[:, None] & cm[None, :])


def conv_dot_tc(x, wt, b, K, dil=1, relu=True, resid=False):
    N, C, L = x.shape; Co = wt.shape[1]; R = C * K; Lout = L - (K - 1) * dil
    BM = _BM; BN = min(_BN, max(16, triton.next_power_of_2(Co))); BK = _BK
    y = torch.empty((N, Co, Lout), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(Lout, BM), triton.cdiv(Co, BN), N)
    h = _im2col_dot_tc_kernel[grid](x, wt, b, y, L, Lout, dil, R, C, K, Co, relu, resid, BM, BN, BK, num_warps=_NW, num_stages=_NS)
    COMPILED[("dot_tc", C, Co, K, dil, relu, resid, BM, BN, BK)] = h
    return y


class K1BPNetV3TC(K1BPNetV3):
    """The sub-model on the tensor-core convs: iconv + the dilated stack on conv_dot_tc; the profile head, the counts head and every epilogue as K1BPNetV3."""
    def forward(self, x):
        h = conv_dot_tc(x, self.iwt, self.ib, self.iK, dil=1, relu=True, resid=False)
        for wt, b, d in zip(self.rwt, self.rb, self.dils):
            h = conv_dot_tc(h, wt, b, 3, dil=d, relu=True, resid=True)
        if self.head == "serial":
            prof = head_serial(h, self.fw_flat, self.fb1, self.fK)
        else:
            prof = conv_dot(h, self.fwt, self.fb, self.fK, dil=1, relu=False, resid=False, BLOCK_N=self.head_pad)[:, 0, :]
        Lp = prof.shape[1]; s = (Lp - self.out_len) // 2
        prof = prof[:, s:s + self.out_len]
        cnt = self.linear(h.mean(dim=2)).reshape(-1, 1) if self.counts_mode == "torch" else exact_counts(h, self.linear, self.counts_form)
        return prof, cnt
