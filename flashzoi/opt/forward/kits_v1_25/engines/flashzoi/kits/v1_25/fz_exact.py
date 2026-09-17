"""The Flashzoi kit's kernel module: the Triton kernels of the exact levers (stage 1, the pool -> BatchNorm -> GELU sites, max-pool,
the NHWC->NCHW bias pass, up-sample + add, the fused LayerNorms, the softplus head), the fused NCHW stack `FastNCHW` that composes them over a
stock Borzoi model's modules, the one-time fp16 weight pre-cast, the CUDA-graph holder `Graphed` and the pinned output lease pool. The
activation dtype is fp16, the stock's own (torch.autocast's default). Imported by the wrapper only (`_wrap.build`); nothing outside the kit
is imported here."""
import collections
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import triton, triton.language as tl
try:
    from triton.language.extra import libdevice
except ImportError:  # triton 3.1 layout
    from triton.language.extra.cuda import libdevice

L = 524288
dev = torch.device("cuda")
ACT_DTYPE = torch.float16                     # the activation dtype: the stock's own (torch.autocast("cuda") default)
HEAD_ROWS = 7611                                  # the human head's rows: the GEMM shape the head lever serves (7611 x 1920); any other head runs upstream's conv
TL_DT = tl.float16
KBETA = float(np.float32(np.float64(1.4142135623730951) * np.float64(1.1283791670955126) * np.float64(0.5))); KKAPPA = float(np.float32(0.044715))


@torch.no_grad()
def fwd(m, x):
    with torch.autocast("cuda", dtype=ACT_DTYPE): return m(x)


def bn_invstd(bn):
    """the exact fp32 invstd PyTorch uses in eval mode (rsqrtf(var+eps), batch_norm_calc_invstd) — native_batch_norm returns it"""
    probe = torch.zeros(1, bn.num_features, 2, device=dev, dtype=ACT_DTYPE)
    return torch.native_batch_norm(probe, bn.weight, bn.bias, bn.running_mean, bn.running_var, False, 0.0, bn.eps)[2].contiguous()


def precast_bf16(m, dtype=None):
    dtype = dtype or ACT_DTYPE
    n = 0
    for name, mod in m.named_modules():
        if name in ("human_head", "mouse_head"): continue
        if isinstance(mod, (nn.Conv1d, nn.Linear)):
            mod.weight.data = mod.weight.data.to(dtype); n += 1
            if mod.bias is not None: mod.bias.data = mod.bias.data.to(dtype)
    return n


# ------------------------------------------------------------------------------------------------ Triton kernels
@triton.jit
def rnd(v, DT: tl.constexpr):
    """round an fp32 value to DT and back to fp32 with an UNFOLDABLE rounding: a `.to(DT).to(f32)` cast pair can
    be folded by the compiler (and the neighbouring mul+add contracted), silently removing the intermediate rounding the stock performs.
    fp16: explicit PTX cvt.rn.f16.f32 / cvt.f32.f16 via inline asm; bf16: integer-bit RNE on the fp32 bit pattern."""
    if DT == tl.float16:
        return tl.inline_asm_elementwise("{ .reg .b16 t; cvt.rn.f16.f32 t, $1; cvt.f32.f16 $0, t; }", "=r,r", [v], dtype=tl.float32, is_pure=True, pack=1)
    else:
        b = v.to(tl.int32, bitcast=True)
        b = (b + 0x7FFF + ((b >> 16) & 1)) & -65536          # RNE to bf16 on the bit pattern (& 0xFFFF0000)
        return b.to(tl.float32, bitcast=True)


@triton.jit
def k_stage1_fma(IDX, WMAT, BIAS, MEAN, INVSTD, GAMMA, BETA, Y, Lx, Lo, KBETA, KKAPPA, OUT_NHWC: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_C: tl.constexpr, DT: tl.constexpr, ORDER: tl.constexpr):
    """one-hot conv (4->512,k15,'same') as 60 rank-1 fp32 FMA updates + bf16 round + bias (bf16) + maxpool2 + BN + GELU, one pass.
    exact: products are 0 or the bf16 weight; sums of <=15 bf16 weights are exactly representable in fp32 (measured: 5 orders bitwise)."""
    pid_q = tl.program_id(0); pid_c = tl.program_id(1); n = tl.program_id(2)
    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q); cidx = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    acc_e = tl.zeros((BLOCK_Q, BLOCK_C), dtype=tl.float32); acc_o = tl.zeros((BLOCK_Q, BLOCK_C), dtype=tl.float32)
    for k in tl.static_range(15):
        pe = 2 * q + (k - 7); po = pe + 1
        be = tl.load(IDX + n * Lx + pe, mask=(q < Lo) & (pe >= 0) & (pe < Lx), other=127).to(tl.int32)
        bo = tl.load(IDX + n * Lx + po, mask=(q < Lo) & (po >= 0) & (po < Lx), other=127).to(tl.int32)
        for b in tl.static_range(4):
            w = tl.load(WMAT + (k * 4 + b) * 512 + cidx)
            ae = (be == b).to(tl.float32); ao = (bo == b).to(tl.float32)
            acc_e = tl.fma(ae[:, None], w[None, :], acc_e); acc_o = tl.fma(ao[:, None], w[None, :], acc_o)
    bias = tl.load(BIAS + cidx).to(tl.float32)
    ce = rnd(rnd(acc_e, DT) + bias[None, :], DT)
    co = rnd(rnd(acc_o, DT) + bias[None, :], DT)
    v = tl.maximum(ce, co)
    mean = tl.load(MEAN + cidx); invstd = tl.load(INVSTD + cidx); gamma = tl.load(GAMMA + cidx); beta = tl.load(BETA + cidx)
    t1 = gamma[None, :] * (v - mean[None, :])
    v = rnd(tl.fma(t1, invstd[None, :], beta[None, :]), DT)
    x3 = v * v * v; inner = KBETA * tl.fma(KKAPPA, x3, v); v = 0.5 * v * (1.0 + libdevice.tanh(inner))
    if OUT_NHWC:
        tl.store(Y + (n * Lo + q[:, None]) * 512 + cidx[None, :], v.to(DT), mask=(q[:, None] < Lo))
    else:
        tl.store(Y + n * (512 * Lo) + cidx[None, :] * Lo + q[:, None], v.to(DT), mask=(q[:, None] < Lo))


@triton.jit
def k_stage1_gather(IDX, WMAT, BIAS, MEAN, INVSTD, GAMMA, BETA, Y, Lx, Lo, KBETA, KKAPPA, OUT_NHWC: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_C: tl.constexpr, DT: tl.constexpr, ORDER: tl.constexpr):
    """same as k_stage1_fma but each tap gathers its weight row by base (row 60 = zeros for N / padding) and adds it in fp32, in
    accumulation order ORDER (see _s1_gather_sum); the order matters only when the 15-term sum is inexact in fp32 (fp16 weights)."""
    pid_q = tl.program_id(0); pid_c = tl.program_id(1); n = tl.program_id(2)
    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q); cidx = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    acc_e = tl.zeros((BLOCK_Q, BLOCK_C), dtype=tl.float32); acc_o = tl.zeros((BLOCK_Q, BLOCK_C), dtype=tl.float32)
    for kk in tl.static_range(15):
        k = kk if ORDER == 0 else 14 - kk          # 0: tap ascending, 1: tap descending (fp32 FMA orders; exact only when the 15-term sum is exact)
        pe = 2 * q + (k - 7); po = pe + 1
        be = tl.load(IDX + n * Lx + pe, mask=(q < Lo) & (pe >= 0) & (pe < Lx), other=127).to(tl.int32)
        bo = tl.load(IDX + n * Lx + po, mask=(q < Lo) & (po >= 0) & (po < Lx), other=127).to(tl.int32)
        re = tl.where(be < 4, k * 4 + be, 60); ro = tl.where(bo < 4, k * 4 + bo, 60)
        acc_e = acc_e + tl.load(WMAT + re[:, None] * 512 + cidx[None, :])
        acc_o = acc_o + tl.load(WMAT + ro[:, None] * 512 + cidx[None, :])
    bias = tl.load(BIAS + cidx).to(tl.float32)
    ce = rnd(rnd(acc_e, DT) + bias[None, :], DT)
    co = rnd(rnd(acc_o, DT) + bias[None, :], DT)
    v = tl.maximum(ce, co)
    mean = tl.load(MEAN + cidx); invstd = tl.load(INVSTD + cidx); gamma = tl.load(GAMMA + cidx); beta = tl.load(BETA + cidx)
    t1 = gamma[None, :] * (v - mean[None, :])
    v = rnd(tl.fma(t1, invstd[None, :], beta[None, :]), DT)
    x3 = v * v * v; inner = KBETA * tl.fma(KKAPPA, x3, v); v = 0.5 * v * (1.0 + libdevice.tanh(inner))
    if OUT_NHWC:
        tl.store(Y + (n * Lo + q[:, None]) * 512 + cidx[None, :], v.to(DT), mask=(q[:, None] < Lo))
    else:
        tl.store(Y + n * (512 * Lo) + cidx[None, :] * Lo + q[:, None], v.to(DT), mask=(q[:, None] < Lo))


@triton.jit
def k_stage1_mma(IDX, W2, BIAS, MEAN, INVSTD, GAMMA, BETA, Y, Lx, Lo, KBETA, KKAPPA, OUT_NHWC: tl.constexpr, BLOCK_Q: tl.constexpr, BLOCK_C: tl.constexpr, DT: tl.constexpr, ORDER: tl.constexpr):
    """stage-1 with the conv as a TENSOR-CORE implicit GEMM exactly like cuDNN's fp16 xmma kernel: A = one-hot im2col rows (64 = 15 taps x 4
    bases, k index s*4+c, zero-padded to 64), B = W2 (64, 512) fp16, fp32 accumulate by the MMA unit (same k-grouping -> same arithmetic);
    then round to DT (the conv's fp16 output), + bias, pool, BN, GELU as in the FMA kernels."""
    pid_q = tl.program_id(0); pid_c = tl.program_id(1); n = tl.program_id(2)
    q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q); cidx = pid_c * BLOCK_C + tl.arange(0, BLOCK_C)
    col = tl.arange(0, 64); sidx = col // 4; cb = col % 4
    B = tl.load(W2 + col[:, None] * 512 + cidx[None, :])                                             # (64, BLOCK_C) fp16
    pe = 2 * q[:, None] + (sidx[None, :] - 7)                                                         # (BLOCK_Q, 64) input positions, even outputs
    me = (q[:, None] < Lo) & (pe >= 0) & (pe < Lx) & (col[None, :] < 60)
    be = tl.load(IDX + n * Lx + pe, mask=me, other=127).to(tl.int32)
    Ae = tl.where(be == cb[None, :], 1.0, 0.0).to(tl.float16)
    acc_e = tl.dot(Ae, B, out_dtype=tl.float32)
    po = pe + 1; mo = (q[:, None] < Lo) & (po >= 0) & (po < Lx) & (col[None, :] < 60)
    bo = tl.load(IDX + n * Lx + po, mask=mo, other=127).to(tl.int32)
    Ao = tl.where(bo == cb[None, :], 1.0, 0.0).to(tl.float16)
    acc_o = tl.dot(Ao, B, out_dtype=tl.float32)
    bias = tl.load(BIAS + cidx).to(tl.float32)
    ce = rnd(rnd(acc_e, DT) + bias[None, :], DT)
    co = rnd(rnd(acc_o, DT) + bias[None, :], DT)
    v = tl.maximum(ce, co)
    mean = tl.load(MEAN + cidx); invstd = tl.load(INVSTD + cidx); gamma = tl.load(GAMMA + cidx); beta = tl.load(BETA + cidx)
    t1 = gamma[None, :] * (v - mean[None, :])
    v = rnd(tl.fma(t1, invstd[None, :], beta[None, :]), DT)
    x3 = v * v * v; inner = KBETA * tl.fma(KKAPPA, x3, v); v = 0.5 * v * (1.0 + libdevice.tanh(inner))
    if OUT_NHWC:
        tl.store(Y + (n * Lo + q[:, None]) * 512 + cidx[None, :], v.to(DT), mask=(q[:, None] < Lo))
    else:
        tl.store(Y + n * (512 * Lo) + cidx[None, :] * Lo + q[:, None], v.to(DT), mask=(q[:, None] < Lo))


@triton.jit
def k_pool_bn_gelu_nhwc(X, Y, MEAN, INVSTD, GAMMA, BETA, BIAS0, n_win, C, KBETA, KKAPPA, ADD_BIAS0: tl.constexpr, BLOCK: tl.constexpr, DT: tl.constexpr):
    # X (N, 2Lo, C) row-major (= channels_last (N,C,1,2Lo)); Y (N, Lo, C); program_id(1) = the window n, whose slabs start at 64-bit bases (n * 2Lo*C in X,
    # n * Lo*C in Y — a batch whose whole tensor exceeds 2**31 elements indexes correctly); within the window out row i reads input rows 2i, 2i+1 at int32 offsets
    n64 = tl.program_id(1).to(tl.int64); Xn = X + n64 * n_win * 2; Yn = Y + n64 * n_win     # n_win = Lo * C, the window's output element count
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); mask = offs < n_win
    c = offs % C; row = offs // C
    a = tl.load(Xn + (2 * row) * C + c, mask=mask, other=0.0); b = tl.load(Xn + (2 * row + 1) * C + c, mask=mask, other=0.0)
    if ADD_BIAS0:
        b0 = tl.load(BIAS0 + c, mask=mask, other=0.0).to(tl.float32)
        a = rnd(a.to(tl.float32) + b0, DT); b = rnd(b.to(tl.float32) + b0, DT)
        v = tl.maximum(a, b)
    else:
        v = tl.maximum(a, b).to(tl.float32)
    mean = tl.load(MEAN + c, mask=mask, other=0.0); invstd = tl.load(INVSTD + c, mask=mask, other=1.0)
    gamma = tl.load(GAMMA + c, mask=mask, other=1.0); beta = tl.load(BETA + c, mask=mask, other=0.0)
    t1 = gamma * (v - mean)
    v = rnd(tl.fma(t1, invstd, beta), DT)
    x3 = v * v * v; inner = KBETA * tl.fma(KKAPPA, x3, v); v = 0.5 * v * (1.0 + libdevice.tanh(inner))
    tl.store(Yn + offs, v.to(DT), mask=mask)


@triton.jit
def k_bn_gelu(X, Y, MEAN, INVSTD, GAMMA, BETA, n, Lx, C, KBETA, KKAPPA, BLOCK: tl.constexpr, DT: tl.constexpr):
    """decoder sites: BatchNorm(eval) -> GELU(tanh) in one pass, NCHW, same formula/roundings as k_pool_bn_gelu without the pool."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); mask = offs < n
    c = (offs // Lx) % C
    v = tl.load(X + offs, mask=mask, other=0.0).to(tl.float32)
    mean = tl.load(MEAN + c, mask=mask, other=0.0); invstd = tl.load(INVSTD + c, mask=mask, other=1.0)
    gamma = tl.load(GAMMA + c, mask=mask, other=1.0); beta = tl.load(BETA + c, mask=mask, other=0.0)
    t1 = gamma * (v - mean)
    v = rnd(tl.fma(t1, invstd, beta), DT)
    x3 = v * v * v; inner = KBETA * tl.fma(KKAPPA, x3, v); v = 0.5 * v * (1.0 + libdevice.tanh(inner))
    tl.store(Y + offs, v.to(DT), mask=mask)


def bn_gelu(x, bn, inv):
    SITE_CALLS["bn_gelu_nchw"] += 1
    x = x.contiguous(); y = torch.empty_like(x); n = y.numel()
    k_bn_gelu[(triton.cdiv(n, 1024),)](x, y, bn.running_mean, inv, bn.weight, bn.bias, n, x.shape[-1], x.shape[1], KBETA, KKAPPA, BLOCK=1024, DT=TL_DT, num_warps=4, enable_fp_fusion=False)
    return y


@triton.jit
def k_pool_bn_gelu(X, Y, MEAN, INVSTD, GAMMA, BETA, BIAS0, n_out, Lo, C, KBETA, KKAPPA, ADD_BIAS0: tl.constexpr, BLOCK: tl.constexpr, DT: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); mask = offs < n_out
    c = (offs // Lo) % C
    a = tl.load(X + 2 * offs, mask=mask, other=0.0); b = tl.load(X + 2 * offs + 1, mask=mask, other=0.0)
    if ADD_BIAS0:   # the conv's bias, added like ATen's separate add_ (fp32 opmath, one bf16 rounding)
        b0 = tl.load(BIAS0 + c, mask=mask, other=0.0).to(tl.float32)
        a = rnd(a.to(tl.float32) + b0, DT); b = rnd(b.to(tl.float32) + b0, DT)
        v = tl.maximum(a, b)
    else:
        v = tl.maximum(a, b).to(tl.float32)
    mean = tl.load(MEAN + c, mask=mask, other=0.0); invstd = tl.load(INVSTD + c, mask=mask, other=1.0)
    gamma = tl.load(GAMMA + c, mask=mask, other=1.0); beta = tl.load(BETA + c, mask=mask, other=0.0)
    t1 = gamma * (v - mean)
    v = rnd(tl.fma(t1, invstd, beta), DT)
    x3 = v * v * v; inner = KBETA * tl.fma(KKAPPA, x3, v); v = 0.5 * v * (1.0 + libdevice.tanh(inner))
    tl.store(Y + offs, v.to(DT), mask=mask)


@triton.jit
def k_maxpool2(X, Y, n_out, BLOCK: tl.constexpr):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); mask = offs < n_out
    a = tl.load(X + 2 * offs, mask=mask, other=0.0); b = tl.load(X + 2 * offs + 1, mask=mask, other=0.0)
    tl.store(Y + offs, tl.maximum(a, b), mask=mask)


def maxpool2_triton(x):
    N, C, Lx = x.shape; y = torch.empty(N, C, Lx // 2, dtype=x.dtype, device=x.device); n = y.numel()
    k_maxpool2[(triton.cdiv(n, 2048),)](x, y, n, BLOCK=2048, num_warps=8); return y


SITE_CALLS = collections.Counter()
def fused_site(x, bn, inv, bias0=None):
    SITE_CALLS["fused_site_nchw"] += 1
    N, C, Lx = x.shape; Lo = Lx // 2; y = torch.empty(N, C, Lo, dtype=x.dtype, device=x.device); n = y.numel()
    k_pool_bn_gelu[(triton.cdiv(n, 1024),)](x, y, bn.running_mean, inv, bn.weight, bn.bias, bias0 if bias0 is not None else bn.bias, n, Lo, C, KBETA, KKAPPA,
                                           ADD_BIAS0=bias0 is not None, BLOCK=1024, DT=TL_DT, num_warps=4, enable_fp_fusion=False)
    return y


def conv_nobias(conv, x):
    """the stock Conv1d's cuDNN conv WITHOUT its bias (the bias is folded into the following fused site kernel)"""
    return F.conv1d(x, conv.weight, None, conv.stride, conv.padding, conv.dilation, conv.groups)


def onehot_to_idx(x):
    """(N,4,L) one-hot with all-zero columns allowed -> int8 base index, 4 for zero columns (N / padding)"""
    return (x.argmax(1) + 4 * (x.amax(1) == 0)).to(torch.int8).contiguous()


class Stage1:
    def __init__(self, m, variant="fma", BLOCK_Q=64, BLOCK_C=64, num_warps=4, order=0):
        conv = m.conv_dna.conv_layer; bn = m.res_tower[0].norm
        w = conv.weight.detach().to(ACT_DTYPE).float()
        wm = torch.zeros(64, 512, device=dev, dtype=torch.float32)
        for k in range(15):
            for b in range(4): wm[k * 4 + b] = w[:, b, k]
        self.wm = wm.contiguous(); self.bias = conv.bias.detach().to(ACT_DTYPE).contiguous()
        self.w2 = wm.to(ACT_DTYPE).contiguous()      # (64, 512) fp16, row s*4+c = cuDNN's nhwc-krsc reduction order (c fastest), rows 60..63 zero
        self.bn = bn; self.inv = bn_invstd(bn); self.variant = variant; self.cfg = (BLOCK_Q, BLOCK_C, num_warps); self.order = order
    def __call__(self, idx, nhwc=False):
        N, Lx = idx.shape; Lo = Lx // 2; BQ, BC, nw = self.cfg
        if nhwc: y = torch.empty(N, 512, 1, Lo, dtype=ACT_DTYPE, device=dev, memory_format=torch.channels_last)
        else: y = torch.empty(N, 512, Lo, dtype=ACT_DTYPE, device=dev)
        kern = {"fma": k_stage1_fma, "gather": k_stage1_gather, "mma": k_stage1_mma}[self.variant]
        wm = self.w2 if self.variant == "mma" else self.wm
        kern[(triton.cdiv(Lo, BQ), 512 // BC, N)](idx, wm, self.bias, self.bn.running_mean, self.inv, self.bn.weight, self.bn.bias, y,
                                                   Lx, Lo, KBETA, KKAPPA, OUT_NHWC=nhwc, BLOCK_Q=BQ, BLOCK_C=BC, DT=TL_DT, ORDER=self.order, num_warps=nw, enable_fp_fusion=False)
        return y


class FastNCHW:
    """exact stack (NCHW): stage-1 kernel + fused pool/BN/GELU sites + Triton pool + early crop + head as cuBLAS baddbmm.
    apply precast_bf16(m) before constructing for the pre-cast lever; wrap with Graphed for the CUDA-graph lever."""
    def __init__(self, m, stage1_variant="fma", stage1_cfg=(64, 64, 4), crop=True, bias_fold=False, head="baddbmm", stage1_order=0, tower="nchw", decoder_fused=False, transformer_fused=False, relu_epilogue=False, skip_fused=False, decoder_fused2=False):
        self.m = m; self.st = Stage1(m, stage1_variant, *stage1_cfg, order=stage1_order); self.crop = crop; self.bias_fold = bias_fold; self.head = head
        self.decoder_fused = decoder_fused   # decoder BN->GELU sites as one Triton pass (k_bn_gelu) instead of cuDNN BN + ATen GELU
        self.transformer_fused = transformer_fused   # residual adds fused into the following LayerNorm (needs FusedLayerNorm-compatible LNs: nn.LayerNorm or FusedLayerNorm)
        self.relu_epilogue = relu_epilogue; self.skip_fused = skip_fused; self.decoder_fused2 = decoder_fused2
        self._tr = (lambda t: transformer_fused_forward(m.transformer, t, relu_epilogue)) if transformer_fused else m.transformer
        self.counters = collections.Counter()   # lever counters (no silent fallback: every call stamps the path it took)
        self.tower = tower   # "nhwc": the k=5 tower convs run on channels_last tensors (cuDNN's own NHWC xmma kernel without its nchw<->nhwc transposes; bitwise per layer on cuDNN 9.1 fp16), sites = NHWC Triton kernels, decoder stays NCHW
        if tower == "nhwc":
            cl = torch.channels_last; self.w4 = {name: mod.weight.detach().unsqueeze(2).contiguous(memory_format=cl) for name, mod in m.named_modules() if isinstance(mod, nn.Conv1d)}
        self.inv = {name: bn_invstd(mod) for name, mod in m.named_modules() if isinstance(mod, nn.BatchNorm1d)}
        self.b16 = {name: mod.bias.detach().to(ACT_DTYPE).contiguous() for name, mod in m.named_modules() if isinstance(mod, nn.Conv1d) and mod.bias is not None}
    def site(self, x, bname, cb, prev_conv_name=None):
        if self.bias_fold:   # x = previous conv WITHOUT bias; add it here, then the next conv also without bias
            y = fused_site(x, cb.norm, self.inv[bname + ".norm"], self.b16[prev_conv_name])
            return conv_nobias(cb.conv_layer, y)
        return cb.conv_layer(fused_site(x, cb.norm, self.inv[bname + ".norm"]))
    def _cb(self, name, cb, x):           # ConvBlock = norm -> GELU -> conv (decoder: no pool); fused or stock
        if self.decoder_fused: return cb.conv_layer(bn_gelu(x, cb.norm, self.inv[name + ".norm"]))
        return cb(x)
    def _up1(self, x):  return F.interpolate(self._cb("upsampling_unet1.0", self.m.upsampling_unet1[0], x), scale_factor=2, mode="nearest") if self.decoder_fused else self.m.upsampling_unet1(x)
    def _up0(self, x):  return F.interpolate(self._cb("upsampling_unet0.0", self.m.upsampling_unet0[0], x), scale_factor=2, mode="nearest") if self.decoder_fused else self.m.upsampling_unet0(x)
    def _sep(self, name, sep, x): return sep(x)   # separable ConvBlock: norm/activation are Identity (depthwise k3 + pointwise k1 only) -> nothing to fuse
    def _fjc(self, x):
        if self.decoder_fused: return F.gelu(self._cb("final_joined_convs.0", self.m.final_joined_convs[0], x), approximate="tanh")
        return self.m.final_joined_convs(x)
    def _head(self, xf, hm):
        """The fp32 head on the features xf (N, 1920, T). The head lever (cuBLAS bmm + fused bias/softplus, or baddbmm + softplus) serves the
        head SHAPE it serves — the 7,611-row human head; any other head module (a track subset set by upstream's
        set_track_subset, the 2,608-row mouse head) runs upstream's own convolution + softplus on the same features (bitwise by construction)."""
        m = self.m
        if self.head == "conv" or int(hm.weight.shape[0]) != HEAD_ROWS:
            return m.final_softplus(hm(xf))                                        # the stock head (cuDNN fp32 conv; TF32 under torch defaults)
        w = hm.weight.squeeze(-1); b = hm.bias; N_, _, T_ = xf.shape
        if self.decoder_fused2: return head_gemm_softplus(xf, w, b)                # the fused fp32 GEMM + softplus(x + bias) head
        out = torch.baddbmm(b.view(1, -1, 1).expand(N_, w.shape[0], T_), w.unsqueeze(0).expand(N_, -1, -1), xf)
        return m.final_softplus(out)
    @torch.no_grad()
    def __call__(self, x, *, upto=None, head_module=None, aux_head=None, with_embeddings=False):
        """forward(x) of the attached model. Defaults = upstream's forward(x, is_human=True): the human head's (N, 7611, T) fp32 output.
        upto="crop": stop where upstream's get_embs_after_crop stops and return those features (N, 1536, T); head_module: the head to
        apply (m.human_head | m.mouse_head); aux_head: upstream's data_parallel_training form (+ 0 * aux_head(features).sum());
        with_embeddings: return (out, features after final_joined_convs) as forward(return_embeddings=True) does."""
        c = self.counters; c["forwards"] += 1; c["decoder_fused"] += int(bool(self.decoder_fused))
        c["ln_fused"] += sum(1 for mod in self.m.transformer.modules() if isinstance(mod, FusedLayerNorm)); c["transformer_fused"] += int(bool(self.transformer_fused)); c["skip_fused"] += int(bool(self.skip_fused)); c["decoder_fused2"] += int(bool(self.decoder_fused2)); c["relu_epilogue"] += int(bool(self.relu_epilogue)); c[f"stage1_{self.st.variant}"] += 1; c[f"tower_{self.tower}"] += 1
        c[f"crop_{self.crop}"] += 1; c[f"head_{self.head}"] += 1; c["bias_fold"] += int(bool(self.bias_fold)); c["batch_units"] += int(x.shape[0])
        m = self.m
        with torch.autocast("cuda", dtype=ACT_DTYPE):
            rt = m.res_tower
            if self.tower == "nhwc":
                assert self.bias_fold, "the NHWC tower uses the bias-folded NHWC sites"
                def conv4(name, mod, t):
                    return F.conv2d(t, self.w4[name], None, padding=(0, mod.kernel_size[0] // 2), groups=mod.groups)
                t = self.st(onehot_to_idx(x), nhwc=True)                                        # (N,512,1,L/2) channels_last
                t = conv4("res_tower.0.conv_layer", rt[0].conv_layer, t)
                for bname, prev in (("res_tower.2", "res_tower.0.conv_layer"), ("res_tower.4", "res_tower.2.conv_layer"), ("res_tower.6", "res_tower.4.conv_layer")):
                    cb = rt[int(bname.split(".")[1])]
                    t = conv4(bname + ".conv_layer", cb.conv_layer, fused_site_nhwc(t, cb.norm, self.inv[bname + ".norm"], self.b16[prev]))
                cb = rt[8]; t = conv4("res_tower.8.conv_layer", cb.conv_layer, fused_site_nhwc(t, cb.norm, self.inv["res_tower.8.norm"], self.b16["res_tower.6.conv_layer"]))
                if self.skip_fused:   # v7: bias add + NHWC->NCHW transpose of the skip tensors in one Triton pass; the next site folds the same bias (bias0)
                    x_unet0 = bias_to_nchw(t, self.b16["res_tower.8.conv_layer"])
                    cb = m.unet1[1]; t = conv4("unet1.1.conv_layer", cb.conv_layer, fused_site_nhwc(t, cb.norm, self.inv["unet1.1.norm"], self.b16["res_tower.8.conv_layer"]))
                    x_unet1 = bias_to_nchw(t, self.b16["unet1.1.conv_layer"])
                    x = (F.max_pool2d(t, (1, 2)) + self.b16["unet1.1.conv_layer"].view(1, -1, 1, 1)).squeeze(2)   # pool then bias == bias then pool (monotone rounding)
                else:
                    t = t + self.b16["res_tower.8.conv_layer"].view(1, -1, 1, 1)                  # x_unet0 is consumed twice: add its bias once, as ATen does
                    x_unet0 = t.squeeze(2).contiguous()                                                # -> NCHW 3D (one transpose of the 16384-long skip tensor)
                    cb = m.unet1[1]; t = conv4("unet1.1.conv_layer", cb.conv_layer, fused_site_nhwc(t, cb.norm, self.inv["unet1.1.norm"])) + self.b16["unet1.1.conv_layer"].view(1, -1, 1, 1)
                    x_unet1 = t.squeeze(2).contiguous()                                                # -> NCHW 3D (8192-long)
                    x = F.max_pool2d(t, (1, 2)).squeeze(2)                                             # NHWC pool (bitwise); (N,C,4096) view of a channels_last tensor -> the transformer's permute(0,2,1) is free
            else:
                s1 = self.st(onehot_to_idx(x))
            if self.tower == "nhwc":
                pass
            elif self.bias_fold:
                x = conv_nobias(rt[0].conv_layer, s1)
                x = self.site(x, "res_tower.2", rt[2], "res_tower.0.conv_layer"); x = self.site(x, "res_tower.4", rt[4], "res_tower.2.conv_layer")
                x = self.site(x, "res_tower.6", rt[6], "res_tower.4.conv_layer"); x_unet0 = self.site(x, "res_tower.8", rt[8], "res_tower.6.conv_layer")
                x_unet0 = x_unet0 + self.b16["res_tower.8.conv_layer"].view(1, -1, 1)          # x_unet0 is consumed twice (skip + unet1): add its bias once, as ATen does
                x_unet1 = m.unet1[1].conv_layer(fused_site(x_unet0, m.unet1[1].norm, self.inv["unet1.1.norm"]))
            else:
                x = rt[0].conv_layer(s1)
                x = self.site(x, "res_tower.2", rt[2]); x = self.site(x, "res_tower.4", rt[4]); x = self.site(x, "res_tower.6", rt[6])
                x_unet0 = self.site(x, "res_tower.8", rt[8]); x_unet1 = self.site(x_unet0, "unet1.1", m.unet1[1])
            if self.tower != "nhwc": x = maxpool2_triton(x_unet1)
            if self.crop == "aligned":   # early crop with halo AND every intermediate LENGTH a multiple of 8 (cuDNN 9.1 picks other kernels for unaligned lengths)
                c8 = lambda n: -(-n // 8) * 8
                T = m.crop.target_length; L16 = x_unet0.shape[-1]; a16 = (L16 - T) // 2; b16 = a16 + T      # needed at 16384: [a16, b16)
                lo8, hi8 = (a16 - 1) // 2, b16 // 2 + 1                                                       # needed at 8192 (+1 halo for separable0)
                lo4, hi4 = (lo8 - 1) // 2, (lo8 - 1) // 2 + c8(-(-(hi8 + 1) // 2) - (lo8 - 1) // 2)           # at 4096: covers (n8 +/- 1)/2, length % 8 == 0
                lo8a, hi8a = 2 * lo4, 2 * hi4                                                                 # upsampled window at 8192 (len % 16 == 0)
                if self.decoder_fused2:   # v8: horizontal-conv BN+GELU sites fused; upsample+add in one pass (fp16 out); strided views instead of slice copies
                    x_unet1_c = self._cb("horizontal_conv1", m.horizontal_conv1, x_unet1[..., lo8a:hi8a].contiguous())
                    x = self._tr(x.permute(0, 2, 1)).permute(0, 2, 1)
                    x = self._cb("upsampling_unet1.0", m.upsampling_unet1[0], x[..., lo4:hi4].contiguous()); x = up2_add(x, x_unet1_c); x = self._sep("separable1", m.separable1, x)
                    hi8b = lo8 + c8(hi8 - lo8); assert lo8a + 1 <= lo8 and hi8b <= hi8a - 1, (lo8a, lo8, hi8b, hi8a)
                    x = x[..., lo8 - lo8a:hi8b - lo8a]
                    x_unet0_c = self._cb("horizontal_conv0", m.horizontal_conv0, x_unet0[..., 2 * lo8:2 * hi8b].contiguous())
                    x = self._cb("upsampling_unet0.0", m.upsampling_unet0[0], x.contiguous()); x = up2_add(x, x_unet0_c); x = self._sep("separable0", m.separable0, x)
                else:
                    x_unet1_c = m.horizontal_conv1(x_unet1[..., lo8a:hi8a])
                    x = self._tr(x.permute(0, 2, 1)).permute(0, 2, 1)
                    x = self._up1(x[..., lo4:hi4]); x = x + x_unet1_c; x = self._sep("separable1", m.separable1, x)            # valid on [lo8a+1, hi8a-1)
                    hi8b = lo8 + c8(hi8 - lo8)                                                                    # [lo8, hi8b): length % 8 == 0
                    assert lo8a + 1 <= lo8 and hi8b <= hi8a - 1, (lo8a, lo8, hi8b, hi8a)
                    x = x[..., lo8 - lo8a:hi8b - lo8a]
                    x_unet0_c = m.horizontal_conv0(x_unet0[..., 2 * lo8:2 * hi8b])
                    x = self._up0(x); x = x + x_unet0_c; x = self._sep("separable0", m.separable0, x)                          # valid on [2*lo8+1, 2*hi8b-1)
                assert 2 * lo8 + 1 <= a16 and b16 <= 2 * hi8b - 1
                x = x[..., a16 - 2 * lo8:b16 - 2 * lo8]
            elif self.crop:   # early crop with minimal halo: exact on cuDNN 9.10 (torch 2.8), NOT on cuDNN 9.1 (torch 2.5.1) at batch 1
                T = m.crop.target_length; L16 = x_unet0.shape[-1]; a16 = (L16 - T) // 2; b16 = a16 + T
                lo8 = (a16 - 1) // 2; hi8 = b16 // 2 + 1; lo8m, hi8m = lo8 - 1, hi8 + 1; lo4, hi4 = lo8m // 2, (hi8m - 1) // 2 + 1
                x_unet1_c = m.horizontal_conv1(x_unet1[..., lo8m:hi8m]); x_unet0_c = m.horizontal_conv0(x_unet0[..., 2 * lo8:2 * hi8])
                x = self._tr(x.permute(0, 2, 1)).permute(0, 2, 1)
                x = self._up1(x[..., lo4:hi4]); x = x + x_unet1_c; x = self._sep("separable1", m.separable1, x); x = x[..., 1:-1]
                x = self._up0(x); x = x + x_unet0_c; x = self._sep("separable0", m.separable0, x); x = x[..., 2:-2]
            else:
                x_unet1 = m.horizontal_conv1(x_unet1); x_unet0 = m.horizontal_conv0(x_unet0)
                x = self._tr(x.permute(0, 2, 1)).permute(0, 2, 1)
                x = self._up1(x); x = x + x_unet1; x = self._sep("separable1", m.separable1, x)
                x = self._up0(x); x = x + x_unet0; x = self._sep("separable0", m.separable0, x)
                x = m.crop(x.permute(0, 2, 1)).permute(0, 2, 1)
            if upto == "crop":
                return x                                                                  # upstream's get_embs_after_crop(x): the features before final_joined_convs
            x = self._fjc(x)
            with torch.autocast("cuda", enabled=False):
                xf = x.float()
                out = self._head(xf, m.human_head if head_module is None else head_module)
                if aux_head is not None:
                    out = out + 0 * aux_head(xf).sum()                                        # upstream's data_parallel_training form, verbatim
        if with_embeddings:
            return out, x                                                                 # forward(return_embeddings=True): (out, the final_joined_convs features)
        return out



@triton.jit
def k_bias_nhwc_to_nchw(X, B, Y, C, L, BL: tl.constexpr, BC: tl.constexpr, DT: tl.constexpr):
    """skip tensors: (N, C, 1, L) channels_last fp16 conv output + bias (fp32 opmath, one RNE rounding — ATen's Half add) -> (N, C, L) NCHW fp16.
    Tiled transpose: loads [BL, BC] with c fastest (coalesced NHWC), stores with l fastest (coalesced NCHW)."""
    pid_l = tl.program_id(0); pid_c = tl.program_id(1); n = tl.program_id(2)
    l = pid_l * BL + tl.arange(0, BL); c = pid_c * BC + tl.arange(0, BC)
    m = (l < L)[:, None] & (c < C)[None, :]
    x = tl.load(X + n * C * L + l[:, None] * C + c[None, :], mask=m, other=0.0).to(tl.float32)
    b = tl.load(B + c, mask=c < C, other=0.0).to(tl.float32)
    y = rnd(x + b[None, :], DT)
    tl.store(Y + n * C * L + c[None, :] * L + l[:, None], y, mask=m)


def bias_to_nchw(t, bias, BL=64, BC=64, num_warps=4):
    """(N,C,1,L) channels_last-contiguous -> (N,C,L) contiguous NCHW with the conv bias added (bitwise = (t + bias.view(1,-1,1,1)).squeeze(2).contiguous())"""
    assert t.is_contiguous(memory_format=torch.channels_last) and t.dim() == 4 and t.shape[2] == 1
    N, C, _, L = t.shape; y = torch.empty((N, C, L), dtype=t.dtype, device=t.device)
    k_bias_nhwc_to_nchw[(triton.cdiv(L, BL), triton.cdiv(C, BC), N)](t, bias, y, C, L, BL=BL, BC=BC, DT=TL_DT, num_warps=num_warps, enable_fp_fusion=False)
    SITE_CALLS["bias_to_nchw"] += 1
    return y


@triton.jit
def k_up2_add(Xp, S, Y, n, T2, C, sx_c, sx_n, ss_c, ss_n, BLOCK: tl.constexpr, DT: tl.constexpr):
    """decoder skip join: y[n,c,l] = fp16( fp32(x[n,c,l//2]) + fp32(s[n,c,l]) ) — ATen: upsample_nearest1d (fp32 under autocast) + add (fp32) + the
    separable conv's fp16 cast = the same two roundings (fp32 add, then RNE to fp16). x, s may be strided views (per-channel / per-batch strides)."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = offs < n
    row = offs // T2; l = offs % T2; nb = row // C; cidx = row % C
    xv = tl.load(Xp + nb * sx_n + cidx * sx_c + l // 2, mask=m, other=0.0).to(tl.float32)
    sv = tl.load(S + nb * ss_n + cidx * ss_c + l, mask=m, other=0.0).to(tl.float32)
    tl.store(Y + offs, rnd(xv + sv, DT), mask=m)


def up2_add(x, sk):
    """x: (N, C, T) (any strides along L contiguous); sk: (N, C, 2T) -> (N, C, 2T) fp16 contiguous == (F.interpolate(x, 2, 'nearest').float() + sk.float()).to(fp16)"""
    N, C, T = x.shape; assert sk.shape == (N, C, 2 * T) and x.stride(-1) == 1 and sk.stride(-1) == 1, (x.shape, sk.shape, x.stride(), sk.stride())
    y = torch.empty((N, C, 2 * T), dtype=ACT_DTYPE, device=x.device); n = y.numel()
    k_up2_add[(triton.cdiv(n, 1024),)](x, sk, y, n, 2 * T, C, x.stride(1), x.stride(0), sk.stride(1), sk.stride(0), BLOCK=1024, DT=TL_DT, num_warps=4, enable_fp_fusion=False)
    SITE_CALLS["up2_add"] += 1; return y


@triton.jit
def k_softplus_bias(X, B, Y, n, T, BLOCK: tl.constexpr):
    """head: y = softplus(x + b) in fp32 — ATen's softplus (beta 1, threshold 20): z > 20 ? z : log1p(exp(z)), libdevice expf/log1pf; z = fp32(x + b) is the
    cuBLAS beta=1 epilogue's single rounding of (acc + b) since the stored fp32 acc is exact."""
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK); m = offs < n
    t = (offs // T); b = tl.load(B + t % 7611, mask=m, other=0.0)
    z = tl.load(X + offs, mask=m, other=0.0) + b
    y = tl.where(z > 20.0, z, libdevice.log1p(libdevice.exp(z)))
    tl.store(Y + offs, y, mask=m)


def head_gemm_softplus(xf, w, b):
    """(N, 1920, T) fp32 -> softplus(w @ xf + b): bmm (no bias) + fused bias+softplus; == final_softplus(baddbmm(b, w, xf)) by bytes (tested)"""
    N_, _, T_ = xf.shape; out = torch.bmm(w.unsqueeze(0).expand(N_, -1, -1), xf); assert w.shape[0] == HEAD_ROWS; assert out.is_contiguous()
    y = torch.empty_like(out); n = y.numel()
    k_softplus_bias[(triton.cdiv(n, 1024),)](out, b, y, n, T_, BLOCK=1024, num_warps=4)
    SITE_CALLS["head_gemm_softplus"] += 1; return y

def fused_site_nhwc(x, bn, inv, bias0=None):   # x: (N,C,1,2Lo) channels_last-contiguous -> (N,C,1,Lo) channels_last
    SITE_CALLS["fused_site_nhwc"] += 1
    assert x.is_contiguous(memory_format=torch.channels_last), "NHWC site needs channels_last input"
    N, C, _, Lx = x.shape; Lo = Lx // 2
    y = torch.empty(N, C, 1, Lo, dtype=x.dtype, device=x.device, memory_format=torch.channels_last); n_win = Lo * C     # one grid row per window: the offsets stay per-window
    k_pool_bn_gelu_nhwc[(triton.cdiv(n_win, 1024), N)](x, y, bn.running_mean, inv, bn.weight, bn.bias, bias0 if bias0 is not None else bn.bias, n_win, C, KBETA, KKAPPA,
                                                ADD_BIAS0=bias0 is not None, BLOCK=1024, DT=TL_DT, num_warps=4, enable_fp_fusion=False)
    return y


class OutputLeasePool:
    """Pinned host buffers for the kit's clock-1 pipeline with EXPLICIT lease/release: a buffer is handed out only when no
    lease is outstanding on it (assertion), the async D2H lands in the leased buffer, the consumer releases it when it has finished
    reading (or copied out) — the consumer's copy/read cost is inside clock 1 because the loop joins every consumer before the clock stops.
    Each lease also carries a device staging copy so a CUDA-graph-owned output cannot be overwritten by the next replay."""
    def __init__(self, example, n=6):
        self.n = n; self.pins = [torch.empty_like(example, device="cpu").pin_memory() for _ in range(n)]
        self.devs = [torch.empty_like(example) for _ in range(n)]; self.evs = [torch.cuda.Event() for _ in range(n)]
        self.leased = [False] * n; self.stream = torch.cuda.Stream(); self.next = 0; self.n_leases = 0
    def lease(self, out, wait_for=None):
        """copy `out` (device) into a free slot: device staging on the current stream, then pinned async D2H on the copy stream.
        Returns a slot id; the caller passes it to `release` after reading `pinned(slot)`. Blocks on `wait_for` (a callable) only
        if the next slot is still leased — never reuses a leased slot."""
        k = self.next % self.n
        if self.leased[k]:
            if wait_for is not None: wait_for(k)
            assert not self.leased[k], f"slot {k} still leased at lease #{self.n_leases}: the consumer must release before the pool wraps"
        self.leased[k] = True; self.next += 1; self.n_leases += 1
        self.devs[k].copy_(out)
        self.stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.stream):
            self.pins[k].copy_(self.devs[k], non_blocking=True); self.evs[k].record(self.stream)
        return k
    def pinned(self, k):
        assert self.leased[k], f"slot {k} read without a lease"
        self.evs[k].synchronize(); return self.pins[k]
    def release(self, k):
        assert self.leased[k], f"double release of slot {k}"
        self.leased[k] = False
    def outstanding(self): return sum(self.leased)
    def status(self): return {"leases": self.n_leases, "leases_outstanding": self.outstanding()}


# ------------------------------------------------------------------------------------------------ fused LayerNorm (bitwise to ATen's vectorized_layer_norm_kernel<float,float>, torch 2.5.1)
@triton.jit
def _ln_online(val, mean, s2, cnt):
    # cuWelfordOnlineSum with nvcc's default contraction (both sites FMA; probed bitwise on real activations, lnfused_20260824b)
    delta = val - mean; nc = cnt + 1.0; coef = libdevice.div_rn(1.0, nc)
    nm = tl.fma(delta, coef, mean); ns = tl.fma(delta, val - nm, s2)
    return nm, ns, nc


@triton.jit
def _ln_combine(mB, sB, cB, mA, sA, cA):
    # cuWelfordCombine(dataB=own, dataA=other)
    delta = mB - mA; count = cA + cB; coef = libdevice.div_rn(1.0, count); nA = cA * coef; nB = cB * coef
    mean = tl.fma(nA, mA, nB * mB); s2 = tl.fma(delta * delta * cA, nB, sA + sB)
    return mean, s2, count


@triton.jit
def _ln_halve(m, s, c, W: tl.constexpr, H: tl.constexpr):
    # shfl_down tree step: lane i (own) combines with lane i+H (other); (W, 2H) -> (W, H)
    m3 = tl.permute(tl.reshape(m, (W, 2, H)), (0, 2, 1)); s3 = tl.permute(tl.reshape(s, (W, 2, H)), (0, 2, 1))
    mo, mx = tl.split(m3); so, sx = tl.split(s3)
    return _ln_combine(mo, so, c, mx, sx, c)


@triton.jit
def k_layernorm_fused(X, G, B, Y, eps, BLOCK: tl.constexpr):
    """one program per row of 1536 fp16 values: ATen's 128-thread / vec-4 Welford (thread t reads vectors t, t+128, t+256), warp shfl_down
    tree (16..1), inter-warp C(C(w0,w2), C(w1,w3)), sigma2/N, rstd = rsqrtf(var+eps), y = fma(gamma, rstd*(x-mean), beta) -> fp16 (RNE),
    i.e. the stock's fp16->fp32 cast + fp32 LayerNorm + fp32->fp16 cast in one pass.  Fixed N = 1536."""
    row = tl.program_id(0); t = tl.arange(0, 128)
    mean = tl.zeros((128,), tl.float32); s2 = tl.zeros((128,), tl.float32); cnt = 0.0
    for c in tl.static_range(3):
        for j in tl.static_range(4):
            v = tl.load(X + row * 1536 + (c * 128 + t) * 4 + j).to(tl.float32)
            mean, s2, cnt = _ln_online(v, mean, s2, cnt)
    m = tl.reshape(mean, (4, 32)); sg = tl.reshape(s2, (4, 32)); cc = cnt
    m, sg, cc = _ln_halve(m, sg, cc, 4, 16); m, sg, cc = _ln_halve(m, sg, cc, 4, 8); m, sg, cc = _ln_halve(m, sg, cc, 4, 4)
    m, sg, cc = _ln_halve(m, sg, cc, 4, 2); m, sg, cc = _ln_halve(m, sg, cc, 4, 1)
    mm = tl.reshape(m, (2, 2)); ss = tl.reshape(sg, (2, 2))                                       # [[w0,w1],[w2,w3]]
    ma, mb = tl.split(tl.permute(mm, (1, 0))); sa, sb = tl.split(tl.permute(ss, (1, 0)))           # [w0,w1] own, [w2,w3] other
    m2, s2b, c2 = _ln_combine(ma, sa, cc, mb, sb, cc)                                              # [C(w0,w2), C(w1,w3)]
    mA_, mB_ = tl.split(tl.reshape(m2, (1, 2))); sA_, sB_ = tl.split(tl.reshape(s2b, (1, 2)))
    mf, sf, cf = _ln_combine(mA_, sA_, c2, mB_, sB_, c2)
    mf = tl.reshape(mf, (1,)); sf = tl.reshape(sf, (1,))
    rstd = libdevice.rsqrt(libdevice.div_rn(sf, 1536.0) + eps)
    offs = tl.arange(0, BLOCK); mask = offs < 1536
    x = tl.load(X + row * 1536 + offs, mask=mask, other=0.0).to(tl.float32)
    g = tl.load(G + offs, mask=mask, other=0.0); b = tl.load(B + offs, mask=mask, other=0.0)
    y = tl.fma(g, rstd * (x - mf), b)
    tl.store(Y + row * 1536 + offs, y.to(tl.float16), mask=mask)


class FusedLayerNorm(nn.Module):
    """drop-in for the transformer's nn.LayerNorm(1536) under fp16 autocast: consumes the fp16 activation, returns the fp16 tensor the
    next Linear/MHA would have received (RNE of the stock's fp32 LN output). Exact only for fp16 activations + fp32 affine + N = 1536
    on the pinned stack (the kernel reproduces ATen's reduction tree for 128 threads / vec 4)."""
    def __init__(self, ln):
        super().__init__(); assert tuple(ln.normalized_shape) == (1536,), ln.normalized_shape
        self.weight = ln.weight; self.bias = ln.bias; self.eps = float(ln.eps); self.calls = 0
    def forward(self, x):
        assert x.dtype == torch.float16 and x.shape[-1] == 1536, (x.dtype, x.shape)      # no silent fallback: refuse other dtypes/shapes
        self.calls += 1; x2 = x.reshape(-1, 1536).contiguous(); y = torch.empty_like(x2)
        k_layernorm_fused[(x2.shape[0],)](x2, self.weight, self.bias, y, self.eps, BLOCK=2048, num_warps=4, enable_fp_fusion=False)
        return y.view(x.shape)


def swap_layernorms(m):
    """replace every nn.LayerNorm(1536) inside m.transformer by FusedLayerNorm; returns the count (lever counter 'ln_fused_modules')"""
    n = 0
    for name, mod in list(m.transformer.named_modules()):
        for cname, child in list(mod.named_children()):
            if isinstance(child, nn.LayerNorm):
                setattr(mod, cname, FusedLayerNorm(child)); n += 1
    SITE_CALLS["ln_fused_modules"] += n; return n


@triton.jit
def k_layernorm_fused_add(X, R, G, B, S, Y, eps, BLOCK: tl.constexpr):
    """residual add + LayerNorm: s = fp16(x + r) (ATen's Half add: fp32 opmath, one RNE rounding) is written to S (the new residual stream)
    and LayerNorm(s) -> fp16 to Y, with exactly the reduction of k_layernorm_fused."""
    row = tl.program_id(0); t = tl.arange(0, 128)
    mean = tl.zeros((128,), tl.float32); s2 = tl.zeros((128,), tl.float32); cnt = 0.0
    for c in tl.static_range(3):
        for j in tl.static_range(4):
            e = (c * 128 + t) * 4 + j
            v = rnd(tl.load(X + row * 1536 + e).to(tl.float32) + tl.load(R + row * 1536 + e).to(tl.float32), tl.float16)
            mean, s2, cnt = _ln_online(v, mean, s2, cnt)
    m = tl.reshape(mean, (4, 32)); sg = tl.reshape(s2, (4, 32)); cc = cnt
    m, sg, cc = _ln_halve(m, sg, cc, 4, 16); m, sg, cc = _ln_halve(m, sg, cc, 4, 8); m, sg, cc = _ln_halve(m, sg, cc, 4, 4)
    m, sg, cc = _ln_halve(m, sg, cc, 4, 2); m, sg, cc = _ln_halve(m, sg, cc, 4, 1)
    mm = tl.reshape(m, (2, 2)); ss = tl.reshape(sg, (2, 2))
    ma, mb = tl.split(tl.permute(mm, (1, 0))); sa, sb = tl.split(tl.permute(ss, (1, 0)))
    m2, s2b, c2 = _ln_combine(ma, sa, cc, mb, sb, cc)
    mA_, mB_ = tl.split(tl.reshape(m2, (1, 2))); sA_, sB_ = tl.split(tl.reshape(s2b, (1, 2)))
    mf, sf, cf = _ln_combine(mA_, sA_, c2, mB_, sB_, c2)
    mf = tl.reshape(mf, (1,)); sf = tl.reshape(sf, (1,))
    rstd = libdevice.rsqrt(libdevice.div_rn(sf, 1536.0) + eps)
    offs = tl.arange(0, BLOCK); mask = offs < 1536
    sm = rnd(tl.load(X + row * 1536 + offs, mask=mask, other=0.0).to(tl.float32) + tl.load(R + row * 1536 + offs, mask=mask, other=0.0).to(tl.float32), tl.float16)
    tl.store(S + row * 1536 + offs, sm.to(tl.float16), mask=mask)
    g = tl.load(G + offs, mask=mask, other=0.0); b = tl.load(B + offs, mask=mask, other=0.0)
    y = tl.fma(g, rstd * (sm - mf), b)
    tl.store(Y + row * 1536 + offs, y.to(tl.float16), mask=mask)


def ln_fused_add(x, r, ln):
    """(x + r) -> new residual stream (fp16) and LayerNorm of it (fp16); ln = FusedLayerNorm or nn.LayerNorm(1536)"""
    SITE_CALLS["ln_fused_add"] += 1
    assert x.dtype == torch.float16 and r.dtype == torch.float16 and x.shape == r.shape and x.shape[-1] == 1536
    x2 = x.reshape(-1, 1536).contiguous(); r2 = r.reshape(-1, 1536).contiguous(); s_ = torch.empty_like(x2); y = torch.empty_like(x2)
    k_layernorm_fused_add[(x2.shape[0],)](x2, r2, ln.weight, ln.bias, s_, y, float(ln.eps), BLOCK=2048, num_warps=4, enable_fp_fusion=False)
    return s_.view(x.shape), y.view(x.shape)


def transformer_fused_forward(tr, x, relu_epilogue=False):
    """the stock transformer (Sequential of [Residual(Seq(LN, attn, drop)), Residual(Seq(LN, fc1, relu, drop, fc2, drop))]) with every
    residual add fused into the following LayerNorm (k_layernorm_fused_add); the first LN and the final add are the plain ops.
    Dropouts are identity in eval. Exact by construction: the same roundings at the same places."""
    pending = None                      # residual branch output not yet added to x
    for layer in tr:
        res_attn, res_mlp = layer[0], layer[1]; fn_a, fn_m = res_attn.fn, res_mlp.fn
        ln_a, attn = fn_a[0], fn_a[1]; ln_m = fn_m[0]
        if pending is None: h = ln_a(x)
        else: x, h = ln_fused_add(x, pending, ln_a)
        a = attn(h)
        x, h = ln_fused_add(x, a, ln_m)
        mods = list(fn_m)[1:]          # stock order: Linear(1536->3072), {Dropout, ReLU} in the package's order, Linear(3072->1536), Dropout
        assert isinstance(mods[0], nn.Linear) and isinstance(mods[3], nn.Linear) and sum(isinstance(q, nn.ReLU) for q in mods[1:3]) == 1 \
            and all(isinstance(q, (nn.ReLU, nn.Dropout)) for q in mods[1:3] + mods[4:]) and not fn_m.training, [type(q).__name__ for q in mods]
        if relu_epilogue:   # fc1 bias + ReLU in the cuBLASLt epilogue (torch._addmm_activation): bitwise vs addmm + clamp_min on real activations (E9); Dropout = identity in eval
            fc1 = mods[0]; h2 = h.reshape(-1, 1536)
            mm = mods[3](torch._addmm_activation(fc1.bias, h2, fc1.weight.t(), use_gelu=False).view(*h.shape[:-1], -1)); SITE_CALLS["relu_epilogue"] += 1
        else:
            mm = h
            for q in mods[:4]: mm = q(mm)
        pending = mm
    return x + pending


class Graphed:
    def __init__(self, fn, example, warmup=3):
        self.fn = fn; self.x = example.clone()
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(warmup): fn(self.x)
        torch.cuda.current_stream().wait_stream(s)
        self.g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.g): self.y = fn(self.x)
        self.replays = 0
    def __call__(self, x):
        assert tuple(x.shape) == tuple(self.x.shape) and x.dtype == self.x.dtype and x.device == self.x.device, \
            f"no captured graph for shape {tuple(x.shape)}/{x.dtype} (captured {tuple(self.x.shape)}/{self.x.dtype}) — refusing a silent eager fallback"
        self.x.copy_(x); self.g.replay(); self.replays += 1; return self.y
    def status(self):
        d = {"graph_replays": self.replays, "graph_shape": list(self.x.shape)}
        if hasattr(self.fn, "counters"): d.update(self.fn.counters)
        return d


# ------------------------------------------------------------------------------------------------ panel reader (data-card rule)
