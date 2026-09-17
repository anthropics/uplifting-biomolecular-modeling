"""K1 v3: the same serial order expressed as an im2col tl.dot with input_precision='ieee' (Triton lowers fp32 'ieee' dots to FFMA
chains, sequential over the K index per output element — verified bitwise against cuDNN IMPLICIT_GEMM / TF algo 0 on real activations).
A[m, j] = X[n, c(r0+j), l_m + k(r0+j)*dil], B[j, n] = WT[r0+j, co_n], r = c*K + k; K tiles of BK consecutive r's; acc carried across tiles.
Epilogue: + bias (one rounding) -> ReLU -> + residual (one rounding), as the stock.
"""
import os, torch, triton, triton.language as tl

_BM = int(os.environ.get("K1_BLOCK_M", "64")); _BN = int(os.environ.get("K1_BLOCK_N", "512")); _BK = int(os.environ.get("K1_BLOCK_K", "16"))
_NW = int(os.environ.get("K1_NUM_WARPS", "16")); _NS = int(os.environ.get("K1_NUM_STAGES", "3"))
COMPILED = {}


@triton.jit
def _im2col_dot_kernel(X, WT, B, Y, L, Lout, dil, R, C: tl.constexpr, K: tl.constexpr, Co: tl.constexpr, RELU: tl.constexpr, RESID: tl.constexpr,
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
        acc = tl.dot(a, w, acc, input_precision="ieee")
    b = tl.load(B + co, mask=cm, other=0.0)
    y = acc + b[None, :]
    if RELU:
        y = tl.maximum(y, 0.0)
    if RESID:
        xr = tl.load(xbase + co[None, :] * L + l[:, None] + dil * (K // 2), mask=lm[:, None] & cm[None, :], other=0.0)
        y = xr + y
    tl.store(Y + n * Co * Lout + co[None, :] * Lout + l[:, None], y, mask=lm[:, None] & cm[None, :])


def conv_dot(x, wt, b, K, dil=1, relu=True, resid=False, BLOCK_M=None, BLOCK_N=None, BLOCK_K=None):
    N, C, L = x.shape; Co = wt.shape[1]; R = C * K; Lout = L - (K - 1) * dil
    BM = BLOCK_M or _BM; BN = BLOCK_N or min(_BN, max(16, triton.next_power_of_2(Co))); BK = BLOCK_K or _BK
    y = torch.empty((N, Co, Lout), device=x.device, dtype=torch.float32)
    grid = (triton.cdiv(Lout, BM), triton.cdiv(Co, BN), N)
    h = _im2col_dot_kernel[grid](x, wt, b, y, L, Lout, dil, R, C, K, Co, relu, resid, BM, BN, BK, num_warps=_NW, num_stages=_NS)
    COMPILED[("dot", C, Co, K, dil, relu, resid, BM, BN, BK)] = h
    return y


@triton.jit
def _head_serial_kernel(X, W, B, Y, L, Lout, C: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr):
    # single-output-channel valid conv (profile head, K=75): one serial chain over r = c*K + k per position; W flat [C*K]
    n = tl.program_id(1); pid_m = tl.program_id(0)
    l = pid_m * BLOCK_M + tl.arange(0, BLOCK_M); lm = l < Lout
    xbase = X + n * C * L + l
    acc = tl.zeros((BLOCK_M,), dtype=tl.float32)
    for c in range(0, C):
        for k in tl.static_range(K):
            a = tl.load(xbase + c * L + k, mask=lm, other=0.0)
            w = tl.load(W + c * K + k)
            acc = tl.math.fma(a, w, acc)
    b = tl.load(B)
    y = acc + b
    tl.store(Y + n * Lout + l, y, mask=lm)


def head_serial(x, wflat, b, K, BLOCK_M=256, num_warps=8):
    """Single-output-channel valid conv (profile head, K=75): one serial fma chain over r = c*K + k per position; W flat [C*K]."""
    N, C, L = x.shape; Lout = L - (K - 1)
    y = torch.empty((N, Lout), device=x.device, dtype=torch.float32)
    h = _head_serial_kernel[(triton.cdiv(Lout, BLOCK_M), N)](x, wflat, b, y, L, Lout, C, K, BLOCK_M, num_warps=num_warps)
    COMPILED[("head_serial", C, K)] = h
    return y


def transpose_weight(w, pad_to=None):
    Co, C, K = w.shape
    wt = w.detach().float().permute(1, 2, 0).reshape(C * K, Co)
    if pad_to and Co < pad_to:
        wt = torch.cat([wt, torch.zeros(C * K, pad_to - Co, device=wt.device, dtype=wt.dtype)], dim=1)
    return wt.contiguous()


class K1BPNetV3(torch.nn.Module):
    def __init__(self, bpnet, head_pad=16, counts_mode="exact", head="serial", counts_form=None):
        super().__init__(); self.counts_mode = counts_mode; self.head = head; self.counts_form = counts_form   # k1r4.22: the per-arch dense order (None = the H100 order)
        self.iwt = transpose_weight(bpnet.iconv.weight); self.ib = bpnet.iconv.bias.detach().float().contiguous(); self.iK = int(bpnet.iconv.kernel_size[0])
        self.rwt = [transpose_weight(c.weight) for c in bpnet.rconvs]; self.rb = [c.bias.detach().float().contiguous() for c in bpnet.rconvs]
        self.dils = [int(c.dilation[0]) for c in bpnet.rconvs]
        self.fwt = transpose_weight(bpnet.fconv.weight, pad_to=head_pad); self.fb = torch.cat([bpnet.fconv.bias.detach().float(), torch.zeros(head_pad - 1, device=bpnet.fconv.bias.device)]).contiguous()
        self.fK = int(bpnet.fconv.kernel_size[0]); self.head_pad = head_pad
        self.fw_flat = self.fwt[:, 0].contiguous(); self.fb1 = self.fb[:1].contiguous()
        self.linear = bpnet.linear; self.out_len = 1000

    def forward(self, x):
        h = conv_dot(x, self.iwt, self.ib, self.iK, dil=1, relu=True, resid=False)
        for wt, b, d in zip(self.rwt, self.rb, self.dils):
            h = conv_dot(h, wt, b, 3, dil=d, relu=True, resid=True)
        if self.head == "serial":
            prof = head_serial(h, self.fw_flat, self.fb1, self.fK)
        else:
            prof = conv_dot(h, self.fwt, self.fb, self.fK, dil=1, relu=False, resid=False, BLOCK_N=self.head_pad)[:, 0, :]
        Lp = prof.shape[1]; s = (Lp - self.out_len) // 2
        prof = prof[:, s:s + self.out_len]
        cnt = self.linear(h.mean(dim=2)).reshape(-1, 1) if self.counts_mode == "torch" else exact_counts(h, self.linear, self.counts_form)
        return prof, cnt


# ---------------------------------------------------------------- exact counts head (orders found by probing the stock head)
@triton.jit
def _gap_serial_kernel(X, Y, L, C: tl.constexpr, BLOCK_C: tl.constexpr):
    # GlobalAveragePooling as the TF stock computes it under DET: one serial fp32 sum over positions (in order), then / L (div.rn)
    n = tl.program_id(1); c = tl.program_id(0) * BLOCK_C + tl.arange(0, BLOCK_C); cm = c < C
    base = X + n * C * L + c * L
    acc = tl.zeros((BLOCK_C,), dtype=tl.float32)
    for l in range(0, L):
        acc = acc + tl.load(base + l, mask=cm, other=0.0)
    y = tl.math.div_rn(acc, L.to(tl.float32))
    tl.store(Y + n * C + c, y, mask=cm)


@triton.jit
def _dense_chains_kernel(G, W, Part, C: tl.constexpr, P: tl.constexpr, BLOCK_P: tl.constexpr, CONTIG: tl.constexpr, HALVES: tl.constexpr):
    # Dense(C -> 1) partial chains: P lanes, lane p owns channels c = p + P*i (interleaved; or contiguous blocks), its steps split into HALVES
    # contiguous sub-chains (each a serial fma chain from 0); Part[n, h, p] = sub-chain h of lane p
    n = tl.program_id(0); p = tl.arange(0, BLOCK_P); pm = p < P
    S: tl.constexpr = C // P
    for h in tl.static_range(HALVES):
        acc = tl.zeros((BLOCK_P,), dtype=tl.float32)
        for i in range(h * (S // HALVES), (h + 1) * (S // HALVES)):
            if CONTIG:
                c = p * S + i
            else:
                c = i * P + p
            g = tl.load(G + n * C + c, mask=pm, other=0.0)
            w = tl.load(W + c, mask=pm, other=0.0)
            acc = tl.math.fma(g, w, acc)
        tl.store(Part + n * HALVES * P + h * P + p, acc, mask=pm)


def gap_serial(h, BLOCK_C=64):
    N, C, L = h.shape; y = torch.empty((N, C), device=h.device, dtype=torch.float32)
    hh = _gap_serial_kernel[(triton.cdiv(C, BLOCK_C), N)](h, y, L, C, BLOCK_C, num_warps=2); COMPILED[("gap", C)] = hh
    return y


def combine_explicit(a, how):
    a = list(a)
    if how == "seq":
        r = a[0]
        for x in a[1:]: r = r + x
        return r
    if how == "rev":
        r = a[-1]
        for x in a[-2::-1]: r = x + r
        return r
    if how == "tree":
        while len(a) > 1: a = [a[i] + a[i + 1] if i + 1 < len(a) else a[i] for i in range(0, len(a), 2)]
        return a[0]
    if how == "shfl":
        s = 1
        while s < len(a): s *= 2
        s //= 2
        while s >= 1:
            a = [a[i] + a[i + s] if i + s < len(a) else a[i] for i in range(s)]; s //= 2
        return a[0]
    raise ValueError(how)


def dense_exact(g, w, b, P=16, contiguous=False, combine="seq", halves=1):
    """Dense(C->1) in the stock's (TF 2.8 / cuBLAS 11) order: P interleaved lanes, each lane's steps split into `halves` serial fma
    sub-chains (summed per lane in order), lanes combined by `combine` (torch adds on (N,) vectors), bias added after."""
    N, C = g.shape; part = torch.empty((N, halves, P), device=g.device, dtype=torch.float32)
    hh = _dense_chains_kernel[(N,)](g, w, part, C, P, max(16, triton.next_power_of_2(P)), contiguous, halves, num_warps=1); COMPILED[("dense", C, P, contiguous, halves)] = hh
    lanes = []
    for p in range(P):
        r = part[:, 0, p]
        for h in range(1, halves): r = r + part[:, h, p]
        lanes.append(r)
    r = combine_explicit(lanes, combine)
    return (r + b).reshape(-1, 1)


def logsumexp_tf(a, b):
    """tf.math.reduce_logsumexp over two values: m = max(a, b); log(exp(a - m) + exp(b - m)) + m (each op one rounding; CUDA expf/logf)."""
    m = torch.maximum(a, b)
    return torch.log(torch.exp(a - m) + torch.exp(b - m)) + m


DENSE_STRUCT = {128: dict(P=16, contiguous=False, combine="seq", halves=1), 512: dict(P=16, contiguous=False, combine="seq", halves=2)}   # from probing the stock head, then a fit on sentinel inputs


def dense_groups_bfly(g, w, b, group=128):
    """Dense(C->1) in the sm_89 (L40S) stock order (TF 2.8 / cuBLAS 11 on the L40S; found by a CPU search over harvested stock outputs: 4096/4096 rows
    bitwise on both heads, C=512 and C=128): the products rounded per element, a stride-halving butterfly within each CONTIGUOUS group of `group` elements,
    the group partials accumulated sequentially, the bias added last. Pure elementwise torch ops (one rounding each; no reduction kernel, no atomics) -> the
    same bytes on any device. (The pool is the sequential sum / L on both classes: 131072/131072 cells — gap_serial already is.)"""
    N, C = g.shape; prod = g * w.reshape(1, -1); acc = None
    for s0 in range(0, C, group):
        p = prod[:, s0:s0 + group]; s = p.shape[1]
        while s > 1:
            s //= 2; p = p[:, :s] + p[:, s:2 * s]
        part = p[:, 0]; acc = part if acc is None else acc + part
    return (acc + b).reshape(-1, 1)

COUNTS_FORMS = {"h100_lanes": "the sm_90 stock order: 16 interleaved lanes of serial fma chains (halves), lanes summed in order, bias last (dense_exact; the H100 lock)",
                "groups_prod_bfly_seq": "the sm_89 stock order: per-element products, a stride-halving butterfly within contiguous groups of <group>, groups summed sequentially, bias last (dense_groups_bfly)"}

def exact_counts(h, linear, form=None):
    """form=None -> the H100 order (DENSE_STRUCT); form={'form': 'groups_prod_bfly_seq', 'group': 128} -> the per-arch order (k1r4.22)."""
    if form and form.get("form") not in (None, "h100_lanes", "groups_prod_bfly_seq"): raise ValueError(f"unknown counts form {form}")   # refused before any launch
    C = h.shape[1]; g = gap_serial(h)
    w = linear.weight.detach().float().reshape(-1).contiguous(); b = linear.bias.detach().float().reshape(())
    if form and form.get("form") == "groups_prod_bfly_seq": return dense_groups_bfly(g, w, b, group=int(form.get("group", 128)))
    return dense_exact(g, w, b, **DENSE_STRUCT[C])
