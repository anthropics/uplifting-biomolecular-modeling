"""protenix_fpf_ditfast.atom_kernels — 2-D row-tile Triton kernels of the fused atom-transformer block (lever atom_fused) (c_atom = 128 channels, M = N_sample*N_atom rows).
Conditioning operands (AdaLN scale/shift, output gates) are the hoisted, sample-invariant [Ns = N_atom, C] tensors, read with row-modulo;
scales/gates arrive PRE-sigmoided (the hoist caches sigmoid(linear(...)) once per item). fp32 statistics; activations out in the GEMM dtype.
  adaln2         : an = sc_a*LN(A)+sh_a ; kvn = sc_kv*LN(an)+sh_kv                          (block entry: two outputs)
  resgate_adaln2 : A += g*x (fp32, in place); then adaln2 of the new A (KV=1) or single AdaLN (KV=0)
  resgate        : A += g*x
  gate2d         : out[r, h*Cd+c] = o[b,h,n,c] * sigmoid(graw[r, .])  (graw = the g columns of the q|g GEMM output, NOT pre-sigmoided)
  swiglu2d       : out = silu(x[:, :F]) * x[:, F:]
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sig(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _ln_rows(a, m2, C, eps):
    mean = tl.sum(a, axis=1) / C
    d = tl.where(m2, a - mean[:, None], 0.0)
    var = tl.sum(d * d, axis=1) / C
    return d * (1.0 / tl.sqrt(var + eps))[:, None]


@triton.jit
def _adaln2_kernel(A, SCA, SHA, SCK, SHK, AN, KVN, M, Ns, C, s_a, s_c, s_an, eps,
                   KV: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    rm = rows < M
    cm = cols < C
    m2 = rm[:, None] & cm[None, :]
    r64 = rows.to(tl.int64); rs64 = (rows % Ns).to(tl.int64)
    a = tl.load(A + r64[:, None] * s_a + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    an = _ln_rows(a, m2, C, eps)
    sca = tl.load(SCA + rs64[:, None] * s_c + cols[None, :], mask=m2, other=0.0)
    sha = tl.load(SHA + rs64[:, None] * s_c + cols[None, :], mask=m2, other=0.0)
    an = sca * an + sha
    an = tl.where(m2, an, 0.0)
    tl.store(AN + r64[:, None] * s_an + cols[None, :], an.to(AN.dtype.element_ty), mask=m2)
    if KV:
        kv = _ln_rows(an, m2, C, eps)
        sck = tl.load(SCK + rs64[:, None] * s_c + cols[None, :], mask=m2, other=0.0)
        shk = tl.load(SHK + rs64[:, None] * s_c + cols[None, :], mask=m2, other=0.0)
        kv = sck * kv + shk
        tl.store(KVN + r64[:, None] * s_an + cols[None, :], kv.to(KVN.dtype.element_ty), mask=m2)


@triton.jit
def _resgate_adaln2_kernel(G, X, A, SCA, SHA, SCK, SHK, AN, KVN, M, Ns, C, s_g, s_x, s_a, s_c, s_an, eps,
                           LN: tl.constexpr, KV: tl.constexpr, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    cols = tl.arange(0, BLOCK_C)
    rm = rows < M
    cm = cols < C
    m2 = rm[:, None] & cm[None, :]
    r64 = rows.to(tl.int64); rs64 = (rows % Ns).to(tl.int64)
    g = tl.load(G + rs64[:, None] * s_g + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    x = tl.load(X + r64[:, None] * s_x + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    a = tl.load(A + r64[:, None] * s_a + cols[None, :], mask=m2, other=0.0).to(tl.float32)
    a = g * x + a
    a = tl.where(m2, a, 0.0)
    tl.store(A + r64[:, None] * s_a + cols[None, :], a, mask=m2)
    if LN:
        an = _ln_rows(a, m2, C, eps)
        sca = tl.load(SCA + rs64[:, None] * s_c + cols[None, :], mask=m2, other=0.0)
        sha = tl.load(SHA + rs64[:, None] * s_c + cols[None, :], mask=m2, other=0.0)
        an = sca * an + sha
        an = tl.where(m2, an, 0.0)
        tl.store(AN + r64[:, None] * s_an + cols[None, :], an.to(AN.dtype.element_ty), mask=m2)
        if KV:
            kv = _ln_rows(an, m2, C, eps)
            sck = tl.load(SCK + rs64[:, None] * s_c + cols[None, :], mask=m2, other=0.0)
            shk = tl.load(SHK + rs64[:, None] * s_c + cols[None, :], mask=m2, other=0.0)
            kv = sck * kv + shk
            tl.store(KVN + r64[:, None] * s_an + cols[None, :], kv.to(KVN.dtype.element_ty), mask=m2)


@triton.jit
def _gate2d_kernel(O, G, OUT, M, N, Cd, HC, s_ob, s_oh, s_on, s_g, s_out, BLOCK_R: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    j = tl.arange(0, BLOCK_C)
    rm = rows < M
    m2 = rm[:, None] & (j < HC)[None, :]
    b = (rows // N).to(tl.int64); n = (rows % N).to(tl.int64)
    h = (j // Cd).to(tl.int64); c = j % Cd
    o = tl.load(O + b[:, None] * s_ob + h[None, :] * s_oh + n[:, None] * s_on + c[None, :], mask=m2, other=0.0).to(tl.float32)
    g = tl.load(G + rows.to(tl.int64)[:, None] * s_g + j[None, :], mask=m2, other=0.0).to(tl.float32)
    tl.store(OUT + rows.to(tl.int64)[:, None] * s_out + j[None, :], (o * _sig(g)).to(OUT.dtype.element_ty), mask=m2)


@triton.jit
def _swiglu2d_kernel(X, OUT, M, F, s_x, s_o, BLOCK_R: tl.constexpr, BLOCK_F: tl.constexpr):
    pid = tl.program_id(0)
    rows = pid * BLOCK_R + tl.arange(0, BLOCK_R)
    j = tl.arange(0, BLOCK_F)
    m2 = (rows < M)[:, None] & (j < F)[None, :]
    r64 = rows.to(tl.int64)
    x1 = tl.load(X + r64[:, None] * s_x + j[None, :], mask=m2, other=0.0).to(tl.float32)
    x2 = tl.load(X + r64[:, None] * s_x + F + j[None, :], mask=m2, other=0.0).to(tl.float32)
    tl.store(OUT + r64[:, None] * s_o + j[None, :], (x1 * _sig(x1) * x2).to(OUT.dtype.element_ty), mask=m2)


def _tile(C):
    bc = triton.next_power_of_2(C)
    br = max(1, 2048 // bc)
    return br, bc


def adaln2(A, sca, sha, sck, shk, out_dtype, eps=1e-5, kv=True):
    M, C = A.shape; Ns = sca.shape[0]
    assert A.stride(1) == 1 and sca.stride(1) == 1 and M % Ns == 0
    for t in (sha, sck, shk):
        assert t is None or (t.shape == sca.shape and t.stride() == sca.stride())
    an = torch.empty((M, C), dtype=out_dtype, device=A.device)
    kvn = torch.empty((M, C), dtype=out_dtype, device=A.device) if kv else an
    br, bc = _tile(C)
    _adaln2_kernel[(triton.cdiv(M, br),)](A, sca, sha, sck if kv else sca, shk if kv else sha, an, kvn, M, Ns, C, A.stride(0), sca.stride(0), an.stride(0), eps,
                                           KV=kv, BLOCK_R=br, BLOCK_C=bc, num_warps=4)
    return (an, kvn) if kv else an


def resgate_adaln2(g, x, A, sca=None, sha=None, sck=None, shk=None, out_dtype=None, eps=1e-5, ln=True, kv=True):
    """In place A += g[r%Ns] * x ; returns (an, kvn) | an | None depending on ln/kv."""
    M, C = x.shape; Ns = g.shape[0]
    assert A.shape == (M, C) and A.dtype == torch.float32 and A.stride(1) == 1 and x.stride(1) == 1 and g.stride(1) == 1 and M % Ns == 0
    br, bc = _tile(C)
    if ln:
        assert sca.stride(1) == 1 and sca.shape[0] == Ns
        an = torch.empty((M, C), dtype=out_dtype, device=A.device)
        kvn = torch.empty((M, C), dtype=out_dtype, device=A.device) if kv else an
        s_c = sca.stride(0)
    else:
        an = kvn = A; sca = sha = sck = shk = g; s_c = g.stride(0); kv = False
    _resgate_adaln2_kernel[(triton.cdiv(M, br),)](g, x, A, sca, sha, sck if kv else sca, shk if kv else sha, an, kvn, M, Ns, C,
                                                   g.stride(0), x.stride(0), A.stride(0), s_c, an.stride(0), eps,
                                                   LN=ln, KV=kv, BLOCK_R=br, BLOCK_C=bc, num_warps=4)
    if not ln:
        return None
    return (an, kvn) if kv else an


def gate2d(o, graw, N, out_dtype):
    """o [B, H, N, Cd] any strides (col stride 1); graw [B*N, H*Cd] view -> [B*N, H*Cd] = o(row-major) * sigmoid(graw)."""
    B, H, N_, Cd = o.shape; assert N_ == N and o.stride(3) == 1 and graw.stride(1) == 1
    M, HC = graw.shape; assert M == B * N and HC == H * Cd
    out = torch.empty((M, HC), dtype=out_dtype, device=o.device)
    br, bc = _tile(HC)
    _gate2d_kernel[(triton.cdiv(M, br),)](o, graw, out, M, N, Cd, HC, o.stride(0), o.stride(1), o.stride(2), graw.stride(0), out.stride(0),
                                           BLOCK_R=br, BLOCK_C=bc, num_warps=4)
    return out


def swiglu2d(x12, out_dtype):
    M, F2 = x12.shape; F_ = F2 // 2; assert x12.stride(1) == 1
    out = torch.empty((M, F_), dtype=out_dtype, device=x12.device)
    br, bf = _tile(F_)
    _swiglu2d_kernel[(triton.cdiv(M, br),)](x12, out, M, F_, x12.stride(0), out.stride(0), BLOCK_R=br, BLOCK_F=bf, num_warps=4)
    return out
