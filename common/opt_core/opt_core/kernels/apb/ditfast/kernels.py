"""protenix_fpf_ditfast.kernels — Triton row kernels of the fused token DiffusionTransformer block (lever dit_fused).

All kernels are dtype-generic (fp32 or bf16 activations, fp32 math inside) and take the conditioning operands with a row-modulo `Ns`
(the sample-deduplicated conditioning has N rows while the activation stream has 5N rows: row r reads conditioning row r % Ns).
  adaln    : out[r] = sigmoid(x1[r%Ns]) * LN(a[r]) + x2[r%Ns]      (LN over C without affine, eps, fp32 statistics; a fp32)  -> act dtype
  gate     : out[r, h*Cd+c] = o[b, h, n, c] * sigmoid(g[r, h*Cd+c])  (r = b*N + n; o = SDPA output in any stride layout)      -> act dtype
  swiglu   : out[r, f] = silu(x[r, f]) * x[r, F+f]                  (x = the merged a1|a2 GEMM output [M, 2F])                -> act dtype
  resgate  : out[r] = sigmoid(gl[r%Ns]) * x[r] + res[r]              (block output gate + residual; res fp32)                    -> fp32
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _sigmoid(x):
    return 1.0 / (1.0 + tl.exp(-x))


@triton.jit
def _adaln_kernel(A, X1, X2, OUT, Ns, C, s_a, s_x1, s_x2, s_o, eps, BLOCK_C: tl.constexpr):
    r = tl.program_id(0)
    rs = r % Ns
    cols = tl.arange(0, BLOCK_C)
    m = cols < C
    a = tl.load(A + r.to(tl.int64) * s_a + cols, mask=m, other=0.0).to(tl.float32)
    mean = tl.sum(a, axis=0) / C
    d = tl.where(m, a - mean, 0.0)
    var = tl.sum(d * d, axis=0) / C
    an = d * (1.0 / tl.sqrt(var + eps))
    x1 = tl.load(X1 + rs.to(tl.int64) * s_x1 + cols, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(X2 + rs.to(tl.int64) * s_x2 + cols, mask=m, other=0.0).to(tl.float32)
    out = _sigmoid(x1) * an + x2
    tl.store(OUT + r.to(tl.int64) * s_o + cols, out.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _gate_kernel(O, G, OUT, N, Cd, HC, s_ob, s_oh, s_on, s_g, s_out, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    b = r // N
    n = r % N
    j = tl.arange(0, BLOCK)
    m = j < HC
    h = j // Cd
    c = j % Cd
    o = tl.load(O + b.to(tl.int64) * s_ob + h.to(tl.int64) * s_oh + n.to(tl.int64) * s_on + c, mask=m, other=0.0).to(tl.float32)
    g = tl.load(G + r.to(tl.int64) * s_g + j, mask=m, other=0.0).to(tl.float32)
    tl.store(OUT + r.to(tl.int64) * s_out + j, (o * _sigmoid(g)).to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _swiglu_kernel(X, OUT, F, s_x, s_o, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    j = tl.arange(0, BLOCK)
    m = j < F
    x1 = tl.load(X + r.to(tl.int64) * s_x + j, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(X + r.to(tl.int64) * s_x + F + j, mask=m, other=0.0).to(tl.float32)
    y = x1 * _sigmoid(x1) * x2
    tl.store(OUT + r.to(tl.int64) * s_o + j, y.to(OUT.dtype.element_ty), mask=m)


@triton.jit
def _resgate_kernel(GL, X, RES, OUT, Ns, C, s_gl, s_x, s_res, s_o, BLOCK_C: tl.constexpr):
    r = tl.program_id(0)
    rs = r % Ns
    j = tl.arange(0, BLOCK_C)
    m = j < C
    gl = tl.load(GL + rs.to(tl.int64) * s_gl + j, mask=m, other=0.0).to(tl.float32)
    x = tl.load(X + r.to(tl.int64) * s_x + j, mask=m, other=0.0).to(tl.float32)
    res = tl.load(RES + r.to(tl.int64) * s_res + j, mask=m, other=0.0).to(tl.float32)
    out = _sigmoid(gl) * x + res
    tl.store(OUT + r.to(tl.int64) * s_o + j, out.to(OUT.dtype.element_ty), mask=m)


def _rows2d(t):
    assert t.dim() == 2 and t.stride(1) == 1, (t.shape, t.stride())
    return t


def adaln(a, x1, x2, out_dtype, eps=1e-5):
    """a [M, C] fp32 (row stride any, col stride 1); x1, x2 [Ns, C] views (col stride 1) -> [M, C] out_dtype contiguous."""
    _rows2d(a); _rows2d(x1); _rows2d(x2)
    M, C = a.shape; Ns = x1.shape[0]
    assert x2.shape[0] == Ns and x1.shape[1] == C and x2.shape[1] == C and M % Ns == 0
    out = torch.empty((M, C), dtype=out_dtype, device=a.device)
    _adaln_kernel[(M,)](a, x1, x2, out, Ns, C, a.stride(0), x1.stride(0), x2.stride(0), out.stride(0), eps,
                        BLOCK_C=triton.next_power_of_2(C), num_warps=4)
    return out


def gate(o, g, N, out_dtype):
    """o [B, H, N, Cd] (any strides, col stride 1); g [B*N, H*Cd] view (col stride 1) -> [B*N, H*Cd] = transpose(o)*sigmoid(g)."""
    B, H, N_, Cd = o.shape
    assert N_ == N and o.stride(3) == 1
    _rows2d(g); M, HC = g.shape
    assert M == B * N and HC == H * Cd
    out = torch.empty((M, HC), dtype=out_dtype, device=o.device)
    _gate_kernel[(M,)](o, g, out, N, Cd, HC, o.stride(0), o.stride(1), o.stride(2), g.stride(0), out.stride(0),
                       BLOCK=triton.next_power_of_2(HC), num_warps=4)
    return out


def swiglu(x12, out_dtype):
    """x12 [M, 2F] (col stride 1) -> [M, F] = silu(x12[:, :F]) * x12[:, F:]."""
    _rows2d(x12); M, F2 = x12.shape; F_ = F2 // 2
    out = torch.empty((M, F_), dtype=out_dtype, device=x12.device)
    _swiglu_kernel[(M,)](x12, out, F_, x12.stride(0), out.stride(0), BLOCK=triton.next_power_of_2(F_), num_warps=4)
    return out


def resgate(gl, x, res, out=None):
    """gl [Ns, C] view; x [M, C]; res [M, C] fp32 -> fp32 [M, C] = sigmoid(gl[r%Ns]) * x + res  (out may alias res: each row reads before it writes)."""
    _rows2d(gl); _rows2d(x); _rows2d(res)
    M, C = x.shape; Ns = gl.shape[0]
    assert res.shape == (M, C) and gl.shape[1] == C and M % Ns == 0
    if out is None:
        out = torch.empty((M, C), dtype=torch.float32, device=x.device)
    _resgate_kernel[(M,)](gl, x, res, out, Ns, C, gl.stride(0), x.stride(0), res.stride(0), out.stride(0),
                          BLOCK_C=triton.next_power_of_2(C), num_warps=4)
    return out


@triton.jit
def _resgate_adaln_kernel(GL, X, RES, X1, X2, AN, Ns, C, s_gl, s_x, s_res, s_x1, s_x2, s_an, eps, BLOCK_C: tl.constexpr):
    # A[r] = sigmoid(gl[r%Ns]) * x[r] + A[r]   (fp32, in place)   ;   an[r] = sigmoid(x1[r%Ns]) * LN(A[r]) + x2[r%Ns]   (act dtype)
    r = tl.program_id(0)
    rs = r % Ns
    j = tl.arange(0, BLOCK_C)
    m = j < C
    r64 = r.to(tl.int64); rs64 = rs.to(tl.int64)
    gl = tl.load(GL + rs64 * s_gl + j, mask=m, other=0.0).to(tl.float32)
    x = tl.load(X + r64 * s_x + j, mask=m, other=0.0).to(tl.float32)
    res = tl.load(RES + r64 * s_res + j, mask=m, other=0.0).to(tl.float32)
    a = _sigmoid(gl) * x + res
    a = tl.where(m, a, 0.0)
    tl.store(RES + r64 * s_res + j, a, mask=m)
    mean = tl.sum(a, axis=0) / C
    d = tl.where(m, a - mean, 0.0)
    var = tl.sum(d * d, axis=0) / C
    anorm = d * (1.0 / tl.sqrt(var + eps))
    x1 = tl.load(X1 + rs64 * s_x1 + j, mask=m, other=0.0).to(tl.float32)
    x2 = tl.load(X2 + rs64 * s_x2 + j, mask=m, other=0.0).to(tl.float32)
    out = _sigmoid(x1) * anorm + x2
    tl.store(AN + r64 * s_an + j, out.to(AN.dtype.element_ty), mask=m)


def resgate_adaln(gl, x, res, x1, x2, out_dtype, eps=1e-5):
    """In place: res[r] += sigmoid(gl[r%Ns]) * x[r]; returns an = AdaLN(res_new, x1, x2) in out_dtype. res must be fp32 contiguous rows."""
    _rows2d(gl); _rows2d(x); _rows2d(res); _rows2d(x1); _rows2d(x2)
    M, C = x.shape; Ns = gl.shape[0]
    assert res.shape == (M, C) and res.dtype == torch.float32 and x1.shape[0] == Ns and x2.shape[0] == Ns and M % Ns == 0
    an = torch.empty((M, C), dtype=out_dtype, device=x.device)
    _resgate_adaln_kernel[(M,)](gl, x, res, x1, x2, an, Ns, C, gl.stride(0), x.stride(0), res.stride(0), x1.stride(0), x2.stride(0), an.stride(0), eps,
                                BLOCK_C=triton.next_power_of_2(C), num_warps=4)
    return an
