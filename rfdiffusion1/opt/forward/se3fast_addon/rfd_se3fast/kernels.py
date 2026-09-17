# rfd_se3fast/kernels.py — Triton kernels for the dense SE(3) layer of RFdiffusion v1 (Tier-2: fp32 in, fp32 accumulate,
# re-associated reductions; NO TF32 — every tl.dot is called with allow_tf32=False / input_precision="ieee").
#
#  K_trunk  radial_trunk_kernel : the first two layers of all FOUR RadialProfile MLPs of one SE(3) layer, which share
#           the same per-edge input inv = [edge_feats(32|64), |r|]:  for g in 0..3:
#              h = ReLU(LN(inv @ W1g^T + b1g)); hid_g = ReLU(LN(h @ W2g^T + b2g))          -> hid (4, E, 32)
#           Stock issues 4 x (3 GEMM + 2 LayerNorm + 2 ReLU) kernels on (E, 32) rows; here ONE kernel, registers only.
#  K_conv   radial_conv_kernel : fused last radial layer (32 -> C_out*J, no bias) x per-edge contraction with
#           tmp = features @ basis (E, J, K):   out[e, co, k] = sum_j ( sum_h hid[e,h] * W3[co*J+j, h] ) * tmp[e, j, k]
#           == stock `radial_weights.view(E, C_out, J) @ tmp` with the (E, C_out*J) radial weight tensor never
#           materialised.  One program = BLOCK_E edges; loop over co; tl.dot (BLOCK_E x H) @ (H x JP) in fp32.
# Launch geometry (BLOCK_E per compute capability, num_warps) is geometry.py's table, read per device at launch when the caller
# passes none.
# Both kernels are shape-specialised (constexpr) and CUDA-graph capturable (no host syncs, static shapes).
import torch
import triton
import triton.language as tl

from . import geometry


def _dot_kwargs():
    # triton >= 3.0: input_precision="ieee"; triton 2.x: allow_tf32=False
    import inspect
    try:
        sig = inspect.signature(tl.dot)
        if "input_precision" in sig.parameters:
            return dict(mode=1)
    except (ValueError, TypeError):
        pass
    return dict(mode=0)


_DOT_MODE = _dot_kwargs()["mode"]


@triton.jit
def _dot_ieee(a, b, MODE: tl.constexpr):
    if MODE == 1:
        return tl.dot(a, b, input_precision="ieee")
    else:
        return tl.dot(a, b, allow_tf32=False)


@triton.jit
def _ln_relu(h, gamma, beta, N: tl.constexpr, EPS: tl.constexpr):
    # h: (BE, N) ; LayerNorm over axis 1 (biased variance, eps inside sqrt = torch.nn.LayerNorm) then ReLU
    mean = tl.sum(h, axis=1) / N
    hc = h - mean[:, None]
    var = tl.sum(hc * hc, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + EPS)
    y = hc * rstd[:, None] * gamma[None, :] + beta[None, :]
    return tl.maximum(y, 0.0)


@triton.jit
def radial_trunk_kernel(inv_ptr, w1_ptr, b1_ptr, g1_ptr, be1_ptr, w2_ptr, b2_ptr, g2_ptr, be2_ptr, hid_ptr,
                        E, D_IN: tl.constexpr, DP: tl.constexpr, H: tl.constexpr, G: tl.constexpr,
                        EPS: tl.constexpr, BLOCK_E: tl.constexpr, MODE: tl.constexpr):
    """inv (E, D_IN) row-major fp32 ; w1 (G, H, D_IN) ; b1,g1,be1 (G, H) ; w2 (G, H, H) ; b2,g2,be2 (G, H) ; hid (G, E, H)."""
    pid = tl.program_id(0)
    e_off = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_off < E
    e_row = e_off.to(tl.int64)     # per-edge row offsets in 64-bit: E * (row width) passes 2**31 elements at large L (E = L*L); addresses only
    dcol = tl.arange(0, DP)
    hcol = tl.arange(0, H)
    x = tl.load(inv_ptr + e_row[:, None] * D_IN + dcol[None, :], mask=e_mask[:, None] & (dcol[None, :] < D_IN), other=0.0)   # (BE, DP)
    for g in range(G):
        # W1g^T as (DP, H): element [d, h] = w1[g, h, d]
        w1t = tl.load(w1_ptr + g * H * D_IN + hcol[None, :] * D_IN + dcol[:, None], mask=(dcol[:, None] < D_IN), other=0.0)
        h1 = _dot_ieee(x, w1t, MODE)                                                       # (BE, H)
        b1 = tl.load(b1_ptr + g * H + hcol); ga1 = tl.load(g1_ptr + g * H + hcol); bt1 = tl.load(be1_ptr + g * H + hcol)
        h1 = _ln_relu(h1 + b1[None, :], ga1, bt1, H, EPS)
        w2t = tl.load(w2_ptr + g * H * H + hcol[None, :] * H + hcol[:, None])              # (H, H): [i, o] = w2[g, o, i]
        h2 = _dot_ieee(h1, w2t, MODE)
        b2 = tl.load(b2_ptr + g * H + hcol); ga2 = tl.load(g2_ptr + g * H + hcol); bt2 = tl.load(be2_ptr + g * H + hcol)
        h2 = _ln_relu(h2 + b2[None, :], ga2, bt2, H, EPS)
        tl.store(hid_ptr + (e_row[:, None] + g * E) * H + hcol[None, :], h2, mask=e_mask[:, None])


@triton.jit
def radial_conv_kernel(hid_ptr, w3_ptr, tmp_ptr, out_ptr, E,
                       H: tl.constexpr, CO: tl.constexpr, J: tl.constexpr, JP: tl.constexpr, K: tl.constexpr, KP: tl.constexpr,
                       BLOCK_E: tl.constexpr, MODE: tl.constexpr):
    """hid (E, H); w3 (CO*J, H); tmp (E, J, K); out (E, CO, K)  — all fp32 contiguous."""
    pid = tl.program_id(0)
    e_off = pid * BLOCK_E + tl.arange(0, BLOCK_E)
    e_mask = e_off < E
    e_row = e_off.to(tl.int64)     # per-edge row offsets in 64-bit (as in radial_trunk_kernel): tmp and out pass 2**31 elements at large L
    hcol = tl.arange(0, H)
    jcol = tl.arange(0, JP)
    kcol = tl.arange(0, KP)
    hid = tl.load(hid_ptr + e_row[:, None] * H + hcol[None, :], mask=e_mask[:, None], other=0.0)            # (BE, H)
    t = tl.load(tmp_ptr + e_row[:, None, None] * (J * K) + jcol[None, :, None] * K + kcol[None, None, :],
                mask=e_mask[:, None, None] & (jcol[None, :, None] < J) & (kcol[None, None, :] < K), other=0.0)  # (BE, JP, KP)
    for co in range(CO):
        # W3 rows co*J .. co*J+J-1 transposed -> (H, JP)
        w = tl.load(w3_ptr + (co * J + jcol[None, :]) * H + hcol[:, None], mask=(jcol[None, :] < J), other=0.0)
        r = _dot_ieee(hid, w, MODE)                                                                           # (BE, JP)
        o = tl.sum(r[:, :, None] * t, axis=1)                                                                 # (BE, KP)
        tl.store(out_ptr + e_row[:, None] * (CO * K) + co * K + kcol[None, :], o, mask=e_mask[:, None] & (kcol[None, :] < K))


def _next_pow2(n):
    p = 1
    while p < n:
        p *= 2
    return p


_CC = {}   # device index -> geometry row key ("major.minor"), read once per device


def _cc(device):
    i = device.index if device.index is not None else torch.cuda.current_device()
    if i not in _CC:
        _CC[i] = geometry.capability_key(torch.cuda.get_device_capability(i))
    return _CC[i]


def radial_trunk(inv, W1, B1, G1, BE1, W2, B2, G2, BE2, eps=1e-5, BLOCK_E=None, num_warps=geometry.NUM_WARPS):
    """inv (E, D_IN); stacked params for G MLPs: W1 (G,H,D_IN) etc.  Returns hid (G, E, H).  BLOCK_E None: the card's (geometry.py)."""
    E, D_IN = inv.shape
    G, H, _ = W1.shape
    hid = torch.empty((G, E, H), device=inv.device, dtype=torch.float32)
    DP = max(16, _next_pow2(D_IN))
    if BLOCK_E is None:
        BLOCK_E = geometry.trunk_block_e(_cc(inv.device))
    grid = (triton.cdiv(E, BLOCK_E),)
    radial_trunk_kernel[grid](inv, W1, B1, G1, BE1, W2, B2, G2, BE2, hid, E, D_IN=D_IN, DP=DP, H=H, G=G, EPS=eps,
                              BLOCK_E=BLOCK_E, MODE=_DOT_MODE, num_warps=num_warps)
    return hid


def radial_conv(hid, w3, tmp, CO, J, K, BLOCK_E=None, num_warps=geometry.NUM_WARPS):
    """hid (E, H) contiguous; w3 (CO*J, H); tmp (E, J, K) contiguous -> out (E, CO, K).  BLOCK_E None: the card's for this tile (geometry.py)."""
    E, H = hid.shape
    assert w3.shape == (CO * J, H) and tuple(tmp.shape) == (E, J, K), (tuple(w3.shape), tuple(tmp.shape), CO, J, K)
    assert hid.is_contiguous() and tmp.is_contiguous() and w3.is_contiguous()
    out = torch.empty((E, CO, K), device=hid.device, dtype=torch.float32)
    JP = max(16, _next_pow2(J)); KP = _next_pow2(K)
    if BLOCK_E is None:
        BLOCK_E = geometry.conv_block_e(_cc(hid.device), JP * KP)
    grid = (triton.cdiv(E, BLOCK_E),)
    radial_conv_kernel[grid](hid, w3, tmp, out, E, H=H, CO=CO, J=J, JP=JP, K=K, KP=KP, BLOCK_E=BLOCK_E, MODE=_DOT_MODE,
                             num_warps=num_warps)
    return out
