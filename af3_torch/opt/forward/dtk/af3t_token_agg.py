# af3t_token_agg.py -- the atom -> token aggregation of the AF3 atom-attention encoder as ONE kernel.
# Stock (xfold nn/atom_cross_attention.py AtomCrossAttEncoder.forward, per denoiser step):
#     token_atoms_act = atom_layout.convert(queries_to_token_atoms, q)          # gather [.., N*A rows] of q [.., Q*32, C] + `*= gather_mask`
#     token_act = mask_mean(token_atoms_mask[.., None], relu(token_atoms_act), dim=-2)
#                = sum_a m[n,a] * relu(g[n,a] * q[.., idx[n,a], :]) / clamp(sum_a m[n,a], 1e-10)
# = five passes over a [S, N, A, C] intermediate (gather, mask, relu, mask*value, sum). Here: for every (sample s, token n, channel block) one
# program reads the A gathered rows of q once (masked slots not read), applies relu and the weights w[n,a] = m[n,a]*g[n,a] and the hoisted
# 1/clamp(sum_a m[n,a], eps), and writes token_act [S, N, C] fp32 -- no intermediate. Accumulation fp32 over the A (= 24) slots in slot order
# (torch's sum uses its own reduction order: the tolerance class, |diff| ~ 1e-6 relative); sample s of a batched call is program-for-program the
# single-sample call (bitwise across batch sizes). Any float dtype of q (loaded, widened to fp32); idx int32/int64.
import torch
import triton
import triton.language as tl


@triton.jit
def _token_mean_relu_kernel(X, IDX, W, INV, OUT, N, C, s_xs, s_xm, s_os, s_on,
                            A: tl.constexpr, BLOCK_C: tl.constexpr):
    pid = tl.program_id(0)                     # sample * N + token
    cb = tl.program_id(1)                      # channel block
    s = pid // N
    n = pid % N
    cols = cb * BLOCK_C + tl.arange(0, BLOCK_C)
    cm = cols < C
    acc = tl.zeros([BLOCK_C], dtype=tl.float32)
    s64 = s.to(tl.int64)
    for a in range(A):                         # A is constexpr: unrolled; masked slots issue no memory traffic (load fully predicated off)
        w = tl.load(W + n * A + a)
        row = tl.load(IDX + n * A + a).to(tl.int64)
        x = tl.load(X + s64 * s_xs + row * s_xm + cols, mask=cm & (w != 0.0), other=0.0).to(tl.float32)
        acc += w * tl.maximum(x, 0.0)
    inv = tl.load(INV + n)
    tl.store(OUT + s64 * s_os + n * s_on + cols, acc * inv, mask=cm)


def operands(gather_idxs, gather_mask, token_atoms_mask, eps=1e-10):
    """The step-invariant operands (hoist per trajectory): w [N, A] fp32 = token_atoms_mask * gather_mask; inv [N] fp32 = 1/clamp(sum_a mask, eps);
    idx [N, A] (the flattened queries-layout row of every token atom slot, as atom_layout.convert indexes)."""
    m = token_atoms_mask.to(torch.float32)
    w = (m * gather_mask.to(torch.float32)).contiguous()
    inv = (1.0 / torch.clamp(m.sum(dim=-1), min=eps)).contiguous()
    idx = gather_idxs.contiguous()
    assert w.shape == idx.shape and w.dim() == 2, (w.shape, idx.shape)
    return {"agg_w": w, "agg_inv": inv, "agg_idx": idx}


def token_mean_relu(x, idx, w, inv, out=None, BLOCK_C=256, num_warps=2):
    """x [S, M, C] (unit stride on C; any float dtype), idx / w [N, A], inv [N] -> token_act [S, N, C] fp32 (see module header)."""
    S, M, C = x.shape
    N, A = w.shape
    assert x.stride(-1) == 1 and idx.shape == (N, A) and inv.shape == (N,), (x.shape, x.stride(), idx.shape, inv.shape)
    assert w.dtype == torch.float32 and inv.dtype == torch.float32 and w.is_contiguous() and idx.is_contiguous() and inv.is_contiguous()
    if out is None:
        out = torch.empty((S, N, C), device=x.device, dtype=torch.float32)
    assert out.shape == (S, N, C) and out.stride(-1) == 1
    grid = (S * N, triton.cdiv(C, BLOCK_C))
    _token_mean_relu_kernel[grid](x, idx, w, inv, out, N, C, x.stride(0), x.stride(1), out.stride(0), out.stride(1),
                                  A=A, BLOCK_C=BLOCK_C, num_warps=num_warps)
    return out


def reference(x, idx, gather_mask, token_atoms_mask, eps=1e-10):
    """The stock statements' arithmetic on the same operands (torch): gather, mask, relu, masked mean -> [S, N, C] fp32."""
    S, M, C = x.shape
    N, A = idx.shape
    g = x[:, idx.reshape(-1).long(), :].reshape(S, N, A, C) * gather_mask.reshape(1, N, A, 1).to(x.dtype)
    m = token_atoms_mask.reshape(1, N, A, 1).to(torch.float32)
    return torch.sum(m * torch.relu(g), dim=-2) / torch.clamp(torch.sum(m, dim=-2), min=eps)


def warmup(C, A=24, device="cuda"):
    """Compile (or load from Triton's cache) the kernel for this channel count / slots-per-token on a dummy problem (bf16 and fp32 inputs)."""
    N, S, M = 4, 2, 3 * 32
    idx = torch.randint(0, M, (N, A), device=device)
    w = torch.ones((N, A), device=device); inv = torch.full((N,), 1.0 / A, device=device)
    for dt in (torch.bfloat16, torch.float32):
        token_mean_relu(torch.zeros((S, M, C), device=device, dtype=dt), idx, w, inv)
    torch.cuda.synchronize()
