import torch

import triton
import triton.language as tl

from xfold.fastnn import config as fastnn_config


def get_cuda_autotune_config():
    return [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64, 'GROUP_SIZE_M': 8}, num_stages=3,
                      num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=4,
                      num_warps=4),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 32, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
        triton.Config({'BLOCK_SIZE_M': 32, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 32, 'GROUP_SIZE_M': 8}, num_stages=5,
                      num_warps=2),
    ]


@triton.autotune(
    configs=get_cuda_autotune_config(),
    key=['M', 'N', 'K'],
)
@triton.jit
def _glu_kernel(
        x_ptr, w_ptr, v_ptr, out_ptr,
        M, N, K,
        stride_xm, stride_xk,  #
        stride_wk, stride_wn,  #
        stride_vk, stride_vn,  #
        stride_om, stride_on,
        # Meta-parameters
        BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,  #
        GROUP_SIZE_M: tl.constexpr,  #
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = ((pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M).to(tl.int64)   # int64 BEFORE `* stride_xm`: rows * 128 passes 2**31 - 1 past 4096 tokens (upstream issue XFOLD-003)
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm +
                      offs_k[None, :] * stride_xk)
    w_ptrs = w_ptr + (offs_k[:, None] * stride_wk +
                      offs_bn[None, :] * stride_wn)
    v_ptrs = v_ptr + (offs_k[:, None] * stride_vk +
                      offs_bn[None, :] * stride_vn)

    acc0 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        x = tl.load(x_ptrs, mask=offs_k[None, :]
                    < K - k * BLOCK_SIZE_K, other=0.0)
        w = tl.load(w_ptrs, mask=offs_k[:, None]
                    < K - k * BLOCK_SIZE_K, other=0.0)
        v = tl.load(v_ptrs, mask=offs_k[:, None]
                    < K - k * BLOCK_SIZE_K, other=0.0)

        acc0 = tl.dot(x, w, acc0)
        acc1 = tl.dot(x, v, acc1)
        # Advance the ptrs to the next K block.
        x_ptrs += BLOCK_SIZE_K * stride_xk
        w_ptrs += BLOCK_SIZE_K * stride_wk
        v_ptrs += BLOCK_SIZE_K * stride_vk

    out = (acc0 * tl.sigmoid(acc0) * acc1)

    offs_cm = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)).to(tl.int64)          # int64 BEFORE `stride_om *`: rows * 512 passes 2**31 - 1 past 2048 tokens (XFOLD-003)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + stride_om * \
        offs_cm[:, None] + stride_on * offs_cn[None, :]
    out_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(out_ptrs, out, mask=out_mask)


def gated_linear_unit_triton(x, weight):
    x = x.contiguous()
    x_shape = x.shape
    x = x.view(-1, x.shape[-1])
    M, K = x.shape
    weight = weight.to(dtype=x.dtype)
    w_gate, w_proj = weight.chunk(2, dim=1)
    K, N = w_gate.shape
    out = torch.empty((M, N), device=x.device, dtype=x.dtype)
    def grid(META): return (triton.cdiv(
        M, META['BLOCK_SIZE_M']) * triton.cdiv(N, META['BLOCK_SIZE_N']), )
    _glu_kernel[grid](
        x, w_gate, w_proj, out,
        M, N, K,
        x.stride(0), x.stride(1),
        w_gate.stride(0), w_gate.stride(1),
        w_proj.stride(0), w_proj.stride(1),
        out.stride(0), out.stride(1),
    )
    return out.view(*x_shape[:-1], N)


def gated_linear_unit_torch(x, weight):
    y = torch.matmul(x, weight)
    a, b = torch.chunk(y, 2, dim=-1)
    out = torch.nn.functional.silu(a) * b
    return out


def gated_linear_unit(x, weight):
    if fastnn_config.gated_linear_unit_implementation == "torch":
        out = gated_linear_unit_torch(x, weight)
    if fastnn_config.gated_linear_unit_implementation == "triton":
        out = gated_linear_unit_triton(x, weight)
    return out
