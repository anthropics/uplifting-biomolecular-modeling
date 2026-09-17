import math
from typing import Optional

import torch

import triton
import triton.language as tl

from xfold.fastnn import config as fastnn_config


# CREDITS: Initially inspired by the Triton tutorial


@triton.jit
def _attention_core(Q, K, V, mask, bias, sm_scale, Out, stride_qz, stride_qh, stride_qm,
                    stride_qk, stride_kz, stride_kh, stride_kn, stride_kk, stride_vz, stride_vh,
                    stride_vn, stride_vk, stride_oz, stride_oh, stride_om, stride_on, Z, H, N_CTX,
                    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
                    use_mask_1d: tl.constexpr, use_mask_2d: tl.constexpr, use_bias: tl.constexpr):
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1).to(tl.int64)   # int64 BEFORE `* stride_qh` / `* stride_oh` / `* (N_CTX * N_CTX)`: over q [N, 4, N, 32] (triangle attention) the head offset passes 2**31 - 1 past 4096 tokens (upstream issue XFOLD-003)
    # initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    off_q = off_hz * stride_qh + \
        offs_m[:, None] * stride_qm + offs_d[None, :] * stride_qk
    off_k = off_hz * stride_qh + \
        offs_n[:, None] * stride_kn + offs_d[None, :] * stride_kk
    off_v = off_hz * stride_qh + \
        offs_n[:, None] * stride_qm + offs_d[None, :] * stride_qk
    # Initialize pointers to Q, K, V
    q_ptrs = Q + off_q
    k_ptrs = K + off_k
    v_ptrs = V + off_v

    # Initialize pointers to bias, mask
    if use_bias:
        off_hz_bias = (off_hz % H)
        offs_base_bias = off_hz_bias * \
            (N_CTX * N_CTX) + offs_m[:, None] * N_CTX

    offs_base_mask = 0
    if use_mask_2d:
        off_hz_mask = (off_hz // H)
        offs_base_mask = off_hz_mask * N_CTX

    # initialize pointer to m and l
    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) - float("inf")
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    # load q: it will stay in SRAM throughout
    q_load_mask = offs_m[:, None] < N_CTX
    q = tl.load(q_ptrs, mask=q_load_mask, other=0.0)
    # loop over k, v and update accumulator
    for start_n in range(0, N_CTX, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        load_mask = (start_n + offs_n)[:, None] < N_CTX

        # -- compute qk ----
        k = tl.load(k_ptrs + start_n * stride_kn, mask=load_mask, other=0.0)
        qk = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        qk += tl.dot(q, tl.trans(k))
        qk *= sm_scale

        if use_bias:
            bias_load_mask = (offs_m[:, None] < N_CTX) & ((start_n + offs_n)[None, :] < N_CTX)
            bias_data = tl.load(bias + offs_base_bias + offs_n + start_n,
                                mask=bias_load_mask,
                                other=0.0)
            qk += bias_data

        if use_mask_1d or use_mask_2d:
            mask_data = tl.load(mask + offs_base_mask + offs_n + start_n,
                                mask=(start_n + offs_n) < N_CTX,
                                other=0.)
            qk = tl.where(mask_data[None, :] == 0., float("-1e9"), qk)

        qk = tl.where(offs_m[:, None] >= N_CTX, float("-1e9"), qk)
        qk = tl.where((start_n + offs_n)[None, :] >= N_CTX, float("-1e9"), qk)

        # -- compute m_ij, p, l_ij
        m_ij = tl.max(qk, 1)
        p = tl.exp(qk - m_ij[:, None])
        l_ij = tl.sum(p, 1)
        # -- update m_i and l_i
        m_i_new = tl.maximum(m_i, m_ij)
        alpha = tl.exp(m_i - m_i_new)
        beta = tl.exp(m_ij - m_i_new)
        l_i_new = alpha * l_i + beta * l_ij
        # -- update output accumulator --
        # scale p
        p_scale = beta / l_i_new
        p = p * p_scale[:, None]
        # scale acc
        acc_scale = l_i / l_i_new * alpha
        acc = acc * acc_scale[:, None]
        # update acc
        load_mask = (start_n + offs_n)[:, None] < N_CTX
        v = tl.load(v_ptrs + start_n * stride_vn, mask=load_mask, other=0.)
        p = p.to(Q.dtype.element_ty)
        acc += tl.dot(p, v)
        # update m_i and l_i
        l_i = l_i_new
        m_i = m_i_new
    # rematerialize offsets to save registers
    start_m = tl.program_id(0)
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    # write back l and m
    # l_ptrs = L + off_hz * N_CTX + offs_m
    # m_ptrs = M + off_hz * N_CTX + offs_m
    # tl.store(l_ptrs, l_i)
    # tl.store(m_ptrs, m_i)
    # initialize pointers to output
    offs_n = tl.arange(0, BLOCK_DMODEL)
    off_o = off_hz * stride_oh + \
        offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    out_ptrs = Out + off_o

    out_store_mask = offs_m[:, None] < N_CTX
    tl.store(out_ptrs, acc, mask=out_store_mask)


def dot_product_attention_triton(q: torch.Tensor,
                                 k: torch.Tensor,
                                 v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 bias: Optional[torch.Tensor] = None):
    assert (q.dtype in [torch.float16,
                        torch.bfloat16]), "triton flash attention only support float16/bfloat16 now"

    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    if mask is not None:
        mask = mask.contiguous()
    if bias is not None:
        bias = bias.contiguous()

    sm_scale = 1. / math.sqrt(q.size(-1))
    BLOCK = 64
    # shape constraints
    Lq, Lk, Lv = q.shape[-1], k.shape[-1], v.shape[-1]
    assert Lq == Lk and Lk == Lv
    assert Lk in {16, 32, 64, 128}
    o = torch.empty_like(q)
    grid = (triton.cdiv(q.shape[2], BLOCK), q.shape[0] * q.shape[1])
    num_warps = 4 if Lk <= 64 else 8
    num_stages = 2

    _attention_core[grid](
        q,
        k,
        v,
        mask,
        bias,
        sm_scale,
        o,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        q.stride(3),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        k.stride(3),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        v.stride(3),
        o.stride(0),
        o.stride(1),
        o.stride(2),
        o.stride(3),
        q.shape[0],
        q.shape[1],
        q.shape[2],
        BLOCK_M=BLOCK,
        BLOCK_N=BLOCK,
        BLOCK_DMODEL=Lk,
        use_mask_1d=(mask != None) and len(mask.shape) == 1,
        use_mask_2d=(mask != None) and len(mask.shape) == 2,
        use_bias=(bias != None),
        num_warps=num_warps,
        num_stages=num_stages,
    )

    return o

def dot_product_attention_torch(q: torch.Tensor,
                                k: torch.Tensor,
                                v: torch.Tensor,
                                mask: Optional[torch.Tensor] = None,
                                bias: Optional[torch.Tensor] = None):
    scaling = q.size(-1) ** -0.5
    q = q * scaling
    logits = torch.matmul(q, k.transpose(-1, -2))

    if bias is not None:
        logits += bias

    if mask is not None:
        if mask.dim() == 1:
            mask = mask[None, None, None, :].to(dtype=torch.bool)
        elif mask.dim() == 2:
            mask = mask[:, None, None, :].to(dtype=torch.bool)
        logits.masked_fill_(~mask, -1e9)

    weights = torch.softmax(logits, dim=-1)

    return torch.matmul(weights, v)


def dot_product_attention(q: torch.Tensor,
                          k: torch.Tensor,
                          v: torch.Tensor,
                          mask: Optional[torch.Tensor] = None,
                          bias: Optional[torch.Tensor] = None):
    # if q,k,v is 3-dimensional, add a batch dimension
    qkv_dims = q.dim()
    if qkv_dims == 3:
        q = q.unsqueeze(0)
        k = k.unsqueeze(0)
        v = v.unsqueeze(0)

    if fastnn_config.dot_product_attention_implementation == "torch":
        out = dot_product_attention_torch(q, k, v, mask, bias)
    elif fastnn_config.dot_product_attention_implementation == "triton":
        if k.shape[-1] not in {16, 32, 64, 128}:
            out = dot_product_attention_torch(q, k, v, mask, bias)
        else:
            out = dot_product_attention_triton(q, k, v, mask, bias)
    else:
        raise ValueError(
            f"Unknown dot_product_attention_implementation: {fastnn_config.dot_product_attention_implementation}"
        )

    if qkv_dims == 3:
        out = out.squeeze(0)

    return out
