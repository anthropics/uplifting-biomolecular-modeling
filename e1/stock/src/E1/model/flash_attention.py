import os

import torch

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None

from .flash_attention_utils import _unpad_input, pad_input


def is_flash_attention_available() -> bool:
    return (
        flash_attn_func is not None and flash_attn_varlen_func is not None and (os.getenv("USE_FLASH_ATTN", "1") == "1")
    )


def flash_attention_func(
    query_states: torch.Tensor,  # (bs, seqlen, nh, hs)
    key_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    value_states: torch.Tensor,  # (bs, seqlen, nkv, hs)
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
    causal: bool = False,
) -> torch.Tensor:  # (bs, seqlen, nh, hs)
    # Contains at least one padding token in the sequence. Note: ignore attention mask if causal.
    if not is_flash_attention_available():
        raise ImportError("Flash Attention is not available. Please install flash-attn.")

    if not causal:
        batch_size, q_len = query_states.shape[0], query_states.shape[1]
        (
            query_states,
            key_states,
            value_states,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        ) = _unpad_input(query_states, key_states, value_states, q_sequence_ids, k_sequence_ids)

        attn_output_unpad = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_in_batch_q,
            max_seqlen_k=max_seqlen_in_batch_k,
            causal=False,
        )
        attn_output = pad_input(attn_output_unpad, indices_q, batch_size, q_len)

    else:
        attn_output = flash_attn_func(query_states, key_states, value_states, causal=True)

    return attn_output
