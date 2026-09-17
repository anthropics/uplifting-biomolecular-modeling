import os

try:
    import torch
    from torch.nn.attention.flex_attention import (
        BlockMask,
        _create_sparse_block_from_block_mask,
        create_block_mask,
        flex_attention,
    )

    from .flash_attention_utils import _unpad_input, pad_input

    if torch.cuda.is_available():
        flex_attention = torch.compile(flex_attention, dynamic=True)
except ImportError:
    flex_attention = None


def block_min_max_seq_ids(SLEN: torch.Tensor, block_size: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    device = SLEN.device
    total_tokens = torch.sum(SLEN)
    B = (total_tokens + block_size - 1) // block_size
    padding_tokens = B * block_size - total_tokens
    SLEN = torch.cat([SLEN, torch.Tensor([padding_tokens]).to(device)], dim=0)

    assert torch.sum(SLEN) == B * block_size

    # Cumulative ends (exclusive) for each sequence; cum[i] == end offset of seq i
    cum = torch.cumsum(SLEN.to(torch.long), dim=0)  # (N,)
    total_tokens = cum[-1].item()

    # Block start/end offsets [start, end) in token index space
    block_starts = torch.arange(0, B * block_size, block_size, device=device, dtype=torch.long)  # (B,)
    block_ends = torch.minimum(block_starts + block_size, torch.tensor(total_tokens, device=device))  # (B,)

    # MIN_SEQ_ID[i] = first sequence whose end > block_start
    # searchsorted with right=True returns first index where cum > value
    MIN_SEQ_ID = torch.searchsorted(cum, block_starts, right=True)

    # MAX_SEQ_ID[i] = sequence containing the last token in the block (block_end - 1)
    # For empty tail beyond total_tokens we already clipped block_ends.
    last_token_in_block = torch.clamp(block_ends - 1, min=0)  # valid only if block has at least 1 token
    MAX_SEQ_ID = torch.searchsorted(cum, last_token_in_block, right=True)

    return MIN_SEQ_ID, MAX_SEQ_ID


def get_overlapping_blocks(SLEN_Q: torch.Tensor, SLEN_K: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    MIN_Q, MAX_Q = block_min_max_seq_ids(SLEN_Q)
    MIN_K, MAX_K = block_min_max_seq_ids(SLEN_K)

    cond1 = MIN_Q.unsqueeze(1) <= MAX_K.unsqueeze(0)
    cond2 = MIN_K.unsqueeze(0) <= MAX_Q.unsqueeze(1)
    overlap = cond1 & cond2

    cond1 = (MIN_Q == MAX_Q).unsqueeze(1)
    cond2 = (MIN_K == MAX_K).unsqueeze(0)
    same_seq_in_qk = cond1 & cond2

    full_blocks = overlap & same_seq_in_qk
    partial_blocks = overlap & ~same_seq_in_qk

    return full_blocks, partial_blocks


def direct_block_mask(SLEN_Q: torch.Tensor, SLEN_K: torch.Tensor) -> BlockMask:
    full_blocks, partial_blocks = get_overlapping_blocks(SLEN_Q, SLEN_K)
    partial_blocks = partial_blocks[None, None]
    full_blocks = full_blocks[None, None]

    q_doc_id = torch.repeat_interleave(SLEN_Q)
    k_doc_id = torch.repeat_interleave(SLEN_K)

    def doc_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return q_doc_id[q_idx] == k_doc_id[kv_idx]

    total_q_len = q_doc_id.shape[0]
    total_k_len = k_doc_id.shape[0]

    return _create_sparse_block_from_block_mask(
        (partial_blocks, full_blocks),
        doc_mask,
        seq_lengths=(total_q_len, total_k_len),
        Q_BLOCK_SIZE=128,
        KV_BLOCK_SIZE=128,
    )


def doc_id_mask(SLEN_Q: torch.Tensor, SLEN_K: torch.Tensor) -> BlockMask:
    q_doc_id = torch.repeat_interleave(SLEN_Q)
    k_doc_id = torch.repeat_interleave(SLEN_K)

    def doc_mask(b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor) -> torch.Tensor:
        return q_doc_id[q_idx] == k_doc_id[kv_idx]

    total_q_len = q_doc_id.shape[0]
    total_k_len = k_doc_id.shape[0]

    return create_block_mask(doc_mask, 1, 1, total_q_len, total_k_len, BLOCK_SIZE=128, device=SLEN_Q.device)


block_mask_creator = direct_block_mask if os.getenv("FAST_BLOCK_MASK", "1") == "1" else doc_id_mask


def varlen_flex_attention_func(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    q_sequence_ids: torch.Tensor,
    k_sequence_ids: torch.Tensor,
) -> torch.Tensor:
    batch_size, q_len = query_states.shape[0], query_states.shape[1]
    (
        query_states,
        key_states,
        value_states,
        indices_q,
        (cu_seqlens_q, cu_seqlens_k),
        (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
    ) = _unpad_input(query_states, key_states, value_states, q_sequence_ids, k_sequence_ids)

    query_states = query_states.unsqueeze(0).transpose(1, 2).contiguous()
    key_states = key_states.unsqueeze(0).transpose(1, 2).contiguous()
    value_states = value_states.unsqueeze(0).transpose(1, 2).contiguous()

    seqlens_q = cu_seqlens_q[1:] - cu_seqlens_q[:-1]
    seqlens_k = cu_seqlens_k[1:] - cu_seqlens_k[:-1]
    block_mask = block_mask_creator(seqlens_q, seqlens_k)

    attn_output_unpad = flex_attention(
        query_states,
        key_states,
        value_states,
        block_mask=block_mask,
        enable_gqa=query_states.shape[1] != key_states.shape[1],
    )

    attn_output = pad_input(attn_output_unpad.transpose(1, 2).squeeze(0), indices_q, batch_size, q_len)

    return attn_output
