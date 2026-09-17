from enum import Enum
from typing import TypedDict

import torch
import torch.nn as nn
from transformers.utils import logging

from ..config import E1Config
from ..dynamic_cache import DynamicCache
from .flash_attention import flash_attention_func, is_flash_attention_available
from .flex_attention import FlexAttentionArgs, flex_attention_func
from .varlen_flex_attention import varlen_flex_attention_func

logger = logging.get_logger(__name__)


class AttentionMethod(Enum):
    FLASH = "flash"
    FLEX = "flex"


class AttentionLayerType(Enum):
    WITHIN_SEQ = "within_seq"
    GLOBAL = "global"


class AttentionArgs(TypedDict, total=False):
    flex_attention_args: FlexAttentionArgs


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to (batch,
    num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self, dim: int, max_position_embeddings: int = 2048, base: int = 10000, device: torch.device | None = None
    ):
        super().__init__()

        self.dim = dim
        self.base = base
        self.max_position_embeddings = max_position_embeddings
        inv_freq = base ** -(torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_sin_cos_cache(seq_len=max_position_embeddings, device=self.inv_freq.device)

    @staticmethod
    def rotate_half(x: torch.Tensor) -> torch.Tensor:
        """Rotates half the hidden dims of the input."""
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def _set_sin_cos_cache(self, seq_len: int, device: torch.device) -> None:
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        self.max_seq_len_cached = seq_len
        t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
        angles = torch.outer(t, self.inv_freq.to(device))
        angles = torch.cat((angles, angles), dim=1)
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(
        self, q: torch.Tensor, k: torch.Tensor, position_ids: torch.LongTensor, seq_len: int | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [bsz, seq_len, num_attention_heads, head_size]
        device, dtype = q.device, q.dtype
        seq_len = position_ids.max().item() + 1 if seq_len is None else seq_len

        if seq_len > self.max_seq_len_cached:
            self._set_sin_cos_cache(seq_len=seq_len, device=device)

        # angles_cached[position_ids] gets us something of shape (batch_size, seq_len, head_dim),
        # so unsqueeze dimension -2 to broadcast to (batch_size, seq_len, n_heads, head_dim).
        idxs = position_ids.to(device)
        cos = self.cos_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]
        sin = self.sin_cached.to(device=device, dtype=dtype).unsqueeze(-2)[idxs]

        # Apply rotary positional embeddings to q and k (treating them as complex numbers). The first half is
        # Re[x exp(it)] = Re[x] cos(t) - Im[x] sin(t), while the second half is
        # Im[x exp(it)] = Im[x] cos(t) + Re[x] sin(t). This works b/c both halves of cos/sin are the same.
        q_embed = (q * cos) + (self.rotate_half(q) * sin)
        k_embed = (k * cos) + (self.rotate_half(k) * sin)
        return q_embed, k_embed


class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper."""

    def __init__(self, config: E1Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_kv_heads
        self.max_num_seqs = config.max_num_sequences
        self.clip_qkv = config.clip_qkv

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )
        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, self.hidden_size, bias=False)

        if self.config.global_attention_every_n_layers > 0:
            self.layer_type = (
                AttentionLayerType.GLOBAL
                if (self.layer_idx + 1) % self.config.global_attention_every_n_layers == 0
                else AttentionLayerType.WITHIN_SEQ
            )
        else:
            self.layer_type = AttentionLayerType.WITHIN_SEQ

        self.rope_theta = (
            config.rope_theta_within_seq
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else config.rope_theta_global
        )
        self.max_position_embeddings = (
            config.max_num_positions_within_seq
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else config.max_num_positions_global
        )

        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, max_position_embeddings=self.max_position_embeddings, base=self.rope_theta
        )

    def prepare_qkv(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.LongTensor,
        past_key_value: DynamicCache | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, q_len, _ = hidden_states.size()
        query_states: torch.Tensor = self.q_proj(hidden_states)
        key_states: torch.Tensor = self.k_proj(hidden_states)
        val_states: torch.Tensor = self.v_proj(hidden_states)

        query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        key_states = key_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)
        val_states = val_states.view(bsz, q_len, self.num_kv_heads, self.head_dim)

        if self.clip_qkv is not None:
            query_states = query_states.clamp(-self.clip_qkv, self.clip_qkv)
            key_states = key_states.clamp(-self.clip_qkv, self.clip_qkv)
            val_states = val_states.clamp(-self.clip_qkv, self.clip_qkv)

        query_states, key_states = self.rotary_emb(query_states, key_states, position_ids)

        if use_cache and past_key_value is not None:
            key_states, val_states = past_key_value.update(key_states, val_states, self.layer_idx)

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if torch.is_autocast_enabled():
            target_dtype = torch.get_autocast_gpu_dtype()
        else:
            target_dtype = self.q_proj.weight.dtype
        if input_dtype != target_dtype:
            logger.warning_once(
                f"The input hidden states seems to be silently casted in {input_dtype}. "
                f"This might be because you have upcasted embedding or layer norm layers "
                f"in {input_dtype}. We will cast back the input in {target_dtype}."
            )
            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            val_states = val_states.to(target_dtype)

        return query_states, key_states, val_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        within_seq_position_ids: torch.LongTensor,
        global_position_ids: torch.LongTensor,
        sequence_ids: torch.LongTensor,
        attention_args: AttentionArgs | None = None,
        past_key_value: DynamicCache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, DynamicCache | None]:
        is_cache_prefilled = (
            use_cache and past_key_value is not None and past_key_value.get_seq_length(self.layer_idx) > 0
        )

        query_states, key_states, val_states = self.prepare_qkv(
            hidden_states=hidden_states,
            position_ids=within_seq_position_ids
            if self.layer_type == AttentionLayerType.WITHIN_SEQ
            else global_position_ids,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )

        # Note: We fallback to using flash attention in inference mode when cache is filled with kv values
        # for global attention layers instead of flex attention. This is because once the cache is filled,
        # the last sequence attends to everything in the cache, so we can make things faster by using a
        # bidirectional flash attention instead of block-causal flex attention.
        if self.layer_type == AttentionLayerType.WITHIN_SEQ or is_cache_prefilled:
            attention_type = AttentionMethod.FLASH
        else:
            attention_type = AttentionMethod.FLEX

        attn_output, attn_weights = self._attn(
            attention_type=attention_type,
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            output_attentions=output_attentions,
        )

        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights, past_key_value

    def _attn(
        self,
        attention_type: AttentionMethod,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        match attention_type:
            case AttentionMethod.FLASH:
                f = self._flash_attn
            case AttentionMethod.FLEX:
                f = self._flex_attn
            case _:
                raise ValueError(f"No attention implementation found for {attention_type}")
        return f(
            query_states=query_states,
            key_states=key_states,
            val_states=val_states,
            sequence_ids=sequence_ids,
            attention_args=attention_args,
            output_attentions=output_attentions,
        )

    def _flash_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Flash attention implementation.

        Calls the public API of flash attention and deals with padding tokens if any are present.
        """
        assert not output_attentions, "Flash attention doesn't support returning attention masks"
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        _, kv_len = key_states.shape[0], key_states.shape[1]

        if self.layer_type == AttentionLayerType.GLOBAL:  # Only happens in inference
            q_sequence_ids = sequence_ids
            if q_len < kv_len:
                # Assumes query contain only one sequence
                # and all tokens in query (except padding) will attend to all tokens in KV
                first_token_id = sequence_ids[:, 0].unsqueeze(1)
                k_sequence_ids = torch.cat([first_token_id.expand(bsz, kv_len - q_len), sequence_ids], dim=-1)
            else:
                k_sequence_ids = sequence_ids
        else:
            if q_len < kv_len:  # Only happens in inference
                key_states = key_states[:, -q_len:]
                val_states = val_states[:, -q_len:]
            q_sequence_ids = k_sequence_ids = sequence_ids

        if is_flash_attention_available():
            attn_output = flash_attention_func(
                query_states,
                key_states,
                val_states,
                q_sequence_ids=q_sequence_ids,
                k_sequence_ids=k_sequence_ids,
                causal=False,
            )
        else:
            attn_output = varlen_flex_attention_func(
                query_states, key_states, val_states, q_sequence_ids=q_sequence_ids, k_sequence_ids=k_sequence_ids
            )

        attn_output = attn_output.reshape(bsz, q_len, self.hidden_size).contiguous()
        return attn_output, None

    def _flex_attn(
        self,
        query_states: torch.Tensor,
        key_states: torch.Tensor,
        val_states: torch.Tensor,
        sequence_ids: torch.Tensor,
        attention_args: AttentionArgs | None = None,
        output_attentions: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        bsz, q_len = query_states.shape[0], query_states.shape[1]
        flex_attention_args = attention_args.get("flex_attention_args", None) if attention_args is not None else None
        block_mask = flex_attention_args.get("block_mask", None) if flex_attention_args is not None else None
        score_mod = flex_attention_args.get("score_mod", None) if flex_attention_args is not None else None
        outputs = flex_attention_func(query_states, key_states, val_states, score_mod=score_mod, block_mask=block_mask)

        outputs = outputs.reshape(bsz, q_len, self.hidden_size).contiguous()
        return outputs, None
