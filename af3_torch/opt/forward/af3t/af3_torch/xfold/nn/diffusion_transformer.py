# Copyright 2024 xfold authors
# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import einops

from xfold.nn import atom_layout
from xfold import fastnn
from xfold import of3


class AdaptiveLayerNorm(nn.Module):
    def __init__(self,
                 c_x: int,
                 c_single_cond: int,
                 use_single_cond: bool = False) -> None:

        super(AdaptiveLayerNorm, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        if self.use_single_cond is True:
            self.layer_norm = fastnn.LayerNorm(
                self.c_x, elementwise_affine=False, bias=False)
            self.single_cond_layer_norm = fastnn.LayerNorm(
                self.c_single_cond, bias=False)
            self.single_cond_scale = nn.Linear(
                self.c_single_cond, self.c_x, bias=True)
            self.single_cond_bias = nn.Linear(
                self.c_single_cond, self.c_x, bias=False)
        else:
            self.layer_norm = fastnn.LayerNorm(self.c_x)

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert (single_cond is None) == (self.use_single_cond is False)

        if self.use_single_cond is True:
            x = self.layer_norm(x)
            single_cond = self.single_cond_layer_norm(single_cond)
            single_scale = self.single_cond_scale(single_cond)
            single_bias = self.single_cond_bias(single_cond)
            return torch.sigmoid(single_scale) * x + single_bias
        else:
            return self.layer_norm(x)


class AdaLNZero(nn.Module):
    def __init__(self,
                 c_in: int,
                 c_out: int,
                 c_single_cond: int,
                 use_single_cond: bool = False) -> None:
        super(AdaLNZero, self).__init__()

        self.c_in = c_in
        self.c_out = c_out
        self.c_single_cond = c_single_cond
        self.use_single_cond = use_single_cond

        self.transition2 = nn.Linear(self.c_in, self.c_out, bias=False)
        if self.use_single_cond is True:
            self.adaptive_zero_cond = nn.Linear(
                self.c_single_cond, self.c_out, bias=True)

    def forward(self,
                x: torch.Tensor,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:

        assert (single_cond is None) == (self.use_single_cond is False)

        output = self.transition2(x)
        if self.use_single_cond is True:
            cond = self.adaptive_zero_cond(single_cond)
            output = torch.sigmoid(cond) * output
        return output


class DiffusionTransition(nn.Module):
    def __init__(self,
                 c_x: int,
                 c_single_cond: int,
                 num_intermediate_factor: int = 2,
                 use_single_cond: bool = False) -> None:
        super(DiffusionTransition, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_intermediate_factor = num_intermediate_factor
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)
        self.transition1 = nn.Linear(
            self.c_x, 2 * self.c_x * self.num_intermediate_factor, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.num_intermediate_factor * self.c_x,
            self.c_x,
            self.c_single_cond,
            self.use_single_cond
        )

    def forward(self, x: torch.Tensor, single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = self.adaptive_layernorm(x, single_cond)
        c = fastnn.gated_linear_unit(x, self.transition1.weight.T)
        return self.adaptive_zero_init(c, single_cond)


class SelfAttention(nn.Module):
    def __init__(self,
                 c_x: int = 768,
                 c_single_cond: int = 384,
                 num_head: int = 16,
                 use_single_cond: bool = False) -> None:

        super(SelfAttention, self).__init__()

        self.c_x = c_x
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.qkv_dim = self.c_x // self.num_head
        self.use_single_cond = use_single_cond

        self.adaptive_layernorm = AdaptiveLayerNorm(
            self.c_x, self.c_single_cond, self.use_single_cond)

        self.q_projection = nn.Linear(self.c_x, self.c_x, bias=True)
        self.k_projection = nn.Linear(self.c_x, self.c_x, bias=False)
        self.v_projection = nn.Linear(self.c_x, self.c_x, bias=False)

        self.gating_query = nn.Linear(self.c_x, self.c_x, bias=False)

        self.adaptive_zero_init = AdaLNZero(
            self.c_x, self.c_x, self.c_single_cond, self.use_single_cond)

    def forward(self,
                x: torch.Tensor,
                mask: torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None,
                single_cond: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (num_tokens, ch)
            mask (torch.Tensor): (num_tokens,)
            pair_logits (torch.Tensor, optional): (num_heads, num_tokens, num_tokens)
        """

        assert (single_cond is None) == (self.use_single_cond is False)

        x = self.adaptive_layernorm(x, single_cond)

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)

        if x.dim() == 3:                                     # lever 'sbatch': x (S, num_tokens, ch) — the sample axis rides where the call below puts its singleton
            q, k, v = map(lambda t: einops.rearrange(            # batch axis (the attention kernel's grid axis: per sample the same program over the same
                t, 's n (h c) -> s h n c', h=self.num_head), [q, k, v])   # [H, N, N] pair logits and [N] mask); every other statement here is row-wise

            weighted_avg = fastnn.dot_product_attention(
                q, k, v, mask=mask, bias=pair_logits
            )

            weighted_avg = einops.rearrange(weighted_avg, 's h q c -> s q (h c)')
        else:
            q, k, v = map(lambda t: einops.rearrange(
                t, 'n (h c) -> h n c', h=self.num_head).unsqueeze(0), [q, k, v])

            weighted_avg = fastnn.dot_product_attention(
                q, k, v, mask=mask, bias=pair_logits
            )

            weighted_avg = weighted_avg.squeeze(0)
            weighted_avg = einops.rearrange(weighted_avg, 'h q c -> q (h c)')

        gate_logits = self.gating_query(x)
        weighted_avg *= torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond)

    def forward_batched(self,
                        x: torch.Tensor,             # (S, num_tokens, ch): S samples of ONE conditioning
                        mask: torch.Tensor,          # (num_tokens,)
                        pair_logits: torch.Tensor,   # (num_heads, num_tokens, num_tokens), shared by the S samples
                        single_cond: torch.Tensor    # (num_tokens, c_single_cond), shared by the S samples
                        ) -> torch.Tensor:
        """lever 'sbatch' on the xfold route (DTK absent or stepped aside) when the attention core of this class is PATCHED (the kit's 'apb' lever
        replaces `forward` with an SDPA statement over (num_tokens, ch) activations): this block over a leading sample axis in that lever's own
        formulation: the adaptive layer norms, the projections, the gate and AdaLNZero are the module's ops (their [N, .] conditioning
        broadcasts over S); the attention core is ONE scaled_dot_product_attention over the S samples with the additive mask
        `pair_logits (+ -1e9 on masked keys)` [1, H, N, N] broadcast, never [S, H, N, N] materialised. With the class's own `forward` in place
        (no kernel lever: `--mode exact` / `off`'s base) the transformer calls `forward` itself, which carries the sample axis (above)."""
        assert x.dim() == 3 and single_cond is not None and self.use_single_cond
        S, N, _ = x.shape

        x = self.adaptive_layernorm(x, single_cond)

        q = self.q_projection(x)
        k = self.k_projection(x)
        v = self.v_projection(x)
        q, k, v = (t.unflatten(-1, (self.num_head, self.qkv_dim)).transpose(1, 2) for t in (q, k, v))   # [S, H, N, Dh] views

        am = pair_logits[None].to(q.dtype)                                                                 # [1, H, N, N]
        if mask is not None:
            am = am + ((1.0 - mask.to(q.dtype)) * -1e9)[None, None, None, :]
        weighted_avg = F.scaled_dot_product_attention(q, k, v, attn_mask=am, scale=self.qkv_dim ** -0.5)  # [S, H, N, Dh]
        weighted_avg = weighted_avg.transpose(1, 2).reshape(S, N, self.num_head * self.qkv_dim)

        gate_logits = self.gating_query(x)
        weighted_avg = weighted_avg * torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond)


_SELF_ATTENTION_FORWARD = SelfAttention.forward     # the class's own attention statements (a kernel lever that patches `forward` is told apart by identity)


class DiffusionTransformer(nn.Module):
    def __init__(self,
                 c_act: int = 768,
                 c_single_cond: int = 384,
                 c_pair_cond: int = 128,
                 num_head: int = 16,
                 num_blocks: int = 24,
                 super_block_size: int = 4) -> None:

        super(DiffusionTransformer, self).__init__()

        self.c_act = c_act
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond
        self.num_head = num_head
        self.num_blocks = num_blocks
        self.super_block_size = super_block_size

        self.num_super_blocks = self.num_blocks // self.super_block_size

        self.of3 = of3.OF3
        if self.of3:
            # OF3 layout: every block owns a pair LayerNorm (no offset) + Linear(c_pair, num_head)
            self.pair_input_layer_norm = nn.ModuleList(
                [fastnn.LayerNorm(self.c_pair_cond, bias=False) for _ in range(self.num_blocks)])
            self.pair_logits_projection = nn.ModuleList(
                [nn.Linear(self.c_pair_cond, self.num_head, bias=False) for _ in range(self.num_blocks)])
        else:
            self.pair_input_layer_norm = fastnn.LayerNorm(self.c_pair_cond, bias=False)
            self.pair_logits_projection = nn.ModuleList(
                [nn.Linear(self.c_pair_cond, self.super_block_size * self.num_head, bias=False) for _ in range(self.num_super_blocks)])

        self.self_attention = nn.ModuleList(
            [SelfAttention(self.c_act, self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])
        self.transition_block = nn.ModuleList(
            [DiffusionTransition(self.c_act, self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])

    def pair_logits_for_block(self, pair_cond: torch.Tensor, block_idx: int, _cache: dict) -> torch.Tensor:
        """[num_head, N, N] pair logits for block `block_idx` (both layouts). `_cache` memoises per-superblock work."""
        if self.of3:
            pair_act = self.pair_input_layer_norm[block_idx](pair_cond)
            return einops.rearrange(self.pair_logits_projection[block_idx](pair_act), 'n s h -> h n s')
        sb, j = divmod(block_idx, self.super_block_size)
        if "pair_act" not in _cache:
            _cache["pair_act"] = self.pair_input_layer_norm(pair_cond)
        if ("sb", sb) not in _cache:
            pl = self.pair_logits_projection[sb](_cache["pair_act"])
            _cache[("sb", sb)] = einops.rearrange(pl, 'n s (b h) -> b h n s', h=self.num_head)
        return _cache[("sb", sb)][j]

    def precompute_pair_logits(self, pair_cond: torch.Tensor) -> torch.Tensor:
        """All blocks' pair logits [num_blocks, num_head, N, N]. They depend only on pair_cond, which is constant over
        the diffusion trajectory, so a sampler may compute them once per sample() call (HOIST lever; same arithmetic)."""
        cache = {}
        return torch.stack([self.pair_logits_for_block(pair_cond, i, cache) for i in range(self.num_blocks)], dim=0)

    def forward(self,
                act: torch.Tensor,
                mask: torch.Tensor,
                single_cond: torch.Tensor,
                pair_cond:  torch.Tensor,
                pair_logits: Optional[torch.Tensor] = None):
        cache = {}
        if act.dim() == 3:                                   # lever 'sbatch': [S, N, ch] — S samples of one conditioning (the DTK swap, when live, replaced this forward and batches itself)
            patched = SelfAttention.forward is not _SELF_ATTENTION_FORWARD   # the attention core replaced by a kernel lever ('apb'): its batched SDPA form; else the class's own statements
            for i in range(self.num_blocks):
                pl = pair_logits[i] if pair_logits is not None else self.pair_logits_for_block(pair_cond, i, cache)
                sa = self.self_attention[i]
                act += sa.forward_batched(act, mask, pl, single_cond) if patched else sa(act, mask, pl, single_cond)
                act += self.transition_block[i](act, single_cond)
            return act
        for i in range(self.num_blocks):
            pl = pair_logits[i] if pair_logits is not None else self.pair_logits_for_block(pair_cond, i, cache)
            act += self.self_attention[i](act, mask, pl, single_cond)
            act += self.transition_block[i](act, single_cond)
        return act


class CrossAttention(nn.Module):
    def __init__(self, key_dim: int = 128, value_dim: int = 128, c_single_cond: int = 128, num_head: int = 4) -> None:
        super(CrossAttention, self).__init__()

        self.key_dim = key_dim
        self.value_dim = value_dim
        self.c_single_cond = c_single_cond
        self.num_head = num_head

        self.key_dim_per_head = self.key_dim // self.num_head
        self.value_dim_per_head = self.value_dim // self.num_head

        self.q_scale = self.key_dim_per_head ** (-0.5)

        self.q_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True)
        self.k_adaptive_layernorm = AdaptiveLayerNorm(
            c_x=self.key_dim, c_single_cond=self.c_single_cond, use_single_cond=True)

        self.q_projection = nn.Linear(self.key_dim, self.key_dim, bias=True)
        self.k_projection = nn.Linear(self.key_dim, self.key_dim, bias=False)
        self.v_projection = nn.Linear(
            self.value_dim, self.value_dim, bias=False)

        self.gating_query = nn.Linear(self.key_dim, self.value_dim, bias=False)
        self.adaptive_zero_init = AdaLNZero(
            self.value_dim, self.value_dim, self.key_dim, use_single_cond=True)

    def forward(
        self,
        x_q: torch.Tensor,
        x_k: torch.Tensor,
        mask_q: torch.Tensor,
        mask_k: torch.Tensor,
        pair_logits: Optional[torch.Tensor] = None,
        single_cond_q: Optional[torch.Tensor] = None,
        single_cond_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        assert 1 <= len(mask_q.shape) <= len(x_q.shape) - \
            1, f'{mask_q.shape}, {x_q.shape}'                   # x may carry leading sample axes the masks lack (lever 'sbatch'): every mask term below broadcasts
        assert len(mask_k.shape) == len(mask_q.shape) and len(x_k.shape) == len(x_q.shape), \
            f'{mask_k.shape}, {x_k.shape}'

        bias = (
            1e9
            * mask_q.logical_not()[..., None, :, None]
            * mask_k.logical_not()[..., None, None, :]
        )

        x_q = self.q_adaptive_layernorm(x_q, single_cond_q)
        x_k = self.k_adaptive_layernorm(x_k, single_cond_k)

        q = self.q_projection(x_q)
        k = self.k_projection(x_k)
        q = torch.reshape(q, q.shape[:-1] +
                          (self.num_head, self.key_dim_per_head))
        k = torch.reshape(k, k.shape[:-1] +
                          (self.num_head, self.key_dim_per_head))

        logits = torch.einsum('...qhc,...khc->...hqk',
                              q * self.q_scale, k) + bias
        if pair_logits is not None:
            logits += pair_logits
        weights = torch.softmax(logits, axis=-1)

        v = self.v_projection(x_k)
        v = torch.reshape(v, v.shape[:-1] +
                          (self.num_head, self.value_dim_per_head))
        weighted_avg = torch.einsum('...hqk,...khc->...qhc', weights, v)
        weighted_avg = torch.reshape(
            weighted_avg, weighted_avg.shape[:-2] + (-1,))

        gate_logits = self.gating_query(x_q)
        weighted_avg *= torch.sigmoid(gate_logits)

        return self.adaptive_zero_init(weighted_avg, single_cond_q)


class DiffusionCrossAttTransformer(nn.Module):
    def compute_pair_logits(self, pair_cond: torch.Tensor) -> torch.Tensor:
        """[num_blocks, num_subsets, num_head, q, k] pair logits; depends only on pair_cond (step-invariant in the sampler)."""
        pair_act = self.pair_input_layer_norm(pair_cond)
        pair_logits = self.pair_logits_projection(pair_act)
        return einops.rearrange(pair_logits, 'n q k (b h) -> b n h q k', h=self.num_head)

    # ------------------------------------------------------------------ lever 'atom_window': the blocks' attention half on the support library's
    # sequence-local atom-attention kernels (atom_window: ln_qkvg + window_attn). The queries layout is the flat atom list cut into blocks of 32
    # (num_subsets x num_queries); every block's 128 keys are a CONTIGUOUS window of that list (queries_to_keys gathers rows [start, start + 128),
    # the window centred on the block and shifted inward to stay inside the real atoms) — so the kernels read the keys in place from per-block
    # window starts instead of gathering them, compute the key-side AdaLN and k / v once per atom instead of once per gathered copy, and fuse
    # LayerNorm + both AdaLNs + q|k|v|gate, then logits + pair bias + mask + softmax + p v + gate + output projection + AdaLN-Zero gate +
    # residual, into two launches per block. Their step-invariant operands (window_operands: the AdaLN modulations and the AdaLN-Zero gate from
    # the per-atom conditioning rows, the atom mask, the window starts) are computed once per trajectory with this module's own layers
    # (DiffusionHead.prime_static) and handed in as `window`; a leading sample axis on queries_act (lever 'sbatch') is the kernels' sample axis
    # (conditioning rows, pair logits and mask shared by the samples, never expanded). Only the query blocks that hold real atoms are given to
    # the kernels (the first ceil(n_real / 32) blocks: a padding block's rows never reach a real atom's output — its keys are never read by a
    # real block, the transition is per row, the aggregation to tokens and the position update are masked); the padding rows keep their input.
    # Numerics: fp32 in / out with TF32 round-to-nearest tensor-core dots (the kernels' stated class) where this route runs bf16 matmuls under
    # autocast — the tolerance class, at or inside the bf16 route's distance from an fp32 reference; the transition half is this module's own.
    WINDOW_KERNEL = None            # the routed opt_core.kernels.atom_window module (af3_torch_api.build_model binds it when the lever is on)
    WINDOW_PRECISION = "tf32rn"     # the kernels' dot precision word (tf32rn: cuBLAS-TF32 class; tf32x3 ~ fp32; ieee)
    WINDOW_LN_KW: dict = {}         # the ln_qkvg launch of this card (af3_torch_api.ATOM_WINDOW_CELLS: the row tile that fits its shared memory)
    WINDOW_COUNTS = {"calls": 0, "blocks_real": 0, "blocks_total": 0, "refused": {}}   # census (forward.json items: atom_window)
    WINDOW_ROWS_ONLY = False        # lever 'atom_rows' (af3_torch_api.enable_atom_rows): the three blocks — attention AND transition — run on ONE contiguous fp32
    ROWS_COUNTS = {"calls": 0, "refused": {}}   # working copy of the real query blocks' rows, written back once (census forward.json items: atom_rows)

    def window_geometry_ok(self, queries_act: torch.Tensor, queries_to_keys) -> Optional[str]:
        """None when the kernels serve this call, else the refusal word (the stock statements run, counted under it)."""
        AW = DiffusionCrossAttTransformer.WINDOW_KERNEL
        if AW is None:
            return "kernel_absent"
        if not queries_act.is_cuda:
            return "device_cpu"
        nq = int(queries_act.shape[-2]); nk = int(queries_to_keys.gather_idxs.shape[-1])
        if (nq, nk) != (32, 128):
            return "window_geom_%dx%d" % (nq, nk)
        C = int(queries_act.shape[-1]); H = int(self.num_head)
        if C % H or C // H not in (16, 32, 64) or C % 16 or C > 128:
            return "head_geom"
        if queries_act.dim() not in (3, 4):
            return "rank_%d" % queries_act.dim()
        return None

    def window_operands(self, queries_single_cond: torch.Tensor, queries_mask: torch.Tensor) -> dict:
        """The step-invariant kernel operands of every block for one trajectory (all [A, C] fp32 contiguous over the flat atom list, A =
        num_subsets * 32): per block b `gq{b}` / `lsq{b}` (sigmoid(scale), bias of the query-side AdaLN), `gk{b}` / `lsk{b}` (key side, on the
        same per-atom rows the key windows read), `zg{b}` (the AdaLN-Zero gate sigmoid(adaptive_zero_cond(cond))); `amask` [1, A]; `n_real`
        int32 [1]; `ks` int32 [num_subsets] window starts; `rows` (0-dim int64: 32 * ceil(n_real / 32), the rows handed to the kernels)."""
        AW = DiffusionCrossAttTransformer.WINDOW_KERNEL
        ns, nq, C = int(queries_single_cond.shape[-3]), int(queries_single_cond.shape[-2]), int(queries_single_cond.shape[-1])
        A = ns * nq
        s = queries_single_cond.reshape(A, C)
        out = {}
        for b, ca in enumerate(self.cross_attention):
            lq, lk = ca.q_adaptive_layernorm, ca.k_adaptive_layernorm
            sq = lq.single_cond_layer_norm(s); sk = lk.single_cond_layer_norm(s)
            out["gq%d" % b] = torch.sigmoid(lq.single_cond_scale(sq)).float().contiguous()
            out["lsq%d" % b] = lq.single_cond_bias(sq).float().contiguous()
            out["gk%d" % b] = torch.sigmoid(lk.single_cond_scale(sk)).float().contiguous()
            out["lsk%d" % b] = lk.single_cond_bias(sk).float().contiguous()
            out["zg%d" % b] = torch.sigmoid(ca.adaptive_zero_init.adaptive_zero_cond(s)).float().contiguous()
        amask = queries_mask.reshape(1, A).to(torch.float32).contiguous()
        n_real = amask.sum(-1)                                        # real atoms (the flat list holds them first, padding after): stays on the device
        out["amask"] = amask
        out["n_real"] = n_real.round().to(torch.int32)
        out["ks"] = AW.window_starts(A, n_real[0], nq, 128, queries_mask.device).to(torch.int32).contiguous()
        out["rows"] = (torch.ceil(n_real / nq) * nq).to(torch.int64).reshape(())   # read back ONCE per trajectory by the caller (prime_static), never per step
        return out

    def forward_windowed_rows(self, queries_act, queries_single_cond, pair_logits, window: dict, R: int, NB: int) -> torch.Tensor:
        """Lever 'atom_rows': forward_windowed() on ONE contiguous fp32 working copy of the first R atom rows (the query blocks that hold real
        atoms): the window kernels read and write it in place of a strided view (no per-block contiguity copies), the compiled transition and
        its residual run on those R rows instead of on every row of the padded layout, and the rows are cast back into the caller's activation
        once (IN PLACE: the caller's tensor is returned). Rows >= R hold no real atom: they keep their input values here where forward_windowed
        gives them the transition's output — both callers multiply the result by the atom mask right after, which zeroes those rows either way.
        The transition on R rows is the same row-local statement (compiled with a dynamic row count: af3_torch_api.enable_atom_rows); its
        GEMMs may pick another algorithm for the shorter operand (the tolerance class, as across padding buckets)."""
        AW = DiffusionCrossAttTransformer.WINDOW_KERNEL
        prec = DiffusionCrossAttTransformer.WINDOW_PRECISION
        shape = queries_act.shape
        ns, nq, C = int(shape[-3]), int(shape[-2]), int(shape[-1])
        A = ns * nq; H = int(self.num_head)
        qa = queries_act.view((-1, A, C))                            # [S, A, C]: a view (the callers hand a contiguous activation; checked by the dispatcher)
        a = qa[:, :R].to(dtype=torch.float32, memory_format=torch.contiguous_format, copy=True)   # [S, R, C] fp32 contiguous: the three blocks' residual stream
        cond = queries_single_cond.reshape((A, -1))[:R]              # [R, c_cond]: the rows' single conditioning (a leading slice: contiguous, no copy)
        for b, ca in enumerate(self.cross_attention):
            qkvg = AW.ln_qkvg(a, window["gq%d" % b][:R], window["lsq%d" % b][:R], window["gk%d" % b][:R], window["lsk%d" % b][:R],
                              ca.q_projection, ca.k_projection, ca.v_projection, ca.gating_query, ca.q_adaptive_layernorm.layer_norm.eps,
                              float(ca.q_scale), precision=prec, **DiffusionCrossAttTransformer.WINDOW_LN_KW)
            bias = window.get("bias%d" % b)
            if bias is None or int(bias.shape[0]) != NB:
                bias = pair_logits[b][:NB].to(torch.float32).contiguous()
            y = AW.window_attn(qkvg, a, bias, window["ks"][:NB], window["n_real"], window["amask"][:, :R],
                               window["zg%d" % b][:R], ca.adaptive_zero_init.transition2.weight, ca.adaptive_zero_init.transition2.bias, H,
                               n_query=nq, n_key=128, inf=1e9, precision=prec)   # a + gate * W_o(attention) on the R rows: a fresh [S, R, C] fp32
            tb = self.transition_block[b]
            a = y
            a += getattr(tb, "forward_rows", tb.forward)(a, cond)   # the compiled transition on the R rows (+ its residual), dynamic in R
        qa[:, :R].copy_(a)                                            # ONE cast back into the caller's activation; rows >= R untouched (masked by the caller)
        cnt = DiffusionCrossAttTransformer.WINDOW_COUNTS
        cnt["calls"] += 1; cnt["blocks_real"] += NB; cnt["blocks_total"] += ns
        DiffusionCrossAttTransformer.ROWS_COUNTS["calls"] += 1
        return queries_act

    def forward_windowed(self, queries_act, queries_single_cond, pair_logits, window: dict, rows: int) -> torch.Tensor:
        """forward() with every block's attention half on the window kernels over the first `rows` atom rows; same output shape and dtype."""
        AW = DiffusionCrossAttTransformer.WINDOW_KERNEL
        prec = DiffusionCrossAttTransformer.WINDOW_PRECISION
        shape, dtype = queries_act.shape, queries_act.dtype
        ns, nq, C = int(shape[-3]), int(shape[-2]), int(shape[-1])
        A = ns * nq; H = int(self.num_head)
        R = max(nq, min(int(rows), A)); NB = R // nq
        if DiffusionCrossAttTransformer.WINDOW_ROWS_ONLY:            # lever 'atom_rows': the rows-only restatement above, unless this call's layout refuses it (named)
            why = None if (queries_act.is_contiguous() and queries_act.dim() in (3, 4) and queries_single_cond.is_contiguous()
                           and tuple(queries_single_cond.shape[:-1]) == (ns, nq)) else "layout"
            if why is None:
                return self.forward_windowed_rows(queries_act, queries_single_cond, pair_logits, window, R, NB)
            refused = DiffusionCrossAttTransformer.ROWS_COUNTS["refused"]; refused[why] = refused.get(why, 0) + 1
        a = queries_act.reshape(-1, A, C).to(torch.float32)         # [S, A, C] fp32 working copy (the residual stream of the three blocks stays fp32)
        if a.data_ptr() == queries_act.data_ptr():
            a = a.clone()
        for b, ca in enumerate(self.cross_attention):
            qkvg = AW.ln_qkvg(a[:, :R], window["gq%d" % b][:R], window["lsq%d" % b][:R], window["gk%d" % b][:R], window["lsk%d" % b][:R],
                              ca.q_projection, ca.k_projection, ca.v_projection, ca.gating_query, ca.q_adaptive_layernorm.layer_norm.eps,
                              float(ca.q_scale), precision=prec, **DiffusionCrossAttTransformer.WINDOW_LN_KW)
            bias = window.get("bias%d" % b)                           # the hoisted logits of the real blocks, fp32 with the keys contiguous (the kernel's operand): laid once per
            if bias is None or int(bias.shape[0]) != NB:              # trajectory with the other window operands (DiffusionHead.prime_static), else converted here
                bias = pair_logits[b][:NB].to(torch.float32).contiguous()
            y = AW.window_attn(qkvg, a[:, :R], bias, window["ks"][:NB], window["n_real"], window["amask"][:, :R],
                               window["zg%d" % b][:R], ca.adaptive_zero_init.transition2.weight, ca.adaptive_zero_init.transition2.bias, H,
                               n_query=nq, n_key=128, inf=1e9, precision=prec)
            a[:, :R].copy_(y)                                         # a + gate * W_o(attention): the block's residual is inside the kernel
            a4 = a.view((-1,) + tuple(shape[-3:])) if queries_act.dim() == 4 else a.view(shape)
            a4 += self.transition_block[b](a4, queries_single_cond)
        cnt = DiffusionCrossAttTransformer.WINDOW_COUNTS
        cnt["calls"] += 1; cnt["blocks_real"] += NB; cnt["blocks_total"] += ns
        return (a.view((-1,) + tuple(shape[-3:])) if queries_act.dim() == 4 else a.view(shape)).to(dtype)

    def __init__(self, c_query: int = 128, c_single_cond: int = 128, c_pair_cond: int = 16, num_blocks: int = 3, num_head: int = 4) -> None:
        super(DiffusionCrossAttTransformer, self).__init__()

        self.c_query = c_query
        self.c_single_cond = c_single_cond
        self.c_pair_cond = c_pair_cond

        self.num_blocks = num_blocks
        self.num_head = num_head

        self.pair_input_layer_norm = fastnn.LayerNorm(self.c_pair_cond, bias=False)
        self.pair_logits_projection = nn.Linear(
            self.c_pair_cond, self.num_blocks * self.num_head, bias=False)

        self.cross_attention = nn.ModuleList(
            [CrossAttention(num_head=self.num_head) for _ in range(self.num_blocks)])

        self.transition_block = nn.ModuleList(
            [DiffusionTransition(c_x=self.c_query, c_single_cond=self.c_single_cond, use_single_cond=True) for _ in range(self.num_blocks)])

    def forward(
        self,
        queries_act: torch.Tensor,  # (num_subsets, num_queries, ch)
        queries_mask: torch.Tensor,  # (num_subsets, num_queries)
        queries_to_keys: atom_layout.GatherInfo,  # (num_subsets, num_keys)
        keys_mask: torch.Tensor,  # (num_subsets, num_keys)
        queries_single_cond: torch.Tensor,  # (num_subsets, num_queries, ch)
        keys_single_cond: torch.Tensor,  # (num_subsets, num_keys, ch)
        pair_cond: torch.Tensor,  # (num_subsets, num_queries, num_keys, ch)
        pair_logits: Optional[torch.Tensor] = None,  # HOIST: precomputed by compute_pair_logits (step-invariant)
        window: Optional[dict] = None,  # lever 'atom_window': window_operands() of this trajectory (+ rows_host), hoisted by DiffusionHead.prime_static
    ) -> torch.Tensor:

        if pair_logits is None:
            pair_logits = self.compute_pair_logits(pair_cond)

        if window is not None:                                       # lever 'atom_window' (above): the operands were hoisted for this trajectory
            why = self.window_geometry_ok(queries_act, queries_to_keys)
            if why is None:
                return self.forward_windowed(queries_act, queries_single_cond, pair_logits, window, int(window["rows_host"]))
            refused = DiffusionCrossAttTransformer.WINDOW_COUNTS["refused"]; refused[why] = refused.get(why, 0) + 1   # named, counted: the statements below run

        for block_idx in range(self.num_blocks):
            keys_act = atom_layout.convert(
                queries_to_keys, queries_act, layout_axes=(-3, -2)
            )

            queries_act += self.cross_attention[block_idx](
                x_q=queries_act,
                x_k=keys_act,
                mask_q=queries_mask,
                mask_k=keys_mask,
                pair_logits=pair_logits[block_idx,...],
                single_cond_q=queries_single_cond,
                single_cond_k=keys_single_cond,
            )
            queries_act += self.transition_block[block_idx](
                queries_act,
                queries_single_cond,
            )

        return queries_act
