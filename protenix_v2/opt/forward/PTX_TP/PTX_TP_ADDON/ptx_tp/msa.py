# Portions derived from Protenix v2.0.0 (https://github.com/bytedance/Protenix), Copyright 2024 ByteDance and/or its affiliates,
# used under the Apache License, Version 2.0.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tensor-parallel (row-sharded pair rep / token-sharded MSA rep) re-statement of
Protenix-2.x ``MSAModule.forward`` for inference.

Stock source of truth (protenix 2.0.0 wheel):
  protenix/model/modules/pairformer.py : MSAPairWeightedAveraging (343-424),
      MSAStack (425-574), MSABlock (575-681), MSAModule (682-918)
  protenix/model/triangular/layers.py  : OuterProductMean (657-787)

Conventions (shared PTX-TP contract):
  * pair rep     z_shard = z[r0:r1, :, :]      (row shard; plain bf16 tensor OR a ZStore)
  * MSA rep      m_shard = m[:, r0:r1, :]      (token shard, plain tensor)
  * s_inputs, input_feature_dict               replicated on every rank
  * layout.replicated -> every function runs the stock single-device statement.

z access: z shards are plain bf16 tensors; op code reads row blocks with
zrows(z, i0, i1) (LOCAL rows, multiples of 128 except the tail) and updates with
zadd(z, i0, i1, delta) - thin tensor helpers in ptx_tp._h2util.

P-invariance: every output element's full contraction is computed inside ONE
GEMM on ONE rank; stock chunk grids are iterated in GLOBAL row coordinates;
the only P-dependence of any kernel is the row count M of row-local GEMMs.
No split-K across ranks, no partial-sum all-reduce anywhere.  All collectives
are bit-preserving copies.

The MSA pair stack (tri_mul out/in, tri_att start/end, pair transition) is
delegated to
    ptx_tp.trimul.tp_trimul, ptx_tp.triatt.tp_triatt, ptx_tp.rowlocal.tp_transition
resolved lazily at call time.

Call under the SAME ambient autocast context as the stock trunk
(runner/inference.py: torch.autocast("cuda", dtype=bf16)).
"""
from __future__ import annotations

from typing import Any, Dict, Iterator, Optional, Tuple

import torch
import torch.nn.functional as F  # noqa: F401

from protenix.model.utils import is_fp16_enabled, sample_indices

from ptx_tp.dist import Layout, all_gather_rows, allreduce_checksum
from ptx_tp._h2util import ZBLK, broadcast_tensor, zadd, zblocks, zrows, zwrite  # noqa: F401

__all__ = [
    "tp_msa_module",
    "tp_msa_embed",
    "tp_msa_block",
    "tp_msa_stack",
    "tp_msa_pair_weighted_averaging",
    "tp_outer_product_mean",
    "tp_outer_product_mean_iter",
    "tp_pair_stack_block",
    "all_gather_tokens",
    "zadd_rows",
    "MEMORY_KNOBS",
]

# Opt-in memory levers for very large N.  Default None == stock GEMM shapes (bitwise
# parity with stock).  Both iterate P-INVARIANT global grids, so TP results stay
# identical across P; vs stock they change a GEMM's M / N-dim (rounding-level
# differences possible).
#   opm_j_block : OuterProductMean column (j) block size (global grid range(0, N, J)).
#   pwa_s_sub   : MSAPairWeightedAveraging MSA-depth sub-chunk inside one msa_chunk.
MEMORY_KNOBS: Dict[str, Optional[int]] = {
    "opm_j_block": None,
    "pwa_s_sub": None,
    # OuterProductMean computes LN(m), a = linear_1(ln), b = linear_2(ln) on the FULL token range
    # ([S, N, c_m] gathered from the token shards; the verbatim stock statements, identical GEMM shapes
    # AND strides) whenever S*N*c_m*elem_size <= this budget.  Reason: at small MSA depth (S = 1) the
    # shard-shaped projections (M = S*R rows, out = 32) are not kernel-invariant vs stock's M = S*N on
    # every card.  The default (16 GiB) covers every input size stock accepts (N <= 2,560, S <= 16,384
    # -> <= 10.7 GB); above it the token-sharded projections are used (M = S*R is then large).
    # 0 -> always sharded.
    "opm_full_ab_max_bytes": 16 << 30,
}


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _h1_api():
    """Resolve the pair-stack primitives lazily (contract names)."""
    from ptx_tp.trimul import tp_trimul  # noqa: WPS433
    from ptx_tp.triatt import tp_triatt  # noqa: WPS433
    from ptx_tp.rowlocal import tp_transition  # noqa: WPS433

    return tp_trimul, tp_triatt, tp_transition


def all_gather_tokens(x_shard: torch.Tensor, layout: Layout) -> torch.Tensor:
    """Token-dim gather for MSA-like tensors: [S, R_r, C] -> [S, N, C] (view, not
    necessarily contiguous).  Bit-preserving."""
    assert x_shard.dim() == 3, f"expected [S, R, C], got {tuple(x_shard.shape)}"
    xt = x_shard.transpose(0, 1).contiguous()  # [R, S, C]
    full = all_gather_rows(xt, layout)  # [N, S, C]
    return full.transpose(0, 1)  # [S, N, C]


def zadd_rows(z, delta: torch.Tensor, layout: Layout, l0: int = 0) -> None:
    """z[l0:l0+len(delta)] += delta, applied in 128-row local blocks via zadd."""
    n = delta.shape[0]
    for (a, b) in zblocks(n):
        zadd(z, l0 + a, l0 + b, delta[a:b])


def _zmeta(z, layout: Layout):
    """(dtype, device, C) of the compute view of a z shard (plain tensor or store)."""
    blk = zrows(z, 0, min(layout.R, 1))
    return blk.dtype, blk.device, blk.shape[-1]


def _stock_pair_kwargs(triangle_multiplicative, triangle_attention, inplace_safe, chunk_size):
    return dict(
        triangle_multiplicative=triangle_multiplicative,
        triangle_attention=triangle_attention,
        inplace_safe=inplace_safe,
        chunk_size=chunk_size,
    )


# --------------------------------------------------------------------------- #
# (M1) MSA sampling + one-hot + embedding, token-sharded
# --------------------------------------------------------------------------- #
def tp_msa_embed(
    msa_module,
    input_feature_dict: Dict[str, Any],
    s_inputs: torch.Tensor,
    layout: Layout,
    *,
    strict_rng: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reproduces MSAModule.forward lines 854-900 and returns (m_shard [S, R, c_m], indices).

    RNG: the stock sampler (protenix.model.utils.sample_indices: CPU torch.randint for
    the sample size + device torch.randperm) is executed on EVERY rank so that RNG
    consumption is identical to stock on every rank; the sampled indices are then
    asserted bitwise-identical across ranks (allreduce_checksum).  With
    strict_rng=False rank 0's indices are broadcast instead (RNG still consumed
    identically everywhere).
    """
    mod = msa_module
    dim_dict = {feat_name: -2 for feat_name in mod.input_feature}
    cutoff = mod.msa_configs["train_cutoff"] if mod.training else mod.msa_configs["test_cutoff"]
    lower_bound = mod.msa_configs["train_lowerb"] if mod.training else mod.msa_configs["test_lowerb"]
    msa = input_feature_dict["msa"]
    msa_len = msa.size(dim=dim_dict["msa"])
    # -- identical statements to sample_msa_feature_dict_random_without_replacement --
    indices = sample_indices(
        n=msa_len, device=msa.device, lower_bound=lower_bound, strategy=mod.msa_configs["strategy"]
    )
    if cutoff > 0:
        indices = indices[:cutoff]
    if strict_rng:
        allreduce_checksum(indices, "tp_msa_embed.sample_indices")
    else:
        indices = broadcast_tensor(indices.contiguous(), src=0)

    tok = slice(0, msa.shape[-1]) if layout.replicated else layout.rows()
    # index_select exactly as stock, then token-shard BEFORE the one-hot (memory /P)
    msa_feat = {
        feat_name: torch.index_select(input=input_feature_dict[feat_name], dim=dim, index=indices)[..., tok]
        for feat_name, dim in dim_dict.items()
    }
    n_token_full = msa.shape[-1]  # stock tests z.shape[-2] (= N_token of the full z)
    if (not mod.training) and n_token_full > 2000:
        msa_feat["msa"] = mod.one_hot_fp32(msa_feat["msa"], num_classes=mod.input_feature["msa"])
    else:
        msa_feat["msa"] = torch.nn.functional.one_hot(  # pylint: disable=not-callable
            msa_feat["msa"], num_classes=mod.input_feature["msa"]
        )
    target_shape = msa_feat["msa"].shape[:-1]
    msa_sample = torch.cat(
        [msa_feat[name].reshape(*target_shape, d) for name, d in mod.input_feature.items()], dim=-1
    )  # [S, R, 32+1+1]
    del msa_feat
    m_shard = mod.linear_no_bias_m(msa_sample)  # row-local GEMM (M = S*R)
    del msa_sample
    # s_inputs is replicated: run the stock-shaped GEMM on the full tensor, slice rows.
    s_emb = mod.linear_no_bias_s(s_inputs)  # [N, c_m]
    m_shard = m_shard + s_emb[..., tok, :]
    return m_shard, indices


# --------------------------------------------------------------------------- #
# (M2) MSAPairWeightedAveraging: row-local in z, all_gather of LN(m) tokens for v
# --------------------------------------------------------------------------- #
def tp_msa_pair_weighted_averaging(pwa, m_shard: torch.Tensor, z_shard, layout: Layout,
                                   *, s_sub: Optional[int] = None) -> torch.Tensor:
    """o_si = sum_j softmax_j(linear(LN z_i.))_ijh * v_sjhc for the rank's rows i in I.

    w needs FULL rows of z (row-local; read in 128-row blocks through zrows); v needs
    ALL tokens j: LN(m) token shards are all-gathered (c_m channels, half the bytes of
    gathering v) and v = linear_mv(LN(m)_full) is the stock-shaped GEMM on every rank.
    Gate g, output projection: row-local on the shard.  Returns the m update [S, R, c_m].
    s_sub (default MEMORY_KNOBS['pwa_s_sub'] = None): optional MSA-depth sub-chunk that
    bounds the replicated v to [s_sub, N, H*c] (lever for very large N; changes GEMM
    shapes vs stock -> rounding-level differences possible; P-invariant).
    """
    if layout.replicated:
        return pwa(m_shard, z_shard)
    s_sub = MEMORY_KNOBS["pwa_s_sub"] if s_sub is None else s_sub
    m_ln = _ln_blocked(pwa.layernorm_m, m_shard)  # [S, R, c_m]  (per-S-block above 2^31 elements)
    # b = linear_z(LN(z)) read block-wise through the accessor (thin tensor helper)
    if layout.R > 0:
        b = torch.cat([pwa.linear_no_bias_z(pwa.layernorm_z(zrows(z_shard, i0, i1)))
                       for (i0, i1) in zblocks(layout.R)], dim=0)  # [R, N, H]
    else:
        b = pwa.linear_no_bias_z(pwa.layernorm_z(zrows(z_shard, 0, 0)))
    w = pwa.softmax_w(b)  # softmax over j (dim=-2): full rows on this rank
    del b
    S = m_ln.shape[0]
    pieces = [(0, S)] if not s_sub else [(s0, min(S, s0 + s_sub)) for s0 in range(0, S, s_sub)]
    out = None
    for (s0, s1) in pieces:
        m_ln_p = m_ln if (s0 == 0 and s1 == S) else m_ln[s0:s1]
        m_ln_full = all_gather_tokens(m_ln_p, layout).contiguous()  # [s, N, c_m]
        v = pwa.linear_no_bias_mv(m_ln_full)  # [s, N, H*c]   (stock statement / shape when s_sub None)
        del m_ln_full
        v = v.reshape(*v.shape[:-1], pwa.n_heads, pwa.c)  # [s, N, H, c]
        g = torch.sigmoid(pwa.linear_no_bias_mg(m_ln_p))  # [s, R, H*c]
        g = g.reshape(*g.shape[:-1], pwa.n_heads, pwa.c)  # [s, R, H, c]
        wv = torch.einsum("...ijh,...mjhc->...mihc", w, v)  # [s, R, H, c]
        o = g * wv
        o = o.reshape(*o.shape[:-2], pwa.n_heads * pwa.c)  # [s, R, H*c]
        o = pwa.linear_no_bias_out(o)  # [s, R, c_m]
        del v, g, wv
        if len(pieces) == 1:
            out = o
        else:
            if out is None:
                out = o.new_empty((S,) + tuple(o.shape[1:]))
            out[s0:s1] = o
        del o
    del w, m_ln
    return out


# --------------------------------------------------------------------------- #
# (M2+M3) MSAStack inference path (in-place S-chunk loop, msa_chunk_size semantics)
# --------------------------------------------------------------------------- #
def tp_msa_stack(stack, m_shard: torch.Tensor, z_shard, layout: Layout) -> torch.Tensor:
    """MSAStack.inference_forward on the token shard (pairformer.py 547-572).

    The training branch (msa_max_size padding, pairformer.py 470-491) exists only to
    keep DDP graphs static and is not reproduced; eval dropout_row is a no-op.
    """
    if stack.training:
        raise NotImplementedError("ptx_tp.msa: MSAStack TP is inference-only (module.eval())")
    if layout.replicated:
        return stack(m_shard, z_shard)
    chunk_size = stack.msa_chunk_size
    num_msa = m_shard.shape[-3]
    no_chunks = num_msa // chunk_size + (num_msa % chunk_size != 0)
    for i in range(no_chunks):
        start = i * chunk_size
        end = min((i + 1) * chunk_size, num_msa)
        m_shard[start:end, :, :] += tp_msa_pair_weighted_averaging(
            stack.msa_pair_weighted_averaging, m_shard[start:end, :, :], z_shard, layout
        )
        # Transition is row-local: the stock statement on the shard IS the TP statement.
        m_shard[start:end, :, :] += stack.transition_m(m_shard[start:end, :, :])
    return m_shard


# --------------------------------------------------------------------------- #
# (M4) OuterProductMean: contraction over MSA depth kept whole on one rank
# --------------------------------------------------------------------------- #
def tp_outer_product_mean_iter(
    opm,
    m_shard: torch.Tensor,
    layout: Layout,
    *,
    chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    j_block: Optional[int] = None,
) -> Iterator[Tuple[int, int, torch.Tensor]]:
    """Yields (l0, l1, dz_rows [l1-l0, N, c_z]) over the STOCK chunk grid
    (chunk_layer over a's rows: global range(0, N, chunk_size) restricted to rows I;
    chunk_size=None -> one piece = the whole shard, as stock's single _opm call).
    l0/l1 are LOCAL row indices.  Replicates OuterProductMean.forward incl. the dtype
    branch (layers.py 776-787): fp16 autocast -> autocast disabled + float32; the
    runner's bf16 autocast (is_fp16_enabled() False) -> as-is.
    LN / linear_1 / linear_2 run on the full token range (stock shapes+strides) when
    S*N*c_m*elem <= MEMORY_KNOBS['opm_full_ab_max_bytes'] (default 16 GiB = whole stock envelope);
    the einsum/linear_out chunk GEMMs are row-sharded on stock's global chunk grid as before.
    j_block (default MEMORY_KNOBS['opm_j_block'] = None): optional GLOBAL column grid
    range(0, N, J) for the einsum/linear_out (lever for very large N; P-invariant;
    changes the GEMM N-dim vs stock -> rounding-level differences possible)."""
    assert not layout.replicated
    j_block = MEMORY_KNOBS["opm_j_block"] if j_block is None else j_block
    if is_fp16_enabled():
        with torch.amp.autocast("cuda", enabled=False):
            yield from _tp_opm_iter(opm, m_shard.float(), layout, chunk_size, inplace_safe, j_block)
    else:
        yield from _tp_opm_iter(opm, m_shard, layout, chunk_size, inplace_safe, j_block)


def _tp_opm_iter(opm, m: torch.Tensor, layout: Layout, chunk_size, inplace_safe, j_block=None):
    # m: [S, R, c_m]; MSABlock never passes a mask -> stock mask = ones (layers.py 740-741)
    assert m.dim() == 3, f"expected m_shard [S, R, c_m], got {tuple(m.shape)}"
    S, R = m.shape[-3], m.shape[-2]
    N = layout.N
    budget = MEMORY_KNOBS.get("opm_full_ab_max_bytes")
    full_ab = budget is not None and (S * N * m.shape[-1] * m.element_size()) <= int(budget)
    if full_ab:
        # the projections are the VERBATIM stock statements on the full token range (layers.py
        # 738-757): same shapes, same strides -> same kernels as stock for every S (incl. S = 1).
        m_full = all_gather_tokens(m, layout).contiguous()  # [S, N, c_m]  (bit copy of every rank's shard)
        mask = m_full.new_ones(m_full.shape[:-1])  # [S, N]
        ln = _ln_blocked(opm.layer_norm, m_full)  # [S, N, c_m]  (per-S-block above 2^31 elements: deep MSA x large N)
        del m_full
        mask = mask.unsqueeze(-1)  # [S, N, 1]
        a = opm.linear_1(ln)
        a = a * mask
        b = opm.linear_2(ln)
        b = b * mask
        del ln
        a = a.transpose(-2, -3)  # [N, S, C] view, strides (C, N*C, 1) exactly as stock
        b_full = b.transpose(-2, -3)  # [N, S, C] view exactly as stock's b
        del b
        a = a[layout.r0:layout.r1]  # rows I (view; stock's chunk views a[i:i+cs] have the same strides)
        mask_rows = mask[:, layout.r0:layout.r1]  # [S, R, 1]
        mask_full = mask  # [S, N, 1]
    else:
        mask = m.new_ones(m.shape[:-1])  # [S, R]
        ln = _ln_blocked(opm.layer_norm, m)  # [S, R, c_m]
        mask = mask.unsqueeze(-1)  # [S, R, 1]
        a = opm.linear_1(ln)
        a = a * mask
        b = opm.linear_2(ln)
        b = b * mask
        del ln
        a = a.transpose(-2, -3)  # [R, S, C]
        b = b.transpose(-2, -3)  # [R, S, C]
        b_full = all_gather_rows(b.contiguous(), layout)  # [N, S, C]  (all tokens j)
        del b
        mask_rows = mask
        mask_full = m.new_ones((S, N, 1))
    # [R, N, 1] mask-norm exactly as stock, restricted to rows I (all-ones -> the integer S, exact in fp32
    # accumulation, then the same bf16/fp32 rounding as stock's [N, N, 1] statement)
    norm = torch.einsum("...abc,...adc->...bdc", mask_rows, mask_full)
    norm = norm + opm.eps
    r0 = layout.r0
    grid = [(0, R)] if chunk_size is None else [(c0 - r0, c1 - r0) for (c0, c1) in layout.chunks(chunk_size)]
    for (l0, l1) in grid:
        if not j_block:
            outer = opm._opm(a[l0:l1], b_full)  # [l1-l0, N, c_z]  einsum + linear_out as stock
        else:
            outer = None
            for j0 in range(0, N, j_block):
                j1 = min(N, j0 + j_block)
                pj = opm._opm(a[l0:l1], b_full[j0:j1])  # [l1-l0, j1-j0, c_z]
                if outer is None:
                    outer = pj.new_zeros((l1 - l0, N) + tuple(pj.shape[2:]))
                outer[:, j0:j1] = pj
                del pj
        if inplace_safe:
            outer /= norm[l0:l1]
        else:
            outer = outer / norm[l0:l1]
        yield (l0, l1, outer)
        del outer
    del a, b_full, norm


def tp_outer_product_mean(
    opm,
    m_shard: torch.Tensor,
    layout: Layout,
    *,
    chunk_size: Optional[int] = None,
    inplace_safe: bool = False,
    j_block: Optional[int] = None,
) -> torch.Tensor:
    """Materialised variant: returns the z update for the rank's rows [R, N, c_z]
    (chunk pieces written into a pre-allocated buffer exactly like stock chunk_layer)."""
    if layout.replicated:
        return opm(m_shard, chunk_size=chunk_size, inplace_safe=inplace_safe)
    out = None
    for (l0, l1, piece) in tp_outer_product_mean_iter(opm, m_shard, layout, chunk_size=chunk_size,
                                                       inplace_safe=inplace_safe, j_block=j_block):
        if chunk_size is None:
            return piece
        if out is None:
            out = piece.new_zeros((layout.R,) + tuple(piece.shape[1:]))
        out[l0:l1] = piece
    if out is None:  # R == 0 on this rank
        out = torch.zeros((0, layout.N, opm.c_z), dtype=m_shard.dtype, device=m_shard.device)
    return out


# --------------------------------------------------------------------------- #
# pair stack of a PairformerBlock with c_s == 0 (MSABlock.pair_stack, template blocks)
# --------------------------------------------------------------------------- #

def _ln_blocked(ln_mod, x, limit=None):
    """Evaluate a LayerNorm over dim-0 blocks into a preallocated output when the
    input has >= PTX_TP_LN_NUMEL_LIMIT (2^31) elements (torch layer_norm is wrong above 2^32; per-row arithmetic identical; below the limit the
    single call is kept so bytes are unchanged)."""
    import os as _os
    lim = int(_os.environ.get("PTX_TP_LN_NUMEL_LIMIT", str(2 ** 31))) if limit is None else int(limit)
    if x.numel() < lim or x.shape[0] <= 1:
        return ln_mod(x)
    per_row = max(1, x.numel() // x.shape[0])
    step = max(1, (lim - 1) // per_row)
    out = None
    for s0 in range(0, x.shape[0], step):
        y = ln_mod(x[s0:s0 + step])
        if out is None:
            out = torch.empty((x.shape[0],) + tuple(y.shape[1:]), dtype=y.dtype, device=y.device)
        out[s0:s0 + step] = y
        del y
    return out


def _opmem(tag, layout):
    """Per-rank memory line before each pair op at large N (every rank prints its own numbers; no collectives)."""
    try:
        import os as _os
        if layout.N >= int(_os.environ.get("PTX_TP_PAIROP_MEM_N", "15000")) and torch.cuda.is_available():
            print(f"[ptx_tp pairop r{layout.rank}] {tag}: allocated {torch.cuda.memory_allocated()/2**30:.2f} GiB | reserved {torch.cuda.memory_reserved()/2**30:.2f} | max {torch.cuda.max_memory_allocated()/2**30:.2f}", flush=True)
    except Exception:
        pass


def tp_pair_stack_block(
    block,
    z_shard,
    layout: Layout,
    *,
    triangle_multiplicative: str = "torch",
    triangle_attention: str = "torch",
    inplace_safe: bool = False,
    chunk_size: Optional[int] = None,
):
    """PairformerBlock.forward (pairformer.py 138-216) for c_s == 0, composed from the
    pair-stack primitives on the block's own sub-modules.  DropoutRowwise / dropout_add_rowwise
    are eval no-ops (residual + x).  pair_mask is None/all-ones in both MSA and template
    contexts (the pair-stack API carries no mask).  Updates are applied with blockwise zadd.

    Primitive semantics (contract): tp_trimul(with_add=True) returns the z shard with
    the update added (stock `_add_with_inplace=True`); tp_triatt(starting=False) handles
    the transposed (ending-node) frame internally and returns the update in z's row
    frame; tp_transition returns the update.
    inplace_safe=False: a plain-tensor input is cloned first and the same in-place
    composition is applied to the clone (eval numerics identical to stock's
    z = z + update sequence for the inference kernels).
    """
    if layout.replicated:
        _, z = block(
            None, z_shard, None,
            **_stock_pair_kwargs(triangle_multiplicative, triangle_attention, inplace_safe, chunk_size),
        )
        return z
    assert getattr(block, "c_s", 0) == 0, "tp_pair_stack_block handles the c_s == 0 pair stack only"
    tp_trimul, tp_triatt, tp_transition = _h1_api()
    att_kw = dict(triangle_attention=triangle_attention, chunk_size=chunk_size, inplace_safe=inplace_safe)
    z = z_shard
    if (not inplace_safe) and torch.is_tensor(z):
        z = z.clone()
    c = z.shape[-1] if torch.is_tensor(z) else -1
    _opmem(f"c={c} before tri_mul_out", layout)
    z = tp_trimul(block.tri_mul_out, z, layout, outgoing=True, with_add=True)
    _opmem(f"c={c} before tri_mul_in", layout)
    z = tp_trimul(block.tri_mul_in, z, layout, outgoing=False, with_add=True)
    try:                                                   # in-place, two-shard-safe variants when the modules provide them
        from ptx_tp.triatt import tp_triatt_ as _tta_
        from ptx_tp.rowlocal import tp_transition_add_ as _tra_
    except ImportError:
        _tta_ = _tra_ = None
    if _tta_ is not None:
        _opmem(f"c={c} before tri_att_start", layout)
        _tta_(block.tri_att_start, z, layout, starting=True, **att_kw)
        _opmem(f"c={c} before tri_att_end", layout)
        _tta_(block.tri_att_end, z, layout, starting=False, **att_kw)
        _opmem(f"c={c} before transition", layout)
        _tra_(block.pair_transition, z, layout)
        _opmem(f"c={c} after transition", layout)
        return z
    upd = tp_triatt(block.tri_att_start, z, layout, starting=True, **att_kw)
    zadd_rows(z, upd, layout)
    del upd
    upd = tp_triatt(block.tri_att_end, z, layout, starting=False, **att_kw)
    zadd_rows(z, upd, layout)
    del upd
    upd = tp_transition(block.pair_transition, z)
    zadd_rows(z, upd, layout)
    del upd
    return z


# --------------------------------------------------------------------------- #
# MSABlock / MSAModule
# --------------------------------------------------------------------------- #
def tp_msa_block(
    block,
    m_shard: torch.Tensor,
    z_shard,
    layout: Layout,
    *,
    triangle_multiplicative: str = "torch",
    triangle_attention: str = "torch",
    inplace_safe: bool = False,
    chunk_size: Optional[int] = None,
):
    """MSABlock.forward (pairformer.py 659-679).  z += OPM(m) is streamed per stock-grid
    chunk into the shard with zadd (stock allocates z + opm; identical elementwise sum)."""
    if layout.replicated:
        return block(
            m_shard, z_shard, None,
            **_stock_pair_kwargs(triangle_multiplicative, triangle_attention, inplace_safe, chunk_size),
        )
    z = z_shard
    if (not inplace_safe) and torch.is_tensor(z):
        z = z.clone()  # keep the caller's tensor intact like stock's out-of-place z = z + opm
    # Communication
    for (l0, l1, piece) in tp_outer_product_mean_iter(
        block.outer_product_mean_msa, m_shard, layout, chunk_size=chunk_size, inplace_safe=inplace_safe
    ):
        zadd_rows(z, piece, layout, l0)
        del piece
    if not block.is_last_block:
        m_shard = tp_msa_stack(block.msa_stack, m_shard, z, layout)
    z = tp_pair_stack_block(
        block.pair_stack, z, layout,
        triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
        inplace_safe=True, chunk_size=chunk_size,  # z is already private here
    ) if inplace_safe else tp_pair_stack_block(
        block.pair_stack, z, layout,
        triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
        inplace_safe=False, chunk_size=chunk_size,
    )
    if not block.is_last_block:
        return m_shard, z
    return None, z  # stock: ensure m is not used after the last block


def tp_msa_module(
    msa_module,
    input_feature_dict: Dict[str, Any],
    z_shard,
    s_inputs: torch.Tensor,
    layout: Layout,
    *,
    triangle_multiplicative: str = "torch",
    triangle_attention: str = "torch",
    inplace_safe: bool = False,
    chunk_size: Optional[int] = None,
    strict_rng: bool = True,
):
    """Tensor-parallel MSAModule.forward (pairformer.py 809-916); returns z_shard.

    pair_mask: the stock trunk calls the MSA module with pair_mask=None (protenix.py);
    the pair-stack API carries no mask accordingly.  With inplace_safe=True the
    incoming shard is updated in place (the trunk only uses the returned z).
    """
    mod = msa_module
    if mod.n_blocks < 1:
        return z_shard
    if "msa" not in input_feature_dict:
        return z_shard
    if input_feature_dict["msa"].dim() < 2:
        return z_shard
    if layout.replicated:
        return mod(
            input_feature_dict, z_shard, s_inputs, None,
            **_stock_pair_kwargs(triangle_multiplicative, triangle_attention, inplace_safe, chunk_size),
        )
    if mod.training:
        raise NotImplementedError("ptx_tp.msa: inference only (call msa_module.eval())")
    m_shard, _ = tp_msa_embed(mod, input_feature_dict, s_inputs, layout, strict_rng=strict_rng)
    z = z_shard
    for block in mod.blocks:  # checkpoint_blocks with blocks_per_ckpt=None == sequential
        m_shard, z = tp_msa_block(
            block, m_shard, z, layout,
            triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
            inplace_safe=inplace_safe, chunk_size=chunk_size,
        )
    return z


# Re-exports (the driver calls ptx_tp.msa.tp_template_embedder); placed at module end to keep
# the ptx_tp.template -> ptx_tp.msa import acyclic at definition time.
from ptx_tp.template import (compact_template_features, template_embedder_is_active,  # noqa: E402
                             template_pair_rows, tp_template_embedder)

__all__ += ["tp_template_embedder", "template_embedder_is_active", "compact_template_features",
            "template_pair_rows"]
