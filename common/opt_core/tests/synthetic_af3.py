"""A tiny synthetic AF3-family model built ONLY from torch primitives, with the statement structure of an AF3-family cofold engine — the statements the
row-sharded (``--mode big --n_gpu P``) seams of ``opt_core.mem.rowpair`` distribute: pair init (outer sum + relpos one-hot rows + token bonds) -> recycling
(``z = z_init + linear(LN(z_prev))``) -> template embedder (T templates, per-template pair stack, mean / relu / linear, ``z +=``) -> MSA
module (outer-product-mean, pair-weighted averaging, MSA transition, pair block) -> Pairformer (pair block + attention-pair-bias +
single transition) -> distogram head -> confidence head (pair embedding from s_input / distances of the predicted coordinates,
pairformer blocks, PAE / PDE / pLDDT logits, AF3 summaries pTM / ipTM / per-chain tables) and diffusion (pair conditioning once per
rollout, single conditioning per step, a DiffusionTransformer of attention-pair-bias blocks with AdaLN, the EDM sampler loop).

Every module exposes its DENSE statement (``forward`` / ``*_dense``: whole ``[N, N, C]`` tensors, one call, no layout) and, for the
statements that are inherently pair-indexed constructions (outer sums / differences, one-hot rows of a pair feature), the ROW form
``*_rows(..., g0, g1)`` = the same per-element statements with the ``i`` operand sliced to global rows ``g0:g1`` (the form a kit's adapter
hands to the core as a row callable). Row-LOCAL statements (LayerNorm, Linear, transitions, gates) need no row form: the sub-module applied
to a row slab IS the row statement.

Shapes (no leading batch dim; fp32): s [N, c_s], s_input [N, c_s_in], z [N, N, c_z], m [S, N, c_m], template slab u [T, N, N, c_t],
coordinates x [n_samples, N, 3] (one atom per token: the atom-attention encoder/decoder of AF3 — replicated by design under sharding,
sequence-local, no N x N term — is not modelled). Random weights; the 'final' projections that AF3 zero-initialises are random here so
every path moves the outputs. Not part of the package.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================================================================================================================
# configuration / features
# =====================================================================================================================================
@dataclass
class Config:
    N: int = 37
    c_z: int = 16
    c_m: int = 8
    c_s: int = 12
    c_s_in: int = 10
    c_t: int = 8                 # template pair-stack channels
    T: int = 2                   # templates
    S_full: int = 7              # MSA rows available
    S: int = 5                   # MSA rows sampled per cycle
    f_msa: int = 6               # raw MSA feature width
    n_cycles: int = 2
    n_templ_blocks: int = 2
    n_msa_blocks: int = 2
    n_pair_blocks: int = 4       # Pairformer blocks
    n_conf_blocks: int = 2
    n_dit_blocks: int = 2
    n_steps: int = 3             # diffusion steps
    n_samples: int = 2
    relpos_k: int = 4            # clipped offset -> 2k+2 bins
    n_dgram_templ: int = 8       # template distogram bins
    n_dist_bins: int = 64        # distogram head bins
    n_contact_bins: int = 20     # contact prob = sum of the first bins of softmax(distogram)
    n_pae_bins: int = 64
    n_plddt_bins: int = 10
    n_conf_dbins: int = 8        # confidence pair-embedding distance bins
    H_tri: int = 2
    c_tri: int = 8
    c_mul: int = 16
    H_apb: int = 2
    c_apb: int = 6
    c_opm: int = 4
    H_pwa: int = 2
    c_pwa: int = 4
    c_a: int = 12                # diffusion token activation width
    H_dit: int = 2
    c_fourier: int = 8
    sigma_data: float = 16.0
    n_chains: int = 2
    eps: float = 1e-8
    inf: float = 1e9


def make_features(cfg: Config, seed: int) -> Dict[str, torch.Tensor]:
    """O(N) / O(T·N) / O(S·N) precursors only (plus the [N, N] token-bond matrix, an input): every pair FEATURE is built per row block from
    these by the model's ``*_rows`` statements (relpos rows, template feature rows) — no ``[N, N, c]`` feature tensor exists."""
    g = torch.Generator().manual_seed(int(seed))
    N = cfg.N
    sizes = [N // cfg.n_chains + (1 if i < N % cfg.n_chains else 0) for i in range(cfg.n_chains)]
    asym_id = torch.cat([torch.full((s,), i, dtype=torch.long) for i, s in enumerate(sizes)])
    residue_index = torch.cat([torch.arange(s) for s in sizes]).long()
    bonds = (torch.rand((N, N), generator=g) > 0.9).float()
    bonds = ((bonds + bonds.t()) > 0).float()
    feats = {
        "asym_id": asym_id,
        "residue_index": residue_index,
        "token_mask": torch.ones(N),
        "token_bonds": bonds,                                                      # [N, N] input (a kit keeps it on the host and slices rows)
        "s_input_feat": torch.randn((N, cfg.c_s_in), generator=g),
        "msa_feat": torch.randn((cfg.S_full, N, cfg.f_msa), generator=g),         # raw MSA (replicated / host)
        "msa_mask_full": (torch.rand((cfg.S_full, N), generator=g) > 0.1).float(),
        "templ_xyz": torch.randn((cfg.T, N, 3), generator=g) * 6.0,               # template pseudo-beta coordinates
        "templ_mask": (torch.rand((cfg.T, N), generator=g) > 0.2).float(),
        "templ_restype": torch.randn((cfg.T, N, 5), generator=g),
        "has_frame": (torch.rand((N,), generator=g) > 0.2),
        "x_init_noise_seed": torch.tensor(int(seed) * 7919 + 13),                  # the diffusion / MSA-sampling RNG stream seed (replicated)
    }
    return feats


def binned_one_hot(x: torch.Tensor, boundaries: torch.Tensor) -> torch.Tensor:
    """one-hot of the bin index ``sum(x[..., None] > boundaries)`` over ``len(boundaries)+1`` bins (fp32)."""
    idx = (x.unsqueeze(-1) > boundaries).sum(-1)
    return F.one_hot(idx, num_classes=int(boundaries.numel()) + 1).to(torch.float32)


# =====================================================================================================================================
# generic layers (AF3-family statement structure)
# =====================================================================================================================================
class Transition(nn.Module):
    """LayerNorm -> SwiGLU -> down projection, per position (row-local). ``forward(x, mask)``."""

    def __init__(self, c: int, n: int = 4):
        super().__init__()
        self.layer_norm = nn.LayerNorm(c)
        self.linear_1 = nn.Linear(c, n * c, bias=False)
        self.linear_2 = nn.Linear(c, n * c, bias=False)
        self.linear_out = nn.Linear(n * c, c, bias=False)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.layer_norm(x)
        h = self.linear_out(F.silu(self.linear_1(h)) * self.linear_2(h))
        if mask is not None:
            h = h * mask.unsqueeze(-1)
        return h


class TriangleMultiplication(nn.Module):
    """``x_ij = sum_k a_ik b_jk`` (outgoing) / ``sum_k a_ki b_kj`` (incoming) with LN_in, gated projections, mask, LN_out, output
    projection and output gate — the AF3-family triangle-multiplication statements. ``projections(z_rows, mask_rows)`` -> (a, b) rows
    ``[rows, N, c]``; ``epilogue(x_rows, z_rows)`` -> the update rows; ``forward(z, mask)`` = the dense statement (returns the update)."""

    def __init__(self, c_z: int, c: int, outgoing: bool):
        super().__init__()
        self.outgoing = outgoing
        self.layer_norm_in = nn.LayerNorm(c_z)
        self.linear_a_p = nn.Linear(c_z, c)
        self.linear_a_g = nn.Linear(c_z, c)
        self.linear_b_p = nn.Linear(c_z, c)
        self.linear_b_g = nn.Linear(c_z, c)
        self.layer_norm_out = nn.LayerNorm(c)
        self.linear_z = nn.Linear(c, c_z)
        self.linear_g = nn.Linear(c_z, c_z)

    def projections(self, z_rows: torch.Tensor, mask_rows: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.layer_norm_in(z_rows)
        m = mask_rows.unsqueeze(-1)
        a = torch.sigmoid(self.linear_a_g(h)) * self.linear_a_p(h) * m
        b = torch.sigmoid(self.linear_b_g(h)) * self.linear_b_p(h) * m
        return a, b

    def epilogue(self, x_rows: torch.Tensor, z_rows: torch.Tensor) -> torch.Tensor:
        g = torch.sigmoid(self.linear_g(self.layer_norm_in(z_rows)))
        return g * self.linear_z(self.layer_norm_out(x_rows))

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        a, b = self.projections(z, mask)
        if self.outgoing:
            x = torch.einsum("ikc,jkc->ijc", a, b)
        else:
            x = torch.einsum("kic,kjc->ijc", a, b)
        return self.epilogue(x, z)


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, biases: List[torch.Tensor]) -> torch.Tensor:
    """``softmax(q k^T + sum(biases)) v`` over the LAST key axis; q [*, H, Q, c] (pre-scaled), k / v [*, H, K, c] -> [*, H, Q, c] (fp32 softmax)."""
    logits = torch.matmul(q, k.transpose(-1, -2))
    for b in biases:
        logits = logits + b
    w = torch.softmax(logits, dim=-1)
    return torch.matmul(w, v)


class TriangleAttention(nn.Module):
    """Starting-node triangle attention around row i (the ENDING node is this module applied to the transposed pair; the pair block does the
    transpose). ``bias_rows(x_ln_rows)`` -> ``[rows, N, H]``;
    ``attn_rows(x_ln_rows, mask_rows, tb_full)`` = q/k/v/gate of those rows, logits ``q_ij·k_ik + tb[j, k, h]``, softmax over k, output
    projection -> ``[rows, N, c_z]``; ``forward(x, mask)`` = LN of the whole pair, bias from all rows, core over all rows (the update)."""

    def __init__(self, c_z: int, c: int, H: int, inf: float):
        super().__init__()
        self.c, self.H, self.inf = c, H, inf
        self.layer_norm = nn.LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, H, bias=False)          # triangle bias
        self.linear_q = nn.Linear(c_z, H * c, bias=False)
        self.linear_k = nn.Linear(c_z, H * c, bias=False)
        self.linear_v = nn.Linear(c_z, H * c, bias=False)
        self.linear_g = nn.Linear(c_z, H * c)
        self.linear_o = nn.Linear(H * c, c_z)

    def bias_rows(self, x_ln_rows: torch.Tensor) -> torch.Tensor:
        return self.linear_z(x_ln_rows)                                              # [rows, N, H]

    def attn_rows(self, x_ln_rows: torch.Tensor, mask_rows: torch.Tensor, tb_full: torch.Tensor) -> torch.Tensor:
        r, N = int(x_ln_rows.shape[0]), int(x_ln_rows.shape[1])
        H, c = self.H, self.c

        def heads(t):                                                                # [rows, N, H*c] -> [rows, H, N, c]
            return t.view(r, N, H, c).transpose(-2, -3)
        q = heads(self.linear_q(x_ln_rows)) * (1.0 / math.sqrt(c))
        k = heads(self.linear_k(x_ln_rows))
        v = heads(self.linear_v(x_ln_rows))
        mask_bias = (self.inf * (mask_rows - 1.0))[:, None, None, :]                 # [rows, 1, 1, N(k)]
        triangle_bias = tb_full.permute(2, 0, 1).unsqueeze(0)                        # [1, H, N(j), N(k)]
        o = attention_core(q, k, v, [mask_bias, triangle_bias])                      # [rows, H, N(j), c]
        o = o.transpose(-2, -3).reshape(r, N, H * c)
        o = torch.sigmoid(self.linear_g(x_ln_rows)) * o
        return self.linear_o(o)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        x_ln = self.layer_norm(x)
        tb = self.bias_rows(x_ln)                                                    # [N, N, H] from ALL rows
        return self.attn_rows(x_ln, mask, tb)


class PairBlock(nn.Module):
    """AF3-family pair block: ``z += tri_mul_out(z); z += tri_mul_in(z); z += tri_att_start(z); zT = z^T; zT += tri_att_end(zT); z = zT^T;
    z += transition(z)`` (dropout = identity at inference). ``forward(z, mask)`` -> new z (dense)."""

    def __init__(self, c_z: int, cfg: Config):
        super().__init__()
        self.tri_mul_out = TriangleMultiplication(c_z, cfg.c_mul, outgoing=True)
        self.tri_mul_in = TriangleMultiplication(c_z, cfg.c_mul, outgoing=False)
        self.tri_att_start = TriangleAttention(c_z, cfg.c_tri, cfg.H_tri, cfg.inf)
        self.tri_att_end = TriangleAttention(c_z, cfg.c_tri, cfg.H_tri, cfg.inf)
        self.pair_transition = Transition(c_z)

    def forward(self, z: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        z = z + self.tri_mul_out(z, mask)
        z = z + self.tri_mul_in(z, mask)
        z = z + self.tri_att_start(z, mask)
        zT = z.transpose(0, 1).contiguous()
        zT = zT + self.tri_att_end(zT, mask.transpose(0, 1))
        z = zT.transpose(0, 1).contiguous()
        z = z + self.pair_transition(z, mask)
        return z


class AdaLN(nn.Module):
    """AF3 AdaLN: ``sigmoid(linear_s(LN_s(s))) * LN_a(a) + linear_b(LN_s(s))`` (row-local over tokens)."""

    def __init__(self, c_a: int, c_s: int):
        super().__init__()
        self.ln_a = nn.LayerNorm(c_a, elementwise_affine=False)
        self.ln_s = nn.LayerNorm(c_s, bias=False)
        self.linear_s = nn.Linear(c_s, c_a)
        self.linear_b = nn.Linear(c_s, c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        s = self.ln_s(s)
        return torch.sigmoid(self.linear_s(s)) * self.ln_a(a) + self.linear_b(s)


class AttentionPairBias(nn.Module):
    """Attention over TOKENS with a bias from the pair representation (Pairformer: plain LN; DiffusionTransformer: AdaLN on the condition
    ``s`` + the ada output gate). ``prep(a, s)`` -> (a_ln, q, k, v) on ALL tokens (every per-token linear keeps M = N);
    ``bias_rows(z_rows)`` -> ``[H, rows, N]``; ``core_rows(q_rows, k, v, bias_rows, mask)`` -> ``[rows, H*c]``; ``wrap_up(o, a_ln, s)`` = gate +
    output projection (+ ada gate). ``forward(a, z, mask, s)`` = the dense statement (the update)."""

    def __init__(self, c_a: int, c_z: int, H: int, c: int, inf: float, c_s: Optional[int] = None):
        super().__init__()
        self.H, self.c, self.inf, self.ada = H, c, inf, c_s is not None
        self.layer_norm_a = AdaLN(c_a, c_s) if self.ada else nn.LayerNorm(c_a)
        self.layer_norm_z = nn.LayerNorm(c_z)
        self.linear_z = nn.Linear(c_z, H, bias=False)
        self.linear_q = nn.Linear(c_a, H * c)
        self.linear_k = nn.Linear(c_a, H * c, bias=False)
        self.linear_v = nn.Linear(c_a, H * c, bias=False)
        self.linear_g = nn.Linear(c_a, H * c, bias=False)
        self.linear_o = nn.Linear(H * c, c_a, bias=False)
        if self.ada:
            self.linear_ada_out = nn.Linear(c_s, c_a)
            with torch.no_grad():
                self.linear_ada_out.bias.fill_(-2.0)

    def prep(self, a: torch.Tensor, s: Optional[torch.Tensor] = None):
        a_ln = self.layer_norm_a(a, s) if self.ada else self.layer_norm_a(a)
        N = int(a_ln.shape[0])

        def heads(t):                                                                # [N, H*c] -> [H, N, c]
            return t.view(N, self.H, self.c).transpose(0, 1)
        q = heads(self.linear_q(a_ln)) * (1.0 / math.sqrt(self.c))
        k = heads(self.linear_k(a_ln))
        v = heads(self.linear_v(a_ln))
        return a_ln, q, k, v

    def bias_rows(self, z_rows: torch.Tensor) -> torch.Tensor:
        return self.linear_z(self.layer_norm_z(z_rows)).permute(2, 0, 1)             # [H, rows, N]

    def core_rows(self, q_rows: torch.Tensor, k: torch.Tensor, v: torch.Tensor, bias_rows: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        mask_bias = (self.inf * (mask - 1.0))[None, None, :]                         # [1, 1, N(k)]
        o = attention_core(q_rows, k, v, [mask_bias, bias_rows])                     # [H, rows, c]
        r = int(q_rows.shape[1])
        return o.transpose(0, 1).reshape(r, self.H * self.c)                         # [rows, H*c]

    def wrap_up(self, o: torch.Tensor, a_ln: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        o = self.linear_o(torch.sigmoid(self.linear_g(a_ln)) * o)
        if self.ada:
            o = torch.sigmoid(self.linear_ada_out(s)) * o
        return o

    def forward(self, a: torch.Tensor, z: torch.Tensor, mask: torch.Tensor, s: Optional[torch.Tensor] = None) -> torch.Tensor:
        a_ln, q, k, v = self.prep(a, s)
        o = self.core_rows(q, k, v, self.bias_rows(z), mask)
        return self.wrap_up(o, a_ln, s)


class PairformerBlock(nn.Module):
    """``z = pair_block(z); s += apb(s, z); s += transition(s)``."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.pair = PairBlock(cfg.c_z, cfg)
        self.attn_pair_bias = AttentionPairBias(cfg.c_s, cfg.c_z, cfg.H_apb, cfg.c_apb, cfg.inf)
        self.single_transition = Transition(cfg.c_s)

    def forward(self, s, z, single_mask, pair_mask):
        z = self.pair(z, pair_mask)
        s = s + self.attn_pair_bias(s, z, single_mask)
        s = s + self.single_transition(s, single_mask)
        return s, z


# =====================================================================================================================================
# trunk pieces
# =====================================================================================================================================
class PairInit(nn.Module):
    """InputEmbedder pair init: ``z_ij = linear_i(s_input)_i + linear_j(s_input)_j + linear_relpos(relpos_ij) + linear_bonds(bonds_ij)``.
    ``single(feats)`` -> (s_input, s, emb_i, emb_j) replicated; ``rows(feats, emb_i, emb_j, g0, g1)`` -> ``z_init[g0:g1]`` (the one-hot
    relpos slab exists only for the row block); ``dense(feats)`` -> whole z_init."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.k = cfg.relpos_k
        self.linear_s_in = nn.Linear(cfg.c_s_in, cfg.c_s_in)
        self.linear_s = nn.Linear(cfg.c_s_in, cfg.c_s)
        self.linear_z_i = nn.Linear(cfg.c_s_in, cfg.c_z)
        self.linear_z_j = nn.Linear(cfg.c_s_in, cfg.c_z)
        self.linear_relpos = nn.Linear((2 * cfg.relpos_k + 2) + 1, cfg.c_z)
        self.linear_bonds = nn.Linear(1, cfg.c_z)

    def single(self, feats):
        s_input = torch.relu(self.linear_s_in(feats["s_input_feat"]))
        return s_input, self.linear_s(s_input), self.linear_z_i(s_input), self.linear_z_j(s_input)

    def relpos_rows(self, feats, g0: int, g1: int) -> torch.Tensor:
        res, asym = feats["residue_index"], feats["asym_id"]
        same_chain = asym[g0:g1, None] == asym[None, :]
        offset = res[g0:g1, None] - res[None, :]
        clipped = torch.clamp(offset + self.k, min=0, max=2 * self.k)
        final = torch.where(same_chain, clipped, (2 * self.k + 1) * torch.ones_like(clipped))
        boundaries = torch.arange(0, 2 * self.k + 1, device=res.device)
        rel_pos = binned_one_hot(final, boundaries)                                  # [rows, N, 2k+2]
        return torch.cat([rel_pos, same_chain[..., None].to(rel_pos.dtype)], dim=-1)

    def rows(self, feats, emb_i, emb_j, g0: int, g1: int) -> torch.Tensor:
        bonds_emb = self.linear_bonds(feats["token_bonds"][g0:g1, :].unsqueeze(-1))
        z = emb_i[g0:g1, :].unsqueeze(-2) + emb_j.unsqueeze(-3)
        z = z + self.linear_relpos(self.relpos_rows(feats, g0, g1))
        z = z + bonds_emb
        return z

    def dense(self, feats):
        s_input, s, emb_i, emb_j = self.single(feats)
        return s_input, s, self.rows(feats, emb_i, emb_j, 0, int(emb_i.shape[0]))


class TemplateEmbedder(nn.Module):
    """``u_t = linear_z(LN_z(z)) + embed(templ feats_t)`` per template, a template pair stack per template, final LN, ``t = sum_t u_t / T``,
    relu, ``linear_t``; the caller adds ``t`` to z. Template pair FEATURES are built per row block from ``[T, N]`` precursors
    (``feat_rows``: distogram one-hot of template distances, mask products, restype outer concat — a template featuriser's per-row statements)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        f_t = cfg.n_dgram_templ + 1 + 2 * 5
        self.layer_norm_z = nn.LayerNorm(cfg.c_z)
        self.linear_z = nn.Linear(cfg.c_z, cfg.c_t, bias=False)
        self.linear_a = nn.Linear(f_t, cfg.c_t)
        self.pair_stack = nn.ModuleList([PairBlock(cfg.c_t, cfg) for _ in range(cfg.n_templ_blocks)])
        self.layer_norm_out = nn.LayerNorm(cfg.c_t)
        self.linear_t = nn.Linear(cfg.c_t, cfg.c_z)
        lower = torch.linspace(2.0, 12.0, cfg.n_dgram_templ - 1)
        self.register_buffer("dgram_bounds", lower ** 2, persistent=False)

    def feat_rows(self, feats, slots: List[int], g0: int, g1: int) -> torch.Tensor:
        xyz = feats["templ_xyz"][slots]                                              # [S, N, 3]
        diff = xyz[:, g0:g1, None, :] - xyz[:, None, :, :]
        d2 = (diff ** 2).sum(-1)                                                     # [S, rows, N]
        dgram = binned_one_hot(d2, self.dgram_bounds)                                # [S, rows, N, bins]
        tm = feats["templ_mask"][slots]
        pair_mask = (tm[:, g0:g1, None] * tm[:, None, :])[..., None]                 # [S, rows, N, 1]
        same_chain = (feats["asym_id"][g0:g1, None] == feats["asym_id"][None, :])[None, :, :, None].to(dgram.dtype)
        rt = feats["templ_restype"][slots]                                           # [S, N, 5]
        rt_i = rt[:, g0:g1, None, :].expand(-1, -1, int(rt.shape[1]), -1)
        rt_j = rt[:, None, :, :].expand(-1, g1 - g0, -1, -1)
        return torch.cat([dgram * pair_mask * same_chain, pair_mask, rt_i, rt_j], dim=-1)

    def u_rows(self, feats, z_rows: torch.Tensor, slots: List[int], g0: int, g1: int) -> torch.Tensor:
        """Alg.16 lines 1-5 + 8 for rows g0:g1 of slots: ``[S, rows, N, c_t]``."""
        return self.linear_z(self.layer_norm_z(z_rows)).unsqueeze(0) + self.linear_a(self.feat_rows(feats, slots, g0, g1))

    def closing_rows(self, u_rows_all_slots: torch.Tensor) -> torch.Tensor:
        """``[T, rows, N, c_t]`` (after the stack's final LN) -> ``linear_t(relu(sum_t / T))`` rows ``[rows, N, c_z]``."""
        t = torch.sum(u_rows_all_slots, dim=0) / self.cfg.T
        return self.linear_t(torch.relu(t))

    def forward(self, feats, z: torch.Tensor, pair_mask: torch.Tensor) -> torch.Tensor:
        N = int(z.shape[0])
        us = []
        for t in range(self.cfg.T):
            u = self.u_rows(feats, z, [t], 0, N)[0]
            for blk in self.pair_stack:
                u = blk(u, pair_mask)
            us.append(self.layer_norm_out(u))
        return self.closing_rows(torch.stack(us, dim=0))


class MSAEmbedder(nn.Module):
    """Per-cycle random MSA subsample (``S`` of ``S_full`` rows, drawn from the replicated generator) -> ``m = linear_m(msa_feat[idx]) +
    linear_s(s_input)``; returns (m, msa_mask). REPLICATED by design under sharding (identical RNG stream on all ranks,
    proven by checksum)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.linear_m = nn.Linear(cfg.f_msa, cfg.c_m)
        self.linear_s = nn.Linear(cfg.c_s_in, cfg.c_m)

    def forward(self, feats, s_input, gen: torch.Generator):
        idx = torch.randperm(self.cfg.S_full, generator=gen)[: self.cfg.S]
        m = self.linear_m(feats["msa_feat"][idx]) + self.linear_s(s_input).unsqueeze(0)
        return m, feats["msa_mask_full"][idx]


class OuterProductMean(nn.Module):
    """``operands(m, msa_mask)`` -> (a, b) ``[S, N, c]`` (LN + masked projections); ``outer_rows(a_rows, b, norm_rows)`` -> ``[rows, N, c_z]``
    (einsum over S, flatten, linear_out, / (norm + eps)); ``norm_rows(msa_mask, g0, g1)`` -> ``[rows, N, 1]``; ``forward`` = dense (the update)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.eps = 1e-3
        self.layer_norm = nn.LayerNorm(cfg.c_m)
        self.linear_1 = nn.Linear(cfg.c_m, cfg.c_opm)
        self.linear_2 = nn.Linear(cfg.c_m, cfg.c_opm)
        self.linear_out = nn.Linear(cfg.c_opm * cfg.c_opm, cfg.c_z)

    def operands(self, m, msa_mask):
        ln = self.layer_norm(m)
        mk = msa_mask.unsqueeze(-1)
        return self.linear_1(ln) * mk, self.linear_2(ln) * mk

    def norm_rows(self, msa_mask, g0: int, g1: int):
        return torch.einsum("si,sj->ij", msa_mask[:, g0:g1], msa_mask).unsqueeze(-1)

    def outer_rows(self, a_rows, b, norm_rows):
        outer = torch.einsum("sic,sjd->ijcd", a_rows, b)
        outer = self.linear_out(outer.reshape(outer.shape[0], outer.shape[1], -1))
        return outer / (norm_rows + self.eps)

    def forward(self, m, msa_mask):
        a, b = self.operands(m, msa_mask)
        return self.outer_rows(a, b, self.norm_rows(msa_mask, 0, int(m.shape[1])))


class MSAPairWeightedAveraging(nn.Module):
    """``v = linear_v(LN_m(m))`` / gate on ALL tokens (M = S·N); weights ``w[i, j, h] = softmax_j(linear_z(LN_z(z_ij)) + mask
    bias)`` need whole ROWS i (row-local); ``o[s, i] = sum_j w[i, j] v[s, j]``; ``rows_fn(m, z_rows, g0, g1, pair_mask)`` -> the gated
    per-row outputs ``[S, rows, H*c]`` BEFORE ``linear_o`` (which runs on the row-gathered slab, M = S·N); ``forward`` = dense update."""
    def __init__(self, cfg: Config):
        super().__init__()
        self.H, self.c, self.inf = cfg.H_pwa, cfg.c_pwa, cfg.inf
        self.layer_norm_m = nn.LayerNorm(cfg.c_m)
        self.layer_norm_z = nn.LayerNorm(cfg.c_z)
        self.linear_z = nn.Linear(cfg.c_z, cfg.H_pwa, bias=False)
        self.linear_v = nn.Linear(cfg.c_m, cfg.H_pwa * cfg.c_pwa, bias=False)
        self.linear_g = nn.Linear(cfg.c_m, cfg.H_pwa * cfg.c_pwa, bias=False)
        self.linear_o = nn.Linear(cfg.H_pwa * cfg.c_pwa, cfg.c_m, bias=False)

    def rows_fn(self, m, z_rows, g0: int, g1: int, pair_mask):
        S, N = int(m.shape[0]), int(m.shape[1])
        m_ln = self.layer_norm_m(m)
        v = self.linear_v(m_ln).view(S, N, self.H, self.c).permute(2, 0, 1, 3)      # [H, S, N(j), c]
        b = self.linear_z(self.layer_norm_z(z_rows)).permute(2, 0, 1)                # [H, rows, N]
        b = b + (self.inf * (pair_mask[g0:g1] - 1.0))[None]
        w = torch.softmax(b, dim=-1)                                                 # [H, rows, N(j)]
        o = torch.einsum("hij,hsjc->sihc", w, v)                                     # [S, rows, H, c]
        g = torch.sigmoid(self.linear_g(m_ln)).view(S, N, self.H, self.c)[:, g0:g1]
        return (o * g).reshape(S, g1 - g0, self.H * self.c)

    def forward(self, m, z, pair_mask):
        return self.linear_o(self.rows_fn(m, z, 0, int(m.shape[1]), pair_mask))


class MSABlock(nn.Module):
    """AF3-family MSA-module block: ``z += opm(m); m += pwa(m, z); m += transition(m); z = pair_block(z)``."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.outer_product_mean = OuterProductMean(cfg)
        self.msa_att_row = MSAPairWeightedAveraging(cfg)
        self.msa_transition = Transition(cfg.c_m)
        self.pair = PairBlock(cfg.c_z, cfg)

    def forward(self, m, z, msa_mask, pair_mask):
        z = z + self.outer_product_mean(m, msa_mask)
        m = m + self.msa_att_row(m, z, pair_mask)
        m = m + self.msa_transition(m, msa_mask)
        z = self.pair(z, pair_mask)
        return m, z


# =====================================================================================================================================
# heads
# =====================================================================================================================================
class DistogramHead(nn.Module):
    """``logits = linear(z); logits = logits + logits^T`` -> ``[N, N, bins]``; contact probability = the first ``n_contact_bins`` of softmax."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.linear = nn.Linear(cfg.c_z, cfg.n_dist_bins)

    def forward(self, z):
        logits = self.linear(z)
        return logits + logits.transpose(0, 1)

    def contact_from_logits(self, logits_rows):
        return torch.softmax(logits_rows, dim=-1)[..., : self.cfg.n_contact_bins].sum(-1)


class ConfidenceHead(nn.Module):
    """AF3 confidence head per diffusion sample: ``z_c = z_trunk + linear_i(s_input)_i + linear_j(s_input)_j + linear_d(one_hot(d_ij(x)))``
    (``embed_rows``), ``s_c = s_trunk + linear_s(s_input)``, ``n_conf_blocks`` Pairformer blocks, then ``pae = linear_pae(z_c)``,
    ``pde = linear_pde(z_c) + linear_pde(z_c)^T`` (rows: ``pde_rows(z_rows, zT_rows)``), ``plddt = linear_plddt(s_c)``. Summaries
    (pTM / ipTM / per-chain / gPDE tables) are the AF3 statements over the full ``[N, N]`` expected-value matrices (``summaries_dense``)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.linear_i = nn.Linear(cfg.c_s_in, cfg.c_z, bias=False)
        self.linear_j = nn.Linear(cfg.c_s_in, cfg.c_z, bias=False)
        self.linear_d = nn.Linear(cfg.n_conf_dbins, cfg.c_z, bias=False)
        self.linear_s = nn.Linear(cfg.c_s_in, cfg.c_s)
        self.blocks = nn.ModuleList([PairformerBlock(cfg) for _ in range(cfg.n_conf_blocks)])
        self.linear_pae = nn.Linear(cfg.c_z, cfg.n_pae_bins)
        self.linear_pde = nn.Linear(cfg.c_z, cfg.n_pae_bins)
        self.linear_plddt = nn.Linear(cfg.c_s, cfg.n_plddt_bins)
        self.register_buffer("d_bounds", torch.linspace(3.0, 60.0, cfg.n_conf_dbins - 1), persistent=False)

    def embed_rows(self, s_input, z_rows, x, g0: int, g1: int):
        """``z_c[g0:g1]`` for ONE sample's coordinates ``x [N, 3]``."""
        d = torch.sqrt(((x[g0:g1, None, :] - x[None, :, :]) ** 2).sum(-1) + self.cfg.eps)
        z = z_rows + self.linear_i(s_input)[g0:g1, None, :] + self.linear_j(s_input)[None, :, :]
        return z + self.linear_d(binned_one_hot(d, self.d_bounds))

    def single(self, s_input, s_trunk):
        return s_trunk + self.linear_s(s_input)

    def pae_rows(self, z_rows):
        return self.linear_pae(z_rows)

    def pde_rows(self, z_rows, zT_rows):
        return self.linear_pde(z_rows) + self.linear_pde(zT_rows)

    def forward(self, s_input, s_trunk, z_trunk, x, single_mask, pair_mask):
        """dense, ONE sample: -> (pae_logits [N,N,b], pde_logits [N,N,b], plddt_logits [N, b_plddt])."""
        N = int(z_trunk.shape[0])
        z = self.embed_rows(s_input, z_trunk, x, 0, N)
        s = self.single(s_input, s_trunk)
        for blk in self.blocks:
            s, z = blk(s, z, single_mask, pair_mask)
        return self.pae_rows(z), self.pde_rows(z, z.transpose(0, 1)), self.linear_plddt(s)


def tm_bin_centers(min_bin: float, max_bin: float, no_bins: int) -> torch.Tensor:
    width = (max_bin - min_bin) / no_bins
    return torch.arange(min_bin + 0.5 * width, max_bin, width)


def summaries_dense(pae_logits, pde_logits, contact, asym_id, has_frame, eps=1e-8) -> Dict[str, torch.Tensor]:
    """The AF3-family confidence summaries on FULL matrices — the independent single-device statements
    (``tests/test_rowpair_logic.py::_dense_confidence_reference``: ptm, iptm, chain_ptm ``[1,C]``, chain_pair_iptm ``[1,C,C]``, chain_iptm,
    gpde, chain_gpde, chain_pair_gpde, token_pair_pde_f32 ``[1,N,N]``) plus the full expected-PAE matrix ``token_pair_pae [N,N]``."""
    from tests.test_rowpair_logic import _dense_confidence_reference
    out = _dense_confidence_reference(pae_logits, pde_logits, contact, asym_id.long(), has_frame.bool(),
                                      torch.zeros_like(asym_id, dtype=torch.bool), eps=eps)
    out["token_pair_pde"] = out.pop("token_pair_pde_f32")[0]
    centers = tm_bin_centers(0.0, 32.0, int(pae_logits.shape[-1])).to(pae_logits.device)
    out["token_pair_pae"] = (torch.softmax(pae_logits.unsqueeze(0), -1) @ centers)[0]
    return out


# =====================================================================================================================================
# diffusion
# =====================================================================================================================================
class DiffusionConditioning(nn.Module):
    """pair path (noise-independent -> once per rollout): ``z_cond = linear_z(LN(cat[z_trunk, relpos])) ; z_cond += transition_1(z_cond);
    z_cond += transition_2(z_cond)`` — ``pair_rows(feats, z_rows, g0, g1)``; single path (per step, replicated): ``s_cond = linear_s(LN(cat[
    s_trunk, s_input])) + linear_n(fourier(t)) ; += transition``."""

    def __init__(self, cfg: Config, pair_init: PairInit):
        super().__init__()
        self.cfg = cfg
        self._relpos = pair_init                                                      # the relpos statement is shared with the trunk
        f_rel = (2 * cfg.relpos_k + 2) + 1
        self.layer_norm_z = nn.LayerNorm(cfg.c_z + f_rel)
        self.linear_z = nn.Linear(cfg.c_z + f_rel, cfg.c_z, bias=False)
        self.trans_z = nn.ModuleList([Transition(cfg.c_z, 2), Transition(cfg.c_z, 2)])
        self.layer_norm_s = nn.LayerNorm(cfg.c_s + cfg.c_s_in)
        self.linear_s = nn.Linear(cfg.c_s + cfg.c_s_in, cfg.c_s, bias=False)
        self.register_buffer("fourier_w", torch.randn(cfg.c_fourier) * 2.0, persistent=True)
        self.register_buffer("fourier_b", torch.rand(cfg.c_fourier), persistent=True)
        self.layer_norm_n = nn.LayerNorm(cfg.c_fourier)
        self.linear_n = nn.Linear(cfg.c_fourier, cfg.c_s, bias=False)
        self.trans_s = Transition(cfg.c_s, 2)

    def pair_rows(self, feats, z_rows, g0: int, g1: int):
        rel = self._relpos.relpos_rows(feats, g0, g1)
        z = self.linear_z(self.layer_norm_z(torch.cat([z_rows, rel], dim=-1)))
        for tr in self.trans_z:
            z = z + tr(z)
        return z

    def single(self, feats, s_trunk, s_input, t_hat: torch.Tensor):
        s = self.linear_s(self.layer_norm_s(torch.cat([s_trunk, s_input], dim=-1)))
        c_noise = 0.25 * torch.log(t_hat / self.cfg.sigma_data)
        n = torch.cos(2.0 * math.pi * (c_noise * self.fourier_w + self.fourier_b))
        s = s + self.linear_n(self.layer_norm_n(n)).unsqueeze(0)
        return s + self.trans_s(s)


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.attn = AttentionPairBias(cfg.c_a, cfg.c_z, cfg.H_dit, cfg.c_apb, cfg.inf, c_s=cfg.c_s)
        self.ada_ln = AdaLN(cfg.c_a, cfg.c_s)
        self.transition = Transition(cfg.c_a, 2)
        self.linear_ada_out = nn.Linear(cfg.c_s, cfg.c_a)

    def transition_rows(self, a_rows, s_rows):
        """conditioned transition (row-local over tokens): ``sigmoid(linear(s)) * transition(AdaLN(a, s))``."""
        h = self.ada_ln(a_rows, s_rows)
        h = self.transition.linear_out(F.silu(self.transition.linear_1(h)) * self.transition.linear_2(h))
        return torch.sigmoid(self.linear_ada_out(s_rows)) * h

    def forward(self, a, s, z, mask):
        a = a + self.attn(a, z, mask, s)
        return a + self.transition_rows(a, s)


class DiffusionModule(nn.Module):
    """one denoising call: token activations ``a = linear_x(x_scaled) + linear_s(s_cond)``, the DiffusionTransformer, ``x_update =
    linear_out(LN(a))``, EDM skip scaling. ``forward(x_noisy [N,3], t_hat, s_cond, z_cond, mask)`` -> denoised ``[N, 3]`` (one sample)."""

    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.linear_x = nn.Linear(3, cfg.c_a)
        self.linear_s = nn.Linear(cfg.c_s, cfg.c_a, bias=False)
        self.layer_norm_s = nn.LayerNorm(cfg.c_s)
        self.blocks = nn.ModuleList([DiffusionTransformerBlock(cfg) for _ in range(cfg.n_dit_blocks)])
        self.layer_norm_out = nn.LayerNorm(cfg.c_a)
        self.linear_out = nn.Linear(cfg.c_a, 3, bias=False)

    def token_act(self, x_noisy, t_hat, s_cond):
        sd = self.cfg.sigma_data
        r = x_noisy / torch.sqrt(t_hat ** 2 + sd ** 2)
        return self.linear_x(r) + self.linear_s(self.layer_norm_s(s_cond))

    def finish(self, a, x_noisy, t_hat):
        sd = self.cfg.sigma_data
        upd = self.linear_out(self.layer_norm_out(a))
        return (sd ** 2 / (sd ** 2 + t_hat ** 2)) * x_noisy + (sd * t_hat / torch.sqrt(sd ** 2 + t_hat ** 2)) * upd

    def forward(self, x_noisy, t_hat, s_cond, z_cond, mask):
        a = self.token_act(x_noisy, t_hat, s_cond)
        for blk in self.blocks:
            a = blk(a, s_cond, z_cond, mask)
        return self.finish(a, x_noisy, t_hat)


def noise_schedule(cfg: Config) -> torch.Tensor:
    smax, smin, rho = 3.0, 4e-4, 7.0
    t = torch.linspace(0, 1, cfg.n_steps + 1)
    return cfg.sigma_data * (smax ** (1 / rho) + t * (smin ** (1 / rho) - smax ** (1 / rho))) ** rho


def sampler_dense(cfg: Config, dmod: DiffusionModule, cond: DiffusionConditioning, feats, s_input, s_trunk, z_trunk, mask, gen):
    """AF3 Alg. 18 (no augmentation): per sample, x ~ sigma_0·N(0,1); per step: ``x_noisy = x + sqrt(t_hat^2 - c_prev^2)·eps``, denoise,
    Euler update. z_cond computed once per rollout (noise-independent). Returns ``[n_samples, N, 3]``."""
    sig = noise_schedule(cfg)
    N = int(s_trunk.shape[0])
    z_cond = cond.pair_rows(feats, z_trunk, 0, N)
    xs = []
    for _ in range(cfg.n_samples):
        x = sig[0] * torch.randn((N, 3), generator=gen)
        for i in range(1, cfg.n_steps + 1):
            c_prev, c = sig[i - 1], sig[i]
            gamma = 0.4 if c > 1.0 else 0.0
            t_hat = c_prev * (gamma + 1.0)
            eps = 1.003 * torch.sqrt((t_hat ** 2 - c_prev ** 2).clamp(min=0)) * torch.randn((N, 3), generator=gen)
            x_noisy = x + eps
            s_cond = cond.single(feats, s_trunk, s_input, t_hat)
            x_den = dmod(x_noisy, t_hat, s_cond, z_cond, mask)
            delta = (x_noisy - x_den) / t_hat
            x = x_noisy + 1.5 * (c - t_hat) * delta
        xs.append(x)
    return torch.stack(xs, 0)


# =====================================================================================================================================
# the model
# =====================================================================================================================================
class SyntheticAF3(nn.Module):
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.pair_init = PairInit(cfg)
        self.layer_norm_z = nn.LayerNorm(cfg.c_z)           # recycling
        self.linear_z = nn.Linear(cfg.c_z, cfg.c_z)
        self.layer_norm_s = nn.LayerNorm(cfg.c_s)
        self.linear_s = nn.Linear(cfg.c_s, cfg.c_s)
        self.template_embedder = TemplateEmbedder(cfg)
        self.msa_embedder = MSAEmbedder(cfg)
        self.msa_blocks = nn.ModuleList([MSABlock(cfg) for _ in range(cfg.n_msa_blocks)])
        self.pairformer = nn.ModuleList([PairformerBlock(cfg) for _ in range(cfg.n_pair_blocks)])
        self.distogram = DistogramHead(cfg)
        self.confidence = ConfidenceHead(cfg)
        self.cond = DiffusionConditioning(cfg, self.pair_init)
        self.diffusion = DiffusionModule(cfg)

    def recycle(self, z_prev_rows):
        """``linear_z(LN_z(z_prev))`` (the caller adds z_init)."""
        return self.linear_z(self.layer_norm_z(z_prev_rows))


def build_model(cfg: Config, seed: int) -> SyntheticAF3:
    """Deterministic random weights (CPU init from ``torch.manual_seed`` is reproducible across processes); LayerNorm affine parameters and
    every bias randomised so no statement is a no-op."""
    torch.manual_seed(int(seed))
    model = SyntheticAF3(cfg)
    g = torch.Generator().manual_seed(int(seed) + 1)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if p.dim() == 1:                                   # LN weights / biases, linear biases
                p.add_(0.1 * torch.randn(p.shape, generator=g))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def rng(feats) -> torch.Generator:
    """the replicated RNG stream (MSA subsample per cycle, then diffusion noise): identically seeded on every rank."""
    return torch.Generator().manual_seed(int(feats["x_init_noise_seed"]))


@torch.no_grad()
def forward_dense(model: SyntheticAF3, feats) -> Dict[str, torch.Tensor]:
    """The single-device statement of the whole model (the reference)."""
    cfg = model.cfg
    gen = rng(feats)
    N = cfg.N
    token_mask = feats["token_mask"]
    pair_mask = token_mask[:, None] * token_mask[None, :]
    s_input, s_init, z_init = model.pair_init.dense(feats)
    s = torch.zeros_like(s_init)
    z = torch.zeros_like(z_init)
    for _cycle in range(cfg.n_cycles):
        z = z_init + model.recycle(z)
        z = z + model.template_embedder(feats, z, pair_mask)
        m, msa_mask = model.msa_embedder(feats, s_input, gen)
        for blk in model.msa_blocks:
            m, z = blk(m, z, msa_mask, pair_mask)
        s = s_init + model.linear_s(model.layer_norm_s(s))
        for blk in model.pairformer:
            s, z = blk(s, z, token_mask, pair_mask)
    out = {"s_trunk": s, "z_trunk": z}
    dist_logits = model.distogram(z)
    contact = model.distogram.contact_from_logits(dist_logits)
    out["distogram_logits"], out["contact_probs"] = dist_logits, contact
    x = sampler_dense(cfg, model.diffusion, model.cond, feats, s_input, s, z, token_mask, gen)
    out["coords"] = x
    for si in range(cfg.n_samples):
        pae, pde, plddt = model.confidence(s_input, s, z, x[si], token_mask, pair_mask)
        summ = summaries_dense(pae, pde, contact, feats["asym_id"], feats["has_frame"])
        out[f"plddt_logits.{si}"] = plddt
        out[f"pae_logits.{si}"] = pae
        for k, v in summ.items():
            out[f"{k}.{si}"] = v
    return out


if __name__ == "__main__":
    cfg = Config()
    model = build_model(cfg, 0)
    feats = make_features(cfg, 0)
    out = forward_dense(model, feats)
    for k, v in out.items():
        print(k, tuple(v.shape), float(v.float().abs().mean()))
