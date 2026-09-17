"""A miniature torch engine with OpenDDE 1.0.0's ATTRIBUTE NAMES and DENSE statements (the ones opendde_opt.tp re-states on row shards), for
the CPU seam tests: `install_modules()` registers stub modules under the stock's module paths (`opendde.model.opendde`,
`opendde.model.modules.pairformer`, `opendde.model.modules.confidence`, `opendde.model.utils`, `opendde.model.sample_confidence`) so tp's
import-time patches and lazy imports resolve to them; `build(seed)` returns a deterministic (model, feats, coords). The dense forwards here
are the reference the sharded run is compared against (same weights, same statements, whole tensors). Nothing of the real stock is imported.
Dims are tiny (N=40, c_z=8): the point is the statement algebra and the seam census, not speed."""
from __future__ import annotations

import math
import sys
import types

import torch
from torch import nn

N, C_Z, C_S, C_SIN, C_M, S_MSA, T_TEMPL, C_T, N_CLS, DBINS, CBINS, PBINS, ATOMS_PER_TOK, C_RELP = 40, 8, 6, 5, 4, 6, 2, 7, 5, 7, 5, 4, 2, 3   # C_T != C_Z: template-track tensors are distinguishable from the pair track by shape
STD_RESIDUES_WITH_GAP = tuple(range(N_CLS))


def permute_final_dims(tensor, inds):
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def one_hot(x, lower_bins, upper_bins):
    """== opendde.model.utils.one_hot: membership of x in [lower, upper) bins as float one-hots."""
    m = (x.unsqueeze(-1) >= lower_bins) & (x.unsqueeze(-1) < upper_bins)
    return m.to(x.dtype)


def broadcast_token_to_atom(x_token, atom_to_token_idx):
    return x_token[..., atom_to_token_idx.long(), :]


def get_bin_params(cfg):
    return {"min_bin": cfg["min_bin"], "max_bin": cfg["max_bin"], "no_bins": cfg["no_bins"]}


def compute_contact_prob(distogram_logits, min_bin, max_bin, no_bins, thres=8.0):
    prob = torch.nn.functional.softmax(distogram_logits, dim=-1)
    tops = torch.linspace(min_bin, max_bin, no_bins, device=distogram_logits.device, dtype=distogram_logits.dtype)
    return prob[..., tops <= thres].sum(-1)


class LN(nn.LayerNorm):
    pass


def Lin(i, o):
    return nn.Linear(i, o, bias=False)


class MHA(nn.Module):
    """Attention along the second-to-last axis with additive biases broadcast to [..., H, Nq, Nk]."""
    def __init__(self, c, H, d):
        super().__init__()
        self.H, self.d = H, d
        self.q, self.k, self.v, self.o = Lin(c, H * d), Lin(c, H * d), Lin(c, H * d), Lin(H * d, c)

    def _prep_qkv(self, q_x, kv_x, apply_scale=True):                  # == layers.Attention._prep_qkv: heads split, q pre-scaled by 1/sqrt(d)
        H, d = self.H, self.d
        q = self.q(q_x).view(*q_x.shape[:-1], H, d).transpose(-2, -3)
        k = self.k(kv_x).view(*kv_x.shape[:-1], H, d).transpose(-2, -3)
        v = self.v(kv_x).view(*kv_x.shape[:-1], H, d).transpose(-2, -3)
        if apply_scale:
            q = q / math.sqrt(d)
        return q, k, v

    def _wrap_up(self, o, q_x):                                          # == layers.Attention._wrap_up without the gate: [.., Q, H, d] -> linear_o
        return self.o(o.reshape(*q_x.shape[:-1], self.H * self.d))

    def forward(self, q_x, kv_x, biases, **_kw):
        q, k, v = self._prep_qkv(q_x, kv_x)
        return self._wrap_up(_attention(q, k, v, biases).transpose(-2, -3), q_x)


def _attention(query, key, value, biases):                                # == layers._attention: logits = q·kᵀ + Σ biases, softmax, · v
    a = torch.matmul(query, key.transpose(-1, -2))
    for b in biases:
        a = a + b
    return torch.matmul(torch.softmax(a, dim=-1), value)


class Transition(nn.Module):
    def __init__(self, c, n=2):
        super().__init__()
        self.ln, self.up, self.down = LN(c), Lin(c, n * c), Lin(n * c, c)

    def forward(self, x):
        return self.down(torch.relu(self.up(self.ln(x))))


class TriangleMultiplication(nn.Module):
    """== TriangleMultiplicationOutgoing/Incoming (triangular.py): LN_in, gated a/b projections under the mask, the contraction, LN_out, linear_z, gate."""
    def __init__(self, c, outgoing):
        super().__init__()
        self.outgoing = outgoing
        self.layer_norm_in, self.layer_norm_out = LN(c), LN(c)
        self.linear_a_g, self.linear_a_p, self.linear_b_g, self.linear_b_p, self.linear_z, self.linear_g = (Lin(c, c) for _ in range(6))
        self.sigmoid = nn.Sigmoid()

    def forward(self, z, mask):
        from opt_core.mem.rowpair.trimul import trimul_dense
        zl = self.layer_norm_in(z)
        mk = mask.unsqueeze(-1).to(zl.dtype)
        a = mk * self.sigmoid(self.linear_a_g(zl)) * self.linear_a_p(zl)
        b = mk * self.sigmoid(self.linear_b_g(zl)) * self.linear_b_p(zl)
        x = trimul_dense(a, b, self.outgoing)
        return self.linear_z(self.layer_norm_out(x)) * self.sigmoid(self.linear_g(zl))


class TriangleAttention(nn.Module):
    """== TriangleAttention (starting form; the block transposes z for the ending node): LN, mask bias of the rows, triangle bias, mha."""
    def __init__(self, c, H=2, d=4):
        super().__init__()
        self.inf = 1e9
        self.layer_norm, self.linear, self.mha = LN(c), Lin(c, H), MHA(c, H, d)

    def forward(self, z, mask):
        x = self.layer_norm(z)
        mask_bias = (self.inf * (mask.to(x.dtype) - 1))[..., :, None, None, :]
        tb = permute_final_dims(self.linear(x), (2, 0, 1)).unsqueeze(-4)
        return self.mha(q_x=x, kv_x=x, biases=[mask_bias, tb])


class AttentionPairBias(nn.Module):
    """== AttentionPairBias (transformer.py, has_s=False): queries/keys/values from LN(s), bias from LN(z) rows."""
    def __init__(self, c_s, c_z, H=2, d=3):
        super().__init__()
        self.layernorm_a, self.layernorm_z, self.linear_nobias_z = LN(c_s), LN(c_z), Lin(c_z, H)
        self.mha = MHA(c_s, H, d)

    def _align_bias_to_query(self, bias, q, n_pair_dims=2):
        return bias

    def attention(self, q_x, kv_x, attn_bias):
        return self.mha(q_x, kv_x, [attn_bias])

    def forward(self, s, z, extra_attn_bias=None):
        a = self.layernorm_a(s)
        bias = permute_final_dims(self.linear_nobias_z(self.layernorm_z(z)), [2, 0, 1])
        if extra_attn_bias is not None:
            bias = bias + extra_attn_bias
        return self.attention(q_x=a, kv_x=a, attn_bias=self._align_bias_to_query(bias, a))


class PairformerBlock(nn.Module):
    def __init__(self, c_z, c_s):
        super().__init__()
        self.c_s = c_s
        self.tri_mul_out, self.tri_mul_in = TriangleMultiplication(c_z, True), TriangleMultiplication(c_z, False)
        self.tri_att_start, self.tri_att_end = TriangleAttention(c_z), TriangleAttention(c_z)
        self.pair_transition = Transition(c_z)
        if c_s > 0:
            self.attention_pair_bias, self.single_transition = AttentionPairBias(c_s, c_z), Transition(c_s)

    def forward(self, s, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None, extra_attn_bias=None):
        """== the pin's PairformerBlock.forward (pairformer.py:273-323): the Fold-CP hook (nothing at 1 GPU) then the source block."""
        return self.forward_source(s, z, pair_mask, triangle_multiplicative=triangle_multiplicative, triangle_attention=triangle_attention,
                                   inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=extra_attn_bias)

    def forward_source(self, s, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None, extra_attn_bias=None):
        if pair_mask is None:
            pair_mask = z.new_ones(z.shape[:-1])
        z = z + self.tri_mul_out(z, pair_mask)
        z = z + self.tri_mul_in(z, pair_mask)
        z = z + self.tri_att_start(z, pair_mask)
        z = z + self.tri_att_end(z.transpose(-2, -3), pair_mask.transpose(-1, -2)).transpose(-2, -3)
        z = z + self.pair_transition(z)
        if self.c_s > 0 and s is not None:
            s = s + self.attention_pair_bias(s, z, extra_attn_bias)
            s = s + self.single_transition(s)
        return s, z


class PairformerStack(nn.Module):
    def __init__(self, n, c_z, c_s):
        super().__init__()
        self.blocks = nn.ModuleList([PairformerBlock(c_z, c_s) for _ in range(n)])

    def forward(self, s, z, pair_mask, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None, extra_attn_bias=None):
        """== the pin's PairformerStack.forward -> forward_source -> _prep_source_blocks: the blocks are reached through ``b.forward_source``
        (pairformer.py:591-632), never through ``b.__call__``."""
        for b in self.blocks:
            s, z = b.forward_source(s, z, pair_mask, extra_attn_bias=extra_attn_bias)
        return s, z


class TemplateEmbedder(nn.Module):
    """== TemplateEmbedder.forward (pairformer.py:1563): per template v = linear_z(LN(z)) + linear_a(features under the multichain mask), its
    pair stack (c_s=0), u += LN_v(v); u / (T + 1e-7); linear_u(relu(u))."""
    def __init__(self, n_blocks=1):
        super().__init__()
        self.n_blocks = n_blocks
        a_dim = DBINS + 1 + N_CLS + N_CLS + 3 + 1
        self.layernorm_z, self.linear_no_bias_z, self.linear_no_bias_a = LN(C_Z), Lin(C_Z, C_T), Lin(a_dim, C_T)
        self.pairformer_stack = PairformerStack(n_blocks, C_T, 0)
        self.layernorm_v, self.relu, self.linear_no_bias_u = LN(C_T), nn.ReLU(), Lin(C_T, C_Z)

    def forward(self, feats, z, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        asym = feats["asym_id"]
        mc = (asym[:, None] == asym[None, :]).to(z.dtype)
        zl = self.layernorm_z(z)
        T = int(feats["template_aatype"].shape[0])
        u = 0
        for t in range(T):
            aat = torch.nn.functional.one_hot(feats["template_aatype"][t], num_classes=N_CLS).to(z.dtype)
            at = torch.cat([feats["template_distogram"][t] * mc[..., None], (feats["template_pseudo_beta_mask"][t] * mc)[..., None],
                            aat[None, :, :].expand(asym.shape[0], asym.shape[0], N_CLS), aat[:, None, :].expand(asym.shape[0], asym.shape[0], N_CLS),
                            feats["template_unit_vector"][t] * mc[..., None], (feats["template_backbone_frame_mask"][t] * mc)[..., None]], dim=-1)
            v = self.linear_no_bias_z(zl) + self.linear_no_bias_a(at)
            _, v = self.pairformer_stack(None, v, None)
            u = u + self.layernorm_v(v)
        u = u / (1e-7 + T)
        return self.linear_no_bias_u(self.relu(u))


class MSAPairWeightedAveraging(nn.Module):
    def __init__(self, H=2, c=2):
        super().__init__()
        self.n_heads, self.c = H, c
        self.layernorm_m, self.layernorm_z = LN(C_M), LN(C_Z)
        self.linear_no_bias_mv, self.linear_no_bias_mg, self.linear_no_bias_z, self.linear_no_bias_out = Lin(C_M, H * c), Lin(C_M, H * c), Lin(C_Z, H), Lin(H * c, C_M)
        self.softmax_w = nn.Softmax(dim=-2)

    def forward(self, m, z):
        mn = self.layernorm_m(m)
        v = self.linear_no_bias_mv(mn).reshape(*mn.shape[:-1], self.n_heads, self.c)
        g = torch.sigmoid(self.linear_no_bias_mg(mn)).reshape(*mn.shape[:-1], self.n_heads, self.c)
        w = self.softmax_w(self.linear_no_bias_z(self.layernorm_z(z)))
        wv = torch.einsum("ijh,mjhc->mihc", w, v)
        return self.linear_no_bias_out((g * wv).reshape(*wv.shape[:-2], self.n_heads * self.c))


class MSAStack(nn.Module):
    def __init__(self):
        super().__init__()
        self.msa_chunk_size = 4
        self.msa_pair_weighted_averaging, self.transition_m = MSAPairWeightedAveraging(), Transition(C_M)

    def forward(self, m, z):
        m = m.clone()
        S = m.shape[-3]
        for i0 in range(0, S, self.msa_chunk_size):
            i1 = min(S, i0 + self.msa_chunk_size)
            m[i0:i1] += self.msa_pair_weighted_averaging(m[i0:i1], z)
            m[i0:i1] += self.transition_m(m[i0:i1])
        return m


class OuterProductMean(nn.Module):
    def __init__(self, c=3):
        super().__init__()
        self.eps = 1e-3
        self.layer_norm, self.linear_1, self.linear_2, self.linear_out = LN(C_M), Lin(C_M, c), Lin(C_M, c), Lin(c * c, C_Z)

    def forward(self, m, inplace_safe=False, chunk_size=None):
        ln = self.layer_norm(m)
        a, b = self.linear_1(ln), self.linear_2(ln)
        outer = torch.einsum("sic,sjd->ijcd", a, b)
        return self.linear_out(outer.reshape(*outer.shape[:-2], -1)) / (float(m.shape[-3]) + self.eps)


class MSABlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.msa_stack, self.outer_product_mean_msa, self.pair_stack = MSAStack(), OuterProductMean(), PairformerBlock(C_Z, 0)

    def forward(self, m, z):
        m = self.msa_stack(m, z)
        z = z + self.outer_product_mean_msa(m)
        _, z = self.pair_stack(None, z, None)
        return m, z


class MSAModule(nn.Module):
    def __init__(self, n_blocks=2):
        super().__init__()
        self.blocks = nn.ModuleList([MSABlock() for _ in range(n_blocks)])
        self.linear_no_bias_m, self.linear_no_bias_s = Lin(3, C_M), Lin(C_SIN, C_M)

    def _prepare_msa_sample(self, input_feature_dict, s_inputs, z_token_dim):
        return self.linear_no_bias_m(input_feature_dict["msa_feat"]) + self.linear_no_bias_s(s_inputs)

    def forward(self, feats, z, s_inputs, pair_mask=None, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=False, chunk_size=None):
        m = self._prepare_msa_sample(feats, s_inputs, z.shape[-2])
        for b in self.blocks:
            m, z = b(m, z)
        return z


class LazyRelativePositionEncodingFeatures(object):
    """Stub of the stock's lazy relative-position features: rows materialized on demand from a dense table kept aside (the adapter forces the lazy
    form under n_gpu>1 and materializes ITS rows)."""
    def __init__(self, table):
        self._table = table

    def materialize(self, row_slice=slice(None), col_slice=slice(None), feature_slice=slice(None)):
        return self._table[row_slice, col_slice, feature_slice]


class RelPos(nn.Module):
    """Stub RelativePositionEncoding: ``generate_relp(feats, lazy)`` like the stock (eager ``[N, N, C_RELP]`` or the lazy object over the same
    values), ``forward`` materializes a lazy argument whole like the stock (the adapter refuses that by name under n_gpu>1)."""
    def __init__(self):
        super().__init__()
        self.linear_no_bias = Lin(C_RELP, C_Z)

    def generate_relp(self, input_feature_dict, lazy=False):
        table = input_feature_dict["relp_table"]
        input_feature_dict["relp"] = LazyRelativePositionEncodingFeatures(table) if lazy else table.clone()
        return input_feature_dict

    def forward(self, relp):
        if isinstance(relp, LazyRelativePositionEncodingFeatures):
            relp = relp.materialize()
        return self.linear_no_bias(relp)


class DistogramHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = Lin(C_Z, CBINS)

    def forward(self, z):
        logits = self.linear(z)
        return logits + logits.transpose(-2, -3)


class ConfidenceHead(nn.Module):
    """== ConfidenceHead.forward (confidence.py:204-384, non-foldcp path + memory_efficient_forward)."""
    def __init__(self):
        super().__init__()
        self.input_strunk_ln = LN(C_S)
        self.linear_no_bias_s1, self.linear_no_bias_s2 = Lin(C_SIN, C_Z), Lin(C_SIN, C_Z)
        self.lower_bins = torch.linspace(0.0, 6.0, PBINS)
        self.upper_bins = torch.cat([self.lower_bins[1:], torch.tensor([1e6])])
        self.linear_no_bias_d, self.linear_no_bias_d_wo_onehot = Lin(PBINS, C_Z), Lin(1, C_Z)
        self.pairformer_stack = PairformerStack(2, C_Z, C_S)
        self.pae_ln, self.pde_ln, self.linear_no_bias_pae, self.linear_no_bias_pde = LN(C_Z), LN(C_Z), Lin(C_Z, PBINS), Lin(C_Z, PBINS)
        self.plddt_ln, self.resolved_ln = LN(C_S), LN(C_S)
        self.plddt_weight = nn.Parameter(torch.randn(ATOMS_PER_TOK, C_S, 3))
        self.resolved_weight = nn.Parameter(torch.randn(ATOMS_PER_TOK, C_S, 2))

    def _select_distogram_rep_atom_mask(self, input_feature_dict, n_token):
        return input_feature_dict["distogram_rep_atom_mask"].bool()

    def forward(self, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_mask, x_pred_coords, z_trunk_spec=None, triangle_multiplicative="torch",
                triangle_attention="torch", inplace_safe=False, chunk_size=None, compute_plddt=True, compute_pae=True, compute_pde=True, compute_resolved=True):
        s_trunk = self.input_strunk_ln(torch.clamp(s_trunk, min=-512, max=512))
        x_rep = x_pred_coords[..., self._select_distogram_rep_atom_mask(input_feature_dict, s_trunk.shape[-2]), :]
        z = z_trunk + self.linear_no_bias_s1(s_inputs)[None, :, :] + self.linear_no_bias_s2(s_inputs)[:, None, :]
        outs = ([], [], [], [])
        for i in range(x_rep.size(-3)):
            zc, sc = z.clone(), s_trunk.clone()
            d = torch.cdist(x_rep[i], x_rep[i])
            zc = zc + self.linear_no_bias_d(one_hot(d, self.lower_bins, self.upper_bins)) + self.linear_no_bias_d_wo_onehot(d.unsqueeze(-1))
            sc, zc = self.pairformer_stack(sc, zc, pair_mask, extra_attn_bias=input_feature_dict.get("structural_pair_attn_bias"))
            a = broadcast_token_to_atom(sc, input_feature_dict["atom_to_token_idx"])
            outs[0].append(torch.einsum("nc,ncb->nb", self.plddt_ln(a), self.plddt_weight[input_feature_dict["atom_to_tokatom_idx"]]))
            outs[1].append(self.linear_no_bias_pae(self.pae_ln(zc)))
            outs[2].append(self.linear_no_bias_pde(self.pde_ln(zc + zc.transpose(-2, -3))))
            outs[3].append(torch.einsum("nc,ncb->nb", self.resolved_ln(a), self.resolved_weight[input_feature_dict["atom_to_tokatom_idx"]]))
        return torch.stack(outs[0], -3), torch.stack(outs[1], -4), torch.stack(outs[2], -4), torch.stack(outs[3], -3)


class OpenDDE(nn.Module):
    """== OpenDDE (opendde.py): get_pairformer_output :777-888 (non-foldcp), expand_to_structural_tokens (expansion DISABLED here: the structural
    stage is not part of the residue-track seams), compute_distogram_contact_probs :1113-1139, the confidence head call."""
    def __init__(self, n_cycle=2):
        super().__init__()
        self.N_cycle = n_cycle
        self.configs = types.SimpleNamespace(triangle_multiplicative="torch", triangle_attention="torch",
                                             confidence=types.SimpleNamespace(distogram={"min_bin": 2.0, "max_bin": 22.0, "no_bins": CBINS},
                                                                              pae={"min_bin": 0.0, "max_bin": 32.0, "no_bins": PBINS},
                                                                              pde={"min_bin": 0.0, "max_bin": 32.0, "no_bins": PBINS}))
        self.enable_structural_token_expansion = False
        self.input_embedder_w = Lin(7, C_SIN)
        self.linear_no_bias_sinit = Lin(C_SIN, C_S)
        self.linear_no_bias_zinit1, self.linear_no_bias_zinit2 = Lin(C_S, C_Z), Lin(C_S, C_Z)
        self.relative_position_encoding = RelPos()
        self.linear_no_bias_token_bond = Lin(1, C_Z)
        self.layernorm_z_cycle, self.linear_no_bias_z_cycle = LN(C_Z), Lin(C_Z, C_Z)
        self.layernorm_s, self.linear_no_bias_s = LN(C_S), Lin(C_S, C_S)
        self.template_embedder = TemplateEmbedder(1)
        self.msa_module = MSAModule(2)
        self.pairformer_stack = PairformerStack(2, C_Z, C_S)
        self.distogram_head = DistogramHead()
        self.confidence_head = ConfidenceHead()

    def input_embedder(self, feats, inplace_safe=False, chunk_size=None):
        return self.input_embedder_w(feats["s_raw"])

    def input_features(self, input_feature_dict):
        """The stock forward's feature step before the trunk: ``generate_relp(lazy=True)`` — the pin's inference relp is lazy (opendde.py:1906)."""
        return self.relative_position_encoding.generate_relp(input_feature_dict, lazy=True)

    def get_pairformer_output(self, input_feature_dict, N_cycle, inplace_safe=False, chunk_size=None):
        s_inputs = self.input_embedder(input_feature_dict)
        s_init = self.linear_no_bias_sinit(s_inputs)
        z_init = self.linear_no_bias_zinit1(s_init)[..., None, :] + self.linear_no_bias_zinit2(s_init)[..., None, :, :]
        z_init = z_init + self.relative_position_encoding(input_feature_dict["relp"])
        z_init = z_init + self.linear_no_bias_token_bond(input_feature_dict["token_bonds"].unsqueeze(dim=-1))
        z, s = torch.zeros_like(z_init), torch.zeros_like(s_init)
        for _ in range(N_cycle):
            z = z_init + self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z))
            if self.template_embedder.n_blocks > 0:
                z = z + self.template_embedder(input_feature_dict, z)
            z = self.msa_module(input_feature_dict, z, s_inputs, pair_mask=None)
            s = s_init + self.linear_no_bias_s(self.layernorm_s(s))
            s, z = self.pairformer_stack(s, z, pair_mask=None)
        return s_inputs, s, z

    def expand_to_structural_tokens(self, input_feature_dict, s_inputs, s, z, inplace_safe=False, chunk_size=None, lazy_relp=False):   # the pin's signature (opendde.py:422-431)
        return input_feature_dict, s_inputs, s, z

    def compute_distogram_contact_probs(self, pair_z, pair_z_spec=None):
        return compute_contact_prob(self.distogram_head(pair_z), **get_bin_params(self.configs.confidence.distogram))

    def pipeline(self, feats, coords):
        """The residue-track stages of _main_inference_loop in order: trunk -> (structural stage: equality here) -> distogram -> confidence."""
        feats = self.input_features(feats)
        s_inputs, s, z = self.get_pairformer_output(feats, self.N_cycle)
        _fd, _si, _s, _z = self.expand_to_structural_tokens(input_feature_dict=feats, s_inputs=s_inputs, s=s, z=z)
        contact = self.compute_distogram_contact_probs(z)
        plddt, pae, pde, resolved = self.confidence_head(input_feature_dict=feats, s_inputs=s_inputs, s_trunk=s, z_trunk=z, pair_mask=None, x_pred_coords=coords)
        return {"s_inputs": s_inputs, "s": s, "z": z, "contact": contact, "plddt": plddt, "pae": pae, "pde": pde, "resolved": resolved}


def build(seed=0, n=N):
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = OpenDDE().eval()
    for p in model.parameters():                       # LayerNorm affine included: nothing is an equality by luck
        with torch.no_grad():
            p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    feats = {"s_raw": torch.randn(n, 7, generator=g), "relp_table": torch.randn(n, n, C_RELP, generator=g), "token_bonds": (torch.rand(n, n, generator=g) > 0.8).float(),
             "asym_id": (torch.arange(n) >= n // 2).long(), "msa_feat": torch.randn(S_MSA, n, 3, generator=g),
             "template_aatype": torch.randint(0, N_CLS, (T_TEMPL, n), generator=g), "template_distogram": torch.rand(T_TEMPL, n, n, DBINS, generator=g),
             "template_unit_vector": torch.randn(T_TEMPL, n, n, 3, generator=g), "template_pseudo_beta_mask": (torch.rand(T_TEMPL, n, n, generator=g) > 0.3).float(),
             "template_backbone_frame_mask": (torch.rand(T_TEMPL, n, n, generator=g) > 0.3).float(),
             "distogram_rep_atom_mask": torch.tensor(([1] + [0] * (ATOMS_PER_TOK - 1)) * n), "atom_to_token_idx": torch.arange(n).repeat_interleave(ATOMS_PER_TOK),
             "atom_to_tokatom_idx": torch.arange(ATOMS_PER_TOK).repeat(n), "is_ligand": torch.zeros(n * ATOMS_PER_TOK, dtype=torch.long),
             "has_frame": (torch.rand(n, generator=g) > 0.2)}
    coords = torch.randn(2, n * ATOMS_PER_TOK, 3, generator=g) * 3.0          # 2 samples
    return model, feats, coords


def install_modules():
    """Register the stub modules under the stock's module paths (idempotent). Returns the `opendde.model.opendde` stub module."""
    if "opendde.model.opendde" in sys.modules and getattr(sys.modules["opendde.model.opendde"], "_tp_stub", False):
        return sys.modules["opendde.model.opendde"]
    mods = {}
    for name in ("opendde", "opendde.model", "opendde.model.modules", "opendde.model.opendde", "opendde.model.modules.pairformer",
                 "opendde.model.modules.confidence", "opendde.model.modules.embedders", "opendde.model.utils", "opendde.model.sample_confidence"):
        m = types.ModuleType(name)
        m.__path__ = []                                # package-like: sub-imports resolve through sys.modules
        m._tp_stub = True
        mods[name] = m
    mods["opendde.model.opendde"].OpenDDE = OpenDDE
    pf = mods["opendde.model.modules.pairformer"]
    pf.PairformerStack, pf.PairformerBlock, pf.STD_RESIDUES_WITH_GAP = PairformerStack, PairformerBlock, STD_RESIDUES_WITH_GAP
    mods["opendde.model.modules.confidence"].ConfidenceHead = ConfidenceHead
    emb = mods["opendde.model.modules.embedders"]
    emb.RelativePositionEncoding, emb.LazyRelativePositionEncodingFeatures = RelPos, LazyRelativePositionEncodingFeatures
    ut = mods["opendde.model.utils"]
    ut.permute_final_dims, ut.one_hot, ut.broadcast_token_to_atom = permute_final_dims, one_hot, broadcast_token_to_atom
    sc = mods["opendde.model.sample_confidence"]
    sc.compute_contact_prob, sc.get_bin_params = compute_contact_prob, get_bin_params
    sys.modules.update(mods)
    return mods["opendde.model.opendde"]
