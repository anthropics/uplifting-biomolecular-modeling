"""The structural diffusion stage on rows (opendde_opt.tp_diffusion) vs the DENSE engine statements, CPU, gloo, P in {2, 3} through the
kit launcher (``opt_core.mem.rowpair.launch.run_sharded``): a miniature torch diffusion engine with OpenDDE 1.0.0's ATTRIBUTE NAMES and
DENSE statements (DiffusionConditioning.prepare_cache / forward, AttentionPairBias incl. the structural extra bias and the fused conv2d
form, DiffusionTransformer, ConditionedTransitionBlock, the atom encoder's token-pair gather, DiffusionModule.forward, the shared-variables
cache and the sampler loop) is the reference; the same weights run through tp_diffusion's binding of the core drivers in every rank.
Checked per rank (verdicts AND-ed onto rank 0): z_cond rows, every block's pair-bias rows, the atom-pair band term at valid slots and the
sampled coordinates vs dense (fp32 max|diff| <= 1e-5; torch.equal REPORTED), the census (zcond_rows, dit_local_queries, denoise calls,
bias variant), no op output shaped ``[.., N_st, .., N_st, ..]`` and max op-output numel < N_st*N_st*16 while the sharded stage runs, the draw
guard refusing a divergent rank BY NAME, and ``P = 1`` refusing by name. Tiny dims (N_st=48, 2-3 atoms/token, c_pair=8, 2 DiT blocks,
2 steps x 2 samples): the point is the statement algebra, the seams and the census."""
from __future__ import annotations

import copy
import math
import os
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("opt_core.mem.rowpair.diffusion")
from torch import nn                                                        # noqa: E402
import torch.nn.functional as F                                             # noqa: E402

N_RES, EXPAND, C_Z, C_PD, C_S, C_SIN, C_TOK, C_ATOM, C_AP, H_DIT, N_BLK, NQ, NK, C_REL, C_NOISE = 24, 2, 12, 8, 6, 5, 10, 6, 4, 2, 2, 4, 8, 5, 4
N_ST = N_RES * EXPAND
TOL = 1e-5


# ============================================================================================================ engine stubs (module paths)
def _dense_trunk(n: int, nq: int, nk: int):
    """== rearrange_qk_to_dense_trunk's index geometry for a length-n axis: query slots padded with 0 at the end to nb*nq; key window b =
    padded[b*nq : b*nq + nk] of the axis padded with 0 by (nk-nq)//2 on the left; mask[b, i, j] = (query real) & (key real)."""
    nb = -(-n // nq)
    q_pad = nb * nq - n
    pad_left = (nk - nq) // 2
    pad_right = max(0, (nb - 1) * nq + pad_left + nk - (n + pad_left)) + 0
    ar = torch.arange(n)
    qi = F.pad(ar, (0, q_pad), value=-1).view(nb, nq)
    kp = F.pad(ar, (pad_left, pad_right), value=-1)
    ki = torch.stack([kp[b * nq: b * nq + nk] for b in range(nb)], 0)
    mask = (qi[:, :, None] >= 0) & (ki[:, None, :] >= 0)
    return qi.clamp(min=0), ki.clamp(min=0), mask, q_pad


def rearrange_qk_to_dense_trunk(q, k, dim_q, dim_k, n_queries=32, n_keys=128, compute_mask=True):
    """Index-map form (1-D token-index tensors), the call tp_diffusion makes: (idx_q [nb, nq], idx_k [nb, nk], pad_info)."""
    assert q.dim() == 1 and k.dim() == 1 and q.shape == k.shape
    qi, ki, mask, q_pad = _dense_trunk(int(q.shape[0]), n_queries, n_keys)
    return q[qi], k[ki], {"mask_trunked": mask.to(q.dtype) if compute_mask else None, "q_pad": q_pad}


def _attention(q, k, v, attn_bias=None, use_efficient_implementation=False, inplace_safe=False):
    """== primitives._attention (matmul / softmax path)."""
    k = k.transpose(-1, -2)
    w = q @ k
    if attn_bias is not None:
        w = w + attn_bias
    w = F.softmax(w, dim=-1)
    return w @ v


def autocasting_disable_decorator(disable_casting):
    def deco(fn):
        return fn
    return deco


def install_modules():
    """Register (or extend) stub modules under the stock's paths for the names tp_diffusion imports lazily."""
    def mod(name):
        m = sys.modules.get(name)
        if m is None:
            m = types.ModuleType(name)
            m._tp_stub = True
            sys.modules[name] = m
            parent, _, leaf = name.rpartition(".")
            if parent:
                setattr(mod(parent), leaf, m)
        return m
    prim = mod("opendde.model.modules.primitives")
    prim._attention, prim.rearrange_qk_to_dense_trunk = _attention, rearrange_qk_to_dense_trunk
    tu = mod("opendde.utils.torch_utils")
    tu.autocasting_disable_decorator = autocasting_disable_decorator


# ============================================================================================================ miniature engine (dense reference)
def LayerNorm(c, create_scale=True, create_offset=True):
    if not create_scale and not create_offset:
        return nn.LayerNorm(c, elementwise_affine=False)
    return nn.LayerNorm(c, bias=create_offset)


def LinearNoBias(i, o, **kw):
    return nn.Linear(i, o, bias=False)


class Transition(nn.Module):
    def __init__(self, c, n=2):
        super().__init__()
        self.layernorm1, self.linear_no_bias_a, self.linear_no_bias_b, self.linear_no_bias = LayerNorm(c), LinearNoBias(c, n * c), LinearNoBias(c, n * c), LinearNoBias(n * c, c)

    def forward(self, x):
        y = self.layernorm1(x)
        return self.linear_no_bias(F.silu(self.linear_no_bias_a(y)) * self.linear_no_bias_b(y))


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, c_a, c_s):
        super().__init__()
        self.layernorm_a, self.layernorm_s = LayerNorm(c_a, create_scale=False, create_offset=False), LayerNorm(c_s, create_offset=False)
        self.linear_s, self.linear_nobias_s = nn.Linear(c_s, c_a), LinearNoBias(c_s, c_a)

    def forward(self, a, s):
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        return torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)


class Attention(nn.Module):
    """== primitives.Attention (global attention, gating, q bias)."""
    def __init__(self, c_q, c_k, c_v, c_hidden, num_heads, gating=True):
        super().__init__()
        self.c_hidden, self.num_heads, self.gating, self.use_efficient_implementation = c_hidden, num_heads, gating, False
        self.linear_q = nn.Linear(c_q, c_hidden * num_heads, bias=True)
        self.linear_k, self.linear_v = LinearNoBias(c_k, c_hidden * num_heads), LinearNoBias(c_v, c_hidden * num_heads)
        self.linear_o = LinearNoBias(c_hidden * num_heads, c_q)
        self.linear_g = LinearNoBias(c_q, c_hidden * num_heads) if gating else None
        self.sigmoid = nn.Sigmoid()

    def _prep_qkv(self, q_x, kv_x, apply_scale=True):
        q, k, v = self.linear_q(q_x), self.linear_k(kv_x), self.linear_v(kv_x)
        q = q.view(q.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)
        k = k.view(k.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)
        v = v.view(v.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)
        if apply_scale:
            q = q / math.sqrt(self.c_hidden)
        return q, k, v

    def _wrap_up(self, o, q_x):
        if self.linear_g is not None:
            g = self.sigmoid(self.linear_g(q_x))
            g = g.view(g.shape[:-1] + (self.num_heads, -1))
            o = o * g
        o = o.reshape(o.shape[:-2] + (-1,))
        return self.linear_o(o)

    def forward(self, q_x, kv_x, attn_bias=None, inplace_safe=False, **kw):
        q, k, v = self._prep_qkv(q_x, kv_x, apply_scale=True)
        if attn_bias is not None and len(attn_bias.shape) != len(q.shape):
            attn_bias = attn_bias.unsqueeze(-3)
        o = _attention(q, k, v, attn_bias=attn_bias, inplace_safe=inplace_safe)
        return self._wrap_up(o.transpose(-2, -3), q_x)


def permute_final_dims(t, inds):
    z = -len(inds)
    first = list(range(len(t.shape[:z])))
    return t.permute(first + [z + i for i in inds])


class AttentionPairBias(nn.Module):
    """== transformer.AttentionPairBias (has_s, standard attention path, structural extra bias, fused conv2d form)."""
    def __init__(self, c_a, c_s, c_z, n_heads):
        super().__init__()
        self.has_s, self.cross_attention_mode, self.n_heads, self.c_a, self.c_z = True, False, n_heads, c_a, c_z
        self.layernorm_a = AdaptiveLayerNorm(c_a, c_s)
        self.linear_a_last = nn.Linear(c_s, c_a)
        nn.init.constant_(self.linear_a_last.bias, -2.0)
        self.layernorm_z = LayerNorm(c_z, create_offset=False)
        self.linear_nobias_z = LinearNoBias(c_z, n_heads)
        self.attention = Attention(c_a, c_a, c_a, c_a // n_heads, n_heads, gating=True)

    @staticmethod
    def _add_extra_attn_bias_to_chunk(bias, extra_attn_bias, row_start, row_end):
        if extra_attn_bias is None:
            return bias
        extra = extra_attn_bias[..., row_start:row_end, :]
        while len(extra.shape) < len(bias.shape) - 1:
            extra = extra.unsqueeze(0)
        if len(extra.shape) == len(bias.shape) - 1:
            extra = extra.unsqueeze(-3)
        return bias + extra

    @staticmethod
    def _align_bias_to_query(bias, q, n_pair_dims):
        target_ndim = len(q.shape[:-2]) + 1 + n_pair_dims
        while len(bias.shape) < target_ndim:
            bias = bias.unsqueeze(len(bias.shape) - (1 + n_pair_dims))
        return bias

    def forward(self, a, s, z, extra_attn_bias=None, inplace_safe=False, enable_efficient_fusion=False, **kw):
        a = self.layernorm_a(a, s)
        if enable_efficient_fusion:                                            # z = normalize(pair) permuted to [c, N, N]
            w = (self.linear_nobias_z.weight * self.layernorm_z.weight[None, :])[:, :, None, None]
            bias = F.conv2d(z, w)                                              # [H, N, N]
        else:
            bias = permute_final_dims(self.linear_nobias_z(self.layernorm_z(z)), [2, 0, 1])
        if extra_attn_bias is not None:
            extra = extra_attn_bias
            while len(extra.shape) < len(bias.shape) - 1:
                extra = extra.unsqueeze(0)
            if len(extra.shape) == len(bias.shape) - 1:
                extra = extra.unsqueeze(-3)
            bias = bias + extra
        bias = self._align_bias_to_query(bias, a, n_pair_dims=2)
        a = self.attention(q_x=a, kv_x=a, attn_bias=bias, inplace_safe=inplace_safe)
        return torch.sigmoid(self.linear_a_last(s)) * a


class ConditionedTransitionBlock(nn.Module):
    def __init__(self, c_a, c_s, n=2):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(c_a, c_s)
        self.linear_nobias_a1, self.linear_nobias_a2, self.linear_nobias_b = LinearNoBias(c_a, n * c_a), LinearNoBias(c_a, n * c_a), LinearNoBias(n * c_a, c_a)
        self.linear_s = nn.Linear(c_s, c_a)
        nn.init.constant_(self.linear_s.bias, -2.0)

    def forward(self, a, s):
        a = self.adaln(a, s)
        b = F.silu(self.linear_nobias_a1(a)) * self.linear_nobias_a2(a)
        return torch.sigmoid(self.linear_s(s)) * self.linear_nobias_b(b)


class DiffusionTransformerBlock(nn.Module):
    def __init__(self, c_a, c_s, c_z, n_heads):
        super().__init__()
        self.attention_pair_bias = AttentionPairBias(c_a, c_s, c_z, n_heads)
        self.conditioned_transition_block = ConditionedTransitionBlock(c_a, c_s)

    def forward(self, a, s, z, **kw):
        attn_out = self.attention_pair_bias(a, s, z, **kw)
        attn_out = attn_out + a
        return self.conditioned_transition_block(attn_out, s) + attn_out


class DiffusionTransformer(nn.Module):
    def __init__(self, c_a, c_s, c_z, n_blocks, n_heads):
        super().__init__()
        self.n_blocks, self.n_heads = n_blocks, n_heads
        self.blocks = nn.ModuleList([DiffusionTransformerBlock(c_a, c_s, c_z, n_heads) for _ in range(n_blocks)])

    def forward(self, a, s, z, **kw):
        for b in self.blocks:
            a = b(a, s, z, **kw)
        return a


class FourierEmbedding(nn.Module):
    def __init__(self, c, seed=42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.w, self.b = nn.Parameter(torch.randn(c, generator=g), requires_grad=False), nn.Parameter(torch.randn(c, generator=g), requires_grad=False)

    def forward(self, t_hat_noise_level):
        return torch.cos(2 * math.pi * (t_hat_noise_level.unsqueeze(-1) * self.w + self.b))


class LazyRelp(object):
    """== embedders' lazy relative-position features: materialize(row_slice, col_slice) -> [rows, cols, C_REL] (never whole here)."""
    def __init__(self, tok_idx, asym):
        self.tok_idx, self.asym = tok_idx, asym

    def materialize(self, row_slice=slice(None), col_slice=slice(None), feature_slice=slice(None)):
        ti, tj = self.tok_idx[row_slice], self.tok_idx[col_slice]
        d = torch.clip(ti[:, None] - tj[None, :] + (C_REL - 2) // 2, 0, C_REL - 2)
        same = (self.asym[row_slice][:, None] == self.asym[col_slice][None, :]).to(torch.float32)
        return torch.cat([F.one_hot(d, C_REL - 1).to(torch.float32) * same[..., None], same[..., None]], -1)[..., feature_slice]


class RelativePositionEncoding(nn.Module):
    def __init__(self, c_z):
        super().__init__()
        self.linear_no_bias = LinearNoBias(C_REL, c_z)

    def forward(self, relp_feature):
        if hasattr(relp_feature, "materialize"):
            relp_feature = relp_feature.materialize()
        return self.linear_no_bias(relp_feature)


class DiffusionConditioning(nn.Module):
    """== diffusion.DiffusionConditioning (compress_pair_z path: c_z != c_z_pair_diffusion)."""
    def __init__(self):
        super().__init__()
        self.sigma_data, self.c_z, self.c_z_pair_diffusion, self.c_s = 16.0, C_Z, C_PD, C_S
        self.compress_pair_z = True
        self.layernorm_z_trunk, self.linear_no_bias_z_trunk = LayerNorm(C_Z), LinearNoBias(C_Z, C_PD)
        self.relpe = RelativePositionEncoding(C_PD)
        self.layernorm_z, self.linear_no_bias_z = LayerNorm(2 * C_PD), LinearNoBias(2 * C_PD, C_PD)
        self.transition_z1, self.transition_z2 = Transition(C_PD), Transition(C_PD)
        self.layernorm_s, self.linear_no_bias_s = LayerNorm(C_S + C_SIN), LinearNoBias(C_S + C_SIN, C_S)
        self.fourier_embedding = FourierEmbedding(C_NOISE)
        self.layernorm_n, self.linear_no_bias_n = LayerNorm(C_NOISE), LinearNoBias(C_NOISE, C_S)
        self.transition_s1, self.transition_s2 = Transition(C_S), Transition(C_S)

    def _project_z_trunk(self, z):
        return self.linear_no_bias_z_trunk(self.layernorm_z_trunk(z))

    def _project_pair_z(self, z):
        return self.linear_no_bias_z(self.layernorm_z(z))

    def _apply_pair_z_transitions(self, pair_z, inplace_safe):
        flat = pair_z.view(-1, pair_z.shape[-1])
        flat += self.transition_z1(flat)
        flat += self.transition_z2(flat)
        return pair_z

    def prepare_cache(self, relp_feature, z_trunk, inplace_safe):
        pair_z = torch.cat([self._project_z_trunk(z_trunk), self.relpe(relp_feature)], -1)
        pair_z = self._project_pair_z(pair_z)
        return self._apply_pair_z_transitions(pair_z, inplace_safe)

    def forward(self, t_hat_noise_level, relp_feature, s_inputs, s_trunk, z_trunk, pair_z=None, inplace_safe=False, use_conditioning=True):
        if pair_z is None:
            pair_z = self.prepare_cache(relp_feature, z_trunk, inplace_safe)
        elif inplace_safe:
            pair_z = pair_z.clone()
        single_s = torch.cat([s_trunk, s_inputs], -1)
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_ratio = (t_hat_noise_level / self.sigma_data).clamp(min=1e-10)
        noise_n = self.fourier_embedding(t_hat_noise_level=torch.log(input=noise_ratio) / 4).to(single_s.dtype)
        single_s = single_s.unsqueeze(-3) + self.linear_no_bias_n(self.layernorm_n(noise_n)).unsqueeze(-2)
        single_s += self.transition_s1(single_s)
        single_s += self.transition_s2(single_s)
        return single_s, pair_z


class AtomAttentionEncoder(nn.Module):
    """A small atom encoder with the stock's cache surface: prepare_cache (p_lm incl. the DENSE token-pair gather z[idx_q, idx_k] when r_l is
    given, c_l) and forward on the cache (windowed attention with p_lm as the pair bias -> the token-pair term reaches the positions)."""
    def __init__(self):
        super().__init__()
        self.n_queries, self.n_keys = NQ, NK
        self.linear_no_bias_ref_pos, self.linear_no_bias_d = LinearNoBias(3, C_ATOM), LinearNoBias(3, C_AP)
        self.layernorm_z, self.linear_no_bias_z = LayerNorm(C_PD, create_offset=False), LinearNoBias(C_PD, C_AP)
        self.layernorm_s, self.linear_no_bias_s = LayerNorm(C_S, create_offset=False), LinearNoBias(C_S, C_ATOM)
        self.linear_no_bias_r = LinearNoBias(3, C_ATOM)
        self.linear_cq, self.linear_ck, self.linear_pb = LinearNoBias(C_ATOM, C_AP), LinearNoBias(C_ATOM, C_AP), LinearNoBias(C_AP, 1)
        self.linear_qkv, self.linear_no_bias_q = LinearNoBias(C_ATOM, 3 * C_ATOM), LinearNoBias(C_ATOM, C_TOK)

    def _add_token_pair_context_to_atom_pair(self, p_lm, z, atom_to_token_idx):
        idx_q, idx_k, _ = rearrange_qk_to_dense_trunk(atom_to_token_idx, atom_to_token_idx, -1, -1, self.n_queries, self.n_keys, compute_mask=False)
        p_lm = p_lm.unsqueeze(dim=-5)
        z_token_pair = z[idx_q.long()[:, :, None], idx_k.long()[:, None, :]]              # == gather_pair_embedding_in_dense_trunk
        z_token_pair = self.linear_no_bias_z(self.layernorm_z(z_token_pair))
        p_lm[..., :, :, :, :] = p_lm[..., :, :, :, :] + z_token_pair.unsqueeze(dim=-5)
        return p_lm

    def prepare_cache(self, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, atom_to_token_idx, d_lm, v_lm, pad_info,
                      r_l=None, z=None, inplace_safe=False):
        c_l = self.linear_no_bias_ref_pos(ref_pos)
        p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info["mask_trunked"].unsqueeze(-1)
        if r_l is not None:
            p_lm = self._add_token_pair_context_to_atom_pair(p_lm, z, atom_to_token_idx)
        return p_lm, c_l

    def forward(self, atom_to_token_idx, ref_pos, ref_charge, ref_mask, ref_atom_name_chars, ref_element, d_lm, v_lm, pad_info, r_l=None, s=None,
                z=None, p_lm=None, c_l=None, inplace_safe=False, chunk_size=None):
        assert z is not None and p_lm is not None and c_l is not None
        n_atom, n_tok = int(ref_pos.shape[-2]), int(s.shape[-2])
        a2t = atom_to_token_idx.long()
        c_l = c_l + self.linear_no_bias_s(self.layernorm_s(s))[..., a2t, :]                # [S, N_atom, c_atom]
        q_l = c_l + self.linear_no_bias_r(r_l)
        qi, ki, mask, q_pad = _dense_trunk(n_atom, self.n_queries, self.n_keys)
        p = p_lm + self.linear_cq(F.relu(c_l[..., qi, :]))[..., :, :, None, :] + self.linear_ck(F.relu(c_l[..., ki, :]))[..., :, None, :, :]
        bias = self.linear_pb(p)[..., 0] + (~mask).to(p.dtype) * (-1e10)                  # [S, nb, nq, nk]; padded keys masked like rearrange_to_dense_trunk
        q, k, v = self.linear_qkv(q_l).chunk(3, -1)
        logits = torch.einsum("...bqc,...bkc->...bqk", q[..., qi, :], k[..., ki, :]) / math.sqrt(C_ATOM) + bias
        o = torch.einsum("...bqk,...bkc->...bqc", torch.softmax(logits, -1), v[..., ki, :]).reshape(q_l.shape[:-2] + (-1, C_ATOM))[..., :n_atom, :]
        q_l = q_l + o
        x = F.relu(self.linear_no_bias_q(q_l))
        a = torch.zeros(x.shape[:-2] + (n_tok, C_TOK), dtype=x.dtype).index_add_(-2, a2t, x) / torch.bincount(a2t, minlength=n_tok).clamp(min=1)[:, None]
        return a, q_l, c_l, p


class AtomAttentionDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear_no_bias_a, self.layernorm_q, self.linear_no_bias_out, self.linear_p = LinearNoBias(C_TOK, C_ATOM), LayerNorm(C_ATOM, create_offset=False), LinearNoBias(C_ATOM, 3), LinearNoBias(C_AP, C_ATOM)

    def forward(self, atom_to_token_idx, a, q_skip, c_skip, p_skip, inplace_safe=False, chunk_size=None):
        mask = _dense_trunk(int(atom_to_token_idx.shape[-1]), NQ, NK)[2].to(p_skip.dtype)[..., None]     # padded pair slots never reach an output (the engine masks them with -inf)
        pctx = self.linear_p((p_skip * mask).sum((-2, -3)) / mask.sum((-2, -3))).mean(-2, keepdim=True)
        q = self.linear_no_bias_a(a)[..., atom_to_token_idx.long(), :] + q_skip + pctx
        return self.linear_no_bias_out(self.layernorm_q(q))


class DiffusionModule(nn.Module):
    """== diffusion.DiffusionModule.forward / f_forward (single-process, cached-pair path) — the DENSE reference denoiser."""
    def __init__(self):
        super().__init__()
        self.sigma_data, self.c_z_pair_diffusion = 16.0, C_PD
        self.diffusion_conditioning = DiffusionConditioning()
        self.atom_attention_encoder = AtomAttentionEncoder()
        self.layernorm_s, self.linear_no_bias_s = LayerNorm(C_S, create_offset=False), LinearNoBias(C_S, C_TOK)
        self.diffusion_transformer = DiffusionTransformer(C_TOK, C_S, C_PD, N_BLK, H_DIT)
        self.layernorm_a = LayerNorm(C_TOK, create_offset=False)
        self.atom_attention_decoder = AtomAttentionDecoder()
        self.normalize = LayerNorm(C_PD, create_scale=False, create_offset=False)

    def f_forward(self, x_noisy, r_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, inplace_safe=False,
                  chunk_size=None, use_conditioning=True, enable_efficient_fusion=False):
        f = input_feature_dict
        s_single, z_pair = self.diffusion_conditioning(t_hat_noise_level, f["relp"], s_inputs, s_trunk, z_trunk, pair_z=pair_z, inplace_safe=inplace_safe)
        a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(f["atom_to_token_idx"], f["ref_pos"], f["ref_charge"], f["ref_mask"], f["ref_atom_name_chars"],
                                                                      f["ref_element"], f["d_lm"], f["v_lm"], f["pad_info"], r_l=r_noisy, s=s_trunk.unsqueeze(-3),
                                                                      z=z_pair, p_lm=p_lm, c_l=c_l, inplace_safe=inplace_safe, chunk_size=chunk_size)
        a_token = a_token.to(torch.float32)
        a_token += self.linear_no_bias_s(self.layernorm_s(s_single))
        if enable_efficient_fusion:
            z = self.normalize(z_pair.to(torch.float32))
            z = torch.permute(z, (2, 0, 1)).contiguous()
        else:
            z = z_pair.to(torch.float32)
        a_token = self.diffusion_transformer(a_token, s_single.to(torch.float32), z, extra_attn_bias=f.get("structural_pair_attn_bias"),
                                             inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        a_token = self.layernorm_a(a_token)
        return self.atom_attention_decoder(f["atom_to_token_idx"], a_token, q_skip, c_skip, p_skip)

    def forward(self, x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk=None, pair_z=None, p_lm=None, c_l=None, inplace_safe=False,
                chunk_size=None, use_conditioning=True, enable_efficient_fusion=False, pair_z_spec=None):
        r_noisy = x_noisy / torch.sqrt(self.sigma_data ** 2 + t_hat_noise_level ** 2)[..., None, None]
        r_update = self.f_forward(x_noisy, r_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l,
                                  inplace_safe=inplace_safe, chunk_size=chunk_size, enable_efficient_fusion=enable_efficient_fusion)
        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(r_update.dtype)
        return (1 / (1 + s_ratio ** 2) * x_noisy + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio ** 2) * r_update).to(r_update.dtype)


def sample_diffusion(denoise_net, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, noise_schedule, N_sample, gamma0=0.8, gamma_min=1.0,
                     noise_scale_lambda=1.003, step_scale_eta=1.5, diffusion_chunk_size=None, inplace_safe=False, attn_chunk_size=None,
                     enable_efficient_fusion=False, rollout_seed=None, pair_z_spec=None):
    """== generator.sample_diffusion (one sample chunk; seeded generator; centre + random translation augmentation; predictor step)."""
    n_atom = int(input_feature_dict["atom_to_token_idx"].shape[-1])
    g = torch.Generator().manual_seed(int(rollout_seed))
    x_l = noise_schedule[0] * torch.randn((N_sample, n_atom, 3), generator=g)
    for c_tau_last, c_tau in zip(noise_schedule[:-1], noise_schedule[1:]):
        x_l = x_l - x_l.mean(-2, keepdim=True) + torch.randn((N_sample, 1, 3), generator=g)
        gamma = float(gamma0) if c_tau > gamma_min else 0.0
        t_hat = c_tau_last * (gamma + 1)
        delta_noise = torch.sqrt(t_hat ** 2 - c_tau_last ** 2)
        x_noisy = x_l + noise_scale_lambda * delta_noise * torch.randn(x_l.shape, generator=g)
        t_hat = t_hat.reshape(1).expand(N_sample)
        x_den = denoise_net(x_noisy=x_noisy, t_hat_noise_level=t_hat, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk,
                            z_trunk=z_trunk, pair_z=pair_z, pair_z_spec=pair_z_spec, p_lm=p_lm, c_l=c_l, chunk_size=attn_chunk_size,
                            inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
        delta = (x_noisy - x_den) / t_hat[..., None, None]
        dt = c_tau - t_hat
        x_l = x_noisy + step_scale_eta * dt[..., None, None] * delta
    return x_l


class Model(nn.Module):
    """== OpenDDE's diffusion-stage surface: prepare_diffusion_cache_for_sampling (shared-variables cache on) + run_sample_diffusion_stage."""
    def __init__(self, fusion: bool):
        super().__init__()
        self.diffusion_module = DiffusionModule()
        self.enable_efficient_fusion, self.enable_diffusion_shared_vars_cache = fusion, True
        self.configs = types.SimpleNamespace(skip_amp=types.SimpleNamespace(sample_diffusion=True), sample_diffusion={"guidance": None},
                                             infer_setting=types.SimpleNamespace(sample_diffusion_chunk_size=None))

    def prepare_diffusion_cache_for_sampling(self, input_feature_dict, z, pair_z_spec=None, force_enable_cache=False):
        f, dm = input_feature_dict, self.diffusion_module
        cache = {"pair_z_spec": None}
        cache["pair_z"] = dm.diffusion_conditioning.prepare_cache(f["relp"], z, False)
        cache["p_lm/c_l"] = list(dm.atom_attention_encoder.prepare_cache(ref_pos=f["ref_pos"], ref_charge=f["ref_charge"], ref_mask=f["ref_mask"], ref_element=f["ref_element"],
                                                                  ref_atom_name_chars=f["ref_atom_name_chars"], atom_to_token_idx=f["atom_to_token_idx"], d_lm=f["d_lm"],
                                                                  v_lm=f["v_lm"], pad_info=f["pad_info"], r_l=True, z=cache["pair_z"], inplace_safe=False))
        return cache

    def run_sample_diffusion_stage(self, *, pred_dict, input_feature_dict, s_inputs, s, z, pair_z_spec, cache, N_sample, noise_schedule, chunk_size, inplace_safe):
        sample_pair_z = cache["pair_z"]
        sample_z_trunk = None if sample_pair_z is not None else z
        pred_dict["coordinate"] = sample_diffusion(self.diffusion_module, input_feature_dict, s_inputs, s, sample_z_trunk, sample_pair_z, cache["p_lm/c_l"][0],
                                                   cache["p_lm/c_l"][1], noise_schedule, N_sample, diffusion_chunk_size=self.configs.infer_setting.sample_diffusion_chunk_size,
                                                   inplace_safe=inplace_safe, attn_chunk_size=chunk_size, enable_efficient_fusion=self.enable_efficient_fusion,
                                                   rollout_seed=int(input_feature_dict["inference_seed"]), pair_z_spec=cache["pair_z_spec"])
        return pred_dict["coordinate"]


def build(seed: int, fusion: bool):
    """Deterministic (model, structural feature dict, s_inputs, s, z_struct [N_st, N_st, C_Z] host)."""
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    model = Model(fusion).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.requires_grad:
                p.copy_(torch.randn(p.shape, generator=g) * (0.35 if p.dim() > 1 else 0.1))
        for m in model.modules():
            if isinstance(m, nn.LayerNorm) and m.elementwise_affine:
                m.weight.copy_(1.0 + 0.1 * torch.randn(m.weight.shape, generator=g))
    apt = torch.tensor([2 + int(i % 3 == 0) for i in range(N_ST)])                             # 2-3 atoms per structural token
    a2t = torch.repeat_interleave(torch.arange(N_ST), apt)
    n_atom = int(a2t.numel())
    qi, ki, mask, q_pad = _dense_trunk(n_atom, NQ, NK)
    tok_idx = torch.arange(N_ST) // EXPAND
    asym = (torch.arange(N_ST) >= (N_ST * 2) // 3).long()
    ref_pos = torch.randn((n_atom, 3), generator=g)
    d = ref_pos[qi] [:, :, None, :] - ref_pos[ki][:, None, :, :]
    feats = {"atom_to_token_idx": a2t, "ref_pos": ref_pos, "ref_charge": torch.zeros(n_atom), "ref_mask": torch.ones(n_atom), "ref_element": torch.zeros(n_atom, 4),
             "ref_atom_name_chars": torch.zeros(n_atom, 4), "d_lm": d, "v_lm": (d.norm(dim=-1, keepdim=True) < 2.0).to(torch.float32),
             "pad_info": {"mask_trunked": mask.to(torch.float32), "q_pad": q_pad}, "relp": LazyRelp(tok_idx, asym), "inference_seed": torch.tensor(11),
             "structural_pair_attn_bias": 0.5 * torch.randn((N_ST, N_ST), generator=g)}
    s_inputs, s = torch.randn((N_ST, C_SIN), generator=g), torch.randn((N_ST, C_S), generator=g)
    z_struct = torch.randn((N_ST, N_ST, C_Z), generator=g)
    return model, feats, s_inputs, s, z_struct


class _HostPair(object):
    """The offload unit's HostPair surface the stage reads: ``.t`` = the host-resident [N_st, N_st, c] tensor."""
    def __init__(self, z):
        self.t, self.n, self.c = z, int(z.shape[0]), int(z.shape[-1])


def _pair_guard(N):
    """Dispatch mode recording every aten op output with >= 2 dims equal to N, and the max output numel, while active."""
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.offenders, self.max_numel, self.max_shape = [], 0, None

        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for leaf in tree_flatten(out)[0]:
                if isinstance(leaf, torch.Tensor):
                    if sum(1 for dd in leaf.shape if int(dd) == N) >= 2:
                        self.offenders.append((str(func), tuple(leaf.shape)))
                    if leaf.numel() > self.max_numel:
                        self.max_numel, self.max_shape = int(leaf.numel()), (str(func), tuple(leaf.shape))
            return out
    return Guard()


# ============================================================================================================ per-rank entry
def _stage_kw(feats, s_inputs, s, z, cache):
    return dict(pred_dict={}, input_feature_dict=feats, s_inputs=s_inputs, s=s, z=z, pair_z_spec=None, cache=cache, N_sample=2,
                noise_schedule=torch.tensor([40.0, 9.0, 1.6]), chunk_size=None, inplace_safe=True)


def _rank_entry(seed: int, fusion: bool, src_kind: str):
    """Every rank: dense reference (whole tensors, the stub engine's own statements) -> the same model through tp_diffusion -> compare HERE."""
    install_modules()
    torch.set_num_threads(1)
    from opendde_opt import tp, tp_diffusion as TD
    from opt_core.mem.rowpair import RowpairRefused, diffusion as DF
    model, feats, s_inputs, s, z_struct = build(seed, fusion)
    res = {"checks": {}, "metrics": {}, "bitwise": {}, "facts": {}}
    chk, met, bit, facts = res["checks"], res["metrics"], res["bitwise"], res["facts"]

    def cmp(name, got, ref, tol=TOL):
        d = float((got.contiguous().to(torch.float32) - ref.contiguous().to(torch.float32)).abs().max())
        met[name], bit[name], chk[name] = d, bool(torch.equal(got.contiguous(), ref.contiguous())), d <= tol

    with torch.no_grad():
        # ---------------------------------------------------------------- dense reference
        cache_d = model.prepare_diffusion_cache_for_sampling(copy.copy(feats), z_struct.clone())
        pair_z = cache_d["pair_z"]
        coords_d = model.run_sample_diffusion_stage(**_stage_kw(feats, s_inputs, s, z_struct, cache_d))
        dm = model.diffusion_module
        if fusion:
            zf = torch.permute(dm.normalize(pair_z), (2, 0, 1)).contiguous()
            bias_d = [F.conv2d(zf, (b.attention_pair_bias.linear_nobias_z.weight * b.attention_pair_bias.layernorm_z.weight[None, :])[:, :, None, None])
                      for b in dm.diffusion_transformer.blocks]                                                       # [H, N, N]
        else:
            bias_d = [permute_final_dims(b.attention_pair_bias.linear_nobias_z(b.attention_pair_bias.layernorm_z(pair_z)), [2, 0, 1]) for b in dm.diffusion_transformer.blocks]
        enc = dm.atom_attention_encoder
        proj_d = enc.linear_no_bias_z(enc.layernorm_z(pair_z))                                                      # [N, N, C_AP]
        # ---------------------------------------------------------------- sharded: this rank's rows through tp_diffusion
        P, r = tp._group()
        lay = TD.struct_layout(N_ST)
        facts.update(P=P, rank=r, R=lay.R, r0=lay.r0, layout=lay.facts())
        src = {"hostpair": _HostPair(z_struct.clone()), "host_full": z_struct.clone(), "host_rows": z_struct[lay.r0:lay.r1].clone(),
               "kit_wrappers": tp.RowShard(z_struct[lay.r0:lay.r1].clone(), lay)}[src_kind]
        z0, q0 = int(tp.STATS.get("zcond_rows", 0)), int(tp.STATS.get("dit_local_queries", 0))
        guard = _pair_guard(N_ST)
        if src_kind == "kit_wrappers":                                             # the KIT's stage wrappers (tp._make_diffusion_cache_rows / tp._make_sample_rows around the stub's
            class _Cfg(dict):                                                      # stage methods): the statements a rank executes in production, incl. the marks and the cache release
                __getattr__ = dict.get                                             # the engine's config nodes answer both attribute and .get access
            tp.STATS["n_gpu"], tp.STATS["group"] = P, True
            model.configs.sample_diffusion = _Cfg(N_sample=2, guidance=None)       # tp._n_sample_per_call reads N_sample (the stub's chunk size >= 2: one denoiser call carries both samples)
            released, release_orig = [], TD.release
            TD.release = released.append                                           # the roll-out's release is DEFERRED to the end of this entry so the rows can be compared first
            prep, samp = tp._make_diffusion_cache_rows(Model.prepare_diffusion_cache_for_sampling), tp._make_sample_rows(Model.run_sample_diffusion_stage)
            feats_w = dict(feats)                                                  # the wrapper cuts the structural extra bias to this rank's rows IN the dict it is handed (production: the
            try:                                                                   # model's own structural feature dict); the dense checks below keep the whole one
                with guard:
                    cache_s = prep(model, input_feature_dict=feats_w, z=src)
                    ctx = cache_s["pair_z"]
                    coords_s = samp(model, **_stage_kw(feats_w, s_inputs, s, src, cache_s))
            finally:
                TD.release = release_orig
            chk["kit_wrapper_released_ctx"] = released == [ctx]
        else:
            with guard:
                ctx = TD.prime(model, feats, src, n_sample=2)
                cache_s = TD.prepare_cache_sharded(model, feats, ctx)
                coords_s = TD.run_sample_diffusion_stage(model, Model.run_sample_diffusion_stage, ctx, **_stage_kw(feats, s_inputs, s, src, cache_s))
        facts["pair_shaped_allocations"] = guard.offenders[:5]
        facts["max_op_numel"], facts["max_op"] = guard.max_numel, guard.max_shape
        chk["never_whole_pair"] = len(guard.offenders) == 0
        chk["max_numel_lt_N2x16"] = guard.max_numel < N_ST * N_ST * 16
        chk["denoiser_hook_removed"] = "forward" not in dm.__dict__
        # ---------------------------------------------------------------- rows vs dense slices
        r0, r1 = lay.r0, lay.r1
        cmp("zcond_rows", ctx.zc, pair_z[r0:r1])
        for i in range(N_BLK):
            got = ctx.cache.store[i] if (ctx.cache.enabled and i in ctx.cache.store) else DF.pair_bias_rows(ctx.blocks[i].bias, ctx.zc, lay)
            cmp(f"bias_rows_block{i}", got, bias_d[i][:, r0:r1, :])
        idx_q, idx_k, _ = rearrange_qk_to_dense_trunk(feats["atom_to_token_idx"], feats["atom_to_token_idx"], -1, -1, NQ, NK, compute_mask=False)
        valid = feats["pad_info"]["mask_trunked"].bool()
        term = TD.token_pair_term(enc, feats, ctx)
        ref_term = proj_d[idx_q.long()[:, :, None], idx_k.long()[:, None, :]]
        cmp("band_term_valid_slots", term[valid], ref_term[valid])
        p_d, p_s = cache_d["p_lm/c_l"][0], cache_s["p_lm/c_l"][0]
        cmp("p_lm_valid_slots", p_s[..., valid, :], p_d[..., valid, :])
        cmp("c_l", cache_s["p_lm/c_l"][1], cache_d["p_lm/c_l"][1])
        cmp("coordinates", coords_s, coords_d)
        facts["coord_absmax"] = float(coords_d.abs().max())
        # ---------------------------------------------------------------- census
        n_calls = 2                                                                                                   # 2 steps x (2 samples in one chunk)
        chk["census_zcond_rows"] = int(tp.STATS["zcond_rows"]) - z0 == lay.R and TD.STATS["zcond_rows"] >= lay.R
        chk["census_dit_local_queries"] = int(tp.STATS["dit_local_queries"]) - q0 == lay.R * N_BLK * n_calls
        chk["census_denoise_calls"] = ctx.denoise_calls == n_calls
        fl = dict(TD.fields())
        facts["fields"] = {k: (v if isinstance(v, (int, float, str, type(None))) else str(v)) for k, v in fl.items()}
        chk["census_named"] = (fl["dit_bias"] == ("fused_conv" if fusion else "ln_linear") and fl["dit_extra_bias"] == ("rows" if src_kind == "kit_wrappers" else "full")
                               and fl["zstruct_src"] == {"hostpair": "hostpair_full", "host_full": "host_full", "host_rows": "host_rows", "kit_wrappers": "host_rows"}[src_kind]   # a RowShard on CPU ranks reads host_rows (device_rows on a card)
                               and fl["diff_stage"] == "rows" and "atom_encoder" in fl["diff_replicated_by_design"])
        chk["schedule_recorded"] = "cond:" in str(fl.get("diff_rows_source")) and fl.get("diff_bias_cache") in ("on", "off") and int(fl.get("diff_cond_rows", 0)) >= 1
        # ---------------------------------------------------------------- extra bias handed as rows == handed whole
        feats_rows = dict(feats)
        TD.shard_extra_bias(feats_rows, lay)
        chk["shard_extra_bias_rows"] = tuple(feats_rows[TD.EXTRA_BIAS_KEY].shape) == (lay.R, N_ST) and feats_rows[TD.EXTRA_BIAS_ROWS_KEY] is True
        fns_rows = TD.dit_block_fns(dm.diffusion_transformer.blocks[0], lay, extra=feats_rows[TD.EXTRA_BIAS_KEY], fusion=fusion, normalize=dm.normalize)
        a0 = torch.randn((2, N_ST, C_TOK), generator=torch.Generator().manual_seed(5))
        s0 = torch.randn((2, N_ST, C_S), generator=torch.Generator().manual_seed(6))
        b0 = DF.pair_bias_rows(ctx.blocks[0].bias, ctx.zc, lay)                                                      # [H, R, N]
        o_full = DF.dit_block_sharded(ctx.blocks[0], a0, s0, b0, lay, q_rows=None)
        o_rows = DF.dit_block_sharded(fns_rows, a0, s0, b0, lay, q_rows=None)
        bit["extra_bias_rows_vs_full"] = chk["extra_bias_rows_vs_full"] = bool(torch.equal(o_full, o_rows))
        o_dense = dm.diffusion_transformer.blocks[0](a0, s0, (torch.permute(dm.normalize(pair_z), (2, 0, 1)).contiguous() if fusion else pair_z),
                                                     extra_attn_bias=feats["structural_pair_attn_bias"], enable_efficient_fusion=fusion)
        cmp("dit_block0_vs_dense", o_full, o_dense)
        # ---------------------------------------------------------------- refusals by name
        def refuses(fn, *needles):
            try:
                fn()
            except RowpairRefused as e:
                return all(n in str(e) for n in needles) or f"wrong message: {e}"
            return False
        chk["refuses_fusion_mismatch"] = refuses(lambda: TD.denoise_sharded(dm, ctx, x_noisy=coords_d, t_hat_noise_level=torch.ones(2), input_feature_dict=feats,
                                                                              s_inputs=s_inputs, s_trunk=s, p_lm=p_s, c_l=cache_s["p_lm/c_l"][1],
                                                                              enable_efficient_fusion=not fusion), "enable_efficient_fusion") is True
        chk["refuses_pair_z_clone"] = refuses(lambda: ctx.clone(), "pair_z.clone") is True
        chk["refuses_bad_source_shape"] = refuses(lambda: TD.prime(model, feats, z_struct[:5].clone(), lay=lay), "structural pair source") is True
        os.environ["OPENDDE_FOLDCP_MODE"] = "distributed"
        chk["refuses_foldcp_variant"] = refuses(lambda: TD.prime(model, feats, src, lay=lay), "OPENDDE_FOLDCP_MODE") is True
        os.environ.pop("OPENDDE_FOLDCP_MODE")
        # ---------------------------------------------------------------- the draw guard: identical passes, a divergent rank is refused BY NAME
        same = torch.full((7, 3), 2.5)
        chk["guard_passes_identical"] = DF.sync_replicated(same, "diffusion.x_noisy", mode="guard") is same
        mine = torch.randn((7, 3), generator=torch.Generator().manual_seed(100 + r))
        chk["guard_refuses_divergent_by_name"] = refuses(lambda: DF.sync_replicated(mine, "diffusion.x_noisy", mode="guard"), "diffusion.x_noisy", "differs across ranks") is True
        TD.release(ctx)
        chk["released"] = ctx.zc is None and len(ctx.cache.store) == 0
    # -------------------------------------------------------------------- every rank's verdict reaches rank 0
    import torch.distributed as tdist
    allres = [None] * P
    tdist.all_gather_object(allres, res)
    out = dict(res)
    out["all_ok"] = all(all(v is True for v in x["checks"].values()) for x in allres)
    out["failed"] = sorted({f"rank{i}:{k}={v}" for i, x in enumerate(allres) for k, v in x["checks"].items() if v is not True})
    out["bitwise_all_ranks"] = {k: all(x["bitwise"].get(k) for x in allres) for k in res["bitwise"]}
    out["metrics_max"] = {k: max(float(x["metrics"].get(k, 0.0)) for x in allres) for k in res["metrics"]}
    return out


def _mp(P, *args):
    from opendde_opt.tests.conftest import run_sharded_or_skip
    return run_sharded_or_skip(P, _rank_entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=900)


@pytest.mark.parametrize("P,fusion,src_kind", [(2, False, "hostpair"), (3, False, "host_full"), (2, True, "host_rows"), (3, True, "hostpair"), (2, True, "kit_wrappers"), (3, False, "kit_wrappers")])
def test_mp_struct_diffusion_rows_vs_dense(P, fusion, src_kind):
    res = _mp(P, 1, fusion, src_kind)
    print(f"\nP={P} fusion={fusion} src={src_kind}: facts={res['facts']}\n  metrics_max={res['metrics_max']}\n  bitwise_all_ranks={res['bitwise_all_ranks']}\n  failed={res['failed']}")
    assert res["all_ok"], res["failed"]


def test_p1_refuses_by_name():
    """At --n_gpu 1 the engine's own diffusion module runs: the stage's entry refuses a P = 1 layout by name (never a silent dense fallback)."""
    install_modules()
    from opendde_opt import tp_diffusion as TD
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    model, feats, s_inputs, s, z_struct = build(1, False)
    with pytest.raises(RowpairRefused, match="n_gpu=1|P=1|P == 1|single rank|not sharded"):
        TD.prime(model, feats, z_struct, lay=D.Layout(N_ST, 1, 0))


def test_module_is_engine_free_at_import():
    """Importing the stage imports neither torch.distributed users nor the engine (lazy imports; the kit's --help path stays light)."""
    before = set(sys.modules)
    import importlib
    importlib.import_module("opendde_opt.tp_diffusion")
    fresh = set(sys.modules) - before
    assert not any(m.startswith("opendde.") for m in fresh), sorted(m for m in fresh if m.startswith("opendde"))