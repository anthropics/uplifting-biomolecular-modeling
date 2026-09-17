"""The diffusion seam's binding (``protenix_opt.tp_bind.diffusion``) through the core at P in {2, 3} on CPU (gloo ranks spawned by
``opt_core.mem.rowpair.launch.run_sharded``): a synthetic Protenix-shaped diffusion module (this file's stub: the attribute paths and the
stock statements the binding calls, small random fp32 weights identical on every rank) runs (a) DENSE — the stub's own whole-tensor
``prepare_cache`` / ``sample_diffusion`` — and (b) through the binding on this rank's pair rows; the conditioned pair shard equals the dense
rows, the sampled coordinates equal the dense ones (fp32, max|diff| <= tolerance; bitwise reported), no aten op inside the binding ever
yields a tensor with two ``N``-sized dims (nothing pair-shaped is whole on a rank), the schedule census carries the diffusion words, the
binding's ``report()`` states the regime ``protenix_opt.report.unit_record`` reads, and the per-rank RNG streams (seeded apart on purpose) are
rank 0's after the binding's sync. Plus: the source census of ``tp_bind`` (imports only the core's rowpair package, torch and protenix; no
collective and no row loop of its own) and the by-name refusals (P == 1 layout, missing skip_amp, sampler hook, pair cache off)."""
import ast
import json
import math
import os
import sys
import types

import pytest

from protenix_opt.tests.conftest import needs_gloo

HERE = os.path.dirname(os.path.abspath(__file__))
BIND_DIR = os.path.join(os.path.dirname(HERE), "tp_bind")
TOL = 1e-4                     # fp32, 3 sampler steps x 2 blocks of accumulated GEMM-order differences (row blocks vs whole); observed values are printed
SEED = 1234


def _torch():
    return pytest.importorskip("torch")


def _core():
    return pytest.importorskip("opt_core.mem.rowpair.diffusion")


# ================================================================================================================ source census
def _tree(name):
    return ast.parse(open(os.path.join(BIND_DIR, name)).read())


def test_source_census_bind_imports_only_core_torch_protenix():
    for name in ("diffusion.py", "relpos.py", "__init__.py"):
        for node in ast.walk(_tree(name)):
            if isinstance(node, ast.Import):
                for a in node.names:
                    assert a.name in ("os", "time", "torch", "torch.nn.functional"), (name, a.name)   # stdlib os/time + torch: the rule keeps ENGINE modules out
            elif isinstance(node, ast.ImportFrom):
                mod = node.module or ""
                ok = (node.level == 1 and mod == "relpos") or mod == "typing" or mod.startswith("opt_core.mem.rowpair") or mod.startswith("protenix.")
                assert ok, (name, node.level, mod)
                assert not mod.startswith("torch.distributed"), (name, mod)


def test_source_census_no_row_loop_or_collective_of_its_own():
    banned_calls = {"iter_row_blocks", "row_blocks", "chunks", "local_blocks", "all_gather", "all_gather_rows", "all_gather_into_rows", "all_reduce",
                    "allreduce", "broadcast", "transpose_blocks", "init_process_group"}
    for name in ("diffusion.py", "relpos.py"):
        t = _tree(name)
        for node in ast.walk(t):
            if isinstance(node, ast.Call):
                fn = node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", "")
                assert fn not in banned_calls, (name, fn, node.lineno)
            if isinstance(node, ast.Attribute) and node.attr == "distributed" and isinstance(node.value, ast.Name) and node.value.id == "torch":
                raise AssertionError(f"{name}:{node.lineno}: torch.distributed")
        src = open(os.path.join(BIND_DIR, name)).read()
        assert "PTX_TP_DIFF_REPLICATE_BELOW" not in src and "GATHER_BELOW" not in src           # no size gate words at all


# ================================================================================================================ the stub engine (Protenix-shaped)
def _utils():
    """``protenix.model.utils`` / ``protenix.utils.torch_utils`` / ``protenix.model.modules.primitives`` functions the binding imports, with
    the stock statements; registered under those module names when the real package is absent (CPU test environment)."""
    torch = _torch()
    import torch.nn.functional as F
    from contextlib import nullcontext

    def permute_final_dims(tensor, inds):
        zero_index = -1 * len(inds)
        first_inds = list(range(len(tensor.shape[:zero_index])))
        return tensor.permute(first_inds + [zero_index + i for i in inds])

    def flatten_final_dims(t, num_dims):
        return t.reshape(t.shape[:-num_dims] + (-1,))

    def expand_at_dim(x, dim, n):
        x = x.unsqueeze(dim=dim)
        if dim < 0:
            dim = x.dim() + dim
        before_shape = x.shape[:dim]
        after_shape = x.shape[dim + 1:]
        return x.expand(*before_shape, n, *after_shape)

    def autocasting_disable_decorator(disable_casting):
        def func_wrapper(func):
            def new_func(*args, **kwargs):
                ctx = torch.autocast(device_type="cuda", enabled=False) if disable_casting else nullcontext()
                cast = (lambda t: t.to(dtype=torch.float32) if (disable_casting and isinstance(t, torch.Tensor) and torch.is_floating_point(t)) else t)
                with ctx:
                    return func(*[cast(a) for a in args], **{k: cast(v) for k, v in kwargs.items()})
            return new_func
        return func_wrapper

    def _attention(q, k, v, attn_bias=None, use_efficient_implementation=True, inplace_safe=False):
        assert k.shape == v.shape
        input_dtype = q.dtype
        q = q.to(dtype=torch.float32)
        k = k.to(dtype=torch.float32)
        if attn_bias is not None:
            attn_bias = attn_bias.to(dtype=torch.float32)
        if use_efficient_implementation:
            return F.scaled_dot_product_attention(query=q, key=k, value=v, attn_mask=attn_bias, scale=1.0)
        k = k.transpose(-1, -2)
        w = q @ k
        if attn_bias is not None:
            w = w + attn_bias
        w = F.softmax(w, dim=-1)
        return w.to(dtype=input_dtype) @ v

    def rearrange_qk_to_dense_trunk(q, k, dim_q, dim_k, n_queries=32, n_keys=128, compute_mask=True):
        """1-D index vectors only (``atom_to_token_idx``): q blocks of ``n_queries`` consecutive atoms, k windows of ``n_keys`` centred on the
        block (left pad ``(n_keys - n_queries) // 2``), zero padding; ``pad_info['mask_trunked'][b, i, j]`` = both atoms real."""
        assert q.dim() == 1 and k.dim() == 1 and dim_q == -1 and dim_k == -1
        n = int(q.shape[0])
        n_tr = (n + n_queries - 1) // n_queries
        pad_q = n_tr * n_queries - n
        padl = (n_keys - n_queries) // 2
        padr = pad_q + (n_keys - n_queries) - padl
        q_t = F.pad(q, (0, pad_q)).reshape(n_tr, n_queries)
        k_t = F.pad(k, (padl, padr)).unfold(0, n_keys, n_queries)[:n_tr]
        info = {}
        if compute_mask:
            qr = F.pad(torch.ones(n, dtype=torch.bool), (0, pad_q)).reshape(n_tr, n_queries)
            kr = F.pad(torch.ones(n, dtype=torch.bool), (padl, padr)).unfold(0, n_keys, n_queries)[:n_tr]
            info["mask_trunked"] = (qr[:, :, None] & kr[:, None, :]).to(torch.float32)
        return q_t, k_t, info

    def broadcast_token_to_local_atom_pair(z_token, atom_to_token_idx, n_queries, n_keys, compute_mask=True):
        iq, ik, info = rearrange_qk_to_dense_trunk(atom_to_token_idx, atom_to_token_idx, -1, -1, n_queries, n_keys, compute_mask)
        return z_token[..., iq[:, :, None], ik[:, None, :], :], info

    return types.SimpleNamespace(permute_final_dims=permute_final_dims, flatten_final_dims=flatten_final_dims, expand_at_dim=expand_at_dim,
                                 autocasting_disable_decorator=autocasting_disable_decorator, _attention=_attention,
                                 rearrange_qk_to_dense_trunk=rearrange_qk_to_dense_trunk, broadcast_token_to_local_atom_pair=broadcast_token_to_local_atom_pair)


def _install_protenix_stub():
    """Register the stub under the protenix module names the binding imports lazily (only when the real package is not importable)."""
    try:
        import protenix.model.modules.primitives as _p  # noqa: F401
        if hasattr(_p, "_attention"):
            return "real"
    except Exception:  # noqa: BLE001
        pass
    U = _utils()
    mods = {}
    for name in ("protenix", "protenix.model", "protenix.model.modules", "protenix.utils"):
        m = types.ModuleType(name); m.__path__ = []; mods[name] = m
    prim = types.ModuleType("protenix.model.modules.primitives")
    prim._attention, prim.rearrange_qk_to_dense_trunk, prim.broadcast_token_to_local_atom_pair = U._attention, U.rearrange_qk_to_dense_trunk, U.broadcast_token_to_local_atom_pair
    mu = types.ModuleType("protenix.model.utils")
    mu.permute_final_dims, mu.expand_at_dim, mu.flatten_final_dims = U.permute_final_dims, U.expand_at_dim, U.flatten_final_dims
    tu = types.ModuleType("protenix.utils.torch_utils"); tu.autocasting_disable_decorator = U.autocasting_disable_decorator
    mods.update({"protenix.model.modules.primitives": prim, "protenix.model.utils": mu, "protenix.utils.torch_utils": tu})
    sys.modules.update(mods)
    return "stub"


def _build(N_tok: int, *, n_blocks=2, fusion=False, N_step=3, N_sample=2, seed=7):
    """The Protenix-shaped stub model + features (identical on every rank: one seeded generator)."""
    torch = _torch()
    import torch.nn as nn
    import torch.nn.functional as F
    _install_protenix_stub()
    from protenix.model.modules.primitives import _attention, broadcast_token_to_local_atom_pair, rearrange_qk_to_dense_trunk
    from protenix.model.utils import expand_at_dim, flatten_final_dims, permute_final_dims
    from protenix.utils.torch_utils import autocasting_disable_decorator
    g = torch.Generator().manual_seed(seed)
    c_z, c_s, c_si, c_a, H, c_atom, C_pair, nq, nk, r_max, s_max, c_noise = 8, 12, 10, 16, 4, 8, 4, 4, 8, 4, 1, 6

    def init_(mod):
        with torch.no_grad():
            for p in mod.parameters():
                p.copy_(torch.randn(p.shape, generator=g) * (0.3 if p.dim() > 1 else 0.1) + (1.0 if p.dim() == 1 and p.shape[0] > 0 and getattr(p, "_is_scale", False) else 0.0))
        return mod

    class LinearNoBias(nn.Linear):
        def __init__(self, i, o, precision=None, **_):
            super().__init__(i, o, bias=False); self.precision = precision

    def LayerNorm(c, create_scale=True, create_offset=True):
        if not create_scale and not create_offset:
            return nn.LayerNorm(c, elementwise_affine=False)
        m = nn.LayerNorm(c, bias=create_offset)
        m.weight._is_scale = True
        return m

    class Transition(nn.Module):
        def __init__(self, c, n=2):
            super().__init__(); self.ln = LayerNorm(c, create_offset=False); self.a = LinearNoBias(c, n * c); self.b = LinearNoBias(c, n * c); self.o = LinearNoBias(n * c, c)
        def forward(self, x):
            x = self.ln(x); return self.o(F.silu(self.a(x)) * self.b(x))

    class AdaLN(nn.Module):
        def __init__(self, c_a_, c_s_):
            super().__init__(); self.ln_a = nn.LayerNorm(c_a_, elementwise_affine=False); self.ln_s = LayerNorm(c_s_, create_offset=False)
            self.lin_s = nn.Linear(c_s_, c_a_); self.lin_nb = LinearNoBias(c_s_, c_a_)
        def forward(self, a, s):
            a = self.ln_a(a); s = self.ln_s(s); return torch.sigmoid(self.lin_s(s)) * a + self.lin_nb(s)

    class Attention(nn.Module):
        def __init__(self):
            super().__init__(); self.num_heads, self.c_hidden = H, c_a // H; self.use_efficient_implementation = True
            self.linear_q = nn.Linear(c_a, c_a); self.linear_k = LinearNoBias(c_a, c_a); self.linear_v = LinearNoBias(c_a, c_a)
            self.linear_g = LinearNoBias(c_a, c_a); self.linear_o = LinearNoBias(c_a, c_a); self.sigmoid = nn.Sigmoid()
        def _prep_qkv(self, q_x, kv_x, apply_scale=True):
            q, k, v = self.linear_q(q_x), self.linear_k(kv_x), self.linear_v(kv_x)
            q = q.view(q.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)
            k = k.view(k.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)
            v = v.view(v.shape[:-1] + (self.num_heads, -1)).transpose(-2, -3)
            if apply_scale:
                q = q / math.sqrt(self.c_hidden)
            return q, k, v
        def _wrap_up(self, o, q_x):
            gg = self.sigmoid(self.linear_g(q_x)); gg = gg.view(gg.shape[:-1] + (self.num_heads, -1)); o = o * gg
            return self.linear_o(flatten_final_dims(o, num_dims=2))
        def forward(self, q_x, kv_x, attn_bias=None, inplace_safe=False, **_):
            q, k, v = self._prep_qkv(q_x, kv_x, True)
            if attn_bias is not None and len(attn_bias.shape) != len(q.shape):
                attn_bias = attn_bias.unsqueeze(-3)
            o = _attention(q, k, v, attn_bias, use_efficient_implementation=self.use_efficient_implementation, inplace_safe=inplace_safe)
            return self._wrap_up(o.transpose(-2, -3), q_x)

    class AttentionPairBias(nn.Module):
        def __init__(self):
            super().__init__(); self.n_heads, self.has_s = H, True; self.layernorm_a = AdaLN(c_a, c_s); self.attention = Attention()
            self.layernorm_z = LayerNorm(c_z, create_offset=False); self.linear_nobias_z = LinearNoBias(c_z, H); self.linear_a_last = nn.Linear(c_s, c_a)
        def forward(self, a, s, z, inplace_safe=False, enable_efficient_fusion=False, **_):
            a = self.layernorm_a(a, s)
            if enable_efficient_fusion:
                weight = (self.linear_nobias_z.weight * self.layernorm_z.weight[None, :])[:, :, None, None]
                bias = F.conv2d(z, weight)
            else:
                bias = permute_final_dims(self.linear_nobias_z(self.layernorm_z(z)), [2, 0, 1])
            a = self.attention(q_x=a, kv_x=a, attn_bias=bias, inplace_safe=inplace_safe)
            if inplace_safe:
                a *= torch.sigmoid(self.linear_a_last(s))
            else:
                a = torch.sigmoid(self.linear_a_last(s)) * a
            return a

    class CondTransition(nn.Module):
        def __init__(self):
            super().__init__(); self.adaln = AdaLN(c_a, c_s); self.a1 = LinearNoBias(c_a, 2 * c_a); self.a2 = LinearNoBias(c_a, 2 * c_a)
            self.b = LinearNoBias(2 * c_a, c_a); self.s = nn.Linear(c_s, c_a)
        def forward(self, a, s):
            a = self.adaln(a, s); b = F.silu(self.a1(a)) * self.a2(a); return torch.sigmoid(self.s(s)) * self.b(b)

    class Block(nn.Module):
        def __init__(self):
            super().__init__(); self.attention_pair_bias = AttentionPairBias(); self.conditioned_transition_block = CondTransition()
        def forward(self, a, s, z, inplace_safe=False, enable_efficient_fusion=False, **_):
            attn_out = self.attention_pair_bias(a=a, s=s, z=z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
            if inplace_safe:
                attn_out += a
            else:
                attn_out = attn_out + a
            ff_out = self.conditioned_transition_block(a=attn_out, s=s)
            return ff_out + attn_out, s, z

    class DiT(nn.Module):
        def __init__(self):
            super().__init__(); self.blocks = nn.ModuleList([Block() for _ in range(n_blocks)])
        def forward(self, a, s, z, inplace_safe=False, chunk_size=None, enable_efficient_fusion=False):
            for b in self.blocks:
                a, s, z = b(a, s, z, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
            return a

    class RelPE(nn.Module):
        def __init__(self):
            super().__init__(); self.r_max, self.s_max, self.c_z = r_max, s_max, c_z; self.linear_no_bias = LinearNoBias(4 * r_max + 2 * s_max + 7, c_z)
        def forward(self, relp):
            return self.linear_no_bias(relp)
        def generate_relp(self, f):                                   # embedders.py 150-215 (dense; the reference)
            same_chain = (f["asym_id"][:, None] == f["asym_id"][None, :]).long()
            same_res = (f["residue_index"][:, None] == f["residue_index"][None, :]).long()
            same_ent = (f["entity_id"][:, None] == f["entity_id"][None, :]).long()
            d_res = torch.clip(f["residue_index"][:, None] - f["residue_index"][None, :] + r_max, 0, 2 * r_max) * same_chain + (1 - same_chain) * (2 * r_max + 1)
            a_pos = F.one_hot(d_res, 2 * (r_max + 1))
            d_tok = torch.clip(f["token_index"][:, None] - f["token_index"][None, :] + r_max, 0, 2 * r_max) * same_chain * same_res + (1 - same_chain * same_res) * (2 * r_max + 1)
            a_tok = F.one_hot(d_tok, 2 * (r_max + 1))
            d_chain = torch.clip(f["sym_id"][:, None] - f["sym_id"][None, :] + s_max, 0, 2 * s_max) * same_ent + (1 - same_ent) * (2 * s_max + 1)
            a_chain = F.one_hot(d_chain, 2 * (s_max + 1))
            return torch.cat([a_pos, a_tok, same_ent[..., None], a_chain], dim=-1).float()

    class Fourier(nn.Module):
        def __init__(self):
            super().__init__(); self.w = nn.Parameter(torch.zeros(c_noise)); self.b = nn.Parameter(torch.zeros(c_noise))
        def forward(self, t_hat_noise_level):
            return torch.cos(2 * math.pi * (t_hat_noise_level.unsqueeze(-1) * self.w + self.b))

    class Conditioning(nn.Module):
        def __init__(self):
            super().__init__(); self.sigma_data, self.c_z, self.c_s = 16.0, c_z, c_s; self.relpe = RelPE()
            self.layernorm_z = LayerNorm(2 * c_z, create_offset=False); self.linear_no_bias_z = LinearNoBias(2 * c_z, c_z, precision=torch.float32)
            self.transition_z1, self.transition_z2 = Transition(c_z), Transition(c_z)
            self.layernorm_s = LayerNorm(c_s + c_si, create_offset=False); self.linear_no_bias_s = LinearNoBias(c_s + c_si, c_s)
            self.fourier_embedding = Fourier(); self.layernorm_n = LayerNorm(c_noise, create_offset=False); self.linear_no_bias_n = LinearNoBias(c_noise, c_s)
            self.transition_s1, self.transition_s2 = Transition(c_s), Transition(c_s)
        def prepare_cache(self, relp_feature, z_trunk, inplace_safe=False):   # diffusion.py 86-107
            pair_z = torch.cat([z_trunk, self.relpe(relp_feature)], dim=-1)
            pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
            if inplace_safe:
                pair_z += self.transition_z1(pair_z); pair_z += self.transition_z2(pair_z)
            else:
                pair_z = pair_z + self.transition_z1(pair_z); pair_z = pair_z + self.transition_z2(pair_z)
            return pair_z
        def forward(self, t_hat_noise_level, relp_feature, s_inputs, s_trunk, z_trunk, pair_z, inplace_safe=False, use_conditioning=True):
            if pair_z is None:
                if not use_conditioning:
                    s_trunk, z_trunk = 0 * s_trunk, 0 * z_trunk
                pair_z = self.prepare_cache(relp_feature, z_trunk, inplace_safe)
            elif inplace_safe:
                pair_z = pair_z.clone()
            single_s = self.linear_no_bias_s(self.layernorm_s(torch.cat([s_trunk, s_inputs], dim=-1)))
            noise_n = self.fourier_embedding(t_hat_noise_level=torch.log(t_hat_noise_level / self.sigma_data) / 4).to(single_s.dtype)
            single_s = single_s.unsqueeze(-3) + self.linear_no_bias_n(self.layernorm_n(noise_n)).unsqueeze(-2)
            single_s = single_s + self.transition_s1(single_s); single_s = single_s + self.transition_s2(single_s)
            return single_s, pair_z

    class Encoder(nn.Module):
        def __init__(self):
            super().__init__(); self.has_coords, self.c_atompair, self.c_z, self.n_queries, self.n_keys = True, C_pair, c_z, nq, nk
            self.linear_no_bias_ref_pos = LinearNoBias(3, c_atom); self.linear_no_bias_d = LinearNoBias(3, C_pair); self.linear_no_bias_v = LinearNoBias(1, C_pair)
            self.layernorm_z = LayerNorm(c_z, create_offset=False); self.linear_no_bias_z = LinearNoBias(c_z, C_pair)
            self.layernorm_s = LayerNorm(c_s, create_offset=False); self.linear_no_bias_s = LinearNoBias(c_s, c_atom); self.linear_no_bias_r = LinearNoBias(3, c_atom)
            self.linear_q2t = LinearNoBias(c_atom, c_a)
        def prepare_cache(self, ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, atom_to_token_idx, d_lm, v_lm, pad_info, r_l=None, z=None, inplace_safe=False):
            c_l = self.linear_no_bias_ref_pos(ref_pos) * ref_mask.unsqueeze(-1)
            p_lm = (self.linear_no_bias_d(d_lm) * v_lm) * pad_info["mask_trunked"].unsqueeze(-1)
            p_lm = p_lm + self.linear_no_bias_v(v_lm.to(dtype=p_lm.dtype))
            if r_l is not None:                                                # transformer.py 806-817
                p_lm = p_lm.unsqueeze(dim=-5) + broadcast_token_to_local_atom_pair(z_token=self.linear_no_bias_z(self.layernorm_z(z)), atom_to_token_idx=atom_to_token_idx,
                                                                                   n_queries=self.n_queries, n_keys=self.n_keys, compute_mask=False)[0]
            return p_lm, c_l
        def forward(self, atom_to_token_idx, ref_pos, ref_charge, ref_mask, ref_atom_name_chars, ref_element, d_lm, v_lm, pad_info, r_l=None, s=None, z=None,
                    p_lm=None, c_l=None, inplace_safe=False, chunk_size=None):
            assert z is not None
            if p_lm is None or c_l is None:
                p_lm, c_l = self.prepare_cache(ref_pos, ref_charge, ref_mask, ref_element, ref_atom_name_chars, atom_to_token_idx, d_lm, v_lm, pad_info, r_l=r_l, z=z)
            n_atom = int(atom_to_token_idx.shape[-1])
            q_l = c_l + self.linear_no_bias_s(self.layernorm_s(s))[..., atom_to_token_idx, :] + self.linear_no_bias_r(r_l)      # [Ns, N_atom, c_atom]
            ar = torch.arange(n_atom)
            _, ik, info = rearrange_qk_to_dense_trunk(ar, ar, -1, -1, self.n_queries, self.n_keys, compute_mask=True)
            valid = info["mask_trunked"] > 0
            logits = p_lm.sum(-1).masked_fill(~valid, -1e9)                          # [1, n_tr, nq, nk]: the pair term steers a local attention
            w = torch.softmax(logits, dim=-1)
            vals = q_l[..., ik, :]                                                     # [Ns, n_tr, nk, c_atom]
            o = torch.einsum("sbqk,sbkc->sbqc", w.expand(q_l.shape[0], -1, -1, -1), vals).reshape(q_l.shape[0], -1, q_l.shape[-1])[:, :n_atom]
            a_atom = q_l + o
            n_tok = int(s.shape[-2])
            up = self.linear_q2t(a_atom)                                               # [Ns, N_atom, c_a]
            a_token = torch.zeros(up.shape[0], n_tok, up.shape[-1], dtype=up.dtype).index_add_(1, atom_to_token_idx, up)
            cnt = torch.zeros(n_tok, dtype=up.dtype).index_add_(0, atom_to_token_idx, torch.ones(n_atom, dtype=up.dtype)).clamp(min=1)
            return a_token / cnt[None, :, None], q_l, c_l, p_lm

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__(); self.lin_a = LinearNoBias(c_a, c_atom); self.lin_out = LinearNoBias(c_atom, 3)
        def forward(self, atom_to_token_idx, a, q_skip, c_skip, p_skip, inplace_safe=False, chunk_size=None):
            return self.lin_out(F.silu(self.lin_a(a)[..., atom_to_token_idx, :] + q_skip))

    class DiffusionModule(nn.Module):
        def __init__(self):
            super().__init__(); self.sigma_data, self.c_z, self.c_atompair = 16.0, c_z, C_pair
            self.diffusion_conditioning, self.atom_attention_encoder, self.diffusion_transformer, self.atom_attention_decoder = Conditioning(), Encoder(), DiT(), Decoder()
            self.layernorm_s = LayerNorm(c_s, create_offset=False); self.linear_no_bias_s = LinearNoBias(c_s, c_a)
            self.layernorm_a = LayerNorm(c_a, create_offset=False); self.normalize = LayerNorm(c_z, create_scale=False, create_offset=False)
        def f_forward(self, r_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, inplace_safe=False, chunk_size=None,
                      use_conditioning=True, enable_efficient_fusion=False):                      # diffusion.py 392-500
            f = input_feature_dict
            s_single, z_pair = self.diffusion_conditioning(t_hat_noise_level, f["relp"], s_inputs, s_trunk, z_trunk, pair_z, inplace_safe=inplace_safe, use_conditioning=use_conditioning)
            s_trunk = expand_at_dim(s_trunk, dim=-3, n=1)
            z_pair = expand_at_dim(z_pair, dim=-4, n=1)
            a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(f["atom_to_token_idx"], f["ref_pos"], f["ref_charge"], f["ref_mask"], f["ref_atom_name_chars"], f["ref_element"],
                                                                          f["d_lm"], f["v_lm"], f["pad_info"], r_l=r_noisy, s=s_trunk, z=z_pair, p_lm=p_lm, c_l=c_l,
                                                                          inplace_safe=inplace_safe, chunk_size=chunk_size)
            a_token = a_token.to(dtype=torch.float32)
            a_token = a_token + self.linear_no_bias_s(self.layernorm_s(s_single))
            if enable_efficient_fusion:
                z = permute_final_dims(self.normalize(z_pair.to(dtype=torch.float32)), [2, 0, 1]).contiguous()
            else:
                z = z_pair.to(dtype=torch.float32)
            a_token = self.diffusion_transformer(a=a_token.to(torch.float32), s=s_single.to(torch.float32), z=z, inplace_safe=inplace_safe, chunk_size=chunk_size,
                                                 enable_efficient_fusion=enable_efficient_fusion)
            a_token = self.layernorm_a(a_token)
            return self.atom_attention_decoder(atom_to_token_idx=f["atom_to_token_idx"], a=a_token, q_skip=q_skip, c_skip=c_skip, p_skip=p_skip, inplace_safe=inplace_safe)
        def forward(self, x_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, inplace_safe=False, chunk_size=None,
                    use_conditioning=True, enable_efficient_fusion=False):                        # diffusion.py 540-600
            r_noisy = x_noisy / torch.sqrt(self.sigma_data ** 2 + t_hat_noise_level ** 2)[..., None, None]
            r_update = self.f_forward(r_noisy, t_hat_noise_level, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, inplace_safe, chunk_size, use_conditioning, enable_efficient_fusion)
            s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(r_update.dtype)
            return 1 / (1 + s_ratio ** 2) * x_noisy + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio ** 2) * r_update

    def generator_sample_diffusion(denoise_net, input_feature_dict, s_inputs, s_trunk, z_trunk, pair_z, p_lm, c_l, noise_schedule, N_sample=1, gamma0=0.8, gamma_min=1.0,
                                   noise_scale_lambda=1.003, step_scale_eta=1.5, diffusion_chunk_size=None, inplace_safe=False, attn_chunk_size=None,
                                   enable_efficient_fusion=False, guidance_configs=None):          # generator.py 123-288 (no sample chunking, no guidance)
        n_atom = int(input_feature_dict["atom_to_token_idx"].shape[-1]); dtype = s_inputs.dtype
        x_l = noise_schedule[0] * torch.randn((N_sample, n_atom, 3), dtype=dtype)
        for c_last, c_tau in zip(noise_schedule[:-1], noise_schedule[1:]):
            x_l = x_l - x_l.mean(dim=-2, keepdim=True) + torch.randn((N_sample, 1, 3), dtype=dtype)         # centre + random translation (replicated draws)
            gamma = float(gamma0) if c_tau > gamma_min else 0.0
            t_hat = c_last * (gamma + 1)
            x_noisy = x_l + noise_scale_lambda * torch.sqrt(t_hat ** 2 - c_last ** 2) * torch.randn(x_l.shape, dtype=dtype)
            t_hat = t_hat.reshape((1,)).expand(N_sample).to(dtype)
            x_den = denoise_net(x_noisy=x_noisy, t_hat_noise_level=t_hat, input_feature_dict=input_feature_dict, s_inputs=s_inputs, s_trunk=s_trunk, z_trunk=z_trunk,
                                pair_z=pair_z, p_lm=p_lm, c_l=c_l, chunk_size=attn_chunk_size, inplace_safe=inplace_safe, enable_efficient_fusion=enable_efficient_fusion)
            delta = (x_noisy - x_den) / t_hat[..., None, None]
            x_l = x_noisy + step_scale_eta * (c_tau - t_hat[..., None, None]) * delta
        return x_l

    class _Cfg(dict):
        __getattr__ = dict.__getitem__
        def to_dict(self):
            return dict(self)

    class Model(nn.Module):                                                     # protenix.py: the plumbing the binding reads
        def __init__(self):
            super().__init__(); self.diffusion_module = DiffusionModule(); self.enable_efficient_fusion = bool(fusion); self.enable_diffusion_shared_vars_cache = True
            self.configs = _Cfg(sample_diffusion=_Cfg(gamma0=0.8, gamma_min=1.0, noise_scale_lambda=1.003, step_scale_eta=1.5, N_step=N_step, N_sample=N_sample, guidance=None),
                                skip_amp=_Cfg(sample_diffusion=True), infer_setting=_Cfg(chunk_size=None, sample_diffusion_chunk_size=None))
        def inference_noise_scheduler(self, N_step, device=None, dtype=torch.float32):
            s_max, s_min, p, sd = 160.0, 4e-4, 7.0, 16.0
            t = torch.linspace(0, 1, N_step + 1, dtype=torch.float64)
            sig = sd * (s_max ** (1 / p) + t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p
            sig[-1] = 0.0
            return sig.to(dtype)
        def sample_diffusion(self, **kwargs):                                    # protenix.py 314-343
            cfg = {k: self.configs.sample_diffusion.get(k) for k in ("gamma0", "gamma_min", "noise_scale_lambda", "step_scale_eta")}
            cfg.update(attn_chunk_size=self.configs.infer_setting.chunk_size, diffusion_chunk_size=self.configs.infer_setting.sample_diffusion_chunk_size,
                       guidance_configs=self.configs.sample_diffusion.to_dict().get("guidance"))
            return autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(generator_sample_diffusion)(**cfg, **kwargs)

    model = init_(Model()).eval().requires_grad_(False)
    # ---------------------------------------------------------------------------------------------------------- features (2 chains, 1-3 atoms per token)
    apt = torch.tensor([[1, 3, 2, 1, 2][i % 5] for i in range(N_tok)])
    a2t = torch.repeat_interleave(torch.arange(N_tok), apt)
    n_atom = int(a2t.shape[0])
    half = N_tok // 2
    feats = {"asym_id": torch.cat([torch.zeros(half), torch.ones(N_tok - half)]).long(), "entity_id": torch.zeros(N_tok).long(),
             "sym_id": torch.cat([torch.zeros(half), torch.ones(N_tok - half)]).long(), "residue_index": torch.cat([torch.arange(half), torch.arange(N_tok - half)]).long(),
             "token_index": torch.arange(N_tok).long(), "atom_to_token_idx": a2t, "ref_pos": torch.randn((n_atom, 3), generator=g),
             "ref_charge": torch.randn((n_atom,), generator=g), "ref_mask": torch.ones(n_atom), "ref_element": torch.zeros(n_atom, 4), "ref_atom_name_chars": torch.zeros(n_atom, 4)}
    _, _, info = rearrange_qk_to_dense_trunk(a2t, a2t, -1, -1, nq, nk, compute_mask=True)
    n_tr = int(info["mask_trunked"].shape[0])
    feats["pad_info"] = info
    feats["d_lm"], feats["v_lm"] = torch.randn((n_tr, nq, nk, 3), generator=g), (torch.rand((n_tr, nq, nk, 1), generator=g) > 0.3).float()
    feats["relp"] = model.diffusion_module.diffusion_conditioning.relpe.generate_relp(feats)          # the dense [N, N, 25] relp (the DENSE reference reads it)
    z = torch.randn((N_tok, N_tok, c_z), generator=g)
    s_inputs, s_trunk = torch.randn((N_tok, c_si), generator=g), torch.randn((N_tok, c_s), generator=g)
    return types.SimpleNamespace(model=model, feats=feats, z=z, s_inputs=s_inputs, s_trunk=s_trunk, N_step=N_step, N_sample=N_sample, N=N_tok, n_atom=n_atom)


def _pair_guard(N: int):
    torch = _torch()
    from torch.utils._python_dispatch import TorchDispatchMode
    from torch.utils._pytree import tree_flatten

    class Guard(TorchDispatchMode):
        def __init__(self):
            super().__init__(); self.offenders = []
        def __torch_dispatch__(self, func, types_, args=(), kwargs=None):
            out = func(*args, **(kwargs or {}))
            for leaf in tree_flatten(out)[0]:
                if isinstance(leaf, torch.Tensor) and sum(1 for d in leaf.shape if int(d) == N) >= 2:
                    self.offenders.append((str(func), tuple(leaf.shape)))
            return out
    return Guard()


# ================================================================================================================ the rank entry (spawned by run_sharded)
def _entry(N_tok: int, B: int, fusion: bool, N_sample: int = 2):
    import torch
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import dist as DI
    from opt_core.mem.rowpair import evidence as EV
    from opt_core.mem.rowpair.shard import shard_rows
    import faulthandler
    torch.set_num_threads(1)
    hang_s = int(os.environ.get("PTX2_TEST_HANG_DUMP_S", "0") or 0)
    if hang_s:
        faulthandler.dump_traceback_later(hang_s, repeat=True, file=sys.stderr)      # a stuck rank names its statement in the rank log
    P, r = DI.world()
    lay = DI.Layout.checked(N_tok, P, r, B)
    E = _build(N_tok, fusion=fusion, N_sample=N_sample)
    say = lambda *a: (sys.stderr.write(f"[ptx2-test rank{r}] " + " ".join(str(x) for x in a) + "\n"), sys.stderr.flush())   # noqa: E731
    say("layout", lay.r0, lay.r1, "R", lay.R, "Rmax", lay.Rmax, "B", B)
    import protenix_opt.tp_bind.diffusion as BD
    m, f, dm = E.model, E.feats, E.model.diffusion_module
    res = {"rank": r, "P": P, "R": lay.R, "checks": {}, "metrics": {}, "bitwise": {}}
    chk, met, bit = res["checks"], res["metrics"], res["bitwise"]
    with torch.no_grad():
        ns = m.inference_noise_scheduler(E.N_step)
        # ------------------------------------------------------------------ dense (the stub's whole-tensor statements = stock semantics)
        torch.manual_seed(SEED)
        pz_d = dm.diffusion_conditioning.prepare_cache(f["relp"], E.z, False)
        kw = {k: f[k] for k in BD.ATOM_KEYS}
        p_lm_d, c_l_d = dm.atom_attention_encoder.prepare_cache(**kw, r_l=True, z=pz_d, inplace_safe=False)
        coords_d = m.sample_diffusion(denoise_net=dm, input_feature_dict=f, s_inputs=E.s_inputs, s_trunk=E.s_trunk, z_trunk=None, pair_z=pz_d, p_lm=p_lm_d, c_l=c_l_d,
                                      N_sample=E.N_sample, noise_schedule=ns, inplace_safe=True, enable_efficient_fusion=fusion)
        # ------------------------------------------------------------------ through the binding on this rank's rows
        say("dense done")
        torch.manual_seed(SEED + r)                                            # ranks seeded APART: the binding continues rank 0's stream everywhere
        z_loc = shard_rows(E.z, lay, dim=-3).contiguous()
        guard = _pair_guard(N_tok)
        with guard:
            pz = BD.tp_prepare_cache(dm.diffusion_conditioning, f, z_loc, lay, skip_amp=True, inplace_safe=True)
            say("prepare_cache done", tuple(pz.shape))
            coords = BD.tp_sample_diffusion(m, f, E.s_inputs, E.s_trunk, None, pz, lay, E.N_sample, E.N_step, skip_amp=True, inplace_safe=True, noise_schedule=ns)
        say("sample_diffusion done", tuple(coords.shape))
        res["pair_shaped"] = guard.offenders[:6]
        chk["never_whole_pair"] = not guard.offenders

        def cmp(name, got, ref, tol=TOL):                                    # max|diff| relative to max(1, max|ref|): coordinates are O(sigma_max)
            same = got.shape == ref.shape and got.dtype == ref.dtype
            d = float((got.float() - ref.float()).abs().max()) / max(1.0, float(ref.float().abs().max())) if same else float("inf")
            met[name] = d; bit[name] = bool(same and torch.equal(got, ref)); chk[name] = same and d <= tol
        cmp("pair_z_rows", pz, pz_d[lay.r0:lay.r1].contiguous(), tol=1e-5)
        cmp("coords", coords, coords_d)
        # ------------------------------------------------------------------ census / report / refusals
        fields = dict(EV.schedule_fields())
        src = str(fields.get("diff_rows_source", ""))
        chk["rows_given"] = all(f"{k}:given" in src for k in ("cond", "bias", "q", "band")) and int(fields.get("diff_cond_rows", -1)) == B and int(fields.get("diff_bias_rows", -1)) == B
        chk["census_words"] = all(k in fields for k in ("diff_bias_cache", "diff_band_W", "diff_noise_sync", "diff_engine", "diff_q_len", "diff_rng_sync"))
        met["census"] = {k: fields[k] for k in sorted(fields) if k.startswith("diff_")}
        sd = BD.report()["modes"].get("sample_diffusion") or {}
        chk["report_regime"] = sd.get("mode") == "tp" and sd.get("skip_amp") is True and sd.get("N") == N_tok and sd.get("P") == P and sd.get("denoise_calls") == E.N_step and sd.get("rng_sync") == "rank0"
        chk["report_prepare"] = (BD.report()["modes"].get("prepare_cache") or {}).get("rows") == B
        refused = {}
        for name, fn in {"skip_amp_missing": lambda: BD.tp_prepare_cache(dm.diffusion_conditioning, f, z_loc, lay),
                         "pair_cache_off": lambda: BD.tp_sample_diffusion(m, f, E.s_inputs, E.s_trunk, None, None, lay, E.N_sample, E.N_step, skip_amp=True),
                         "unknown_kw": lambda: BD.tp_sample_diffusion(m, f, E.s_inputs, E.s_trunk, None, pz, lay, E.N_sample, E.N_step, skip_amp=True, force_mode="replicated")}.items():
            try:
                fn(); refused[name] = False
            except RowpairRefused as e:
                refused[name] = "protenix_opt.tp_bind.diffusion" in str(e)
        os.environ[BD.SAMPLER_HOOK_ENVS[0]] = "some.module:factory"
        try:
            BD.tp_sample_diffusion(m, f, E.s_inputs, E.s_trunk, None, pz, lay, E.N_sample, E.N_step, skip_amp=True); refused["sampler_hook"] = False
        except RowpairRefused as e:
            refused["sampler_hook"] = BD.SAMPLER_HOOK_ENVS[0] in str(e)
        finally:
            os.environ.pop(BD.SAMPLER_HOOK_ENVS[0])
        chk["refusals_by_name"] = all(refused.values()); met["refusals"] = refused
    res["ok"] = all(chk.values())
    import torch.distributed as tdist
    allres = [None] * P
    tdist.all_gather_object(allres, res)
    return {"P": P, "ranks": allres, "rows": [x["R"] for x in allres], "ok": all(x["ok"] for x in allres)}


def _mp(P, *args):
    _core()
    from opt_core.mem.rowpair import launch
    os.environ["ROWPAIR_TEST_DEVICE"] = "cpu"
    return launch.run_sharded(P, _entry, *args, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120,
                              run_timeout_s=float(os.environ.get("PTX2_TEST_RUN_TIMEOUT_S", "600")), log_dir=os.environ.get("PTX2_TEST_LOG_DIR") or None)


@needs_gloo
@pytest.mark.parametrize("P,N,B,fusion,uneven,N_sample", [(2, 40, 16, True, True, 2), (3, 48, 16, False, False, 2), (2, 40, 16, False, True, 2),
                                                        (3, 36, 8, True, True, 2), (4, 52, 8, False, True, 2),   # rows 16/16/4 and 16/16/16/4: a ragged LAST rank with R=4 < B, < the atom key window, < the other ranks' q_len
                                                        (2, 40, 16, False, True, 4)])                             # N_sample == n_heads (4): the bias enters SDPA at q's rank [1, H, q, N], never unsqueezed at the head axis
def test_diffusion_binding_vs_dense(P, N, B, fusion, uneven, N_sample):
    _torch(); _core()
    res = _mp(P, N, B, fusion, N_sample)
    print("RESULT " + json.dumps(res, sort_keys=True, default=str))
    assert [x["rank"] for x in res["ranks"]] == list(range(P))
    assert (len(set(res["rows"])) > 1) is uneven, res["rows"]
    bad = {x["rank"]: [k for k, v in x["checks"].items() if not v] for x in res["ranks"] if not x["ok"]}
    assert not bad, (bad, [(x["rank"], x["metrics"], x["pair_shaped"]) for x in res["ranks"]])


def test_p1_layout_refused_by_name():
    torch = _torch(); _core()
    from opt_core.mem.rowpair import RowpairRefused
    from opt_core.mem.rowpair import dist as DI
    _install_protenix_stub()
    import protenix_opt.tp_bind.diffusion as BD
    lay = DI.Layout(12, 1, 0)
    with pytest.raises(RowpairRefused, match="n_gpu=1"):
        BD.tp_prepare_cache(None, {}, torch.zeros(12, 12, 2), lay, skip_amp=True)
    with pytest.raises(RowpairRefused, match="n_gpu=1"):
        BD.tp_sample_diffusion(None, {}, None, None, None, torch.zeros(12, 12, 2), lay, 1, 1, skip_amp=True)


if __name__ == "__main__":                                                   # python test_tp_rebase_diffusion.py [P N B fusion]
    P, N, B, fu, ns = (sys.argv[1:6] + ["2", "40", "16", "1", "2"][len(sys.argv) - 1:])[:5]
    out = _mp(int(P), int(N), int(B), bool(int(fu)), int(ns))
    print("RESULT " + json.dumps(out, sort_keys=True, default=str))
    sys.exit(0 if out["ok"] else 1)
