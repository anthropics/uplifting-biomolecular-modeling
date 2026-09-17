"""The `atom_window` adapter on this kit: the core kernel's window addressing (`opt_core.kernels.atom_window.window_starts` + the kernel's key /
pair validity rule) selects, position by position, the keys of THIS engine's shifted-window gather (`atom_attention_block_utils.get_block_indices`)
and its block pair mask (`convert_single_rep_to_blocks`) when openfold3 0.5.0 imports (CPU); and `plan()` refuses by name through the adapter's
binding. The layout sweep against the transliterated rule and the kernels' numerics are the tree's tests (the OpenFold3 0.4.x kit's test of this name,
opt_core's tests/gpu/test_atom_window_gpu.py) — one implementation, not repeated here."""
import math

import pytest

torch = pytest.importorskip("torch", reason="needs torch")
from openfold3_ob0_opt.cells import atom_window as AWC  # noqa: E402

aw = pytest.importorskip("opt_core.kernels.atom_window", reason="needs opt_core's atom_window kernel module (Triton importable)")
NQ, NK = 32, 128


def _kernel_side(atom_mask):
    S, A = atom_mask.shape
    n_real = int(atom_mask[0].sum().item())
    ks = aw.window_starts(A, torch.tensor(float(n_real)), NQ, NK, atom_mask.device).long()
    nb = ks.numel()
    k_idx = ks[:, None] + torch.arange(NK)[None, :]
    k_in = (k_idx >= 0) & (k_idx < A)
    am_k = torch.zeros((S, nb, NK)); am_k[:, k_in] = atom_mask[:, k_idx[k_in]]
    key_valid = k_in[None] & (k_idx[None] < n_real) & (am_k > 0.5)
    q_rows = torch.arange(nb * NQ).reshape(nb, NQ); q_in = q_rows < A
    am_q = torch.zeros((S, nb, NQ)); am_q[:, q_in] = atom_mask[:, q_rows[q_in]]
    return k_idx, key_valid, (am_q[..., :, None] > 0.5) & key_valid[..., None, :]


def test_window_addressing_equals_this_engines_gather():
    abu = pytest.importorskip("openfold3.core.utils.atom_attention_block_utils", reason="openfold3 0.5.0 not importable here")
    bad = []
    for A in list(range(1, 300, 7)) + [31, 32, 33, 127, 128, 129, 383, 384, 385, 1000, 3260, 6473]:
        for n_real in sorted({A, max(1, A - 1), max(1, A - 31), max(1, A // 2)}):
            S = 1 if A > 600 else 2
            m = torch.zeros((S, A)); m[:, :n_real] = 1.0
            safe, invalid = abu.get_block_indices(atom_mask=m, n_query=NQ, n_key=NK, device=m.device)
            _, _, pair = abu.convert_single_rep_to_blocks(torch.zeros((S, A, 4)), NQ, NK, m)
            k_idx, key_valid, pair_k = _kernel_side(m)
            valid_ref = ~invalid & (torch.gather(m[:, None, :].expand(-1, safe.shape[1], -1), 2, safe.long()) > 0.5)
            ok = (torch.equal(key_valid, valid_ref) and bool(((k_idx[None].expand_as(safe) == safe.long()) | ~valid_ref).all()) and torch.equal(pair_k, pair > 0.5))
            if not ok:
                bad.append((A, n_real, S))
    assert not bad, bad[:10]


def test_plan_refuses_by_name_through_the_adapter():
    ns = lambda **kw: type("NS", (), kw)()   # noqa: E731
    lin = lambda cin, cout: ns(in_features=cin, out_features=cout, weight=torch.zeros(cout, cin), bias=None)   # noqa: E731
    C, H, cz, S, A = 128, 4, 16, 2, 100
    mha = ns(no_heads=H, linear_q=lin(C, C), linear_k=lin(C, C), linear_v=lin(C, C), linear_g=lin(C, C), linear_o=lin(C, C))
    apb = ns(mha=mha, n_query=32, n_key=128, inf=1e9, layer_norm_a_q=ns(eps=1e-5), layer_norm_a_k=ns(eps=1e-5), linear_z=lin(cz, H), linear_ada_out=lin(C, C))
    blk = ns(attention_pair_bias=apb, use_cross_attention=True)
    nb = math.ceil(A / 32)
    a = torch.zeros(1, S, A, C); s = torch.zeros(1, A, C); z = torch.zeros(1, nb, 32, 128, cz); mask = torch.ones(1, S, A)
    assert AWC.plan(blk, a, s, z, mask, {"use_high_precision_attention": True})[1] == "device_cpu"      # the word is accepted; a CPU tensor is the first miss
    assert AWC.plan(blk, a, s, z, mask, {"chunk_size": 4})[1] == "kwargs:chunk_size"
    assert AWC.plan(ns(attention_pair_bias=ns(mha=mha), use_cross_attention=True), a, s, z, mask, {})[1] == "module_layout"
    assert AWC._core.SERVED_WINDOW == (32, 128)
