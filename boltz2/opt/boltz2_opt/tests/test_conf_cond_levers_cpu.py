"""CONF track — CPU unit tests of the `condproj` lever (opt/forward/conf/cond_levers.py through the boltz2_opt.conf loader): the fused
shared-statistics normalisation + folded GEMM equals stock's 24 / 3 / 3 separate LayerNorm+Linear projections of DiffusionConditioning
to fp32 rounding (Tier 2: not a bitwise claim), returns stock's shapes/dtypes and torch.cat order, folds once, and refuses by name on a
structure that is not the pinned one. Pure-python boltz 2.2.1 model tree required; no GPU.
"""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("boltz.model.modules.diffusion_conditioning", reason="the pure-python boltz 2.2.1 model tree is required (pip install --no-deps boltz==2.2.1 einops scipy)")

from boltz.model.modules.diffusion_conditioning import DiffusionConditioning  # noqa: E402

from boltz2_opt import conf as ADAPTER  # noqa: E402

TOKEN_S, TOKEN_Z, ATOM_S, ATOM_Z = 32, 24, 16, 8


@pytest.fixture()
def CL():
    cl = ADAPTER._module("cond_levers")
    yield cl
    cl.uninstall()
    for k in list(cl.STATS):
        cl.STATS[k] = 0


def make_module(seed=0):
    torch.manual_seed(seed)
    m = DiffusionConditioning(token_s=TOKEN_S, token_z=TOKEN_Z, atom_s=ATOM_S, atom_z=ATOM_Z, atoms_per_window_queries=32, atoms_per_window_keys=128,
                              atom_encoder_depth=3, atom_encoder_heads=4, token_transformer_depth=24, token_transformer_heads=16,
                              atom_decoder_depth=3, atom_decoder_heads=4, atom_feature_dim=3 + 1 + 4 + 4 * 64)
    with torch.no_grad():
        for lst in (m.token_trans_proj_z, m.atom_enc_proj_z, m.atom_dec_proj_z):
            for seq in lst:                                       # non-trivial gamma/beta so the folding is really exercised
                seq[0].weight.copy_(1.0 + 0.3 * torch.randn_like(seq[0].weight)); seq[0].bias.copy_(0.2 * torch.randn_like(seq[0].bias))
                seq[1].weight.copy_(0.4 * torch.randn_like(seq[1].weight))
    return m.eval()


def stock_lists(m, z, p):
    with torch.no_grad():
        tok = torch.cat([layer(z) for layer in m.token_trans_proj_z], dim=-1)
        enc = torch.cat([layer(p) for layer in m.atom_enc_proj_z], dim=-1)
        dec = torch.cat([layer(p) for layer in m.atom_dec_proj_z], dim=-1)
    return tok, enc, dec


def test_fused_projection_equals_stock_lists_fp32(CL):
    m = make_module(); N, K = 20, 3
    z = torch.randn(1, N, N, TOKEN_Z) * 2.0 + 0.5; p = torch.randn(1, K, 32, 128, ATOM_Z)
    tok, enc, dec = stock_lists(m, z, p)
    with torch.no_grad():
        ftok = CL.fused_projection(z, CL.fold(m.token_trans_proj_z)); fenc = CL.fused_projection(p, CL.fold(m.atom_enc_proj_z)); fdec = CL.fused_projection(p, CL.fold(m.atom_dec_proj_z))
    assert ftok.shape == tok.shape == (1, N, N, 24 * 16) and fenc.shape == enc.shape == (1, K, 32, 128, 12) and fdec.shape == dec.shape
    assert ftok.dtype == tok.dtype == torch.float32
    for a, b in ((ftok, tok), (fenc, enc), (fdec, dec)):
        err = (a - b).abs().max().item(); scale = b.abs().max().item()
        assert err <= 2e-5 * max(1.0, scale), (err, scale)          # exact in real arithmetic; fp32 rounding only (gamma folded, beta*W as bias)
    # cat order is layer-major: block l of the fused output is layer l's projection
    assert torch.allclose(ftok[..., 16 * 5:16 * 6], m.token_trans_proj_z[5](z), atol=1e-4)


def test_forward_patched_equals_stock_forward_and_counts(CL):
    m = make_module(); N = 16; n_atoms = 64
    feats = fake_feats(N, n_atoms)
    s = torch.randn(1, N, TOKEN_S); z = torch.randn(1, N, N, TOKEN_Z); rel = torch.randn(1, N, N, TOKEN_Z)
    with torch.no_grad():
        ref = m(s_trunk=s, z_trunk=z, relative_position_encoding=rel, feats=feats)
        CL.install(["condproj"], model_hook=False)
        got = m(s_trunk=s, z_trunk=z, relative_position_encoding=rel, feats=feats)
    assert len(ref) == len(got) == 6
    for i, (a, b) in enumerate(zip(ref, got)):
        if i == 2:                                                     # to_keys is a functools.partial over the same indexing matrix
            probe = torch.randn(1, n_atoms, 5)
            assert torch.equal(a(probe), b(probe)); continue
        assert a.shape == b.shape and a.dtype == b.dtype, i
        if i < 2:
            assert torch.equal(a, b), f"output {i} (q, c) must be untouched"
        else:
            assert torch.allclose(a, b, atol=1e-4, rtol=1e-4), (i, (a - b).abs().max().item())
    assert CL.STATS["calls"] == 1 and CL.STATS["token_lists_fused"] == 1 and CL.STATS["atom_lists_fused"] == 2 and CL.STATS["layers_folded"] == 30
    with torch.no_grad():
        m(s_trunk=s, z_trunk=z, relative_position_encoding=rel, feats=feats)
    assert CL.STATS["layers_folded"] == 30, "folded weights are computed once per instance"


def test_structure_violation_refuses_by_name(CL):
    m = make_module()
    m.token_trans_proj_z[3][1].bias = torch.nn.Parameter(torch.zeros(16))     # a Linear with a bias is not the pinned structure
    CL.install(["condproj"], model_hook=False)
    N = 8; feats = fake_feats(N, 32)
    with pytest.raises(RuntimeError, match="conf.condproj REFUSED: DiffusionConditioning.token_trans_proj_z: layer 3"):
        with torch.no_grad():
            m(s_trunk=torch.randn(1, N, TOKEN_S), z_trunk=torch.randn(1, N, N, TOKEN_Z), relative_position_encoding=torch.randn(1, N, N, TOKEN_Z), feats=feats)
    assert CL.check_list(torch.nn.ModuleList()) == "empty ModuleList"


def test_adapter_knows_condproj():
    assert "condproj" in ADAPTER.LEVERS and ADAPTER.MODULE_OF["condproj"] == "cond_levers" and ADAPTER.TIER_OF["condproj"] == 2
    assert ADAPTER.parse_words("condproj,tfeat") == ["condproj", "tfeat"]


def fake_feats(N, n_atoms):
    """The AtomEncoder's feats (encodersv2.py AtomEncoder.forward) for one unpadded item: n_atoms atoms (multiple of 32) over N tokens."""
    g = torch.Generator().manual_seed(3)
    atom_to_token_idx = torch.sort(torch.randint(0, N, (n_atoms,), generator=g)).values
    a2t = torch.nn.functional.one_hot(atom_to_token_idx, N).float()[None]
    return {"ref_pos": torch.randn(1, n_atoms, 3, generator=g), "atom_pad_mask": torch.ones(1, n_atoms), "ref_charge": torch.randn(1, n_atoms, generator=g),
            "ref_element": torch.nn.functional.one_hot(torch.randint(0, 4, (1, n_atoms), generator=g), 4).float(),
            "ref_atom_name_chars": torch.nn.functional.one_hot(torch.randint(0, 64, (1, n_atoms, 4), generator=g), 64).float(),
            "ref_space_uid": atom_to_token_idx[None].clone(), "atom_to_token": a2t}
