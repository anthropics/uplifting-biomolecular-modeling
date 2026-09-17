"""apb_core's installers against protenix-v2's module topology (stubs of protenix.model.modules.{primitives,transformer,diffusion} on CPU):
the DiffusionModule's 24 token-attention modules (16 x 48), its atom encoder + decoder AtomTransformers (local_cross_attention, 4 x 32,
32 x 128 windows, gated) and the Pairformer AttentionPairBias (has_s=False, 16 gated heads) BIND through the provider by the tier word; an
unknown topology steps aside BY NAME at install (apb_levers: `<MARK>aside(install: ...)`, no raise, the modules keep the stock statement)."""
import io
import sys
import types
from contextlib import redirect_stderr

import pytest
import torch

from protenix_opt import apb_core as core, apb_levers as apb


class _Attention(torch.nn.Module):                      # protenix.model.modules.primitives.Attention's surface the levers read
    def __init__(self, c_in, num_heads, c_hidden, gating=True, local_attention_method="global"):
        super().__init__()
        self.c_q = self.c_k = self.c_v = c_in; self.num_heads = num_heads; self.c_hidden = c_hidden; self.gating = gating
        self.local_attention_method = local_attention_method
        self.linear_q = torch.nn.Linear(c_in, num_heads * c_hidden); self.linear_k = torch.nn.Linear(c_in, num_heads * c_hidden, bias=False)
        self.linear_v = torch.nn.Linear(c_in, num_heads * c_hidden, bias=False); self.linear_g = torch.nn.Linear(c_in, num_heads * c_hidden, bias=False)
        self.linear_o = torch.nn.Linear(num_heads * c_hidden, c_in)
        self.stock_calls = []

    def forward(self, q_x, kv_x, attn_bias=None, trunked_attn_bias=None, n_queries=None, n_keys=None, inf=1e10, inplace_safe=False, chunk_size=None):
        self.stock_calls.append((tuple(q_x.shape), n_queries, n_keys)); return ("stock", q_x)


class _AttentionPairBias(torch.nn.Module):
    def __init__(self, has_s, c_a, num_heads, c_hidden, c_z=128, method="global", cross_attention_mode=False):
        super().__init__()
        self.has_s = has_s; self.cross_attention_mode = cross_attention_mode
        self.attention = _Attention(c_a, num_heads, c_hidden, gating=True, local_attention_method=method)
        self.layernorm_z = torch.nn.LayerNorm(c_z); self.linear_nobias_z = torch.nn.Linear(c_z, num_heads, bias=False)

    def standard_multihead_attention(self, q, kv, z, inplace_safe=False, enable_efficient_fusion=False):
        return ("stock_smha", q)


class _Block(torch.nn.Module):
    def __init__(self, apb_mod):
        super().__init__(); self.attention_pair_bias = apb_mod


class _DiffusionTransformer(torch.nn.Module):
    def __init__(self, blocks):
        super().__init__(); self.blocks = torch.nn.ModuleList(blocks)


class _AtomTransformer(torch.nn.Module):
    def __init__(self, n_blocks, method):
        super().__init__()
        self.diffusion_transformer = _DiffusionTransformer([_Block(_AttentionPairBias(True, 128, 4, 32, c_z=16, method=method)) for _ in range(n_blocks)])


class _AtomPart(torch.nn.Module):
    def __init__(self, method):
        super().__init__(); self.atom_transformer = _AtomTransformer(3, method)


class _DiffusionModule(torch.nn.Module):
    def __init__(self, atom_method="local_cross_attention", dit_heads=16, dit_c_hidden=48):
        super().__init__()
        self.atom_attention_encoder = _AtomPart(atom_method); self.atom_attention_decoder = _AtomPart(atom_method)
        self.diffusion_transformer = _DiffusionTransformer([_Block(_AttentionPairBias(True, 768, dit_heads, dit_c_hidden, c_z=128)) for _ in range(24)])


class _Model(torch.nn.Module):
    def __init__(self, **kw):
        super().__init__()
        self.diffusion_module = _DiffusionModule(**kw)
        self.pairformer_stack = torch.nn.ModuleList([_AttentionPairBias(False, 384, 16, 24, c_z=128) for _ in range(48)])
        self.confidence_head = torch.nn.ModuleList([_AttentionPairBias(False, 384, 16, 24, c_z=128) for _ in range(4)])


@pytest.fixture
def protenix_stubs(monkeypatch):
    """protenix.model.modules.{primitives,transformer,diffusion} with the classes the installers match by isinstance."""
    P = types.ModuleType("protenix.model.modules.primitives"); P.Attention = _Attention
    T = types.ModuleType("protenix.model.modules.transformer"); T.AtomTransformer = _AtomTransformer; T.DiffusionTransformer = _DiffusionTransformer; T.AttentionPairBias = _AttentionPairBias
    D = types.ModuleType("protenix.model.modules.diffusion"); D.DiffusionModule = _DiffusionModule
    M = types.ModuleType("protenix.model.modules"); M.primitives, M.transformer, M.diffusion = P, T, D; M.__path__ = []
    for name, mod in {"protenix": types.ModuleType("protenix"), "protenix.model": types.ModuleType("protenix.model"), "protenix.model.modules": M,
                      "protenix.model.modules.primitives": P, "protenix.model.modules.transformer": T, "protenix.model.modules.diffusion": D}.items():
        if not hasattr(mod, "__path__"):
            mod.__path__ = []
        monkeypatch.setitem(sys.modules, name, mod)
    core._reset_for_tests()
    monkeypatch.setattr(apb, "_STATE", {"on": [], "patch": None, "installed": {}, "errors": {}, "named": {}, "precision": None, "models": 0, "gate": {}})
    yield
    core._reset_for_tests()


def test_atom_attn_binds_the_local_cross_attention_topology(protenix_stubs):
    """protenix-v2's atom encoder + decoder attention: local_cross_attention, 4 heads, c_q 128 -> head dim 32, gated -> BOUND by the tier word."""
    model = _Model()
    for word in ("fast", "big", "exact"):
        core._reset_for_tests(); model = _Model()
        st = core.install_atom_attn(model, word=word)
        assert st["installed_on"] == 6 and st["cell"] == {"word": word, "face": "kernels.apb", "geometry": "atom_h4d32w32x128"} and st["aside"] is None
        atts = [blk.attention_pair_bias.attention for part in (model.diffusion_module.atom_attention_encoder, model.diffusion_module.atom_attention_decoder)
                for blk in part.atom_transformer.diffusion_transformer.blocks]
        assert len(atts) == 6 and all("forward" in vars(a) and a.forward._fpf_apb == "atom_attn" and a._fpf_apb == "atom_attn" for a in atts)
    att = atts[0]                                                    # a call outside the windowed contract (no window bias) takes the module's own statement BY NAME, counted -- never raised
    out = att.forward(torch.zeros(2, 40, 128), torch.zeros(2, 40, 128), attn_bias=torch.zeros(1, 4, 40, 40))
    assert out[0] == "stock" and core.STATE["atom_attn"]["aside"] == {"global_call": 1} and core.STATE["atom_attn"]["calls"] == 1 and att.stock_calls


def test_dit_attn_and_pf_attn_bind_their_topologies(protenix_stubs):
    model = _Model()
    st = core.install_dit_attn(model, word="fast", fp16=True)
    assert st["installed_on"] == 24 and st["precision"] == "per_cell" and st["cell"]["fp16_arms"] == "admissible" and core.STATE["dit_attn_fp16"]["on"] is True
    dits = [blk.attention_pair_bias.attention for blk in model.diffusion_module.diffusion_transformer.blocks]
    assert all(a.forward._fpf_apb == "dit_attn" for a in dits)
    out = dits[0].forward(torch.zeros(5, 20, 768), torch.zeros(5, 20, 768), attn_bias=None)          # no pair bias: the statement BY NAME, counted
    assert out[0] == "stock" and core.STATE["dit_attn"]["aside"] == {"no_bias": 1}
    out = dits[0].forward(torch.zeros(5, 64, 768), torch.zeros(5, 64, 768), trunked_attn_bias=torch.zeros(1, 16, 2, 32, 128), n_queries=32, n_keys=128)
    assert out[0] == "stock" and core.STATE["dit_attn"]["aside"] == {"no_bias": 1, "local_call": 1}
    pst = core.install_pf_attn(model, word="fast")
    assert pst["installed_on"] == 52 and pst["skipped_has_s"] == 0                                   # 48 pairformer + 4 confidence; the DiffusionModule's has_s modules are outside the scan
    assert all(getattr(m.standard_multihead_attention, "_fpf_apb", None) == "pf_attn" for m in list(model.pairformer_stack) + list(model.confidence_head))
    assert not any("standard_multihead_attention" in vars(blk.attention_pair_bias) for blk in model.diffusion_module.diffusion_transformer.blocks)


def test_an_unknown_topology_raises_by_name_in_the_installer(protenix_stubs):
    with pytest.raises(RuntimeError, match=r"atom_attn: local_attention_method='some_new_method'"):
        core.install_atom_attn(_Model(atom_method="some_new_method"), word="fast")
    with pytest.raises(RuntimeError, match=r"dit_attn: token attention geometry heads=8 c_hidden=96"):
        core.install_dit_attn(_Model(dit_heads=8, dit_c_hidden=96), word="fast")


def test_the_seam_answers_an_install_failure_with_a_stated_aside_never_a_raise(protenix_stubs, monkeypatch):
    """Doctrine: a lever that cannot engage steps aside BY NAME -- apb_levers' installer prints `ATOM_ATTN:aside(install: ...)`, records the
    reason (reconcile names the lever a fallback with it), leaves the stock statement on the modules and returns; rc stays the run's own."""
    monkeypatch.setattr(core, "tier_word", lambda mode=None: "fast")
    monkeypatch.setattr(apb, "triton_note", lambda: None)
    monkeypatch.setattr(apb, "gate_installed", lambda lever, model: None)
    model = _Model(atom_method="some_new_method")
    runner = types.SimpleNamespace(model=model)
    apb._STATE["on"] = ["dit_attn", "dit_attn_fp16", "atom_attn"]
    err = io.StringIO()
    with redirect_stderr(err):
        apb._installer("atom_attn")(runner)                        # unknown atom topology: aside, no raise
        apb._installer("dit_attn")(runner)                         # the DiT site of the same model binds
    lines = err.getvalue().splitlines()
    assert any(l.startswith("[protenix-opt] ATOM_ATTN:aside(install: RuntimeError(\"atom_attn: local_attention_method='some_new_method'") for l in lines), lines
    assert apb._STATE["errors"]["atom_attn"].startswith("RuntimeError(") and apb._STATE["installed"]["atom_attn"] == 0 and apb._STATE["named"]["atom_attn"].startswith("aside at install:")
    atts = [blk.attention_pair_bias.attention for blk in model.diffusion_module.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks]
    assert not any("forward" in vars(a) for a in atts)              # the modules keep the stock statement
    assert any(l.startswith("[protenix-opt] DIT_ATTN:on(24 modules; cell=") and "word=fast" in l for l in lines), lines
    assert apb._STATE["installed"]["dit_attn"] == 24 and "[protenix-opt] DIT_ATTN_FP16:on" in lines


def test_the_seam_binds_the_atom_topology_by_the_tier_word(protenix_stubs, monkeypatch):
    monkeypatch.setattr(core, "tier_word", lambda mode=None: "big")
    monkeypatch.setattr(apb, "triton_note", lambda: None)
    monkeypatch.setattr(apb, "gate_installed", lambda lever, model: None)
    runner = types.SimpleNamespace(model=_Model())
    apb._STATE["on"] = ["atom_attn"]
    err = io.StringIO()
    with redirect_stderr(err):
        apb._installer("atom_attn")(runner)
    line = [l for l in err.getvalue().splitlines() if l.startswith("[protenix-opt] ATOM_ATTN:")]
    assert len(line) == 1 and line[0].startswith("[protenix-opt] ATOM_ATTN:on(6 modules; cell=") and "word=big" in line[0] and "geometry=atom_h4d32w32x128" in line[0], line
    assert apb._STATE["installed"]["atom_attn"] == 6 and not apb._STATE["errors"]


def test_pf_attn_takes_the_memory_lines_chunked_producer_when_it_hands_one(protenix_stubs, monkeypatch):
    """big above apb_bias_chunk's gate: pf_attn's pair bias comes from big.apb_bias_chunked (the stock statement on row blocks) instead of the
    provider's producer row; the core still serves the attention; the LEVER tally names the producer `apb_bias_chunk`."""
    from opt_core.kernels import apb as A
    model = _Model()
    core.install_pf_attn(model, word="big")
    m = model.pairformer_stack[0]
    N, H, D = 12, 16, 24
    handed = torch.arange(H * N * N, dtype=torch.float32).reshape(H, N, N)
    seen = {}
    monkeypatch.setattr(core, "_big_bias", lambda apb_mod, zz: handed if apb_mod is m else None)
    monkeypatch.setattr(core, "_select", lambda A_, site, t, **kw: types.SimpleNamespace(row="sdpa", variant="auto"))
    def fake_core(q, k, v, bias, key_mask, gate, **kw):
        seen["bias"] = bias; seen["cell"] = kw.get("cell"); return torch.zeros(q.shape[0], q.shape[1], q.shape[2], q.shape[3]), kw["selection"]
    monkeypatch.setattr(A, "pair_bias_attention", fake_core)
    monkeypatch.setattr(A, "pair_bias_planes", lambda *a, **k: (_ for _ in ()).throw(AssertionError("the provider producer must not run above the gate")))
    q = torch.zeros(1, N, 384); z = torch.zeros(1, N, N, 128)
    out = m.standard_multihead_attention(q, q, z)
    assert out.shape == (1, N, 384) and seen["cell"] == "pf_h16d24"
    assert seen["bias"].shape == (1, H, N, N) and torch.equal(seen["bias"][0], handed)
    assert core.STATE["pf_attn"]["producer"] == {"apb_bias_chunk": 1} and core.STATE["pf_attn"]["served"] == {"sdpa:auto": 1}
    monkeypatch.setattr(core, "_big_bias", lambda apb_mod, zz: None)                            # below the gate / fast: the provider's producer row again
    monkeypatch.setattr(A, "pair_bias_planes", lambda zz, lw, lb, w, **k: (torch.zeros(H, N, 16), types.SimpleNamespace(row="ln_proj", variant=None)))
    monkeypatch.setattr(A, "arm_word", lambda row, variant: row + (":" + variant if variant else ""))
    m.standard_multihead_attention(q, q, z)
    assert core.STATE["pf_attn"]["producer"] == {"apb_bias_chunk": 1, "ln_proj": 1}
