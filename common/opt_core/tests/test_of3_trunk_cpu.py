"""opt_core.of3_trunk.templ_embed on CPU: the adapter switch grammar through configure(), install()'s by-name refusals (unbound engine,
missing kernel entry points), the patched forward's routing (not serving -> the stock forward untouched; a refused pass -> the stock forward,
counted by name; the census line), and the kernels' stock-statement references against a plain restatement. GPU numerics of the kernels are
tests/gpu/test_templ_embed_gpu.py; in-model numerics are the kits' GPU checks."""
import sys
import types

import pytest

torch = pytest.importorskip("torch")

from opt_core.of3_trunk import templ_embed as TE  # noqa: E402


@pytest.fixture
def bound(monkeypatch):
    """templ_embed bound to a throw-away kit's names and a stub engine module `x_engine.template_module`; restored after the test."""
    saved = {k: getattr(TE, k) for k in TE.CONFIGURABLE}
    state0 = {k: (v.copy() if isinstance(v, (dict, list)) else v) for k, v in TE.STATE.items()}
    mod = types.ModuleType("x_engine.template_module")

    class TemplateEmbedderAllAtom(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.calls = []

        def forward(self, batch, z, pair_mask, chunk_size=None, _mask_trans=True, use_deepspeed_evo_attention=False, use_cueq_triangle_kernels=False,
                    use_triton_triangle_kernels=False, use_lma=False, inplace_safe=False):
            self.calls.append((chunk_size, use_cueq_triangle_kernels, inplace_safe))
            return z + 1.0
    mod.TemplateEmbedderAllAtom = TemplateEmbedderAllAtom
    monkeypatch.setitem(sys.modules, "x_engine.template_module", mod)
    TE.configure(PREFIX="[x-kit/templ_embed]", ENV="X_TEMPL_EMBED", M_TEMPLATE="x_engine.template_module")
    yield mod
    TE.configure(**saved)
    TE.STATE.clear(); TE.STATE.update(state0)
    TE._K["mod"] = None


def test_switch_grammar_through_configure(bound):
    with pytest.raises(KeyError):
        TE.configure(NOT_A_SETTING=1)
    assert TE.requested({}) is False and TE.requested({"X_TEMPL_EMBED": ""}) is False and TE.requested({"X_TEMPL_EMBED": "1"}) is True
    with pytest.raises(ValueError):
        TE.requested({"X_TEMPL_EMBED": "on"})
    TE.configure(ENV=None)
    assert TE.requested({"X_TEMPL_EMBED": "1"}) is False                       # unbound: nothing requested whatever the environment says


def test_install_not_requested_touches_nothing(bound):
    orig = bound.TemplateEmbedderAllAtom.forward
    assert TE.install({}) is TE.STATE and TE.STATE["installed"] is False and TE.STATE["state"] == "off"
    assert bound.TemplateEmbedderAllAtom.forward is orig


def test_install_refuses_by_name_without_an_engine_binding(bound):
    TE.configure(M_TEMPLATE=None)
    with pytest.raises(RuntimeError, match="not bound to an engine"):
        TE.install({"X_TEMPL_EMBED": "1"})
    assert TE.STATE["installed"] is False


def test_install_refuses_by_name_when_the_kernel_lacks_its_entry_points(bound, monkeypatch):
    fake = types.ModuleType("templ_embed"); fake.__file__ = "/nowhere/templ_embed.py"
    monkeypatch.setitem(sys.modules, "templ_embed", fake)
    import opt_core.kernels as KS
    monkeypatch.setattr(KS, "route", lambda name: None)
    with pytest.raises(RuntimeError, match="carries no embed / tail / pack_weights"):
        TE.install({"X_TEMPL_EMBED": "1"})
    assert TE.STATE["installed"] is False and TE.STATE["state"] == "off"


def _install_with_fake_kernel(bound, monkeypatch):
    fake = types.ModuleType("templ_embed"); fake.__file__ = "/fake/templ_embed.py"

    class Unsupported(Exception):
        def __init__(self, event, detail=""):
            super().__init__(event); self.event = event
    fake.Unsupported = Unsupported
    fake.C_T, fake.C_DG, fake.C_AA, fake.SERVED_CZ = 64, 39, 32, (64, 128, 256)
    fake.pack_weights = lambda **kw: {"wt": None}
    fake.embed = lambda *a, **k: None
    fake.tail = lambda *a, **k: None
    monkeypatch.setitem(sys.modules, "templ_embed", fake)
    import opt_core.kernels as KS
    monkeypatch.setattr(KS, "route", lambda name: None)
    monkeypatch.setattr(TE.atexit, "register", lambda f: None)
    st = TE.install({"X_TEMPL_EMBED": "1"})
    assert st["installed"] and st["state"] == "on" and st["impl"] == "/fake/templ_embed.py"
    assert st["patched"] == ["x_engine.template_module.TemplateEmbedderAllAtom.forward"]
    assert getattr(bound.TemplateEmbedderAllAtom.forward, "_of3opt_templ_embed", False)
    return fake


def test_patched_forward_routes_refused_passes_to_the_stock_forward_counted(bound, monkeypatch):
    _install_with_fake_kernel(bound, monkeypatch)
    TE.install({"X_TEMPL_EMBED": "1"})                                           # idempotent: one wrapper
    assert bound.TemplateEmbedderAllAtom.forward.__wrapped__.__name__ == "forward" and not hasattr(bound.TemplateEmbedderAllAtom.forward.__wrapped__, "_of3opt_templ_embed")
    m = bound.TemplateEmbedderAllAtom()
    z = torch.zeros(1, 3, 3, 128)
    out = m(batch={}, z=z, pair_mask=torch.ones(1, 3, 3), chunk_size=4, use_cueq_triangle_kernels=True, inplace_safe=True)
    assert torch.equal(out, z + 1.0) and m.calls == [(4, True, True)]           # CPU tensor: refused `cuda:no`, the stock forward ran with the caller's arguments
    assert TE.STATE["served"] == 0 and TE.STATE["fallback"] == {"cuda:no": 1}
    m.train()
    m(batch={}, z=z, pair_mask=torch.ones(1, 3, 3))
    assert TE.STATE["fallback"] == {"cuda:no": 2}                               # the device check precedes the training check
    line = TE.census_line()
    assert line.startswith("[x-kit/templ_embed] LEVER name=templ_embed state=on impl=/fake/templ_embed.py served=0 fallback=cuda:no:2") and "first=" not in line
    # not serving (state off): the wrapper is inert
    TE.STATE["state"] = "off"
    m.eval(); m.calls.clear()
    assert torch.equal(m(batch={}, z=z, pair_mask=torch.ones(1, 3, 3)), z + 1.0) and m.calls == [(None, False, False)] and TE.STATE["fallback"] == {"cuda:no": 2}


def test_refusal_words(bound, monkeypatch):
    """refusal() names the first property outside the domain: device, training, autocast, batch, dims, feature layout."""
    _install_with_fake_kernel(bound, monkeypatch)
    lin = lambda o, i: torch.nn.Linear(i, o, bias=False)                        # noqa: E731
    pe = types.SimpleNamespace(dgram_linear=lin(64, 39), aatype_linear_1=lin(64, 32), linear_z=lin(64, 128))
    mod = types.SimpleNamespace(training=False, template_pair_embedder=pe, linear_t=lin(128, 64))
    T, N = 2, 5
    batch = {"template_distogram": torch.zeros(1, T, N, N, 39), "template_restype": torch.zeros(1, T, N, 32), "template_unit_vector": torch.zeros(1, T, N, N, 3),
             "template_pseudo_beta_mask": torch.ones(1, T, N), "template_backbone_frame_mask": torch.ones(1, T, N), "asym_id": torch.zeros(1, N)}
    z = torch.zeros(1, N, N, 128)
    assert TE.refusal(mod, batch, z) == "cuda:no"
    if not torch.cuda.is_available():
        return
    zc = z.cuda()
    if torch.cuda.get_device_capability() < (8, 0):
        assert TE.refusal(mod, batch, zc).startswith("cc:")
        return
    mod.training = True
    assert TE.refusal(mod, batch, zc) == "training"
    mod.training = False
    assert TE.refusal(mod, batch, zc) == "autocast:off"
    with torch.autocast("cuda", dtype=torch.float16):
        assert TE.refusal(mod, batch, zc) == "autocast:float16"
    with torch.autocast("cuda", dtype=torch.bfloat16):
        assert TE.refusal(mod, batch, zc) is None
        assert TE.refusal(mod, {k: v for k, v in batch.items() if k != "asym_id"}, zc) == "feat:missing:asym_id"
        assert TE.refusal(mod, batch, torch.zeros(2, N, N, 128, device="cuda")).startswith("batch:2x")
        mod.linear_t = lin(512, 64)
        assert TE.refusal(mod, batch, zc).startswith("dims:")
        mod.linear_t = lin(128, 64)
        bad = dict(batch, template_distogram=torch.zeros(1, T, N + 1, N + 1, 39))
        assert TE.refusal(mod, bad, zc).startswith("feat:")
        assert TE.refusal(mod, dict(batch, template_distogram=torch.zeros(1, T, N, N, 39, dtype=torch.int64)), zc) == "feat:dtype:int64"


def test_reference_statements_restate_the_stock_chain():
    """reference_embed / reference_tail (fp32) == a line-by-line restatement of the stock statements on CPU tensors."""
    pytest.importorskip("triton")
    from opt_core.kernels import route
    route("templ_embed")
    import importlib
    K = importlib.import_module("templ_embed")
    g = torch.Generator().manual_seed(3)
    T, N = 3, 7
    dg = torch.nn.functional.one_hot(torch.randint(0, 39, (T, N, N), generator=g), 39).float()
    uv = torch.randn(T, N, N, 3, generator=g)
    pbm = (torch.rand(T, N, generator=g) > 0.3).float(); bbm = (torch.rand(T, N, generator=g) > 0.3).float()
    asym = torch.tensor([0, 0, 0, 1, 1, 2, 2])
    restype = torch.nn.functional.one_hot(torch.randint(0, 32, (T, N), generator=g), 32).float()
    zp = torch.randn(N, N, 64, generator=g)
    W = {k: torch.randn(*s, generator=g) * 0.2 for k, s in (("w_dgram", (64, 39)), ("w_pb", (64, 1)), ("w_aa1", (64, 32)), ("w_aa2", (64, 32)), ("w_x", (64, 1)),
                                                           ("w_y", (64, 1)), ("w_z", (64, 1)), ("w_bb", (64, 1)))}
    got = K.reference_embed(dg, uv, pbm, bbm, asym, restype, zp, **W)
    same = (asym[:, None] == asym[None, :]).float()
    a = dg @ W["w_dgram"].t() + ((pbm[:, :, None] * pbm[:, None, :]) * same)[..., None] @ W["w_pb"].t()
    a = a + (restype @ W["w_aa1"].t())[:, :, None, :] + (restype @ W["w_aa2"].t())[:, None, :, :]
    a = a + uv[..., 0:1] @ W["w_x"].t() + uv[..., 1:2] @ W["w_y"].t() + uv[..., 2:3] @ W["w_z"].t() + ((bbm[:, :, None] * bbm[:, None, :]) * same)[..., None] @ W["w_bb"].t()
    assert torch.allclose(got, zp[None] + a, atol=1e-5, rtol=1e-5) and got.shape == (T, N, N, 64)
    s = torch.randn(T, N, N, 64, generator=g); w_t = torch.randn(128, 64, generator=g)
    assert torch.allclose(K.reference_tail(s, w_t), torch.relu(s.sum(0) / T) @ w_t.t(), atol=1e-5)
    with pytest.raises(K.Unsupported, match="dims:w_t"):
        K.pack_weights(w_dgram=W["w_dgram"], w_pb=W["w_pb"], w_x=W["w_x"], w_y=W["w_y"], w_z=W["w_z"], w_bb=W["w_bb"], w_t=torch.zeros(512, 64))
    with pytest.raises(K.Unsupported, match="dims:w_dgram"):
        K.pack_weights(w_dgram=torch.zeros(64, 40), w_pb=W["w_pb"], w_x=W["w_x"], w_y=W["w_y"], w_z=W["w_z"], w_bb=W["w_bb"], w_t=w_t)
    P = K.pack_weights(w_t=w_t, **{k: v for k, v in W.items() if k not in ("w_aa1", "w_aa2")})
    assert P["wd"].shape == (64, 64) and P["wd"].dtype == torch.bfloat16 and bool((P["wd"][39:] == 0).all()) and P["wt"].shape == (64, 128)
    with pytest.raises(K.Unsupported, match="device:cpu"):
        K.check_tail(s.to(torch.bfloat16), P["wt"])
