"""The rollout_bf16 lever's token gate and scope (cells/rollout.py): below a caller's OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS (default 0 = no gate; the fixture pins 600) the sampler call is gated
(its denoiser calls stay fp32) and the census says gated=<n>; at or above it the call is served — bf16 autocast entered around the denoiser's
token-level submodules (`diffusion_conditioning`, `diffusion_transformer`) only, never around the atom encoder/decoder or the sampler's own
coordinate arithmetic; an unreadable batch is served and counted unsized; the threshold's grammar; fallbacks() stays empty on both sides (a gate
is not a fallback). CPU only: the hook bodies are called directly with a stand-in batch and a stand-in denoiser."""
import pytest

torch = pytest.importorskip("torch")

from openfold3_ob0_opt.cells import rollout as R


@pytest.fixture()
def fresh():
    R.STATE.update({"installed": False, "calls": 0, "served": 0, "gated": 0, "unsized": 0, "denoiser_calls": 0, "regions": 0, "islands": 0, "graphed_step": 0,
                    "dtype": "bfloat16", "min_tokens": 600, "n_tokens": None, "logged": set()})
    del R._CTXS[:]; del R._REGIONS[:]; del R._ISLANDS[:]
    yield R
    del R._CTXS[:]; del R._REGIONS[:]; del R._ISLANDS[:]


def _batch(n):
    return ({"token_mask": torch.ones(1, n)},)


def test_threshold_grammar():
    assert R.min_tokens({}) == 0 == R.MIN_TOKENS_DEFAULT                       # no gate unless the caller sets one
    assert R.min_tokens({R.ENV_MIN_TOKENS: "0"}) == 0 and R.min_tokens({R.ENV_MIN_TOKENS: " 800 "}) == 800
    for bad in ("abc", "-1", "6e2"):
        with pytest.raises(ValueError):
            R.min_tokens({R.ENV_MIN_TOKENS: bad})
    with pytest.raises(ValueError):                                   # a mistyped threshold is refused even with the lever off
        R.requested({R.ENV_MIN_TOKENS: "x"})
    assert R.requested({}) is False and R.requested({R.ENV: "bf16", R.ENV_MIN_TOKENS: "600"}) is True


def test_below_threshold_is_gated_by_name(fresh, capsys):
    R.observe_model_call(_batch(400))
    assert R.STATE["n_tokens"] == 400 and R.gate_decision() == (False, "n_tokens=400 < 600")
    R.sampler_pre(torch); R.sampler_post()
    assert (R.STATE["calls"], R.STATE["served"], R.STATE["gated"], R.STATE["unsized"]) == (1, 0, 1, 0) and not R._CTXS
    err = capsys.readouterr().err
    assert "rollout_bf16: n_tokens=400 < 600 (OPENFOLD3_OB0_OPT_ROLLOUT_MIN_TOKENS) -> upstream's fp32 rollout" in err and "by design" in err
    line = R.census_line()
    assert "LEVER name=rollout_bf16" in line and "served=0" in line and "gated=1" in line and "min_tokens=600" in line
    assert R.fallbacks() == {}                                          # a gate is not a fallback


def test_at_threshold_and_above_is_served(fresh):
    for n in (600, 800):
        R.observe_model_call(_batch(n))
        assert R.gate_decision()[0] is True
        R.sampler_pre(torch)
        assert R._CTXS and R._CTXS[-1] is not None and R.serving()     # the call is marked served: its denoiser calls enter the bf16 regions
        R.sampler_post()
        assert not R.serving()
    assert (R.STATE["calls"], R.STATE["served"], R.STATE["gated"]) == (2, 2, 0) and not R._CTXS
    assert "served=2 gated=0" in R.census_line() and R.fallbacks() == {}


def test_unreadable_batch_is_served_and_counted_unsized(fresh):
    R.observe_model_call(())                                            # the model called with keyword arguments only: no positional batch
    assert R.STATE["n_tokens"] is None and R.gate_decision() == (True, "unsized")
    R.sampler_pre(torch); R.sampler_post()
    R.observe_model_call(({"no_token_mask_here": 1},))
    R.sampler_pre(torch); R.sampler_post()
    assert (R.STATE["served"], R.STATE["unsized"], R.STATE["gated"]) == (2, 2, 0) and "unsized=2" in R.census_line()


def test_threshold_zero_serves_everything(fresh):
    R.STATE["min_tokens"] = 0
    R.observe_model_call(_batch(61))
    assert R.gate_decision() == (True, "n_tokens=61 >= 0")


class _Denoiser(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.diffusion_conditioning = torch.nn.Linear(4, 4)
        self.atom_attn_enc = torch.nn.Linear(4, 4)
        self.diffusion_transformer = torch.nn.Linear(4, 4)
        self.atom_attn_dec = torch.nn.Linear(4, 4)

    def forward(self, x):
        for name in ("diffusion_conditioning", "atom_attn_enc", "diffusion_transformer", "atom_attn_dec"):
            x = getattr(self, name)(x)
        return x


def test_scope_inside_a_served_call_is_the_step_with_fp32_atom_islands(fresh):
    assert R.FP32_ISLANDS == ("atom_attn_enc", "atom_attn_dec")
    d = _Denoiser()
    assert R.attach_islands(torch, d, R.serving) == 2 and R.attach_islands(torch, d, R.serving) == 0 and R.STATE["islands"] == 2
    seen = {}
    for name in ("diffusion_conditioning", "diffusion_transformer", "atom_attn_enc", "atom_attn_dec"):
        getattr(d, name).register_forward_pre_hook(lambda m, a, name=name: seen.__setitem__(name, (len(R._ISLANDS), R._ISLANDS[-1] if R._ISLANDS else None)))
    R.observe_model_call(_batch(400)); R.sampler_pre(torch)               # gated: neither the region nor an island is entered
    R.region_enter(torch, R.serving()); d(torch.zeros(2, 4)); R.region_leave()
    assert R.STATE["regions"] == 0 and seen["atom_attn_enc"] == (1, None) and not R.serving()
    R.sampler_post(); assert not R._REGIONS and not R._ISLANDS
    R.observe_model_call(_batch(800)); R.sampler_pre(torch)               # served: the region around the step, an fp32 island while the atom modules run, none for the token-level ones
    R.region_enter(torch, R.serving()); d(torch.zeros(2, 4)); R.region_leave()
    assert seen["atom_attn_enc"][0] == 1 and seen["atom_attn_enc"][1] is not None and seen["atom_attn_dec"][1] is not None
    assert seen["diffusion_conditioning"] == (0, None) and seen["diffusion_transformer"] == (0, None)
    R.sampler_post()
    assert R.STATE["regions"] == 1 and not R._REGIONS and not R._ISLANDS and not R._CTXS and "bf16_regions=1" in R.census_line() and "fp32_islands=2" in R.census_line()


def test_graphed_step_wrap_follows_the_gate(fresh):
    import types
    inflight = []
    cls = type("GraphedStep", (), {"__call__": lambda self, *a, **kw: inflight.append((len(R._REGIONS), R._REGIONS[-1] if R._REGIONS else None))})
    mods = {R.GRAPHS_MODULE: types.SimpleNamespace(**{R.GRAPHS_STEP: cls})}
    assert R.wrap_graphed_step(torch, R.serving, modules=mods) is True and R.wrap_graphed_step(torch, R.serving, modules=mods) is False and R.STATE["graphed_step"] == 1
    g = cls()
    R.observe_model_call(_batch(400)); R.sampler_pre(torch); g(); R.sampler_post()
    R.observe_model_call(_batch(800)); R.sampler_pre(torch); g(); R.sampler_post()
    assert inflight[0] == (1, None) and inflight[1][0] == 1 and inflight[1][1] is not None and R.STATE["regions"] == 1 and not R._REGIONS
