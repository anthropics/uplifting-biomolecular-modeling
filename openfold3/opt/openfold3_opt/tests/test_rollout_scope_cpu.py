"""The rollout_bf16 lever's scope (cells/rollout.py): the bf16 region is entered around the denoiser step call — `DiffusionModule.__call__` and
the fast-inference kit's `GraphedStep.__call__` when that module is loaded — and fp32 islands are entered around the denoiser's `atom_attn_enc` /
`atom_attn_dec`; hooks attach once per instance / once per class; every enter is left. CPU only: stand-in modules with the same attribute names and a
stand-in `of3_graphs` module; `torch.autocast("cuda", …)` entered on a CPU box is a disabled context, so the test reads the lever's own bookkeeping."""
import types

import pytest

torch = pytest.importorskip("torch")

from openfold3_opt.cells import rollout as R


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


@pytest.fixture()
def fresh():
    R.STATE.update({"installed": False, "denoiser_calls": 0, "regions": 0, "islands": 0, "graphed_step": 0, "dtype": "bfloat16"})
    del R._REGIONS[:]; del R._ISLANDS[:]
    yield R
    del R._REGIONS[:]; del R._ISLANDS[:]


def _probe(d, lst):
    """Record, for each submodule call, how many entries of `lst` are in flight and the innermost one (registered after the lever's own pre-hooks)."""
    seen = {}
    for name in ("diffusion_conditioning", "diffusion_transformer", "atom_attn_enc", "atom_attn_dec"):
        getattr(d, name).register_forward_pre_hook(lambda m, a, name=name: seen.__setitem__(name, (len(lst), lst[-1] if lst else None)))
    return seen


def test_islands_are_the_atom_attention_modules_only(fresh):
    assert R.FP32_ISLANDS == ("atom_attn_enc", "atom_attn_dec")
    d = _Denoiser()
    assert R.attach_islands(torch, d) == 2 and R.attach_islands(torch, d) == 0 and R.STATE["islands"] == 2      # idempotent per instance
    seen = _probe(d, R._ISLANDS)
    R.region_enter(torch); d(torch.zeros(2, 4)); R.region_leave()                                                 # one eager step: the region around the denoiser call
    assert seen["atom_attn_enc"][0] == 1 and seen["atom_attn_enc"][1] is not None                                # an fp32 island in flight while the atom modules run
    assert seen["atom_attn_dec"][0] == 1 and seen["atom_attn_dec"][1] is not None
    assert seen["diffusion_conditioning"] == (0, None) and seen["diffusion_transformer"] == (0, None)             # none while the token-level modules run (they are in the bf16 region)
    assert R.STATE["regions"] == 1 and not R._REGIONS and not R._ISLANDS                                          # balanced


def test_not_serving_enters_neither_region_nor_islands(fresh):
    d = _Denoiser()
    flag = {"serve": False}
    R.attach_islands(torch, d, serving=lambda: flag["serve"])
    seen = _probe(d, R._ISLANDS)
    R.region_enter(torch, flag["serve"]); d(torch.zeros(2, 4)); R.region_leave()
    assert R.STATE["regions"] == 0 and seen["atom_attn_enc"] == (1, None) and not R._ISLANDS and not R._REGIONS
    flag["serve"] = True
    R.region_enter(torch, flag["serve"]); d(torch.zeros(2, 4)); R.region_leave()
    assert R.STATE["regions"] == 1 and seen["atom_attn_enc"][1] is not None and not R._ISLANDS and not R._REGIONS


def test_graphed_step_call_is_wrapped_once_and_runs_in_the_region(fresh):
    inflight = []

    class GraphedStep:                                                   # stands in for of3_graphs.GraphedStep: __call__(self, batch, xl_noisy, ...)
        def __call__(self, *a, **kw):
            inflight.append(len(R._REGIONS)); return "out"

    mods = {R.GRAPHS_MODULE: types.SimpleNamespace(**{R.GRAPHS_STEP: GraphedStep})}
    assert R.wrap_graphed_step(torch, modules={}) is False and R.STATE["graphed_step"] == 0                       # graphs off: nothing to wrap
    assert R.wrap_graphed_step(torch, modules=mods) is True and R.wrap_graphed_step(torch, modules=mods) is False  # once, class-wide
    step = GraphedStep()
    assert step(1, x=2) == "out" and inflight == [1] and R.STATE["regions"] == 1 and not R._REGIONS              # the add-ons' eager work inside the call sees the region
    flag = {"serve": False}
    mods2 = {R.GRAPHS_MODULE: types.SimpleNamespace(**{R.GRAPHS_STEP: type("GraphedStep", (), {"__call__": lambda self: len(R._REGIONS)})})}
    R.wrap_graphed_step(torch, serving=lambda: flag["serve"], modules=mods2)
    g = mods2[R.GRAPHS_MODULE].GraphedStep()
    assert g() == 1 and R.STATE["regions"] == 1                                                                   # gated: an inert entry, no bf16 region counted
    flag["serve"] = True
    assert g() == 1 and R.STATE["regions"] == 2 and not R._REGIONS


def test_census_words(fresh):
    R.STATE["installed"] = True
    line = R.census_line()
    for word in ("LEVER name=rollout_bf16 state=on", "denoiser_calls=0", "bf16_regions=0", "fp32_islands=0", "graphed_step=0", "dtype=bfloat16"):
        assert word in line
    assert R.requested({}) is False and R.requested({R.ENV: "bf16"}) is True
    with pytest.raises(ValueError):
        R.requested({R.ENV: "fp16"})


def test_install_wraps_the_two_classes_once_and_registers_no_process_wide_hooks(fresh, monkeypatch):
    """install(): SampleDiffusion.forward / DiffusionModule.forward wrapped class-wide, once (a second install or wrap is a no-op); the
    process-wide module hook registries are untouched; a denoiser call counts, attaches the islands to the instance and runs in the region."""
    import sys
    hooks_mod = torch.nn.modules.module
    n_pre, n_post = len(hooks_mod._global_forward_pre_hooks), len(hooks_mod._global_forward_hooks)

    class SampleDiffusion(torch.nn.Module):
        def forward(self, x):
            return self.dm(x)

    class DiffusionModule(_Denoiser):
        pass
    dm_mod = types.ModuleType("openfold3.core.model.structure.diffusion_module")
    dm_mod.SampleDiffusion, dm_mod.DiffusionModule = SampleDiffusion, DiffusionModule
    monkeypatch.setitem(sys.modules, "openfold3.core.model.structure.diffusion_module", dm_mod)
    monkeypatch.setattr(R.atexit, "register", lambda f: None)
    st = R.install({R.ENV: "bf16"})
    assert st["installed"] and st["wrapped"] == ["DiffusionModule.forward", "SampleDiffusion.forward"]
    assert R.wrap_classes(torch, SampleDiffusion, DiffusionModule) == []                                            # once, class-wide
    assert (len(hooks_mod._global_forward_pre_hooks), len(hooks_mod._global_forward_hooks)) == (n_pre, n_post)         # no process-wide hooks
    s = SampleDiffusion(); s.dm = DiffusionModule()
    seen = _probe(s.dm, R._REGIONS)
    s(torch.zeros(2, 4)); s(torch.zeros(2, 4))
    assert R.STATE["denoiser_calls"] == 2 and R.STATE["regions"] == 2 and R.STATE["islands"] == 2 and not R._REGIONS and not R._ISLANDS
    assert seen["diffusion_transformer"][0] == 1 and seen["atom_attn_enc"][0] == 1                                    # the region is in flight around the denoiser body
    assert "hooks=class" in R.census_line()
