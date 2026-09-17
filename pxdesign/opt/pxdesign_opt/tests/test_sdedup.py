"""sdedup: the hook module the carried hoist lever consults (pxd_xattempt.fuse) evaluates the single conditioning on one sample row through
opt_core.capture.hoist.RowDedup — one row out when t_hat is uniform across the sample dimension (served), the stock rows when it is not
(fallback, counted), one row when there is one sample (single); apply() binds the module only when the carried lever consults the hook."""
import sys
import types

import pytest

from pxdesign_opt import sdedup

torch = None                                   # real torch, imported lazily by the `site` fixture (a stub torch installed by another test file is a skip, never an import at collection)


def _real_torch():
    t = pytest.importorskip("torch")
    if not hasattr(t, "arange") or not hasattr(t, "Generator"):
        pytest.skip("torch in this interpreter is a stub")
    return t


def _Cond(c_s=8, c_noise=6):
    """A stand-in DiffusionConditioning single path: fourier_embedding ([..., N] -> [..., N, c]), layernorm_n, linear_no_bias_n, transition_s1/s2."""
    nn = torch.nn

    class Fourier(nn.Module):
        def __init__(self, c):
            super().__init__(); g = torch.Generator().manual_seed(0)
            self.w = nn.Parameter(torch.randn(c, generator=g)); self.b = nn.Parameter(torch.randn(c, generator=g))

        def forward(self, t_hat_noise_level):
            return torch.cos(2 * torch.pi * (t_hat_noise_level.unsqueeze(-1) * self.w + self.b))

    class Cond(nn.Module):
        def __init__(self):
            super().__init__(); torch.manual_seed(1)
            self.sigma_data = 16.0
            self.fourier_embedding = Fourier(c_noise)
            self.layernorm_n = nn.LayerNorm(c_noise)
            self.linear_no_bias_n = nn.Linear(c_noise, c_s, bias=False)
            self.transition_s1 = nn.Sequential(nn.LayerNorm(c_s), nn.Linear(c_s, c_s))
            self.transition_s2 = nn.Sequential(nn.LayerNorm(c_s), nn.Linear(c_s, c_s))
    return Cond()


@pytest.fixture()
def site():
    global torch
    torch = _real_torch()
    sdedup.reset_for_tests()
    sdedup._STATE["on"] = True
    yield sdedup
    sdedup.reset_for_tests()


def _full(cond, t_hat, base):                  # the stock rows: the same statements on every sample row
    return sdedup._single_path(cond, base, inplace_safe=False)(t_hat)


def test_uniform_t_hat_is_served_on_one_row(site):
    cond, base = _Cond(), torch.randn(5, 8)                      # N_tok 5, c_s 8
    t_hat = torch.full((4,), 3.5)                                # N_sample 4, one noise level (the sampler's case)
    y = site._cond_forward_single_dedup(cond, t_hat, None, None, False, single_base=base)
    assert y.shape == (1, 5, 8)
    full = _full(cond, t_hat, base)
    assert full.shape == (4, 5, 8) and torch.allclose(y.expand(4, 5, 8), full, atol=1e-6)
    st = site.stats()
    assert st["served"] == 1 and st["fallback"] == 0 and st["rows_saved"] == 3 and site.gate() == []
    y2 = site._cond_forward_single_dedup(cond, t_hat, None, None, True, single_base=base)   # inplace_safe branch: same value
    assert torch.allclose(y2, y, atol=1e-6) and site.stats()["served"] == 2


def test_nonuniform_t_hat_runs_the_stock_rows_counted(site, capsys):
    cond, base = _Cond(), torch.randn(5, 8)
    t_hat = torch.tensor([3.5, 3.5, 2.0])
    y = site._cond_forward_single_dedup(cond, t_hat, None, None, False, single_base=base)
    assert y.shape == (3, 5, 8) and torch.equal(y, _full(cond, t_hat, base))
    st = site.stats()
    assert st["fallback"] == 1 and st["fallback_by"] == {"nonuniform": 1} and st["served"] == 0
    problems = site.gate()
    assert problems and any("nonuniform" in p or "fallback" in p for p in problems) and any("served" in p for p in problems)


def test_one_sample_and_no_base(site):
    cond, base = _Cond(), torch.randn(5, 8)
    y = site._cond_forward_single_dedup(cond, torch.tensor([3.5]), None, None, False, single_base=base)
    assert y.shape == (1, 5, 8) and site.stats()["single"] == 1
    assert site.gate() == []                                        # an N_sample 1 run: every call `single`, nothing to deduplicate — the exit gate passes (warm --mode fast is this job)
    D = site.dedup(); D.stats_["calls"] += 2                        # dedup-able calls that were never served: the gate names it
    assert site.gate() and "never served" in site.gate()[0]
    D.stats_["calls"] -= 2
    assert site._cond_forward_single_dedup(cond, torch.tensor([3.5, 3.5]), None, None, False, single_base=None) is None   # no step-invariant base: hoist runs its own stock lines
    site._STATE["on"] = False
    assert site._cond_forward_single_dedup(cond, torch.tensor([3.5, 3.5]), None, None, False, single_base=base) is None and site.state() == {"sdedup": False, "qkvg": False}


def _hoist_consulting():
    def _f_forward_hoisted(self, *a, **kw):        # the carried lever's call into the hook, by source
        _fuse = None
        return _fuse._cond_forward_single_dedup(self)  # noqa
    m = types.ModuleType("pxd_xattempt.hoist"); m._f_forward_hoisted = _f_forward_hoisted
    return m


def _hoist_not_consulting():
    def _f_forward_hoisted(self, *a, **kw):
        return None
    m = types.ModuleType("pxd_xattempt.hoist"); m._f_forward_hoisted = _f_forward_hoisted
    return m


def test_apply_binds_the_hook_only_when_the_lever_consults_it(monkeypatch):
    sdedup.reset_for_tests()
    pkg = types.ModuleType("pxd_xattempt")
    monkeypatch.setitem(sys.modules, "pxd_xattempt", pkg)
    fields = sdedup.apply(_hoist_not_consulting())
    assert fields["hook"] is False and "source guard" in fields["reason"] and not hasattr(pkg, "fuse") and sdedup.state()["sdedup"] is False
    fields = sdedup.apply(_hoist_consulting())
    assert fields == {"hook": True, "class": "tolerance"} and pkg.fuse is sdedup and sys.modules["pxd_xattempt.fuse"] is sdedup and sdedup.state()["sdedup"] is True
    from pxd_xattempt import fuse                   # the carried lever's own import statement resolves to this module
    assert fuse is sdedup
    other = types.ModuleType("someone_elses_fuse"); pkg.fuse = other
    fields = sdedup.apply(_hoist_consulting())
    assert fields["hook"] is False and "already bound" in fields["reason"]
    pkg.fuse = sdedup
    sdedup.reset_for_tests()
    assert not hasattr(pkg, "fuse") and "pxd_xattempt.fuse" not in sys.modules


def test_lever_lines(monkeypatch):
    sdedup.reset_for_tests()
    off = sdedup.lever_line("pxdesign-opt", planned=False)
    assert off.startswith("[pxdesign-opt] LEVER name=sdedup state=off") and "strategy=F7.row_dedup" in off and "origin=core" in off
    skipped = sdedup.lever_line("pxdesign-opt", planned=True)
    assert "state=skipped reason=not_applied" in skipped
    sdedup._STATE["on"] = True; sdedup.dedup()
    on = sdedup.lever_line("pxdesign-opt", planned=True)
    assert on.startswith("[pxdesign-opt] LEVER name=sdedup state=on") and "impl=row_dedup" in on and "class=tolerance" in on and "served=0" in on and on.count("strategy=") == 1
    assert sdedup.dedup().verify_every == 0                 # a tolerance site: the core's per-call equality demand stays off
    sdedup.reset_for_tests()
