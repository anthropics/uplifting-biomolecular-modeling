"""sched_host — the sampler loop's two per-step host round-trips answered on the host / made non-blocking (opendde_opt/schedhost.py):
composition on the lines, the HostSchedule view's semantics (the ONE host-answered op is `0-d element > python number`, everything else is
the stock device op with a plain result), torch's comparison semantics kept per dtype, and the stock sampler bitwise through the view."""
import os
import sys

import pytest

from opendde_opt import modes, ran, registry, schedhost

HERE = os.path.dirname(os.path.abspath(__file__))


def test_composition_on_the_lines():
    assert registry.LEVERS["sched_host"].tier == "exact" and registry.LEVERS["sched_host"].kit == registry.HOUSE
    for name in ("S1", "LSTAR2A", "BIG_F"):
        assert "sched_host" in modes.LINES[name].levers, name
    assert "sched_host" not in modes.LINES["BIG_TP"].levers                                # the rank processes: stock loop (modes.BIG_TP_DROP)
    assert "sched_host" in modes.LEVER_SWITCHES and "sched_host" in ran.COUNTERS
    out = modes.line_without(modes.LINES["S1"], ("sched_host",))
    assert "sched_host" not in out.levers and "stepgraph" in out.levers                       # individually switchable; the step graph does not depend on it


@pytest.fixture
def cpu_view(monkeypatch):
    torch = pytest.importorskip("torch")
    schedhost._reset()
    monkeypatch.setitem(schedhost.STATS, "wrap_cpu", True)
    yield torch
    schedhost._reset()


def test_host_schedule_semantics(cpu_view):
    torch = cpu_view
    base = torch.linspace(3.0, 0.01, 9)
    hs = schedhost.host_schedule(base.clone())
    cls = schedhost._schedule_class()
    assert isinstance(hs, cls) and torch.equal(hs, base) and len(hs) == 9 and schedhost.STATS["sched_calls"] == 1
    assert isinstance(hs[:-1], cls) and isinstance(hs[1:], cls) and isinstance(hs[0], cls)
    pairs = list(zip(hs[:-1], hs[1:]))
    assert len(pairs) == 8 and all(isinstance(a, cls) and isinstance(b, cls) for a, b in pairs)
    n0 = schedhost.STATS["host_cmp"]
    for (a, b), (pa, pb) in zip(pairs, zip(base[:-1], base[1:])):
        g = (b > 1.0)
        assert isinstance(g, bool) and g == bool(pb > 1.0)                                   # the host-answered op: a python bool, torch's result
        t_hat = a * (0.8 + 1)                                                                 # every other op: the stock op, a PLAIN tensor, the same bits
        assert type(t_hat) is torch.Tensor and torch.equal(t_hat, pa * 1.8)
        d = torch.sqrt(t_hat ** 2 - a ** 2)
        assert type(d) is torch.Tensor and torch.equal(d, torch.sqrt((pa * 1.8) ** 2 - pa ** 2))
        assert type(b - t_hat) is torch.Tensor
    assert schedhost.STATS["host_cmp"] == n0 + 8
    x = hs[0] * torch.ones(2, 3)
    assert type(x) is torch.Tensor and torch.equal(x, base[0] * torch.ones(2, 3))
    assert type(hs > 1.0) is torch.Tensor                                                     # a 1-D comparison is not the loop's op: the stock device op
    assert schedhost.host_schedule(torch.ones(2, 2)) is not None and schedhost.STATS["aside"].get("schedule_not_1d_float") == 1


def test_comparison_semantics_follow_the_schedule_dtype(cpu_view):
    """`element > gamma_min` on the host copy = torch's own tensor-vs-python-scalar comparison in the schedule's dtype (bf16 / fp16 / fp32),
    including thresholds the dtype cannot represent and elements equal to the rounded threshold."""
    torch = cpu_view
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        vals = torch.tensor([2.0, 1.0009765625, 1.0, 0.99951171875, 0.05, 0.0500001, 0.0499999, 0.0], dtype=dtype)
        hs = schedhost.host_schedule(vals.clone())
        for g in (1.0, 0.05, 0.0500001, 1.0004, 0.0):
            want = [bool(v > g) for v in vals]                                                # torch: 0-d tensor vs python float
            got = [e > g for e in hs]
            assert got == want, (dtype, g, got, want)


def _upstream():
    torch = pytest.importorskip("torch")
    pytest.importorskip("opendde.model.modules.diffusion")
    sys.path.insert(0, HERE)
    import _tiny_sampler
    return torch, _tiny_sampler


@pytest.fixture(scope="module", autouse=True)
def _upstream_modules_leave_with_this_module():
    before = {k for k in sys.modules if k.split(".")[0] == "opendde"}
    yield
    for k in [k for k in sys.modules if k.split(".")[0] == "opendde" and k not in before]:
        del sys.modules[k]
    for k in [k for k in sys.modules if k.rsplit(".", 1)[-1] == "_tiny_sampler"]:
        del sys.modules[k]


def test_the_stock_sampler_is_bitwise_through_the_view(cpu_view):
    """The real upstream sampler at tiny widths (CPU): schedule built by the patched scheduler (HostSchedule), every per-step `c_tau >
    gamma_min` answered on the host (host_cmp = steps), the roll-out bitwise the stock one."""
    torch, tiny = _upstream()
    import opendde.model.generator as G
    b = tiny.build()
    ref = tiny.run_sampler(b, n_step=6, n_sample=2)                                          # stock (the scheduler unpatched or its view inert on CPU)
    schedhost.install()
    assert getattr(G.InferenceNoiseScheduler.__call__, "_sched_host", False) and getattr(G.centre_random_augmentation, "_sched_host", False)
    schedhost._reset(); schedhost.STATS["wrap_cpu"] = True
    out = tiny.run_sampler(b, n_step=6, n_sample=2)
    st = schedhost.kit_stats()
    assert torch.equal(out, ref)
    assert st["sched_calls"] == 1 and st["host_cmp"] == 6 and st["installed"] is True, st
    assert st["aside"].get("coords_not_cuda") == 6 and st["rot_copies"] == 0                 # CPU coordinates: the rotation copy path steps aside by name
    assert schedhost.fallbacks(["sched_host"]) == [] and ran.count("sched_host") == 6
