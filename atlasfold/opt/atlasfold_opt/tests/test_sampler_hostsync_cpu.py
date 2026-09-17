"""CPU tests of lever sampler_hostsync: the re-stated DiffusionHead.sample (mask predicate once per roll-out, c_noise schedule as one table) returns
torch.equal coordinates vs the stock roll-out on a small random-parameter head (same seed), alone and under sampler_hoist; the schedule table holds
exactly the per-step `torch.tensor(c_noise(t_hat))` values; install refuses by name when sample() is already wrapped or a re-stated text changed;
AFO_SAMPLER_HOSTSYNC=0 runs the stock body counted `disabled`; registry / modes / installers coverage and row order (innermost: before every
lever that wraps sample())."""
import pytest
import torch

from atlasfold_opt.hooks import sampler_hostsync as HS
from atlasfold_opt.tests.test_sampler_hoist_cpu import _head, _batch, _sample, stock_sites   # noqa: F401  (the same small stock head + batch builders; the autouse site fixture)


@pytest.fixture
def restore_sample(stock_sites):
    from atlasfold.model.network.diffusion_head import DiffusionHead
    stock = DiffusionHead.sample
    assert not hasattr(stock, "__wrapped_stock__")            # the site fixture hands every test the innermost callable
    yield
    DiffusionHead.sample = stock


def test_restated_rollout_is_bitwise_stock_and_counts_syncs(restore_sample, monkeypatch):
    monkeypatch.delenv(HS.ENV_SWITCH, raising=False)
    head = _head(21); batch, s, z = _batch(B=2, L=20, seed=22)
    x0 = _sample(head, batch, s, z, num_samples=3, steps=4)
    ins = HS.install("exact", "atlasfold-opt", {})
    assert ins.applied, ins.reason
    x1 = _sample(head, batch, s, z, num_samples=3, steps=4)
    assert torch.equal(x0, x1)
    led = ins.facts["ledger"]
    assert led.served == 1 and led.get("rollouts") == 1 and led.get("steps") == 4
    assert led.get("host_syncs") == 1 and led.get("host_syncs_avoided") == 3 * 4 - 1
    assert ins.gates[0]().ok and "sampler_hostsync" in ins.lines[0]()


def test_two_chunks_and_fully_masked_item_bitwise(restore_sample, monkeypatch):
    monkeypatch.delenv(HS.ENV_SWITCH, raising=False)
    head = _head(23); batch, s, z = _batch(B=1, L=12, seed=24)
    x0 = _sample(head, batch, s, z, num_samples=7, steps=3)
    ins = HS.install("exact", "atlasfold-opt", {})
    x1 = _sample(head, batch, s, z, num_samples=7, steps=3)
    assert torch.equal(x0, x1) and ins.facts["ledger"].served == 1


def test_schedule_table_holds_the_per_step_scalars():
    from atlasfold.model.network.diffusion_head import SamplingConfig, DiffusionHead
    cfg = SamplingConfig(num_steps=200)
    sch = HS.schedule(cfg)
    head = DiffusionHead.__new__(DiffusionHead); head.sigma_data = 16.0
    table = torch.tensor([head.c_noise(t) for t in sch["t_hats"]]).view(-1, 1, 1)
    assert table.dtype == torch.float32 and table.shape == (200, 1, 1)
    sigmas = cfg.get_sigmas()
    for step in (1, 2, 57, 199, 200):                                   # the loop's statements, verbatim, per step
        sigma_tm, sigma_t = sigmas[step - 1], sigmas[step]
        gamma = cfg.gamma_0 * (sigma_t > cfg.gamma_min)
        t_hat = sigma_tm * (1 + gamma)
        assert torch.equal(table[step - 1], torch.tensor(head.c_noise(t_hat)).view(1, 1))


def test_composes_under_sampler_hoist(restore_sample, monkeypatch):
    """Row order: hostsync innermost, sampler_hoist over it — still the stock numbers."""
    from atlasfold_opt.hooks import sampler_hoist as SH
    monkeypatch.delenv(HS.ENV_SWITCH, raising=False); monkeypatch.delenv(SH.ENV_SWITCH, raising=False); monkeypatch.delenv(SH.ENV_HOISTS, raising=False)
    head = _head(25); batch, s, z = _batch(B=2, L=16, seed=26)
    x0 = _sample(head, batch, s, z)
    ins1 = HS.install("exact", "atlasfold-opt", {}); ins2 = SH.install("exact", "atlasfold-opt", {})
    try:
        assert ins1.applied and ins2.applied
        x1 = _sample(head, batch, s, z)
        assert torch.equal(x0, x1)
        assert ins1.facts["ledger"].served == 1 and ins2.facts["ledger"].served == 1
    finally:
        ins2.facts["restore"](); SH.remove_memos(head); ins1.facts["restore"]()


def test_refuses_by_name_when_not_innermost(restore_sample):
    from atlasfold.model.network.diffusion_head import DiffusionHead
    from atlasfold_opt.hooks import rebind
    stock = DiffusionHead.sample

    def wrapper(self, *a, **k):
        return stock(self, *a, **k)
    wrapper.__qualname__ = "DiffusionHead.sample[some:other]"
    rebind(DiffusionHead, "sample", wrapper, stock)
    ins = HS.install("fast", "atlasfold-opt", {})
    assert not ins.applied and ins.reason.startswith("not_innermost:")
    DiffusionHead.sample = stock


def test_refuses_by_name_on_a_changed_stock_text(restore_sample, monkeypatch):
    monkeypatch.setitem(HS.SOURCE_SHA256, "get_center", ("0" * 64,))
    ins = HS.install("exact", "atlasfold-opt", {})
    assert not ins.applied and ins.reason == "source:get_center" and len(ins.facts["digests"]["get_center"]) == 64


def test_switch_off_runs_the_stock_body_counted_disabled(restore_sample, monkeypatch):
    monkeypatch.setenv(HS.ENV_SWITCH, "0")
    head = _head(27); batch, s, z = _batch(B=1, L=8, seed=28)
    x0 = _sample(head, batch, s, z)
    ins = HS.install("exact", "atlasfold-opt", {})
    x1 = _sample(head, batch, s, z)
    assert torch.equal(x0, x1)
    led = ins.facts["ledger"]
    assert led.served == 0 and led.fallbacks == {"disabled": 1} and ins.gates[0]().ok and "AFO_SAMPLER_HOSTSYNC=0" in ins.lines[0]()


def test_registry_modes_installers_and_row_order():
    from atlasfold_opt import modes, registry
    from atlasfold_opt.hooks import installers
    assert registry.LEVERS["sampler_hostsync"]["cls"] == "exact" and registry.LEVERS["sampler_hostsync"]["expected"] == ("disabled",)
    assert installers()["sampler_hostsync"] is HS.install
    ex, fa = modes.MODES[modes.EXACT], modes.MODES[modes.FAST]
    assert "sampler_hostsync" in ex and "sampler_hostsync" in fa                                   # R3: exact ⊂ fast
    assert ex.index("sampler_hostsync") < ex.index("sampler_hoist")                                # innermost: before every lever that wraps sample()
    for later in ("diffusion_bf16", "denoiser_graph", "sampler_hoist"):
        if later in fa:
            assert fa.index("sampler_hostsync") < fa.index(later), later
