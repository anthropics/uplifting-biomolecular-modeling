"""The host-sync census: counts torch's sync warnings by call site under a stand-in torch, restores the previous debug mode, re-emits foreign
warnings, and its at_most gate — CPU, no torch, no engine."""
import warnings

import pytest

from opt_core.diffusion_loop import sync_census as sc


class FakeCuda:
    def __init__(self):
        self.mode, self.history = "default", []

    def get_sync_debug_mode(self):
        return self.mode

    def set_sync_debug_mode(self, m):
        self.mode = m
        self.history.append(m)


class FakeTorch:
    def __init__(self):
        self.cuda = FakeCuda()


def _item():           # a stand-in for tensor.item(): torch warns at the call site
    warnings.warn("called a synchronizing CUDA operation (Triggered internally)", UserWarning, stacklevel=2)
    return 1.0


def step():
    total = 0.0
    for _ in range(3):
        total += _item()                       # site A, three times
    total += _item()                           # site B, once
    warnings.warn("unrelated deprecation", DeprecationWarning)
    return total


def test_sync_census_counts_by_site_and_restores_mode():
    fake = FakeTorch()
    with warnings.catch_warnings(record=True) as outer:
        warnings.simplefilter("always")
        census = sc.sync_census(step, torch_module=fake)
    assert census.syncs == 4 and census.result == 4.0
    assert [s.count for s in census.sites] == [3, 1] and all(":" in s.where for s in census.sites)
    assert census.sites[0].text.startswith("called a synchronizing CUDA operation")
    assert fake.cuda.history == ["warn", "default"] and fake.cuda.mode == "default"
    assert [str(w.message) for w in outer] == ["unrelated deprecation"]        # foreign warnings are re-emitted, not swallowed
    assert census.evidence_fields() == {"syncs": 4, "sites": 2}


def test_sync_census_restores_mode_when_the_step_raises():
    fake = FakeTorch()

    def boom():
        _item()
        raise RuntimeError("step failed")

    with pytest.raises(RuntimeError):
        sc.sync_census(boom, torch_module=fake)
    assert fake.cuda.mode == "default"


def test_at_most_gate():
    fake = FakeTorch()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        census = sc.sync_census(step, torch_module=fake)
    g = census.at_most(4)
    assert g.ok and g.details["syncs"] == 4 and g.details["limit"] == 4
    g = census.at_most(0, name="step0")
    assert not g.ok and g.name == "step0" and "step made 4 host syncs > 0 allowed" in g.reason
