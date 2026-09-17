"""A mode is ALL of its levers on the card: a GPU below the kit modes' floor (sm80) makes each of the kit's modes refuse BY NAME before anything runs
(stack.activate: a named NOT ACTIVE problem, not a note); an untested class at or above the floor engages every lever and names the uncertainty."""
import pytest

from openfold3_opt import stack


def _probe(cc, name="NVIDIA Test GPU"):
    g = {"name": name, "cc": cc, "sm": "sm" + cc.replace(".", ""), "memory_mib": 16000, "probe": "test"}
    g["supported"] = float(cc) >= stack.MIN_CC
    return g


def test_a_card_below_sm80_is_a_named_refusal_of_the_mode(monkeypatch):
    monkeypatch.setattr(stack, "gpu_probe", lambda environ=None: _probe("7.5", "NVIDIA T4"))
    rep = stack.activate("exact", dry_run=True, environ={"PATH": "/bin"}, strict=False)
    assert rep["active"] is False and "below the kit modes' floor (sm80+" in (rep.get("reason") or "") and "--mode off runs stock" in rep["reason"]
    with pytest.raises(stack.ActivationError):
        stack.activate("fast", dry_run=True, environ={"PATH": "/bin"}, strict=True)


def test_an_untested_card_at_or_above_the_floor_is_not_refused_for_that(monkeypatch):
    monkeypatch.setattr(stack, "gpu_probe", lambda environ=None: _probe("8.9", "NVIDIA L40S"))
    rep = stack.activate("exact", dry_run=True, environ={"PATH": "/bin"}, strict=False)
    assert "below the kit modes' floor" not in (rep.get("reason") or "")
