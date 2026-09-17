"""The exact row on compute capability 8.0 (A100) after the sm_80 proofs [0.3.5]: the token transformer's fused softmax / glue replicas (dit_smx,
dit_glue: ditexact PROVEN_CC) and the MSA transition's replica (trans2x, lever msa_trans2_exact: msa2.EXACT_PIN per unit) are admitted on 8.0
— their run-time per-class bit-compare stays the guard —, the MSA PairWeightedAveraging's exact cell (pwa2x, lever msa_pwa_exact) stays pinned to
9.0 by name. CPU tests: the card is simulated."""
import importlib.util
import os
import sys
import types

import pytest

from .. import modes, msa2

HERE = os.path.dirname(os.path.abspath(__file__))


def test_card_drops_on_8_0_name_only_the_levers_without_an_sm_80_proof():
    assert modes.CARD_DROPS["8.0"]["exact"] == {"msa_pwa_exact": {"reason": "pin:cc80"}}, "the exact tri-attention lever rides on 8.0: the core provider's `exact` word names the card's row"
    assert {m: set(d) for m, d in modes.CARD_DROPS["8.0"].items() if m != "exact"} == {"big": {"atom_fused"}}, \
        "fast on 8.0: no lever leaves (the tri-attention site is the provider's tier word); the memory row names off fast's fused atom kernels there (measured device memory outside the allocator on sm_80), by name"
    assert {"dit_smx", "dit_glue", "msa_trans2_exact"}.isdisjoint(modes.CARD_DROPS["8.0"]["exact"]) and {"dit_smx", "dit_glue", "msa_trans2_exact"} <= set(modes.CARD_DROPS["10.0"]["exact"]), \
        "admitted on 8.0 (proven there), still leaving on 10.0 (no proof there)"


def test_msa2_exact_pin_is_per_unit_trans2x_admits_8_0_pwa2x_does_not(monkeypatch):
    assert msa2.EXACT_PIN["cc"] == {"pwa2x": ((9, 0),), "trans2x": ((9, 0), (8, 0))} and msa2.EXACT_PIN["torch"] == "2.12." and msa2.EXACT_PIN["cuda"] == "13.0"
    torch = pytest.importorskip("torch")
    fake = types.SimpleNamespace(is_available=lambda: True, get_device_capability=lambda i=0: (8, 0))
    monkeypatch.setattr(torch, "cuda", fake, raising=False)
    monkeypatch.setattr(torch, "__version__", "2.12.0+cu130", raising=False)
    monkeypatch.setattr(torch, "version", types.SimpleNamespace(cuda="13.0"), raising=False)
    assert msa2._pin_word("trans2x") is None and msa2._pin_word("pwa2x") == "pin:cc80", "8.0: the transition replica is inside its pin, the PWA cell outside (by name)"
    fake.get_device_capability = lambda i=0: (9, 0)
    assert msa2._pin_word("trans2x") is None and msa2._pin_word("pwa2x") is None, "9.0: both inside"
    fake.get_device_capability = lambda i=0: (10, 0)
    assert msa2._pin_word("trans2x") == "pin:cc100" == msa2._pin_word("pwa2x"), "10.0: both outside, by name"
    fake.get_device_capability = lambda i=0: (8, 0)
    monkeypatch.setattr(torch, "__version__", "2.13.0", raising=False)
    assert msa2._pin_word("trans2x") == "pin:torch2.13.0", "the stack pin holds on every card"
    fake.is_available = lambda: False
    assert msa2._pin_word("trans2x") == "pin:no_cuda"


def test_ditexact_replicas_are_proven_on_sm_80_and_sm_90():
    pytest.importorskip("torch")
    src = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "ditexact", "src"))
    if "dx_token" in sys.modules:
        dx = sys.modules["dx_token"]
    else:
        spec = importlib.util.spec_from_file_location("dx_token", os.path.join(src, "dx_token.py"))
        dx = importlib.util.module_from_spec(spec)
        try:
            spec.loader.exec_module(dx)
        except Exception as e:  # noqa: BLE001 — a stack without the module's imports: a named skip, not a pass
            pytest.skip(f"dx_token not importable here: {type(e).__name__}: {e}")
    assert set(dx.PROVEN_CC) == {"smx", "glue"} and dx.PROVEN_CC["smx"] == {(9, 0), (8, 0)} == dx.PROVEN_CC["glue"], "sm_100 stays refused at install (cc_refused), by name"
