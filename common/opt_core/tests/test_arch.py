"""opt_core.arch: the card-class registry — normalisation, declarations, verdict words, the activation-line words, the readers with
mocked capability (no GPU, no torch and no jax needed)."""
from __future__ import annotations

import types

import pytest

from opt_core import arch, modes, report


@pytest.fixture(autouse=True)
def _clean_registry():
    before = dict(arch.declared())
    yield
    for k in list(arch.declared()):
        if k not in before:
            arch._forget(k)


def test_sm_of_reads_every_form():
    P = types.SimpleNamespace
    assert arch.sm_of("9.0") == "sm90" and arch.sm_of((8, 0)) == "sm80" and arch.sm_of([10, 3]) == "sm103"
    assert arch.sm_of("sm90") == "sm90" and arch.sm_of("SM_90a") == "sm90" and arch.sm_of("sm100") == "sm100"
    assert arch.sm_of({"sm": "sm90", "cc": "9.0"}) == "sm90" and arch.sm_of({"sm": None, "cc": "10.0"}) == "sm100"
    assert arch.sm_of({"sm": None, "cc": None, "probe": "nvidia-smi rc=9"}) is None
    assert arch.sm_of(P(major=10, minor=3)) == "sm103" and arch.sm_of(P(compute_capability="8.0")) == "sm80"
    assert arch.sm_of(None) is None and arch.sm_of("H100") is None and arch.sm_of("sm") is None and arch.sm_of("12") is None
    assert arch.sm_key("sm90") == (9, 0) and arch.sm_key("sm103") == (10, 3) and arch.sm_key("8.6") == (8, 6)
    assert sorted(["sm103", "sm90", "sm100", "sm80"], key=arch.sm_key) == ["sm80", "sm90", "sm100", "sm103"]
    with pytest.raises(ValueError):
        arch.sm_key("A100")


def test_card_labels_are_labels_only():
    assert arch.CARD_SM == {"A100": "sm80", "H100": "sm90", "H200": "sm90", "B200": "sm100", "B300": "sm103"}
    assert set(arch.SM_CLASSES) == {"sm80", "sm90", "sm100", "sm103"}


def test_declare_and_verdict_words():
    sup = arch.declare("LOCAL.acme.fused_pair", certified=("9.0",), min_sm="sm80", exclude={"sm100": "tmem_528_gt_512", (10, 3): "tmem_528_gt_512"},
                       note="a citation, never parsed")
    assert sup.certified == ("sm90",) and sup.min_sm == "sm80" and sup.excluded() == {"sm100": "tmem_528_gt_512", "sm103": "tmem_528_gt_512"}
    assert arch.declared("LOCAL.acme.fused_pair") == sup
    v = arch.supports("LOCAL.acme.fused_pair", "sm90")
    assert (v.ok, v.word, v.sm, v.certified) == (True, "supported", "sm90", True)
    assert arch.supports("LOCAL.acme.fused_pair", "8.0").word == "uncertified" and arch.supports("LOCAL.acme.fused_pair", "8.0").ok
    assert arch.supports("LOCAL.acme.fused_pair", "sm100").word == "unsupported:tmem_528_gt_512"
    assert not arch.supports("LOCAL.acme.fused_pair", {"cc": "10.3"}).ok
    assert arch.supports("LOCAL.acme.fused_pair", "7.5").word == "unsupported:below_sm80"
    assert arch.supports("LOCAL.acme.fused_pair", None).word == "no_gpu"
    assert arch.supports("LOCAL.acme.nobody_declared", "sm90").word == "undeclared" and arch.supports("LOCAL.acme.nobody_declared", "sm90").ok


def test_one_producer_per_lever():
    arch.declare("LOCAL.acme.fp8_gemm", min_sm="sm90", certified=("sm90",))
    arch.declare("LOCAL.acme.fp8_gemm", min_sm="sm90", certified=("sm90",))            # same content: a no-op
    with pytest.raises(ValueError, match="one producer per lever"):
        arch.declare("LOCAL.acme.fp8_gemm", min_sm="sm80")
    arch.declare("LOCAL.acme.fp8_gemm", min_sm="sm90", certified=("sm90", "sm100"), replace=True)
    assert arch.declared("LOCAL.acme.fp8_gemm").certified == ("sm90", "sm100")
    with pytest.raises(ValueError, match="below min_sm"):
        arch.declare("LOCAL.acme.bad1", min_sm="sm90", certified=("sm80",))
    with pytest.raises(ValueError, match="both certified and excluded"):
        arch.declare("LOCAL.acme.bad2", certified=("sm90",), exclude={"sm90": "x"})
    with pytest.raises(ValueError, match="blank-free"):
        arch.declare("LOCAL.acme.bad3", exclude={"sm100": "two words"})
    with pytest.raises(ValueError, match="not an sm class"):
        arch.declare("LOCAL.acme.bad4", certified=("H100",))


def test_lever_state_words_render_through_the_lever_grammar():
    arch.declare("LOCAL.acme.fused_pair", certified=("sm90",), exclude={"sm100": "tmem_528_gt_512"})
    on = arch.lever_state("LOCAL.acme.fused_pair", "sm90")
    assert (on["state"], on["reason"], on["evidence"]) == ("on", None, {"sm": "sm90"})
    assert on["words"] == []                                                            # certified: nothing to name
    unc = arch.lever_state("LOCAL.acme.fused_pair", "sm80")
    assert (unc["state"], unc["evidence"]) == ("on", {"sm": "sm80", "card_support": "uncertified:sm80"})
    assert unc["words"] == ["card_support=uncertified:sm80"]                            # the named uncertainty the lever engages under
    und = arch.lever_state("LOCAL.acme.other", "sm80")
    assert (und["state"], und["evidence"]["card_support"], und["words"]) == ("on", "undeclared", ["card_support=undeclared"])
    strict = arch.lever_state("LOCAL.acme.fused_pair", "sm80", strict=True)              # certification is a word, never a state: strict changes nothing
    assert {k: strict[k] for k in ("state", "reason", "evidence", "words")} == {k: unc[k] for k in ("state", "reason", "evidence", "words")}
    assert not hasattr(arch, "REASON_UNCERTIFIED")
    off = arch.lever_state("LOCAL.acme.fused_pair", "sm100")
    assert (off["state"], off["reason"], off["evidence"]) == ("off", "unsupported_card:sm100", {"card_reason": "tmem_528_gt_512"})
    assert off["words"] == []                                                           # cannot run: a refusal reason, not a word
    nog = arch.lever_state("LOCAL.acme.fused_pair", None)
    assert (nog["state"], nog["reason"], nog["evidence"]) == ("off", "unsupported_card:none", {"card_reason": "no_gpu"})
    line = arch.card_line("acme-opt", "LOCAL.acme.fused_pair", "10.0", impl="fused_pair@1.2", origin="kit")
    assert line == "[acme-opt] LEVER name=LOCAL.acme.fused_pair state=off reason=unsupported_card:sm100 impl=fused_pair@1.2 origin=kit card_reason=tmem_528_gt_512"
    line_on = arch.card_line("acme-opt", "LOCAL.acme.fused_pair", "sm90", impl="fused_pair@1.2", origin="kit", served=3)
    assert line_on == "[acme-opt] LEVER name=LOCAL.acme.fused_pair state=on impl=fused_pair@1.2 origin=kit sm=sm90 served=3"
    # a shared strategy declared under its canonical id serves a kit line of another name
    arch.declare("F7.pair_offload", certified=("sm90",), exclude={"sm80": "needs_tma"})
    shared = arch.card_line("acme-opt", "pair_offload_v2", "sm80", impl="mem.torch_hostpair", origin="core", strategy="F7.pair_offload")
    assert "state=off reason=unsupported_card:sm80" in shared and "strategy=F7.pair_offload" in shared and shared.endswith("card_reason=needs_tma")


def test_require_refuses_by_name_with_a_mode_error():
    arch.declare("LOCAL.acme.fused_pair", certified=("sm90",), exclude={"sm100": "tmem_528_gt_512"})
    assert arch.require("LOCAL.acme.fused_pair", "sm90", mode="fast").word == "supported"
    assert arch.require("LOCAL.acme.fused_pair", "sm80", mode="fast").word == "uncertified"
    with pytest.raises(arch.ArchRefused, match=r"mode 'big' needs lever 'LOCAL.acme.fused_pair', which cannot run on this card \(sm100: unsupported:tmem_528_gt_512\)") as ex:
        arch.require("LOCAL.acme.fused_pair", "sm100", mode="big")
    assert isinstance(ex.value, modes.ModeError)
    from opt_core import gates
    assert gates.is_cannot_run(ex.value) and arch.ArchRefused.cannot_run is True           # a mode refusal by name: the cannot-run event class
    assert arch.require("LOCAL.acme.fused_pair", "sm80", mode="exact", strict=True).word == "uncertified"   # uncertified runs under strict too (named on the line)
    with pytest.raises(arch.ArchRefused, match="no GPU class read"):
        arch.require("LOCAL.acme.fused_pair", None, mode="fast")


def test_card_table_is_the_applicability_projection():
    arch.declare("LOCAL.acme.fused_pair", certified=("sm90",), exclude={"sm100": "tmem_528_gt_512"})
    arch.declare("LOCAL.acme.fp8_gemm", min_sm="sm90", certified=("sm90",))
    t = arch.card_table(["LOCAL.acme.fused_pair", "LOCAL.acme.fp8_gemm", "LOCAL.acme.other"])
    assert list(t["LOCAL.acme.fused_pair"]) == ["sm80", "sm90", "sm100", "sm103"]
    assert t["LOCAL.acme.fused_pair"] == {"sm80": "uncertified", "sm90": "supported", "sm100": "unsupported:tmem_528_gt_512", "sm103": "uncertified"}
    assert t["LOCAL.acme.fp8_gemm"] == {"sm80": "unsupported:below_sm90", "sm90": "supported", "sm100": "uncertified", "sm103": "uncertified"}
    assert set(t["LOCAL.acme.other"].values()) == {"undeclared"}
    assert arch.card_table(["LOCAL.acme.fp8_gemm"], sms=["sm90"]) == {"LOCAL.acme.fp8_gemm": {"sm90": "supported"}}


def test_current_sm_uses_a_given_probe_and_never_imports_torch(monkeypatch):
    import sys
    sm, probe = arch.current_sm(probe={"name": "SOME CARD", "cc": "10.0", "sm": "sm100", "memory_mib": 183359, "probe": "nvidia-smi"})
    assert sm == "sm100" and probe["memory_mib"] == 183359
    if "torch" in sys.modules:                                        # order-dependent otherwise: a torch imported earlier in the session cannot be un-imported cleanly
        pytest.skip("torch already imported in this interpreter: the reader's no-import property is held on a fresh interpreter (run this file alone)")
    had_torch = sys.modules.get("torch")
    monkeypatch.delitem(sys.modules, "torch", raising=False)          # as if this process never imported torch (restored at teardown)
    from opt_core import gates
    monkeypatch.setattr(gates, "nvidia_smi_probe", lambda timeout_s=20, index=0, keys=None: {"name": None, "cc": None, "sm": None, "memory_mib": None, "probe": "nvidia-smi unavailable (FileNotFoundError)"})
    sm2, probe2 = arch.current_sm()
    assert sm2 is None and probe2["probe"].startswith("nvidia-smi unavailable")
    mem = arch.device_memory()
    assert mem == {"total_bytes": None, "free_bytes": None, "source": "nvidia-smi unavailable (FileNotFoundError)"}
    monkeypatch.setattr(gates, "nvidia_smi_probe", lambda timeout_s=20, index=0, keys=None: {"name": "CARD", "cc": "9.0", "sm": "sm90", "memory_mib": 81559, "probe": "nvidia-smi"})
    assert arch.current_sm()[0] == "sm90"
    assert arch.device_memory() == {"total_bytes": 81559 * 1024 * 1024, "free_bytes": None, "source": "nvidia-smi"}
    assert "torch" not in sys.modules                                                   # the reader imported nothing
    if had_torch is not None:
        monkeypatch.setitem(sys.modules, "torch", had_torch)


def test_torch_reader_when_torch_is_already_imported(monkeypatch):
    fake_cuda = types.SimpleNamespace(is_available=lambda: True, mem_get_info=lambda index=0: (7 * 2 ** 30, 40 * 2 ** 30),
                                      get_device_properties=lambda index=0: types.SimpleNamespace(major=8, minor=0, total_memory=40 * 2 ** 30),
                                      get_device_name=lambda index=0: "FAKE CARD")
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setitem(arch.sys.modules, "torch", fake_torch)
    assert arch.device_memory() == {"total_bytes": 40 * 2 ** 30, "free_bytes": 7 * 2 ** 30, "source": "torch"}
    sm, probe = arch.current_sm()
    assert sm == "sm80" and probe["probe"] == "torch" and probe["memory_mib"] == 40 * 1024


def test_jax_readers_with_mock_devices():
    from opt_core import jax_arch
    D = types.SimpleNamespace
    gpu = D(platform="gpu", device_kind="SOME CARD", compute_capability="9.0", memory_stats=lambda: {"bytes_limit": 100, "bytes_in_use": 30})
    assert jax_arch.jax_sm(gpu) == ("sm90", {"platform": "gpu", "device_kind": "SOME CARD", "probe": "jax"})
    assert jax_arch.jax_sm(devices=[gpu])[0] == "sm90"
    assert jax_arch.jax_sm(devices=[])[0] is None
    cpu = D(platform="cpu", device_kind="cpu")
    sm, info = jax_arch.jax_sm(cpu)
    assert sm is None and "platform cpu" in info["probe"]
    assert jax_arch.jax_device_memory(gpu) == {"total_bytes": 100, "free_bytes": 70, "source": "jax"}
    assert jax_arch.jax_device_memory(cpu)["source"].startswith("jax: memory_stats unavailable")
    nostats = D(platform="gpu", compute_capability="10.0", memory_stats=lambda: None)
    assert jax_arch.jax_device_memory(nostats)["source"] == "jax: memory_stats without bytes_limit" and jax_arch.jax_sm(nostats)[0] == "sm100"


def test_reason_words_are_valid_lever_line_tokens():
    # every word lever_state can produce must pass report.lever_line's blank rule
    arch.declare("LOCAL.acme.x", certified=("sm90",), min_sm="sm80", exclude={"sm100": "why_not"})
    for sm in (None, "sm75", "sm80", "sm90", "sm100"):
        for strict in (False, True):
            st = arch.lever_state("LOCAL.acme.x", sm, strict=strict)
            report.lever_line("t", "LOCAL.acme.x", st["state"], reason=st["reason"], impl="m", origin="kit", **st["evidence"])
