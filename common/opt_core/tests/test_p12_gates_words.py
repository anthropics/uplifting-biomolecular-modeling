"""P12 lever engagement in the core's gates, records and caches: the three outcomes (words · cannot-run · accounting) on the certified
inputs (nothing changes: no words, the same verdicts) and on the uncertain ones (a word, never a refusal; a cannot-run raise, never a
quiet subset). CPU only; torch-dependent cases skip without torch."""
import os
import types

import pytest

from opt_core import counters, gates, jit_cache, report
from opt_core.capture import graphs, pool, xla_cache
from opt_core.jax_design import subbatch_policy as sp
from opt_core.precision import policy


# ------------------------------------------------------------------------------------------------------------ the convention itself

def test_cannot_run_marker_and_reader():
    e = gates.cannot_run(RuntimeError("fpf: none:CompilationError"))
    assert gates.is_cannot_run(e) and str(e) == "fpf: none:CompilationError"
    assert not gates.is_cannot_run(RuntimeError("plain")) and not gates.is_cannot_run(ValueError())
    for cls in (graphs.CaptureUnavailable, graphs.CaptureFailed, xla_cache.XlaCacheUnavailable, jit_cache.StackKeyUnknown):
        assert cls.cannot_run is True and gates.is_cannot_run(cls("x")), cls
    assert gates.is_cannot_run(graphs.CaptureRefused("rng", "drew under forbid"))
    from opt_core import arch
    assert gates.is_cannot_run(arch.ArchRefused("mode 'fast' needs lever 'x'"))


def test_graph_cache_default_is_strict_raise():
    """The certified lines' form is unchanged: a GraphCache raises its cannot-run events (strict=True by default); strict=False stays the
    named development opt-in."""
    assert graphs.GraphCache("site").strict is True and graphs.GraphCache("dev", strict=False).strict is False


# --------------------------------------------------------------------------------------------------------------- Ledger words

def test_ledger_words_ride_on_the_line_the_fields_and_the_gate_and_never_refuse():
    L = counters.Ledger("F2.fpf_trimul", impl="fpf_trimul_v4@4", origin="core", expected=("below_min_tokens",))
    L.serve("384x128")
    clean = L.gate()
    assert clean.ok and clean.words == () and L.words() == [] and L.fields()["words"] == []          # certified: nothing named
    line0 = L.line("acme-opt")
    assert L.word("settings", "safe:no_cell:384x128") == "settings=safe:no_cell:384x128"
    L.word("card_support", "uncertified:sm89")
    assert L.words() == ["settings=safe:no_cell:384x128", "card_support=uncertified:sm89"]
    g = L.gate()
    assert g.ok and g.words == ("settings=safe:no_cell:384x128", "card_support=uncertified:sm89") and g.details["words"] == list(g.words)
    line = L.line("acme-opt")
    assert line.startswith(line0) and " card_support=uncertified:sm89" in line and " settings=safe:no_cell:384x128" in line   # facts print sorted after the pairs
    L.fallback("kernel_unavailable:ImportError")                                                    # an undeclared reason: accounting, refused — the words still ride
    bad = L.gate()
    assert not bad.ok and bad.reason == "unexpected fallback: fallback_by=kernel_unavailable:ImportError:1" and bad.words == g.words
    L.clear()
    assert L.words() == [] and L.fields()["words"] == []


# ------------------------------------------------------------------------------------------------------------ capture policies (M)

def test_measured_off_capture_decisions_carry_the_not_measured_name():
    t = pool.Table({"k1": 2})
    assert t.decide("k1", 2) == ("capture", "table: sighting 2 >= K=2", False)                     # a pinned row: unchanged
    action, reason, final = t.decide("k9", 1)
    assert (action, final) == ("eager", True) and reason == "table: no capture rule for key 'k9' (off: not measured on this key)"
    p = pool.Planned({"a": 5}, pricing=pool.FixedK(2))
    assert p.decide("a", 1)[0] == "capture"
    action, reason, final = p.decide("zz", 1)
    assert (action, final) == ("eager", True) and reason == "planned: key 'zz' not in the job's plan (off: not measured on this key)"
    assert pool.Planned({"a": 5}, unplanned="pricing").decide("zz", 2)[0] == "capture"             # the pricing opt-in is untouched


# ------------------------------------------------------------------------------------------------------------------ xla_cache tier

class _FakeSer:
    """jax.experimental.serialize_executable stand-in: serialize may fail; deserialize_and_load returns a marked object."""

    def __init__(self, fail_serialize=False, fail_load=False):
        self.fail_serialize, self.fail_load = fail_serialize, fail_load

    def serialize(self, compiled):
        if self.fail_serialize:
            raise RuntimeError("serialize broke")
        return b"EXE:" + compiled.encode(), "in", "out"

    def deserialize_and_load(self, serialized, in_tree, out_tree):
        if self.fail_load:
            raise RuntimeError("load broke")
        return ("loaded", serialized)


class _Lowered:
    def __init__(self, tag):
        self.tag = tag

    def compile(self):
        return f"compiled-{self.tag}"


def _cache(tmp_path, monkeypatch, ser, **kw):
    c = xla_cache.ExecutableCache("af2_fwd", str(tmp_path / "xla_exec"), {"model": 3}, verbose=False, log=lambda s: None, **kw)
    monkeypatch.setattr(c, "_ser", lambda: ser)
    c._stack, c._hash = {"jax": "0.5.3", "jaxlib": "0.5.3"}, "cfg0123"
    return c


def test_xla_cache_certified_path_is_unchanged_in_both_tiers(tmp_path, monkeypatch):
    for exact in (True, False):
        c = _cache(tmp_path / str(exact), monkeypatch, _FakeSer(), exact=exact)
        exe, event = c.load_or_compile("b512", lambda: _Lowered("b512"))
        assert event == "compiled+stored" and exe == ("loaded", b"EXE:compiled-b512")
        exe2, event2 = c.load_or_compile("b512", lambda: _Lowered("b512"))
        assert event2 == "memory"
        c._mem.clear()
        assert c.load_or_compile("b512", lambda: _Lowered("nope"))[1] == "loaded"                  # the next process: from the artefact
        assert c.words() == [] and c.misses() == 0 and " cache=miss" not in c.evidence_line("kit")
        assert not hasattr(c, "partial")                                                          # nothing in this record is 'partial'


def test_xla_cache_miss_is_a_word_under_fast_and_a_cannot_run_raise_under_exact(tmp_path, monkeypatch):
    fast = _cache(tmp_path / "fast", monkeypatch, _FakeSer(fail_serialize=True), exact=False)
    exe, event = fast.load_or_compile("b256", lambda: _Lowered("b256"))
    assert exe == "compiled-b256" and event.startswith("compiled(unstored: ") and fast.stats_["unstored"] == 1
    assert fast.words() == ["cache=miss(1)"] and fast.evidence_line("kit").endswith(" cache=miss(1)")
    assert report.words_of(fast, ["cache=miss(1)"]) == ["cache=miss(1)"]
    exact = _cache(tmp_path / "exact", monkeypatch, _FakeSer(fail_serialize=True))                  # default tier: exact
    assert exact.exact is True
    with pytest.raises(xla_cache.XlaCacheUnavailable, match=r"af2_fwd: compiled\(unstored:b256\): serialization failed .* under exact") as ex:
        exact.load_or_compile("b256", lambda: _Lowered("b256"))
    assert gates.is_cannot_run(ex.value) and exact.stats_["unstored"] == 1
    # a stored artefact that fails to load: the same rule
    good = _cache(tmp_path / "reload", monkeypatch, _FakeSer())
    good.load_or_compile("b64", lambda: _Lowered("b64"))
    broken = _cache(tmp_path / "reload", monkeypatch, _FakeSer(fail_load=True), exact=False)
    assert broken.load_or_compile("b64", lambda: _Lowered("b64"))[0] == "compiled-b64" and broken.words() == ["cache=miss(2)"]   # load failed, then the reload after store failed
    strict = _cache(tmp_path / "reload", monkeypatch, _FakeSer(fail_load=True))
    with pytest.raises(xla_cache.XlaCacheUnavailable, match="reloaded-failed"):
        strict.load_or_compile("b64", lambda: _Lowered("b64"))


# ---------------------------------------------------------------------------------------------------------------- jit_cache run path

def test_jit_cache_key_facts_isolates_an_unknown_part_and_names_it(monkeypatch):
    monkeypatch.setattr(gates, "nvidia_smi_probe", lambda **kw: {"name": None, "cc": None, "probe": "nvidia-smi unavailable"})
    full = jit_cache.key_facts("2.7.0+cu126", cc="9.0")
    assert full == {"key": "torch2.7.0-cu126-sm90", "parts": {"version": "2.7.0", "cuda": "126", "cc": "9.0"}, "unknown": [], "word": None}
    assert full["key"] == jit_cache.key("2.7.0+cu126", cc="9.0")                                  # the certified box: byte-identical to key()
    tok = jit_cache.isolation_token()
    f = jit_cache.key_facts("2.7.0+cu126")                                                        # no GPU visible: cc unknown
    assert f["key"] == f"torch2.7.0-cu126-smunknown{tok}" and f["unknown"] == ["cc"] and f["word"] == "cache_key=unknown(cc)"
    assert f["key"] != jit_cache.key("2.7.0+cu126", strict=False) == "torch2.7.0-cu126-smunknown"  # the display form stays a plain word
    with pytest.raises(jit_cache.StackKeyUnknown) as ex:                                          # the cache verbs' form still raises by name
        jit_cache.key("2.7.0+cu126")
    assert gates.is_cannot_run(ex.value)


# --------------------------------------------------------------------------------------------------------------------- subbatch

def test_subbatch_no_device_decision_is_a_word():
    peak = sp.QuadraticPeak(a=1e-4, b=0.0, c=1.0, unit=1e9)
    fits = sp.choose(tokens=100, requested="auto", device_bytes=80 * 10**9, peak_estimator=peak, stock_value=4)
    assert fits.source == "auto:fits" and fits.words() == []
    nodev = sp.choose(tokens=100, requested="auto", device_bytes=None, peak_estimator=peak, stock_value=4)
    assert (nodev.value, nodev.source) == (4, "auto:no_device") and nodev.words() == ["subbatch=no_device(stock)"]
    unch = sp.choose(tokens=100, requested="auto", device_bytes=None, peak_estimator=peak, stock_value=4, when_device_unknown="unchunked")
    assert unch.value is None and unch.words() == ["subbatch=no_device(unchunked)"] and unch.words("attn_chunk") == ["attn_chunk=no_device(unchunked)"]


# -------------------------------------------------------------------------------------------------------------- precision policy tier

def test_tf32_override_gate_refuses_under_exact_and_names_under_fast():
    assert policy.tf32_override_gate({}).ok and policy.tf32_override_gate({}, exact=False).words == ()
    ex = policy.tf32_override_gate({"NVIDIA_TF32_OVERRIDE": "0"})
    assert not ex.ok and ex.reason == policy.tf32_override_refusal(["NVIDIA_TF32_OVERRIDE"]) and ex.words == ()
    fast = policy.tf32_override_gate({"NVIDIA_TF32_OVERRIDE": "0", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": "1"}, exact=False)
    assert fast.ok and fast.reason is None and fast.words == ("tf32_override=NVIDIA_TF32_OVERRIDE=0,TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1",)
    assert fast.details["hits"] == ["NVIDIA_TF32_OVERRIDE", "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"]


def test_expect_refuses_drift_under_exact_and_names_it_under_fast():
    torch = pytest.importorskip("torch")
    snap = policy.snapshot(torch)
    try:
        torch.set_float32_matmul_precision("highest")
        torch.backends.cudnn.allow_tf32 = False
        ok = policy.expect(policy.FP32_STRICT, torch)
        assert ok["changed"] == [] and ok["words"] == [] and ok["drift"] == {}                      # the certified arm: nothing named
        assert policy.expect(policy.FP32_STRICT, torch, exact=False)["words"] == []
        torch.set_float32_matmul_precision("high")
        with pytest.raises(policy.PolicyMismatch, match="matmul:declared=highest,live=high"):
            policy.expect(policy.FP32_STRICT, torch)
        fast = policy.expect(policy.FP32_STRICT, torch, exact=False)
        assert fast["words"] == ["numerics=drift(matmul:high!=highest)"] and fast["drift"] == {"matmul": {"declared": "highest", "live": "high"}}
    finally:
        policy.restore(snap, torch)
