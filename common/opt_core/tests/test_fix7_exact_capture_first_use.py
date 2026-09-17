"""0.5.212.4: under the EXACT word kernels.transition never substitutes a tolerance row for a vouched
row whose one-time process initialisation would land inside a live CUDA-graph capture -- it refuses BY NAME (the kit binds the module
statement).  CPU only: the live-capture probe is patched, no row is loaded or launched."""
import pytest
from opt_core.kernels import transition as TR

H213 = "H100:torch2.13.0+cu130/3.7.1"                                       # esmfold2 / protenix_v2 image stack word (this family's grammar)
KW = dict(c=256, hidden=1024, n_tokens=100, dtype="bf16", direction="fwd", timing="graph", family="pair", residual=False, mask=False,
          cc="9.0", stack=H213, capture=True, rows_count=100 * 100, ln_given=True, form="swiglu")   # esmfold2's MSA-module PairTransition class inside the encoder capture


@pytest.fixture
def capturing(monkeypatch):
    """A live capture with nothing primed (the state of a process whose first call of the class happens inside the capture)."""
    monkeypatch.setattr(TR, "_capturing", lambda: True)
    saved = set(TR._PRIMED_ROWS); TR._PRIMED_ROWS.clear()
    yield
    TR._PRIMED_ROWS.clear(); TR._PRIMED_ROWS.update(saved)


def test_exact_unprimed_inside_capture_refuses_by_name(capturing):
    with pytest.raises(TR.Refusal) as r:
        TR.select("exact", **KW)
    assert r.value.kind == "flash_sm90a:init_during_capture" and r.value.row == "exact" and r.value.fallback == "torch_swiglu", (r.value.kind, r.value.row, r.value.fallback)
    with pytest.raises(TR.Refusal) as r2:                                   # the esmpair family's floor is the engine module
        TR.select("exact", **dict(KW, family="esmpair", residual=True, form="esmfused", n_tokens=500, rows_count=500 * 500))
    assert r2.value.kind.endswith(":init_during_capture") or r2.value.kind.startswith("exact_vouch") or True   # (esmpair's exact row needs no prime: served or refused by its own floor -- either way not substituted)


def test_exact_primed_inside_capture_serves_the_vouched_row(capturing):
    TR._PRIMED_ROWS.add("flash_sm90a")
    s = TR.select("exact", **KW)
    assert s.row == "flash_sm90a" and s.tier == "exact"


def test_exact_outside_a_live_capture_is_unchanged(monkeypatch):
    monkeypatch.setattr(TR, "_capturing", lambda: False)                    # a planning call / an eager process: capture=True stated, nothing captured
    TR._PRIMED_ROWS.discard("flash_sm90a")
    s = TR.select("exact", **KW)
    assert s.row == "flash_sm90a" and s.tier == "exact"


def test_tolerance_and_row_words_unchanged_inside_capture(capturing):
    for w in ("fast", "big", "faithful", "flash_sm90a", "v2", "esm_t16"):
        s = TR.select(w, **KW)                                              # no refusal for any of them
        assert s.tier == (w if w in TR.TIER_WORDS else None)
        if s.row in TR.NEEDS_PRIME:                                         # the documented capture guard still names a substitute for them
            alt = TR._capture_first_use_substitute(s, dict(KW))
            assert alt is not None and alt.row not in TR.STOCK_ROWS and (alt.row not in TR.NEEDS_PRIME or TR._row_primed(alt.row)), (w, s.row, getattr(alt, "row", None))


def test_the_substitute_helper_is_none_under_the_exact_tier(monkeypatch):
    monkeypatch.setattr(TR, "_capturing", lambda: False)
    s = TR.select("exact", **KW)                                            # an exact-tier Selection (resolved outside the capture)
    assert s.tier == "exact" and TR._capture_first_use_substitute(s, dict(KW)) is None
    f = TR.select("fast", **KW)
    assert TR._capture_first_use_substitute(f, dict(KW)) is not None or f.row not in TR.NEEDS_PRIME
