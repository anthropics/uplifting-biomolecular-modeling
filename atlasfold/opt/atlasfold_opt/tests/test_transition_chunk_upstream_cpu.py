"""conf_transition_chunk / pair_transition_chunk census: a pair-shaped call that a transition KERNEL lever (transition_exact, fast's pair_transition —
installed outside the row-block wrapper) served first is counted on the chunk ledger that would have blocked it, BY NAME (upstream_<lever>),
inside the confidence heads on the conf ledger, elsewhere on the pair ledger; non-pair shapes and unknown words are ignored; never raises."""
import types
import pytest


class _Ledger:
    def __init__(self): self.fb = {}
    def fallback(self, w): self.fb[w] = self.fb.get(w, 0) + 1


def test_note_upstream_counts_on_the_right_ledger(monkeypatch):
    torch = pytest.importorskip("torch")
    from atlasfold_opt.hooks import transition_chunk as TC
    conf, pair = {"ledger": _Ledger()}, {"ledger": _Ledger()}
    monkeypatch.setitem(TC._STATE, "conf", conf); monkeypatch.setitem(TC._STATE, "pair", pair)
    z = torch.zeros(1, 8, 8, 4); s = torch.zeros(1, 8, 4)
    TC.note_upstream(z, "upstream_transition_exact")                      # outside the heads -> pair ledger
    TC.note_upstream(s, "upstream_transition_exact")                      # rank-3 single transition: ignored
    TC.note_upstream(z, "bogus")                                          # unknown word: ignored
    TC._IN_HEAD.depth = getattr(TC._IN_HEAD, "depth", 0) + 1
    try:
        TC.note_upstream(z, "upstream_pair_transition")                   # inside a confidence head -> conf ledger
    finally:
        TC._IN_HEAD.depth -= 1
    assert pair["ledger"].fb == {"upstream_transition_exact": 1} and conf["ledger"].fb == {"upstream_pair_transition": 1}
    monkeypatch.setitem(TC._STATE, "pair", None)
    TC.note_upstream(z, "upstream_transition_exact")                      # no pair lever in the row: nothing to count, no error
    TC.note_upstream(types.SimpleNamespace(), "upstream_transition_exact")   # not a tensor: never raises


def test_expected_words_and_call_sites():
    import inspect
    from atlasfold_opt.hooks import transition_chunk as TC, transition_exact as TX, pair_transition_fused as PT
    assert set(TC.UPSTREAM_WORDS) == {"upstream_transition_exact", "upstream_pair_transition"}
    assert 'note_upstream(x, "upstream_transition_exact")' in inspect.getsource(TX)
    assert 'note_upstream(x, "upstream_pair_transition")' in inspect.getsource(PT)
    src = inspect.getsource(TC)
    assert '"not_pair") + UPSTREAM_WORDS' in src and '"disabled") + UPSTREAM_WORDS' in src        # both ledgers list the words as expected fallbacks (the gate stays ok on an all-upstream run)
