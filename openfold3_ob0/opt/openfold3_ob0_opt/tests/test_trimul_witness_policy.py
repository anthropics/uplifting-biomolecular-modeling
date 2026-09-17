"""of3_trimul.Router consults the core's witness policy (opt_core.kernels.trimul.witness_policy) before a row's
first-call WITNESS: under the tier word `big` above the core's WITNESS_BIG_MAX_N tokens no reference statement is issued (an N^2-class
allocation at the memory line's ceiling would raise OutOfMemoryError there) and the census carries the
core's skip token; under `fast`, and under `big` at small N, the witness runs."""
import pytest

torch = pytest.importorskip("torch", reason="needs torch")

from opt_core.kernels import trimul as KT
from openfold3_ob0_opt import of3_trimul as M


class _Sel:
    cell = "cell"


class _Prov:
    def fn(self, call):
        return call.z + 1.0


def _plan():
    return M.Plan("row", row="native", prov=_Prov(), sel=_Sel(), key="9.0|bf16|C128|H128|N<=4096|in|fwd")


def _serve(word, n, monkeypatch, calls):
    r = M.Router(word, weights_of=lambda mod: {})
    real = KT.witness_policy
    def spy(w, n_tokens):
        calls.append((w, n_tokens)); return real(w, n_tokens)
    monkeypatch.setattr(KT, "witness_policy", spy)
    issued = []
    monkeypatch.setattr(r, "_witness", lambda out, call, ref_fn: (issued.append(1), ("maxabs:0/relrms:0(vs:ref)", True))[1])
    z4 = torch.zeros(1, n, n, 4)
    out = r.serve_row(_plan(), mod=None, z4=z4, m3=None, direction="incoming", ref_fn=lambda: z4 + 1.0)
    return r, out, issued


def test_big_above_the_cores_bound_issues_no_witness_statement(monkeypatch):
    calls = []
    n = KT.WITNESS_BIG_MAX_N + 1
    r, out, issued = _serve("big", n, monkeypatch, calls)
    assert out is not None and calls == [("big", n)]                          # the binder consulted the policy with its word and this call's N
    assert issued == []                                                         # ... and issued NO reference statement
    assert list(r.witness.values()) == [KT.WITNESS_SKIP_TOKEN]                  # the census carries the core's token instead
    assert KT.WITNESS_SKIP_TOKEN in r.fields()


@pytest.mark.parametrize("word,n", [("big", 64), ("fast", KT.WITNESS_BIG_MAX_N + 1), ("fast", 64)])
def test_otherwise_the_witness_runs_as_before(monkeypatch, word, n):
    calls = []
    r, out, issued = _serve(word, n, monkeypatch, calls)
    assert out is not None and calls == [(word, n)] and issued == [1]
    assert list(r.witness.values()) == ["maxabs:0/relrms:0(vs:ref)"]


def test_the_policy_itself_at_the_pinned_core():
    assert KT.witness_policy("big", KT.WITNESS_BIG_MAX_N + 1) == ("skip", KT.WITNESS_SKIP_TOKEN)
    assert KT.witness_policy("big", KT.WITNESS_BIG_MAX_N) == ("run", "")
    assert KT.witness_policy("fast", 10 ** 6) == ("run", "")
