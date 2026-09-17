"""The pair transition's own row block (``pairstack.bind(transition_rows=)``) and ``transition.transition_face_fn`` (the
shared core's transition provider by tier word, per row block) — CPU.

* ``bind(transition_rows=None)`` drives the transition over blocks of ``chunk`` rows exactly as before (same blocks, same result, no census word);
  ``transition_rows=k`` drives k-row blocks with the attention chunk untouched; census ``transition_rows``.
* ``transition_update_(residual_in_fn=True)`` (a callable returning ``x + delta``, in place or not) == the delta form, bitwise.
* ``transition_face_fn`` on CPU: the provider refuses by name (``x_not_cuda``) -> the bound ``engine_fn`` serves, bitwise today's result, sticky,
  census ``transition=engine:fallback(x_not_cuda)``; ``residual=True`` carries ``residual_in_fn`` and lands ``x + delta`` in place; without
  ``engine_fn`` the provider's stock statement row answers (tolerance-equal to the fp64 reference); ``max_rows_hidden`` arithmetic and sub-blocking.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import dist as D  # noqa: E402
from opt_core.mem.rowpair import evidence  # noqa: E402
from opt_core.mem.rowpair import pairstack as PS  # noqa: E402
from opt_core.mem.rowpair import transition as TR  # noqa: E402
from opt_core.kernels import transition as KT  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("ROWPAIR_TRANSITION_FACE", raising=False)
    monkeypatch.delenv("ROWPAIR_TRANSITION_ROWS", raising=False)
    evidence.reset_schedule()
    yield


def test_face_and_rows_words(monkeypatch):
    """ROWPAIR_TRANSITION_FACE unset / off words = the face is OFF: transition_face_fn hands back engine_fn ITSELF (today's binding; nothing packed or
    recorded); a word turns it on; ROWPAIR_TRANSITION_ROWS feeds bind(transition_rows=None); bad words refused by name."""
    engine = lambda x, m: x * 0                                                          # noqa: E731
    assert TR.transition_face_word() == "" and TR.transition_face_word("big") == "big" and TR.transition_face_word(" OFF ") == ""
    for w in ("", "0", "off", "engine"):
        monkeypatch.setenv("ROWPAIR_TRANSITION_FACE", w)
        assert TR.transition_face_fn(w_a=None, n_tokens=8, engine_fn=engine) is engine
    assert "transition" not in evidence.schedule()
    with pytest.raises(RowpairRefused, match="face is off"):
        TR.transition_face_fn(n_tokens=8)
    monkeypatch.setenv("ROWPAIR_TRANSITION_FACE", "big")
    w_a, w_b, w_o, ln_w, ln_b = _weights(64, factor=2)
    fn = TR.transition_face_fn(w_a=w_a, w_b=w_b, w_o=w_o, ln_w=ln_w, ln_b=ln_b, n_tokens=8, engine_fn=engine)
    assert fn is not engine and fn.word == "big" and evidence.schedule()["transition"] == "face:big"
    # rows word
    assert PS.transition_rows_word() is None and PS.transition_rows_word(7) == 7
    monkeypatch.setenv("ROWPAIR_TRANSITION_ROWS", "chunk")
    assert PS.transition_rows_word() is None
    monkeypatch.setenv("ROWPAIR_TRANSITION_ROWS", "256")
    assert PS.transition_rows_word() == 256
    spy = _Spy()
    z = torch.randn(9, 5, 4)
    _bind(spy, 2).transition(z, torch.ones(9, 5), D.Layout(5, 2, 0, 8))
    assert spy.rows == [9] and evidence.schedule()["transition_rows"] == 256               # the env word reached bind(transition_rows=None)
    spy2 = _Spy()
    _bind(spy2, 2, transition_rows=4).transition(z, torch.ones(9, 5), D.Layout(5, 2, 0, 8))
    assert spy2.rows == [4, 4, 1]                                                        # the argument wins
    for bad in ("-3", "lots"):
        monkeypatch.setenv("ROWPAIR_TRANSITION_ROWS", bad)
        with pytest.raises(RowpairRefused, match="ROWPAIR_TRANSITION_ROWS"):
            PS.transition_rows_word()


class _Spy(object):
    """an engine-statement stand-in: delta = tanh(x) * mask, recording the row counts it was called with"""

    def __init__(self):
        self.rows = []

    def __call__(self, x_rows, mask_u_rows):
        self.rows.append(int(x_rows.shape[0]))
        return torch.tanh(x_rows) * mask_u_rows


def _bind(transition, chunk, **kw):
    return PS.bind(trimul_out=None, trimul_in=None, triatt_start=None, triatt_end=None, transition=transition, chunk=chunk,
                   trimul_kw=dict(inplace_chunk=4), **kw)


def test_bind_transition_rows_default_is_chunk_and_decoupled_when_given():
    g = torch.Generator().manual_seed(0)
    R, N, C = 11, 9, 4
    z = torch.randn(R, N, C, generator=g)
    mask = (torch.rand(R, N, generator=g) > 0.2).float()
    lay = D.Layout(N * 1, 2, 0, 8)                                                       # only handed through (the transition is row-local)
    spy0 = _Spy()
    ref = PS.transition_update_(spy0, z.clone(), mask, 3)
    assert spy0.rows == [3, 3, 3, 2]
    # None = chunk, byte for byte, no census word
    spy1 = _Spy()
    out1 = _bind(spy1, 3).transition(z.clone(), mask, lay)
    assert torch.equal(out1, ref) and spy1.rows == [3, 3, 3, 2] and "transition_rows" not in evidence.schedule()
    # transition_rows=8 with chunk 3: 8-row blocks, same values (row-local statement), census word
    spy2 = _Spy()
    out2 = _bind(spy2, 3, transition_rows=8).transition(z.clone(), mask, lay)
    assert torch.equal(out2, ref) and spy2.rows == [8, 3] and evidence.schedule()["transition_rows"] == 8
    # transition_rows larger than R: one block
    spy3 = _Spy()
    _bind(spy3, 3, transition_rows=256).transition(z.clone(), mask, lay)
    assert spy3.rows == [11]


def test_transition_update_residual_in_fn_equals_delta_form():
    g = torch.Generator().manual_seed(1)
    R, N, C = 10, 6, 4
    z = torch.randn(R, N, C, generator=g)
    mask = (torch.rand(R, N, generator=g) > 0.3).float()
    delta_fn = lambda x, m: torch.sin(x) * m                                             # noqa: E731
    ref = PS.transition_update_(delta_fn, z.clone(), mask, 4)

    def inplace_fn(x, m):                                                                # x + delta written IN PLACE, returned
        x.add_(torch.sin(x) * m)
        return x

    def outofplace_fn(x, m):                                                             # x + delta as a NEW tensor (landed by the driver)
        return x + torch.sin(x) * m

    a = PS.transition_update_(inplace_fn, z.clone(), mask, 4, residual_in_fn=True)
    b = PS.transition_update_(outofplace_fn, z.clone(), mask, 4, residual_in_fn=True)
    assert torch.equal(a, ref) and torch.equal(b, ref)
    # bind reads the callable's residual_in_fn attribute
    inplace_fn.residual_in_fn = True
    lay = D.Layout(N, 2, 0, 8)
    c = _bind(inplace_fn, 4).transition(z.clone(), mask, lay)
    assert torch.equal(c, ref) and evidence.schedule()["transition_residual"] == "fn"
    d = _bind(inplace_fn, 4, transition_rows=7).transition(z.clone(), mask, lay)
    assert torch.equal(d, ref)


def _weights(c=128, factor=4, seed=2):
    g = torch.Generator().manual_seed(seed)
    hidden = factor * c
    w_a = torch.randn(hidden, c, generator=g) / c ** 0.5
    w_b = torch.randn(hidden, c, generator=g) / c ** 0.5
    w_o = torch.randn(c, hidden, generator=g) / hidden ** 0.5
    ln_w = 1.0 + 0.1 * torch.randn(c, generator=g)
    ln_b = 0.1 * torch.randn(c, generator=g)
    return w_a, w_b, w_o, ln_w, ln_b


def _engine_of(w_a, w_b, w_o, ln_w, ln_b, eps=1e-5):
    """today's engine statement (the AF3-family SwiGLU transition form): LN -> a|b -> silu(a)*b -> out, * mask -> delta"""
    import torch.nn.functional as F

    def engine(x_rows, mask_u_rows):
        n = F.layer_norm(x_rows, (x_rows.shape[-1],), ln_w, ln_b, eps)
        y = F.linear(F.silu(F.linear(n, w_a)) * F.linear(n, w_b), w_o)
        return y * mask_u_rows
    return engine


def test_max_rows_hidden():
    assert TR.max_rows_hidden(10120, 128) == (2 ** 31 - 1) // (10120 * 512) == 414
    assert TR.max_rows_hidden(70320, 128) == 59 and TR.max_rows_hidden(4, 4, 4) == (2 ** 31 - 1) // 64
    assert TR.max_rows_hidden(10 ** 9, 512) == 1


@pytest.mark.parametrize("residual", [False, True])
def test_face_fn_refuses_on_cpu_by_name_and_engine_serves_bitwise(residual):
    c, N, rows = 128, 8, 5
    w_a, w_b, w_o, ln_w, ln_b = _weights(c)
    engine = _engine_of(w_a, w_b, w_o, ln_w, ln_b)
    g = torch.Generator().manual_seed(5)
    x = torch.randn(rows, N, c, generator=g)
    mask_u = (torch.rand(rows, N, 1, generator=g) > 0.25).float()
    lines = []
    fn = TR.transition_face_fn(w_a=w_a, w_b=w_b, w_o=w_o, ln_w=ln_w, ln_b=ln_b, word="big", n_tokens=N, engine_fn=engine, residual=residual,
                               log=lines.append)
    assert fn.residual_in_fn is residual and fn.n_tokens == N and fn.cap_rows == TR.max_rows_hidden(N, c)
    assert evidence.schedule()["transition"] == "face:big" and evidence.schedule()["transition_face_ntokens"] == N
    ref_delta = engine(x, mask_u)
    xin = x.clone()
    y = fn(xin, mask_u)
    if residual:
        assert y is xin and torch.equal(y, x + ref_delta)                                 # x + delta landed in place
    else:
        assert torch.equal(y, ref_delta) and torch.equal(xin, x)                         # the delta; x untouched
    sched = evidence.schedule()
    kind = fn.face_state["refusal"]                                                      # by NAME: x_not_cuda (a bf16 cell's kernel row on a CPU tensor) or
    assert kind and sched["transition"] == f"engine:fallback({kind})" and fn.face_state["served"] == 0   # big:tier_winner_takes_no_mask:<row> (the fp32 cell's winner folds no mask)
    assert kind in ("x_not_cuda",) or "takes_no_mask" in kind or "tier" in kind, kind
    assert len(lines) == 1 and "refused word='big' by name" in lines[0]
    # sticky: the second call goes straight to the engine (one refusal line in all)
    y2 = fn(x.clone(), mask_u)
    assert torch.equal(y2, x + ref_delta if residual else ref_delta) and len(lines) == 1 and fn.face_state["engine"] == 2
    # through the driver: bind(residual attribute) == the plain engine binding, bitwise
    R = 9
    z = torch.randn(R, N, c, generator=g)
    m2 = (torch.rand(R, N, generator=g) > 0.25).float()
    lay = D.Layout(N, 2, 0, 8)
    ref = _bind(engine, 2).transition(z.clone(), m2, lay)
    got = _bind(fn, 2, transition_rows=4).transition(z.clone(), m2, lay)
    assert torch.equal(got, ref)


def test_face_fn_without_engine_answers_with_the_stock_row_and_refusals():
    c, N, rows = 128, 6, 3
    w_a, w_b, w_o, ln_w, ln_b = _weights(c, seed=3)
    W = KT.pack(w_a=w_a, w_b=w_b, w_o=w_o, ln_w=ln_w, ln_b=ln_b)
    fn = TR.transition_face_fn(W, word="big", n_tokens=N)                              # engine_fn=None: the provider's torch_swiglu row answers a refusal
    g = torch.Generator().manual_seed(6)
    x = torch.randn(rows, N, c, generator=g)
    mask_u = torch.ones(rows, N, 1)
    y = fn(x.clone(), mask_u)
    ref = KT.reference(x.reshape(-1, c), W, mask=mask_u.reshape(-1)).reshape(x.shape).float()
    assert y.shape == x.shape and torch.allclose(y, ref, atol=2e-4, rtol=2e-4) and fn.face_state["refusal"] is not None
    with pytest.raises(RowpairRefused, match="packed c=128"):
        fn(torch.randn(2, N, 64), torch.ones(2, N, 1))
    with pytest.raises(RowpairRefused, match="n_tokens=0"):
        TR.transition_face_fn(W, word="big", n_tokens=0)


def test_face_fn_sub_blocks_below_the_hidden_budget(monkeypatch):
    """rows/call = max_rows_hidden: a served path is exercised on CPU by asking the provider for its stock row word directly (torch_swiglu serves on
    any device), with a tiny cap so a 5-row block takes three face calls; == one call, tolerance-free (the statement is row-wise)."""
    c, N, rows = 64, 4, 5
    w_a, w_b, w_o, ln_w, ln_b = _weights(c, factor=2, seed=4)
    W = KT.pack(w_a=w_a, w_b=w_b, w_o=w_o, ln_w=ln_w, ln_b=ln_b)
    g = torch.Generator().manual_seed(8)
    x = torch.randn(rows, N, c, generator=g)
    mask_u = (torch.rand(rows, N, 1, generator=g) > 0.3).float()
    whole = TR.transition_face_fn(W, word="torch_swiglu", n_tokens=N)
    y_whole = whole(x.clone(), mask_u)
    assert whole.face_state["served"] == 1 and whole.face_state["refusal"] is None
    assert evidence.schedule()["transition_face"].startswith("transition:torch_swiglu") and evidence.schedule()["transition_face_calls"] == 1
    monkeypatch.setattr(TR, "max_rows_hidden", lambda N_, c_, factor=4, cap_elems=None: 2)
    split = TR.transition_face_fn(W, word="torch_swiglu", n_tokens=N)
    assert split.cap_rows == 2
    y_split = split(x.clone(), mask_u)
    assert split.face_state["served"] == 3 and torch.allclose(y_split, y_whole, atol=1e-5, rtol=1e-5)   # row-wise statement; the CPU GEMM's blocking
    res = TR.transition_face_fn(W, word="torch_swiglu", n_tokens=N, residual=True)                       # follows M, so equal to tolerance, not bits
    xin = x.clone()
    y_res = res(xin, mask_u)
    assert y_res is xin and torch.allclose(y_res, x + y_whole, atol=1e-5, rtol=1e-5)
