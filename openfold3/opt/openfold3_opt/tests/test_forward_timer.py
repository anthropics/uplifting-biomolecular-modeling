"""The per-item forward-time line (report.wrap_forward_timer / install_forward_timer): `[openfold3-opt] Model forward time: <s>s item=<query_id>
seed=<seed> tokens=<N> route=<kit|stock>` once per OpenFold3.forward call on every route — the kit routes wrap with the instance counter
(stack._wrap_model_class), the stock route in stock_pred before the stock call — parsed by report.FORWARD_TIME_RE; numerically inert; idempotent.
Right after it the phase line (report.phase_line / PHASE_RE): the same call split at run_trunk / the sample_diffusion module call / _rollout."""
import re
import sys

import pytest

from openfold3_opt import report
from openfold3_opt.tests import _stubs


class _Seed:                                   # a tensor-like seed (flatten()[0].item()) without torch
    shape = (1,)

    def flatten(self):
        return [self]

    def item(self):
        return 7


class _Mask:
    shape = (1, 5, 311)


def test_forward_time_line_grammar_and_wrap(capsys):
    class M:
        def forward(self, batch, extra=None):
            return {"out": batch, "extra": extra}
    assert report.wrap_forward_timer(M, route="kit") is True
    assert report.wrap_forward_timer(M, route="kit") is False                       # idempotent: one line per item
    out = M().forward({"query_id": ["1abc"], "seed": [42], "token_mask": _Mask()}, extra=3)
    assert out["extra"] == 3 and out["out"]["seed"] == [42]                            # arguments and the return value pass through
    M().forward({"query_id": ["7x y", "B"], "seed": _Seed()})                            # a tensor-like seed, two ids, no token mask
    M().forward({})                                                                     # nothing named: the line still prints
    raw = capsys.readouterr().err.strip().splitlines()
    err = [l for l in raw if report.FORWARD_TIME_WORDS in l]
    assert len(err) == 3 and all(l.startswith("[openfold3-opt] Model forward time: ") for l in err), err
    phase = [l for l in raw if l.startswith("[openfold3-opt] PHASE item=")]
    assert len(phase) == 3 and all(re.search(report.PHASE_RE, l) for l in phase), raw          # one phase line per item; a model without the phase callables: NA
    p0 = re.search(report.PHASE_RE, phase[0])
    assert (p0["name"], p0["lm"], p0["trunk"], p0["sampler"], p0["conf"], p0["seed"], p0["route"]) == ("1abc", "-", "NA", "NA", "NA", "42", "kit")
    notes = [l for l in raw if "PHASE-NOTE" in l]
    assert len(notes) == 3 and {l.split()[2] for l in notes} == {"trunk", "sampler", "conf"}, notes   # once per process per phase, not per item
    m = [re.search(report.FORWARD_TIME_RE, l) for l in err]
    assert all(m), err
    assert (m[0]["name"], m[0]["seed"], m[0]["tokens"], m[0]["route"]) == ("1abc", "42", "311", "kit")
    assert (m[1]["name"], m[1]["seed"], m[1]["tokens"]) == ("7x_y_B", "7", "-1")
    assert (m[2]["name"], m[2]["tokens"]) == ("?", "-1") and m[2]["seed"].isdigit() and int(m[2]["seed"]) == report.pass_seed()   # no seed in the batch: the seed in force, digits
    assert float(m[0]["s"]) >= 0.0 and report.FORWARD_TIME_WORDS in report.forward_time_line(1.5, "q", 1, 10, "kit")
    M().forward({"query_id": ["s"], "seed": ["notanint"]}); M().forward({"query_id": ["t"], "seed": [3.0]})
    err2 = [l for l in capsys.readouterr().err.strip().splitlines() if report.FORWARD_TIME_WORDS in l]
    assert re.search(report.FORWARD_TIME_RE, err2[0])["seed"] == str(report.pass_seed()) and re.search(report.FORWARD_TIME_RE, err2[1])["seed"] == "3"


def test_a_forward_that_raises_prints_no_line(capsys):
    class M:
        def forward(self, batch):
            raise MemoryError("device")
    report.wrap_forward_timer(M, route="kit")
    with pytest.raises(MemoryError):
        M().forward({"query_id": ["q"]})
    err = capsys.readouterr().err
    assert "Model forward time" not in err and "PHASE item=" not in err


def test_install_forward_timer_on_the_stock_route(capsys):
    """stock_pred's form: the model module imported by name and wrapped; the record names the target; absent module = a named record + line."""
    saved = {n: sys.modules.pop(n) for n in list(sys.modules) if n == report.MODEL_MODULE}
    try:
        sys.modules[report.MODEL_MODULE] = None                                        # import refused: the timer is NOT installed, by name
        rec = report.install_forward_timer("stock")
        assert rec["installed"] is False and "error" in rec and rec["target"].endswith("OpenFold3.forward")
        assert "forward timer NOT installed (stock)" in capsys.readouterr().err
        del sys.modules[report.MODEL_MODULE]
        m = _stubs.stub_model_module()
        rec = report.install_forward_timer("stock")
        assert rec == {"target": report.MODEL_MODULE + ".OpenFold3.forward", "words": "Model forward time:", "route": "stock", "installed": True}
        assert report.install_forward_timer("stock")["installed"] is True              # again: still one wrap
        m.OpenFold3().forward({"query_id": ["5xyz"], "seed": [1], "token_mask": _Mask()})
        raw = capsys.readouterr().err.strip().splitlines()
        err = [l for l in raw if report.FORWARD_TIME_WORDS in l]
        assert len(err) == 1 and re.search(report.FORWARD_TIME_RE, err[0])["route"] == "stock" and "item=5xyz seed=1 tokens=311" in err[0]
        ph = [l for l in raw if " PHASE item=" in l]
        assert len(ph) == 1 and re.search(report.PHASE_RE, ph[0])["route"] == "stock"        # the same wrap on the stock arm
    finally:
        sys.modules.pop(report.MODEL_MODULE, None)
        sys.modules.update(saved)


class _HookedModule:
    """A torch-free stand-in for an nn.Module: forward pre-hooks / forward hooks fire around __call__ the way Module.__call__ fires them."""
    def __init__(self, fn):
        self._fn, self._pre, self._post = fn, [], []

    def register_forward_pre_hook(self, h):
        self._pre.append(h)

    def register_forward_hook(self, h):
        self._post.append(h)

    def __call__(self, *a, **k):
        for h in self._pre:
            h(self, a)
        out = self._fn(*a, **k)
        for h in self._post:
            h(self, a, out)
        return out


def _model_class(sleep):
    class OpenFold3:                                             # the stock statement order: run_trunk, then _rollout = sampler call + heads
        def __init__(self):
            self.sample_diffusion = _HookedModule(lambda **k: sleep(0.02) or "x")

        def run_trunk(self, batch, num_cycles, inplace_safe=False):
            sleep(0.03)
            return 1, 2, 3

        def _rollout(self, batch, si_input, si_trunk, zij_trunk, inplace_safe=False):
            x = self.sample_diffusion(batch=batch)
            sleep(0.01)                                          # the confidence heads
            return {"x": x}

        def forward(self, batch):
            s = self.run_trunk(batch=batch, num_cycles=4)
            sleep(0.005)                                         # glue outside every phase
            return self._rollout(batch=batch, si_input=s[0], si_trunk=s[1], zij_trunk=s[2])
    return OpenFold3


def test_phase_line_splits_the_forward_at_the_model_callables(capsys):
    import time
    M = _model_class(time.sleep)
    report._PHASE["notes"].clear()
    assert report.wrap_forward_timer(M, route="kit")
    m = M()
    assert m.forward({"query_id": ["q1"], "seed": [5]}) == {"x": "x"}
    assert M.run_trunk.__name__ == "run_trunk" and M._rollout.__name__ == "_rollout"      # the class attributes untouched (the lines patch and `is`-check them)
    assert set(m._of3_phase_timers) == {"trunk", "rollout", "sampler"}
    m.forward({"query_id": ["q2"], "seed": [6]})                                            # a second item on the same instance: attached once, timed again
    raw = capsys.readouterr().err.strip().splitlines()
    fwd = [re.search(report.FORWARD_TIME_RE, l) for l in raw if report.FORWARD_TIME_WORDS in l]
    ph = [re.search(report.PHASE_RE, l) for l in raw if " PHASE item=" in l]
    assert len(fwd) == 2 and len(ph) == 2 and not [l for l in raw if "PHASE-NOTE" in l], raw
    for f_, p in zip(fwd, ph):
        assert p["name"] == f_["name"] and p["seed"] == f_["seed"] and p["route"] == "kit" and p["lm"] == "-"
        trunk, sampler, conf, total = (float(p[k]) for k in ("trunk", "sampler", "conf", "total"))
        assert trunk >= 0.03 and sampler >= 0.02 and conf >= 0.01, p.string
        assert trunk + sampler + conf <= total + 1e-6 and abs(total - float(f_["s"])) < 0.006, p.string   # phases sum below the forward line's total (2-decimal there)
    assert raw.index([l for l in raw if " PHASE item=q1" in l][0]) == raw.index([l for l in raw if "item=q1 seed=5" in l and "forward time" in l][0]) + 1   # right after its forward line


def test_a_line_that_replaces_the_class_methods_is_what_gets_timed(capsys):
    """A line's replacement of OpenFold3.run_trunk / _rollout installed on the class (before or after the first forward) is reached through the
    instance timers; a rollout that drives the sampler below Module.__call__ prints sampler NA and conf NA with their PHASE-NOTE, never a split."""
    import time
    M = _model_class(time.sleep)
    report._PHASE["notes"].clear()
    report.wrap_forward_timer(M, route="kit")
    m = M()
    calls = []
    stock_rt = M.run_trunk
    M.run_trunk = lambda self, batch, num_cycles, inplace_safe=False: calls.append("line") or stock_rt(self, batch, num_cycles, inplace_safe)
    M._rollout = lambda self, batch, si_input, si_trunk, zij_trunk, inplace_safe=False: {"x": self.sample_diffusion._fn(batch=batch)}   # the sampler's internals, not its __call__
    m.forward({"query_id": ["q3"], "seed": [1]})
    raw = capsys.readouterr().err.strip().splitlines()
    p = re.search(report.PHASE_RE, [l for l in raw if " PHASE item=" in l][0])
    assert calls == ["line"] and float(p["trunk"]) >= 0.03 and (p["sampler"], p["conf"]) == ("NA", "NA")
    notes = [l for l in raw if "PHASE-NOTE" in l]
    assert len(notes) == 2 and "sampler not separable" in notes[0] and "conf not separable" in notes[1], notes
