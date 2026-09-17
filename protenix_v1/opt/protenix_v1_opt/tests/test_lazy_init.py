"""CPU tests for lever lazy_init (lib/ptx1_lazy_init.py): the random initialisers are no-op'd only inside the context and restored; install()
patches a runner class's init_model / load_checkpoint; the recheck mode compares the stock-initialised+loaded model with the lazily
constructed+loaded one tensor by tensor. torch-dependent parts skip without torch; the protenix import inside install() is stubbed."""
import os
import sys
import types

import pytest

from protenix_v1_opt import kit as K, modes as M, report as R

LIB = os.path.join(K.kit_home(), "lib")


@pytest.fixture()
def LZ(monkeypatch):
    torch = pytest.importorskip("torch")
    if LIB not in sys.path:
        monkeypatch.syspath_prepend(LIB)
    for name in ("protenix", "protenix.model", "protenix.model.protenix", "protenix.model.modules", "protenix.model.modules.primitives",
                 "protenix.model.triangular", "protenix.model.triangular.layers"):          # install() imports the model tree: stub modules (no protenix here)
        if name not in sys.modules:
            monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    helper_calls = {"n": 0}
    def trunc_normal_init_(w, scale=1.0):                                                     # a protenix-style helper at an import site the lever must find and no-op
        helper_calls["n"] += 1; torch.nn.init.normal_(w)
    sys.modules["protenix.model.triangular.layers"].trunc_normal_init_ = trunc_normal_init_
    import importlib
    mod = importlib.import_module("ptx1_lazy_init"); mod = importlib.reload(mod)
    mod._helper_calls = helper_calls
    yield mod
    sys.modules.pop("ptx1_lazy_init", None)


def test_membership_and_strategy():
    for mode in ("exact", "fast", "big"):
        assert "lazy_init" in M.resolve(mode, environ={}).levers
    assert M.LEVERS["lazy_init"]["class"] == "exact" and R.STRATEGY_IDS["lazy_init"] == "F6.weights_residency_init" and R.lever_impl("lazy_init") == ("lib/ptx1_lazy_init.py", "kit")


def test_skip_context_noops_and_restores(LZ):
    import torch
    w = torch.zeros(4, 4)
    with LZ.skip_random_init() as patched:
        assert "torch.nn.init.trunc_normal_" in patched and "protenix.model.triangular.layers.trunc_normal_init_" in patched
        torch.nn.init.normal_(w); torch.nn.init.kaiming_uniform_(w); sys.modules["protenix.model.triangular.layers"].trunc_normal_init_(w)
        assert float(w.abs().sum()) == 0.0                                                    # nothing wrote
        torch.nn.init.ones_(w); assert float(w.sum()) == 16.0                                 # the cheap constant inits stay live
    torch.nn.init.normal_(w); assert float(w.abs().sum()) != 16.0                             # restored
    sys.modules["protenix.model.triangular.layers"].trunc_normal_init_(w); assert LZ._helper_calls["n"] == 1


def _runner_cls(LZ):
    import torch
    ckpt = {"lin.weight": torch.arange(16.0).reshape(4, 4), "lin.bias": torch.arange(4.0)}
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__(); self.lin = torch.nn.Linear(4, 4)
            torch.nn.init.trunc_normal_(self.lin.weight); sys.modules["protenix.model.triangular.layers"].trunc_normal_init_(self.lin.bias.data.unsqueeze(0))
            self.register_buffer("np_buf", torch.full((3,), 7.0), persistent=False)          # a non-persistent buffer the checkpoint does not cover: hashed too
    class Runner:
        lines = []
        def __init__(self):
            self.init_model(); self.load_checkpoint()
        def init_model(self):
            self.model = Model()
        def load_checkpoint(self):
            self.model.load_state_dict(ckpt, strict=True)
    return Runner


def test_install_lazy_mode(LZ):
    import torch
    Runner = _runner_cls(LZ)
    st = LZ.install(Runner, mode="1", log=Runner.lines.append)
    assert st["installed"] and Runner.init_model.__wrapped__ is not None
    r = Runner()
    rep = LZ.report()
    assert rep["constructs"] == 1 and rep["patched"] >= 7 and rep["recheck"] is None and rep["mode"] == "1"
    assert torch.equal(r.model.lin.weight, torch.arange(16.0).reshape(4, 4)) and any("random init skipped" in l for l in Runner.lines)
    with pytest.raises(ValueError):
        LZ._STATE["installed"] = False; LZ.install(Runner, mode="check")


def test_install_recheck_mode_is_identical(LZ):
    Runner = _runner_cls(LZ)
    LZ.install(Runner, mode="recheck", log=Runner.lines.append)
    Runner()
    rc = LZ.report()["recheck"]
    assert rc["verdict"] == "IDENTICAL" and rc["n_equal"] == rc["n_tensors"] == 3 and rc["unequal"] == []      # weight, bias, <nonpersistent>np_buf
    assert any("RECHECK" in l and "3/3" in l for l in Runner.lines)


def test_evidence_and_line():
    res = M.resolve("fast", environ={})
    acct = {"cfg": {"trimul": "fast"}, "counts": {}, "lazy_init": {"installed": True, "constructs": 1, "lazy_construct_s": 4.2, "patched": 9,
            "recheck": {"verdict": "IDENTICAL", "n_equal": 5187, "n_tensors": 5187}}}
    ev = R.kit_evidence(acct, "fast", ("lazy_init",))
    assert ev["lazy_init"]["served"] == 1 and ev["lazy_init"]["fallback"] == {} and ev["lazy_init"]["recheck"] == "IDENTICAL:5187/5187"
    line = [l for l in R.lever_lines(ev, {"mode": "fast"}) if l.endswith(" lever=lazy_init")][0]
    assert line == "[protenix-v1-opt] LEVER name=F6.weights_residency_init state=on impl=lib/ptx1_lazy_init.py origin=kit strategy=F6.weights_residency_init served=1 construct_s=4.2 patched=9 recheck=IDENTICAL:5187/5187 lever=lazy_init"
    bad = R.kit_evidence(dict(acct, lazy_init={"installed": True, "constructs": 1, "patched": 9, "recheck": {"verdict": "DIFFERENT", "n_equal": 5000, "n_tensors": 5187}}), "fast", ("lazy_init",))
    assert bad["lazy_init"]["fallback"] == {"recheck": "DIFFERENT:5000/5187"}                     # a differing recheck is a failed lever (partial), never a pass
    none = R.kit_evidence(dict(acct, lazy_init={"installed": False}), "fast", ("lazy_init",))
    assert none["lazy_init"]["served"] == 0 and "lazy_init_installed" in none["lazy_init"]["fallback"]
