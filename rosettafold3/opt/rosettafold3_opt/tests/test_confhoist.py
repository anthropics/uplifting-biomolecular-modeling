"""The confhoist lever (confhoist.py): the confidence head's sample-invariant prologue computed once per item — source pin, cache discipline,
the per-item bound, the n_gpu>1 decline, the CPU helper's named no-op, and its registry / report words (CPU: no torch, stand-in tensors)."""
import os
import sys
import types

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from rosettafold3_opt import confhoist as ch  # noqa: E402
from rosettafold3_opt import registry, report  # noqa: E402


class _T:
    """A stand-in tensor: identity, an in-place version counter, clone()."""
    is_cuda = True

    def __init__(self, name):
        self.name = name; self._version = 0

    def clone(self):
        c = _T(self.name + ".clone"); c.src = self; return c


class _Model:
    def forward(self, *a, **k):
        return "outputs"


class _Head:
    use_Cb_distances = False

    def forward(self, S_inputs_I, S_trunk_I, Z_trunk_II, X_pred_L, seq, rep_atoms, frame_atom_idxs=None):  # pragma: no cover - replaced
        return "stock"


@pytest.fixture
def hoist_on(monkeypatch):
    """confhoist installed on the stand-in head class with recorders for the prologue / tail (the torch statements are upstream's; the
    discipline around them is this module's and is what these tests pin)."""
    rec = {"prologue": [], "tail": []}
    monkeypatch.setattr(ch, "_prologue", lambda self, S_in, S, Z: (rec["prologue"].append((S_in, S, Z)) or (_T("S_pro"), _T("Z_pro"))))
    monkeypatch.setattr(ch, "_tail", lambda self, S_pro, Z_pro, X, seq, rep, fai: (rec["tail"].append((S_pro, Z_pro, X)) or {"pae_logits": X}))
    monkeypatch.setattr(ch, "FORWARD_NORM_SHA256", ch.norm_sha256(__import__("inspect").getsource(_Head.forward)))
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    mod = types.SimpleNamespace(ConfidenceHead=_Head)
    d = ch.enable(mod)
    assert d["installed"] and d["on"] and _Head.forward is ch.forward, d
    assert ch.wrap_model(types.SimpleNamespace(RF3WithConfidence=_Model)) and _Model.forward is ch._model_forward
    yield rec
    ch.disable()
    for k in ("calls", "prologues", "reused", "cleared", "stale", "passthrough"):
        ch.STATE[k] = 0
    ch.STATE.update(armed=False, reason=None, refusal=None, conflict=None)
    assert _Head.forward is not ch.forward and _Model.forward is not ch._model_forward


def test_norm_sha256_is_indentation_and_blank_line_neutral():
    a = "def f(x):\n    return x + 1\n"
    b = "        def f(x):\n\n            return x + 1   \n\n"
    assert ch.norm_sha256(a) == ch.norm_sha256(b) != ch.norm_sha256("def f(x):\n    return x + 2\n")


def test_upstream_digest_looks_through_report_only_wrappers():
    class C:
        def forward(self):
            return 1
    dg = ch.upstream_digest(C)
    orig = C.forward

    def wrapper(self):
        return orig(self)
    wrapper.__wrapped__ = orig
    C.forward = wrapper
    assert ch.upstream_digest(C) == dg                       # a functools-style wrapper is looked through: the pin is judged on upstream's code


def test_five_samples_run_the_prologue_once_and_the_item_bound_drops_it(hoist_on):
    rec = hoist_on
    head, S_in, S, Z = _Head(), _T("S_in"), _T("S"), _T("Z")
    outs = [head.forward(S_in, S, Z, _T(f"X{i}"), _T("seq"), _T("rep")) for i in range(5)]
    assert len(rec["prologue"]) == 1 and len(rec["tail"]) == 5
    assert ch.describe()["census"] == {"calls": 5, "prologues": 1, "reused": 4, "cleared": 0, "stale": 0, "passthrough": 0}
    assert all(t[1].name == "Z_pro" for t in rec["tail"])                       # every sample reads the one cached pair prologue
    assert all(t[0].name == "S_pro.clone" for t in rec["tail"])                 # the single track is handed over as a fresh clone per sample
    assert [o["pae_logits"].name for o in outs] == [f"X{i}" for i in range(5)]
    # the model wrapper's bound: the cache never outlives RF3WithConfidence.forward
    assert _Model().forward("inputs") == "outputs" and ch.describe()["census"]["cleared"] == 1 and ch._CACHE["entry"] is None
    assert ch.problems() == []
    head.forward(S_in, S, Z, _T("X5"), _T("seq"), _T("rep"))
    assert len(rec["prologue"]) == 2                                             # after the bound, the same operands recompute (a new item may reuse addresses, never entries)


def test_other_operands_or_an_in_place_write_recompute_never_read_stale(hoist_on):
    rec = hoist_on
    head, S_in, S, Z = _Head(), _T("S_in"), _T("S"), _T("Z")
    head.forward(S_in, S, Z, _T("X0"), _T("seq"), _T("rep"))
    head.forward(S_in, S, _T("Z2"), _T("X1"), _T("seq"), _T("rep"))            # another pair tensor object: a new prologue
    assert len(rec["prologue"]) == 2
    Z2 = rec["prologue"][-1][2]
    Z2._version += 1                                                             # an in-place write on a cached operand: recompute, counted 'stale'
    head.forward(S_in, S, Z2, _T("X2"), _T("seq"), _T("rep"))
    assert len(rec["prologue"]) == 3 and ch.describe()["census"]["stale"] == 1
    assert ch.problems() == ["confhoist:stale_operands:1"]                       # a stale operand is a named verdict failure (it never happens on the stock call path)
    other = _Head()
    head.forward(S_in, S, Z2, _T("X3"), _T("seq"), _T("rep")); other.forward(S_in, S, Z2, _T("X4"), _T("seq"), _T("rep"))
    assert len(rec["prologue"]) == 4                                             # another head instance: its own prologue


def test_a_foundry_whose_forward_is_not_the_pins_is_refused_by_name(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)

    class Other:
        def forward(self, *a, **k):
            return "another foundry"
    try:
        d = ch.enable(types.SimpleNamespace(ConfidenceHead=Other))
        assert not d["installed"] and not d["on"] and d["refusal"].startswith("upstream_bytes:") and Other.forward is not ch.forward
        assert ch.problems() == [f"confhoist:refused:{d['refusal']}"] and d["ok"] is False
    finally:
        ch.STATE.update(reason=None, refusal=None, upstream_sha12=None)


def test_a_cuda_less_helper_process_is_a_named_no_op(monkeypatch):
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False))
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    try:
        d = ch.enable(types.SimpleNamespace(ConfidenceHead=_Head))
        assert d["on"] is False and d["installed"] is False and d["reason"] == ch.CPU_PROCESS and ch.problems() == [] and d["ok"] is False
        assert _Head.forward is not ch.forward
    finally:
        ch.STATE.update(reason=None, refusal=None)


def test_under_n_gpu_above_one_the_lever_steps_aside_by_name():
    rep = {"levers": ["graph", "confhoist"], "n_gpu": 2}
    watches = []
    try:
        ch.arm(rep, lambda trigger, cb, rep_: watches.append(trigger))
        assert watches == [] and rep["confhoist"]["conflict"] == "n_gpu" and rep["confhoist"]["ok"] is True and not ch.STATE["installed"]
    finally:
        ch.STATE.update(armed=False, conflict=None, reason=None, on=False)


def test_arm_installs_watches_only_for_rows_naming_the_lever(monkeypatch):
    for m in (ch.HEADS_MODULE, ch.MODEL_MODULE):
        monkeypatch.delitem(sys.modules, m, raising=False)
    watches = []
    rep = {"levers": ["graph", "xtr"], "n_gpu": 1}
    ch.arm(rep, lambda trigger, cb, rep_: watches.append(trigger))
    assert watches == [] and "confhoist" not in rep and not ch.STATE["armed"]
    rep = {"levers": ["graph", "confhoist"], "n_gpu": 1}
    try:
        ch.arm(rep, lambda trigger, cb, rep_: watches.append(trigger))
        assert watches == [ch.HEADS_MODULE, ch.MODEL_MODULE] and rep["confhoist"]["armed"] is True
    finally:
        ch.STATE.update(armed=False)


def test_registry_and_report_words():
    lv = registry.LEVERS["confhoist"]
    assert lv.switch is None and lv.kit == registry.PACKAGE and lv.probe == ("confhoist", "reused") and lv.strategy == "LOCAL.step_invariant_hoist"
    assert registry.IMPL_OF["confhoist"] == ("confhoist.py", "kit")
    t = {"confhoist": dict(ch.describe(), census={"calls": 10, "prologues": 2, "reused": 8, "cleared": 2, "stale": 0, "passthrough": 0})}
    assert report._probe_count(lv, t) == 8
    ev = report._evidence("confhoist", t)
    assert ev["calls"] == 10 and ev["prologues"] == 2 and ev["reused"] == 8 and ev["cleared"] == 2


def test_confln_rides_the_prologue_and_is_refused_by_name_without_confhoist(monkeypatch):
    """confln (fast class): the prologue's whole-tensor layer norms by _whole_layer_norm when the row names confln WITH confhoist; named
    alone it is refused by name (ok=False, the sentence names confhoist); its tally block counts calls."""
    ch.STATE.update(armed=False, fastln=False, fastln_calls=0, fastln_refusal=None)
    rep = {"levers": ["graph", "confln"], "n_gpu": 1}
    try:
        ch.arm(rep, lambda trigger, cb, rep_: None)
        d = rep["confln"]
        assert d["ok"] is False and "confhoist" in d["reason"] and d["on"] is False and not ch.STATE["armed"]
    finally:
        ch.STATE.update(armed=False, fastln=False, fastln_refusal=None)
    for m in (ch.HEADS_MODULE, ch.MODEL_MODULE):
        monkeypatch.delitem(sys.modules, m, raising=False)
    rep = {"levers": ["graph", "confhoist", "confln"], "n_gpu": 1}
    try:
        ch.arm(rep, lambda trigger, cb, rep_: None)
        assert ch.STATE["fastln"] is True and rep["confln"]["name"] == "confln" and rep["confln"]["eps"] == 1e-5
        # the prologue statement choice: fastln -> _whole_layer_norm x3 (stand-in records calls)
        calls = []
        monkeypatch.setattr(ch, "_whole_layer_norm", lambda x, eps=ch.LN_EPS: calls.append(x) or x)
        if "torch" not in sys.modules:                                          # CPU box without torch: the prologue's `import torch.nn.functional` resolves to a stand-in
            t = types.ModuleType("torch"); tn = types.ModuleType("torch.nn"); tf = types.ModuleType("torch.nn.functional")
            tf.layer_norm = lambda x, normalized_shape=None: x; t.nn = tn; tn.functional = tf
            for name, mod in (("torch", t), ("torch.nn", tn), ("torch.nn.functional", tf)):
                monkeypatch.setitem(sys.modules, name, mod)
        class _X:
            shape = (2, 3)
            def detach(self): return self
            def float(self): return self
            def unsqueeze(self, d): return 0
            def __add__(self, o): return self
        head = types.SimpleNamespace(layer_norm_along_feature_dimension=False, process_s_inputs_right=lambda x: x, process_s_inputs_left=lambda x: x)
        ch._prologue(head, _X(), _X(), _X())
        assert len(calls) == 3
    finally:
        ch.STATE.update(armed=False, fastln=False, fastln_calls=0)


def test_confln_registry_words_and_withhold_dependency(monkeypatch):
    from rosettafold3_opt import leversoff
    lv = registry.LEVERS["confln"]
    assert lv.switch is None and lv.probe == ("confln", "calls") and lv.strategy == "LOCAL.layernorm_kernel" and registry.IMPL_OF["confln"] == ("confhoist.py", "kit")
    assert leversoff.REQUIRES["confln"] == ("confhoist",)
    t = {"confln": {"name": "confln", "on": True, "calls": 3, "eps": 1e-5, "ok": True}}
    assert report._probe_count(lv, t) == 3 and report._evidence("confln", t) == {"calls": 3, "eps": 1e-5}
