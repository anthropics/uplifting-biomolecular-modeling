"""A guard/composition test runs against the REAL component objects, never a hand-built state. This test composes the eager kit's guard
with the REAL ew1: the real `engines.e1.kits.ew1.patches` module object (importable torch-free under the test's fake torch), its real
`_state` (the bindings record) and its real `_bind` writer / `unapply` restorer. Only the MODEL is a stub (a component is never stubbed;
the model is the user's object)."""
import importlib
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)


class _Ids:
    """The batch's input_ids stand-in: .numel() and .shape (the guard reads nothing else)."""
    def __init__(self, B, T):
        self.shape = (B, T)

    def numel(self):
        return self.shape[0] * self.shape[1]


class _Layer:
    def forward(self, x):                       # the class-level = the stock's
        return ("stock_layer", x)


class _Inner:
    """E1Model's stand-in: the class-level forward is the stock's; ew1 binds an instance-level forward on it."""
    def __init__(self):
        self.layer = _Layer()
        self.config = type("C", (), {"hidden_size": 1024, "intermediate_size": 3072})()

    def forward(self, input_ids, **kw):
        return ("stock", self.layer.forward(input_ids.numel()))


class _Outer:
    def __init__(self):
        self.model = _Inner()
        self.config = self.model.config


def _real_ew1_patches():
    P = importlib.import_module("engines.e1.kits.ew1.patches")
    assert "originals" in P._state and callable(P._bind) and callable(P.unapply)          # the real component's surface
    return P


def test_guard_composes_with_the_real_ew1_bindings_record():
    P = _real_ew1_patches()
    G = importlib.import_module("engines.e1.kits.eager.guard")
    P.unapply()                                                                             # a clean real state (ew1's own restorer)
    m = _Outer()
    log = []

    def ew1_forward(self, input_ids, **kw):                                                 # what ew1 binds on E1Model (the fused path)
        log.append("ew1_outer"); return ("ew1", self.layer.forward(input_ids.numel()))

    def ew1_layer(self, x):
        log.append("ew1_layer"); return ("ew1_layer", x)
    P._bind(m.model, "forward", ew1_forward)                                                # the REAL writer into the REAL record
    P._bind(m.model.layer, "forward", ew1_layer)
    assert set(P._state["originals"]) == {(id(m.model), "forward"), (id(m.model.layer), "forward")}
    rec = G.install(m, P._state, m.config.hidden_size, m.config.intermediate_size)          # the composition kits/eager's apply performs (ew1.patches._state)
    assert rec["sites_wrapped"] == 2 and rec["bound_tokens"] == G.bound_tokens(1024, 3072) and G._G["installed"]
    below, above = _Ids(8, 516), _Ids(2048, 516)
    assert m.model.forward(below) == ("ew1", ("ew1_layer", 8 * 516)) and log == ["ew1_outer", "ew1_layer"]
    log.clear()
    assert m.model.forward(above) == ("stock", ("stock_layer", 2048 * 516)) and log == []       # the WHOLE forward on the class-level stock methods
    c = G.counters()
    assert c["guard:forwards"] == 2 and c["guard:stock_path_forwards"] == 1 and c["guard:above_int32_bound_forwards"] == 1 and c["guard:stock_path_sites"] == 1
    assert "GLU intermediate" in G.line() and str(rec["bound_tokens"]) in G.line()
    G.uninstall()
    assert m.model.__dict__["forward"].__func__ is ew1_forward and m.model.layer.__dict__["forward"].__func__ is ew1_layer   # ew1's bindings back beneath
    P.unapply()                                                                             # ew1's restorer: the originals popped, the record cleared
    assert "forward" not in m.model.__dict__ and "forward" not in m.model.layer.__dict__ and not P._state["originals"]


def test_v1_24_call_shape_fails_against_the_real_ew1_init_state():
    """Passing ew1.__init__._state (not ew1.patches._state) to the guard raises KeyError 'originals' against the REAL object."""
    E = importlib.import_module("engines.e1.kits.ew1")
    G = importlib.import_module("engines.e1.kits.eager.guard")
    P = _real_ew1_patches(); P.unapply()
    m = _Outer(); P._bind(m.model, "forward", lambda self, ids, **kw: None)
    assert "originals" not in E._state
    with pytest.raises(KeyError):
        G.install(m, E._state, 1024, 3072)
    P.unapply()
    assert not G._G["installed"]


def test_kit_eager_installs_the_guard_on_the_real_record_by_source():
    src = open(os.path.join(ROOT, "engines", "e1", "kits", "eager", "__init__.py")).read()
    seg = src[src.index("def apply("):src.index("def unapply(")]
    assert "from ..ew1 import patches as _ew1p" in seg and "guard.install(model, _ew1p._state," in seg and "_ew1._state" not in seg
    assert "bound=guard.INT64_BOUND" in seg and "ew1_i64.install()" in seg          # ew1_i64 composed: the int32 bound is counted, not routed
    assert 'anylen.install("hybrid")' in seg                                          # the any-length flex route is part of the set
    assert seg.index("pins.apply_autotune_pin(size)") < seg.index("v1_2.apply(") < seg.index("guard.install(") < seg.index("anylen.install(")   # the one order


def test_a3_off_is_worded_under_the_levers_own_name():
    """The A3 lever's line: `state=off reason="OFF on <class>: …"` where the class table says so, `note="untested class …" state=on` on a GPU
    of no tested class — keyed by the lever's own name in v1.LEVERS (read by source: the kit is not importable on a CPU host)."""
    import re
    src = open(os.path.join(ROOT, "engines", "e1", "kits", "eager", "__init__.py")).read()
    m = re.search(r'^A3_LEVER = "([^"]+)"', src, re.M)
    assert m and m.group(1) == "attn:A3_flex_kernel_options"
    v1src = open(os.path.join(ROOT, "engines", "e1", "kits", "v1", "__init__.py")).read()
    attnsrc = open(os.path.join(ROOT, "engines", "e1", "kits", "attn", "__init__.py")).read()
    assert 'tuple(f"attn:{l}" for l in _components.attn.LEVERS)' in v1src and '"A3_flex_kernel_options"' in attnsrc   # v1 names it attn:<attn lever>; attn's lever id is the suffix
    assert 'levers[A3_LEVER] = {"on": False, "reason": f"OFF on {rec12[\'gpu_class\']}: {rec12[\'A3_cite\']}"}' in src
    assert 'levers[A3_LEVER]["note"] = rec12["A3_cite"]' in src
    assert 'LEVERS = ("rmsnorm_autotune_pin",) + tuple(v1.LEVERS) + ("ew1_i64_offsets", "anylen_flex")' in src
