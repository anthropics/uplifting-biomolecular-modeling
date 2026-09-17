"""The `fastjson` cell (exact class, in bytes): the engine's confidence-JSON writer's encoder class replaced by a subclass whose encode() renders
the same characters row-wise at C level. CPU-only: the text is compared with `json.dumps(..., cls=<the stock class>)` byte for byte on the
structures the writer renders (numpy float32 scalars, 1-D per-atom and 2-D per-token-pair float32 blocks, nested dicts keyed by chain ids)
and on the port's edge cases (NaN / ±Infinity, float16 / float64 / integer / boolean / object / 0-d / 3-D / empty arrays, unicode, escapes,
empty containers, tuples, bools, None, big ints, non-str keys, sort_keys, other indents and separators, ensure_ascii off, allow_nan off,
skipkeys, circular references, objects the encoder refuses) for the 0.4.x conversion (`tolist()`) and the 0.5.x one (rounded); the install
on a stand-in writer module before and after its import; the route knob; refusals by name; the lines and the registry row."""
import json
import os
import sys
import types

import pytest

np = pytest.importorskip("numpy")

from openfold3_opt import modes, registry, stack
from openfold3_opt.cells import fastjson, of3_fastjson as core
from openfold3_opt.tests import _stubs


class NumpyEncoder(json.JSONEncoder):                      # openfold3 0.4.1 core/runners/writer.py:33-44, verbatim in behaviour
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


class NumpyEncoder050(json.JSONEncoder):                   # openfold3 0.5.0's: float16 / float32 rounded to 3 / 6 decimals
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            if obj.dtype == np.float16:
                return obj.astype(np.float64).round(3).tolist()
            elif obj.dtype == np.float32:
                return obj.astype(np.float64).round(6).tolist()
            return obj.tolist()
        elif isinstance(obj, np.generic):
            if obj.dtype == np.float16:
                return round(obj.item(), 3)
            elif obj.dtype == np.float32:
                return round(obj.item(), 6)
            return obj.item()
        return super().default(obj)


def _writer_payloads(n_tok=37, n_atom=290, seed=3):
    rs = np.random.RandomState(seed)
    pae = rs.uniform(0, 31.75, size=(n_tok, n_tok)).astype(np.float32)
    pde = rs.uniform(0, 31.75, size=(n_tok, n_tok)).astype(np.float32)
    plddt = rs.uniform(0, 100, size=(n_atom,)).astype(np.float32)
    aggregated = {"avg_plddt": np.mean(plddt), "gpde": np.float32(0.73105854), "ptm": np.array(0.8351, dtype=np.float32), "iptm": np.array([0.42], dtype=np.float32)[0],
                  "disorder": 0.0, "has_clash": np.bool_(False), "sample_ranking_score": np.float64(0.61234567890123),
                  "chain_ptm": {"A": np.float32(0.81), "B": np.float32(0.79)}, "chain_pair_iptm": {"A": {"A": np.float32(0.81), "B": np.float32(0.4)}, "B": {"A": np.float32(0.4), "B": np.float32(0.79)}},
                  "bespoke_iptm": {"(A, B)": np.float32(0.40000001)}}
    full = {"plddt": plddt, "pde": pde, "pae": pae}
    return aggregated, full


def _edge_payload():
    f32 = (np.arange(20, dtype=np.float32).reshape(4, 5) / np.float32(3.0))
    f32[1, 2] = np.nan
    f32[2, 0] = np.inf
    f32[3, 4] = -np.inf
    return {
        "f32": f32, "f64": f32.astype(np.float64) * 1e-9, "f16": np.array([0.1, 65504.0, -2.5], dtype=np.float16), "row": np.array([1e16, 1e-5, 123456789.0, -0.0, 5e-324, 1.7976931348623157e308]),
        "i64": np.arange(6, dtype=np.int64).reshape(2, 3) - 3, "u8": np.arange(3, dtype=np.uint8), "b": np.array([True, False]), "obj": np.array([1, "x", None], dtype=object),
        "d0": np.array(2.5, dtype=np.float32), "d3": np.ones((2, 1, 2), dtype=np.float32), "e1": np.zeros((0,), dtype=np.float32), "e2": np.zeros((3, 0)), "e3": np.zeros((0, 3)),
        "scalars": [np.float32(0.1), np.float64(0.1), np.int32(-7), np.uint64(2 ** 63), np.bool_(True), np.float16(1.5), float("nan"), float("inf"), -float("inf")],
        "py": [None, True, False, 0, -1, 10 ** 30, 1.0, 0.1, 1e22, 1e-7, "", "a\"b\\c\n\t\u00e9\u2192\U0001f9ec", [], {}, (), [[[]]], {"": {"": []}}],
        "keys": {1: "int", 2.5: "float", True: "true", None: "null", "z": 0, "a": 1},
        "nested": {"A": {"B": [np.float32(1.0), [np.array([1.5, 2.5], dtype=np.float32), {"c": np.array([[1, 2], [3, 4]], dtype=np.int32)}]]}},
    }


@pytest.fixture()
def patched_classes():
    out = {}
    for base in (NumpyEncoder, NumpyEncoder050):
        out[base.__name__] = (base, core.make_encoder_class(base))
    keep = {k: (dict(v) if isinstance(v, dict) else v) for k, v in core.STATE.items()}
    core.STATE["route"] = "fast"
    yield out
    core.STATE.clear()
    core.STATE.update(keep)


@pytest.mark.parametrize("kw", [dict(indent=4), dict(indent=2), dict(indent=0), dict(indent="\t"), dict(indent=4, sort_keys=True), dict(indent=4, ensure_ascii=False),
                                dict(indent=3, separators=(" ,", "= ")), dict(indent=4, separators=(",", ":")), dict(indent=4, skipkeys=True), dict(indent=None), dict()])
def test_text_equals_the_stock_encoders_byte_for_byte(patched_classes, kw):
    aggregated, full = _writer_payloads()
    edge = _edge_payload()
    for name, (base, fast) in patched_classes.items():
        for payload in (aggregated, full, edge, [full, aggregated], "plain string \u00e9", 3.5, None, [], {}):
            if kw.get("sort_keys") and payload is edge:
                payload = {k: v for k, v in edge.items() if k != "keys"}          # mixed-type keys do not sort (TypeError on both routes, covered below)
            want = json.dumps(payload, cls=base, **kw)
            got = json.dumps(payload, cls=fast, **kw)
            assert got == want, (name, kw, len(want), next((i for i, (a, b) in enumerate(zip(got, want)) if a != b), None))
    assert core.STATE["fallback"] == 0, core.STATE["fallback_by"]
    if kw.get("indent") is not None:                                                   # indented calls are the fast route's; indent None is the stock C encoder's own call, not counted
        assert core.STATE["served"] > 0 and core.STATE["arrays"] > 0 and core.STATE["bytes"] > 0
    else:
        assert core.STATE["served"] == 0 and core.STATE["stock"] == 0


def test_the_writers_call_form_is_served_not_fallen_back(patched_classes):
    """json.dumps(obj, indent=4, cls=NumpyEncoder) on the aggregated dict and on the full plddt/pde/pae dict: both served, the three arrays and
    the per-atom vector rendered by the row route, nothing through the stock encoder."""
    base, fast = patched_classes["NumpyEncoder"]
    aggregated, full = _writer_payloads(n_tok=64, n_atom=500)
    s0 = dict(core.STATE)
    assert json.dumps(full, indent=4, cls=fast) == json.dumps(full, indent=4, cls=base)
    assert json.dumps(aggregated, indent=4, cls=fast) == json.dumps(aggregated, indent=4, cls=base)
    assert core.STATE["served"] == s0["served"] + 2 and core.STATE["fallback"] == s0["fallback"] and core.STATE["arrays"] == s0["arrays"] + 3 and core.STATE["stock"] == s0["stock"]
    text = json.dumps(full, indent=4, cls=fast)
    assert text.startswith('{\n    "plddt": [\n        ') and text.endswith("\n        ]\n    ]\n}") and "\n" + " " * 12 in text and not text.endswith("\n")


def test_errors_are_the_stock_encoders_own(patched_classes):
    """What the stock encoder refuses, the patched class refuses with the same exception type (the stock route answers the call): NaN under
    allow_nan=False, keys of an unsupported type, mixed-type keys under sort_keys, a circular reference, an object default() rejects."""
    loop = []
    loop.append(loop)
    cases = [(dict(indent=4, allow_nan=False), {"x": np.array([1.0, np.nan])}), (dict(indent=4, allow_nan=False), [float("inf")]), (dict(indent=4), {(1, 2): 3}),
             (dict(indent=4, sort_keys=True), {1: "a", "b": 2}), (dict(indent=4), loop), (dict(indent=4), {"o": object()}), (dict(indent=4), {"c": complex(1, 2)})]
    for name, (base, fast) in patched_classes.items():
        for kw, payload in cases:
            with pytest.raises(Exception) as w:
                json.dumps(payload, cls=base, **kw)
            with pytest.raises(Exception) as g:
                json.dumps(payload, cls=fast, **kw)
            assert type(g.value) is type(w.value), (name, kw, type(w.value), type(g.value))


def test_selftest_holds_for_both_engine_conversions(patched_classes):
    for name, (base, fast) in patched_classes.items():
        assert core.selftest(fast, base) is None, name
    s0 = {k: core.STATE[k] for k in ("served", "fallback", "bytes")}

    class Trailing(NumpyEncoder):                          # a base whose encode() adds to the text: the port does not reproduce it — the self-test says so
        def encode(self, o):
            return json.JSONEncoder.encode(self, o) + "\n"
    assert core.selftest(core.make_encoder_class(Trailing), Trailing) == "selftest_mismatch"
    assert {k: core.STATE[k] for k in ("served", "fallback", "bytes")} == s0     # the probe's own calls never reach the census


def _standin_writer(name):
    mod = types.ModuleType(name)
    mod.json = json
    mod.NumpyEncoder = type("NumpyEncoder", (NumpyEncoder,), {})

    def write_confidence_scores(payload, indent=4):        # the engine's call form (writer.py:191-207)
        return mod.json.dumps(payload, indent=indent, cls=mod.NumpyEncoder)
    mod.write_confidence_scores = write_confidence_scores
    return mod


@pytest.fixture()
def bound(monkeypatch):
    """The core bound to a stand-in writer module name for one test; the kit binding and the record restored after."""
    keep_cfg = {k: getattr(core, k) for k in core.CONFIGURABLE}
    keep_state = {k: (dict(v) if isinstance(v, dict) else v) for k, v in core.STATE.items()}
    name = "of3opt_standin_writer_%d" % id(object())
    core.configure(M_WRITER=name)
    core.STATE.update(installed=False, state="off", reason="", patched=0, served=0, stock=0, fallback=0, fallback_by={}, arrays=0, bytes=0, seconds=0.0)
    yield name
    core.uninstall()
    sys.modules.pop(name, None)
    core.configure(**keep_cfg)
    core.STATE.clear()
    core.STATE.update(keep_state)


def test_install_patches_an_imported_writer_and_serves_its_calls(bound, capsys, monkeypatch):
    monkeypatch.setitem(sys.modules, "openfold3_opt.cells.fastjson", fastjson)                             # the probe reads the cell module of THIS package instance (another test may have re-imported the package)
    mod = _standin_writer(bound)
    sys.modules[bound] = mod
    stock_cls = mod.NumpyEncoder
    aggregated, full = _writer_payloads()
    want_full, want_agg = mod.write_confidence_scores(full), mod.write_confidence_scores(aggregated)
    st = core.install({"OPENFOLD3_OPT_FASTJSON": "1"})
    assert st["installed"] and st["state"] == "on" and st["patched"] == 1 and st["route"] == "fast" and st["selftest"] == "ok"
    assert mod.NumpyEncoder is not stock_cls and getattr(mod.NumpyEncoder, core.MARK) is stock_cls and issubclass(mod.NumpyEncoder, stock_cls)
    assert mod.write_confidence_scores(full) == want_full and mod.write_confidence_scores(aggregated) == want_agg
    assert core.STATE["served"] == 2 and core.STATE["fallback"] == 0 and core.STATE["arrays"] == 3 and core.STATE["bytes"] == len(want_full) + len(want_agg)
    line = core.census_line()
    assert line.startswith("[openfold3-opt/fastjson] LEVER name=fastjson state=on route=fast patched=1 served=2 fallback=0 stock=0 arrays=3 bytes=%d s=" % (len(want_full) + len(want_agg)))
    assert "installed: %s.NumpyEncoder -> NumpyEncoder_fastjson" % bound in capsys.readouterr().err
    assert core.install({"OPENFOLD3_OPT_FASTJSON": "1"}) is st and mod.NumpyEncoder.__mro__[1] is stock_cls          # idempotent: one subclass, never stacked
    assert stack._PROBES["fastjson"](_stubs.tree_home()) is True and fastjson.serving()


def test_install_before_the_writers_import_arms_a_finder_that_patches_it_at_import(bound, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "openfold3_opt.cells.fastjson", fastjson)
    pkg_dir = tmp_path / "standin_pkg"
    pkg_dir.mkdir()
    (pkg_dir / (bound + ".py")).write_text(
        "import json\nimport numpy as np\n\n\nclass NumpyEncoder(json.JSONEncoder):\n    def default(self, obj):\n        if isinstance(obj, np.ndarray):\n"
        "            return obj.tolist()\n        elif isinstance(obj, np.generic):\n            return obj.item()\n        return super().default(obj)\n\n\n"
        "def write(payload):\n    return json.dumps(payload, indent=4, cls=NumpyEncoder)\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(pkg_dir))
    assert bound not in sys.modules
    st = core.install({"OPENFOLD3_OPT_FASTJSON": "1", "OPENFOLD3_OPT_FASTJSON_ROUTE": "fast"})
    assert st["state"] == "armed" and st["patched"] == 0 and core.FINDER in sys.meta_path
    assert stack._PROBES["fastjson"](_stubs.tree_home()) in (None, False) and not fastjson.serving()      # armed is not serving: the record says so until the writer is met
    import importlib
    mod = importlib.import_module(bound)
    assert st["state"] == "on" and st["patched"] == 1 and getattr(mod.NumpyEncoder, core.MARK, None) is not None
    _, full = _writer_payloads(n_tok=9, n_atom=30)
    assert mod.write(full) == json.dumps(full, indent=4, cls=getattr(mod.NumpyEncoder, core.MARK))
    assert st["served"] == 1 and st["arrays"] == 3


def test_route_stock_keeps_the_class_installed_and_answers_with_the_stock_encoder(bound):
    mod = _standin_writer(bound)
    sys.modules[bound] = mod
    _, full = _writer_payloads(n_tok=11, n_atom=50)
    want = mod.write_confidence_scores(full)
    st = core.install({"OPENFOLD3_OPT_FASTJSON": "1", "OPENFOLD3_OPT_FASTJSON_ROUTE": "stock"})
    assert st["state"] == "on" and st["route"] == "stock" and st["patched"] == 1
    assert mod.write_confidence_scores(full) == want
    assert st["served"] == 0 and st["stock"] == 1 and st["fallback"] == 0 and " route=stock patched=1 served=0 fallback=0 stock=1 " in core.census_line()


def test_switch_words_the_cell_does_not_know_are_refused_by_name(bound):
    assert core.install({}) is core.STATE and not core.STATE["installed"]                                  # not requested: nothing installed, nothing armed
    assert core.install({"OPENFOLD3_OPT_FASTJSON": ""})["installed"] is False
    with pytest.raises(ValueError, match="OPENFOLD3_OPT_FASTJSON='yes' is not one of 1"):
        core.requested({"OPENFOLD3_OPT_FASTJSON": "yes"})
    with pytest.raises(ValueError, match="OPENFOLD3_OPT_FASTJSON_ROUTE='quick' is not one of fast|stock"):
        core.route({"OPENFOLD3_OPT_FASTJSON_ROUTE": "quick"})
    st = core.install({"OPENFOLD3_OPT_FASTJSON": "1", "OPENFOLD3_OPT_FASTJSON_ROUTE": "quick"})       # at activation the lever steps aside by name, the writer untouched — never a crash
    assert st["installed"] and st["state"] == "refused" and st["reason"] == "bad_word:OPENFOLD3_OPT_FASTJSON_ROUTE" and st["patched"] == 0 and core.FINDER not in sys.meta_path
    assert " state=refused reason=bad_word:OPENFOLD3_OPT_FASTJSON_ROUTE " in core.census_line()
    core.STATE.update(installed=False, state="off", reason="")
    mod = types.ModuleType(bound)                                                                          # a writer module without the encoder class: refused by name, the module untouched
    sys.modules[bound] = mod
    st = core.install({"OPENFOLD3_OPT_FASTJSON": "1"})
    assert st["installed"] and st["state"] == "refused" and st["reason"] == "no_encoder:%s.NumpyEncoder" % bound and st["patched"] == 0
    assert " state=refused reason=no_encoder:" in core.census_line() and not fastjson.serving()


def test_the_kit_binding_names_this_engines_writer():
    assert (core.ENV, core.ENV_ROUTE, core.M_WRITER, core.ENCODER_ATTR, core.PREFIX) == ("OPENFOLD3_OPT_FASTJSON", "OPENFOLD3_OPT_FASTJSON_ROUTE", "openfold3.core.runners.writer", "NumpyEncoder", "[openfold3-opt/fastjson]")
    assert fastjson.ENV == core.ENV and fastjson.VALUES == ("1",) and fastjson.ROUTES == ("fast", "stock") and registry.T_WRITER == core.M_WRITER
    src = open(os.path.join(_stubs.tree_home(), "stock", "src", "openfold3", "core", "runners", "writer.py"), encoding="utf-8").read()   # the pinned engine's writer: the class and the call form the cell binds
    assert "class NumpyEncoder(json.JSONEncoder):" in src and src.count("cls=NumpyEncoder") == 2 and "indent=4" in src and "return obj.tolist()" in src


def test_every_kit_line_carries_the_lever_and_the_registry_row_names_it():
    for key, ln in modes.LINES.items():
        if key[0] == "off":
            assert "OPENFOLD3_OPT_FASTJSON" not in ln.env and "fastjson" not in ln.levers
            continue
        assert ln.env.get("OPENFOLD3_OPT_FASTJSON") == "1" and "fastjson" in ln.levers and "OPENFOLD3_OPT_FASTJSON_ROUTE" not in ln.env, key   # the route knob is the caller's, never a line's
    lv = registry.LEVERS["fastjson"]
    assert (lv.kit, lv.cls, lv.tier, lv.env_keys, lv.target) == ("cells", registry.OUTPUT, registry.EXACT, modes.FASTJSON_ENVS, registry.T_WRITER)
    assert set(lv.modes) == {"exact", "fast", "big/resident", "big/tp"} and lv.marker == "[openfold3-opt/fastjson] installed" and lv.source == "opt/openfold3_opt/cells/fastjson.py"
    assert registry.tested_sm("fastjson") == ("sm80", "sm90") and "fastjson" in registry.ARCH_NOTES   # served on both cards (the writer runs wherever the kit does)
    assert modes.FASTJSON_ENVS in modes.CELL_FAMILIES and "fastjson" in modes.ACTIVATION_CELLS and "fastjson" in stack._PROBES


def test_the_route_knob_is_the_callers_on_every_kit_line_and_foreign_to_none():
    home = _stubs.tree_home()
    for mode, n_gpu, extra in (("exact", None, {}), ("fast", None, {}), ("big", None, {}), ("big", "2", {"OF3TP_RANK": "0", "OF3TP_WORLD": "2"})):
        res = modes.resolve(mode, home, environ={"PATH": "/bin", "OPENFOLD3_OPT_FASTJSON_ROUTE": "stock", **extra}, n_tokens=612 if mode != "big" else 2565, n_gpu=n_gpu)
        assert res.conflicts == [], (mode, n_gpu, res.conflicts)
        assert res.exports.get("OPENFOLD3_OPT_FASTJSON") == "1"
    res = modes.resolve("fast", home, environ={"PATH": "/bin", "OPENFOLD3_OPT_FASTJSON": "0"}, n_tokens=612)          # the arming switch is the line's: another value preset contradicts it, named
    assert any("OPENFOLD3_OPT_FASTJSON='0' preset" in c for c in res.conflicts), res.conflicts
