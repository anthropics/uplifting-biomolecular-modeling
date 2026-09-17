"""The model-instance count behind the late-activation rule: a constructor wrap on stock's Protenix class, registered at activation —
on the class when its module is already imported (existing instances found once by a gc scan), at the module's import otherwise (a
meta-path finder) — and the gc scan only when the class cannot be wrapped. The activation report notes that fallback."""
import gc
import importlib
import inspect
import io
import os
import sys
import textwrap
import types
from contextlib import redirect_stderr
from unittest import mock

import pytest

import protenix_opt
from protenix_opt import stack
from opt_core import instances

H100 = {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90", "cc": "9.0", "probe": "nvidia-smi"}
MARKERS = ["T1:fused", "BLK:2(pro+cueq+epi+fusion_transition)", "DEADSKIP:hooked", "LAZY_INIT:1"]


def _entry() -> dict:
    """The core's counter entry for the model class under test (opt_core.instances: {"state", "finder"})."""
    return instances._entry(stack.MODEL_MODULE, stack.MODEL_CLASS)


def _state() -> dict:
    return _entry()["state"]


class _Frozen(type):
    def __setattr__(cls, name, value):
        raise TypeError("frozen class")


@pytest.fixture
def counter(monkeypatch):
    """A fresh counter on a test-private module name; activation state and the meta path restored afterwards."""
    name = "_ptx_counter_test_module"
    monkeypatch.setattr(stack, "MODEL_MODULE", name)
    monkeypatch.setattr(instances, "_COUNTERS", {})
    monkeypatch.setattr(stack, "_REPORT", None)
    saved = list(sys.meta_path)
    yield name
    sys.meta_path[:] = saved
    sys.modules.pop(name, None)


def _module(name, cls):
    mod = types.ModuleType(name)
    setattr(mod, stack.MODEL_CLASS, cls)
    return mod


class Model:
    def __init__(self, configs, *inputs, **kwargs):
        self.configs = configs


def test_not_imported_means_no_instances(counter):
    assert stack.instance_check() == {"n": 0, "method": "none", "built": 0}
    assert stack.instance_check()["n"] == 0


def test_counted_path_on_an_imported_class(counter):
    cls = type("Protenix", (Model,), {})
    sys.modules[counter] = _module(counter, cls)
    before = cls("pre")                                                   # exists before the counter: found once by the gc scan
    sig = inspect.signature(cls)
    assert stack.register_instance_counter() == {"n": 1, "method": "counted", "built": 0}
    assert _state()["gc_seeded"] == 1
    assert inspect.signature(cls) == sig, "the class signature is kept"
    a = cls("a"); b = cls("b", 1, x=2)
    assert stack.instance_check() == {"n": 3, "method": "counted", "built": 2}
    assert (a.configs, b.configs) == ("a", "b")
    del a, before
    assert stack.instance_check()["n"] == 1, "no gc pass needed: a released instance leaves the count at once"
    stack.register_instance_counter()                                     # idempotent: one wrap per class
    cls("c")
    assert stack.instance_check()["built"] == 3 and stack.instance_check()["n"] == 1
    del b


def test_finder_path_wraps_the_class_at_its_import(counter, tmp_path):
    (tmp_path / f"{counter}.py").write_text(textwrap.dedent(f'''
        class {stack.MODEL_CLASS}:
            def __init__(self, configs):
                self.configs = configs
    '''))
    sys.path.insert(0, str(tmp_path))
    try:
        assert stack.register_instance_counter() == {"n": 0, "method": "none", "built": 0}
        assert isinstance(sys.meta_path[0], instances._ClassFinder) and _entry()["finder"] is sys.meta_path[0]
        stack.register_instance_counter()                                 # idempotent: one finder
        assert sum(isinstance(f, instances._ClassFinder) for f in sys.meta_path) == 1
        mod = importlib.import_module(counter)                            # the import happens on the program's own schedule
        assert not any(isinstance(f, instances._ClassFinder) for f in sys.meta_path), "the finder fires once and removes itself"
        cls = getattr(mod, stack.MODEL_CLASS)
        assert _state()["cls"] is cls
        assert stack.instance_check() == {"n": 0, "method": "counted", "built": 0}
        m = cls("cfg")
        assert stack.instance_check() == {"n": 1, "method": "counted", "built": 1} and m.configs == "cfg"
        del m
        assert stack.instance_check()["n"] == 0
    finally:
        sys.path.remove(str(tmp_path))


def test_gc_scan_when_the_class_cannot_be_wrapped(counter):
    cls = _Frozen(stack.MODEL_CLASS, (), {"__init__": lambda self, configs=None: None})
    sys.modules[counter] = _module(counter, cls)
    assert stack.register_instance_counter() == {"n": 0, "method": "gc", "built": 0}
    assert _state()["wrapped"] is False
    m = cls()
    assert stack.instance_check() == {"n": 1, "method": "gc", "built": 0}
    del m
    gc.collect()
    assert stack.instance_check()["n"] == 0


# ------------------------------------------------------------------------------------------------ through enable() ----
def _protenix_stubs(model_path=None):
    protenix = types.ModuleType("protenix"); protenix.__path__ = []
    model = types.ModuleType("protenix.model"); model.__path__ = [model_path] if model_path else []
    return {"protenix": protenix, "protenix.model": model}


def _enable():
    apply = mock.Mock(return_value=(list(MARKERS), {}))
    with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(stack, "gpu_probe_smi", return_value=H100), \
         mock.patch.object(stack, "gpu_info", return_value=H100), mock.patch.object(stack, "_apply", apply), redirect_stderr(io.StringIO()):
        return protenix_opt.enable("exact")


@pytest.fixture
def activation(monkeypatch):
    monkeypatch.setattr(instances, "_COUNTERS", {})
    monkeypatch.setattr(stack, "_REPORT", None)
    monkeypatch.delenv("PROTENIX_OPT", raising=False)
    monkeypatch.delenv("PROTENIX_OPT_FORCE", raising=False)
    saved_env, saved_path = dict(os.environ), list(sys.meta_path)
    yield
    sys.meta_path[:] = saved_path
    os.environ.clear(); os.environ.update(saved_env)
    stack._REPORT = None


def test_activation_counts_from_the_model_import_on(activation, tmp_path):
    """enable() before protenix.model.protenix is imported: the finder wraps the class at its import; a later instance is counted."""
    (tmp_path / "protenix.py").write_text("class Protenix:\n    def __init__(self, configs):\n        self.configs = configs\n")
    mods = _protenix_stubs(str(tmp_path))
    with mock.patch.dict(sys.modules, mods):
        sys.modules.pop(stack.MODEL_MODULE, None)
        r = _enable()
        assert r["active"] and not [n for n in r["notes"] if "gc scan" in n]
        assert any(isinstance(f, instances._ClassFinder) for f in sys.meta_path)
        pm = importlib.import_module(stack.MODEL_MODULE)
        assert not any(isinstance(f, instances._ClassFinder) for f in sys.meta_path)
        assert stack.instance_check() == {"n": 0, "method": "counted", "built": 0}
        m = pm.Protenix({"a": 1})
        assert stack.instance_check() == {"n": 1, "method": "counted", "built": 1} and m.configs == {"a": 1}
        del m
        assert stack.instance_check() == {"n": 0, "method": "counted", "built": 1}


def test_activation_refusal_names_the_method_and_the_gc_fallback_is_noted(activation):
    mods = _protenix_stubs()
    cls = _Frozen("Protenix", (), {"__init__": lambda self, configs=None: None})          # a class the wrap cannot touch
    mods[stack.MODEL_MODULE] = _module(stack.MODEL_MODULE, cls)
    instance = cls()
    with mock.patch.dict(sys.modules, mods):
        r = _enable()
        assert not r["active"] and "a Protenix model instance already exists in this process (1 live, gc)" in r["reason"]
        assert stack.instance_check() == {"n": 1, "method": "gc", "built": 0}
        del instance
        gc.collect()
        stack._REPORT = None
        r = _enable()
        assert r["active"]
        assert "Protenix instances are counted by a gc scan: the class could not be wrapped" in r["notes"]


def test_activation_refusal_on_a_counted_instance(activation):
    mods = _protenix_stubs()
    cls = type("Protenix", (Model,), {})
    mods[stack.MODEL_MODULE] = _module(stack.MODEL_MODULE, cls)
    instance = cls({})
    with mock.patch.dict(sys.modules, mods):
        r = _enable()
        assert not r["active"] and "a Protenix model instance already exists in this process (1 live, counted)" in r["reason"]
        del instance
        stack._REPORT = None
        r = _enable()
        assert r["active"] and not [n for n in r["notes"] if "gc scan" in n]
        cls({})
        assert stack.instance_check()["built"] == 1
