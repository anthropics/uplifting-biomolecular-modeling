"""Out-of-memory is the caller's: the kit's one classifier (esmc_opt/_oom.py, which the kits import too)
recognises torch's OutOfMemoryError (by class when torch is importable, by name otherwise), the 'CUDA out of memory' RuntimeError, the
host's MemoryError and a wrapper raised FROM one of them; the package's served apply path (stack._apply_model_lever / stack._apply_pipe —
the levers of every load) lets an out-of-memory raised inside a kit's apply() PROPAGATE, while any other exception keeps the named
route (the lever's record says applied False + refused '<Type>: <message>', the activation's PARTIAL rule decides the exit)."""
import importlib
import os
import types

import pytest

from esmc_opt import _oom, stack
from esmc_opt._oom import is_oom

from ._paths import KITS, PKG


class OutOfMemoryError(RuntimeError):
    """Stand-in with torch's class NAME (torch.OutOfMemoryError derives from RuntimeError): classified by name when torch is absent."""


class KitRefused(RuntimeError):
    pass


def _oom_classes():
    out = [OutOfMemoryError]
    try:
        import torch                                                   # noqa: F401 — the real class when this interpreter has torch (the CPU wheel carries it)
        for k in (getattr(torch, "OutOfMemoryError", None), getattr(torch.cuda, "OutOfMemoryError", None)):
            if isinstance(k, type) and k not in out:
                out.append(k)
    except ImportError:
        pass
    return out


@pytest.mark.parametrize("cls", _oom_classes(), ids=lambda c: f"{c.__module__}.{c.__qualname__}")
def test_is_oom_true_for_the_oom_classes(cls):
    assert is_oom(cls("CUDA out of memory. Tried to allocate more than the device holds"))
    assert is_oom(cls(""))                                             # by class, not only by message


def test_is_oom_true_for_messages_memoryerror_and_one_level_of_chaining():
    assert is_oom(RuntimeError("CUDA out of memory. Tried to allocate more than the device holds"))          # older torch: a plain RuntimeError
    assert is_oom(RuntimeError("CUDA error: out of memory"))
    assert is_oom(MemoryError())
    try:
        try:
            raise OutOfMemoryError("CUDA out of memory")
        except Exception as inner:                                     # a kit wrapper that re-raises by name keeps the cause
            raise KitRefused("prewarm failed") from inner
    except KitRefused as wrapped:
        assert is_oom(wrapped)
    XlaRuntimeError = type("XlaRuntimeError", (RuntimeError,), {})
    assert is_oom(XlaRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate"))
    assert not is_oom(XlaRuntimeError("INTERNAL: something else"))
    assert is_oom(type("ResourceExhaustedError", (Exception,), {})("OOM when allocating tensor"))


def test_is_oom_false_for_everything_else():
    for e in (KitRefused("kit v1: the census failed"), RuntimeError("shape mismatch"), ValueError("bad"), KeyError("k"), OSError("disk"),
              Exception("REFUSED: graph budget reached")):
        assert not is_oom(e), repr(e)


def test_one_classifier_one_home():
    """ONE spelling: the package's _oom.py; the kits import it (no second copy in the kits' dir)."""
    assert not os.path.exists(os.path.join(KITS, "oom.py"))
    assert _oom.__all__ == ["is_oom"]


def test_every_served_reroute_site_asks_is_oom_first():
    """The served-site census (each listed handler's first statement is `if is_oom(<e>): raise`)."""
    sites = {                                                          # file (relative to opt/) -> number of served reroute handlers in it
        "esmc_opt/stack.py": 2,
        "esmc_opt/kits/fused/__init__.py": 2,                       # the qk-ladder warm-up and the template-priming forward: both re-raise an OOM first
        "esmc_opt/kits/fused/_patch.py": 1,
        "esmc_opt/kits/fused/_hs_write.py": 1,
        "esmc_opt/kits/thin/thin_launch.py": 1,
    }
    opt = os.path.dirname(PKG)
    for rel, n in sites.items():
        lines = open(os.path.join(opt, rel), encoding="utf-8").read().splitlines()
        hits = [i for i, ln in enumerate(lines) if ln.strip().startswith("if is_oom(") and ln.split("#")[0].strip().endswith(": raise")]
        assert len(hits) == n, (rel, len(hits), n)
        for i in hits:                                                 # the statement right after an `except ... as <var>:` line (comment lines may sit between)
            j = i - 1
            while lines[j].strip().startswith("#") or not lines[j].strip():
                j -= 1
            assert lines[j].strip().startswith("except") and " as " in lines[j], (rel, i + 1, lines[j].strip()[:60])
        imports = [ln for ln in lines if ln.strip() == "from esmc_opt._oom import is_oom" or ln.strip().endswith("; from esmc_opt._oom import is_oom")]
        assert len(imports) == 1, (rel, imports)                       # ONE import line per file


class _FakeKit(types.ModuleType):
    def __init__(self, exc):
        super().__init__("fake_kit")
        self.exc = exc
        self.calls = 0

    def apply(self, *a, **kw):                                         # the innermost callable of the served apply path
        self.calls += 1
        raise self.exc

    def resolve_levers(self, route, wd):
        return ("imports", "boot", "tok"), True


@pytest.fixture
def fake_kit(monkeypatch):
    holder = {}

    def _install(exc):
        kit = _FakeKit(exc)
        monkeypatch.setattr(stack, "_kit", lambda name: kit)
        monkeypatch.setattr(stack, "weights_dir", lambda variant: "/nonexistent/weights")
        holder["kit"] = kit
        return kit
    return _install


@pytest.mark.parametrize("cls", _oom_classes(), ids=lambda c: f"{c.__module__}.{c.__qualname__}")
@pytest.mark.parametrize("lever", ["fused"])
def test_served_model_lever_apply_propagates_oom(fake_kit, cls, lever):
    kit = fake_kit(cls("CUDA out of memory. Tried to allocate more than the device holds"))
    with pytest.raises(cls):
        stack._apply_model_lever(lever, types.SimpleNamespace(model=object()), {"variant": "300m"})
    assert kit.calls == 1


@pytest.mark.parametrize("cls", _oom_classes(), ids=lambda c: f"{c.__module__}.{c.__qualname__}")
def test_served_pipe_apply_propagates_oom(fake_kit, cls):
    fake_kit(cls("CUDA out of memory."))
    with pytest.raises(cls):
        stack._apply_pipe({"variant": "300m"})


@pytest.mark.parametrize("lever", ["fused"])
def test_served_model_lever_apply_keeps_the_named_route_for_other_exceptions(fake_kit, lever):
    fake_kit(KitRefused("kit says no: not a served shape"))
    rec = stack._apply_model_lever(lever, types.SimpleNamespace(model=object()), {"variant": "300m"})
    assert rec["applied"] is False and rec["refused"].startswith("KitRefused: kit says no"), rec
    assert rec["lever"] == lever


def test_served_pipe_apply_keeps_the_named_route_for_other_exceptions(fake_kit):
    fake_kit(KitRefused("pins unreadable"))
    rec = stack._apply_pipe({"variant": "300m"})
    assert rec["applied"] is False and rec["refused"].startswith("KitRefused: pins unreadable"), rec
