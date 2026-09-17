"""A model the kit's kernels are not built for (not on a CUDA device / parameters not float32) is served by upstream's own forward under an
active mode — said once on one ASIDE line, never refused (no NOT ACTIVE, no exit 3); a later .to()/.float() makes the next forward look again."""
import types
import pytest

torch = pytest.importorskip("torch")
from flashzoi_opt import stack, report


class _Dev:
    def __init__(self, t): self.type = t
    def __str__(self): return self.type


def _active(monkeypatch):
    monkeypatch.setattr(stack, "_REPORT", {"active": True, "mode": "exact", "resolution": None, "models": []})
    monkeypatch.setattr(stack, "_HOOK", dict(stack._HOOK, original=lambda self, x, *a, **k: ("upstream forward", tuple(x.shape))))
    stack._ASIDE.clear()


def test_cpu_model_is_served_aside_named_once(monkeypatch, capsys):
    _active(monkeypatch)
    m = torch.nn.Linear(2, 2)                                                        # parameters on the CPU
    rec = stack.apply_to(m, trigger="forward")
    assert rec["trigger"] == "aside" and "device cpu" in rec["aside"] and rec["components_applied"] == []
    assert m.forward(torch.zeros(1, 2)) == ("upstream forward", (1, 2))                  # upstream's own forward bound on the instance: the hook is not re-entered
    stack.apply_to(m, trigger="forward")                                             # a second look says nothing new
    err = capsys.readouterr().err + capsys.readouterr().out
    assert err.count("ASIDE model=0:") == 1 and "NOT ACTIVE" not in err, err
    assert report.aside_line(0, "x").startswith("[flashzoi-opt] ASIDE model=0: x")


def test_non_float32_model_is_served_aside_and_a_move_makes_the_next_forward_look_again(monkeypatch, capsys):
    _active(monkeypatch)
    monkeypatch.setattr(stack, "_model_device", lambda model: _Dev("cuda"))
    m = torch.nn.Linear(2, 2).half()
    rec = stack.apply_to(m)
    assert rec["trigger"] == "aside" and "float16" in rec["aside"] and "forward" in m.__dict__
    m.float()                                                                        # the watched move: the instance forward is dropped so the next forward re-enters the hook
    assert "forward" not in m.__dict__ and id(m) not in stack._ASIDE and m.weight.dtype == torch.float32
    assert stack._not_for_the_kit(m) is None                                           # a CUDA float32 model now (by the patched device): the kit would attach at the next forward
