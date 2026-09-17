import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
OPT = os.path.dirname(os.path.dirname(HERE))          # progen2/opt
TREE = os.path.dirname(OPT)                            # progen2/

# The pins gate passed, as one line of Python: the CPU test box is not the pinned stack and the fake stocks differ from the pins, so the
# tests that reach activation replace `stack.pins_gate` — in this process (`gate_ok`) or in a child interpreter (`run_package(gate_ok=True)`).
GATE_OK_PRELUDE = ("import progen2_opt.stack as _s; "
                   "_s.pins_gate = lambda p: (True, {'have': _s.stack_versions(), 'want': {}, 'mismatch': [], 'files': {}}, None)")


@pytest.fixture(scope="session")
def tree():
    return TREE


@pytest.fixture(autouse=True)
def home(monkeypatch):
    """`stack.package_home()` pinned to this tree's progen2/opt (the kit dirs and SUMS the tests read); returns that path."""
    for k in list(os.environ):
        if k.startswith("PROGEN2_"):
            monkeypatch.delenv(k, raising=False)
    from progen2_opt import stack
    monkeypatch.setattr(stack, "package_home", lambda: OPT)
    stack._STATE.update({"report": None})
    yield OPT


@pytest.fixture
def gate_ok(monkeypatch):
    """`stack.pins_gate` passed in this process (stack + stock files as pinned)."""
    from progen2_opt import stack
    monkeypatch.setattr(stack, "pins_gate", lambda p: (True, {"have": stack.stack_versions(), "want": {}, "mismatch": [], "files": {}}, None))


@pytest.fixture
def fake_stock(tmp_path):
    """A fake stock checkout: the stock model module + main scripts, the way the real ones import (`from models.progen.modeling_progen ...`)."""
    d = tmp_path / "stock"
    (d / "models" / "progen").mkdir(parents=True)
    (d / "models" / "__init__.py").write_text("")
    (d / "models" / "progen" / "__init__.py").write_text("")
    (d / "models" / "progen" / "modeling_progen.py").write_text("class ProGenForCausalLM:\n    def __init__(self):\n        pass\n    def to(self, *a, **k):\n        return self\n")
    body = 'from models.progen.modeling_progen import ProGenForCausalLM\nm = ProGenForCausalLM().to("cuda:0")\nprint("stock main continued")\n'
    for n in ("likelihood.py", "sample.py"):
        (d / n).write_text(body)
    return d


def subprocess_env(extra=None, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if not k.startswith("PROGEN2_")}
    if pythonpath:
        env["PYTHONPATH"] = str(pythonpath)
    env.update(extra or {})
    return env


def run_package(args, extra_env=None, gate_ok=False):
    """`python -m progen2_opt <args>` in a fresh interpreter (cli.main on the same argv); `gate_ok` patches the pins gate in that child."""
    code = (GATE_OK_PRELUDE + "\n" if gate_ok else "") + "import sys; from progen2_opt.cli import main; sys.exit(main(sys.argv[1:]))"
    r = subprocess.run([sys.executable, "-c", code, *args], env=subprocess_env(extra_env), capture_output=True, text=True)
    return r.returncode, r.stdout + r.stderr, r
