"""Out-of-memory errors propagate; no fallback is applied.

The borzoi kit's served path carries NO broad `except` that continues the run on a substitute route: the forward call the `exact`
composition installs (`v17/kitlib/forward.py` `ForwardCall`, `KIT_FWD=1`) wraps the stock's Keras call (eager on the first call,
one traced graph after it) with no handler at all, so a `tf.errors.ResourceExhaustedError` (or anything else) raised by the model call is the process's own failure. This test
proves it through that served wrapper with the innermost callable — the model — mocked to raise a TensorFlow-named
`ResourceExhaustedError`, and locks the census: the served modules' broad handlers are exactly the audited set (an import-time
availability guard, the host cores probe's named `probe_ok=false` records that the package's partial-activation gate refuses, a
reader's `close`) — a new broad handler on the served path fails here by name. CPU only; TensorFlow is not imported.
"""
import ast
import importlib.util
import os

import numpy as np
import pytest

from ._fixtures import REAL_KIT

FORWARD = os.path.join(REAL_KIT, "v17", "kitlib", "forward.py")
SERVED_DIRS = (os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),           # opt/borzoi_opt (the package)
               os.path.join(REAL_KIT, "v17"), os.path.join(REAL_KIT, "v17", "kitlib"))                # the kit
# (relative file, enclosing function) of every broad handler on the served path, each read and classified below:
AUDITED_BROAD_HANDLERS = {
    ("v17/kitlib/onehot.py", "<module>"),               # baskerville absent at import: the delegated paths then raise by name (onehot.py dna_1hot)
    ("v17/kitlib/sad_post.py", "probe_effective_cores"),  # the host cores probe: probe_ok=false, named; the package's gate: partial, exit 3
    ("v17/kitlib/sad_post.py", "start_probe_async"),
    ("v17/kitlib/sad_post.py", "join_probe"),             # two handlers (kill on a failed join; the same named record)
}


class ResourceExhaustedError(Exception):
    """Stand-in for tf.errors.ResourceExhaustedError (that class name, a tensorflow module)."""
    __module__ = "tensorflow.python.framework.errors_impl"


def _load_forward():
    spec = importlib.util.spec_from_file_location("kitlib_forward_under_test", FORWARD)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class _Model:
    """seqnn_model stand-in: `ensemble` None, `model` the Keras callable — here one that raises what the BFC allocator raises."""
    ensemble = None

    def __init__(self, exc):
        self._exc = exc

    def model(self, x):
        raise self._exc


@pytest.mark.parametrize("exc", [ResourceExhaustedError("OOM when allocating tensor with shape[1,524288,1536]"), MemoryError(),
                                 ValueError("any other failure of the model call")])
def test_forward_call_has_no_fallback(monkeypatch, exc):
    monkeypatch.setenv("KIT_FWD", "1")
    fwd = _load_forward()
    wrapped, stamp = fwd.install(_Model(exc))
    assert stamp["enabled"] and stamp["kit_fwd"] == "graph_copy_free"
    with pytest.raises(type(exc)):
        wrapped(np.zeros((1, 8, 4), dtype="float32"))
    assert stamp["n_calls"] == 0 and stamp["n_stock_calls"] == 0 and stamp["n_graph_calls"] == 0   # nothing was counted as done, nothing rerouted


def _broad_handlers(path):
    tree = ast.parse(open(path, encoding="utf-8").read(), path)
    out = []

    def walk(node, fn):
        for child in ast.iter_child_nodes(node):
            name = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else fn
            if isinstance(child, ast.ExceptHandler):
                t = child.type
                names = ([] if t is None else [t.id] if isinstance(t, ast.Name) else
                         [e.id for e in t.elts if isinstance(e, ast.Name)] if isinstance(t, ast.Tuple) else ["?"])
                if t is None or {"Exception", "BaseException", "RuntimeError"} & set(names):
                    out.append(fn)
            walk(child, name)
    walk(tree, "<module>")
    return out


def test_served_broad_handlers_are_the_audited_set():
    found = set()
    for d in SERVED_DIRS:
        for f in sorted(os.listdir(d)):
            if f.endswith(".py"):
                rel = os.path.relpath(os.path.join(d, f), REAL_KIT) if d.startswith(REAL_KIT) else "borzoi_opt/" + f
                for fn in _broad_handlers(os.path.join(d, f)):
                    found.add((rel, fn))
    assert found == AUDITED_BROAD_HANDLERS, sorted(found ^ AUDITED_BROAD_HANDLERS)
