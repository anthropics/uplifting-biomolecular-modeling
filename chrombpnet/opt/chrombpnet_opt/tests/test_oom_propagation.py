"""Out-of-memory errors propagate; no fallback is applied.

On the fast mode's line every broad `except` that continues the run on a substitute route (the TF route for a failed K1 import, the
unknown-class / class-table route for a failed probe, the stock JIT for a refused shipped cache, the stock module for a failed writer
source check …) re-raises an out-of-memory error FIRST — `if is_oom(e): raise` — through the kit's one classifier
(`chrombpnet_fastkit/_oom.py`, carried byte-identically in `opt/kit_ho/tf` and, for the torch side,
`opt/kit/torch/chrombpnet_k1/_oom.py`). This test enters the served route decision — the kit's `fastdefault` module loaded by path
the way the package's `check` and the entry script load it, `resolve()` the entry script's own call — with the innermost callable,
the `nvidia-smi` subprocess, mocked: a torch-named `OutOfMemoryError`, a TensorFlow-named `ResourceExhaustedError` and `MemoryError`
propagate by name; a `FileNotFoundError` (no nvidia-smi) takes the existing named route (`nvidia-smi unavailable (…)`: the unknown
class, never the K1 forward). Plus the classifier on framework-named stand-ins and the three carried copies byte-identical. CPU only: no
TensorFlow, torch or GPU (the torch probe reports the stack missing, itself a named route)."""
import importlib.util
import os
import unittest
from unittest import mock

from chrombpnet_opt import stack

OPT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))          # opt/
COPIES = [os.path.join(OPT, "kit_ho", "tf", "chrombpnet_fastkit", "_oom.py"), os.path.join(OPT, "kit", "torch", "chrombpnet_k1", "_oom.py")]


class OutOfMemoryError(RuntimeError):
    """Stand-in for torch.OutOfMemoryError (that class name, module `torch`)."""
    __module__ = "torch"


class ResourceExhaustedError(Exception):
    """Stand-in for tf.errors.ResourceExhaustedError (that class name, a tensorflow module)."""
    __module__ = "tensorflow.python.framework.errors_impl"


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestRouteDecisionPropagatesOom(unittest.TestCase):
    """fastdefault.resolve() — the route decision the fast mode's line makes first (pred_bw_fast.py `_fd.resolve(...)`)."""

    def setUp(self):
        self.kit_ho = stack.kit_home()                                   # opt/kit_ho: the fast mode's kit directory
        self.kit = stack.kit_tables_root(self.kit_ho)                    # opt/kit: the carried kit (the torch side, the tables)
        self.assertTrue(os.path.isdir(os.path.join(self.kit, "torch", "chrombpnet_k1")), self.kit)

    def _resolve_with(self, exc):
        fd = stack.fastdefault(self.kit_ho)                             # a fresh module each call (its nvidia-smi answer is memoized per module)
        env = {k: v for k, v in os.environ.items() if not k.startswith("CHROMBPNET_")}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.object(fd.subprocess, "run", side_effect=exc):
            return fd.resolve(kit_root=self.kit)

    def test_out_of_memory_propagates(self):
        for exc in (OutOfMemoryError("CUDA out of memory."), ResourceExhaustedError("OOM when allocating tensor"), MemoryError()):
            with self.subTest(exc=type(exc).__name__), self.assertRaises(type(exc)):
                self._resolve_with(exc)

    def test_other_failure_keeps_the_named_route(self):
        r = self._resolve_with(FileNotFoundError("nvidia-smi"))
        self.assertEqual(r["gpu_class"], "unknown")
        self.assertIn("nvidia-smi unavailable (", r["gpu_name"])
        self.assertNotEqual(r["forward"], "k1")                        # the TF-side route, said on the line — never K1 on an unknown class


class TestClassifier(unittest.TestCase):
    def setUp(self):
        self.is_oom = _load(COPIES[0], "chrombpnet_fastkit_oom_under_test").is_oom

    def test_the_carried_copies_are_one_text(self):
        texts = [open(p, "rb").read() for p in COPIES]
        self.assertTrue(all(t == texts[0] for t in texts), [p for p, t in zip(COPIES, texts) if t != texts[0]])

    def test_out_of_memory_spellings(self):
        cuda_oom = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch.cuda"})
        xla = type("XlaRuntimeError", (RuntimeError,), {"__module__": "jaxlib.xla_extension"})
        wrapped = ImportError("the K1 stack failed to import"); wrapped.__cause__ = cuda_oom("CUDA out of memory")
        during = KeyError("raised while handling"); during.__context__ = ResourceExhaustedError("OOM")
        for e in (OutOfMemoryError("x"), cuda_oom("x"), ResourceExhaustedError("OOM when allocating tensor"), MemoryError(),
                  RuntimeError("CUDA out of memory. Tried to allocate a block"), RuntimeError("CUDA error: out of memory"),
                  xla("RESOURCE_EXHAUSTED: Out of memory while trying to allocate"), wrapped, during):
            self.assertTrue(self.is_oom(e), repr(e))

    def test_everything_else_is_not(self):
        xla = type("XlaRuntimeError", (RuntimeError,), {"__module__": "jaxlib.xla_extension"})
        two_deep = ValueError("top"); two_deep.__cause__ = ValueError("mid"); two_deep.__cause__.__cause__ = MemoryError()
        for e in (RuntimeError("boom"), ValueError("out of memory?"), OSError(12, "Cannot allocate memory"), FileNotFoundError("nvidia-smi"),
                  xla("INTERNAL: ptxas failed"), two_deep, KeyboardInterrupt(), None):
            self.assertFalse(self.is_oom(e), repr(e))


if __name__ == "__main__":
    unittest.main()
