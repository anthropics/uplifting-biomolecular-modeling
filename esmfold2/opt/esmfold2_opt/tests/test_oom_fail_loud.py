"""GPU out-of-memory is fail-loud in every served path: a fold that runs out of memory RAISES / is recorded as failed BY NAME — no retry on
the stock eager path, no seed-batch or cross-design-batch halving, no refusal or lever-disable re-wording
OOM detection is the release tree's one classifier `opt_core.oom.is_oom` (the kit pins opt_core; one helper, no kit-local
spelling). One test per site plus two served-entry-point propagation tests; CPU only (the callee is mocked to raise an OutOfMemoryError); the
carried drivers are read from the kit (`_stubs.require_kit`)."""
import ast
import glob
import importlib.util
import io
import json
import os
import sys
import threading
import unittest
from unittest import mock

from ._stubs import require_kit

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)


def _oom_class():
    """torch's OutOfMemoryError when this interpreter has torch, else a class of that NAME (what opt_core.oom.is_oom keys on) — no GPU either way."""
    try:
        import torch
        k = getattr(torch, "OutOfMemoryError", None) or getattr(torch.cuda, "OutOfMemoryError", None)
        if isinstance(k, type):
            return k
    except Exception:                                        # torch absent: the named class below is what the classifier recognises
        pass
    return type("OutOfMemoryError", (RuntimeError,), {})


def _is_oom_or_skip():
    try:
        from opt_core.oom import is_oom
    except ImportError as e:
        raise unittest.SkipTest(f"opt_core.oom not importable (opt_core < 0.5.8?): {e!r}")
    return is_oom


def _torch_or_skip():
    try:
        import torch
    except Exception as e:                                   # named skip, never a silent pass
        raise unittest.SkipTest(f"torch not importable on this interpreter: {e!r}")
    return torch, _oom_class()


def _driver_file(name):
    kit = require_kit()
    p = os.path.join(kit, "driver", name)
    if not os.path.isfile(p):
        raise unittest.SkipTest(f"{p} not carried")
    return p


SERVED = {                                                   # served-path modules: every broad `except` that reroutes around a lever / kernel launch re-raises OOM first
    "house": ("stack.py", "big.py", "cli.py", "rowpair.py", "rowpair_feats.py"),
    "driver": ("ef2_server.py", "ef2_opt.py", "ef2_w4.py"),
    "addon": ("EF2_XL_ADDON_v1/ef2_xl.py",),
}


class TestOneClassifier(unittest.TestCase):
    def test_the_core_classifier_recognises_torch_oom_memoryerror_and_the_cuda_text_only(self):
        is_oom = _is_oom_or_skip(); OOM = _oom_class()
        self.assertTrue(is_oom(OOM("CUDA out of memory. Tried to allocate 2.00 GiB")))
        self.assertTrue(is_oom(RuntimeError("CUDA out of memory. Tried to allocate 20.00 MiB")))
        self.assertTrue(is_oom(MemoryError("no tier can hold 41.0 GiB")))
        self.assertFalse(is_oom(RuntimeError("CUDA error: an illegal memory access was encountered")))
        self.assertFalse(is_oom(ValueError("out of memory")))

    def test_no_second_spelling_in_the_kit(self):
        """Every served module that classifies OOM imports `opt_core.oom.is_oom`; no module of the package or the carried tree defines its own."""
        kit = require_kit(); fwd = os.path.dirname(kit)
        files = glob.glob(os.path.join(PKG, "*.py")) + glob.glob(os.path.join(kit, "driver", "*.py")) + glob.glob(os.path.join(fwd, "EF2_*", "*.py"))
        self.assertTrue(files)
        users = 0
        for f in files:
            src = open(f, encoding="utf-8", errors="replace").read()
            self.assertNotRegex(src, r"(?m)^\s*def _?is_oom\(", f"{f} defines its own OOM classifier")
            if "is_oom(" in src:
                users += 1; self.assertIn("from opt_core.oom import is_oom", src, f)
        self.assertGreaterEqual(users, 7, users)                                        # stack, big, cli, rowpair, ef2_opt, ef2_w4, ef2_xl

    def test_every_rerouting_broad_handler_on_the_served_path_reraises_oom_first(self):
        """In the served modules, a broad `except` whose body disables a lever,
        refuses, or substitutes a stock route carries `if is_oom(e): raise` (or routes to the server's named OOM record) BEFORE that route."""
        kit = require_kit(); fwd = os.path.dirname(kit)
        paths = [os.path.join(PKG, f) for f in SERVED["house"]] + [os.path.join(kit, "driver", f) for f in SERVED["driver"]] + [os.path.join(fwd, f) for f in SERVED["addon"]]
        REROUTE = ("_disable_lever(", "ActivationError(", "apply_failed=True", '_TX["why"]', "stock layout", 'return {"active": False', "same = False")
        checked = 0
        for p in paths:
            src = open(p, encoding="utf-8").read(); tree = ast.parse(src)
            for node in ast.walk(tree):
                if not isinstance(node, ast.ExceptHandler) or node.type is None and False:
                    continue
                tsrc = ast.get_source_segment(src, node.type) if node.type is not None else ""
                if node.type is not None and "Exception" not in tsrc:
                    continue
                body = "\n".join(ast.get_source_segment(src, b) or "" for b in node.body)
                if not any(w in body for w in REROUTE):
                    continue
                if "reset_after_error" in body and "healthy = False" in body:                 # the CUDA health probe after a recorded failure: not a route
                    continue
                head = "\n".join(ast.get_source_segment(src, b) or "" for b in node.body[:3])
                self.assertTrue("if is_oom(e): raise" in head or "if is_oom(e):" in head and "_oom_rows(" in body,
                                f"{os.path.basename(p)}:{node.lineno}: a rerouting handler must re-raise (or record by name) a GPU out-of-memory error first:\n{head[:200]}")
                checked += 1
        self.assertGreaterEqual(checked, 9, checked)


class TestServedEntryPointsPropagateOom(unittest.TestCase):
    """A mocked OOM raised under a served entry point reaches the caller unchanged (no NOT-ACTIVE re-wording, no ActivationError)."""
    def test_cli_activate_propagates_oom(self):
        _is_oom_or_skip(); OOM = _oom_class()
        import esmfold2_opt
        from esmfold2_opt import cli
        with mock.patch.object(esmfold2_opt, "enable", side_effect=OOM("CUDA out of memory. Tried to allocate 1.00 GiB")):
            with self.assertRaises(OOM):
                cli.activate("fast", "full_msa")
        with mock.patch.object(esmfold2_opt, "enable", side_effect=RuntimeError("weights missing")):     # a non-OOM failure keeps its NOT ACTIVE report
            rep = cli.activate("fast", "full_msa")
        self.assertFalse(rep["active"]); self.assertIn("weights missing", rep["reason"])


if __name__ == "__main__":
    unittest.main()
