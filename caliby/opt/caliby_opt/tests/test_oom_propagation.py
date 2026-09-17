"""Out-of-memory on the served paths. The kit's one classifier is the shared core's (`opt_core.oom.is_oom`); the rule it serves: a broad
`except` around a lever or a kernel launch (a fallback to a stock or degraded route, or a rename of the failure) re-raises an out-of-memory
before anything else, because a fallback needs at least the memory the fast path needed and a rename would hide the out-of-memory. This file is the census, by syntax tree, not by grep:

* the package's own served modules (`caliby_opt/*.py`) carry NO broad handler around a lever or kernel — the broad handlers that exist (`PACKAGE_SERVED_OTHER`)
  cannot originate an out-of-memory or reroute a lever: the finder's `find_spec` delegation (a sibling finder raising = not found
  there), the stderr tee's fallback-line counter (`write`: counting never breaks stderr, the write itself always happens), the GPU probe
  (`gpu_info`: `import torch` failing is recorded by name and `exact` refuses "no GPU" by name), the distribution-version read for the
  activation line's notes (`dist_version`: unreadable metadata = no version to name), the background CIF writers' join
  (`bg_writers.Writers.join`: the core writer's failure is named on stderr and re-raised — the process exits non-zero, nothing is rerouted),
  the install step's weights fetch (`weights.fetch`: upstream's downloader errors relayed by name; CPU, never in a design process), the
  script runner (`design_run.run_main`: an exception the writer raised — a lever's named stop, or an out-of-memory that reached the top —
  is printed whole and becomes exit code 1 so the exit rule still runs; nothing is rerouted or retried);
* the carried lever files the overlay loads under the upstream module names (`opt/forward/xattempt_addon/fast/*`) carry exactly three
  broad handlers around an allocating call — `complexity_lcp` (the fused LCP Triton kernel launch → the named stop `XLcpKernelError`,
  never the stock lines), `clean_pdbs` (the clean lever's DataLoader route → the stock serial loop) — and each starts with `if is_oom(e): raise`;
  plus two that cannot allocate: the import-layout guards at `complexity.py` module level (a Triton import or `@triton.jit` definition
  failing = the lever disabled by name);
* the kit owner's unit tests (`xattempt_addon/tests/*`) are not on the served path
  (the overlay's file map, `stack.TOUCHED`, names neither) and are out of scope.
So an out-of-memory propagates on every served path and no fallback is applied to it; when a lever's fallback line fires for any other
reason, it is named on its own stderr line and on the EXIT line (`partial=`), recorded, and the design process ends NOT ACTIVE by name
(exit 3: a mode is all of its levers, never a subset under its name — `test_stock_arm_and_cli`). The
`clean_pdbs` handlers are also driven here from the carried bytes (the function's source loaded by syntax tree, the upstream `clean_pdb`
stubbed — the served entry, `caliby.api.clean_pdbs` inside a design process, needs the upstream stack this suite does not install): a mocked
`torch.cuda.OutOfMemoryError` from the loader route is raised to the caller, an ordinary loader failure still takes the serial fallback."""
import ast
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

from .. import stack

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                 # caliby_opt/
FORWARD = os.path.join(stack.opt_home(), "forward")
BROAD_NAMES = {"Exception", "BaseException", "RuntimeError"}
OWNER_DEV = ("xattempt_addon/tests/",)   # the owner's unit tests: never loaded by the overlay

PACKAGE_SERVED_OTHER = [("_autoload.py", "find_spec"), ("bg_writers.py", "join"), ("design_run.py", "run_main"), ("report.py", "write"), ("stack.py", "dist_version"),
                        ("stack.py", "gpu_info"), ("weights.py", "fetch"), ("weights.py", "fetch"), ("weights.py", "fetch")]   # cannot allocate, reroute no lever                                       # the install step's weights fetch (CPU, never in a design process): upstream's downloader errors relayed by name
CARRIED_REROUTE = [("xattempt_addon/fast/api.py", "clean_pdbs"),
                   ("xattempt_addon/fast/complexity.py", "complexity_lcp")]                                                # around an allocating lever / kernel call (a reroute, or complexity_lcp's named stop)
CARRIED_OTHER = [("xattempt_addon/fast/complexity.py", "<module>"), ("xattempt_addon/fast/complexity.py", "<module>")]   # import-layout guards


def _is_broad(handler: ast.ExceptHandler) -> bool:
    if handler.type is None:
        return True
    names = {n.id for n in ast.walk(handler.type) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(handler.type) if isinstance(n, ast.Attribute)}
    return bool(names & BROAD_NAMES)


def _reraises_oom_first(handler: ast.ExceptHandler) -> bool:
    """The handler's first statement is exactly `if is_oom(<its name>): raise`."""
    first = handler.body[0]
    if not (isinstance(first, ast.If) and isinstance(first.test, ast.Call) and not first.orelse and len(first.body) == 1):
        return False
    f = first.test.func
    called = f.id if isinstance(f, ast.Name) else (f.attr if isinstance(f, ast.Attribute) else None)
    arg_ok = len(first.test.args) == 1 and isinstance(first.test.args[0], ast.Name) and first.test.args[0].id == handler.name
    return called == "is_oom" and arg_ok and isinstance(first.body[0], ast.Raise) and first.body[0].exc is None


def broad_handlers(path: str, rel: str) -> list:
    """[(rel, innermost enclosing def or '<module>', handler)] for every broad handler in the file, in source order."""
    with open(path, encoding="utf-8") as fh:
        tree = ast.parse(fh.read(), path)
    found = []

    def visit(node, scope):
        for child in ast.iter_child_nodes(node):
            inner = child.name if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)) else scope
            if isinstance(child, ast.ExceptHandler) and _is_broad(child):
                found.append((rel, scope, child))
            visit(child, inner)
    visit(tree, "<module>")
    return found


def carried_handlers() -> list:
    out = []
    for d, dns, fns in os.walk(FORWARD):
        dns[:] = sorted(x for x in dns if x != "__pycache__")
        for f in sorted(fns):
            rel = os.path.relpath(os.path.join(d, f), FORWARD)
            if f.endswith(".py") and not rel.startswith(OWNER_DEV):
                out += broad_handlers(os.path.join(d, f), rel)
    return out


class TestOomClassifier(unittest.TestCase):
    def test_is_oom_is_the_cores_and_needs_no_torch(self):
        """One classifier, the core's; torch's class is recognised by name (this suite installs no torch), the host's MemoryError too,
        and one level down a wrapper's __cause__; an ordinary lever failure is not an out-of-memory."""
        from opt_core.oom import is_oom
        OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch.cuda"})   # torch.cuda.OutOfMemoryError's shape, without torch
        self.assertTrue(is_oom(OutOfMemoryError("CUDA out of memory (mock)")))
        self.assertTrue(is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))
        self.assertTrue(is_oom(MemoryError()))
        wrapped = RuntimeError("DataLoader worker failed")
        wrapped.__cause__ = OutOfMemoryError("CUDA out of memory (mock)")
        self.assertTrue(is_oom(wrapped))
        self.assertFalse(is_oom(RuntimeError("Triton JIT: no C compiler")))
        self.assertFalse(is_oom(ValueError("x")))


class TestServedHandlers(unittest.TestCase):
    def test_package_served_modules_reroute_nothing(self):
        """caliby_opt/*.py: the broad handlers are exactly the six that cannot allocate and reroute no lever plus the install step's three fetch relays (file, enclosing def)."""
        got = [h for f in sorted(os.listdir(PKG)) if f.endswith(".py") for h in broad_handlers(os.path.join(PKG, f), f)]
        self.assertEqual(sorted((rel, scope) for rel, scope, _ in got), sorted(PACKAGE_SERVED_OTHER), [(r, s, h.lineno) for r, s, h in got])

    def test_carried_lever_files_broad_handlers_are_the_named_set(self):
        """The carried tree outside the owner's tests: exactly the three rerouting handlers and the three that cannot allocate — a kit
        re-issue that adds, moves or removes one revisits this census."""
        got = carried_handlers()
        self.assertEqual(sorted((rel, scope) for rel, scope, _ in got), sorted(CARRIED_REROUTE + CARRIED_OTHER), [(r, s, h.lineno) for r, s, h in got])

    def test_carried_reroutes_reraise_oom_first(self):
        """Each of the three rerouting handlers starts with `if is_oom(<name>): raise`, and each of the two files imports the classifier once,
        from the core (`from opt_core.oom import is_oom`) — no kit-local spelling."""
        got = [(rel, scope, h) for rel, scope, h in carried_handlers() if (rel, scope) in CARRIED_REROUTE]
        self.assertEqual(len(got), len(CARRIED_REROUTE))
        for rel, scope, h in got:
            self.assertTrue(_reraises_oom_first(h), (rel, scope, h.lineno))
        for rel in sorted({rel for rel, _ in CARRIED_REROUTE}):
            with open(os.path.join(FORWARD, rel), encoding="utf-8") as fh:
                src = fh.read()
            self.assertEqual(src.count("from opt_core.oom import is_oom"), 1, rel)
            self.assertNotIn("def is_oom", src, rel)
        pkg_src = "".join(open(os.path.join(PKG, f), encoding="utf-8").read() for f in os.listdir(PKG) if f.endswith(".py"))
        self.assertNotIn("def is_oom", pkg_src)                                                  # never a kit-local classifier

    def test_owner_tests_are_not_served(self):
        """What OWNER_DEV excludes is never loaded by the overlay: no kit file of the overlay's map lives there."""
        served = {os.path.join(FORWARD, stack.KIT_ADDON, "fast", name) for name in stack.TOUCHED}
        for path in served:
            rel = os.path.relpath(path, FORWARD)
            self.assertFalse(rel.startswith(OWNER_DEV), rel)
        self.assertTrue(any(os.path.isfile(p) for p in served))


def _carried_clean_pdbs():
    """`clean_pdbs` compiled from the carried `xattempt_addon/fast/api.py` (its own source, by syntax tree) into a namespace holding what the
    function uses at module level (os, tempfile, Path, is_oom); the upstream `clean_pdb` it imports lazily is a stub module in sys.modules."""
    from pathlib import Path
    from opt_core.oom import is_oom
    path = os.path.join(FORWARD, "xattempt_addon", "fast", "api.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    body = ast.parse(src, path).body
    fn = next(n for n in body if isinstance(n, ast.FunctionDef) and n.name == "clean_pdbs")
    helpers = [n for n in body if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in ("_XCleanDataset", "_x_identity")]   # the loader route's dataset + collate, module level in api.py
    ns = {"os": os, "tempfile": tempfile, "Path": Path, "Any": object, "is_oom": is_oom, "__name__": "carried_api_clean_pdbs"}
    exec(compile(ast.Module(body=helpers + [fn], type_ignores=[]), path, "exec"), ns)   # noqa: S102 — the carried function and its two helpers, nothing else of the module
    return ns["clean_pdbs"]


class TestCarriedCleanLeverUnderMockedOom(unittest.TestCase):
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch.cuda"})   # torch.cuda.OutOfMemoryError without torch

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="oom_clean_")
        stub = types.ModuleType("caliby.data.preprocessing.atomworks.clean_pdbs")
        stub.clean_pdb = lambda pdb_path, out_dir: os.path.join(str(out_dir), os.path.basename(str(pdb_path)) + ".cif")   # the serial route's unit of work
        parents = {n: types.ModuleType(n) for n in ("caliby", "caliby.data", "caliby.data.preprocessing", "caliby.data.preprocessing.atomworks")}
        joblib = types.ModuleType("joblib"); joblib.Parallel = joblib.delayed = None                                # imported by the function, used only on the stock parallel route
        self.torch_data = types.ModuleType("torch.utils.data")                                                       # the loader route's one import: `from torch.utils.data import DataLoader`
        torch_pkg, torch_utils = types.ModuleType("torch"), types.ModuleType("torch.utils")
        self.modules = mock.patch.dict(sys.modules, {**parents, stub.__name__: stub, "joblib": joblib,
                                                     "torch": torch_pkg, "torch.utils": torch_utils, "torch.utils.data": self.torch_data})
        self.modules.start(); self.addCleanup(self.modules.stop)
        self.env = mock.patch.dict(os.environ, {"CALIBY_X_CLEAN": "loader"})
        self.env.start(); self.addCleanup(self.env.stop)
        self.clean_pdbs = _carried_clean_pdbs()

    def _loader_raising(self, exc):
        def DataLoader(*args, **kwargs):                                                                           # noqa: N802 — the class's name in torch
            raise exc
        self.torch_data.DataLoader = DataLoader

    def test_oom_from_the_loader_route_is_raised_not_rerouted(self):
        self._loader_raising(self.OutOfMemoryError("CUDA out of memory (mock)"))
        with redirect_stderr(io.StringIO()) as err:
            with self.assertRaises(self.OutOfMemoryError):
                self.clean_pdbs(["a.pdb", "b.pdb"], out_dir=self.tmp, num_workers=2)
        self.assertNotIn("serial fallback", err.getvalue())

    def test_an_ordinary_loader_failure_still_takes_the_serial_fallback(self):
        self._loader_raising(RuntimeError("DataLoader worker exited unexpectedly (mock)"))
        with redirect_stderr(io.StringIO()) as err:
            got = self.clean_pdbs(["a.pdb", "b.pdb"], out_dir=self.tmp, num_workers=2)
        self.assertEqual(got, [os.path.join(self.tmp, "a.pdb.cif"), os.path.join(self.tmp, "b.pdb.cif")])
        self.assertIn("[CALIBY_X_CLEAN=loader] failed", err.getvalue())
        self.assertIn("-> serial fallback", err.getvalue())


if __name__ == "__main__":
    unittest.main()
