"""CALIBY_X_LCP (xattempt_addon X004, `fast/complexity.py`): a fused LCP kernel that cannot build or launch is a named stop at the first
served call — `XLcpKernelError: [CALIBY_X_LCP] fused LCP kernel could not build/launch on this device: <exc> — the mode refuses by name (its lever
set cannot run whole here); nothing is substituted` — never the stock lines (a mode is all of its levers; caliby_opt ends the process NOT ACTIVE); an out-of-memory propagates as itself; calls the kernel does not serve (lever off, CPU tensors,
another estimator, an alphabet wider than the kernel's row) take the stock lines, whose values are upstream's (the kit's stock copy of the module on the same input).
The kernel's launch geometry is one constant on every card (`_X_LCP_GEOMETRY` = 8 rows per program × 2 warps — launch speed only, the
same H bits at every geometry) and the launch takes that pair.
The shape rules are read from the source (no torch needed); the behaviour is exercised with the module loaded on CPU against stand-in
`chroma.constants` / `chroma.layers.graph` modules (upstream's AA20 and `collect_neighbors`), skipped by name where torch is not installed."""
import ast
import importlib.util
import os
import sys
import types
import unittest
from unittest import mock

from ._fixtures import kit_sources

try:
    import torch
except ImportError:                                                     # the CPU rules suite may run without torch; the behaviour tests say so by name
    torch = None

STOCK, ADDON = kit_sources()
FAST_PATH, STOCK_PATH = ADDON["complexity.py"], STOCK["complexity.py"]
ESCAPE = "the mode refuses by name (its lever set cannot run whole here); nothing is substituted"
PREFIX = "[CALIBY_X_LCP] fused LCP kernel could not build/launch on this device: "


def _fn(tree, name):
    return next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == name)


class SourceRules(unittest.TestCase):
    """The stop is in the source: complexity_lcp's one broad handler re-raises an out-of-memory, records the failure and raises
    XLcpKernelError — no stock reroute, no printed fallback line — and _x_lcp_usable raises it for a recorded failure."""

    def setUp(self):
        with open(FAST_PATH, encoding="utf-8") as fh:
            self.src = fh.read()
        self.tree = ast.parse(self.src)

    def test_the_call_site_stops_by_name(self):
        fn = _fn(self.tree, "complexity_lcp")
        handlers = [n for n in ast.walk(fn) if isinstance(n, ast.ExceptHandler)]
        self.assertEqual(len(handlers), 1)
        h = handlers[0]
        self.assertEqual((ast.unparse(h.type), h.name), ("Exception", "exc"))
        self.assertEqual(ast.unparse(h.body[0]), "if is_oom(exc):\n    raise")                  # an out-of-memory is never rerouted or renamed
        self.assertEqual([ast.unparse(b) for b in h.body[1:]], ["global _X_LCP_DISABLED_REASON", "_X_LCP_DISABLED_REASON = repr(exc)",
                                                            "raise XLcpKernelError(repr(exc)) from exc"])   # record the failure, then stop — nothing computes H_ij here
        self.assertFalse([n for n in ast.walk(fn) if isinstance(n, ast.Call) and ast.unparse(n.func) == "print"])   # complexity_lcp prints nothing
        self.assertNotIn("falling back", self.src)

    def test_the_error_names_the_lever_and_the_escape(self):
        cls = next(n for n in ast.walk(self.tree) if isinstance(n, ast.ClassDef) and n.name == "XLcpKernelError")
        self.assertEqual([ast.unparse(b) for b in cls.bases], ["RuntimeError"])
        text = ast.unparse(cls)
        self.assertIn(PREFIX, text)
        self.assertIn(ESCAPE, text)
        usable = ast.unparse(_fn(self.tree, "_x_lcp_usable"))
        self.assertIn("raise XLcpKernelError(_X_LCP_DISABLED_REASON)", usable)                # triton / the kernel did not import: the first served call stops
        with open(STOCK_PATH, encoding="utf-8") as fh:
            self.assertNotIn("CALIBY_X_LCP", fh.read())                                        # the stock copy is upstream's file


def _collect_neighbors(node_h, edge_idx):
    """upstream chroma.layers.graph.collect_neighbors: node_h [B, N, C], edge_idx [B, N, K] -> [B, N, K, C]."""
    b, n, k = edge_idx.shape
    idx = edge_idx.reshape(b, n * k, 1).expand(-1, -1, node_h.shape[-1])
    return torch.gather(node_h, 1, idx).reshape(b, n, k, node_h.shape[-1])


def _load(path, name):
    """The module at ``path`` under ``name`` with stand-in chroma.constants (AA20) / chroma.layers.graph (collect_neighbors) in sys.modules."""
    stubs = {"chroma": types.ModuleType("chroma"), "chroma.constants": types.ModuleType("chroma.constants"),
             "chroma.layers": types.ModuleType("chroma.layers"), "chroma.layers.graph": types.ModuleType("chroma.layers.graph")}
    stubs["chroma.constants"].AA20 = "ACDEFGHIKLMNPQRSTVWY"
    stubs["chroma.layers.graph"].collect_neighbors = _collect_neighbors
    saved = {k: sys.modules.get(k) for k in stubs}
    sys.modules.update(stubs)
    try:
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v


@unittest.skipUnless(torch is not None, "torch is not installed: the CALIBY_X_LCP behaviour tests need CPU torch")
class KernelStop(unittest.TestCase):
    OutOfMemoryError = type("OutOfMemoryError", (RuntimeError,), {"__module__": "torch.cuda"})   # torch.cuda.OutOfMemoryError's shape

    @classmethod
    def setUpClass(cls):
        cls.fast = _load(FAST_PATH, "_x_lcp_fast_complexity")
        cls.stock = _load(STOCK_PATH, "_x_lcp_stock_complexity")
        g = torch.Generator().manual_seed(7)
        cls.S = torch.softmax(4.0 * torch.randn(2, 50, 20, generator=g), dim=-1)              # (B, N, Q) float: the differentiable route
        cls.C = torch.ones(2, 50, dtype=torch.long)

    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"CALIBY_X_LCP": "1"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_a_kernel_that_cannot_build_or_launch_stops_the_call_by_name(self):
        cause = RuntimeError("Triton Error [CUDA]: no C compiler on PATH (mock)")
        self.addCleanup(setattr, self.fast, "_X_LCP_DISABLED_REASON", self.fast._X_LCP_DISABLED_REASON)
        served = types.SimpleNamespace(is_cuda=True, dtype=torch.float32, shape=(1, 50, 20))
        with mock.patch.object(self.fast, "_x_lcp_usable", return_value=True), mock.patch.object(self.fast, "_x_fused_entropy_rows", side_effect=cause) as kern:
            with self.assertRaises(self.fast.XLcpKernelError) as cm:
                self.fast.complexity_lcp(self.S, self.C)
        kern.assert_called_once()                                                                 # the first failure stops the run: one attempt, no reroute
        msg = str(cm.exception)
        self.assertTrue(msg.startswith(PREFIX + repr(cause)), msg)
        self.assertTrue(msg.endswith(ESCAPE), msg)
        self.assertIs(cm.exception.__cause__, cause)
        self.assertIsInstance(cm.exception, RuntimeError)
        self.assertEqual(self.fast._X_LCP_DISABLED_REASON, repr(cause))                         # the failure is the module's record (the EXIT line's lcp_kernel=disabled(...))
        with mock.patch.object(self.fast, "_x_fused_entropy_rows") as again:                    # a later served call stops again, by name, without another build
            with self.assertRaises(self.fast.XLcpKernelError) as cm2:
                self.fast._x_lcp_usable(served, "naive")
            self.assertEqual(str(cm2.exception), PREFIX + repr(cause) + " \u2014 " + ESCAPE)
            again.assert_not_called()

    def test_an_out_of_memory_propagates_as_itself(self):
        with mock.patch.object(self.fast, "_x_lcp_usable", return_value=True), \
                mock.patch.object(self.fast, "_x_fused_entropy_rows", side_effect=self.OutOfMemoryError("CUDA out of memory (mock)")):
            with self.assertRaises(self.OutOfMemoryError):
                self.fast.complexity_lcp(self.S, self.C)

    def test_an_import_failure_stops_the_first_served_call(self):
        served = types.SimpleNamespace(is_cuda=True, dtype=torch.float32, shape=(1, 50, 20))     # a call the kernel serves: CUDA fp32, Q = 20
        with mock.patch.object(self.fast, "_X_LCP_DISABLED_REASON", "ImportError('no triton (mock)')"):
            with self.assertRaises(self.fast.XLcpKernelError) as cm:
                self.fast._x_lcp_usable(served, "naive")
            self.assertEqual(str(cm.exception), PREFIX + "ImportError('no triton (mock)')" + " \u2014 " + ESCAPE)
            self.assertFalse(self.fast._x_lcp_usable(served, "chao-shen"))                         # routing is decided before the record is read:
            self.assertFalse(self.fast._x_lcp_usable(types.SimpleNamespace(is_cuda=False, dtype=torch.float32, shape=(1, 50, 20)), "naive"))
            self.assertFalse(self.fast._x_lcp_usable(types.SimpleNamespace(is_cuda=True, dtype=torch.float32, shape=(1, 50, 80)), "naive"))   # an alphabet the kernel does not serve
            with mock.patch.dict(os.environ, {"CALIBY_X_LCP": "0"}):
                self.assertFalse(self.fast._x_lcp_usable(served, "naive"))                         # the escape itself: lever off, stock lines, no stop
        if self.fast._X_LCP_DISABLED_REASON is None:                                            # triton imported here: the same call is the kernel's
            self.assertTrue(self.fast._x_lcp_usable(served, "naive"))

    def test_calls_the_kernel_does_not_serve_take_upstreams_lines(self):
        """CPU tensors are not the kernel's: the fast module's values are the stock module's, lever on or off, bit for bit."""
        want = self.stock.complexity_lcp(self.S, self.C)
        want_idx = self.stock.complexity_lcp(self.S.argmax(-1), self.C)
        for level in ("1", "0"):
            with mock.patch.dict(os.environ, {"CALIBY_X_LCP": level}), mock.patch.object(self.fast, "_x_fused_entropy_rows", side_effect=AssertionError("kernel called on CPU")):
                self.assertTrue(torch.equal(self.fast.complexity_lcp(self.S, self.C), want), level)
                self.assertTrue(torch.equal(self.fast.complexity_lcp(self.S.argmax(-1), self.C), want_idx), level)
                for method in ("chao-shen", "miller-maddow"):
                    self.assertTrue(torch.equal(self.fast.complexity_lcp(self.S, self.C, method=method), self.stock.complexity_lcp(self.S, self.C, method=method)), (level, method))


class LaunchGeometry(unittest.TestCase):
    """`_X_LCP_GEOMETRY`: one (BLOCK, num_warps) pair for every card, and the source launches with it (no literal geometry at the launch)."""

    def test_the_constant_and_the_launch_in_the_source(self):
        with open(FAST_PATH, encoding="utf-8") as fh:
            tree = ast.parse(fh.read())
        geom = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "_X_LCP_GEOMETRY"]
        self.assertEqual([ast.literal_eval(n.value) for n in geom], [(8, 2)])                            # one pair, stated once
        rows = _fn(tree, "_x_fused_entropy_rows")
        self.assertIn("BLOCK, warps = _X_LCP_GEOMETRY", ast.unparse(rows))
        self.assertNotIn("get_device_capability", ast.unparse(rows))                                     # the same geometry on every card
        launch = next(n for n in ast.walk(rows) if isinstance(n, ast.Call) and ast.unparse(n.func).startswith("_x_lcp_rows_kernel["))
        kw = {k.arg: ast.unparse(k.value) for k in launch.keywords}
        self.assertEqual((kw["BLOCK"], kw["num_warps"], kw["enable_fp_fusion"], kw["L"]), ("BLOCK", "warps", "False", "L"))   # fp fusion stays off (numerics)


if __name__ == "__main__":
    unittest.main()
