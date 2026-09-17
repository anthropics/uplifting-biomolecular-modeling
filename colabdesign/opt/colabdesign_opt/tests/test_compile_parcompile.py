"""Lever `parcompile` (launchpad_parcompile.py): the flag composition over upstream's XLA_FLAGS assignment (kept verbatim, ours appended once, the
last occurrence wins), the thread rule min(16, cpus) >= 2, install() on the stand-in stack (colabdesign importable, no backend) — the environment
written, idempotent — and the LEVER line grammar off / on."""

import os
import re
import shutil
import sys
import tempfile
import unittest
from unittest import mock

from opt_core import gates

try:
    import pytest
except ImportError:                                   # the image: unittest only
    from colabdesign_opt.tests import _noptest as pytest

from colabdesign_opt import launchpad_parcompile as pc

UP = "--xla_gpu_enable_triton_gemm=false"          # colabdesign/__init__.py's assignment


class TestFlags(unittest.TestCase):
    def test_thread_rule(self):
        self.assertEqual([pc.thread_count(c) for c in (1, 2, 8, 16, 17, 128)], [2, 2, 8, 16, 16, 16])
        self.assertTrue(2 <= pc.thread_count() <= 16)

    def test_compose_appends_once_after_upstream(self):
        s = pc.compose(UP, 16)
        self.assertEqual(s, f"{UP} --xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=16")
        self.assertEqual(pc.compose(s, 16), s)                                              # idempotent on the string
        self.assertEqual(pc.compose("", 4), "--xla_gpu_enable_llvm_module_compilation_parallelism=true --xla_gpu_force_compilation_parallelism=4")
        self.assertEqual(pc.compose(None, 4), pc.compose("", 4))

    def test_compose_appends_after_another_value(self):
        s = pc.compose(f"{UP} --xla_gpu_force_compilation_parallelism=1", 8)
        self.assertTrue(s.endswith("--xla_gpu_force_compilation_parallelism=8"), s)          # XLA takes the last occurrence
        self.assertIn("--xla_gpu_force_compilation_parallelism=1 ", s)                       # the earlier word is left as it was (nothing of the caller's is edited)


class TestInstallAndLine(unittest.TestCase):
    def setUp(self):
        pc.reset_for_tests()
        self._saved = os.environ.get("XLA_FLAGS")
        self._path = list(sys.path)
        try:                                                                                # install() imports colabdesign: the real one on the pinned stack, else the suite's stand-in
            import colabdesign  # noqa: F401
        except ImportError:
            from colabdesign_opt.tests import _stubs
            self._tmp = tempfile.mkdtemp(prefix="cd_opt_pc_")
            sys.path.insert(0, _stubs.make_stub_stack(self._tmp))

    def tearDown(self):
        pc.reset_for_tests()
        sys.path[:] = self._path
        for m in [m for m in sys.modules if m == "colabdesign" or m.startswith("colabdesign.")]:
            if getattr(sys.modules[m], "__file__", "") and getattr(self, "_tmp", None) and sys.modules[m].__file__.startswith(self._tmp):
                del sys.modules[m]
        if getattr(self, "_tmp", None):
            shutil.rmtree(self._tmp, ignore_errors=True)
        if self._saved is None:
            os.environ.pop("XLA_FLAGS", None)
        else:
            os.environ["XLA_FLAGS"] = self._saved

    def test_off_line_and_protocol_words(self):
        self.assertEqual(pc.line_of(), "[colabdesign-opt] LEVER name=parcompile state=off impl=xla_llvm_module_parallelism@kit origin=kit numerics=exact")
        for attr in ("install", "installed", "uninstall", "off_line", "evidence", "REFUSALS", "NUMERICS"):
            self.assertTrue(hasattr(pc, attr), attr)
        self.assertEqual(pc.NUMERICS, "exact"); self.assertEqual(pc.REFUSALS, (pc.ParcompileError,))
        self.assertEqual(pc.off_line("ablated"), "[colabdesign-opt] LEVER name=parcompile state=off reason=ablated impl=xla_llvm_module_parallelism@kit origin=kit numerics=exact")

    def test_refused_by_name_when_backends_already_initialised(self):
        with mock.patch.object(pc, "_backends_initialized", return_value=True):
            with self.assertRaises(pc.ParcompileError) as cm:
                pc.install({"XLA_FLAGS": UP}, threads=8)
        self.assertTrue(gates.is_cannot_run(cm.exception))
        self.assertIn("backends are already initialised", str(cm.exception))
        self.assertFalse(pc.installed())

    def test_install_writes_the_environment_idempotently(self):
        """The composition and the record (the backend probe is pinned to 'not yet': in a suite process another test may have touched a device —
        the refusal that then follows is the test above)."""
        env = {"XLA_FLAGS": UP}
        with mock.patch.object(pc, "_backends_initialized", return_value=False):
            info = pc.install(env, threads=8)
        self.assertEqual(info["threads"], 8)
        self.assertEqual(info["xla_flags_kept"], UP)
        self.assertEqual(env["XLA_FLAGS"], pc.compose(UP, 8))
        self.assertEqual(os.environ["XLA_FLAGS"], env["XLA_FLAGS"])                          # the process environment too (XLA reads os.environ)
        self.assertEqual(pc.install(env, threads=3), info)                                  # idempotent: the first install stands
        line = pc.line_of()
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=parcompile state=on impl=xla_llvm_module_parallelism@kit origin=kit numerics=exact threads=8 cpus="), line)
        self.assertIn(" flags=--xla_gpu_enable_llvm_module_compilation_parallelism=true,--xla_gpu_force_compilation_parallelism=8 ", line)
        self.assertIn(f" xla_flags_kept={UP} ", line)
        self.assertIn(f" source=install pid={os.getpid()}", line)
        fields = dict(re.findall(r"(\S+?)=(\S+)", line))
        self.assertEqual((fields["name"], fields["state"]), ("parcompile", "on"))
        self.assertTrue(pc.installed())
        pc.uninstall(env)                                                                   # XLA_FLAGS back to upstream's word, record off
        self.assertFalse(pc.installed())
        self.assertEqual((env["XLA_FLAGS"], os.environ["XLA_FLAGS"]), (UP, UP))
        self.assertIn(" state=off ", pc.line_of())


class TestCgroupQuota(unittest.TestCase):
    def test_v2_v1_quota_forms(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            open(os.path.join(t, "cpu.max"), "w").write("1600000 100000\n")
            self.assertEqual(pc.cgroup_cpu_quota([t]), 16)
            open(os.path.join(t, "cpu.max"), "w").write("250000 100000\n")
            self.assertEqual(pc.cgroup_cpu_quota([t]), 3)                                        # ceil(2.5)
            open(os.path.join(t, "cpu.max"), "w").write("max 100000\n")
            self.assertIsNone(pc.cgroup_cpu_quota([t]))
            v1 = os.path.join(t, "v1"); os.makedirs(v1)
            open(os.path.join(v1, "cpu.cfs_quota_us"), "w").write("800000"); open(os.path.join(v1, "cpu.cfs_period_us"), "w").write("100000")
            self.assertEqual(pc.cgroup_cpu_quota(["/nonexistent", v1]), 8)
            open(os.path.join(v1, "cpu.cfs_quota_us"), "w").write("-1")
            self.assertIsNone(pc.cgroup_cpu_quota([v1]))
            self.assertIsNone(pc.cgroup_cpu_quota(["/nonexistent/x"]))

    def test_proc_self_cgroup_paths(self):
        import tempfile
        with tempfile.TemporaryDirectory() as t:
            proc = os.path.join(t, "cgroup"); open(proc, "w").write("0::/my/slice\n7:cpu,cpuacct:/docker/abc\n3:memory:/x\n")
            dirs = pc._cgroup_dirs(root=t, proc=proc)
            self.assertEqual(dirs[:3], [os.path.join(t, "my/slice"), os.path.join(t, "cpu", "docker/abc"), os.path.join(t, "cpu,cpuacct", "docker/abc")])
            self.assertIn(t, dirs)

    def test_cpu_count_positive_and_threads_capped(self):
        self.assertGreaterEqual(pc.cpu_count(), 1)
        self.assertEqual(pc.thread_count(10**6), pc.MAX_THREADS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
