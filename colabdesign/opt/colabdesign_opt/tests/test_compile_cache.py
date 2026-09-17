"""Lever `compilecache` (compilecache_jax.py): where the cache goes (the two directory rules, pure), the refusal by name on an unwritable
directory, the LEVER line grammar (opt_core.report.lever_line) off / on, and — where a real jax with its persistent-cache module is importable
(the pinned GPU stack; skipped on the CPU stand-in stack) — install(): jax.config + environment in force before any compile, idempotent, the
`reinit` repair named, one exit line."""

import importlib.util
import os
import re
import tempfile
import unittest

try:
    import pytest
except ImportError:                                   # the image: unittest only
    from colabdesign_opt.tests import _noptest as pytest

from opt_core import gates
from opt_core.jax_design import pcc

from colabdesign_opt import compilecache_jax as cc

KEY = "jax0.6.0-jaxlib0.6.0-cuda12plugin0.6.0-nvidia-h100-80gb-hbm3"


def real_jax() -> bool:
    """A real jax with its persistent-cache module is importable — decided from the package's files WITHOUT importing jax (this runs at module
    import under `unittest discover`, in the process the stand-in tests share)."""
    try:
        spec = importlib.util.find_spec("jax")
    except (ImportError, ValueError):
        return False
    if spec is None or not spec.submodule_search_locations:
        return False
    root = list(spec.submodule_search_locations)[0]
    return os.path.isfile(os.path.join(root, "_src", "compilation_cache.py")) and os.path.isfile(os.path.join(root, "monitoring.py"))


class TestDirectoryRules(unittest.TestCase):
    def test_rule1_jax_variable_adopted_as_given(self):
        with tempfile.TemporaryDirectory() as t:
            r = cc.resolve_dir({cc.ENV_DIR: os.path.join(t, "my cache"), cc.ENV_XDG: "/elsewhere"}, key=KEY)
            self.assertEqual(r, {"dir": os.path.join(t, "my cache"), "dir_source": "JAX_COMPILATION_CACHE_DIR", "key": "adopted", "root": None})

    def test_rule2_keyed_default_under_xdg_cache_home(self):
        r = cc.resolve_dir({cc.ENV_XDG: "/x/cache"}, key=KEY)
        self.assertEqual(r["dir"], f"/x/cache/colabdesign_opt/pcc/{KEY}/xla")
        self.assertEqual((r["dir_source"], r["key"], r["root"]), ("default", KEY, "/x/cache/colabdesign_opt/pcc"))
        self.assertEqual(r["dir"], pcc.cache_dir(r["root"], KEY))                       # the core's placement, not a kit convention

    def test_rule2_home_cache_when_xdg_unset(self):
        r = cc.resolve_dir({}, key=KEY)
        self.assertEqual(r["dir"], os.path.join(os.path.expanduser("~"), ".cache", "colabdesign_opt", "pcc", KEY, "xla"))
        self.assertEqual(r["dir_source"], "default")

    def test_blank_variable_is_unset(self):
        self.assertEqual(cc.resolve_dir({cc.ENV_DIR: "  ", cc.ENV_XDG: ""}, key=KEY)["dir_source"], "default")

    def test_unwritable_directory_refused_by_name(self):
        with tempfile.TemporaryDirectory() as t:
            blocker = os.path.join(t, "file"); open(blocker, "w").close()
            with self.assertRaises(cc.CompileCacheError) as cm:
                cc._writable_dir(os.path.join(blocker, "sub"))
            self.assertTrue(gates.is_cannot_run(cm.exception))
            for word in ("compilecache:", "not writable", "JAX_COMPILATION_CACHE_DIR", "XDG_CACHE_HOME"):
                self.assertIn(word, str(cm.exception))

    def test_no_autotune_pin_no_xla_flags(self):
        """The lever exports the cache directory and the store-everything thresholds; XLA_FLAGS is never among its writes."""
        ex = pcc.plan("/d", autotune="off", environ={"XLA_FLAGS": "--xla_gpu_enable_triton_gemm=false"}, exact=True)["exports"]
        self.assertEqual(ex["JAX_COMPILATION_CACHE_DIR"], "/d")
        self.assertEqual((ex["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"], ex["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"]), ("0", "0"))
        self.assertEqual(ex.get("XLA_FLAGS"), "--xla_gpu_enable_triton_gemm=false")     # carried through unchanged, nothing appended


class TestLeverLine(unittest.TestCase):
    def setUp(self):
        cc.reset_for_tests()

    tearDown = setUp

    def test_off_before_install(self):
        line = cc.line_of()
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=compilecache state=off impl=jax_design.pcc@"), line)
        self.assertIn(" origin=core numerics=exact", line)
        self.assertFalse(cc.installed())

    def test_protocol_words(self):
        for attr in ("install", "installed", "uninstall", "off_line", "evidence", "REFUSALS", "NUMERICS"):
            self.assertTrue(hasattr(cc, attr), attr)
        self.assertEqual(cc.NUMERICS, "exact"); self.assertEqual(cc.REFUSALS, (cc.CompileCacheError,))
        line = cc.off_line("ablated")
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=compilecache state=off reason=ablated impl=jax_design.pcc@"), line)
        self.assertIn(" origin=core", line)

    def test_on_line_grammar(self):
        ev = {"installed": True, "dir": "/c/my dir/xla", "dir_source": "default", "key": KEY, "autotune": "jax_xla_cache", "entries_install": 0,
              "entries_now": 104, "reinit": 0, "requests": 104, "hits": 2, "misses": 102, "compiled_s": 7.94, "saved_s": 92.36, "via": "jax.config+environ"}
        line = cc.line_of(ev)
        self.assertTrue(line.startswith("[colabdesign-opt] LEVER name=compilecache state=on impl=jax_design.pcc@"), line)
        for part in (" origin=core ", " numerics=exact ", " dir=/c/my_dir/xla ", " dir_source=default ", f" key={KEY} ", " autotune=jax_xla_cache ", " requests=104 hits=2 misses=102 compiled_s=7.9 saved_s=92.4 ",
                     " entries=0->104 ", " reinit=0 ", " source=exit ", f" pid={os.getpid()}"):
            self.assertIn(part, line)
        self.assertNotRegex(line, r"=\S*\s\S*=(?!\S)")                                    # every field is one blank-free k=v token
        fields = dict(re.findall(r"(\S+?)=(\S+)", line))
        self.assertEqual(fields["name"], "compilecache"); self.assertEqual(fields["state"], "on")


@unittest.skipUnless(real_jax(), "install() needs a real jax with jax._src.compilation_cache (the pinned stack)")
class TestInstallRealJax(unittest.TestCase):
    def setUp(self):
        cc.reset_for_tests()
        import jax
        from jax._src import compilation_cache as jcc
        self.jax, self.jcc = jax, jcc
        jcc.reset_cache()
        jax.config.update("jax_compilation_cache_dir", None)

    def tearDown(self):
        cc.reset_for_tests()
        self.jcc.reset_cache()
        self.jax.config.update("jax_compilation_cache_dir", None)

    def test_install_puts_the_cache_in_force_idempotently(self):
        with tempfile.TemporaryDirectory() as t:
            env = {cc.ENV_DIR: os.path.join(t, "c")}
            info = cc.install(env, key=KEY)
            self.assertEqual((info["dir"], info["dir_source"], info["key"], info["entries_install"], info["reinit"]), (os.path.join(t, "c"), "JAX_COMPILATION_CACHE_DIR", "adopted", 0, 0))
            self.assertTrue(os.path.isdir(info["dir"]))
            self.assertEqual(self.jax.config.jax_compilation_cache_dir, info["dir"])
            self.assertEqual(float(self.jax.config.jax_persistent_cache_min_compile_time_secs), 0.0)
            self.assertEqual(int(self.jax.config.jax_persistent_cache_min_entry_size_bytes), 0)
            self.assertEqual((env["JAX_COMPILATION_CACHE_DIR"], env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"]), (info["dir"], "0"))
            self.assertNotIn("XLA_FLAGS", env)
            self.assertEqual(cc.install(env, key=KEY), info)                                   # idempotent: the record, nothing redone
            ev = cc.evidence()
            self.assertEqual((ev["installed"], ev["requests"], ev["hits"], ev["misses"], ev["entries_now"], ev["compiled_s"], ev["saved_s"]), (True, 0, 0, 0, 0, 0.0, 0.0))
            self.assertIn(" state=on ", cc.exit_line())
            self.assertTrue(cc.installed())
            cc.uninstall(env)                                                                    # out of force: jax defaults back, exports gone, record off
            self.assertFalse(cc.installed())
            self.assertIsNone(self.jax.config.jax_compilation_cache_dir)
            self.assertNotIn("JAX_COMPILATION_CACHE_DIR", env)
            self.assertIn(" state=off ", cc.exit_line())

    def test_reinit_named_when_jax_cache_state_was_already_consulted(self):
        with tempfile.TemporaryDirectory() as t:
            self.jcc._cache_initialized = True; self.jcc._cache = None                         # what a compile before install leaves behind with no directory set
            info = cc.install({cc.ENV_DIR: os.path.join(t, "c")}, key=KEY)
            self.assertEqual(info["reinit"], 1)
            self.assertFalse(self.jcc._cache_initialized)                                      # reset: the next compile initialises the cache at our directory
            self.assertTrue(self.jcc.is_persistent_cache_enabled() if hasattr(self.jcc, "is_persistent_cache_enabled") else True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
