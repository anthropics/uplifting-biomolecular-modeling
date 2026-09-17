"""XLA_CACHE, the compile-cache deployment lever: a pre-set JAX_COMPILATION_CACHE_DIR is kept; without a root it is skipped by name; with a root
it is keyed <root>/<stack key>/<recipe>/jax through opt_core.capture.xla_cache (a core without that module is a skip by name, never a local
substitute); the recipe word (det | default) read from the caller's XLA_FLAGS; under det the autotune-cache fence
(JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none for children, jax.config in-process) on the keyed directory, a kept directory read as given; its
LEVER line at exit; the registry files it as a deployment lever outside every mode's `levers=`. No jax, no GPU (a stand-in `jax` module records config updates)."""
import io
import os
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr

from colabfold_opt import modes, registry, report, stack, xla_cache

KEY = "jax0.5.3-cu12.9-sm90"
DET = "--xla_gpu_autotune_level=0 --xla_gpu_deterministic_ops=true"     # the deterministic recipe (README Notes)
FENCE_ENV, FENCE = "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "none"


def fake_core_capture(log):
    """A stand-in opt_core.capture.xla_cache with the two functions the lever calls (the F3 module contract)."""
    cap = types.ModuleType("opt_core.capture"); cap.__path__ = []
    xc = types.ModuleType("opt_core.capture.xla_cache")

    def persistent_cache_env(root, cache_key, sub="jax"):
        return {"JAX_COMPILATION_CACHE_DIR": os.path.join(root, cache_key, sub), "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0",
                "JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES": "-1"}

    def enable_persistent_cache(directory):
        log.append(directory); os.makedirs(directory, exist_ok=True); return {"jax_compilation_cache_dir": directory}
    xc.persistent_cache_env, xc.enable_persistent_cache = persistent_cache_env, enable_persistent_cache
    cap.xla_cache = xc
    return {"opt_core.capture": cap, "opt_core.capture.xla_cache": xc}


def fake_jax(updates, has_option=True):
    """A stand-in `jax` whose config.update records (option, value); without the coupling option it raises AttributeError as jax < 0.4.36 does."""
    jx = types.ModuleType("jax")

    class _Config:
        def update(self, name, value):
            if name == xla_cache.XLA_CACHES_OPTION and not has_option:
                raise AttributeError(f"Unrecognized config option: {name}")
            updates.append((name, value))
    jx.config = _Config()
    return jx


class TestRecipeWord(unittest.TestCase):
    def test_forms(self):
        for flags, want in (
            (None, "default"), ("", "default"), ("--xla_gpu_enable_triton_gemm=false", "default"),
            ("--xla_gpu_deterministic_ops=true", "det"), ("--xla_gpu_deterministic_ops", "det"), ("--xla_gpu_deterministic_ops=1", "det"),
            ("--xla_gpu_deterministic_ops=True", "det"), ("--xla_gpu_deterministic_ops=false", "default"),
            ("--xla_gpu_autotune_level=0", "det"), ("--xla_gpu_autotune_level=4", "default"), ("--xla_gpu_autotune_level", "default"),
            (DET, "det"), ("  --xla_gpu_force_compilation_parallelism=1   --xla_gpu_autotune_level=0 ", "det"),
            ("--xla_gpu_deterministic_opsx=true", "default"),
        ):
            self.assertEqual(xla_cache.recipe(flags), want, flags)
        self.assertEqual(xla_cache.RECIPES, ("default", "det"))


class TestPlacement(unittest.TestCase):
    def setUp(self):
        xla_cache.reset_for_tests(); stack.reset_for_tests()
        self.tmp = tempfile.mkdtemp()
        self.saved = {n: sys.modules.get(n) for n in ("opt_core.capture", "opt_core.capture.xla_cache", "jax")}

    def tearDown(self):
        for n, m in self.saved.items():
            sys.modules.pop(n, None)
            if m is not None:
                sys.modules[n] = m
        xla_cache.reset_for_tests(); stack.reset_for_tests()

    def test_preset_is_kept(self):
        sys.modules.pop("jax", None)
        env = {"JAX_COMPILATION_CACHE_DIR": "/work/jit/x/jax", xla_cache.ROOT_ENV: self.tmp}
        st = xla_cache.apply(KEY, environ=env)
        self.assertEqual((st["state"], st["source"], st["dir"], st["reason"], st["recipe"], st["root"], st["xla_caches"]),
                         ("on", "kept", "/work/jit/x/jax", None, "default", "caller", "jax_default"))
        self.assertEqual(env, {"JAX_COMPILATION_CACHE_DIR": "/work/jit/x/jax", xla_cache.ROOT_ENV: self.tmp})        # default recipe: nothing exported over the caller's choice

    def test_preset_kept_under_det_is_read_as_given(self):
        updates = []
        sys.modules["jax"] = fake_jax(updates)                                                                           # jax imported already (the model process)
        env = {"JAX_COMPILATION_CACHE_DIR": "/work/jit/x/jax", "XLA_FLAGS": DET}
        st = xla_cache.apply(KEY, environ=env)
        self.assertEqual((st["state"], st["source"], st["dir"], st["recipe"], st["xla_caches"]), ("on", "kept", "/work/jit/x/jax", "det", "jax_default"))
        self.assertEqual(env, {"JAX_COMPILATION_CACHE_DIR": "/work/jit/x/jax", "XLA_FLAGS": DET})                        # nothing set beside the caller's directory: stock reads it the same way (exact = stock on it)
        self.assertEqual(updates, [])                                                                                    # no jax.config switch either
        ev = xla_cache.exit_evidence()
        self.assertEqual((ev["source"], ev["recipe"], ev["root"], ev["dir"], ev["xla_caches"]), ("kept", "det", "caller", "/work/jit/x/jax", "jax_default"))
        xla_cache.reset_for_tests()
        env = {"JAX_COMPILATION_CACHE_DIR": "/c", "XLA_FLAGS": DET, FENCE_ENV: "all"}                                    # the caller's own coupling word is reported as set
        self.assertEqual(xla_cache.apply(KEY, environ=env)["xla_caches"], "all")

    def test_keyed_det_on_an_uncoupled_jax(self):
        updates = []
        sys.modules.update(fake_core_capture([])); sys.modules["jax"] = fake_jax(updates, has_option=False)             # a jax without jax_persistent_cache_enable_xla_caches: nothing to fence, said by name
        env = {xla_cache.ROOT_ENV: self.tmp, "XLA_FLAGS": "--xla_gpu_deterministic_ops"}
        st = xla_cache.apply(KEY, environ=env)
        self.assertEqual((st["state"], st["source"], st["recipe"], st["xla_caches"]), ("on", "keyed", "det", "uncoupled")); self.assertEqual(env[FENCE_ENV], FENCE)
        self.assertEqual(st["dir"], os.path.join(self.tmp, KEY, "det", "jax"))

    def test_no_root_is_a_skip_by_name(self):
        st = xla_cache.apply(KEY, environ={})
        self.assertEqual((st["state"], st["reason"], st["enabled"], st["recipe"]), ("skipped", "no_cache_root", False, "default"))
        st = xla_cache.apply(KEY, environ={"XLA_FLAGS": DET})
        self.assertEqual((st["state"], st["reason"], st["recipe"]), ("skipped", "no_cache_root", "det"))

    def test_core_without_capture_is_a_skip_by_name(self):
        blocker = types.ModuleType("opt_core.capture")                                                                  # importable name, no xla_cache inside
        sys.modules["opt_core.capture"] = blocker; sys.modules.pop("opt_core.capture.xla_cache", None)
        env = {xla_cache.ROOT_ENV: self.tmp}
        st = xla_cache.apply(KEY, environ=env)
        self.assertEqual((st["state"], st["reason"]), ("skipped", "opt_core.capture_unavailable")); self.assertEqual(env, {xla_cache.ROOT_ENV: self.tmp})

    def test_root_is_keyed_through_the_core_by_stack_and_recipe(self):
        log = []
        sys.modules.update(fake_core_capture(log)); sys.modules.pop("jax", None)
        env = {xla_cache.ROOT_ENV: self.tmp}
        st = xla_cache.apply(KEY, environ=env)
        want = os.path.join(self.tmp, KEY, "default", "jax")
        self.assertEqual((st["state"], st["source"], st["dir"], st["root"], st["recipe"], st["xla_caches"]), ("on", "keyed", want, self.tmp, "default", "jax_default"))
        self.assertEqual(env[ "JAX_COMPILATION_CACHE_DIR"], want); self.assertEqual(env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"], "0")
        self.assertNotIn(FENCE_ENV, env)                                                                                # default recipe: jax's own coupling stays
        self.assertEqual(log, [])                                                                                       # in-process switch only when jax is imported
        self.assertTrue(os.path.isdir(want)); self.assertEqual(xla_cache.entries(), 0)
        open(os.path.join(want, "exe1"), "w").write("x")
        self.assertEqual(xla_cache.exit_evidence(), {"source": "keyed", "recipe": "default", "root": self.tmp, "dir": want, "xla_caches": "jax_default", "entries": 1})
        self.assertFalse(os.path.exists(os.path.join(self.tmp, KEY, "jax")))                                             # the 0.2.8 layout is neither made nor read

    def test_root_keyed_under_det_is_its_own_directory_with_the_fence(self):
        log, updates = [], []
        sys.modules.update(fake_core_capture(log)); sys.modules["jax"] = fake_jax(updates)
        for flags in (DET, "--xla_gpu_deterministic_ops", "--xla_gpu_autotune_level=0"):
            xla_cache.reset_for_tests(); log.clear(); updates.clear()
            env = {xla_cache.ROOT_ENV: self.tmp, "XLA_FLAGS": flags}
            st = xla_cache.apply(KEY, environ=env)
            want = os.path.join(self.tmp, KEY, "det", "jax")
            self.assertEqual((st["state"], st["source"], st["dir"], st["recipe"], st["xla_caches"]), ("on", "keyed", want, "det", "none"), flags)
            self.assertEqual((env["JAX_COMPILATION_CACHE_DIR"], env[FENCE_ENV]), (want, FENCE), flags)                   # children: the recipe's directory + the fence
            self.assertEqual(log, [want], flags)                                                                            # this process: the cache switched on in jax.config …
            self.assertEqual(updates, [(xla_cache.XLA_CACHES_OPTION, FENCE)], flags)                                        # … and XLA's caches uncoupled from it
        default_dir = os.path.join(self.tmp, KEY, "default", "jax")
        self.assertFalse(os.path.exists(default_dir))                                                                   # a det process touches nothing of default's

    def test_lever_line_forms(self):
        sys.modules.update(fake_core_capture([])); sys.modules.pop("jax", None)
        xla_cache.apply(KEY, environ={xla_cache.ROOT_ENV: self.tmp})
        ev = xla_cache.exit_evidence()
        self.assertEqual(report.lever_line("XLA_CACHE", "on", None, **ev), f"[colabfold-opt] LEVER name=XLA_CACHE state=on impl=opt_core.capture.xla_cache origin=core strategy=F3.jit_cache_keyed source=keyed recipe=default root={self.tmp} dir={ev['dir']} xla_caches=jax_default entries=0")
        xla_cache.reset_for_tests()
        xla_cache.apply(KEY, environ={"JAX_COMPILATION_CACHE_DIR": "/c/jax", "XLA_FLAGS": DET})
        self.assertEqual(report.lever_line("XLA_CACHE", "on", None, **xla_cache.exit_evidence()), "[colabfold-opt] LEVER name=XLA_CACHE state=on impl=opt_core.capture.xla_cache origin=core strategy=F3.jit_cache_keyed source=kept recipe=det root=caller dir=/c/jax xla_caches=jax_default entries=none")
        xla_cache.reset_for_tests(); sys.modules["jax"] = fake_jax([])
        xla_cache.apply(KEY, environ={xla_cache.ROOT_ENV: self.tmp, "XLA_FLAGS": DET})
        det_dir = os.path.join(self.tmp, KEY, "det", "jax")
        self.assertEqual(report.lever_line("XLA_CACHE", "on", None, **xla_cache.exit_evidence()), f"[colabfold-opt] LEVER name=XLA_CACHE state=on impl=opt_core.capture.xla_cache origin=core strategy=F3.jit_cache_keyed source=keyed recipe=det root={self.tmp} dir={det_dir} xla_caches=none entries=0")
        self.assertEqual(report.lever_line("XLA_CACHE", "skipped", "no_cache_root"), "[colabfold-opt] LEVER name=XLA_CACHE state=skipped reason=no_cache_root impl=opt_core.capture.xla_cache origin=core strategy=F3.jit_cache_keyed")


class TestRegistryFiling(unittest.TestCase):
    def test_deployment_lever_outside_every_mode(self):
        self.assertIn("XLA_CACHE", registry.LEVERS); self.assertEqual(registry.DEPLOYMENT, ("XLA_CACHE",))
        self.assertNotIn("XLA_CACHE", registry.IN_MODE); self.assertNotIn("XLA_CACHE", registry.NOT_WIRED)
        for mode in modes.KIT_MODE_NAMES:
            self.assertNotIn("XLA_CACHE", modes.resolve(mode)["levers"])
        self.assertEqual(xla_cache.ROOT_ENV, "COLABFOLD_OPT_JIT_ROOT")
        self.assertEqual((xla_cache.XLA_CACHES_ENV, xla_cache.XLA_CACHES_OPTION, xla_cache.XLA_CACHES_FENCE),
                         ("JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES", "jax_persistent_cache_enable_xla_caches", "none"))   # jax's names (jax/_src/config.py), locked


if __name__ == "__main__":
    unittest.main()
