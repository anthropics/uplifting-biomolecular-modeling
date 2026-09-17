"""The mode table (modes.py) against the kits' own files: off | exact | fast | big, the default, fast's row over exact's, the switches read from
the runner, the PYTHONPATH order of the mode table, the imports read from sitecustomize / the runner, the excluded switch values."""
import os
import re
import unittest

from boltzgen_opt import modes, registry, stack

HOME = stack.opt_home()


class TestModesTable(unittest.TestCase):
    def test_modes_are_off_exact_fast_and_big(self):
        self.assertEqual(modes.MODES, ("off", "exact", "fast", "big"))
        self.assertEqual(modes.FAST_MODE, "fast")
        self.assertEqual(set(modes.KIT_MODES), {"exact", "fast", "big"})
        for name in ("VARIANTS", "Variant", "check_variant", "route_refusal", "ENV_VARIANT", "ROUTES", "KIT_WORKER"):   # one row per mode: no variants, no per-verb routes, no worker kit
            self.assertFalse(hasattr(modes, name), name)
        import dataclasses
        self.assertEqual([f.name for f in dataclasses.fields(modes.KitMode)], ["kits", "exports_defaults", "levers", "env_overrides", "house_kits", "own_imports", "kernels", "kernels_replaced"])
        self.assertEqual(modes.KITS, (modes.KIT_XATTEMPT, modes.KIT_PARTNER, modes.KIT_SIZE_LEVERS, modes.KIT_FAST_LEVERS, modes.KIT_HOST_LEVERS))   # the five carried kit directories
        self.assertFalse(hasattr(modes, "SERVING_LEVERS")); self.assertFalse(hasattr(modes, "PACKER"))   # no packed form: the four modes are the whole surface
        self.assertEqual(modes.KIT_SWITCH_PREFIXES, ("BG_", "XA_", "SZ_", "FL_", "HL_"))
        self.assertEqual(modes.OBSERVABILITY_SWITCHES, ("BG_TIMING_FILE",))

    def test_fast_is_exacts_row_plus_the_fast_levers(self):
        """fast = exact's kits, defaults and levers (async_writer included) + cond_dedup / attn_bf16 / attn_cudnn / dit_fused switched on by
        its own exports; its own modules imported last (fl_levers, then hl_levers like every kit mode); the same runner."""
        ex, fa = modes.resolve("exact", HOME), modes.resolve("fast", HOME)
        self.assertEqual(fa.levers, ex.levers + ("cond_dedup", "attn_bf16", "attn_cudnn", "dit_fused"))
        self.assertIn("async_writer", ex.levers)
        self.assertEqual(fa.env, dict(ex.env, FL_COND_DEDUP="1", FL_ATTN_BF16="1", FL_ATTN_BACKEND="cudnn", FL_DIT_FUSED="1"))
        self.assertEqual(fa.kits, (modes.KIT_XATTEMPT, modes.KIT_PARTNER, modes.KIT_FAST_LEVERS, modes.KIT_SIZE_LEVERS, modes.KIT_HOST_LEVERS))
        self.assertEqual(ex.kits, (modes.KIT_XATTEMPT, modes.KIT_PARTNER, modes.KIT_SIZE_LEVERS, modes.KIT_HOST_LEVERS))
        self.assertEqual((fa.imports[:-3], fa.imports[-3:]), (ex.imports[:-2], ("fl_levers", "sz_levers", "hl_levers")))
        self.assertEqual(ex.imports[-2:], ("sz_levers", "hl_levers"))                          # every kit mode's own imports: the out-of-memory trace, then the background writer
        self.assertEqual(fa.runner, ex.runner)
        import inspect
        self.assertEqual(list(inspect.signature(modes.resolve).parameters), ["mode", "opt_home"])   # resolve(mode, opt_home): no variant argument
        for name, switch, probe in (("cond_dedup", "FL_COND_DEDUP", ("stats", "fl_levers", "dedup_enabled")), ("attn_bf16", "FL_ATTN_BF16", ("stats", "fl_levers", "attn_enabled")),
                                    ("attn_cudnn", "FL_ATTN_BACKEND", ("mode", "fl_levers", "attn_backend")), ("dit_fused", "FL_DIT_FUSED", ("stats", "fl_levers", "dit_enabled"))):
            lv = registry.LEVERS[name]
            self.assertEqual((lv.kit, lv.file, lv.switch, lv.tier, lv.probe), (modes.KIT_FAST_LEVERS, "src/fl_levers.py", switch, "2", probe))
            self.assertIn(name, stack.RUNNER_LINES); self.assertIn(name, stack.RUNNER_DISABLED_LINES)
        lv = registry.LEVERS["async_writer"]
        self.assertEqual((lv.kit, lv.file, lv.switch, lv.tier, lv.probe), (modes.KIT_HOST_LEVERS, "src/hl_levers.py", "HL_ASYNC_WRITER", "exact", ("stats", "hl_levers", "writer_enabled")))
        self.assertIn("async_writer", stack.RUNNER_LINES); self.assertIn("async_writer", stack.RUNNER_DISABLED_LINES)
        for m in ("exact", "fast", "big"):
            self.assertIn("async_writer", modes.KIT_MODES[m].levers); self.assertEqual(modes.resolve(m, HOME).env["HL_ASYNC_WRITER"], "1")
        self.assertTrue(os.path.isfile(os.path.join(stack.kit_src(modes.KIT_FAST_LEVERS), "fl_levers.py")))
        self.assertTrue(os.path.isfile(os.path.join(stack.kit_src(modes.KIT_HOST_LEVERS), "hl_levers.py")))

    def test_default_is_fast(self):
        self.assertEqual(modes.DEFAULT_MODE, "fast")
        self.assertEqual(modes.check_mode(None), "fast")
        self.assertEqual(modes.check_mode(""), "fast")
        self.assertEqual(modes.resolve(None, HOME).mode, "fast")                     # every verb resolves the one default

    def test_an_unknown_mode_is_refused_by_name(self):
        self.assertEqual(modes.check_mode("fast"), "fast")
        with self.assertRaises(ValueError) as cm:
            modes.check_mode("exact2")
        self.assertIn("off|exact|fast|big", str(cm.exception))

    def test_env_unset_or_empty_is_off(self):
        self.assertIsNone(modes.mode_from_env({}))
        self.assertIsNone(modes.mode_from_env({"BOLTZGEN_OPT": ""}))
        self.assertEqual(modes.mode_from_env({"BOLTZGEN_OPT": "exact"}), "exact")

    def test_switch_defaults_are_read_from_the_runner(self):
        runner = os.path.join(HOME, modes.KIT_XATTEMPT, modes.RUNNER)
        d = modes.kit_defaults(runner)
        self.assertEqual(d, {"BG_GRAPH": "graph", "XA_FAST_INIT": "1", "XA_HOIST": "1"})
        # the same three names appear on the runner's own line 19, in this order
        line = open(runner, encoding="utf-8").read().splitlines()[18]
        self.assertEqual(re.findall(r'setdefault\("([A-Z_]+)", "([a-z0-9]+)"\)', line), list(d.items()))

    def test_pythonpath_order_is_the_mode_tables(self):
        res = modes.resolve("exact", HOME)
        self.assertEqual(res.kits, (modes.KIT_XATTEMPT, modes.KIT_PARTNER, modes.KIT_SIZE_LEVERS, modes.KIT_HOST_LEVERS))
        self.assertEqual(modes.KIT_MODES["exact"].kits + modes.KIT_MODES["exact"].house_kits, res.kits)

    def test_imports_are_the_kits_own_launch(self):
        res = modes.resolve("exact", HOME)
        self.assertEqual(res.imports, ("bg_hook", "xa_fastinit", "xa_hoist", "sz_levers", "hl_levers"))   # the vendor imports in the kits' own order, then the mode's own modules
        sc = open(os.path.join(HOME, modes.KIT_PARTNER, modes.SITECUSTOMIZE), encoding="utf-8").read()
        self.assertIn("import bg_hook", sc)

    def test_exact_resolution(self):
        res = modes.resolve("exact", HOME)
        self.assertTrue(res.active)
        self.assertEqual(res.env, {"BG_GRAPH": "graph", "XA_FAST_INIT": "1", "XA_HOIST": "1", "HL_ASYNC_WRITER": "1"})
        self.assertEqual(res.levers, ("inproc", "graph_sampler", "fastinit", "hoist", "async_writer"))
        self.assertEqual(res.runner, os.path.join(modes.KIT_XATTEMPT, "src", "xa_run.py"))
        self.assertFalse(hasattr(res, "worker")); self.assertFalse(hasattr(res, "variant"))
        self.assertEqual(modes.describe_line(res), "BG_GRAPH=graph,XA_FAST_INIT=1,XA_HOIST=1,HL_ASYNC_WRITER=1")
        for k in res.kits:
            self.assertEqual(stack.kit_missing(k), [], k)
        self.assertFalse(os.path.exists(os.path.join(HOME, "serving")))          # no worker kit in the tree

    def test_off_resolution_sets_nothing(self):
        res = modes.resolve("off", HOME)
        self.assertFalse(res.active)
        self.assertEqual((res.env, res.imports, res.levers, res.kits), ({}, (), (), ()))
        self.assertIsNone(res.runner)
        self.assertEqual(modes.describe_line(res), "stock")

    def test_big_resolution(self):
        res = modes.resolve("big", HOME)
        self.assertTrue(res.active)
        self.assertEqual(res.env, {"BG_GRAPH": "off", "XA_FAST_INIT": "1", "XA_HOIST": "0", "SZ_TD_CHUNK": "64",
                                    "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True", "HL_ASYNC_WRITER": "1"})
        self.assertEqual(res.levers, ("inproc", "fastinit", "td_chunk", "async_writer"))
        self.assertNotIn("graph_sampler", res.levers)                     # deliberately off for this mode, not attempted-and-fallen-back
        self.assertNotIn("hoist", res.levers)
        self.assertEqual(res.kits, (modes.KIT_XATTEMPT, modes.KIT_PARTNER, modes.KIT_SIZE_LEVERS, modes.KIT_HOST_LEVERS))
        self.assertIn("sz_levers", res.imports)
        self.assertEqual(res.imports[:3], ("bg_hook", "xa_fastinit", "xa_hoist"))  # the vendor imports, unchanged order, then the house import
        self.assertEqual(res.imports[-2:], ("sz_levers", "hl_levers"))
        for k in (modes.KIT_XATTEMPT, modes.KIT_PARTNER, modes.KIT_SIZE_LEVERS, modes.KIT_HOST_LEVERS):
            self.assertEqual(stack.kit_missing(k), [], k)

    def test_report_dict_keys_are_one_set_for_every_mode(self):
        """A locked regression guard: no mode carries a report-dict key another lacks — the dict is serialized into opt_manifest.json,
        whose keys are one schema for every command and every mode."""
        base_keys = {"active", "dry_run", "mode", "form", "switches", "pythonpath", "imports", "levers_planned",
                     "boltzgen_version", "package_version", "gpu", "stack_key", "python", "torch",
                     "notes"}
        for name in modes.MODES:
            res = modes.resolve(name, HOME)
            rep = stack._base_report(res, None, dry_run=True)
            self.assertEqual(set(rep.keys()), base_keys, f"{name}: unexpected report keys {set(rep.keys()) - base_keys} or missing {base_keys - set(rep.keys())}")

    def test_excluded_values_never_in_a_mode_except_the_modes_own_declared_override(self):
        """A resolved mode's env never carries an excluded value UNLESS the mode's own table declares that exact
        override (big's BG_GRAPH=off / XA_HOIST=0 are the mode's own declared values; only specific VALUES of a
        switch are excluded, never the switch itself)."""
        for name, km in modes.KIT_MODES.items():
            res = modes.resolve(name, HOME)
            for k, v in res.env.items():
                if km.env_overrides.get(k) == v:
                    continue                                              # the mode's own declared row — reviewed, not a drift
                self.assertNotIn(v, modes.EXCLUDED_VALUES.get(k, ()), f"{name}: {k}={v} (not from env_overrides)")

    def test_excluded_values_checked_against_the_raw_runner_defaults(self):
        """The runner's raw setdefault() values (before ANY mode's own overrides) never carry an excluded value —
        this is the drift check env_overrides is layered on top of, never bypassed."""
        runner = os.path.join(HOME, modes.KIT_XATTEMPT, modes.RUNNER)
        raw = modes.kit_defaults(runner)
        for k, v in raw.items():
            self.assertNotIn(v, modes.EXCLUDED_VALUES.get(k, ()), f"raw runner default: {k}={v}")

    def test_every_mode_lever_is_in_the_registry_with_a_kit_file(self):
        """Every lever of `exact` and `big` is tier \"exact\"; `fast`'s own four levers are the tree's only tier-2 levers."""
        tier2 = {("fast", "cond_dedup"), ("fast", "attn_bf16"), ("fast", "attn_cudnn"), ("fast", "dit_fused")}
        for name, km in modes.KIT_MODES.items():
            for lever in km.levers:
                lv = registry.LEVERS[lever]
                self.assertTrue(os.path.isfile(os.path.join(HOME, lv.kit, lv.file)), f"{lever}: {lv.kit}/{lv.file}")
                self.assertIn(lv.klass, ("forward", "datapath", "serving", "orchestration"))
                self.assertEqual(lv.tier, "2" if (name, lever) in tier2 else "exact", (name, lever))
        self.assertEqual({(m, l) for m, km in modes.KIT_MODES.items() for l in km.levers if registry.LEVERS[l].tier != "exact"}, tier2)

    def test_every_registry_lever_belongs_to_a_mode(self):
        """The registry is the modes' lever vocabulary and nothing more: every registry lever is planned by some mode — no orphan row for a
        lever no mode can switch on."""
        planned = {l for km in modes.KIT_MODES.values() for l in km.levers}
        self.assertEqual(set(registry.LEVERS), planned)

    def test_registry_switches_are_accounted_for_by_some_mode(self):
        """Every registry switch belongs to at least one mode's exports — nothing in the registry is an orphan switch no mode ever sets."""
        switched = {lv.switch for lv in registry.LEVERS.values() if lv.switch}
        accounted = set()
        for name in modes.KIT_MODES:
            accounted |= set(modes.resolve(name, HOME).env)
        self.assertEqual(switched, switched & accounted, f"registry switches with no mode: {switched - accounted}")

    def test_exact_mode_switches_are_unchanged_by_the_big_addition(self):
        """exact's own resolved env is exactly the runner's three switches — big's new SZ_*/allocator switches and
        env_overrides never leak into exact (env_overrides is per-KitMode, not shared state)."""
        self.assertEqual(set(modes.resolve("exact", HOME).env), {"BG_GRAPH", "XA_FAST_INIT", "XA_HOIST", "HL_ASYNC_WRITER"})

    def test_mode_env_drops_caller_switches_and_orders_pythonpath(self):
        res = modes.resolve("exact", HOME)
        base = {"PATH": "/bin", "BG_GRAPH": "predraw", "XA_HOIST": "0", "BG_GRAPH_STEPS": "design,folding", "SZ_TD_CHUNK": "32", "FL_ATTN_BF16": "1", "HL_WRITER_MODE": "spawn",
                "BG_TIMING_FILE": "/t.jsonl", "BOLTZGEN_OPT": "exact", "PYTHONPATH": "/somewhere"}
        env, dropped = stack.mode_env(res, base)
        self.assertEqual(dropped, sorted(["BG_GRAPH", "XA_HOIST", "BG_GRAPH_STEPS", "SZ_TD_CHUNK", "FL_ATTN_BF16", "HL_WRITER_MODE", "PYTHONPATH"]))   # every caller-set switch under BG_/XA_/SZ_/FL_/HL_ is dropped and reported; BOLTZGEN_OPT=exact is handed over (the child activates the package to import the mode's own modules)
        self.assertFalse([k for k in env if k.startswith(("SZ_", "FL_")) or k == "HL_WRITER_MODE"])
        self.assertEqual({k: env[k] for k in res.env}, res.env)
        self.assertEqual(env["BG_TIMING_FILE"], "/t.jsonl")
        self.assertEqual((env["BOLTZGEN_OPT"], env.get(stack.ENV_HANDOVER)), ("exact", "1"))
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep), [stack.kit_src(modes.KIT_XATTEMPT), stack.kit_src(modes.KIT_PARTNER), stack.kit_src(modes.KIT_SIZE_LEVERS), stack.kit_src(modes.KIT_HOST_LEVERS)] + stack.package_roots())   # the kits' src dirs in the table's order, then the package's and the core's own directories (stack.package_roots)
        env2, _ = stack.mode_env(res, base, keep_pythonpath=True)
        self.assertEqual(env2["PYTHONPATH"].split(os.pathsep)[-1], "/somewhere")
        off_env, off_dropped = stack.mode_env(modes.resolve("off", HOME), base)
        self.assertNotIn("BG_GRAPH", off_env)

    def test_mode_env_in_process_form_keeps_the_package_switch(self):
        res = modes.resolve("exact", HOME)
        kits = [stack.kit_src(modes.KIT_XATTEMPT), stack.kit_src(modes.KIT_PARTNER), stack.kit_src(modes.KIT_SIZE_LEVERS), stack.kit_src(modes.KIT_HOST_LEVERS)]
        base = {"BG_GRAPH": "graph", "XA_HOIST": "0", "BOLTZGEN_OPT": "exact", "PYTHONPATH": os.pathsep.join(kits + ["/somewhere"])}
        env, dropped = stack.mode_env(res, base, keep_pythonpath=True, keep_package_env=True)
        self.assertEqual(dropped, ["XA_HOIST"])                                                # a caller value equal to the mode's export is not dropped
        self.assertEqual(env["BOLTZGEN_OPT"], "exact")                                          # the children of an activated process take the env route
        self.assertEqual(env["PYTHONPATH"].split(os.pathsep), kits + stack.package_roots() + ["/somewhere"])   # no duplicate kit entries down a process tree; the package roots follow the kits, the caller's own entries last
        env, dropped = stack.mode_env(res, base)
        self.assertEqual(dropped, ["PYTHONPATH", "XA_HOIST"])                                  # the process form: the mode is handed over (own_imports), the caller's PYTHONPATH and stray switch dropped

    def test_autoload_copies_equal_the_table(self):
        """_autoload.py is import-free at interpreter start, so it carries its own copies of the names it needs: held equal here."""
        from boltzgen_opt import _autoload, cli, codes
        self.assertEqual(tuple(_autoload.MODES), tuple(modes.MODES))
        self.assertEqual((_autoload.EXIT_NOT_ACTIVE, cli.EXIT_NOT_ACTIVE), (codes.EXIT_NOT_ACTIVE,) * 2)   # codes.py is the one definition (stack.py raises ActivationError; the exit is its callers')
        self.assertEqual((_autoload.ENV, _autoload.STEP_ENV, _autoload.ENV_KERNELS), (modes.ENV, stack.ENV_STEP, stack.ENV_KERNELS))
        self.assertFalse(hasattr(_autoload, "ENV_VARIANT"))
        self.assertEqual(tuple(_autoload.CPU_STEPS), tuple(stack.CPU_STEPS))
        self.assertEqual(_autoload.EXIT_NOT_ACTIVE, cli.EXIT_NOT_ACTIVE)
        self.assertEqual(_autoload.TRIGGER, stack.BOLTZ_MODULE.split(".")[0])

    def test_switch_table_is_read_from_the_kit_files_not_transcribed(self):
        """On a copy of the two kit files the table is read from, every edit changes the resolution (nothing is typed into modes.py)."""
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="bgopt_modes_")
        self.addCleanup(shutil.rmtree, tmp, True)
        files = {"runner": os.path.join(modes.KIT_XATTEMPT, modes.RUNNER), "sitecustomize": os.path.join(modes.KIT_PARTNER, modes.SITECUSTOMIZE)}

        def home(edits):
            h = os.path.join(tmp, str(len(os.listdir(tmp))))
            for key, rel in files.items():
                src = os.path.join(HOME, rel)
                text = open(src, encoding="utf-8").read()
                for old, new in edits.get(key, ()):
                    self.assertIn(old, text, f"{rel}: {old!r}")
                    text = text.replace(old, new, 1)
                dst = os.path.join(h, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                open(dst, "w", encoding="utf-8").write(text)
            return h

        res0 = modes.resolve("exact", home({}))                                                  # the untouched copies: the tree's own resolution
        self.assertEqual((res0.env, res0.imports, res0.kits), (modes.resolve("exact", HOME).env, modes.resolve("exact", HOME).imports, modes.resolve("exact", HOME).kits))
        res = modes.resolve("exact", home({"runner": [('setdefault("XA_HOIST", "1")', 'setdefault("XA_HOIST", "2")')]}))
        self.assertEqual(res.env["XA_HOIST"], "2")                                               # a runner default follows the runner's bytes
        res = modes.resolve("exact", home({"runner": [("import xa_fastinit", "import xa_fastinit2")]}))
        self.assertIn("xa_fastinit2", res.imports)                                               # the runner's lever imports follow the runner's bytes
        res = modes.resolve("exact", home({"sitecustomize": [("import bg_hook", "import bg_hook_v3")]}))
        self.assertEqual(res.imports[0], "bg_hook_v3")                                           # the hook import follows the partner's sitecustomize
        with self.assertRaises(ValueError) as cm:
            modes.resolve("exact", home({"runner": [('setdefault("BG_GRAPH", "graph")', 'setdefault("BG_GRAPH", "predraw")')]}))
        self.assertIn("a value outside the kit line", str(cm.exception))                          # a runner default outside the kit line is refused, not aliased

    def test_env_overrides_never_bypass_the_exclusion_check_for_a_value_the_mode_did_not_declare(self):
        """The two cases the exception mechanism must NOT swallow: (1) a runner default that drifted to an excluded
        value under a mode with NO override for that switch (exact) is still refused; (2) a runner default drifted to
        a DIFFERENT excluded value than the one a mode's own row declares (big declares BG_GRAPH=off; a raw runner
        default of BG_GRAPH=predraw is a different value it never declared) is still refused, checked before the
        mode's own override is ever applied."""
        import shutil
        import tempfile
        tmp = tempfile.mkdtemp(prefix="bgopt_modes_excl_")
        self.addCleanup(shutil.rmtree, tmp, True)

        def home_with_runner_edit(old, new):
            h = os.path.join(tmp, str(len(os.listdir(tmp)) if os.path.isdir(tmp) else 0))
            for rel in (modes.RUNNER,):
                src = os.path.join(HOME, modes.KIT_XATTEMPT, rel)
                text = open(src, encoding="utf-8").read()
                self.assertIn(old, text)
                text = text.replace(old, new, 1)
                dst = os.path.join(h, modes.KIT_XATTEMPT, rel)
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                open(dst, "w", encoding="utf-8").write(text)
            sc_src = os.path.join(HOME, modes.KIT_PARTNER, modes.SITECUSTOMIZE)
            sc_dst = os.path.join(h, modes.KIT_PARTNER, modes.SITECUSTOMIZE)
            os.makedirs(os.path.dirname(sc_dst), exist_ok=True)
            open(sc_dst, "w", encoding="utf-8").write(open(sc_src, encoding="utf-8").read())
            return h

        # (1) a runner default drifted to BG_GRAPH=off under exact (exact's env_overrides is empty: no exemption) is refused
        h1 = home_with_runner_edit('setdefault("BG_GRAPH", "graph")', 'setdefault("BG_GRAPH", "off")')
        with self.assertRaises(ValueError) as cm:
            modes.resolve("exact", h1)
        self.assertIn("a value outside the kit line", str(cm.exception))

        # (2) a runner default drifted to BG_GRAPH=predraw under big is STILL refused, even though big's own row
        #     later sets BG_GRAPH=off — the check runs on the raw runner default, before env_overrides is applied
        h2 = home_with_runner_edit('setdefault("BG_GRAPH", "graph")', 'setdefault("BG_GRAPH", "predraw")')
        with self.assertRaises(ValueError) as cm:
            modes.resolve("big", h2)
        self.assertIn("a value outside the kit line", str(cm.exception))

        # sanity: big's OWN declared override (BG_GRAPH=off, from an UNMODIFIED runner default of "graph") still resolves fine
        res = modes.resolve("big", HOME)
        self.assertEqual(res.env["BG_GRAPH"], "off")


if __name__ == "__main__":
    unittest.main()
