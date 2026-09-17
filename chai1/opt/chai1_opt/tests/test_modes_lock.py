"""The mode table (modes.KIT_MODES) is locked to the kits' own spellings: every kit mode runs the driver at its default --levels and the
eager stack's `tier1` (the stack's own LEVERS) + the DSTEP lever `hoist2` (the add-on's own ALL_LEVERS: hoist2, compiled, dit_attn); the driver accepts exactly
the levers the modes name; fast at any fold settings (KitMode.stack documents the pinned stack); the default mode `fast` (modes.DEFAULT_MODE;
CHAI1_OPT unset = no activation) in the package and in run.sh (executed, with a stub python recording what run.sh hands the package); the
refusal rules. No torch, no GPU."""
import glob
import os
import re
import stat
import subprocess
import sys
import tempfile
import unittest

from chai1_opt import _autoload, cli, modes, registry, settings, stack, warm
from chai1_opt.tests import _kitspell

DRIVER = stack.driver_path()
EAGER_STACK = stack.eager_stack_path()
DSTEP_STACKX = stack.dstep_stackx_path()

PKG = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))            # opt/chai1_opt
OPT = os.path.dirname(PKG)                                                    # opt/
TREE = os.path.dirname(OPT)                                                   # chai1/


def _modules():
    return {os.path.basename(f): open(f, encoding="utf-8").read() for f in glob.glob(os.path.join(PKG, "*.py"))}


class TestModeTable(unittest.TestCase):
    def test_row_levers_every_name_a_mode_composes(self):
        """modes.row_levers: the names MODEL_OPT_LEVERS_OFF resolves against, per row (driver + eager + DSTEP + pair-track + memory line + gated + implied)."""
        self.assertEqual(modes.row_levers(modes._ROWS["exact"]), ("W1", "W2", "W5", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo", "prefetch", "msa_chunk", "nograph", "alloc"))
        self.assertEqual(modes.row_levers(modes._ROWS["fast"]), ("W1", "W2", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo", "prefetch", "msa_rows", "msa_chunk", "nograph", "tf32", "alloc"))
        self.assertEqual(modes.row_levers(modes._ROWS["big"]), ("W1", "W2", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo", "prefetch", "msa_rows", "msa_chunk", "trunk_chunk", "opm_chunk", "nograph", "tf32", "alloc"))
        self.assertEqual(set(modes.NOT_SWITCHABLE), {"tier1", "W1"}); self.assertEqual(modes.ENV_LEVERS_OFF, "MODEL_OPT_LEVERS_OFF")

    def test_rows(self):
        self.assertEqual(modes.MODES, ("off", "exact", "fast", "big"))
        self.assertEqual(modes.KIT_MODES["exact"].levels, ("W1", "W2", "W5"))
        self.assertEqual(modes.KIT_MODES["fast"].levels, ("W1", "W2", "W5"))
        self.assertEqual(modes.driver_levels("exact"), "W1,W2,W5")
        self.assertEqual(modes.driver_levels("fast"), "W1,W2,W5")
        self.assertEqual(modes.KIT_MODES["exact"].eager, "tier1"); self.assertEqual(modes.KIT_MODES["fast"].eager, "tier1")
        self.assertEqual(modes.KIT_MODES["exact"].pairtrack, ("templ_empty", "exactln", "msa_pad", "transition")); self.assertEqual(modes.KIT_MODES["fast"].pairtrack, ("templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n"))
        self.assertEqual(modes.KIT_MODES["exact"].lever_names, ("W1", "W2", "W5", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo", "prefetch"))
        self.assertEqual(modes.KIT_MODES["fast"].lever_names, ("W1", "W2", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo", "prefetch"))
        self.assertEqual(modes.KIT_MODES["exact"].dstep, ("hoist2",)); self.assertEqual(modes.KIT_MODES["fast"].dstep, ("hoist2", "compiled", "dit_attn")); self.assertEqual(modes.KIT_MODES["big"].dstep, ("hoist2", "compiled", "dit_attn"))
        self.assertEqual(modes.kit_mode("fast").dstep, ("hoist2", "compiled", "dit_attn")); self.assertEqual(modes.KIT_MODES["fast"].dstep_arg, "hoist2,compiled,dit_attn"); self.assertEqual(modes.KIT_MODES["exact"].dstep_arg, "hoist2")
        # the default is a value (fast: the house rule, default = fast wherever fast ships), never a rule computed from the table;
        # fast runs at any fold settings; its kernels need a C compiler on the box — refused by name without one (stack.mode_stack_check)
        self.assertIn(modes.DEFAULT_MODE, modes.MODES)
        self.assertEqual(modes.DEFAULT_MODE, "fast")
        self.assertIn('DEFAULT_MODE: str = "fast"', open(modes.__file__, encoding="utf-8").read())
        fast = modes.KIT_MODES["fast"]
        self.assertIsNone(modes.refusal("fast"))
        self.assertEqual(modes.DEFAULT_MODE, "fast")
        self.assertEqual(modes.KIT_MODES["exact"].stack, modes.PINNED_STACK); self.assertEqual(modes.KIT_MODES["exact"].stack_rule, "bitwise")                 # exact: bitwise vs stock on one stack, tested on the pinned stack
        self.assertEqual(fast.stack, modes.PINNED_STACK); self.assertEqual(fast.stack_rule, "triton"); self.assertRegex(modes.PINNED_STACK, r"^torch \d+\.\d+\.\d+\+cu\d+$")
        self.assertEqual(modes.DEFAULT_MODE, "fast")
        self.assertFalse(hasattr(modes.KIT_MODES["fast"], "presets"))
        big = modes.KIT_MODES["big"]                                     # composes fast by reference; its own levers are the memory line + implied alloc
        self.assertEqual((big.levels, big.eager, big.dstep, big.stack), (fast.levels, fast.eager, fast.dstep, fast.stack))
        self.assertEqual(big.memory, ("msa_rows", "msa_chunk", "trunk_chunk", "opm_chunk", "nograph")); self.assertEqual(big.implied_optin, ("tf32", "alloc"))
        self.assertEqual(modes.KIT_MODES["exact"].memory, ()); self.assertEqual(fast.memory, ()); self.assertEqual(fast.implied_optin, ("tf32", "alloc"))
        self.assertEqual(big.pairtrack, fast.pairtrack); self.assertEqual(modes.KIT_MODES["exact"].implied_optin, ("alloc",))   # the allocator policy rides every kit row (K.38)
        self.assertEqual(set(big.in_process) - set(fast.in_process), set(big.memory) | {"alloc"})

    def test_exact_is_the_drivers_own_default(self):
        self.assertEqual(_kitspell.driver_default_levels(DRIVER), modes.driver_levels("exact"))

    def test_the_driver_accepts_exactly_the_modes_levers(self):
        accepted = set(_kitspell.driver_accepted_levers(DRIVER))              # the driver refuses any other --levels name (ap.error, rc 2)
        self.assertEqual(accepted, set(modes.DRIVER_LEVERS))
        for km in modes.KIT_MODES.values():
            self.assertEqual(set(km.levels), set(modes.KIT_MODES["exact"].levels)); self.assertTrue(set(km.levels) <= accepted)

    def test_eager_levers_are_the_stacks_own(self):
        own = _kitspell.eager_stack_levers(EAGER_STACK)
        self.assertEqual(own, ("stock", "tier1"))
        self.assertEqual(modes.EAGER_LEVERS, tuple(n for n in own if n != "stock"))
        for km in modes.KIT_MODES.values():
            self.assertIn(km.eager, (None,) + modes.EAGER_LEVERS)
        self.assertEqual(modes.KIT_MODES["exact"].eager, "tier1")          # the bitwise row carries the eager stack's tier1 line (bitwise vs stock under the recipe)
        self.assertEqual(modes.KIT_MODES["fast"].eager, "tier1")
        self.assertEqual(set(registry.EAGER_LEVERS), set(modes.EAGER_LEVERS))

    def test_dstep_levers_are_the_addons_own(self):
        own = _kitspell.dstep_all_levers(DSTEP_STACKX)
        self.assertEqual(own, ("hoist2", "compiled", "dit_attn")); self.assertEqual(modes.DSTEP_LEVERS, own)                    # hoist2 in every row; compiled, dit_attn in fast / big
        for km in modes.KIT_MODES.values():
            for n in km.dstep:
                self.assertIn(n, own)
            if km.dstep:
                self.assertEqual(km.eager, "tier1")                        # the add-on's levers ride the tier1 line (DSTEP README.md)
        self.assertEqual(set(registry.DSTEP_LEVERS), set(modes.DSTEP_LEVERS))
        for lv in registry.EAGER_LEVERS.values():
            self.assertEqual(lv.kit_file, modes.EAGER_STACK_RELPATH.replace(os.sep, "/")); self.assertEqual(lv.probe, ("eager", lv.name))
            self.assertFalse(lv.default_on)

    def test_every_lever_name_is_the_drivers(self):
        names = set(_kitspell.driver_lever_names(DRIVER))                   # W1 / W5 at import, W2 in the loop (`"Wn" in levels` tests)
        for km in modes.KIT_MODES.values():
            self.assertTrue(set(km.levels) <= names, (km.name, km.levels, names))
        self.assertEqual(set(registry.LEVERS), set(modes.DRIVER_LEVERS) | set(modes.EAGER_LEVERS) | set(modes.DSTEP_LEVERS) | set(modes.OPTIN_LEVERS) | set(modes.PAIRTRACK_LEVERS) | set(modes.POSTPROC_LEVERS) | set(modes.KIT_MODES["big"].memory))
        from chai1_opt import featfast, pairtrack, postproc
        self.assertEqual(modes.POSTPROC_LEVERS, postproc.LEVERS + featfast.LEVERS); self.assertEqual(set(registry.POSTPROC_LEVERS), set(modes.POSTPROC_LEVERS))   # the host-side levers ride every kit row, by the modules' own names
        for km in modes.KIT_MODES.values():
            self.assertEqual(km.postproc, modes.POSTPROC_LEVERS, km.name); self.assertTrue(set(postproc.LEVERS) <= set(km.in_process)); self.assertNotIn("prefetch", km.in_process)   # prefetch is the driver loop's (the next uid), like W2
        self.assertEqual(modes.PAIRTRACK_LEVERS, pairtrack.LEVERS); self.assertEqual(set(registry.PAIRTRACK_LEVERS), set(pairtrack.LEVERS)); self.assertEqual(set(registry.PAIRTRACK_PROBES), set(pairtrack.LEVERS))
        for km in modes.KIT_MODES.values():
            self.assertTrue(set(km.pairtrack) <= set(pairtrack.LEVERS), (km.name, km.pairtrack))
            self.assertTrue(set(km.pairtrack) <= set(km.in_process))          # the trunk levers install in-process like the eager line they ride
            if km.pairtrack:
                self.assertEqual(km.eager, "tier1")                            # they attach to the eager trunk
        self.assertEqual(pairtrack.EXPECTED_FALLBACKS, {"v4trimul": (), "templ_empty": ("templates_present",), "exactln": ("statement", "refused"), "triattn": ("stock_statement", "no_cell", "row_error"), "msa_pad": ("no_tail",), "transition": ("stock", "no_cell", "refused", "class_differs"), "trunk_n": ("no_pad", "layout")})   # the refusal words each trunk lever counts as expected; any other word refuses the run at exit
        self.assertEqual(set(registry.MEMORY_LEVERS), set(modes.KIT_MODES["big"].memory)); self.assertEqual(set(registry.BIG_PROBES), set(registry.MEMORY_LEVERS))
        self.assertEqual(set(registry.DRIVER_LEVERS), set(modes.DRIVER_LEVERS))
        self.assertEqual(set(registry.OPTIN_LEVERS), set(modes.OPTIN_LEVERS))
        self.assertEqual(set(registry.FLAG_PROBES) | set(registry.ENV_PROBES), set(modes.OPTIN_LEVERS))
        self.assertEqual(set(modes.OPTIN_LEVERS), {"tf32", "alloc"})                 # the package's implied levers (a row carries them as KitMode.implied_optin)
        self.assertEqual({n: oi.modes for n, oi in modes.OPTIN_LEVERS.items()}, {"tf32": ("fast", "big"), "alloc": ("exact", "fast", "big")})   # numerics-changing levers never under exact; none under off; alloc implied by big
        for n, oi in modes.OPTIN_LEVERS.items():
            if "fast" in oi.modes:
                self.assertIn("big", oi.modes, n)                                  # big carries every fast lever, fast's opt-ins included
        for name, km in modes.KIT_MODES.items():
            self.assertIsNone(modes.optin_refusal(name, km.implied_optin), name)       # a row admits its own implied opt-ins (fast: tf32; big: tf32 + alloc)

        self.assertEqual(set(registry.FAMILY), set(registry.LEVERS))
        for n, oi in modes.OPTIN_LEVERS.items():
            self.assertEqual(oi.name, n); self.assertFalse(registry.LEVERS[n].default_on)
            self.assertTrue(oi.modes and set(oi.modes) <= set(modes.KIT_MODES), (n, oi.modes))
            for km in modes.KIT_MODES.values():
                self.assertNotIn(n, km.lever_names)
        self.assertEqual({n for n, lv in registry.DRIVER_LEVERS.items() if lv.default_on}, set(modes.KIT_MODES["exact"].levels))

    def test_every_mode_runs_at_any_fold_settings(self):
        for km in modes.KIT_MODES.values():
            self.assertFalse(hasattr(km, "presets"))                                    # no settings restriction on a mode: stock's flags pass through every one
        for m in modes.MODES:
            self.assertIsNone(modes.refusal(m))

    def test_in_process_forms(self):
        self.assertEqual(modes.KIT_MODES["exact"].in_process, ("W1", "W5", "tier1", "hoist2", "templ_empty", "exactln", "msa_pad", "transition", "rankcc", "tailasync", "confmemo"))
        self.assertEqual(modes.KIT_MODES["fast"].in_process, ("W1", "W5", "tier1", "hoist2", "compiled", "dit_attn", "templ_empty", "v4trimul", "exactln", "triattn", "msa_pad", "transition", "trunk_n", "rankcc", "tailasync", "confmemo"))
        for m in modes.KIT_MODE_NAMES:
            self.assertIsNone(modes.refusal(m, in_process=True))
            self.assertNotIn("W2", modes.KIT_MODES[m].in_process)         # the driver loop's lever: not applicable in a program of its own

    def test_in_process_levers_are_the_kit_blocks(self):
        self.assertEqual(_kitspell.kit_lever_names_in_block(DRIVER), ("W1", "W5"))
        nodes, text = stack.kit_lever_statements(DRIVER)
        self.assertIn("chai1.load_exported = load_exported_cached", text)
        self.assertIn("esm_mod.esm_model = esm_model_resident", text)
        self.assertIn("esm_mod._get_esm_contexts_for_sequences = _get_ctx_cached", text)
        self.assertNotIn("run_folding", text)                          # the block installs attributes only, never folds


class TestDefaultMode(unittest.TestCase):
    def test_cli_resolves_the_default_mode(self):
        os.environ.pop(modes.ENV, None)
        self.assertEqual(cli.resolve_mode(None), modes.DEFAULT_MODE)
        self.assertEqual(cli.resolve_mode("exact"), "exact"); self.assertEqual(cli.resolve_mode("off"), "off")
        with self.assertRaises(cli.CliError):
            cli.resolve_mode("turbo")
        os.environ[modes.ENV] = "fast"
        try:
            self.assertEqual(cli.resolve_mode(None), "fast")
            with self.assertRaises(cli.CliError):
                cli.resolve_mode("exact")
        finally:
            os.environ.pop(modes.ENV, None)

    def _run_sh(self, args, env_extra):
        """Execute run.sh with a stub `python` first on PATH that passes `-c` through to this interpreter, answers the pin check with
        rc 0 and records the argv run.sh hands `python -m chai1_opt` (nothing else runs)."""
        tmp = tempfile.mkdtemp(prefix="chai1_opt_runsh_")
        rec = os.path.join(tmp, "argv.txt")
        stub = os.path.join(tmp, "python")
        with open(stub, "w") as fh:
            fh.write("#!/bin/bash\n"
                     f'if [ "$1" = -m ] && [ "$2" = chai1_opt ]; then printf \'%s\\n\' "$@" > "{rec}"; exit 0; fi\n'
                     'if [ "$1" = -I ] && [[ "$2" == *check_pins.py ]]; then exit 0; fi\n'
                     f'exec "{sys.executable}" "$@"\n')
        os.chmod(stub, os.stat(stub).st_mode | stat.S_IEXEC)
        env = {k: v for k, v in os.environ.items() if k != modes.ENV}
        env["PATH"] = tmp + os.pathsep + env.get("PATH", ""); env.update(env_extra)
        r = subprocess.run(["bash", os.path.join(stack.tree_home(), "run.sh")] + args, capture_output=True, text=True, env=env, cwd=tmp)
        argv = open(rec).read().splitlines() if os.path.isfile(rec) else None
        return r.returncode, r.stderr, argv

    def test_run_sh_resolves_the_mode_like_the_package(self):
        pred = ["pred", "--input", "x.fasta", "--out_dir", "o"]
        rc, err, argv = self._run_sh(pred, {})                                        # no --mode, no CHAI1_OPT: the package default
        self.assertEqual(rc, 0, err); self.assertIn("the package default mode, " + modes.DEFAULT_MODE, err)
        self.assertEqual(argv, ["-m", "chai1_opt", "pred", "--mode", modes.DEFAULT_MODE] + pred[1:])
        rc, err, argv = self._run_sh(["check", "--mode", "fast"], {})
        self.assertEqual(rc, 0, err); self.assertEqual(argv, ["-m", "chai1_opt", "check", "--mode", "fast"]); self.assertNotIn("default", err)
        rc, err, argv = self._run_sh(["check"], {modes.ENV: "off"})
        self.assertEqual(rc, 0, err); self.assertEqual(argv, ["-m", "chai1_opt", "check", "--mode", "off"])
        rc, err, argv = self._run_sh(["check", "--mode", "exact"], {modes.ENV: "fast"})
        self.assertEqual(rc, 2); self.assertIn("disagrees", err); self.assertIsNone(argv)
        rc, err, argv = self._run_sh(["check", "--mode", "turbo"], {})
        self.assertEqual(rc, 2); self.assertIn("not a mode", err); self.assertIsNone(argv)
        rc, err, argv = self._run_sh(["bogus"], {modes.ENV: "off"})                                 # not a verb of this kit: usage, exit 2
        self.assertEqual(rc, 2); self.assertIn("run.sh pred", err); self.assertIsNone(argv)

    def test_run_sh_spells_no_mode_of_its_own(self):
        sh = open(os.path.join(stack.tree_home(), "run.sh"), encoding="utf-8").read()
        self.assertEqual(sh.count("MODE=${MODE:-$ENVMODE}"), 1)
        self.assertNotRegex(sh, r"\{[A-Z]+:-(exact|fast|off)\}", "run.sh must not carry a default mode of its own")
        self.assertIn("# Modes: " + " | ".join(modes.MODES) + " (opt/chai1_opt/modes.py MODES)", sh)   # the usage header names the modes, in MODES order
        self.assertIn("print(modes.DEFAULT_MODE)", sh)                            # the default is read from the package's one table
        self.assertRegex(sh, r'case "\$MODE" in off\|exact\|fast\|big\)')


class TestOneTableEach(unittest.TestCase):
    """One copy of each concern: the mode table, the settings table, the stock caller, the deterministic recipe, the driver launcher, the
    command list, and the documented pairs (_autoload.MODES <-> modes.KIT_MODE_NAMES, warm.WARM_FOLD within settings.FOLD_KEYS). Source-level,
    line-anchored over every line of every module."""
    def test_one_mode_table(self):
        """modes.KIT_MODES is the only place a --levels list is spelled in package code."""
        for f, s in _modules().items():
            if f == "modes.py":
                continue
            self.assertNotRegex(s, r'["\']W1,W2,W5["\']', f"{f} spells a lever list: modes.KIT_MODES is the one table")
            names = "KIT_MODES|DEFAULT_MODE" if f == "_autoload.py" else "KIT_MODES|MODES|DEFAULT_MODE"   # _autoload.MODES: the documented pair below
            self.assertNotRegex(s, re.compile(rf"^({names})\s*=", re.M), f"{f} carries a second mode table")
        self.assertEqual(tuple(modes.MODES), ("off", "exact", "fast", "big"))

    def test_documented_pairs(self):
        """The second copies that must exist are locked to their source: _autoload.MODES (the .pth path imports nothing at start) ==
        modes.KIT_MODE_NAMES; warm's fold constant names run_inference keywords the kit driver serves; the registry spells no lever composition of its own."""
        self.assertEqual(tuple(_autoload.MODES), tuple(modes.KIT_MODE_NAMES))
        self.assertEqual(_autoload.ENV, modes.ENV)
        self.assertLessEqual(set(warm.WARM_FOLD), set(settings.FOLD_KEYS)); self.assertEqual(settings.to_driver_run_kw(settings.from_values(warm.WARM_FOLD))["num_diffn_samples"], 1); self.assertFalse(hasattr(warm, "WARM_PRESET"))
        self.assertFalse(hasattr(registry, "BITWISE"))
        self.assertEqual({n for n, lv in registry.DRIVER_LEVERS.items() if lv.default_on}, set(modes.KIT_MODES["exact"].levels))

    def test_one_settings_table(self):
        for f, s in _modules().items():
            if f == "settings.py":
                continue
            self.assertNotRegex(s, re.compile(r"^(FOLD_KEYS|STOCK_FLAGS|DRIVER_FIXED)\s*=", re.M), f"{f} carries a second settings table")
            if f != "warm.py":                                                            # warm's WARM_FOLD constant is its own (not a command-line choice)
                self.assertNotRegex(s, r"num_trunk_recycles\s*=\s*\d", f"{f} spells a fold setting: settings.py resolves them from stock's flags")
            self.assertNotRegex(s, r"add_argument\(\"--settings\"", f"{f} carries a --settings flag")

    def test_one_stock_caller_one_recipe_one_launcher(self):
        mods = _modules()
        callers = [f for f, s in mods.items() if "run_inference(" in s and f not in ("settings.py",)]
        self.assertEqual(callers, ["stock_fold.py"], "stock_fold.py is the one stock caller")
        recipes = [f for f, s in mods.items() if "use_deterministic_algorithms(" in s]
        self.assertEqual(recipes, ["det.py"], "det.py is the one deterministic recipe")
        launchers = [f for f, s in mods.items() if "exec(" in s and '"__name__": "__main__"' in s]
        self.assertEqual(launchers, ["driver.py"], "driver.py is the one place the kit driver runs as __main__")
        blocks = [f for f, s in mods.items() if "kit_lever_statements()" in s and "exec(" in s]
        self.assertEqual(blocks, ["stack.py"], "stack.py is the one place the kit's W1/W5 statements run in-process")
        self.assertNotRegex(mods["stock_fold.py"], r"import (chai_proto|chai_worker)")

    def test_no_kit_code_transcribed(self):
        """The three kit functions the package relies on are called or exec'd from the kit file, never restated."""
        mods = _modules()
        for f, s in mods.items():
            self.assertNotIn("def load_exported_cached", s, f); self.assertNotIn("def esm_model_resident", s, f)
            self.assertNotIn("def _get_ctx_cached", s, f); self.assertNotIn("def fasta_spec", s, f)


class TestRunShAndConfig(unittest.TestCase):
    def setUp(self):
        self.sh = open(os.path.join(TREE, "run.sh"), encoding="utf-8").read()
        self.cfg = open(os.path.join(TREE, "configs", "h100.env"), encoding="utf-8").read()

    def test_commands_in_step(self):
        m = re.search(r'case "\$CMD" in ([a-z|]+)\)', self.sh)
        self.assertEqual(set(m.group(1).split("|")), set(cli.COMMANDS) | {"install"})   # the package's verbs + `install`, the script's own step (pip: the core, then the kit; then the pin check) — the package cannot install itself
        self.assertIn('python -m pip install -e "$HERE/../common/opt_core" -e "$HERE/opt"', self.sh)
        self.assertIn('python -I "$HERE/stock/check_pins.py"', self.sh)

    def test_modes_in_step(self):
        m = re.search(r'case "\$MODE" in ([a-z|]+)\)', self.sh)
        self.assertEqual(set(m.group(1).split("|")), set(modes.MODES))
        self.assertIn("python -m chai1_opt \"$CMD\" --mode \"$MODE\"", self.sh)
        self.assertIn("check_pins.py", self.sh)

    def test_config_carries_no_lever_switch(self):
        exported = re.findall(r"^export ([A-Z0-9_]+)=", self.cfg, re.M)
        self.assertEqual(sorted(exported), ["MODEL_OPT", "MODEL_OPT_TARGET_GPU", "PYTHONDONTWRITEBYTECODE"])   # the weights directory CHAI_DOWNLOADS_DIR is the caller's
                                                                                                                # variable (README, Variables), never a default here
        for name in ("CHAI1_OPT", "CHAI_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG", "CHAI_JIT_PROFILING_OFF", "--levels", "W1"):
            self.assertNotRegex(self.cfg, rf"^export {re.escape(name)}", name)
            self.assertNotRegex(self.cfg, rf"export {re.escape(name)}=", name)
        self.assertIn("import chai1_opt", self.cfg)

    def test_one_config(self):
        self.assertEqual(sorted(os.listdir(os.path.join(TREE, "configs"))), ["a100.env", "h100.env", "h200.env"])   # a100.env = h100.env with MODEL_OPT_TARGET_GPU pre-set (test_locks.TestNoPlatformCoupling.test_configs_one_shape)


class TestKitDirectoryHasNoHouseFile(unittest.TestCase):
    def test_house_files_outside_the_kit(self):
        kit = stack.kit_home()
        self.assertTrue(os.path.samefile(kit, os.path.join(OPT, "forward", "fast_inference")))
        self.assertFalse(glob.glob(os.path.join(kit, "**", "*chai1_opt*"), recursive=True))     # no house file leaks into the vendored kit dir
