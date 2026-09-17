"""The stock route (`design --mode off`) on the stubbed upstream: configure in a clean environment with the caller's own configure
arguments (a spec's keys are upstream's to read — nothing of it is refused here), one clean subprocess per step under the shared core's
stock-proof contract with BOLTZGEN_PIPELINE_STEP set — seeded `seed + i` when the caller passed `--seed`, upstream's own unseeded run when
not (the seed slot's word `none`, the ENV-CLEAN line's `seed=none`) —, the core's proof written and clean even when the caller's environment
carries kit switches, the manifest beside the outputs; the stock caller refuses (exit 3, nothing runs) when a kit variable, a kit directory
on sys.path or the armed finder is present; the seed recipe is applied (the stub records the last seed)."""
import ast
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

from opt_core import stock_proof

from boltzgen_opt import codes, design, modes, stack, stock_design
from boltzgen_opt.tests import _stubs

SPEC = "entities:\n  - protein:\n      id: B\n      sequence: 60..80\n"
SPEC_MSA = ("entities:\n  - protein:\n      id: A\n      sequence: MKTAYIAKQR\n      msa: ./a.a3m\n  - protein:\n      id: B\n      sequence: 60..80\n"
            "templates:\n  - cif: ./t.cif\n")
CLEAN = "[boltzgen-opt stock] environment proven clean; step={step} seed={seed} config="


class TestStockRoute(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="bgopt_stock_")
        cls.site = _stubs.materialize(cls.tmp)
        cls.spec = os.path.join(cls.tmp, "spec.yaml")
        open(cls.spec, "w").write(SPEC)
        try:                                                                     # the stub upstream importable in the stock child, which strips PYTHONPATH (a site .pth, as the
            cls.pth = _stubs.site_pth(cls.site)                                  # installed wheel is on a box); the package itself reaches the child by design.stock_preamble
        except RuntimeError as e:                                                # (this tree's roots first on its sys.path) — no install of it is needed for the stock arm
            raise unittest.SkipTest(str(e))

    @classmethod
    def tearDownClass(cls):
        _stubs.remove_site_pth(cls.pth)
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def cli(self, args, **env_more):
        env = _stubs.clean_env(self.site, **env_more)
        r = subprocess.run([sys.executable, "-m", "boltzgen_opt"] + args, env=env, capture_output=True, text=True, timeout=300, cwd=self.tmp)
        return r.returncode, r.stdout, r.stderr

    def test_design_off_runs_every_step_clean_and_seeded(self):
        out = os.path.join(self.tmp, "off_run")
        rc, so, se = self.cli(["design", "--mode", "off", self.spec, "--output", out, "--seed", "7", "--num_designs", "2", "--steps", "design", "inverse_folding"],
                              BG_GRAPH="graph", XA_HOIST="1", SZ_TD_CHUNK="8", BOLTZGEN_OPT="off", BOLTZGEN_CACHE=os.path.join(self.tmp, "hf"))
        self.assertEqual(rc, 0, se)
        pins = stack.pins()["stock_environment"]
        log = open(os.path.join(out, "opt_run.log"), encoding="utf-8").read()
        for i, step in enumerate(("design", "inverse_folding")):
            done = json.load(open(os.path.join(out, step + ".done")))
            self.assertEqual(done["step_env"], step)
            self.assertEqual(done["kit_modules"], [])
            self.assertEqual(done["env_kit"], [])
            self.assertIsNone(done["pythonpath"])
            self.assertEqual(done["seeded"], 7 + i)                              # step i runs with seed + i
            proof = json.load(open(os.path.join(out, f"stock_env_proof_{step}.json")))
            self.assertTrue(proof["ok"], proof)
            self.assertTrue({"ok", "env_absent", "forbidden_present", "kit_modules_loaded", "kit_dirs_on_path", "autoload_armed", "kit_sitecustomize",
                             "torch_loaded_before_proof", "core_modules_loaded", "no_user_site", "det_exception", "step", "pid"} <= set(proof), sorted(proof))   # the core's proof (opt_core.stock_proof.env_proof) plus the caller's step and pid
            self.assertEqual((proof["step"], proof["no_user_site"], proof["forbidden_present"], proof["kit_dirs_on_path"], proof["autoload_armed"], proof["torch_loaded_before_proof"], proof["det_exception"]),
                             (step, True, [], [], [], False, None))
            self.assertEqual(proof["env_absent"], list(pins["must_be_absent_prefixes"]) + list(pins["must_be_absent_names"]))   # every must-be-absent prefix and name of stock/PINS.json, proven by the child itself
            self.assertIn(CLEAN.format(step=step, seed=7 + i) + os.path.join(out, "config", step + ".yaml") + " absent=", log)   # the kit's ENV-CLEAN line with the core's clean sentence
        self.assertIn(" kit_modules=none kit_dirs=none no_user_site=True", log)
        self.assertFalse(os.path.exists(os.path.join(out, "folding.done")))
        conf = open(os.path.join(out, "configure_argv.txt")).read().splitlines()
        self.assertEqual(conf[0], f"configure {self.spec} --output {out} --cache {os.path.join(self.tmp, 'hf')} --num_designs 2 --steps design inverse_folding")   # `--cache $BOLTZGEN_CACHE` (the caller named none), then the caller's flags as given
        self.assertEqual(conf[1], "ENV ")                                    # configure ran with no kit switch, no PYTHONPATH, no BOLTZGEN_OPT
        man = json.load(open(os.path.join(out, "opt_manifest.json")))
        self.assertEqual((man["mode"], man["form"], man["active"], man["exit_code"], man["seed"], man["command"]), ("off", "process", False, 0, 7, "design"))
        self.assertNotIn("settings", man)                                      # no presets: the configure line is the caller's
        self.assertEqual(man["configure"]["cmd"][-5:], ["--num_designs", "2", "--steps", "design", "inverse_folding"])   # the caller's own order
        self.assertEqual(man["configure"]["kernels_resolution"]["line"], "Using kernels: False [device capability: (0, 0)]")   # the stub box has no card: upstream ships the kernels off, the census expects off
        self.assertEqual(man["boltzgen"]["pin"], "0.3.2")
        self.assertEqual(len(man["activation_report"]["stock_env_proof"]), 2)
        self.assertEqual([s["seed"] for s in man["run"]["steps"]], [7, 8])
        self.assertIn("[boltzgen-opt] NOT ACTIVE: mode off: stock", se)
        self.assertEqual(se.count("[boltzgen-opt stock] environment proven clean; "), 2)   # the children's lines pass through the caller's stderr

    def test_design_off_without_seed_is_upstreams_unseeded_run(self):
        """No `--seed` on the stock route: each step runs unseeded (upstream's own form) — the stock child gets the word `none` in the seed
        slot, skips the seed line and says `seed=none`; the manifest records seed null."""
        out = os.path.join(self.tmp, "off_unseeded")
        rc, so, se = self.cli(["design", "--mode", "off", self.spec, "--output", out, "--num_designs", "2", "--steps", "design", "inverse_folding"])
        self.assertEqual(rc, 0, se)
        self.assertNotIn("NOT ACTIVE: mode off runs", se)                    # a kit mode needs --seed; off never does
        log = open(os.path.join(out, "opt_run.log"), encoding="utf-8").read()
        for step in ("design", "inverse_folding"):
            done = json.load(open(os.path.join(out, step + ".done")))
            self.assertIn(done["seeded"], (None, False), (step, done))       # the seed line never ran: pytorch_lightning was not even imported for it
            self.assertIn(CLEAN.format(step=step, seed="none"), log)
            self.assertTrue(json.load(open(os.path.join(out, f"stock_env_proof_{step}.json")))["ok"])
        man = json.load(open(os.path.join(out, "opt_manifest.json")))
        self.assertEqual((man["exit_code"], man["seed"], [s["seed"] for s in man["run"]["steps"]]), (0, None, [None, None]))

    def test_a_relative_output_and_spec_resolve_to_one_absolute_run_dir(self):
        """`--output` and the spec are made absolute before anything consumes them: a two-component relative `--output` yields ONE run
        directory holding both the launcher's opt_configure.log and configure's steps.yaml (configure and the steps run with the run
        directory's parent as cwd, so a path left relative would land one level off), and absolute inputs pass through unchanged."""
        rel = os.path.join("rel", "two")                                         # typed relative to the caller's cwd (self.tmp), like the spec below
        rc, so, se = self.cli(["design", "--mode", "off", "spec.yaml", "--output", rel, "--seed", "1", "--num_designs", "1", "--steps", "design"])
        self.assertEqual(rc, 0, se)
        out = os.path.join(self.tmp, rel)
        for f in ("opt_configure.log", "steps.yaml", "design.done", "opt_manifest.json"):
            self.assertTrue(os.path.exists(os.path.join(out, f)), f)
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "rel", "rel")))                                              # no second tree beside it
        self.assertTrue(open(os.path.join(out, "configure_argv.txt")).read().startswith(f"configure {self.spec} --output {out} "))  # configure was handed absolute paths

    def test_absolute_output_and_spec_pass_through_unchanged(self):
        out = os.path.join(self.tmp, "abs_run")
        rc, so, se = self.cli(["design", "--mode", "off", self.spec, "--output", out, "--seed", "1", "--num_designs", "1", "--steps", "design"])
        self.assertEqual(rc, 0, se)
        self.assertTrue(open(os.path.join(out, "configure_argv.txt")).read().startswith(f"configure {self.spec} --output {out} "))  # byte-identical to what was typed
        self.assertEqual(json.load(open(os.path.join(out, "opt_manifest.json")))["exit_code"], 0)

    def test_a_spec_naming_msa_and_templates_is_configured_not_refused(self):
        """A spec's `msa` / `templates` keys are upstream's to consume: the kit hands the spec to `boltzgen configure` as given (class H,
        pass-through) — no refusal, no exit 2, the steps configured and run."""
        spec = os.path.join(self.tmp, "spec_b.yaml")
        open(spec, "w").write(SPEC_MSA)
        out = os.path.join(self.tmp, "off_spec_b")
        rc, so, se = self.cli(["design", "--mode", "off", spec, "--output", out, "--seed", "1", "--num_designs", "1", "--steps", "design"])
        self.assertEqual(rc, 0, se)
        self.assertNotIn("refus", se.lower()); self.assertNotIn("ERROR", se); self.assertNotIn(" msa", se.lower())   # no refusal, no usage error, no word about the spec's keys
        self.assertTrue(os.path.exists(os.path.join(out, "steps.yaml")))
        self.assertTrue(os.path.exists(os.path.join(out, "design.done")))
        self.assertTrue(open(os.path.join(out, "configure_argv.txt")).read().startswith(f"configure {spec} --output {out} "))

    def test_upstreams_options_reach_configure_as_given(self):
        """Upstream's `run` options (cli.UPSTREAM_OPTIONS) are forwarded to its configure as given, in the caller's order, wherever they stand
        (before the spec, after it, behind a bare `--`), behind the caller's `--cache` (which wins over $BOLTZGEN_CACHE)."""
        out = os.path.join(self.tmp, "off_passthrough")
        rc, so, se = self.cli(["design", "--mode", "off", "--num_designs", "3", self.spec, "--output", out, "--seed", "1", "--steps", "design",
                               "--cache", os.path.join(self.tmp, "mine"), "--protocol", "peptide-anything", "--", "--budget", "5"], BOLTZGEN_CACHE=os.path.join(self.tmp, "hf"))
        self.assertEqual(rc, 0, se)
        conf = open(os.path.join(out, "configure_argv.txt")).read().splitlines()[0]
        self.assertEqual(conf, f"configure {self.spec} --output {out} --cache {os.path.join(self.tmp, 'mine')} --num_designs 3 --steps design --protocol peptide-anything --budget 5")   # the caller's --cache (not $BOLTZGEN_CACHE), then upstream's options in the caller's order
        self.assertIn("num_designs: 3", open(os.path.join(out, "config", "design.yaml")).read())

    def stock_cmd(self, proof, with_step=True):
        """The stock child's argv exactly as `design` builds it (design.stock_step_cmd: the core's contract + this kit's PINS lists and kit
        dirs); without the step words when ``with_step`` is False."""
        cmd = design.stock_step_cmd(None, os.path.join(self.tmp, "never", "config", "design.yaml"), proof)
        return cmd if with_step else cmd[:cmd.index("--")]

    def test_stock_caller_refuses_with_a_kit_variable_or_kit_path(self):
        proof = os.path.join(self.tmp, "proof.json")
        base = self.stock_cmd(proof)
        clean, dropped = design.stock_env(_stubs.clean_env(self.site))         # what `design` hands the child: no PYTHONPATH, nothing under BOLTZGEN_OPT* or a kit switch prefix
        self.assertTrue({"PYTHONPATH", "BOLTZGEN_OPT_TEST_GPU", "BOLTZGEN_OPT_HOME"} <= set(dropped), dropped)
        env = dict(clean, PYTHONPATH=self.site)
        r = subprocess.run(base, env=env, capture_output=True, text=True)
        self.assertEqual(r.returncode, codes.EXIT_NOT_ACTIVE, r.stderr)     # PYTHONPATH itself is a forbidden name for the stock arm: exit 3, nothing ran
        self.assertIn("[boltzgen-opt stock] NOT STOCK: forbidden env ['PYTHONPATH'], kit modules [], kit dirs []", r.stderr)
        self.assertIn(f"[boltzgen-opt stock] refused: the stock step does not run with the kits within reach (exit {codes.EXIT_NOT_ACTIVE})", r.stderr)
        self.assertNotIn("environment proven clean", r.stderr)
        rec = json.load(open(proof))
        self.assertEqual((rec["ok"], rec["forbidden_present"]), (False, ["PYTHONPATH"]))
        self.assertFalse(os.path.exists(os.path.join(self.tmp, "never", "design.done")))
        r = subprocess.run(base, env=dict(clean, BOLTZGEN_OPT_WHATEVER="1"), capture_output=True, text=True)
        self.assertEqual(r.returncode, 3)
        self.assertIn("NOT STOCK: forbidden env ['BOLTZGEN_OPT_WHATEVER']", r.stderr)   # the names of stock/PINS.json are read as prefixes, as the core reads them: any BOLTZGEN_OPT* is a kit variable there
        kit_src = stack.kit_src(modes.KIT_XATTEMPT)
        code = "import sys; sys.path[:0] = %r; import boltzgen_opt.stock_design as s; sys.exit(s.main(%r))" % ([kit_src] + stack.package_roots(), base[base.index("--proof-json"):])
        r = subprocess.run([sys.executable, "-s", "-c", code], env=dict(clean, XA_FAST_INIT="1"), capture_output=True, text=True)
        self.assertEqual(r.returncode, 3, r.stderr)
        self.assertIn("NOT STOCK: forbidden env ['XA_FAST_INIT'], kit modules [], kit dirs ['%s']" % kit_src, r.stderr)   # a kit directory on sys.path is named too
        rec = json.load(open(proof))
        self.assertEqual((rec["ok"], rec["forbidden_present"], rec["kit_dirs_on_path"]), (False, ["XA_FAST_INIT"], [kit_src]))

    def test_stock_caller_with_a_clean_environment_writes_its_proof_before_anything_else(self):
        """The child's contract is the core's (`--proof-json --env-absent --kit-dirs --module-prefixes -- <seed|none> <config>`):
        a clean call without the two step words proves and writes its environment, then stops at usage (2) — nothing of upstream imported."""
        proof = os.path.join(self.tmp, "proof_clean.json")
        own = self.stock_cmd(proof, with_step=False)
        clean, _ = design.stock_env(_stubs.clean_env(self.site))
        r = subprocess.run(own, env=clean, capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("usage: ... -- <seed|none> <config.yaml>", r.stderr)
        rec = json.load(open(proof))
        pins = stack.pins()["stock_environment"]
        self.assertEqual((rec["ok"], rec["env_absent"], rec["kit_modules_loaded"], rec["kit_dirs_on_path"], rec["torch_loaded_before_proof"]),
                         (True, list(pins["must_be_absent_prefixes"]) + list(pins["must_be_absent_names"]), [], [], False))
        i = own.index("--proof-json")
        self.assertFalse(hasattr(stock_design, "split_kernels")); self.assertFalse(hasattr(stock_design, "arm_census"))   # no census in a stock process: the caller has no option of its own beyond the core's contract

    def test_the_stock_caller_is_a_runnable_module(self):
        """`python -s -m boltzgen_opt.stock_design` runs main(): the module carries its `__main__` clause (a caller whose module body only defines
        functions would prove nothing and run nothing, exit 0 — the failure mode this test names)."""
        tree = ast.parse(open(stock_design.__file__, encoding="utf-8").read())
        mains = [n for n in tree.body if isinstance(n, ast.If) and isinstance(n.test, ast.Compare) and getattr(n.test.left, "id", None) == "__name__"]
        self.assertEqual(len(mains), 1)
        self.assertIn("main", ast.dump(mains[0]))

    def test_the_stock_child_starts_this_trees_caller_before_site(self):
        """design.stock_step_cmd: the core's contract line with the module started by `-S -s -c <preamble>` — this tree's package and core
        roots (stack.package_roots(), the parent's own copies) first on sys.path, then site, then the caller as __main__; the own options and
        the stock words follow unchanged."""
        proof = os.path.join(self.tmp, "proof_form.json")
        cmd = design.stock_step_cmd(5, "/x/config/design.yaml", proof)
        self.assertEqual(cmd[:4], [sys.executable, "-S", "-s", "-c"])
        self.assertEqual(cmd[4], design.stock_preamble())
        pre = cmd[4]
        self.assertTrue(pre.startswith("import sys; sys.path[:0] = [p for p in %r if p not in sys.path]; import boltzgen_opt, opt_core.stock_proof; import site; site.main(); "
                                       % (stack.package_roots(),)), pre)
        self.assertIn("runpy.run_module('boltzgen_opt.stock_design', run_name='__main__', alter_sys=True)", pre)
        self.assertEqual(cmd[5:7], ["--proof-json", proof])
        self.assertEqual(cmd[cmd.index("--"):], ["--", "5", "/x/config/design.yaml"])
        self.assertNotIn("-m", cmd)

    def test_a_second_copy_of_the_package_within_reach_does_not_replace_this_trees_caller(self):
        """The defect the preamble closes: another copy of `boltzgen_opt` that the child's interpreter resolves while its site initialises — on a
        box, a container image's own editable install imported by its autoload .pth before any path entry of the mounted tree exists; here a
        site .pth that puts the impostor first on sys.path AND imports it, the strongest form — must not run in place of this tree's stock caller.
        The child proves and stops at usage (2) with THIS tree's lines; the impostor (exit 9, its own word) never runs. 0.5.2's form
        (`-s -m boltzgen_opt.stock_design`) runs the impostor — recorded here so the difference stays visible."""
        import site as _site
        other = os.path.join(self.tmp, "other_copy")
        os.makedirs(os.path.join(other, "boltzgen_opt"), exist_ok=True)
        open(os.path.join(other, "boltzgen_opt", "__init__.py"), "w").write("IMPOSTOR = True\n")
        open(os.path.join(other, "boltzgen_opt", "stock_design.py"), "w").write("import sys\nprint('IMPOSTOR stock caller ran', file=sys.stderr)\nsys.exit(9)\n")
        pth = os.path.join(os.path.dirname(self.pth), "aaa_bgopt_second_copy_test.pth")     # sorts first among the site's .pth files, as an editable finder's does
        proof = os.path.join(self.tmp, "proof_other.json")
        own = self.stock_cmd(proof, with_step=False)
        clean, _ = design.stock_env(_stubs.clean_env(self.site))
        with open(pth, "w", encoding="utf-8") as fh:
            fh.write("import sys; sys.path.insert(0, %r); exec('import boltzgen_opt')\n" % other)
        try:
            r = subprocess.run(own, env=clean, capture_output=True, text=True)
            old = [sys.executable, "-s", "-m", "boltzgen_opt.stock_design"] + own[own.index("--proof-json"):]
            r0 = subprocess.run(old, env=clean, capture_output=True, text=True)
        finally:
            os.remove(pth)
        self.assertNotIn("IMPOSTOR", r.stderr + r.stdout)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("[boltzgen-opt stock] usage: ... -- <seed|none> <config.yaml>", r.stderr)
        self.assertTrue(json.load(open(proof))["ok"])
        self.assertEqual((r0.returncode, "IMPOSTOR stock caller ran" in r0.stderr), (9, True), r0.stderr)

    def test_seed_default_rng_is_the_hooks_derivation_draw_for_draw(self):
        """`--seed` seeds upstream's featurizer generator on every arm alike: under one seeded numpy stream, the stock caller's
        seed_default_rng and the partner hook (bg_hook.py, imported at interpreter start on the kit arms) hand `np.random.default_rng(None)`
        the same seeds call after call, and neither advances the stream; an explicit seed passes through untouched."""
        probe = ("import json, numpy as np\nnp.random.seed(2026)\n{install}\n"
                 "draws = [int(np.random.default_rng().integers(0, 2**31)) for _ in range(3)]\n"
                 "print(json.dumps([draws, int(np.random.get_state()[2]), int(np.random.default_rng(5).integers(0, 2**31))]))\n")
        stock = subprocess.run([sys.executable, "-c", probe.format(install="from boltzgen_opt import stock_design; stock_design.seed_default_rng(np)")],
                               capture_output=True, text=True, env=_stubs.clean_env(self.site))
        hook_env = dict(_stubs.clean_env(self.site)); hook_env["PYTHONPATH"] = os.pathsep.join([stack.kit_src(modes.KIT_PARTNER), hook_env.get("PYTHONPATH", "")])
        hook = subprocess.run([sys.executable, "-c", probe.format(install="import bg_hook")], capture_output=True, text=True, env=hook_env)
        self.assertEqual(stock.returncode, 0, stock.stderr[-400:]); self.assertEqual(hook.returncode, 0, hook.stderr[-400:])
        a, b = json.loads(stock.stdout.strip().splitlines()[-1]), json.loads(hook.stdout.strip().splitlines()[-1])
        self.assertEqual(a, b)                                                    # same seeds draw for draw, same untouched stream position, same explicit-seed passthrough
        self.assertEqual(len(set(a[0])), 3)                                       # distinct per call
        plain = subprocess.run([sys.executable, "-c", "import numpy as np; np.random.seed(2026); print(int(np.random.get_state()[2]))"], capture_output=True, text=True)
        self.assertEqual(a[1], int(plain.stdout))                                 # the seeded stream was not advanced by the three generator constructions

    def test_seed_words_and_the_seed_line(self):
        self.assertEqual(stock_design.NO_SEED, "none")
        self.assertIsNone(stock_design.parse_seed("none"))
        self.assertEqual((stock_design.parse_seed("0"), stock_design.parse_seed("12")), (0, 12))
        self.assertTrue(stock_design.SEED_LINE.startswith("pl.seed_everything(seed, workers=True); random.seed(seed);"))
        compile(stock_design.SEED_LINE, "<seed>", "exec")
        self.assertEqual(stock_design.EXIT_NOT_STOCK, stock_proof.EXIT_NOT_STOCK)
        self.assertEqual(stock_design.PREFIX, "[boltzgen-opt stock]")
        self.assertFalse(hasattr(stock_design, "prove"))                      # the proof is the core's one (opt_core.stock_proof.env_proof), composed, not restated


if __name__ == "__main__":
    unittest.main()
