"""Every command in a fresh interpreter on a stand-in tree whose gates the test interpreter meets: check, pred on both arms end to end
(the ACTIVE / STOCK lines, the driver's argv from its own proc_start record, the proof, the manifest, the exit tally), the refusals."""
import json
import re
import os
import tempfile
import sys
import unittest
from unittest import mock

from opt_core import gates as core_gates

from af2ig_opt import cli, modes, registry, report
from . import _stubs


class TestCli(unittest.TestCase):
    PEAK_RE = re.compile(r"^\[af2ig-opt\] PEAK item=(\S+) inuse_gib=([0-9.]+)$")                       # one line per item, key=value tokens only
    NOTE_RE = re.compile(r"^\[af2ig-opt\] PEAK-NOTE scope=pass items=(\d+) inuse_gib=([0-9.]+) source=peak_bytes_in_use reset=none( unread=(\d+) reason=\S+| reason=\S+)?$")

    def assert_peak_lines(self, err, names, gib="4.66"):
        """Both arms print, on stderr with the LEVER/EXIT lines, one PEAK line per completed design (item = the design record's tag) and ONE PEAK-NOTE (the stub driver's
        peak_bytes_in_use is 5000000000 = 4.66 GiB)."""
        peaks = [m.groups() for m in map(self.PEAK_RE.match, err.splitlines()) if m]
        notes = [m.groups() for m in map(self.NOTE_RE.match, err.splitlines()) if m]
        self.assertEqual(sorted(p[0] for p in peaks), sorted(names), err); self.assertEqual({p[1] for p in peaks}, {gib}, err)
        self.assertEqual(len(notes), 1, err); self.assertEqual(notes[0][:3], (str(len(names)), gib, None), err)
        self.assertNotIn(" PEAK ", "\n".join(l for l in err.splitlines() if not self.PEAK_RE.match(l)))   # the word PEAK is the per-item line's alone

    def test_check(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            rc, out, err = _stubs.run_cli(["check", "--mode", "exact"], env)
            self.assertEqual(rc, 0, err)
            self.assertIn("DRY-RUN mode=exact line=-fast -precompile 6 -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,L1,U1,L7,L13,L15,L16,ccache precompile=6 jit=on:", err); self.assertIn(" core=ok pins=ok checkout=ok weights=ok jax=", err)
            rc, out, err = _stubs.run_cli(["check", "--mode", "fast", "--json"], env)
            self.assertEqual(rc, 0, err)
            self.assertIn("line=-fast -precompile 6 -subbatch 128 -flash_attn -fused_triattn -fused_trimul -opm_reassoc -tmpl_pointwise_sub 8192 -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,L1,U1,L7,L9,L8,L10,L11,L12,L18,L19,L13,L15,L16,ccache", err)
            rep = json.loads(out)
            self.assertFalse(rep["active"]); self.assertTrue(rep["dry_run"]); self.assertEqual(rep["mode"], "fast")
            rc, out, err = _stubs.run_cli(["check", "--mode", "off"], env)
            self.assertEqual(rc, 0, err)
            self.assertIn("DRY-RUN mode=off line=none levers=none", err)
            # the default mode when neither --mode nor AF2IG_OPT is given: fast (modes.DEFAULT_MODE)
            rc, out, err = _stubs.run_cli(["check"], env)
            self.assertIn("mode=fast", err); self.assertEqual((modes.DEFAULT_MODE, modes.FOLDED), ("fast", {}))   # 0.7.2: the default word resolves to the big line
            # AF2IG_OPT read as the mode; a disagreeing --mode refused
            rc, out, err = _stubs.run_cli(["check"], dict(env, AF2IG_OPT="exact"))
            self.assertIn("mode=exact", err)
            rc, out, err = _stubs.run_cli(["check", "--mode", "fast"], dict(env, AF2IG_OPT="exact"))
            self.assertEqual(rc, cli.EXIT_USAGE)
            # an unknown selection under the variable is refused loudly (exit 2), never run as stock
            rc, out, err = _stubs.run_cli(["check"], dict(env, AF2IG_OPT="bogus"))
            self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("unknown mode 'bogus'", err)
            rc, out, err = _stubs.run_cli(["pred", "--pdbdir", tree, "--out", os.path.join(tmp, "x")], dict(env, AF2IG_OPT="bogus"))
            self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("unknown mode 'bogus'", err); self.assertFalse(os.path.exists(os.path.join(tmp, "x", "pdbs")))
            # a mistyped switch name under the package prefix is refused, never stripped silently
            for verb in (["check"], ["pred", "--mode", "exact", "--pdbdir", tree, "--out", os.path.join(tmp, "y")]):
                rc, out, err = _stubs.run_cli(verb, dict(env, AF2IG_OPT_MODE="exact"))
                self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("undeclared AF2IG_OPT* variable(s) ['AF2IG_OPT_MODE']", err)
            self.assertFalse(os.path.exists(os.path.join(tmp, "y")))

    def test_pins_drift_runs_named(self):
        """A pinned distribution installed at another version is drift: named on the printed line (`pins=drift`, `note='pins drift …'`), never a refusal —
        `check` and `pred` proceed on both arms (uncertainty about the environment is stated, not a reason to refuse)."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp, pins_ok=False)
            rc, out, err = _stubs.run_cli(["check", "--mode", "exact"], env)
            self.assertEqual(rc, 0, err); self.assertIn("pins=drift", err); self.assertNotIn("would_refuse", err); self.assertRegex(err, r"note='pins drift \(stock/PINS\.json; runs, named\): \S+=\S+ \(want 0\.0\.1\)")
            in_dir, names = _stubs.inputs(tmp, 1)
            rc, out, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o")], env)
            self.assertEqual(rc, 0, err); self.assertIn("] ACTIVE mode=exact", err); self.assertIn("note='pins drift (stock/PINS.json; runs, named): ", err); self.assertNotIn("NOT ACTIVE", err)
            rc, out, err = _stubs.run_cli(["pred", "--mode", "off", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o2")], env)
            self.assertEqual(rc, 0, err); self.assertNotIn("NOT ACTIVE", err)

    def test_absent_pin_refuses_and_force_runs(self):
        """A required distribution that is not installed at all is the pins gate's one refusal (nothing of the stack could run): NOT ACTIVE, exit 3,
        `pins=unmet`; AF2IG_OPT_FORCE=1 records it (`pins=forced`) and runs."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp, pins_absent=True)
            rc, out, err = _stubs.run_cli(["check", "--mode", "exact"], env)
            self.assertEqual(rc, cli.EXIT_INACTIVE); self.assertIn("would_refuse", err); self.assertIn("pins=unmet", err); self.assertIn("af2ig-no-such-distribution: not installed; want 1.0", err)
            rc, out, err = _stubs.run_cli(["check", "--mode", "exact"], dict(env, AF2IG_OPT_FORCE="1"))
            self.assertEqual(rc, 0); self.assertIn("pins=forced", err)
            in_dir, names = _stubs.inputs(tmp, 1)
            rc, out, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o")], env)
            self.assertEqual(rc, cli.EXIT_INACTIVE); self.assertIn("NOT ACTIVE", err)
            rc, out, err = _stubs.run_cli(["pred", "--mode", "off", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o2")], env)
            self.assertEqual(rc, cli.EXIT_INACTIVE); self.assertIn("NOT ACTIVE", err)

    def test_pred_both_arms(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            in_dir, names = _stubs.inputs(tmp, 2)
            # stock
            out = os.path.join(tmp, "off")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "off", "--pdbdir", in_dir, "--out", out, "--det", "1"], env)
            self.assertEqual(rc, 0, err)
            self.assertIn("[det] applied level=1 XLA_FLAGS='--xla_gpu_autotune_level=0'", err)
            self.assertIn("[af2ig-opt] STOCK line=", err); self.assertNotIn("ACTIVE mode", err)
            self.assertRegex(err, r"(?m)^\[af2ig-opt\] STOCK line='-pdbdir \S+ -outpdbdir \S+ -scorefilename \S+ -checkpoint_name \S+ -af2_dir \S+ -timers \S+' det=level=1 stripped=\S+$")   # the package's STOCK line: its words unchanged
            self.assertRegex(err, r"(?m)^\[af2ig-opt stock\] ENV-CLEAN ok: absent=AF2IG_OPT,\S+ kit_modules=none kit_dirs=none no_user_site=True autoload=none package_modules=af2ig_opt det=1$")   # then the caller's census (stock_cli.census_line)
            self.assertLess(err.index("[af2ig-opt] STOCK line="), err.index("[af2ig-opt stock] ENV-CLEAN ok: "))
            self.assertIn("[af2ig-opt] LEVER name=L6 state=off reason=the_stock_arm_applies_no_lever impl=predict_pdb origin=kit strategy=F6.host_sync_elimination flag=-host_outputs arm=off", err); self.assertNotIn("state=on", err)   # the stock arm applies nothing: every lever `off`
            self.assertIn("EXIT pid=", err); self.assertIn("mode=off route=cli items=2/2 det=on driver_rc=0", err)
            self.assert_peak_lines(err, names)                   # the stock arm: the driver's design records carry the peak too
            self.assertEqual(_stubs.run_files(out), ["check.point", "out.sc", "pdbs"])                                # <out> holds the driver's own outputs only: no manifest, proof or timers file
            lines = _stubs.records(so)                                     # the driver's -timers records, relayed to stdout
            self.assertEqual(lines[0]["kind"], "proc_start"); self.assertNotIn("-fast", lines[0]["argv"]); self.assertEqual(lines[0]["env"]["XLA_FLAGS"], "--xla_gpu_autotune_level=0")
            self.assertEqual([l["kind"] for l in lines].count("design"), 2); self.assertIn('"kind": "design"', so)
            for n in names:
                self.assertTrue(os.path.isfile(os.path.join(out, "pdbs", n + "_af2pred.pdb")))
            # exact
            out = os.path.join(tmp, "exact")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", out], env)
            self.assertEqual(rc, 0, err)
            self.assertIn("[af2ig-opt] ACTIVE mode=exact route=cli line=-fast -precompile 6 -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,L1,U1,L7,L13,L15,L16,ccache precompile=6 jit=on:", err)
            self.assertIn("mode=exact route=cli items=2/2 det=off driver_rc=0", err)
            self.assert_peak_lines(err, names)
            self.assertIn("partial=none", err); self.assertEqual([l for l in err.splitlines() if " state=on " in l and "arm=exact" in l].__len__(), 8)   # L6, L1, U1 (+ L7, L13, L15, L16, ccache) evidenced: nothing partial
            first = _stubs.records(so)[0]
            self.assertEqual(first["argv"][-9:], ["-fast", "-precompile", "6", "-program_cache", "<programs>", "-prefetch", "2", "-overlap_output", "1"]); self.assertIsNone(first["env"]["XLA_FLAGS"])
            self.assertEqual(_stubs.run_files(out), ["check.point", "out.sc", "pdbs"])
            # fast
            out = os.path.join(tmp, "fast")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", out, "--det", "1"], env)
            self.assertEqual(rc, 0, err)
            self.assertIn("ACTIVE mode=fast route=cli line=-fast -precompile 6 -subbatch 128 -flash_attn -fused_triattn -fused_trimul -opm_reassoc -tmpl_pointwise_sub 8192 -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,L1,U1,L7,L9,L8,L10,L11,L12,L18,L19,L13,L15,L16,ccache", err)
            self.assert_peak_lines(err, names)
            first = _stubs.records(so)[0]
            self.assertEqual(first["argv"][-17:], ["-fast", "-precompile", "6", "-subbatch", "128", "-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-tmpl_pointwise_sub", "8192", "-program_cache", "<programs>", "-prefetch", "2", "-overlap_output", "1"]); self.assertEqual(first["env"]["XLA_FLAGS"], "--xla_gpu_autotune_level=0")

    def test_pred_partial_exits_3_unless_allowed(self):
        """A lever of the mode without the driver's evidence of it (here L11: the fused triangle-multiplication block served no call) is a partial activation:
        exit 3 with the one NOT ACTIVE line naming the lever and the escape; `--allow-partial` records it and keeps the run's exit (no environment form)."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            in_dir, names = _stubs.inputs(tmp, 2)
            env = dict(env, STUB_FTRIMUL_SERVED="0")
            out = os.path.join(tmp, "fast")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", out], env)
            self.assertEqual(rc, cli.EXIT_INACTIVE, err)
            self.assertRegex(err, r"\[af2ig-opt\] NOT ACTIVE: partial activation — .*L11.*; exit 3 \(--allow-partial records and proceeds\)")
            self.assertIn("partial=L11 allow_partial=off", err); self.assertNotIn("PARTIAL allowed", err)
            self.assertIn("items=2/2", err); self.assertTrue(all(os.path.isfile(os.path.join(out, "pdbs", n + "_af2pred.pdb")) for n in names))      # the outputs stay; nothing incomplete
            out2 = os.path.join(tmp, "fast_allowed")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", out2, "--allow-partial"], env)
            self.assertEqual(rc, 0, err)
            self.assertRegex(err, r"\[af2ig-opt\] PARTIAL allowed: .*L11.* \(--allow-partial, recorded\)")
            self.assertIn("partial=L11 allow_partial=on", err); self.assertNotIn("NOT ACTIVE", err)
            out3 = os.path.join(tmp, "fast_env")                            # no environment twin of the flag: AF2IG_OPT_ALLOW_PARTIAL is an undeclared AF2IG_OPT* name, refused by name, nothing runs
            rc, so, err = _stubs.run_cli(["pred", "--pdbdir", in_dir, "--out", out3], dict(env, AF2IG_OPT="fast", AF2IG_OPT_ALLOW_PARTIAL="1"))
            self.assertEqual(rc, cli.EXIT_USAGE, err); self.assertIn("undeclared AF2IG_OPT* variable(s) ['AF2IG_OPT_ALLOW_PARTIAL']", err); self.assertFalse(os.path.exists(out3))

    def test_pred_incomplete_exits_1_never_partial(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            in_dir, names = _stubs.inputs(tmp, 2)
            out = os.path.join(tmp, "o")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", out, "--allow-partial"], dict(env, STUB_DRIVER_SKIP_TAG=names[0]))
            self.assertEqual(rc, cli.EXIT_FAIL, err); self.assertIn("items=1/2", err)
            self.assertIn("partial=none", err); self.assertNotIn("NOT ACTIVE", err)               # incomplete (items=1/2, exit 1), never a partial activation
            # a run that failed on its own (outputs short) WITH a lever unevidenced keeps its own code (1): the lever is named on its LEVER line and the EXIT line,
            # never by a `NOT ACTIVE: partial activation … exit 3` sentence the process does not honour (the shared core's exit rule and partial line)
            rc, so, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o2")], dict(env, STUB_DRIVER_SKIP_TAG=names[0], STUB_FLASH_SERVED="0"))
            self.assertEqual(rc, cli.EXIT_FAIL, err); self.assertIn("items=1/2", err); self.assertIn("partial=L8 allow_partial=off", err)
            self.assertIn("LEVER name=L8 state=skipped reason=the_kernel_is_in_no_compiled_program", err); self.assertNotIn("NOT ACTIVE", err); self.assertNotIn("exit 3", err); self.assertNotIn("PARTIAL allowed", err)

    def test_pred_without_driver_records_names_the_absence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            in_dir, names = _stubs.inputs(tmp, 1)
            out = os.path.join(tmp, "o")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", out], dict(env, STUB_DRIVER_NO_TIMERS="1"))
            self.assertEqual(rc, cli.EXIT_INACTIVE, err)                # no driver record: no lever can be evidenced — every planned lever is partial (exit 3), never silently applied
            self.assertIn("partial=L6,L1,U1,L7,L13,L15,L16,ccache allow_partial=off", err); self.assertIn("items=1/1", err); self.assertNotIn(" state=on ", "\n".join(l for l in err.splitlines() if " arm=" in l))   # the package's per-lever lines (arm=…); the stub driver prints its own L13 exit line
            self.assertIn("LEVER name=L6 state=skipped reason=no_driver_record_in_timers.jsonl_(the_lever_cannot_be_evidenced) impl=predict_pdb origin=kit strategy=F6.host_sync_elimination flag=-host_outputs arm=exact", err)
            self.assertIn("NOT ACTIVE: partial activation — levers=L6,L1,U1,L7,L13,L15,L16,ccache (L6: no driver record in timers.jsonl (the lever cannot be evidenced); L1: ", err); self.assertIn("); exit 3 (--allow-partial records and proceeds)", err)
            rc, so, err = _stubs.run_cli(["pred", "--mode", "exact", "--allow-partial", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o2")], dict(env, STUB_DRIVER_NO_TIMERS="1"))
            self.assertEqual(rc, 0, err); self.assertIn("PARTIAL allowed: levers=L6,L1,U1,L7,L13,L15,L16,ccache (L6: no driver record in timers.jsonl (the lever cannot be evidenced); L1: ", err)

    def test_usage_refusals(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            in_dir, names = _stubs.inputs(tmp, 1)
            rc, out, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o"), "--", "-subbatch", "256"], env)
            self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("lever flags", err)
            rc, out, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", os.path.join(tmp, "nowhere"), "--out", os.path.join(tmp, "o")], env)
            self.assertEqual(rc, cli.EXIT_USAGE)

    def test_precompile_usage_refusals(self):
        """--precompile [N] (L7, every kit mode since 0.7.2): off refuses by name, and N < 1 refuses by name."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 1)
            rc, _, err = _stubs.run_cli(["pred", "--mode", "off", "--precompile", "--pdbdir", in_dir, "--out", os.path.join(tmp, "off_pre")], env)
            self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn("--precompile: a lever of the kit modes only", err); self.assertIn("stock takes no lever", err)   # 0.7.2: big precompiles too; only stock refuses
            self.assertFalse(os.path.exists(os.path.join(tmp, "off_pre", "pdbs")))
            for bad in ("0", "-1", "-3"):                               # every typed N < 1 is refused by name: neither 0 nor -1 is a spelling of "the default N"
                rc, _, err = _stubs.run_cli(["pred", "--mode", "exact", "--precompile", bad, "--pdbdir", in_dir, "--out", os.path.join(tmp, "bad" + bad)], env)
                self.assertEqual(rc, cli.EXIT_USAGE); self.assertIn(f"--precompile {bad}: N must be >= 1 (or omitted for the default N)", err)
                self.assertFalse(os.path.exists(os.path.join(tmp, "bad" + bad, "pdbs")))

    def test_warm_runs_the_stock_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            rc, out, err = _stubs.run_cli(["warm", "--out", os.path.join(tmp, "w")], env)
            self.assertEqual(rc, 0, err)
            self.assertIn("WARM PASS", err); self.assertIn("STOCK line=", err); self.assertNotIn("ACTIVE mode", err)
            self.assertTrue(os.path.isfile(os.path.join(tmp, "w", "out", "pdbs", "1brs_bb_1to1_af2pred.pdb")))
            rc, out, err = _stubs.run_cli(["warm"], env)
            self.assertEqual(rc, 0, err); self.assertIn("out=discarded", err)


class TestFlashAttnLever(unittest.TestCase):
    """L8 (-flash_attn, tier 2): composed on the fast line (fast, big on its fast base), never on a tier-1 line; evidenced by the driver's flash_attn census record (served >= 1)."""

    def test_resolve_and_refusals(self):
        kit = _stubs.KIT
        r = modes.resolve("fast", kit)
        self.assertEqual(r.flags[-12:-8], ["-flash_attn", "-fused_triattn", "-fused_trimul", "-opm_reassoc"]); self.assertEqual(r.levers[-10:], ["L8", "L10", "L11", "L12", "L18", "L19", "L13", "L15", "L16", "ccache"])   # 0.7.4: fast carries L18 (rows 8192), no memory line   # the fast line composes both kernel levers, the fused block last
        self.assertNotIn("-flash_attn", modes.resolve("exact", kit).flags)                                 # tier 2: never on the exact line
        self.assertIn("-flash_attn", modes.resolve("big", kit).flags)                                    # big inherits the fast line's levers

    def test_switch_defaults_are_pinned(self):
        """0.8.0: no kit size gate — the provider decides per cell; configure() names it."""
        from af2ig_opt import flash_attn
        with mock.patch.object(flash_attn, "_serve") as sv:
            sv.return_value = mock.Mock(ledger=mock.Mock(return_value=object()), kernel_origin=mock.Mock(return_value="core"), FALLBACK_REASONS=("no_pair_bias",))
            conf = flash_attn.configure()
        self.assertEqual((conf["min_tokens"], conf["gate_source"], conf["all_calls"]), (0, "provider", False))
        self.assertFalse(hasattr(flash_attn, "MIN_TOKENS"))

    def test_levers_off_and_ccache(self):
        """MODEL_OPT_LEVERS_OFF=<ids> (the tree's uniform ablation switch): the mode's composition minus the named levers — their flags leave the driver's argv, the ACTIVE
        line says levers_off=<ids>, each dropped lever's LEVER line reads state=off reason=levers_off:MODEL_OPT_LEVERS_OFF, an unknown id is named and ignored; the run is
        complete (never partial, never exit 3). The deployment lever ccache is placed in every kit mode (jit=on:<dir> on the line, JAX_COMPILATION_CACHE_DIR in the child,
        its LEVER line on with the directory's entries), kept when the caller set JAX_COMPILATION_CACHE_DIR (jit=kept:<dir>), dropped the same way (jit=off:levers_off);
        the stock line never carries it and stays ENV-CLEAN under a caller's JAX_COMPILATION_CACHE_DIR."""
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, names = _stubs.inputs(tmp, 2)
            rc, so, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "e")], env)
            self.assertEqual(rc, 0, err)
            jit_dir = os.path.join(tmp, "cache", "jit")                                                           # the package default root: <AF2IG_OPT_CACHE_DIR>/jit
            m = re.search(r" levers=L6,L1,U1,L7,L13,L15,L16,ccache precompile=6 jit=on:(\S+) jax=", err); self.assertTrue(m, err)
            self.assertTrue(m.group(1).startswith(jit_dir + os.sep) and m.group(1).endswith(os.sep + "jax"), m.group(1)); self.assertTrue(os.path.isdir(m.group(1)))
            self.assertRegex(err, r"LEVER name=ccache state=on impl=capture\.xla_cache origin=core strategy=F3\.jit_cache_keyed flag=JAX_COMPILATION_CACHE_DIR arm=exact evidence=child_env:_JAX_COMPILATION_CACHE_DIR=\S+/jax_source=default_key=jax\S+_entries=0->0_stored=0\n")
            self.assertIn("partial=none", err); self.assertNotIn("levers_off=", err)
            # the ablation switch on the fast line: L8 and the deployment lever dropped, FOO unknown
            abl = dict(env, MODEL_OPT_LEVERS_OFF="L8,ccache,FOO")
            rc, so, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "f")], abl)
            self.assertEqual(rc, 0, err)
            self.assertIn("ACTIVE mode=fast route=cli line=-fast -precompile 6 -subbatch 128 -fused_triattn -fused_trimul -opm_reassoc -tmpl_pointwise_sub 8192 -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,L1,U1,L7,L9,L10,L11,L12,L18,L19,L13,L15,L16 levers_off=L8,ccache precompile=6 jit=off:levers_off jax=", err)
            self.assertIn("note='MODEL_OPT_LEVERS_OFF: unknown id(s) FOO ignored'", err)
            self.assertIn("LEVER name=L8 state=off reason=levers_off:MODEL_OPT_LEVERS_OFF impl=opt_core.pallas.serve.attention origin=core strategy=F1.flash_triatt flag=-flash_attn arm=fast", err)
            self.assertIn("LEVER name=ccache state=off reason=levers_off:MODEL_OPT_LEVERS_OFF impl=capture.xla_cache origin=core strategy=F3.jit_cache_keyed flag=JAX_COMPILATION_CACHE_DIR arm=fast", err)
            self.assertIn("partial=none", err); self.assertNotIn("NOT ACTIVE", err)
            first = _stubs.records(so)[0]
            self.assertEqual(first["argv"][-16:], ["-fast", "-precompile", "6", "-subbatch", "128", "-fused_triattn", "-fused_trimul", "-opm_reassoc", "-tmpl_pointwise_sub", "8192", "-program_cache", "<programs>", "-prefetch", "2", "-overlap_output", "1"]); self.assertNotIn("-flash_attn", first["argv"])
            self.assertEqual(sorted(l.split(" name=")[1].split()[0] for l in err.splitlines() if " state=on " in l and "arm=fast" in l), sorted(["L6", "L1", "U1", "L7", "L9", "L10", "L11", "L12", "L18", "L19", "L13", "L15", "L16"]))
            # a preset lever dropped: the remaining preset levers by their own flags
            rc, so, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "x")], dict(env, MODEL_OPT_LEVERS_OFF="L1"))
            self.assertEqual(rc, 0, err); self.assertIn(" line=-host_outputs -sort_by_length -precompile 6 -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,U1,L7,L13,L15,L16,ccache levers_off=L1 precompile=6 jit=on:", err)
            self.assertEqual(_stubs.records(so)[0]["argv"][-10:], ["-host_outputs", "-sort_by_length", "-precompile", "6", "-program_cache", "<programs>", "-prefetch", "2", "-overlap_output", "1"]); self.assertIn("LEVER name=L6 state=on ", err); self.assertIn("LEVER name=L1 state=off reason=levers_off:MODEL_OPT_LEVERS_OFF ", err)
            rc, _, err = _stubs.run_cli(["check", "--mode", "exact"], dict(env, MODEL_OPT_LEVERS_OFF="U1 L9"))
            self.assertEqual(rc, 0, err); self.assertIn("DRY-RUN mode=exact line=-host_outputs -device_params -precompile 6 -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,L1,L7,L13,L15,L16,ccache levers_off=U1 precompile=6 jit=on:", err)
            self.assertIn("note='MODEL_OPT_LEVERS_OFF: L9 not in the composition of exact (ignored)'", err)
            # the caller's own JAX_COMPILATION_CACHE_DIR: kept on a kit mode, absent from the stock child (the stock rule) — the switch and the cache never touch the stock line
            preset = os.path.join(tmp, "preset")
            kept = dict(env, JAX_COMPILATION_CACHE_DIR=preset, MODEL_OPT_LEVERS_OFF="L8")
            rc, _, err = _stubs.run_cli(["check", "--mode", "fast"], kept)
            self.assertEqual(rc, 0, err); self.assertIn(f" levers=L6,L1,U1,L7,L9,L10,L11,L12,L18,L19,L13,L15,L16,ccache levers_off=L8 precompile=6 jit=kept:{preset} ", err); self.assertTrue(os.path.isdir(preset))
            rc, so, err = _stubs.run_cli(["pred", "--mode", "off", "--pdbdir", in_dir, "--out", os.path.join(tmp, "s"), "--det", "1"], kept)
            self.assertEqual(rc, 0, err); self.assertRegex(err, r"(?m)^\[af2ig-opt stock\] ENV-CLEAN ok: absent=AF2IG_OPT,JAX_COMPILATION_CACHE_DIR,\S+ ")
            self.assertNotIn("levers_off=", err); self.assertNotIn(" jit=", err); self.assertNotIn("state=on", err); self.assertIn("stripped=", err); self.assertIn("JAX_COMPILATION_CACHE_DIR", err.split("stripped=")[1].split()[0])

    def test_pred_fast_flash_attn_evidence(self):
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 2)
            out = os.path.join(tmp, "o")
            rc, _, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", out], env)
            self.assertEqual(rc, 0, err)
            self.assertIn("ACTIVE mode=fast route=cli line=-fast -precompile 6 -subbatch 128 -flash_attn -fused_triattn -fused_trimul -opm_reassoc -tmpl_pointwise_sub 8192 -program_cache <programs> -prefetch 2 -overlap_output 1 levers=L6,L1,U1,L7,L9,L8,L10,L11,L12,L18,L19,L13,L15,L16,ccache precompile=6 jit=on:", err)
            self.assertIn("partial=none", err); self.assertEqual(sorted(l.split(" name=")[1].split()[0] for l in err.splitlines() if " state=on " in l and "arm=fast" in l), sorted(["L6", "L1", "U1", "L7", "L9", "L8", "L10", "L11", "L12", "L18", "L19", "L13", "L15", "L16", "ccache"]))
            self.assertIn("LEVER name=L9 state=on impl=subbatch_policy origin=core strategy=F7.chunked_eval flag=-subbatch arm=fast evidence=argv:_-subbatch;_subbatch_record(s):_2_compiled_length(s)_at_128_rows_per_chunk_(stock_4;_source_requested;_opt_core.jax_design.subbatch_policy)", err)   # two inputs of distinct lengths, unpadded: two programs
            self.assertIn("evidence=argv:_-fused_triattn;_fused_triattn_census_(1_record(s)):_4_TriangleAttention_call(s)_traced_onto_the_block_fpf_pallas_f32_(origin_core),_0_left_on_the_stock_body_(reasons:_none);_precision=tf32;_tiles=own:9.0", err)
            self.assertIn(" evidence=argv:_-fused_trimul;_fused_trimul_census_(1_record(s)):_4_TriangleMultiplication_call(s)_traced_onto_the_block_fpf_pallas_f32_(origin_core),_0_left_on_the_stock_body_(reasons:_none);_precision=tf32;_tiles=own:9.0", err)   # L11
            rc, _, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o3")], dict(env, STUB_FUSED_UNEXPECTED="c_not_multiple_of_16"))
            self.assertEqual(rc, cli.EXIT_INACTIVE, err); self.assertIn("L10: TriangleAttention call(s) left on the stock body for undeclared reason(s)", err)   # an undeclared fallback is a partial activation, never silent
            rc, _, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o4")], dict(env, STUB_FUSED_SERVED="0"))
            self.assertEqual(rc, cli.EXIT_INACTIVE, err); self.assertIn("L10: the block is in no compiled program", err)
            rc, _, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o5")], dict(env, STUB_FTRIMUL_SERVED="0"))
            self.assertEqual(rc, cli.EXIT_INACTIVE, err); self.assertIn("L11: the block is in no compiled program", err)
            self.assertIn("evidence=argv:_-flash_attn;_flash_attn_census_(1_record(s)):_12_Attention_call(s)_traced_onto_the_kernel_pallas_attn_(origin_core),_4_left_on_the_stock_ops_(reasons:_no_pair_bias_x4);_min_tokens=0;_precision=tf32", err)   # L8 (the flash-attention kernel) evidenced on this run too
            self.assertIn("[af2ig-opt] LEVER name=L8 state=on impl=opt_core.pallas.serve.attention origin=core strategy=F1.flash_triatt flag=-flash_attn arm=fast evidence=", err)
            # the kernel in no compiled program (every call gated or ineligible): L8 asked and never applied = partial, exit 3
            rc, _, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o2")], dict(env, STUB_FLASH_SERVED="0"))
            self.assertEqual(rc, cli.EXIT_INACTIVE, err)
            self.assertIn("NOT ACTIVE: partial activation — levers=L8 (L8: the kernel is in no compiled program: 0 Attention calls served, 16 left on the stock ops (below_size_rule x16)); exit 3 (--allow-partial records and proceeds)", err)
            self.assertIn("LEVER name=L8 state=skipped reason=the_kernel_is_in_no_compiled_program:_0_Attention_calls_served,_16_left_on_the_stock_ops_(below_size_rule_x16) impl=opt_core.pallas.serve.attention origin=core strategy=F1.flash_triatt flag=-flash_attn arm=fast", err)
            # a call left on the stock ops for a reason OUTSIDE the lever's declared set (flash_attn.EXPECTED_FALLBACKS) while others were served: L8 is fail-closed like L10/L11 — partial, exit 3, never `on`
            rc, _, err = _stubs.run_cli(["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o6")], dict(env, STUB_FLASH_UNEXPECTED="bias_form"))
            self.assertEqual(rc, cli.EXIT_INACTIVE, err)
            self.assertIn("NOT ACTIVE: partial activation — levers=L8 (L8: Attention call(s) left on the stock ops for undeclared reason(s) / kernel error(s) {'bias_form': 1} (12 served)); exit 3", err)
            self.assertRegex(err, r"LEVER name=L8 state=skipped reason=Attention_call\(s\)_left_on_the_stock_ops_for_undeclared_reason\(s\)_/_kernel_error\(s\)_\{'bias_form':_1\}_\(12_served\) impl=opt_core.pallas.serve.attention "); self.assertIn("partial=L8 allow_partial=off", err)
            self.assertNotIn("LEVER name=L8 state=on", err); self.assertTrue(os.path.isfile(os.path.join(tmp, "o6", "out.sc")))                 # the outputs stay; the promise is withdrawn by name


class TestStrategyIds(unittest.TestCase):
    """Every registry lever names a strategy id of the LEVER line's form (opt_core.report.strategy_form: F<k>.<name> or LOCAL.af2ig.<name>)."""

    def test_registry_strategies_have_the_lever_lines_form(self):
        from opt_core import report as core_report
        for lv, lever in registry.LEVERS.items():
            self.assertTrue(lever.strategy, lv)
            self.assertEqual(core_report.strategy_form(lever.strategy), lever.strategy)


class TestWeightsWarnAndRun(unittest.TestCase):
    """AF2_PARAMS: a present parameter file is ALWAYS accepted — the pin's bytes print `weights=<name> sha256=<12> (pinned)`, other bytes print ONE line
    `weights sha256=<12> not pinned (pinned: <12>) — proceeding` and the run proceeds (exit 0, no flag); a missing file is
    refused by name (exit 3). The manifest records the digest and the word (stack.gates.weights.sha256 / .pinned)."""

    def test_pinned_not_pinned_missing(self):
        import hashlib
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); env.pop("AF2IG_OPT_FORCE", None); in_dir, _ = _stubs.inputs(tmp, 1)
            wp = os.path.join(env["AF2_PARAMS"], "params", "params_model_1_ptm.npz")
            pinned12 = hashlib.sha256(open(wp, "rb").read()).hexdigest()[:12]
            rc, _, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "c")], env)      # the pinned bytes
            self.assertEqual(rc, 0, err); self.assertIn(f"[af2ig-opt] weights=params_model_1_ptm.npz sha256={pinned12} (pinned)", err); self.assertNotIn("not pinned", err)
            with open(wp, "wb") as fh:                                                                                                # a user's own checkpoint: other bytes
                fh.write(b"a user checkpoint of other bytes")
            other12 = hashlib.sha256(b"a user checkpoint of other bytes").hexdigest()[:12]
            rc, _, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "u")], env)
            self.assertEqual(rc, 0, err); self.assertIn("[af2ig-opt] ACTIVE mode=exact", err)
            self.assertEqual(err.count(f"[af2ig-opt] weights sha256={other12} not pinned (pinned: {pinned12}) — proceeding"), 1, err)
            self.assertTrue(os.path.exists(os.path.join(tmp, "u", "pdbs")))
            rc, _, err = _stubs.run_cli(["check", "--mode", "fast"], env)                                                           # the dry run names it too, exit 0
            self.assertEqual(rc, 0, err); self.assertIn("weights=ok", err); self.assertIn(f"weights sha256={other12} not pinned", err)
            os.remove(wp)                                                                                                             # missing: the one refusal, by name
            rc, _, err = _stubs.run_cli(["pred", "--mode", "exact", "--pdbdir", in_dir, "--out", os.path.join(tmp, "m")], env)
            self.assertEqual(rc, 3, err); self.assertIn("[af2ig-opt] NOT ACTIVE: weights: ", err); self.assertIn("params_model_1_ptm.npz: not found", err); self.assertNotIn("Traceback", err)
            self.assertFalse(os.path.exists(os.path.join(tmp, "m", "pdbs")))


class TestEntryProbePassthrough(unittest.TestCase):
    """run.sh's probe and the config's probe (`python -c "… import af2ig_opt …; gate(…)"`) reserve the `package_missing` word (rc 3) for a genuine absence of the
    package (find_spec None); a package whose import itself fails or refuses ends the script with THAT python's stderr and exit status — no word added, no status
    coerced. `python` on PATH is a shim over this interpreter with -S, so only PYTHONPATH supplies packages."""

    def _env(self, tmp, *pythonpath):
        shim = os.path.join(tmp, "bin"); os.makedirs(shim, exist_ok=True)
        with open(os.path.join(shim, "python"), "w") as fh:
            fh.write("#!/bin/bash\nexec %s -S \"$@\"\n" % sys.executable)
        os.chmod(os.path.join(shim, "python"), 0o755)
        env = {k: v for k, v in os.environ.items() if not k.startswith(("AF2IG_OPT", "PYTHON"))}
        env.update(PATH=shim + os.pathsep + env.get("PATH", ""), PYTHONPATH=os.pathsep.join(pythonpath), PYTHONDONTWRITEBYTECODE="1")
        return env

    def _fake_package(self, tmp, name, body):
        d = os.path.join(tmp, name); os.makedirs(os.path.join(d, "af2ig_opt"))
        with open(os.path.join(d, "af2ig_opt", "__init__.py"), "w") as fh:
            fh.write(body)
        return d

    def _run(self, argv, env):
        import subprocess
        r = subprocess.run(argv, env=env, capture_output=True, text=True, cwd=_stubs.TREE)
        return r.returncode, r.stderr

    def test_probe_passthrough(self):
        import subprocess
        run_sh = os.path.join(_stubs.TREE, "run.sh"); cfg = os.path.join(_stubs.TREE, "configs", "h100.env")
        with tempfile.TemporaryDirectory() as tmp:
            refusing = self._fake_package(tmp, "refusing", "import sys\nsys.stderr.write('[af2ig-opt] NOT ACTIVE: reason=a_refusal_raised_while_importing\\n')\nraise SystemExit(5)\n")
            broken = self._fake_package(tmp, "broken", "raise RuntimeError('boom while importing af2ig_opt')\n")
            cases = [                                                                      # (PYTHONPATH dirs, rc, words that must be there, words that must not)
                ((refusing,), 5, ["NOT ACTIVE: reason=a_refusal_raised_while_importing"], ["package_missing", "Traceback"]),
                ((broken,), 1, ["Traceback", "boom while importing af2ig_opt"], ["package_missing"]),
                ((tmp,), 3, ["[af2ig-opt] NOT ACTIVE: package_missing:af2ig_opt on "], ["Traceback"]),        # nothing named af2ig_opt importable: the genuine absence
            ]
            for paths, rc_want, present, absent in cases:
                env = self._env(tmp, *paths)
                for argv in (["bash", run_sh, "check", "--mode", "exact"], ["bash", run_sh, "check", "--config", "h100", "--mode", "exact"],
                             ["bash", run_sh, "check", "--config", "a100", "--mode", "exact"], ["bash", "-c", "source %s && echo SOURCED-OK" % cfg]):
                    rc, err = self._run(argv, env)
                    self.assertEqual(rc, rc_want, (paths, argv, err))
                    for w in present: self.assertIn(w, err, (paths, argv))
                    for w in absent: self.assertNotIn(w, err, (paths, argv))
                    self.assertEqual(err.count("NOT ACTIVE"), 1 if rc_want in (3, 5) else 0, (paths, argv, err))     # one line: the first probe to run ends the script


class TestCoreMissing(unittest.TestCase):
    """The core pin gate (af2ig_opt/_core_gate.py, the tree's kit_template copy) runs as statement one of every entry — `python -m af2ig_opt`
    (= the console script), `bash run.sh <verb>` (its probe, then the entry it execs), the config's probe, the
    driver-side hook `af2ig_opt.pairstack.install` — BEFORE any opt_core import: an ABSENT core, an OLDER-VERSION core, an UNVERSIONED core and an
    unreadable pin are one `[af2ig-opt] NOT ACTIVE: reason=…` line and exit 3, never a traceback; a core that meets the pin's version but
    lacks a sub-module this package imports is named by the second guard (`NOT ACTIVE: core_missing:<module>`)."""
    GATE = "[af2ig-opt] NOT ACTIVE: reason="

    def test_core_absent(self):
        """No opt_core importable at all (python -S: site-packages off; only the package on PYTHONPATH): reason=core_missing:opt_core, rc 3, on every verb."""
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 1)
            env["PYTHONPATH"] = _stubs.OPT                                                     # nothing but the package on the path; with -S (no site-packages) no opt_core is importable anywhere
            for verb in (["check", "--mode", "exact"], ["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o")], ["check", "--mode", "big"]):
                rc, _, err = _stubs.run_cli(verb, env, no_site=True)
                self.assertEqual(rc, 3, (verb, err)); self.assertIn(self.GATE + "core_missing:opt_core (pinned >= v", err); self.assertNotIn("Traceback", err)
            e = env
            r = subprocess.run([sys.executable, "-S", "-c", "from af2ig_opt import pairstack; pairstack.install(trimul=(256, 1473))"], env=e, capture_output=True, text=True)   # the driver's hook
            self.assertEqual(r.returncode, 3, r.stderr); self.assertIn(self.GATE + "core_missing:opt_core", r.stderr); self.assertNotIn("Traceback", r.stderr)
            self.assertFalse(os.path.exists(os.path.join(tmp, "o", "pdbs")))

    def test_core_stale_or_unversioned(self):
        """A core PRESENT but not meeting the pin — (a) an OLDER __version__ literal (FLOOR semantics: older refuses, newer passes), (b) no
        __version__ literal at all — through `python -m af2ig_opt <verb>`, `bash run.sh check --config h100` and the driver hook:
        reason=core_mismatch with the pinned and installed facts, rc 3."""
        import subprocess, sys
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp); in_dir, _ = _stubs.inputs(tmp, 1)
            stale = _stubs.prepend_pythonpath(env, _stubs.shadow_core(tmp, "stale", version="0.2.5"))
            for verb in (["check", "--mode", "exact"], ["pred", "--mode", "fast", "--pdbdir", in_dir, "--out", os.path.join(tmp, "o")], ["check", "--mode", "big"]):
                rc, _, err = _stubs.run_cli(verb, stale)
                self.assertEqual(rc, 3, (verb, err)); self.assertIn(self.GATE + "core_mismatch: opt_core pinned >= v", err); self.assertIn("installed v0.2.5 at ", err); self.assertNotIn("Traceback", err)
            r = subprocess.run(["bash", os.path.join(_stubs.TREE, "run.sh"), "check", "--config", "h100", "--mode", "fast"], env=stale, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stderr); self.assertIn(self.GATE + "core_mismatch: opt_core pinned >= v", r.stderr); self.assertNotIn("Traceback", r.stderr)
            self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, r.stderr)                                     # the config's probe refuses; run.sh stops there (one line, not three)
            r = subprocess.run([sys.executable, "-c", "from af2ig_opt import pairstack; pairstack.install(trimul=(256, 1473))"], env=stale, capture_output=True, text=True, cwd=_stubs.OPT)
            self.assertEqual(r.returncode, 3, r.stderr); self.assertIn(self.GATE + "core_mismatch", r.stderr); self.assertNotIn("Traceback", r.stderr)
            unversioned = _stubs.prepend_pythonpath(env, _stubs.shadow_core(tmp, "unversioned", strip_version=True))
            rc, _, err = _stubs.run_cli(["check", "--mode", "exact"], unversioned)
            self.assertEqual(rc, 3, err); self.assertIn(self.GATE + "core_mismatch: opt_core pinned >= v", err); self.assertIn("installed v? at ", err); self.assertNotIn("Traceback", err)
            self.assertFalse(os.path.exists(os.path.join(tmp, "o", "pdbs")))

    def test_older_core_passing_the_pin_but_lacking_a_module(self):
        """The second guard: a core whose version meets the pin but which lacks a sub-module the command line imports (opt_core.precision here) is
        `NOT ACTIVE: core_missing:opt_core.precision…` rc 3 through python -m and run.sh — the words after the pin gate, never a traceback."""
        import subprocess
        with tempfile.TemporaryDirectory() as tmp:
            tree, env = _stubs.make_tree(tmp)
            older = _stubs.prepend_pythonpath(env, _stubs.shadow_core(tmp, "older", drop=("precision",)))
            rc, _, err = _stubs.run_cli(["check", "--mode", "exact"], older)
            self.assertEqual(rc, 3, err); self.assertIn("[af2ig-opt] NOT ACTIVE: core_missing:opt_core.precision", err); self.assertNotIn("Traceback", err)
            r = subprocess.run(["bash", os.path.join(_stubs.TREE, "run.sh"), "check", "--config", "h100", "--mode", "exact"], env=older, capture_output=True, text=True)
            self.assertEqual(r.returncode, 3, r.stderr); self.assertIn("[af2ig-opt] NOT ACTIVE: core_missing:opt_core.precision", r.stderr); self.assertNotIn("Traceback", r.stderr)


class TestCorePin(unittest.TestCase):
    def test_pin_is_the_imported_core(self):
        """opt/pyproject.toml's [tool.opt_core] pin is met by the opt_core this interpreter actually imports: the pin is a floor (opt_core.gates.core_pin_check —
        the tree's core may run ahead of it), and it names the tree's own core by path."""
        pyproject = os.path.join(_stubs.TREE, "opt", "pyproject.toml")
        g = core_gates.core_pin_check(pyproject)
        self.assertTrue(g.ok, g.reason)
        pin = core_gates.core_pin(pyproject)
        self.assertLessEqual(core_gates.version_tuple(pin["version"]), core_gates.version_tuple(core_gates.imported_core()["version"]))
        self.assertEqual(pin["path"], "../../common/opt_core")


class TestPeakLinesCompleteness(unittest.TestCase):
    """report.peak_lines' completeness accounting: a design counts even when its allocator-peak value is unusable, and a process with no
    completed design still gets a properly-reasoned line -- never silence. (The happy-path line/note format, every design carrying a valid
    value, is exercised end to end by TestCli.assert_peak_lines on every real pred run above; these two cover what that path never hits.)"""
    GIB = 2 ** 30

    def test_a_record_without_the_value_is_counted_not_dropped(self):
        recs = [{"kind": "design", "tag": "a", "peak_bytes_in_use": 2 * self.GIB}, {"kind": "design", "tag": "b", "peak_bytes_in_use": None},
                {"kind": "design", "tag": "c"}, {"kind": "design", "tag": "d", "peak_bytes_in_use": True}]     # a bool is not a byte count
        lines = report.peak_lines(recs)
        self.assertEqual(lines, ["[af2ig-opt] PEAK item=a inuse_gib=2.00",
                                  "[af2ig-opt] PEAK-NOTE scope=pass items=1 inuse_gib=2.00 source=peak_bytes_in_use reset=none unread=3 reason=memory_stats_unavailable"])

    def test_no_design_record_is_named(self):
        self.assertEqual(report.peak_lines([{"kind": "proc_start"}, {"kind": "failed", "tag": "x"}]),
                          ["[af2ig-opt] PEAK-NOTE scope=pass items=0 inuse_gib=0.00 source=peak_bytes_in_use reset=none reason=no_design_record"])
        self.assertEqual(report.peak_lines([]), ["[af2ig-opt] PEAK-NOTE scope=pass items=0 inuse_gib=0.00 source=peak_bytes_in_use reset=none reason=no_design_record"])

