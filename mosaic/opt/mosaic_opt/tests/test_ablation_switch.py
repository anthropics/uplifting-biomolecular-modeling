"""The ablation switch and the tier rows' warm-free QoL levers (0.3.7): `MODEL_OPT_LEVERS_OFF=<id>[,<id>…]` skips the named levers BY NAME
in every route — the ONE installer for the per-step levers (LEVER … state=off reason=levers_off:MODEL_OPT_LEVERS_OFF, label `<mode>-<ID>…`),
the resolver for the row levers (P1's environment, P2 / P3's flags dropped, the mode keeps its word); a lever whose `needs` the request
switches off is refused by name (F8 needs F6); an unknown id is refused by name. The fast and big rows carry P1 in its TRANSPARENT form
(the compilation cache alone, its own directory per shape, pcc `autotune=off`; it steps aside by name without MOSAIC_OPT_CACHE_ROOT) and P2;
P3 stays exact-only. CPU-only: fake lever modules, no jax computation."""
import contextlib
import dataclasses
import io
import os
import tempfile
import unittest
from unittest import mock

from . import _fake_lever, core_src  # noqa: F401  (core on sys.path from a bare checkout)
from mosaic_opt import levers, modes, registry, report, stack

FAKE = "mosaic_opt.tests._fake_lever"


@contextlib.contextmanager
def fake_modules():
    """Every per-step lever of the registry served by the fake module (records calls, patches nothing real)."""
    fakes = {k: dataclasses.replace(v, module=FAKE) for k, v in registry.LEVERS.items() if v.route == "install"}
    with mock.patch.dict(registry.LEVERS, fakes), mock.patch.dict(levers.LEVERS, fakes), mock.patch.dict(modes.LEVERS, fakes):
        _fake_lever.reset(); levers.reset_for_tests()
        try:
            yield
        finally:
            levers.reset_for_tests(); _fake_lever.reset()


class TestLeversOffWord(unittest.TestCase):

    def test_parse_in_registry_order_and_refuse_unknown(self):
        self.assertEqual(levers.ENV_LEVERS_OFF, "MODEL_OPT_LEVERS_OFF")                     # the family's one spelling
        self.assertFalse(levers.ENV_LEVERS_OFF.startswith(tuple(stack.stock_env_absent())))  # not a stock-stripped prefix: it rides the arm's environment and means nothing to stock
        self.assertEqual(levers.levers_off({}), ()); self.assertEqual(levers.levers_off({"MODEL_OPT_LEVERS_OFF": " "}), ())
        self.assertEqual(levers.levers_off({"MODEL_OPT_LEVERS_OFF": "F8, K1,P2"}), ("P2", "K1", "F8"))   # registry order, whatever the spelling order
        self.assertEqual(levers.levers_off({"MODEL_OPT_LEVERS_OFF": "K1+F8"}), ("K1", "F8"))
        with self.assertRaises(levers.LeverError) as cm:
            levers.levers_off({"MODEL_OPT_LEVERS_OFF": "K1,Z9"})
        self.assertIn("levers_off_unknown", str(cm.exception)); self.assertIn("Z9", str(cm.exception))

    def test_registry_declares_needs_and_env_ownership(self):
        self.assertEqual(registry.LEVERS["F8"].needs, ("F6",))                                 # F8 rides F6's served call
        self.assertEqual([k for k, v in registry.LEVERS.items() if v.env], ["P1"])            # P1 alone owns environment
        self.assertIn("JAX_COMPILATION_CACHE_DIR", registry.LEVERS["P1"].env); self.assertIn("XLA_FLAGS", registry.LEVERS["P1"].env)
        for tag, row in modes.ROWS.items():                                                    # every variable a row assigns is P1's to drop
            self.assertLessEqual(set(modes.parse_row(row["text"])["env"]), set(registry.LEVERS["P1"].env), tag)


class TestInstallerSkipsByName(unittest.TestCase):

    def test_plan_label_and_off_lines(self):
        with fake_modules(), mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "K1,F8,P2"}), mock.patch("sys.stderr", new_callable=io.StringIO) as err:
            p = levers.plan("fast")
            self.assertEqual((p["label"], p["levers"], p["levers_off"]), ("fast-K1-F8", ["E1", "P6", "E10", "F6", "F9", "P7"], ["K1", "F8"]))   # P2 is the resolver's (a row lever), not the installer's
            info = levers.install("fast")
            self.assertEqual((info["mode"], info["label"], info["levers"], info["levers_off"]), ("fast", "fast-K1-F8", ["E1", "P6", "E10", "F6", "F9", "P7"], ["K1", "F8"]))
            text = err.getvalue()
            self.assertIn("[mosaic-opt] LEVER name=K1 state=off reason=levers_off:MODEL_OPT_LEVERS_OFF", text)
            self.assertIn("[mosaic-opt] LEVER name=F8 state=off reason=levers_off:MODEL_OPT_LEVERS_OFF", text)
            self.assertIn("[mosaic-opt] LEVER name=P5 state=off reason=mode:fast-K1-F8", text)   # a lever outside the row: off by the mode's label, as before
            self.assertRegex(text, r"LEVER name=E1 state=on"); self.assertNotIn("name=K1 state=on", text)
            self.assertEqual(levers.manifest_record().keys(), {"E1", "P6", "E10", "F6", "F9", "P7"})           # the driver's probes: the skipped levers are not shown applied, and not asked for

    def test_a_needed_lever_switched_off_is_refused_by_name(self):
        with fake_modules(), mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "F6"}):
            with self.assertRaises(levers.LeverError) as cm:
                levers.plan("fast")
            self.assertIn("lever_needs: F8 needs F6", str(cm.exception)); self.assertIn("MODEL_OPT_LEVERS_OFF=F6,F8", str(cm.exception))
            with self.assertRaises(levers.LeverError):
                levers.install("big")
            self.assertEqual(levers.installed()["levers"], [])                                     # nothing half-installed
        with fake_modules(), mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "F6,F8"}):
            self.assertEqual(levers.plan("big")["label"], "big-F6-F8")
        with fake_modules(), mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "K1"}):        # P5's chunked attention delegates to whichever attention is served: no declared need
            self.assertEqual(levers.plan("big")["levers"], ["E1", "P6", "E10", "F6", "F8", "F9", "P5", "P7"])

    def test_a_setting_for_a_switched_off_lever_names_the_switch(self):
        with fake_modules(), mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "K1"}):
            with self.assertRaises(levers.LeverError) as cm:
                levers.plan("fast", K1="triattn")
            self.assertIn("switched off by MODEL_OPT_LEVERS_OFF", str(cm.exception))

    def test_unset_changes_nothing(self):
        with fake_modules(), mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("MODEL_OPT_LEVERS_OFF", None)
            p = levers.plan("fast")
            self.assertEqual((p["label"], p["levers"], p["levers_off"]), ("fast", list(modes.install_levers_of("fast")), []))


class TestResolverDropsRowLevers(unittest.TestCase):

    def test_p1_env_and_p2_flag_leave_the_row(self):
        res = modes.resolve("fast", cache_dir="/c/x", levers_off=("P1", "P2", "K1"))
        self.assertEqual((res.row, res.env, res.flags, res.levers_off), ("T_fast", {}, ["--levers", "fast"], ("P1", "P2", "K1")))   # the mode keeps its word and its ONE flag; the driver skips K1 itself
        self.assertEqual(res.levers, ("E1", "P6", "E10", "F6", "F8", "F9", "P7")); self.assertEqual(modes.describe_line(res), "T_fast[E1,P6,E10,F6,F8,F9,P7]-off[P1,P2,K1]")
        res = modes.resolve("exact", cache_dir="/c/x", features="/f.npz", features_sha="ab", levers_off=("P3",))
        self.assertEqual((res.levers, res.flags), (("P1", "P2"), ["--weights", "fastinit"]))     # P3's two flags gone; P1's load flag stays
        self.assertIn("xla_autotune_results.pb", res.env["XLA_FLAGS"])
        res = modes.resolve("exact", cache_dir="/c/x", features="/f.npz", features_sha="ab", levers_off=("P1",))
        self.assertEqual((res.levers, res.env), (("P2", "P3"), {})); self.assertFalse(any("autotune results" in n for n in res.notes))
        with self.assertRaises(ValueError):
            modes.resolve("fast", cache_dir="/c/x", levers_off=("nope",))

    def test_mode_needs_leaves_out_what_is_off(self):
        self.assertEqual(stack.mode_needs("exact", ("P3",))["features"], False)                  # no frozen features to read: nothing to warm for P3
        self.assertEqual(stack.mode_needs("exact", ("P1",))["p1"], False); self.assertIsNone(stack.mode_needs("exact", ("P1",))["populate"])
        n = stack.mode_needs("fast", ("K1", "P2"))
        self.assertEqual((n["per_step"], n["flag_levers"], n["levers_off"], n["p1_form"]), (["E1", "P6", "E10", "F6", "F8", "F9", "P7"], [], ["P2", "K1"], "transparent"))
        self.assertEqual(stack.mode_needs("big")["populate"], None); self.assertEqual(stack.mode_needs("exact")["populate"], "B_p1populate_p2")

    def test_lines_name_the_switch_and_the_aside(self):
        rep = {"active": True, "mode": "fast", "route": "driver", "row_line": "T_fast[E1,P6]-off[K1]", "levers_applied": ["E1", "P6"], "levers_off": ["K1"],
               "levers_unavailable": [], "p1": {"aside": "MOSAIC_OPT_CACHE_ROOT unset"}, "upstream": {}, "gpu": None}
        line = report.activation_line(rep)
        self.assertIn(" levers=E1,P6 levers_off=K1 unavailable=none p1=none(MOSAIC_OPT_CACHE_ROOT_unset) ", line)
        rep = dict(rep, p1={"cache_dir": "/c/L80_c1/xla_cache_fast", "autotune": "off"}, levers_off=[])
        self.assertIn(" p1=off:/c/L80_c1/xla_cache_fast ", report.activation_line(rep)); self.assertNotIn("levers_off=", report.activation_line(rep))


class TestTierRowsCarryTheWarmFreeQolLevers(unittest.TestCase):

    def test_exact_subset_fast_subset_big(self):
        self.assertEqual(modes.TIER_QOL, ("P1", "P2"))                                            # P3 needs `warm`'s frozen features: exact-only
        for w in ("fast", "big"):
            self.assertEqual(modes.KIT_MODES[w]["levers"][:2], ("P1", "P2")); self.assertNotIn("P3", modes.KIT_MODES[w]["levers"])
            self.assertIsNone(modes.KIT_MODES[w]["populate"]); self.assertEqual(modes.p1_form(w), "transparent")
        self.assertLessEqual(set(modes.KIT_MODES["fast"]["levers"]), set(modes.KIT_MODES["big"]["levers"])); self.assertEqual(modes.BIG_EXCLUDES, ())
        self.assertEqual(modes.install_levers_of("fast"), ("E1", "P6", "K1", "E10", "F6", "F8", "F9", "P7"))   # the per-step install order is unchanged
        self.assertEqual(modes.install_levers_of("exact"), ())                                     # E1 measured not bitwise at 195 tokens under the exact step recipe: not composed by exact

    def test_transparent_p1_directories(self):
        from mosaic_opt import inputs
        sh = inputs.Shape(binder_length=80, target_copies=1)
        self.assertEqual(inputs.p1_dir("/r", sh), "/r/L80_c1/xla_cache"); self.assertEqual(inputs.p1_dir("/r", sh, "fast"), "/r/L80_c1/xla_cache_fast")
        self.assertEqual(stack.p1_state(None, "transparent")["autotune"], None)
        import tempfile
        d = tempfile.mkdtemp(prefix="mosaic_opt_p1t_")
        self.assertEqual(stack.p1_state(d, "transparent")["autotune"], "off"); self.assertEqual(stack.p1_state(d)["autotune"], "dump")
        with mock.patch.dict(os.environ, {}):
            os.environ.pop(stack.ENV_CACHE_ROOT, None); os.environ.pop(stack.ENV_CACHE_DIR, None)
            self.assertEqual(stack._p1_dir_for(None, "transparent", "fast"), (None, None, stack.P1_ASIDE_REASON))   # steps aside by name
            why, dd, aside = stack._p1_dir_for(None, "pinned")
            self.assertIn(stack.ENV_CACHE_ROOT, why); self.assertEqual((dd, aside), (None, None))                  # a pinned P1 without a directory is a refusal by name
            root = tempfile.mkdtemp(prefix="mosaic_opt_root_")
            os.environ[stack.ENV_CACHE_ROOT] = root
            self.assertEqual(stack._p1_dir_for(None, "transparent", "big"), (None, os.path.join(root, "campaign_big"), None)); self.assertTrue(os.path.isdir(os.path.join(root, "campaign_big")))
            self.assertEqual(stack._p1_dir_for(None, "pinned"), (None, os.path.join(root, "campaign"), None))
            os.chmod(root, 0o555)                                                                            # a root this process cannot create under: the transparent P1 steps aside by name, a pinned one is pcc's to refuse
            try:
                if not os.access(root, os.W_OK):                                                             # (root can always write: the case does not arise)
                    why, dd, aside = stack._p1_dir_for(None, "transparent", "fast")
                    self.assertEqual((why, dd), (None, None)); self.assertIn("not writable", aside)
                    self.assertIsNone(stack.transparent_dir_unusable(os.path.join(root, "campaign_big")))  # an existing directory is usable read-only
            finally:
                os.chmod(root, 0o755)

    def test_design_refuses_an_unknown_id_by_name(self):
        """The design route: an unknown id in MODEL_OPT_LEVERS_OFF is the NOT ACTIVE line and exit 3 — never ignored, never a traceback."""
        import argparse
        from mosaic_opt import cli, inputs
        ns = argparse.Namespace(mode="fast", det=0, binder_length=80, target_copies=1, target_fasta=None, first_record=False, msa=None, epitope=None,
                                steps1=None, steps2=None, out=tempfile.mkdtemp(prefix="mosaic_opt_q7_"), tag=None, seed=0, allow_partial=False, timeout=None)
        with mock.patch.dict(os.environ, {"MODEL_OPT_LEVERS_OFF": "Q7"}), mock.patch.object(cli, "_shape", lambda a: inputs.Shape(binder_length=80, target_copies=1)), \
             mock.patch.object(cli.stack, "kit_home", lambda: "/nonexistent"), mock.patch.object(cli.settings, "effective", lambda *a, **k: {}), \
             mock.patch.object(cli.det, "precision_refusal", lambda: None), mock.patch.object(cli.stack, "pins", lambda: {}), \
             mock.patch.object(cli.stack, "pins_gate", lambda p, force=False: (True, {}, None)), mock.patch.object(cli.stack, "data_path_gate", lambda: (None, [])), \
             mock.patch("sys.stderr", new_callable=io.StringIO):
            with self.assertRaises(cli.CliError) as cm:                                     # main() prints `[mosaic-opt] NOT ACTIVE: <reason>` and exits with the code
                cli.cmd_design(ns)
        self.assertEqual(cm.exception.code, cli.EXIT_NOT_ACTIVE)
        self.assertIn("levers_off_unknown: MODEL_OPT_LEVERS_OFF='Q7'", str(cm.exception))

    def test_warm_is_the_pinned_rows_only(self):
        from mosaic_opt import cli, warm, inputs
        self.assertEqual(cli.P1_MODE, "exact")                                                     # warm's default: the mode with a populate row, though fast (the package default) now carries P1 too
        sh = inputs.Shape(binder_length=80, target_copies=1)
        with mock.patch.dict(os.environ, {}):
            os.environ.pop(stack.ENV_CACHE_ROOT, None)
            with self.assertRaises(stack.ActivationError):                                              # fast / big: nothing to warm without a cache root — said by name
                warm.warm("fast", sh)
            with self.assertRaises(ValueError):
                warm.warm("off", sh)                                                                     # stock has no P1
            root = tempfile.mkdtemp(prefix="mosaic_opt_warmt_")
            os.environ[stack.ENV_CACHE_ROOT] = root
            launched = {}
            def fake_launch(argv, env, cwd=None, timeout=None, on_line=None):                            # the ONE warm-up design: the row's own environment ($C1 = xla_cache_fast) and flags, short stages
                launched.update(argv=list(argv), env=dict(env)); os.makedirs(env["JAX_COMPILATION_CACHE_DIR"], exist_ok=True)
                open(os.path.join(env["JAX_COMPILATION_CACHE_DIR"], "entry"), "w").close(); return 0, False
            with mock.patch.object(warm, "stage_weights", lambda sdir, env, log, timeout: {"step": "weights", "exit_code": 0}), \
                 mock.patch.object(warm.driver, "launch", fake_launch), mock.patch.object(warm.driver, "compose", lambda res, **kw: (["drv"] + list(res.flags) + kw["settings_flags"], dict(res.env), {"cwd": root, "row_env": dict(res.env)})), \
                 mock.patch.object(warm.outputs, "read_results", lambda out_dir, tag: {"status": "ok", "manifest": {}}), mock.patch.object(warm.outputs, "classify", lambda r: {}), \
                 mock.patch.object(warm.stack, "kit_home", lambda: root), mock.patch.object(warm.settings, "effective", lambda kit, steps1=None, steps2=None: {"--steps1": steps1, "--steps2": steps2}), \
                 mock.patch.object(warm.stack, "upstream_versions", lambda: {}), mock.patch.object(warm.stack, "gpu_identity", lambda: {}):
                res = warm.warm("fast", sh, log=lambda s: None)
            self.assertEqual((res["status"], res["exit"], res["mode"]), ("PASS", 0, "fast"))
            self.assertEqual(launched["env"]["JAX_COMPILATION_CACHE_DIR"], inputs.p1_dir(root, sh, "fast")); self.assertNotIn("XLA_FLAGS", launched["env"])   # transparent: the cache, no autotune pin
            self.assertEqual(launched["argv"][-4:], ["--steps1", "3", "--steps2", "3"]); self.assertIn("--levers", launched["argv"]); self.assertIn("fast", launched["argv"])
            self.assertEqual((res["p1"]["n_cache_entries"], res["p1"]["entries_before"], res["p1"]["form"]), (1, 0, "transparent"))
            self.assertIn("WARM PASS mode=fast shape=L80_c1 features=n/a autotune=off cache_entries=1 entries_before=0", warm.summary_line(res))


if __name__ == "__main__":
    unittest.main()
