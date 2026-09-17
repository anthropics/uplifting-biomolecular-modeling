"""The lever registry as the one source of truth (modes derive their lever sets from it; `exact` exists only while an exact-class lever is
registered), the mode word's subtractive form `<mode>-no-<lever>` (modes.resolve: base, dropped levers, the one unknown-mode text for anything
else), the lines such a run prints (ACTIVE base=/ablated=, tier none, class ablation; EVIDENCE/EXIT ablated=; the dropped lever's LEVER line
state=off reason=ablated) — and that a table mode's lines are byte-for-byte what they were before the form existed — the kernels' one exit
printer, and the form end to end: the kit arm and `design` on the stub stack with `fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj` (compilecache + nosub + nosub_fn
installed; pallas dropped — no kernel on the stand-in — and hoist_prev dropped: the stand-in design surface has no ColabDesign recycle loop for it to re-state)."""

import os
import re
import subprocess
import sys
import tempfile
import unittest

from opt_core import report as core_report

from . import _stubs
from .test_levers import APPLIED, APPLIED_FN, CACHE_OK, LOWERCACHE_OK, HOIST_OK, PALLAS_OK, LN_OK, OPM_OK, PALLAS_REPLACED, PROJ_OK, PARCOMPILE_OK, TRIATT_OK, TRIMUL_OK, GATED, GATED_FN, TXLA_OK, TXLA_GATED, TXLA_ASIDE, TRANSITION_OK, TRANSITION_SMALL
from colabdesign_opt.kernels import triatt_lever
from colabdesign_opt import _autoload, evidence, kernels, launch, levers, modes, names, nosub, pallas, registry, report

GUARANTEE_FAST = modes.MODES["fast"].guarantee
REP_FAST = {"active": True, "mode": "fast", "route": "subprocess", "tier": 2, "numerics_class": "precision", "levers": ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"], "arm": "kit",
            "line": GUARANTEE_FAST, "settings": "default_4stage_multimer",
            "upstream": {"colabdesign": {"version": "1.1.3", "commit": "e31a56fe1d9b4de25c8697f3a28b75892941cc72"}, "jax": "0.6.0", "dm-haiku": "0.0.17"},
            "gpu": {"name": "NVIDIA H100 80GB HBM3", "cc": "9.0", "memory_mib": 81559}}


# ============================================================ the registry: one source of truth; modes derive from it
class TestRegistry(unittest.TestCase):
    def test_registry_order_and_classes(self):
        self.assertEqual(registry.ORDER, ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "pallas", "triatt", "opm_fold", "ln", "proj", "transition", "txla"))
        self.assertEqual(registry.NUMERICS, ("exact", "precision", "approx"))
        self.assertEqual({k: (l.numerics, l.module, l.line_name) for k, l in registry.LEVERS.items()},
                         {"compilecache": ("exact", "colabdesign_opt.compilecache_jax", "compilecache"), "lowercache": ("exact", "colabdesign_opt.lowercache", "lowercache"), "parcompile": ("exact", "colabdesign_opt.launchpad_parcompile", "parcompile"), "hoist_prev": ("exact", "colabdesign_opt.hoist_prev", "hoist_prev"), "nosub": ("precision", "colabdesign_opt.nosub", "nosub"), "nosub_fn": ("precision", "colabdesign_opt.nosub_fn", "nosub_fn"), "trimul": ("precision", "colabdesign_opt.kernels.trimul_fused", "trimul_pallas"), "pallas": ("precision", "colabdesign_opt.pallas", "F1.pallas_attn"), "triatt": ("precision", "colabdesign_opt.kernels.triatt_lever", "triatt"), "opm_fold": ("precision", "colabdesign_opt.kernels.layers_opm", "opm_fold"), "ln": ("precision", "colabdesign_opt.kernels.layers_ln", "ln"), "proj": ("precision", "colabdesign_opt.kernels.proj_attn", "proj"), "transition": ("precision", "colabdesign_opt.kernels.layers_transition", "transition"), "txla": ("precision", "colabdesign_opt.txla", "txla")})
        self.assertEqual((nosub.NUMERICS, pallas.NUMERICS), (registry.LEVERS["nosub"].numerics, registry.LEVERS["pallas"].numerics))   # the module states the class the registry records
        self.assertEqual(registry.levers_of_numerics("precision"), ("nosub", "nosub_fn", "trimul", "pallas", "triatt", "opm_fold", "ln", "proj", "transition", "txla")); self.assertEqual(registry.levers_of_numerics("exact"), ("compilecache", "lowercache", "parcompile", "hoist_prev"))
        with self.assertRaises(ValueError):
            registry.levers_of_numerics("bitwise")
        self.assertEqual(evidence.LINE_NAMES, registry.line_names())

    def test_modes_derive_from_the_registry(self):
        self.assertEqual(modes.levers_of("fast"), tuple(l for l in registry.ORDER if l != "pallas")); self.assertEqual(modes.MODES["fast"].levers, registry.mode_set(registry.ORDER))   # fast = every registered lever minus the replaced one (triatt supersedes pallas), registry order
        self.assertEqual(modes.levers_of("exact"), ("compilecache", "lowercache", "parcompile", "hoist_prev")); self.assertEqual(modes.MODES["exact"].levers, registry.levers_of_numerics("exact"))   # exact = the exact-class subset
        self.assertEqual(modes.levers_of("off"), ())
        self.assertEqual(names.MODE_NAMES, tuple(modes.MODES))                                                                    # the .pth hook's literal copy == the derived table

    def test_supersession(self):
        """A lever that declares `supersedes` removes the lever it replaces from a mode's set (that one stays registered, prints state=off
        reason=replaced); dropping the superseding lever with the subtractive word restores the replaced one (mode_set over what remains)."""
        self.assertEqual(registry.LEVERS["triatt"].supersedes, ("pallas",)); self.assertEqual(registry.replaced_by("pallas"), ("triatt",)); self.assertEqual(registry.replaced_by("nosub"), ())
        self.assertEqual([k for k, l in registry.LEVERS.items() if l.supersedes], ["triatt"])
        self.assertEqual(registry.mode_set(registry.ORDER), ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"))   # registry order minus the lever triatt supersedes
        self.assertEqual(registry.mode_set(("nosub", "pallas")), ("nosub", "pallas"))            # without the superseding lever among the candidates nothing is dropped
        self.assertEqual(registry.mode_set(tuple(l for l in registry.ORDER if l != "triatt")), ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "pallas", "opm_fold", "ln", "proj", "transition", "txla"))   # = fast-no-triatt: pallas restored
        self.assertEqual(modes.resolve("fast-no-triatt").restored, ("pallas",)); self.assertEqual(modes.resolve("fast-no-pallas-no-triatt").restored, ())
        self.assertEqual((levers.ABLATED_REASON, levers.REPLACED_REASON, levers.MODE_REASON), ("ablated", "replaced", "mode"))

    def test_lever_modules_follow_the_protocol(self):
        for lever in registry.ORDER:
            mod = levers.module_of(lever)
            for attr in ("install", "installed", "uninstall", "off_line", "evidence", "REFUSALS", "NUMERICS"):
                self.assertTrue(hasattr(mod, attr), f"{registry.LEVERS[lever].module}.{attr}")
            line = mod.off_line("ablated")
            self.assertTrue(line.startswith(f"[colabdesign-opt] LEVER name={registry.LEVERS[lever].line_name} state=off reason=ablated impl="), line)
            self.assertRegex(line, r" origin=(core|kit)\b")                                          # core: the lever adapts a mechanism of opt_core (impl=<mechanism>@<core version>); kit: the implementation lives in this kit (impl=<name>@kit)
            self.assertEqual(mod.NUMERICS, registry.LEVERS[lever].numerics)


# ============================================================ the subtractive word: resolve
class TestSubtractiveWord(unittest.TestCase):
    UNKNOWN_TURBO = "unknown mode 'turbo' (expected off|exact|fast, or exact-no-<lever>[-no-<lever>...] without the named levers of exact's compilecache+lowercache+parcompile+hoist_prev, fast-no-<lever>[-no-<lever>...] without the named levers of fast's compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla)"

    def test_split(self):
        self.assertEqual(names.split_mode_word(" FAST-no-Pallas "), ("fast", ("pallas",)))
        self.assertEqual(names.split_mode_word("fast-no-pallas-no-nosub"), ("fast", ("pallas", "nosub")))
        self.assertEqual(names.split_mode_word("fast"), ("fast", ())); self.assertEqual(names.split_mode_word(None), ("", ()))
        self.assertEqual((names.ABLATION_SEP, names.ABLATION_CLASS), ("-no-", "ablation"))

    def test_resolve(self):
        r = modes.resolve("fast-no-triatt")                                                          # dropping the lever that supersedes pallas RESTORES pallas, named (registry Supersession)
        self.assertEqual((r.mode, r.levers, r.tier, r.numerics_class, r.route, r.base, r.ablated, r.restored), ("fast-no-triatt", ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "pallas", "opm_fold", "ln", "proj", "transition", "txla"), None, "ablation", "kit", "fast", ("triatt",), ("pallas",)))
        self.assertEqual(r.line, "ablation: fast without triatt (restored: pallas) - an A/B attribution run, none of the table's modes, no guarantee against stock claimed")
        r = modes.resolve("fast-no-triatt-no-pallas")                                                # ... and dropping both runs stock's attention; canonical word = registry order
        self.assertEqual((r.mode, r.levers, r.ablated, r.restored), ("fast-no-pallas-no-triatt", ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "opm_fold", "ln", "proj", "transition", "txla"), ("pallas", "triatt"), ()))
        with self.assertRaises(ValueError):
            modes.resolve("fast-no-pallas")                                                              # a replaced lever alone takes nothing out of the run: refused like any unknown word
        r = modes.resolve(" FAST-NO-NOSUB ")
        self.assertEqual((r.mode, r.levers, r.base, r.ablated, r.restored), ("fast-no-nosub", ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"), "fast", ("nosub",), ()))
        self.assertEqual(modes.describe(r), "mode=fast-no-nosub levers=compilecache+lowercache+parcompile+hoist_prev+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla route=kit base=fast ablated=nosub")
        plain = modes.resolve("fast")
        self.assertEqual((plain.base, plain.ablated), (None, ()))                                     # a table mode: no base, nothing ablated

    def test_refusals_share_the_one_text(self):
        for bad in ("turbo", "Exact-No-Bogus", "off-no-nosub", "fast-no-bogus", "fast-no-pallas-no-pallas", "fast-no-", "fast-no-compilecache-no-lowercache-no-parcompile-no-hoist_prev-no-nosub-no-nosub_fn-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla", "exact-no-nosub", "exact-no-compilecache-no-lowercache-no-parcompile-no-hoist_prev", "-no-pallas", "fast-no-F1.pallas_attn"):
            with self.assertRaises(ValueError, msg=bad) as cm:
                modes.resolve(bad)
            self.assertTrue(str(cm.exception).startswith(f"unknown mode {bad.strip()!r} (expected off|exact|fast, or exact-no-<lever>[-no-<lever>...] without the named levers of exact's compilecache+lowercache+parcompile+hoist_prev, fast-no-<lever>[-no-<lever>...] without the named levers of fast's compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla)"), str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            modes.resolve("turbo")
        self.assertEqual(str(cm.exception), self.UNKNOWN_TURBO)
        with self.assertRaises(levers.LeverError) as cm:                                              # the installer refuses with the same text (the kit arm prints REFUSED: <it>)
            levers.install("fast-no-bogus")
        self.assertIn("unknown mode 'fast-no-bogus' (expected off|exact|fast, or ", str(cm.exception))

    def test_autoload_accepts_the_form_and_refuses_a_bad_base(self):
        for word, refusal in (("fast-no-pallas", None), ("fast-no-bogus", None),                       # a bad lever under a good base is refused at the trigger (modes.resolve), not here
                              ("exact-no-nosub", None),                                          # exact is a base: its bad lever is refused at the trigger too
                              ("turbo-no-nosub", "unknown COLABDESIGN_OPT='turbo-no-nosub' (expected off|exact|fast, or <mode>-no-<lever>[-no-<lever>...])")):
            f = _autoload.install({"COLABDESIGN_OPT": word})
            try:
                self.assertIsInstance(f, _autoload.Finder); self.assertEqual(f.refusal, refusal)
            finally:
                sys.meta_path[:] = [x for x in sys.meta_path if x is not f]

    def test_launcher_takes_the_word(self):
        a = launch.parse_args("kit", ["--pins", "p.json", "--mode", "fast-no-pallas", "--", "--out", "o"])
        self.assertEqual((a.mode, a.script_args), ("fast-no-pallas", ["--", "--out", "o"]))            # validated by the installer after the proof, not by argparse


# ============================================================ the lines: a table mode's bytes unchanged; the subtractive word's keys
class TestLines(unittest.TestCase):
    def test_table_mode_lines_are_unchanged(self):
        """The plain words print exactly what the line printed before the subtractive form existed (no base=, no ablated=, anywhere)."""
        self.assertEqual(report.active_line(REP_FAST),
                         "ACTIVE mode=fast route=subprocess tier=2 class=precision colabdesign=1.1.3@e31a56fe jax=0.6.0 haiku=0.0.17 gpu=NVIDIA-H100-80GB-HBM3(sm90,81559MiB) "
                         f"levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla skipped=none arm=kit settings=default_4stage_multimer line={GUARANTEE_FAST}")
        off = dict(REP_FAST, mode="off", tier=None, numerics_class=None, levers=[], arm="stock", line=modes.MODES["off"].guarantee)
        self.assertEqual(report.active_line(off), "ACTIVE mode=off route=subprocess tier=none class=stock colabdesign=1.1.3@e31a56fe jax=0.6.0 haiku=0.0.17 "
                         "gpu=NVIDIA-H100-80GB-HBM3(sm90,81559MiB) levers=none skipped=none arm=stock settings=default_4stage_multimer line=stock: the design script in a subprocess proven clean of every kit variable and module")
        ev = evidence.classify([CACHE_OK, LOWERCACHE_OK, PARCOMPILE_OK, HOIST_OK, APPLIED, APPLIED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_OK], ("compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"))
        self.assertEqual(report.evidence_line(ev), "EVIDENCE levers=compilecache,lowercache,parcompile,hoist_prev,nosub,nosub_fn,trimul,triatt,opm_fold,ln,proj,transition,txla gated=none fallback=none missing=none skipped=none grad_subbatch=none fn_subbatch=none compile_cache=0/104 compile_threads=16 prev_on_device=141 trimul_served=12 trimul_fallback_by=none triatt_served=96 triatt_fallback_by=below_keys_rule:8 opm_fold_served=52 opm_fold_fallback_by=none ln_served=830 ln_fallback_by=channels_not_pow2:96 proj_served=88 proj_fallback_by=head_dim_lt_16:8 transition_served=5 transition_fallback_by=none txla_served=8 txla_fallback_by=small_call:1 pallas_served=n/a pallas_fallback=n/a pallas_fallback_by=none")
        self.assertEqual(report.exit_line(0, modes.resolve("fast"), ev, "/o"), "EXIT rc=0 mode=fast levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla evidence=applied=compilecache,lowercache,parcompile,hoist_prev,nosub,nosub_fn,trimul,triatt,opm_fold,ln,proj,transition,txla;fallback=none;missing=none out=/o")
        ev = evidence.classify([CACHE_OK, LOWERCACHE_OK, PARCOMPILE_OK, HOIST_OK], ("compilecache", "lowercache", "parcompile", "hoist_prev"))
        self.assertEqual(report.exit_line(0, modes.resolve("exact"), ev, "/o"), "EXIT rc=0 mode=exact levers=compilecache+lowercache+parcompile+hoist_prev evidence=applied=compilecache,lowercache,parcompile,hoist_prev;fallback=none;missing=none out=/o")
        self.assertEqual(report.exit_line(0, modes.resolve("off"), None, "/o"), "EXIT rc=0 mode=off levers=none evidence=n/a out=/o")

    def test_subtractive_word_lines(self):
        res = modes.resolve("fast-no-triatt")                                                        # triatt dropped: pallas, which it replaces in fast, is restored and named on every line
        rep = dict(REP_FAST, mode=res.mode, tier=res.tier, numerics_class=res.numerics_class, levers=list(res.levers), line=res.line, base=res.base, ablated=list(res.ablated), restored=list(res.restored))
        line = report.active_line(rep)
        self.assertTrue(line.startswith("ACTIVE mode=fast-no-triatt base=fast ablated=triatt restored=pallas route=subprocess tier=none class=ablation colabdesign=1.1.3@e31a56fe "), line)
        self.assertIn(" levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+pallas+opm_fold+ln+proj+transition+txla skipped=none arm=kit settings=default_4stage_multimer line=ablation: fast without triatt (restored: pallas) - ", line)
        off = triatt_lever.off_line("ablated")
        self.assertTrue(off.startswith("[colabdesign-opt] LEVER name=triatt state=off reason=ablated impl=triatt_attn"), off)
        ev = evidence.classify([off, CACHE_OK, LOWERCACHE_OK, PARCOMPILE_OK, HOIST_OK, APPLIED, APPLIED_FN, TRIMUL_OK, PALLAS_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_ASIDE], res.levers, res.ablated, res.restored)
        self.assertEqual((ev["state"], ev["ablated"], ev["restored"], ev["off"]), ({"compilecache": "applied", "lowercache": "applied", "parcompile": "applied", "hoist_prev": "applied", "nosub": "applied", "nosub_fn": "applied", "trimul": "applied", "pallas": "applied", "opm_fold": "applied", "ln": "applied", "proj": "applied", "transition": "applied", "txla": "skipped"}, ["triatt"], ["pallas"], ["triatt"]))   # the dropped lever is recorded, never classified; the restored one is classified like any lever of the run
        self.assertEqual(evidence.verdict(ev), {"partial": {}, "gated": {}, "skipped": {"txla": "cannot_run: NeedsTriatt:_lever_triatt_does_not_serve_colabdesign.af.alphafold.model.modules_in_this_run"}})   # txla composes on triatt: stepped aside by name, exit 0
        self.assertEqual(report.evidence_line(ev), "EVIDENCE levers=compilecache,lowercache,parcompile,hoist_prev,nosub,nosub_fn,trimul,pallas,opm_fold,ln,proj,transition ablated=triatt restored=pallas gated=none fallback=none missing=none skipped=txla grad_subbatch=none fn_subbatch=none compile_cache=0/104 compile_threads=16 prev_on_device=141 trimul_served=12 trimul_fallback_by=none triatt_served=n/a triatt_fallback_by=n/a opm_fold_served=52 opm_fold_fallback_by=none ln_served=830 ln_fallback_by=channels_not_pow2:96 proj_served=88 proj_fallback_by=head_dim_lt_16:8 transition_served=5 transition_fallback_by=none txla_served=n/a txla_fallback_by=n/a pallas_served=8 pallas_fallback=1 pallas_fallback_by=head_dim_lt_16:1")
        self.assertEqual(report.exit_line(0, res, ev, "/o"), "EXIT rc=0 mode=fast-no-triatt ablated=triatt restored=pallas levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+pallas+opm_fold+ln+proj+transition+txla evidence=applied=compilecache,lowercache,parcompile,hoist_prev,nosub,nosub_fn,trimul,pallas,opm_fold,ln,proj,transition;fallback=none;missing=none out=/o")
        ev = evidence.classify([off, CACHE_OK, LOWERCACHE_OK, PARCOMPILE_OK, HOIST_OK, APPLIED_FN, TRIMUL_OK, PALLAS_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_OK, TXLA_ASIDE], res.levers, res.ablated, res.restored)        # a kept lever missing: the run is all of ITS levers (partial, exit 3)
        self.assertEqual(list(evidence.verdict(ev)["partial"]), ["nosub"])
        res2 = modes.resolve("fast-no-nosub")
        ev = evidence.classify([nosub.off_line("ablated"), CACHE_OK, LOWERCACHE_OK, PARCOMPILE_OK, HOIST_OK, GATED, GATED_FN, TRIMUL_OK, PALLAS_REPLACED, TRIATT_OK, OPM_OK, LN_OK, PROJ_OK, TRANSITION_SMALL, TXLA_GATED], res2.levers, res2.ablated)   # a stray line of a dropped lever never counts
        self.assertEqual((ev["state"], ev["applied"], ev["gated"]), ({"compilecache": "applied", "lowercache": "applied", "parcompile": "applied", "hoist_prev": "applied", "nosub_fn": "gated", "trimul": "applied", "triatt": "applied", "opm_fold": "applied", "ln": "applied", "proj": "applied", "transition": "applied", "txla": "applied"}, ["compilecache", "lowercache", "parcompile", "hoist_prev", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"], ["nosub_fn"]))


# ============================================================ the kernels' one exit printer
class TestExitPrinter(unittest.TestCase):
    def setUp(self):
        kernels.reset_for_tests()
        self.saved = dict(core_report._TALLIES); core_report._TALLIES.pop(names.TAG, None)

    def tearDown(self):
        kernels.reset_for_tests(); core_report._TALLIES.clear(); core_report._TALLIES.update(self.saved)

    def test_one_printer_registry_order(self):
        self.assertTrue(kernels.register_exit_line("pallas", lambda: "P")); self.assertFalse(kernels.register_exit_line("pallas", lambda: "P2"))
        self.assertEqual(kernels.exit_lines(), "P2"); self.assertEqual(kernels.registered(), ["pallas"])
        self.assertIs(core_report._TALLIES[names.TAG], kernels.exit_lines)                            # the package's one exit slot holds the multiplexer
        kernels.register_exit_line("nosub", lambda: "N")                                               # (nosub has no exit census; registration order != registry order here on purpose)
        self.assertEqual(kernels.exit_lines(), "N\nP2")                                                # registry order, one line each
        with self.assertRaises(ValueError):
            kernels.register_exit_line("bogus", lambda: "x")


# ============================================================ end to end on the stub stack: the kit arm and `design` with fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj
@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
@unittest.skipIf(_stubs.bindcraft_stack_missing(), _stubs.SKIP_BINDCRAFT)
class TestSubtractiveWordProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cd_opt_abl_")
        cls.site = _stubs.make_stub_stack(cls.tmp)
        cls.target = _stubs.write_target(os.path.join(cls.tmp, "t.pdb"), ("A",), 108)
        cls.params = _stubs.make_params(os.path.join(cls.tmp, "params"))
        cls.env = _stubs.child_env(cls.site, PATH=_stubs.nvidia_smi_stub(cls.tmp) + os.pathsep + os.environ["PATH"], COLABDESIGN_OPT_HOME=_stubs.TREE, XDG_CACHE_HOME=os.path.join(cls.tmp, "xdg"))
        cls.arm_env = {k: v for k, v in cls.env.items() if k != "COLABDESIGN_OPT_HOME"}          # an arm's own environment holds no package variable (design composes it so; here by hand)

    def design_args(self, out, binder_len=100):
        return ["--", "--starting-pdb", self.target, "--chains", "A", "--binder-len", str(binder_len), "--seed", "0", "--params-dir", self.params, "--out", out, "--binder-name", "design"]

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_kit_arm_fast_without_pallas(self):
        """The kit arm with `--mode fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj` on the stub stack: the proof passes (the word travels as the launcher's argument, no variable),
        pallas prints its state=off reason=ablated line at install and is never imported into the model, nosub installs and prints its line at
        model build, the design runs to its outputs (the plain `fast` arm is refused here for want of the kernel: test_launchers)."""
        out = os.path.join(self.tmp, "kit_abl")
        cmd = _stubs.arm_cmd("kit", out + "_probe.json", ["--pins", _stubs.PINS, "--mode", "fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj", *self.design_args(out, 300)])   # 108 + 300 tokens: above the size gate, nosub on
        r = subprocess.run(cmd, env=self.arm_env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 0, r.stderr[-2500:])
        self.assertIn("[colabdesign-opt kit] ENV-CLEAN ok:", r.stderr)
        lever = [l for l in r.stderr.splitlines() if l.startswith("[colabdesign-opt] LEVER ")]
        self.assertEqual(len([l for l in lever if l.startswith("[colabdesign-opt] LEVER name=F1.pallas_attn state=off reason=ablated impl=pallas_attn origin=core")]), 1, lever)
        self.assertTrue(any(l.startswith("[colabdesign-opt] LEVER name=nosub state=on ") for l in lever), lever)
        fn = [l for l in lever if l.startswith("[colabdesign-opt] LEVER name=nosub_fn state=on ")]
        self.assertEqual(len(fn), 1, lever); self.assertIn(" fn_subbatch=none fn_subbatch_source=kit stock_fn_subbatch=4 gate.nosub_fn=min385 ", fn[0])
        self.assertFalse(any("state=on" in l and "F1.pallas_attn" in l for l in lever), lever)
        self.assertEqual(len([l for l in lever if l.startswith("[colabdesign-opt] LEVER name=hoist_prev state=off reason=ablated impl=hoist_prev@kit origin=kit")]), 1, lever)
        cc = [l for l in lever if l.startswith("[colabdesign-opt] LEVER name=compilecache state=on ")]          # installed before the first compile; its exit line counts jax's cache events
        self.assertEqual(len(cc), 1, lever); self.assertIn(" dir_source=default ", cc[0]); self.assertIn(" requests=2 hits=0 misses=2 ", cc[0]); self.assertIn(" source=exit ", cc[0])   # the stand-in model's two programs
        self.assertTrue(os.path.isdir(os.path.join(self.tmp, "xdg", "colabdesign_opt", "pcc")), os.listdir(self.tmp))                      # the keyed default under XDG_CACHE_HOME
        self.assertTrue(all(os.path.isfile(os.path.join(out, f)) for f in ("design.pdb", "design.fasta", "trajectory.jsonl")))
        bad = subprocess.run(_stubs.arm_cmd("kit", out + "_probe2.json", ["--pins", _stubs.PINS, "--mode", "fast-no-bogus", *self.design_args(out + "2")]), env=self.arm_env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(bad.returncode, 3); self.assertIn("[colabdesign-opt kit] REFUSED: unknown mode 'fast-no-bogus' (expected off|exact|fast, or exact-no-<lever>", bad.stderr)

    @unittest.skipUnless(_stubs.dssp_present(), _stubs.SKIP_DSSP)
    def test_design_fast_without_pallas(self):
        """`design --mode fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla`: ACTIVE names the word, its base and the dropped lever with tier none / class ablation; the arm runs;
        EVIDENCE and EXIT carry ablated=pallas; at 208 tokens nosub is gated (stock's programs), which is named and exit 0."""
        out = os.path.join(self.tmp, "design_abl")
        r = subprocess.run([sys.executable, "-m", "colabdesign_opt", "design", "--mode", "fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla", "--starting-pdb", self.target, "--chains", "A", "--binder-len", "100",
                            "--seed", "0", "--out", out, "--params-dir", self.params], env=self.env, capture_output=True, text=True, cwd=self.tmp)
        lines = [l for l in r.stderr.splitlines() if l.startswith("[colabdesign-opt]")]
        self.assertEqual(r.returncode, 0, r.stderr[-2500:])
        self.assertTrue(lines[0].startswith("[colabdesign-opt] ACTIVE mode=fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla base=fast ablated=hoist_prev,trimul,pallas,triatt,opm_fold,ln,proj,transition,txla route=subprocess tier=none class=ablation "), lines[0])
        self.assertIn(" levers=compilecache+lowercache+parcompile+nosub+nosub_fn skipped=none arm=kit settings=default_4stage_multimer line=ablation: fast without hoist_prev,trimul,pallas,triatt,opm_fold,ln,proj,transition,txla - an A/B attribution run, none of the table's modes, no guarantee against stock claimed", lines[0])
        self.assertTrue(lines[1].startswith("[colabdesign-opt] RUN mode=fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla "), lines[1])
        self.assertTrue(any(l.startswith("[colabdesign-opt] EVIDENCE levers=compilecache,lowercache,parcompile ablated=hoist_prev,trimul,pallas,triatt,opm_fold,ln,proj,transition,txla gated=nosub,nosub_fn fallback=none missing=none skipped=none grad_subbatch=none fn_subbatch=none compile_cache=0/2 compile_threads=") for l in lines), lines)
        self.assertTrue(lines[-1].startswith(f"[colabdesign-opt] EXIT rc=0 mode=fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla ablated=hoist_prev,trimul,pallas,triatt,opm_fold,ln,proj,transition,txla levers=compilecache+lowercache+parcompile+nosub+nosub_fn evidence=applied=compilecache,lowercache,parcompile;fallback=none;missing=none out={out}"), lines[-1])   # 208 tokens: nosub/nosub_fn gated (exit 0), compilecache applied
        env_word = subprocess.run([sys.executable, "-m", "colabdesign_opt", "check", "--mode", "fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla"], env=dict(self.env, COLABDESIGN_OPT="fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla"), capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(env_word.returncode, 0, env_word.stderr[-1500:]); self.assertIn("ACTIVE mode=fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla base=fast ablated=hoist_prev,trimul,pallas,triatt,opm_fold,ln,proj,transition,txla route=check tier=none class=ablation ", env_word.stderr)
        clash = subprocess.run([sys.executable, "-m", "colabdesign_opt", "check", "--mode", "fast"], env=dict(self.env, COLABDESIGN_OPT="fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla"), capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(clash.returncode, 2); self.assertIn("disagrees with COLABDESIGN_OPT=fast-no-hoist_prev-no-trimul-no-pallas-no-triatt-no-opm_fold-no-ln-no-proj-no-transition-no-txla", clash.stderr)   # the variable and --mode must agree, word for word


if __name__ == "__main__":
    unittest.main()
