"""Every surface that can bring the kit's optimizations into force, and every gate that can refuse to: the API/env/check routes
in-process (gates by name -- pins, core pin, no GPU; late-activation and mode-change refusal; `off`; `exact` refused as an unknown name;
`fast` refused by name where the kernel cannot serve; the partial line; `status()` before and after); the .pth-triggered autoload
hook across a real subprocess (nothing of the core/jax/colabdesign is imported at interpreter start; the env-route trigger on `import
colabdesign`; a stale or absent core is refused by name, never a bare exit or a traceback); and the same hook through a REAL editable
install in a fresh venv (pip's own `.pth` sort order, the config/run.sh import probes, the env-route trigger) -- proving the auto-activation
path is correct however the package actually reaches an interpreter, not just when tests import it directly."""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

try:
    import pytest
except ImportError:                                        # an interpreter without pytest (the stack image): the kit's stand-in
    from colabdesign_opt.tests import _noptest as pytest

from . import _stubs
import colabdesign_opt
from colabdesign_opt import _autoload, stack


# ============================================================ the activation contract in-process: gates, refusals, partial activation, status()
def modes_levers(word):
    from colabdesign_opt import modes
    return list(modes.resolve(word).levers)


def fake_hoist(mp):
    """The stand-in design surface is not ColabDesign's pinned design loop (hoist_prev refuses it by name): for the activation-plumbing tests,
    `hoist_prev.install` marks the stand-in class and reports itself; the lever's own numerics test runs on the image (test_hoist_prev)."""
    from colabdesign_opt import hoist_prev

    def install():
        from colabdesign.af.design import _af_design
        for n in hoist_prev.PATCHED:
            f = (lambda self, *a, **k: None); setattr(f, hoist_prev.MARKER, True); setattr(_af_design, n, f)
        return {"impl": hoist_prev.IMPL, "patched": list(hoist_prev.PATCHED), "already": False}
    mp.setattr(hoist_prev, "install", install)


def fake_kernel(mp):
    """The stub stack has no Pallas kernel: for the activation-plumbing tests, `pallas.install` and `kernels.trimul_fused.install` are replaced by
    stand-ins that mark the stub `Attention` / `TriangleMultiplication` classes as served and report themselves (`impl=stand-in`). The kernels'
    own refusals on this stack: test_fast_refused_*."""
    from colabdesign_opt import pallas
    from colabdesign_opt.kernels import trimul_fused

    def install():
        from colabdesign.af.alphafold.model import modules as CDM
        setattr(CDM.Attention, pallas.MARKER, True)
        return {"impl": "stand-in", "origin": "test", "scope": pallas.SCOPE, "expected_fallbacks": list(pallas.EXPECTED_FALLBACKS), "jax": None}
    mp.setattr(pallas, "install", install)

    def install_trimul():
        from colabdesign.af.alphafold.model import modules as CDM
        if not hasattr(CDM, "TriangleMultiplication"):
            CDM.TriangleMultiplication = type("TriangleMultiplication", (), {})
        setattr(CDM.TriangleMultiplication, trimul_fused.MARKER, True)
        return {"impl": "stand-in", "origin": "test", "numerics": trimul_fused.NUMERICS, "expected_fallbacks": list(trimul_fused.EXPECTED_FALLBACKS), "jax": None}
    mp.setattr(trimul_fused, "install", install_trimul)

    from colabdesign_opt.kernels import triatt_lever                                # this kit's attention kernels: the same interception, no Triton on the stand-in

    def install_triatt():
        from opt_core.counters import Ledger
        triatt_lever.LEDGER = Ledger(triatt_lever.LEVER, impl="triatt_attn@5tub5tub", origin="kit", min_tokens=0, expected=triatt_lever.EXPECTED_FALLBACKS)
        A = sys.modules["colabdesign.af.alphafold.model.modules"].Attention
        setattr(A, triatt_lever.MARKER, True); A._ledger = triatt_lever.LEDGER
        return {"impl": triatt_lever.LEDGER.impl, "origin": "kit"}
    mp.setattr(triatt_lever, "install", install_triatt)

    from colabdesign_opt.kernels import layers_opm, layers_ln                          # the layer levers: class markers on the stand-in modules

    def install_opm():
        from colabdesign.af.alphafold.model import modules as CDM
        if not hasattr(CDM, "OuterProductMean"):
            CDM.OuterProductMean = type("OuterProductMean", (), {})
        setattr(CDM.OuterProductMean, layers_opm.MARKER, True); layers_opm._STATE["installed"] = True
        return {"lever": layers_opm.NAME, "impl": layers_opm.IMPL}
    mp.setattr(layers_opm, "install", install_opm)

    def install_ln():
        layers_ln._STATE["installed"] = True
        return {"lever": layers_ln.NAME, "impl": layers_ln.IMPL}
    mp.setattr(layers_ln, "install", install_ln)

    from colabdesign_opt.kernels import layers_transition                            # the fused Transition MLP: a class marker on the stand-in modules, like the layer levers

    def install_transition():
        layers_transition._STATE["installed"] = True
        return {"lever": layers_transition.NAME, "impl": layers_transition.IMPL}
    mp.setattr(layers_transition, "install", install_transition)

    from colabdesign_opt.kernels import proj_attn                                     # the fused projections: this lever's own ledger and marker on the stand-in class

    def install_proj():
        from opt_core.counters import Ledger
        proj_attn.LEDGER = Ledger(proj_attn.LEVER, impl=proj_attn.IMPL, origin="kit", expected=proj_attn.EXPECTED_FALLBACKS)
        A = sys.modules["colabdesign.af.alphafold.model.modules"].Attention
        setattr(A, proj_attn.MARKER, True); proj_attn._STATE["installed_on"] = A
        return {"impl": proj_attn.IMPL, "origin": "kit"}
    mp.setattr(proj_attn, "install", install_proj)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
@unittest.skipIf("numpy" in _stubs.bindcraft_stack_missing(), "the stand-in model builds its arrays with numpy: pip install -e opt[test]")
class TestActivation(unittest.TestCase):
    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.tmp = tempfile.mkdtemp(prefix="cd_opt_act_")
        self.site = _stubs.make_stub_stack(self.tmp); sys.path.insert(0, self.site)
        for k in ("COLABDESIGN_OPT", "AF2M_LEVERS", "AF_PALLAS_ATTN", "MODEL_OPT_TARGET_GPU"):
            self.mp.delenv(k, raising=False)
        self.mp.setenv("COLABDESIGN_OPT_HOME", _stubs.TREE)
        self.gpu = dict(_stubs.GPU)
        self.mp.setattr(stack, "nvidia_smi_probe", lambda: dict(self.gpu))
        self.mp.setenv("XDG_CACHE_HOME", os.path.join(self.tmp, "xdg"))                          # lever compilecache's keyed default directory: under the test's tmp, never ~/.cache
        self.mp.setenv("PATH", _stubs.nvidia_smi_stub(self.tmp) + os.pathsep + os.environ["PATH"])   # the cache key names the GPU product (opt_core pcc reads nvidia-smi)

    def tearDown(self):
        self.mp.undo(); self.state.restore()

    def test_status_before(self):
        st = colabdesign_opt.status()
        self.assertFalse(st["active"]); self.assertIn("has not run", st["reason"])

    def test_dry_run_and_gates(self):
        rep = stack.activate("fast", route="check", dry_run=True)
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["route"], "check"); self.assertTrue(rep["core"]["ok"]); self.assertNotIn("levers_installed", rep)
        self.assertEqual((rep["levers"], rep["arm"], rep["numerics_class"]), (["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"], "kit", "precision"))
        self.assertFalse(colabdesign_opt.status()["active"])                                  # a dry run activates nothing
        self.gpu = {}                                                                          # no GPU visible
        from colabdesign_opt import report
        rep = stack.activate("fast", route="check", dry_run=True)                              # no card: the levers that need the GPU backend will step aside BY NAME in the arm — foretold on the dry-run route
        self.assertTrue(rep["active"], rep.get("reason"))                                      # (the kit rule: a lever that cannot engage never refuses the mode), the rest engage
        self.assertEqual(list(rep["skipped"]), ["compilecache", "lowercache", "trimul", "triatt", "ln", "proj", "transition", "txla"], rep["skipped"])   # levers.NEEDS_GPU ∩ fast
        self.assertTrue(rep["skipped"]["trimul"].startswith("no_gpu: no GPU visible (nvidia-smi)"), rep["skipped"])
        self.assertEqual(rep["levers"], ["parcompile", "hoist_prev", "nosub", "nosub_fn", "opm_fold"])
        self.assertIn(" levers=parcompile+hoist_prev+nosub+nosub_fn+opm_fold skipped=compilecache,lowercache,trimul,triatt,ln,proj,transition,txla arm=kit ", report.active_line(rep))
        rep = stack.activate("exact", route="check", dry_run=True)                             # exact without a card: its compile cache has no GPU product to be keyed by — it steps aside, named; the mode is served
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual((list(rep["skipped"]), rep["levers"]), (["compilecache", "lowercache"], ["parcompile", "hoist_prev"]))
        self.gpu = dict(_stubs.GPU)
        rep = stack.activate("exact", route="check", dry_run=True)
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual((rep["mode"], rep["tier"], rep["numerics_class"], rep["levers"]), ("exact", 1, "exact", ["compilecache", "lowercache", "parcompile", "hoist_prev"]))
        self.assertTrue(stack.activate("off", route="check", dry_run=True)["active"])          # stock has no GPU gate

    def test_pin_refusal(self):
        site = _stubs.make_stub_stack(os.path.join(self.tmp, "other"), commit="0" * 40)
        sys.path.insert(0, site)
        rep = stack.activate("fast", route="check", dry_run=True)
        self.assertFalse(rep["active"]); self.assertIn("installed from commit 000000000000", rep["reason"])
        with self.assertRaises(stack.ActivationError):
            stack.activate("fast", route="check", dry_run=True, strict=True)

    def test_core_gate_facts_on_the_record(self):
        """The kit route's record carries the core pin's facts from THE gate (_core_gate.py, run at every entry): the pin is a FLOOR — installed >= pinned here."""
        rep = stack.activate("fast", route="check", dry_run=True)
        self.assertTrue(rep["active"], rep.get("reason")); self.assertTrue(rep["core"]["ok"])
        from colabdesign_opt import _core_gate
        self.assertLessEqual(_core_gate.version_tuple(rep["core"]["pinned"]["version"]), _core_gate.version_tuple(rep["core"]["installed"]["version"]))   # the pin is a floor (>=), not an equality
        self.mp.setattr(_core_gate, "installed_core", lambda: {"package_dir": "/x/opt_core", "root": "/x", "version": "0.0.0"})
        rep = stack.activate("fast", route="check", dry_run=True)                              # in-process (a caller that skipped the entry gate): refused with the gate's own words
        self.assertFalse(rep["active"]); self.assertIn("core_mismatch", rep["reason"]); self.assertIn("v0.0.0", rep["reason"])
        self.assertTrue(stack.activate("off", route="check", dry_run=True)["active"])         # stock imports no core lever

    def test_api_route_applies_once(self):
        fake_kernel(self.mp); fake_hoist(self.mp)
        rep = colabdesign_opt.enable("fast")
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["levers_installed"], ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition"]); self.assertEqual(list(rep["skipped"]), ["txla"]); self.assertTrue(rep["skipped"]["txla"].startswith("cannot_run: NeedsTriatt:"), rep["skipped"]); self.assertEqual(rep["route"], "api")   # txla composes on triatt's served class, which the stub kernel does not serve: stepped aside by name
        self.assertEqual(stack.kit_markers(), {"compilecache": True, "lowercache": True, "parcompile": True, "hoist_prev": True, "nosub": True, "nosub_fn": True, "trimul": True, "pallas": False, "triatt": True, "opm_fold": True, "ln": True, "proj": True, "transition": True, "txla": False})
        again = colabdesign_opt.enable("fast")
        self.assertEqual(again["levers_installed"], ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition"]); self.assertTrue(again["active"])
        other = colabdesign_opt.enable("off")
        self.assertFalse(other["active"]); self.assertIn("already activated mode=fast", other["reason"])
        with self.assertRaises(colabdesign_opt.ActivationError):
            colabdesign_opt.enable("off", strict=True)
        self.assertEqual(colabdesign_opt.status()["levers_installed"], ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition"])
        import colabdesign
        m = colabdesign.mk_afdesign_model(protocol="binder", use_multimer=True)
        m.prep_inputs(_stubs.write_target(os.path.join(self.tmp, "t.pdb"), ("A",), 108), "A", 400)      # 508 tokens: the design step unchunked, fn at stock's chunk
        from colabdesign_opt import nosub
        b = nosub.builds()
        self.assertEqual((len(b), b[0]["tokens"], b[0]["grad"]["value"], b[0]["fn"]["value"]), (1, 508, None, None)); self.assertTrue(b[0]["gate"].startswith("served"), b[0]["gate"])   # fast: both executables unchunked (nosub + nosub_fn)
        self.assertEqual(b[0]["policy"], {"grad": "kit", "fn": "kit"}); self.assertEqual(len(b[0]["lines"]), 2); self.assertIn(" name=nosub_fn state=on ", b[0]["lines"][1])
        self.assertIsNone(m._model["runner"].config.model.global_config.subbatch_size)             # the config the design step was built with: unchunked

    def test_exact_applies_its_two_levers(self):
        """`exact` = the exact-class levers (compilecache + hoist_prev; no kernel, so the stand-in stack hosts both): the API route installs them,
        the report names tier 1 / class 1 and the two levers; an unknown word is refused with the one text, nothing installed."""
        from colabdesign_opt import report
        fake_kernel(self.mp); fake_hoist(self.mp)
        rep = colabdesign_opt.enable("exact", strict=True)
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual((rep["mode"], rep["tier"], rep["numerics_class"], rep["levers"], rep["levers_installed"]), ("exact", 1, "exact", ["compilecache", "lowercache", "parcompile", "hoist_prev"], ["compilecache", "lowercache", "parcompile", "hoist_prev"]))
        self.assertEqual(stack.kit_markers(), {"compilecache": True, "lowercache": True, "parcompile": True, "hoist_prev": True, "nosub": False, "nosub_fn": False, "trimul": False, "pallas": False, "triatt": False, "opm_fold": False, "ln": False, "proj": False, "transition": False, "txla": False})
        from colabdesign_opt import compilecache_jax, hoist_prev
        self.assertTrue(compilecache_jax.evidence()["dir"].startswith(os.path.join(self.tmp, "xdg", "colabdesign_opt", "pcc")), compilecache_jax.evidence())   # the keyed default under XDG_CACHE_HOME
        self.assertEqual(hoist_prev.evidence()["patched"], ["_recycle", "run"])
        stack.reset_for_tests()
        self.mp.undo(); self.mp.setenv("COLABDESIGN_OPT_HOME", _stubs.TREE); self.mp.setattr(stack, "nvidia_smi_probe", lambda: dict(self.gpu)); self.mp.setenv("XDG_CACHE_HOME", os.path.join(self.tmp, "xdg")); self.mp.setenv("PATH", _stubs.nvidia_smi_stub(self.tmp) + os.pathsep + os.environ["PATH"])
        rep = colabdesign_opt.enable("exact")                                                     # the real hoist_prev on the stand-in design surface (not ColabDesign's pinned loop): the LEVER steps aside by name, the mode is served with the rest
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual((rep["levers"], list(rep["skipped"])), (["compilecache", "lowercache", "parcompile"], ["hoist_prev"]))
        self.assertTrue(rep["skipped"]["hoist_prev"].startswith("cannot_run: HoistPrevError: "), rep["skipped"]); self.assertIn("defines no `_recycle`", rep["skipped"]["hoist_prev"])
        self.assertIn(" levers=compilecache+lowercache+parcompile skipped=hoist_prev arm=kit ", report.active_line(rep))
        stack.reset_for_tests()
        with self.assertRaises(colabdesign_opt.ActivationError) as cm:
            colabdesign_opt.enable("turbo", strict=True)
        self.assertIn("unknown mode 'turbo' (expected off|exact|fast, or exact-no-<lever>[-no-<lever>...] without the named levers of exact's compilecache+lowercache+parcompile+hoist_prev, fast-no-<lever>[-no-<lever>...] without the named levers of fast's compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla)", str(cm.exception))

    def test_fast_levers_step_aside_by_name_without_the_kernel(self):
        """The stub stack has no Pallas kernel: `fast` is SERVED — every lever that cannot engage here steps aside by name (one `state=skipped
        reason=cannot_run` line each, the kernel's own refusal text in `detail=`), the levers that compose on an attention kernel step aside as
        `no_attention_kernel`, and the rest engage: the kit rule (a lever never refuses the mode). Every lever of the set is accounted for."""
        import contextlib, io
        from colabdesign_opt import levers, report
        fake_hoist(self.mp)                                                                       # past hoist_prev (its own step-aside on the stand-in surface: test_exact_applies_its_two_levers) to the kernels'
        word = "fast-no-opm_fold"                                                                 # the stand-in modules carry no OuterProductMean class for opm_fold to rebind (its coverage: test_layers_opm on the real stack): dropped by the word, named
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            rep = colabdesign_opt.enable(word, strict=True)                                         # strict raises only for configuration: nothing here
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["mode"], word)
        self.assertEqual(sorted(rep["levers"] + list(rep["skipped"])), sorted(modes_levers(word)))     # every lever of the run: on, or stepped aside by name — none unnamed
        self.assertIn("trimul", rep["skipped"]); self.assertTrue(rep["skipped"]["trimul"].startswith("cannot_run: "), rep["skipped"])
        for kl in levers.ATTENTION_KERNEL_LEVERS:
            self.assertNotIn(kl, rep["levers"])                                                   # no attention kernel engaged on the stub …
        for l in levers.NEEDS_ATTENTION_KERNEL:
            self.assertTrue(rep["skipped"][l].startswith("no_attention_kernel: "), (l, rep["skipped"]))   # … so the levers composing on one stepped aside by that name
        for l, why in rep["skipped"].items():
            self.assertRegex(err.getvalue(), rf"\[colabdesign-opt\] LEVER name={levers.registry.LEVERS[l].line_name.replace('.', '[.]')} state=skipped reason={why.split(':')[0]} ", l)   # one skipped line per lever, printed at install
        self.assertIn(f" skipped={','.join(rep['skipped'])} arm=kit ", report.active_line(rep)); self.assertIn(f" levers={'+'.join(rep['levers'])} ", report.active_line(rep))
        self.assertRegex(err.getvalue(), r"\[colabdesign-opt\] LEVER name=opm_fold state=off reason=ablated ")   # the dropped lever's one line
        self.assertEqual(colabdesign_opt.status()["levers_installed"], rep["levers"])
        stack.reset_for_tests()

    def test_a_missing_lever_at_activation_refuses_the_mode(self):
        """A lever of the mode the installer neither put in place nor stepped aside is an installer DEFECT, refused by name on the env / api
        routes (never a silent subset): not active, the reason names the levers, `strict` raises."""
        from colabdesign_opt import levers, modes, report
        self.mp.setattr(levers, "install", lambda mode: {"mode": mode, "levers": ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn", "trimul", "triatt", "opm_fold", "ln", "proj", "transition", "txla"], "levers_installed": ["compilecache", "lowercache", "parcompile", "hoist_prev", "nosub", "nosub_fn"], "pallas": None})
        rep = stack.activate("fast", route="api")
        self.assertFalse(rep["active"]); self.assertEqual(rep["reason"], "mode fast: lever(s) trimul,triatt,opm_fold,ln,proj,transition,txla neither installed nor stepped aside at activation (installed=compilecache,lowercache,parcompile,hoist_prev,nosub,nosub_fn skipped=none) — an installer defect")
        self.assertEqual(report.active_line(rep), f"NOT ACTIVE reason={rep['reason']}"); self.assertNotIn("levers_installed", rep)
        stack.reset_for_tests()
        with self.assertRaises(stack.ActivationError):
            stack.activate("fast", route="api", strict=True)
        self.assertFalse(hasattr(modes, "ENV_ALLOW_PARTIAL")); self.assertFalse(hasattr(stack, "PartialActivation"))   # no opt-out: nothing runs under a mode's name with a subset of its levers

    def test_stack_drift_is_named_not_refused(self):
        """A stack distribution off its pin (stock/PINS.json pins_asserted) is `stack_drift={dist: installed!=pinned}` in the report and on the ACTIVE
        line; the activation stands and the levers engage. Only the colabdesign commit is a gate (test_pin_refusal)."""
        from colabdesign_opt import report
        p = _stubs.pins()
        py = ".".join(map(str, sys.version_info[:3]))
        asserted = [n for n in (p.get("pins_asserted") or []) if n != "colabdesign"]
        self.assertTrue(asserted, p.get("pins_asserted"))
        name = asserted[0]
        fake = dict(p, pins=dict(p["pins"], **{name: "0.0.0.dev0"}), python=py)
        drift = stack.stack_drift(fake)
        self.assertIn(name, drift); self.assertTrue(drift[name].endswith("!=0.0.0.dev0"), drift); self.assertNotIn("python", drift)
        self.assertEqual(stack.stack_drift(dict(p, pins_asserted=["colabdesign"], python=py)), {})          # nothing asserted but the commit (pin_refusal's), the interpreter as pinned: no drift
        vi = sys.version_info
        patch = f"{vi[0]}.{vi[1]}.{vi[2] + 3}"
        self.assertEqual(stack.stack_drift(dict(p, pins_asserted=["colabdesign"], python=patch)), {"python": f"{py}!={patch}"})   # another patch release of the pinned series: named (stock/check_pins.py python_pin, the rule `run.sh install` warns-and-runs on), never refused
        self.assertEqual(stack.load_check_pins().python_pin({"python": patch})["status"], "not_pinned")
        self.mp.setattr(stack, "pins", lambda environ=None: fake)
        rep = stack.activate("fast", route="check", dry_run=True)
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["stack_drift"], drift)
        self.assertIn(f" stack_drift={name}:{drift[name]}", report.active_line(rep) + " ")            # named on the ACTIVE line, the activation stands

    def test_late_activation_refused(self):
        fake_kernel(self.mp); fake_hoist(self.mp)
        import colabdesign
        m = colabdesign.mk_afdesign_model(protocol="binder", use_multimer=True)
        rep = colabdesign_opt.enable("fast")
        self.assertFalse(rep["active"]); self.assertIn("1 mk_af_model instance(s) already exist", rep["reason"])
        del m
        import gc; gc.collect()
        self.assertEqual(stack.model_instances(), 0)
        rep = colabdesign_opt.enable("fast")
        self.assertTrue(rep["active"], rep.get("reason"))                                      # no instance left: activation allowed

    def test_markers_from_another_caller_refused(self):
        from colabdesign_opt import nosub
        import colabdesign  # noqa: F401
        nosub.install()                                                                          # another caller put the patch in place, not activate()
        rep = colabdesign_opt.enable("fast")
        self.assertFalse(rep["active"]); self.assertIn("already installed in this process by another caller", rep["reason"])

    def test_off_applies_nothing(self):
        rep = colabdesign_opt.enable("off")
        self.assertTrue(rep["active"]); self.assertEqual(rep["levers"], []); self.assertNotIn("levers_installed", rep)
        self.assertEqual(stack.kit_markers(), {"compilecache": False, "lowercache": False, "parcompile": False, "hoist_prev": False, "nosub": False, "nosub_fn": False, "trimul": False, "pallas": False, "triatt": False, "opm_fold": False, "ln": False, "proj": False, "transition": False, "txla": False})

    def test_unknown_name(self):
        rep = colabdesign_opt.enable("turbo")
        self.assertFalse(rep["active"]); self.assertIn("turbo", rep["reason"])
        with self.assertRaises(colabdesign_opt.ActivationError):
            colabdesign_opt.enable("turbo", strict=True)

    def test_card_check(self):
        p = _stubs.pins()
        self.assertTrue(stack.card_check(p, _stubs.GPU)["ok"])
        self.assertNotIn("note", stack.card_check(p, _stubs.GPU))                                       # the tested card: no note key at all
        a100 = stack.card_check(p, {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920, "cc": "8.0"})
        self.assertEqual((a100["ok"], a100["reason"], a100["note"]), (True, None, "card 'NVIDIA A100-SXM4-80GB' != 'NVIDIA H100 80GB HBM3' — not the tested card"))   # another card: ok, a note
        self.assertIn("memory", stack.card_check(p, {"name": _stubs.GPU["name"], "memory_mib": 40000, "cc": "9.0"})["note"])
        v100 = stack.card_check(p, {"name": "Tesla V100-SXM2-32GB", "memory_mib": 32768, "cc": "7.0"})
        self.assertEqual((v100["ok"], v100["reason"], v100["note"]), (True, None, "card 'Tesla V100-SXM2-32GB' != 'NVIDIA H100 80GB HBM3'; memory 32768 MiB outside 1% of 81559 MiB; compute capability 7.0 < 8.0, the Pallas kernel's floor — not the tested card"))   # below the kernel's floor: named, never refused — the levers engage
        self.assertTrue(stack.card_check(p, {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920})["ok"])                    # capability unknown: not refused
        self.assertEqual(stack.card_check(p, {})["reason"], "no GPU visible")

    def test_a_card_below_the_kernel_floor_is_named_and_engaged(self):
        """No visible card is refused: one below PINS.json gpu.cc_min (the Pallas kernel's floor) activates every mode with the same lever set,
        named on the ACTIVE line and in the card's note; another card at or above it likewise (the A100); `off` has no GPU logic at all."""
        from colabdesign_opt import report
        self.gpu = {"name": "Tesla V100-SXM2-32GB", "memory_mib": 32768, "cc": "7.0"}
        for mode, levers in (("fast", "compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla"), ("exact", "compilecache+lowercache+parcompile+hoist_prev")):
            rep = stack.activate(mode, route="check", dry_run=True)
            self.assertTrue(rep["active"], rep.get("reason")); self.assertIn("compute capability 7.0 < 8.0, the Pallas kernel's floor — not the tested card", rep["card"]["note"])
            self.assertIn(f" gpu=Tesla-V100-SXM2-32GB(sm70,32768MiB) levers={levers} skipped=none arm=kit ", report.active_line(rep))   # the card names itself; the lever set is the mode's, whole
        self.assertTrue(stack.activate("off", route="check", dry_run=True)["active"])
        self.gpu = {"name": "NVIDIA A100-SXM4-80GB", "memory_mib": 81920, "cc": "8.0"}
        rep = stack.activate("fast", route="check", dry_run=True)
        self.assertTrue(rep["active"], rep.get("reason")); self.assertEqual(rep["card"]["note"], "card 'NVIDIA A100-SXM4-80GB' != 'NVIDIA H100 80GB HBM3' — not the tested card")
        self.assertIn(" gpu=NVIDIA-A100-SXM4-80GB(sm80,81920MiB) levers=compilecache+lowercache+parcompile+hoist_prev+nosub+nosub_fn+trimul+triatt+opm_fold+ln+proj+transition+txla skipped=none arm=kit ", report.active_line(rep))   # the A100 names itself on the ACTIVE line; no other token changes


# ============================================================ the .pth-triggered autoload hook, across a real subprocess
WATCHED_ROOTS = ("colabdesign_opt", "opt_core", "colabdesign", "jax", "jaxlib", "haiku", "numpy", "af2m_levers", "colabdesign_pallas_attn", "af2_flash_pallas")
STARTUP_PROBE = ("import sys\nprint('LOADED', sorted(m for m in sys.modules if m.split('.')[0] in " + repr(WATCHED_ROOTS) + "))\n"
                 "print('TRIG', [type(f).__name__ for f in sys.meta_path if type(f).__module__.startswith('colabdesign_opt')])\n"
                 "print('HASHLIB', 'hashlib' in sys.modules)")
STARTUP_CHAIN = ["colabdesign_opt", "colabdesign_opt._autoload"]                               # interpreter start, COLABDESIGN_OPT unset or off
STARTUP_CHAIN_MODE = ["colabdesign_opt", "colabdesign_opt._autoload", "colabdesign_opt.names"]   # a mode set: + the import-free names for the mode-name check


class TestAutoloadUnit(unittest.TestCase):
    def tearDown(self):
        sys.meta_path[:] = [f for f in sys.meta_path if not isinstance(f, _autoload.Finder)]

    def test_install(self):
        self.assertIsNone(_autoload.install({}))
        self.assertIsNone(_autoload.install({"COLABDESIGN_OPT": "off"}))
        f = _autoload.install({"COLABDESIGN_OPT": "turbo"})                        # a refusing finder: nothing printed until the trigger, exit 3 there
        self.assertIsInstance(f, _autoload.Finder); self.assertIn("unknown COLABDESIGN_OPT='turbo'", f.refusal); sys.meta_path.remove(f)
        for name in ("exact", "fast"):
            f = _autoload.install({"COLABDESIGN_OPT": name})
            self.assertIsInstance(f, _autoload.Finder); self.assertIsNone(f.refusal); self.assertIs(_autoload.install({"COLABDESIGN_OPT": name}), f); self.assertIn(f, sys.meta_path)
            sys.meta_path.remove(f)

    def test_env_name_is_the_tables(self):
        from colabdesign_opt import modes
        self.assertEqual(_autoload.ENV, modes.ENV)                                     # restated import-free in the hook; held equal here

    def test_driver_process_refusal(self):
        for a0 in ("stock_design.py", "kit_launch.py", "/x/y/stock_launch.py"):
            self.assertIsNotNone(_autoload.driver_process_refusal([a0]), a0)
        self.assertIsNone(_autoload.driver_process_refusal(["/home/u/my_pipeline.py"]))
        self.assertIsNone(_autoload.driver_process_refusal(["-c"]))


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestAutoloadProcess(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="cd_opt_autoload_")
        cls.site = _stubs.make_stub_stack(cls.tmp)
        d = _stubs.nvidia_smi_stub(cls.tmp)
        cls.env = _stubs.child_env(cls.site, PATH=d + os.pathsep + os.environ["PATH"], COLABDESIGN_OPT_HOME=_stubs.TREE)

    def run_py(self, code, **env):
        return subprocess.run([sys.executable, "-c", "import colabdesign_opt._autoload\n" + code], env=dict(self.env, **env), capture_output=True, text=True, cwd=self.tmp)

    def test_env_route_activates_on_import(self):
        r = self.run_py("import colabdesign, colabdesign_opt\nprint('reached', colabdesign_opt.status()['active'], '+'.join(colabdesign_opt.status()['levers_installed']))", COLABDESIGN_OPT="exact")
        self.assertEqual(r.returncode, 0, r.stderr[-1500:])                                    # the stand-in design surface is not ColabDesign's pinned loop: hoist_prev STEPS ASIDE by name at the trigger, exact is served with the rest (the kit rule)
        self.assertRegex(r.stderr, r"\[colabdesign-opt\] LEVER name=hoist_prev state=skipped reason=cannot_run impl=hoist_prev@kit origin=kit detail=HoistPrevError:\S+ source=install pid=\d+")
        self.assertIn("[colabdesign-opt] ACTIVE mode=exact route=env tier=1 class=exact ", r.stderr); self.assertIn(" levers=compilecache+lowercache+parcompile skipped=hoist_prev arm=kit ", r.stderr)
        self.assertIn("reached True compilecache+lowercache+parcompile", r.stdout); self.assertNotIn("NOT ACTIVE", r.stderr)
        r = self.run_py("import colabdesign, colabdesign_opt\nprint('unreachable')", COLABDESIGN_OPT="turbo")
        self.assertEqual(r.returncode, 3); self.assertIn("[colabdesign-opt] NOT ACTIVE reason=unknown COLABDESIGN_OPT='turbo' (expected off|exact|fast, or <mode>-no-<lever>[-no-<lever>...])", r.stderr); self.assertNotIn("unreachable", r.stdout)   # configuration refuses, by name
        r = self.run_py("import colabdesign, colabdesign_opt\nst = colabdesign_opt.status(); print('reached', st['active'], '+'.join(st['levers_installed']), ','.join(st.get('skipped') or {}))", COLABDESIGN_OPT="fast")
        self.assertEqual(r.returncode, 0, r.stderr[-2500:]); self.assertIn("reached True ", r.stdout)   # the stand-in is neither ColabDesign's pinned loop nor a Pallas host: those levers step aside by name, fast runs the rest
        self.assertIn("[colabdesign-opt] ACTIVE mode=fast route=env tier=2 class=precision ", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
        act = [l for l in r.stderr.splitlines() if l.startswith("[colabdesign-opt] ACTIVE ")][0]
        on, aside = act.split(" levers=")[1].split(" ")[0].split("+"), act.split(" skipped=")[1].split(" ")[0].split(",")
        self.assertIn("hoist_prev", aside); self.assertIn("trimul", aside); self.assertIn("nosub", on)
        for l in aside:
            self.assertRegex(r.stderr, rf"\[colabdesign-opt\] LEVER name=\S+ state=skipped reason=(cannot_run|no_attention_kernel) (\S+ )*source=install", l)
        self.assertEqual(sorted(on + aside), sorted(modes_levers("fast")))                       # every lever of the mode named: on, or stepped aside
        r = self.run_py("import colabdesign, colabdesign_opt, sys\nprint('ST', colabdesign_opt.status()['active'], [type(f).__name__ for f in sys.meta_path if 'colabdesign_opt' in type(f).__module__])")
        self.assertEqual(r.returncode, 0); self.assertIn("ST False []", r.stdout)             # unset: nothing attaches

    def test_startup_imports_nothing(self):
        """After `import colabdesign_opt._autoload` (what the .pth line does at interpreter start) the interpreter holds exactly the package and
        the hook of everything kit / core / upstream — measured ABSOLUTELY, so the assertion is the same whether this interpreter reached the package
        through PYTHONPATH or through an installed .pth that already ran the line (TestInstalledPthCoreAbsent, below, covers the installed layout); with a
        mode set, `names` (import-free) joins for the mode-name check and the Finder is armed; unset / off arm nothing."""
        for env, expect, finders in (({}, STARTUP_CHAIN, []), ({"COLABDESIGN_OPT": "off"}, STARTUP_CHAIN, []),
                                     ({"COLABDESIGN_OPT": "fast"}, STARTUP_CHAIN_MODE, ["Finder"]),
                                     ({"COLABDESIGN_OPT": "turbo"}, STARTUP_CHAIN_MODE, ["Finder"])):
            r = subprocess.run([sys.executable, "-c", "import colabdesign_opt._autoload\n" + STARTUP_PROBE], env=dict(self.env, **env), capture_output=True, text=True, cwd=self.tmp)
            self.assertEqual(r.returncode, 0, r.stderr[-800:])
            self.assertEqual(r.stdout.splitlines()[0], f"LOADED {expect}", env)                    # no colabdesign, no jax, no numpy, no opt_core, no further module of the kit
            self.assertEqual(r.stdout.splitlines()[1], f"TRIG {finders}", env)
            self.assertEqual(r.stdout.splitlines()[2], "HASHLIB False", env)                        # names.py keeps hashlib inside its one function

    def test_env_route_refused_name_exits_3_at_the_trigger(self):
        r = self.run_py("print('before')\nimport colabdesign\nprint('unreachable')", COLABDESIGN_OPT="turbo")
        self.assertEqual(r.returncode, 3); self.assertIn("before", r.stdout); self.assertNotIn("unreachable", r.stdout)
        self.assertEqual(r.returncode, 3); self.assertIn("unknown COLABDESIGN_OPT='turbo'", r.stderr)
        r = self.run_py("import json\nprint('no trigger, no refusal')", COLABDESIGN_OPT="turbo")
        self.assertEqual(r.returncode, 0); self.assertNotIn("NOT ACTIVE", r.stderr)

    def test_env_route_core_missing_exits_3(self):
        """A core on the path that is not the pinned one (a shadowing opt_core with no readable `__version__`) at the trigger: the
        pre-import gate's NOT ACTIVE line and exit 3 (the hook runs `_core_gate.gate` at the trigger, before any opt_core import; nothing
        of that core is imported)."""
        hide = os.path.join(self.tmp, "hide"); os.makedirs(os.path.join(hide, "opt_core"), exist_ok=True)
        open(os.path.join(hide, "opt_core", "__init__.py"), "w").write("raise ImportError('opt_core hidden for the test')\n")
        r = self.run_py("import colabdesign\nprint('unreachable')", COLABDESIGN_OPT="fast", PYTHONPATH=hide + os.pathsep + self.env["PYTHONPATH"])
        self.assertEqual(r.returncode, 3, r.stderr[-600:]); self.assertNotIn("unreachable", r.stdout)
        self.assertIn("[colabdesign-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned ", r.stderr); self.assertIn("installed v? at ", r.stderr); self.assertNotIn("Traceback", r.stderr)   # the pre-import gate (_core_gate.py) names the shadowing core; nothing of it was imported

    def test_pth_route_core_missing_exits_3(self):
        """Through a REAL .pth (site.addsitedir on a directory holding the kit's colabdesign_opt_autoload.pth): startup imports nothing of the
        core (the .pth line succeeds), and at the trigger with no core on the path the process prints the NOT ACTIVE line and exits 3."""
        sitedir = os.path.join(self.tmp, "sitedir"); os.makedirs(sitedir, exist_ok=True)
        shutil.copy(os.path.join(_stubs.OPT_DIR, "colabdesign_opt_autoload.pth"), sitedir)
        hide = os.path.join(self.tmp, "hide2"); os.makedirs(os.path.join(hide, "opt_core"), exist_ok=True)
        open(os.path.join(hide, "opt_core", "__init__.py"), "w").write("raise ImportError('opt_core hidden for the test')\n")
        code = f"import site; site.addsitedir({sitedir!r}); print('started'); import colabdesign; print('unreachable')"
        env = dict(self.env, COLABDESIGN_OPT="fast", PYTHONPATH=hide + os.pathsep + self.env["PYTHONPATH"])
        r = subprocess.run([sys.executable, "-S", "-c", code], env=env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 3, r.stderr[-800:]); self.assertIn("started", r.stdout); self.assertNotIn("unreachable", r.stdout)
        self.assertIn("[colabdesign-opt] NOT ACTIVE: reason=core_mismatch: opt_core pinned ", r.stderr); self.assertNotIn("Traceback", r.stderr)
        self.assertNotIn("Error processing line", r.stderr)                                      # the .pth line itself never fails

    @unittest.skipIf(_stubs.host_nvidia_smi(), "a real nvidia-smi is on PATH: the no-GPU cases need a host without one")
    def test_env_route_without_a_gpu(self):
        r = self.run_py("import colabdesign\nprint('reached')", COLABDESIGN_OPT="fast", PATH=os.environ["PATH"])   # no nvidia-smi stand-in: no GPU — the levers that need the GPU backend step aside BY NAME, fast runs the rest (the kit rule)
        self.assertEqual(r.returncode, 0, r.stderr[-2500:]); self.assertIn("reached", r.stdout); self.assertNotIn("NOT ACTIVE", r.stderr)
        self.assertIn("[colabdesign-opt] ACTIVE mode=fast route=env ", r.stderr); self.assertRegex(r.stderr, r"\[colabdesign-opt\] LEVER name=compilecache state=skipped reason=cannot_run ")   # no GPU product to key the cache by: named
        act = [l for l in r.stderr.splitlines() if l.startswith("[colabdesign-opt] ACTIVE ")][0]; self.assertIn("compilecache", act.split(" skipped=")[1].split(" ")[0].split(","))
        r = self.run_py("import colabdesign\nprint('reached')", COLABDESIGN_OPT="off", PATH=os.environ["PATH"])        # off installs no finder: stock runs, GPU or not
        self.assertEqual(r.returncode, 0, r.stderr[-800:]); self.assertIn("reached", r.stdout); self.assertNotIn("NOT ACTIVE", r.stderr)

    def test_env_route_refusals_exit_3(self):
        script = os.path.join(self.tmp, "stock_design.py"); open(script, "w").write("import colabdesign_opt._autoload\nimport colabdesign\nprint('unreachable')\n")
        r = subprocess.run([sys.executable, script], env=dict(self.env, COLABDESIGN_OPT="fast"), capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 3); self.assertIn("this process is stock_design.py", r.stderr)


# ============================================================ the same hook through a REAL editable install in a fresh venv (pip's .pth sort order)
GATE = "[colabdesign-opt] NOT ACTIVE: reason=core_missing:opt_core (pinned >= v"


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestInstalledPthCoreAbsent(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import venv
        from colabdesign_opt import __version__
        cls.tmp = tempfile.mkdtemp(prefix="cd_opt_pth_")
        cls.venv = os.path.join(cls.tmp, "venv")
        venv.EnvBuilder(with_pip=False, symlinks=True).create(cls.venv)
        cls.py = os.path.join(cls.venv, "bin", "python")
        sp = subprocess.run([cls.py, "-c", "import site; print(site.getsitepackages()[0])"], capture_output=True, text=True, check=True).stdout.strip()
        with open(os.path.join(sp, f"__editable__.colabdesign_opt-{__version__}.pth"), "w") as fh:      # the editable install's path entry (pip's name)
            fh.write(_stubs.OPT_DIR + "\n")
        with open(os.path.join(_stubs.OPT_DIR, "colabdesign_opt_autoload.pth")) as src, open(os.path.join(sp, "colabdesign_opt_autoload.pth"), "w") as dst:
            dst.write(src.read())                                                                       # the kit's autoload .pth, as the wheel ships it
        cls.env = {k: v for k, v in os.environ.items() if not k.startswith(("COLABDESIGN_OPT", "AF2M_", "AF_PALLAS_", "PYTHON"))}
        cls.env.update(PATH=os.path.join(cls.venv, "bin") + os.pathsep + os.environ.get("PATH", ""), COLABDESIGN_OPT="fast")
        r = subprocess.run([cls.py, "-c", "import importlib.util as u; print(u.find_spec('opt_core'))"], env=cls.env, capture_output=True, text=True)
        assert r.stdout.strip() == "None", f"the test venv must see no opt_core: {r.stdout!r} {r.stderr!r}"

    def probe_line(self, path):
        """The import-probe command of configs/h100.env / run.sh: the line that starts with `python -c "import importlib.util`."""
        lines = [l for l in open(path, encoding="utf-8").read().splitlines() if l.startswith('python -c "import importlib.util')]
        self.assertEqual(len(lines), 1, path)
        return lines[0]

    def test_startup_imports_nothing_of_the_core_and_kills_nothing(self):
        r = subprocess.run([self.py, "-c", "import sys, colabdesign_opt; print('ok', 'opt_core' in sys.modules)"], env=self.env, capture_output=True, text=True)
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "ok False"), r.stderr)

    def test_interpreter_start_holds_the_hook_chain_only(self):
        """No explicit import at all: what the installed .pth line left in the interpreter — the package and the hook (+ the import-free names
        when a mode is set), no further module of the kit, nothing of the core / jax / colabdesign / numpy, no hashlib; the Finder armed only under a mode."""
        base = {k: v for k, v in self.env.items() if k != "COLABDESIGN_OPT"}
        for env, expect, finders in ((base, STARTUP_CHAIN, []), (dict(base, COLABDESIGN_OPT="off"), STARTUP_CHAIN, []),
                                     (dict(base, COLABDESIGN_OPT="fast"), STARTUP_CHAIN_MODE, ["Finder"])):
            r = subprocess.run([self.py, "-c", STARTUP_PROBE], env=env, capture_output=True, text=True, cwd=self.tmp)
            self.assertEqual(r.returncode, 0, r.stderr[-600:])
            self.assertEqual(r.stdout.splitlines(), [f"LOADED {expect}", f"TRIG {finders}", "HASHLIB False"], env.get("COLABDESIGN_OPT"))

    def test_config_and_runsh_probes_name_the_missing_core(self):
        for path in (os.path.join(_stubs.TREE, "configs", "h100.env"), os.path.join(_stubs.TREE, "run.sh")):
            probe = self.probe_line(path)
            r = subprocess.run(["bash", "-c", probe], env=dict(self.env, MODEL_OPT=_stubs.TREE, HERE=_stubs.TREE), capture_output=True, text=True, cwd=self.tmp)
            self.assertEqual(r.returncode, 3, (path, r.stderr)); self.assertIn(GATE, r.stderr, path); self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, r.stderr)
            self.assertNotIn("not importable", r.stderr); self.assertNotIn("package_missing", r.stderr); self.assertNotIn("Traceback", r.stderr)
        r = subprocess.run(["bash", "-c", f"source {os.path.join(_stubs.TREE, 'configs', 'h100.env')}; echo after=$?"], env=self.env, capture_output=True, text=True, cwd=self.tmp)
        self.assertIn(GATE, r.stderr); self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, r.stderr); self.assertNotIn("after=0", r.stdout)      # the config stops at its probe
        r = subprocess.run(["bash", os.path.join(_stubs.TREE, "run.sh"), "check", "--config", "h100", "--mode", "fast"], env=self.env, capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn(GATE, r.stderr); self.assertEqual(r.stderr.count("NOT ACTIVE"), 1, r.stderr); self.assertNotIn("Traceback", r.stderr)

    def test_env_route_trigger_names_the_missing_core(self):
        stub = os.path.join(self.tmp, "stub"); os.makedirs(os.path.join(stub, "colabdesign"), exist_ok=True)
        open(os.path.join(stub, "colabdesign", "__init__.py"), "w").write("")
        r = subprocess.run([self.py, "-c", "import colabdesign; print('unreachable')"], env=dict(self.env, PYTHONPATH=stub), capture_output=True, text=True, cwd=self.tmp)
        self.assertEqual(r.returncode, 3, r.stderr); self.assertIn(GATE, r.stderr); self.assertNotIn("unreachable", r.stdout); self.assertNotIn("Traceback", r.stderr)


if __name__ == "__main__":
    unittest.main()
