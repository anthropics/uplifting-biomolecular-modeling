
"""One pipeline, end to end: a mode word resolves to its row (modes.resolve, the row the launcher composes from), levers.install() applies
that row's per-step levers (the ONE installer -- install()/installed()/MODES/env_required -- proven atomic: a bad describe() or a failed
gate leaves nothing installed, a rollback failure is still named), every served lever's real module meets the installer's own describe()
evidence contract (no stand-in, by kit_file, before/after served traffic/after restore), an out-of-memory from the activation lever is
re-raised to the caller rather than folded into the kit's NOT ACTIVE report (`if is_oom(e): raise` is the handler's first statement), and
driver.compose() turns a resolved row into the subprocess argv/environment (forbidden prefixes stripped, the recipe's own variables kept,
pre-set P1 variables dropped on `exact`) -- proven against a live dirty process via stock_design.env_proof and the boltz checkpoint gate.
Unwired tier words are refused by name at every one of these layers, first."""
import contextlib
import dataclasses
import importlib.util
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

import mosaic_opt
import pytest

from . import _fake_lever, _stubs, core_src
from mosaic_opt import ActivationError, det, driver, levers, modes, registry, stack, stock_design
from opt_core.oom import is_oom


FAKE = "mosaic_opt.tests._fake_lever"


@contextlib.contextmanager
def big_wired(module=FAKE):
    """P5 given a module and the `big` word given its row: what the stitch of a memory lever does to the tables."""
    lv = dataclasses.replace(registry.LEVERS["P5"], module=module, wired=True, flag="--levers P5")
    km = dict(modes.KIT_MODES["big"], levers=("P5",), row="U_big", specs={})     # the fake lever at its default setting (the real row's P5 pin is memlevers grammar, not the fake's)
    served = tuple(w for w in modes.WORDS if (km if w == "big" else modes.KIT_MODES[w])["row"] is not None)
    with mock.patch.dict(registry.LEVERS, {"P5": lv}), mock.patch.dict(levers.LEVERS, {"P5": lv}), mock.patch.dict(modes.LEVERS, {"P5": lv}), \
         mock.patch.dict(modes.KIT_MODES, {"big": km}), mock.patch.object(modes, "MODES", served), mock.patch.object(modes, "NOT_WIRED", ()):
        _fake_lever.reset(); levers.reset_for_tests()
        try:
            yield
        finally:
            levers.reset_for_tests(); _fake_lever.reset()


def quiet():
    return mock.patch("sys.stderr", new_callable=io.StringIO)


class TestUnwiredTierWordsAreRefused(unittest.TestCase):

    def test_served_words(self):
        self.assertEqual(modes.WORDS, ("fast", "exact", "big", "off"))
        self.assertEqual(modes.MODES, ("fast", "exact", "big", "off"))        # fast: the tier row T_fast; big: the tier row U_big
        self.assertIs(levers.MODES, modes.MODES)                                # one home
        self.assertEqual(modes.NOT_WIRED, ())                                    # every tier word has a row
        fast = list(modes.install_levers_of("fast"))                             # the fast row's per-step levers in install order (modes.KIT_MODES, one home)
        self.assertEqual(levers.plan("fast"), {"mode": "fast", "label": "fast", "levers": fast, "specs": {k: modes.specs_of("fast").get(k) for k in fast}, "levers_off": []})   # pinned settings from the mode table, None = the module's record

    def test_refused_by_name(self):
        for word in modes.NOT_WIRED:
            with self.assertRaises(levers.LeverError) as cm:
                levers.plan(word)
            self.assertIn(f"unknown mode {word!r}: {word!r} is a tier word of this kit line with no ", str(cm.exception))
            with self.assertRaises(ValueError) as cm2:
                modes.resolve(word)
            self.assertIn("What is not wired", str(cm2.exception))
            with self.assertRaises(levers.LeverError):
                levers.install(word)
            self.assertEqual(levers.installed()["mode"], None); self.assertEqual(levers.installed()["levers"], [])

    def test_exact_installs_no_per_step_lever_and_accounts_for_all(self):
        levers.reset_for_tests()
        with quiet() as err:
            info = levers.install("exact")
        try:
            self.assertEqual(info["levers"], []); self.assertEqual(info["mode"], "exact"); self.assertEqual(info["label"], "exact")
            self.assertEqual(len(info["lines"]), len(registry.INSTALL))            # one LEVER line per per-step lever of the registry, all off
            for lid, line in zip(registry.INSTALL, info["lines"]):
                self.assertIn(f"[mosaic-opt] LEVER name={lid} state=off reason=mode:exact impl=", line); self.assertIn(line, err.getvalue())
            with self.assertRaises(levers.LeverError) as cm:                       # a second install() raises, whatever it requests
                levers.install("exact")
            self.assertIn("lever_already_installed", str(cm.exception))
        finally:
            levers.reset_for_tests()

    def test_parse_ids(self):
        with big_wired():
            self.assertEqual(levers.parse_ids("P5"), [("P5", None)]); self.assertEqual(levers.parse_ids("P5=tri64+pf8"), [("P5", "tri64+pf8")])
            self.assertEqual(levers.parse_ids(""), []); self.assertEqual(levers.parse_ids(None), [])
            for text, word in (("Z9", "lever_unknown"), ("P1", "lever_not_per_step"), ("P2", "lever_not_per_step"), ("P5,P5", "lever_repeated")):
                with self.assertRaises(levers.LeverError) as cm:
                    levers.parse_ids(text)
                self.assertTrue(str(cm.exception).startswith(word), (text, str(cm.exception)))

    def test_off_takes_no_plus(self):
        with big_wired():
            with self.assertRaises(levers.LeverError) as cm:
                levers.plan("off", plus="P5")
            self.assertIn("mode 'off' is stock", str(cm.exception))
            self.assertEqual(levers.plan("off")["levers"], [])


class TestInstallOverAWiredLever(unittest.TestCase):

    def test_install_the_tier_word(self):
        with big_wired(), quiet() as err:
            self.assertEqual(modes.MODES, ("fast", "exact", "big", "off"))
            self.assertEqual(levers.plan("big"), {"mode": "big", "label": "big", "levers": ["P5"], "specs": {"P5": None}, "levers_off": []})
            info = levers.install("big")
            self.assertEqual(info["mode"], "big"); self.assertEqual(info["label"], "big"); self.assertEqual(info["levers"], ["P5"])
            self.assertEqual(set(info), {"mode", "label", "levers", "lines", "effective", "env_required", "specs", "levers_off"})
            self.assertEqual(info["effective"]["P5"], {"impl": "fake_lever@test", "origin": "kit", "chunk": "of_record", "installed": True})
            self.assertEqual(info["env_required"], {})
            self.assertEqual(_fake_lever.CALLS, [("install", None), ("configure", None)])
            on = [l for l in info["lines"] if "name=P5" in l][0]
            self.assertTrue(on.startswith("[mosaic-opt] LEVER name=P5 state=on impl=fake_lever@test origin=kit chunk=of_record installed=True"), on)
            self.assertIn("[mosaic-opt] LEVER name=E1 state=off reason=mode:big", err.getvalue())
            self.assertEqual(levers.installed(), info)
            self.assertEqual(levers.manifest_record()["P5"]["state"], "on")               # registry.probe_key's evidence
            with self.assertRaises(levers.LeverError):
                levers.install("big")
            self.assertEqual(levers.uninstall(), ["P5"]); self.assertEqual(levers.installed()["levers"], [])

    def test_spec_and_plus(self):
        with big_wired(), quiet():
            info = levers.install("big", P5="tri64+pf8")
            self.assertEqual(info["specs"], {"P5": "tri64+pf8"}); self.assertEqual(info["effective"]["P5"]["chunk"], "tri64+pf8")
            self.assertEqual(_fake_lever.CALLS[:2], [("install", None), ("configure", "tri64+pf8")])
        with big_wired(), quiet():
            info = levers.install("exact", plus="P5=pf8")                                 # a development request: exact's (no per-step lever) plus P5 — labelled, never the tier word
            self.assertEqual((info["mode"], info["label"], info["levers"], info["specs"]), ("exact", "exact+P5=pf8", ["P5"], {"P5": "pf8"}))
        with big_wired():
            for kw, word in (({"E1": "x"}, "lever_setting_unused"), ({"nope": "x"}, "lever_setting_unknown")):
                with self.assertRaises(levers.LeverError) as cm:
                    levers.plan("big", **kw)
                self.assertTrue(str(cm.exception).startswith(word), str(cm.exception))

    def test_env_required_is_declared_never_set(self):
        with big_wired(), quiet():
            _fake_lever.ENV_REQUIRED["MOSAIC_TEST_LEVER_FLAG"] = "1"
            self.assertEqual(levers.env_required("big"), {"MOSAIC_TEST_LEVER_FLAG": "1"})
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("MOSAIC_TEST_LEVER_FLAG", None)
                with self.assertRaises(levers.LeverError) as cm:
                    levers.install("big")
                self.assertIn("lever_env_required: export before the interpreter starts: MOSAIC_TEST_LEVER_FLAG='1'", str(cm.exception))
                self.assertEqual(levers.installed()["levers"], []); self.assertNotIn(("install", None), _fake_lever.CALLS)
            with mock.patch.dict(os.environ, {"MOSAIC_TEST_LEVER_FLAG": "1"}):
                info = levers.install("big")
                self.assertEqual(info["env_required"], {"MOSAIC_TEST_LEVER_FLAG": "1"}); self.assertEqual(os.environ["MOSAIC_TEST_LEVER_FLAG"], "1")

    def test_refusals_leave_nothing_installed(self):
        with big_wired(), quiet():
            _fake_lever.BEHAVIOUR["install_raises"] = "tri_chunk 0 is not a chunk size"
            with self.assertRaises(levers.LeverError) as cm:
                levers.install("big")
            self.assertIn("lever_refused: P5: RuntimeError: tri_chunk 0 is not a chunk size", str(cm.exception))
            self.assertEqual(levers.installed()["mode"], None)
        with big_wired(), quiet():
            _fake_lever.BEHAVIOUR["describe"] = {"impl": "x"}                             # no origin: the registry entry's origin word fills it (one home for module path / origin)
            info = levers.install("big")
            self.assertEqual((info["effective"]["P5"]["impl"], info["effective"]["P5"]["origin"]), ("x", registry.LEVERS["P5"].origin))
            levers.uninstall()
        with big_wired(), quiet():
            _fake_lever.BEHAVIOUR["describe"] = {"chunk": "64"}                           # neither impl nor origin: both from the registry (impl = the registry module)
            info = levers.install("big")
            self.assertEqual((info["effective"]["P5"]["impl"], info["effective"]["P5"]["origin"], info["effective"]["P5"]["chunk"]), (FAKE, "kit", "64"))
            levers.uninstall()
        with big_wired(), quiet(), mock.patch.dict(levers.LEVERS, {"P5": dataclasses.replace(levers.LEVERS["P5"], origin="")}):
            _fake_lever.BEHAVIOUR["describe"] = {"impl": "x"}                             # nothing gives origin: refused by name, the install rolled back
            with self.assertRaises(levers.LeverError) as cm:
                levers.install("big")
            self.assertIn("lever_describe_incomplete: P5: neither describe() nor the registry gives origin", str(cm.exception))
            self.assertIn(("uninstall", None), _fake_lever.CALLS)                           # the half-installed lever was removed
            self.assertEqual(levers.installed()["levers"], [])
        with big_wired(module="mosaic_opt.tests._no_such_module"), quiet():
            with self.assertRaises(levers.LeverError) as cm:
                levers.install("big")
            self.assertIn("lever_import_failed: P5 (mosaic_opt.tests._no_such_module)", str(cm.exception))

    def test_invalid_evidence_is_atomic(self):
        """origin outside the LEVER grammar / a blank token: refused by name AFTER install() ran, the lever uninstalled, the process as it was —
        and a following good install succeeds exactly once."""
        for facts, word in (({"impl": "x", "origin": "opt_core"}, "lever_evidence_invalid: P5: origin='opt_core'"),
                            ({"impl": "has blank", "origin": "kit"}, "lever_evidence_invalid: P5: describe() keys and values must be single non-blank tokens"),
                            ({"impl": "", "origin": "kit"}, "lever_evidence_invalid: P5")):
            with big_wired(), quiet():
                _fake_lever.BEHAVIOUR["describe"] = facts
                with self.assertRaises(levers.LeverError) as cm:
                    levers.install("big")
                self.assertTrue(str(cm.exception).startswith(word), (facts, str(cm.exception)))
                self.assertIn(("install", None), _fake_lever.CALLS); self.assertIn(("uninstall", None), _fake_lever.CALLS)
                self.assertIsNone(levers.installed()["mode"]); self.assertEqual(levers.installed()["levers"], [])
                _fake_lever.BEHAVIOUR["describe"] = None
                self.assertEqual(levers.install("big")["levers"], ["P5"])
                with self.assertRaises(levers.LeverError) as cm2:
                    levers.install("big")
                self.assertIn("lever_already_installed", str(cm2.exception))

    def test_rollback_failure_is_named(self):
        with big_wired(), quiet():
            _fake_lever.BEHAVIOUR["describe"] = {"impl": "x", "state": "on"}                # a post-install refusal (a reserved key) whose rollback then fails
            with mock.patch.object(_fake_lever, "uninstall", side_effect=RuntimeError("cannot unpatch")):
                with self.assertRaises(levers.LeverError) as cm:
                    levers.install("big")
            self.assertIn("lever_evidence_invalid: P5", str(cm.exception)); self.assertIn("rollback_incomplete: P5: RuntimeError: cannot unpatch", str(cm.exception))


    def test_install_order_is_the_requests(self):
        """Row order, then the extra levers — installs happen in that order and the `off` accounting lines follow."""
        lv1 = dataclasses.replace(registry.LEVERS["E1"], module=FAKE)
        with big_wired(), quiet(), mock.patch.dict(levers.LEVERS, {"E1": lv1}), mock.patch.dict(registry.LEVERS, {"E1": lv1}), mock.patch.dict(modes.LEVERS, {"E1": lv1}):
            info = levers.install("big", plus="E1")
            self.assertEqual(info["levers"], ["P5", "E1"]); self.assertEqual(info["label"], "big+E1")
            self.assertEqual([l.split()[2] for l in info["lines"]], ["name=P5"] + [f"name={k}" for k in registry.INSTALL if k != "P5"])   # the installed lever first (request order), then one off line per other per-step lever

    def test_reserved_keys_refused(self):
        with big_wired(), quiet():
            _fake_lever.BEHAVIOUR["describe"] = {"impl": "x", "origin": "kit", "state": "on"}
            with self.assertRaises(levers.LeverError) as cm:
                levers.install("big")
            self.assertIn("lever_evidence_invalid: P5: describe() carries the LEVER line's own slots state", str(cm.exception))

    def test_finalize_runs_census_and_gate(self):
        with big_wired(), quiet():
            self.assertEqual(levers.finalize("t"), {})                                       # nothing installed: nothing to gate
            levers.install("big")
            rec = levers.finalize("t")
            self.assertEqual(rec["P5"]["state"], "on"); self.assertIn(("emit_line", "t"), _fake_lever.CALLS); self.assertIn(("gate", None), _fake_lever.CALLS)
            _fake_lever.BEHAVIOUR["gate_raises"] = "fallback_undeclared: below_size_rule:3"
            with self.assertRaises(levers.LeverError) as cm:
                levers.finalize("t")
            self.assertTrue(str(cm.exception).startswith("lever_gate: P5: RuntimeError: fallback_undeclared"), str(cm.exception))
            _fake_lever.BEHAVIOUR["gate_raises"] = None

    def test_no_module(self):
        with big_wired(module=None):
            pass                                                                            # (the registry's own import assert forbids wired+install+no module; plan-time refusal below uses an E1 with no module)
        lv1 = dataclasses.replace(registry.LEVERS["E1"], module=None)
        with big_wired(), quiet(), mock.patch.dict(levers.LEVERS, {"E1": lv1}), mock.patch.dict(registry.LEVERS, {"E1": lv1}), mock.patch.dict(modes.LEVERS, {"E1": lv1}):
            with self.assertRaises(levers.LeverError) as cm:
                levers.install("exact", plus="E1")                                          # a per-step lever with no module
            self.assertIn("lever_no_module: E1", str(cm.exception))


class TestTheDriversFlagUsesTheInstaller(unittest.TestCase):
    """tools/recipe.py install_levers is the driver's `--levers WORD[+ID[=SPEC]...]`: empty = nothing imported; WORD = a served mode word,
    literally `levers.install(WORD, plus=...)` (the step probe's call); `off` refused by name (the stock arm omits the flag); a refusal is a
    RecipeError naming the lever / word."""

    @classmethod
    def setUpClass(cls):
        from .. import recipe as accessor
        cls.M = accessor.module()

    def test_an_unwired_tier_word_is_refused_by_name(self):
        for word in modes.NOT_WIRED:
            with quiet(), self.assertRaises(self.M.RecipeError) as cm:
                self.M.install_levers(word)
            self.assertIn(f"unknown mode {word!r}: {word!r} is a tier word", str(cm.exception))

    def test_empty_imports_nothing(self):
        self.assertIsNone(self.M.install_levers(""))
        self.assertIsNone(self.M.install_levers(None))
        self.assertIsNone(self.M.install_levers("   "))

    def test_word_installs_the_words_levers(self):
        with big_wired(), quiet():
            info = self.M.install_levers("big")
            self.assertEqual((info["mode"], info["label"], info["levers"]), ("big", "big", ["P5"]))
            self.assertEqual(self.M.levers_record()["P5"]["state"], "on"); self.assertIsNone(self.M.levers_record()["P5"]["spec"])
            levers.uninstall()
        with big_wired(), quiet():
            info = self.M.install_levers("big+P5=tri64+pf8")                             # the word's own lever with a setting: the first `+` separates the word, the spec keeps its `+`
            self.assertEqual((info["label"], info["specs"]), ("big+P5=tri64+pf8", {"P5": "tri64+pf8"})); self.assertEqual(self.M.levers_record()["P5"]["spec"], "tri64+pf8")   # a caller-set spec is spelled in the label
            levers.uninstall()
        with big_wired(), quiet():
            info = self.M.install_levers("exact+P5")                                        # a development request: exact's set (none) plus P5 — labelled, never the tier word
            self.assertEqual((info["mode"], info["label"], info["levers"]), ("exact", "exact+P5", ["P5"]))
            levers.uninstall()
        with big_wired(), quiet():
            info = self.M.install_levers("exact")                                           # exact composes no per-step lever: the accounting lines only
            self.assertEqual((info["mode"], info["levers"]), ("exact", []))

    def test_refusals_are_named(self):
        with big_wired(), quiet():
            for text, word in (("off", "lever_word_off"), ("exact+P1", "lever_not_per_step"), ("exact+Z9", "lever_unknown"),
                               ("P5", "unknown mode 'p5'")):
                with self.assertRaises(self.M.RecipeError) as cm:
                    self.M.install_levers(text)
                self.assertTrue(str(cm.exception).startswith(word), (text, str(cm.exception)))
                levers.reset_for_tests()


class TestKitModuleDoor(unittest.TestCase):
    """recipe.kit_module(stem): ONE module object per process for a kit file, whichever caller asks; the helpers' kit_modules() goes through it."""

    @classmethod
    def setUpClass(cls):
        from .. import recipe as accessor
        cls.M = accessor.module()

    def test_one_object(self):
        m1, r1 = self.M.kit_module("numstate"); m2, r2 = self.M.kit_module("numstate")
        self.assertIs(m1, m2); self.assertEqual(r1, r2); self.assertIn(r1["source"], ("mosaic.fast", "kit_src"))
        try:
            import jax  # noqa: F401  (fastload imports jax at module level: the helpers' pair loads only where jax is installed)
            fl, ns, rec = self.M.kit_modules()
            self.assertIs(ns, m1)                                                               # the helpers' door is the same door
        except ImportError:
            pass
        with self.assertRaisesRegex(self.M.RecipeError, "kit_module_unknown"):
            self.M.kit_module("no_such_kit_file")

    def test_levers_resolve_kit_modules_through_the_door(self):
        lv = dataclasses.replace(registry.LEVERS["P5"], module="mosaic.fast.numstate", wired=True, flag="--levers big")   # a kit stem standing in as P5's module
        with mock.patch.dict(levers.LEVERS, {"P5": lv}):
            mod, rec = self.M.kit_module("numstate")
            if rec["source"] == "mosaic.fast":                                                 # the installed package carries the kit's files (the kit venv): the SAME object
                self.assertIs(levers._module("P5"), mod)
            else:                                                                               # a tools/ or source-tree copy is never a per-step lever's module: refused by name
                with self.assertRaises(levers.LeverError) as cm:
                    levers._module("P5")
                self.assertTrue(str(cm.exception).startswith("lever_module_not_installed: P5: mosaic.fast.numstate is not carried by the installed mosaic"), str(cm.exception))


class TestPinnedSettingsAndLabels(unittest.TestCase):
    """A mode pins settings for its own levers explicitly (modes.KIT_MODES[mode]["specs"], read by plan(); None = the module's setting of
    record); whatever the CALLER adds or changes is spelled in the label, so `fast` with one lever re-set is `fast+<ID>=<SPEC>`, never `fast`."""

    def test_pinned_specs_reach_the_plan(self):
        with mock.patch.dict(modes.KIT_MODES["fast"], {"specs": {"K1": "pinned-word"}}):
            p = levers.plan("fast")
            self.assertEqual((p["label"], p["specs"]["K1"]), ("fast", "pinned-word"))
            self.assertTrue(all(v is None for k, v in p["specs"].items() if k != "K1"))
            q = levers.plan("fast", K1="other")                                        # the caller changes a pinned setting: labelled
            self.assertEqual((q["label"], q["specs"]["K1"]), ("fast+K1=other", "other"))
            self.assertEqual(levers.plan("fast", K1="pinned-word")["label"], "fast")   # asking for the pinned word itself changes nothing

    def test_caller_settings_are_in_the_label(self):
        own = list(modes.install_levers_of("fast"))
        p = levers.plan("fast", **{own[0]: "x"})
        self.assertEqual(p["label"], f"fast+{own[0]}=x")
        q = levers.plan("exact", plus=",".join(own))                                   # composition by plus list on the exact word: order = the list
        self.assertEqual((q["levers"], q["label"]), (own, "exact" + "".join("+" + l for l in own)))
        r = levers.plan("exact", plus=f"{own[0]}=y")
        self.assertEqual(r["label"], f"exact+{own[0]}=y")



def _load(lid):
    """The lever's kit-home module imported by path; the array-stack stand-ins exist ONLY around this import (a module that consults the JAX
    backend later — a kernel lever's probe — must see the box as it is, never a stand-in)."""
    path = os.path.join(_stubs.KIT, registry.LEVERS[lid].kit_file)
    spec = importlib.util.spec_from_file_location(f"lever_{lid}_describe_contract", path)
    mod = importlib.util.module_from_spec(spec)
    with mock.patch.dict(sys.modules, _stubs._standins()):
        spec.loader.exec_module(mod)
    return mod


def _probe_reset():
    """Forget the shared core's cached backend probe (opt_core.kernels.pallas_attn_serve) so this test neither inherits another test's answer
    nor leaves its own behind; returns the server module or None when the core does not carry it."""
    try:
        from opt_core.kernels import pallas_attn_serve as F1
    except ImportError:
        return None
    F1._PROBE = None
    return F1


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestEveryServedLeversDescribeMeetsTheInstaller(unittest.TestCase):

    def setUp(self):
        self.F1 = _probe_reset()

    def tearDown(self):
        _probe_reset()

    def _no_backend_named(self, message):
        """True when the refusal carries the core probe's own structured no-usable-backend outcome on this box (probe ok False; its `kind` word —
        jax_missing, jax_older_than_floor, … — is the word in the message): the tolerance is keyed on the probe's answer, never on a guessed literal."""
        if self.F1 is None:
            return False
        p = self.F1.probe(refresh=True, require_gpu=False)
        return (not p.get("ok")) and bool(p.get("kind")) and p["kind"] in message

    def _checked(self, lid, mod, when):
        """describe() through the installer's own evidence check (raises LeverError by name) + one token per value; returns the facts."""
        try:
            facts = levers._describe(lid, mod)
        except levers.LeverError as e:
            if _stubs.NO_JAX and when == "after served traffic" and self._no_backend_named(str(e)):   # a kernel lever names its kernel's origin from the core's backend probe: a box
                return None                                                                 # without jax cannot hold its served state at all (configure refuses there by the same
            raise AssertionError(f"{lid} {when}: {e}") from None                            # word) — tolerated BY THAT WORD only; the kit image's CPU suite enforces this state
        self.assertTrue(facts["impl"], (lid, when)); self.assertIn(facts["origin"], ("kit", "core"), (lid, when))
        for k, v in facts.items():
            self.assertEqual(len(str(v).split()), 1, (lid, when, k, v))
        return facts

    def test_describe_of_every_lever_of_every_served_mode(self):
        seen = set()
        for word in modes.MODES:
            for lid in modes.install_levers_of(word):
                if lid in seen:
                    continue
                seen.add(lid)
                with self.subTest(mode=word, lever=lid):
                    mod = _load(lid)
                    self._checked(lid, mod, "before install")                                  # describe() passes the installer's scalar-token rule
        self.assertEqual(seen, {l for w in modes.MODES for l in modes.install_levers_of(w)})
        self.assertTrue(seen >= {"E1", "P6", "K1", "P5"}, seen)


class OutOfMemoryError(RuntimeError):
    """Stand-in for torch.cuda.OutOfMemoryError (torch.OutOfMemoryError): the class name the classifier keys on without importing torch."""


class XlaRuntimeError(RuntimeError):
    """Stand-in for jaxlib's XlaRuntimeError: an out-of-memory on this engine's stack is this class with a RESOURCE_EXHAUSTED status."""


TORCH_OOM = OutOfMemoryError("CUDA out of memory (mock)")


JAX_OOM = XlaRuntimeError("RESOURCE_EXHAUSTED: Out of memory while trying to allocate 68719476736 bytes. (mock)")


HOST_OOM = MemoryError("host out of memory (mock)")


NOT_OOM = (RuntimeError("pcc: the autotune file is not a serialized AutotuneResults (mock)"), XlaRuntimeError("INTERNAL: cuDNN launch failure (mock)"),
           OSError(28, "No space left on device (mock)"))


class RaisingLever(_stubs.FakeReproCache):
    """The tests' P1 lever stand-in (`stack.p1_lever()`), whose enable() raises the given exception instead of applying anything."""
    def __init__(self, exc):
        super().__init__()
        self.exc = exc

    def enable(self, cache_dir, autotune="auto", autotune_file=None):
        self.calls.append((cache_dir, autotune))
        raise self.exc


def test_the_classifier_is_the_cores_and_knows_this_engines_oom():
    assert is_oom.__module__ == "opt_core.oom" and stack.is_oom is is_oom                     # one spelling: the kit imports the core's symbol, no kit-local classifier
    assert is_oom(TORCH_OOM) and is_oom(JAX_OOM) and is_oom(HOST_OOM)
    wrapped = RuntimeError("lever failed")
    wrapped.__cause__ = JAX_OOM
    assert is_oom(wrapped)                                                                       # one level down: a wrapper that re-raised the OOM as something else
    assert not any(is_oom(e) for e in NOT_OOM) and not is_oom(RuntimeError("wrapped"))


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestOomPropagatesThroughEnable(unittest.TestCase):
    """The served entry a program calls, `mosaic_opt.enable("exact")`, past every gate (stubbed: pins, stack, an H100), up to the lever."""

    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.tmp = tempfile.mkdtemp(prefix="mosaic_opt_oom_")
        for k in ("XLA_FLAGS", "JAX_COMPILATION_CACHE_DIR", "JAX_ENABLE_X64", "JAX_DEFAULT_MATMUL_PRECISION", "MOSAIC_OPT", "MOSAIC_OPT_CACHE_DIR",
                  "MOSAIC_OPT_CACHE_ROOT", "MOSAIC_OPT_FORCE", "MOSAIC_CACHE_DIR", "MODEL_OPT_TARGET_GPU", "MOSAIC_OPT_ALLOW_PARTIAL"):
            self.mp.delenv(k, raising=False)
        self.mp.setenv("MOSAIC_OPT_HOME", _stubs.TREE)

    def tearDown(self):
        self.mp.undo(); self.state.restore()

    def _lever(self, exc):
        """A fresh package state (no report), the gates stubbed past (pins, stack, an H100), and the raising lever in place."""
        self.state.restore()
        _stubs.stub_gates(self.mp)
        self.mp.setenv("MOSAIC_OPT_CACHE_DIR", os.path.join(self.tmp, "p1"))               # the P1 directory of this process (after the restore: the state snapshot holds the environment too)
        lever = RaisingLever(exc)
        self.mp.setattr(stack, "p1_lever", lambda: lever)
        return lever

    def test_oom_in_the_lever_is_raised_to_the_caller_not_reported_inactive(self):
        for exc in (TORCH_OOM, JAX_OOM, HOST_OOM):
            with self.subTest(exc=type(exc).__name__):
                lever = self._lever(exc)
                with pytest.raises(type(exc)) as info:
                    mosaic_opt.enable("exact")
                self.assertIs(info.value, exc); self.assertEqual(len(lever.calls), 1)      # the lever was reached (every gate passed) and its OOM came out unchanged
                self.assertFalse(mosaic_opt.status().get("active"))                          # nothing recorded an activation
                self._lever(exc)
                with pytest.raises(type(exc)):
                    stack.activate("exact", strict=True)                                     # the hook's strict form: the OOM itself, not ActivationError

    def test_any_other_lever_failure_is_still_the_named_not_active_report(self):
        for exc in NOT_OOM:
            with self.subTest(exc=repr(exc)[:40]):
                lever = self._lever(exc)
                rep = mosaic_opt.enable("exact")                                             # behaviour unchanged: an inactive report by name, the program continues on stock
                self.assertFalse(rep["active"]); self.assertEqual(len(lever.calls), 1)
                self.assertTrue(rep["reason"].startswith("pcc.enable("), rep["reason"]); self.assertIn(type(exc).__name__, rep["reason"])
                self._lever(exc)
                with self.assertRaises(ActivationError):
                    stack.activate("exact", strict=True)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestStripAndCompose(unittest.TestCase):

    def setUp(self):
        self.state = _stubs.StubState(); self.state.restore()
        self.mp = pytest.MonkeyPatch()
        self.mp.setenv("MOSAIC_OPT_HOME", _stubs.TREE)
        self.pins = stack.pins()
        self.dirty = {"PATH": "/usr/bin", "HOME": "/h", "MOSAIC_CACHE_DIR": "/w", "MOSAIC_OPT": "exact", "MOSAIC_OPT_CACHE_ROOT": "/jc", "MOSAIC_OPT_CACHE_DIR": "/p1",
                      "MOSAIC_OPT_FORCE": "1", "JAX_COMPILATION_CACHE_DIR": "/old", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0", "XLA_FLAGS": "--xla_gpu_load_autotune_results_from=/old/x.pb",
                      "PYTHONUNBUFFERED": "0"}

    def tearDown(self):
        self.mp.undo(); self.state.restore()

    def test_stock_environment(self):
        se = self.pins["stock_environment"]
        self.assertEqual(se["must_be_absent_prefixes"], ["JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_", "XLA_FLAGS", "MOSAIC_OPT"])
        self.assertEqual(se["allowed_exceptions"], ["MOSAIC_CACHE_DIR"])                      # the one exception: the library's own weights cache
        for k in stack.PATH_ENV + stack.PACKAGE_ENV:                                        # the package's own variables all carry the prefix
            self.assertTrue(k.startswith("MOSAIC_OPT")); self.assertNotIn(k, se["allowed_exceptions"])
        self.assertEqual(se["reads"], ["MOSAIC_CACHE_DIR"])
        self.assertEqual(tuple(se["must_be_unset_every_arm"]), det.MUST_BE_UNSET)

    def test_strip_env(self):
        env, dropped = driver.strip_env(self.dirty, stack.stock_env_absent(self.pins), stack.stock_env_exceptions(self.pins))
        self.assertEqual(sorted(dropped), ["JAX_COMPILATION_CACHE_DIR", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "MOSAIC_OPT", "MOSAIC_OPT_CACHE_DIR",
                                           "MOSAIC_OPT_CACHE_ROOT", "MOSAIC_OPT_FORCE", "XLA_FLAGS"])
        self.assertEqual(env["MOSAIC_CACHE_DIR"], "/w"); self.assertNotIn("MOSAIC_OPT_CACHE_ROOT", env); self.assertEqual(env["PATH"], "/usr/bin")

    def test_compose_off(self):
        res = modes.resolve("off")
        argv, env, notes = driver.compose(res, seed=3, out_dir="/o", tag="t", shape_flags=["--binder-length", "80", "--target-copies", "1"], settings_flags=["--steps1", "75", "--steps2", "50"],
                                          det_level=1, base_env=self.dirty)
        for k in ("JAX_COMPILATION_CACHE_DIR", "XLA_FLAGS", "MOSAIC_OPT", "MOSAIC_OPT_CACHE_DIR", "MOSAIC_OPT_FORCE", "MOSAIC_OPT_CACHE_ROOT"):
            self.assertNotIn(k, env)
        self.assertEqual(env["MOSAIC_CACHE_DIR"], "/w")
        self.assertNotIn("XLA_PYTHON_CLIENT_PREALLOCATE", env)                              # the recipe sets no allocator variable at any level
        self.assertEqual(env["PYTHONUNBUFFERED"], "0")                                      # a set value stands
        self.assertEqual(argv[1:], [os.path.join(stack.kit_home(), "tools", "public_design_run.py"), "--tag", "t", "--out", "/o", "--seed", "3",
                                    "--binder-length", "80", "--target-copies", "1", "--steps1", "75", "--steps2", "50"])
        self.assertEqual(notes["cwd"], os.path.join(stack.kit_home(), "tools"))
        argv0, env0, _ = driver.compose(res, seed=3, out_dir="/o", tag="t", shape_flags=[], settings_flags=[], det_level=0, base_env={"PATH": "/usr/bin"})
        self.assertNotIn("XLA_PYTHON_CLIENT_PREALLOCATE", env0); self.assertNotIn("PYTHONUNBUFFERED", env0)   # --det 0: nothing exported

    def test_compose_off_on_frozen_features_is_the_D_row(self):
        res = modes.resolve("off", features="/f.npz", features_sha="abc")
        self.assertEqual((res.row, res.levers, res.env), ("D_stock2", (), {}))
        argv, env, notes = driver.compose(res, seed=1, out_dir="/o", tag="t", shape_flags=[], settings_flags=[], det_level=1, base_env=self.dirty)
        self.assertEqual(argv[-4:], ["--features-in", "/f.npz", "--features-sha", "abc"])
        self.assertNotIn("--weights", argv)                                                   # the row has no --weights: the driver's default (torch)
        self.assertNotIn("JAX_COMPILATION_CACHE_DIR", env)

    def test_compose_exact_exports_the_row_and_drops_pre_set_p1_variables(self):
        res = modes.resolve("exact", cache_dir="/c1", features="/f.npz", features_sha="abc")
        argv, env, notes = driver.compose(res, seed=0, out_dir="/o", tag="t", shape_flags=[], settings_flags=[], det_level=1, base_env=self.dirty)
        self.assertEqual(env["JAX_COMPILATION_CACHE_DIR"], "/c1")
        self.assertEqual(env["XLA_FLAGS"], "--xla_gpu_load_autotune_results_from=/c1/xla_autotune_results.pb")
        self.assertEqual(env["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS"], "0"); self.assertEqual(env["JAX_PERSISTENT_CACHE_MIN_ENTRY_SIZE_BYTES"], "0")
        self.assertIn("XLA_FLAGS", notes["dropped"]); self.assertIn("JAX_COMPILATION_CACHE_DIR", notes["dropped"])
        self.assertNotIn("MOSAIC_OPT", env)
        self.assertEqual(argv[-6:], ["--weights", "fastinit", "--features-in", "/f.npz", "--features-sha", "abc"])
        self.assertEqual(notes["row_env"], res.env)


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestEnvProof(unittest.TestCase):

    def setUp(self):
        self.pins = json.load(open(os.path.join(_stubs.TREE, "stock", "PINS.json"), encoding="utf-8"))

    def test_clean(self):
        p = stock_design.env_proof(self.pins, environ={"PATH": "/x", "MOSAIC_CACHE_DIR": "/w"}, modules={"os": None, "sys": None}, path=["/usr/lib"])
        self.assertTrue(p["clean"]); self.assertEqual(p["env_present_forbidden"], []); self.assertEqual(p["kit_modules_loaded_before_call"], [])
        self.assertEqual(p["values"]["MOSAIC_CACHE_DIR"], "/w")
        self.assertIn("tools/public_design_run.py (the driver)", p["kit_code_on_this_path"]); self.assertEqual(len(p["kit_code_on_this_path"]), 3)
        self.assertIsNone(p["mosaic_fast_installed"]); self.assertIsNone(p["stock_install"])   # no kit directory given: layout not checked
        p = stock_design.env_proof(self.pins, environ={"MOSAIC_OPT_CACHE_ROOT": "/jc"}, modules={}, path=[])
        self.assertFalse(p["clean"]); self.assertEqual(p["env_present_forbidden"], ["MOSAIC_OPT_CACHE_ROOT"])   # a package path variable is not an exception

    def test_install_layout_seen_by_the_proof(self):
        """The proof finds the installed `mosaic` package without importing it and compares an installed `mosaic/fast/` with the kit's own
        lever files: steps 1-2 (no sub-package) is the stock install; steps 1-3 with every kit file present is recorded as not of
        record and stays clean; a copy missing a kit file is refused."""
        import shutil
        src = os.path.join(_stubs.KIT, "mosaic_fast")
        tmp = tempfile.mkdtemp(prefix="mosaic_opt_layout_")
        site = os.path.join(tmp, "site"); pkg = os.path.join(site, "mosaic"); os.makedirs(pkg); open(os.path.join(pkg, "__init__.py"), "w").close()
        fc = stack.installed_fast_check(_stubs.KIT, mosaic_dir=pkg)
        self.assertEqual((fc["installed"], fc["clean"], fc["path"]), (False, True, None))
        self.assertEqual(stack.installed_fast_label(fc), "not-installed")
        shutil.copytree(src, os.path.join(pkg, "fast"))                                    # what run.sh install places
        fc = stack.installed_fast_check(_stubs.KIT, mosaic_dir=pkg)
        self.assertTrue(fc["installed"] and fc["clean"]); self.assertEqual(set(fc["files"].values()), {"present"}); self.assertEqual(fc["extra"], [])
        self.assertEqual(stack.installed_fast_label(fc), f"kit-bytes@{os.path.join(pkg, 'fast')}")
        self.assertTrue(stock_design.install_layout(fc).startswith("the pinned stack with the kit's lever files"))
        open(os.path.join(pkg, "fast", "stray.py"), "w").close()                             # an extra file the kit does not ship: named (a note), not a refusal — nothing of the kit imports it
        fc = stack.installed_fast_check(_stubs.KIT, mosaic_dir=pkg)
        self.assertTrue(fc["clean"]); self.assertEqual(fc["extra"], ["stray.py"]); self.assertEqual(stack.installed_fast_label(fc), f"kit-bytes+1extra@{os.path.join(pkg, 'fast')}")
        os.remove(os.path.join(pkg, "fast", "stray.py"))
        os.remove(os.path.join(pkg, "fast", "fastload.py"))
        fc = stack.installed_fast_check(_stubs.KIT, mosaic_dir=pkg)
        self.assertFalse(fc["clean"]); self.assertEqual(fc["files"]["fastload.py"], "absent")
        self.assertIn("ABSENT(fastload.py)", stack.installed_fast_label(fc)); self.assertIn("not the kit's lever files", stack.installed_fast_refusal(fc))
        # found without importing: the locator reads sys.modules first, else the path-based finder (never the meta path)
        self.assertEqual(stack.installed_mosaic_dir(modules={"mosaic": type("M", (), {"__path__": [pkg]})()}), pkg)
        mp = pytest.MonkeyPatch(); mp.syspath_prepend(site)
        try:
            self.assertEqual(stack.installed_mosaic_dir(modules={}), pkg)
        finally:
            mp.undo()
        # the proof carries the layout
        real = stock_design.env_proof(self.pins, environ={}, modules={}, path=[], kit_dir=_stubs.KIT)
        self.assertIn("install_layout", real); self.assertIsNotNone(real["mosaic_fast_installed"])
        self.assertEqual(real["stock_install"], not real["mosaic_fast_installed"]["installed"])
        self.assertIn("mosaic.fast=", stock_design.env_line(real)); self.assertIn("stock_install=", stock_design.env_line(real))

    def test_dirty_named(self):
        p = stock_design.env_proof(self.pins, environ={"XLA_FLAGS": "x", "MOSAIC_OPT": "exact", "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS": "0"}, modules={}, path=[])
        self.assertFalse(p["clean"]); self.assertEqual(p["env_present_forbidden"], ["JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS", "MOSAIC_OPT", "XLA_FLAGS"])
        p = stock_design.env_proof(self.pins, environ={"JAX_ENABLE_X64": "1"}, modules={}, path=[])
        self.assertFalse(p["clean"]); self.assertEqual(p["env_precision_set"], ["JAX_ENABLE_X64"])
        p = stock_design.env_proof(self.pins, environ={}, modules={"jax": None, "mosaic.fast.repro_cache": None}, path=[])
        self.assertFalse(p["clean"]); self.assertEqual(p["kit_modules_loaded_before_call"], ["jax", "mosaic.fast.repro_cache"])
        p = stock_design.env_proof(self.pins, environ={}, modules={}, path=["/site/mosaic/fast"])
        self.assertFalse(p["clean"]); self.assertEqual(p["mosaic_fast_on_sys_path"], ["/site/mosaic/fast"])

    def test_stock_design_refuses_a_dirty_process(self):
        tmp = tempfile.mkdtemp(prefix="mosaic_opt_proof_")
        env = {**os.environ, "PYTHONPATH": _stubs.OPT_DIR + os.pathsep + core_src(), "XLA_FLAGS": "--xla_gpu_load_autotune_results_from=/x"}
        out = subprocess.run([sys.executable, "-s", "-m", "mosaic_opt.stock_design", "--driver", os.path.join(_stubs.KIT, "tools", "public_design_run.py"),
                              "--pins", os.path.join(_stubs.TREE, "stock", "PINS.json"), "--", "--tag", "t", "--out", tmp], capture_output=True, text=True, env=env)
        self.assertEqual(out.returncode, 3, out.stderr)
        self.assertIn("ENV-CLEAN FAIL", out.stderr); self.assertIn("present_forbidden=XLA_FLAGS", out.stderr); self.assertIn("REFUSED: the stock environment is not clean", out.stderr)
        self.assertEqual(os.listdir(tmp), [])                                                  # nothing written: the proof is the printed line


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestDataPathGate(unittest.TestCase):
    """boltz has no offline switch: a missing checkpoint refuses rather than proceeding to boltz's own unguarded download
    (src/mosaic/losses/boltz2.py:51). Shape-only -- no download exercised."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="mosaic_opt_datapath_")

    def test_checkpoint_present_passes(self):
        os.makedirs(os.path.join(self.tmp, "boltz"))
        open(os.path.join(self.tmp, "boltz", "boltz2_conf.ckpt"), "wb").write(b"x")
        why, notes = stack.data_path_gate({"MOSAIC_CACHE_DIR": self.tmp})
        self.assertIsNone(why); self.assertEqual(notes, [])

    def test_checkpoint_absent_refuses(self):
        os.makedirs(os.path.join(self.tmp, "boltz"))                                          # dir exists, checkpoint does not
        why, notes = stack.data_path_gate({"MOSAIC_CACHE_DIR": self.tmp})
        self.assertIsNotNone(why)
        self.assertIn("boltz2_conf.ckpt", why); self.assertIn("absent", why)

    def test_missing_cache_dir_refuses(self):
        why, notes = stack.data_path_gate({"MOSAIC_CACHE_DIR": os.path.join(self.tmp, "does_not_exist")})
        self.assertIsNotNone(why); self.assertIn("does not exist", why)

    def test_unset_refuses(self):
        """An unset MOSAIC_CACHE_DIR resolves to boltz's own default cache -- exactly where its unguarded download lands -- so this
        refuses too, the same as a missing checkpoint under a set directory."""
        why, notes = stack.data_path_gate({})
        self.assertIsNotNone(why); self.assertIn("unset", why); self.assertEqual(notes, [])


@unittest.skipUnless(_stubs.tree_present(), "release tree not present around the package")
class TestModesTable(unittest.TestCase):

    def test_resolver_composes_from_the_row(self):
        parsed = modes.parse_row(modes.ROWS["C_p1warm_p2"]["text"])
        res = modes.resolve("exact", cache_dir="/c1", features="/f.npz", features_sha="deadbeef")
        self.assertEqual(res.row, "C_p1warm_p2"); self.assertEqual(res.levers, ("P1", "P2", "P3"))
        self.assertEqual(set(res.env), set(parsed["env"]))
        for k, v in parsed["env"].items():
            self.assertEqual(res.env[k], v.replace("$C1", "/c1"))
        self.assertEqual(res.env["XLA_FLAGS"], "--xla_gpu_load_autotune_results_from=/c1/xla_autotune_results.pb")
        self.assertEqual(res.flags, ["--weights", "fastinit", "--features-in", "/f.npz", "--features-sha", "deadbeef"])
        pop = modes.resolve("exact", cache_dir="/c1", features="/f.npz", features_sha="deadbeef", phase="populate")
        self.assertEqual(pop.row, "B_p1populate_p2")
        self.assertEqual(pop.env["XLA_FLAGS"], "--xla_gpu_dump_autotune_results_to=/c1/xla_autotune_results.pb")
        self.assertEqual({k: v for k, v in pop.env.items() if k != "XLA_FLAGS"}, {k: v for k, v in res.env.items() if k != "XLA_FLAGS"})
        self.assertEqual(pop.flags, res.flags)

    def test_exact_needs_every_placeholder(self):
        with self.assertRaises(ValueError):
            modes.resolve("exact")
        with self.assertRaises(ValueError):
            modes.resolve("exact", cache_dir="/c1")

    def test_off_resolves_to_the_stock_row(self):
        res = modes.resolve("off")
        self.assertEqual((res.row, res.levers, res.env, res.flags), ("A_stock1", (), {}, []))
        from opt_core.jax_design import pcc
        for word, tag in (("big", "U_big"), ("fast", "T_fast")):                         # the tier rows: P1 transparent (the cache alone: pcc autotune=off), P2, the ONE lever flag
            with self.assertRaises(ValueError):
                modes.resolve(word)                                                         # the row names $C1: a directory or the aside reason, never a literal
            res = modes.resolve(word, cache_dir=f"/c/L80_c1/xla_cache_{word}")
            self.assertEqual((res.row, res.levers, res.flags), (tag, modes.KIT_MODES[word]["levers"], ["--weights", "fastinit", "--levers", word]))
            self.assertEqual(res.env, pcc.env(f"/c/L80_c1/xla_cache_{word}", autotune="off", environ={}))
            self.assertEqual((res.levers_off, res.aside), ((), {}))
            res = modes.resolve(word, p1_aside="MOSAIC_OPT_CACHE_ROOT unset")               # no cache root: P1 steps aside by name, the row runs without it
            self.assertEqual((res.row, res.levers, res.env, res.flags), (tag, tuple(l for l in modes.KIT_MODES[word]["levers"] if l != "P1"), {}, ["--weights", "fastinit", "--levers", word]))
            self.assertEqual(res.aside, {"P1": "MOSAIC_OPT_CACHE_ROOT unset"}); self.assertTrue(any("P1 steps aside" in n for n in res.notes))
        self.assertEqual(modes.p1_form("fast"), "transparent"); self.assertEqual(modes.p1_form("exact"), "pinned"); self.assertIsNone(modes.p1_form("off"))
        res = modes.resolve("exact", features=None, aside={"P1": "shape not warm", "P3": "shape not warm"})   # a cold shape: P1 and P3 step aside by name, P2 stays, nothing warm runs in the command
        self.assertEqual((res.levers, res.env, res.flags, sorted(res.aside)), (("P2",), {}, ["--weights", "fastinit"], ["P1", "P3"])); self.assertTrue(modes.describe_line(res).endswith("-aside[P1,P3]"))
        # on frozen features: the stock arm D (`:103`), its flags from the row (no --weights: the driver's default, torch)
        res = modes.resolve("off", features="/f.npz", features_sha="deadbeef")
        self.assertEqual((res.row, res.levers, res.env), ("D_stock2", (), {}))
        self.assertEqual(res.flags, ["--features-in", "/f.npz", "--features-sha", "deadbeef"])
        self.assertEqual(modes.KIT_MODES["off"]["features_row"], "D_stock2")
        with self.assertRaises(ValueError):                                                # D needs the sha placeholder too
            modes.resolve("off", features="/f.npz")

    def test_default_mode_literal(self):
        """The default mode is the literal `exact` (no `fast` ships for this engine; the package default is `fast` wherever one ships);
        the environment switch unset means off — no finder is installed and nothing attaches (_autoload.install)."""
        from mosaic_opt import _autoload, cli
        self.assertEqual(modes.DEFAULT_MODE, "fast"); self.assertIn("big", modes.MODES)
        self.assertIsNone(_autoload.install({})); self.assertIsNone(_autoload.install({"MOSAIC_OPT": "off"}))
        import pytest
        mp = pytest.MonkeyPatch(); mp.delenv("MOSAIC_OPT", raising=False)
        try:
            self.assertEqual(cli._mode_arg(None), modes.DEFAULT_MODE)                        # the CLI without --mode: the package default
            mp.setenv("MOSAIC_OPT", "off"); self.assertEqual(cli._mode_arg(None), "off")     # the environment switch, when set, decides
            self.assertEqual(cli._mode_arg("exact"), "exact")                                 # --mode over the environment
        finally:
            mp.undo()

    def test_the_row_does_not_carry_the_seed(self):
        for tag in modes.ROWS:
            self.assertIn("--seed 0", modes.ROWS[tag]["text"])
        res = modes.resolve("exact", cache_dir="/c", features="/f", features_sha="s")
        self.assertNotIn("--seed", res.flags)

    def test_jit_cache_key(self):
        self.assertEqual(modes.jit_cache_key("0.10.2", "0.10.2", "0.10.2", "NVIDIA H100 80GB HBM3"), "jax0.10.2-jaxlib0.10.2-cuda12plugin0.10.2-nvidia-h100-80gb-hbm3")
        self.assertEqual(modes.gpu_slug(None), "unknown-gpu")


if __name__ == "__main__":
    unittest.main()

