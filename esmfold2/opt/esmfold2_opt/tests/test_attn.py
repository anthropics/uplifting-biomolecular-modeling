"""The fast-environment reader (esmfold2_opt/attn.py) on a CPU box with none of flash-attn / transformer-engine / xformers: the words from fake
upstream modules (FLASH_ATTN_AVAILABLE, the ESMC switches) and fake models (atom-attention modules with the stock forward, the fast line's U1
forward, the banded forward, an unknown one; TE-like and pure-PyTorch LM classes); the SETTINGS / ACTIVE / DRY-RUN / APPLIED words; the
ESMFOLD2_OPT_REQUIRE_FAST_ENV refusal sentence (run-time and dry-run forms); the stack gate; stock/check_pins.py's image-pins assertion with a stubbed
metadata query; the stock/PINS.json "image" block (the pinned stack's accelerated layer: packages and wheels, no platform identifiers)."""
import contextlib
import importlib.util
import json
import os
import re
import sys
import types
import unittest
from unittest import mock

from esmfold2_opt import attn, report, stack, stock_fold

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))                     # esmfold2/
PINS = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
FAST = {"atom_attn": "flash_attn", "esmc_mlp": "te", "esmc_attn": "sdpa(chain_mask)", "esmc_rope": "flash_attn_triton",
        "versions": {"flash_attn": "2.8.3.post1", "transformer_engine": "2.6.0", "xformers": "0.0.33"}}   # a state on the pinned image (versions illustrative)
SLOW = {"atom_attn": "sdpa", "esmc_mlp": "torch", "esmc_attn": "sdpa(chain_mask)", "esmc_rope": "torch", "versions": {}, "import_errors": {"flash_attn": "ModuleNotFoundError: No module named 'flash_attn'"}}


def fake_common(flag: bool):
    m = types.ModuleType(attn.COMMON_MODULE); m.FLASH_ATTN_AVAILABLE = flag; return m


def fake_esmc(te: bool, rope: bool, xf: bool = False, fa: bool = False):
    m = types.ModuleType(attn.ESMC_MODULE)
    m._te_available, m._flash_attn_rotary_available, m._xformers_available, m._flash_attn_available = te, rope, xf, fa
    return m


def bare_box():
    """A box with none of the accelerated packages and upstream not importable, whatever this interpreter has: the two upstream modules masked in
    sys.modules (``None`` = the import raises), the distribution metadata empty, every import error named — so the tests encode the reader's logic,
    not the image they run on (these tests run on the pinned image, where all three packages import, and on the torch-only test image)."""
    stack_ctx = contextlib.ExitStack()
    stack_ctx.enter_context(mock.patch.dict(sys.modules, {attn.COMMON_MODULE: None, attn.ESMC_MODULE: None}))
    stack_ctx.enter_context(mock.patch.object(attn, "versions", lambda: {}))
    stack_ctx.enter_context(mock.patch.object(attn, "import_error", lambda mod: f"ModuleNotFoundError: No module named '{mod.split('.')[0]}'"))
    return stack_ctx


class SWA3DRoPEAttention:                                                         # named as upstream's class: the reader finds instances by class name
    def forward(self, x, attention_params):
        return x


def _swa_forward_cached(self, x, attention_params):                              # named as the fast line's U1 forward (driver/ef2_opt.py)
    return x


def _forward_banded(self, x, attention_params):                                  # a stand-in for atom_swa's forward, passed through the words map
    return x


def _something_else(self, x, attention_params):
    return x


LayerNormMLP = type("LayerNormMLP", (), {"__module__": "transformer_engine.pytorch.module.layernorm_mlp"})   # TE-like: classified by the class's module path
_PyTorchLayerNormMLP = type("_PyTorchLayerNormMLP", (), {})                      # the fork's pure-PyTorch fallback: classified by class name
MultiHeadAttention = type("MultiHeadAttention", (), {})
_FlashMultiHeadAttention = type("_FlashMultiHeadAttention", (), {})


class FakeModel:
    def __init__(self, mods):
        self._mods = list(mods)

    def modules(self):
        yield self
        yield from self._mods


def atoms(n_stock=0, n_u1=0, n_other=0):
    out = [SWA3DRoPEAttention() for _ in range(n_stock)]
    for _ in range(n_u1):
        a = SWA3DRoPEAttention(); a.forward = types.MethodType(_swa_forward_cached, a); out.append(a)
    for _ in range(n_other):
        a = SWA3DRoPEAttention(); a.forward = types.MethodType(_something_else, a); out.append(a)
    return out


class ReaderTest(unittest.TestCase):
    def test_flag_state_reads_the_module_constant(self):
        self.assertEqual(attn.flag_state(fake_common(True))["available"], True)
        self.assertEqual(attn.flag_state(fake_common(False))["available"], False)
        with bare_box():
            self.assertIsNone(attn.flag_state()["available"])                    # upstream's module not imported: unread, not guessed
        with self.assertRaises(AttributeError):
            attn.flag_state(types.ModuleType("no_flag"))

    def test_forward_resolution_words(self):
        a = atoms(2, 1, 1); b = SWA3DRoPEAttention(); b.forward = types.MethodType(_forward_banded, b)
        res = attn.forward_resolution(a + [b], {_forward_banded: attn.BANDED_WORD})
        self.assertEqual(res, {"banded": 1, "instance:_something_else": 1, "instance:_swa_forward_cached": 1, "stock": 2})

    def test_kernel_word(self):
        self.assertEqual(attn.kernel_word(True, {}), "flash_attn"); self.assertEqual(attn.kernel_word(False, {}), "sdpa"); self.assertEqual(attn.kernel_word(None, {}), "unread")
        self.assertEqual(attn.kernel_word(True, {"stock": 6, "instance:_swa_forward_cached": 3}), "flash_attn")   # U1 passes through under the flag
        self.assertEqual(attn.kernel_word(False, {"instance:_swa_forward_cached": 9}), "sdpa")
        self.assertEqual(attn.kernel_word(True, {"stock": 8, "banded": 1}), "mixed(flash_attnx8,sdpax1)")
        self.assertEqual(attn.kernel_word(True, {"instance:_something_else": 2}), "unknown(instance:_something_else)")

    def test_esmc_state_module_and_bound(self):
        st = attn.esmc_state(esmc=fake_esmc(te=True, rope=True, xf=True))            # no model: what the factories will bind
        self.assertEqual((st["esmc_mlp"], st["esmc_attn"], st["esmc_rope"], st["esmc_source"]), ("te", "sdpa(chain_mask)", "flash_attn_triton", "module"))
        st = attn.esmc_state(esmc=fake_esmc(te=False, rope=False))
        self.assertEqual((st["esmc_mlp"], st["esmc_rope"]), ("torch", "torch"))
        with bare_box():
            self.assertEqual(attn.esmc_state()["esmc_mlp"], "unread")
        m = FakeModel([LayerNormMLP(), LayerNormMLP(), MultiHeadAttention()])
        st = attn.esmc_state(m, esmc=fake_esmc(te=True, rope=True))
        self.assertEqual((st["esmc_mlp"], st["esmc_attn"], st["esmc_source"], st["esmc_bound"]), ("te", "sdpa(chain_mask)", "bound", {"mlp": {"te": 2}, "attn": {"sdpa(chain_mask)": 1}}))
        st = attn.esmc_state(FakeModel([LayerNormMLP(), _PyTorchLayerNormMLP()]), esmc=fake_esmc(te=True, rope=True))
        self.assertEqual(st["esmc_mlp"], "mixed(tex1,torchx1)")                      # the bound classes, not the flag, decide the word
        st = attn.esmc_state(FakeModel([_PyTorchLayerNormMLP(), _FlashMultiHeadAttention()]), esmc=fake_esmc(te=False, rope=False))
        self.assertEqual((st["esmc_mlp"], st["esmc_attn"]), ("torch", "flash_attn_varlen"))

    def test_state_and_words_with_a_model(self):
        m = FakeModel(atoms(6, 3) + [LayerNormMLP(), MultiHeadAttention()])
        with mock.patch.dict("sys.modules", {attn.ESMC_MODULE: fake_esmc(te=True, rope=True, xf=True)}), \
                mock.patch.object(attn, "versions", lambda: dict(FAST["versions"])), mock.patch.object(attn, "import_error", lambda mod: None):
            st = attn.state(m, common=fake_common(True))
        self.assertEqual((st["atom_attn"], st["atom_modules"], st["atom_forward"]), ("flash_attn", 9, {"instance:_swa_forward_cached": 3, "stock": 6}))
        self.assertEqual(attn.words(st), "atom_attn=flash_attn atom_forward=instance:_swa_forward_cachedx3,stockx6 esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton "
                                         "flash_attn=2.8.3.post1 transformer_engine=2.6.0 xformers=0.0.33")
        self.assertEqual(attn.words(st, forward=False), "atom_attn=flash_attn esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton flash_attn=2.8.3.post1 transformer_engine=2.6.0 xformers=0.0.33")
        with bare_box():
            st = attn.state(FakeModel(atoms(9)), common=fake_common(False))        # a bare box: nothing accelerated, the words say so
        self.assertEqual((st["atom_attn"], st["upstream_flag"]), ("sdpa", False))
        self.assertTrue(attn.words(st).endswith("flash_attn=absent transformer_engine=absent xformers=absent"))
        self.assertIn("flash_attn", st["import_errors"])                            # named: why the accelerated package does not import here

    def test_metadata_words(self):
        self.assertEqual(attn.metadata_words({}), "flash_attn=absent transformer_engine=absent xformers=absent")
        self.assertEqual(attn.metadata_words({"flash_attn": "2.8.3.post1"}), "flash_attn=2.8.3.post1 transformer_engine=absent xformers=absent")


class LinesTest(unittest.TestCase):
    def test_settings_line_carries_the_words(self):
        state = {"kernel_backend": "'fused'", "chunk": "None", "opm_chunk": "None"}
        base = stock_fold.settings_line("shipped", [], state)
        self.assertNotIn("atom_attn", base)                                          # without a reading the line is unchanged
        line = stock_fold.settings_line("shipped", [], state, dict(FAST, atom_forward={"stock": 9}))
        self.assertTrue(line.startswith(base + " atom_attn=flash_attn atom_forward=stockx9 esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton flash_attn=2.8.3.post1"), line)

    def test_active_dry_run_and_applied_lines(self):
        rep = {"active": True, "mode": "fast", "variant": "fast", "server_line": "opt14_msa", "levers_applied": [], "levers_fallback": [], "applied": "deferred",
               "attn": dict(FAST, atom_forward={})}
        self.assertIn(" applied=deferred atom_attn=flash_attn esmc_mlp=te esmc_attn=sdpa(chain_mask) esmc_rope=flash_attn_triton flash_attn=2.8.3.post1", report.activation_line(rep))
        dry = dict(rep, active=False, dry_run=True, server_mode="opt14_msa", attn_metadata="flash_attn=absent transformer_engine=absent xformers=absent"); dry.pop("attn")
        self.assertIn(" flash_attn=absent transformer_engine=absent xformers=absent", report.activation_line(dry))
        rec = {"server_mode": "opt14_msa", "levers_applied": ["tg"], "levers_fallback": [], "levers_not_for_variant": [], "attn": dict(FAST, atom_forward={"instance:_swa_forward_cached": 9})}
        self.assertIn(" atom_attn=flash_attn atom_forward=instance:_swa_forward_cachedx9 esmc_mlp=te ", report.applied_line({}, rec))


class RequireTest(unittest.TestCase):
    def _enter(self, cm):                                                                     # unittest's enterContext, for interpreters older than 3.11
        cm.__enter__(); self.addCleanup(cm.__exit__, None, None, None)

    IMG = {"python_packages": {"flash_attn": "2.8.3.post1", "transformer_engine": "2.6.0"}}

    def test_require_value(self):
        self.assertFalse(attn.require_value({})); self.assertFalse(attn.require_value({attn.ENV_REQUIRE: "0"})); self.assertFalse(attn.require_value({attn.ENV_REQUIRE: ""}))
        self.assertTrue(attn.require_value({attn.ENV_REQUIRE: "1"}))
        with self.assertRaisesRegex(ValueError, "ESMFOLD2_OPT_REQUIRE_FAST_ENV='yes': expected 1"):
            attn.require_value({attn.ENV_REQUIRE: "yes"})

    def test_refusal_sentence(self):
        self.enterContext(bare_box()) if hasattr(self, "enterContext") else self._enter(bare_box())
        on = {attn.ENV_REQUIRE: "1"}
        self.assertIsNone(attn.require_refusal(SLOW, environ={}, image=self.IMG))             # switch unset: report only
        self.assertIsNone(attn.require_refusal(FAST, environ=on, image=self.IMG))            # every required word at its accelerated value
        self.assertEqual(attn.failing_words(FAST), [])
        why = attn.require_refusal(SLOW, environ=on, image=self.IMG)
        self.assertTrue(why.startswith("ESMFOLD2_OPT_REQUIRE_FAST_ENV=1 requires the pinned stack's accelerated paths and this interpreter lacks "
                                       "atom_attn=sdpa (expected flash_attn; import flash_attn: ModuleNotFoundError: No module named 'flash_attn'), esmc_mlp=torch (expected te; import transformer_engine.pytorch: "), why)
        self.assertIn("esmc_rope=torch (expected flash_attn_triton; import flash_attn.ops.triton.rotary: ", why)
        self.assertIn("— the pinned stack's accelerated layer (flash_attn 2.8.3.post1, transformer_engine 2.6.0; stock/PINS.json \"image\") is not live here: refused (rc 3), no fallback to the slow paths", why)
        self.assertNotRegex(why, r"\bim-[A-Za-z0-9]+")                                              # no platform identifier in the sentence
        self.assertNotIn("esmc_attn", why)                                                   # reported, never required (chain-aware sequence_id: sdpa on every image)
        part = dict(FAST, esmc_mlp="torch")                                                  # one word off is enough
        self.assertRegex(attn.require_refusal(part, environ=on, image=self.IMG), r"lacks esmc_mlp=torch \(expected te; import transformer_engine\.pytorch: \w+Error")
        unread = attn.require_refusal({}, environ=on, image=self.IMG)                         # nothing read (a stand-in upstream): every required word fails by name
        for w in ("atom_attn=unread", "esmc_mlp=unread", "esmc_rope=unread"):
            self.assertIn(w, unread)

    def test_refusal_metadata_form(self):
        on = {attn.ENV_REQUIRE: "1"}
        self.assertIsNone(attn.require_refusal_metadata(environ={}, image=self.IMG, vers={}))
        self.assertIsNone(attn.require_refusal_metadata(environ=on, image=self.IMG, vers={"flash_attn": "2.8.3.post1", "transformer_engine": "2.6.0"}))
        why = attn.require_refusal_metadata(environ=on, image=self.IMG, vers={"flash_attn": "2.8.3.post1"})
        self.assertTrue(why.startswith("ESMFOLD2_OPT_REQUIRE_FAST_ENV=1 requires the pinned stack's accelerated paths and the distribution metadata lacks transformer_engine "
                                       "(flash_attn=2.8.3.post1 transformer_engine=absent xformers=absent) — the pinned stack's accelerated layer (flash_attn 2.8.3.post1, "
                                       "transformer_engine 2.6.0; stock/PINS.json \"image\") is not live here: refused (rc 3)"), why)

    def test_stack_attn_gate(self):
        self.enterContext(bare_box()) if hasattr(self, "enterContext") else self._enter(bare_box())
        base = {"dry_run": True}
        self.assertIsNone(stack.attn_gate(base, {"image": self.IMG}, environ={}))
        self.assertEqual(base["attn_metadata"], "flash_attn=absent transformer_engine=absent xformers=absent")
        why = stack.attn_gate({"dry_run": True}, {"image": self.IMG}, environ={attn.ENV_REQUIRE: "1"})
        self.assertIn("the distribution metadata lacks flash_attn, transformer_engine", why); self.assertIn("stock/PINS.json \"image\"", why)
        self.assertIn("expected 1", stack.attn_gate({"dry_run": True}, {}, environ={attn.ENV_REQUIRE: "2"}))
        base = {}                                                                             # an activation on this box: upstream's modules do not import → unread, refused by name under the switch
        why = stack.attn_gate(base, {"image": self.IMG}, environ={attn.ENV_REQUIRE: "1"})
        self.assertIn("atom_attn=unread", why); self.assertEqual(base["attn"]["atom_attn"], "unread")
        self.assertIsNone(stack.attn_gate({}, {"image": self.IMG}, environ={}))

    def test_the_switch_is_declared_and_reaches_the_stock_subprocess(self):
        from esmfold2_opt import _autoload
        self.assertIn(attn.ENV_REQUIRE, _autoload.ENV_NAMES)                                 # a mistyped name under the prefix is refused at interpreter start; this one is declared
        self.assertTrue(attn.ENV_REQUIRE.startswith(stack.ENV_MODE + "_"))                  # under the package prefix: _autoload refuses undeclared names there
        self.assertNotIn(attn.ENV_REQUIRE, stack.PACKAGE_ENV)                               # not stripped from the stock subprocess (cli: prefixes ending in '_' strip by prefix, other entries by exact name)
        self.assertFalse(any((attn.ENV_REQUIRE.startswith(s) if s.endswith("_") else attn.ENV_REQUIRE == s) for s in stack.stock_env_absent(PINS)))
        cfg = open(os.path.join(TREE, "configs", "h100.env")).read()
        self.assertRegex(cfg, r"(?m)^export ESMFOLD2_OPT_REQUIRE_FAST_ENV=\$\{ESMFOLD2_OPT_REQUIRE_FAST_ENV:-1\}")
        last = [l for l in cfg.splitlines() if l.startswith("export ")][-1]
        self.assertTrue(last.startswith("export ESMFOLD2_OPT_REQUIRE_FAST_ENV="))            # the LAST export: the interpreter probes earlier in the file never see it


class ImagePinsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("check_pins_under_test", os.path.join(TREE, "stock", "check_pins.py"))
        cls.cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cls.cp)

    def _with_versions(self, table):
        def version(name):
            key = name.replace("-", "_")
            if key in table:
                return table[key]
            raise self.cp.md.PackageNotFoundError(name)
        return mock.patch.object(self.cp.md, "version", version)

    def test_pins_image_block(self):
        """The layer is pinned by content (package versions, wheel names + sha256, the install step), never by a platform identifier: the file
        carries no image / sandbox id at all; pinned_stack.tested_on names the card and the stack versions (a documentation line nothing compares against)."""
        img = PINS["image"]
        self.assertNotIn("id", img); self.assertNotIn("registry", img); self.assertIsInstance(img["base"], str)
        self.assertNotIn("image", PINS["pinned_stack"]); self.assertIsInstance(PINS["pinned_stack"]["tested_on"], str)
        self.assertEqual(re.findall(r"\b(?:im|sb|ap)-[A-Za-z0-9]{10,}", json.dumps(PINS)), [])          # no platform identifier anywhere in the pin card
        self.assertIn("H100", PINS["pinned_stack"]["tested_on"]); self.assertIn(PINS["pinned_stack"]["torch"], PINS["pinned_stack"]["tested_on"])
        self.assertTrue(all("build" not in w or "BUILD_RECORD" not in w["build"] for w in img["wheels"]))   # no build-record pointers; wheel identity = name + sha256 + cuda_archs
        self.assertEqual(img["python_packages"]["flash_attn"], PINS["pinned_stack"]["flash_attn"])
        for w in img["wheels"]:
            self.assertRegex(w["sha256"], r"^[0-9a-f]{64}$"); self.assertTrue(w["name"].endswith(".whl")); self.assertNotIn("stored", w)
        self.assertTrue(any(w["name"].startswith("flash_attn-2.8.3.post1") for w in img["wheels"]))
        self.assertNotIn("flash_attn", PINS["system"]["optional_absent"])

    def test_absent_package_is_refused_by_name(self):
        img = {"python_packages": {"flash_attn": "2.8.3.post1", "transformer_engine": "2.6.0"}}
        with self._with_versions({}):
            bad, detail = self.cp.image_pins(img)
        self.assertEqual(len(bad), 2, bad)
        self.assertEqual(bad[0], "flash_attn: expected 2.8.3.post1 got None — the pinned stack's accelerated layer (stock/PINS.json \"image\") is not installed here")
        self.assertEqual(bad[1], "transformer_engine: expected 2.6.0 got None — the pinned stack's accelerated layer (stock/PINS.json \"image\") is not installed here")
        self.assertEqual(detail["flash_attn"], {"expected": "2.8.3.post1", "got": None, "pinned": False})

    def test_wrong_version_is_refused_and_the_pin_passes(self):
        img = PINS["image"]
        with self._with_versions({"flash_attn": "2.7.4", **{k: v for k, v in img["python_packages"].items() if k != "flash_attn"}}):
            bad, _ = self.cp.image_pins(img)
        self.assertEqual(len(bad), 1); self.assertIn(f"expected {img['python_packages']['flash_attn']} got 2.7.4", bad[0])
        with self._with_versions(dict(img["python_packages"])):
            bad, detail = self.cp.image_pins(img)
        self.assertEqual(bad, []); self.assertTrue(all(d["pinned"] for d in detail.values()))
        self.assertEqual(self.cp.image_pins(None), ([], {}))                        # no image block pins nothing

    def test_main_exits_3_naming_the_layer(self):
        import io, contextlib
        err = io.StringIO()
        with self._with_versions({}), mock.patch.object(self.cp, "check", lambda pins: ([], {})), \
                mock.patch.object(self.cp.sys, "argv", ["check_pins.py", "--quiet"]), contextlib.redirect_stderr(err), self.assertRaises(SystemExit) as cm:
            self.cp.main()
        self.assertEqual(cm.exception.code, 3)
        self.assertIn(f"check_pins: image pins NOT MET: flash_attn: expected {PINS['image']['python_packages']['flash_attn']} got None — the pinned stack's accelerated layer (stock/PINS.json \"image\") is not installed here", err.getvalue())


class ActivationOrderTest(unittest.TestCase):
    """The memory lines export their allocator policy BEFORE anything of torch / upstream is imported by the activation: the stack gates apply
    the fail-loud switch in its metadata form (no import), and the run-time words / gate — which import upstream's two modeling modules (on the
    pinned image transformer-engine and xformers initialise CUDA at import) — run after big's alloc_export and before the kit import."""

    def test_metadata_form_imports_nothing(self):
        base = {"dry_run": False}
        with mock.patch.object(attn, "state", side_effect=AssertionError("attn.state must not run in the metadata form")), \
                mock.patch.object(attn, "flag_state", side_effect=AssertionError("no upstream import in the metadata form")):
            self.assertIsNone(stack.attn_gate(base, {"image": {}}, environ={}, metadata_only=True))
        self.assertIn("attn_metadata", base); self.assertNotIn("attn", base)

    def test_stack_gates_use_the_metadata_form(self):
        calls = []
        passing = lambda *a, **k: (True, {}, None)
        with mock.patch.object(stack, "attn_gate", lambda base, p, **k: calls.append(k) or "stop here (test)"), \
                mock.patch.object(stack, "version_gate", passing), mock.patch.object(stack, "pins_gate", passing), \
                mock.patch("esmfold2_opt._producers.refusal", lambda: None), \
                mock.patch("esmfold2_opt._core_gate.gate", lambda *a, **k: {"pinned": {}, "installed": {}}):
            why = stack._stack_gates({"gpu": {}, "notes": [], "target_gpu": None}, {"image": {}})
        self.assertEqual(why, "stop here (test)")                                          # the attn gate's refusal is the stack gates' reason, before any GPU probe
        self.assertTrue(calls, "the stack gates call attn_gate"); self.assertTrue(all(k.get("metadata_only") for k in calls), calls)

    def test_big_alloc_export_precedes_the_runtime_gate(self):
        order = []
        res = types.SimpleNamespace(notes=[], overrides={"EF2_TEST_OVERRIDE": "1"}, composition={"xl": True, "alloc_strict": True})   # the memory mode (modes.KitMode.alloc_strict)
        def runtime_gate(base, p, **k):
            order.append("metadata" if k.get("metadata_only") else "runtime")
            return None if k.get("metadata_only") else "stop here (test)"
        gates = lambda base, variant: (base.update(kit_home=os.path.join(TREE, "opt", "forward")), (None, {"image": {}}))[1]
        with mock.patch.object(stack, "_gates", gates), \
                mock.patch.object(stack, "register_instance_counter", lambda: {"n": 0, "method": "counted"}), \
                mock.patch.object(stack, "kit_levers_applied", lambda: []), \
                mock.patch.object(stack, "resolve", lambda *a, **k: res), \
                mock.patch.object(stack._big, "alloc_export", lambda: order.append("alloc_export") or {"alloc": "test"}), \
                mock.patch.object(stack, "attn_gate", runtime_gate), \
                mock.patch.dict(os.environ, {}, clear=False):
            rep = stack._activate_body("big", "fast", strict=False, trigger=None)
            self.assertNotIn("EF2_TEST_OVERRIDE", os.environ)                            # a refused process keeps the caller's environment
        self.assertEqual(order, ["alloc_export", "runtime"], order)
        self.assertFalse(rep.get("active")); self.assertEqual(rep.get("reason"), "stop here (test)")


if __name__ == "__main__":
    unittest.main()
