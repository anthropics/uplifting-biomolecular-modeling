"""The `xln` lever (driver/ef2_xln.py: the release tree's exactln LayerNorm row at the pair-sized nn.LayerNorm sites): its registration in the
kit's tables, the site selection, the per-call routing (rows floor, dtype forms, exactln's per-call words -> the verbatim statement), the
install-time refusals BY NAME, and the LEVER line's words for an engaged and for a refused install.  CPU only: the kernel itself is the
release tree's row (its bitwise cells live in opt_core.kernels.ln.LN_CELLS); a device run is the kit's identity procedure (README)."""
import os
import sys
import unittest
from unittest import mock

import torch
import torch.nn.functional as F

from esmfold2_opt import modes, registry, report, stack

HERE = os.path.dirname(os.path.abspath(__file__))
DRIVER = os.path.normpath(os.path.join(HERE, "..", "..", "forward", "fast_inference", "driver"))
ef2_xln = None                     # imported in setUpModule, dropped in tearDownModule: a driver module in sys.modules at collection time would hand the
                                   # other tests' EXIT-tally / LEVER-line reads a record this process never made


def setUpModule():  # noqa: N802
    global ef2_xln
    if DRIVER not in sys.path:
        sys.path.insert(0, DRIVER)
    import ef2_xln as _m
    ef2_xln = _m


class _Sites(torch.nn.Module):
    """A module tree with the model's LayerNorm site names: served (msa / parcae / base_z / confidence / conditioning / atom) and excluded ones."""

    def __init__(self):
        super().__init__()
        ln = torch.nn.LayerNorm
        self.msa_encoder = torch.nn.ModuleDict({"blocks": torch.nn.ModuleList([torch.nn.ModuleDict({"pair_transition": torch.nn.ModuleDict({"norm": ln(256)}),
                                                                                                     "outer_product_mean": torch.nn.ModuleDict({"norm": ln(128)})})])})
        self.parcae_input_norm = ln(256)
        self.language_model = torch.nn.ModuleDict({"base_z_mlp": torch.nn.Sequential(torch.nn.Identity(), ln(256)), "base_z_linear": torch.nn.Sequential(ln(2560)),
                                                  "model": torch.nn.ModuleDict({"norm": ln(256)})})          # under the LM: excluded by name even at a served width
        self._esmc = torch.nn.ModuleDict({"transformer": torch.nn.ModuleDict({"norm": ln(2560)})})
        self.confidence_head = torch.nn.ModuleDict({"z_norm": ln(256), "pae_ln": ln(256), "plddt_ln": ln(384), "s_inputs_norm": ln(451)})   # 451 % 4 != 0: not served
        dm = torch.nn.ModuleDict({"conditioning": torch.nn.ModuleDict({"z_input_norm": ln(512)}), "token_norm": ln(768), "s_step_norm": ln(768),
                                  "token_transformer": torch.nn.ModuleDict({"a_norm": ln(768, elementwise_affine=False), "s_norm": ln(768, bias=False), "q_norm": ln(768)}),
                                  "atom_encoder": torch.nn.ModuleDict({"atom_norm": ln(128)})})
        self.structure_head = torch.nn.ModuleDict({"diffusion_module": dm})


class _FakeCuda(torch.Tensor):
    """A CPU tensor that answers is_cuda=True, so the routing below the device check can be exercised on a CPU host."""
    @property
    def is_cuda(self):  # noqa: D401
        return True


class _FakeE:
    """Stands in for opt_core.kernels.ln.exactln: records the call and returns F.layer_norm's own result (or raises its Unsupported word)."""
    class Unsupported(Exception):
        def __init__(self, reason):
            super().__init__(reason); self.reason = reason
    calls = []
    refuse = None

    @classmethod
    def layer_norm(cls, x, shape, w, b, eps, widen=False):
        cls.calls.append({"shape": tuple(x.shape), "dtype": x.dtype, "w": (w.dtype if w is not None else None), "widen": widen, "eps": eps})
        if cls.refuse:
            raise cls.Unsupported(cls.refuse)
        xx = x.float() if widen else x
        return F.layer_norm(torch.Tensor(xx) if isinstance(xx, _FakeCuda) else xx, shape, w, b, eps)


class TestRegistration(unittest.TestCase):
    def test_registry_row_and_strategy(self):
        lv = registry.LEVERS["xln"]
        self.assertEqual((lv.field, lv.kit_file, lv.class_4, lv.tier_vs_kit_line, lv.probe), (registry.FIELD_LN, "ef2_xln.py", "forward", "bitwise", ("ln", "xln")))
        self.assertEqual(set(lv.variants), set(registry.ALL_VARIANTS))            # both models: the Fast model's parcae / base_z / confidence sites, the Full model's MSA module too
        self.assertEqual(registry.STRATEGY["xln"], "F5.row_layernorm")
        self.assertIn("ln", modes.GROUPS); self.assertLess(modes.GROUP_INSTALL_ORDER.index("hoist"), modes.GROUP_INSTALL_ORDER.index("ln")); self.assertLess(modes.GROUP_INSTALL_ORDER.index("ln"), modes.GROUP_INSTALL_ORDER.index("dit"))

    def test_every_kit_mode_carries_it_in_the_server_table(self):
        kit = stack.kit_home()
        if not os.path.isfile(os.path.join(kit, modes.SERVER_RELPATH)):
            self.skipTest("kit tree not present")
        for mode in ("exact", "fast", "big"):
            for variant in ("fast", "full_msa"):
                res = modes.resolve(mode, variant, kit)
                self.assertEqual(modes.parse_extra(res.entry[3]).get("ln"), ["xln"], (mode, variant))
                self.assertIn("xln", res.levers_for_variant, (mode, variant))
        srv = open(os.path.join(kit, modes.SERVER_RELPATH)).read()
        self.assertIn('"ln"', srv.split("GROUPS = (")[1].split(")")[0])                 # ef2_server.GROUPS knows the group (an unknown group is a table error)
        self.assertIn("import ef2_xln", srv); self.assertIn("ef2_xln.install(model", srv)


class TestSites(unittest.TestCase):
    def test_site_selection(self):
        names = sorted(n for n, _ in ef2_xln.eligible_sites(_Sites()))
        self.assertEqual(names, sorted(["msa_encoder.blocks.0.pair_transition.norm", "msa_encoder.blocks.0.outer_product_mean.norm", "parcae_input_norm", "language_model.base_z_mlp.1",
                                        "confidence_head.z_norm", "confidence_head.pae_ln", "confidence_head.plddt_ln", "structure_head.diffusion_module.conditioning.z_input_norm",
                                        "structure_head.diffusion_module.atom_encoder.atom_norm"]))
        for excluded in ("language_model.base_z_linear.0", "_esmc.transformer.norm", "language_model.model.norm", "confidence_head.s_inputs_norm",
                         "structure_head.diffusion_module.token_norm", "structure_head.diffusion_module.s_step_norm", "structure_head.diffusion_module.token_transformer.q_norm"):
            self.assertNotIn(excluded, names)


class TestRouting(unittest.TestCase):
    def setUp(self):
        _FakeE.calls = []; _FakeE.refuse = None
        ef2_xln.STATS.clear(); ef2_xln._STATE["E"] = _FakeE
        self.m = torch.nn.LayerNorm(256)
        import types as _t
        self.m.forward = _t.MethodType(ef2_xln._xln_forward, self.m)

    def _x(self, rows, dtype=torch.float32, fake_cuda=True):
        t = torch.randn(rows, 256, dtype=torch.float32).to(dtype)
        return t.as_subclass(_FakeCuda) if fake_cuda else t

    def test_cpu_tensor_runs_the_statement(self):
        y = self.m(self._x(8192, fake_cuda=False))
        self.assertEqual(ef2_xln.STATS["not_cuda"], 1); self.assertEqual(_FakeE.calls, []); self.assertEqual(tuple(y.shape), (8192, 256))

    def test_below_the_floor_runs_the_statement(self):
        self.m(self._x(ef2_xln.FLOOR_ROWS - 1))
        self.assertEqual(ef2_xln.STATS["below_floor"], 1); self.assertEqual(_FakeE.calls, [])

    def test_fp32_at_the_floor_is_served(self):
        x = self._x(ef2_xln.FLOOR_ROWS)
        y = self.m(x)
        self.assertEqual(ef2_xln.STATS["served"], 1); self.assertEqual(len(_FakeE.calls), 1); self.assertFalse(_FakeE.calls[0]["widen"])
        self.assertTrue(torch.equal(torch.Tensor(y), F.layer_norm(torch.Tensor(x), (256,), self.m.weight, self.m.bias, self.m.eps)))

    def test_bf16_rows_with_fp32_parameters_take_the_widen_form(self):
        self.m(self._x(ef2_xln.FLOOR_ROWS, dtype=torch.bfloat16))          # the model's form under its bf16 autocast: bf16 x, fp32 parameters -> fp32 result
        self.assertEqual(len(_FakeE.calls), 1); self.assertTrue(_FakeE.calls[0]["widen"]); self.assertEqual(_FakeE.calls[0]["w"], torch.float32)

    def test_other_dtypes_run_the_statement_counted(self):
        self.m.double()
        self.m(self._x(ef2_xln.FLOOR_ROWS, dtype=torch.float64))
        self.assertEqual(ef2_xln.STATS["dtype:float64"], 1); self.assertEqual(_FakeE.calls, [])

    def test_exactln_per_call_words_fall_through_counted(self):
        _FakeE.refuse = "width"
        x = self._x(ef2_xln.FLOOR_ROWS)
        y = self.m(x)
        self.assertEqual(ef2_xln.STATS["fallback:width"], 1)
        self.assertTrue(torch.equal(torch.Tensor(y), F.layer_norm(torch.Tensor(x), (256,), self.m.weight, self.m.bias, self.m.eps)))
        st = ef2_xln.stats()
        self.assertEqual((st["xln_served"], st["xln_fallback"], st["xln_fallback_width"]), (0, 1, 1))


class TestInstallRefusals(unittest.TestCase):
    def setUp(self):
        ef2_xln.uninstall(); ef2_xln._STATE["refusal"] = None

    def test_no_cuda_steps_aside_by_name(self):
        if torch.cuda.is_available():
            self.skipTest("a CUDA device is visible")
        model = _Sites()
        d = ef2_xln.install(model)
        self.assertEqual(d["levers"], {"xln": False}); self.assertEqual(ef2_xln.refusal(), "no_cuda"); self.assertFalse(ef2_xln.levers_on()["xln"])
        self.assertFalse(any("forward" in vars(m) for _, m in ef2_xln.eligible_sites(model)))     # nothing patched
        with self.assertRaises(ef2_xln.XlnUnavailable) as cm:
            ef2_xln.install(model, strict=True)
        self.assertEqual(cm.exception.reason, "no_cuda")

    def test_core_below_the_floor_steps_aside_by_name(self):
        import opt_core
        with mock.patch.object(opt_core, "__version__", "0.5.27.9"):
            d = ef2_xln.install(_Sites())
        self.assertEqual(d["refusal"], "core_below_0.5.28.0:0.5.27.9"); self.assertEqual(d["levers"], {"xln": False})

    def test_xln_false_installs_nothing(self):
        d = ef2_xln.install(_Sites(), xln=False)
        self.assertEqual(d["levers"], {"xln": False}); self.assertIsNone(d["refusal"])

    def test_uninstall_restores_the_class_forward(self):
        model = _Sites(); sites = ef2_xln.eligible_sites(model)
        import types as _t
        for _, m in sites:
            m.forward = _t.MethodType(ef2_xln._xln_forward, m); ef2_xln._STATE["patched"].append(m)
        ef2_xln._STATE["on"] = True
        ef2_xln.uninstall(model)
        self.assertFalse(any("forward" in vars(m) for _, m in sites)); self.assertFalse(ef2_xln.levers_on()["xln"])


class TestLeverLine(unittest.TestCase):
    """The LEVER line's words: engaged -> state=on + the module's evidence; refused at install -> state=skipped reason=guard:off:<word>."""

    def _lines(self, rec):
        rep = {"mode": "exact", "variant": "full_msa", "gpu": {"cc": "9.0"}, "ablate": [], "levers_dropped": []}
        return {l.split("name=")[1].split()[0]: l for l in report.lever_lines(rep, rec)}

    def test_engaged(self):
        fake = type(sys)("ef2_xln")
        fake.evidence = lambda: {"sites": 57, "floor_rows": 4096, "core": "0.5.32.0", "kernels": 6, "cc": "9.0", "provider": "opt_core.kernels.ln.exactln"}
        with mock.patch.dict(sys.modules, {"ef2_xln": fake}):
            line = self._lines({"levers_applied": ["xln"], "model_index": 0})["xln"]
        for word in ("state=on", "impl=ef2_xln.py", "origin=kit", "strategy=F5.row_layernorm", "tier_vs_stock=T2", "sites=57", "floor_rows=4096", "core=0.5.32.0", "provider=opt_core.kernels.ln.exactln"):
            self.assertIn(word, line)

    def test_refused(self):
        line = self._lines({"levers_applied": [], "levers_out_of_scope": {"xln": "off:cc:10.0_not_proven"}, "model_index": 0})["xln"]
        self.assertIn("state=skipped", line); self.assertIn("reason=guard:off:cc:10.0_not_proven", line)


def tearDownModule():  # noqa: N802 — leave no ef2_xln record behind for the other tests' exit-tally / lever-line reads of sys.modules
    ef2_xln.uninstall(); ef2_xln.STATS.clear(); ef2_xln._STATE["sites"] = {}
    sys.modules.pop("ef2_xln", None)


if __name__ == "__main__":
    unittest.main()
