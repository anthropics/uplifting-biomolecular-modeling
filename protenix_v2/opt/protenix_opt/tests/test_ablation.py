"""MODEL_OPT_LEVERS_OFF — the ablation switch: registry lever names, refused by name when unknown / outside the mode / switchless /
under mode off; applied after modes.resolve() by removing the levers' switches; `ablated=<names>` on the ACTIVE / DRY-RUN / FINAL lines;
an ablated lever's LEVER line reads state=off reason=ablated; unset = the mode's lines byte for byte."""
import io
import os
import sys
import unittest
from contextlib import redirect_stderr
from unittest import mock

from protenix_opt import ablation, modes, report, stack
from protenix_opt.registry import LEVERS
from protenix_opt.tests.test_enable_gates import _Base

K2B = "k2b_flash_triattention"
NATIVE = "triattn_native"                                  # cc 9.0 / 8.0 (protenix_opt 0.3.51): the row's Tier-2 attention through the shared core's provider by the tier word (PTX_T_ATT=fast)
H100 = {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90", "cc": "9.0", "probe": "nvidia-smi"}


def _dry_run(mode, levers_off=None):
    """The core's dry run of `mode` on an H100 row (9.0|3.7) with MODEL_OPT_LEVERS_OFF=levers_off (None: unset); (report, stderr)."""
    err = io.StringIO()
    env = {ablation.ENV: levers_off} if levers_off is not None else {}
    with mock.patch.dict(os.environ, env), mock.patch.object(stack, "protenix_version", return_value="2.0.0"), \
         mock.patch.object(modes, "triton_version", return_value="3.7.1"), mock.patch.object(stack, "gpu_probe_smi", return_value=dict(H100)), \
         redirect_stderr(err):
        if levers_off is None:
            os.environ.pop(ablation.ENV, None)
        r = stack.activate(mode, dry_run=True)
    return r, err.getvalue()


class TestRequest(unittest.TestCase):
    def test_unset_or_blank_is_no_ablation(self):
        self.assertEqual(ablation.requested({}), [])
        self.assertEqual(ablation.requested({ablation.ENV: ""}), [])
        self.assertEqual(ablation.requested({ablation.ENV: " , "}), [])

    def test_comma_list_order_kept_duplicates_folded(self):
        self.assertEqual(ablation.requested({ablation.ENV: f" {K2B}, pad8 ,{K2B}"}), [K2B, "pad8"])

    def test_env_name_is_the_house_name(self):
        self.assertEqual(ablation.ENV, "MODEL_OPT_LEVERS_OFF"); self.assertEqual(ablation.REASON, report.ABLATED)

    def test_switches_are_env_keys_plus_conditional(self):
        self.assertEqual(ablation.switches(K2B), ("PTX_BLK_ATT",))
        self.assertEqual(ablation.switches("trimul_core_exact"), ("PTX_TRIMUL", "FPF_OPS"))
        self.assertEqual(ablation.switches("cache_release"), ())


class TestValidate(unittest.TestCase):
    def test_member_lever_passes(self):
        self.assertEqual(ablation.validate("fast", [K2B]), [K2B])
        self.assertEqual(ablation.validate("exact", ["trimul_core_exact", "pad8"]), ["trimul_core_exact", "pad8"])
        self.assertEqual(ablation.validate("big", [K2B, "guard_lift"]), [K2B, "guard_lift"])
        self.assertEqual(ablation.validate("fast", []), [])

    def test_unknown_name_refused_by_name(self):
        with self.assertRaises(ablation.AblationError) as cm:
            ablation.validate("fast", ["bogus_lever"])
        self.assertIn("MODEL_OPT_LEVERS_OFF refused", str(cm.exception)); self.assertIn("bogus_lever: not a lever of this kit", str(cm.exception))

    def test_lever_outside_the_mode_refused_by_name(self):
        with self.assertRaises(ablation.AblationError) as cm:
            ablation.validate("fast", ["trimul_core_exact"])            # exact's TriMul is not in fast's set
        self.assertIn("trimul_core_exact: not in mode fast's lever set", str(cm.exception))
        with self.assertRaises(ablation.AblationError):
            ablation.validate("exact", [K2B])

    def test_switchless_big_memory_lever_refused_by_name(self):
        with self.assertRaises(ablation.AblationError) as cm:
            ablation.validate("big", ["cache_release"])
        self.assertIn("cache_release: applied by the big line without a switch of its own", str(cm.exception))

    def test_mode_off_refuses_any_name(self):
        with self.assertRaises(ablation.AblationError) as cm:
            ablation.validate("off", [K2B])
        self.assertIn("mode off applies no lever", str(cm.exception))

    def test_emptying_the_mode_is_refused(self):
        every = [n for n in modes.MODES["fast"] if ablation.switches(n)]
        with self.assertRaises(ablation.AblationError) as cm:
            ablation.validate("fast", every)
        self.assertIn("removes every lever of mode fast", str(cm.exception))


class TestApply(unittest.TestCase):
    def test_switches_leave_exports_and_join_unsets(self):
        res = modes.Resolution(mode="fast", exports={"PTX_BLK_ATT": "k2b", "PTX_BLK": "2", "PTX_E_PAD8": "1"}, unsets=["FPF_TRIMUL_CONTRACT"],
                               extras={"PTX_E_PAD8": "1"}, pre_exports={"PTX_T_TRIMUL": "v4"})
        removed = ablation.apply(res, [K2B, "pad8"])
        self.assertEqual(removed, {K2B: ["PTX_BLK_ATT"], "pad8": ["PTX_E_PAD8", "PTX_E_PAD8_MIN_TOKENS"]})
        self.assertEqual(res.exports, {"PTX_BLK": "2"}); self.assertEqual(res.extras, {}); self.assertEqual(res.pre_exports, {"PTX_T_TRIMUL": "v4"})
        self.assertEqual(res.unsets, ["FPF_TRIMUL_CONTRACT", "PTX_BLK_ATT", "PTX_E_PAD8", "PTX_E_PAD8_MIN_TOKENS"])
        self.assertNotIn("PTX_BLK_ATT", res.final({"PTX_BLK_ATT": "cudac", "HOME": "/x"}), "a caller's pre-set switch is unset too")

    def test_leaks_read_the_kits_own_evidence(self):
        fam, good = stack.MARKERS[K2B]
        self.assertEqual(ablation.leaks([K2B], [f"{fam}{good[0]}"], {}), {K2B: f"{fam}{good[0]}"})
        self.assertEqual(ablation.leaks([K2B], [f"{fam}unavailable(x)"], {}), {})
        self.assertEqual(ablation.leaks(["lever_report"], [], {"PTX_LEVER_REPORT": "/tmp/x.jsonl"}), {"lever_report": "switch still set: PTX_LEVER_REPORT=/tmp/x.jsonl"})
        self.assertEqual(ablation.leaks(["lever_report"], [], {}), {})


class TestLines(unittest.TestCase):
    def _rep(self, ablated=()):
        on = [n for n in modes.MODES["fast"] if n not in ablated]
        return {"active": True, "mode": "fast", "n_gpu": 1, "protenix_version": "2.0.0", "gpu": dict(H100), "levers_applied": on,
                "levers_fallback": [], "levers_ablated": list(ablated), "reconciled": {"records": ["clisampler"], "moves": {}}}

    def test_active_and_final_lines_gain_the_token_only_when_ablated(self):
        plain, abl = self._rep(), self._rep([K2B])
        self.assertNotIn("ablated=", report.activation_line(plain)); self.assertNotIn("ablated=", report.final_line(plain))
        self.assertTrue(report.activation_line(abl).endswith(f" fallbacks=none ablated={K2B}"), report.activation_line(abl))
        self.assertIn(f" fallbacks=none ablated={K2B} partial=false ", report.final_line(abl))
        self.assertEqual(report.activation_line(abl).replace(f" ablated={K2B}", "").replace(f"{K2B},", "").replace(f",{K2B}", ""),
                         report.activation_line(plain).replace(f"{K2B},", "").replace(f",{K2B}", ""), "the token is the only change to the line's grammar")

    def test_lever_line_of_an_ablated_lever_reads_off_ablated(self):
        lines = report.lever_lines(self._rep([K2B]))
        k2b = [l for l in lines if l.endswith(f" lever={K2B}")]
        self.assertEqual(len(k2b), 1); self.assertIn(" state=off ", k2b[0]); self.assertIn(" reason=ablated ", k2b[0])
        others = [l for l in lines if " lever=" in l and not l.endswith(f" lever={K2B}") and l.rsplit(" lever=", 1)[1] in modes.MODES["fast"]]
        self.assertTrue(others) ; self.assertTrue(all(" state=on " in l for l in others), [l for l in others if " state=on " not in l])
        self.assertFalse([l for l in report.lever_lines(self._rep()) if "reason=ablated" in l], "no ablation, no ablated line")


class TestDryRun(_Base):
    """The core's dry run (run.sh check) end to end on a mocked H100 row: resolve, validate the ablation by name, remove the switches, report."""

    def test_unset_is_the_mode_byte_for_byte(self):
        r, err = _dry_run("fast")
        self.assertNotIn("levers_ablated", r); self.assertIn(NATIVE, r["levers_applied"]); self.assertEqual(r["env"]["PTX_BLK_ATT"], "ptx_native_core:attn")   # H100: the row's attention word (kit 0.3.36: native)
        self.assertIn("[protenix-opt] DRY-RUN mode=fast", err); self.assertNotIn("ablated=", err)

    def test_ablating_k2b_on_h100_is_refused_by_name(self):
        """cc 9.0: k2b_flash_triattention is not part of the arm (triattn_native serves its slot): its ablation there is refused by name."""
        r, err = _dry_run("fast", K2B)
        self.assertFalse(r["active"]); self.assertIn(f"{K2B}: not part of the arm on kernel key 9.0|3.7 (its slot is served by triattn_native there — ablate triattn_native instead)", r["reason"])
        self.assertIn("NOT ACTIVE", err)

    def test_ablating_triattn_native_in_fast(self):
        """The A arm of triattn_native's A/B (protenix_opt 0.3.51): the displaced lever k2b_flash_triattention has no PTX_T_ATT word of its own, so env.sh is
        sourced again WITHOUT the row's word -> its own default PTX_BLK_ATT=k2b (that lever); only the attention words change in the exports."""
        plain, perr = _dry_run("fast")
        r, err = _dry_run("fast", NATIVE)
        self.assertEqual(r["levers_ablated"], [NATIVE]); self.assertEqual(r["ablated_switches"], {NATIVE: ["PTX_T_ATT"]})
        self.assertEqual(r["ablation_restored"], {})
        self.assertNotIn(NATIVE, r["levers_applied"]); self.assertNotIn(NATIVE, r["levers_fallback"]); self.assertIn(K2B, r["levers_applied"])
        self.assertEqual(r["env"]["PTX_BLK_ATT"], "k2b"); self.assertNotIn("PTX_T_ATT", r["env"])
        self.assertNotIn("PTX_BLK_ATT_PROVIDER_MIN_TOKENS", r["env"]); self.assertNotIn("PTX_NATIVE_MIN_TOKENS", r["env"])
        self.assertEqual([n for n in plain["levers_applied"] if n != NATIVE], [n for n in r["levers_applied"] if n != K2B], "the ablated lever leaves the plan, the slot's keeper joins it")
        att = ("PTX_BLK_ATT", "PTX_T_ATT", "PTX_BLK_ATT_PROVIDER_MIN_TOKENS")
        self.assertEqual({k: v for k, v in plain["env"].items() if k not in att}, {k: v for k, v in r["env"].items() if k not in att}, "only the attention words change in the exports")
        line = [l for l in err.splitlines() if l.startswith("[protenix-opt] DRY-RUN mode=fast")]
        self.assertEqual(len(line), 1); self.assertIn(f" ablated={NATIVE}", line[0]); self.assertIn(K2B, line[0])

    def test_a_callers_preset_switch_is_unset_too(self):
        with mock.patch.dict(os.environ, {"PTX_GLUE_V2": "1"}):
            r, _ = _dry_run("fast", "glue_v2")
        self.assertIn("PTX_GLUE_V2", r["unset"]); self.assertNotIn("PTX_GLUE_V2", r["env"])

    def test_unknown_name_is_not_active_by_name(self):
        r, err = _dry_run("fast", "bogus_lever")
        self.assertFalse(r["active"]); self.assertTrue(r["refused"]); self.assertIn("bogus_lever: not a lever of this kit", r["reason"])
        self.assertIn("[protenix-opt] NOT ACTIVE: MODEL_OPT_LEVERS_OFF refused", err); self.assertNotIn("env", r, "refused before anything resolved is reported")

    def test_lever_outside_the_mode_is_not_active_by_name(self):
        r, err = _dry_run("fast", "trimul_core_exact")
        self.assertFalse(r["active"]); self.assertIn("trimul_core_exact: not in mode fast's lever set", r["reason"]); self.assertIn("NOT ACTIVE", err)

    def test_mode_off_with_a_request_is_refused(self):
        r, err = _dry_run("off", K2B)
        self.assertFalse(r["active"]); self.assertTrue(r["refused"]); self.assertIn("mode off applies no lever", r["reason"]); self.assertIn("NOT ACTIVE", err)

    def test_strict_raises(self):
        with mock.patch.dict(os.environ, {ablation.ENV: "bogus_lever"}), mock.patch.object(stack, "protenix_version", return_value="2.0.0"), \
             mock.patch.object(stack, "gpu_probe_smi", return_value=dict(H100)), redirect_stderr(io.StringIO()):
            with self.assertRaises(stack.ActivationError):
                stack.activate("fast", dry_run=True, strict=True)

    def test_row_levers_drop_the_ablated_names(self):
        self.assertNotIn(K2B, stack._row_levers("fast", {ablation.ENV: K2B}))
        self.assertIn(K2B, stack._row_levers("fast", {}))
        self.assertEqual(stack._row_levers("exact", {ablation.ENV: K2B}), stack._row_levers("exact", {}), "validation, not _row_levers, refuses a stranger")


if __name__ == "__main__":
    unittest.main()


class WordSelectedProviderLeaks(unittest.TestCase):
    """ablation.leaks: a word-selected marker lever (trimul_core_exact, PTX_TRIMUL=exact) shares its family marker (FPF:enabled(trimul_in,trimul_out))
    with the registry's other TriMul word: ablated, the switch no longer carries its word — not a leak whatever the marker says; still on its word
    with its provider module (src/ptx_trimul_routes.py) loaded — a leak; the marker without the module is not the lever's proof."""
    APPLIED = ["FPF:enabled(trimul_in,trimul_out)"]

    def test_ablated_word_lever_is_not_a_leak_when_the_switch_carries_another_word(self):
        self.assertEqual(ablation.leaks(["trimul_core_exact"], self.APPLIED, {"PTX_TRIMUL": "tier"}), {})
        self.assertEqual(ablation.leaks(["trimul_core_exact"], self.APPLIED, {}), {})

    def test_word_lever_still_on_its_word_with_its_module_loaded_is_a_leak(self):
        import sys, types
        name = "ptx_trimul_routes"; had = name in sys.modules; keep = sys.modules.get(name)
        try:
            sys.modules.pop(name, None)
            self.assertEqual(ablation.leaks(["trimul_core_exact"], self.APPLIED, {"PTX_TRIMUL": "exact"}), {})      # word still exact but the provider never loaded: the marker is not its proof
            sys.modules[name] = types.ModuleType(name)
            self.assertEqual(ablation.leaks(["trimul_core_exact"], self.APPLIED, {"PTX_TRIMUL": "exact"}), {"trimul_core_exact": self.APPLIED[0]})
        finally:
            sys.modules.pop(name, None)
            if had: sys.modules[name] = keep

