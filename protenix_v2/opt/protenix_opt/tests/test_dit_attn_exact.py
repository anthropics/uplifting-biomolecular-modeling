"""dit_attn_exact — the exact DiT attention kernel lever: registry row (EXACT, marker probe, extra + row_dependent, replaced_by dit_attn), membership
(MODES exact / fast / big; the cc-9.0 exact compositions export its switch, no fast row does: never two DiT attention providers), the marker
grammar the kit classifies by, the carried files, and the LEVER evidence off a stubbed unit report."""
import json
import os
import re
import sys
import types
import unittest
from unittest import mock

from protenix_opt import modes, registry, report, stack, tp

LEVER = "dit_attn_exact"
PKG = "protenix_fpf_dit_attn_exact"                       # the kit copy before protenix_opt 0.3.51 (deleted; PKG_DIR must not exist)
REC = "protenix_opt.apb_core"                             # the module whose report() holds the hook record now
HERE = os.path.dirname(os.path.abspath(__file__))
FPF = os.path.abspath(os.path.join(HERE, "..", "..", "forward", "flashpairformer"))
PKG_DIR = os.path.join(FPF, "third_party", PKG)


class TestRegistryAndRows(unittest.TestCase):
    def test_registry_row(self):
        lv = registry.LEVERS[LEVER]
        self.assertEqual((lv.tier, lv.probe, lv.env_keys, lv.extra, lv.row_dependent, lv.replaced_by), (registry.EXACT, "marker", ("PTX_DIT_ATTN_EXACT",), True, True, ("dit_attn",)))
        self.assertEqual(report.STRATEGY_IDS[LEVER], "LOCAL.protenix_v2.dit_attn_exact")
        self.assertEqual(report.IMPL[LEVER], ("opt_core.kernels.apb", "core"))                     # protenix_opt 0.3.51: apb_core.py binds the provider's dit_exact row by the tier word exact
        self.assertEqual(stack.MARKERS[LEVER], ("DITATTN:", ("DITATTN:on(",)))
        self.assertIn(LEVER, tp.TALLY)

    def test_membership_and_rows(self):
        for mode in ("exact", "fast", "big"):
            self.assertIn(LEVER, modes.MODES[mode], mode)
        self.assertIn(LEVER, modes.big_levers())
        for key, row in modes.README_ROWS.items():
            e_post, t = row["exact"]["post"], row["fast"]
            if key.startswith("9.0|"):
                self.assertEqual(e_post.get("PTX_DIT_ATTN_EXACT"), "1", key)                       # the exact composition of the cc-9.0 rows exports it
            else:
                self.assertNotIn("PTX_DIT_ATTN_EXACT", e_post, key)                                 # another card's row does not list it (never a refusal there)
            self.assertNotIn("PTX_DIT_ATTN_EXACT", {**t["pre"], **t["post"]}, key)                 # no fast row exports it: dit_attn serves the DiT site there
        self.assertNotIn("PTX_DIT_ATTN_EXACT", modes.OTHER_ROW["exact"]["post"])

    def test_env_sh_does_not_export_the_switch(self):
        env_sh = open(os.path.join(FPF, "env.sh"), encoding="utf-8").read()
        self.assertNotIn("PTX_DIT_ATTN_EXACT", env_sh, "row-dependent: exported by modes.README_ROWS on cc 9.0 only, never by env.sh (an A100 would refuse)")


class TestMarkers(unittest.TestCase):
    ON = ("DITATTN:on(opt_core.kernels.apb 0.5.111.0 word=exact: dit_exact v7 torch2.13.0-cu130-sm90 serves 12/12 DiT classes (S5 N256..2048 eager|graph) vouched on "
          "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1; loadcheck=3cases-bit-equal digests=match; NVIDIA H100 80GB HBM3 cc=9.0; EXACT-BITWISE vs mem-efficient SDPA fp32 D=48)")
    FLOOR = ("DITATTN:on(opt_core.kernels.apb 0.5.111.0 word=exact: exact word -> sdpa:auto on H100:torch9.9.9+cu130/3.7.1/cueq0.11.1 (no class vouched for dit_exact here): "
             "every call keeps the statement BY NAME; loadcheck=none(kernel not loaded); NVIDIA H100 80GB HBM3 cc=9.0; EXACT-BITWISE vs mem-efficient SDPA fp32 D=48)")

    def test_on_marker_classifies_on_and_failures_do_not(self):
        family, good = stack.MARKERS[LEVER]
        ok = lambda a: a.startswith(family) and any(m.casefold() in a.casefold() for m in good) and not any(b in a for b in stack.BAD)
        self.assertTrue(ok(self.ON))
        self.assertTrue(ok(self.FLOOR))                                                              # an unvouched stack: installed, the statement serves by name (stated) -- not a refusal
        self.assertFalse(ok("DITATTN:unavailable(RuntimeError('dit_attn_exact: binary sha256 mismatch'))"))
        self.assertFalse(ok("DITATTN:inactive(PTX_DIT_ATTN_EXACT unset)"))

    def test_the_binding_prints_this_grammar(self):
        from protenix_opt import apb_core as core
        src = open(core.__file__, encoding="utf-8").read()
        self.assertIn('f"DITATTN:on({FACE} {core_version()} word=exact: {served}; {check}; ', src)
        self.assertIn('raise RuntimeError(f"{EXACT_LEVER}: ', src)                                      # refusal by name
        self.assertIn('if os.environ.get("PTX_DIT_ATTN_EXACT", "0") != "1":\n        return None', src)   # inert without the switch (the hook only calls it under the switch)
        self.assertEqual((core.EXACT_LEVER, core.MARK["dit_attn_exact"], core.CELLS["dit_attn"], core.DIT_HEAD_DIM), ("dit_attn_exact", "DITATTN:", "dit_h16d48", 48))
        with mock.patch.dict(os.environ, {"PTX_DIT_ATTN_EXACT": "0"}):
            self.assertIsNone(core.apply())
        self.assertEqual(core.report()["unit"], "dit_attn_exact")
        self.assertFalse(core.report()["installed"])

    def test_the_hook_imports_the_binding(self):
        src = open(os.path.join(FPF, "src", "ptx_trunk2_levers.py"), encoding="utf-8").read()
        self.assertIn("from protenix_opt import apb_core as _dx", src)
        self.assertNotIn("import protenix_fpf_dit_attn_exact", src)


class TestFiles(unittest.TestCase):
    def test_the_kit_carries_no_copy_the_provider_row_does(self):
        self.assertFalse(os.path.exists(PKG_DIR), "protenix_opt 0.3.51: the kit copy is deleted; the provider's dit_exact row carries the kernel")
        from opt_core.kernels import apb as A
        self.assertIn("dit_exact", A.EXACT_ROWS)
        d = os.path.join(os.path.dirname(A.__file__), "dit_exact")
        for rel in ("__init__.py", "csrc/dit_attn_exact.cu", "prebuilt/torch2.13.0-cu130-sm90/dit_attn_exact.so", "prebuilt/torch2.13.0-cu130-sm90/manifest.json"):
            self.assertTrue(os.path.isfile(os.path.join(d, rel)), rel)
        man = json.load(open(os.path.join(d, "prebuilt", "torch2.13.0-cu130-sm90", "manifest.json")))
        for k in ("so_sha256", "source_sha256", "torch", "cuda", "kernel_version", "loadcheck_cases", "loadcheck_digests"):
            self.assertIn(k, man, k)
        self.assertEqual((man["loadcheck_cases"], len(man["loadcheck_digests"])), (3, 3), "three load-check cases, run in every process")
        self.assertIn("torch2.13.0-cu130-sm90", A.DIT_EXACT_ABIS)


class TestEvidence(unittest.TestCase):
    def test_unit_none_without_the_module(self):
        saved = sys.modules.pop(REC, None)
        try:
            self.assertEqual(report.dit_attn_exact_evidence(), [("unit", "none")])
        finally:
            if saved is not None:
                sys.modules[REC] = saved

    def test_pairs_off_a_stub_report(self):
        m = types.ModuleType(REC)
        m.report = lambda: {"unit": "dit_attn_exact", "calls": 30, "routes": {"kernel": 24, "head_dim": 6}, "installed": True, "loadcheck": {"cases": 3, "ok": True, "digest_match": True},
                            "row": "dit_exact", "stack": "H100:torch2.13.0+cu130/3.7.1/cueq0.11.1"}
        with mock.patch.dict(sys.modules, {REC: m}):
            self.assertEqual(report.dit_attn_exact_evidence(), [("installed", 1), ("calls", 30), ("kernel", 24), ("other", 6), ("loadcheck", "3cases-bit-equal"), ("row", "dit_exact"), ("word", "exact")])
        m.report = lambda: {"unit": "dit_attn_exact", "calls": 30, "routes": {"exact_word:sdpa:auto": 30}, "installed": True, "loadcheck": None, "row": "statement", "stack": "H100:torch9.9.9"}
        with mock.patch.dict(sys.modules, {REC: m}):                                              # an unvouched stack: installed, every call kept the statement by name
            self.assertEqual(report.dit_attn_exact_evidence(), [("installed", 1), ("calls", 30), ("kernel", 0), ("other", 30), ("loadcheck", "none"), ("row", "statement"), ("word", "exact")])
        m.report = lambda: {"installed": False}
        with mock.patch.dict(sys.modules, {REC: m}):
            self.assertEqual(report.dit_attn_exact_evidence(), [("unit", "none")])


if __name__ == "__main__":
    unittest.main()
