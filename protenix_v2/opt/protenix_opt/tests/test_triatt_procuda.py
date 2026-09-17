"""triatt_prologue_cuda — the CUDA tri-attention prologue: registry row (EXACT, marker probe, extra + row_dependent), membership (exact / fast / big; the
cc-9.0 README rows export the switch in the exact AND fast compositions, never env.sh), the marker grammar (unavailable / REFUSED -> the mode refuses
by name), the hook order (after MK-PF and headsplit, before the block path import), the LEVER evidence + EXIT token off the live tally, the carried
files, and the package surface (reads its switch, the kit's PTX_BLK / FPF_MKPF_LN words and the lever-report path only)."""
import json
import os
import re
import sys
import types
import unittest
from unittest import mock

from protenix_opt import modes, registry, report, stack, tp
from protenix_opt.tests import test_enable_gates as _teg

LEVER = "triatt_prologue_cuda"
PKG = "protenix_fpf_triatt_procuda"
HERE = os.path.dirname(os.path.abspath(__file__))
FPF = os.path.abspath(os.path.join(HERE, "..", "..", "forward", "flashpairformer"))
PKG_DIR = os.path.join(FPF, "third_party", PKG)
ON = "PROCUDA:on(sm90a,so=e9009ba57e,sites=f1+f1hm)"
TALLY = {"cuda": 1045, "rm": 0, "hm": 1045, "pad": 1045, "pass_shape": 0, "pass_wx": 0, "chk_eq": 1, "chk_ne": 0, "chk_defer": 0, "chk_pending": 0}


class TestRowAndRows(unittest.TestCase):
    def test_registry_row_and_ids(self):
        lv = registry.LEVERS[LEVER]
        self.assertEqual((lv.tier, lv.probe, lv.env_keys, lv.extra, lv.row_dependent), (registry.EXACT, "marker", ("PTX_TRIATT_PROCUDA",), True, True))
        self.assertEqual(stack.MARKERS[LEVER], ("PROCUDA:", ("PROCUDA:on(",)))
        self.assertEqual(report.IMPL[LEVER], ("forward/flashpairformer/third_party/protenix_fpf_triatt_procuda", "kit")); self.assertIn(LEVER, report.STRATEGY_IDS); self.assertIn(LEVER, tp.TALLY)
        self.assertIn("procuda", report.TALLY_KEYS)

    def test_membership_and_rows(self):
        for mode in ("exact", "fast", "big"):
            self.assertIn(LEVER, modes.MODES[mode], mode)
        for key, row in modes.README_ROWS.items():
            for arm in ("exact", "fast"):
                got = row[arm]["post"].get("PTX_TRIATT_PROCUDA")
                self.assertEqual(got, "1" if key.startswith("9.0|") else None, (key, arm))
        self.assertNotIn("PTX_TRIATT_PROCUDA", open(os.path.join(FPF, "env.sh"), encoding="utf-8").read())

    def test_hook_order(self):
        src = open(os.path.join(FPF, "src", "ptx_trunk2_levers.py"), encoding="utf-8").read()
        body = src[src.index("def _apply_blk2"):]
        i_mk, i_pc, i_pf = body.index('os.environ.get("PTX_MK_PF"'), body.index("import protenix_fpf_triatt_procuda as _PC"), body.index("import protenix.model.modules.pairformer as PF")
        self.assertLess(i_mk, i_pc); self.assertLess(i_pc, i_pf); self.assertNotIn('os.environ.get("PTX_TRIATT_HEADSPLIT"', body)   # protenix_opt 0.3.51: the head split is the shared core's exact_headsplit row (no kit hook)
        self.assertIn('f"PROCUDA:unavailable({_e!r})"', body)


class TestMarkers(unittest.TestCase):
    tc = _teg.TestClassify()
    FAST = ["T1:fused", _teg.TestClassify.TRIATTN_LINE, "NM:1", "PWAZ:True", "DEADSKIP:hooked(load_checkpoint)",
            "v02:GRAPH:installed", "FPF:enabled(trimul_in,trimul_out)", "TEMPL_DEDUPE:1", "SAMPLER:hook(meta_path)", "LAZY_INIT:1"]

    def test_grammar(self):
        family, good = stack.MARKERS[LEVER]
        ok = lambda a: a.startswith(family) and any(m.casefold() in a.casefold() for m in good) and not any(b in a for b in stack.BAD)
        self.assertTrue(ok(ON)); self.assertTrue(ok("PROCUDA:on(sm90a,so=e9009ba57e,sites=f1)"))
        for bad in ("PROCUDA:unavailable(Refused('triatt_prologue_cuda: MK-PF F1 is not installed on the prologue'))", "PROCUDA:refused(x)"):
            self.assertFalse(ok(bad), bad)

    def test_unavailable_is_a_fallback_in_exact_and_fast_on_9_0(self):
        bad = "PROCUDA:unavailable(Refused('triatt_prologue_cuda: cc 8.0 has no binary'))"
        for mode, howto in (("exact", list(self.tc.HOWTO_E)), ("fast", list(self.FAST))):
            marks = [m for m in howto if not m.startswith("PROCUDA:")] + [bad]
            with mock.patch.dict(sys.modules, {"fpf_trimul_exact": types.ModuleType("fpf_trimul_exact")}):
                on, fb, skipped, why = self.tc._classify(mode, marks, self.tc._env(mode))
            self.assertIn(LEVER, fb, mode); self.assertIn("unavailable(", why[LEVER])
            with mock.patch.dict(sys.modules, {"fpf_trimul_exact": types.ModuleType("fpf_trimul_exact")}):
                on, fb, skipped, why = self.tc._classify(mode, [m for m in howto if not m.startswith("PROCUDA:")] + [ON], self.tc._env(mode))
            self.assertIn(LEVER, on, mode)

    def test_the_package_surface(self):
        src = open(os.path.join(PKG_DIR, "__init__.py"), encoding="utf-8").read()
        self.assertRegex(src, r'(?m)^__version__ = "0\.9\.1"'); self.assertIn('LEV._STATS["applied"].append(marker()); LEV._STATS["procuda"] = _T', src)
        reads = sorted(set(re.findall(r"""environ(?:\.get)?\s*[\[(]\s*["']([A-Z][A-Z0-9_]+)["']""", src)) | ({"PTX_TRIATT_PROCUDA"} if 'SWITCH = "PTX_TRIATT_PROCUDA"' in src else set()))
        self.assertEqual([k for k in reads if k not in ("PTX_TRIATT_PROCUDA", "PTX_BLK", "FPF_MKPF_LN", "PTX_LEVER_REPORT")], [], reads)


class TestFilesAndEvidence(unittest.TestCase):
    def test_carried_files_and_manifest(self):
        for rel in ("__init__.py", "binding.py", "build.py", "csrc/triatt_procuda_sm90.cu", "prebuilt/libtriatt_procuda_sm90a.so", "prebuilt/manifest.json", "prebuilt/NOTICE.md"):
            self.assertTrue(os.path.isfile(os.path.join(PKG_DIR, rel)), rel)
        man = json.load(open(os.path.join(PKG_DIR, "prebuilt", "manifest.json")))
        self.assertEqual((man["arch"], man["so_sha256"][:10], man["src_sha256"][:8]), ("sm_90a", "e9009ba57e", "fc120646"))

    def test_evidence_and_exit_token(self):
        saved = sys.modules.get("ptx_trunk2_levers")
        try:
            sys.modules.pop("ptx_trunk2_levers", None)
            self.assertEqual(report.procuda_evidence(), [("tally", "none")])
            m = types.ModuleType("ptx_trunk2_levers"); m._STATS = {"procuda": dict(TALLY)}
            with mock.patch.dict(sys.modules, {"ptx_trunk2_levers": m}):
                self.assertEqual(report.procuda_evidence(), [(k, TALLY[k]) for k in report.PROCUDA_TALLY])
        finally:
            if saved is not None:
                sys.modules["ptx_trunk2_levers"] = saved
        fields = report.tally_fields({"applied": [ON], "procuda": dict(TALLY)})
        tok = [f for f in fields if f.startswith("procuda=")]
        self.assertEqual(tok, ["procuda=cuda:1045,rm:0,hm:1045,pad:1045,pass_shape:0,pass_wx:0,chk_eq:1,chk_ne:0,chk_defer:0,chk_pending:0"]); self.assertTrue(all(" " not in f for f in fields))


if __name__ == "__main__":
    unittest.main()
