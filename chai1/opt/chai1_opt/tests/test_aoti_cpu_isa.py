"""chai1_fastln 0.3.13 / chai1_opt 0.4.33 — an ahead-of-time package whose C++ launcher was compiled (-march=native) for CPU instructions this host
lacks is REFUSED BY NAME at load (`cpu_isa_mismatch`) and the EAGER hoisted step serves the crop; a package the host's CPU covers takes the AOTI
route unchanged. CPU-only: the archive is a real zip carrying torch's `<hash>.wrapper_metadata.json` (`AOTI_CPU_ISA`), the host's word is declared
through `CHAI1_OPT_AOTI_HOST_ISA` (the hook that declares the host ISA by hand), torch's loader is replaced by a sentinel so the test proves WHICH route was
chosen without a GPU: refused → the loader is never reached; loadable → the loader is reached. The probe words (`cpu_isa.py`, stdlib) are checked on
synthetic /proc/cpuinfo flag sets (AMD Milan → AVX2; Ice Lake → AVX512 AVX512_VNNI; Sapphire Rapids → … AMX_TILE; Skylake-SP → AVX512), the
implication rule (package features <= host features; unknown vocabulary → equality; no recorded word → not judged), the ACTIVE token
(`modes.compile_word`: off:cpu_isa_mismatch | on:aoti) and the EXIT census rendering (`report.eager_tally_fields`)."""
import importlib.util
import json
import os
import sys
import tempfile
import unittest
import zipfile
from unittest import mock

from chai1_opt import modes, report, stack

torch = None
A = None          # chai1_fastln.aoti (needs torch: imported in TestLoadRoute.setUpClass)


def probe():
    """chai1_fastln/cpu_isa.py the way modes loads it (file spec, no package __init__, no torch)."""
    path = os.path.join(stack.dstep_home(), "chai1_fastln", "cpu_isa.py")
    spec = importlib.util.spec_from_file_location("_cpu_isa_under_test", path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def write_pt2(path, isa="AVX512 AVX512_VNNI", where="wrapper", extra=None):
    """A pt2-shaped zip: <stem>/data/aotinductor/model/<hash>.{wrapper,kernel}_metadata.json with torch's device-info keys (+ a dummy .so member)."""
    stem = os.path.basename(path)[:-4]
    meta = {"AOTI_DEVICE_KEY": "cuda", "AOTI_PLATFORM": "linux", "AOTI_MACHINE": "x86_64", "AOTI_COMPUTE_CAPABILITY": "90"}
    if isa is not None: meta["AOTI_CPU_ISA"] = isa
    if extra: meta.update(extra)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(f"{stem}/archive_format", "pt2")
        z.writestr(f"{stem}/data/aotinductor/model/cabc123.wrapper.so", b"\x7fELF-not-really")
        z.writestr(f"{stem}/data/aotinductor/model/cabc123.{where}_metadata.json", json.dumps(meta))


META_LAYOUTS = {"kinds": "iic", "consts_digest": "x", "constants": {}, "inputs": ["arg:atom_coords", "arg:noise_sigma"], "input_strides": [[15, 3, 1], [5, 1]], "input_dtypes": ["torch.float32"] * 2}

MILAN = "fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush mmx fxsr sse sse2 ht syscall nx mmxext fxsr_opt pdpe1gb rdtscp lm constant_tsc rep_good nopl nonstop_tsc cpuid extd_apicid aperfmperf rapl pni pclmulqdq monitor ssse3 fma cx16 pcid sse4_1 sse4_2 x2apic movbe popcnt aes xsave avx f16c rdrand lahf_lm cmp_legacy svm extapic cr8_legacy abm sse4a misalignsse 3dnowprefetch osvw ibs skinit wdt tce topoext perfctr_core perfctr_nb bpext perfctr_llc mwaitx cpb cat_l3 cdp_l3 invpcid_single hw_pstate ssbd mba ibrs ibpb stibp vmmcall fsgsbase bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a rdseed adx smap clflushopt clwb sha_ni xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local clzero irperf xsaveerptr rdpru wbnoinvd amd_ppin arat npt lbrv svm_lock nrip_save tsc_scale vmcb_clean flushbyasid decodeassists pausefilter pfthreshold v_vmsave_vmload vgif v_spec_ctrl umip pku ospke vaes vpclmulqdq rdpid overflow_recov succor smca"
ICELAKE = "fpu sse sse2 ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand abm 3dnowprefetch fsgsbase bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg avx512_vpopcntdq rdpid"
SAPPHIRE = ICELAKE + " avx512_bf16 amx_bf16 avx512_fp16 amx_tile amx_int8"
SKYLAKE = "fpu sse sse2 ssse3 fma cx16 sse4_1 sse4_2 movbe popcnt aes xsave avx f16c rdrand abm fsgsbase bmi1 avx2 smep bmi2 erms invpcid avx512f avx512dq rdseed adx smap clflushopt clwb avx512cd avx512bw avx512vl xsaveopt xsavec"


class TestProbeWords(unittest.TestCase):
    """cpu_isa.py — stdlib only (loaded from its file, as chai1_opt.modes loads it before any model exists)."""

    @classmethod
    def setUpClass(cls):
        cls.P = probe()

    def test_the_word_is_one_word_across_the_kit(self):
        self.assertEqual(self.P.WORD, "cpu_isa_mismatch"); self.assertEqual(modes.COMPILE_CPU_ISA_MISMATCH, self.P.WORD)
        self.assertEqual(self.P.ENV_HOST_ISA, "CHAI1_OPT_AOTI_HOST_ISA"); self.assertEqual(stack.ENV_AOTI_HOST_ISA, self.P.ENV_HOST_ISA)
        from chai1_opt import _autoload
        self.assertIn(self.P.ENV_HOST_ISA, stack.DECLARED_ENV); self.assertIn(self.P.ENV_HOST_ISA, _autoload.DECLARED_ENV); self.assertEqual(_autoload.undeclared({self.P.ENV_HOST_ISA: "AVX2"}), [])   # declared: the activation gate admits it

    def test_cpuinfo_words_are_torchs_words(self):
        P = self.P
        self.assertEqual(P.cpuinfo_isa(set(MILAN.split())), "AVX2")                               # AMD EPYC Milan (AWS H100 nodes): no AVX-512
        self.assertEqual(P.cpuinfo_isa(set(ICELAKE.split())), "AVX512 AVX512_VNNI")               # the reference build hosts
        self.assertEqual(P.cpuinfo_isa(set(SAPPHIRE.split())), "AVX512 AVX512_VNNI AMX_TILE")     # str(VecAMX()).upper()
        self.assertEqual(P.cpuinfo_isa(set(SKYLAKE.split())), "AVX512")                           # AVX-512 without VNNI
        self.assertEqual(P.cpuinfo_isa({"sse2"}), "INVALID_VEC_ISA")
        self.assertEqual(P.cpuinfo_flags("processor : 0\nflags\t\t: avx2 fma AVX512F\n\nprocessor : 1\nflags : sse\n"), {"avx2", "fma", "avx512f"})

    def test_loadable_is_package_features_within_host_features(self):
        L = self.P.loadable
        self.assertIs(L("AVX512 AVX512_VNNI", "AVX2"), False)                    # the mismatch case: an AVX-512 launcher on an AVX2-only host
        self.assertIs(L("AVX512 AVX512_VNNI", "AVX512 AVX512_VNNI"), True)       # the reference hosts
        self.assertIs(L("AVX512 AVX512_VNNI", "AVX512 AVX512_VNNI AMX_TILE"), True)   # a richer host runs it (torch would still warn: unequal strings)
        self.assertIs(L("AVX512 AVX512_VNNI", "AVX512"), False)                  # a host without VNNI lacks an instruction family the build had
        self.assertIs(L("AVX2", "AVX512"), True); self.assertIs(L("AVX512", "AVX2"), False)
        self.assertIs(L("avx512  avx512_vnni", "AVX512 AVX512_VNNI"), True)      # spacing / case as recorded vs as probed
        self.assertIsNone(L(None, "AVX2")); self.assertIsNone(L("", "AVX2"))      # no recorded word: not judged (served as before)
        self.assertIs(L("SVE256", "SVE256"), True); self.assertIs(L("SVE256", "AVX512"), False)   # outside the x86 vocabulary: torch's own rule (equality)
        self.assertIs(L("AVX512 AVX512_VNNI", "INVALID_VEC_ISA"), False)         # a host with no vector ISA torch knows runs no AVX-512 launcher

    def test_package_word_is_read_from_the_archive_and_the_host_word_can_be_declared(self):
        P = self.P
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "dstep_c256_s5_hoist2+dit_attn_0123456789ab.pt2")
            write_pt2(p, "AVX512 AVX512_VNNI"); self.assertEqual(P.package_isa(p), "AVX512 AVX512_VNNI")
            write_pt2(p, "avx2", where="kernel"); self.assertEqual(P.package_isa(p), "AVX2")        # a kernel_metadata.json alone carries it too
            write_pt2(p, None); self.assertIsNone(P.package_isa(p))                                    # a torch that recorded no ISA word
            open(p, "wb").write(b"not a zip"); self.assertIsNone(P.package_isa(p))
            write_pt2(p, "AVX512 AVX512_VNNI")
            v = P.verdict(p, {"CHAI1_OPT_AOTI_HOST_ISA": "avx2"})
            self.assertEqual((v["ok"], v["package"], v["host"], v["source"], v["word"]), (False, "AVX512 AVX512_VNNI", "AVX2", "env", "cpu_isa_mismatch"))
            self.assertIs(P.verdict(p, {"CHAI1_OPT_AOTI_HOST_ISA": "AVX512 AVX512_VNNI"})["ok"], True)
            w, src = P.host_isa({})                                                                     # no override: torch's word if torch is imported in this process, else cpuinfo's
            self.assertIn(src, ("torch", "cpuinfo")); self.assertTrue(w)

    def test_active_token_probes_the_archives_against_the_host(self):
        """modes.compile_word: loadable packages whose launchers this host cannot run -> off:cpu_isa_mismatch (each refused by name at its crop, the
        eager hoisted step serves); a host that covers them (or a package with no recorded word) -> on:aoti, unchanged; no_layout_meta keeps its own word."""
        from chai1_opt import jit
        rep = {"mode": "big", "levers_applied": ["tier1", "hoist2", "compiled", "dit_attn"]}
        with tempfile.TemporaryDirectory() as root:
            d = os.path.join(root, "torchX-cuY-h100", "aoti"); os.makedirs(d)
            base = {"MODEL_OPT_JIT_ROOT": root, jit.ENV_KEY: "torchX-cuY-h100"}
            name = "dstep_c512_s5_hoist2+dit_attn_0123456789ab"
            write_pt2(os.path.join(d, name + ".pt2"), "AVX512 AVX512_VNNI")
            open(os.path.join(d, name + ".meta.json"), "w").write(json.dumps(dict(META_LAYOUTS, name=name)))
            self.assertEqual(modes.aoti_packages_layout_state(base), "meta")
            avx2 = dict(base, CHAI1_OPT_AOTI_HOST_ISA="AVX2"); same = dict(base, CHAI1_OPT_AOTI_HOST_ISA="AVX512 AVX512_VNNI"); amx = dict(base, CHAI1_OPT_AOTI_HOST_ISA="AVX512 AVX512_VNNI AMX_TILE")
            self.assertEqual(modes.aoti_packages_isa_state(avx2), "cpu_isa_mismatch"); self.assertEqual(modes.compile_word(rep, environ=avx2), "off:cpu_isa_mismatch")
            self.assertEqual(modes.aoti_packages_isa_state(same), "ok"); self.assertEqual(modes.compile_word(rep, environ=same), "on:aoti")
            self.assertEqual(modes.compile_word(rep, environ=amx), "on:aoti")
            self.assertEqual(modes.compile_word(rep, environ=dict(avx2, MODEL_OPT_TARGET_GPU="A100-SXM4-80GB")), "off:cpu_isa_mismatch")   # both cards: the word is the host CPU's
            write_pt2(os.path.join(d, name + ".pt2"), None)                                                       # no recorded word: not judged -> served
            self.assertEqual(modes.aoti_packages_isa_state(avx2), "ok"); self.assertEqual(modes.compile_word(rep, environ=avx2), "on:aoti")
            open(os.path.join(d, name + ".meta.json"), "w").write(json.dumps({"name": name, "kinds": "iic"}))     # no_layout_meta comes first
            write_pt2(os.path.join(d, name + ".pt2"), "AVX512 AVX512_VNNI")
            self.assertIsNone(modes.aoti_packages_isa_state(avx2)); self.assertEqual(modes.compile_word(rep, environ=avx2), "off:no_layout_meta")
        self.assertEqual(modes.compile_word(rep, environ={"CHAI1_OPT_AOTI_HOST_ISA": "AVX2"}), "on:dynamo")           # no JIT root: nothing to judge

    def test_exit_census_renders_the_word(self):
        """report.eager_tally_fields: aoti=0/<n>(refused:cpu_isa_mismatch:<n>) compiled=stepped_aside:cpu_isa_mismatch — the same shape as no_layout_meta's."""
        stats = {"diffusion_events": [(256, "aoti_refused:cpu_isa_mismatch", 0.0), (256, "compile_aside:cpu_isa_mismatch", 0.0)],
                 "dstep_aoti": {"n_loaded": 0, "n_asked": 2, "refused": ["dstep_c256_s5_hoist2+dit_attn_6ea93441912b:cpu_isa_mismatch", "dstep_c512_s5_hoist2+dit_attn_1bd606867918:cpu_isa_mismatch"],
                                "n_aside": 2, "aside_reason": "cpu_isa_mismatch", "errors": [], "mismatch": [], "n_realigned": 0}}
        f = report.eager_tally_fields(stats, "tier1", items=1, ok=1)
        self.assertIn("aoti=0/2(refused:cpu_isa_mismatch:2)", f); self.assertIn("compiled=stepped_aside:cpu_isa_mismatch", f)


class TestLoadRoute(unittest.TestCase):
    """aoti.Package / aoti.Dispatch with torch (CPU): the verdict acts BEFORE torch's loader — refused → the loader is never reached and the eager
    hoisted step serves by name; loadable → the loader is reached (here a sentinel; on the box the package serves)."""

    @classmethod
    def setUpClass(cls):
        global torch, A
        try:
            import torch as _t
        except ImportError as e:
            raise unittest.SkipTest(f"the load-route checks need torch (CPU is enough): {e}")
        torch = _t
        for home in (stack.eager_home(), stack.dstep_home()):
            home in sys.path or sys.path.insert(0, home)
        from chai1_fastln import aoti as _aoti
        A = _aoti

    def _hf(self, crop):
        """The hoisted step's surface aoti reads (test_aoti.FakeHF's shape) at a crop of its own: aoti's step-aside is said once per (crop, n_samples)
        per PROCESS (module STATS), so each check here takes a crop no other test in the suite uses."""
        class FakeHF:
            method = f"forward_{crop}"
            def __init__(self):
                self.argnames = ["atom_coords", "noise_sigma", "mask"]
                self.cache = {"pair": torch.zeros(2, 3), "n_blocks": 4, "shape0": torch.tensor(7), "biases": [torch.ones(2), None], "flag": None}
                self.src = "def _step(root, atom_coords, noise_sigma, mask, cache):\n    return atom_coords * 2\n"
                self.root = torch.nn.Linear(1, 1)
                self.compiled, self._compile_mode, self.stats = "default", "default", {}
                self._step = lambda root, *rest: ("eager", len(rest))
        return FakeHF()

    def _package_on_disk(self, d, hf, isa):
        name = A.package_name(A.crop_of(hf), 5, "hoist2", A.step_digest(hf))
        write_pt2(os.path.join(d, name), isa)
        open(os.path.join(d, name[:-4] + ".meta.json"), "w").write(json.dumps(dict(META_LAYOUTS, name=name[:-4])))
        return name

    def test_a_package_built_for_instructions_this_host_lacks_is_refused_by_name_before_the_loader(self):
        hf = self._hf(1088)
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"CHAI1_OPT_AOTI_HOST_ISA": "AVX2"}), \
             mock.patch("torch._inductor.aoti_load_package", side_effect=AssertionError("the loader must not be reached for a refused package")) as loader:
            name = self._package_on_disk(d, hf, "AVX512 AVX512_VNNI")
            self.assertEqual(A.ASIDE_ISA, "cpu_isa_mismatch"); self.assertEqual(A.cpu_isa.WORD, modes.COMPILE_CPU_ISA_MISMATCH)
            n_isa = len(A.STATS["isa"])
            with self.assertRaises(A.Refused) as cm:
                A.Package(hf, os.path.join(d, name))
            self.assertEqual((cm.exception.word, cm.exception.package), ("cpu_isa_mismatch", name[:-4])); loader.assert_not_called()
            self.assertTrue(A.STATS["isa"][n_isa].endswith(":AVX512 AVX512_VNNI|AVX2|env|lacking"), A.STATS["isa"][n_isa:])
            # through the Dispatch: counted, said, the EAGER hoisted step serves (no Inductor build), the wrapper carries the aside word for the EXIT line
            built, events, n0 = [], [], len(A.STATS["refused"])
            class W: compile_aside = None
            w = W()
            disp = A.Dispatch(hf, d, "hoist2", lambda: built.append(1) or (lambda root, *rest: ("dynamo", len(rest))), events=events, wrapper=w, eager=lambda root, *rest: ("eager", len(rest)))
            out = disp(hf.root, torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool), hf.cache)
            self.assertEqual(out[0], "eager"); self.assertIsNone(disp.pkg); self.assertEqual(built, []); loader.assert_not_called()
            self.assertEqual(w.compile_aside, "cpu_isa_mismatch"); self.assertEqual(A.STATS["aside_reason"], "cpu_isa_mismatch")
            self.assertEqual(A.STATS["refused"][n0:], [f"{name[:-4]}:cpu_isa_mismatch"])
            self.assertTrue(any(e == "aoti_refused:cpu_isa_mismatch" for _, e, _ in events), events); self.assertTrue(any(e == "compile_aside:cpu_isa_mismatch" for _, e, _ in events), events)
            out2 = disp(hf.root, torch.zeros(1, 5, 3), torch.ones(1, 5), torch.zeros(4, dtype=torch.bool), hf.cache)     # later steps: the eager step, nothing re-tried
            self.assertEqual(out2[0], "eager"); loader.assert_not_called()
            s = A.report(); self.assertIn(f"{name[:-4]}:cpu_isa_mismatch", s["refused"]); self.assertTrue(any(x.endswith("|lacking") for x in s["isa"]))
            mine = dict(s, refused=[r for r in s["refused"] if r.startswith(name[:-4])], n_asked=1, n_loaded=0, n_realigned=0, errors=[], mismatch=[])   # this test's package only (STATS is the process's)
            f = report.eager_tally_fields({"diffusion_events": events, "dstep_aoti": mine}, "tier1", items=1, ok=1)
            self.assertIn("aoti=0/1(refused:cpu_isa_mismatch:1)", f); self.assertIn("compiled=stepped_aside:cpu_isa_mismatch", f)

    def test_a_package_the_host_covers_takes_the_aoti_route(self):
        """Equal words (the reference hosts) and a richer host: the verdict passes and torch's loader IS reached — the route is unchanged there."""
        class Reached(RuntimeError): pass
        for crop, host in ((1152, "AVX512 AVX512_VNNI"), (1216, "AVX512 AVX512_VNNI AMX_TILE")):
            hf = self._hf(crop)
            with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"CHAI1_OPT_AOTI_HOST_ISA": host}), \
                 mock.patch("torch._inductor.aoti_load_package", side_effect=Reached("loader reached")) as loader:
                name = self._package_on_disk(d, hf, "AVX512 AVX512_VNNI")
                n_isa = len(A.STATS["isa"])
                with self.assertRaises(Reached):
                    A.Package(hf, os.path.join(d, name))
                loader.assert_called_once(); self.assertTrue(A.STATS["isa"][n_isa].endswith(f"|{host}|env|ok"), A.STATS["isa"][n_isa:])
        hf = self._hf(1280)                                                                               # a package with no recorded word: not judged, loader reached
        with tempfile.TemporaryDirectory() as d, mock.patch.dict(os.environ, {"CHAI1_OPT_AOTI_HOST_ISA": "AVX2"}), \
             mock.patch("torch._inductor.aoti_load_package", side_effect=Reached("loader reached")) as loader:
            name = self._package_on_disk(d, hf, None)
            with self.assertRaises(Reached):
                A.Package(hf, os.path.join(d, name))
            loader.assert_called_once(); self.assertTrue(A.STATS["isa"][-1].endswith("|unrecorded"))

    def test_the_host_word_without_an_override_is_torchs_own_expression(self):
        w, src = A.cpu_isa.host_isa({})
        self.assertEqual(src, "torch")
        from torch._inductor import cpu_vec_isa as V
        self.assertEqual(w, " ".join(str(V.pick_vec_isa()).split()).upper())          # == codecache.get_device_information(...)["AOTI_CPU_ISA"], the word _load_aoti compares


if __name__ == "__main__":
    unittest.main()
