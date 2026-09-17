"""enable(): "off" is a no-op, the version gate refuses anything but protenix 2.0.0 (PROTENIX_OPT_FORCE=1 overrides with a
warning), the GPU gate refuses without CUDA, strict mode raises, the kit's sitecustomize makes protenix_opt stand down, and
the activation report keeps total accounting (every lever of a mode lands in exactly one list)."""
import io
import os
import sys
import types
import unittest
from contextlib import redirect_stderr
from unittest import mock

import protenix_opt
from protenix_opt import ActivationError, stack
from opt_core import instances
from protenix_opt.modes import MODES


class _Base(unittest.TestCase):
    def setUp(self):
        stack._REPORT = None
        self._env = dict(os.environ)
        self._meta = list(sys.meta_path)                                  # an activation installs finders (the instance counter, the kernel routes)
        os.environ.pop("PROTENIX_OPT", None)
        os.environ.pop("PROTENIX_OPT_FORCE", None)

    def tearDown(self):
        stack._REPORT = None
        instances._COUNTERS.clear()
        sys.meta_path[:] = self._meta
        os.environ.clear()
        os.environ.update(self._env)


class TestOff(_Base):
    def test_off_is_a_noop(self):
        env, mods = dict(os.environ), set(sys.modules)
        err = io.StringIO()
        with redirect_stderr(err):
            r = protenix_opt.enable("off")
        self.assertFalse(r["active"]); self.assertEqual(r["mode"], "off"); self.assertIn("stock", r["reason"])
        self.assertEqual(dict(os.environ), env, "off must not touch the environment")
        self.assertFalse({"torch", "protenix", "ptx_trunk2_levers"} & (set(sys.modules) - mods), "off must not import the stack")
        self.assertEqual(err.getvalue(), "")
        self.assertEqual(protenix_opt.status()["mode"], "off")
        self.assertEqual(protenix_opt.enable("off"), r)

    def test_unknown_mode_is_an_error(self):
        with self.assertRaises(ValueError):
            protenix_opt.enable("turbo")

    def test_status_before_enable(self):
        s = protenix_opt.status()
        self.assertFalse(s["active"]); self.assertIn("reason", s)


class TestVersionGate(_Base):
    def test_refuses_on_a_fake_version(self):
        with mock.patch.object(stack, "protenix_version", return_value="2.1.0"):
            err = io.StringIO()
            with redirect_stderr(err):
                r = protenix_opt.enable("exact")
        self.assertFalse(r["active"]); self.assertIn("2.1.0", r["reason"]); self.assertIn("2.0.0", r["reason"])
        self.assertEqual(r["protenix_version"], "2.1.0")
        self.assertIn("[protenix-opt] NOT ACTIVE: protenix==2.1.0", err.getvalue())
        self.assertNotIn("PTX_BLK", os.environ, "a refused activation sets nothing")

    def test_refuses_when_not_installed(self):
        with mock.patch.object(stack, "protenix_version", return_value=None):
            with redirect_stderr(io.StringIO()):
                r = protenix_opt.enable("fast")
        self.assertFalse(r["active"]); self.assertIn("not installed", r["reason"])

    def test_strict_raises(self):
        with mock.patch.object(stack, "protenix_version", return_value="2.1.0"), redirect_stderr(io.StringIO()):
            with self.assertRaises(ActivationError):
                protenix_opt.enable("exact", strict=True)

    def test_force_overrides_with_a_warning(self):
        os.environ["PROTENIX_OPT_FORCE"] = "1"
        err = io.StringIO()
        with mock.patch.object(stack, "protenix_version", return_value="2.1.0"), \
             mock.patch.object(stack, "gpu_probe_smi", return_value={"name": None, "sm": None, "cc": None, "probe": "stub"}), \
             mock.patch.object(stack, "resolve", side_effect=RuntimeError("env.sh stub")), redirect_stderr(err):
            r = protenix_opt.enable("exact")
        self.assertIn("WARNING", err.getvalue()); self.assertIn("2.1.0", err.getvalue())
        self.assertFalse(r["active"]); self.assertIn("env.sh could not be sourced", r["reason"], "passed the version and GPU gates (forced), stopped at the next one")

    def test_accepts_the_pin(self):
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"):
            self.assertEqual(stack.version_gate(), (True, "2.0.0", None))


class TestGpuGate(_Base):
    def test_refuses_without_cuda(self):
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), \
             mock.patch.object(stack, "gpu_probe_smi", return_value={"name": None, "sm": None, "cc": None, "probe": "nvidia-smi absent"}), redirect_stderr(io.StringIO()):
            r = protenix_opt.enable("exact")
        self.assertFalse(r["active"]); self.assertIn("no CUDA device", r["reason"])
        self.assertNotIn("PTX_BLK", os.environ)

    def test_dry_run_resolves_and_gates_without_applying(self):
        """check: stack.activate(dry_run=True) sources env.sh, gates, reports the planned levers — exports nothing, applies nothing."""
        from protenix_opt import modes
        err = io.StringIO()
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(modes, "triton_version", return_value="3.3.1"), \
             mock.patch.object(stack, "gpu_probe_smi", return_value={"name": "NVIDIA H100 80GB HBM3", "sm": "sm90", "cc": "9.0", "probe": "nvidia-smi"}), \
             redirect_stderr(err):
            r = stack.activate("exact", dry_run=True)
        self.assertTrue(r["dry_run"]); self.assertFalse(r["active"])
        self.assertIn("PTX_BLK", r["env"]); self.assertNotIn("PTX_BLK", os.environ)
        self.assertEqual(r["row_key"], "9.0|3.3"); self.assertEqual(r["levers_not_in_arm"], ["mk_pf"])
        self.assertIn("ptx_trimul_routes:trimul_out", r["env"]["FPF_OPS"]); self.assertEqual(r["env"]["PTX_TRIMUL"], "exact")   # protenix_opt 0.3.51: the exact TriMul is the shared core's exact tier word on every card (no row word)
        self.assertEqual(r["extras"]["PTX_E_PAD8"], "1"); self.assertIn("trimul_core_exact", r["levers_applied"]); self.assertNotIn("levers_external", r)
        self.assertEqual(r["kit_version"], stack.kit_version()); self.assertNotIn("base_kit_version", r)
        self.assertNotIn("kits", r)                                   # no per-unit file account in the report
        self.assertEqual([p for p in r["sys_path"] if p in stack.kit_sys_path()], stack.kit_sys_path(), "env.sh's entries, in env.sh's order (a caller's PYTHONPATH keeps its place among them)")
        self.assertTrue(r["sys_path"][-1].endswith("blockfuse_addon"), "env.sh's last entry is the last sys.path entry")
        self.assertIn("[protenix-opt] DRY-RUN mode=exact", err.getvalue()); self.assertIn("env_vars=", err.getvalue())
        self.assertIsNone(stack._REPORT, "a dry run is not an activation")
        with mock.patch.object(stack, "protenix_version", return_value="2.1.0"), redirect_stderr(io.StringIO()):
            r = stack.activate("exact", dry_run=True)
        self.assertFalse(r["active"]); self.assertIn("2.1.0", r["reason"])

    def test_direct_apply_signature_is_an_activation_error(self):
        with mock.patch.object(stack, "_run_kit_sitecustomize", return_value="[sitecustomize] trunk-II levers applied (direct): ['T1:fused']\n"):
            with self.assertRaises(ActivationError) as cm:
                stack._apply(stack.kit_home(), "exact")
        self.assertIn("direct branch", str(cm.exception))


class TestSitecustomize(_Base):
    def test_stands_down_when_the_kit_sitecustomize_is_present(self):
        fake = types.ModuleType("sitecustomize")
        fake.__file__ = os.path.join(stack.kit_home(), "src", "sitecustomize.py")
        with mock.patch.dict(sys.modules, {"sitecustomize": fake}), mock.patch.object(stack, "protenix_version", return_value="2.0.0"), \
             redirect_stderr(io.StringIO()):
            r = protenix_opt.enable("exact", strict=True)          # strict must not raise: env.sh owns the process, that is not a failure
        self.assertFalse(r["active"]); self.assertEqual(r.get("activated_by"), "sitecustomize")
        self.assertNotIn("PTX_BLK", os.environ)


H100 = {"name": "NVIDIA H100 80GB HBM3", "sm": "sm90", "cc": "9.0", "probe": "nvidia-smi"}
MARKERS = ["T1:fused", "TRCORE:on(word=exact bind=tier:exact provider=opt_core.kernels.transition opt_core=0.5.111.0 sites=Transition.forward+block_pair_transition)", "BLK:2(pro+cueq+epi+transition_core)", "DEADSKIP:hooked", "LAZY_INIT:1"]


class TestLateActivation(_Base):
    """The explicit API's late-activation rule: allowed after `import protenix.model`, refused (named) once a model instance exists, the
    kit reports a lever applied, or pairformer is imported; a second enable() is idempotent."""

    def _stubs(self, with_protenix_module=False, with_pairformer=False):
        protenix = types.ModuleType("protenix"); protenix.__path__ = []
        model = types.ModuleType("protenix.model"); model.__path__ = []           # stock 2.0.0: protenix/model/__init__.py is empty
        mods = {"protenix": protenix, "protenix.model": model}
        if with_protenix_module:
            pm = types.ModuleType("protenix.model.protenix")
            pm.Protenix = type("Protenix", (), {})
            mods["protenix.model.protenix"] = pm
        if with_pairformer:
            mods["protenix.model.modules"] = types.ModuleType("protenix.model.modules")
            mods["protenix.model.modules.pairformer"] = types.ModuleType("protenix.model.modules.pairformer")
        return mods

    def _enable(self, strict=False):
        apply = mock.Mock(return_value=(list(MARKERS), {}))
        with mock.patch.object(stack, "protenix_version", return_value="2.0.0"), mock.patch.object(stack, "gpu_probe_smi", return_value=H100), \
             mock.patch.object(stack, "gpu_info", return_value=H100), mock.patch.object(stack, "_apply", apply), redirect_stderr(io.StringIO()):
            r = protenix_opt.enable("exact", strict=strict)
            r2 = protenix_opt.enable("exact", strict=strict)
        return r, r2, apply

    def test_enable_before_any_protenix_import(self):
        with mock.patch.dict(sys.modules):
            for m in [m for m in sys.modules if m == "protenix" or m.startswith("protenix.")]:
                del sys.modules[m]                                              # nothing of the protenix family imported yet
            r, r2, apply = self._enable()
        self.assertTrue(r["active"]); self.assertEqual(apply.call_count, 1); self.assertEqual(r2["mode"], "exact"); self.assertTrue(r2["active"])
        self.assertEqual(protenix_opt.status()["mode"], "exact")

    def test_enable_after_import_protenix_model_before_instantiation(self):
        with mock.patch.dict(sys.modules, self._stubs()):
            r, r2, apply = self._enable()
        self.assertTrue(r["active"], r.get("reason")); self.assertEqual(apply.call_count, 1, "idempotent: the second enable() applies nothing")
        self.assertEqual(r2, r)

    def test_enable_after_instantiation_is_refused_by_name(self):
        mods = self._stubs(with_protenix_module=True, with_pairformer=True)
        instance = mods["protenix.model.protenix"].Protenix()                     # a live model instance
        with mock.patch.dict(sys.modules, mods):
            r, r2, apply = self._enable()
            self.assertFalse(r["active"]); self.assertIn("late activation refused", r["reason"]); self.assertIn("model instance already exists in this process (1 live, counted)", r["reason"])
            self.assertEqual(apply.call_count, 0); self.assertNotIn("PTX_BLK", os.environ)
            stack._REPORT = None
            with self.assertRaises(ActivationError) as cm:
                self._enable(strict=True)
            self.assertIn("model instance already exists", str(cm.exception))
        del instance

    def test_enable_after_pairformer_import_is_refused_by_name(self):
        with mock.patch.dict(sys.modules, self._stubs(with_pairformer=True)):
            r, r2, apply = self._enable()
        self.assertFalse(r["active"]); self.assertIn("pairformer was imported before activation", r["reason"]); self.assertEqual(apply.call_count, 0)

    def test_kit_reported_levers_are_refused_by_name(self):
        lev = types.ModuleType("ptx_trunk2_levers"); lev._STATS = {"applied": ["T1:fused"]}
        with mock.patch.dict(sys.modules, {"ptx_trunk2_levers": lev}):
            r, r2, apply = self._enable()
        self.assertFalse(r["active"]); self.assertIn("levers already applied", r["reason"]); self.assertEqual(apply.call_count, 0)


class TestClassify(unittest.TestCase):
    HOWTO_E = ["LAZY_INIT:1", "PRED_RELEASE:patched", "SAMPLER:hook", "T1:fused", "ATOMATTNEXACT:patched", "TRIATT_EXACT:on(opt_core.kernels.triattn 0.5.225.0 word=exact bind=tier:exact sites=tl+pad8 cc=9.0 select=S256:cueq,S512:cueq,S1536:cueq; no cell vouches an exact-class row on 9.0|torch2.13.0+cu130|cueq0.11.1: every call takes the library op BY NAME through the provider; EXACT-BITWISE vs the library triangle_attention D=32)", "BLK:2(pro+cueq+epi+fusion_transition,chunked=1)", "DEADSKIP:hooked(load_checkpoint)", "PROCUDA:on(sm90a,so=e9009ba57e,sites=f1+f1hm)", "FLASH_TR:on(sm90a,prebuilt=torch2.13.0-cu130,cell=256x1024,class=EXACT,ln=fused,loadcheck=silu65536+tiles(8,333,65536)+lnrows,0.4s)", "HEADSPLIT:on(n>1024,L2=50MB,g=1,prologue=hm,sites=tl+pad8,cell=sm90:fp32acc)", "DITATTN:on(protenix_fpf_dit_attn_exact v7 torch2.13.0-cu130-sm90 loadcheck=3cases-bit-equal digests=match; EXACT-BITWISE vs mem-efficient SDPA fp32 D=48)", "KEEP_POOL:on(skip=3of5,over=empty_cache)", "SUMHOST:on(inference-call;d2h=2,h2d=2/item)", "PWAZ:True",
               "NM:1", "v02:GRAPH:installed", "FPF:enabled(trimul_in,trimul_out)", "TEMPL_DEDUPE:1", "SAMPLER:hook(meta_path)", "DIT_FUSE:ada,gate,res,swiglu(armed)", "TRCORE:on(word=exact bind=tier:exact provider=opt_core.kernels.transition opt_core=0.5.111.0 sites=Transition.forward+block_pair_transition)"]

    ROW = "9.0|3.3"

    def _env(self, mode):
        from protenix_opt.modes import resolve
        return resolve(mode, {"PATH": os.environ["PATH"]}, stack.kit_home(), compute_cap="9.0", triton="3.3.1").exports

    def _row(self, mode):
        from protenix_opt.modes import readme_row
        return readme_row(self.ROW, mode)

    def _classify(self, mode, applied, env):
        from protenix_opt.modes import readme_row
        return stack._classify(mode, applied, env, row=readme_row(self.ROW, mode), row_key=self.ROW)

    def test_total_accounting_exact(self):
        env = self._env("exact")
        env.pop("INFOPT_FASTLN_PREBUILT", None)          # no prebuilt fast-LN selected by env.sh for the installed torch: the lever falls back by name
        with mock.patch.dict(sys.modules, {}):                                               # the exact TriMul's provider module (src/ptx_trimul_routes.py) NOT loaded here
            sys.modules.pop("ptx_trimul_routes", None)
            on0, fb0, skipped0, why0 = self._classify("exact", self.HOWTO_E, env)
        self.assertIn("trimul_core_exact", fb0); self.assertIn("ptx_trimul_routes not loaded", why0["trimul_core_exact"])   # the FPF:enabled marker without the provider module in this process: a fallback by name, never assumed
        tm_mod = types.ModuleType("ptx_trimul_routes")
        with mock.patch.dict(sys.modules, {"ptx_trimul_routes": tm_mod}):
            on, fb, skipped, why = self._classify("exact", self.HOWTO_E, env)
            on2, fb2, _s2, _w2 = self._classify("exact", self.HOWTO_E, dict(env, INFOPT_FASTLN_PREBUILT="/kit/third_party/fastln_prebuilt_cu130"))
        self.assertEqual(sorted(on + fb + skipped), sorted(MODES["exact"]), "every lever lands in exactly one list")
        self.assertEqual(fb, ["fastln_prebuilt"]); self.assertIn("fastln_prebuilt*/manifest.json", why["fastln_prebuilt"])
        self.assertIn("fastln_prebuilt", on2); self.assertEqual(fb2, [], "a prebuilt selected by env.sh is the lever on")
        self.assertEqual(skipped, ["mk_pf"], "MK-PF is not in the row: not part of this arm, not unavailable"); self.assertIn("9.0|3.3", why["mk_pf"])
        for name in ("pad8", "glue_v2"):
            self.assertIn(name, on, name)
        for name in ("deadskip", "trimul_core_exact", "lazy_init", "stackgraph", "sampler_graph", "sampler_fuse", "template_dedupe", "blk2_chunked_exact"):
            self.assertIn(name, on, name)

    def test_fallbacks_are_named(self):
        applied = [a for a in self.HOWTO_E if not a.startswith("DEADSKIP")] + ["DEADSKIP:unavailable(ImportError('deadskip'))", "LAZY_INIT:unavailable(x)"]
        applied = [a for a in applied if a != "LAZY_INIT:1"]
        with mock.patch.dict(sys.modules, {"ptx_trimul_routes": types.ModuleType("ptx_trimul_routes")}):
            on, fb, skipped, why = self._classify("exact", applied, self._env("exact"))
        self.assertIn("deadskip", fb); self.assertIn("ImportError", why["deadskip"])
        self.assertIn("lazy_init", fb)
        self.assertNotIn("deadskip", on)

    TRIATTN_LINE = "BLK:2(pro+PROVIDER(ptx_native_core:attn,tier2,min_tokens=0,below=k2b)+epi+fusion_transition,chunked=k2b)"   # ptx_trunk2_levers' BLK marker with the cc-9.0 attention word (kit 0.3.36: lever triattn_native in the provider slot; 0.3.18-0.3.32: TRIATTN_CUDA(tier2,...))

    def test_fast_needs_smalln_loaded(self):
        applied = ["T1:fused", self.TRIATTN_LINE, "NM:1", "PWAZ:True", "DEADSKIP:hooked(load_checkpoint)",
                   "v02:GRAPH:installed", "FPF:enabled(trimul_in,trimul_out)", "TEMPL_DEDUPE:1", "SAMPLER:hook(meta_path)", "LAZY_INIT:1"]
        env = self._env("fast")
        with mock.patch.dict(sys.modules, {"ptx_trimul_routes": types.ModuleType("ptx_trimul_routes")}):
            sys.modules.pop("fpf_smalln", None)
            on, fb, skipped, why = self._classify("fast", applied, env)       # fpf_smalln not in sys.modules
        self.assertIn("smalln_size_gate", fb); self.assertIn("fpf_smalln not loaded", why["smalln_size_gate"])
        self.assertIn("triattn_native", on); self.assertIn("blk2_chunked_k2b", on); self.assertIn("trimul_core", on)
        self.assertIn("k2b_flash_triattention", skipped); self.assertIn("its slot is served by triattn_native", why["k2b_flash_triattention"])   # cc 9.0: not in the arm, never a fallback
        self.assertIn("glue_v2", on); self.assertEqual(skipped, ["k2b_flash_triattention", "mk_pf", "atom_attn_exact", "dit_attn_exact"])
        self.assertIn("its slot is served by atom_attn,atom_fused on this row", why["atom_attn_exact"])
        self.assertIn("its slot is served by dit_attn on this row", why["dit_attn_exact"])                                   # fast on cc 9.0: the DiT site belongs to dit_attn; the exact kernel is not part of this arm, never a fallback
        with mock.patch.dict(sys.modules, {"fpf_smalln": types.ModuleType("fpf_smalln"), "ptx_trimul_routes": types.ModuleType("ptx_trimul_routes")}):
            on, fb, skipped, why = self._classify("fast", applied, env)
        self.assertIn("smalln_size_gate", on)
        self.assertEqual(sorted(on + fb + skipped), sorted(MODES["fast"]))

    def test_triattn_marker_and_its_unavailable_form(self):
        """cc 9.0: `BLK:2(pro+PROVIDER(ptx_native_core:attn` is the on marker of triattn_native; the slot's `BLK_ATT:provider ptx_native_core:attn unavailable(<exc>)`
        (no CUDA / no usable opt_core.kernels.triattn / the tier word refused at install) carries BAD 'unavailable(' -> the lever is a fallback and fast is NOT ACTIVE."""
        env = self._env("fast")
        real = ["LAZY_INIT:1", "T1:fused", self.TRIATTN_LINE, "DEADSKIP:hooked(load_checkpoint)", "PWAZ:True", "NM:1", "v02:GRAPH:installed",
                "FPF:enabled(trimul_in,trimul_out)", "TEMPL_DEDUPE:1", "SAMPLER:hook(meta_path)"]
        with mock.patch.dict(sys.modules, {"fpf_smalln": types.ModuleType("fpf_smalln")}):
            on, fb, skipped, why = self._classify("fast", real, env)
        self.assertIn("triattn_native", on); self.assertNotIn("triattn_native", fb); self.assertIn("k2b_flash_triattention", skipped)
        back = [a.replace(self.TRIATTN_LINE, "BLK:2(pro+cueq+epi+fusion_transition,chunked=k2b)") for a in real] + ["BLK_ATT:provider ptx_native_core:attn unavailable(NativeUnavailable('package: no_prebuilt:v10@torch2.99')) -> cueq"]
        with mock.patch.dict(sys.modules, {"fpf_smalln": types.ModuleType("fpf_smalln")}):
            on, fb, skipped, why = self._classify("fast", back, env)
        self.assertIn("triattn_native", fb); self.assertIn("blk2_block_path", on); self.assertIn("k2b_flash_triattention", skipped)

    def test_k2b_marker_matches_the_kits_real_line(self):
        """ptx_trunk2_levers L1379 prints attname = sel.upper() + "(tier2)": `BLK:2(pro+K2B(tier2),min_tokens=0(off)+epi+fusion_transition,chunked=k2b)`.
        A lowercase marker never matched it and fast was reported partial with k2b_flash_triattention unavailable although the kit applied it.
        K2B keeps the slot on the cc 10.x rows (B200 here): the row without PTX_T_ATT."""
        from protenix_opt.modes import readme_row, resolve
        env = resolve("fast", {"PATH": os.environ["PATH"]}, stack.kit_home(), compute_cap="10.0", triton="3.7.1").exports
        _classify = lambda mode, applied, env: stack._classify(mode, applied, env, row=readme_row("10.0|3.7", mode), row_key="10.0|3.7", sm="sm100")
        real = ["LAZY_INIT:1", "T1:fused", "BLK:2(pro+K2B(tier2),min_tokens=0(off)+epi+fusion_transition,chunked=k2b)", "DEADSKIP:hooked(load_checkpoint)",
                "PWAZ:True", "NM:1", "v02:GRAPH:installed", "FPF:enabled(trimul_in,trimul_out)", "TEMPL_DEDUPE:1", "SAMPLER:hook(meta_path)"]
        with mock.patch.dict(sys.modules, {"fpf_smalln": types.ModuleType("fpf_smalln")}):
            on, fb, skipped, why = _classify("fast", real, env)
        self.assertIn("k2b_flash_triattention", on); self.assertIn("blk2_chunked_k2b", on); self.assertNotIn("k2b_flash_triattention", fb)
        # the provider form (L1369) also carries K2B; the cueq fallback (L1372) does not
        prov = [a.replace("BLK:2(pro+K2B(tier2),min_tokens=0(off)", "BLK:2(pro+PROVIDER(k2b,tier2,min_tokens=300,below=cueq)") for a in real]
        with mock.patch.dict(sys.modules, {"fpf_smalln": types.ModuleType("fpf_smalln")}):
            on, fb, skipped, why = _classify("fast", prov, env)
        self.assertIn("k2b_flash_triattention", on)
        back = [a.replace("BLK:2(pro+K2B(tier2),min_tokens=0(off)", "BLK:2(pro+cueq") for a in real] + ["BLK_ATT:provider k2b unavailable(ImportError('x')) -> cueq"]
        with mock.patch.dict(sys.modules, {"fpf_smalln": types.ModuleType("fpf_smalln")}):
            on, fb, skipped, why = _classify("fast", back, env)
        self.assertIn("k2b_flash_triattention", fb); self.assertIn("blk2_block_path", on)

    def test_other_key_row_switches_are_not_part_of_the_arm(self):
        from protenix_opt.modes import OTHER_ROW, resolve
        env = resolve("exact", {"PATH": os.environ["PATH"]}, stack.kit_home(), compute_cap="8.9", triton="3.3.1").exports   # a cc with no row (8.0 has its broad 8.0|* row)
        with mock.patch.dict(sys.modules, {"ptx_trimul_routes": types.ModuleType("ptx_trimul_routes")}):
            on, fb, skipped, why = stack._classify("exact", self.HOWTO_E, env, row=OTHER_ROW["exact"], row_key="other")
        self.assertEqual(skipped, ["pad8", "glue_v2", "mk_pf", "dit_attn_exact", "triatt_exact", "atom_attn_exact", "triatt_prologue_cuda"]); self.assertNotIn("PTX_E_PAD8", env); self.assertNotIn("PTX_DIT_ATTN_EXACT", env); self.assertNotIn("PTX_E_TRIMUL", env)
        self.assertEqual(sorted(on + fb + skipped), sorted(MODES["exact"]))


class TestPaths(unittest.TestCase):
    def test_kit_paths_are_package_relative(self):
        opt = stack.opt_home()
        self.assertEqual(stack.kit_home(), os.path.join(opt, "forward", "flashpairformer"))
        self.assertEqual(stack.kit_class_dirs(), [os.path.join(opt, "forward")])
        sp = stack.kit_sys_path()
        self.assertEqual(sp[0], os.path.join(stack.kit_home(), "src"))
        self.assertTrue(sp[-1].endswith("blockfuse_addon"), "env.sh's last entry is the last sys.path entry")
        self.assertEqual(stack.kit_sys_path(mode="off"), [], "off puts nothing on sys.path")
        self.assertEqual(stack.protenix_pin(), "2.0.0"); self.assertTrue(os.path.isfile(stack.pins_path()))

    def test_cells_view(self):
        c = stack.cells("sm90")
        self.assertNotIn("error", c, c.get("error"))
        self.assertIn("sm90", c["blk2_arch"]); self.assertTrue(c["blk2"])
        self.assertTrue(stack.cells("sm80")["blk2"])                 # A100: blk2_arch carries sm80
        self.assertFalse(stack.cells("sm86")["blk2"])                # an arch without BLK2 cells


if __name__ == "__main__":
    unittest.main()
