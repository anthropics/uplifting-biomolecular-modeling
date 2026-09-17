"""The registry describes levers without values; the fold settings are the library's own knobs and defaults, read from upstream's own sources;
the deterministic recipe in det.py keeps one spelling in the package and in the core's recipe form."""
import os
import unittest

from esmfold2_opt import det, modes, registry, settings
from esmfold2_opt.tests import _stubs


class TestRegistry(unittest.TestCase):
    def test_fields(self):
        fields = {registry.FIELD_BASE, registry.FIELD_OPT, registry.FIELD_W4, registry.FIELD_MK, registry.FIELD_MSA, registry.FIELD_XL,
                  registry.FIELD_ATOM, registry.FIELD_FEATS, registry.FIELD_MSA2, registry.FIELD_PAIR, registry.FIELD_HOIST, registry.FIELD_DIT, registry.FIELD_TRIMUL, registry.FIELD_LN,
                  registry.FIELD_RC}
        for name, lv in registry.LEVERS.items():
            self.assertEqual(name, lv.name)
            self.assertIn(lv.field, fields)
            self.assertIn(lv.class_4, ("forward", "datapath", "serving", "packing"))
            self.assertTrue(set(lv.variants) <= set(registry.ALL_VARIANTS) and lv.variants)
            self.assertIn(lv.probe[0], ("w4", "opt", "msa", "mk", "base", "xl", "atom", "feats", "msa2", "pair", "cute", "trimul", "hoist", "dit", "ln", "xte", "rc"))
            self.assertTrue(lv.what and lv.kit_file)
            if lv.field in (registry.FIELD_XL, registry.FIELD_RC):       # the memory levers and the route's row-chunking levers: their word vs stock is their own (bitwise / T2 by label)
                self.assertIn(lv.tier_vs_stock, ("bitwise", "T2"))
            else:
                self.assertEqual(lv.tier_vs_stock, "T2")                 # every fused-line kit mode inherits the fused backend

    def test_no_switch_values_transcribed(self):
        """registry.py is a pure lever-name index: switch values live once, in the settings/preset modules, never copied in here."""
        import inspect
        src = inspect.getsource(registry)
        self.assertNotRegex(src, r"EF2_\w+\s*=\s*['\"]")            # no switch value transcribed

    def test_msa_levers_act_on_the_full_model(self):
        """The kit applies its MSA-encoder levers (M1 = `msa`, t11-t14) to every model with an msa_encoder — both Full variants, MSA or
        not (the GPU proof: ef2_msa.enable(model, t12) runs on full_nomsa); only `fast` (no msa_encoder) is outside their reach."""
        self.assertEqual(registry.FULL_MODEL, ("full_msa", "full_nomsa"))
        for n in ("t11", "t12", "t13", "t14", "msa"):
            self.assertEqual(registry.LEVERS[n].variants, registry.FULL_MODEL)
        self.assertTrue(registry.acts_on(registry.LEVERS["t12"], None))
        self.assertTrue(registry.acts_on(registry.LEVERS["t12"], "full_nomsa") and registry.acts_on(registry.LEVERS["msa"], "full_nomsa"))
        self.assertFalse(registry.acts_on(registry.LEVERS["t12"], "fast"))
        self.assertEqual((registry.LEVERS["msa"].tier_vs_kit_line, registry.LEVERS["msa"].tier_vs_stock), ("T2", "T2"))
        self.assertNotIn("FULL_MSA_ONLY", open(registry.__file__, encoding="utf-8").read())

    def test_jit_cache_key(self):
        """The JIT cache key rule torch<version sans local tag>-cu<torch.version.cuda sans dot>-sm<cc> (configs/h100.env exports it as MODEL_OPT_STACK_KEY)."""
        from esmfold2_opt.modes import jit_cache_key
        self.assertEqual(jit_cache_key("2.13.0+cu130", None, "9.0"), "torch2.13.0-cu130-sm90")
        self.assertEqual(jit_cache_key("2.7.1+cu126", None, "9.0"), "torch2.7.1-cu126-sm90")
        self.assertEqual(jit_cache_key("2.13.0", "13.0", "10.0"), "torch2.13.0-cu130-sm100")
        from esmfold2_opt import stack
        env = open(os.path.join(stack.tree_home(), "configs", "h100.env"), encoding="utf-8").read()
        self.assertIn("from esmfold2_opt.modes import jit_cache_key; print(jit_cache_key())", env)
        self.assertNotIn("_mo_torch", env)                                                    # no second derivation in the config

    def test_config_h100_keys_the_triton_cache(self):
        """`source configs/h100.env`: MODEL_OPT_STACK_KEY = jit_cache_key() (a stub torch 2.13.0 whose version has no local tag, CUDA from
        torch/version.py read without importing torch, a fake nvidia-smi 9.0); with MODEL_OPT_JIT_ROOT set, TRITON_CACHE_DIR = <root>/<key>/triton and the
        weights memo dir = <root>/weights unless pre-set; with it unset neither is exported (the tools' own defaults) and a pre-set cache directory is
        kept as given; HF_HOME is never defaulted by the config."""
        import subprocess, sys, tempfile
        from esmfold2_opt import stack
        tmp = tempfile.mkdtemp()
        stub = os.path.join(tmp, "stub", "torch"); os.makedirs(stub)
        open(os.path.join(stub, "__init__.py"), "w").write("from .version import __version__, cuda\n")
        open(os.path.join(stub, "version.py"), "w").write("__version__ = '2.13.0'\ncuda = '13.0'\n")         # the wheel layout the resolver reads WITHOUT importing torch (torch/version.py: a version with no local tag, cuda beside it)
        dist = os.path.join(tmp, "stub", "torch-2.13.0.dist-info"); os.makedirs(dist)                  # and the stub's DISTRIBUTION METADATA beside it: the resolver reads the installed torch's metadata
        open(os.path.join(dist, "METADATA"), "w").write("Metadata-Version: 2.1\nName: torch\nVersion: 2.13.0\n")   # first (importlib.metadata, sys.path order), so the stub — first on PYTHONPATH — is
        open(os.path.join(dist, "RECORD"), "w").write("")                                                # authoritative whatever torch the interpreter running the tests has installed
        b = os.path.join(tmp, "bin"); os.makedirs(b)
        open(os.path.join(b, "python"), "w").write(f"#!/bin/bash\nexec {sys.executable} \"$@\"\n")
        open(os.path.join(b, "nvidia-smi"), "w").write("#!/bin/bash\necho 'NVIDIA H100 80GB HBM3, 9.0, 81559'\n")     # the probe's query form: name, compute_cap, memory.total (nounits)
        os.chmod(os.path.join(b, "python"), 0o755); os.chmod(os.path.join(b, "nvidia-smi"), 0o755)
        env = {k: v for k, v in os.environ.items() if k not in ("MODEL_OPT_STACK_KEY", "TRITON_CACHE_DIR", "MODEL_OPT", "HF_HOME", "ESMCFOLD_CCD_PATH",
                                                             "MODEL_OPT_JIT_ROOT", "ESMFOLD2_OPT_WEIGHTS_MEMO_DIR")}
        env["PATH"] = b + os.pathsep + env["PATH"]; env["PYTHONPATH"] = os.path.join(tmp, "stub") + os.pathsep + env.get("PYTHONPATH", "")
        cfg = os.path.join(stack.tree_home(), "configs", "h100.env")
        run = lambda cmd: subprocess.run(["bash", "-c", f"source {cfg} && {cmd}"], env=env, capture_output=True, text=True)
        unset = "${MODEL_OPT_STACK_KEY:-unset} ${TRITON_CACHE_DIR:-unset} ${ESMFOLD2_OPT_WEIGHTS_MEMO_DIR:-unset} ${HF_HOME:-unset} ${ESMCFOLD_CCD_PATH:-unset}"
        r = run(f"echo {unset}")                                                               # no JIT root: the key is computed, no cache path is invented, no weights root is invented
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        self.assertEqual(r.stdout.split(), ["torch2.13.0-cu130-sm90", "unset", "unset", "unset", "unset"])
        root = os.path.join(tmp, "jit")
        env["MODEL_OPT_JIT_ROOT"] = root                                                        # the optional persistent JIT root: <root>/<key>/triton and <root>/weights
        r = run(f"echo {unset}")
        self.assertEqual(r.returncode, 0, r.stderr[-600:])
        self.assertEqual(r.stdout.split(), ["torch2.13.0-cu130-sm90", os.path.join(root, "torch2.13.0-cu130-sm90", "triton"), os.path.join(root, "weights"), "unset", "unset"])
        env["TRITON_CACHE_DIR"] = "/state/torch_ext/triton"                                   # a pre-set cache directory is kept as given, existing or not
        r = run("echo $TRITON_CACHE_DIR")
        self.assertEqual(r.stdout.split(), ["/state/torch_ext/triton"])
        del env["MODEL_OPT_JIT_ROOT"]; env["HF_HOME"] = os.path.join(tmp, "hf")                 # a given weights root: the CCD path pinned derives from it
        r = run("echo $HF_HOME && echo $ESMCFOLD_CCD_PATH")
        self.assertEqual(r.stdout.split(), [os.path.join(tmp, "hf"), os.path.join(tmp, "hf", "hub", "models--biohub--ESMFold2", "snapshots", "1ebf0e3481a5184eb6171d40615c79e384b48796", "ccd.pkl")])

    def test_data_path_gate_names_the_variable_and_the_path(self):
        """A data variable that names a path which does not exist is a refusal naming both; an UNSET HF_HOME is a refusal too (frozen weights:
        the library's default cache is never read); a root lacking a pinned file, or holding it at another byte count, is refused naming the
        file; a complete root (every pinned file at its byte count) passes with the CCD note when ESMCFOLD_CCD_PATH is unset."""
        import tempfile
        from esmfold2_opt import stack
        from esmfold2_opt.tests import _stubs
        pins = _stubs.require_pins()
        why, notes = stack.data_path_gate({"HF_HOME": "/state/weights/esmfold2/hf"}, pins=pins)
        self.assertEqual(why[:len("HF_HOME=/state/weights/esmfold2/hf does not exist")], "HF_HOME=/state/weights/esmfold2/hf does not exist")
        why, notes = stack.data_path_gate({"HF_HOME": os.getcwd(), "ESMCFOLD_CCD_PATH": "/state/weights/esmfold2/ccd.pkl"}, pins=pins)
        self.assertTrue(why.startswith("ESMCFOLD_CCD_PATH=/state/weights/esmfold2/ccd.pkl does not exist"))
        why, notes = stack.data_path_gate({}, pins=pins)
        self.assertTrue(why.startswith("HF_HOME is not set"), why); self.assertIn("PINS.json", why); self.assertIn("HF_HUB_OFFLINE", why)
        tmp = tempfile.mkdtemp(prefix="ef2_weights_")
        try:
            why, notes = stack.data_path_gate({"HF_HOME": tmp}, variant="fast", pins=pins)              # an existing root with nothing in it
            self.assertTrue(why.startswith(f"HF_HOME={tmp} lacks the weights files variant fast loads"), why); self.assertIn("config.json absent", why); self.assertIn("more", why)
            hf = _stubs.write_fake_weights_root(tmp, pins)
            why, notes = stack.data_path_gate({"HF_HOME": hf}, variant="fast", pins=pins)
            self.assertIsNone(why, why)
            self.assertEqual([n for n in notes if not n.startswith("WEIGHTS")], ["ESMCFOLD_CCD_PATH is not set: the library reads ccd.pkl from the pinned snapshot under HF_HOME (stock/PINS.json \"ccd\")"])
            self.assertEqual(len([n for n in notes if n.startswith("WEIGHTS unknown")]), 1, notes)          # the stub root's one-byte files: an unknown checkpoint by sha256, WARN + proceed
            _repo, rel, size = next(x for x in stack.pinned_weight_files(pins, "fast") if x[1].endswith("model.safetensors"))
            with open(os.path.join(hf, rel), "wb") as fh:
                fh.truncate(size - 1)                                                           # a short file = an UNKNOWN checkpoint: WARN by name (the file and both counts) and proceed
            why, notes = stack.data_path_gate({"HF_HOME": hf}, variant="fast", pins=pins)
            self.assertIsNone(why, why)
            warn = [n for n in notes if n.startswith("WEIGHTS unknown")]
            self.assertEqual(len(warn), 1, notes); self.assertIn("proceeding", warn[0])
            _absent, unknown = stack.weight_files_check(hf, pins, "fast")                          # the per-file entries (the note prints the first four)
            ent = [u for u in unknown if u.startswith(rel)]
            self.assertEqual(len(ent), 1, unknown); self.assertIn("sha256=", ent[0]); self.assertIn(f"{size - 1} bytes (pinned {size})", ent[0])
            why, notes = stack.data_path_gate({"HF_HOME": hf}, variant="full_msa", pins=pins)   # another variant's files are complete
            self.assertIsNone(why, why)
        finally:
            import shutil
            shutil.rmtree(tmp, ignore_errors=True)


class TestSettings(unittest.TestCase):
    def test_kit_driver_readers(self):
        """The kit driver's own constants and the server's fused base are read from their files."""
        kit = _stubs.require_kit()
        c = settings.kit_constants(os.path.join(kit, modes.DRIVER_RELPATH))
        self.assertEqual(set(c), set(settings.KIT_CONSTANTS))
        self.assertEqual(settings.kit_model_calls(os.path.join(kit, modes.SERVER_RELPATH)), {"kernel_backend": "fused", "chunk_size": None})   # the server's fused base, read from configure()

    def test_defaults_from_pins_when_esm_absent(self):
        pins = _stubs.require_pins()
        try:
            import esm.models.esmfold2.processor  # noqa: F401
            has_esm = True
        except Exception:
            has_esm = False
        st = settings.resolve(pins=pins)
        d = pins["library_defaults"]["fold"]
        if not has_esm:
            self.assertEqual(st.source, "stock/PINS.json library_defaults")
        self.assertEqual((st.num_loops, st.num_sampling_steps, st.msa_max_depth), (d["num_loops"], d["num_sampling_steps"], d["msa_max_depth"]))
        self.assertIsNone(st.msa_read_depth)
        self.assertFalse(st.msa_remove_insertions)                                   # the library's from_a3m default: insertions kept
        m = pins["library_defaults"]["model"]
        self.assertIsNone(st.seeds)                                                  # unseeded: --seeds is required

    def test_no_flag_is_the_library_defaults_field_by_field(self):
        """A run with no fold flag is a pass-through to the upstream API at its defaults: every field equals stock/PINS.json
        library_defaults (fold keywords, the two model calls, the A3M read form), and it carries no seed list — the seeds are the
        caller's (nothing else is added: the stock caller sets no torch flag and no variable; test_merge_locks holds that)."""
        pins = _stubs.require_pins()
        st = settings.resolve(pins=pins)
        lib = pins["library_defaults"]
        self.assertEqual(st.fold_kwargs(), {k: lib["fold"][k] for k in st.fold_kwargs()})
        self.assertEqual(set(st.fold_kwargs()), {"num_loops", "num_sampling_steps", "num_diffusion_samples", "msa_max_depth", "lm_dropout", "msa_column_mask_rate"})
        self.assertEqual(st.msa_read_kwargs(), {"remove_insertions": lib["msa_read"]["remove_insertions"], "max_sequences": lib["msa_read"]["max_sequences"]})
        self.assertIsNone(st.seeds)

    def test_flags_are_the_library_keywords_by_name(self):
        """Every pass-through flag is a keyword of ESMFold2InputBuilder.fold / MSA.from_a3m, spelled as the library spells it, and lands on
        the settings by name; an unknown name is refused by name."""
        pins = _stubs.require_pins()
        self.assertEqual(settings.STOCK_FLAGS, ("num_loops", "num_sampling_steps", "num_diffusion_samples", "msa_max_depth", "lm_dropout", "msa_column_mask_rate",
                                                "noise_scale", "step_scale", "max_inference_sigma", "lm_mask_pct", "early_exit",
                                                "remove_insertions", "max_sequences"))
        self.assertEqual(settings.SAMPLER_OVERRIDES, ("noise_scale", "step_scale", "max_inference_sigma", "lm_mask_pct", "early_exit"))
        self.assertTrue(set(settings.FOLD_KWARGS) <= set(pins["library_defaults"]["fold"]))
        self.assertEqual(set(settings.MSA_READ_KWARGS), {k for k in pins["library_defaults"]["msa_read"] if k != "note"})
        st = settings.resolve(pins=pins, num_loops=10, num_sampling_steps=100, num_diffusion_samples=5, remove_insertions=True, max_sequences=2048)
        self.assertEqual((st.num_loops, st.num_sampling_steps, st.num_diffusion_samples, st.msa_remove_insertions, st.msa_read_depth), (10, 100, 5, True, 2048))
        self.assertEqual(st.executed_steps, 68)                                       # 100 scheduled -> 68 executed (the paper's count)
        self.assertIn("--num_loops 10", st.source)
        with self.assertRaises(ValueError):
            settings.resolve(pins=pins, recycles=3)
        import argparse
        ap = argparse.ArgumentParser(); settings.add_stock_flags(ap)
        a = ap.parse_args(["--num_loops", "1", "--num_sampling_steps", "4", "--remove_insertions", "false", "--lm_dropout", "0.0"])
        self.assertEqual(settings.flags_given(a), {"num_loops": 1, "num_sampling_steps": 4, "lm_dropout": 0.0, "remove_insertions": False})
        st = settings.resolve(a, pins=pins)
        self.assertEqual((st.num_loops, st.num_sampling_steps, st.lm_dropout, st.msa_remove_insertions, st.num_diffusion_samples), (1, 4, 0.0, False, pins["library_defaults"]["fold"]["num_diffusion_samples"]))
        self.assertEqual(vars(ap.parse_args([])), {k: None for k in settings.STOCK_FLAGS})   # absent = the library default, by name
        # fold()'s optional overrides: forwarded only when given (no-flag fold kwargs stay the six), kit modes refuse them by name
        st = settings.resolve(pins=pins)
        self.assertIsNone(st.fold_overrides); self.assertNotIn("noise_scale", st.fold_kwargs())
        a = ap.parse_args(["--noise_scale", "1.05", "--early_exit", "true", "--max_inference_sigma", "160"])
        st = settings.resolve(a, pins=pins)
        self.assertEqual(st.fold_overrides, {"noise_scale": 1.05, "max_inference_sigma": 160.0, "early_exit": True})
        self.assertEqual((st.fold_kwargs()["noise_scale"], st.fold_kwargs()["early_exit"]), (1.05, True)); self.assertIsNone(st.executed_steps)   # another sigma cut: the executed count is not recorded
        import json as _json
        self.assertEqual(settings.from_json(_json.dumps(st.as_dict())).fold_overrides, st.fold_overrides)

    def test_executed_steps_recorded_beside_num_sampling_steps(self):
        """ONE source for the executed-step table: stock/PINS.json library_defaults.executed_steps (the package reads it; nothing is retyped)."""
        pins = _stubs.require_pins()
        table = {int(k): v for k, v in pins["library_defaults"]["executed_steps"].items() if k != "note"}
        self.assertEqual(settings.EXECUTED_STEPS.table(pins), table)
        self.assertEqual({200, 100, 68, 14}, set(table)); self.assertEqual(settings.EXECUTED_STEPS.get(100), table[100])
        self.assertIsNone(settings.EXECUTED_STEPS.get(7))
        src = open(settings.__file__, encoding="utf-8").read()
        self.assertNotIn("{200: 134", src)                                                    # no literal copy in the package
        self.assertEqual(settings.resolve(pins=pins).executed_steps, 134)
        self.assertEqual(settings.resolve(pins=pins, num_sampling_steps=68).executed_steps, 46)

    def test_det_overrides_fold_kwargs(self):
        pins = _stubs.require_pins()
        st = settings.resolve(pins=pins)
        kw = st.fold_kwargs(det=True)
        self.assertEqual(kw["lm_dropout"], 0.0); self.assertEqual(kw["msa_column_mask_rate"], 0.0)
        kw0 = st.fold_kwargs()
        self.assertNotEqual(kw0["lm_dropout"], 0.0)

    def test_json_round_trip(self):
        pins = _stubs.require_pins()
        st = settings.resolve(pins=pins, num_loops=3)
        import json
        self.assertEqual(settings.from_json(json.dumps(st.as_dict())), st)


class TestDetRecipe(unittest.TestCase):
    def test_recipe_in_the_core_shape_has_the_kit_words(self):
        r1, r2, r0 = det.recipe(1), det.recipe(2), det.recipe(0)
        self.assertEqual(dict(r1.env), det.ENV); self.assertEqual(dict(r2.env), det.ENV)     # the words cannot drift between the kit table and the core's recipe form
        self.assertEqual((r0.level, dict(r0.env)), (0, {}))
        self.assertEqual((r1.level, r2.level), (1, 2))

    def test_apply_env_setdefault(self):
        env = {"CUBLAS_WORKSPACE_CONFIG": ":16:8"}
        added = det.apply_env(env)
        self.assertEqual(env["CUBLAS_WORKSPACE_CONFIG"], ":16:8")       # a caller's value is kept (setdefault)
        self.assertEqual(set(added), set(det.ENV) - {"CUBLAS_WORKSPACE_CONFIG"})


if __name__ == "__main__":
    unittest.main()


class TestMkReporting(unittest.TestCase):
    """F2: the MK hoist is reported from the preset at plan time and from the sampler's own counters after the run."""

    def test_plan_note_from_the_preset(self):
        from esmfold2_opt import stack
        self.assertIsNone(stack.mk_plan_note(["t6", "t3"], 5))
        self.assertEqual(stack.mk_plan_note(["t6", "mk"], 5), "mk: installed; inactive at num_diffusion_samples>1 (kit guard)")
        self.assertEqual(stack.mk_plan_note(["mk"], 1), "mk: installed; active at num_diffusion_samples=1")

    def test_after_run_never_applied_when_every_step_fell_back(self):
        import collections, sys, types
        from esmfold2_opt import stack
        saved = sys.modules.get(stack.MK_MODULE)
        try:
            sys.modules.pop(stack.MK_MODULE, None)
            self.assertIsNone(stack.mk_after_run())
            rep = {"levers_applied": ["t6", "mk"], "levers_fallback": [], "partial": []}
            self.assertEqual(stack.settle_mk(rep), rep)                                  # module never imported: nothing to settle
            sys.modules[stack.MK_MODULE] = types.SimpleNamespace(STATS=collections.Counter(fallback_batch_or_args=68, folds_seen=0))
            out = stack.settle_mk(rep)                                                   # the kit guard's own fallback: a declared precondition -> gated by name, not partial
            self.assertEqual((out["levers_applied"], out["levers_fallback"], out["partial"]), (["t6"], ["mk"], []))
            self.assertEqual(len(out["gated"]), 1); self.assertTrue(out["gated"][0].startswith("mk: every step fell back"))
            self.assertIn("every step fell back", out["fallback_reasons"]["mk"]); self.assertFalse(out["mk"]["active"])
            sys.modules[stack.MK_MODULE] = types.SimpleNamespace(STATS=collections.Counter(fallback_batch_or_args=0, folds_seen=0))
            out = stack.settle_mk(rep)                                                   # no fold reached the sampler: no evidence of application -> partial
            self.assertEqual((out["levers_fallback"], out["partial"], out.get("gated")), (["mk"], ["mk"], None))
            self.assertEqual(out["fallback_reasons"]["mk"], "no fold reached the sampler")
            sys.modules[stack.MK_MODULE] = types.SimpleNamespace(STATS=collections.Counter(fallback_batch_or_args=0, folds_seen=3, side_steps=200))
            out = stack.settle_mk(rep)
            self.assertEqual((out["levers_applied"], out["levers_fallback"]), (["t6", "mk"], [])); self.assertTrue(out["mk"]["active"])
        finally:
            if saved is not None:
                sys.modules[stack.MK_MODULE] = saved
            else:
                sys.modules.pop(stack.MK_MODULE, None)
