"""Locks against the kit bytes and the stock pin: the deterministic recipe pair (det.py == the kit's apply_deterministic_mode), the
items translation (the kit's own fasta_spec, uid form), the fold settings (stock's run_inference defaults from the pinned source,
stock's own flags; warm's constant == chai_proto.RUN_KW), the carried kit files (by directory
listing; no house file inside the kit directories), the stock caller's environment proof, the exit tally at interpreter exit
(subprocesses). No torch, no GPU."""
import hashlib
import json
import os
import re
import subprocess
import sys
import tempfile
import unittest

from chai1_opt import det, inputs, modes, outputs, report, settings, stack, stock_fold, warm
from chai1_opt.tests import _kitspell

TREE = stack.tree_home()
KIT = stack.kit_home()
EAGER = stack.eager_home()
DSTEP = stack.dstep_home()
ERRATA02 = stack.errata02_home()
PROTO = os.path.join(KIT, modes.PROTO_RELPATH)


def _kit_proto():
    d = os.path.dirname(PROTO)
    if d not in sys.path:
        sys.path.insert(0, d)
    import chai_proto
    return chai_proto


def _sha(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


class TestDetRecipePair(unittest.TestCase):
    def test_det_apply_is_the_kits_four_statements(self):
        k = _kitspell.kit_recipe(PROTO)
        self.assertEqual(k["env_switch"], det.ENV_KIT)
        self.assertEqual(k["jit_optout"], det.ENV_JIT)
        self.assertTrue(k["jit_profiling"])
        self.assertEqual(tuple(k["cublas"]), det.CUBLAS_ENV)
        self.assertTrue(k["cudnn_deterministic"] and k["cudnn_benchmark_false"] and k["deterministic_algorithms"])
        # the package's apply() performs the same statements on a stub torch
        from chai1_opt.tests import _stubs
        t, _, _, saved = _stubs.install()
        try:
            env = {}
            state = det.apply(1, t, environ=env)
            self.assertEqual(t._C.calls, [("profiling", False)])
            self.assertEqual(env[det.CUBLAS_ENV[0]], det.CUBLAS_ENV[1])
            self.assertTrue(t.backends.cudnn.deterministic); self.assertFalse(t.backends.cudnn.benchmark)
            self.assertEqual(t._det, (True, False))
            self.assertEqual(state["level"], 1)
            self.assertEqual(det.apply(0, t, environ={}), {"level": 0})
        finally:
            _stubs.remove(saved)
        self.assertEqual(det.env_for_driver(1), {"CHAI_DETERMINISTIC": "1"}); self.assertEqual(det.env_for_driver(0), {})
        self.assertEqual(det.env_before_torch(1), {"CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        with self.assertRaises(ValueError):
            det.level(2)
        # the .pth route reads the level from the kit's own switch
        self.assertEqual(det.level_from_env({}), 0); self.assertEqual(det.level_from_env({"CHAI_DETERMINISTIC": "0"}), 0)
        self.assertEqual(det.level_from_env({"CHAI_DETERMINISTIC": "1"}), 1)
        for bad in ("warn", "2", "yes"):
            with self.assertRaises(ValueError):
                det.level_from_env({"CHAI_DETERMINISTIC": bad})


class TestInputsTranslation(unittest.TestCase):
    def test_kit_key_and_uid(self):
        cp = _kit_proto()
        pack = os.path.join(KIT, "tests", "public_inputs", "PACK.json")
        its = inputs.load(pack)
        self.assertEqual(len(its.items), 4)
        inputs.resolve_keys(its, cp)
        for it in its.items:
            spec = cp.fasta_spec(it.fasta)
            self.assertEqual((it.key, it.uid), (spec["key"], f"fasta:{it.fasta}"))
            self.assertEqual(it.key, os.path.splitext(os.path.basename(it.fasta))[0])                          # the key is the FASTA stem
        self.assertEqual(its.items[0].key, "1BRS_1to1")
        self.assertEqual(inputs.uids_arg(its.items[:2]), its.items[0].uid + "," + its.items[1].uid)
        one = inputs.load(os.path.join(KIT, "tests", "public_inputs", "1BRS_1to1.fasta"))
        self.assertEqual(len(one.items), 1)
        inputs.resolve_keys(one, cp)
        self.assertEqual(one.items[0].key, cp.fasta_spec(one.items[0].fasta)["key"])

    def test_role_less_inputs(self):
        """Inputs are what chai-lab accepts: a chai FASTA folded verbatim, no target / binder / ligand role anywhere. A monomer, a plain
        three-chain complex and a protein + ligand FASTA with no role field load, key by the FASTA stem, translate to the driver's
        `fasta:<path>` spec and to the stock caller's plan (id / key / fasta / seeds only); the kit's own input_spec agrees on the key and
        carries no role field, its write_fasta is a byte copy and its MSA directory is the caller's as given. A stray role field in an items
        record is not read; two items with one stem are refused by name."""
        cp = _kit_proto()
        cases = {"mono": ">protein|name=A\nMKVL\n", "tri": ">protein|name=A\nMKV\n>protein|name=B\nGGS\n>protein|name=C\nWWL\n",
                 "pl": ">protein|name=A\nMKV\n>ligand|name=L\nCCO\n"}
        with tempfile.TemporaryDirectory() as td:
            for stem, txt in cases.items():
                fa = os.path.join(td, stem + ".fasta"); open(fa, "w").write(txt)
                its = inputs.resolve_keys(inputs.load(fa), cp); it = its.items[0]
                self.assertEqual((it.id, it.key, it.uid), (stem, stem, f"fasta:{fa}"))
                spec = cp.input_spec(it.uid)
                self.assertEqual(spec, {"uid": stem, "key": stem, "fasta_src": fa})                                # the kit's spec: no role field
                self.assertEqual(cp.input_spec(fa), spec)                                                         # a bare .fasta path is the same spec
                dst = os.path.join(td, stem + ".copy.fasta"); cp.write_fasta(spec, dst)
                self.assertEqual(open(dst, "rb").read(), open(fa, "rb").read())                                     # folded verbatim
                self.assertEqual(os.fspath(cp.ensure_msa_pqt(spec, os.path.join(td, "msas"))), os.path.join(td, "msas"))
                plan = inputs.to_plan(its, its.items, {it.key: [7]}, settings.from_values({}), 0, None)
                self.assertEqual(plan["items"], [{"id": stem, "key": stem, "fasta": fa, "seeds": [7]}])
            tri = os.path.join(td, "tri.fasta"); items_p = os.path.join(td, "items.json")
            json.dump({"items": [{"fasta": "tri.fasta", "seeds": [1]}, {"id": "m", "fasta": "mono.fasta"}]}, open(items_p, "w"))
            its = inputs.resolve_keys(inputs.load(items_p), cp)
            self.assertEqual([(i.id, i.key, i.uid, i.seeds) for i in its.items], [("tri", "tri", f"fasta:{tri}", [1]), ("m", "mono", f"fasta:{os.path.join(td, 'mono.fasta')}", None)])
            json.dump({"items": [{"fasta": "tri.fasta"}, {"id": "again", "fasta": "tri.fasta"}]}, open(items_p, "w"))
            with self.assertRaises(ValueError) as cm:
                inputs.resolve_keys(inputs.load(items_p), cp)
            self.assertIn("share the kit key 'tri' (the same FASTA stem: one output directory)", str(cm.exception))
            with self.assertRaises(SystemExit):
                cp.input_spec("some_opaque_id")                                                             # not a FASTA: refused by name

    def test_plan_carries_what_the_stock_caller_needs(self):
        cp = _kit_proto()
        its = inputs.resolve_keys(inputs.load(os.path.join(KIT, "tests", "public_inputs", "1BRS_2to2.fasta")), cp)
        st = settings.from_values({"use_esm_embeddings": False, "low_memory": False})
        plan = inputs.to_plan(its, its.items, {its.items[0].key: [42, 43]}, st, 1, "/tmp/msa")
        self.assertEqual((plan["fold"], plan["non_default"], plan["schema"]), (st["fold"], {"use_esm_embeddings": False, "low_memory": False}, "chai1_opt.plan/2")); self.assertNotIn("preset", plan)
        self.assertEqual(plan["items"][0]["seeds"], [42, 43]); self.assertEqual(plan["det"], 1); self.assertIsNone(plan["device"])
        plan = inputs.to_plan(its, its.items, {its.items[0].key: None}, settings.from_values({}, device="cuda:1"), 0, None)   # no seed named, stock's --device given
        self.assertEqual((plan["items"][0]["seeds"], plan["device"], plan["non_default"]), (None, "cuda:1", {}))
        self.assertEqual(plan["items"][0]["key"], its.items[0].key)


class TestFoldSettings(unittest.TestCase):
    """settings.py: stock's own run_inference knobs as pass-through flags — stock's names, stock's defaults (the pinned source, AST; never an
    import of the upstream), the kit driver's RUN_KW shape, the driver's by-name refusal of a value it cannot serve, the seed fallback."""

    def test_defaults_are_the_pinned_signatures(self):
        src = os.path.join(TREE, "stock", "src", "chai_lab", "chai1.py")
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))
        d = settings.library_defaults(src)
        self.assertEqual(d, settings.library_defaults()); self.assertEqual(d, {k: pins["library_defaults"][k] for k in settings.FOLD_KEYS})   # the AST read == PINS' copy
        self.assertEqual(d, {"msa_server_url": "https://api.colabfold.com", "constraint_path": None, "template_hits_path": None,
                             "use_esm_embeddings": True, "use_msa_server": False, "use_templates_server": False, "recycle_msa_subsample": 0, "num_trunk_recycles": 3,
                             "num_diffn_timesteps": 200, "num_diffn_samples": 5, "num_trunk_samples": 1, "low_memory": True})       # chai1.py:487-502, as shipped
        st = settings.from_values({})                                                        # no flag given: stock's defaults, nothing else
        self.assertEqual((st["fold"], st["non_default"], st["flags"], settings.label(st)), (d, {}, [], "stock"))
        self.assertEqual(settings.from_values({"use_esm_embeddings": True, "num_diffn_samples": 5})["non_default"], {})   # a flag AT stock's default is no departure
        self.assertIsNone(pins["library_defaults"]["seed"])                                  # the signature is unseeded and stock sets no seed then:
        self.assertRegex(open(src, encoding="utf-8").read(), r"if seed is not None:\n\s+set_seed\(\[seed\]\)")   # chai1.py:570-572 — off without a seed is exactly that call
        self.assertEqual((settings.seeds_for(None, None), settings.seeds_for([7], [1, 2]), settings.seeds_for(None, [1, 2])), (None, [7], [1, 2]))   # no seed named = None, never a made-up one
        self.assertIsNone(settings.seed_refusal("off", {"a": None}))
        self.assertIsNone(settings.seed_refusal("exact", {"a": [0], "b": [1, 2]}))
        self.assertEqual(settings.seed_refusal("fast", {"a": [0], "b": None}),
                         "kit modes need --seed <int> (or --seeds 0,1 / the items' own seeds): outputs are written per seed, <key>/seed_<s>/ — no seed named for b (--mode off without a seed is stock's unseeded call)")
        self.assertNotIn("from chai_lab", open(settings.__file__, encoding="utf-8").read().split("def installed_signature_defaults")[0])
        self.assertIsNone(settings.installed_signature_defaults())                           # never imports the upstream

    def test_flags_are_stocks_own(self):
        main_src = open(os.path.join(TREE, "stock", "src", "chai_lab", "main.py"), encoding="utf-8").read()
        self.assertIn('app.command("fold", help="Run Chai-1 to fold a complex.")(run_inference)', main_src)   # stock's CLI is typer over run_inference: its flags are the keyword names, dashed
        self.assertEqual(settings.STOCK_FLAGS, {k: "--" + k.replace("_", "-") for k in settings.FOLD_KEYS})
        self.assertEqual(settings.STOCK_FLAGS["num_trunk_recycles"], "--num-trunk-recycles"); self.assertEqual(settings.STOCK_FLAGS["low_memory"], "--low-memory")
        import argparse
        ap = argparse.ArgumentParser(); settings.add_fold_arguments(ap)
        self.assertEqual({k: getattr(ap.parse_args([]), k) for k in settings.FOLD_KEYS}, {k: None for k in settings.FOLD_KEYS})   # not given = None = stock's default
        for given in ({"low_memory": False}, {}, {"use_esm_embeddings": False, "low_memory": False}, {"use_esm_embeddings": False}, {"num_diffn_samples": 1, "low_memory": False},
                      {"num_trunk_recycles": 4, "num_diffn_timesteps": 100, "recycle_msa_subsample": 2, "num_trunk_samples": 2, "use_msa_server": True, "use_templates_server": True},
                      {"msa_server_url": "https://example.invalid/msa", "constraint_path": "/c/restraints.csv", "template_hits_path": "/t/hits.m8"}):
            st = settings.from_values(given)
            back = settings.from_args(ap.parse_args(st["flags"]))                            # flags out -> flags in: the same settings
            self.assertEqual((back["fold"], back["non_default"], back["flags"]), (st["fold"], st["non_default"], st["flags"]), given)
        self.assertEqual(settings.fold_flags({"use_esm_embeddings": False, "low_memory": False}), ["--no-use-esm-embeddings", "--no-low-memory"])
        self.assertEqual(settings.fold_flags({"num_diffn_samples": 1, "low_memory": True}), ["--num-diffn-samples", "1", "--low-memory"])
        self.assertEqual(settings.from_args(ap.parse_args(["--low-memory", "false"]))["non_default"], {"low_memory": False})   # the valued form reads too
        self.assertEqual(settings.fold_flags({"constraint_path": "/c/r.csv", "msa_server_url": "u"}), ["--msa-server-url", "u", "--constraint-path", "/c/r.csv"])   # FOLD_KEYS order, the value verbatim
        self.assertTrue(os.path.isabs(ap.parse_args(["--template-hits-path", "hits.m8"]).template_hits_path))                 # a path flag is made absolute (the driver runs in the output directory)
        with self.assertRaises(KeyError):
            settings.from_values({"num_recycles": 3})

    def test_warms_fold_is_the_kits_run_kw(self):
        cp = _kit_proto()
        st = settings.from_values(warm.WARM_FOLD)                                            # warm's constant: one diffusion sample, the batch on the GPU
        self.assertEqual(settings.to_driver_run_kw(st), dict(cp.RUN_KW))                    # == the kit's own RUN_KW literal (kit/chai_proto.py:52-53)
        self.assertEqual(st["flags"], ["--num-diffn-samples", "1", "--no-low-memory"])
        self.assertEqual(sorted(settings.to_driver_run_kw(settings.from_values({}))), sorted(settings.DRIVER_RUN_KW_KEYS))

    def test_every_fold_flag_rides_the_kits_run_kw(self):
        """Every keyword and --device ride the kit's RUN_KW, whose feature-context call, trunk loop and fold calls read them
        (errata_02/chai_worker.py) — num_trunk_samples included: the worker runs run_inference's own loop over trunk samples."""
        self.assertFalse(hasattr(settings, "DRIVER_FIXED")); self.assertFalse(hasattr(settings, "driver_refusal"))
        self.assertEqual((settings.from_values({})["device"], settings.from_values({}, device="cuda:1")["device"]), (None, "cuda:1"))   # stock's --device: None = not given = stock's default
        rk = settings.to_driver_run_kw(settings.from_values({"recycle_msa_subsample": 3, "template_hits_path": "/t/hits.m8", "use_templates_server": True, "num_trunk_samples": 2}, device="cuda:1"))
        self.assertEqual((rk["device"], rk["recycle_msa_subsample"], rk["template_hits_path"], rk["use_templates_server"], rk["constraint_path"], rk["num_trunk_samples"]), ("cuda:1", 3, "/t/hits.m8", True, None, 2))
        self.assertEqual(settings.to_driver_run_kw(settings.from_values({}))["device"], "cuda:0")                     # not given: cuda:0, what stock resolves None to
        self.assertEqual(settings.to_driver_run_kw(settings.from_values({}))["num_trunk_samples"], 1)
        src = open(stack.driver_path(), encoding="utf-8").read()                                                       # the worker reads them where run_inference passes them
        for frag in ('use_msa_server=cp.RUN_KW["use_msa_server"]', 'msa_server_url=cp.RUN_KW["msa_server_url"]', 'constraint_path=_opt_path(cp.RUN_KW["constraint_path"])',
                     'use_templates_server=cp.RUN_KW["use_templates_server"]', 'templates_path=_opt_path(cp.RUN_KW["template_hits_path"])',
                     'recycle_msa_subsample=cp.RUN_KW["recycle_msa_subsample"]', 'device = torch.device(cp.RUN_KW.get("device") or "cuda:0")',
                     'n_trunk = int(cp.RUN_KW["num_trunk_samples"])', 'for trunk_idx in range(n_trunk):',                 # run_inference's trunk loop (chai1.py:526-540), verbatim
                     'output_dir=(od / f"trunk_{trunk_idx}" if n_trunk > 1 else od)', 'seed=s + trunk_idx if s is not None else None',
                     'cand = chai1.StructureCandidates.concat(all_candidates)'):
            self.assertIn(frag, src, frag)
        self.assertEqual(src.count('use_msa_server=cp.RUN_KW["use_msa_server"]'), 1)                                   # the one context call site (the per-seed pass)

    def _on_disk(self, root):
        return sorted(os.path.relpath(os.path.join(dp, f), os.path.join(TREE, "opt")) for dp, _, fn in os.walk(root) for f in fn if "__pycache__" not in dp)


    def test_driver_kit_carried_files(self):
        prefix = "forward/fast_inference/"
        carried = sorted(rel[len(prefix):] for rel in self._on_disk(KIT))
        self.assertEqual(carried, sorted(["KNOWN_ISSUES.md", "LICENSE", "NOTICE", "README.md",
                                          "kit/chai_proto.py"] +
                                         ["tests/public_inputs/" + n for n in ("1BRS_1to1.fasta", "1BRS_2to2.fasta", "1BRS_3to3.fasta", "1BRS_4to4.fasta", "PACK.json")]))

    def test_errata02_carries_exactly_the_worker(self):
        self.assertEqual(sorted(rel[len("forward/errata_02/"):] for rel in self._on_disk(ERRATA02)), ["chai_worker.py"])

    def test_the_worker_imports_the_kit_and_carries_no_diagnostics(self):
        """The worker imports the kit's modules through its $KIT switch, applies the kit's deterministic recipe, and carries no diagnostic arm
        (RNG probes, per-module stage timers, a process-global numerics audit, coordinate digests) and no weight download — a content check."""
        self.assertEqual(stack.driver_path(), os.path.join(TREE, "opt", "forward", "errata_02", modes.WORKER_RELPATH))
        w = open(stack.driver_path(), encoding="utf-8").read()
        self.assertIn('if os.environ.get("KIT"): sys.path.insert(0, os.environ["KIT"])', w); self.assertIn("import chai_proto as cp", w)
        self.assertIn("DET_MODE = cp.apply_deterministic_mode()", w)
        for gone in ("--probes", "--stages", "global_state", "coords_sha", "ensure_weights", "CHAI_WEIGHTS", "urllib", "requests"):
            self.assertNotIn(gone, w, gone)
        proto = open(os.path.join(os.path.dirname(os.path.dirname(stack.driver_path())), "fast_inference", "kit", "chai_proto.py"), encoding="utf-8").read()
        for gone in ("def ensure_weights", "CHAI_WEIGHTS", "chaiassets.com", "urlopen", "def coords_sha"):
            self.assertNotIn(gone, proto, gone)                                                           # weights come from the install; nothing is fetched at run time

    def test_eager_stack_oom_reraise_is_in_the_source(self):
        src = open(os.path.join(TREE, "opt", "forward", "eager_trunk", "chai1_eager", "stack.py"), encoding="utf-8").read()
        self.assertIn("from opt_core.oom import is_oom", src); self.assertIn("if is_oom(e):", src)

    def test_eager_stack_carried_files(self):
        prefix = "forward/eager_trunk/"
        carried = sorted(rel[len(prefix):] for rel in self._on_disk(EAGER))
        self.assertEqual(carried, ["LICENSE", "NOTICE", "README.md"] +
                         ["chai1_eager/" + n for n in ("__init__.py", "hoist.py", "kernels.py", "msa_kernels.py", "stack.py", "transition_core.py", "trunk.py", "ts2eager.py")])
        self.assertFalse(os.path.isdir(os.path.join(EAGER, "results")))
        # the stack's own version and its LEVERS, from the carried files (what modes.py composes against)
        init = open(os.path.join(EAGER, "chai1_eager", "__init__.py"), encoding="utf-8").read()
        self.assertIn('__version__ = "0.4.8+errata01"', init)                       # the errata drop-in (+ hoist2 stand-down hook 0.4.3, scalar-cache folding 0.4.4)
        self.assertEqual(_kitspell.eager_stack_levers(os.path.join(EAGER, "chai1_eager", "stack.py")), ("stock", "tier1"))
        self.assertEqual(_sha(os.path.join(EAGER, "LICENSE")), _sha(os.path.join(KIT, "LICENSE")))       # the same Apache-2.0 text as the driver kit's

    def test_dstep_addon_carried_files(self):
        prefix = "forward/dstep_megakernel/"
        carried = sorted(rel[len(prefix):] for rel in self._on_disk(DSTEP))
        self.assertEqual(carried, ["README.md"] + ["chai1_fastln/" + n for n in ("__init__.py", "aoti.py", "cpu_isa.py", "dit_attn.py", "stackx.py")])   # fastln.py (lever flnbig) retired at 0.3.2; aoti.py (the compiled step ahead of time) at 0.3.3; cpu_isa.py (a package's launcher ISA vs the host's, stdlib) at 0.3.13
        init = open(os.path.join(DSTEP, "chai1_fastln", "__init__.py"), encoding="utf-8").read()
        self.assertIn('__version__ = "0.3.13"', init)
        self.assertEqual(_kitspell.dstep_all_levers(os.path.join(DSTEP, modes.DSTEP_STACKX_RELPATH)), modes.DSTEP_LEVERS)


class TestStockPassThrough(unittest.TestCase):
    """The stock route passes upstream nothing but its own defaults (plus inputs, outputs, the seed) unless a flag was given: defaults_check
    holds the plan's keywords to the INSTALLED signature and names every departure (the pass-through gate allows exactly the flags given)."""

    def test_defaults_check_names_exactly_the_flags_given(self):
        def run_inference(fasta_file, *, output_dir, use_esm_embeddings=True, use_msa_server=False, use_templates_server=False, recycle_msa_subsample=0,
                          num_trunk_recycles=3, num_diffn_timesteps=200, num_diffn_samples=5, num_trunk_samples=1, seed=None, device=None, low_memory=True,
                          msa_server_url="https://api.colabfold.com", constraint_path=None, template_hits_path=None):
            return None
        n = len(settings.FOLD_KEYS)
        chk = stock_fold.defaults_check(settings.from_values({})["fold"], run_inference)
        self.assertEqual((chk["checked"], chk["mismatch"], chk["not_in_signature"]), (n, {}, []))          # no flag: the installed defaults exactly
        chk = stock_fold.defaults_check(settings.from_values({"low_memory": False})["fold"], run_inference)
        self.assertEqual((chk["checked"], chk["mismatch"], chk["not_in_signature"]), (n, {"low_memory": {"passed": False, "default": True}}, []))
        chk = stock_fold.defaults_check(settings.from_values({"use_esm_embeddings": False, "low_memory": False})["fold"], run_inference)
        self.assertEqual(chk["mismatch"], {"low_memory": {"passed": False, "default": True}, "use_esm_embeddings": {"passed": False, "default": True}})
        chk = stock_fold.defaults_check(settings.from_values({"use_esm_embeddings": False})["fold"], run_inference)
        self.assertEqual((chk["checked"], chk["mismatch"], chk["not_in_signature"]), (n, {"use_esm_embeddings": {"passed": False, "default": True}}, []))

class TestStockEnvProof(unittest.TestCase):
    def test_forbidden_names(self):
        env = {"PATH": "/bin", "CHAI_DOWNLOADS_DIR": "/data/chai_downloads"}
        proof = stock_fold.env_proof(environ=env, path=["/usr/lib/python3"], modules={"os": None, "sys": None})
        self.assertTrue(proof["clean"])
        for k in ("CHAI1_OPT", "CHAI1_OPT_HOME", "CHAI_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG", "CHAI_JIT_PROFILING_OFF"):
            with self.assertRaises(RuntimeError):
                stock_fold.env_proof(environ=dict(env, **{k: "1"}), path=[], modules={})
        with self.assertRaises(RuntimeError):
            stock_fold.env_proof(environ=env, path=[os.path.join(KIT, "kit")], modules={})
        with self.assertRaises(RuntimeError):
            stock_fold.env_proof(environ=env, path=[], modules={"chai_proto": None})
        cleaned = stock_fold.clean_environment(dict(env, CHAI1_OPT="exact", CHAI_DETERMINISTIC="1", CUBLAS_WORKSPACE_CONFIG=":4096:8"))
        self.assertEqual(cleaned, env)
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))["stock_environment"]
        for name in pins["must_be_absent"]:
            name = name.rstrip("*")
            with self.assertRaises(RuntimeError, msg=name):
                stock_fold.env_proof(environ=dict(env, **{name: "1"}), path=[], modules={})


class TestExitTally(unittest.TestCase):
    NS = ("{'levels': {'W1', 'W2', 'W5'}, '_MODULE_CACHE': {'a': 1, 'b': 2}, 'LOAD_LOG': [('a', 1.5), ('b', 0.5)], "
          "'ESM_STATS': {'hits': 3, 'misses': 1}, 'ESM_CACHE': {'x': 1}, 'summary': [{'n_ok': 2, 'n': 2, 'rows': []}]}")

    def _exit_lines(self, code, env_extra=None):
        env = {k: v for k, v in os.environ.items() if k not in ("CHAI1_OPT", "CHAI_DETERMINISTIC", "CUBLAS_WORKSPACE_CONFIG")}
        env.update(env_extra or {})
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        lines = [ln for ln in r.stderr.splitlines() if ln.strip()]
        return r.returncode, lines

    def test_tally_printed_at_interpreter_exit(self):
        rc, lines = self._exit_lines(f"import chai1_opt.report as r; r.register_exit_tally({self.NS}, route='driver')")
        self.assertEqual(rc, 0, lines)
        self.assertRegex(lines[-1], r"^\[chai1-opt\] EXIT pid=\d+ route=driver source=memory levels=W1,W2,W5 modules_resident=2 loads=2 "
                                    r"load_s=2\.0 esm_hits=3 esm_misses=1 esm_memo=1 seed_folds_ok=2/2 inputs=1$")
        self.assertEqual(sum("EXIT pid=" in ln for ln in lines), 1)                       # once per process

    def test_tally_says_so_when_the_lever_statements_never_ran(self):
        rc, lines = self._exit_lines("import chai1_opt.report as r; r.register_exit_tally()")
        self.assertEqual(rc, 0, lines)
        self.assertRegex(lines[-1], r"^\[chai1-opt\] EXIT pid=\d+ route=in-process no lever counters: the kit's lever statements never ran in this process$")
        rc, lines = self._exit_lines("import chai1_opt.report")                          # no registration: no line, never silent otherwise
        self.assertEqual(rc, 0); self.assertFalse(any("EXIT pid=" in ln for ln in lines))

    def test_env_route_refuses_a_partial_by_name_unless_the_opt_out(self):
        """The .pth route (CHAI1_OPT=<mode>) and enable(): a partial activation (a lever of the mode the run's own switches turned off) is
        REFUSED by name — NOT ACTIVE naming the levers and the opt-out, exit 3, never a silent stock run — unless CHAI1_OPT_ALLOW_PARTIAL=1 (the
        environment route's word, that route only; pred / the driver take --allow-partial) / enable(allow_partial=True): then RECORDED (ACTIVE line
        with PARTIAL off=..., one NOTE line) and the process runs on the levers that applied."""
        stub = ("from chai1_opt.tests import _stubs; from chai1_opt import stack; _stubs.install(); _stubs.gates_pass(stack); _stubs.eager_stub(stack); "
                "real = stack.classify\n"
                "def cw(*a, **k):\n    c = real(*a, **k); c['applied'] = [n for n in c['applied'] if n != 'W5']; c['off'] = sorted(set(c['off']) | {'W5'}); return c\n"
                "stack.classify = cw\n")
        fire = (stub + "from chai1_opt import _autoload; import sys; f = _autoload.__dict__[[n for n in dir(_autoload) if n.endswith('Finder')][0]]('exact'); "
                       "f._fire('test'); import chai1_opt; rep = chai1_opt.status(); assert rep['active'] and rep['partial'] and rep.get('partial_note') and rep['allow_partial'], rep")
        rc, lines = self._exit_lines(fire)                                                # no opt-out: refused by name, exit 3
        self.assertEqual(rc, 3, lines)
        self.assertTrue(any(ln.startswith("[chai1-opt] NOT ACTIVE: partial activation: lever(s) W5 of mode exact not applied") and "CHAI1_OPT_ALLOW_PARTIAL=1" in ln for ln in lines), lines)
        self.assertFalse(any("] ACTIVE mode=" in ln for ln in lines), lines)
        rc, lines = self._exit_lines(fire, {"CHAI1_OPT_ALLOW_PARTIAL": "1"})              # the route's opt-out: recorded, proceeds
        self.assertEqual(rc, 0, lines)
        self.assertTrue(any("ACTIVE mode=exact" in ln and "PARTIAL off=W5" in ln for ln in lines), lines)
        self.assertTrue(any("NOTE partial activation: lever(s) W5 not applied" in ln for ln in lines), lines)
        rc, lines = self._exit_lines(fire, {"CHAI1_OPT_ALLOW_PARTIAL": "true"})           # a value other than 0/1: refused by name
        self.assertEqual(rc, 3, lines); self.assertTrue(any("NOT ACTIVE: CHAI1_OPT_ALLOW_PARTIAL='true'" in ln for ln in lines), lines)
        rc2, lines2 = self._exit_lines(stub + "import chai1_opt; rep = chai1_opt.enable('exact'); sys = __import__('sys'); sys.exit(0 if rep['active'] else 3)")
        self.assertEqual(rc2, 3, lines2)
        self.assertTrue(any("NOT ACTIVE: partial activation: lever(s) W5 of mode exact not applied" in ln for ln in lines2), lines2)
        rc3, lines3 = self._exit_lines(stub + "import chai1_opt; rep = chai1_opt.enable('exact', allow_partial=True); sys = __import__('sys'); sys.exit(0 if rep['active'] else 3)")
        self.assertEqual(rc3, 0, lines3)
        self.assertTrue(any("NOTE partial activation: lever(s) W5 not applied" in ln for ln in lines3), lines3); self.assertFalse(any("NOT ACTIVE" in ln for ln in lines3), lines3)

    def test_tally_at_exit_through_enable_on_stubs(self):
        code = ("from chai1_opt.tests import _stubs; from chai1_opt import stack; _stubs.install(); _stubs.gates_pass(stack); _stubs.eager_stub(stack); "
                "import chai1_opt; rep = chai1_opt.enable('exact'); assert rep['active'], rep")
        rc, lines = self._exit_lines(code)
        self.assertEqual(rc, 0, lines)
        self.assertTrue(any(ln.startswith("[chai1-opt] ACTIVE mode=exact levels=W1,W2,W5 eager=tier1 dstep=hoist2 optin=alloc levers_applied=W1,W5,tier1,hoist2,templ_empty,exactln,msa_pad,transition,rankcc,tailasync,confmemo,alloc not_applicable=W2,prefetch") for ln in lines), lines)
        self.assertRegex(lines[-1], r"^\[chai1-opt\] EXIT pid=\d+ route=in-process source=memory n_gpu=1 sharding=none levels=W1,W5 modules_resident=0 loads=0 load_s=0\.0 esm_hits=0 esm_misses=0 esm_memo=0 eager=tier1 eager_graph_fallbacks=0 eager_events=0( .*)?$")

    def test_lines(self):
        line = report.exit_tally_line()
        self.assertTrue(line.startswith("[chai1-opt] EXIT pid="))
        ns = {"levels": {"W1", "W2", "W5"}, "_MODULE_CACHE": {"a": 1, "b": 2}, "LOAD_LOG": [("a", 1.5), ("b", 0.5)], "ESM_STATS": {"hits": 3, "misses": 1},
              "ESM_CACHE": {"x": 1}, "summary": [{"n_ok": 2, "n": 2, "rows": [{}, {}]}]}
        f = " ".join(report.tally_fields(ns))
        self.assertEqual(f, "levels=W1,W2,W5 modules_resident=2 loads=2 load_s=2.0 esm_hits=3 esm_misses=1 esm_memo=1 seed_folds_ok=2/2 inputs=1")
        report.register_exit_tally(ns, route="driver")
        self.assertIn("route=driver source=memory levels=W1,W2,W5", report.exit_tally_line())
        self.assertTrue(report._TALLY["registered"])


class TestOutputs(unittest.TestCase):
    def test_names_and_hook(self):
        """Per (item, seed) the kept files are upstream's own two per sample — pred.model_idx_<k>.cif + scores.model_idx_<k>.npz — and nothing
        else: the kit's save_seed_outputs keeps sample 0's pair, the package's hook adds the pairs of samples k >= 1; no array file is written."""
        import numpy as np
        self.assertEqual(outputs.upstream_names(0), ["pred.model_idx_0.cif", "scores.model_idx_0.npz"]); self.assertEqual(sum(len(outputs.upstream_names(k)) for k in range(5)), 10)
        self.assertFalse(any(hasattr(outputs, n) for n in ("SAMPLE_ARRAYS", "array_name", "save_sample_arrays")))
        cp = _kit_proto()
        orig = cp.save_seed_outputs
        outputs.install_extra_samples_hook(cp); outputs.install_extra_samples_hook(cp)
        self.assertTrue(cp.save_seed_outputs._chai1_opt_hook); self.assertIs(cp.save_seed_outputs._chai1_opt_orig, orig)
        class Cand:                                                                               # two samples; arrays a writer would have saved
            def __init__(self, od):
                self.cif_paths = [os.path.join(od, f"pred.model_idx_{k}.cif") for k in range(2)]
                self.pae = np.zeros((2, 4, 4)); self.plddt = np.zeros((2, 4)); self.pde = np.zeros((2, 4, 4))
        try:
            with tempfile.TemporaryDirectory() as td:
                od = os.path.join(td, "fold"); sd = os.path.join(td, "keep"); os.makedirs(od)
                for k in range(2):
                    open(os.path.join(od, f"pred.model_idx_{k}.cif"), "w").write(f"cif{k}")
                    np.savez(os.path.join(od, f"scores.model_idx_{k}.npz"), aggregate_score=np.float32(0.5 + k), ptm=np.float32(0.4), iptm=np.float32(0.3))
                row = cp.save_seed_outputs(Cand(od), od, sd)                                      # the hooked function: sample 0 by the kit, sample 1 by the hook
                self.assertEqual(sorted(os.listdir(sd)), ["pred.model_idx_0.cif", "pred.model_idx_1.cif", "scores.model_idx_0.npz", "scores.model_idx_1.npz"])
                self.assertEqual((row["aggregate_score"], row["pae_shape"], row["n_samples"], row["extra_sample_files"]), (0.5, [4, 4], 2, ["pred.model_idx_1.cif", "scores.model_idx_1.npz"]))
                for k in range(2):
                    self.assertEqual(open(os.path.join(sd, f"pred.model_idx_{k}.cif")).read(), f"cif{k}")            # copied as written
                self.assertNotIn("side_files", row); self.assertFalse(os.path.exists(os.path.join(sd, "msa_depth.pdf")))   # no MSA: upstream wrote no coverage plot, none is kept
                open(os.path.join(od, "msa_depth.pdf"), "w").write("plot")                                           # an input with an MSA: run_folding_on_context wrote its plot beside the samples
                row = cp.save_seed_outputs(Cand(od), od, sd)
                self.assertEqual(row["side_files"], ["msa_depth.pdf"]); self.assertEqual(open(os.path.join(sd, "msa_depth.pdf")).read(), "plot")   # kept where upstream put it, upstream's name
                # two trunk samples (num_trunk_samples=2): upstream writes trunk_<i>/pred|scores.model_idx_<k> (run_inference's layout); the kit keeps
                # the first sample's pair where it lies (trunk_0/) and the hook every other pair, sub-directories preserved, nothing renamed
                class Cand2(Cand):
                    def __init__(self, od):
                        self.cif_paths = [os.path.join(od, f"trunk_{i}", f"pred.model_idx_{k}.cif") for i in range(2) for k in range(2)]
                        self.pae = np.zeros((4, 4, 4)); self.plddt = np.zeros((4, 4)); self.pde = np.zeros((4, 4, 4))
                od2 = os.path.join(td, "fold2"); sd2 = os.path.join(td, "keep2")
                for i in range(2):
                    os.makedirs(os.path.join(od2, f"trunk_{i}"))
                    for k in range(2):
                        open(os.path.join(od2, f"trunk_{i}", f"pred.model_idx_{k}.cif"), "w").write(f"cif{i}{k}")
                        np.savez(os.path.join(od2, f"trunk_{i}", f"scores.model_idx_{k}.npz"), aggregate_score=np.float32(0.25 * (i + k)), ptm=np.float32(0.4), iptm=np.float32(0.3))
                row2 = cp.save_seed_outputs(Cand2(od2), od2, sd2)
                self.assertEqual(sorted(os.listdir(sd2)), ["trunk_0", "trunk_1"])
                self.assertEqual([sorted(os.listdir(os.path.join(sd2, f"trunk_{i}"))) for i in range(2)], [["pred.model_idx_0.cif", "pred.model_idx_1.cif", "scores.model_idx_0.npz", "scores.model_idx_1.npz"]] * 2)
                self.assertEqual((row2["aggregate_score"], row2["n_samples"], row2["extra_sample_files"]), (0.0, 4, ["trunk_0/pred.model_idx_1.cif", "trunk_0/scores.model_idx_1.npz", "trunk_1/pred.model_idx_0.cif", "trunk_1/scores.model_idx_0.npz", "trunk_1/pred.model_idx_1.cif", "trunk_1/scores.model_idx_1.npz"]))
                self.assertEqual(open(os.path.join(sd2, "trunk_1", "pred.model_idx_0.cif")).read(), "cif10")           # copied as written
                self.assertNotIn("side_files", row2)
                for i in range(2):
                    open(os.path.join(od2, f"trunk_{i}", "msa_depth.pdf"), "w").write(f"plot{i}")                     # one plot per run_folding_on_context call: per trunk_<i>/
                row2 = cp.save_seed_outputs(Cand2(od2), od2, sd2)
                self.assertEqual(row2["side_files"], ["trunk_0/msa_depth.pdf", "trunk_1/msa_depth.pdf"])
                self.assertEqual([open(os.path.join(sd2, f"trunk_{i}", "msa_depth.pdf")).read() for i in range(2)], ["plot0", "plot1"])
                self.assertEqual(outputs.UPSTREAM_SIDE_FILES, ("msa_depth.pdf",))
        finally:
            cp.save_seed_outputs = orig


class TestConfigsPlatformNeutral(unittest.TestCase):
    """configs/h100.env fills only platform-neutral values: the weights directory comes from the caller's CHAI_DOWNLOADS_DIR (refused by
    name when unset) and no filesystem path is a default of the file."""

    def test_h100_env_defaults(self):
        env = open(os.path.join(TREE, "configs", "h100.env"), encoding="utf-8").read()
        self.assertNotRegex(env, r"(?m)^export CHAI_DOWNLOADS_DIR=")                                                  # the caller's variable, never a default of this file
        self.assertNotRegex(env, r"(?m)^export [A-Z_]+=/")                                                              # no absolute path baked in

    def test_configs_one_shape(self):
        """configs/ holds h100.env and a100.env; a100.env pre-sets the one differing value (MODEL_OPT_TARGET_GPU=A100, compute capability 8.0)
        and sources h100.env for everything else — one body of configuration logic, two targets."""
        self.assertEqual(sorted(os.listdir(os.path.join(TREE, "configs"))), ["a100.env", "h100.env", "h200.env"])
        a100 = open(os.path.join(TREE, "configs", "a100.env"), encoding="utf-8").read()
        code = [l for l in a100.split("\n") if l.strip() and not l.lstrip().startswith("#")]
        self.assertEqual(len(code), 2, code)
        self.assertTrue(code[0].startswith("export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-A100}"), code[0])
        self.assertEqual(code[1], 'source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/h100.env"')
        h100 = open(os.path.join(TREE, "configs", "h100.env"), encoding="utf-8").read()
        self.assertIn("export MODEL_OPT_TARGET_GPU=${MODEL_OPT_TARGET_GPU:-H100}", h100)                                   # pre-set wins: h100.env fills only what is unset

    def test_weights_directory_unset_is_refused_by_name(self):
        saved = os.environ.pop("CHAI_DOWNLOADS_DIR", None)
        try:
            ok, why, _ = stack.weights_check()
            self.assertFalse(ok); self.assertTrue(why.startswith("CHAI_DOWNLOADS_DIR is not set — see README Variables"), why)
            ok, why, _ = stack.weights_check("/nonexistent/chai_downloads")
            self.assertEqual((ok, why), (False, "CHAI_DOWNLOADS_DIR=/nonexistent/chai_downloads does not exist"))
        finally:
            if saved is not None: os.environ["CHAI_DOWNLOADS_DIR"] = saved


class TestStackOfRecord(unittest.TestCase):
    """The tree knows one stack: stock/PINS.json `stack` = `stacks`' only entry (every mode; stock and kit arms on one torch / CUDA stack);
    modes.PINNED_STACK documents its torch build; the stack's packages are named in PINS `stack` itself (no freeze file beside it)."""

    def test_pins_name_exactly_the_pinned_stack(self):
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))
        names = [k for k in pins["stacks"] if k != "note"]
        self.assertEqual(names, [stack.stack_key(torch_v=pins["stack"]["torch"], cuda=pins["stack"]["cuda"], cc="9.0")])   # one pinned stack, keyed by the tree's stack key (torch2.13.0-cu130-sm90), never by a container tag
        self.assertNotIn("image", pins["stack"]); self.assertTrue(pins["stack"]["tested_on"].startswith("the pinned stack below (torch 2.13.0+cu130, CUDA 13.0"), pins["stack"]["tested_on"])   # the stack by label + versions; no platform, no id
        self.assertEqual(pins["stacks"][names[0]]["modes"], ["off", "exact", "fast", "big"])
        self.assertEqual(modes.PINNED_STACK, "torch " + pins["stack"]["torch"]); self.assertEqual(pins["stacks"][names[0]]["torch"], pins["stack"]["torch"])
        self.assertNotIn("freeze", pins["stack"]); self.assertEqual([f for f in os.listdir(os.path.join(TREE, "stock")) if "freeze" in f], [])   # the stack's pins live in PINS.json; no unread freeze file

    def test_pins_stack_restates_the_environment_lock(self):
        """environment/requirements.lock is the stack's one package list (the image and route C install it); the entries PINS `stack` names for
        the pin check and the docs equal the lock's (torch sans its local tag: the lock pins PyPI's dist version, torch.version carries +cu130)."""
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json")))["stack"]
        lock = {}
        for line in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "==" in line:
                k, v = line.split("==", 1); lock[k.lower().replace("-", "_")] = v
        self.assertEqual(lock["torch"], pins["torch"].split("+")[0]); self.assertTrue(pins["torch"].endswith("+cu" + pins["cuda"].replace(".", "")))
        for k in ("triton", "numpy", "rdkit", "gemmi", "pandas", "antipickle", "einops"): self.assertEqual(lock[k], pins[k], k)
        self.assertEqual(lock["nvidia_cudnn_cu13"].rsplit(".", 1)[0], pins["cudnn"])          # 9.20.0.48 → 9.20.0, torch.backends.cudnn's number
        self.assertEqual(lock["chai_lab"], json.load(open(os.path.join(TREE, "stock", "PINS.json")))["upstream"]["version"])
        self.assertTrue(pins["python"].startswith("3.11."))

    def test_environment_dockerfile_builds_the_pinned_stack(self):
        """environment/Dockerfile's base image, interpreter and install steps are the pinned stack's: the CUDA base tag carries PINS `cuda`, the
        python-build-standalone tarball PINS `python`, the lock and the stock wheel are the files it installs, `run.sh install` is its kit step."""
        pins = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8")); st = pins["stack"]
        df = open(os.path.join(TREE, "environment", "Dockerfile"), encoding="utf-8").read()
        m = re.search(r"^FROM (\S+)$", df, re.M); self.assertIsNotNone(m)
        self.assertTrue(m.group(1).startswith(f"nvidia/cuda:{st['cuda']}.") and "-runtime-" in m.group(1), m.group(1))
        self.assertIn(f"cpython-{st['python']}+", df)
        self.assertIn("COPY chai1/environment/requirements.lock ", df); self.assertIn(f"chai1/stock/chai_lab-{pins['upstream']['version']}-py3-none-any.whl", df)
        self.assertIn("RUN bash run.sh install", df)
        self.assertRegex(open(os.path.join(TREE, "environment", "apptainer.def"), encoding="utf-8").read(), r"(?m)^From: chai1-kit:")

