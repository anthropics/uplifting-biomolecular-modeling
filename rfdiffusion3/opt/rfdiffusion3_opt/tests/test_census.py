"""The KERNELS census (census.py): the routes table's expectations, the line's grammar, the readers on fake bound objects, the REQUIRE
guard's refusals (exit 5, the accelerator and the route named), and the two process routes end to end on fake interpreters — the stock
child (`design --mode off [--compile]`: route stock | default) and the kit process (`design --mode exact|fast [--compile]`).

The fakes model upstream's shapes, not its numerics: the engine stub (tests/_fixtures.stub_engine) carries `_COMPILE_TARGETS`, the walk
to the diffusion module and the compile step of rfd3/engine.py:212-269, the one atom-attention path record of attention.py:403-404 and the
per-batch clock of engine.py:402 and the per-call dense/sparse decision of attention.py:409-454 behind the module attribute upstream's
call site looks up; the fake torch carries `torch.compile` -> `torch._dynamo.eval_frame.OptimizedModule`, the dynamo disable switch and
the counters. The record's grammar is locked against the vendored upstream file itself (the stock archive); the report-only families
beside the census (the ATTN_PATH line, the KERNELS line's stack facts) are locked to the letter the consumer parses; nothing is printed per item
(the kit prints no per-item timing line).

Two of the census's own accelerators, tested directly: the gather lever's census words (gather.final_word, census.KINDS — engaged only
when every atom call was served, a distinct `partial` kind when some were refused by name, `fallback` when none were served or the kernel
could not install); and out-of-memory on the served paths (opt_core.oom.is_oom as the one classifier, the graph sampler's capture-failure
reroute re-raising an out-of-memory uncounted instead of falling back to the eager denoiser).
"""
import io
import json
import logging
import os
import re
import subprocess
import sys
import tarfile
import tempfile
import textwrap
import unittest
from unittest import mock

from .. import cli, gather, census, manifest, modes, registry, report, routes, settings
from . import _fixtures as fx

TREE_OOM = registry.tree_home()
GRAPH_SAMPLER_OOM = "opt/forward/xattempt_addon/patched/rfd3/model/cudagraph_sampler.py"
HOIST_OOM = "opt/forward/xattempt_addon/patched/rfd3/model/hoist.py"

PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))                        # the package's source directory


class _Sub:
    def parameters(self):
        return iter(())


class _FakeOptimized(_Sub):
    def __init__(self, mod=None):
        self._orig_mod = mod


class LocalAttentionPairBias:                                                                   # the atom-level attention module: the pre-flight reads its n_head
    n_head = 4


class _AtomSub(_Sub):
    def __init__(self):
        self.attn = LocalAttentionPairBias()


class _DM:                                                                                      # the diffusion module: called once per denoising step, X_noisy_L (D, L, 3)
    n_attn_keys = 128

    def forward(self, X_noisy_L=None, t=None, f=None, **kw):
        return {"X_L": X_noisy_L}

    def __call__(self, *a, **k):
        return self.forward(*a, **k)


class _T:                                                                                       # a tensor's shape / device surface
    def __init__(self, shape, device="cuda:0"):
        self.shape, self.device, self.is_cuda = tuple(shape), device, str(device).startswith("cuda")


def _step(engine, D=8, L=3210):
    """One denoising step through the located diffusion module (the samplers' per-step call): the pre-flight decision's hook."""
    return census.locate_diffusion_module(engine)(X_noisy_L=_T((D, L, 3)), t=None, f={})


def _fake_kit_hoist(applied=True):
    """The kit's hoist module as the census reads it on a fast pass (census.kit_compile_state): RFD3_COMPILE armed at import and, once a roll-out ran,
    the core record's census of an applied compile (4/4 targets, frames compiled, an empty gate)."""
    m = type(sys)("rfd3.model.hoist"); m.COMPILE = True
    m.compile_describe = lambda: {"RFD3_COMPILE": True, "installs": 1 if applied else 0, "wrapped": 4 if applied else 0, "declared": 4, "graphs": 6 if applied else 0,
                                  "graph_breaks": 8, "frames_ok": 44, "frames_total": 44, "frames_skipped": 0, "frames_skipped_allowed": 11, "recompile_limit_hits": 0, "dynamic": "auto",
                                  "targets": "encoder,diffusion_token_encoder,diffusion_transformer,decoder", "gate": ""}
    return m


_APEX = type(sys)(census.APEX_NORM_MODULE)                                                  # the apex norm module of the test process: the pinned stack carries apex, a fake engine's norms are its FusedRMSNorm
_APEX.FusedRMSNorm = type("FusedRMSNorm", (), {"__module__": census.APEX_NORM_MODULE, "__init__": lambda self, *a, **k: None})
sys.modules.setdefault(census.APEX_NORM_MODULE, _APEX)


def _engine(compile_model=False, wrap=(), targets=("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder"), locate=True, norms=None):
    """A fake engine after initialize(): trainer.state['model']._forward_module.shadow.diffusion_module with the four children, `wrap` of
    them replaced by the fake OptimizedModule; the net's `modules()` walk yields `norms` (default: two apex FusedRMSNorm instances — the
    process's `apex.normalization.fused_layer_norm` module is _APEX unless a test patches it)."""
    dm = _DM()
    for n in targets:
        setattr(dm, n, _FakeOptimized(_AtomSub()) if n in wrap else _AtomSub())
    if norms is None:
        cls = getattr(sys.modules.get(census.APEX_NORM_MODULE), "FusedRMSNorm", _APEX.FusedRMSNorm)
        try:
            norms = [cls(8), cls(8)]
        except TypeError:                                                                       # a test's own norm class without a constructor argument
            norms = [cls(), cls()]
    norms = list(norms)
    net = type("Net", (), {"diffusion_module": dm, "modules": lambda self: iter([self] + norms)})() if locate else type("NoNet", (), {})()
    ema = type("EMA", (), {"shadow": net})()
    fabric = type("Fabric", (), {"_forward_module": ema})()
    trainer = type("Trainer", (), {"state": {"model": fabric}})()
    cls = type("RFD3InferenceEngine", (), {"_COMPILE_TARGETS": tuple(targets)})
    e = cls()
    e.trainer, e.compile_model = trainer, compile_model
    return e


def _fake_dynamo_modules(disable=False, unique_graphs=4, frames=(4, 4)):
    import collections
    ef = type(sys)("torch._dynamo.eval_frame"); ef.OptimizedModule = _FakeOptimized
    cfg = type(sys)("torch._dynamo.config"); cfg.disable = disable
    utils = type(sys)("torch._dynamo.utils")
    utils.counters = collections.defaultdict(collections.Counter)
    utils.counters["stats"]["unique_graphs"] = unique_graphs
    utils.counters["frames"]["ok"], utils.counters["frames"]["total"] = frames
    utils.compilation_time_metrics = {"_compile.compile_inner": [2.0, 3.0]}
    torch = type(sys)("torch"); torch.__version__ = "2.13.0+cu130"
    torch.empty = lambda shape, device=None, **kw: _T(shape, device)                            # the pre-flight probe's zero-byte tensors
    torch.cuda = type(sys)("torch.cuda"); torch.cuda.is_available = lambda: False; torch.cuda.is_current_stream_capturing = lambda: False
    torch.cuda.mem_get_info = lambda device=None: (70 * 2 ** 30, 80 * 2 ** 30)
    att = type(sys)(census.ATTENTION_MODULE); att._DENSE_PATH_REPORTED = False
    att.use_dense_sdpa_pairbias = lambda Q, indices, full, H: os.environ.get("RFD3_DENSE_SDPA_ATTENTION", "1") == "1" and not full   # upstream's decision surface, dense unless the switch says sparse
    return {"torch": torch, "torch.cuda": torch.cuda, "torch._dynamo.eval_frame": ef, "torch._dynamo.config": cfg, "torch._dynamo.utils": utils, census.ATTENTION_MODULE: att}


UPSTREAM_RECORD = "Atom attention path: {chosen} ({reason}) [{shape}]. Set RFD3_DENSE_SDPA_ATTENTION=0 to force the original sparse path."


class TestCensusIsImportFreeOnTheStockRoute(unittest.TestCase):
    """The stock route's census runs in the PRISTINE interpreter (stock_design's child: no opt_core, no torch at census time, nothing of the package's
    mode table): importing the census and the routes table, observing a stock pass, the expectation kinds and the withheld read import neither the
    shared core nor torch nor `.modes`, in a fresh interpreter where those imports FAIL."""

    PROBE = r"""
import importlib.abc, json, os, sys

class Refuse(importlib.abc.MetaPathFinder):                     # any import of the shared core, torch, or the package's mode table / stack / report fails here
    NAMES = ("opt_core", "torch", "rfdiffusion3_opt.modes", "rfdiffusion3_opt.stack", "rfdiffusion3_opt.report")
    def find_spec(self, name, path=None, target=None):
        if name in self.NAMES or name.split(".")[0] in ("opt_core", "torch"):
            raise ImportError(f"REFUSED import of {name} on the stock route's census")

sys.meta_path.insert(0, Refuse())
from rfdiffusion3_opt import census, routes
lines = []
obs = census.Observer("stock", "off", ["inference_engine.compile_model=true", "diffusion_batch_size=8"], emit=lines.append)
out = {"route": obs.route, "route_of": routes.route_of("off", ["compile_model=true"]).name,
       "stock": census.expected_kinds("stock"), "default": census.expected_kinds("default"), "fast_compile": census.expected_kinds("fast")["compile"],
       "heavy_loaded": sorted(m for m in sys.modules if m.split(".")[0] in ("opt_core", "torch") or m in Refuse.NAMES)}
print(json.dumps(out))
"""

    def test_observe_and_expectations_import_nothing_heavy(self):
        import subprocess
        here = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))           # opt/: the package's parent, importable without an install
        env = dict(os.environ, PYTHONPATH=here + os.pathsep + os.environ.get("PYTHONPATH", ""))
        r = subprocess.run([sys.executable, "-c", self.PROBE], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr[-2000:])
        out = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(out["heavy_loaded"], [], out)                                                # nothing of the core, torch, or the mode table was imported
        self.assertEqual((out["route"], out["route_of"]), ("stock", "stock"))
        self.assertEqual(out["stock"]["compile"], "engaged"); self.assertEqual(out["default"]["compile"], "off-by-route")
        self.assertEqual(out["fast_compile"], "engaged")                                             # a route's expectation is fixed: no run-parameter moves it


class TestRoutesTable(unittest.TestCase):
    def test_every_route_names_every_accelerator_with_a_known_kind(self):
        self.assertEqual(census.ACCELERATORS, modes.KERNEL_ACCELERATORS)            # the import-free restatement is locked here
        self.assertEqual(census.ACCELERATORS[-1], "rmsnorm")                            # the fifth guarded word
        self.assertEqual(census.LAYER_UTILS_MODULE, modes.KIT_FZT_MODULE)             # one module path, restated import-free, locked here
        for name, r in modes.ROUTES.items():
            self.assertEqual(r.name, name)
            if name in ("exact", "fast"):
                self.assertEqual(tuple(r.expected), census.ACCELERATORS, name)      # the kit routes: every word judged, in the line's order
            else:                                                                    # the stock routes: upstream through its own surface — the attention path it picks (and, on `default`, the RMSNorm class its interpreter binds) are printed as found, never judged
                self.assertEqual(tuple(r.expected), tuple(a for a in census.ACCELERATORS if a != "atom_attn" and not (name == "default" and a == "rmsnorm")), name)
            for accel, kind in r.expected.items():
                self.assertIn(kind, census.KINDS, (name, accel))
            self.assertIn(r.mode, modes.MODES)
            self.assertEqual(census.expected_kinds(name), r.expected)

    def test_the_expectations_per_route(self):
        E = {n: r.expected for n, r in modes.ROUTES.items()}
        self.assertEqual(E["stock"]["compile"], "engaged")                           # STOCK = upstream's fastest documented environment: compile_model=true
        self.assertEqual(E["default"]["compile"], "off-by-route")                    # DEFAULT = upstream as shipped
        self.assertEqual(E["exact"]["compile"], "off-by-route")                      # exact's class is stated against eager upstream
        self.assertEqual(E["fast"]["compile"], "engaged")                           # fast compiles upstream's four targets itself (RFD3_COMPILE, the `engaged:kit:` word)
        self.assertEqual(set(modes.ROUTES), {"stock", "default", "exact", "fast"})   # upstream's compile_model=true composes with no kit mode (COMPILE_UNDER_FAST / _EXACT); its sparse switches are upstream's own, refused under a kit mode at activation
        for n in modes.ROUTES:
            self.assertEqual(E[n].get("atom_attn"), "engaged" if n in ("exact", "fast") else None, n)   # dense (exact) / gather (fast) expected on the kit routes; the stock routes take whatever path upstream picks
            self.assertEqual(E[n].get("rmsnorm"), None if n == "default" else "engaged", n)             # default = the README install (apex or not: printed as found); stock's image carries apex; the kit routes need it
            self.assertEqual(E[n]["cueq"], "n/a-upstream", n)
            self.assertEqual(E[n]["ds4sci"], "n/a-upstream", n)
        import dataclasses
        self.assertEqual([f.name for f in dataclasses.fields(modes.Route)], ["name", "mode", "compile", "expected", "preflight_gate", "attn_pin", "enforce", "doc"])   # routes.Route: no attention level, no booking flag
        self.assertEqual({n for n, r in modes.ROUTES.items() if r.enforce}, {"exact", "fast"})   # the kit routes REFUSE a missed expectation (exit 5); the stock routes report it and proceed (census.route_enforced)
        self.assertEqual({n: census.route_enforced(n) for n in modes.ROUTES}, {"stock": False, "default": False, "exact": True, "fast": True})
        self.assertEqual({n for n, r in modes.ROUTES.items() if r.preflight_gate}, {"exact", "fast"})   # the pre-flight dense bound stops the KIT routes (their captured / compiled dense step cannot serve sparse); the stock routes name the path and run it
        with self.assertRaises(TypeError):
            modes.route_of("off", [settings.COMPILE_TOKEN], "sparse")                     # no attention argument: the line's compile switch and the mode are the whole key
        self.assertEqual((modes.ROUTES["stock"].mode, modes.ROUTES["stock"].compile), ("off", True))
        self.assertEqual((modes.ROUTES["default"].mode, modes.ROUTES["default"].compile), ("off", False))

    def test_route_of(self):
        line = ["diffusion_batch_size=8", "seed=101"]
        self.assertEqual(modes.route_of("off", line).name, "default")
        self.assertEqual(modes.route_of("off", line + [settings.COMPILE_TOKEN]).name, "stock")
        self.assertEqual(modes.route_of("off", line + ["compile_model=True"]).name, "stock")           # hydra's other true spelling
        self.assertEqual(modes.route_of("off", line + ["++compile_model=1"]).name, "stock")
        self.assertEqual(modes.route_of("off", line + ["compile_model=false"]).name, "default")
        self.assertEqual(modes.route_of("exact", line).name, "exact")
        self.assertEqual(modes.route_of("fast", line).name, "fast")
        for mode in ("exact", "fast"):                                                    # upstream's compile switch under a kit mode: refused by name
            with self.assertRaises(ValueError) as cm:
                modes.route_of(mode, line + [settings.COMPILE_TOKEN])
            self.assertIn(f"compile_model=true under {mode}", str(cm.exception))
            self.assertEqual(modes.route_refusal(line + [settings.COMPILE_TOKEN], mode), modes.COMPILE_REFUSED[mode])
            self.assertIsNone(modes.route_refusal(line, mode))
        self.assertEqual(modes.route_refusal(line + [settings.COMPILE_TOKEN], "exact"), modes.COMPILE_UNDER_EXACT)
        self.assertIsNone(modes.route_refusal(line, "exact"))
        with self.assertRaises(ValueError):
            census.expected_kinds("big")                                          # the kit ships no big mode: no such route

class TestGrammar(unittest.TestCase):
    def test_settings_compile_token(self):
        """upstream's compile switch is read from the line's own token (settings.compile_requested: OmegaConf's truth spellings); the kit composes none."""
        self.assertEqual((settings.COMPILE_KEY, settings.COMPILE_TOKEN), ("compile_model", "compile_model=true"))
        for tok, want in (("compile_model=true", True), ("compile_model=True", True), ("compile_model=1", True), ("compile_model=false", False), ("compile_model=0", False), ("seed=101", False)):
            self.assertEqual(settings.compile_requested([tok]), want, tok)
        self.assertTrue(settings.compile_requested(["compile_model=false", "compile_model=true"]))     # the last token of a key wins, as under hydra
        self.assertFalse(hasattr(settings, "compose")); self.assertFalse(hasattr(settings, "as_args"))

    def test_run_parameters_pass_through_verbatim(self):
        """The design line is the caller's tokens, verbatim and in order (settings.line); kv_of reads them (hydra's +/~ prefixes stripped, the
        last token of a key wins, a token without `=` skipped, never refused — upstream judges its own grammar); the kit's three keys are named once."""
        self.assertEqual(settings.line([]), ["rfd3", "design"])                                     # upstream's configuration untouched
        toks = ["inputs=/s.json", "out_dir=/o", "seed=3", "n_batches=3", "diffusion_batch_size=16", "seed=5", "+extra=1", "~dropped", "ckpt_path=/w.ckpt"]
        self.assertEqual(settings.line(toks), ["rfd3", "design"] + toks)
        kv = settings.kv_of(toks)
        self.assertEqual((kv["seed"], kv["diffusion_batch_size"], kv["extra"], kv["inputs"], kv["ckpt_path"]), ("5", "16", "1", "/s.json", "/w.ckpt")); self.assertNotIn("dropped", kv)
        self.assertEqual((settings.INPUTS_KEY, settings.OUT_DIR_KEY, settings.CKPT_KEY, settings.BATCH_KEY), ("inputs", "out_dir", "ckpt_path", "diffusion_batch_size"))

    def test_the_path_record_regex_matches_the_vendored_upstream_f_string(self):
        """attention.py:403-404 in the stock archive is the record's source; the census parses exactly that shape."""
        with tarfile.open(fx.ARCHIVE, "r:gz") as tf:
            src = tf.extractfile(fx.STOCK_RFD3 + "rfd3/model/layers/attention.py").read().decode()
        self.assertIn('f"Atom attention path: {chosen} ({reason}) [{shape}]. "', src)
        self.assertIn('"Set RFD3_DENSE_SDPA_ATTENTION=0 to force the original sparse path."', src)
        for reason in ("full=True", "grad enabled (training)", "disabled via env var", "low memory mode", "not on CUDA"):
            self.assertIn(f'_report_attention_path("SPARSE", "{reason}", shape)', src, reason)      # the static reasons the census reads / traps, verbatim
        self.assertIn('_report_attention_path("SPARSE", f"L={L} <= k={k}", shape)', src)            # the data-dependent reasons: size ...
        self.assertIn('f"needs {dense_bytes / 2**30:.1f} GiB of {free_bytes / 2**30:.1f} GiB free"', src)   # ... and free memory (attention.py:447-451)
        self.assertIn('_report_attention_path("DENSE-SDPA", f"{dense_bytes / 2**30:.1f} GiB bias", shape)', src)
        self.assertIn("use_kernel = False", src)                                                 # cueq: dead code at the pin (the n/a-upstream word's evidence)
        with tarfile.open(fx.ARCHIVE, "r:gz") as tf:
            eng = tf.extractfile(fx.STOCK_RFD3 + "rfd3/engine.py").read().decode()
            pf = tf.extractfile(fx.STOCK_RFD3 + "rfd3/model/layers/pairformer_layers.py").read().decode()
        self.assertIn('_COMPILE_TARGETS = (\n        "encoder",\n        "diffusion_token_encoder",\n        "diffusion_transformer",\n        "decoder",\n    )', eng)
        self.assertIn('"Could not locate the diffusion module; skipping torch.compile."', eng)
        self.assertIn(census.COMPILE_SKIPPED_TEXT, eng.replace('"\n                "', ""))
        self.assertIn('ranked_logger.info(f"Finished inference batch in {t_end - t0:.2f} seconds.")', eng)
        self.assertIn("torch.compile(submodule, dynamic=False)", eng)
        self.assertIn("self.use_deepspeed_evo = False", pf)                                      # ds4sci: hard-off at the pin
        w = census.word_of_path_record(UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="default", shape="D=8 L=3210 k=128 H=4"))
        self.assertEqual(census.kind_of(w[0]), "engaged"); self.assertIn("dense-sdpa", w[0])
        w = census.word_of_path_record(UPSTREAM_RECORD.format(chosen="SPARSE", reason="needs 3.10 GiB of 2.00 GiB free", shape="D=8 L=9 k=128 H=4"))
        self.assertEqual(w[0], "fallback:sparse(needs-3.10-GiB-of-2.00-GiB-free)")               # one whitespace-free token
        self.assertIsNone(census.word_of_path_record("Low memory mode enabled."))

    def test_line_format(self):
        words = {"compile": "engaged:torch.compile(inductor,dynamic=False;4/4)@torch2.13.0+cu130", "atom_attn": "engaged:dense-sdpa(F.scaled_dot_product_attention)@torch2.13.0+cu130",
                 "cueq": census.CUEQ_WORD, "ds4sci": census.DS4SCI_WORD}
        line = census.kernels_line("stock", "off", "8", words, {"compile_s": 88.1, "graphs": 12, "frames": "12/12", "graph_breaks": 3, "skipped": None})
        self.assertTrue(line.startswith("[rfdiffusion3-opt] KERNELS route=stock mode=off B=8 compile=engaged:"), line)   # `KERNELS route=` is the cross-engine grep token
        toks = line.split()
        self.assertEqual(toks[1:3], ["KERNELS", "route=stock"])
        got = [t.split("=", 1)[0] for t in toks if "=" in t]
        self.assertEqual(got, ["route", "mode", "B", "compile", "atom_attn", "cueq", "ds4sci", "rmsnorm", "compile_s", "graphs", "frames", "graph_breaks"])
        for t in toks[2:]:
            self.assertRegex(t, r"^[A-Za-z0-9_]+=\S+$")                                             # every field one whitespace-free key=value token
        self.assertIn("cueq=n/a-upstream:dead-code(attention.py:282", line)
        partial = census.kernels_line("default", "off", None, {"cueq": census.CUEQ_WORD})
        self.assertIn("compile=unread atom_attn=unread", partial)                                  # a refusal printed before the fact was read
        self.assertNotIn(" B=", partial)
        self.assertEqual(census.refusal_line("stock", "compile", "fallback:x", "engaged"),
                         "[rfdiffusion3-opt] NOT ACTIVE: KERNELS refused on route stock — compile=fallback:x, expected engaged; exit 5")
        self.assertNotIn("KERNELS route=", census.refusal_line("stock", "compile", "fallback:x", "engaged"))   # that grep counts passes: one line each
        self.assertEqual(census.EXIT_KERNELS, 5)
        self.assertEqual(report.EXIT_KERNELS, census.EXIT_KERNELS)
        self.assertNotIn(census.EXIT_KERNELS, (report.EXIT_OK, report.EXIT_FAIL, report.EXIT_USAGE, report.EXIT_NOT_ACTIVE))


class TestReaders(unittest.TestCase):
    def test_census_rmsnorm(self):
        """The word names the RMSNorm class upstream's import guard bound (layer_utils.py:13-23), read from the built net's modules and the module attribute."""
        class TorchRMS: pass
        class ApexRMS: pass
        torch = type(sys)("torch"); torch.__version__ = "2.13.0+cu130"; torch.nn = type(sys)("torch.nn"); torch.nn.RMSNorm = TorchRMS
        apexmod = type(sys)(census.APEX_NORM_MODULE); apexmod.FusedRMSNorm = ApexRMS
        ext = type(sys)(census.APEX_CUDA_EXT)
        lu = type(sys)(census.LAYER_UTILS_MODULE)
        def engine_with(norms, raises=None):
            e = _engine()
            def walk():
                yield object()
                yield from norms
                if raises: raise raises
            census.locate_net(e).modules = walk
            return e
        with mock.patch.dict(sys.modules, {"torch": torch, census.LAYER_UTILS_MODULE: lu}):
            sys.modules.pop(census.APEX_NORM_MODULE, None); sys.modules.pop(census.APEX_CUDA_EXT, None)
            lu.RMSNorm_ = TorchRMS
            w, d = census.census_rmsnorm(engine_with([TorchRMS(), TorchRMS()]))
            self.assertEqual(w, "fallback:torch.nn.RMSNorm(2-modules)@torch2.13.0+cu130"); self.assertEqual((d["apex_modules"], d["torch_modules"], d["apex_imported"], d["cuda_ext_loaded"]), (0, 2, False, False))
            self.assertEqual(d["bound"], f"{TorchRMS.__module__}.{TorchRMS.__qualname__}")
            w, _ = census.census_rmsnorm(engine_with([]))
            self.assertEqual(w, "fallback:torch.nn.RMSNorm(bound;modules-unread)@torch2.13.0+cu130")       # no norm module on the walk: the module binding names the class
            w, d = census.census_rmsnorm(engine_with([TorchRMS()], raises=RuntimeError("walk broke")))
            self.assertEqual(w, "fallback:torch.nn.RMSNorm(1-modules)@torch2.13.0+cu130")                 # a walk that raises midway keeps what it counted
            e = _engine(); census.locate_net(e).modules = None                                     # a net without a callable .modules: the binding alone
            self.assertEqual(census.census_rmsnorm(e)[0], "fallback:torch.nn.RMSNorm(bound;modules-unread)@torch2.13.0+cu130")
            with mock.patch.dict(sys.modules, {census.APEX_NORM_MODULE: apexmod, census.APEX_CUDA_EXT: ext}), mock.patch("importlib.metadata.version", return_value="0.1"):
                lu.RMSNorm_ = ApexRMS
                w, d = census.census_rmsnorm(engine_with([ApexRMS()] * 3))
                self.assertEqual(w, "engaged:apex.FusedRMSNorm(3-modules+ext)@apex0.1"); self.assertEqual((d["apex_modules"], d["torch_modules"], d["apex_imported"], d["cuda_ext_loaded"]), (3, 0, True, True))
                w, _ = census.census_rmsnorm(engine_with([ApexRMS(), TorchRMS()]))
                self.assertEqual(w, "fallback:mixed(apex=1,torch=1)@apex0.1,torch2.13.0+cu130")
                self.assertEqual(census.census_rmsnorm(engine_with([]))[0], "engaged:apex.FusedRMSNorm(bound;modules-unread+ext)@apex0.1")
                self.assertEqual(census.census_rmsnorm(None, net=census.locate_net(engine_with([ApexRMS()])))[0], "engaged:apex.FusedRMSNorm(1-modules+ext)@apex0.1")   # the net handed over, no engine walk
            with mock.patch.dict(sys.modules, {census.APEX_NORM_MODULE: apexmod}), mock.patch("importlib.metadata.version", side_effect=RuntimeError("no dist")):
                sys.modules.pop(census.APEX_CUDA_EXT, None)
                self.assertEqual(census.census_rmsnorm(engine_with([ApexRMS()]))[0], "engaged:apex.FusedRMSNorm(1-modules)@apexunknown")   # a source tree without metadata
        with mock.patch.dict(sys.modules, {"torch": torch}):
            sys.modules.pop(census.LAYER_UTILS_MODULE, None); sys.modules.pop(census.APEX_NORM_MODULE, None)
            self.assertEqual(census.census_rmsnorm(_engine(locate=False))[0], census.UNREAD)         # no net, no binding: unread
            obs = census.Observer("default", "off", ["diffusion_batch_size=8"], emit=lambda l: None)
            ex = obs._extras()
            self.assertNotIn("rmsnorm", ex); self.assertNotIn("compile_s", ex); self.assertEqual(list(ex)[-1], "ckpt")   # rmsnorm is a WORD of the line (ACCELERATORS), not a tail field; the dynamo fields only on a compiled pass
            self.assertIsNone(obs.words.get("rmsnorm"))                                                # unread: not yet a word (refused at the pass's end if it stays so)
            with mock.patch.object(census, "census_rmsnorm", side_effect=ValueError("boom")):
                obs = census.Observer("default", "off", [], emit=lambda l: None); obs._read_rmsnorm(_engine())
                self.assertEqual(obs.rmsnorm, "unread:ValueError"); self.assertIn("boom", obs.detail["rmsnorm"]["error"])   # a read that raises is named, never fatal there ...
                self.assertEqual(census.kind_of(obs.words["rmsnorm"]), census.UNREAD)                   # ... and its kind is unread: the guard refuses it against `engaged`
        self.assertIn(" ds4sci=x rmsnorm=engaged:y compile_s=1.0", census.kernels_line("stock", "off", "8", {"ds4sci": "x", "rmsnorm": "engaged:y"}, {"compile_s": 1.0}))
        self.assertEqual(census.ACCELERATORS[-1], "rmsnorm"); self.assertEqual({r: census.expected_kinds(r).get("rmsnorm") for r in modes.ROUTES}, {"stock": "engaged", "default": None, "exact": "engaged", "fast": "engaged"})   # expected engaged where the pinned stack's apex is the contract (stock's image, the kit routes); `default` = the README install, printed as found

    def test_guard_refuses_a_pass_whose_norms_are_not_apex(self):
        """`rmsnorm` is a guarded word, expected `engaged` (apex's FusedRMSNorm) on every route: an apex-bound net finishes 0 everywhere; a
        torch-bound net (an interpreter without apex) is REFUSED right after initialize() on a kit route — `rmsnorm=fallback:…, expected
        engaged; exit 5` — and REPORTED on a stock route (`KERNELS report-only on route default: rmsnorm=fallback:… — upstream's run proceeds`,
        the run's own exit code); a word never read likewise at the pass's end."""
        class ApexRMS: pass
        class TorchRMS: pass
        apexmod = type(sys)(census.APEX_NORM_MODULE); apexmod.FusedRMSNorm = ApexRMS
        torch = self._torch_for_rmsnorm = type(sys)("torch"); torch.__version__ = "2.13.0+cu130"; torch.nn = type(sys)("torch.nn"); torch.nn.RMSNorm = TorchRMS
        words = {"compile": "off-by-route:compile_model=false", "atom_attn": "engaged:dense-sdpa(F.scaled_dot_product_attention)@torch2.13.0+cu130"}
        for route in modes.ROUTES:
            if route in ("stock", "fast"):
                continue                                                                            # (stock and fast also expect compile engaged — upstream's form, the kit's form; the rmsnorm rule is the same there)
            enforced = modes.ROUTES[route].enforce                                                  # exact: refused (5); default: no rmsnorm expectation at all — the word rides the KERNELS line as found, the run's rc (0), no report line
            miss_rc, tail, head = (5, "; exit 5", f"NOT ACTIVE: KERNELS refused on route {route} — ") if enforced else (0, "", "")
            for norms, mods, want_rc, want in (([ApexRMS(), ApexRMS()], {census.APEX_NORM_MODULE: apexmod}, 0, "rmsnorm=engaged:apex.FusedRMSNorm(2-modules)@apex0.1"),
                                                ([TorchRMS(), TorchRMS()], {"torch": torch}, miss_rc, head + "rmsnorm=fallback:torch.nn.RMSNorm(2-modules)@torch2.13.0+cu130" + (", expected engaged" + tail if enforced else " ")),
                                                ([], {}, miss_rc, (head + "rmsnorm=unread, expected engaged" + tail) if enforced else " rmsnorm=unread ")):
                census.reset(); lines = []
                with mock.patch.dict(sys.modules, mods), mock.patch("importlib.metadata.version", return_value="0.1"):
                    for k in [census.APEX_NORM_MODULE, census.APEX_CUDA_EXT, census.LAYER_UTILS_MODULE]:
                        if k not in mods: sys.modules.pop(k, None)
                    obs = census.Observer(route, modes.ROUTES[route].mode, ["diffusion_batch_size=8"], emit=lines.append)
                    e = _engine(norms=norms)
                    obs.words.update(words)
                    rc = None
                    try:
                        obs.after_initialize(e)
                        if modes.ROUTES[route].preflight_gate:
                            obs.preflight.append({"route": route, "B": 8, "n_atoms": 999, "L": 100, "atom_sparse": 0, "dense": True})   # (the pre-flight read this test does not exercise)
                        obs.items.append({"item": "batch:0", "clock_s": "1.0", "B": "8"})
                        rc = obs.finish(0)
                    except SystemExit as ex_:
                        rc = ex_.code
                self.assertEqual(rc, want_rc, (route, want, lines))
                self.assertTrue(any(want in l for l in lines), (route, want, lines))
                if not enforced:
                    self.assertFalse([l for l in lines if "report-only" in l or "NOT ACTIVE" in l], (route, lines))   # the stock routes judge no RMSNorm class

    def test_census_compile(self):
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            w, d = census.census_compile(_engine(False))
            self.assertEqual(w, "off-by-route:compile_model=false")
            w, d = census.census_compile(_engine(True, wrap=("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")))
            self.assertEqual(w, "engaged:torch.compile(inductor,dynamic=False;4/4)@torch2.13.0+cu130", d)
            self.assertEqual(d["wrapped"], ["encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder"])
            w, _ = census.census_compile(_engine(True, wrap=("encoder",)))                        # the flag on, three children eager: not the flag, the objects
            self.assertEqual(w, "fallback:not-wrapped(1/4:missing=diffusion_token_encoder,diffusion_transformer,decoder)")
            w, _ = census.census_compile(_engine(True, locate=False))                             # upstream's 'Could not locate the diffusion module' case
            self.assertEqual(w, "fallback:diffusion-module-not-located(engine.py:247-251)")
            w, _ = census.census_compile(_engine(False, wrap=("encoder",)))
            self.assertTrue(w.startswith("fallback:wrapped-without-compile_model"), w)
            w, _ = census.census_compile(_engine(True, targets=()))
            self.assertTrue(w.startswith("absent:"), w)
        with mock.patch.dict(sys.modules, _fake_dynamo_modules(disable=True)):
            w, _ = census.census_compile(_engine(True, wrap=("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")))
            self.assertEqual(w, "fallback:dynamo-disabled(torch._dynamo.config.disable)")         # wrapped objects that run eager: a silent fallback, named

    def test_static_atom_attention_reasons(self):
        self.assertIsNone(census.census_atom_attn_static({}, "cuda"))
        self.assertEqual(census.census_atom_attn_static({"RFD3_DENSE_SDPA_ATTENTION": "0"}), "fallback:sparse(disabled-via-RFD3_DENSE_SDPA_ATTENTION=0)")
        self.assertEqual(census.census_atom_attn_static({"RFD3_LOW_MEMORY_MODE": "1"}), "fallback:sparse(low-memory-mode,RFD3_LOW_MEMORY_MODE=1)")
        self.assertEqual(census.census_atom_attn_static({}, "cpu"), "fallback:sparse(not-on-CUDA,device=cpu)")

    def test_dynamo_counters(self):
        with mock.patch.dict(sys.modules, _fake_dynamo_modules(unique_graphs=7, frames=(6, 7))):
            c = census.dynamo_counters()
        self.assertEqual((c["unique_graphs"], c["frames_ok"], c["frames_total"], c["compile_s"]), (7, 6, 7, 5.0))
        with mock.patch.dict(sys.modules, {"torch._dynamo.utils": None}):
            self.assertIsNone(census.dynamo_counters()["unique_graphs"])



class TestReportOnlyWords(unittest.TestCase):
    """The report-only families beside the census: the counting shim behind ATTN_PATH (transparent by construction) and the KERNELS line's
    stack facts — grammar to the letter, and the census record that carries them; upstream's clock records are kept as data, never printed."""

    CONSUMER_ATTN_RE = r"^\[rfdiffusion3-opt\] ATTN_PATH atom_dense=(\d+) atom_sparse=(\d+) token_full=(\d+) compiled=([01]) calls_uncounted=([01])$"
    CONSUMER_PREFLIGHT_RE = r"^\[rfdiffusion3-opt\] ATTN_PREFLIGHT route=(\w+) B=(\d+) n_atoms=(\d+) L=(\d+|unread) atom_sparse=([01]) free_gib=([\d.]+) est_dense_gib=([\d.]+) pinned=([01])$"   # the line's full grammar; the consumer's activation rules are prefix matches ending at `atom_sparse=0 ` (an external activation matcher's rules end there)

    def setUp(self):
        census.reset()
        self.lines = []
        self.td = tempfile.TemporaryDirectory()
        self.env = mock.patch.dict(os.environ, {"RFDIFFUSION3_OPT_CACHE": os.path.join(self.td.name, "cache")})   # the digest cache of ckpt= stays in the sandbox
        self.env.start()

    def tearDown(self):
        self.env.stop()
        census.reset()
        self.td.cleanup()

    def test_dense_bias_estimate_is_upstreams_own_expression(self):
        """est_dense_gib restates attention.py:445 — locked against the vendored upstream file, and the arithmetic spelled out."""
        self.assertEqual(census.dense_bias_bytes(2, 4, 999), 2 * 2 * 4 * 999 * 999 * 2)
        self.assertEqual(census.dense_bias_bytes(16, 4, 2456), 2 * 16 * 4 * 2456 * 2456 * 2)
        with tarfile.open(fx.ARCHIVE, "r:gz") as tf:
            src = tf.extractfile(fx.STOCK_RFD3 + "rfd3/model/layers/attention.py").read().decode()
            dm = tf.extractfile(fx.STOCK_RFD3 + "rfd3/model/RFD3_diffusion_module.py").read().decode()
        self.assertIn("    dense_bytes = 2 * D * H * L * L * 2\n", src)                            # the estimate the 0.35 x free rule compares (attention.py:445-446)
        self.assertIn("    if dense_bytes >= 0.35 * free_bytes:\n", src)
        self.assertIn("    free_bytes, _ = torch.cuda.mem_get_info(Q.device)\n", src)
        self.assertIn("    D, L, _ = Q.shape\n    k = indices.shape[-1]\n", src)                    # what the pre-flight probe's zero-byte tensors must carry: shapes only
        self.assertRegex(dm, r"def forward\(\s*self,\s*X_noisy_L,")                              # DM_X_PARAM: the step call's (D, L, 3) argument
        self.assertIn("n_attn_keys=self.n_attn_keys,", dm)                                        # DM_KEYS_ATTR
        self.assertEqual((census.DM_X_PARAM, census.DM_KEYS_ATTR, census.ATOM_ATTENTION_CLASS), ("X_noisy_L", "n_attn_keys", "LocalAttentionPairBias"))
        self.assertIn("class LocalAttentionPairBias(nn.Module):", src)

    def test_graph_replays_and_atom_heads(self):
        g = type(sys)("rfd3.model.cudagraph_sampler"); g.GRAPH_STATS = {"installs": 1, "captures": 1, "hits": 1, "fallbacks": 0}
        with mock.patch.dict(sys.modules, {"rfd3.model.cudagraph_sampler": g}):
            self.assertEqual(census.graph_replays(), 2)
        self.assertEqual(census.graph_replays(), 0)                                               # module absent: nothing replayed
        A = type("LocalAttentionPairBias", (), {}); a4, a2, tok = A(), A(), A(); a4.n_head, a2.n_head, tok.n_head = 4, 2, 16
        enc = type("Enc", (), {})(); enc.attn = a4
        dec = type("Dec", (), {})(); dec.attn = a2
        wrapped = type("OptimizedModule", (), {})(); wrapped._orig_mod = dec                     # a compiled child: unwrapped to the module it wraps
        trf = type("Trf", (), {})(); trf.attn = tok                                               # the token-level transformer's heads are not the atom decision's
        dm = type("DM", (), {})(); dm.encoder, dm.decoder, dm.diffusion_transformer = enc, wrapped, trf
        self.assertEqual(sorted(census.atom_attention_heads(dm)), [2, 4])

    def _preflight_rig(self, route, dense=True, free_gib=71.5, kit_file=False):
        """An observer on `route`, a fake torch with empty / cuda.mem_get_info, upstream's decision surface as a module, a diffusion module
        object with forward / __call__ / n_attn_keys / encoder.attn — the pre-flight wrapper installed on it."""
        torch = type(sys)("torch")
        class T:
            def __init__(self, shape, device=None): self.shape, self.device, self.is_cuda = tuple(shape), device, str(device).startswith("cuda")
        torch.empty = lambda shape, device=None, **kw: T(shape, device)
        torch.cuda = type(sys)("torch.cuda"); torch.cuda.is_current_stream_capturing = lambda: False; torch.cuda.is_available = lambda: False
        torch.cuda.mem_get_info = lambda device=None: (int(free_gib * 2 ** 30), 80 * 2 ** 30)
        att = type(sys)(census.ATTENTION_MODULE); att._DENSE_PATH_REPORTED = False; att.calls = []; att.memo_calls = []
        def use_dense_sdpa_pairbias(Q, indices, full, H):
            att.calls.append((tuple(Q.shape), tuple(indices.shape), full, H, att._DENSE_PATH_REPORTED)); return dense
        att.use_dense_sdpa_pairbias = use_dense_sdpa_pairbias
        if kit_file:                                                                                 # the kit's patched file: the public entry memoises per shape under the graph sampler; the rule keeps upstream's body
            def _memo_entry(Q, indices, full, H):
                att.memo_calls.append((tuple(Q.shape), full)); return dense
            att.use_dense_sdpa_pairbias = _memo_entry
            att._use_dense_sdpa_pairbias_stock = use_dense_sdpa_pairbias
        A = type("LocalAttentionPairBias", (), {}); a = A(); a.n_head = 4
        class DM:
            n_attn_keys = 128
            def __init__(s): s.encoder = type("E", (), {})(); s.encoder.attn = a; s.decoder = None; s.steps = 0
            def forward(s, X_noisy_L, t=None, f=None, **kw): s.steps += 1; return {"X_L": X_noisy_L}
            def __call__(s, *a_, **k_): return s.forward(*a_, **k_)
        dm = DM()
        obs = census.Observer(route, "off" if route in ("stock", "default") else route, ["out_dir=" + self.td.name], emit=self.lines.append)
        mods = {"torch": torch, "torch.cuda": torch.cuda, census.ATTENTION_MODULE: att}
        return obs, dm, att, torch, T, mods

    def test_preflight_runs_upstreams_decision_once_per_shape_before_the_step(self):
        obs, dm, att, torch, T, mods = self._preflight_rig("stock", dense=True)
        with mock.patch.dict(sys.modules, mods):
            obs._wrap_attention(att)                                                               # the counting shim is on: the probe must bypass it
            obs._install_preflight(dm); obs._install_preflight(dm)                                # idempotent
            self.assertTrue(dm.forward._kernels_observed); self.assertEqual(obs.detail["preflight"], "installed")
            out = dm(X_noisy_L=T((16, 2456, 3), "cuda:0"), t=None, f={})                          # keyword, as the samplers call it
            self.assertEqual(out["X_L"].shape, (16, 2456, 3)); self.assertEqual(dm.steps, 1)     # the step ran, its arguments and value untouched
            dm(T((16, 2456, 3), "cuda:0")); dm(X_noisy_L=T((4, 2456, 3), "cuda:0"))               # same shape: nothing; a new D (the last, smaller diffusion batch): a second decision
        self.assertEqual(self.lines, ["[rfdiffusion3-opt] ATTN_PREFLIGHT route=stock B=16 n_atoms=2456 L=unread atom_sparse=0 free_gib=71.500 est_dense_gib=%.3f pinned=0" % (2 * 16 * 4 * 2456 * 2456 * 2 / 2 ** 30),
                                      "[rfdiffusion3-opt] ATTN_PREFLIGHT route=stock B=4 n_atoms=2456 L=unread atom_sparse=0 free_gib=71.500 est_dense_gib=%.3f pinned=0" % (2 * 4 * 4 * 2456 * 2456 * 2 / 2 ** 30)])
        for line in self.lines:
            self.assertRegex(line, self.CONSUMER_PREFLIGHT_RE)
        self.assertEqual(att.calls, [((16, 2456, 0), (0, 0, 128), False, 4, True), ((4, 2456, 0), (0, 0, 128), False, 4, True)])   # zero-byte probes at the step's shape; the once-flag held during the probe ...
        self.assertFalse(att._DENSE_PATH_REPORTED)                                                 # ... and restored: upstream's own first call still logs its record
        self.assertEqual(obs.attn_counts, {"atom_dense": 0, "atom_sparse": 0, "token_full": 0})   # the probe is not a counted call
        self.assertEqual([(p["B"], p["n_atoms"], p["L"], p["k"], p["H"], p["dense"]) for p in obs.preflight], [(16, 2456, None, 128, 4, True), (4, 2456, None, 128, 4, True)])   # L (tokens) unread: the rig passes no token map
        self.assertIsNone(obs.refused)
        census.reset(); self.lines.clear()
        obs, dm, att, torch, T, mods = self._preflight_rig("fast", dense=True, kit_file=True)      # on the kit's patched file the probe calls the RULE, never the memoising entry the model's calls go through
        with mock.patch.dict(sys.modules, mods):
            obs._wrap_attention(att); obs._install_preflight(dm)
            dm(X_noisy_L=T((8, 1400, 3), "cuda:0"))
        self.assertEqual((len(att.calls), att.memo_calls), (1, []))
        self.assertEqual(self.lines, ["[rfdiffusion3-opt] ATTN_PREFLIGHT route=fast B=8 n_atoms=1400 L=unread atom_sparse=0 free_gib=71.500 est_dense_gib=%.3f pinned=1" % (2 * 8 * 4 * 1400 * 1400 * 2 / 2 ** 30)])   # a kit route: the decision is pinned for the process
        self.assertEqual(obs.attn_pins, {(8, 1400, 128, 4): True})
        self.assertEqual(census.DECISION_RULE_KIT, "_use_dense_sdpa_pairbias_stock")
        patched = open(os.path.join(fx.KIT, "patched", "rfd3", "model", "layers", "attention.py"), encoding="utf-8").read()
        self.assertIn("def _use_dense_sdpa_pairbias_stock(Q, indices, full, H) -> bool:", patched)   # the kit file's rule ...
        self.assertIn("return _cg.cg_dense_decision(D, L, k, H, full, lambda: _use_dense_sdpa_pairbias_stock(Q, indices, full, H))", patched)   # ... behind the per-shape memo of its public entry
        with tarfile.open(fx.ARCHIVE, "r:gz") as tf:
            self.assertNotIn("_use_dense_sdpa_pairbias_stock", tf.extractfile(fx.STOCK_RFD3 + "rfd3/model/layers/attention.py").read().decode())   # the pristine file: the public function is the rule

    def test_preflight_sparse_refuses_the_gated_routes_only(self):
        for route, gated in (("stock", False), ("fast", True), ("exact", True), ("default", False)):   # the kit routes only (exact replays captured steps: gated like fast); compiled stock takes upstream's path
            census.reset(); self.lines.clear()
            obs, dm, att, torch, T, mods = self._preflight_rig(route, dense=False, free_gib=3.25)
            self.assertEqual(census.preflight_gated(route), gated); self.assertEqual(modes.ROUTES[route].preflight_gate, gated)   # the routes table is the one source
            with mock.patch.dict(sys.modules, mods):
                obs._install_preflight(dm)
                if gated:
                    with self.assertRaises(census.KernelsRefused) as cm:
                        dm(X_noisy_L=T((8, 999, 3), "cuda:0"))
                    self.assertEqual(cm.exception.code, census.EXIT_KERNELS)
                    self.assertEqual(dm.steps, 0)                                                  # refused before the step ran
                    self.assertEqual([l.split()[1] for l in self.lines], ["ATTN_PREFLIGHT", "KERNELS", "NOT"])
                    self.assertIn(f"NOT ACTIVE: KERNELS refused on route {route} — attn_preflight=sparse(B=8,n_atoms=999,free_gib=3.250,est_dense_gib=", self.lines[-1])
                    self.assertEqual(obs.refused["accel"], "attn_preflight")
                else:
                    dm(X_noisy_L=T((8, 999, 3), "cuda:0"))
                    self.assertEqual(dm.steps, 1); self.assertIsNone(obs.refused)
                    self.assertEqual(self.lines, [f"[rfdiffusion3-opt] ATTN_PREFLIGHT route={route} B=8 n_atoms=999 L=unread atom_sparse=1 free_gib=3.250 est_dense_gib=%.3f pinned=0" % (2 * 8 * 4 * 999 * 999 * 2 / 2 ** 30)])

    def _rule_rig(self, route, free_gib):
        """The pre-flight rig with upstream's RULE as the decision surface (attention.py:417-454, its terms verbatim: the force-sparse switches, L <= k,
        dense iff 2·D·H·L·L·2 < 0.35 × torch.cuda.mem_get_info free) and its once-per-process path record, on a box whose free memory the test moves."""
        obs, dm, att, torch, T, mods = self._preflight_rig(route, kit_file=True)
        state = {"free_gib": free_gib}
        torch.cuda.mem_get_info = lambda device=None: (int(state["free_gib"] * 2 ** 30), 80 * 2 ** 30)
        att.records = []
        def _report_attention_path(chosen, reason, shape):
            if att._DENSE_PATH_REPORTED: return
            att._DENSE_PATH_REPORTED = True; att.records.append(f"Atom attention path: {chosen} ({reason}) [{shape}].")
        def rule(Q, indices, full, H):
            D, L, _ = Q.shape; k = indices.shape[-1]; shape = f"D={D} L={L} k={k} H={H}"; att.calls.append((D, L, k, H, full))
            if full: _report_attention_path("SPARSE", "full=True", shape); return False
            if os.environ.get("RFD3_DENSE_SDPA_ATTENTION", "1") == "0": _report_attention_path("SPARSE", "disabled via env var", shape); return False
            if os.environ.get("RFD3_LOW_MEMORY_MODE", "0") == "1": _report_attention_path("SPARSE", "low memory mode", shape); return False
            if L <= k: _report_attention_path("SPARSE", f"L={L} <= k={k}", shape); return False
            free_bytes, _ = torch.cuda.mem_get_info(Q.device); dense_bytes = 2 * D * H * L * L * 2
            if dense_bytes >= 0.35 * free_bytes: _report_attention_path("SPARSE", f"needs {dense_bytes / 2**30:.1f} GiB of {free_bytes / 2**30:.1f} GiB free", shape); return False
            _report_attention_path("DENSE-SDPA", f"{dense_bytes / 2**30:.1f} GiB bias", shape); return True
        att._report_attention_path = _report_attention_path
        att._use_dense_sdpa_pairbias_stock = rule                                                    # the kit file's rule ...
        att.use_dense_sdpa_pairbias = lambda Q, indices, full, H: rule(Q, indices, full, H)          # ... and its public entry (the memo is the graph sampler's; eager here)
        return obs, dm, att, torch, T, mods, state

    def test_the_preflight_decision_is_pinned_for_the_process(self):
        """L500 B=16 on an H100 (the reference shape): the pre-flight, before the step's activations, reads 39.45 GiB free — upstream's rule says dense
        (11.7 < 0.35 × 39.45); the model's own call runs with the activations live at 31.5 GiB free, where the rule per call would say sparse (11.7 ≥ 11.0).
        Pinned: the call answers dense from the pre-flight WITHOUT evaluating the rule, is counted, and makes upstream's path record in its own words with
        the pin named; a key the pre-flight did not evaluate runs the rule per call (a miss, counted); a token call (full) is never pinned."""
        obs, dm, att, torch, T, mods, state = self._rule_rig("exact", free_gib=39.45)
        self.assertTrue(obs.pin); self.assertTrue(census.attn_pinned("exact")); self.assertFalse(census.attn_pinned("stock")); self.assertFalse(census.attn_pinned("default"))
        with mock.patch.dict(sys.modules, mods):
            obs._wrap_attention(att); obs._install_preflight(dm)
            dm(X_noisy_L=T((16, 7000, 3), "cuda:0"))                                                # the first step of the shape: the pre-flight evaluates and pins
            self.assertEqual(self.lines, ["[rfdiffusion3-opt] ATTN_PREFLIGHT route=exact B=16 n_atoms=7000 L=unread atom_sparse=0 free_gib=39.450 est_dense_gib=%.3f pinned=1" % (2 * 16 * 4 * 7000 * 7000 * 2 / 2 ** 30)])
            self.assertEqual(obs.attn_pins, {(16, 7000, 128, 4): True}); self.assertEqual(att.records, [])   # the probe made no path record
            n_rule = len(att.calls)
            state["free_gib"] = 31.5                                                                 # the step's activations are live now
            self.assertIs(att.use_dense_sdpa_pairbias(Q=T((16, 7000, 128), "cuda:0"), indices=T((16, 7000, 128)), full=False, H=4), True)   # the model's call: dense, from the pin ...
            self.assertEqual(len(att.calls), n_rule)                                                 # ... the rule not evaluated again
            self.assertEqual(att.records, ["Atom attention path: DENSE-SDPA (11.7 GiB bias; pinned at pre-flight) [D=16 L=7000 k=128 H=4]."])   # upstream's once-per-process record, the pin named
            self.assertIs(att._use_dense_sdpa_pairbias_stock(Q=T((16, 7000, 128), "cuda:0"), indices=T((16, 7000, 128)), full=False, H=4), False)   # (what the rule per call says at this free memory)
            self.assertIs(att.use_dense_sdpa_pairbias(T((16, 7000, 128), "cuda:0"), T((16, 7000, 128)), False, 4), True)   # positional: the same key, the same pin
            self.assertIs(att.use_dense_sdpa_pairbias(Q=T((16, 500, 32), "cuda:0"), indices=T((16, 500, 500)), full=True, H=16), False)   # the token transformer's call: the rule's (never a memory decision)
            self.assertIs(att.use_dense_sdpa_pairbias(Q=T((32, 7000, 128), "cuda:0"), indices=T((32, 7000, 128)), full=False, H=4), False)   # a key the pre-flight did not evaluate (another D): the rule, per call (23.4 ≥ 0.35 × 31.5: sparse)
        self.assertEqual(obs.attn_counts, {"atom_dense": 2, "atom_sparse": 1, "token_full": 1, "pin_hits": 2, "pin_misses": 1})
        rec = obs._attn_record()
        self.assertEqual((rec["pin_hits"], rec["pin_misses"], rec["preflight"][0]["pinned"], rec["preflight"][0]["decisions"]), (2, 1, 1, {"4": True}))
        self.assertIsNone(obs.refused)

    def test_every_atom_head_count_is_pinned(self):
        """Encoder and decoder atom attention of different head counts: the rule is evaluated once per distinct H at the pre-flight (largest on the line), each pinned."""
        obs, dm, att, torch, T, mods, state = self._rule_rig("fast", free_gib=60.0)
        A = type("LocalAttentionPairBias", (), {}); a2 = A(); a2.n_head = 2
        dm.decoder = type("D", (), {})(); dm.decoder.attn = a2
        with mock.patch.dict(sys.modules, mods):
            obs._wrap_attention(att); obs._install_preflight(dm)
            dm(X_noisy_L=T((8, 3000, 3), "cuda:0"))
        self.assertEqual(obs.attn_pins, {(8, 3000, 128, 4): True, (8, 3000, 128, 2): True})
        self.assertEqual([c[3] for c in att.calls], [4, 2]); self.assertIn(" B=8 n_atoms=3000 L=unread atom_sparse=0 ", self.lines[0]); self.assertTrue(self.lines[0].endswith(" pinned=1"))
        self.assertEqual(obs.preflight[0]["decisions"], {"2": True, "4": True}); self.assertEqual(obs.preflight[0]["H"], 4)

    def test_upstreams_force_sparse_switches_decide_the_preflight_itself(self):
        """RFD3_DENSE_SDPA_ATTENTION=0 / RFD3_LOW_MEMORY_MODE=1 make upstream's rule say sparse at the pre-flight: a gated route refuses BEFORE the first step,
        as without the pin, and no dense pin is ever written."""
        for env in ({"RFD3_DENSE_SDPA_ATTENTION": "0"}, {"RFD3_LOW_MEMORY_MODE": "1"}):
            census.reset(); self.lines.clear()
            with mock.patch.dict(os.environ, env):
                obs, dm, att, torch, T, mods, state = self._rule_rig("fast", free_gib=70.0)
                with mock.patch.dict(sys.modules, mods):
                    obs._wrap_attention(att); obs._install_preflight(dm)
                    with self.assertRaises(census.KernelsRefused):
                        dm(X_noisy_L=T((8, 999, 3), "cuda:0"))
                self.assertEqual(dm.steps, 0); self.assertEqual(obs.refused["accel"], "attn_preflight")
                self.assertTrue(self.lines[0].endswith(" pinned=1")); self.assertIn(" atom_sparse=1 ", self.lines[0])
                self.assertEqual(obs.attn_pins, {(8, 999, 128, 4): False}); self.assertNotIn(True, obs.attn_pins.values())

    def test_a_pinned_dense_path_that_does_not_fit_raises(self):
        """The pin never guards the attention call: an out-of-memory error of the dense path propagates out of the model's call (the shim only answers the
        decision; house rule: no try-then-fallback to sparse)."""
        obs, dm, att, torch, T, mods, state = self._rule_rig("exact", free_gib=39.45)
        class OutOfMemoryError(RuntimeError): pass
        with mock.patch.dict(sys.modules, mods):
            obs._wrap_attention(att); obs._install_preflight(dm)
            dm(X_noisy_L=T((16, 7000, 3), "cuda:0"))
            def attention_call():                                                                    # the model's call site: the decision, then the dense kernel
                if att.use_dense_sdpa_pairbias(Q=T((16, 7000, 128), "cuda:0"), indices=T((16, 7000, 128)), full=False, H=4):
                    raise OutOfMemoryError("CUDA out of memory. Tried to allocate 11.68 GiB")
            with self.assertRaises(OutOfMemoryError):
                attention_call()
        src = open(census.__file__, encoding="utf-8").read()
        shim_src = src[src.index("    def use_dense_sdpa_pairbias(*args, **kwargs):"): src.index("    use_dense_sdpa_pairbias._kernels_observed = True")]
        self.assertNotIn("try:", shim_src); self.assertNotIn("except", shim_src)                     # the call wrapper carries no handler at all

    def test_kit_compile_gate_sentence_refuses_the_fast_pass(self):
        """The kit record's gate sentences (opt_core CompileRecord.gate: a recompile-limit hit, a target not wrapped, no frame compiled — compiled code
        did not serve: the lever cannot run as the mode names it) make the compile word a fallback on route fast: `fallback:kit-compile-gate(<sentences>)`,
        expected engaged — refused by name by the guard (a mode is all of its levers)."""
        hoist = _fake_kit_hoist()
        d = hoist.compile_describe(); d.update(recompile_limit_hits=2, gate="RFD3_COMPILE: recompile_limit_hits=2 (dynamo fell back to eager past cache_size_limit=64)")
        hoist.compile_describe = lambda: d
        with mock.patch.dict(sys.modules, dict(_fake_dynamo_modules(), **{"rfd3.model.hoist": hoist})):
            word, detail = census.census_compile(_engine(False))
        self.assertTrue(word.startswith("fallback:kit-compile-gate(RFD3_COMPILE: recompile_limit_hits=2"), word)
        self.assertEqual(census.kind_of(word), "fallback")
        self.assertEqual(census.refusal_line("fast", "compile", word, "engaged"), f"[rfdiffusion3-opt] NOT ACTIVE: KERNELS refused on route fast — compile={word}, expected engaged; exit 5")

    def test_frames_skipped_above_the_allowance_is_named_not_refused(self):
        """A torch / dynamo other than the pinned one skips a different number of frames — an untested library version, not a lever that cannot run:
        the count is NAMED — one `COMPILE frames_skipped=<n> above the <allowed> measured on the pinned stack …` line at the pass's end — the compile
        word stays what the graphs say (engaged: every target wrapped, compiled frames ran) and the pass is not refused. The patched hoist no longer
        gates frames (compile_describe: gate without max_frames_skipped); on the pinned stack the count equals the allowance and no line prints."""
        hoist = _fake_kit_hoist()
        d = hoist.compile_describe(); d.update(frames_ok=40, frames_total=55, frames_skipped=15, gate="")   # 15 > the 11 measured; the record's gate is empty (frames are not gated)
        hoist.compile_describe = lambda: d
        obs = census.Observer("fast", "fast", ["diffusion_batch_size=8"], emit=self.lines.append)
        mods = dict(_fake_dynamo_modules(), **{"rfd3.model.hoist": hoist})
        mods["torch.compiler"] = mods["torch"].compiler = type(sys)("torch.compiler"); mods["torch.compiler"].disable = lambda fn=None, recursive=True: fn
        with mock.patch.dict(sys.modules, mods):
            obs.after_initialize(_engine(False))
            self.assertEqual(census.kind_of(obs.words["compile"]), "engaged", obs.words["compile"])
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            obs.preflight.append({"route": "fast", "B": 8, "n_atoms": 999, "L": 100, "atom_sparse": 0, "dense": True})
            obs.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("9.50", "seconds"))
            self.assertEqual(obs.finish(0), 0)                                                     # not refused: the run's own code
        note = [l for l in self.lines if l.startswith("[rfdiffusion3-opt] COMPILE ")]
        self.assertEqual(note, ["[rfdiffusion3-opt] COMPILE frames_skipped=15 above the 11 measured on the pinned stack (torch 2.13.0+cu130): named — those frames run eager, the compiled graphs serve the rest"])
        self.assertEqual(obs.detail["compile_drift_note"], {"frames_skipped": 15, "frames_skipped_allowed": 11}); self.assertIsNone(obs.refused)
        self.assertNotIn("NOT ACTIVE", "\n".join(self.lines))
        d.update(frames_skipped=11, frames_total=51); self.lines.clear()                             # at the allowance (the pinned stack's count): no line
        obs2 = census.Observer("fast", "fast", ["diffusion_batch_size=8"], emit=self.lines.append)
        with mock.patch.dict(sys.modules, mods):
            obs2.after_initialize(_engine(False)); obs2.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            obs2.preflight.append({"route": "fast", "B": 8, "n_atoms": 999, "L": 100, "atom_sparse": 0, "dense": True}); obs2.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("9.50", "seconds"))
            self.assertEqual(obs2.finish(0), 0)
        self.assertEqual([l for l in self.lines if l.startswith("[rfdiffusion3-opt] COMPILE ")], [])
        src = open(os.path.join(registry.tree_home(), "opt", "forward", "xattempt_addon", "patched", "rfd3", "model", "hoist.py"), encoding="utf-8").read()
        self.assertIn("COMPILE_RECORD.gate(require_frames=True, max_frames_skipped=None)", src)   # the patched module does not gate frames

    def test_a_gated_route_that_ran_with_no_preflight_read_is_refused_at_finish(self):
        """No allowance for a vacuous gate: roll-outs on a gated route without a pre-flight decision read -> attn_preflight=unread(...), exit 5."""
        obs = census.Observer("fast", "fast", ["diffusion_batch_size=8"], emit=self.lines.append)
        mods = _fake_dynamo_modules(); mods["rfd3.model.hoist"] = _fake_kit_hoist()                # a fast pass: the kit's compile armed and applied (compile=engaged:kit), so the refusal below is the pre-flight's alone
        mods["torch.compiler"] = mods["torch"].compiler = type(sys)("torch.compiler"); mods["torch.compiler"].disable = lambda fn=None, recursive=True: fn   # the boundary marker the counting shim takes under the kit's compile (opt_core.capture.compile.nodynamo)
        with mock.patch.dict(sys.modules, mods):
            engine = _engine(False)
            census.locate_diffusion_module(engine).__dict__["forward"] = None                     # a diffusion module without a forward to wrap: not installed, by name
            obs.after_initialize(engine)
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            obs.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("9.50", "seconds"))
            self.assertEqual(obs.finish(0), census.EXIT_KERNELS)
        self.assertIn("refused on route fast — attn_preflight=unread(not-installed(diffusion-module-has-no-forward)), expected dense; exit 5", self.lines[-1])
        census.reset(); self.lines.clear()
        obs = census.Observer("stock", "off", ["diffusion_batch_size=8", settings.COMPILE_TOKEN], emit=self.lines.append)   # the compiled STOCK route is not gated: the same pass (installed, no step observed) is clean there
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            obs.after_initialize(_engine(True, wrap=("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")))
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            obs.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("9.50", "seconds"))
            self.assertEqual(obs.finish(0), 0)
        self.assertNotIn("attn_preflight=", "\n".join(self.lines)); self.assertIsNone(obs.refused)
        census.reset(); self.lines.clear()
        obs = census.Observer("default", "off", ["diffusion_batch_size=8"], emit=self.lines.append)   # not gated (the eager route): the same pass is clean
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            obs.after_initialize(_engine(False))
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            obs.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("9.50", "seconds"))
            self.assertEqual(obs.finish(0), 0)

    def test_every_census_line_starts_a_line_after_a_progress_bar(self):
        """upstream's tqdm bar rewrites its line with \\r and ends without a newline; the package's one printer puts every census line at a
        line start regardless (a consumer splits on \\n — and \\r — and matches the tag at the start)."""
        import io
        from .. import _emit, report, stock_design
        buf = io.StringIO()
        buf.write("\rSampling:  99%|#########9| 198/200 [00:19<00:00, 10.31it/s]\rSampling: 100%|##########| 200/200 [00:19<00:00, 10.30it/s]")   # a tqdm fragment, no newline
        _emit.emit(census.kernels_line("default", "off", "8", {"compile": "off-by-route:compile_model=false", "atom_attn": "engaged:x@torch2", "cueq": census.CUEQ_WORD, "ds4sci": census.DS4SCI_WORD}), stream=buf)
        _emit.emit(census.attn_path_line({}), stream=buf)
        text = buf.getvalue()
        starts = [l for l in text.split("\n") if l.startswith("[rfdiffusion3-opt] ")]
        self.assertEqual([l.split()[1] for l in starts], ["KERNELS", "ATTN_PATH"])
        also = [seg for l in text.split("\n") for seg in l.split("\r") if seg.startswith("[rfdiffusion3-opt] ")]   # a reader that also splits on \r sees the same two
        self.assertEqual(len(also), 2)
        self.assertEqual(_emit.fresh("x"), "\nx"); self.assertIs(report.emit, _emit.emit); self.assertIs(stock_design.emit, _emit.emit); self.assertIs(census._default_emit, _emit.emit)
        srcs = {n: open(os.path.join(PKG_DIR, n), encoding="utf-8").read() for n in os.listdir(PKG_DIR) if n.endswith(".py")}
        writers = sorted(n for n, s in srcs.items() if re.search(r"sys\.stderr\.write\(|file=sys\.stderr", s) and n not in ("_emit.py", "_core_gate.py", "install.py"))
        self.assertEqual(writers, [], "census lines go through _emit.emit (the core gate is the core's template; install.py relays pip's own stderr)")

    def test_census_settings_routes_are_core_free_at_import_and_in_use(self):
        """An outside runner's stock arm imports rfdiffusion3_opt.census / .settings from <kit>/opt in the PRISTINE interpreter, which need
        not carry the shared core: import them, observe a pass and finish it with every opt_core import refused — stub-free, the real modules."""
        code = textwrap.dedent(f"""
            import json, sys
            sys.path[:] = [{fx.PKG_PARENT!r}] + [p for p in sys.path if 'opt_core' not in p and 'site-packages' not in p]
            class Refuse:
                def find_spec(self, name, path=None, target=None):
                    if name == 'opt_core' or name.startswith('opt_core.'):
                        raise ImportError('opt_core is not importable in this interpreter (refused by the test)')
            sys.meta_path.insert(0, Refuse())
            import rfdiffusion3_opt.settings as settings, rfdiffusion3_opt.routes as routes, rfdiffusion3_opt.census as census
            tokens = ['diffusion_batch_size=8', 'seed=101', 'out_dir=/nonexistent/o']
            assert routes.route_of('off', tokens + [settings.COMPILE_TOKEN]).name == 'stock'
            lines = []
            obs = census.observe('stock', 'off', tokens + [settings.COMPILE_TOKEN], emit=lines.append)
            rc = census.finish(1)                                                    # a failed run keeps its code; the line still prints
            print(json.dumps({{'rc': rc, 'kinds': [l.split()[1] for l in lines], 'core': sorted(m for m in sys.modules if m.split('.')[0] == 'opt_core'),
                              'B': obs.batch}}))
        """)
        env = {k: v for k, v in os.environ.items() if k not in ("RFDIFFUSION3_OPT", "PYTHONPATH")}
        r = subprocess.run([sys.executable, "-S", "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout.strip().splitlines()[-1])
        self.assertEqual(got, {"rc": 1, "kinds": ["KERNELS"], "core": [], "B": "8"})   # a stock route: the census line alone (no ATTN_PATH: nothing counts calls in a stock process)


    def test_counting_shim_is_transparent(self):
        seen, sentinel = [], object()

        def use_dense_sdpa_pairbias(Q, indices, full, H):
            seen.append((Q, indices, full, H))
            if H == "raise":
                raise KeyError("upstream's own error")
            return {"dense": True, "sparse": False, "odd": sentinel}[Q]

        counts = {}
        shim = census.counting_shim(use_dense_sdpa_pairbias, counts)
        self.assertEqual(shim.__name__, "use_dense_sdpa_pairbias"); self.assertIs(shim.__wrapped__, use_dense_sdpa_pairbias); self.assertTrue(shim._kernels_observed)
        q_obj, idx_obj = "dense", object()
        self.assertIs(shim(Q=q_obj, indices=idx_obj, full=False, H=4), True)                     # keyword call, as attention.py:340-342 calls it
        self.assertEqual(seen[-1], (q_obj, idx_obj, False, 4))                                    # the arguments arrive unchanged (identity)
        self.assertIs(shim("sparse", idx_obj, False, 8), False)                                   # positional: `full` read at upstream's position
        self.assertIs(shim("odd", idx_obj, 0, 2), sentinel)                                       # whatever upstream returns comes back as is (identity, not truthiness)
        self.assertIs(shim(Q="sparse", indices=None, full=True, H=1), False)                      # full truthy: the token-level call
        with self.assertRaises(KeyError):
            shim("dense", None, False, "raise")                                                   # exceptions propagate; a raising call is not counted
        self.assertEqual(counts, {"atom_dense": 2, "atom_sparse": 1, "token_full": 1})            # sentinel is truthy: atom_dense
        self.assertEqual(len(seen), 5)
        self.assertEqual(census.classify_decision(True, True), "token_full")                     # full wins over the value (upstream returns False there anyway)
        self.assertEqual((census.classify_decision(None, True), census.classify_decision(0, False)), ("atom_dense", "atom_sparse"))

    def test_counting_shim_skips_the_count_while_dynamo_traces(self):
        torch = type(sys)("torch"); torch.compiler = type(sys)("torch.compiler"); state = {"tracing": True}
        torch.compiler.is_compiling = lambda: state["tracing"]
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.compiler": torch.compiler}):
            counts = {}
            shim = census.counting_shim(lambda Q, indices, full, H: True, counts)
            self.assertIs(shim(None, None, False, 4), True); self.assertEqual(counts, {})        # traced: the value passes, nothing is counted (no guard on the tally)
            state["tracing"] = False
            self.assertIs(shim(None, None, False, 4), True); self.assertEqual(counts, {"atom_dense": 1})

    def test_attn_path_line_and_file(self):
        self.assertEqual(census.attn_path_line({}), "[rfdiffusion3-opt] ATTN_PATH atom_dense=0 atom_sparse=0 token_full=0 compiled=0 calls_uncounted=0")   # present with zero calls
        self.assertEqual(census.attn_path_line({"atom_dense": 4416, "token_full": 2208}), "[rfdiffusion3-opt] ATTN_PATH atom_dense=4416 atom_sparse=0 token_full=2208 compiled=0 calls_uncounted=0")
        self.assertEqual(census.attn_path_line({}, compiled=True, calls_uncounted=True), "[rfdiffusion3-opt] ATTN_PATH atom_dense=0 atom_sparse=0 token_full=0 compiled=1 calls_uncounted=1")   # route stock: the counts' scope named on the line
        self.assertEqual(census.attn_path_line({"atom_dense": 27}, False, True), "[rfdiffusion3-opt] ATTN_PATH atom_dense=27 atom_sparse=0 token_full=0 compiled=0 calls_uncounted=1")   # mode fast: graph replays uncounted
        self.assertRegex(census.attn_path_line({"atom_sparse": 3}), self.CONSUMER_ATTN_RE)
        self.assertEqual(census.ATTN_PATH_KEYS, ("atom_dense", "atom_sparse", "token_full")); self.assertEqual(census.ATTN_SCOPE_KEYS, ("compiled", "calls_uncounted"))
        pre = {"route": "stock", "B": 16, "n_atoms": 2456, "L": 175, "k": 128, "H": 4, "heads": [4], "atom_sparse": 0, "dense": True, "free_gib": 71.23456, "est_dense_gib": 1.4567, "device": "cuda:0", "after_items": 0}
        self.assertEqual(census.preflight_line("stock", pre), "[rfdiffusion3-opt] ATTN_PREFLIGHT route=stock B=16 n_atoms=2456 L=175 atom_sparse=0 free_gib=71.235 est_dense_gib=1.457 pinned=0")
        self.assertEqual(census.preflight_line("exact", dict(pre, pinned=1)), "[rfdiffusion3-opt] ATTN_PREFLIGHT route=exact B=16 n_atoms=2456 L=175 atom_sparse=0 free_gib=71.235 est_dense_gib=1.457 pinned=1")
        self.assertEqual(census.preflight_line("stock", dict(pre, L=None)), "[rfdiffusion3-opt] ATTN_PREFLIGHT route=stock B=16 n_atoms=2456 L=unread atom_sparse=0 free_gib=71.235 est_dense_gib=1.457 pinned=0")   # no token map on the step: the residues unread, the atoms always
        self.assertRegex(census.preflight_line("fast", dict(pre, atom_sparse=1)), self.CONSUMER_PREFLIGHT_RE)
        class _Map:                                                                                  # f["atom_to_token_map"]: upstream's token count is .max() + 1 (RFD3_diffusion_module.py:193-195)
            def max(self): return 99
        self.assertEqual((census.token_count({"atom_to_token_map": _Map()}), census.token_count({}), census.token_count(None)), (100, None, None))
        with tarfile.open(fx.ARCHIVE, "r:gz") as tf:
            dm_src = tf.extractfile(fx.STOCK_RFD3 + "rfd3/model/RFD3_diffusion_module.py").read().decode()
        self.assertIn('tok_idx = f["atom_to_token_map"]', dm_src); self.assertIn("I = tok_idx.max() + 1  # Number of tokens", dm_src)
        got = census.attn_path_record({"atom_dense": 7}, compiled=False, calls_uncounted=True, preflight=[pre])   # the record the manifest carries (census.attn_path); no file of its own
        self.assertEqual(got, {"atom_dense": 7, "atom_sparse": 0, "token_full": 0, "compiled": False, "calls_uncounted": True, "pin_hits": 0, "pin_misses": 0, "preflight": [pre]})
        self.assertTrue(all(type(got[k]) is int for k in census.ATTN_PATH_KEYS)); self.assertTrue(all(type(got[k]) is bool for k in census.ATTN_SCOPE_KEYS))
        json.dumps(got)
        self.assertEqual(manifest.NOT_OUTPUTS, (manifest.FILENAME,))                              # the manifest is the one kit file beside the designs

    def test_clock_records_are_kept_not_printed(self):
        """Upstream's per-roll-out clock record (`Finished inference batch in <s> seconds`) is kept as data — obs.items {item, clock_s (upstream's
        digits verbatim), B} and obs.batch_seconds — and NOTHING is printed per item: the kit prints no timing line of its own."""
        obs = census.Observer("default", "off", ["out_dir=/x", "diffusion_batch_size=8"], emit=self.lines.append)
        cuda = type(sys)("torch.cuda"); cuda.is_available = lambda: False                    # no CUDA initialised: no PEAK line either
        torch = type(sys)("torch"); torch.cuda = cuda
        with mock.patch.dict(sys.modules, {"torch": torch, "torch.cuda": cuda}):
            for t in ("7.50", "0.125", "12"):
                obs.on_record(census.ENGINE_LOGGER, f"[rank: 0] Finished inference batch in {t} seconds.")
        self.assertEqual(self.lines, [])
        self.assertEqual(obs.items, [{"item": f"batch:{k}", "clock_s": t, "B": "8"} for k, t in enumerate(("7.50", "0.125", "12"))])
        self.assertEqual(obs.batch_seconds, [7.5, 0.125, 12.0])

    def _torch(self, cuda="13.0", tf32=False, cudnn_tf32=True):
        torch = type(sys)("torch"); torch.__version__ = "2.13.0+cu130"
        torch.version = type("V", (), {"cuda": cuda})()
        matmul = type("M", (), {"allow_tf32": tf32})()
        torch.backends = type("B", (), {"cuda": type("C", (), {"matmul": matmul})(), "cudnn": type("D", (), {"allow_tf32": cudnn_tf32})()})()
        return torch

    def test_stack_facts_on_a_fake_torch(self):
        lu = type(sys)(census.LAYER_UTILS_MODULE)
        lu.RMSNorm_ = type("FusedRMSNorm", (), {"__module__": "apex.normalization.fused_layer_norm"})
        with mock.patch.dict(sys.modules, {"torch": self._torch(), census.LAYER_UTILS_MODULE: lu}), mock.patch("importlib.util.find_spec", return_value=None):
            f = census.stack_facts({})
            self.assertEqual(f, {"torch": "2.13.0+cu130", "cuda": "13.0", "tf32_matmul": "False", "tf32_cudnn": "True", "alloc_conf": "unset", "apex": "absent"})   # torch's defaults: matmul TF32 off, cuDNN TF32 on
            self.assertEqual(list(f), ["torch", "cuda", "tf32_matmul", "tf32_cudnn", "alloc_conf", "apex"])   # the order on the line
            self.assertEqual(census.stack_facts({"PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})["alloc_conf"], "expandable_segments:True")
        with mock.patch.dict(sys.modules, {"torch": self._torch(tf32=True, cudnn_tf32=False)}):
            f = census.stack_facts({})
            self.assertEqual((f["tf32_matmul"], f["tf32_cudnn"]), ("True", "False"))                # as found at the pass's end, whatever set it (TORCH_ALLOW_TF32_CUBLAS_OVERRIDE=1 lands in the first)
        saved = sys.modules.pop("torch", None)
        try:
            f = census.stack_facts({})
            self.assertEqual((f["torch"], f["cuda"], f["tf32_matmul"], f["tf32_cudnn"]), ("absent", "absent", "absent", "absent"))   # no torch in the process: facts absent, never `unread`
        finally:
            if saved is not None:
                sys.modules["torch"] = saved
        apex = type(sys)("apex"); apex.__version__ = "0.1"
        with mock.patch.dict(sys.modules, {"apex": apex}), mock.patch("importlib.util.find_spec", return_value=object()), mock.patch("importlib.metadata.version", side_effect=__import__("importlib.metadata").metadata.PackageNotFoundError("apex")):
            self.assertEqual(census.apex_word(), "present:0.1")                                    # importable: its version, without importing it here

    def test_ckpt_word(self):
        import hashlib
        ck = os.path.join(self.td.name, "rfd3_latest.ckpt")
        with open(ck, "wb") as fh:
            fh.write(b"pinned weights")
        want = hashlib.sha256(b"pinned weights").hexdigest()[:12]
        self.assertEqual(census.ckpt_word(ck), f"rfd3_latest.ckpt@{want}")
        self.assertEqual(census.ckpt_word(ck), f"rfd3_latest.ckpt@{want}")                         # second read: the digest cache, same word
        self.assertEqual(census.ckpt_word(None), "unread")                                          # no engine initialized
        self.assertEqual(census.ckpt_word(os.path.join(self.td.name, "gone.ckpt")), "gone.ckpt@absent")

    def test_census_line_carries_the_facts_after_the_compile_fields(self):
        words = {"compile": "off-by-route:compile_model=false", "atom_attn": "engaged:dense-sdpa(F.scaled_dot_product_attention)@torch2.13.0+cu130", "cueq": census.CUEQ_WORD, "ds4sci": census.DS4SCI_WORD}
        words = dict(words, rmsnorm="engaged:apex.FusedRMSNorm(215-modules+ext)@apex0.1")
        extras = {"compile_s": None, "torch": "2.13.0+cu130", "cuda": "13.0", "tf32_matmul": "False", "tf32_cudnn": "True", "alloc_conf": "unset", "apex": "present:0.1", "ckpt": "rfd3_latest.ckpt@9b3f85923e0d"}
        line = census.kernels_line("default", "off", "8", words, extras)
        self.assertEqual(line, "[rfdiffusion3-opt] KERNELS route=default mode=off B=8 compile=off-by-route:compile_model=false "
                               "atom_attn=engaged:dense-sdpa(F.scaled_dot_product_attention)@torch2.13.0+cu130 " + f"cueq={census.CUEQ_WORD} ds4sci={census.DS4SCI_WORD} "
                               "rmsnorm=engaged:apex.FusedRMSNorm(215-modules+ext)@apex0.1 torch=2.13.0+cu130 cuda=13.0 tf32_matmul=False tf32_cudnn=True alloc_conf=unset apex=present:0.1 ckpt=rfd3_latest.ckpt@9b3f85923e0d")
        self.assertEqual([t.split("=", 1)[0] for t in line.split()[2:]], ["route", "mode", "B", "compile", "atom_attn", "cueq", "ds4sci", "rmsnorm", "torch", "cuda", "tf32_matmul", "tf32_cudnn", "alloc_conf", "apex", "ckpt"])
        for t in line.split()[2:]:
            self.assertRegex(t, r"^[A-Za-z0-9_]+=\S+$")

    def test_observer_counts_calls_and_records_the_census_at_finish(self):
        """A KIT route (exact): the shim goes on the module attribute at initialize (the module imported by then), counts every call through it,
        and finish prints KERNELS then ATTN_PATH; the record carries attn_path, items, ckpt_path (no file beside the designs); B is the engine's
        live value. (The stock routes install no shim and print no ATTN_PATH line: test_the_stock_routes_install_handlers_only.)"""
        out = os.path.join(self.td.name, "designs")
        ck = os.path.join(self.td.name, "w.ckpt")
        with open(ck, "wb") as fh:
            fh.write(b"w")
        att = type(sys)(census.ATTENTION_MODULE)
        att.use_dense_sdpa_pairbias = lambda Q, indices, full, H: (not full) and Q != "sparse"
        eng = _engine(False); eng.transform_overrides = {"diffusion_batch_size": 16}; eng.ckpt_path = ck      # the line said 8; the engine runs 16: the engine's value is the B that counts
        obs = census.Observer("exact", "exact", [f"out_dir={out}", "diffusion_batch_size=8", "n_batches=2"], emit=self.lines.append)
        self.assertEqual(obs.batch, "8")                             # the line's own token
        mods = _fake_dynamo_modules(); mods[census.ATTENTION_MODULE] = att
        with mock.patch.dict(sys.modules, mods):
            obs.after_initialize(eng)
            self.assertTrue(getattr(att.use_dense_sdpa_pairbias, "_kernels_observed", False))
            first = att.use_dense_sdpa_pairbias
            obs._wrap_attention(att); self.assertIs(att.use_dense_sdpa_pairbias, first)             # idempotent: one shim, never stacked
            for q, full in (("dense", False), ("dense", True), ("dense", False)):
                att.use_dense_sdpa_pairbias(Q=q, indices=None, full=full, H=4)
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=16 L=999 k=128 H=4"))
            obs.preflight.append({"route": "exact", "B": 16, "n_atoms": 999, "L": 100, "atom_sparse": 0, "dense": True})   # the gated route's pre-flight read (a real step installs it: TestGuard)
            obs.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("33.10", "seconds"))
            self.assertEqual(obs.finish(0), 0, self.lines)
        kinds = [l.split()[1] for l in self.lines]
        self.assertEqual(kinds, ["KERNELS", "ATTN_PATH"])                                  # no per-item stamp: upstream's clock record is kept as data (obs.items), never printed
        self.assertEqual(obs.items[0], {"item": "batch:0", "clock_s": "33.10", "B": "16"})
        self.assertIn(" B=16 ", self.lines[0])
        self.assertRegex(self.lines[0], r" ckpt=w\.ckpt@[0-9a-f]{12}$")
        self.assertEqual(self.lines[1], "[rfdiffusion3-opt] ATTN_PATH atom_dense=2 atom_sparse=0 token_full=1 compiled=0 calls_uncounted=0")
        rec = obs.record()
        self.assertEqual({k: rec["attn_path"][k] for k in ("atom_dense", "atom_sparse", "token_full", "compiled", "calls_uncounted")},
                         {"atom_dense": 2, "atom_sparse": 0, "token_full": 1, "compiled": False, "calls_uncounted": False}); self.assertNotIn("attn_path_file", rec)   # (the pin counts: TestGuard's pin tests)
        self.assertFalse(os.path.exists(os.path.join(out, "attn_path.json")))                     # the census writes no file beside the designs
        self.assertEqual(rec["items"], [{"item": "batch:0", "clock_s": "33.10", "B": "16"}]); self.assertEqual((rec["batch"], rec["ckpt_path"]), ("16", ck))
        self.assertEqual(rec["detail"]["batch_source"], "engine"); self.assertEqual(rec["detail"]["stack"]["ckpt"], self.lines[0].rsplit("ckpt=", 1)[1])
        json.dumps(rec)                                                                              # the record is JSON (manifest / stock proof)

    def test_the_stock_routes_install_handlers_only(self):
        """A STOCK route (stock | default: upstream in the pristine interpreter) installs logging handlers on upstream's three loggers and NOTHING
        else — no wrap of the engine's initialize, no counting shim, no forward probe, no meta-path finder —; every word is read off upstream's
        own records; no ATTN_PATH line (nothing counts calls in a stock process); B and the checkpoint are the line's tokens."""
        ck = os.path.join(self.td.name, "w.ckpt")
        with open(ck, "wb") as fh:
            fh.write(b"w")
        att = type(sys)(census.ATTENTION_MODULE); fn = lambda Q, indices, full, H: True; att.use_dense_sdpa_pairbias = fn
        eng_mod = type(sys)(census.ENGINE_MODULE)
        class E:
            def initialize(self):
                return "cfg"
        eng_mod.RFD3InferenceEngine = E
        meta_before = list(sys.meta_path)
        with mock.patch.dict(sys.modules, {census.ATTENTION_MODULE: att, census.ENGINE_MODULE: eng_mod}):
            obs = census.observe("stock", "off", ["diffusion_batch_size=4", f"ckpt_path={ck}", "compile_model=true"], emit=self.lines.append)
            self.assertIs(att.use_dense_sdpa_pairbias, fn); self.assertIs(eng_mod.RFD3InferenceEngine.initialize, E.initialize)   # nothing re-bound
            self.assertEqual(list(sys.meta_path), meta_before); self.assertIsNone(obs._finder)                                    # no import hook
            self.assertEqual([type(h).__name__ for name in (census.ENGINE_LOGGER, census.ATTENTION_LOGGER, census.LAYER_UTILS_LOGGER) for h in logging.getLogger(name).handlers if h is obs._handler], ["_RecordHandler"] * 3)
            self.assertEqual((obs.batch, obs.ckpt_path, obs.detail["observation"], obs.detail["batch_source"]), ("4", ck, census.STOCK_OBSERVATION, "line"))
            logging.getLogger(census.LAYER_UTILS_LOGGER).info("Fused RMSNorm enabled!")                                   # layer_utils.py:16
            logging.getLogger(census.ENGINE_LOGGER).info("torch.compile enabled for diffusion submodules (encoder, ...).")    # engine.py:265-269
            logging.getLogger(census.ATTENTION_LOGGER).info(UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=4 L=999 k=128 H=4"))
            logging.getLogger(census.ENGINE_LOGGER).info("Finished inference batch in 9.50 seconds.")
            self.assertEqual(census.finish(0), 0, self.lines)
        self.assertEqual([l.split()[1] for l in self.lines], ["KERNELS"])                             # the census once; no ATTN_PATH, no ATTN_PREFLIGHT, no report-only line
        self.assertRegex(self.lines[0], r"^\[rfdiffusion3-opt\] KERNELS route=stock mode=off B=4 compile=engaged:torch\.compile\(record:engine\.py:265-269\) atom_attn=engaged:dense-sdpa\(\S+ cueq=n/a-upstream:\S+ ds4sci=n/a-upstream:\S+ rmsnorm=engaged:apex\.FusedRMSNorm\(record:layer_utils\.py:16\) .* ckpt=w\.ckpt@[0-9a-f]{12}$")
        rec = obs.record()
        self.assertEqual((rec["refused"], rec["report_only"], rec["attn_path_line"], rec["attn_preflight_lines"], rec["batch_seconds"]), (None, {}, None, [], [9.5]))
        self.assertEqual(len(rec["records"]), 3); json.dumps(rec)
        census.reset(); self.lines.clear()
        with mock.patch.dict(sys.modules, {census.ATTENTION_MODULE: att, census.ENGINE_MODULE: eng_mod}):
            obs = census.observe("default", "off", [f"ckpt_path={ck}"], emit=self.lines.append)      # no batch token: upstream's shipped value is the B
            logging.getLogger(census.LAYER_UTILS_LOGGER).warning("Using nn.RMSNorm instead of apex.normalization.fused_layer_norm.FusedRMSNorm.Ensure you're using the correct apptainer")
            logging.getLogger(census.ATTENTION_LOGGER).info(UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            self.assertEqual(census.finish(0), 0, self.lines)
        self.assertEqual([l.split()[1] for l in self.lines], ["KERNELS"])
        self.assertRegex(self.lines[0], r"^\[rfdiffusion3-opt\] KERNELS route=default mode=off B=8 compile=off-by-route:compile_model=false atom_attn=engaged:dense-sdpa\(\S+ .* rmsnorm=fallback:torch\.nn\.RMSNorm\(record:layer_utils\.py:19-21\) ")
        self.assertEqual(obs.detail["batch_source"], "upstream-default(rfdiffusion3.yaml:20)")

    def test_observer_without_out_dir_prints_the_line(self):
        obs = census.Observer("exact", "exact", ["diffusion_batch_size=8"], emit=self.lines.append)
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            obs.after_initialize(_engine(False))
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            self.assertEqual(obs.finish(0), 0)
        self.assertEqual(self.lines[-1], "[rfdiffusion3-opt] ATTN_PATH atom_dense=0 atom_sparse=0 token_full=0 compiled=0 calls_uncounted=0")
        self.assertEqual(obs.record()["attn_path"]["atom_dense"], 0)

    def test_the_finder_wraps_both_upstream_modules_at_their_import(self):
        """observe() on a KIT route before upstream is imported: the meta-path finder lets rfd3.engine and rfd3.model.layers.attention execute,
        wraps each once, and leaves sys.meta_path when both are done. On a STOCK route no finder is installed and nothing is wrapped."""
        site = os.path.join(self.td.name, "site")
        fx._write(os.path.join(site, "rfd3", "__init__.py")); fx._write(os.path.join(site, "rfd3", "model", "__init__.py")); fx._write(os.path.join(site, "rfd3", "model", "layers", "__init__.py"))
        fx._write(os.path.join(site, "rfd3", "engine.py"), "class RFD3InferenceEngine:\n    _COMPILE_TARGETS = ()\n    def initialize(self):\n        return 'cfg'\n")
        fx._write(os.path.join(site, "rfd3", "model", "layers", "attention.py"), "def use_dense_sdpa_pairbias(Q, indices, full, H):\n    return bool(Q)\n")
        for route, mode, want in (("exact", "exact", {"before": True, "after": False, "cfg": "cfg", "r": [True, False, True], "seen": True, "counts": {"atom_dense": 1, "atom_sparse": 1, "token_full": 1}, "shim": True, "init": True}),
                                  ("default", "off", {"before": False, "after": False, "cfg": "cfg", "r": [True, False, True], "seen": False, "counts": {"atom_dense": 0, "atom_sparse": 0, "token_full": 0}, "shim": False, "init": False})):
            code = ("import json, sys; from rfdiffusion3_opt import census; obs = census.observe(%r, %r, ['diffusion_batch_size=8']); " % (route, mode) +
                    "before = obs._finder in sys.meta_path; import rfd3.engine, rfd3.model.layers.attention as att; "
                    "e = rfd3.engine.RFD3InferenceEngine(); cfg = e.initialize(); "
                    "r = [att.use_dense_sdpa_pairbias(1, None, False, 4), att.use_dense_sdpa_pairbias(Q=0, indices=None, full=False, H=4), att.use_dense_sdpa_pairbias(1, None, True, 4)]; "
                    "print(json.dumps({'before': before, 'after': obs._finder in sys.meta_path, 'cfg': cfg, 'r': r, 'seen': obs.engine_seen, 'counts': obs.attn_counts, "
                    "'shim': getattr(att.use_dense_sdpa_pairbias, '_kernels_observed', False), 'init': getattr(rfd3.engine.RFD3InferenceEngine.initialize, '_kernels_observed', False)}))")
            r = fx.run_py(code, site)
            self.assertEqual(r.returncode, 0, r.stderr)
            got = json.loads(r.stdout.strip().splitlines()[-1])
            self.assertEqual(got, want, route)

class TestGuard(unittest.TestCase):
    """The REQUIRE guard in process. On the kit routes (exact | fast: routes.Route.enforce) a missed expectation is a refusal =
    KernelsRefused (SystemExit 5) at the observation point, the KERNELS line printed first, the accelerator and the route named. On the
    stock routes (stock | default) the same miss is one `KERNELS report-only on route <r>: …` line per accelerator and the pass proceeds
    with its own exit code; the pre-flight dense bound still stops every gated route (stock included)."""

    def setUp(self):
        census.reset()
        self.lines = []

    def tearDown(self):
        census.reset()

    def _obs(self, route, mode="off", tokens=None):
        return census.Observer(route, mode, tokens or ["diffusion_batch_size=8"], emit=self.lines.append)

    def test_stock_reports_a_compile_upstream_did_not_log_and_proceeds(self):
        """Route stock expects compile engaged, read off upstream's own record (engine.py:265-269). A pass whose record never came is worded
        `unread:no-compile-record…` at its end and REPORTED — one `KERNELS report-only` line, no refusal, nothing raised — with the run's own
        exit code; upstream's `skipping torch.compile` record (engine.py:247-251) is a fallback word reported the moment it is logged."""
        obs = self._obs("stock")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            obs.on_record(census.LAYER_UTILS_LOGGER, "Fused RMSNorm enabled!")
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=3210 k=128 H=4"))
            obs.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("9.50", "seconds"))
            self.assertEqual(self.lines, [])                                                         # nothing printed before the pass's end
            self.assertEqual(obs.finish(0), 0, self.lines)                                           # the run's own code, never EXIT_KERNELS here
            self.assertEqual(obs.finish(1), 1)
        kinds = [l.split()[1] + ("/ro" if "report-only" in l else "") for l in self.lines]
        self.assertEqual(kinds, ["KERNELS", "KERNELS/ro"], self.lines)                              # the census line once, then the report-only line once; no ATTN_PATH on a stock route
        kline = self.lines[0]
        self.assertTrue(kline.startswith("[rfdiffusion3-opt] KERNELS route=stock mode=off B=8 compile=unread:no-compile-record(engine.py:265-269) atom_attn=engaged:dense-sdpa(F.scaled_dot_product_attention)@torch2.13.0+cu130 "), kline)
        self.assertRegex(kline, r" ds4sci=\S+ rmsnorm=engaged:apex\.FusedRMSNorm\(record:layer_utils\.py:16\) torch=2\.13\.0\+cu130 cuda=none tf32_matmul=absent tf32_cudnn=absent alloc_conf=\S+ apex=\S+ ckpt=unread$")   # no compile fields (nothing engaged); the stack facts; no checkpoint token on this line
        self.assertEqual(self.lines[1], f"[rfdiffusion3-opt] KERNELS report-only on route stock: compile={obs.words['compile']}, expected engaged — upstream's run proceeds")
        rec = obs.record()
        self.assertEqual((rec["refused"], rec["report_only"], rec["line"]), (None, {"compile": {"word": obs.words["compile"], "expected": "engaged", "when": "finish"}}, kline))
        self.assertEqual(census.report_only_line("stock", "compile", "fallback:x", "engaged"), "[rfdiffusion3-opt] KERNELS report-only on route stock: compile=fallback:x, expected engaged — upstream's run proceeds")
        census.reset(); self.lines.clear()
        obs = self._obs("stock")
        obs.on_record(census.ENGINE_LOGGER, census.COMPILE_SKIPPED_TEXT + ".")
        self.assertEqual(self.lines, ["[rfdiffusion3-opt] KERNELS report-only on route stock: compile=fallback:compile-skipped(engine.py:247-251), expected engaged — upstream's run proceeds"])
        self.assertEqual(obs.report_only["compile"]["when"], "before-timing"); self.assertIsNone(obs.refused)

    def test_fast_refuses_an_unwrapped_compile_before_timing(self):
        """The same miss on a KIT route refuses: KernelsRefused (SystemExit 5) at the observation point, the KERNELS line printed first, the
        accelerator and the route named; finish() keeps the 5 and prints the line once."""
        obs = self._obs("fast", mode="fast")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            for k in ("rfd3.model.hoist",):
                sys.modules.pop(k, None)                                                           # no kit hoist module in this process: the census reads upstream's objects
            with self.assertRaises(SystemExit) as cm:
                obs.after_initialize(_engine(True, wrap=()))
        self.assertEqual(cm.exception.code, 5)
        self.assertIsInstance(cm.exception, census.KernelsRefused)
        self.assertEqual(len(self.lines), 2, self.lines)
        self.assertTrue(self.lines[0].startswith("[rfdiffusion3-opt] KERNELS route=fast mode=fast B=8 compile="), self.lines[0])
        self.assertIn(census.kind_of(obs.words["compile"]), ("fallback", "absent"), obs.words["compile"])
        self.assertIn("atom_attn=unread", self.lines[0])
        self.assertEqual(self.lines[1], f"[rfdiffusion3-opt] NOT ACTIVE: KERNELS refused on route fast — compile={obs.words['compile']}, expected engaged; exit 5")
        self.assertEqual((obs.refused["accel"], obs.refused["when"]), ("compile", "before-timing")); self.assertEqual(obs.report_only, {})
        self.assertEqual(obs.finish(5), 5)                                                         # the run unwound with 5: the KERNELS line is not printed twice ...
        self.assertEqual([l.split()[1] for l in self.lines], ["KERNELS", "NOT", "ATTN_PATH"])         # ... the ATTN_PATH line prints once

    def test_graph_breaks_are_named_not_refused(self):
        """dynamo's partial-graph operation at the pin (upstream's modules break the graph): the compiled graphs run — the word stays engaged,
        the counts ride on the line; on the stock route the word is upstream's record and the counts dynamo's own (read at the pass's end)."""
        obs = self._obs("stock")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules(unique_graphs=39, frames=(157, 171))):
            sys.modules["torch._dynamo.utils"].counters["graph_break"]["Backend compiler exception"] = 44
            obs.on_record(census.LAYER_UTILS_LOGGER, "Fused RMSNorm enabled!")
            obs.on_record(census.ENGINE_LOGGER, "torch.compile enabled for diffusion submodules (encoder, ...).")
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            self.assertEqual(obs.finish(0), 0)
        self.assertEqual(census.kind_of(obs.words["compile"]), "engaged")
        self.assertRegex(self.lines[0], r"compile=engaged:torch\.compile\(record:engine\.py:265-269\) .* graphs=39 frames=157/171 graph_breaks=44 torch=2\.13\.0\+cu130 cuda=none ")   # the stack facts follow the compile fields
        self.assertIsNone(obs.refused); self.assertEqual([l.split()[1] for l in self.lines], ["KERNELS"])
        census.reset(); self.lines.clear()
        obs = self._obs("stock")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules(unique_graphs=0, frames=(0, 12))):
            obs.on_record(census.LAYER_UTILS_LOGGER, "Fused RMSNorm enabled!")
            obs.on_record(census.ENGINE_LOGGER, "torch.compile enabled for diffusion submodules (encoder, ...).")
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="0.1 GiB bias", shape="D=8 L=999 k=128 H=4"))
            self.assertEqual(obs.finish(0), 0)                                                     # the record said compiled, yet dynamo compiled no graph: every frame ran eager — a fallback word, REPORTED on the stock route
        self.assertIn("compile=fallback:no-graph-compiled(dynamo-counters)", self.lines[0])
        self.assertEqual(self.lines[-1], "[rfdiffusion3-opt] KERNELS report-only on route stock: compile=fallback:no-graph-compiled(dynamo-counters), expected engaged — upstream's run proceeds")
        self.assertEqual([l.split()[1] for l in self.lines], ["KERNELS", "KERNELS"]); self.assertIsNone(obs.refused)
        self.assertEqual(obs.report_only["compile"]["when"], "finish")

    def test_inductor_cache_state(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(census.inductor_cache_state({"TORCHINDUCTOR_CACHE_DIR": d}), f"cold({d})")
            open(os.path.join(d, "x"), "w").close()
            self.assertEqual(census.inductor_cache_state({"TORCHINDUCTOR_CACHE_DIR": d}), f"warm({d})")
            self.assertTrue(census.inductor_cache_state({"TORCHINDUCTOR_CACHE_DIR": os.path.join(d, "absent")}).startswith("cold("))
        self.assertIn("torchinductor_", census.inductor_cache_dir({}))                             # torch's default location when the variable is unset

    def test_peak_line_per_item_with_reset_at_item_start(self):
        """PEAK item=batch:<k> alloc_gib=<.2f> reserved_gib=<.2f> at each batch clock record, from torch.cuda's peak stats (mocked), the
        stats reset at the item's start (after initialize) and after each line; nothing when CUDA is not initialized."""
        calls = []
        cuda = type(sys)("torch.cuda")                                                        # the surface opt_core.mem.allocator reads
        cuda.is_available = lambda: True
        cuda.current_device = lambda: 0
        cuda.max_memory_allocated = lambda dev=None: 21.5 * 2 ** 30
        cuda.max_memory_reserved = lambda dev=None: 24.25 * 2 ** 30
        cuda.memory_allocated = lambda dev=None: 3.0 * 2 ** 30
        cuda.memory_reserved = lambda dev=None: 4.0 * 2 ** 30
        cuda.reset_peak_memory_stats = lambda dev=None: calls.append("reset")
        mods = _fake_dynamo_modules(); mods["torch"].cuda = cuda
        obs = self._obs("default")
        with mock.patch.dict(sys.modules, mods):
            obs.after_initialize(_engine(False))
            self.assertEqual(calls, ["reset"])                                                # item 0 starts after initialize()
            obs.on_record(census.ENGINE_LOGGER, "[rank: 0] Finished inference batch in %.2f %s." % (18.2, "seconds"))
            obs.on_record(census.ENGINE_LOGGER, "[rank: 0] Finished inference batch in %.2f %s." % (17.9, "seconds"))
        self.assertEqual(self.lines, [                                                          # one PEAK line per clock record, nothing else (no per-item stamp)
                                      "[rfdiffusion3-opt] PEAK item=batch:0 alloc_gib=21.50 reserved_gib=24.25",
                                      "[rfdiffusion3-opt] PEAK item=batch:1 alloc_gib=21.50 reserved_gib=24.25"])
        self.assertEqual(calls, ["reset"] * 3)
        self.assertEqual([p["item"] for p in obs.record()["peaks"]], ["batch:0", "batch:1"])
        self.assertEqual(obs.record()["items"], [{"item": "batch:0", "clock_s": "18.20", "B": "8"}, {"item": "batch:1", "clock_s": "17.90", "B": "8"}])
        for line in self.lines:                                                                # the cross-kit reader's grammar, exactly: nothing after the two groups
            self.assertRegex(line, r"PEAK item=(\S+) alloc_gib=([0-9.]+) reserved_gib=([0-9.]+)$")
        census.reset(); self.lines.clear(); calls.clear()
        cuda.is_available = lambda: False                                                     # no CUDA: no PEAK line, no reset — the clock record kept regardless
        obs = self._obs("default")
        with mock.patch.dict(sys.modules, mods):
            obs.after_initialize(_engine(False)); obs.on_record(census.ENGINE_LOGGER, "[rank: 0] Finished inference batch in %.2f %s." % (1.0, "seconds"))
        self.assertEqual((self.lines, calls, obs.batch_seconds), ([], [], [1.0]))

    def test_default_reports_a_compile_it_did_not_ask_exact_refuses_it(self):
        obs = self._obs("default")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            obs.after_initialize(_engine(True, wrap=("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")))   # no raise on the stock route
        self.assertEqual(len(self.lines), 1, self.lines)
        self.assertTrue(self.lines[0].startswith("[rfdiffusion3-opt] KERNELS report-only on route default: compile=engaged:"), self.lines[0]); self.assertIn(", expected off-by-route — upstream's run proceeds", self.lines[0])
        self.assertIsNone(obs.refused)
        census.reset(); self.lines.clear()
        obs = self._obs("exact", mode="exact")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            sys.modules.pop("rfd3.model.hoist", None)
            with self.assertRaises(census.KernelsRefused):
                obs.after_initialize(_engine(True, wrap=("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")))
        self.assertIn("refused on route exact — compile=engaged:", self.lines[-1]); self.assertIn("expected off-by-route", self.lines[-1])

    def test_sparse_at_the_first_attention_call_is_named_on_default_and_unwinds_exact(self):
        rec = UPSTREAM_RECORD.format(chosen="SPARSE", reason="needs 3.10 GiB of 2.00 GiB free", shape="D=8 L=9 k=128 H=4")
        obs = self._obs("default")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            obs.after_initialize(_engine(False))                                                   # compile off-by-route: fine so far, nothing printed
            self.assertEqual(self.lines, [])
            obs.on_record(census.ATTENTION_LOGGER, rec)                                           # upstream's SPARSE record: upstream's own path on a stock route — nothing printed, the pass goes on
            self.assertEqual(self.lines, [])
            obs.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("9.50", "seconds"))
            self.assertEqual(obs.finish(0), 0)
        self.assertIn("atom_attn=fallback:sparse(needs-3.10-GiB-of-2.00-GiB-free)", obs.printed)   # the KERNELS line names the path
        self.assertEqual((obs.refused, obs.report_only.get("atom_attn")), (None, None))
        self.assertNotIn("report-only", "\n".join(self.lines))
        census.reset(); self.lines.clear()
        obs = self._obs("exact", mode="exact")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            obs.after_initialize(_engine(False))
            n_before = len(self.lines)
            with self.assertRaises(census.KernelsRefused) as cm:
                obs.on_record(census.ATTENTION_LOGGER, rec)
        self.assertEqual(cm.exception.code, census.EXIT_KERNELS)
        self.assertIn("atom_attn=fallback:sparse(needs-3.10-GiB-of-2.00-GiB-free)", self.lines[n_before])
        self.assertIn("refused on route exact — atom_attn=fallback:sparse(", self.lines[n_before + 1])

    def test_a_forced_sparse_path_on_the_stock_routes_is_named_not_refused(self):
        """Upstream's own RFD3_DENSE_SDPA_ATTENTION=0 in a `--mode off` process (it passes through the stock arm): the static census reads the
        forced sparse path right after initialize() and the KERNELS line NAMES it on route default and on route stock alike (the stock routes
        judge no attention path: no report line); the pre-flight decision then reads sparse too and refuses nothing: the pass keeps its own exit
        code. (Under a kit mode the same variable is refused at activation, test_activation; a sparse pre-flight refuses the KIT routes only:
        test_preflight_sparse_refuses_the_gated_routes_only.)"""
        ALL = ("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")
        for route, compile in (("default", False), ("stock", True)):
            census.reset(); self.lines.clear()
            obs = self._obs(route, tokens=["diffusion_batch_size=8"] + (["compile_model=true"] if compile else []))
            with mock.patch.dict(os.environ, {"RFD3_DENSE_SDPA_ATTENTION": "0"}), mock.patch.dict(sys.modules, _fake_dynamo_modules(unique_graphs=9, frames=(9, 9))):
                engine = _engine(compile, wrap=ALL if compile else ())
                obs.after_initialize(engine)                                                       # no KernelsRefused, no report line: the stock routes judge no attention path
                self.assertEqual(self.lines, [], route)
                self.assertTrue(obs.detail["atom_attn"]["static"], obs.detail)
                _step(engine)                                                                      # the first denoising step: the pre-flight decision reads upstream's rule, which says sparse (the switch)
                obs.on_record(census.ENGINE_LOGGER, "Finished inference batch in %s %s." % ("9.50", "seconds"))
                self.assertEqual(obs.finish(0), 0, (route, self.lines))
                self.assertEqual(obs.finish(3), 3)                                                 # whatever the run's code is
            self.assertIsNone(obs.refused, route)
            pre = [l for l in self.lines if "] ATTN_PREFLIGHT " in l]
            self.assertEqual(len(pre), 1, self.lines); self.assertIn(f"ATTN_PREFLIGHT route={route} B=8 n_atoms=3210 ", pre[0]); self.assertIn(" atom_sparse=1 ", pre[0])
            self.assertNotIn("attn_preflight=", "\n".join(self.lines)); self.assertNotIn("NOT ACTIVE", "\n".join(self.lines))
            self.assertIn("atom_attn=fallback:sparse(disabled-via-RFD3_DENSE_SDPA_ATTENTION=0)", obs.printed)   # the KERNELS line names it
            self.assertEqual(obs.record()["report_only"], {})

    def test_static_sparse_refuses_right_after_initialize(self):
        obs = self._obs("exact", mode="exact")
        with mock.patch.dict(os.environ, {"RFD3_DENSE_SDPA_ATTENTION": "0"}), mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            with self.assertRaises(census.KernelsRefused):
                obs.after_initialize(_engine(False))
        self.assertIn("atom_attn=fallback:sparse(disabled-via-RFD3_DENSE_SDPA_ATTENTION=0)", self.lines[0])
        self.assertEqual(obs.refused["when"], "before-timing")

    def test_a_pass_without_a_path_record_is_named_on_default_and_refused_on_exact_at_its_end(self):
        obs = self._obs("default")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            rc = obs.finish(0)                                                                       # a stock route: no engine object is read — the records alone
        self.assertEqual(rc, 0)
        self.assertIn("atom_attn=absent:attention-never-ran", self.lines[0])                        # the KERNELS line names it; the stock routes judge no attention path
        self.assertEqual([l.split()[1] for l in self.lines], ["KERNELS"])                             # no ATTN_PATH on a stock route
        self.assertEqual((obs.refused, obs.report_only.get("atom_attn")), (None, None))
        census.reset(); self.lines.clear()
        obs = self._obs("exact", mode="exact")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            obs.after_initialize(_engine(False))
            n_before = len(self.lines)
            rc = obs.finish(0)
        self.assertEqual(rc, 5)
        self.assertIn("atom_attn=absent:attention-never-ran", self.lines[n_before])
        self.assertIn("refused on route exact — atom_attn=absent:attention-never-ran, expected engaged; exit 5", self.lines[-1])
        self.assertEqual([l.split()[1] for l in self.lines[n_before:]], ["KERNELS", "ATTN_PATH", "NOT"])
        self.assertEqual(obs.refused["when"], "finish")

    def test_an_engaged_compile_that_compiled_no_graph_is_a_fallback_at_the_end(self):
        obs = self._obs("stock")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules(unique_graphs=0, frames=(0, 0))):
            obs.after_initialize(_engine(True, wrap=("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")))
            obs.on_record(census.ATTENTION_LOGGER, UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="default", shape="D=8 L=3210 k=128 H=4"))
            self.assertEqual(obs.finish(0), 0)                                                     # reported on the stock route, the run's own code
        self.assertIn("compile=fallback:no-graph-compiled(dynamo-counters)", self.lines[0])
        self.assertIn("KERNELS report-only on route stock: compile=fallback:no-graph-compiled(dynamo-counters), expected engaged", self.lines[-1])

    def test_a_clean_stock_pass(self):
        """Route stock, records only: upstream's compile record, RMSNorm binding record, path record and clock records → one KERNELS line
        (compile engaged with dynamo's counts beside, the path, apex's norm), no ATTN_PREFLIGHT / ATTN_PATH lines, no report-only line."""
        obs = self._obs("stock")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules(unique_graphs=9, frames=(9, 9))):
            obs.on_record(census.LAYER_UTILS_LOGGER, "[rank: 0] Fused RMSNorm enabled!")
            obs.on_record(census.ENGINE_LOGGER, "[rank: 0] torch.compile enabled for diffusion submodules (encoder, ...).")
            obs.on_record(census.ATTENTION_LOGGER, "[rank: 0] " + UPSTREAM_RECORD.format(chosen="DENSE-SDPA", reason="default", shape="D=8 L=3210 k=128 H=4"))
            obs.on_record(census.ENGINE_LOGGER, "[rank: 0] Finished inference batch in %.2f %s." % (181.2, "seconds"))
            obs.on_record(census.ENGINE_LOGGER, "[rank: 0] Finished inference batch in %.2f %s." % (95.51, "seconds"))
            self.assertEqual(obs.finish(0), 0)
        self.assertEqual([l.split()[1] for l in self.lines], ["KERNELS"])   # the census once (the two clock records are data: obs.items)
        line = self.lines[0]
        self.assertRegex(line, r"^\[rfdiffusion3-opt\] KERNELS route=stock mode=off B=8 compile=engaged:torch\.compile\(record:engine\.py:265-269\) "
                               r"atom_attn=engaged:dense-sdpa\(F\.scaled_dot_product_attention\)@torch2\.13\.0\+cu130 cueq=n/a-upstream:\S+ ds4sci=n/a-upstream:\S+ rmsnorm=engaged:apex\.FusedRMSNorm\(record:layer_utils\.py:16\) compile_s=5\.0 graphs=9 frames=9/9 graph_breaks=0 "
                               r"torch=2\.13\.0\+cu130 cuda=none tf32_matmul=absent tf32_cudnn=absent alloc_conf=\S+ apex=(absent|present:\S+) ckpt=unread$")
        self.assertEqual(obs.items[0], {"item": "batch:0", "clock_s": "181.20", "B": "8"})
        rec = obs.record()
        self.assertEqual(rec["batch_seconds"], [181.2, 95.51])                                     # upstream's own per-roll-out clock, first (compile warm-up) and steady
        self.assertIsNone(rec["refused"]); self.assertEqual(rec["route"], "stock"); self.assertEqual(rec["expected"]["compile"], "engaged"); self.assertEqual(rec["report_only"], {})

    def test_a_failed_run_keeps_its_code_and_still_prints_the_line(self):
        obs = self._obs("default")
        with mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            self.assertEqual(obs.finish(1), 1)
        self.assertEqual(len(self.lines), 1); self.assertIn("atom_attn=unread", self.lines[0]); self.assertIn(" KERNELS route=default ", self.lines[0])

    def test_process_api_is_one_observer(self):
        a = census.observe("default", "off", [])
        b = census.observe("stock", "off", [])                                                    # the first observer stands
        self.assertIs(a, b); self.assertEqual(census.observer().route, "default")
        with self.assertRaises(ValueError):
            census.reset(); census.observe("big", "off", [])
        self.assertEqual(census.finish(7), 7)                                                     # no observer: rc unchanged


class TestDesignRoutesEndToEnd(unittest.TestCase):
    """`design` on fake interpreters: the stock child prints `KERNELS route=stock|default`, the kit process `route=exact|fast|fast_compile`;
    the guard's refusals exit 5 with the accelerator and the route named; the census lands in the manifest and in the stock proof."""

    def setUp(self):
        self.td = tempfile.TemporaryDirectory()
        self.bin = fx.fake_gpu(os.path.join(self.td.name, "bin"))
        self.patched = fx.fake_torch(fx.make_site(os.path.join(self.td.name, "p"), "patched"))
        self.pristine = fx.make_site(os.path.join(self.td.name, "s"), "pristine")
        fx.stub_cli(self.patched)
        fx.stub_cli(self.pristine)
        self.spec = fx.fake_spec(os.path.join(self.td.name, "public.json"))
        self.ckpt = os.path.join(self.td.name, "fake.ckpt")
        open(self.ckpt, "wb").write(b"not a checkpoint")
        self.stockpy = fx.make_venv(self.td.name, self.pristine)
        self.kit_env = {"RFD3_FAKE_IMPORT_HOIST": "1"}                                             # the model build imports the kit modules: a clean (non-partial) kit pass

    def tearDown(self):
        self.td.cleanup()

    def _design(self, site, mode, extra=(), env=None, out="out", python=None, tail=(), ckpt=True, batch=True):
        out_dir = os.path.join(self.td.name, out)
        tail = (["diffusion_batch_size=2"] if batch else []) + list(tail)                    # run parameters: upstream's own tokens after --
        args = (["design", "--mode", mode] + (["--ckpt", self.ckpt] if ckpt else []) + list(extra) + [f"inputs={self.spec}", f"out_dir={out_dir}"] + list(tail))   # the line: upstream's own tokens, verbatim; the kit's flags before them
        return fx.run_cli(args, site, env=env, bin_dir=self.bin, python=python), out_dir

    def _n_batches(self):
        return len(json.load(open(self.spec)))                                                     # the stub runs one diffusion batch per spec key

    def _kernels_line(self, stderr):
        lines = [l for l in stderr.splitlines() if "KERNELS route=" in l]                          # the cross-engine grep
        self.assertEqual(len(lines), 1, stderr)                                                    # exactly one per pass
        return lines[0]

    def test_an_interpreter_without_apex_is_named_on_default_reported_on_stock_and_refused_on_a_kit_route(self):
        """On an interpreter where `apex` does not import, upstream's guard binds torch.nn.RMSNorm and LOGS it (layer_utils.py:19-21). On route
        `default` — upstream as a README install runs it, which ships no apex — that is simply the class the KERNELS line names (no expectation,
        no report line, the run's own exit); on route `stock` (compiled upstream on the pinned stack, which carries apex) the stock child
        REPORTS it off that record (`KERNELS report-only …`) and upstream's run proceeds to its own exit; a kit route is refused by name there,
        exit 5, before any roll-out."""
        r, out = self._design(self.pristine, "off", env={"RFD3_FAKE_NO_APEX": "1"}, python=self.stockpy, out="noapex_s")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("report-only", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
        self.assertRegex(self._kernels_line(r.stderr), r"^\[rfdiffusion3-opt\] KERNELS route=default .* rmsnorm=fallback:torch\.nn\.RMSNorm\(record:layer_utils\.py:19-21\) ")
        self.assertTrue(os.path.exists(os.path.join(out, "cli_seen.json")))                      # upstream ran to its end
        k = manifest.read(out)["stock_env_proof"]["after"]["kernels"]
        self.assertEqual((k["refused"], k["report_only"]), (None, {}))
        self.assertEqual(manifest.read(out)["exit_code"], 0)
        r, out = self._design(self.pristine, "off", extra=["compile_model=true"], env={"RFD3_FAKE_NO_APEX": "1"}, python=self.stockpy, out="noapex_c")   # compiled stock: apex is part of its image, a torch-bound net is reported
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("[rfdiffusion3-opt] KERNELS report-only on route stock: rmsnorm=fallback:torch.nn.RMSNorm(record:layer_utils.py:19-21), expected engaged — upstream's run proceeds", r.stderr)
        self.assertNotIn("NOT ACTIVE", r.stderr)
        k = manifest.read(out)["stock_env_proof"]["after"]["kernels"]
        self.assertEqual((k["refused"], k["report_only"]), (None, {"rmsnorm": {"word": "fallback:torch.nn.RMSNorm(record:layer_utils.py:19-21)", "expected": "engaged", "when": "before-timing"}}))
        r, out = self._design(self.patched, "exact", env=dict(self.kit_env, RFD3_FAKE_NO_APEX="1"), out="noapex_k")
        self.assertEqual(r.returncode, 5, r.stderr)
        self.assertIn("NOT ACTIVE: KERNELS refused on route exact — rmsnorm=fallback:torch.nn.RMSNorm(3-modules)@torch0.0.0+fake, expected engaged; exit 5", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(out, "cli_seen.json")))                     # refused before any roll-out
        self.assertEqual(manifest.read(out)["kernels"]["refused"], {"accel": "rmsnorm", "word": "fallback:torch.nn.RMSNorm(3-modules)@torch0.0.0+fake", "expected": "engaged", "when": "before-timing"})
        self.assertEqual(manifest.read(out)["kernels"]["report_only"], {})

    def test_default_route_in_the_stock_child(self):
        """The stock child on route default: the STOCK proof, upstream's entry point in-process, the census read off upstream's records — one
        KERNELS line, no ATTN_PREFLIGHT / ATTN_PATH lines (nothing counts calls in a stock process), no upstream callable re-bound."""
        r, out = self._design(self.pristine, "off", python=self.stockpy)
        self.assertEqual(r.returncode, 0, r.stderr)
        line = self._kernels_line(r.stderr)
        self.assertIn("KERNELS route=default mode=off B=2 compile=off-by-route:compile_model=false atom_attn=engaged:dense-sdpa(", line)
        self.assertIn(" cueq=n/a-upstream:", line); self.assertIn(" ds4sci=n/a-upstream:", line)
        proof = manifest.read(out)["stock_env_proof"]                                             # the stock child's proof, folded into the manifest
        self.assertTrue(proof["ok"]); self.assertEqual(proof["after"]["exit_code"], 0)
        k = proof["after"]["kernels"]
        self.assertEqual((k["route"], k["refused"], k["line"]), ("default", None, line))
        self.assertEqual(k["batch_seconds"], [0.01])                                               # one spec key = one batch: upstream's clock, trapped
        man = manifest.read(out)
        self.assertEqual(man["kernels"]["route"], "default"); self.assertEqual(man["settings"]["route"], "default")
        seen = json.load(open(os.path.join(out, "cli_seen.json")))
        self.assertNotIn("compile_model=true", seen["argv"]); self.assertEqual(seen["wrapped"], [])
        self.assertNotIn("rfdiffusion3_opt.stack", seen["modules"])                                # the census brought nothing of the activation core into the stock process
        n = self._n_batches()
        self.assertNotIn("] ITEM ", r.stderr)                                                     # nothing printed per item
        self.assertNotIn("] ATTN_PATH ", r.stderr); self.assertNotIn("] ATTN_PREFLIGHT ", r.stderr)   # the stock routes count no calls and probe no step (no shim, no wrap in a stock process)
        self.assertEqual(k["detail"]["observation"], census.STOCK_OBSERVATION)
        self.assertEqual((k["attn_path_line"], k["attn_preflight_lines"], len(k["items"])), (None, [], n))
        self.assertEqual(k["ckpt_path"], self.ckpt); self.assertRegex(line, r" ckpt=fake\.ckpt@[0-9a-f]{12}$")   # the line's own ckpt_path= token, hashed
        self.assertRegex(line, r" rmsnorm=engaged:apex\.FusedRMSNorm\(record:layer_utils\.py:16\) torch=0\.0\.0\+fake cuda=none tf32_matmul=absent tf32_cudnn=absent alloc_conf=\S+ apex=present:0\.1 ckpt=")   # upstream's import guard bound the (fake) site's apex FusedRMSNorm and logged it
        self.assertFalse(os.path.exists(os.path.join(out, "attn_path.json")))                     # no census file beside the designs

    def test_a_sparse_first_decision_is_named_on_the_stock_routes(self):
        """A sparse decision (free memory) on the stock routes — compiled `stock` and eager `default` alike — is NAMED off upstream's own path
        record: the KERNELS line's atom_attn word says sparse, nothing is refused or reported as a miss, no ATTN_PREFLIGHT line is printed (no
        step is probed in a stock process), and the pass keeps its own exit: stock is upstream through its own surface. (The kit routes are
        refused there by name: test_preflight_sparse_refuses_the_gated_routes_only.)"""
        r, out = self._design(self.pristine, "off", extra=["compile_model=true"], env={"RFD3_FAKE_ATTN": "memory"}, python=self.stockpy)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertNotIn("] ATTN_PREFLIGHT ", r.stderr); self.assertNotIn("] ATTN_PATH ", r.stderr)
        self.assertNotIn("attn_preflight=", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr); self.assertNotIn("report-only", r.stderr)
        self.assertIn("atom_attn=fallback:sparse(", self._kernels_line(r.stderr))
        r, out = self._design(self.pristine, "off", env={"RFD3_FAKE_ATTN": "memory"}, python=self.stockpy, out="o2")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("atom_attn=fallback:sparse(", self._kernels_line(r.stderr))
        self.assertNotIn("report-only", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr); self.assertTrue(os.path.exists(os.path.join(out, "cli_seen.json")))

    def test_no_override_runs_upstream_untouched_on_the_default_route(self):
        """No token after --: the child execs `rfd3 design out_dir= inputs= ckpt_path=` and nothing else; KERNELS' B is upstream's shipped
        batch size (the line names none), ckpt= names the line's checkpoint token; one token = exactly that token on the line."""
        r, out = self._design(self.pristine, "off", python=self.stockpy, batch=False)
        self.assertEqual(r.returncode, 0, r.stderr)
        seen = json.load(open(os.path.join(out, "cli_seen.json")))
        self.assertEqual(seen["argv"][1:], [f"inputs={self.spec}", f"out_dir={out}", f"ckpt_path={self.ckpt}"])   # no seed, no batch size, no sampler token
        design_line = next(l for l in r.stderr.splitlines() if l.startswith("[rfdiffusion3-opt] design mode=off "))   # the composed child command, printed before the child starts
        self.assertTrue(design_line.endswith(f" -- design inputs={self.spec} out_dir={out} ckpt_path={self.ckpt}"), design_line)
        line = self._kernels_line(r.stderr)
        self.assertIn("KERNELS route=default mode=off B=8 compile=off-by-route:compile_model=false atom_attn=engaged:dense-sdpa(", line)   # B: upstream's shipped value (settings.BATCH_DEFAULT)
        self.assertRegex(line, r" ckpt=fake\.ckpt@[0-9a-f]{12}$")
        man = manifest.read(out)
        self.assertEqual(man["settings"]["values"], {}); self.assertNotIn("preset", man["settings"])
        self.assertEqual(man["command"], ["rfd3", "design", f"inputs={self.spec}", f"out_dir={out}", f"ckpt_path={self.ckpt}"])
        self.assertEqual(man["kernels"]["batch"], "8"); self.assertEqual(man["kernels"]["detail"]["batch_source"], "upstream-default(rfdiffusion3.yaml:20)")
        r2, out2 = self._design(self.pristine, "off", python=self.stockpy, batch=False, tail=["diffusion_batch_size=16"], out="o2")
        self.assertEqual(r2.returncode, 0, r2.stderr)
        self.assertEqual(json.load(open(os.path.join(out2, "cli_seen.json")))["argv"][1:], [f"inputs={self.spec}", f"out_dir={out2}", "diffusion_batch_size=16", f"ckpt_path={self.ckpt}"])   # exactly one extra token
        self.assertIn(" mode=off B=16 ", self._kernels_line(r2.stderr))
        self.assertEqual(manifest.read(out2)["settings"]["values"], {"diffusion_batch_size": "16"})

    def test_a_mid_run_flip_to_sparse_cannot_happen_on_a_kit_route_and_is_invisible_on_a_stock_route(self):
        """Upstream logs its first decision only; a later SPARSE (free memory) is invisible to the path record. A STOCK route reads that record
        and nothing else (no shim counts calls in a stock process): the word says dense, the pass keeps its own exit. The KIT process PINS the
        pre-flight decision (dense) — the model's calls answer from it, the flip cannot happen — and its shim counts every call."""
        r, out = self._design(self.pristine, "off", env={"RFD3_FAKE_ATTN": "flip"}, python=self.stockpy)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("atom_attn=engaged:dense-sdpa(", self._kernels_line(r.stderr))               # the record said DENSE at the first call
        self.assertNotIn("] ATTN_PATH ", r.stderr); self.assertNotIn("] ATTN_PREFLIGHT ", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
        n = self._n_batches()
        r, out = self._design(self.patched, "exact", env=dict(self.kit_env, RFD3_FAKE_ATTN="flip"), out="o2")   # the kit process
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertRegex(r.stderr, r"(?m)^\[rfdiffusion3-opt\] ATTN_PREFLIGHT route=exact B=2 n_atoms=999 L=120 atom_sparse=0 free_gib=70\.000 est_dense_gib=0\.030 pinned=1$")
        self.assertIn(f"[rfdiffusion3-opt] ATTN_PATH atom_dense={2 * n} atom_sparse=0 token_full={n} compiled=0 calls_uncounted=0", r.stderr)
        self.assertIn("atom_attn=engaged:dense-sdpa(", self._kernels_line(r.stderr))               # upstream's record, made by the first pinned call
        krec = manifest.read(out)["kernels"]
        self.assertTrue(any("Atom attention path: DENSE-SDPA (0.0 GiB bias; pinned at pre-flight) [D=2 L=999 k=128 H=4]." in rec for rec in krec["records"]), krec["records"])   # upstream's record function, called by the first pinned answer
        got = krec["attn_path"]
        self.assertEqual({kk: got[kk] for kk in ("atom_dense", "atom_sparse", "token_full", "pin_hits", "pin_misses")}, {"atom_dense": 2 * n, "atom_sparse": 0, "token_full": n, "pin_hits": 2 * n, "pin_misses": 0})
        self.assertEqual(got["preflight"][0]["decisions"], {"2": True, "4": True})                  # both atom head counts evaluated and pinned

    def test_stock_route_in_the_stock_child(self):
        r, out = self._design(self.pristine, "off", extra=["compile_model=true"], python=self.stockpy)
        self.assertEqual(r.returncode, 0, r.stderr)
        line = self._kernels_line(r.stderr)
        self.assertRegex(line, r"KERNELS route=stock mode=off B=2 compile=engaged:torch\.compile\(record:engine\.py:265-269\) atom_attn=engaged:dense-sdpa\(\S+ cueq=n/a-upstream:\S+ ds4sci=n/a-upstream:\S+ rmsnorm=engaged:apex\.FusedRMSNorm\(record:layer_utils\.py:16\) compile_s=\S+ .*\bgraphs=\d+ frames=\d+/\d+ graph_breaks=\d+ "
                               r"torch=0\.0\.0\+fake cuda=none tf32_matmul=absent tf32_cudnn=absent alloc_conf=\S+ apex=\S+ ckpt=fake\.ckpt@[0-9a-f]{12}$")
        self.assertNotIn("] ATTN_PATH ", r.stderr); self.assertNotIn("] ATTN_PREFLIGHT ", r.stderr); self.assertNotIn("report-only", r.stderr)
        seen = json.load(open(os.path.join(out, "cli_seen.json")))
        self.assertEqual([t for t in seen["argv"] if "=" in t and t.split("=")[0] in ("compile_model", "diffusion_batch_size")], ["compile_model=true", "diffusion_batch_size=2"])      # upstream's own tokens, verbatim, in the caller's order
        self.assertEqual(seen["wrapped"], ["encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder"])
        man = manifest.read(out)
        self.assertEqual(man["settings"]["route"], "stock"); self.assertEqual(man["settings"]["values"], {"compile_model": "true", "diffusion_batch_size": "2"})
        self.assertEqual(man["kernels"]["words"]["compile"], "engaged:torch.compile(record:engine.py:265-269)")
        r2, _ = self._design(self.pristine, "off", tail=["compile_model=true"], python=self.stockpy, out="out2")   # the literal token after -- is the same route
        self.assertEqual(r2.returncode, 0, r2.stderr); self.assertIn("KERNELS route=stock ", r2.stderr)

    def test_stock_route_reports_a_skipped_compile_and_proceeds(self):
        r, out = self._design(self.pristine, "off", extra=["compile_model=true"], env={"RFD3_FAKE_COMPILE_SKIP": "1"}, python=self.stockpy)
        self.assertEqual(r.returncode, 0, r.stderr)                                                # upstream's own exit code: the miss is a report on the stock route
        self.assertIn("KERNELS route=stock mode=off B=2 compile=fallback:", self._kernels_line(r.stderr))
        self.assertRegex(r.stderr, r"(?m)^\[rfdiffusion3-opt\] KERNELS report-only on route stock: compile=fallback:\S+, expected engaged — upstream's run proceeds$")
        self.assertNotIn("NOT ACTIVE", r.stderr)
        self.assertTrue(os.path.exists(os.path.join(out, "cli_seen.json")))                      # the run went on and wrote its designs
        proof = manifest.read(out)["stock_env_proof"]
        self.assertEqual(proof["after"]["exit_code"], 0); self.assertIsNone(proof["after"]["kernels"]["refused"])
        self.assertEqual((list(proof["after"]["kernels"]["report_only"]), proof["after"]["kernels"]["report_only"]["compile"]["when"]), (["compile"], "before-timing"))
        man = manifest.read(out)
        self.assertEqual((man["exit_code"], man["kernels"]["refused"], list(man["kernels"]["report_only"])), (0, None, ["compile"]))

    def test_stock_route_reports_a_compile_that_compiled_nothing(self):
        r, out = self._design(self.pristine, "off", extra=["compile_model=true"], env={"RFD3_FAKE_DYNAMO": "nograph"}, python=self.stockpy)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("compile=fallback:no-graph-compiled(dynamo-counters)", self._kernels_line(r.stderr))
        self.assertIn("[rfdiffusion3-opt] KERNELS report-only on route stock: compile=fallback:no-graph-compiled(dynamo-counters), expected engaged — upstream's run proceeds", r.stderr)
        self.assertEqual(manifest.read(out)["kernels"]["report_only"]["compile"]["when"], "finish")   # dynamo's counters are read at the pass's end

    def test_default_route_names_a_sparse_atom_attention(self):
        """On route default a sparse atom-attention path — upstream's free-memory SPARSE record at the first call, no path record at all, or
        upstream's own low-memory switch on the line (its record says `low memory mode`) — is NAMED: the KERNELS line carries the word read off
        upstream's record, nothing is reported as a miss (the stock routes judge no attention path) and the run keeps its own exit code."""
        cases = (({"RFD3_FAKE_ATTN": "memory"}, [], "fallback:sparse(needs-10.0-GiB-of-1.0-GiB-free)"),
                 ({"RFD3_FAKE_ATTN": "silent"}, [], "absent:attention-never-ran"),
                 ({}, ["low_memory_mode=True"], "fallback:sparse(low-memory-mode;line)"))   # upstream's low-memory path never reaches the decision (no record): the line's own token names it
        for n, (env, tail, word) in enumerate(cases):
            r, out = self._design(self.pristine, "off", env=env, tail=tail, python=self.stockpy, out=f"sparse{n}")
            self.assertEqual(r.returncode, 0, r.stderr)
            self.assertIn(f"atom_attn={word}", self._kernels_line(r.stderr), (n, r.stderr[-2000:]))
            self.assertNotIn("report-only", r.stderr); self.assertNotIn("NOT ACTIVE", r.stderr)
            self.assertTrue(os.path.exists(os.path.join(out, "cli_seen.json")))                  # the designs were written
            k = manifest.read(out)["kernels"]
            self.assertEqual((k["refused"], k["report_only"].get("atom_attn")), (None, None), word)

    def test_exact_route_in_the_kit_process(self):
        r, out = self._design(self.patched, "exact", env=self.kit_env)
        self.assertEqual(r.returncode, 0, r.stderr)
        line = self._kernels_line(r.stderr)
        self.assertIn("KERNELS route=exact mode=exact B=2 compile=off-by-route:compile_model=false atom_attn=engaged:dense-sdpa(", line)
        man = manifest.read(out)
        self.assertEqual(man["kernels"]["route"], "exact"); self.assertEqual(man["kernels"]["line"], line); self.assertIsNone(man["kernels"]["refused"])
        self.assertIn("APPLIED rfd3.model.hoist RFD3_HOIST=True", r.stderr)                       # the exit rule's evidence lines are unchanged beside it
        n = self._n_batches()
        self.assertNotIn("] ITEM ", r.stderr)                                                     # no per-item line from the kit
        self.assertIn(f"[rfdiffusion3-opt] ATTN_PATH atom_dense={2 * n} atom_sparse=0 token_full={n} compiled=0 calls_uncounted=0", r.stderr)
        apj = manifest.read(out)["kernels"]["attn_path"]
        self.assertEqual({kk: apj[kk] for kk in ("atom_dense", "atom_sparse", "token_full", "compiled", "calls_uncounted")}, {"atom_dense": 2 * n, "atom_sparse": 0, "token_full": n, "compiled": False, "calls_uncounted": False})
        self.assertRegex(r.stderr, r"(?m)^\[rfdiffusion3-opt\] ATTN_PREFLIGHT route=exact B=2 n_atoms=999 L=120 atom_sparse=0 ")
        self.assertRegex(line, r" rmsnorm=engaged:apex\.FusedRMSNorm\(3-modules\)@apex0\.1 torch=0\.0\.0\+fake cuda=none tf32_matmul=absent tf32_cudnn=absent alloc_conf=\S+ apex=present:0\.1 ckpt=fake\.ckpt@[0-9a-f]{12}$")   # the kit's layer_utils carries upstream's import guard unchanged: the site's apex FusedRMSNorm
        self.assertEqual(man["kernels"]["rmsnorm"], line.split(" rmsnorm=")[1].split()[0])            # the word reaches the manifest
        self.assertEqual(man["kernels"]["items"][0]["clock_s"], "0.01"); self.assertFalse(os.path.exists(os.path.join(out, "attn_path.json")))

    def test_exact_refuses_compile_by_name_before_anything_runs(self):
        r, out = self._design(self.patched, "exact", extra=["compile_model=true"], env=self.kit_env)
        self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, r.stderr)
        self.assertIn("NOT ACTIVE: compile_model=true under exact", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(out, "cli_seen.json")))
        r, out = self._design(self.patched, "exact", tail=["compile_model=True"], env=self.kit_env, out="o2")
        self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, r.stderr)

    def test_fast_route_and_compile_refused_under_fast(self):
        """(--allow-partial: the stub roll-out never engages the graph sampler, the exit rule records that partial; the census is what is read here.)"""
        r, out = self._design(self.patched, "fast", extra=["--allow-partial"], env=self.kit_env)
        # the stub never rolls out, so the kit's compile lever (armed at the hoist module's import, applied at the first roll-out) never applies:
        # the census names it a fallback and the guard refuses the pass by name (exit 5) — a fast pass without its compile is never timed as fast
        self.assertEqual(r.returncode, census.EXIT_KERNELS, r.stderr)
        self.assertIn("KERNELS route=fast mode=fast B=2 compile=fallback:kit-compile-never-applied", self._kernels_line(r.stderr))
        self.assertIn("refused on route fast — compile=fallback:kit-compile-never-applied", r.stderr)
        self.assertEqual(manifest.read(out)["settings"]["route"], "fast")
        r, out = self._design(self.patched, "fast", extra=["--allow-partial", "compile_model=true"], env=self.kit_env, out="o2")   # no such route: refused by name before anything runs
        self.assertEqual(r.returncode, report.EXIT_NOT_ACTIVE, r.stderr)
        self.assertIn("NOT ACTIVE: compile_model=true under fast", r.stderr)
        self.assertNotIn("KERNELS route=", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(out, "cli_seen.json")))

    def test_kit_route_refusals_exit_5(self):
        r, out = self._design(self.patched, "exact", env=dict(self.kit_env, RFD3_FAKE_ATTN="small"))   # upstream's rule says SPARSE (L <= k): exact replays captured steps, so the pre-flight decision is its gate and refuses first
        self.assertEqual(r.returncode, 5, r.stderr)
        self.assertRegex(r.stderr, r"NOT ACTIVE: KERNELS refused on route exact — attn_preflight=sparse\(B=2,n_atoms=\d+,free_gib=70\.000,est_dense_gib=[\d.]+\), expected dense; exit 5")
        self.assertEqual(manifest.read(out)["exit_code"], 5)
        self.assertEqual(manifest.read(out)["kernels"]["refused"]["accel"], "attn_preflight")
        r, out = self._design(self.patched, "fast", extra=["--allow-partial"], env=dict(self.kit_env, RFD3_FAKE_ATTN="memory"), out="o2")   # upstream's free-memory rule says SPARSE: on fast the pre-flight decision refuses, at the first denoising step
        self.assertEqual(r.returncode, 5, r.stderr)                                                # failed ranks before partial: the refusal's own code
        self.assertIn("NOT ACTIVE: KERNELS refused on route fast — attn_preflight=sparse(B=2,n_atoms=999,free_gib=70.000,est_dense_gib=0.030), expected dense; exit 5", r.stderr)
        self.assertEqual(manifest.read(out)["kernels"]["refused"]["accel"], "attn_preflight")
        self.assertFalse(os.path.exists(os.path.join(out, "cli_seen.json")))                     # unwound before any roll-out completed
        r, out = self._design(self.patched, "exact", env=dict(self.kit_env, RFD3_FAKE_ATTN="memory"), out="o3")   # the same rule on exact (graph replay: gated like fast): the pre-flight decision refuses at the first denoising step
        self.assertEqual(r.returncode, 5, r.stderr)
        self.assertIn("NOT ACTIVE: KERNELS refused on route exact — attn_preflight=sparse(B=2,n_atoms=999,free_gib=70.000,est_dense_gib=0.030), expected dense; exit 5", r.stderr)
        self.assertEqual(manifest.read(out)["kernels"]["refused"]["accel"], "attn_preflight")


class TestGatherWords(unittest.TestCase):
    """The gather lever's final census words (gather.final_word, census.KINDS): engaged only when every atom call was served (stock=0);
    some calls refused by name -> the DISTINCT kind `partial:gather_attn(stock=<m>,stock_by=<reasons>)[served=<n>]`; none served -> fallback:gather-never-served;
    a kernel that could not install -> fallback:gather-unavailable; route fast expects atom_attn engaged, so partial and fallback words are refused (exit 5)."""

    def setUp(self):
        self.saved = {k: (dict(v) if isinstance(v, dict) else v) for k, v in gather.STATS.items()}

    def tearDown(self):
        gather.STATS.clear(); gather.STATS.update(self.saved)

    def _tally(self, served, stock, stock_by=None, unavailable=None, on=True):
        gather.STATS.update({gather.SWITCH: on, "installs": 1 if unavailable is None else 0, "unavailable": unavailable, "served": served, "stock": stock,
                             "stock_by": dict(stock_by or {}), "kernel": "gather_attn/1.0"})

    def test_engaged_only_when_every_call_was_served(self):
        self._tally(27, 0)
        w = gather.final_word(census.gather_word())
        self.assertEqual(w, "engaged:gather_attn(gather_attn/1.0;F5.gather_attn)[served=27,stock=0]"); self.assertEqual(census.kind_of(w), "engaged")
        self.assertEqual(census.expected_kinds("fast")["atom_attn"], "engaged")                    # the fast route's expectation this word meets

    def test_partial_is_a_distinct_kind_refused_on_fast(self):
        self._tally(25, 2, {"cell-uncertified:dh32/k100/bfloat16": 2})
        w = gather.final_word(census.gather_word())
        self.assertEqual(w, "partial:gather_attn(stock=2,stock_by=cell-uncertified:dh32/k100/bfloat16:2)[served=25]")
        self.assertEqual(census.kind_of(w), "partial"); self.assertIn("partial", census.KINDS); self.assertNotEqual(census.kind_of(w), census.expected_kinds("fast")["atom_attn"])
        self.assertEqual(census.refusal_line("fast", "atom_attn", w, "engaged"),
                         f"[rfdiffusion3-opt] NOT ACTIVE: KERNELS refused on route fast — atom_attn={w}, expected engaged; exit 5")
        self._tally(1, 3, {"v-dtype-float32": 2, "grad": 1})                                          # reasons sorted, joined by +
        self.assertEqual(gather.final_word(census.gather_word()), "partial:gather_attn(stock=3,stock_by=grad:1+v-dtype-float32:2)[served=1]")

    def test_never_served_and_unavailable_are_named_fallbacks(self):
        self._tally(0, 27, {"cell-uncertified:dh32/k100/bfloat16": 27})
        w = gather.final_word(census.gather_word()); self.assertEqual(w, "fallback:gather-never-served(stock=27)"); self.assertEqual(census.kind_of(w), "fallback")
        self._tally(0, 0)
        self.assertEqual(gather.final_word(census.gather_word(), calls_seen=False), "engaged:gather_attn(gather_attn/1.0;F5.gather_attn)[served=0,stock=0]")   # no atom call ran: nothing to serve, nothing refused
        self.assertEqual(gather.final_word(census.gather_word(), calls_seen=True), "fallback:gather-never-served(stock=0)")
        self._tally(0, 0, unavailable="no-triton")
        w = gather.final_word(census.gather_word()); self.assertEqual(w, "fallback:gather-unavailable(no-triton)"); self.assertEqual(census.kind_of(w), "fallback")
        self.assertEqual(census.gather_word(), "fallback:gather-unavailable(no-triton)")

    def test_every_word_kind_is_a_known_kind(self):
        for served, stock, un in ((3, 0, None), (3, 1, None), (0, 3, None), (0, 0, "not-cuda")):
            self._tally(served, stock, {"grad": stock} if stock else None, unavailable=un)
            self.assertIn(census.kind_of(gather.final_word(census.gather_word())), census.KINDS)


class TestOomClassifier(unittest.TestCase):
    """Out-of-memory on the served paths: the core's one classifier, `opt_core.oom.is_oom`, correctly separates an out-of-memory from any
    other failure; the graph sampler's capture-failure reroute (the one served handler that could see an out-of-memory) re-raises it
    uncounted instead of falling back to the eager denoiser, while any other capture failure still reroutes and is counted."""

    def test_is_oom_is_the_core_classifier(self):
        from opt_core.oom import is_oom
        self.assertTrue(is_oom(MemoryError()))
        self.assertTrue(is_oom(type("OutOfMemoryError", (RuntimeError,), {})("CUDA out of memory (mock)")))   # torch's class, by name, without torch
        self.assertTrue(is_oom(RuntimeError("CUDA out of memory. Tried to allocate 2.00 GiB")))
        self.assertTrue(is_oom(RuntimeError("CUDA error: out of memory")))
        self.assertFalse(is_oom(RuntimeError("graph capture failed (mock)")))
        self.assertFalse(is_oom(ValueError("x")))


def _load_graph_sampler():
    """The carried target file itself, loaded from the tree (its imports: the standard library, torch, opt_core.oom, and the kit's hoist module for
    the dynamo-boundary marker — hoist.py is registered by path under its package name first; upstream's `rfd3` package is not importable here)."""
    import importlib.util
    if "rfd3.model.hoist" not in sys.modules:
        hs = importlib.util.spec_from_file_location("rfd3.model.hoist", os.path.join(TREE_OOM, HOIST_OOM))
        hoist = importlib.util.module_from_spec(hs)
        sys.modules["rfd3.model.hoist"] = hoist                                       # registered before exec: the import system serves the full name from sys.modules
        hs.loader.exec_module(hoist)
    spec = importlib.util.spec_from_file_location("rfd3_cudagraph_sampler_under_test", os.path.join(TREE_OOM, GRAPH_SAMPLER_OOM))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestOutOfMemoryPropagatesThroughTheServedRollout(unittest.TestCase):
    """The graph sampler's roll-out — `SampleDiffusionWithMotifGraph.sample_diffusion_like_af3`, the function upstream's
    `ConditionalDiffusionSampler` calls under RFD3_CUDAGRAPH=1 (upstream itself is not importable here) — driven in-process on a fake stock
    sampler, with its capture (`_get_entry`: warm-up forwards + CUDA-graph capture, the kernel launch the fallback wraps) mocked to fail.
    Tensor code: it needs a CPU torch and skips by name without one (the package venv of tests/README carries none)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("needs a CPU torch: the served roll-out is tensor code (tests/README.md, Package tests)")
        cls.torch = torch
        cls.OOM = getattr(torch, "OutOfMemoryError", None) or torch.cuda.OutOfMemoryError     # torch >= 2.5 spells it torch.OutOfMemoryError
        cls.mod = _load_graph_sampler()

    def _rollout(self, capture_error):
        torch, mod = self.torch, self.mod
        D, L, I = 2, 5, 3

        class Stock:                                     # the attributes the roll-out reads off the stock sampler (upstream's SampleDiffusionWithMotif)
            use_classifier_free_guidance = False
            allow_realignment = False
            s_jitter_origin = 0.0
            fraction_of_steps_to_fix_motif = 0.0
            gamma_0, gamma_min, noise_scale, step_scale, sigma_data, n_recycle = 0.6, 1.0, 1.003, 1.5, 16.0, 1

            def _construct_inference_noise_schedule(self, device, partial_t=None):
                return torch.tensor([8.0, 2.0, 0.5], device=device)                      # three levels: two denoiser steps

            def sample_diffusion_like_af3(self, **kw):
                raise AssertionError("the stock roll-out route is for unsupported options only")

        class OnDevice:                                  # the graph route is for CUDA tensors; a CPU tensor presented as one reaches the capture
            is_cuda = True

            def __init__(self, t):
                self._t, self.device = t, t.device

            def clone(self):
                return self._t.clone()

        calls = []

        def diffusion_module(*, X_noisy_L, t, f, n_recycle, **io):                          # the eager denoiser the fallback reroutes to
            calls.append(tuple(X_noisy_L.shape))
            return {"X_L": X_noisy_L * 0.5, "sequence_logits_I": torch.zeros(D, I, 20), "sequence_indices_I": torch.zeros(D, I, dtype=torch.long)}

        def failing_capture(*a, **k):
            raise capture_error

        wrapper = mod.SampleDiffusionWithMotifGraph(Stock())
        wrapper._get_entry = failing_capture
        f = {"is_motif_atom_with_fixed_coord": torch.zeros(L, dtype=torch.bool), "ref_element": torch.zeros(L, 3)}
        before = mod.GRAPH_STATS["fallbacks"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("RFD3_CUDAGRAPH_")}
        with mock.patch.dict(os.environ, env, clear=True), torch.no_grad():
            out = wrapper.sample_diffusion_like_af3(f=f, diffusion_module=diffusion_module, diffusion_batch_size=D,
                                                    coord_atom_lvl_to_be_noised=OnDevice(torch.zeros(D, L, 3)),
                                                    initializer_outputs={}, ref_initializer_outputs=None, f_ref=None)
        return wrapper, out, calls, mod.GRAPH_STATS["fallbacks"] - before

    def test_an_out_of_memory_at_capture_propagates_uncounted(self):
        oom = self.OOM("CUDA out of memory (mock)")                                        # constructible without a GPU on the CPU build
        from opt_core.oom import is_oom
        self.assertTrue(is_oom(oom))
        fallbacks_before = self.mod.GRAPH_STATS["fallbacks"]
        with self.assertRaises(self.OOM):
            self._rollout(oom)
        self.assertEqual(self.mod.GRAPH_STATS["fallbacks"], fallbacks_before)              # not counted as a fallback: the design ended with the OOM

    def test_any_other_capture_failure_still_reroutes_to_the_eager_denoiser(self):
        wrapper, out, calls, counted = self._rollout(RuntimeError("graph capture failed (mock)"))
        self.assertEqual(counted, 1)
        self.assertEqual(wrapper.stats["fallbacks"], 1)
        self.assertTrue(wrapper.stats["fallback_errors"][0].startswith("RuntimeError: graph capture failed (mock)"))
        self.assertEqual(len(calls), 2)                                                     # both denoiser steps ran eagerly and the roll-out completed
        self.assertEqual(tuple(out["X_L"].shape), (2, 5, 3))
        self.assertEqual(len(out["sequence_entropy_traj"]), 2)


class TestUnservedAtInitialize(unittest.TestCase):
    """A kit route meeting, at the engine's initialize, a computation the mode's levers do not drive (modes.UNSERVED) that the design line's
    reader could not see: `unserved_by_engine` reads the engine's composed sampler config / the built model's own CFG decision / the
    low-memory switch the engine exported; `Observer.after_initialize` prints ONE NOT ACTIVE line and raises NotServed (SystemExit 3)
    before anything attaches; `finish` then prints no KERNELS line and keeps the NOT ACTIVE code; the record carries `not_served`. The
    stock routes are upstream's own business: nothing is read there."""

    def _cfg_engine(self, asked=None, decided=False, cfg_scale=1.5, kind=None, sampler_cls="SampleDiffusionWithMotifGraph"):
        e = _engine()
        if asked is not None or kind is not None:
            e.inference_sampler_overrides = dict({"cfg_scale": cfg_scale, "num_timesteps": 200}, **({"use_classifier_free_guidance": asked} if asked is not None else {}), **({"kind": kind} if kind else {}))
        net = census.locate_net(e)
        net.use_classifier_free_guidance = decided
        net.inference_sampler = type("ConditionalDiffusionSampler", (), {})()                        # RFD3.inference_sampler: the holder whose `.sampler` the model built (inference_sampler.py:596-614)
        net.inference_sampler.sampler = type(sampler_cls, (), {})()
        return e

    def test_unserved_by_engine_reads_the_engine(self):
        env = {k: v for k, v in os.environ.items() if k != census.LOWMEM_ENV}
        with mock.patch.dict(os.environ, env, clear=True):
            for mode in ("exact", "fast"):
                self.assertIsNone(census.unserved_by_engine(self._cfg_engine(asked=False), mode))
                self.assertIsNone(census.unserved_by_engine(self._cfg_engine(), mode))                       # no overrides attribute, model says no
                self.assertEqual(census.unserved_by_engine(self._cfg_engine(asked=True), mode),
                                 modes.unserved_reason(mode, "classifier-free guidance", "inference_sampler.use_classifier_free_guidance on in the engine's composed config"))
                self.assertEqual(census.unserved_by_engine(self._cfg_engine(asked=False, decided=True), mode),
                                 modes.unserved_reason(mode, "classifier-free guidance", "the built model's use_classifier_free_guidance"))
                with mock.patch.dict(os.environ, {census.LOWMEM_ENV: "1"}):
                    self.assertEqual(census.unserved_by_engine(self._cfg_engine(asked=False), mode),
                                     modes.unserved_reason(mode, "low_memory_mode", f"{census.LOWMEM_ENV}=1 exported by the engine (low_memory_mode on in its config)"))
                with mock.patch.dict(os.environ, {census.LOWMEM_ENV: "0"}):
                    self.assertIsNone(census.unserved_by_engine(self._cfg_engine(asked=False), mode))
                self.assertIsNone(census.unserved_by_engine(self._cfg_engine(asked=True, cfg_scale=1.0), mode))          # upstream's own rule (modes.cfg_active): the switch with cfg_scale 1.0 is CFG off
                self.assertEqual(census.unserved_by_engine(self._cfg_engine(asked="true", cfg_scale="2.0"), mode)[:40], modes.unserved_reason(mode, "classifier-free guidance", "x")[:40])   # string spellings of a composed config
                self.assertEqual(census.unserved_by_engine(self._cfg_engine(kind="symmetry"), mode),               # the symmetry sampler asked in the composed config ...
                                 modes.unserved_reason(mode, "symmetry sampler", "inference_sampler.kind=symmetry in the engine's composed config"))
                self.assertEqual(census.unserved_by_engine(self._cfg_engine(asked=False, sampler_cls="SampleDiffusionWithSymmetry"), mode),   # ... or found built on the model
                                 modes.unserved_reason(mode, "symmetry sampler", "the built model's sampler is SampleDiffusionWithSymmetry"))
                self.assertIsNone(census.unserved_by_engine(self._cfg_engine(asked=False, sampler_cls="SampleDiffusionWithMotif"), mode))     # the default sampler, wrapped or not: served

    def test_after_initialize_refuses_by_name_and_finish_prints_no_kernels_line(self):
        env = {k: v for k, v in os.environ.items() if k != census.LOWMEM_ENV}
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, _fake_dynamo_modules()):
            for route in ("exact", "fast"):
                census.reset(); lines = []
                obs = census.Observer(route, route, ["diffusion_batch_size=8"], emit=lines.append)
                with self.assertRaises(SystemExit) as cm:
                    obs.after_initialize(self._cfg_engine(asked=True))
                self.assertIsInstance(cm.exception, census.NotServed); self.assertEqual(cm.exception.code, 3)
                why = modes.unserved_reason(route, "classifier-free guidance", "inference_sampler.use_classifier_free_guidance on in the engine's composed config")
                self.assertEqual(lines, [f"[rfdiffusion3-opt] NOT ACTIVE: {why}"])
                self.assertEqual(obs.not_served, {"reason": why, "when": "initialize"})
                self.assertEqual(obs.finish(None), 3); self.assertEqual(obs.finish(0), 3); self.assertEqual(obs.finish(1), 1)   # a failed run keeps its own code
                self.assertEqual(lines, [f"[rfdiffusion3-opt] NOT ACTIVE: {why}"])                          # no KERNELS line: no pass ran
                self.assertIsNone(obs.printed); self.assertIsNone(obs.refused)
                rec = obs.record()
                self.assertEqual((rec["not_served"], rec["line"], rec["refused"]), ({"reason": why, "when": "initialize"}, None, None))
            for route in ("stock", "default"):                                                                # upstream's own routes: CFG is upstream's to run
                census.reset(); lines = []
                obs = census.Observer(route, "off", ["diffusion_batch_size=8"], emit=lines.append)
                obs.after_initialize(self._cfg_engine(asked=True, decided=True))
                self.assertIsNone(obs.not_served); self.assertFalse([l for l in lines if "NOT ACTIVE" in l], lines)
            census.reset(); lines = []
            obs = census.Observer("exact", "exact", ["diffusion_batch_size=8"], emit=lines.append)
            obs.after_initialize(self._cfg_engine(asked=False))                                              # a served config attaches as before (exact: compile off by route): no line yet
            self.assertIsNone(obs.not_served); self.assertEqual(lines, [])


class TestGraphLeverLine(unittest.TestCase):
    """report.graph_lever_line: the graph sampler's state in the core's LEVER grammar — `state=on` when every roll-out of the process ran
    the captured step, `state=skipped reason=<word>` naming why one did not; the counters ride the same line."""

    def test_states(self):
        P = "[rfdiffusion3-opt] LEVER name=RFD3_CUDAGRAPH "
        self.assertEqual(report.graph_lever_line({"installs": 1, "captures": 3, "hits": 396, "fallbacks": 0, "declined": 0, "declined_by": "none"}),
                         P + "state=on impl=cuda_graphs origin=kit installs=1 captures=3 hits=396 fallbacks=0 declined=0")
        self.assertEqual(report.graph_lever_line(None), P + "state=skipped reason=not_imported impl=cuda_graphs origin=kit installs=0 captures=0 hits=0 fallbacks=0 declined=0")
        self.assertEqual(report.graph_lever_line({"installs": 0}), P + "state=skipped reason=not_installed impl=cuda_graphs origin=kit installs=0 captures=0 hits=0 fallbacks=0 declined=0")
        self.assertEqual(report.graph_lever_line({"installs": 1, "captures": 1, "hits": 5, "fallbacks": 2, "declined": 1, "declined_by": "cpu"}),
                         P + "state=skipped reason=capture_failed:2 impl=cuda_graphs origin=kit installs=1 captures=1 hits=5 fallbacks=2 declined=1")
        self.assertEqual(report.graph_lever_line({"installs": 1, "captures": 0, "hits": 0, "fallbacks": 0, "declined": 2, "declined_by": "chunked_pairwise_embedder,use_classifier_free_guidance"}),
                         P + "state=skipped reason=not_driven:chunked_pairwise_embedder,use_classifier_free_guidance impl=cuda_graphs origin=kit installs=1 captures=0 hits=0 fallbacks=0 declined=2")
        self.assertEqual(report.graph_lever_line({"installs": 1, "captures": 0, "hits": 0, "fallbacks": 0, "declined": 0}),          # installed, no roll-out ran (a model build without a design): not `on`
                         P + "state=skipped reason=no_rollout impl=cuda_graphs origin=kit installs=1 captures=0 hits=0 fallbacks=0 declined=0")
        self.assertEqual(report.graph_lever_line({"installs": 1, "captures": 1, "hits": 0, "fallbacks": 0, "declined": 0}).split()[3], "state=on")   # one roll-out captured (a single design batch): on
        self.assertEqual(report.graph_lever_line({"installs": 1, "captures": 0, "hits": 2, "fallbacks": 0, "declined": 0}).split()[3], "state=on")   # roll-outs served by a reused graph: on
        for line in (report.graph_lever_line(None), report.graph_lever_line({"installs": 1, "declined": 1, "declined_by": "cpu"})):
            self.assertRegex(line, r"^\[rfdiffusion3-opt\] LEVER name=RFD3_CUDAGRAPH state=(on|skipped)( reason=\S+)? impl=cuda_graphs origin=kit( \w+=\d+){5}$")
        from opt_core import report as core_report                                          # the grammar is the core's: the kit line IS the core's rendering
        self.assertEqual(report.graph_lever_line({"installs": 1, "captures": 2, "hits": 3}),
                         core_report.lever_line("rfdiffusion3-opt", "RFD3_CUDAGRAPH", "on", impl="cuda_graphs", origin="kit", installs=1, captures=2, hits=3, fallbacks=0, declined=0))


class TestGraphRolloutDrivesTheSamplerOptions(unittest.TestCase):
    """The graph roll-out drives the default sampler's options that live outside the denoiser call, with upstream's own statements at
    upstream's points: partial diffusion (`f["partial_t"]` reaches the stock schedule builder), origin jitter (one (D, 1, 3) draw right after
    the initial noise, before any step noise), realignment (upstream's `centre_random_augment_around_motif` once per step with the motif-fix
    schedule's two arguments, then the closing re-insertion + `weighted_rigid_align` when the design has fixed motif atoms), and the two
    roll-out kinds it does not drive (classifier-free guidance, the chunked pair embedder) go to the stock loop, counted. CPU torch; the
    capture is mocked to fail so the eager denoiser serves the steps (the option wiring is the same on both step routes)."""

    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest("needs a CPU torch: the served roll-out is tensor code (tests/README.md, Package tests)")
        cls.torch = torch
        cls.mod = _load_graph_sampler()

    def _run(self, *, realign=False, jitter=0.0, fraction=0.0, partial_t=None, motif=(), cfg=False, chunked=False, insert_motif_at_end=True, ref=False):
        torch, mod = self.torch, self.mod
        D, L, I = 2, 5, 3
        seen = {"partial_t": "unset", "stock_calls": 0}

        class Stock:
            use_classifier_free_guidance = cfg
            allow_realignment = realign
            s_jitter_origin = jitter
            fraction_of_steps_to_fix_motif = fraction
            center_option, s_trans = "all", 1.0
            gamma_0, gamma_min, noise_scale, step_scale, sigma_data, n_recycle = 0.6, 1.0, 1.003, 1.5, 16.0, 1

            def _construct_inference_noise_schedule(self, device, partial_t=None):
                seen["partial_t"] = partial_t
                return torch.tensor([8.0, 2.0, 0.5], device=device)                      # three levels: two denoiser steps

            def sample_diffusion_like_af3(self, **kw):
                seen["stock_calls"] += 1
                return {"X_L": "stock"}
        Stock.insert_motif_at_end = insert_motif_at_end

        class OnDevice:
            is_cuda = True

            def __init__(self, t):
                self._t, self.device = t, t.device

            def clone(self):
                return self._t.clone()

            def __getattr__(self, name):
                return getattr(self._t, name)

        def diffusion_module(*, X_noisy_L, t, f, n_recycle, **io):
            return {"X_L": X_noisy_L * 0.5, "sequence_logits_I": torch.zeros(D, I, 20), "sequence_indices_I": torch.zeros(D, I, dtype=torch.long)}

        realign_calls, align_calls, draws = [], [], []
        fake_is = type(sys)("rfd3.model.inference_sampler")

        def centre_random_augment_around_motif(X_L, coords, mask, **kw):
            realign_calls.append(dict(kw)); return X_L + 0.0, None

        def weighted_rigid_align(coords, X_L, X_exists_L=None):
            align_calls.append(X_exists_L); return X_L
        fake_is.centre_random_augment_around_motif, fake_is.weighted_rigid_align = centre_random_augment_around_motif, weighted_rigid_align
        real_normal = torch.normal

        def normal_spy(*a, **k):
            draws.append(tuple(k.get("size", ())))
            return real_normal(*a, **k)

        wrapper = mod.SampleDiffusionWithMotifGraph(Stock())
        wrapper._get_entry = mock.Mock(side_effect=RuntimeError("graph capture failed (mock)"))
        mask = torch.zeros(L, dtype=torch.bool)
        for i in motif:
            mask[i] = True
        f = {"is_motif_atom_with_fixed_coord": mask, "ref_element": torch.zeros(L, 3)}
        if partial_t is not None:
            f["partial_t"] = partial_t
        io = {"chunked_pairwise_embedder": object()} if chunked else {}
        declined_before = mod.GRAPH_STATS["declined"]
        env = {k: v for k, v in os.environ.items() if not k.startswith("RFD3_CUDAGRAPH_")}
        rfd3_pkg, rfd3_model = sys.modules.get("rfd3") or type(sys)("rfd3"), sys.modules.get("rfd3.model") or type(sys)("rfd3.model")
        with mock.patch.dict(os.environ, env, clear=True), mock.patch.dict(sys.modules, {"rfd3": rfd3_pkg, "rfd3.model": rfd3_model, "rfd3.model.inference_sampler": fake_is}), \
                mock.patch.object(torch, "normal", normal_spy), torch.no_grad():
            out = wrapper.sample_diffusion_like_af3(f=f, diffusion_module=diffusion_module, diffusion_batch_size=D, coord_atom_lvl_to_be_noised=OnDevice(torch.zeros(D, L, 3)),
                                                    initializer_outputs=io, ref_initializer_outputs=({} if (cfg or ref) else None), f_ref=None)
        return out, seen, realign_calls, align_calls, draws, mod.GRAPH_STATS["declined"] - declined_before, wrapper

    def test_partial_t_reaches_the_stock_schedule_builder(self):
        pt = self.torch.tensor([15.0])
        out, seen, *_ = self._run(partial_t=pt)
        self.assertIs(seen["partial_t"], pt)
        out, seen, *_ = self._run()
        self.assertIsNone(seen["partial_t"])
        self.assertEqual(tuple(out["X_L"].shape), (2, 5, 3))

    def test_draw_order_initial_then_jitter_then_one_per_step(self):
        _, _, _, _, draws, _, _ = self._run()
        self.assertEqual(draws, [(2, 5, 3), (2, 5, 3), (2, 5, 3)])                        # initial noise, then one (D, L, 3) draw per step — the stock order
        _, _, _, _, draws, _, _ = self._run(jitter=0.5, motif=(1,))
        self.assertEqual(draws, [(2, 5, 3), (2, 1, 3), (2, 5, 3), (2, 5, 3)])             # the origin jitter's (D, 1, 3) draw sits where upstream draws it: after the initial noise, before step 0

    def test_realignment_runs_upstreams_statements_with_the_motif_fix_schedule(self):
        out, seen, realign_calls, align_calls, draws, declined, _ = self._run(realign=True, fraction=0.5, motif=(0, 3))
        # two steps, threshold_step = (T - 1) * 0.5 = 1.0: centering_affects_motif = max(step - 1, 0) >= 1.0 (False, False); s_trans = 1.0 iff step >= 1.0 (0.0, 1.0)
        self.assertEqual(realign_calls[:2], [{"center_option": "all", "centering_affects_motif": False, "s_trans": 0.0}, {"center_option": "all", "centering_affects_motif": False, "s_trans": 1.0}])
        self.assertEqual(realign_calls[2:], [{"reinsert_motif": True}])                    # the closing re-insertion (insert_motif_at_end) ...
        self.assertEqual(len(align_calls), 1); self.assertEqual(align_calls[0].tolist(), [True, False, False, True, False])   # ... and the alignment to the input motif
        self.assertEqual((declined, seen["stock_calls"]), (0, 0))                          # driven here, not sent to the stock loop
        self.assertEqual(tuple(out["X_L"].shape), (2, 5, 3))
        out, seen, realign_calls, align_calls, *_ = self._run(realign=True, fraction=0.0, motif=(), insert_motif_at_end=False)
        self.assertEqual([c["s_trans"] for c in realign_calls], [1.0, 1.0]); self.assertEqual(align_calls, [])   # no fixed motif atom: per-step re-centring only, no closing alignment (upstream's `torch.any(mask)` guard)
        out, seen, realign_calls, *_ = self._run(realign=False, fraction=0.5, motif=(0,))
        self.assertEqual(realign_calls, [])                                                 # the motif-fix schedule alone changes nothing (upstream reads it only inside the realignment call)

    def test_cfg_and_the_chunked_embedder_go_to_the_stock_loop_counted(self):
        for kw, name in (({"cfg": True}, "use_classifier_free_guidance"), ({"chunked": True}, "chunked_pairwise_embedder")):
            out, seen, realign_calls, align_calls, draws, declined, wrapper = self._run(**kw)
            self.assertEqual((out, seen["stock_calls"], declined, wrapper.stats["declined"]), ({"X_L": "stock"}, 1, 1, 1), name)
            self.assertIn(name, self.mod.GRAPH_STATS["declined_by"].split(","))
            self.assertEqual(draws, [], name)                                               # nothing drawn here: the stock loop owns that roll-out
        out, seen, *_ , declined, wrapper = self._run(ref=True)                                # upstream's `ref_initializer_outputs` present (CFG decided by the model, the sampler switch off): not driven either
        self.assertEqual((out, seen["stock_calls"], declined), ({"X_L": "stock"}, 1, 1))
        self.assertIn("use_classifier_free_guidance", self.mod.GRAPH_STATS["declined_by"].split(","))


if __name__ == "__main__":
    unittest.main()



class TestStockRoutesLoadNoLeverModule(unittest.TestCase):
    """The stock / default routes' observer and final line run in upstream's own process (the kit-hosted stock child): they load the caller's
    standard-library modules and nothing else — never a lever module (gather), whatever the line says. An outside runner's exit census of that
    process admits exactly these names."""

    def test_observe_and_finish_load_only_the_declared_modules(self):
        code = textwrap.dedent("""
            import json, sys
            import rfdiffusion3_opt.census as census
            out = {}
            for route, extra in (("stock", ["compile_model=true"]), ("default", [])):
                census.observe(route, "off", ["diffusion_batch_size=8", "seed=1"] + extra, emit=lambda line: None)
                census.finish(0)
                out[route] = sorted(m.split(".", 1)[1] for m in sys.modules if m.startswith("rfdiffusion3_opt."))
            print(json.dumps(out))
        """)
        env = {k: v for k, v in os.environ.items() if k != "RFDIFFUSION3_OPT"}
        r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=env)
        self.assertEqual(r.returncode, 0, r.stderr)
        got = json.loads(r.stdout.strip().splitlines()[-1])
        for route in ("stock", "default"):
            self.assertEqual(got[route], ["_autoload", "_emit", "census", "routes", "settings"], route)   # no gather, no stack, no modes: the lever module loads on kit routes only
