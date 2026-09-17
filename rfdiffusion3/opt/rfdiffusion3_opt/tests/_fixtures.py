"""Fixtures for the CPU tests: fake interpreters (site-packages trees) built from the bytes the tree carries, and a fake GPU.

A fake site holds an `rfd3` package whose ten target files are real — the kit's `patched/` files (`patched` state) or the eight
upstream files extracted from `stock/foundry-4010e3e.tar.gz` (`pristine` state) — under stub `__init__.py` files that import nothing
(the real ones import pydantic / torch, which these tests do not have), an upstream `foundry/utils/torch.py` from the same archive,
and the distribution metadata `stock/check_pins.py` reads (a `rc_foundry-<version>.dist-info` with a `direct_url.json` at the pin).
`stub_cli(site, ...)` adds an `rfd3/cli.py` whose `app()` records its argv and the environment and writes one fake design, so the
`design` command's two routes run end to end without torch. `fake_gpu(bin_dir)` writes an `nvidia-smi` that answers like an H100.
`fake_torch(site)` adds a `torch` package with the names the kit's `hoist.py` and `layers/layer_utils.py` touch at import (a stub
`torch.nn` with the module classes `layer_utils` subclasses, `torch.nn.functional.silu`), and `make_site` adds the two `foundry` names
and the `numpy` name `layer_utils` imports, so the REAL kit modules can be imported in these tests (they read `RFD3_HOIST` /
`RFD3_FZT` at their import): the stubs then import both on request — `RFD3_FAKE_IMPORT_HOIST=1` for the CLI stub (the engine stub
always does, as the real model build does) — and `RFD3_FAKE_DROP_SWITCH=1` removes `RFD3_HOIST` from the
process environment first (what an upstream `.env` load or a caller's reset after the gate would do), the partial-activation case the
exit rule refuses. `RFD3_FAKE_EXIT=<n>` makes the CLI stub end with that code after writing its outputs (a failed run).
`RFD3_FAKE_BLANKET_IGNORE=1` makes the CLI stub install upstream's construction-time `warnings.filterwarnings("ignore")` before the
import and raise a det-recipe-style `UserWarning` after it (the channel the package re-opens at the kit module's import).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile

from .. import registry, stack

TREE = registry.tree_home()
KIT = registry.kit_home()
PKG_PARENT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))   # opt/: the package importable from source


def core_dir() -> str:
    """The installed core's directory (the core pin gate's `installed.root`: the tree that makes the pinned `opt_core` importable): on the path
    of every test subprocess beside the package."""
    from .._autoload import core_gate
    return core_gate()["installed"]["root"]
CACHE_DIR = tempfile.mkdtemp(prefix="rfd3port_cache_")           # RFDIFFUSION3_OPT_CACHE of every test subprocess
ARCHIVE = os.path.join(TREE, "stock", "foundry-4010e3e.tar.gz")
ARCHIVE_PREFIX = "foundry-4010e3e2/"
STOCK_RFD3 = ARCHIVE_PREFIX + "models/rfd3/src/"
STOCK_TORCH_PY = ARCHIVE_PREFIX + "src/foundry/utils/torch.py"
PINS = stack.pins()
VERSION = PINS["upstream"]["foundry"]["version"]
COMMIT = PINS["upstream"]["foundry"]["commit"]
GPU_LINE = "NVIDIA H100 80GB HBM3, 9.0, 81559 MiB, 580.95.05"


def _write(path: str, text: str = "") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def dist_info(site: str, version: str = VERSION, commit: str = COMMIT, url: str = "https://github.com/RosettaCommons/foundry.git", kind: str = "vcs") -> None:
    """The rc-foundry distribution metadata: `kind` = vcs (a git install at `commit`) | dir (a local checkout / extracted archive,
    `file:` url, no commit) | none (no direct_url.json: an index install)."""
    d = os.path.join(site, f"rc_foundry-{version}.dist-info")
    _write(os.path.join(d, "METADATA"), f"Metadata-Version: 2.1\nName: rc-foundry\nVersion: {version}\n")
    if kind == "vcs":
        _write(os.path.join(d, "direct_url.json"), json.dumps({"url": url, "vcs_info": {"vcs": "git", "commit_id": commit, "requested_revision": commit}}))
    elif kind == "dir":
        _write(os.path.join(d, "direct_url.json"), json.dumps({"url": "file:///opt/foundry", "dir_info": {"editable": True}}))
    _write(os.path.join(d, "RECORD"), "")
    _write(os.path.join(d, "INSTALLER"), "pip\n")


def stock_bytes() -> dict:
    """{rfd3/... or foundry/utils/torch.py: bytes} for the nine upstream files, from the stock archive."""
    out = {}
    with tarfile.open(ARCHIVE, "r:gz") as tf:
        for m in tf.getmembers():
            if m.name.startswith(STOCK_RFD3 + "rfd3/model/") and m.isfile():
                out[m.name[len(STOCK_RFD3):]] = tf.extractfile(m).read()
            elif m.name == STOCK_TORCH_PY:
                out["foundry/utils/torch.py"] = tf.extractfile(m).read()
    return out


def make_site(root: str, state: str = "patched", version: str = VERSION, commit: str = COMMIT) -> str:
    """A fake site-packages under `root` in the given state: patched | pristine | `mixed` (one file stock) | unknown."""
    site = os.path.join(root, "site")
    os.makedirs(site, exist_ok=True)
    _write(os.path.join(site, "rfd3", "__init__.py"),
           "import os, json\nSEEN = {'RFD3_HOIST': os.environ.get('RFD3_HOIST'), 'RFDIFFUSION3_OPT': os.environ.get('RFDIFFUSION3_OPT')}\n"
           "_p = os.environ.get('RFD3_FAKE_SEEN')\n_p and open(_p, 'w').write(json.dumps(SEEN))\n")
    _write(os.path.join(site, "rfd3", "model", "__init__.py"), "")
    _write(os.path.join(site, "rfd3", "model", "layers", "__init__.py"), "")
    _write(os.path.join(site, "foundry", "__init__.py"), "")
    _write(os.path.join(site, "foundry", "utils", "__init__.py"), "")
    _write(os.path.join(site, "foundry", "utils", "ddp.py"),                                     # the logger layer_utils.py constructs at import
           "import logging\ndef RankedLogger(name, rank_zero_only=True):\n    return logging.getLogger(name)\n")
    _write(os.path.join(site, "foundry", "training", "__init__.py"), "")
    _write(os.path.join(site, "foundry", "training", "checkpoint.py"), "def activation_checkpointing(fn):\n    return fn\n")
    _write(os.path.join(site, "numpy", "__init__.py"), "def prod(xs):\n    out = 1\n    for x in xs:\n        out *= x\n    return out\n")   # the one numpy name layer_utils.py uses (inside a method)
    fake_apex(site)                                                                              # the pinned stack carries apex: upstream's import guard (layer_utils.py:13-23) binds its FusedRMSNorm
    sb = stock_bytes()
    with open(os.path.join(site, "foundry", "utils", "torch.py"), "wb") as fh:
        fh.write(sb["foundry/utils/torch.py"])
    targets = registry.target_files(KIT)
    for rel in targets:
        dst = os.path.join(site, rel)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if state == "patched" or (state == "mixed" and rel != "rfd3/model/layers/blocks.py"):
            shutil.copyfile(os.path.join(KIT, "patched", rel), dst)
        elif state in ("pristine", "mixed"):
            if rel in registry.NEW_FILES:
                continue
            with open(dst, "wb") as fh:
                fh.write(sb[rel])
        elif state == "unknown":
            if rel in registry.NEW_FILES:
                continue
            with open(dst, "wb") as fh:
                fh.write(sb[rel] + b"\n# edited\n")
        else:
            raise ValueError(state)
    dist_info(site, version, commit)
    return site


def fake_apex(site: str, version: str = "0.1") -> None:
    """A fake `apex` distribution: `apex.normalization.fused_layer_norm.FusedRMSNorm` (the one name upstream's import guard binds,
    layer_utils.py:14) and the dist-info `importlib.metadata.version("apex")` reads. RFD3_FAKE_NO_APEX=1 in a stub process hides it (the
    engine stub refuses the import before the layers load): upstream's guard then binds torch.nn.RMSNorm — the `fallback:` word."""
    _write(os.path.join(site, "apex", "__init__.py"), f"__version__ = {version!r}\n")
    _write(os.path.join(site, "apex", "normalization", "__init__.py"), "from .fused_layer_norm import FusedRMSNorm\n")
    _write(os.path.join(site, "apex", "normalization", "fused_layer_norm.py"),
           "class FusedRMSNorm:\n    def __init__(self, *a, **k):\n        pass\n    def __call__(self, x, *a, **k):\n        return x\n")
    _write(os.path.join(site, f"apex-{version}.dist-info", "METADATA"), f"Metadata-Version: 2.1\nName: apex\nVersion: {version}\n")


def stub_cli(site: str, out_marker: str = "fake_design") -> None:
    """`rfd3/cli.py` with an `app()` that parses `design key=value` argv, builds the engine stub (`stub_engine`, written with it: upstream's
    CLI builds `RFD3InferenceEngine(**cfg)` and calls `run()`, rfd3/run_inference.py:38-40) with `compile_model` / `low_memory_mode` from the
    line, runs it — `initialize()` (the kit-module imports under RFD3_FAKE_IMPORT_HOIST=1, upstream's compile step under compile_model=true),
    upstream's one atom-attention path record and its per-batch clock line —, writes `<out_dir>/<spec basename>_<key>_0_model_<k>.cif.gz` (+ .json)
    and a record of what it saw (`<out_dir>/cli_seen.json`), then exits 0. A fake `torch` (fake_torch) is written when the site has none: the
    engine stub's compile step and dynamo counters import it lazily, as upstream's process has torch loaded by then."""
    stub_engine(site, from_cli=True)
    if not os.path.isdir(os.path.join(site, "torch")):
        fake_torch(site)
    _write(os.path.join(site, "rfd3", "cli.py"), '''
import gzip, json, os, sys
def app():
    args = sys.argv[1:]
    kv = dict(a.split("=", 1) for a in args[1:] if "=" in a)
    if os.environ.get("RFD3_FAKE_DROP_SWITCH") == "1":
        os.environ.pop("RFD3_HOIST", None)                # the switch reset after the gate: the kit module reads it off
    if os.environ.get("RFD3_FAKE_BLANKET_IGNORE") == "1":
        import warnings
        warnings.filterwarnings("ignore")                 # upstream's engine construction (foundry/utils/logging.py:123)
    from rfd3.engine import RFD3InferenceEngine           # upstream's CLI: engine = RFD3InferenceEngine(**cfg); engine.run(...) (run_inference.py:38-40)
    truthy = lambda v: str(v).strip().lower() in ("true", "1", "yes", "on", "y", "t")
    ckpt = kv.get("ckpt_path") or os.path.join((os.environ.get("FOUNDRY_CHECKPOINT_DIRS") or os.path.expanduser("~/.foundry/checkpoints")).split(":")[0], "rfd3_latest.ckpt")   # base.py:76-95: the token, else the registry's file (checkpoint_registry.py:71-77,86-89)
    engine = RFD3InferenceEngine(compile_model=truthy(kv.get("compile_model", "False")), low_memory_mode=truthy(kv.get("low_memory_mode", "False")),
                                 diffusion_batch_size=int(kv.get("diffusion_batch_size", "8")), from_cli=True, ckpt_path=ckpt)
    spec = json.load(open(kv["inputs"])); base = os.path.splitext(os.path.basename(kv["inputs"]))[0]
    engine.run(n_batches=len(spec))
    if os.environ.get("RFD3_FAKE_BLANKET_IGNORE") == "1":
        import warnings
        warnings.warn("fake_op does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'", UserWarning)
    out = kv["out_dir"]; os.makedirs(out, exist_ok=True)
    for key in spec:
        for k in range(int(kv.get("diffusion_batch_size", "8"))):
            with gzip.open(os.path.join(out, f"{base}_{key}_0_model_{k}.cif.gz"), "wb") as fh:
                fh.write(f"data_{key}\\n_fake.design {k} seed {kv.get('seed')}\\n".encode())
            json.dump({"ckpt_path": kv.get("ckpt_path"), "seed": kv.get("seed"), "k": k}, open(os.path.join(out, f"{base}_{key}_0_model_{k}.json"), "w"))
    seen = {"argv": args, "RFD3_HOIST": os.environ.get("RFD3_HOIST"), "RFDIFFUSION3_OPT": os.environ.get("RFDIFFUSION3_OPT"),
            "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),   # torch's own variable: reaches the stock child unchanged (the strip rule leaves it)
            "RFD3_DENSE_SDPA_ATTENTION": os.environ.get("RFD3_DENSE_SDPA_ATTENTION"),               # upstream's own switch: passes through the stock arm as the caller set it; refused under a kit mode at activation
            "modules": sorted(m for m in sys.modules if m.startswith("rfd3") or m.startswith("rfdiffusion3_opt")), "python": sys.executable,
            "compile_model": engine.compile_model, "wrapped": engine.fake_wrapped()}
    json.dump(seen, open(os.path.join(out, "cli_seen.json"), "w"))
    raise SystemExit(int(os.environ.get("RFD3_FAKE_EXIT") or 0))
''')


def stub_engine(site: str, from_cli: bool = False) -> None:
    """`rfd3/engine.py` with the two names `warm` imports: a config dict and an engine whose `initialize()` imports the kit's module
    when the site has one (as the real model build does; built `from_cli=True` — the CLI stub — only under RFD3_FAKE_IMPORT_HOIST=1, the
    partial-activation cases keeping their meaning) and records what the process saw (`RFD3_FAKE_SEEN`, like the stub package); then, under
    `compile_model=True`, upstream's compile step verbatim in shape (engine.py:236-269: the walk to the net with `diffusion_module`, the four
    `_COMPILE_TARGETS` wrapped by `torch.compile` — the fake torch's OptimizedModule —, the `torch.compile enabled …` record;
    RFD3_FAKE_COMPILE_SKIP=1 takes upstream's 'Could not locate the diffusion module; skipping torch.compile.' branch). The engine carries
    `ckpt_path` and `transform_overrides` as foundry's base engine does (inference_engines/base.py:95,106) and the composed sampler group `inference_sampler_overrides` (engine.py:180; RFD3_FAKE_SAMPLER=<json dict>, default {}). `initialize()` also stands up
    the decision surface of `rfd3.model.layers.attention` as a module object (the real file on the site imports the full torch stack):
    `use_dense_sdpa_pairbias(Q, indices, full, H)` in upstream's shape (attention.py:409-454: `full` -> the 'SPARSE (full=True)' record and
    False; SPARSE 'disabled via env var' under RFD3_DENSE_SDPA_ATTENTION!=1; SPARSE by RFD3_FAKE_ATTN=memory|small: the free-memory / L<=k
    reasons; DENSE-SDPA otherwise; RFD3_FAKE_ATTN=flip: the first three atom-level calls dense (the census's pre-flight probes — one per distinct atom head count, 4 and 2 here — and the pass's first), every later one sparse with no further record — upstream
    logs the first decision of the process only) with the once-per-process path record (attention.py:397-406's f-string). `run(n_batches)` =
    initialize + per batch two denoising-step calls of the diffusion module (`dm(X_noisy_L=(D, L, 3) on 'cuda:0', f={'atom_to_token_map': RFD3_FAKE_TOKENS tokens, default 120}, …)`, L = RFD3_FAKE_ATOMS,
    default 999: the pre-flight decision's hook) + two atom-level calls (`full=False`: encoder, decoder) and one token-level call (`full=True`) of that function
    through the module attribute (upstream's call site, attention.py:340-342; none at all under RFD3_LOW_MEMORY_MODE=1 or RFD3_FAKE_ATTN=silent —
    upstream's low-memory path never reaches the decision) + dynamo's counters when compiled (RFD3_FAKE_DYNAMO=nograph leaves them at zero) +
    one `Finished inference batch in <s> seconds.` record per batch (engine.py:402; RFD3_FAKE_CLOCK=<text> sets the digits, default 0.01)."""
    _write(os.path.join(site, "rfd3", "engine.py"), '''
import json, logging, os, sys
_alog = logging.getLogger("rfd3.model.layers.attention")
_elog = logging.getLogger("rfd3.engine")
class RFD3InferenceConfig(dict):
    def __init__(self, **kw):
        super().__init__(**kw)
class _Sub:                                            # a diffusion submodule (an nn.Module in upstream)
    def parameters(self):
        return iter(())
class LocalAttentionPairBias:                          # attention.py:189 — the pair-bias attention; the census reads the atom-level modules' n_head
    def __init__(self, n_head):
        self.n_head = n_head
class _AtomSub(_Sub):                                  # encoder / decoder: atom transformers holding atom-level attention (full=False)
    def __init__(self, n_head=4):
        self.attn = LocalAttentionPairBias(n_head)
class _DiffusionModule:                                # RFD3_diffusion_module.py: forward(self, X_noisy_L, t, f, …) once per denoising step; n_attn_keys the indices' k
    n_attn_keys = 128
    def __init__(self):
        self.encoder, self.diffusion_token_encoder, self.diffusion_transformer, self.decoder = _AtomSub(4), _Sub(), _Sub(), _AtomSub(2)
        self.steps = []
    def forward(self, X_noisy_L, t=None, f=None, n_recycle=None, **kw):
        self.steps.append(tuple(X_noisy_L.shape))
        return {}
    def __call__(self, *a, **k):                        # nn.Module.__call__: self.forward, the instance attribute first
        return self.forward(*a, **k)
class _Net:                                            # the RFD3 net: the first object with `diffusion_module` on upstream's walk; modules() = nn.Module's recursive walk (here: itself, the norms)
    def __init__(self):
        self.diffusion_module = _DiffusionModule()
        self.norms = []
    def modules(self):
        yield self
        yield from self.norms
class _Fabric:                                         # lightning's _FabricModule: `_forward_module` -> an EMA wrapper with `shadow` -> the net
    def __init__(self):
        self._forward_module = type("EMA", (), {"shadow": _Net()})()
class _Trainer:
    def __init__(self):
        self.state = {"model": _Fabric()}
class RFD3InferenceEngine:
    _COMPILE_TARGETS = ("encoder", "diffusion_token_encoder", "diffusion_transformer", "decoder")   # engine.py:212-217
    def __init__(self, compile_model=False, low_memory_mode=False, diffusion_batch_size=8, from_cli=False, **kw):
        self.ckpt_path = os.path.abspath(kw["ckpt_path"]) if kw.get("ckpt_path") else None   # base.py:95: Path(ckpt_path).resolve()
        self.transform_overrides = {"diffusion_batch_size": diffusion_batch_size}             # base.py:106, as rfd3/engine.py:168 passes it
        self.compile_model, self.compiled_, self.diffusion_batch_size, self.from_cli = compile_model, False, diffusion_batch_size, from_cli
        if low_memory_mode:
            os.environ["RFD3_LOW_MEMORY_MODE"] = "1"      # engine.py:202-205: the engine sets the switch itself
        self.inference_sampler_overrides = json.loads(os.environ.get("RFD3_FAKE_SAMPLER") or "{}")   # engine.py:180: the composed `inference_sampler` group as the engine holds it (RFD3_FAKE_SAMPLER: a JSON dict)
        self.trainer = _Trainer()
    def initialize(self):
        if os.environ.get("RFD3_FAKE_DROP_SWITCH") == "1":
            os.environ.pop("RFD3_HOIST", None)         # the switch reset after the gate: the kit module reads it off
        if os.environ.get("RFD3_FAKE_NO_APEX") == "1":       # an interpreter without apex: upstream's import guard falls to torch.nn.RMSNorm
            class _NoApex:
                def find_spec(self, name, path=None, target=None):
                    if name == "apex" or name.startswith("apex."):
                        raise ImportError("apex hidden by RFD3_FAKE_NO_APEX")
            sys.meta_path.insert(0, _NoApex())
            for m in [m for m in sys.modules if m == "apex" or m.startswith("apex.")]:
                del sys.modules[m]
        if not self.from_cli or os.environ.get("RFD3_FAKE_IMPORT_HOIST") == "1":
            try:
                import rfd3.model.layers.layer_utils       # the model build imports the layers (layer_utils reads RFD3_FZT) ...
            except ImportError:
                pass
            try:
                import rfd3.model.hoist                    # ... and the kit's hoist module (it reads RFD3_HOIST)
            except ImportError:
                pass
            if os.environ.get("RFD3_CUDAGRAPH", "0") == "1":     # inference_sampler.py:610-612: the default sampler's holder installs the graph roll-out at its construction
                try:
                    from rfd3.model.cudagraph_sampler import install as _cg_install
                    holder = type("ConditionalDiffusionSampler", (), {})()
                    holder.sampler = type("SampleDiffusionWithMotif", (), {})()   # the kit wraps upstream's default sampler by its class name (cudagraph_sampler.install)
                    _cg_install(holder)
                except ImportError:
                    pass
        _lulog = logging.getLogger("rfd3.model.layers.layer_utils")   # upstream's model build imports layers/layer_utils.py, whose import guard LOGS the binding (layer_utils.py:16 / :19-21) — the record the stock routes read
        try:                                             # the network's norms, built through upstream's import guard (layer_utils.py:13-30): apex's FusedRMSNorm when importable, else torch.nn.RMSNorm
            from apex.normalization.fused_layer_norm import FusedRMSNorm as _Norm
            _lulog.info("Fused RMSNorm enabled!")
        except ImportError:
            from torch.nn import RMSNorm as _Norm
            _lulog.warning("Using nn.RMSNorm instead of apex.normalization.fused_layer_norm.FusedRMSNorm." "Ensure you're using the correct apptainer")
        self.trainer.state["model"]._forward_module.shadow.norms = [_Norm(8) for _ in range(3)]
        self._attention_module()
        seen = {"RFD3_HOIST": os.environ.get("RFD3_HOIST"), "RFDIFFUSION3_OPT": os.environ.get("RFDIFFUSION3_OPT"), "python": sys.executable,
                "hoist_imported": "rfd3.model.hoist" in sys.modules}
        p = os.environ.get("RFD3_FAKE_SEEN")
        p and open(p, "w").write(json.dumps(seen))
        if self.compile_model and not self.compiled_:      # engine.py:219-224
            self._compile_diffusion_submodules()
            self.compiled_ = True
        return {}
    def _attention_module(self):                       # rfd3/model/layers/attention.py:394-454, in shape: the once-per-process record and the per-call decision
        name = "rfd3.model.layers.attention"
        if name in sys.modules:
            return sys.modules[name]
        import types
        att = types.ModuleType(name)
        att._DENSE_PATH_REPORTED = False
        att._atom_calls = 0
        B = self.diffusion_batch_size
        def _report_attention_path(chosen, reason, shape):          # :397-406
            if att._DENSE_PATH_REPORTED:
                return
            att._DENSE_PATH_REPORTED = True
            _alog.info(f"Atom attention path: {chosen} ({reason}) [{shape}]. "      # attention.py:403-404, verbatim in form
                       "Set RFD3_DENSE_SDPA_ATTENTION=0 to force the original sparse path.")
        def use_dense_sdpa_pairbias(Q, indices, full, H):           # :409-454
            shape = f"D={B} L=999 k=128 H={H}"
            if full:
                _report_attention_path("SPARSE", "full=True", shape)
                return False
            att._atom_calls += 1
            if os.environ.get("RFD3_DENSE_SDPA_ATTENTION", "1") != "1":
                _report_attention_path("SPARSE", "disabled via env var", shape)
                return False
            if os.environ.get("RFD3_FAKE_ATTN") == "small":
                _report_attention_path("SPARSE", "L=9 <= k=128", shape)             # attention.py:440: f"L={L} <= k={k}"
                return False
            if os.environ.get("RFD3_FAKE_ATTN") == "memory" or (os.environ.get("RFD3_FAKE_ATTN") == "flip" and att._atom_calls > 3):
                _report_attention_path("SPARSE", "needs 10.0 GiB of 1.0 GiB free", shape)   # attention.py:449's f-string form
                return False
            _report_attention_path("DENSE-SDPA", "0.6 GiB bias", shape)          # attention.py:453's reason: f"{dense_bytes / 2**30:.1f} GiB bias"
            return True
        def dense_sdpa_pairbias_attention(Q, K, V, B_, indices, H, G=None):   # :457 (named by the census, never called here)
            raise NotImplementedError
        att._report_attention_path, att.use_dense_sdpa_pairbias, att.dense_sdpa_pairbias_attention = _report_attention_path, use_dense_sdpa_pairbias, dense_sdpa_pairbias_attention
        sys.modules[name] = att
        return att
    def _compile_diffusion_submodules(self):           # engine.py:226-269, in shape
        import torch
        model = self.trainer.state["model"]
        net = getattr(model, "_forward_module", model)
        for _ in range(5):
            if hasattr(net, "diffusion_module") and os.environ.get("RFD3_FAKE_COMPILE_SKIP") != "1":
                break
            for attr in ("module", "shadow", "model"):
                if hasattr(net, attr):
                    net = getattr(net, attr)
                    break
        else:
            _elog.warning("Could not locate the diffusion module; skipping torch.compile.")
            return
        dm = net.diffusion_module
        for name in self._COMPILE_TARGETS:
            sub = getattr(dm, name, None)
            if sub is None:
                continue
            setattr(dm, name, torch.compile(sub, dynamic=False))
        _elog.info("torch.compile enabled for diffusion submodules (%s). Expect a one-off warmup on the first diffusion step." % ", ".join(self._COMPILE_TARGETS))
    def fake_wrapped(self):
        om = getattr(getattr(getattr(sys.modules.get("torch"), "_dynamo", None), "eval_frame", None), "OptimizedModule", None)
        dm = self.trainer.state["model"]._forward_module.shadow.diffusion_module
        return [n for n in self._COMPILE_TARGETS if om is not None and isinstance(getattr(dm, n), om)]
    def run(self, n_batches=1, **kw):                   # initialize, then per batch the model forward: the attention decisions (upstream's path record at the first), its clock line
        self.initialize()
        decides = os.environ.get("RFD3_LOW_MEMORY_MODE", "0") != "1" and os.environ.get("RFD3_FAKE_ATTN") != "silent"   # the low-memory path never reaches the decision
        import torch
        dm = self.trainer.state["model"]._forward_module.shadow.diffusion_module
        atoms = int(os.environ.get("RFD3_FAKE_ATOMS") or 999)
        tokens = int(os.environ.get("RFD3_FAKE_TOKENS") or 120)   # the design's token count (residues): f["atom_to_token_map"].max() + 1, RFD3_diffusion_module.py:193-195
        class _TokenMap:                                       # the per-atom token index tensor's one surface the census reads (.max())
            def __init__(self, n): self.n = n
            def max(self): return self.n - 1
            def __len__(self): return atoms
        for b in range(int(n_batches)):
            for _step in range(2):                                # two denoising steps per batch through the diffusion module's __call__ (the samplers' per-step call, X_noisy_L (D, L, 3) by keyword)
                dm(X_noisy_L=torch.empty((self.diffusion_batch_size, atoms, 3), device="cuda:0"), t=None, f={"atom_to_token_map": _TokenMap(tokens)}, n_recycle=2)
            if decides:
                att = sys.modules["rfd3.model.layers.attention"]
                for full in (False, True, False):             # encoder (atom), token transformer (full=True, RFD3_diffusion_module.py:339), decoder (atom) — by name through the module, keyword arguments shaped as attention.py:340-342 passes them (Q (D, L, c), indices (D, L, k), H)
                    L_ = tokens if full else atoms
                    att.use_dense_sdpa_pairbias(Q=torch.empty((self.diffusion_batch_size, L_, 128), device="cuda:0"), indices=torch.empty((self.diffusion_batch_size, L_, L_ if full else dm.n_attn_keys), device="cuda:0"), full=full, H=4)
            if b == 0 and self.fake_wrapped() and os.environ.get("RFD3_FAKE_DYNAMO") != "nograph":
                from torch._dynamo.utils import counters, compilation_time_metrics
                counters["stats"]["unique_graphs"] += 4; counters["frames"]["total"] += 4; counters["frames"]["ok"] += 4
                compilation_time_metrics.setdefault("_compile.compile_inner", []).extend([1.5, 1.5])
            _elog.info("Finished inference batch in %s %s." % (os.environ.get("RFD3_FAKE_CLOCK") or "%.2f" % 0.01, "seconds"))   # engine.py:402's f-string form
''')


def fake_torch(site: str, cuda: str = None, sm: tuple = None) -> str:
    """A `torch` package on the fake site with the names the kit's hoist.py touches at import and in describe(). When `cuda` is
    given, also adds the `version.cuda` / `cuda.get_device_capability` surface `modes.jit_cache_key()` needs, so a CPU-only test
    box can exercise its success path (a fake, deterministic stack, not a real GPU query)."""
    extra = ""
    if cuda is not None:
        cc = sm or (9, 0)
        extra = (f"\n\nclass _FakeVersion:\n    cuda = {cuda!r}\n\n\nversion = _FakeVersion()\n\n\n"
                 f"class _FakeCuda(_FakeCudaBase):\n    @staticmethod\n    def get_device_capability(_index=0):\n        return {tuple(cc)!r}\n\n\n"
                 f"cuda = _FakeCuda()\n")
    _write(os.path.join(site, "torch", "__init__.py"),
           "__version__ = '0.0.0+fake'\nbfloat16 = 'bfloat16'\nfloat32 = 'float32'\nfrom . import nn\ndef is_tensor(x):\n    return False\n"
           "class _FakeTensor:\n"                                                              # torch.empty's value: the shape / device surface the pre-flight probe and the stub's step call use
           "    def __init__(self, shape, device=None):\n        self.shape = tuple(shape); self.device = device; self.is_cuda = str(device).startswith('cuda')\n"
           "def empty(shape, *more, device=None, dtype=None, **kw):\n    return _FakeTensor(tuple(shape) if isinstance(shape, (tuple, list)) else (shape,) + more, device)\n"
           "class _FakeCudaBase:\n"                                                            # torch.cuda without a device: no peak statistics (is_available False), the free-memory query the decision reads
           "    @staticmethod\n    def is_available():\n        return False\n"
           "    @staticmethod\n    def is_current_stream_capturing():\n        return False\n"
           "    @staticmethod\n    def mem_get_info(device=None):\n        import os\n        free = float(os.environ.get('RFD3_FAKE_FREE_GIB') or 70.0) * 2 ** 30\n        return int(free), int(80 * 2 ** 30)\n"
           "cuda = _FakeCudaBase()\n" + extra)
    _write(os.path.join(site, "torch", "nn", "__init__.py"),                                    # the module classes layers/layer_utils.py subclasses / instantiates at import
           "from . import functional\n"
           "class Module:\n    def __init__(self, *a, **k):\n        pass\n"
           "class Linear(Module):\n    pass\n"
           "class RMSNorm(Module):\n    pass\n"
           "class Sequential(Module):\n    pass\n"
           "class Sigmoid(Module):\n    pass\n")
    _write(os.path.join(site, "torch", "nn", "functional.py"), "def silu(x):\n    return x\n")
    _write(os.path.join(site, "torch", "_dynamo", "__init__.py"), "from . import config, eval_frame, utils\n\n\ndef disable(fn=None, recursive=True):\n    return fn if fn is not None else (lambda f: f)\n")          # the names the KERNELS census reads (census.py): the wrapper class, the disable switch, the counters
    _write(os.path.join(site, "torch", "_dynamo", "eval_frame.py"),
           "class OptimizedModule:\n    def __init__(self, mod, **kw):\n        self._orig_mod = mod\n        self.compile_kwargs = kw\n    def parameters(self):\n        return self._orig_mod.parameters()\n")
    _write(os.path.join(site, "torch", "_dynamo", "config.py"), "import os\ndisable = os.environ.get('TORCHDYNAMO_DISABLE') == '1'\nsuppress_errors = False\ncache_size_limit = 8\n")   # cache_size_limit: dynamo's per-code recompile budget, which the kit's compile record raises to its floor (opt_core.capture.compile)
    _write(os.path.join(site, "torch", "compiler", "__init__.py"),
           "def disable(fn=None, recursive=True):\n    if fn is None:\n        return lambda f: f\n    return fn\n\n\ndef is_compiling():\n    return False\n")      # the boundary marker the kit's hoist.py applies under RFD3_COMPILE (torch.compiler.disable) and the tracing predicate the ATTN_PATH shim reads
    _write(os.path.join(site, "torch", "_dynamo", "utils.py"),
           "import collections\ncounters = collections.defaultdict(collections.Counter)\ncompilation_time_metrics = {}\n")
    with open(os.path.join(site, "torch", "__init__.py"), "a", encoding="utf-8") as fh:
        fh.write("\n\ndef compile(mod, **kw):\n    from ._dynamo.eval_frame import OptimizedModule\n    return OptimizedModule(mod, **kw)\n")
    return site


def fake_gpu(bin_dir: str, line: str = GPU_LINE) -> str:
    os.makedirs(bin_dir, exist_ok=True)
    p = os.path.join(bin_dir, "nvidia-smi")
    _write(p, f"#!/bin/sh\necho '{line}'\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def unpin_distribution(site: str, **kw) -> str:
    """Replace the fake site's rc-foundry metadata (make_site: a git install at the pin) with one OFF the pin — by default an index install
    (`kind="none"`: no direct_url.json); `kw` as dist_info's. The rfd3 tree is untouched. Returns `site`."""
    for d in glob.glob(os.path.join(site, "rc_foundry-*.dist-info")):
        shutil.rmtree(d)
    dist_info(site, **dict({"kind": "none"}, **kw))
    return site


def fake_spec(path: str, key: str = "public") -> str:
    """A one-example design spec JSON at `path` (the line's `inputs=` file): the stub CLI reads its keys and writes one design per key
    and batch index, `<basename>_<key>_0_model_<k>`. Returns the path."""
    _write(path, json.dumps({key: {"dialect": 2, "contig": "55-80,/0,A1-193", "length": "55-80"}}))
    return path


def fake_python_on_path(bin_dir: str, site: str) -> str:
    """A `python` wrapper on PATH that runs this interpreter with `site` first on sys.path (what install.sh's `python -c` sees)."""
    os.makedirs(bin_dir, exist_ok=True)
    p = os.path.join(bin_dir, "python")
    _write(p, f"#!/bin/sh\nPYTHONPATH='{site}'${{PYTHONPATH:+:$PYTHONPATH}} exec '{sys.executable}' \"$@\"\n")
    os.chmod(p, os.stat(p).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return bin_dir


def make_venv(root: str, site: str) -> str:
    """A real interpreter for the stock arm: a venv (no pip) whose site-packages carries one .pth pointing at the fake site and at
    this package's parent directory, so `<venv>/bin/python -I stock/check_pins.py` and `-s -m rfdiffusion3_opt.stock_design` work as they
    do in a deployment (isolated mode ignores PYTHONPATH; .pth files are still honoured). Returns the interpreter path."""
    import glob
    import venv
    d = os.path.join(root, "venv")
    venv.EnvBuilder(with_pip=False, symlinks=True).create(d)
    sp = glob.glob(os.path.join(d, "lib", "python*", "site-packages"))[0]
    _write(os.path.join(sp, "fake_stock.pth"), f"{site}\n{PKG_PARENT}\n{core_dir()}\n")
    return os.path.join(d, "bin", "python")


def run_py(code: str, site: str, env: dict = None, bin_dir: str = None, argv: list = None, cwd: str = None, python: str = None) -> subprocess.CompletedProcess:
    """This interpreter, with the fake site first on sys.path (PYTHONPATH), the package importable, and `env` overlaid on a clean
    environment (no lever names, no package switch)."""
    base = {k: v for k, v in os.environ.items() if k not in stack.lever_names() and k not in stack.PACKAGE_ENV and k not in stack.upstream_switches()}
    base["PYTHONPATH"] = os.pathsep.join([site, PKG_PARENT, core_dir()] + ([base["PYTHONPATH"]] if base.get("PYTHONPATH") else []))
    base["PYTHONDONTWRITEBYTECODE"] = "1"
    base["MODEL_OPT"] = TREE
    base["RFDIFFUSION3_OPT_CACHE"] = CACHE_DIR                                   # the digest cache of a test run never touches the user's cache
    if bin_dir:
        base["PATH"] = bin_dir + os.pathsep + base.get("PATH", "")
    if env:
        base.update({k: v for k, v in env.items() if v is not None})
        for k, v in env.items():
            if v is None:
                base.pop(k, None)
    return subprocess.run([python or sys.executable, "-c", code] + list(argv or []), env=base, capture_output=True, text=True, cwd=cwd)


def run_cli(args: list, site: str, env: dict = None, bin_dir: str = None, cwd: str = None, python: str = None) -> subprocess.CompletedProcess:
    return run_py("import sys; from rfdiffusion3_opt.cli import main; sys.exit(main(sys.argv[1:]))", site, env, bin_dir, args, cwd, python)
