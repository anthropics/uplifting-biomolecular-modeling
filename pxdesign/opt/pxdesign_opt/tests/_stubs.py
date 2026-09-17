"""Stub upstream stack for the CPU tests: just enough `torch`, `protenix` and `pxdesign` for the REAL kit module
(`opt/forward/hoist/pxd_xattempt/hoist.py`) to import and for its `install(model)` to run its class-level patches and model tags,
so the package's hook, classify() and the activation rules are exercised against the kit's own code without a GPU.

Nothing here reproduces model numerics: the stub forwards are placeholders that are never called.
"""
import sys
import types


def _module(name, **attrs):
    m = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


class _NoGrad:
    def __call__(self, fn=None):
        if fn is None:
            return self
        return fn

    def __enter__(self):
        return None

    def __exit__(self, *a):
        return False


class _Generator:
    def __init__(self, device=None):
        self.device = device

    def manual_seed(self, s):
        return self


def install_torch(cuda: bool = True, name: str = "NVIDIA H100 80GB HBM3", cc=(9, 0), mem_gib: float = 79.6, version: str = "2.3.1+cu121"):
    torch = _module("torch")
    torch.__version__ = version
    torch.version = types.SimpleNamespace(cuda="12.1" if "+cu" in version else None)
    torch.no_grad = _NoGrad()
    torch.Generator = _Generator
    torch.equal = lambda a, b: a is b
    torch.float32 = "float32"
    torch.are_deterministic_algorithms_enabled = lambda: bool(_state["det"])
    torch.use_deterministic_algorithms = lambda v: _state.__setitem__("det", bool(v))
    props = types.SimpleNamespace(name=name, major=cc[0], minor=cc[1], total_memory=int(mem_gib * 2**30))
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, device_count=lambda: 1 if cuda else 0, get_device_properties=lambda i: props,
                                       get_device_name=lambda i=0: name, get_device_capability=lambda device=None: tuple(cc), synchronize=lambda: None,
                                       manual_seed_all=lambda s: None)
    torch.backends = types.SimpleNamespace(cudnn=types.SimpleNamespace(benchmark=False, deterministic=False, allow_tf32=True, version=lambda: 8902),
                                           cuda=types.SimpleNamespace(matmul=types.SimpleNamespace(allow_tf32=False)))
    torch._fp32_matmul_precision = "highest"                                             # torch.get/set_float32_matmul_precision (opt_core.precision.policy reads and sets them)
    torch.get_float32_matmul_precision = lambda: torch._fp32_matmul_precision

    def _set_prec(word):
        torch._fp32_matmul_precision = word
        torch.backends.cuda.matmul.allow_tf32 = word != "highest"
    torch.set_float32_matmul_precision = _set_prec
    torch.random = types.SimpleNamespace(manual_seed=lambda s: None)
    nn = _module("torch.nn"); torch.nn = nn
    nn.Module = type("Module", (), {})
    F = _module("torch.nn.functional"); nn.functional = F
    F.silu = lambda x: x
    return torch


_state = {"det": False}


class OutOfMemoryError(RuntimeError):
    """The stub stack's `torch.cuda.OutOfMemoryError` (torch's own is a RuntimeError subclass of this name): constructible without a GPU;
    `opt_core.oom.is_oom` recognises it by class through the imported `torch` module and by name."""


def install_oom_class():
    """Put the OOM class on the stub torch (`torch.OutOfMemoryError`, `torch.cuda.OutOfMemoryError`) and return it."""
    torch = sys.modules["torch"]
    torch.OutOfMemoryError = OutOfMemoryError; torch.cuda.OutOfMemoryError = OutOfMemoryError
    return OutOfMemoryError


X_SHA = "2d711642b726b04401627ca9fbac32f5c8530fb1903cc4db02258717921a4881"      # sha256(b"x"): the tests' stand-in checkpoint / CCD bytes


def stage_weights(tmp_path, monkeypatch, pins):
    """A checkpoint directory and a CCD cache holding every pinned file name with the bytes b"x" (PROTENIX_DATA_ROOT_DIR exported); returns the
    checkpoint directory. No 1 GB file: b"x" stands in for every weight and CCD file (preflight checks presence only)."""
    ckpt = tmp_path / "ckpt"; ccd = tmp_path / "ccd"; ckpt.mkdir(); ccd.mkdir()
    monkeypatch.setenv("PROTENIX_DATA_ROOT_DIR", str(ccd))
    for f in [pins["weights"]["checkpoint"]["file"], *pins["weights"]["required_in_dir"]["files"]]:
        (ckpt / f).write_bytes(b"x")
    for f in pins["ccd_cache"]["files"]:
        (ccd / f).write_bytes(b"x")
    return ckpt


class _Block:
    def __init__(self, apb, ctb):
        self.attention_pair_bias = apb()
        self.conditioned_transition_block = ctb()


def install_upstream(n_tok_blocks: int = 16, n_atom_blocks: int = 4):
    """pxdesign / protenix stubs with the class names hoist.install() rebinds. Returns the ProtenixDesign stub class."""
    # protenix
    protenix = _module("protenix")
    _module("protenix.model"); _module("protenix.model.modules"); _module("protenix.utils"); _module("protenix.openfold_local"); _module("protenix.openfold_local.model")

    class DiffusionModule:
        def f_forward(self, *a, **k):
            return "stock_f_forward"

    class AttentionPairBias:
        def forward(self, *a, **k):
            return "stock_apb"

    class ConditionedTransitionBlock:
        def forward(self, *a, **k):
            return "stock_ctb"

    def _local_attention(*a, **k):
        return "stock_local_attention"

    _module("protenix.model.modules.diffusion", DiffusionModule=DiffusionModule)
    _module("protenix.model.modules.transformer", AttentionPairBias=AttentionPairBias, ConditionedTransitionBlock=ConditionedTransitionBlock)
    inits = []

    def trunc_normal_init_(weights, scale=1.0, fan="fan_in"):     # Protenix's construction-time initialiser (openfold_local/model/primitives.py:90); `calls` records each run
        inits.append(scale)
        return weights

    def rearrange_qk_to_dense_trunk(*a, **k):                    # the two dense-trunk primitives the padmask lever rebinds (sizeceil.py)
        return "stock_qk_trunk"

    def rearrange_to_dense_trunk(*a, **k):
        return "stock_dense_trunk"

    _module("protenix.model.modules.primitives", _local_attention=_local_attention, trunc_normal_init_=trunc_normal_init_,
            rearrange_qk_to_dense_trunk=rearrange_qk_to_dense_trunk, rearrange_to_dense_trunk=rearrange_to_dense_trunk)
    seeds = []

    def seed_everything(seed, deterministic):
        seeds.append((seed, deterministic))
        _state["det"] = bool(deterministic)

    _module("protenix.utils.seed", seed_everything=seed_everything, calls=seeds)

    class OpenFoldLayerNorm:
        pass

    class FusedLayerNorm:
        pass

    import os
    prim = _module("protenix.openfold_local.model.primitives", trunc_normal_init_=trunc_normal_init_, calls=inits)   # the defining module (openfold_local/model/primitives.py:90)
    prim.LayerNorm = FusedLayerNorm if os.environ.get("LAYERNORM_TYPE") == "fast_layernorm" else OpenFoldLayerNorm

    # pxdesign
    pxdesign = _module("pxdesign"); _module("pxdesign.model"); _module("pxdesign.runner")

    def sample_diffusion(denoise_net=None, input_feature_dict=None, s_inputs=None, s_trunk=None, z_trunk=None, **kw):
        return "stock_sample_diffusion"

    gen = _module("pxdesign.model.generator", sample_diffusion=sample_diffusion)

    class _Transformer:
        def __init__(self, n):
            self.blocks = [_Block(AttentionPairBias, ConditionedTransitionBlock) for _ in range(n)]

    class _AtomTransformer:
        def __init__(self, n):
            self.diffusion_transformer = _Transformer(n)

    class _DM(DiffusionModule):
        def __init__(self):
            self.diffusion_transformer = _Transformer(n_tok_blocks)
            self.atom_attention_encoder = types.SimpleNamespace(atom_transformer=_AtomTransformer(n_atom_blocks))
            self.atom_attention_decoder = types.SimpleNamespace(atom_transformer=_AtomTransformer(n_atom_blocks))

    class ProtenixDesign:
        def __init__(self, configs=None):
            self.configs = configs
            self.diffusion_module = _DM()
            import protenix.model.modules.primitives as _PR   # a Protenix Linear's init: trunc_normal_init_ through the importing module's reference (primitives.py:71)
            _PR.trunc_normal_init_("weights", scale=1.0)

        def to(self, device):
            return self

        def eval(self):
            return self

        def load_state_dict(self, state_dict, strict=True):   # torch's return value shape: the incompatible keys
            return types.SimpleNamespace(missing_keys=[], unexpected_keys=[])

    P = _module("pxdesign.model.pxdesign", ProtenixDesign=ProtenixDesign, sample_diffusion=sample_diffusion)

    class InferenceRunner:
        def __init__(self, configs=None):
            self.configs = configs
            self.init_model()
            self.load_checkpoint()

        def init_model(self):                              # upstream: runner/inference.py:88
            self.model = ProtenixDesign(self.configs)

        def load_checkpoint(self):                          # upstream: runner/inference.py:91-114 (a strict load_state_dict, then eval())
            self.load_result = self.model.load_state_dict(state_dict={}, strict=True)
            self.model.eval()
            self.checkpoint_loaded = True

        def predict(self, data):                            # upstream: runner/inference.py:128-147 (the model call on one item's features)
            return {"predicted": data.get("sample_name")}

    _module("pxdesign.runner.inference", InferenceRunner=InferenceRunner, ProtenixDesign=ProtenixDesign)
    return ProtenixDesign


UPSTREAM_MODULES = ("torch", "torch.nn", "torch.nn.functional", "protenix", "protenix.model", "protenix.model.modules", "protenix.model.modules.diffusion",
                    "protenix.model.modules.transformer", "protenix.model.modules.primitives", "protenix.utils", "protenix.utils.seed", "protenix.openfold_local",
                    "protenix.openfold_local.model", "protenix.openfold_local.model.primitives", "pxdesign", "pxdesign.model", "pxdesign.model.generator",
                    "pxdesign.model.pxdesign", "pxdesign.runner", "pxdesign.runner.inference", "pxd_xattempt", "pxd_xattempt.hoist")


def uninstall():
    for m in UPSTREAM_MODULES:
        sys.modules.pop(m, None)


def fake_box(monkeypatch, stack, cuda=True, pinned=True, name="NVIDIA H100 80GB HBM3", cc=(9, 0), mem_gib=79.6, torch="2.3.1+cu121"):
    """Make stack gates see a pinned upstream and an H100 (or no GPU)."""
    monkeypatch.setattr(stack, "gpu_probe", lambda: {"name": name if cuda else None, "cc": cc if cuda else None, "sm": f"sm{cc[0]}{cc[1]}" if cuda else None,
                                                     "mem_gib": mem_gib if cuda else None, "torch": torch, "cuda": "12.1", "count": 1 if cuda else 0, "probe": "torch"})
    monkeypatch.setattr(stack, "upstream_versions", lambda: {"pxdesign": "0.1.0", "protenix": "0.5.0+pxd", "pxdbench": "0.1.2"})
    monkeypatch.setattr(stack, "pins_report", lambda pins=None: {"pinned": pinned, "bad": [] if pinned else ["pxdesign 0.1.0: installed from git commit deadbeef; want f788441"], "detail": {},
                                                          "stack": {"status": "OPEN", "pinned_id": "pxd-cu121", "running": {"python": "3.11.5", "torch": "2.3.1+cu121", "cuda": "12.1"},
                                                                    "pinned": {"torch": "2.3.1+cu121"}, "differs": {}, "equal": True}})
