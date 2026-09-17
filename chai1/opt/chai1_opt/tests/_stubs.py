"""Stub ``torch`` and ``chai_lab`` modules for the CPU tests: enough surface for the kit driver's W1/W5 statements to install and for
the package's gates to run, nothing that computes. Installed into ``sys.modules`` by ``install()``; ``remove()`` restores the interpreter.
``eager_stub(stack)`` stands in for the eager stack's ``chai1_eager.stack`` (``stack.eager_module``): its ``install`` patches
``chai1.load_exported`` with a loader wrapping the attribute it finds — the real stack's shape (``StackHandle``: loader, orig, parts,
stats, restore; ``LEVERS``) without a model."""
from __future__ import annotations

import contextlib
import os
import sys
import types

STUB_NAMES = ("torch", "torch.cuda", "torch.jit", "torch.backends", "torch.backends.cudnn", "torch.backends.cuda", "torch.version", "torch._C",
              "chai_lab", "chai_lab.chai1", "chai_lab.data", "chai_lab.data.dataset", "chai_lab.data.dataset.embeddings",
              "chai_lab.data.dataset.embeddings.esm")


class _Props:
    name, major, minor, total_memory = "STUB H100 80GB HBM3", 9, 0, 80 * 2**30


def make_torch(version="2.5.1+cu124", cuda_available=True):
    t = types.ModuleType("torch"); t.__version__ = version
    cuda = types.ModuleType("torch.cuda")
    cuda.is_available = lambda: cuda_available
    cuda.get_device_properties = lambda i=0: _Props()
    cuda.synchronize = lambda *a, **k: None
    cuda.max_memory_allocated = lambda *a, **k: 0
    cuda.max_memory_reserved = lambda *a, **k: 0
    cuda.reset_peak_memory_stats = lambda *a, **k: None
    cuda.empty_cache = lambda: None
    jit = types.ModuleType("torch.jit"); jit.load = lambda path, map_location=None: types.SimpleNamespace(path=path, device=map_location)
    backends = types.ModuleType("torch.backends"); cudnn = types.ModuleType("torch.backends.cudnn"); cudnn.deterministic = False; cudnn.benchmark = False
    backends.cudnn = cudnn
    bcuda = types.ModuleType("torch.backends.cuda"); bcuda.matmul = types.SimpleNamespace(allow_tf32=False)   # the opt-in lever tf32's switch (precision.py)
    backends.cuda = bcuda
    cudnn.allow_tf32 = True                                                                                     # torch's default; read by the numerics signature
    _mm = {"precision": "highest"}

    def set_float32_matmul_precision(word):                             # torch links the word and the flag: high|medium -> allow_tf32 True
        _mm["precision"] = word; bcuda.matmul.allow_tf32 = word != "highest"

    def get_float32_matmul_precision():
        return "high" if (bcuda.matmul.allow_tf32 and _mm["precision"] == "highest") else _mm["precision"]
    t.set_float32_matmul_precision, t.get_float32_matmul_precision = set_float32_matmul_precision, get_float32_matmul_precision
    t.is_autocast_enabled = lambda device_type=None: False
    t.get_autocast_dtype = lambda device_type=None: "none"
    t.is_deterministic_algorithms_warn_only_enabled = lambda: False
    version = types.ModuleType("torch.version"); version.cuda = "12.4"
    _C = types.ModuleType("torch._C"); _C.calls = []; _C._jit_set_profiling_mode = lambda v: _C.calls.append(("profiling", v))
    t.cuda, t.jit, t.backends, t.version, t._C = cuda, jit, backends, version, _C
    t.device = lambda s: s
    t.no_grad = contextlib.nullcontext
    t.use_deterministic_algorithms = lambda mode, warn_only=False: setattr(t, "_det", (mode, warn_only))
    t.are_deterministic_algorithms_enabled = lambda: bool(getattr(t, "_det", (False, False))[0])
    t.get_rng_state = lambda: 0
    t.manual_seed = lambda s: None
    return t, {"torch": t, "torch.cuda": cuda, "torch.jit": jit, "torch.backends": backends, "torch.backends.cudnn": cudnn,
               "torch.backends.cuda": bcuda, "torch.version": version, "torch._C": _C}


def make_chai_lab():
    chai_lab = types.ModuleType("chai_lab"); chai_lab.__version__ = "0.6.1"; chai_lab.__path__ = []
    chai1 = types.ModuleType("chai_lab.chai1")

    def load_exported(comp_key, device):
        return types.SimpleNamespace(comp_key=comp_key, device=device)
    load_exported.__module__ = "chai_lab.chai1"
    chai1.load_exported = load_exported
    chai1.run_inference = lambda **kw: None
    chai1.run_folding_on_context = lambda *a, **kw: None
    chai1.make_all_atom_feature_context = lambda *a, **kw: None
    chai1.set_seed = lambda s: None
    esm = types.ModuleType("chai_lab.data.dataset.embeddings.esm")
    esm._esm_model = []

    @contextlib.contextmanager
    def esm_model(device):
        yield types.SimpleNamespace(device=device)
    esm_model.__module__ = esm.__name__

    def _get_esm_contexts_for_sequences(prot_sequences, device):
        return {}
    _get_esm_contexts_for_sequences.__module__ = esm.__name__
    esm.esm_model = esm_model
    esm._get_esm_contexts_for_sequences = _get_esm_contexts_for_sequences
    data = types.ModuleType("chai_lab.data"); data.__path__ = []
    dataset = types.ModuleType("chai_lab.data.dataset"); dataset.__path__ = []
    embeddings = types.ModuleType("chai_lab.data.dataset.embeddings"); embeddings.__path__ = []
    chai_lab.chai1 = chai1; chai_lab.data = data; data.dataset = dataset; dataset.embeddings = embeddings; embeddings.esm = esm
    return {"chai_lab": chai_lab, "chai_lab.chai1": chai1, "chai_lab.data": data, "chai_lab.data.dataset": dataset,
            "chai_lab.data.dataset.embeddings": embeddings, "chai_lab.data.dataset.embeddings.esm": esm}


def install(cuda_available=True):
    """Install the stubs (replacing any real module of the same name for the test's duration). Returns (torch, chai1, esm)."""
    t, tmods = make_torch(cuda_available=cuda_available)
    cmods = make_chai_lab()
    saved = {n: sys.modules.get(n) for n in STUB_NAMES}
    saved["env:PYTORCH_CUDA_ALLOC_CONF"] = os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)   # the kit rows export the allocator policy in-process (alloc.py): start clean, restore on remove
    sys.modules.update(tmods); sys.modules.update(cmods)
    return t, cmods["chai_lab.chai1"], cmods["chai_lab.data.dataset.embeddings.esm"], saved


def remove(saved):
    v = saved.get("env:PYTORCH_CUDA_ALLOC_CONF")
    if v is None:
        os.environ.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    else:
        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = v
    for n in STUB_NAMES:
        if saved.get(n) is None:
            sys.modules.pop(n, None)
        else:
            sys.modules[n] = saved[n]


def gates_pass(stack, weights_dir="/stub/weights", gpu_name=None, pinned=True):
    """Monkeypatch the gates that read the box — an installed upstream, real weights, the mode's capability check (this box builds every
    row), the torch/CUDA stack record (``pinned``: this box passes for the pinned stack, so no NOTE line rides the reports; False leaves
    the real ``stack_pinning``) — so activation logic can be tested on a CPU box; ``gpu_name`` replaces the stub torch's GPU name."""
    saved = (stack.pins_check, stack.weights_check, stack.gpu_info, stack.mode_stack_check, stack.weights_match, stack.stack_pinning)
    stack.mode_stack_check = lambda km, compiler=None: (None, None)
    if pinned:
        stack.stack_pinning = lambda environ=None: {"stack_pinned": True, "stack_line": None, "stack_strict": False}
    stack.weights_match = lambda d, afresh=False: None                                               # no bytes to hash on the stub box (tests set pinned / unknown explicitly)
    stack.pins_check = lambda: ([], {"chai_lab": {"version": "0.6.1", "pinned": True, "files_checked": 60, "files_differ": []},
                                     "torch": {"version": "2.13.0+cu130", "cuda": "13.0", "declared_range": "torch<2.7,>=2.3.1", "in_range": False,
                                               "pinned_stack": "stack", "pinned": True, "line": None, "strict": False}})
    stack.weights_check = lambda downloads_dir=None: (True, weights_dir, [])
    if gpu_name is not None:
        stack.gpu_info = lambda use_torch=False: {"visible": True, "name": gpu_name, "cc": "8.9", "mem_mib": 24564, "source": "stub"}
    return saved


def gates_restore(stack, saved):
    stack.pins_check, stack.weights_check, stack.gpu_info, stack.mode_stack_check, stack.weights_match, stack.stack_pinning = saved


class PWAStub:
    """A stand-in ``MSAPairWeightedAveraging`` with the carried forward's signature (big's chunked class wraps it)."""
    def forward(self, msa, z, pair_mask, msa_mask):
        return msa


class OPMStub:
    """A stand-in ``OuterProductMean`` (big's opm_chunk re-classes it; never called on the stubs)."""
    def forward(self, msa, *a, **k):
        return msa


class EagerTrunkStub:
    """The real ``EagerTrunkWrapper`` surface the package reads: ``cfg_fn`` (re-applied before each call), ``trunk`` (the eager trunk
    module: its ``msa_module`` carries the pair-weighted-averaging and outer-product-mean lists big's levers re-class); ``forward``
    counts its calls."""
    def __init__(self, cfg_fn=None):
        import types as _types
        self.calls = 0; self.cfg_fn = cfg_fn
        self.trunk = _types.SimpleNamespace(msa_module=_types.SimpleNamespace(msa_pair_weighted_averaging=[PWAStub()], outer_product_mean=[OPMStub()]))

    def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
        self.calls += 1
        if self.cfg_fn:
            self.cfg_fn()
        return ("trunk", crop_size)


class HoistStub:
    """The real ``HoistedForward`` per-item state (``cache``, ``_graph``, ``_static_kw``, ``_static_out``)."""
    def __init__(self):
        self.cache = self._graph = self._static_kw = self._static_out = None


class DiffusionStub:
    """The real ``HoistedDiffusionWrapper`` surface the package reads: ``events``, the DSTEP policy the add-on's builder sets, and the
    hoist it keys (``key`` / ``hf`` / ``forward``: the stack's own rule, a precompute whenever ``key`` differs from the last one; the
    key is what ``key_of`` returns — the data-pointer form of the real stack by default)."""
    def __init__(self, graphed=True, hoister="base"):
        self.graphed, self.events, self.ln_policy, self.hoister = graphed, [], None, hoister
        self.stand_down, self.hoister_used, self.compile, self.compile_failed, self.dit_attn = None, None, None, None, None
        self.hf, self.key, self.precomputes = {}, None, []

    def hoister_now(self):
        return self.stand_down if self.stand_down in ("base",) and self.hoister != "base" else self.hoister

    @staticmethod
    def key_of(crop_size, kw):
        return (crop_size, kw["token_pair_trunk_repr"].data_ptr(), kw["atom_single_input_feats"].data_ptr(), tuple(kw["atom_noised_coords"].shape))

    def forward(self, crop_size, *, return_on_cpu=False, move_to_device=None, **kw):
        key = self.key_of(crop_size, kw)
        hf = self.hf.setdefault(crop_size, HoistStub())
        if key != self.key:
            hf.cache = {"item": kw["token_pair_trunk_repr"].tag}; hf._graph = "graph"; hf._static_kw = dict(kw); hf._static_out = "out"
            self.precomputes.append(kw["token_pair_trunk_repr"].tag)
            self.key = key
        return hf.cache["item"]


class PolicyStub:
    def __init__(self, mode):
        self.mode, self.n_fast, self.n_slow = mode, 0, 0


class EagerHandleStub:
    """The real ``chai1_eager.stack.StackHandle`` surface the package reads: (C1, orig, loader, parts); stats() / restore()."""
    def __init__(self, C1, orig, loader, parts):
        self.C1, self.orig, self.loader, self.parts = C1, orig, loader, parts
        self.levers = "tier1"
        self.events = self.parts["diffusion"].events

    def stats(self):
        return {"levers": self.levers, "diffusion_events": list(self.events)}

    def restore(self):
        self.C1.load_exported = self.orig


def make_eager_module(levers=("stock", "tier1")):
    """A stand-in ``chai1_eager.stack``: ``LEVERS``, ``install(levers=..., device=...)`` patching the stub chai1's ``load_exported``,
    and the lower-level surface the DSTEP path composes (``Components`` / ``build_parts`` / ``make_loader`` / ``StackHandle``)."""
    import types as _t
    mod = _t.ModuleType("chai1_eager.stack"); mod.LEVERS = tuple(levers); mod.installs = []; mod.StackHandle = EagerHandleStub
    mod.HoistedDiffusionWrapper, mod.EagerTrunkWrapper = DiffusionStub, EagerTrunkStub                # the classes hoist.adopt re-classes
    mod._move = lambda kw, dev: kw
    trunk = _t.ModuleType("chai1_eager.trunk")                                                        # the trunk module's CFG hooks (the tier1 triangle-attention rule and big's chunker read/write them)
    trunk.CFG = dict(precast_bf16=False, trimul_impl=None, triattn_impl=None, record_ranges=False, ln_impl=None, msa_impl=None, transition_impl=None)
    for _cls in ("TemplateEmbedder", "TriangleAttention", "Transition", "TriangleMultiplication"):          # the classes pairtrack.py hooks (forward replaced at class level)
        setattr(trunk, _cls, type(_cls, (), {"forward": lambda self, *a, **k: None}))
    mod.trunk = trunk; mod.cfg_calls = []
    sys.modules["chai1_eager.trunk"] = trunk                                                          # importable by name, as on the eager stack's sys.path (removed by eager_restore)
    serve = _t.ModuleType("chai1_exactln_serve")                                                       # pairtrack's `exactln` serving module (opt/forward/chai1_exactln/serve.py) on a CPU box: installs, serves nothing
    serve.installs = []
    serve.install = lambda mode="fast": (serve.installs.append(mode), {"torch": "stub", "cc": "90", "word": mode})[1]   # binds the mode's tier word (exact | fast | big)
    serve.ln = lambda kind, x, c, w=None, b=None, eps=1e-5: NotImplemented
    serve.census = lambda: {"served": 0, "served_by": {}, "rows_served": {}, "statement": {}, "fallback": 0, "fallback_by": {}, "errors": {}, "first_error": None, "selftest": {}, "classes": 0, "word": (serve.installs[-1] if serve.installs else None), "facts": {}}
    serve.gate = lambda: (True, ""); serve.line = lambda tag="[chai1-opt]": f"{tag} LEVER exactln pairtrack served=0"; serve.reset = lambda: None
    serve._S = {"fallback": {}}; serve._cnt = lambda d, k, n=1: d.__setitem__(k, d.get(k, 0) + n)
    serve.bind_flat = lambda flat, name="confidence_head": f"{name}:layer_norm+to"; serve.conf_fields = lambda: {"bound": False}
    sys.modules["chai1_exactln_serve"] = serve
    mk = _t.ModuleType("chai1_eager.msa_kernels")                                                         # pairtrack's `msa_pad` statement module (chai1_eager/msa_kernels.py) on a CPU box: installs, cuts nothing
    def _mk_ledger(expected=("no_tail",)):
        from opt_core.counters import Ledger
        class _L(Ledger):
            def gate(self, name=None, *, require_served=False):
                return Ledger.gate(self, name, require_served=require_served)
        return _L("LOCAL.chai1.msa_pad", impl="chai1_eager.msa_kernels@stub", origin="kit", expected=tuple(expected))
    class _MsaPad:
        chai1_opt_lever = "msa_pad"
        def __init__(self, TR, ledger, selftest=True): self.TR, self.ledger, self.selftest = TR, ledger, selftest
        def describe(self): return {"impl": "chai1_eager.msa_kernels@stub", "block_opm": 4096, "block_mod": 8192, "selftest": True, "classes": {}}
        def __call__(self, *a, **k): return NotImplemented
    mk.new_ledger, mk.MsaPad = _mk_ledger, _MsaPad
    sys.modules["chai1_eager.msa_kernels"] = mk
    tc = _t.ModuleType("chai1_eager.transition_core")                                                     # pairtrack's `transition` binding (chai1_eager/transition_core.py) on a CPU box: installs, serves nothing
    def _tc_ledger(expected=("stock", "no_cell", "refused", "class_differs")):
        from opt_core.counters import Ledger
        class _L(Ledger):
            def gate(self, name=None, *, require_served=False):
                return Ledger.gate(self, name, require_served=require_served)
        return _L("LOCAL.chai1.transition", impl="opt_core.kernels.transition", origin="core", expected=tuple(expected))
    class _TB:
        chai1_opt_lever = "transition"
        def __init__(self, TR, ledger, mode, selftest=True): self.TR, self.ledger, self.mode = TR, ledger, mode; self.tier = {"exact": "exact", "fast": "fast", "big": "big"}[mode]
        def describe(self): return {"provider": "opt_core.kernels.transition", "core": "0", "mode": self.mode, "tier": self.tier, "cc": "none", "stack": "none", "classes": "none", "refused": "none", "stock": "none", "rows": "none"}
        def __call__(self, *a, **k): return NotImplemented
    tc.new_ledger, tc.TransitionBinding, tc.TIER, tc.EXPECTED_FALLBACKS = _tc_ledger, _TB, {"exact": "exact", "fast": "fast", "big": "big"}, ("stock", "no_cell", "refused", "class_differs")
    sys.modules["chai1_eager.transition_core"] = tc

    class Components:
        def __init__(self, downloads_dir=None, device="cuda:0", code_root=None):
            self.C1 = sys.modules["chai_lab.chai1"]; self.orig_load = self.C1.load_exported; self.device = device

        @staticmethod
        def cfg_lc():                                                                                 # the tier1 line's settings, re-applied before every trunk call
            trunk.CFG.update(precast_bf16=True); mod.cfg_calls.append("cfg_lc")

    def build_parts(comps, lv, *, graphed=True, sink=None):
        if lv not in mod.LEVERS:
            raise ValueError(f"unknown levers {lv!r}")
        if lv == "stock":
            return {}
        return {"trunk": EagerTrunkStub(cfg_fn=Components.cfg_lc), "diffusion": DiffusionStub(graphed=graphed), "flat_rest": True}

    def make_loader(comps, parts, sink=None):
        orig = comps.orig_load

        def load(comp_key, device=comps.device):                              # the stack's loader: memoised stock modules underneath
            return _t.SimpleNamespace(comp_key=comp_key, device=device, eager=parts, base=orig(comp_key, device))
        load.__name__ = "load"
        return load

    def install(levers="tier1", *, graphed=None, sink=None, comps=None, device="cuda:0"):
        comps = comps or Components(device=device)
        parts = build_parts(comps, levers, graphed=bool(graphed))
        load = make_loader(comps, parts)
        comps.C1.load_exported = load
        h = EagerHandleStub(comps.C1, comps.orig_load, load, parts); mod.installs.append(h)
        return h
    mod.Components, mod.build_parts, mod.make_loader, mod.install = Components, build_parts, make_loader, install
    return mod


def make_dstep_module(all_levers=("hoist2", "compiled", "dit_attn")):
    """A stand-in ``chai1_fastln.stackx``: ``ALL_LEVERS`` and ``build_lever_parts`` setting the lever's hoister / the add-on's policy on a private denoiser."""
    import types as _t
    mod = _t.ModuleType("chai1_fastln.stackx"); mod.ALL_LEVERS = tuple(all_levers); mod.builds = []

    def build_lever_parts(comps, base_parts, line, lv, *, sink=None, graphed=True, mode="fast"):
        dw = DiffusionStub(graphed=graphed)
        for name in lv:
            if name not in mod.ALL_LEVERS:
                raise ValueError(name)
            if name == "hoist2":
                dw.hoister = "hoist2"
            elif name == "compiled":
                dw.compile = "default"
            elif name == "dit_attn":
                dw.dit_attn = object()
        mod.builds.append((line, tuple(lv)))
        return {"trunk": base_parts[line]["trunk"], "diffusion": dw, "flat_rest": True}
    mod.build_lever_parts = build_lever_parts
    return mod


def eager_stub(stack, levers=("stock", "tier1"), dstep_levers=("hoist2", "compiled", "dit_attn")):
    """Monkeypatch ``stack.eager_module`` / ``stack.dstep_module`` to return stand-in kits; returns (saved, module) — ``module.dstep`` is
    the DSTEP stand-in."""
    saved = (stack.eager_module, stack.dstep_module)
    mod = make_eager_module(levers); mod.dstep = make_dstep_module(dstep_levers)
    stack.eager_module = lambda: mod
    stack.dstep_module = lambda: mod.dstep
    return saved, mod


def eager_restore(stack, saved):
    stack.eager_module, stack.dstep_module = saved
    sys.modules.pop("chai1_eager.trunk", None); sys.modules.pop("chai1_exactln_serve", None); sys.modules.pop("chai1_eager.msa_kernels", None); sys.modules.pop("chai1_eager.transition_core", None)
