"""fpf_engines — the FlashPairformer engine adapter for OpenDDE.

Op names (the REGISTRY below):
  trimul_out, trimul_in, triatt (starting+ending; module.starting tells which), transition, single_attention_pb,
  outer_product_mean, msa_pair_weighted_avg, pairformer_block.

ENGINE-FUNCTION RULE: an op name is bound to one engine's function.  The engine registry below names
the engine's own stock class and the exact stock forward signature; a kernel registered for an engine reproduces
THAT class (its error reference is that class in fp64 on the same inputs).  Kernel callables receive
`fn(module, *args, **kwargs)` with EXACTLY the stock signature of that engine's class, and must return the stock
return value.  They may cache packed weights on `module._fpf_cache`; they must never mutate module parameters.

Environment contract (read at enable time, all default-off; with nothing set every patched class is byte-for-byte
the stock method object -> nothing changes):
  FPF_ENGINE   = opendde                     (which registry; auto-detected from the installed package when unset)
  FPF_OPS      = comma list of op names to enable (e.g. trimul_out,trimul_in,triatt)
  FPF_IMPL     = comma list  op=module.path:attr  (callable per op; an op in FPF_OPS without an impl is a config error)
  FPF_CROSSCHECK   = k   -> the first k calls per op run BOTH the stock method and the kernel; max|d| (fp64) and bitwise
                       equality are recorded in STATS[op]['cross-check'] (list of dicts) ; output returned = the KERNEL's
  FPF_CROSSCHECK_REF = kernel|stock  -> which output is returned during cross-check (default kernel)
  FPF_STRICT   = 1   -> a kernel exception is raised instead of falling back to stock

Fallbacks are COUNTED with a reason (STATS[op]['fallbacks'][reason] += 1) and never silent: unsupported dtype/shape/
mask/device, kernel import failure, kernel exception (unless FPF_STRICT), 'disabled' (op not in FPF_OPS).
"""
from __future__ import annotations
from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import os, sys, json, time, importlib, importlib.util, functools, threading
import torch

__version__ = "0.1.2"
OP_NAMES = ("trimul_out", "trimul_in", "triatt", "transition", "single_attention_pb", "outer_product_mean", "msa_pair_weighted_avg", "pairformer_block")

# ----------------------------------------------------------------------------------------------------------------
# Engine registries: op name -> (module path, class name, method name, signature note).  Signatures are the STOCK ones.
# ----------------------------------------------------------------------------------------------------------------
REGISTRY = {
    "opendde": {
        # OpenDDE 1.x: fp32 weights + activations, TF32 ON (enable_tf32), c_z = 384, c_s = 384, 48 blocks.
        "trimul_out": ("opendde.model.triangular.triangular", "TriangleMultiplicationOutgoing", "forward",
                       "forward(self, z[*,N,N,c_z], mask=None, inplace_safe=False, _add_with_inplace=False, _inplace_chunk_size=256, triangle_multiplicative='torch') -> z(+update when inplace_safe&_add_with_inplace)"),
        "trimul_in": ("opendde.model.triangular.triangular", "TriangleMultiplicationIncoming", "forward",
                      "same as trimul_out"),
        "triatt": ("opendde.model.triangular.triangular", "TriangleAttention", "forward",
                   "forward(self, x[*,I,J,C], mask=None, chunk_size=None, triangle_attention='torch', inplace_safe=False) -> update (self.starting)"),
        "transition": ("opendde.model.modules.primitives", "Transition", "forward",
                       "forward(self, x[...,c]) -> update (chunked internally by row count)"),
        "single_attention_pb": ("opendde.model.modules.transformer", "AttentionPairBias", "forward",
                                "forward(self, a, s, z, n_queries=None, n_keys=None, inplace_safe=False, chunk_size=None, enable_efficient_fusion=False, extra_attn_bias=None) -> update"),
        "outer_product_mean": ("opendde.model.triangular.layers", "OuterProductMean", "forward",
                               "forward(self, m, mask=None, chunk_size=None, inplace_safe=False) -> z_update"),
        "msa_pair_weighted_avg": ("opendde.model.modules.pairformer", "MSAPairWeightedAveraging", "forward",
                                  "forward(self, m, z, z_pair_spec=None, foldcp_mesh=None) -> m_update"),
        "pairformer_block": ("opendde.model.modules.pairformer", "PairformerBlock", "forward",
                             "forward(self, s, z, pair_mask, triangle_multiplicative='torch', triangle_attention='torch', inplace_safe=False, chunk_size=None, extra_attn_bias=None) -> (s, z)"),
    },
}

_ORIG: dict = {}          # (engine, op) -> (cls, original method)
STATS: dict = {}          # op -> {'calls', 'kernel', 'stock', 'fallbacks': {reason: n}, 'cross-check': [...], 'ms_kernel_sum', 'ms_stock_sum'}
_LOCK = threading.Lock()
_CFG: dict = {"engine": None, "ops": [], "impl": {}, "cross-check": 0, "crosscheck_ref": "kernel", "strict": False}
_DIM_GUARD: dict = {}     # op -> set of allowed channel widths (FPF_DIMS="transition=64,128;..."); other widths -> counted fallback without invoking the kernel


class FPFFallback(Exception):
    """Raise inside a kernel callable to fall back to stock with a counted reason (e.g. unsupported shape)."""
    def __init__(self, reason: str):
        super().__init__(reason); self.reason = reason


def detect_engine() -> str | None:
    e = os.environ.get("FPF_ENGINE", "").strip().lower()
    if e: return e
    for name, mod in (("opendde", "opendde"),):
        if importlib.util.find_spec(mod) is not None:
            return name
    return None


def _stat(op):
    if op not in STATS:
        STATS[op] = {"calls": 0, "kernel": 0, "stock": 0, "fallbacks": {}, "cross-check": [], "ms_kernel_sum": 0.0, "ms_stock_sum": 0.0}
    return STATS[op]


def _fallback(op, reason):
    s = _stat(op); s["fallbacks"][reason] = s["fallbacks"].get(reason, 0) + 1; s["stock"] += 1



def _err_context(out, out_stock) -> dict:
    """Context for a cross-check record: the stock output's max magnitude, the bf16 ulp at that magnitude, the element-wise max relative |d| and its
    location. A max_abs_d equal to one bf16 ulp at max|stock| is the stock's own output-rounding class; anything larger is a real discrepancy."""
    try:
        a = out[-1] if isinstance(out, (tuple, list)) else out; b = out_stock[-1] if isinstance(out_stock, (tuple, list)) else out_stock
        if not (torch.is_tensor(a) and torch.is_tensor(b)) or a.shape != b.shape: return {}
        a32 = a.detach().float(); b32 = b.detach().float(); d = (a32 - b32).abs()
        mx = float(b32.abs().max()); idx = int(d.argmax()); dmax = float(d.flatten()[idx]); bref = float(b32.flatten()[idx].abs())
        import math
        ulp = (2.0 ** (math.floor(math.log2(mx)) - 7)) if mx > 0 else 0.0
        return {"max_abs_stock": mx, "bf16_ulp_at_max_stock": ulp, "stock_at_argmax_d": bref, "max_rel_d_at_argmax": (dmax / bref if bref > 0 else None),
                "max_abs_d_in_ulps_of_stock_at_argmax": (dmax / (2.0 ** (math.floor(math.log2(bref)) - 7)) if bref > 0 else None)}
    except Exception as e:  # noqa: BLE001
        return {"err_context_error": repr(e)[:120]}


def _module_dim(module):
    for path in (("norm", "normalized_shape"), ("layer_norm_in", "normalized_shape"), ("norm_in", "normalized_shape"), ("layernorm1", "normalized_shape")):
        o = getattr(module, path[0], None); v = getattr(o, path[1], None) if o is not None else None
        if v: return int(v[0])
    for attr in ("c_in", "dim", "c_z", "c_s"):
        v = getattr(module, attr, None)
        if isinstance(v, int): return v
    return None


def label_instances(model, roots=("template_module", "msa_module", "pairformer_module", "structure_module", "confidence_module", "diffusion_module")) -> dict:
    """Stamp every hooked module with its path in the engine model (module._fpf_path) and return the instance census
    {op: {(root, dim): n_instances}} — an engine can serve several instances of one op (different roots or dims) through the same registry
    entry point; per-instance counters make such rows visible."""
    import collections
    engine = _CFG.get("engine") or detect_engine(); reg = REGISTRY.get(engine, {})
    cls_to_op = {}
    for op, (modpath, clsname, meth, _sig) in reg.items():
        try: cls_to_op[getattr(importlib.import_module(modpath), clsname)] = op
        except Exception: pass
    census = collections.defaultdict(collections.Counter)
    for name, m in model.named_modules():
        for cls, op in cls_to_op.items():
            if isinstance(m, cls):
                m._fpf_path = name; m._fpf_root = name.split(".")[0] if name else "<root>"; m._fpf_dim = _module_dim(m)
                census[op][(m._fpf_root, m._fpf_dim)] += 1
    _CFG["instance_census"] = {op: {f"{r}|dim{d}": n for (r, d), n in c.items()} for op, c in census.items()}
    return _CFG["instance_census"]


def _inst(st, module, kind):
    key = f"{getattr(module, '_fpf_root', '?')}|dim{getattr(module, '_fpf_dim', None) if hasattr(module, '_fpf_dim') else _module_dim(module)}"
    bi = st.setdefault("by_instance", {}); rec = bi.setdefault(key, {"calls": 0, "kernel": 0, "stock": 0}); rec["calls"] += 1; rec[kind] += 1

def _first_tensor(args, kwargs):
    for a in list(args) + list(kwargs.values()):
        if torch.is_tensor(a): return a
    return None


def _maxabs(a, b):
    if isinstance(a, (tuple, list)):
        return max(_maxabs(x, y) for x, y in zip(a, b))
    if a is None or b is None: return 0.0
    return float((a.detach().double() - b.detach().double()).abs().max().item())


def _bitwise(a, b):
    if isinstance(a, (tuple, list)):
        return all(_bitwise(x, y) for x, y in zip(a, b))
    if a is None or b is None: return a is b
    return bool(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b))


def _clone_args(args, kwargs):
    ca = [a.clone() if torch.is_tensor(a) else a for a in args]
    ck = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in kwargs.items()}
    return ca, ck


def _make_wrapper(op, fn, orig, crosscheck_n, crosscheck_ref, strict):
    @functools.wraps(orig)
    def wrapped(self, *args, **kwargs):
        st = _stat(op); st["calls"] += 1
        n_call = st["calls"]
        do_crosscheck = n_call <= crosscheck_n
        out_stock = None
        if do_crosscheck:
            # cross-check: run stock on CLONED inputs first (stock may be in-place), then the kernel on the originals
            sa, sk = _clone_args(args, kwargs)
            t0 = time.perf_counter(); out_stock = orig(self, *sa, **sk)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            st["ms_stock_sum"] += (time.perf_counter() - t0) * 1e3
        try:
            if op in _DIM_GUARD:
                _C = _module_dim(self)
                if _C is not None and _C not in _DIM_GUARD[op]: raise FPFFallback(f"dim_guard_{_C}")
            t0 = time.perf_counter(); out = fn(self, *args, **kwargs)
            if do_crosscheck and torch.cuda.is_available(): torch.cuda.synchronize()
            st["ms_kernel_sum"] += (time.perf_counter() - t0) * 1e3
            st["kernel"] += 1
            _inst(st, self, "kernel")
        except FPFFallback as e:
            _fallback(op, e.reason); _inst(st, self, "stock")
            return out_stock if do_crosscheck else orig(self, *args, **kwargs)
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            if strict: raise
            _fallback(op, f"kernel_exception:{type(e).__name__}")
            _inst(st, self, "stock")
            return out_stock if do_crosscheck else orig(self, *args, **kwargs)
        if do_crosscheck:
            x = _first_tensor(args, kwargs)
            rec = {"call": n_call, "shape": list(x.shape) if x is not None else None, "dtype": str(x.dtype) if x is not None else None,
                   "max_abs_d": _maxabs(out, out_stock), "bitwise": _bitwise(out, out_stock)}
            rec.update(_err_context(out, out_stock))
            rec["instance"] = getattr(self, "_fpf_path", None) or f"dim{_module_dim(self)}"   # max|stock|, bf16 ulp at that magnitude, max relative |d|: classifies a large max|d| as 1-ulp-at-large-magnitude or a real error
            st["cross-check"].append(rec)
            if crosscheck_ref == "stock": return out_stock
        return out
    wrapped._fpf_orig = orig; wrapped._fpf_op = op
    return wrapped


def enable(ops: dict, engine: str | None = None, crosscheck: int = 0, crosscheck_ref: str = "kernel", strict: bool = False) -> dict:
    """ops: {op_name: callable(module, *args, **kwargs)}.  Monkeypatches the engine's stock class method.  Returns report()."""
    engine = engine or detect_engine()
    if engine not in REGISTRY: raise ValueError(f"unknown engine {engine!r}; known: {sorted(REGISTRY)}")
    reg = REGISTRY[engine]
    with _LOCK:
        for op, fn in ops.items():
            if op not in reg: raise KeyError(f"{engine} has no op {op!r}; ops: {sorted(reg)}")
            modname, clsname, attr, _sig = reg[op]
            cls = getattr(importlib.import_module(modname), clsname)
            key = (engine, op)
            if key not in _ORIG: _ORIG[key] = (cls, getattr(cls, attr))
            orig = _ORIG[key][1]
            setattr(cls, attr, _make_wrapper(op, fn, orig, int(crosscheck), crosscheck_ref, bool(strict)))
        _CFG.update(engine=engine, ops=sorted(set(_CFG["ops"]) | set(ops)), crosscheck=int(crosscheck), crosscheck_ref=crosscheck_ref, strict=bool(strict))
    return report()


def disable(names=None):
    with _LOCK:
        for (engine, op), (cls, orig) in list(_ORIG.items()):
            if names is None or op in names:
                attr = REGISTRY[engine][op][2]
                setattr(cls, attr, orig); del _ORIG[(engine, op)]
        _CFG["ops"] = [o for o in _CFG["ops"] if names is not None and o not in names]


def original(op: str, engine: str | None = None):
    """The stock (unpatched) method, for fallback inside a kernel callable: fpf_engines.original('transition')(module, x)."""
    engine = engine or _CFG["engine"] or detect_engine()
    if (engine, op) in _ORIG: return _ORIG[(engine, op)][1]
    modname, clsname, attr, _ = REGISTRY[engine][op]
    return getattr(getattr(importlib.import_module(modname), clsname), attr)


def is_stock(op: str, engine: str | None = None) -> bool:
    """True iff the class method is the stock function object (nothing patched)."""
    engine = engine or _CFG["engine"] or detect_engine()
    modname, clsname, attr, _ = REGISTRY[engine][op]
    m = getattr(getattr(importlib.import_module(modname), clsname), attr)
    return not hasattr(m, "_fpf_orig")


def report() -> dict:
    return {"version": __version__, "engine": _CFG["engine"], "enabled": sorted(op for (_e, op) in _ORIG), "cfg": dict(_CFG),
            "stats": {op: {k: (v if k != "cross-check" else {"n": len(v), "max_abs_d": max([r["max_abs_d"] for r in v], default=None),
                                                         "all_bitwise": all(r["bitwise"] for r in v) if v else None})
                           for k, v in s.items()} for op, s in STATS.items()}}


def _load_callable(spec: str):
    modpath, attr = spec.split(":")
    return getattr(importlib.import_module(modpath.strip()), attr.strip())


def enable_from_env() -> dict | None:
    """FPF_OPS=trimul_out,trimul_in FPF_IMPL=trimul_out=pkg.mod:fn,trimul_in=pkg.mod:fn2 [FPF_CROSSCHECK=k] [FPF_STRICT=1] [FPF_ENGINE=...]"""
    ops = [o.strip() for o in os.environ.get("FPF_OPS", "").split(",") if o.strip()]
    if not ops: return None
    _DIM_GUARD.clear()
    for item in os.environ.get("FPF_DIMS", "").split(";"):
        if "=" in item: o, ds = item.split("="); _DIM_GUARD[o.strip()] = {int(x) for x in ds.split(",") if x.strip()}
    impl = {}
    for item in os.environ.get("FPF_IMPL", "").split(","):
        if not item.strip(): continue
        name, target = item.split("="); impl[name.strip()] = target.strip()
    missing = [o for o in ops if o not in impl]
    if missing: raise RuntimeError(f"FPF_OPS lists {missing} without an FPF_IMPL entry")
    fns = {}
    for o in ops:
        try:
            fns[o] = _load_callable(impl[o])
        except Exception as e:  # noqa: BLE001
            if is_oom(e): raise
            _fallback(o, f"impl_import_failed:{type(e).__name__}")
            if os.environ.get("FPF_STRICT", "0") == "1": raise
            print(f"[fpf_engines] impl import failed for {o}: {e!r} -> stock", file=sys.stderr)
    if not fns: return report()
    return enable(fns, engine=os.environ.get("FPF_ENGINE") or None, crosscheck=int(os.environ.get("FPF_CROSSCHECK", "0") or 0),
                  crosscheck_ref=os.environ.get("FPF_CROSSCHECK_REF", "kernel"), strict=os.environ.get("FPF_STRICT", "0") == "1")


def dims(model, engine: str | None = None) -> dict:
    """Resolve every registry dim from a BUILT engine model (authoritative; never hardcode)."""
    engine = engine or detect_engine()
    out = {"engine": engine}
    if engine == "opendde":
        pb = model.pairformer_stack.blocks[0]
        out["pairformer"] = {"n_blocks": len(model.pairformer_stack.blocks), "c_z": pb.tri_mul_out.c_z, "trimul_c_hidden": pb.tri_mul_out.c_hidden,
                             "triatt_heads": getattr(pb.tri_att_start.mha, "num_heads", None), "triatt_c_hidden": pb.tri_att_start.mha.c_hidden,
                             "pair_transition": {"c_in": pb.pair_transition.c_in, "n": pb.pair_transition.n},
                             "c_s": pb.c_s, "apb_heads": getattr(pb.attention_pair_bias, "n_heads", None) if pb.c_s > 0 else None,
                             "single_transition": {"c_in": pb.single_transition.c_in, "n": pb.single_transition.n} if pb.c_s > 0 else None}
        mm = model.msa_module; mb = mm.blocks[0]
        out["msa_module"] = {"n_blocks": len(mm.blocks), "opm": {"c_m": mb.outer_product_mean_msa.c_m, "c_hidden": mb.outer_product_mean_msa.c_hidden, "c_z": mb.outer_product_mean_msa.c_z},
                             "pwa": None if getattr(mb, "is_last_block", False) else {"c": mb.msa_stack.msa_pair_weighted_averaging.c, "heads": mb.msa_stack.msa_pair_weighted_averaging.n_heads},
                             "pair_stack": {"c_z": mb.pair_stack.tri_mul_out.c_z, "trimul_c_hidden": mb.pair_stack.tri_mul_out.c_hidden}}
    return out
