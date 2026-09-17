"""ptx_msa_adapt.adapters — STOCK-SIGNATURE wrappers for the FPF_SPEC_v0 registry ops of the Protenix-v2 MSA module.

    fpf.enable(ops=ptx_msa_adapt.ops_for_fpf())          # or FPF_OPS="outer_product_mean=ptx_msa_adapt.adapters:opm,..."

Each wrapper has EXACTLY the stock method signature (module first):
    opm(module, m, mask=None, chunk_size=None, inplace_safe=False)          # OuterProductMean.forward
    pwa(module, m, z)                                                       # MSAPairWeightedAveraging.forward
    transition_msa(module, x)                                               # primitives.Transition.forward (routes ONLY instances with c_in == msa c_m
                                                                            #  that live inside an MSAStack; every other Transition -> stock/original)
Backends: a backend is a plain function with the canonical signature of ptx_msa_adapt.api (explicit weights). Register with
    set_backend("opm" | "pwa" | "transition", fn)   or env PTX_MSA_OPM / PTX_MSA_PWA / PTX_MSA_TRANS = "pkg.mod:fn" (resolved lazily at first call).
Default backend = ptx_msa_adapt.torch_ref.*_stockpath (bitwise replica of stock) so enabling the adapters alone is an exactness op test.

Envelope (outside it the wrapper calls the stock forward, so semantics never change):
    eval mode only (module.training False); bf16-or-fp32 CUDA tensors; OPM: mask is None and chunk_size is None (the stock CLI: chunk_size=None below the
    dynamic-chunk threshold ~2000 tokens; masks are None in MSABlock); fp16 autocast branch -> stock; leading batch dims -> stock.
Weights: read from the module once and cached on module._fpf_cache as DETACHED references (no copies, no dtype change: backends receive the fp32
    Parameters' .data and must do their own casting exactly as documented in ptx_msa_adapt.api); cache invalidated if a parameter's data_ptr changes.
"""
import os, importlib, threading
import torch
from . import torch_ref as TR

_BACKENDS = {"opm": None, "pwa": None, "transition": None}
_ENVKEYS = {"opm": "PTX_MSA_OPM", "pwa": "PTX_MSA_PWA", "transition": "PTX_MSA_TRANS"}
_RESOLVED_ENV = set()
_LOCK = threading.Lock()
STATS = {"opm": {"calls": 0, "backend": 0, "stock_fallback": 0}, "pwa": {"calls": 0, "backend": 0, "stock_fallback": 0},
         "transition": {"calls": 0, "backend": 0, "stock_fallback": 0, "not_msa": 0}}
FALLBACK_REASONS = {}


def set_backend(op, fn):
    assert op in _BACKENDS, op
    _BACKENDS[op] = fn


def get_backend(op):
    """returns the registered backend callable for op, resolving FPF_MSA_* env once; None -> default stock-path torch reference."""
    if _BACKENDS[op] is None and op not in _RESOLVED_ENV:
        with _LOCK:
            _RESOLVED_ENV.add(op)
            spec = os.environ.get(_ENVKEYS[op], "").strip()
            if spec:
                modpath, attr = spec.split(":")
                _BACKENDS[op] = getattr(importlib.import_module(modpath), attr)
    return _BACKENDS[op]


def backends_report():
    return {op: (f"{fn.__module__}:{getattr(fn, '__name__', '?')}" if fn else "default:torch_ref stockpath") for op, fn in _BACKENDS.items()} | {"stats": STATS, "fallback_reasons": dict(FALLBACK_REASONS)}


def _fallback(op, why):
    STATS[op]["stock_fallback"] += 1
    FALLBACK_REASONS[(op, why)] = FALLBACK_REASONS.get((op, why), 0) + 1


def _stock(name, module=None):
    """the stock (unpatched) forward: fpf.original(name) if the fpf adapter is active; else the module class's forward
    (unwrapping our own patch via ._fpf_orig if present)."""
    try:
        import fpf
        return fpf.original(name)
    except Exception:
        pass
    if module is not None:
        f = type(module).forward
        return getattr(f, "_fpf_orig", f)
    modname, clsname = {"outer_product_mean": ("protenix.model.triangular.layers", "OuterProductMean"),
                        "msa_pair_weighted_avg": ("protenix.model.modules.pairformer", "MSAPairWeightedAveraging"),
                        "transition": ("protenix.model.modules.primitives", "Transition")}[name]
    f = getattr(getattr(importlib.import_module(modname), clsname), "forward")
    return getattr(f, "_fpf_orig", f)


def _cache(module, build):
    """per-instance weight cache keyed on parameter data_ptrs (detached views; rebuilt if the module's parameters were replaced/moved)."""
    key = tuple(p.data_ptr() for p in module.parameters())
    c = getattr(module, "_fpf_cache", None)
    if c is None or c.get("_key") != key:
        c = build(module); c["_key"] = key
        module._fpf_cache = c
    return c


def _common_envelope_ok(op, module, *tensors):
    if module.training:
        _fallback(op, "training"); return False
    for t in tensors:
        if not (torch.is_tensor(t) and t.is_cuda):
            _fallback(op, "non-cuda"); return False
        if t.dtype not in (torch.bfloat16, torch.float32):
            _fallback(op, f"dtype {t.dtype}"); return False
    if torch.is_autocast_enabled() and torch.get_autocast_dtype("cuda") == torch.float16:
        _fallback(op, "fp16 autocast"); return False
    return True


# ============================================================================================================ OPM
def _opm_weights(module):
    return {"ln": module.layer_norm, "w_ln": module.layer_norm.weight.detach(), "b_ln": module.layer_norm.bias.detach(),
            "w_a": module.linear_1.weight.detach(), "w_b": module.linear_2.weight.detach(),
            "w_out": module.linear_out.weight.detach(), "b_out": (module.linear_out.bias.detach() if module.linear_out.bias is not None else None),
            "eps_ln": float(getattr(module.layer_norm, "eps", 1e-5)), "eps_opm": float(module.eps), "c_hidden": int(module.c_hidden), "c_z": int(module.c_z), "c_m": int(module.c_m)}


def opm(module, m, mask=None, chunk_size=None, inplace_safe=False):
    """OuterProductMean.forward(m, mask=None, chunk_size=None, inplace_safe=False) -> [N,N,c_z] update."""
    STATS["opm"]["calls"] += 1
    if mask is not None or chunk_size is not None or m.dim() != 3 or not _common_envelope_ok("opm", module, m):
        if mask is not None: _fallback("opm", "mask given")
        if chunk_size is not None: _fallback("opm", f"chunk_size={chunk_size}")
        if m.dim() != 3: _fallback("opm", f"m.dim={m.dim()}")
        return _stock("outer_product_mean", module)(module, m, mask=mask, chunk_size=chunk_size, inplace_safe=inplace_safe)
    W = _cache(module, _opm_weights)
    fn = get_backend("opm")
    if fn is None:
        return TR.opm_stockpath(m, W["ln"], W["w_a"], W["w_b"], W["w_out"], W["b_out"], eps_opm=W["eps_opm"], inplace_safe=inplace_safe)
    STATS["opm"]["backend"] += 1
    return fn(m, W["w_ln"], W["b_ln"], W["w_a"], W["w_b"], W["w_out"], W["b_out"], eps_ln=W["eps_ln"], eps_opm=W["eps_opm"])


# ============================================================================================================ PWA
def _pwa_weights(module):
    return {"ln_m": module.layernorm_m, "ln_z": module.layernorm_z,
            "w_ln_m": module.layernorm_m.weight.detach(), "b_ln_m": module.layernorm_m.bias.detach(),
            "w_ln_z": module.layernorm_z.weight.detach(), "b_ln_z": module.layernorm_z.bias.detach(),
            "w_v": module.linear_no_bias_mv.weight.detach(), "w_z": module.linear_no_bias_z.weight.detach(),
            "w_g": module.linear_no_bias_mg.weight.detach(), "w_o": module.linear_no_bias_out.weight.detach(),
            "n_heads": int(module.n_heads), "c": int(module.c), "c_m": int(module.c_m), "c_z": int(module.c_z),
            "eps": float(getattr(module.layernorm_m, "eps", 1e-5))}


def pwa(module, m, z):
    """MSAPairWeightedAveraging.forward(m, z) -> [S,N,c_m] update."""
    STATS["pwa"]["calls"] += 1
    if m.dim() != 3 or z.dim() != 3 or not _common_envelope_ok("pwa", module, m, z):
        if m.dim() != 3 or z.dim() != 3: _fallback("pwa", "batched dims")
        return _stock("msa_pair_weighted_avg", module)(module, m, z)
    W = _cache(module, _pwa_weights)
    fn = get_backend("pwa")
    if fn is None:
        out = TR.pwa_stockpath(m, z, W["ln_m"], W["w_v"], W["ln_z"], W["w_z"], W["w_g"], W["w_o"], n_heads=W["n_heads"], c=W["c"])
    else:
        STATS["pwa"]["backend"] += 1
        out = fn(m, z, W["w_ln_m"], W["b_ln_m"], W["w_v"], W["w_ln_z"], W["b_ln_z"], W["w_z"], W["w_g"], W["w_o"], n_heads=W["n_heads"], c=W["c"], eps=W["eps"])
    if m.shape[-3] > 5120:
        pass   # stock deletes its temporaries here; nothing to do
    return out


# ============================================================================================================ Transition (MSA transition_m only)
_MSA_TRANSITION_IDS = None


def mark_msa_transitions(model):
    """Call once after model build: records id() of every MSAStack.transition_m so transition_msa() routes only those (c_in alone is ambiguous if
    c_m == some other transition width). If never called, the router falls back to `module.c_in == PTX_MSA_CM` (env, default 128) AND module.n == 4
    AND input is 3-D [S,N,c] — adequate for protenix-v2 where no other Transition has c_in=128."""
    global _MSA_TRANSITION_IDS
    import protenix.model.modules.pairformer as PF
    ids = set()
    for mod in model.modules():
        if isinstance(mod, PF.MSAStack):
            ids.add(id(mod.transition_m))
    _MSA_TRANSITION_IDS = ids
    return len(ids)


def _is_msa_transition(module, x):
    if _MSA_TRANSITION_IDS is not None:
        return id(module) in _MSA_TRANSITION_IDS
    return int(module.c_in) == int(os.environ.get("PTX_MSA_CM", "128")) and int(module.n) == 4 and x.dim() == 3


def _tr_weights(module):
    return {"ln": module.layernorm1, "w_ln": module.layernorm1.weight.detach(), "b_ln": module.layernorm1.bias.detach(),
            "w_a": module.linear_no_bias_a.weight.detach(), "w_b": module.linear_no_bias_b.weight.detach(), "w_out": module.linear_no_bias.weight.detach(),
            "c_in": int(module.c_in), "n": int(module.n), "eps": float(getattr(module.layernorm1, "eps", 1e-5))}


def transition_msa(module, x):
    """primitives.Transition.forward(x) — MSA transition_m instances go to the 'transition' backend; all other Transition instances -> stock."""
    STATS["transition"]["calls"] += 1
    if not _is_msa_transition(module, x):
        STATS["transition"]["not_msa"] += 1
        return _stock("transition", module)(module, x)
    if not _common_envelope_ok("transition", module, x):
        return _stock("transition", module)(module, x)
    W = _cache(module, _tr_weights)
    fn = get_backend("transition")
    if fn is None:
        return TR.transition_stockpath(x, W["ln"], W["w_a"], W["w_b"], W["w_out"], c_in=W["c_in"])
    STATS["transition"]["backend"] += 1
    return fn(x, W["w_ln"], W["b_ln"], W["w_a"], W["w_b"], W["w_out"], eps=W["eps"])


def ops_for_fpf(include_transition=True):
    """dict for fpf.enable(ops=...)"""
    d = {"outer_product_mean": opm, "msa_pair_weighted_avg": pwa}
    if include_transition:
        d["transition"] = transition_msa
    return d
