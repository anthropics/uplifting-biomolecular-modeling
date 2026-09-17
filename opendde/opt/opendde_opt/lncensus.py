"""The kernel census of this tree — ONE reader for every upstream accelerator the pinned OpenDDE can engage: the LayerNorm backend
(`LAYERNORM` line, below) and the cuEquivariance triangle kernels (`KERNELS` line, second half of this module), read from the objects
upstream ACTUALLY bound in this process plus the accelerator's own runtime signals, with the REQUIRE guard that refuses a timing run route
whose expected accelerator is absent or fell back (exit 5, `EXIT_KERNELS`).

Part 1 — the LayerNorm backend census: which LayerNorm implementation upstream bound, read from upstream's own
modules (never from the environment variable), printed once as

  [opendde-opt] LAYERNORM requested=<LAYERNORM_TYPE> backend=fast_layer_norm_cuda_v2|torch|unconfirmed [module=<path of the loaded extension>] [reason=<why>]

(`unconfirmed` only when upstream carries no `opendde.model.triangular.layers` module at all — an upstream layout the version gate refuses first)

Upstream selects the implementation at import (`opendde/model/triangular/layers.py`: `LAYERNORM_TYPE=fast_layernorm` -> `FusedLayerNorm`) and
loads — compiling at first use with ninja + nvcc into `$TORCH_EXTENSIONS_DIR` — the CUDA extension `fast_layer_norm_cuda_v2`
(`opendde/model/layer_norm/layer_norm.py`); when the extension cannot be imported or built, upstream logs a warning and computes LayerNorm with
torch. When the caller asks for `fast_layernorm` (`LAYERNORM_TYPE` in the environment, settings.ln_requested), a process whose backend is not the
extension is not running what was asked: `census(strict=True)` raises `NotLoaded` (the callers exit non-zero by name) instead of running on the
other implementation. The census forces upstream's own lazy load (the call upstream makes at its first LayerNorm forward; idempotent), so it
is taken before any weights load or forward pass. Call sites: `cli.cmd_pred` (kit modes, in-process), `stock_pred` (`--mode off`, the clean
stock process).
"""
from __future__ import annotations

import os
import sys

PREFIX = "[opendde-opt]"
EXTENSION = "fast_layer_norm_cuda_v2"                 # upstream's extension module name (layer_norm.py)
LAYERS = "opendde.model.triangular.layers"            # upstream: `_use_fast_layer_norm` decided at import
LOADER = "opendde.model.layer_norm.layer_norm"        # upstream: `_load_fast_layer_norm_cuda_v2()` -> the extension or None


class NotLoaded(RuntimeError):
    """`LAYERNORM_TYPE=fast_layernorm` was requested and upstream is not bound to the extension."""


def take() -> dict:
    """{requested, backend, module, reason} from upstream's modules in this process (imports them; forces the extension's lazy load)."""
    import importlib
    requested = os.environ.get("LAYERNORM_TYPE", "torch")
    out = {"requested": requested, "backend": "torch", "module": None, "reason": None}
    try:
        layers = importlib.import_module(LAYERS)
    except ModuleNotFoundError as e:                      # an upstream without the selection module (not the pin's layout; the version gate owns upstream's equality):
        out["backend"] = "unconfirmed"                     # nothing to read the backend from — named, never claimed either way
        out["reason"] = f"absent:{e.name}"
        return out
    except Exception as e:  # noqa: BLE001 — the module exists and fails to import: upstream's own error, named; the caller's import fails next with upstream's words
        out["reason"] = f"upstream_import:{type(e).__name__}:{e}"
        return out
    if not getattr(layers, "_use_fast_layer_norm", False):
        out["reason"] = ("not_requested" if requested != "fast_layernorm"
                         else "FusedLayerNorm import failed at upstream import (opendde/model/triangular/layers.py:26-30), or LAYERNORM_TYPE was exported after upstream was imported")
        return out
    try:
        import time as _time
        loader = importlib.import_module(LOADER)
        _t0 = _time.perf_counter()
        ext = loader._load_fast_layer_norm_cuda_v2()      # upstream's own lazy load: import the built module, else compile it (ninja + nvcc), else None with upstream's warning
        out["jit_s"] = round(_time.perf_counter() - _t0, 3)   # the load / JIT-build seconds, spent HERE (before any timed item) and named on the line
    except Exception as e:  # noqa: BLE001 — the loader raising is upstream's error; named here, re-raised by nobody: backend stays torch and strict callers refuse by name
        out["reason"] = f"load:{type(e).__name__}:{e}"
        return out
    if ext is None:
        cuda = _cuda_word()
        out["reason"] = f"extension {EXTENSION} unavailable ({cuda}; upstream logged the build/load error above; the build needs ninja + nvcc: pip install ninja)"
        return out
    out["backend"] = EXTENSION
    out["module"] = getattr(ext, "__file__", None) or EXTENSION
    return out


def _cuda_word() -> str:
    try:
        import torch
        return "cuda available" if torch.cuda.is_available() else "no CUDA device in this process — upstream loads the extension only on CUDA"
    except Exception:  # noqa: BLE001
        return "torch unavailable"


def line(c: dict) -> str:
    s = f"{PREFIX} LAYERNORM requested={c['requested']} backend={c['backend']}"
    if c.get("module"):
        s += f" module={c['module']}"
    if c.get("reason"):
        s += f" reason={c['reason']!r}"
    if c.get("jit_s") is not None:
        s += f" jit_s={c['jit_s']}"                     # the extension's load / JIT-build seconds, spent at the census (before any timed item)
    return s


def census(strict: bool = True, stream=None) -> dict:
    """Take the census, print its line (stderr by default), and under ``strict`` raise ``NotLoaded`` when fast_layernorm was requested and the
    extension is not the backend — never a silent run on the other implementation."""
    c = take()
    print(line(c), file=stream or sys.stderr, flush=True)
    if strict and c["requested"] == "fast_layernorm" and c["backend"] not in (EXTENSION, "unconfirmed"):
        raise NotLoaded(f"LAYERNORM_TYPE=fast_layernorm requested but not loaded: {c['reason']}; install ninja (pip install ninja) and a CUDA toolkit (nvcc) "
                        f"so upstream can build {EXTENSION}, or run on a stack where it is built ($TORCH_EXTENSIONS_DIR)")
    return c


# =====================================================================================================================================
# Part 2 — the triangle-kernel census: the `KERNELS` line and the REQUIRE guard
#
#   [opendde-opt <mode>] KERNELS route=<stock|exact|fast|big|…[_x<P>]> fast_layernorm=<word> cueq_triatt=<word>
#                         cueq_trimul=<word> others=n/a-upstream:<names>(…) cueq_dist=<dists> resolved=<mult>,<att> device=<sm> probe_s=<f>
#
# ONE line per pass (per model process; rank 0 of a multi-GPU pass), printed when the pass ends — or at the moment the guard refuses.
# Words: engaged:<impl>@<ver>[-<build>][served=<k>,byrule=<m>(<reasons>)[;replaced=<levers>]] | off-by-route:<reason>[served=0,…] |
# absent:<why> | fallback:<why> | n/a-upstream:<why>. `off-by-route` means the library served NO call on the route (a lever owns the whole site);
# a lever that replaces the site only inside its own scope / contract while the library serves the rest reads `engaged:…[…;replaced=<levers>]`.
#
# What is read, per accelerator (never distribution metadata alone, never find_spec alone):
#   fast_layernorm  Part 1's census (upstream's `_use_fast_layer_norm` flag + the extension object its loader returns).
#   cueq_triatt     the function object upstream's call site binds at call time — `cuequivariance_torch.primitives.triangle.triangle_attention`
#                   (stock/src/opendde/model/triangular/layers.py:486-489 imports it inside `cuequivariance_triangular_attn`) — the ops package
#                   it dispatches to (`cuequivariance_ops_torch`: `__version__`, `__git_commit__`, `has_sm100f_support()`), the device's compute
#                   capability, a functional probe of the kernel on the device BEFORE the model runs, upstream's RESOLVED kernel selection
#                   (`InferenceRunner.configs.triangle_attention` after `apply_runtime_compatibility`, config/inference.py:198-262 — what `auto`
#                   became), and per-call counters: `served` = calls the library ran on its CUDA kernel, `byrule` = calls the library routed to its
#                   torch reference path BY ITS OWN RULES (cuequivariance_ops_torch/triangle_attention.py:683-736: S_qo <= CUEQ_TRIATTN_FALLBACK_THRESHOLD
#                   (100; 200 when hidden_dim < 32) or an unsupported hidden_dim/dtype) — admissible, counted with reasons, never refused.
#   cueq_trimul     likewise: `…triangle.triangle_multiplicative_update` (triangular.py:31-32), reference path `_tri_mul_torch` taken when
#                   N <= CUEQ_TRIMUL_FALLBACK_THRESHOLD (100) (cuequivariance_ops_torch/triangle_multiplicative_update.py:352-373).
#   others          DeepSpeed DS4Sci / CUTLASS, flash-attn, xformers, trifast, transformer_engine, apex, tokamax: n/a-upstream — the pinned
#                   OpenDDE references none of them (0 occurrences in stock/src), so there is nothing to engage.
# Upstream's own pre-call policies — attention through cuEquivariance only for CUDA queries longer than 16 (layers.py:457-461), the
# multiplicative update only when c_z == c_hidden (triangular.py:483-487) — route a call to upstream's torch implementation before the library
# is reached; they are size/shape policies of the model, not fallbacks, and not counted here (such a call never reaches the reader).
#
# The REQUIRE guard (`require`): the expected word-set of the route comes from the modes table (modes.kernel_expectations: which lever of the
# line owns which site, whether the environment requests the fused LayerNorm). It
# REFUSES — one `KERNELS` line carrying the offending word, then exit 5 — when an accelerator the route expects is absent (module missing /
# import failure), when the functional probe fails (no kernel for this device / wrong ops build), when upstream resolved the kernel OFF where the
# table expects it ON (auto -> torch: config/inference.py:162-182) or ON where the route is defined without it, and — at the end of the pass —
# when the library took its reference path for a call its rules say the kernel serves (`violations`). Routes: every `pred` route, the clean stock
# process included (stock_pred arms the same reader before the upstream CLI runs; the resolved selection is read by a wrap of
# `InferenceRunner.__init__` armed through phase.py's one meta-path hook).
# =====================================================================================================================================
import functools as _functools

EXIT_KERNELS = 5                                                     # the guard's exit status (cli / stock_pred map it through unchanged)
KERNEL_SITES = ("cueq_triatt", "cueq_trimul")                        # the upstream accelerator sites a kit lever may own (registry.Lever.owns)
ACCELERATORS = ("fast_layernorm",) + KERNEL_SITES                    # the words of the KERNELS line, in order
FRONT = "cuequivariance_torch.primitives.triangle"                   # the module upstream's two call sites import from at call time
FRONT_FUNCS = {"cueq_triatt": "triangle_attention", "cueq_trimul": "triangle_multiplicative_update"}
OPS = "cuequivariance_ops_torch"
OPS_REF = {                                                          # site -> (ops submodule, its torch-reference function, its threshold constant)
    "cueq_triatt": ("cuequivariance_ops_torch.triangle_attention", "_triangle_attention_torch", "CUEQ_TRIATTN_FALLBACK_THRESHOLD"),
    "cueq_trimul": ("cuequivariance_ops_torch.triangle_multiplicative_update", "_tri_mul_torch", "CUEQ_TRIMUL_FALLBACK_THRESHOLD"),
}
FRONT_DIST, OPS_DIST_PREFIX = "cuequivariance-torch", "cuequivariance-ops-torch"
RUNNER_CLASS = "InferenceRunner"
RESOLVED_ATTRS = {"cueq_triatt": "triangle_attention", "cueq_trimul": "triangle_multiplicative"}   # OpenDDEConfig fields (config/schema.py:330-331)
NA_UPSTREAM = ("deepspeed_ds4sci", "cutlass", "flash_attn", "xformers", "trifast", "transformer_engine", "apex", "tokamax")
NA_REASON = "0_references_in_opendde_{ver}"
MARK = "_opendde_opt_kcensus"


class KernelsRefused(SystemExit):
    """``SystemExit(5)``: the REQUIRE guard refused the route; ``.line`` is the KERNELS line printed, ``.problems`` the named reasons."""

    def __init__(self, line: str, problems: list):
        super().__init__(EXIT_KERNELS)
        self.line, self.problems = line, list(problems)


def _new_counts() -> dict:
    return {"front": 0, "reference": 0, "byrule": {}, "violations": 0, "violation_examples": [], "errors": 0}


K: dict = {                                                          # the census state of this process (one model process = one pass)
    "armed": False, "strict": True, "tag": None, "route": None, "expected": {}, "stream": None,
    "presence": {}, "probe": {}, "resolved": None, "device": None, "counts": {s: _new_counts() for s in KERNEL_SITES},
    "layernorm": None, "line_printed": False, "printed_line": None, "problems": [], "counters_installed": False, "unconfirmed": None,
}


def reset() -> None:
    """Forget everything (tests; a second arm in one interpreter)."""
    K.update({"armed": False, "strict": True, "tag": None, "route": None, "expected": {}, "stream": None, "presence": {},
              "probe": {}, "resolved": None, "device": None, "counts": {s: _new_counts() for s in KERNEL_SITES}, "layernorm": None,
              "line_printed": False, "printed_line": None, "problems": [], "counters_installed": False, "unconfirmed": None})


# ---------------------------------------------------------------------------------------------------------------- presence (bound objects + dists)
def _dist_version(name: str):
    try:
        import importlib.metadata as md
        return md.version(name)
    except Exception:  # noqa: BLE001
        return None


def _ops_dist():
    """(distribution name, version) of the installed cuequivariance-ops-torch-* wheel — its name carries the CUDA-major build tag (-cu12 / -cu13)."""
    try:
        import importlib.metadata as md
        for d in md.distributions():
            n = (d.metadata.get("Name") or "").lower()
            if n.startswith(OPS_DIST_PREFIX):
                return n, d.version
    except Exception:  # noqa: BLE001
        pass
    return None, None


def presence() -> dict:
    """Import what upstream imports, the way upstream imports it, and record what is bound: per site {ok, func (qualified name), file, error},
    plus the library facts {front_version, ops_dist, ops_version, ops_commit, sm100f, device_cc, opendde_version}. Never raises."""
    import importlib
    out = {"sites": {}, "front_version": _dist_version(FRONT_DIST), "opendde_version": _dist_version("opendde")}
    name, ver = _ops_dist()
    out["ops_dist"], out["ops_dist_version"] = name, ver
    out["ops_build"] = name[len(OPS_DIST_PREFIX):].lstrip("-") if name else None          # "cu12" | "cu13" | ""
    try:
        ops = importlib.import_module(OPS)
        out["ops_version"] = getattr(ops, "__version__", None) or ver
        out["ops_commit"] = getattr(ops, "__git_commit__", None)
        try:
            out["sm100f"] = bool(importlib.import_module(OPS + "._ext").has_sm100f_support())
        except Exception:  # noqa: BLE001
            out["sm100f"] = None
        out["ops_error"] = None
    except Exception as e:  # noqa: BLE001
        out["ops_version"], out["ops_commit"], out["sm100f"], out["ops_error"] = None, None, None, f"{type(e).__name__}: {str(e)[:200]}"
    for site, fn in FRONT_FUNCS.items():
        rec = {"ok": False, "func": None, "file": None, "error": None}
        try:
            mod = importlib.import_module(FRONT)                                        # == `from cuequivariance_torch.primitives.triangle import …`
            f = getattr(mod, fn)
            real = getattr(f, "_orig", f) if getattr(f, MARK, None) else f
            rec.update(ok=out["ops_error"] is None, func=f"{FRONT}.{fn}", file=getattr(getattr(real, "__code__", None), "co_filename", None),
                       error=None if out["ops_error"] is None else f"{OPS}: {out['ops_error']}")
            sub, ref, thr = OPS_REF[site]
            m = importlib.import_module(sub)                                            # the ops submodule that holds the reference function and threshold
            rec["threshold"] = getattr(m, thr, None)
            if not callable(getattr(m, ref, None)):
                rec.update(ok=False, error=f"{sub}.{ref} absent (the pinned library layout changed)")
        except Exception as e:  # noqa: BLE001
            rec.update(ok=False, error=f"{type(e).__name__}: {str(e)[:200]}")
        out["sites"][site] = rec
    try:
        import torch
        out["device_cc"] = list(torch.cuda.get_device_capability()) if torch.cuda.is_available() else None
        out["torch"] = torch.__version__
    except Exception:  # noqa: BLE001
        out["device_cc"], out["torch"] = None, None
    return out


# ---------------------------------------------------------------------------------------------------------------- per-call counters
def _byrule_reason(site: str, args: tuple, kwargs: dict):
    """Why the library's own rules send THIS reference-path call to torch (the rule replayed on the call's arguments), or None when the rules say
    the kernel serves it — then the reference path is a violation. attention: cuequivariance_ops_torch/triangle_attention.py:683-736; trimul:
    triangle_multiplicative_update.py:352-373."""
    import sys as _sys
    sub, _ref, thr_name = OPS_REF[site]
    m = _sys.modules.get(sub)
    thr = getattr(m, thr_name, 100) if m is not None else 100
    x = args[0] if args else kwargs.get("q" if site == "cueq_triatt" else "x")
    try:
        if site == "cueq_trimul":
            n = int(x.shape[-2])
            return f"N<={thr}" if n <= thr else None
        s, d = int(x.size(3)), int(x.shape[-1])
        dt = str(x.dtype).replace("torch.", "")
        if dt in ("float16", "bfloat16"):
            if d > 128 or d % 8:                         # (sm100f raises the bound to 256; not this reader's device class — a served call never reaches here anyway)
                return f"dims({dt},hidden={d})"
        elif dt == "float32":
            if d > 32 or d % 4:
                return f"dims({dt},hidden={d})"
        else:
            return f"dtype({dt})"
        if d < 32:
            thr = max(thr, 200)
        return f"S<={thr}" if s <= thr else None
    except Exception:  # noqa: BLE001 — an argument this replay cannot read is NOT explained by the rules: a violation (fail-closed), refused at exit
        return None


def _wrap_front(site: str, orig):
    @_functools.wraps(orig)
    def counted(*args, **kwargs):
        c = K["counts"][site]
        c["front"] += 1
        try:
            return orig(*args, **kwargs)
        except Exception:
            c["errors"] += 1                              # the library raised: counted and RE-RAISED (upstream does not swallow it: the item fails loud)
            raise
    setattr(counted, MARK, "front"); counted._orig = orig
    return counted


def _wrap_reference(site: str, orig):
    @_functools.wraps(orig)
    def counted(*args, **kwargs):
        c = K["counts"][site]
        c["reference"] += 1
        why = _byrule_reason(site, args, kwargs)
        if why is None:
            c["violations"] += 1
            if len(c["violation_examples"]) < 3:
                try:
                    x = args[0]
                    c["violation_examples"].append(f"shape={tuple(x.shape)},dtype={str(x.dtype).replace('torch.', '')}")
                except Exception:  # noqa: BLE001
                    c["violation_examples"].append("unreadable")
        else:
            c["byrule"][why] = c["byrule"].get(why, 0) + 1
        return orig(*args, **kwargs)
    setattr(counted, MARK, "reference"); counted._orig = orig
    return counted


def install_counters() -> dict:
    """Wrap, in place and idempotently, the two front functions upstream binds at call time and the two torch-reference functions inside the
    ops library. Returns {site: 'installed' | 'absent:<why>'}. A kit lever that wraps upstream's OWN site function (the ARM add-on's attention /
    TriMul binds) is untouched: its below-gate delegations still import the front function and are counted."""
    import importlib
    out = {}
    for site, fn in FRONT_FUNCS.items():
        try:
            mod = importlib.import_module(FRONT)
            f = getattr(mod, fn)
            if not getattr(f, MARK, None):
                setattr(mod, fn, _wrap_front(site, f))
            sub, ref, _thr = OPS_REF[site]
            m = importlib.import_module(sub)
            r = getattr(m, ref)
            if not getattr(r, MARK, None):
                setattr(m, ref, _wrap_reference(site, r))
            out[site] = "installed"
        except Exception as e:  # noqa: BLE001
            out[site] = f"absent:{type(e).__name__}: {str(e)[:160]}"
    K["counters_installed"] = True
    return out


# ---------------------------------------------------------------------------------------------------------------- the functional probe
def probe(sites=KERNEL_SITES, device=None) -> dict:
    """Run each site's kernel ONCE on a tiny CUDA input above the library's thresholds (attention: S=128, hidden 32, bf16; multiplicative update:
    N=104, c=64, bf16), through the very front function upstream binds: {site: {ok, served, s, error}}. `served` = the call did NOT take the
    reference path (the counters see the probe; its counts are removed afterwards). No CUDA device -> {ok: False, error: 'no_cuda'}."""
    import importlib
    import time as _time
    out = {}
    try:
        import torch
    except Exception as e:  # noqa: BLE001
        return {s: {"ok": False, "served": False, "s": 0.0, "error": f"torch: {type(e).__name__}"} for s in sites}
    if not torch.cuda.is_available():
        return {s: {"ok": False, "served": False, "s": 0.0, "error": "no_cuda"} for s in sites}
    dev = torch.device(device or "cuda")
    for site in sites:
        snap = {k: (dict(v) if isinstance(v, dict) else (list(v) if isinstance(v, list) else v)) for k, v in K["counts"][site].items()}
        t0 = _time.perf_counter()
        try:
            mod = importlib.import_module(FRONT)
            f = getattr(mod, FRONT_FUNCS[site])
            g = torch.Generator(device="cpu").manual_seed(0)
            with torch.no_grad():
                if site == "cueq_triatt":
                    q = torch.randn(1, 2, 1, 128, 32, generator=g).to(dev, torch.bfloat16)
                    bias = torch.zeros(1, 1, 1, 128, 128, device=dev, dtype=torch.float32)
                    mask = torch.ones(1, 2, 1, 1, 128, device=dev, dtype=torch.bool)
                    y = f(q, q, q, bias, mask=mask, scale=1.0 / (32 ** 0.5))
                else:
                    n, c = 104, 64
                    x = torch.randn(1, n, n, c, generator=g).to(dev, torch.bfloat16)
                    w = lambda *shape: torch.randn(*shape, generator=g).to(dev, torch.bfloat16)  # noqa: E731
                    y = f(x, direction="outgoing", mask=torch.ones(1, n, n, device=dev, dtype=torch.bfloat16),
                          norm_in_weight=w(c), norm_in_bias=w(c), p_in_weight=w(2 * c, c), g_in_weight=w(2 * c, c),
                          norm_out_weight=w(c), norm_out_bias=w(c), p_out_weight=w(c, c), g_out_weight=w(c, c), eps=1e-5)
                y = y[0] if isinstance(y, (tuple, list)) else y
                torch.cuda.synchronize(dev)
                finite = bool(torch.isfinite(y.float()).all().item())
            after = K["counts"][site]
            served = (after["front"] - snap["front"]) >= 1 and (after["reference"] - snap["reference"]) == 0 if K["counters_installed"] else None
            out[site] = {"ok": finite and served is not False, "served": served, "s": round(_time.perf_counter() - t0, 3),
                         "error": None if finite else "nonfinite_output"}
            if served is False and out[site]["error"] is None:
                out[site]["error"] = "reference_path_taken_above_threshold"
        except Exception as e:  # noqa: BLE001
            out[site] = {"ok": False, "served": False, "s": round(_time.perf_counter() - t0, 3), "error": f"{type(e).__name__}: {str(e)[:200]}"}
        K["counts"][site] = snap                                                     # the probe's own calls are not the pass's
    return out


# ---------------------------------------------------------------------------------------------------------------- resolved selection (runner init)
def _on_runner_module(module) -> None:
    """phase.py's hook hands us `runner.inference` once upstream imports it: wrap `InferenceRunner.__init__` so the census reads the RESOLVED
    triangle-kernel selection (what `auto` became on this device) the moment the runner exists — before any weights forward — and the guard runs."""
    cls = getattr(module, RUNNER_CLASS, None)
    if cls is None or getattr(cls.__init__, MARK, None):
        return
    init = cls.__init__

    @_functools.wraps(init)
    def __init__(self, *a, **kw):
        init(self, *a, **kw)
        try:
            cfg = getattr(self, "configs", None)
            K["resolved"] = {site: getattr(cfg, attr, None) for site, attr in RESOLVED_ATTRS.items()}
            K["resolved"]["dtype"] = getattr(cfg, "dtype", None)
            K["device"] = str(getattr(self, "device", None))
        except Exception as e:  # noqa: BLE001
            K["resolved"] = {"error": f"{type(e).__name__}: {e}"}
        enforce("runner_init")
    setattr(__init__, MARK, "init")
    cls.__init__ = __init__


# ---------------------------------------------------------------------------------------------------------------- words, line, guard
def _counts_text(site: str) -> str:
    c = K["counts"][site]
    served = max(c["front"] - c["reference"], 0)
    by = sum(c["byrule"].values())
    reasons = ",".join(f"{k}:{v}" for k, v in sorted(c["byrule"].items()))
    s = f"[served={served},byrule={by}" + (f"({reasons})" if reasons else "") 
    if c["violations"]:
        s += f",violations={c['violations']}"
    if c["errors"]:
        s += f",errors={c['errors']}"
    return s + "]"


def _impl_text(site: str) -> str:
    p = K["presence"] or {}
    cc = p.get("device_cc")
    backend = (f"sm{cc[0]}{cc[1]}" + ("f" if (cc[0] == 10 and p.get("sm100f")) else "")) if cc else "nodevice"
    ver = p.get("ops_version") or p.get("front_version") or "?"
    build = p.get("ops_build")
    return f"{backend}@{ver}" + (f"-{build}" if build else "")


def words() -> dict:
    """accelerator -> word, from the expectations (word head) and the live facts (presence, probe, resolved selection, counters)."""
    out = {}
    exp = K["expected"] or {}
    ln = K["layernorm"] or {}
    e = exp.get("fast_layernorm") or {"kind": "engaged"}
    ver = (K["presence"] or {}).get("opendde_version") or "?"
    if ln.get("backend") == EXTENSION:
        out["fast_layernorm"] = f"engaged:{EXTENSION}@opendde-{ver}" if e["kind"] == "engaged" else f"fallback:engaged_where_route_defines_torch({e.get('reason')})"
    elif e["kind"] == "off-by-route" and ln.get("requested", "torch") != "fast_layernorm":
        out["fast_layernorm"] = f"off-by-route:{e.get('reason')}"
    elif ln.get("backend") == "unconfirmed":
        out["fast_layernorm"] = f"absent:{ln.get('reason')}"
    else:
        out["fast_layernorm"] = "fallback:" + "_".join(str(ln.get("reason") or "not_loaded").split())[:160]
    pres = (K["presence"] or {}).get("sites") or {}
    res = K["resolved"] or {}
    for site in KERNEL_SITES:
        e = exp.get(site) or {"kind": "engaged", "resolved": "cuequivariance", "need_stack": True, "reason": None}
        p = pres.get(site) or {}
        pr = (K["probe"] or {}).get(site)
        counts = _counts_text(site) if K["counters_installed"] else ""
        if e.get("kind") == "user":                                            # upstream's own kernel flag stated for this site (modes.kernel_expectations): the caller's word,
            out[site] = f"user:{e.get('reason')}{counts}"                         # counted, never required and never refused
        elif e.get("need_stack") and not p.get("ok"):
            out[site] = "absent:" + "_".join(str(p.get("error") or "not_importable").split())[:200]
        elif e.get("need_stack") and pr is not None and not pr.get("ok"):
            out[site] = "fallback:probe_failed:" + "_".join(str(pr.get("error")).split())[:200]
        elif res and not res.get("error") and e.get("resolved") and res.get(site) is not None and res.get(site) != e["resolved"]:
            out[site] = f"fallback:resolved_{res.get(site)}(expected_{e['resolved']}:{e.get('reason') or 'modes_table'})"
        elif K["counts"][site]["violations"]:
            out[site] = f"fallback:reference_path_where_rules_say_served{counts}"
        elif e["kind"] == "off-by-route" and K["counts"][site]["front"] > 0:       # the line's lever replaced the site IN PART (its own scope / contract) and the library served
            out[site] = f"engaged:{_impl_text(site)}{counts[:-1]};replaced={e.get('reason')}]"   # the rest: engaged, counted, the lever named — off-by-route means served=0
        elif e["kind"] == "off-by-route":
            out[site] = f"off-by-route:{e.get('reason')}{counts}"
        else:
            out[site] = f"engaged:{_impl_text(site)}{counts}"
    return out


TEST_UNCONFIRMED_ENV = "MODEL_OPT_TEST_KERNELS_UNCONFIRMED"    # the ONE opt-out: a CPU conformance box (the test stubs carry no `opendde.model.triangular.layers`,
                                                             # Part 1's `unconfirmed` key) exports =1 so an unconfirmable census reports instead of refusing — its own token on
                                                             # the line; the prefix MODEL_OPT_TEST_ is must-be-absent on every timing run route (stock/PINS.json)


def unconfirmed_allowed() -> bool:
    return os.environ.get(TEST_UNCONFIRMED_ENV, "") == "1"


def problems_of(w: dict) -> list:
    """The refusal reasons in a word set: every absent / fallback word, accelerator named; an `unconfirmed` census (nothing of the pinned
    upstream's layout in this interpreter) is itself a refusal reason unless the test-only opt-out ``TEST_UNCONFIRMED_ENV`` is exported."""
    probs = [f"{k}={v}" for k, v in w.items() if v.startswith(("absent", "fallback"))]
    if K.get("unconfirmed"):
        return [] if unconfirmed_allowed() else [f"census=unconfirmed({K['unconfirmed']})"] + probs
    return probs


def kernels_line(w: dict | None = None) -> str:
    w = words() if w is None else w
    p = K["presence"] or {}
    tag = K["tag"] or PREFIX
    parts = [f"{tag} KERNELS route={K['route']}"]
    if K.get("unconfirmed"):                                                  # nothing of the pinned upstream to read in this interpreter: the words below claim nothing —
        parts.append(f"census=unconfirmed({K['unconfirmed']}" + (f";allowed_by={TEST_UNCONFIRMED_ENV}=1)" if unconfirmed_allowed() else ")"))   # refused, unless the test-only opt-out says so by name
    parts += [f"{a}={w[a]}" for a in ACCELERATORS]
    parts.append(f"others=n/a-upstream:{','.join(NA_UPSTREAM)}({NA_REASON.format(ver=p.get('opendde_version') or '?')})")
    dist = f"{FRONT_DIST}@{p.get('front_version')},{p.get('ops_dist')}@{p.get('ops_dist_version')}"
    if p.get("ops_commit"):
        dist += f"(g{str(p['ops_commit'])[:7]})"
    parts.append(f"cueq_dist={dist}")
    res = K["resolved"] or {}
    read = res and not res.get("error") and all(res.get(s) is not None for s in KERNEL_SITES)
    parts.append(f"resolved={res.get('cueq_trimul')},{res.get('cueq_triatt')}" if read else f"resolved={'error' if (res or {}).get('error') else 'unread'}")
    thr = ",".join(f"{s.split('_')[1]}:{((p.get('sites') or {}).get(s) or {}).get('threshold')}" for s in KERNEL_SITES)
    parts.append(f"thresholds={thr}")
    if K["probe"]:
        parts.append("probe_s=" + ",".join(f"{s.split('_')[1]}:{(K['probe'].get(s) or {}).get('s')}" for s in KERNEL_SITES if s in K["probe"]))
    parts.append(f"torch={p.get('torch')}")
    return " ".join(parts)


def _emit(line: str) -> None:
    if os.environ.get("RANK", "0") not in ("0", "") and not K["problems"]:      # one line per pass: rank 0 speaks for a multi-GPU pass (a refusing rank speaks for itself)
        K["line_printed"], K["printed_line"] = True, line
        return
    stream = K["stream"] or sys.stderr
    print(line, file=stream, flush=True)
    K["line_printed"], K["printed_line"] = True, line


def enforce(stage: str) -> list:
    """Run the guard now: compute the words; when any is absent/fallback and the census is strict, print THE line and raise KernelsRefused (exit 5).
    Returns the problems (non-strict: printed nothing, raised nothing)."""
    w = words()
    probs = problems_of(w)
    if probs and K["strict"] and K["armed"]:
        K["problems"] = probs
        line = kernels_line(w)
        if not K["line_printed"]:
            _emit(line)
        print(f"{K['tag'] or PREFIX} KERNELS REFUSED route={K['route']} at={stage} exit={EXIT_KERNELS} " + " ".join(probs), file=K["stream"] or sys.stderr, flush=True)
        raise KernelsRefused(line, probs)
    return probs


def arm(route: str, expected: dict, *, tag: str = PREFIX, strict: bool = True, stream=None, layernorm: dict | None = None,
        do_probe: bool | None = None) -> dict:
    """Arm the census for this model process: record the route's expectations (modes.kernel_expectations), read presence, install the counters,
    probe the kernels on the device (when the route expects the library and CUDA is present; `do_probe` forces), subscribe to phase.py's runner
    hook for the resolved selection, and run the guard once now (absent / probe failure refuse HERE, before any weights load). Returns the state."""
    from . import phase as _phase
    ln = layernorm if layernorm is not None else (K["layernorm"] or take())
    reset()                                                                      # one census per model process; a re-arm (tests, an embedded caller) starts clean
    K.update({"armed": True, "strict": strict, "tag": tag, "route": route, "expected": dict(expected or {}), "stream": stream,
              "layernorm": ln, "unconfirmed": (ln.get("reason") if ln.get("backend") == "unconfirmed" else None)})
    K["presence"] = presence()
    install_counters()
    need = any((expected.get(s) or {}).get("need_stack") for s in KERNEL_SITES)
    if do_probe is None:
        do_probe = need and bool((K["presence"] or {}).get("device_cc"))
    if do_probe:
        K["probe"] = probe(tuple(s for s in KERNEL_SITES if (expected.get(s) or {}).get("need_stack")))
    _phase.on_runner_module(_on_runner_module)
    import atexit
    atexit.register(_atexit)                                                     # a pass that dies before its caller reports still prints its one line
    enforce("arm")
    return dict(K)


def _atexit() -> None:
    """Interpreter exit: print THE line if the pass's caller never did (a crash mid-run); a closed stream at teardown is not an event."""
    try:
        if K["armed"] and not K["line_printed"]:
            finish()
    except (ValueError, OSError):
        pass


def finish(rc: int | None = None) -> int:
    """End of the pass: print THE line once (with the final counters) and return the exit status the caller must adopt — EXIT_KERNELS when the
    library took its reference path where its rules say the kernel serves (or any word is absent/fallback), else `rc` (0 when None)."""
    if not K["armed"]:
        return rc or 0
    w = words()
    probs = problems_of(w)
    if not K["line_printed"]:
        _emit(kernels_line(w))
    elif not K["problems"] and K["printed_line"] != kernels_line(w):           # printed early by a refusal-free enforce? never; kept for a re-armed interpreter
        pass
    if probs and K["strict"]:
        if not K["problems"]:
            print(f"{K['tag'] or PREFIX} KERNELS REFUSED route={K['route']} at=exit exit={EXIT_KERNELS} " + " ".join(probs), file=K["stream"] or sys.stderr, flush=True)
        K["problems"] = probs
        return EXIT_KERNELS
    return rc or 0


def record() -> dict:
    """The census as a manifest block: expectations, presence, probe, resolved selection, counters, words, the printed line."""
    return {"route": K["route"], "expected": K["expected"], "presence": K["presence"], "probe": K["probe"],
            "resolved": K["resolved"], "counts": K["counts"], "words": words() if K["armed"] else {}, "line": K["printed_line"], "problems": K["problems"]}
