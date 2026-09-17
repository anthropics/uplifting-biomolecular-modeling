"""The LayerNorm binding (kit-namespaced): the engine's `LayerNorm` primitive (core/model/primitives/normalization.py, ONE class behind every
LayerNorm of the trunk, the MSA / template stacks, the diffusion transformer and the heads) bound to the shared core's LayerNorm PROVIDER
(`opt_core.kernels.ln`: ONE provider over every carried LayerNorm implementation — the NVRTC replica `exactln` (bitwise ATen), the Triton rows
`fastln` / `fastln:lp` / `ln_rows` / `rfd` / `dtk_ln` — and its cell table LN_CELLS.json, keyed per compute capability, operand form, cell
family by row count, token bucket, eager | graph and stack) BY THE LINE'S TIER WORD.  Two levers, one binding:

  exactln      exact class, the `exact` line: the provider is asked with the word `exact` — its cell table names an exact-class row (`exactln`,
               `exactln:widen`) only where the table marks that row bitwise `F.layer_norm` AND at least as fast as ATen on the running card and stack,
               else ATen BY NAME (the statement runs, counted `kept_by=stock:<row>`).  Every exact-class row's first call per signature is proven
               torch.equal against the statement in this process before its result is used (a differing signature retires to the statement, named).
  ln_provider  tolerance class, the `fast` and `big` lines: the provider is asked with the line's word (`fast`: the fastest row inside
               the cell's identity band; `big`: the lowest-peak row among those) — capture-aware (a call met during CUDA-graph capture asks the
               graph cell).  A tolerance row's first call per signature is witnessed against the statement (rel-RMS <= WITNESS_RELRMS, else the
               signature retires to the statement by name); exact-class rows the word names are proven bitwise as above.

What is decided per call, from the live operands only: the statement's fp32 branch and its bf16 branch in the tree's dialect (DIALECTS: `cast16` —
bf16 params, F.layer_norm in bf16; `upcast` — everything .float(), the result cast back: the provider's `bf16o` form, fp32 statistics with a bf16
store), plain [C] affine parameters or none, any width the provider carries, contiguous CUDA operands below 2^31 elements, no autocast inside the
statement's own disabled region.  Everything else runs the statement, counted by name (`kept_by=`): other dtypes / devices / non-contiguous /
autocast-enabled / partial affine / provider refusals (`refused:<kind>`) / cells whose word names ATen (`stock:<row>`) / a signature first met
during capture whose kernel is not yet loaded (`capture_first`) / retired signatures.  Row kernels are made ready OFF the forward where possible:
the NVRTC replica's (width x form) kernels compile and self-prove in a background thread at install (PRECOMPILE_WIDTHS), and an eager call that
decides its class also readies the arm the GRAPH cell of that class names, so a later capture replays it.

Binding: ONE class-wide wrap of `LayerNorm.forward`, installed after the model module executed, outermost over whatever already wraps it (the
castcache lever's memo, whose castcache.cached_cast this binding reuses for the bf16 parameter copies when both serve; the offload port's LN-SAFE
guard on the memory line, reached through its closure).  Standard library only at import; torch and opt_core are imported inside install()."""
import inspect
import math
import os
import sys
import threading
import time
from typing import Any, Dict, Optional, Tuple

ENV: Optional[str] = None                  # <KIT>_EXACTLN=1: the exact-class lever (word `exact` unless <KIT>_EXACTLN_WORD names another)
ENV_WORD: Optional[str] = None             # <KIT>_EXACTLN_WORD=<tier word | row[:variant]>: the caller's explicit word for the exactln lever (ablation)
ENV_PROVIDER: Optional[str] = None         # <KIT>_LN_PROVIDER=1: the tolerance-class lever (the fast / big lines)
ENV_TIER: Optional[str] = None             # <KIT>_LN_TIER=fast|big: the line's tier word (exported by the mode resolver with the lever)
ENV_PROVIDER_WORD: Optional[str] = None    # <KIT>_LN_WORD=<tier word | row[:variant]>: the caller's explicit word for the ln_provider lever (ablation)
VALUES = ("1",)
TIER_VALUES = ("fast", "big")
EXACT_WORD_DEFAULT = "exact"
PROVIDER_WORD_DEFAULT = "fast"             # the ln_provider lever switched on by hand with no <KIT>_LN_TIER
PREFIX = "[exactln]"
PREFIX_PROVIDER = "[ln_provider]"
LEVERS = ("exactln", "ln_provider")
M_NORM = "openfold3.core.model.primitives.normalization"
M_MODEL = "openfold3.projects.of3_all_atom.model"
MODEL_CLASS = "OpenFold3"                  # M_MODEL.<MODEL_CLASS>.forward(batch) reads batch["token_mask"]: the item's token count keys the provider's cells
CASTCACHE_MODULE: Optional[str] = None     # "<kit>_opt.cells.castcache": its cached_cast serves the bf16 parameter copies when that lever is serving too
PRECOMPILE_WIDTHS: Tuple[int, ...] = (64, 128, 256)     # the replica's (width x form) kernels compiled + self-checked in a background thread at install
NUMEL_MAX = 2 ** 31 - 1                    # operands at or above 2^31 elements keep the statement (the LN-SAFE guard's domain on the memory line)
WITNESS_RELRMS = 2e-2                      # a tolerance row's first call per signature: rel-RMS vs the statement above this retires the signature by name
DIALECTS: Dict[str, Tuple[str, ...]] = {   # dialect -> the source needles of the innermost LayerNorm.forward that identify the statements served (tests/test_ln_dialect_source.py checks them against stock/)
    "cast16": ("if d is torch.bfloat16 and not deepspeed_is_initialized:", "weight = self.weight.to(dtype=d) if self.weight is not None else None",
               "bias = self.bias.to(dtype=d) if self.bias is not None else None", "normalized_shape=self.c_in,", "weight=self.weight,", "bias=self.bias,", "eps=self.eps,"),
    "upcast": ("if d in (torch.bfloat16, torch.float16):", "weight = self.weight.float() if self.weight is not None else None",
               "bias = self.bias.float() if self.bias is not None else None", "input=x.float(),", "return out.to(dtype=d)", "normalized_shape=self.c_in,",
               "weight=self.weight,", "bias=self.bias,", "eps=self.eps,"),
}

STATE: Dict[str, Any] = {"installed": False, "state": "off", "reason": None, "lever": None, "dialect": None, "word": None, "tier": None, "nvrtc": None, "torch": None,
                         "torch_pin": None, "cc": None, "stack": None, "stack_listed": None, "has_triton": False, "health": "ok", "precompiled": 0, "precompile_s": None,
                         "precompile_error": None, "served": 0, "kept": 0, "served_by": {}, "kept_by": {}, "proven": [], "witnessed": [], "bits_differ": [], "retired": set(),
                         "ready": set(), "decisions": {}, "cells": {}, "n_tok": None, "compile_s": 0.0, "over": "none"}


def _log(msg: str) -> None:
    sys.stderr.write(f"{STATE.get('prefix') or PREFIX} {msg}\n")


def configure(*, ENV: str, ENV_WORD: str, ENV_PROVIDER: str, ENV_TIER: str, ENV_PROVIDER_WORD: str, PREFIX: str, PREFIX_PROVIDER: str,
              M_NORM: str = M_NORM, M_MODEL: str = M_MODEL, MODEL_CLASS: str = MODEL_CLASS, CASTCACHE_MODULE: Optional[str] = None) -> None:
    g = globals()
    g.update(ENV=ENV, ENV_WORD=ENV_WORD, ENV_PROVIDER=ENV_PROVIDER, ENV_TIER=ENV_TIER, ENV_PROVIDER_WORD=ENV_PROVIDER_WORD, PREFIX=PREFIX, PREFIX_PROVIDER=PREFIX_PROVIDER,
             M_NORM=M_NORM, M_MODEL=M_MODEL, MODEL_CLASS=MODEL_CLASS, CASTCACHE_MODULE=CASTCACHE_MODULE)


def _valid_word(w: str) -> bool:
    from opt_core.kernels import ln as LN
    row, _v = LN.split_word(w)
    return row in LN.TIER_WORDS or (row in LN.ROW_NAMES and row not in LN.BACKWARD_ROWS + ("ln_proj_ln_linear",)) or row in ("aten", "aten_autocast")


def lever(environ=None) -> Optional[str]:
    """Which lever the environment switches on: 'exactln' | 'ln_provider' | None; both at once is a usage error (one binding, one word)."""
    environ = os.environ if environ is None else environ
    ex = bool(ENV and (environ.get(ENV) or "").strip())
    pv = bool(ENV_PROVIDER and (environ.get(ENV_PROVIDER) or "").strip())
    if ex and pv:
        raise ValueError(f"{ENV} and {ENV_PROVIDER} are both set: the LayerNorm binding takes ONE word (the exact line sets {ENV}, the fast / big lines {ENV_PROVIDER})")
    return "exactln" if ex else ("ln_provider" if pv else None)


def requested(environ=None, which: Optional[str] = None) -> bool:
    """True when the named lever (default: either) is switched on; malformed values raise ValueError by name."""
    environ = os.environ if environ is None else environ
    lv = lever(environ)
    if lv is None or (which is not None and lv != which):
        return False
    sw, knob = (ENV, ENV_WORD) if lv == "exactln" else (ENV_PROVIDER, ENV_PROVIDER_WORD)
    v = (environ.get(sw) or "").strip()
    if v not in VALUES:
        raise ValueError(f"{sw}={v!r} is not one of {'|'.join(VALUES)}")
    if lv == "ln_provider":
        t = (environ.get(ENV_TIER) or "").strip()
        if t and t not in TIER_VALUES:
            raise ValueError(f"{ENV_TIER}={t!r} is not one of {'|'.join(TIER_VALUES)}")
    w = (environ.get(knob) or "").strip() if knob else ""
    if w and not _valid_word(w):
        raise ValueError(f"{knob}={w!r} is not a tier word (exact|fast|big|faithful) nor a serving row of opt_core.kernels.ln")
    return True


def word_sourced(environ=None) -> Tuple[Optional[str], str, Optional[str]]:
    """(word, source, tier): the word the provider is asked with — the lever's knob when set (`knob`), else the line's tier word (`line`),
    else the lever's default (`default`: exact for exactln, fast for ln_provider)."""
    environ = os.environ if environ is None else environ
    lv = lever(environ)
    if lv is None:
        return None, "off", None
    if lv == "exactln":
        w = (environ.get(ENV_WORD) or "").strip() if ENV_WORD else ""
        return (w, "knob", "exact") if w else (EXACT_WORD_DEFAULT, "default", "exact")
    t = (environ.get(ENV_TIER) or "").strip() or None
    w = (environ.get(ENV_PROVIDER_WORD) or "").strip() if ENV_PROVIDER_WORD else ""
    if w:
        return w, "knob", t or PROVIDER_WORD_DEFAULT
    return (t, "line", t) if t else (PROVIDER_WORD_DEFAULT, "default", PROVIDER_WORD_DEFAULT)


def word(environ=None) -> Optional[str]:
    return word_sourced(environ)[0]


def serving() -> bool:
    return STATE["state"] == "on"


def _bump(d: dict, k: str, n: int = 1) -> None:
    d[k] = d.get(k, 0) + n


UNWRAP_FREEVARS = ("orig_forward", "orig", "_orig")           # the closure names a class-wide wrapper of LayerNorm.forward keeps the function it wraps under when it sets no
                                                         # __wrapped__ (the offload port's LN-SAFE guard `_ln_safe_forward.<locals>.forward`: orig_forward) — followed like __wrapped__


def _wrapper_chain(fn):
    """[fn, the function it wraps, …, the innermost]: `__wrapped__` first, else a closure cell named in UNWRAP_FREEVARS holding a callable."""
    chain = [fn]
    while len(chain) < 9:
        cur = chain[-1]
        nxt = getattr(cur, "__wrapped__", None)
        if nxt is None:
            code, clo = getattr(cur, "__code__", None), getattr(cur, "__closure__", None)
            if code is not None and clo:
                for name, cell in zip(code.co_freevars, clo):
                    if name in UNWRAP_FREEVARS:
                        try:
                            v = cell.cell_contents
                        except ValueError:
                            continue
                        if callable(v):
                            nxt = v
                            break
        if nxt is None or nxt in chain:
            break
        chain.append(nxt)
    return chain


def _innermost(fn):
    return _wrapper_chain(fn)[-1]


def _over(fn) -> str:
    """The wrappers this binding installs over (their qualnames, outermost first) or 'none' — the census names what the kept calls run through."""
    names = [getattr(f, "__qualname__", getattr(f, "__name__", "?")).replace(".<locals>", "") for f in _wrapper_chain(fn)[:-1]]
    return ",".join(names) or "none"


def _dialect_of(fn) -> Tuple[Optional[str], str]:
    try:
        src = inspect.getsource(_innermost(fn))
    except Exception as e:  # noqa: BLE001
        return None, f"LayerNorm.forward: source unreadable ({type(e).__name__})"
    best, best_missing = None, None
    for name, want in DIALECTS.items():
        missing = [w for w in want if w not in src]
        if not missing:
            return name, ""
        if best_missing is None or len(missing) < len(best_missing):
            best, best_missing = name, missing
    return None, f"LayerNorm.forward differs from the statements this binding serves (nearest dialect {best}: missing {best_missing})"


def _form(x_dtype, torch) -> Optional[str]:
    if x_dtype is torch.float32:
        return "fp32"
    if x_dtype is torch.bfloat16:
        return "bf16"
    return None


def _n_tokens(x, rows: int) -> int:
    """The token count the cell table is keyed by: the running item's (MODEL_CLASS.forward read token_mask), else a pair-shaped operand
    ([.., N, N, C]) names it and is remembered; else round(sqrt(rows)) (the provider's own default)."""
    if STATE["n_tok"]:
        return STATE["n_tok"]
    if x.dim() >= 3 and int(x.shape[-2]) == int(x.shape[-3]) and int(x.shape[-2]) > 1:
        STATE["n_tok_seen"] = int(x.shape[-2])
        return STATE["n_tok_seen"]
    return STATE.get("n_tok_seen") or max(1, int(round(math.sqrt(rows))))


def _form_word(form: str, dialect: str) -> Tuple[str, bool, Optional[str]]:
    """(provider dtype, widen, out word) of a call: fp32 -> fp32; bf16 in the cast16 dialect -> bf16 (bf16 params); bf16 in the upcast dialect ->
    the bf16o form (bf16 x, fp32 params and statistics, bf16 store: widen with out='bf16')."""
    if form == "fp32":
        return "fp32", False, None
    if dialect == "upcast":
        return "bf16", True, "bf16"
    return "bf16", False, None


def _decide(LN, form: str, C: int, rows: int, x, affine: int, aligned: bool, capturing: bool):
    """The provider's Selection (or a kept reason) for a call class, cached per class: (sel | None, kept_reason | None, cell word)."""
    key = (form, C, rows, affine, aligned, capturing)
    hit = STATE["decisions"].get(key)
    if hit is not None:
        return hit
    n_tok = _n_tokens(x, rows)
    cell = LN.cell_for_rows(rows, n_tok, C)                              # the family whose row range holds this call (None: no family for the width)
    dtype, widen, outw = _form_word(form, STATE["dialect"])
    fword = LN.dtype_word(dtype, widen, outw)
    cname = str(cell) if cell is not None else "rows_c%d" % C
    try:
        sel = LN.select(STATE["cc"], dtype, cell, n_tok, word=STATE["word"], widen=widen, out=outw, C=C, capture=capturing, stack=STATE["stack"],
                        has_nvrtc=STATE["nvrtc"] is not None, has_triton=STATE["has_triton"], aligned=aligned, rows=rows)
    except LN.Refusal as r:
        out = (None, "refused:%s" % r.kind, "%s:%s:refused:%s" % (fword, cname, r.kind))
    else:
        arm = LN.arm_word(sel.row, sel.variant)
        tag = "%s:%s%s:%s" % (fword, cname, "@graph" if capturing else "", arm)
        if getattr(sel, "x_stock", None) is not None and sel.row not in LN.STOCK_ROWS:
            tag += "@x%.1f" % sel.x_stock
        if not sel.stack_measured:
            tag += "@ref"                                                # the cell's column is the reference stack's (this stack unlisted): flagged, as the provider's census does
        out = (None, "stock:%s" % arm, tag) if sel.row in LN.STOCK_ROWS else (sel, None, tag)
    STATE["decisions"][key] = out
    _bump(STATE["cells"], out[2])
    return out


def _exit_lines() -> None:
    if not STATE["installed"]:
        return
    _log(census_line())


def census_line() -> str:
    sb = ",".join(f"{k}={v}" for k, v in sorted(STATE["served_by"].items())) or "-"
    kb = ",".join(f"{k}={v}" for k, v in sorted(STATE["kept_by"].items())) or "-"
    cells = ",".join(f"{k}={v}" for k, v in sorted(STATE["cells"].items())) or "-"
    pre = (f" precompiled={STATE['precompiled']} precompile_s={STATE['precompile_s']:.2f}" if STATE.get("precompile_s") is not None else "") + \
          (f" precompile_error={STATE['precompile_error']}" if STATE.get("precompile_error") else "")
    return (f"LEVER name={STATE['lever']} state={STATE['state']} dialect={STATE['dialect']} word={STATE['word']} word_source={STATE.get('word_source')} tier={STATE['tier']} "
            f"stack={STATE['stack']} stack_listed={STATE['stack_listed']} over={STATE['over']} nvrtc={STATE['nvrtc']} torch={STATE['torch']} health={STATE['health']} "
            f"served={STATE['served']} kept={STATE['kept']} served_by={sb} kept_by={kb} cells={cells} proven={len(STATE['proven'])} witnessed={len(STATE['witnessed'])} "
            f"retired={len(STATE['retired'])} bits_differ={','.join(STATE['bits_differ']) or 'none'} compile_s={STATE['compile_s']:.2f}{pre}"
            + (f" reason={STATE['reason']}" if STATE['reason'] else ""))


def _refuse(reason: str) -> Dict[str, Any]:
    STATE.update(state="refused", reason=reason)
    _log(f"not installed: {reason}")
    return STATE


_PRE: Dict[str, Any] = {"thread": None, "lock": threading.Lock()}


def _precompile_worker(dialect: str) -> None:
    """Compile + self-check the replica's kernel for every (width, form) of PRECOMPILE_WIDTHS on synthetic operands (torch.equal vs the statement, own
    CUDA stream): NVRTC and cuModuleLoadData run here, off the forward.  Runs only when the word can name an exact-class row at all."""
    import torch
    import torch.nn.functional as F
    from opt_core.kernels.ln import exactln as E
    t0, n = time.perf_counter(), 0
    try:
        dev = torch.device("cuda", torch.cuda.current_device())
        stream = torch.cuda.Stream(device=dev)
        with torch.no_grad(), torch.cuda.stream(stream):
            g = torch.Generator(device=dev)
            g.manual_seed(1234567)
            for C in PRECOMPILE_WIDTHS:
                w = torch.randn(C, device=dev, generator=g)
                b = torch.randn(C, device=dev, generator=g)
                x32 = torch.randn(4096, C, device=dev, generator=g)
                forms = [("fp32", x32, w, b, False, None)]
                if dialect == "cast16":
                    forms.append(("bf16", x32.to(torch.bfloat16), w.to(torch.bfloat16), b.to(torch.bfloat16), False, None))
                else:
                    forms.append(("bf16o", x32.to(torch.bfloat16), w, b, True, torch.bfloat16))
                for name, x, ww, bb, widen, od in forms:
                    with _PRE["lock"]:
                        y = E.layer_norm(x, (C,), ww, bb, 1e-5, widen=widen, out_dtype=od)
                    ref = F.layer_norm(x.float(), (C,), ww.float(), bb.float(), 1e-5).to(torch.bfloat16) if widen else F.layer_norm(x, (C,), ww, bb, 1e-5)
                    stream.synchronize()
                    if y.dtype != ref.dtype or not torch.equal(y, ref):
                        STATE["bits_differ"].append(f"precheck:{name}:C={C}")
                        STATE["retired"].add(("pre", name, C))
                    else:
                        STATE["ready"].add((name, C))
                    n += 1
        stream.synchronize()
    except Exception as e:  # noqa: BLE001
        STATE["precompile_error"] = f"{type(e).__name__}:{str(e)[:120]}"
    STATE["precompiled"] = n
    STATE["precompile_s"] = time.perf_counter() - t0


def _probe_exactln(E, dev) -> dict:
    """The exactln package's readiness on this device, read through its public facts() plus one NVRTC version query: {"nvrtc": <version or None>,
    "torch", "torch_pin" (tuple), "health" ("ok" or the NVRTC load error), "capability"}.  The package binds NVRTC lazily through ctypes; a library
    that cannot load leaves nvrtc None and the exactln word steps aside by name (every LayerNorm runs the statement)."""
    import torch
    fx = {}
    try:
        fx = E.facts() or {}
    except Exception as e:                                                   # facts() is bookkeeping only; never fatal
        fx = {"health": f"facts:{type(e).__name__}"}
    pin = fx.get("torch_pin")
    pin = tuple(pin) if isinstance(pin, (list, tuple)) else ((pin,) if isinstance(pin, str) and pin else ())
    nv, health = None, fx.get("health") or "ok"
    try:
        ver = getattr(E, "_nvrtc_version", None)
        if ver is not None:
            nv = ver()
        else:                                                                # older packages: the ctypes binding itself is the readiness
            E._bindings("nvrtc"); nv = "?"
    except Exception as e:
        nv, health = None, f"{type(e).__name__}: {str(e)[:100]}"
    return {"nvrtc": nv, "torch": fx.get("torch") or torch.__version__, "torch_pin": pin, "health": health, "capability": tuple(torch.cuda.get_device_capability(dev))}

def install(environ=None) -> Dict[str, Any]:
    environ = os.environ if environ is None else environ
    if STATE["installed"] or STATE["state"] == "refused":
        return STATE
    lv = lever(environ)
    if lv is None:
        return STATE
    requested(environ)
    w, wsrc, tier = word_sourced(environ)
    STATE.update(lever=lv, word=w, word_source=wsrc, tier=tier, prefix=PREFIX if lv == "exactln" else PREFIX_PROVIDER)
    import atexit
    import importlib
    import torch
    import torch.nn.functional as F
    if not torch.cuda.is_available():
        return _refuse("no CUDA device")
    from opt_core.kernels import ln as LN                                 # the shared core's LayerNorm provider: rows, tier words, cells, census
    from opt_core.kernels.ln import exactln as E                          # its exactln row's carried package (facts / NVRTC readiness only)
    dev = torch.device("cuda", torch.cuda.current_device())
    pr = _probe_exactln(E, dev)
    STATE.update(nvrtc=pr.get("nvrtc"), torch=pr.get("torch"), torch_pin=",".join(pr.get("torch_pin") or ()), health=pr.get("health") or "ok",
                 cc="%d.%d" % tuple(pr.get("capability") or torch.cuda.get_device_capability(dev)))
    try:
        import triton  # noqa: F401
        STATE["has_triton"] = True
    except ImportError:
        STATE["has_triton"] = False
    STATE["stack"] = LN.stack_word(dev)
    STATE["stack_listed"] = "y" if STATE["stack"] in (LN.table().get("stacks") or {}) else "n"
    if LN.split_word(w)[0] == "exactln" and STATE["nvrtc"] is None:      # the replica's row asked by name needs NVRTC: absent -> every LayerNorm runs the statement, by name
        return _refuse("word %s: no NVRTC library loadable (%s): every LayerNorm runs the statement" % (w, STATE["health"]))
    try:
        norm = importlib.import_module(M_NORM)
    except ImportError as e:
        return _refuse(f"cannot import {M_NORM}: {e}")
    cls = getattr(norm, "LayerNorm", None)
    if cls is None or not isinstance(cls, type):
        return _refuse(f"{M_NORM}.LayerNorm absent")
    orig = cls.forward
    dialect, why = _dialect_of(orig)
    if dialect is None:
        return _refuse(why)
    STATE["dialect"] = dialect
    STATE["over"] = _over(orig)
    cached_cast = None
    if CASTCACHE_MODULE:
        try:
            cc_mod = importlib.import_module(CASTCACHE_MODULE)
            cached_cast = getattr(cc_mod, "cached_cast", None)
            cc_serving = getattr(cc_mod, "serving", None)
        except ImportError:
            cached_cast = None
    served_by, kept_by, ready, retired = STATE["served_by"], STATE["kept_by"], STATE["ready"], STATE["retired"]
    bf16 = torch.bfloat16
    exact_rows = frozenset(LN.EXACT_ROWS)

    def _keep(reason: str, self, x):
        STATE["kept"] += 1
        _bump(kept_by, reason)
        return orig(self, x)

    def _cast(t):
        if cached_cast is not None and cc_serving is not None and cc_serving():
            return cached_cast(t, bf16)
        return t.to(dtype=bf16)

    def _statement(form: str, x, w_, b_, C: int, eps: float):
        """The statement's own arithmetic for the call (the proof / witness reference): fp32 and cast16-bf16 = F.layer_norm on the operands as
        passed; upcast-bf16 = everything .float(), cast back."""
        if form == "bf16" and dialect == "upcast":
            return F.layer_norm(x.float(), (C,), None if w_ is None else w_.float(), None if b_ is None else b_.float(), eps).to(dtype=bf16)
        return F.layer_norm(x, (C,), w_, b_, eps)

    def _ready_arm(sel, form: str, C: int, affine: int, aligned: bool, x, w_, b_, eps: float, out_dtype):
        """Serve one call through the provider with a fixed Selection and, on the signature's first call, prove (exact-class rows: torch.equal) or
        witness (tolerance rows: rel-RMS) it against the statement.  Returns (y | None) — None = the signature is retired (the caller keeps the statement)."""
        arm = LN.arm_word(sel.row, sel.variant)
        sig = (form, C, affine, aligned, arm)
        if sig in retired:
            return None
        first = sig not in ready
        t0 = time.perf_counter() if first else 0.0
        try:
            y = LN.layer_norm(x, (C,), w_, b_, eps, selection=sel, out_dtype=out_dtype)[0]
        except LN.Refusal as r:                                            # a row that refuses at serving time (kernel absent for the ABI, width): the signature retires by name
            retired.add(sig)
            STATE["bits_differ"].append(f"{arm}:{form}:C={C}(refused:{r.kind})")
            return None
        except Exception as e:  # noqa: BLE001 — a row that raises on its first call refuses the signature by name; later signatures unaffected
            if not first:
                raise
            retired.add(sig)
            STATE["bits_differ"].append(f"{arm}:{form}:C={C}(error:{type(e).__name__})")
            return None
        if first:
            ref = _statement(form, x, w_, b_, C, eps)
            ok_form = (y.dtype == ref.dtype and y.shape == ref.shape)
            if sel.row in exact_rows:
                ok = ok_form and torch.equal(y, ref)
                tag = "proof"
            else:
                if ok_form:
                    d = (y.float() - ref.float())
                    rel = float(d.pow(2).mean().sqrt() / ref.float().pow(2).mean().sqrt().clamp_min(1e-30))
                    ok = math.isfinite(rel) and rel <= WITNESS_RELRMS
                else:
                    ok = False
                tag = "witness"
            STATE["compile_s"] += time.perf_counter() - t0
            if not ok:
                retired.add(sig)
                STATE["bits_differ"].append(f"{arm}:{form}:C={C}:affine={affine}({tag})")
                STATE["kept"] += 1
                _bump(kept_by, "%s_fail" % tag)
                return ref
            ready.add(sig)
            (STATE["proven"] if tag == "proof" else STATE["witnessed"]).append(f"{arm}:{form}:C={C}:affine={affine}:aligned={int(aligned)}")
        STATE["served"] += 1
        _bump(served_by, f"{arm}:{form}:C={C}")
        return y

    def forward(self, x):
        form = _form(x.dtype, torch)
        if form is None:
            return _keep("other_dtype", self, x)
        if not x.is_cuda:
            return _keep("cpu", self, x)
        if torch.is_autocast_enabled():
            return _keep("autocast", self, x)
        c_in = self.c_in
        C = int(c_in[-1]) if isinstance(c_in, (tuple, list)) else int(c_in)
        if len(c_in) != 1 or int(x.shape[-1]) != C:
            return _keep("shape", self, x)
        w, b = self.weight, self.bias
        if (w is None) != (b is None):
            return _keep("partial_affine", self, x)
        affine = 0 if w is None else 1
        if affine and (w.dtype is not torch.float32 or b.dtype is not torch.float32 or w.dim() != 1 or int(w.shape[0]) != C or int(b.shape[0]) != C):
            return _keep("param_form", self, x)
        if not x.is_contiguous():
            return _keep("noncontig", self, x)
        n = x.numel()
        if n == 0 or n > NUMEL_MAX:
            return _keep("numel", self, x)
        rows = n // C
        aligned = (x.data_ptr() % 16 == 0)
        capturing = torch.cuda.is_current_stream_capturing()
        sel, kept, _tag = _decide(LN, form, C, rows, x, affine, aligned, capturing)
        if sel is None:
            return _keep(kept, self, x)
        arm = LN.arm_word(sel.row, sel.variant)
        sig = (form, C, affine, aligned, arm)
        if sig in retired:
            return _keep("retired", self, x)
        eps = float(self.eps)
        if form == "bf16" and dialect == "cast16":
            w_, b_, od = (_cast(w), _cast(b), None) if affine else (None, None, None)
        elif form == "bf16":                                              # upcast dialect: fp32 params, fp32 statistics, bf16 store (the provider's bf16o form)
            w_, b_, od = (w, b, bf16) if affine else (None, None, bf16)
        else:
            w_, b_, od = (w, b, None) if affine else (None, None, None)
        if sig not in ready:
            if capturing:                                                 # a signature first met during capture: its kernel may not be loaded / compiled yet — the statement, by name
                return _keep("capture_first", self, x)
            y = _ready_arm(sel, form, C, affine, aligned, x, w_, b_, eps, od)
            if y is None:
                return _keep("retired", self, x)
            gsel, _gk, _gt = _decide(LN, form, C, rows, x, affine, aligned, True)     # ready the arm the GRAPH cell of this class names too, off any capture
            if gsel is not None:
                garm = LN.arm_word(gsel.row, gsel.variant)
                if (form, C, affine, aligned, garm) not in ready and (form, C, affine, aligned, garm) not in retired:
                    yg = _ready_arm(gsel, form, C, affine, aligned, x, w_, b_, eps, od)
                    if yg is not None:                                    # counted as served above; undo: this was a readiness call, its result is dropped
                        STATE["served"] -= 1
                        _bump(served_by, f"{garm}:{form}:C={C}", -1)
            return y
        y = _ready_arm(sel, form, C, affine, aligned, x, w_, b_, eps, od)
        return _keep("retired", self, x) if y is None else y

    forward.__wrapped__ = orig
    forward.__qualname__ = "ln_binding_forward"
    cls.forward = forward
    # The running item's token count keys the provider's cells (decided per item, before the item's first LayerNorm): MODEL_CLASS.forward(batch)
    # reads batch["token_mask"].  A model class without that entry leaves the count to the operands (pair-shaped operands name it).
    try:
        model_mod = importlib.import_module(M_MODEL)
        mcls = getattr(model_mod, MODEL_CLASS, None)
    except ImportError:
        mcls = None
    if mcls is not None and isinstance(mcls, type) and callable(getattr(mcls, "forward", None)):
        m_orig = mcls.forward

        def model_forward(self, batch, *a, **k):
            try:
                tm = batch.get("token_mask") if hasattr(batch, "get") else None
                n_tok = int(tm.shape[-1]) if tm is not None else None
            except Exception:  # noqa: BLE001
                n_tok = None
            if n_tok and n_tok != STATE["n_tok"]:
                STATE["n_tok"] = n_tok
                STATE["decisions"].clear()                                # a new token count re-keys the cells (signatures stay ready)
            return m_orig(self, batch, *a, **k)

        model_forward.__wrapped__ = m_orig
        mcls.forward = model_forward
        STATE["item_hook"] = f"{MODEL_CLASS}.forward"
    else:
        STATE["item_hook"] = None
    STATE.update(installed=True, state="on")
    atexit.register(_exit_lines)
    can_exact = LN.split_word(w)[0] in ("exact", "fast", "big", "faithful", "exactln")
    if STATE["nvrtc"] is not None and PRECOMPILE_WIDTHS and can_exact and _PRE["thread"] is None:
        _PRE["thread"] = threading.Thread(target=_precompile_worker, args=(dialect,), name="exactln-precompile", daemon=True)
        _PRE["thread"].start()
    _log(f"installed: LayerNorm.forward ({dialect} dialect) bound to opt_core.kernels.ln by the word {w} ({wsrc}; lever {lv}, tier {tier}): the provider's cell table decides "
         f"per call class and capture state on stack {STATE['stack']} (listed={STATE['stack_listed']}, cc {STATE['cc']}); cells naming ATen, refusals and retired signatures run the "
         f"statement, counted; exact-class rows proven torch.equal / tolerance rows witnessed (rel-RMS <= {WITNESS_RELRMS}) on each signature's first call; "
         f"torch {STATE['torch']} (replica pin {STATE['torch_pin']}), nvrtc {STATE['nvrtc']}, triton {'yes' if STATE['has_triton'] else 'no'}; "
         f"item token count from {STATE['item_hook'] or 'the operands'}")
    return STATE
