"""The accelerator proof: ONE reader of what the BUILT model engages, the expected words per (mode, stack), and the guard.

``census(model)`` reads the objects actually bound in a loaded ``EsmcForMaskedLM`` (or the SDK client that holds one): the attention
class of every transformer block, the classes bound at the LayerNorm+QKV / output-projection / feed-forward sites, the rotary class,
the model's resolved ``_use_flash_attn``, and upstream's kernel flags together with the module and version of the function objects those
flags bound (``esm.models.esmc.kernels``) — never distribution metadata alone. One word per accelerator of ``modes.ACCELERATORS``:

    engaged:<impl>@<version>[(<classes>)]   the accelerator's own objects are bound at upstream's call sites
    fallback:<impl>:<why>                   upstream's fallback implementation is bound (why: absent | import-error… | not-cuda | …)
    off-by-rule:<rule>(use_flash_attn=False;sdpa-class;flash_attn@<version>)
                                            the route's call turned the varlen class off by a named rule of the modes table
                                            (``modes.STOCK_USE_FLASH_ATTN`` / ``STOCK_RULE_NAMES``) — the non-varlen class is bound AND flash-attention still imports
                                            with its dense kernel bound (the class's cascade runs it, layers.py:378-388)
    off-by-route:sdpa-class:<why>           the non-varlen class is bound with no rule naming it (a caller's own ``use_flash_attn=False``)
    absent                                  installed nowhere and bound nowhere (xformers: pinned absent)
    present:<version>                       importable where the pin says absent (xformers) — named

``expected(pins, stack, call, rule)`` = the words a run must show: every accelerator ``stock/PINS.json`` pins a version for on the expected
stack is ``engaged:<impl>@<that version>`` — except flash_attn when the route's own call (``modes.load_call``) says ``use_flash_attn=False``
by a named rule (``modes.call_rule``): then ``off-by-rule:<rule>(use_flash_attn=False;sdpa-class;flash_attn@<that version>)``; one it pins
absent is ``fallback:<impl>:absent`` (``absent`` for an accelerator pinned absent). The expected stack is the pinned stack
(``modes.pinned_stack``); the line names it (``stack=<name>``).
``require`` compares word by word; ``prove_once`` = census → expected → the KERNELS line (``report.kernels_line``) → require, once per
model object, the record kept for the manifest (``record()``); a difference is NAMED on a second line (``report.kernels_short_line``:
every differing accelerator with its expected word) and the run proceeds on what the built model engages — an environment short of the
pinned stack is stated, never a refusal (the KERNELS words already say what serves each call). Upstream's own import-time warnings (kernels.py:64-90: Transformer Engine missing, neither attention kernel importable,
the rotary kernel missing) are the same conditions these words name; ``listen()`` captures their text into the record as well.
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import importlib.util
import logging
import re
import sys
from typing import Dict, List, Optional, Tuple

from . import modes, report as _report

UPSTREAM_KERNELS = "esm.models.esmc.kernels"          # upstream's flags and the function objects they bound
UPSTREAM_LAYERS = "esm.models.esmc.layers"            # the classes upstream builds the blocks from
FLASH_ATTN_CLASS = "EsmcFlashMultiHeadAttention"       # layers.py:489 (a subclass of the sdpa class: checked first)
SDPA_ATTN_CLASS = "EsmcMultiHeadAttention"             # layers.py:404
FLAGS = ("TE_INSTALLED", "XFORMERS_INSTALLED", "FLASH_ATTN_INSTALLED", "FLASH_ATTN_ROTARY_INSTALLED")
BOUND_FUNCTIONS = ("flash_attn_varlen_qkvpacked_func", "flash_attn_func", "unpad_input", "pad_input", "apply_triton_rotary", "te", "xops")
VIAS = ("stock_child", "kit")                          # which process served the model: the clean stock child, or the kits in-process

_WARNINGS: List[str] = []
_STATE: Dict[str, object] = {"records": {}, "last": None, "listening": False}


class _Catch(logging.Handler):
    def emit(self, record):                                   # noqa: D401 — a logging.Handler hook
        try:
            _WARNINGS.append(record.getMessage())
        except Exception:                                     # noqa: BLE001 — a listener never breaks the run
            pass


def listen() -> None:
    """Capture upstream's kernel warnings (the logger of esm/models/esmc/kernels.py) from now on; idempotent; install before the upstream import."""
    if _STATE["listening"]:
        return
    logging.getLogger(UPSTREAM_KERNELS).addHandler(_Catch(level=logging.WARNING))
    _STATE["listening"] = True


def warnings_seen() -> List[str]:
    return list(_WARNINGS)


# ----------------------------------------------------------------------------------------------------------------------------- helpers
def _dist_version(name: str) -> Optional[str]:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _importable(modname: str) -> bool:
    try:
        return importlib.util.find_spec(modname) is not None
    except Exception:                                         # noqa: BLE001 — a parent that fails to import = not importable
        return False


def _import_error(modname: str) -> Optional[str]:
    """Re-run an import upstream's try/except swallowed, to name why it failed (None when it imports now)."""
    try:
        importlib.import_module(modname)
        return None
    except Exception as e:                                    # noqa: BLE001 — the text is the point
        return f"{type(e).__name__}: {e}"


def _token(text, limit: int = 96) -> str:
    """A line-safe token: no whitespace (the line grammar: values carry no spaces), bounded length."""
    t = re.sub(r"\s+", "_", str(text)).strip("_")
    return t[:limit]


def _why_not(modname: str, dist: str) -> str:
    """Why an accelerator's module is not bound: ``absent`` (no distribution, not importable) or ``import-error(<the error>)``."""
    if _dist_version(dist) is None and not _importable(modname.split(".")[0]):
        return "absent"
    err = _import_error(modname)
    return f"import-error({_token(err)})" if err else "importable-unbound"


def _module_version(top: str, dist: str) -> Optional[str]:
    mod = sys.modules.get(top)
    v = getattr(mod, "__version__", None) if mod is not None else None
    return str(v) if v else _dist_version(dist)


def _norm_version(v: Optional[str]) -> Optional[str]:
    return None if v is None else str(v).split("+", 1)[0].strip()


def _cls_id(t: type) -> str:
    return f"{t.__module__}.{t.__qualname__}"


def _model_of(x) -> Tuple[object, object]:
    """(the masked-LM model, its core EsmcModel) from an SDK client, an EsmcForMaskedLM or an EsmcModel."""
    m = getattr(x, "model", x)                                # the SDK client (compatibility.py ESMC) holds .model
    core = getattr(m, "esmc", m)                              # EsmcForMaskedLM.esmc (model.py:555)
    return m, core


# ------------------------------------------------------------------------------------------------------------------------------ census
def census(x, rule: Optional[str] = None) -> dict:
    """The one reader. ``x`` = the SDK client, the masked-LM model or the core model — LOADED (the classes are chosen at build:
    layers.py:588-596, model.py:286-298). ``rule`` = the name of the modes-table rule by which the route's own call turned the varlen
    class off (``modes.call_rule``), or None: it only NAMES the reason on the flash_attn word when the non-varlen class is what is bound;
    it never changes what is read."""
    K = sys.modules.get(UPSTREAM_KERNELS)
    L = sys.modules.get(UPSTREAM_LAYERS)
    if K is None or L is None:
        raise RuntimeError("kernels.census: the upstream package is not imported — a model must be built first")
    m, core = _model_of(x)
    blocks = list(getattr(getattr(core, "transformer", None), "blocks", None) or [])
    if not blocks:
        raise RuntimeError("kernels.census: the model has no transformer blocks (esmc.transformer.blocks)")
    p0 = next(core.parameters())
    device, dtype = p0.device.type, str(p0.dtype).replace("torch.", "")
    flash_cls, sdpa_cls = getattr(L, FLASH_ATTN_CLASS), getattr(L, SDPA_ATTN_CLASS)
    attn = {type(b.attn) for b in blocks}
    qkv = {type(b.attn.layernorm_qkv) for b in blocks}
    outp = {type(b.attn.out_proj) for b in blocks}
    ffn = {type(b.ffn) for b in blocks}
    rot = {type(b.attn.rotary) for b in blocks}
    flags = {k: bool(getattr(K, k, False)) for k in FLAGS}
    funcs = {}
    for n in BOUND_FUNCTIONS:
        obj = getattr(K, n, None)
        funcs[n] = None if obj is None else (getattr(obj, "__module__", None) or getattr(obj, "__name__", None) or type(obj).__name__)
    use_flash = bool(getattr(core, "_use_flash_attn", False))
    attn_impl = getattr(getattr(m, "config", None), "attn_implementation", None) or getattr(getattr(core, "config", None), "attn_implementation", None)
    rec = {"device": device, "dtype": dtype, "n_blocks": len(blocks), "flags": flags, "functions": funcs,
           "use_flash_attn_resolved": use_flash, "attn_implementation_requested": attn_impl,
           "bound": {"attn": sorted(_cls_id(t) for t in attn), "layernorm_qkv": sorted(_cls_id(t) for t in qkv), "out_proj": sorted(_cls_id(t) for t in outp),
                     "ffn": sorted(_cls_id(t) for t in ffn), "rotary": sorted(_cls_id(t) for t in rot)},
           "cuda_extensions": sorted(n for n in list(sys.modules) if n.startswith("flash_attn") and n.endswith("_cuda")),
           "versions": {"flash_attn": _module_version("flash_attn", "flash_attn"), "transformer_engine": _module_version("transformer_engine", "transformer_engine"),
                        "xformers": _module_version("xformers", "xformers")},
           "upstream_warnings": warnings_seen()}
    A = modes.ACCELERATORS
    words: Dict[str, str] = {}
    # flash-attention: the flash class on every block, the model's resolved flag, upstream's flag, and the bound varlen function from flash_attn
    all_flash = all(issubclass(t, flash_cls) for t in attn)
    all_sdpa = all(issubclass(t, sdpa_cls) and not issubclass(t, flash_cls) for t in attn)
    fa_fn = funcs.get("flash_attn_varlen_qkvpacked_func") or ""
    if all_flash and use_flash and flags["FLASH_ATTN_INSTALLED"] and fa_fn.startswith("flash_attn"):
        words["flash_attn"] = f"engaged:{A['flash_attn']['engaged']}@{rec['versions']['flash_attn']}"
    elif not flags["FLASH_ATTN_INSTALLED"]:
        words["flash_attn"] = f"fallback:{A['flash_attn']['fallback']}:{_why_not('flash_attn', 'flash_attn')}"
    elif all_sdpa:
        dense = (funcs.get("flash_attn_func") or "").startswith("flash_attn")                        # flash-attention's dense kernel bound: the sdpa class's cascade runs it on an unmasked batch (layers.py:378-388)
        if attn_impl and attn_impl != "flash_attention_2" and rule and dense and device == "cuda":
            words["flash_attn"] = f"off-by-rule:{rule}(use_flash_attn=False;sdpa-class;{A['flash_attn']['engaged']}@{rec['versions']['flash_attn']})"
        elif attn_impl and attn_impl != "flash_attention_2":
            words["flash_attn"] = f"off-by-route:sdpa-class:attn_implementation={_token(attn_impl)}" + ("" if dense else ";flash_attn_func-unbound")   # a caller's own use_flash_attn=False (compatibility.py:169), no rule naming it
        else:
            words["flash_attn"] = f"fallback:sdpa-class:{'not-cuda' if device != 'cuda' else 'unresolved'}"
    else:
        words["flash_attn"] = f"fallback:mixed:{_token(','.join(sorted(t.__name__ for t in attn)))}"
    # the rotary kernel: upstream's flag, the bound function from flash_attn.ops.triton.rotary, a CUDA model (both rotary classes take it then)
    r_fn = funcs.get("apply_triton_rotary") or ""
    if flags["FLASH_ATTN_ROTARY_INSTALLED"] and r_fn.startswith("flash_attn") and device == "cuda":
        words["rotary"] = f"engaged:{A['rotary']['engaged']}@{rec['versions']['flash_attn']}"
    elif not flags["FLASH_ATTN_ROTARY_INSTALLED"]:
        words["rotary"] = f"fallback:{A['rotary']['fallback']}:{_why_not('flash_attn.ops.triton.rotary', 'flash_attn')}"
    else:
        words["rotary"] = f"fallback:{A['rotary']['fallback']}:{'not-cuda' if device != 'cuda' else 'unbound(' + _token(r_fn) + ')'}"
    # Transformer Engine: the classes bound at the three sites come from transformer_engine, upstream's flag, the bound te module
    te_sites = qkv | outp | ffn
    te_bound = all(t.__module__.startswith("transformer_engine") for t in te_sites)
    te_any = any(t.__module__.startswith("transformer_engine") for t in te_sites)
    if flags["TE_INSTALLED"] and te_bound and (funcs.get("te") or "").startswith("transformer_engine"):
        words["te"] = f"engaged:{A['te']['engaged']}@{rec['versions']['transformer_engine']}({','.join(sorted({t.__name__ for t in te_sites}))})"
    elif not flags["TE_INSTALLED"]:
        words["te"] = f"fallback:{A['te']['fallback']}:{_why_not('transformer_engine.pytorch', 'transformer_engine')}"
    elif not te_any:
        words["te"] = f"fallback:{A['te']['fallback']}:{'not-cuda' if device != 'cuda' else 'unresolved'}"
    else:
        words["te"] = f"fallback:mixed:{_token(','.join(sorted(t.__name__ for t in te_sites)))}"
    # xformers: pinned absent; importable = refused (present), a broken install = refused (present-broken)
    if flags["XFORMERS_INSTALLED"]:
        words["xformers"] = f"present:{rec['versions']['xformers']}"
    elif _dist_version("xformers") is not None or _importable("xformers"):
        words["xformers"] = f"present-broken:{rec['versions']['xformers']}"
    else:
        words["xformers"] = "absent"
    assert tuple(words) == tuple(A), (tuple(words), tuple(A))          # one word per accelerator, in table order
    rec["words"] = words
    rec["rule"] = rule
    return rec


# ---------------------------------------------------------------------------------------------------------------------------- expected
def expected(p: dict, stack_name: str, call: Optional[dict] = None, rule: Optional[str] = None) -> Dict[str, str]:
    """The words a run must show on ``stack_name`` (a "stacks" entry of stock/PINS.json), derived from modes.ACCELERATORS, the pins, and the
    route's own call: ``call`` = the ``ESMC.from_pretrained`` keywords the route passed (``modes.load_call``), ``rule`` = ``modes.call_rule`` —
    a call that says ``use_flash_attn=False`` by a named rule expects the off-by-rule word (the non-varlen class bound, flash-attention still
    importable at the pinned version); every other call expects the varlen class engaged."""
    st = (p.get("stacks") or {}).get(stack_name)
    if st is None:
        raise modes.ResolveError(f"stack {stack_name!r} is not a stack of stock/PINS.json")
    flash_off_by_rule = bool(rule) and (call or {}).get("use_flash_attn") is False
    out = {}
    for name, a in modes.ACCELERATORS.items():
        if a["pin"] is None:
            out[name] = "absent"
        elif st.get(a["pin"]):
            if name == "flash_attn" and flash_off_by_rule:
                out[name] = f"off-by-rule:{rule}(use_flash_attn=False;sdpa-class;{a['engaged']}@{st[a['pin']]})"
            else:
                out[name] = f"engaged:{a['engaged']}@{st[a['pin']]}"
        else:
            out[name] = f"fallback:{a['fallback']}:absent"
    return out


def _strip_local_versions(word: str) -> str:
    """`@<version>+<local tag>` -> `@<version>` inside a word (a local build tag on the pinned version is the pinned version)."""
    return re.sub(r"@([^;()+@]+)\+[^;()]*", r"@\1", word or "")


def word_matches(observed: str, exp: str) -> bool:
    """``engaged`` words match on implementation and version (a local ``+suffix`` and the ``(<classes>)`` tail aside); other words exactly
    (a local version tag aside)."""
    if exp.startswith("engaged:"):
        mo = re.match(r"^engaged:([^@]+)@([^()]*)(\(.*\))?$", observed or "")
        me = re.match(r"^engaged:([^@]+)@(.*)$", exp)
        return bool(mo and me and mo.group(1) == me.group(1) and _norm_version(mo.group(2)) == _norm_version(me.group(2)))
    return _strip_local_versions(observed) == _strip_local_versions(exp)


def differences(words: Dict[str, str], exp: Dict[str, str]) -> List[Tuple[str, str, str]]:
    return [(n, words.get(n, "unread"), e) for n, e in exp.items() if not word_matches(words.get(n, "unread"), e)]


def require(rec: dict, printer=print) -> dict:
    """Name every accelerator whose word differs from the expectation (``rec`` from ``prove_once``: words + expected + route context):
    ``verdict`` = "ok" | "short"; a shortfall prints the KERNELS SHORT line and the run proceeds (recorded in the manifest's ``kernels``)."""
    diffs = differences(rec["words"], rec["expected"])
    rec["differences"] = [{"accelerator": n, "observed": o, "expected": e} for n, o, e in diffs]
    rec["verdict"] = "short" if diffs else "ok"
    if diffs:
        rec["short_line"] = _report.kernels_short_line(rec)
        printer(rec["short_line"])
    return rec


def prove_once(x, *, mode: str, via: str, regime: Optional[str] = None, call: Optional[dict] = None,
               rule: Optional[str] = None, p: Optional[dict] = None, printer=print) -> dict:
    """census → expected → the KERNELS line → require, once per model object (a second call for the same model returns its record).
    ``mode`` = the caller's mode (the route word is modes.ROUTE_WORDS[mode]); ``via`` ∈ VIAS;
    ``regime`` = the pass's call shape; ``call`` / ``rule`` = the ``ESMC.from_pretrained`` keywords the route actually passed and the
    modes-table rule that chose them (``modes.load_call`` / ``modes.call_rule`` of the LOADING mode — the stock child serving a kit mode's
    job loads by its own stock-class mode)."""
    m, _core = _model_of(x)
    key = id(m)
    done = _STATE["records"].get(key)                         # type: ignore[union-attr]
    if done is not None:
        return done
    if via not in VIAS:
        raise ValueError(f"via {via!r} is not one of {VIAS}")
    if p is None:
        from . import stack as _stack
        p = _stack.pins()
    sor = modes.pinned_stack(p)
    rec = census(x, rule=rule)
    rec.update({"mode": modes.check_mode(mode), "route": modes.ROUTE_WORDS[modes.check_mode(mode)], "via": via, "regime": modes.check_regime(regime),
                "pinned_stack": sor, "stack_token": sor,
                "call": {k: str(v) for k, v in (call or {}).items()}, "expected": expected(p, sor, call, rule)})
    rec["line"] = _report.kernels_line(rec)
    printer(rec["line"])
    _STATE["records"][key] = rec                              # type: ignore[index]
    _STATE["last"] = rec
    try:
        return require(rec, printer=printer)
    finally:
        sys.stdout.flush()


def record() -> Optional[dict]:
    """The last proof of this process (``esmc_opt.status()["kernels"]`` carries it), or None before any model was proven."""
    return _STATE["last"]                                     # type: ignore[return-value]


def _reset_for_tests() -> None:
    _STATE["records"] = {}
    _STATE["last"] = None
    del _WARNINGS[:]
