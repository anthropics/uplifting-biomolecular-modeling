"""accel.py — the KERNELS proof: ONE reader of the accelerators the running model process BOUND at upstream's dispatch sites, and its
line. Every route passes through here: the stock runner (stock_score.py, mode off, route word stock),
the kit runner (kit_score.py, mode exact) and the env / API arm (stack.py) call `install(...)` before
the upstream tool runs; the proof fires ONCE per process at the first `E1Predictor` construction (`E1/tools/score.py` L43-61 → `E1Scorer`
builds one: the model is on the device and the kit — on its route — has applied) and prints one line (report.kernels_line).

Upstream E1 (the pinned commit, `stock/src/`) engages two optional accelerators when they import and runs a third torch ships:
  flash_attn      flash-attn's varlen kernel for the within-sequence layers — `E1/model/flash_attention.py` L5-9 imports
                  `flash_attn_func, flash_attn_varlen_func` (None on ImportError), `is_flash_attention_available()` L14-17 = both symbols AND
                  env `USE_FLASH_ATTN` (default "1") read PER CALL, dispatched at `E1/model/attention.py` L299-311 inside `Attention._flash_attn`
                  (else `varlen_flex_attention_func` — no log line); the model reaches it through the class attribute `Attention._flash_attn`
                  (L252-253) and the module name `E1.model.attention.flash_attention_func` (bound from `.flash_attention` at import, L10).
  hub_layernorm   the Hugging Face `kernels` hub Triton RMSNorm `kernels-community/triton-layer-norm`, fetched at import (`E1/modeling.py`
                  L22-26: on failure `logger.warning("Failed to load triton layer norm kernel: …")` and `layer_norm = None` = torch `rms_norm`,
                  `RMSNorm.forward` L82-98).
  flex_attention  torch's flex_attention for the global layers, `torch.compile`d when CUDA is available (`E1/model/flex_attention.py` L4-11).

What is read is the OBJECT bound in this process — module globals, the class attribute the model calls through, the accelerator's own
loaded extension / kernel object — never distribution metadata: the word printed is the path the forward will take.
  word := engaged:<impl>@<version>  |  fallback:<what>(<why>)  |  absent(<why>)
EXPECTED: the words the pinned stock prints = the mode table's accelerators for the mode (modes.MODE_TABLE[mode].kernels: WHICH are
engaged) at the pins module's versions (pins.STACK / pins.KERNEL: at WHAT version) — `expected(mode, pins)`. An accelerator whose word
differs on this host (absent, upstream's own fallback taken, another version) is NAMED by that word on the line and recorded
(`fell_back`). Under mode off (the stock untouched) that is all: the stock runs the path upstream itself takes there and the calling
`e1-opt score` lists what fell back on its EXIT line (`kernels_fallback=`). Under mode exact the kit's levers run ON those
accelerators, so an absent or fallen-back one means the mode cannot run in this process: it REFUSES BY NAME — the `NOT ACTIVE: mode <m>:
accelerator(s) not engaged …` line and exit 3 before the first forward — never a subset of the mode under the mode's name. Upstream's own fallback signal (the `E1.modeling` warning above) is trapped by
a logging handler attached before the upstream package is imported (`trap_upstream_logs`) and worded into the hub_layernorm word — the
same reader, not a second mechanism.
"""
from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
from typing import Callable, Dict, List, Optional, Tuple

from . import modes, report
from ._names import EXIT_NOT_ACTIVE

ACCELERATORS = ("flash_attn", "hub_layernorm", "flex_attention")
IMPL = {"flash_attn": "flash_attn", "hub_layernorm": "triton_layer_norm", "flex_attention": "torch.compile"}   # the <impl> of each accelerator's engaged word
STOCK_ROUTES = ("stock", "default")                               # mode off: the stock runner's route words (the documented command | the tool's default line as a pass)
ROUTES = STOCK_ROUTES + modes.KIT_MODES                            # + the kit route (exact): the route word IS the mode
assert ACCELERATORS == modes.KERNELS_ALL == report.KERNELS_ACCELERATORS and tuple(IMPL) == ACCELERATORS and ROUTES == report.KERNELS_ROUTES

#: upstream's own fallback signals: accelerator -> (logger name, message head) — E1/modeling.py L20 `logging.get_logger(__name__)`, L25 the warning
UPSTREAM_FALLBACK_LOGS = {"hub_layernorm": ("E1.modeling", "Failed to load triton layer norm kernel")}
FLASH_ENV = "USE_FLASH_ATTN"                                      # upstream's switch (flash_attention.py L16), read here as upstream reads it
_RE_SNAPSHOT = re.compile(r"snapshots[/\\]([0-9a-f]{40})[/\\]")
_TRAP = {"installed": False, "records": []}
_HOOK = {"installed": [], "fired": False, "result": None, "config": None, "finder": None}


# --------------------------------------------------------------------------------------------------------------------- words
def engaged(accel: str, version) -> str:
    return f"engaged:{IMPL[accel]}@{version}"


def _clean(s, n: int = 72) -> str:
    s = re.sub(r"[^A-Za-z0-9_.:=+@/<>-]+", "_", str(s)).strip("_")
    return s[:n]


def _qual(f) -> str:
    if f is None:
        return "None"
    return f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', getattr(f, '__name__', type(f).__name__))}"


def route_name(mode: str) -> str:
    """The route word: `stock` on the stock route (mode off), else the mode (exact)."""
    return "stock" if mode == "off" else mode


# ------------------------------------------------------------------------------------------------------ upstream's own signals
class _Trap(logging.Handler):
    def emit(self, record):                                       # noqa: D401 — a logging.Handler hook
        try:
            msg = record.getMessage()
        except Exception:                                         # noqa: BLE001
            msg = str(record.msg)
        for accel, (name, head) in UPSTREAM_FALLBACK_LOGS.items():
            if record.name == name and head in msg:
                _TRAP["records"].append((accel, msg))


def trap_upstream_logs() -> None:
    """Attach the trap to upstream's loggers BEFORE the upstream package is imported (its hub-kernel fallback logs at import). Idempotent."""
    if _TRAP["installed"]:
        return
    h = _Trap(level=logging.DEBUG)
    for _accel, (name, _head) in UPSTREAM_FALLBACK_LOGS.items():
        logging.getLogger(name).addHandler(h)
    _TRAP["installed"] = True


def trapped(accel: str) -> Optional[str]:
    """The last upstream fallback message trapped for `accel`, or None."""
    for a, msg in reversed(_TRAP["records"]):
        if a == accel:
            return msg
    return None


# --------------------------------------------------------------------------------------------------------------------- the reader
def _is_compiled(f) -> bool:
    """A torch.compile wrapper around a function (torch._dynamo's eval-frame callable), as opposed to the eager function."""
    return f is not None and (hasattr(f, "_torchdynamo_orig_callable") or type(f).__module__.startswith("torch._dynamo")
                              or getattr(f, "__module__", "").startswith("torch._dynamo"))


def _read_flash_attn(FA, A, words: dict, details: dict) -> None:
    env = os.environ.get(FLASH_ENV, "1")
    if FA is None or A is None:
        words["flash_attn"] = "absent(E1.model.attention_not_loaded)"
        return
    fn = getattr(FA, "flash_attn_varlen_func", None)
    pkg = sys.modules.get("flash_attn")
    hop = getattr(A, "flash_attention_func", None) is getattr(FA, "flash_attention_func", None) is not None   # attention.py L10 binds the name it calls at L300
    ext = "flash_attn_2_cuda" in sys.modules                                                               # flash_attn_interface imports its CUDA extension at import
    details["flash_attn"] = {FLASH_ENV: env, "varlen_func": _qual(fn), "hop_intact": hop, "flash_attn_2_cuda_loaded": ext,
                             "package_version": getattr(pkg, "__version__", None), "package_file": getattr(pkg, "__file__", None)}
    if fn is None:
        try:
            spec = importlib.util.find_spec("flash_attn")
        except (ImportError, ValueError):
            spec = None
        words["flash_attn"] = "absent(flash_attn_not_installed)" if spec is None else "fallback:flex(flash_attn_import_failed)"
    elif env != "1":
        words["flash_attn"] = f"fallback:flex({FLASH_ENV}={_clean(env, 12)})"
    elif pkg is None or fn is not getattr(pkg, "flash_attn_varlen_func", None):
        words["flash_attn"] = f"fallback:unknown(varlen_func_is_{_clean(_qual(fn))})"
    elif not hop:
        words["flash_attn"] = f"fallback:unknown(flash_attention_func_rebound_{_clean(_qual(getattr(A, 'flash_attention_func', None)))})"
    elif not ext:
        words["flash_attn"] = "fallback:unknown(flash_attn_2_cuda_not_loaded)"
    else:
        words["flash_attn"] = engaged("flash_attn", getattr(pkg, "__version__", "unknown"))


def _read_site(A) -> str:
    """The class attribute the within-sequence layers call through (attention.py L252-253 `self._flash_attn`): upstream's own function, or the
    kit's routed one (engines.e1.kits.attn.adapter `_flash_attn_routed`, which calls flash_attn's varlen kernel itself)."""
    if A is None or not hasattr(A, "Attention"):
        return "unknown(E1.model.attention_not_loaded)"
    f = A.Attention.__dict__.get("_flash_attn")
    mod = getattr(f, "__module__", "") or ""
    if mod == "E1.model.attention":
        return "stock"
    if mod.startswith("engines.e1.kits.attn"):
        return "kit_attn"
    return f"other({_clean(_qual(f))})"


def _read_hub_layernorm(M, words: dict, details: dict) -> None:
    if M is None:
        words["hub_layernorm"] = "absent(E1.modeling_not_loaded)"
        return
    ln = getattr(M, "layer_norm", None)
    if ln is None:
        why = trapped("hub_layernorm") or "get_kernel_failed"
        details["hub_layernorm"] = {"loaded": False, "upstream_warning": trapped("hub_layernorm")}
        words["hub_layernorm"] = f"fallback:torch_rmsnorm({_clean(why.split(';')[0], 60)})"
        return
    fn = getattr(ln, "rms_norm_fn", None)                                            # the function the model calls (modeling.py L89)
    f = getattr(ln, "__file__", None) or getattr(getattr(fn, "__code__", None), "co_filename", "") or ""
    m = _RE_SNAPSHOT.search(f)
    rev = m.group(1) if m else None
    try:
        k = fn.__globals__.get("_layer_norm_fwd_1pass_kernel")                       # the kernel that call reaches (pins.rmsnorm_autotuner reads the same object)
        kind = type(k).__name__
    except Exception:                                                                 # noqa: BLE001
        kind = "unreadable"
    details["hub_layernorm"] = {"loaded": True, "file": f, "rev": rev, "kernel_type": kind}
    if kind != "Autotuner":
        words["hub_layernorm"] = f"fallback:unknown(_layer_norm_fwd_1pass_kernel_is_{_clean(kind, 24)})"
    else:
        words["hub_layernorm"] = engaged("hub_layernorm", (rev or "unknown")[:8])


def _read_flex(FX, words: dict, details: dict) -> None:
    if FX is None:
        words["flex_attention"] = "absent(E1.model.flex_attention_not_loaded)"
        return
    f = getattr(FX, "flex_attention", None)
    torch = sys.modules.get("torch")
    details["flex_attention"] = {"callable": _qual(f), "type": _qual(type(f)) if f is not None else None, "compiled": _is_compiled(f)}
    if f is None:
        words["flex_attention"] = "absent(torch_flex_attention_import_failed)"
    elif _is_compiled(f):
        words["flex_attention"] = engaged("flex_attention", getattr(torch, "__version__", "unknown"))
    else:
        words["flex_attention"] = "fallback:eager(cuda_not_available_at_import)"


def read() -> dict:
    """{words: {accel: word}, site, upstream_says, details} from the objects bound in THIS process. Imports nothing: a module the upstream
    package did not load reads `absent(<module>_not_loaded)`."""
    FA, A = sys.modules.get("E1.model.flash_attention"), sys.modules.get("E1.model.attention")
    M, FX = sys.modules.get("E1.modeling"), sys.modules.get("E1.model.flex_attention")
    words: Dict[str, str] = {}
    details: Dict[str, dict] = {}
    _read_flash_attn(FA, A, words, details)
    _read_hub_layernorm(M, words, details)
    _read_flex(FX, words, details)
    ups = None
    if FA is not None and hasattr(FA, "is_flash_attention_available"):
        try:
            ups = bool(FA.is_flash_attention_available())                            # upstream's own word (corroboration, never the verdict)
        except Exception:                                                             # noqa: BLE001
            ups = None
    return {"words": words, "site": _read_site(A), "upstream_says": ups, "details": details}


# ------------------------------------------------------------------------------------------------------------------- REQUIRE
def expected(mode: str, pins) -> Dict[str, str]:
    """The words the mode's model process must print: the mode table names WHICH accelerators are engaged, the pins module at WHAT version."""
    versions = {"flash_attn": pins.STACK["flash_attn"], "hub_layernorm": pins.KERNEL["rev"][:8], "flex_attention": pins.STACK["torch_version_str"]}
    return {a: engaged(a, versions[a]) for a in modes.kernels_expected(mode)}


def kernels_refusal_reason(mode: str, bad: List[Tuple[str, str, str]]) -> str:
    """The mode's refusal BY NAME when an accelerator its lever set runs on is absent or fell back in this process."""
    return (f"mode {mode}: accelerator(s) not engaged in this process — " + ", ".join(f"{a}={w} (the pinned stock engages {x})" for a, w, x in bad)
            + f" — the mode's levers run on them, so the mode cannot run here (mode off runs the stock's own fallback route); exit {EXIT_NOT_ACTIVE}")


def fell_back(words: Dict[str, str], exp: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """[(accel, word, expected)] for every accelerator the pinned stock engages whose word in this process differs — each NAMED on the
    KERNELS line by its own word (`fallback:<what>(<why>)` | `absent(<why>)`); the process runs on what it bound, nothing is refused."""
    return [(a, words.get(a, "absent(not_read)"), e) for a, e in exp.items() if words.get(a) != e]


def proof(mode: str, route: str, pins, emit: Optional[Callable[[str], None]] = None) -> dict:   # noqa: C901
    """Read and print the KERNELS line (one per model process, before its first forward); returns the record. An accelerator that fell back
    or is absent on this host is worded on the line and recorded under `fell_back`; under a kit mode that is the mode's refusal by name
    (NOT ACTIVE, SystemExit EXIT_NOT_ACTIVE); under mode off the stock runs the path upstream itself takes there and the calling run's
    EXIT line lists it under `kernels_fallback=`."""
    out = emit or (lambda s: print(s, flush=True))
    res = read()
    exp = expected(mode, pins)
    bad = fell_back(res["words"], exp)
    out(report.kernels_line(mode, route, res["words"], res["site"], res["upstream_says"]))
    rec = {"mode": mode, "route": route, "words": res["words"], "site": res["site"], "upstream_says": res["upstream_says"], "expected": exp,
           "fell_back": [list(v) for v in bad], "details": res["details"]}
    _HOOK["result"] = rec
    if bad and mode in modes.KIT_MODES:                     # a kit mode is ALL of its levers: an accelerator its lever set runs on that this process did not
        out(report.not_active_line(kernels_refusal_reason(mode, bad)))   # engage means the mode cannot run here — it REFUSES BY NAME (exit 3) before the first forward,
        sys.stdout.flush()                                  # never a subset under the mode's name. Mode off is the stock untouched: its line names the word, nothing refuses
        raise SystemExit(EXIT_NOT_ACTIVE)
    return rec


def result() -> Optional[dict]:
    """The record of the proof that fired in this process (None before the first scorer construction)."""
    return _HOOK["result"]


PREDICTOR_MODULE = "E1.predictor"                                 # the class every route constructs lives here: E1Scorer builds an E1Predictor (scorer.py), the API's is one


def install(mode: str, route: str, pins, emit: Optional[Callable[[str], None]] = None) -> None:
    """Arm the proof — importing nothing of upstream's: trap upstream's fallback logs, then wrap `E1.predictor.E1Predictor.__init__` now if the
    module is already loaded (the env / API arm, the det runner's pin wrap), else the moment upstream's own import chain loads it (the same
    meta-path finder the autoload uses, `_autoload.Finder`, with this wrap as its action — the tool's import order and the kit's lazy-import
    shim are untouched). The FIRST construction in the process, after the original constructor returns, reads + prints + judges (`proof`); the
    kit's own wraps (its size pin / apply at `E1Scorer` construction) run before the predictor is built, so the objects read are the ones the
    forward calls. Idempotent."""
    if route not in ROUTES:
        raise ValueError(f"route {route!r} not in {ROUTES}")
    if (mode == "off" and route not in STOCK_ROUTES) or (mode != "off" and route != mode):
        raise ValueError(f"route {route!r} is not a route of mode {mode!r} (off: {'|'.join(STOCK_ROUTES)}; a kit mode: its own name)")
    trap_upstream_logs()
    _HOOK["config"] = (mode, route, pins, emit)
    if PREDICTOR_MODULE in sys.modules:
        _wrap(sys.modules[PREDICTOR_MODULE])
        return
    if _HOOK.get("finder") is not None:
        return
    from ._autoload import Finder                                  # stdlib-only; the finder imports nothing until upstream's own import fires it
    f = Finder(mode, on_fire=lambda _trigger: _wrap(sys.modules[PREDICTOR_MODULE]), triggers=(PREDICTOR_MODULE,))
    sys.meta_path.insert(0, f)
    _HOOK["finder"] = f


def _wrap(P) -> None:
    cls = P.E1Predictor
    orig = cls.__init__
    if getattr(orig, "_e1_opt_kernels", False):
        return
    mode, route, pins, emit = _HOOK["config"]

    def __init__(self, *args, **kwargs):
        orig(self, *args, **kwargs)
        if not _HOOK["fired"]:
            _HOOK["fired"] = True
            proof(mode, route, pins, emit)
    __init__._e1_opt_kernels = True
    __init__.__wrapped__ = orig
    cls.__init__ = __init__
    _HOOK["installed"].append((cls, orig))


def armed() -> bool:
    """Is the constructor hook in place (wrapped, or waiting on upstream's import)?"""
    return bool(_HOOK["installed"]) or _HOOK.get("finder") is not None


def _reset_for_tests() -> None:
    for cls, orig in _HOOK["installed"]:
        cls.__init__ = orig
    f = _HOOK.get("finder")
    if f is not None:
        try:
            sys.meta_path.remove(f)
        except ValueError:
            pass
    _HOOK.update({"installed": [], "fired": False, "result": None, "config": None, "finder": None})
    _TRAP.update({"installed": False, "records": []})
