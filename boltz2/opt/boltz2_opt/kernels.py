"""The KERNELS census — the one reader of upstream's accelerators, on every route, with the REQUIRE guard.

Upstream boltz 2.2.1 engages two optional accelerators, both cuEquivariance ops behind the one model flag ``use_kernels``
(``boltz predict --no_kernels`` turns them off; ``Boltz2.setup`` turns them off silently without CUDA or below compute capability 8):

    cueq_triatt   fused triangle attention        primitives.py ``kernel_triangular_attn`` -> cuequivariance_torch.primitives.triangle.triangle_attention
                                                  -> cuequivariance_ops_torch.triangle_attention (the CUDA op ``torch.ops.cuequivariance.triangle_attention``)
    cueq_trimul   fused triangle multiplication   triangular_mult.py ``kernel_triangular_mult`` -> cuequivariance_torch...triangle_multiplicative_update
                                                  -> cuequivariance_ops_torch.triangle_multiplicative_update (Triton kernels)

``trifast`` (``primitives.py``) has no caller at this pin. This module reads what is actually bound and what actually ran, in the process
that runs the model — the stock CLI process (``stock_pred``) and the kits' worker process (``worker_launch``), at the same two sites as the
PHASE instrument — and is the guard that refuses a timed route whose accelerator is absent or fell back. Standard library only at import;
torch and the cuEquivariance modules are touched at the first ``Boltz2.predict_step`` (after upstream's own environment writes and
seeding; the probe runs under ``torch.random.fork_rng`` with a private generator: RNG-neutral), so a stock process may hold it
(stock_pred OWN_MODULES).

One line per pass (process), printed at interpreter exit with the per-call census, and one JSON record beside the run's outputs::

    [boltz2-opt <mode>] KERNELS route=<stock|exact|fast|big|big_x<P>> settings=<defaults|flags> cueq_triatt=<word> cueq_trimul=<word> trifast=n/a-upstream:no_caller <info tokens> verdict=<PASS|REFUSED(<accel>,..)|NO-STEP>

    word := engaged:<backend>@<version>-<build>[served=<k>,byrule=<m>(<reason>=<n>;..)]   the library served k calls on its kernels; m calls took its torch
                                                                                        reference path BY ITS OWN DOCUMENTED RULES (S <= CUEQ_TRIATTN_FALLBACK_THRESHOLD /
                                                                                        CUEQ_TRIMUL_FALLBACK_THRESHOLD, dtype / hidden-dim rules) — named, counted, admissible
          | off-by-route:<reason>          the route turns the accelerator off or replaces it by name (``--no_kernels``; ``replaced_by_rowpair`` at n_gpu > 1)
          | absent:<what>                  a distribution / module missing or failing to import
          | fallback:<reason>              use_kernels resolved off on a route that expects on; the functional probe failed; the reference path taken for a
                                            call the rules say is served; a fallback signal (warning / log record matching /fall(ing)?back/) at S > threshold
          | n/a-upstream:<reason>          not an accelerator upstream engages at this pin (informative, never judged)

The expected word kind per accelerator and route is ONE function of the mode table (``modes.kernels_expected``); the parent process passes it
on the command line (``stock_pred --kernels-expect``, ``worker_launch --kernels-expect``). REQUIRE: at the first ``predict_step``, before any
timed work, an accelerator expected ``engaged`` that reads ``absent`` / ``fallback`` prints ``[boltz2-opt <mode>] KERNELS-REFUSED route=<r>
<accel>=<word> (expected <kind>); exit 5`` and the process exits 5 (``EXIT_KERNELS``); a per-call violation (reference path at S > threshold,
fallback signal at S > threshold) refuses at that call the same way; a contradiction visible only in the totals (an ``off-by-route``
accelerator that was called, a verdict other than PASS) is refused by the parent from the line / the JSON (``cli`` / ``worker``: exit 5).
"""
from __future__ import annotations

import atexit
import importlib
import importlib.metadata as md
import importlib.util
import json
import logging
import os
import re
import sys
import threading
import time
import warnings
from typing import Dict, List, Optional, Tuple

TAG = "boltz2-opt"
EXIT_KERNELS = 5                                    # the REQUIRE guard's exit code (report.EXIT_KERNELS restates it): an expected accelerator absent / fell back
MODEL_MODULE = "boltz.model.models.boltz2"          # the census attaches to Boltz2.predict_step (the PHASE instrument's own boundary)
ACCELERATORS = ("cueq_triatt", "cueq_trimul")       # judged words, in line order: what upstream boltz 2.2.1 engages
INFORMATIVE = {"trifast": "n/a-upstream:no_caller"}  # printed after the judged words, never judged: primitives.py:369-410 `_trifast_attn` has no caller at 2.2.1
EXPECT_KINDS = ("engaged", "off-by-route")          # what a route may expect of an accelerator
WORD_KINDS = ("engaged", "off-by-route", "absent", "fallback", "n/a-upstream")
ROUTE_RE = re.compile(r"^(stock|default|exact|fast|big|big_x\d+)$")
LINE_MARK = "KERNELS route="                        # one cross-engine grep finds every pass
REFUSED_MARK = "KERNELS-REFUSED route="             # the point-of-refusal line (in-process exit 5); never counted as a pass line
LINE_RE = re.compile(r"^\[boltz2-opt (?P<mode>\S+)\] KERNELS route=(?P<route>\S+) settings=(?P<settings>\S+) (?P<rest>.*?) verdict=(?P<verdict>\S+)\s*$")
FALLBACK_RE = re.compile(r"fall(?:ing)?[\s_-]*back", re.IGNORECASE)   # upstream's / the library's own runtime fallback words (warnings + logging records)
SIGNAL_SOURCES = ("cuequivariance", "boltz")        # module names / file paths whose warnings and log records the trap reads
PROBE_TOKENS = 128                                  # above both library thresholds (100 by default): the probe reaches the CUDA / Triton kernels, not the reference paths

OPS = {   # per accelerator: upstream's call site, the front-end it imports at call time, the ops module whose public entry the front-end imports at
          # call time (`from cuequivariance_ops_torch import <public>` inside the front-end function, cuequivariance_torch/primitives/triangle.py:128-135,
          # 238-262) and whose module-global reference function the public entry calls by name when its rules route a call to torch
          # (cuequivariance_ops_torch/triangle_attention.py:735-737, triangle_multiplicative_update.py:353-377) — the two counting points
    "cueq_triatt": {"site": ("boltz.model.layers.triangular_attention.primitives", "kernel_triangular_attn"),
                    "frontend": ("cuequivariance_torch.primitives.triangle", "triangle_attention"),
                    "ops_module": "cuequivariance_ops_torch.triangle_attention", "public": "triangle_attention", "reference": "_triangle_attention_torch",
                    "threshold_attr": "CUEQ_TRIATTN_FALLBACK_THRESHOLD", "threshold_env": "CUEQ_TRIATTN_FALLBACK_THRESHOLD"},
    "cueq_trimul": {"site": ("boltz.model.layers.triangular_mult", "kernel_triangular_mult"),
                    "frontend": ("cuequivariance_torch.primitives.triangle", "triangle_multiplicative_update"),
                    "ops_module": "cuequivariance_ops_torch.triangle_multiplicative_update", "public": "triangle_multiplicative_update", "reference": "_tri_mul_torch",
                    "threshold_attr": "CUEQ_TRIMUL_FALLBACK_THRESHOLD", "threshold_env": "CUEQ_TRIMUL_FALLBACK_THRESHOLD"},
}
OPS_PACKAGE = "cuequivariance_ops_torch"
DISTS = {"frontend": ("cuequivariance-torch",), "ops_torch": ("cuequivariance-ops-torch-cu13", "cuequivariance-ops-torch-cu12"),
         "ops": ("cuequivariance-ops-cu13", "cuequivariance-ops-cu12"), "core": ("cuequivariance",)}
CUBIN_LIB = ("cuequivariance_ops", os.path.join("lib", "libcue_ops.so"))   # the CUDA kernels' shared object: its embedded cubin tags (sm_90 …) are the device coverage

_LOCK = threading.RLock()
_CUR = threading.local()                              # the accelerator / sequence length of the public call in flight (the signal trap and the reference counter read it)
_STATE: Dict = {"installed": False, "mode": None, "route": None, "settings": None, "n_gpu": 1, "rank": None, "expected": {}, "json_path": None,
                "static": None, "imports": {}, "use_kernels": None, "stepped": False, "counted": False, "probe": {}, "signals": [], "notes": [], "refused": [],
                "device": None, "has_sm100f": None, "do_probe": True, "line": None, "printed": False, "t_install": None, "orig_step": None}
_COUNTS: Dict[str, dict] = {}


_STATE0 = json.loads(json.dumps({k: v for k, v in _STATE.items()}))


def _reset_counts() -> None:
    for acc in ACCELERATORS:
        _COUNTS[acc] = {"calls": 0, "served": 0, "byrule": {}, "errors": 0, "violations": 0, "violation_first": None}


def reset() -> None:   # tests only: a fresh, un-installed census in this process (restores a wrapped predict_step if the class is still reachable)
    with _LOCK:
        _STATE.clear(); _STATE.update(json.loads(json.dumps(_STATE0))); _STATE["orig_step"] = None
        _reset_counts()


_reset_counts()


# ---------------------------------------------------------------- expectations (the parent computes them from modes.py; the child parses) ----------------------------------------------------------------
def parse_expect(spec: str) -> Dict[str, str]:
    """``cueq_triatt=engaged,cueq_trimul=off-by-route:--no_kernels`` -> {accelerator: expected word}; every ACCELERATORS name exactly once, kinds in EXPECT_KINDS."""
    out: Dict[str, str] = {}
    for tok in (t for t in (spec or "").split(",") if t.strip()):
        name, sep, word = tok.strip().partition("=")
        if not sep or name not in ACCELERATORS:
            raise ValueError(f"--kernels-expect: unknown accelerator {name!r} (the accelerators are {','.join(ACCELERATORS)})")
        if word_kind(word) not in EXPECT_KINDS:
            raise ValueError(f"--kernels-expect: {name}={word!r}: an expectation is one of {'|'.join(EXPECT_KINDS)}[:reason]")
        if name in out:
            raise ValueError(f"--kernels-expect: {name} given twice")
        out[name] = word
    missing = [a for a in ACCELERATORS if a not in out]
    if missing:
        raise ValueError(f"--kernels-expect: no expectation for {','.join(missing)}")
    return out


def format_expect(expected: Dict[str, str]) -> str:
    return ",".join(f"{a}={expected[a]}" for a in ACCELERATORS)


def word_kind(word: str) -> str:
    k = (word or "").split(":", 1)[0].split("[", 1)[0]
    return k if k in WORD_KINDS else "?"


def route_word(mode: str, n_gpu: int = 1) -> str:
    """The line's route token: ``stock`` on the stock CLI (mode off), the mode name on the worker route, ``big_x<P>`` at n_gpu > 1."""
    if mode == "off":
        return "stock"
    return f"{mode}_x{int(n_gpu)}" if int(n_gpu or 1) > 1 else mode


# ---------------------------------------------------------------- static facts (no library import) ----------------------------------------------------------------
def dist_versions() -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    for names in DISTS.values():
        for n in names:
            try:
                out[n] = md.version(n)
            except md.PackageNotFoundError:
                out[n] = None
    return out


def _build_tag(dists: Dict[str, Optional[str]]) -> Optional[str]:
    for n in DISTS["ops_torch"]:
        if dists.get(n):
            return n.rsplit("-", 1)[-1]             # cu13 | cu12
    return None


_CUBIN_CACHE: Dict[str, List[str]] = {}


def cubin_sm_tags(path: Optional[str]) -> List[str]:
    """The ``sm_<NN>[a|f]`` tags embedded in the kernels' shared object (the architectures it carries cubins for); [] when unreadable."""
    if not path or not os.path.isfile(path):
        return []
    if path in _CUBIN_CACHE:
        return _CUBIN_CACHE[path]
    tags = set()
    try:
        with open(path, "rb") as fh:
            tail = b""
            for chunk in iter(lambda: fh.read(1 << 22), b""):
                buf = tail + chunk
                tags.update(m.decode() for m in re.findall(rb"sm_(\d{2,3}[af]?)", buf))
                tail = buf[-8:]
    except OSError:
        return []
    _CUBIN_CACHE[path] = sorted(tags, key=lambda s: (int(re.match(r"\d+", s).group()), s))
    return _CUBIN_CACHE[path]


def static_facts() -> dict:
    """Distributions, module specs (find_spec: located, not imported), the ops library's cubin coverage — what the box carries, before torch is touched."""
    dists = dist_versions()
    specs = {}
    for name in (OPS_PACKAGE, "cuequivariance_ops", "cuequivariance_torch", OPS["cueq_triatt"]["frontend"][0]) + tuple(o["ops_module"] for o in OPS.values()):
        try:
            s = importlib.util.find_spec(name)
            specs[name] = s.origin if s else None
        except (ImportError, ValueError) as e:   # a parent package that fails to import
            specs[name] = f"error:{type(e).__name__}:{e}"
    lib = None
    try:
        s = importlib.util.find_spec(CUBIN_LIB[0])
        if s and s.origin:
            cand = os.path.join(os.path.dirname(s.origin), CUBIN_LIB[1])
            lib = cand if os.path.isfile(cand) else None
    except (ImportError, ValueError):
        lib = None
    return {"dists": dists, "build": _build_tag(dists), "version": dists.get("cuequivariance-torch") or next((v for v in dists.values() if v), None),
            "specs": specs, "cubin_lib": lib, "cubin_sm_tags": cubin_sm_tags(lib)}


# ---------------------------------------------------------------- the signal trap (warnings + logging) ----------------------------------------------------------------
class _LogTrap(logging.Handler):
    def emit(self, record):   # noqa: D401 — a fallback word from the library / upstream is a named event, at S > threshold a refusal
        try:
            msg = record.getMessage()
        except Exception:  # noqa: BLE001
            return
        if any(record.name.startswith(s) or s in (record.pathname or "") for s in SIGNAL_SOURCES):
            _on_signal(f"log:{record.name}", msg)


def _install_trap() -> None:
    warnings.filterwarnings("always", message=r".*[Ff]all(ing)?[\s_-]*back.*")   # a fallback-class warning is always shown (and so always trapped), never deduplicated away
    prev = warnings.showwarning

    def show(message, category, filename, lineno, file=None, line=None):
        text = str(message)
        if any(s in (filename or "") for s in SIGNAL_SOURCES) or FALLBACK_RE.search(text):
            _on_signal(f"warning:{os.path.basename(filename or '?')}:{lineno}", text)
        return prev(message, category, filename, lineno, file, line)

    warnings.showwarning = show
    h = _LogTrap(level=logging.DEBUG)
    for name in ("cuequivariance", "cuequivariance_ops", "cuequivariance_torch", "cuequivariance_ops_torch", "boltz"):
        logging.getLogger(name).addHandler(h)


def _on_signal(source: str, text: str) -> None:
    """A warning / log record from the library or upstream: a fallback word is recorded with the call in flight; at S above the accelerator's
    threshold on a route that expects it engaged it is a refusal at once. Anything else from those sources is a note."""
    acc = getattr(_CUR, "acc", None); S = getattr(_CUR, "S", None)
    rec = {"source": source, "message": text[:300], "acc": acc, "S": S, "utc": _utc()}
    if not FALLBACK_RE.search(text):
        if len(_STATE["notes"]) < 50 and not any(n["message"] == rec["message"] for n in _STATE["notes"]):
            _STATE["notes"].append(rec)
        return
    _STATE["signals"].append(rec)
    thr = _threshold(acc) if acc else None
    if acc and S is not None and thr is not None and int(S) > int(thr) and word_kind(_STATE["expected"].get(acc, "engaged")) == "engaged":
        _violation(acc, f"signal@S={S}>{thr}:{_slug(text)}")


def _slug(text: str, n: int = 60) -> str:
    return re.sub(r"[^A-Za-z0-9_.<=>@/+-]+", "_", text.strip())[:n].strip("_")


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# ---------------------------------------------------------------- the per-call census (counting wrappers at the ops layer) ----------------------------------------------------------------
def _threshold(acc: Optional[str]) -> Optional[int]:
    if not acc:
        return None
    mod = sys.modules.get(OPS[acc]["ops_module"])
    v = getattr(mod, OPS[acc]["threshold_attr"], None) if mod is not None else None
    try:
        return int(v) if v is not None else int(os.environ.get(OPS[acc]["threshold_env"], "100"))
    except (TypeError, ValueError):
        return None


def _seq_len(acc: str, args, kwargs) -> Optional[int]:
    t = args[0] if args else kwargs.get("q" if acc == "cueq_triatt" else "x")
    try:
        return int(t.shape[-2])                      # triangle attention: S_qo = q.shape[-2] (B,N,H,S,D); TriMul: seq_len = x.shape[-2] (B,N,N,D)
    except Exception:  # noqa: BLE001
        return None


def byrule_reason(acc: str, args, kwargs) -> Tuple[str, bool]:
    """(reason, admissible) for a call the library routed to its torch reference path: admissible when its DOCUMENTED rules say so —
    S <= threshold (cuequivariance_ops_torch/triangle_attention.py:735-737, triangle_multiplicative_update.py:353-377); for triangle attention
    also the dtype / hidden-dim support rules (cuequivariance_torch/primitives/triangle.py:84: hidden_dim <= 32 and % 4 for fp32, <= 128 and % 8
    for bf16/fp16; hidden_dim < 32 raises the threshold to 200) — otherwise the call should have been served: a violation."""
    S = _seq_len(acc, args, kwargs); thr = _threshold(acc)
    if S is not None and thr is not None and S <= thr:
        return f"S<={thr}", True
    if acc == "cueq_triatt":
        q = args[0] if args else kwargs.get("q")
        try:
            hd = int(q.shape[-1]); dt = str(q.dtype).replace("torch.", "")
        except Exception:  # noqa: BLE001
            return f"reference@S={S}", False
        if dt in ("bfloat16", "float16") and (hd > 128 or hd % 8):
            return f"hidden_dim={hd}_{dt}", True
        if dt == "float32" and (hd > 32 or hd % 4):
            return f"hidden_dim={hd}_{dt}", True
        if hd < 32 and S is not None and S <= 200:
            return f"S<=200_hidden_dim={hd}", True
        if kwargs.get("return_aux"):
            return f"reference@S={S}_return_aux", False
    return f"reference@S={S}>{thr}", False


def _violation(acc: str, detail: str) -> None:
    c = _COUNTS[acc]
    c["violations"] += 1
    if c["violation_first"] is None:
        c["violation_first"] = detail
    if word_kind(_STATE["expected"].get(acc, "engaged")) == "engaged" and _STATE["installed"]:
        refuse([(acc, f"fallback:{detail}", _STATE["expected"].get(acc, "engaged"))])


def _wrap_public(acc: str, fn):
    if getattr(fn, "_kernels_census", None):
        return fn

    def public(*args, **kwargs):
        c = _COUNTS[acc]
        prev = (getattr(_CUR, "acc", None), getattr(_CUR, "S", None), getattr(_CUR, "ref", False))
        _CUR.acc, _CUR.S, _CUR.ref = acc, _seq_len(acc, args, kwargs), False
        c["calls"] += 1
        try:
            out = fn(*args, **kwargs)
        except BaseException:
            c["errors"] += 1                          # a kernel that raises is LOUD (the exception propagates); counted, never served
            raise
        else:
            if not _CUR.ref:
                c["served"] += 1
            return out
        finally:
            _CUR.acc, _CUR.S, _CUR.ref = prev

    public._kernels_census = acc; public.__wrapped__ = fn; public.__name__ = getattr(fn, "__name__", "public"); public.__doc__ = fn.__doc__
    return public


def _wrap_reference(acc: str, fn):
    if getattr(fn, "_kernels_census", None):
        return fn

    def reference(*args, **kwargs):
        reason, ok = byrule_reason(acc, args, kwargs)
        c = _COUNTS[acc]
        if getattr(_CUR, "acc", None) == acc:
            _CUR.ref = True
        else:                                         # a reference call that did not come through the counted public entry: counted as a call of its own
            c["calls"] += 1
        if ok:
            c["byrule"][reason] = c["byrule"].get(reason, 0) + 1
        else:
            _violation(acc, reason)
        return fn(*args, **kwargs)

    reference._kernels_census = acc; reference.__wrapped__ = fn; reference.__name__ = getattr(fn, "__name__", "reference")
    return reference


def install_counters() -> Dict[str, str]:
    """Import the front-end and ops modules and place the counting wrappers: the ops package's public entry (what the front-end imports at every
    call) and the ops module's reference function (what the public entry calls by name when its rules route to torch). Returns
    {accelerator: '' | 'absent:<what>' | 'fallback:<what>'} — the import verdict."""
    verdict: Dict[str, str] = {}
    try:
        pkg = importlib.import_module(OPS_PACKAGE)
    except Exception as e:  # noqa: BLE001 — a missing / broken ops package is the word `absent`, named
        for acc in ACCELERATORS:
            verdict[acc] = f"absent:{OPS_PACKAGE}({type(e).__name__})"
        _STATE["imports"][OPS_PACKAGE] = f"{type(e).__name__}: {e}"[:300]
        return verdict
    _STATE["imports"][OPS_PACKAGE] = getattr(pkg, "__file__", None)
    trimul_stub = getattr(pkg, "IMPORT_EXCEPTION", None)   # cuequivariance_ops_torch/__init__.py:56-75: the Triton component failed to import — its name is a stub that raises at call
    for acc in ACCELERATORS:
        o = OPS[acc]
        try:
            fe = importlib.import_module(o["frontend"][0])
            if not callable(getattr(fe, o["frontend"][1], None)):
                raise ImportError(f"{o['frontend'][0]} has no {o['frontend'][1]}")
            _STATE["imports"][o["frontend"][0]] = getattr(fe, "__file__", None)
        except Exception as e:  # noqa: BLE001
            verdict[acc] = f"absent:{o['frontend'][0]}({type(e).__name__})"; _STATE["imports"][o["frontend"][0]] = f"{type(e).__name__}: {e}"[:300]; continue
        if acc == "cueq_trimul" and trimul_stub:
            verdict[acc] = "fallback:triton_component_import_failed"; _STATE["imports"][o["ops_module"]] = str(trimul_stub)[-300:]; continue
        try:
            sub = importlib.import_module(o["ops_module"])
        except Exception as e:  # noqa: BLE001
            verdict[acc] = f"absent:{o['ops_module']}({type(e).__name__})"; _STATE["imports"][o["ops_module"]] = f"{type(e).__name__}: {e}"[:300]; continue
        _STATE["imports"][o["ops_module"]] = getattr(sub, "__file__", None)
        pub = getattr(pkg, o["public"], None); ref = getattr(sub, o["reference"], None)
        if not callable(pub) or not callable(ref):
            verdict[acc] = f"absent:{OPS_PACKAGE}.{o['public']}" if not callable(pub) else f"absent:{o['ops_module']}.{o['reference']}"; continue
        wpub = _wrap_public(acc, pub)
        setattr(pkg, o["public"], wpub)
        if getattr(sub, o["public"], None) is pub:
            setattr(sub, o["public"], wpub)
        setattr(sub, o["reference"], _wrap_reference(acc, ref))
        verdict[acc] = ""
    return verdict


def _imported_after_install(acc: str) -> bool:
    """On a route that expects the accelerator off nothing of this tree imports the ops module; upstream imports it only inside a kernel call
    (primitives.py:201, triangular_mult.py:22): its arrival in sys.modules after install is a call by upstream — the contradiction's evidence."""
    return OPS[acc]["ops_module"] in sys.modules and OPS[acc]["ops_module"] not in (_STATE.get("modules_at_install") or ())


def bound_sites() -> Dict[str, Optional[str]]:
    """What is bound at upstream's two call sites right now: ``<module>:<qualname>@<file>`` of ``primitives.kernel_triangular_attn`` /
    ``triangular_mult.kernel_triangular_mult`` (upstream's own functions on every route of this tree; a replacement would show here)."""
    out: Dict[str, Optional[str]] = {}
    for acc, o in OPS.items():
        mod = sys.modules.get(o["site"][0])
        fn = getattr(mod, o["site"][1], None) if mod is not None else None
        if fn is None:
            out[acc] = None; continue
        code = getattr(fn, "__code__", None)
        out[acc] = f"{getattr(fn, '__module__', '?')}:{getattr(fn, '__qualname__', '?')}@{code.co_filename if code else '?'}"
    return out


# ---------------------------------------------------------------- the functional probe (one call per accelerator through upstream's own call site) ----------------------------------------------------------------
def probe(device=None) -> Dict[str, dict]:
    """One call of each accelerator through upstream's bound call-site function at PROBE_TOKENS (> both thresholds) on the CUDA device, under
    ``torch.random.fork_rng`` with a private generator (RNG-neutral) and bf16 autocast (the stock predict context): {accelerator: {ok, path:
    served|byrule|error, ms, error}}. The counters see the calls; they are subtracted afterwards (the census counts the model's calls only)."""
    import torch
    out: Dict[str, dict] = {}
    dev = torch.device(device if device is not None else "cuda")
    S, H, C, D = PROBE_TOKENS, 4, 32, 128
    devs = [dev.index if dev.index is not None else torch.cuda.current_device()] if dev.type == "cuda" else []
    with torch.random.fork_rng(devices=devs), torch.no_grad():
        g = torch.Generator(device=dev); g.manual_seed(0)
        for acc in ACCELERATORS:
            before = {k: (dict(v) if isinstance(v, dict) else v) for k, v in _COUNTS[acc].items()}
            t0 = time.perf_counter(); rec = {"ok": False, "path": None, "ms": None, "error": None, "S": S}
            try:
                mod = sys.modules.get(OPS[acc]["site"][0]) or importlib.import_module(OPS[acc]["site"][0])
                fn = getattr(mod, OPS[acc]["site"][1])
                with torch.autocast(dev.type, dtype=torch.bfloat16):
                    if acc == "cueq_triatt":          # primitives.py:341-357: q,k,v [B,I,H,J,c_hidden] (unscaled), tri_bias [B,1,H,I,J], mask [B,I,1,1,J] bool, scale 1/sqrt(c_hidden)
                        q = torch.randn(1, S, H, S, C, device=dev, dtype=torch.bfloat16, generator=g)
                        k = torch.randn(1, S, H, S, C, device=dev, dtype=torch.bfloat16, generator=g)
                        v = torch.randn(1, S, H, S, C, device=dev, dtype=torch.bfloat16, generator=g)
                        bias = torch.randn(1, 1, H, S, S, device=dev, dtype=torch.float32, generator=g)
                        mask = torch.ones(1, S, 1, 1, S, device=dev, dtype=torch.bool)
                        o = fn(q, k, v, bias, mask, 1.0 / (C ** 0.5))
                    else:                             # triangular_mult.py:91-104: x [B,N,N,D], mask [B,N,N], the module's LayerNorm / Linear weights
                        x = torch.randn(1, S, S, D, device=dev, dtype=torch.float32, generator=g)
                        mask = torch.ones(1, S, S, device=dev, dtype=torch.float32)
                        w = lambda *shape: torch.randn(*shape, device=dev, dtype=torch.float32, generator=g) * 0.02  # noqa: E731
                        o = fn(x, direction="outgoing", mask=mask, norm_in_weight=torch.ones(D, device=dev), norm_in_bias=torch.zeros(D, device=dev),
                               p_in_weight=w(2 * D, D), g_in_weight=w(2 * D, D), norm_out_weight=torch.ones(D, device=dev), norm_out_bias=torch.zeros(D, device=dev),
                               p_out_weight=w(D, D), g_out_weight=w(D, D), eps=1e-5)
                    if dev.type == "cuda":
                        torch.cuda.synchronize(dev)
                if not bool(torch.isfinite(o.float()).all()):
                    raise RuntimeError("non-finite output")
                after = _COUNTS[acc]
                d_served = after["served"] - before["served"]; d_byrule = sum(after["byrule"].values()) - sum(before["byrule"].values())
                rec.update(ok=d_served == 1 and d_byrule == 0, path="served" if d_served == 1 else ("byrule" if d_byrule else "uncounted"))
                if not rec["ok"]:
                    rec["error"] = f"probe call at S={S} took path={rec['path']} (served={d_served} byrule={d_byrule})"
            except BaseException as e:  # noqa: BLE001 — the probe's failure is a word (fallback:probe:…), never a crash of its own
                rec.update(ok=False, path="error", error=f"{type(e).__name__}: {str(e).strip()[:200]}")
            rec["ms"] = round((time.perf_counter() - t0) * 1000.0, 1)
            _COUNTS[acc] = before                     # the probe's own call is not the model's: subtracted
            _COUNTS[acc]["byrule"] = dict(before["byrule"])
            out[acc] = rec
    return out


# ---------------------------------------------------------------- verdicts, the line, the record ----------------------------------------------------------------
def words(final: bool = True) -> Dict[str, str]:
    """The word per accelerator from everything read so far (imports, use_kernels, probe, counters, violations) against the expectation."""
    st = _STATE; facts = st["static"] or {}
    ver = facts.get("version") or "?"; build = facts.get("build") or "?"
    out: Dict[str, str] = {}
    for acc in ACCELERATORS:
        exp = st["expected"].get(acc, "engaged"); c = _COUNTS[acc]; imp = st["imports"].get(f"verdict:{acc}")
        census = f"[served={c['served']},byrule={sum(c['byrule'].values())}" + (f"({';'.join(f'{k}={v}' for k, v in sorted(c['byrule'].items()))})" if c["byrule"] else "") + \
                 (f",errors={c['errors']}" if c["errors"] else "") + "]"
        engaged = f"engaged:{OPS_PACKAGE}@{ver}-{build}{census}"
        spec = (facts.get("specs") or {}).get(OPS[acc]["ops_module"])
        if word_kind(exp) == "off-by-route":            # the route turns the accelerator off / replaces it: the word stands unless something called it anyway
            called = c["calls"] > 0 or (not st["counted"] and st["stepped"] and _imported_after_install(acc))
            out[acc] = exp if not called else (engaged if st["counted"] else f"engaged:{OPS_PACKAGE}@{ver}-{build}[imported]")
            continue
        if imp:                                          # absent:<what> / fallback:<import problem>
            out[acc] = imp; continue
        if not spec or str(spec).startswith("error:"):
            out[acc] = f"absent:{OPS[acc]['ops_module']}"; continue
        if st["use_kernels"] is False:
            out[acc] = "fallback:use_kernels=False(boltz2.py:361-366)"; continue
        pr = st["probe"].get(acc)
        if pr is not None and not pr.get("ok"):
            out[acc] = f"fallback:probe:{_slug(pr.get('error') or pr.get('path') or '?')}"; continue
        if c["violation_first"]:
            out[acc] = f"fallback:{c['violation_first']}{census}"; continue
        out[acc] = engaged
    return out


def judge(expected: Dict[str, str], got: Dict[str, str]) -> List[Tuple[str, str, str]]:
    """[(accelerator, word, expected)] for every accelerator whose word kind is not the expected kind, or whose off-by-route reason differs."""
    bad = []
    for acc in ACCELERATORS:
        exp = expected.get(acc, "engaged"); w = got.get(acc, "absent:unread")
        if word_kind(w) != word_kind(exp) or (word_kind(exp) == "off-by-route" and w != exp):
            bad.append((acc, w, exp))
    return bad


def info_tokens() -> List[str]:
    """The line's informative tokens after the judged words: trifast (n/a), the live use_kernels, the device's compute capability and whether the
    ops library carries a cubin for it, the sm100f build flag, the probe outcome, the library thresholds, the fallback-signal count, n_gpu / rank."""
    st = _STATE; facts = st["static"] or {}
    cc = (st.get("device") or {}).get("cc"); tags = facts.get("cubin_sm_tags") or []
    sm = ("sm%d%d" % (cc[0], cc[1])) if cc else None
    have = {t.rstrip("af") for t in tags}
    toks = [f"{k}={v}" for k, v in INFORMATIVE.items()]
    toks += [f"use_kernels={st['use_kernels'] if st['use_kernels'] is not None else '-'}",
             f"device={sm or '-'}",
             f"cubin={sm}:{'yes' if sm[2:] in have else 'NO'}" if sm else f"cubin={'-' if not tags else 'unprobed'}",
             f"sm100f_build={st['has_sm100f'] if st['has_sm100f'] is not None else '-'}",
             "probe=" + (",".join(f"{a.split('_', 1)[1]}:{'ok' if r.get('ok') else r.get('path')}@{r.get('ms')}ms" for a, r in st["probe"].items()) if st["probe"] else ("skipped" if st["stepped"] else "-")),
             "thresholds=" + ",".join(f"{a.split('_', 1)[1]}<={_threshold(a) if _threshold(a) is not None else '?'}" for a in ACCELERATORS),
             f"signals={len(st['signals'])}", f"n_gpu={st['n_gpu']}"] + ([f"rank={st['rank']}"] if st["rank"] is not None else [])
    return toks


def format_line(mode: str, route: str, settings: str, got: Dict[str, str], verdict: str, extra: Optional[List[str]] = None) -> str:
    toks = [f"{a}={got[a]}" for a in ACCELERATORS] + list(extra or [])
    return f"[{TAG} {mode}] {LINE_MARK}{route} settings={settings} " + " ".join(toks) + f" verdict={verdict}"


def parse_line(line: str) -> Optional[dict]:
    """{mode, route, settings, words{accelerator: word}, tokens{key: value}, verdict} of a KERNELS line; None when it is not one."""
    m = LINE_RE.match(line.strip())
    if not m:
        return None
    toks = dict(t.split("=", 1) for t in m.group("rest").split() if "=" in t)
    return {"mode": m.group("mode"), "route": m.group("route"), "settings": m.group("settings"), "verdict": m.group("verdict"),
            "words": {a: toks.pop(a, None) for a in ACCELERATORS}, "tokens": toks}


def find_lines(text: str) -> List[dict]:
    """Every KERNELS pass line in a transcript, parsed (REFUSED point lines are not pass lines)."""
    return [p for p in (parse_line(l) for l in (text or "").splitlines() if LINE_MARK in l and REFUSED_MARK not in l) if p]


def verdict_word(bad: List[Tuple[str, str, str]]) -> str:
    if not _STATE["stepped"]:
        return "NO-STEP"
    return "PASS" if not bad else "REFUSED(" + ",".join(a for a, _w, _e in bad) + ")"


def record() -> dict:
    got = words(); bad = judge(_STATE["expected"], got)
    return {"tag": TAG, "mode": _STATE["mode"], "route": _STATE["route"], "settings": _STATE["settings"], "n_gpu": _STATE["n_gpu"], "rank": _STATE["rank"],
            "expected": dict(_STATE["expected"]), "words": got, "verdict": verdict_word(bad), "refused": [{"accelerator": a, "word": w, "expected": e} for a, w, e in bad],
            "refused_in_process": list(_STATE["refused"]), "stepped": _STATE["stepped"], "use_kernels": _STATE["use_kernels"], "counts": json.loads(json.dumps(_COUNTS)),
            "probe": _STATE["probe"], "signals": _STATE["signals"], "notes": _STATE["notes"][:20], "static": _STATE["static"], "imports": {k: v for k, v in _STATE["imports"].items() if not k.startswith("verdict:")},
            "sites": bound_sites(), "device": _STATE.get("device"), "has_sm100f_support": _STATE.get("has_sm100f"),
            "thresholds": {OPS[a]["threshold_env"]: _threshold(a) for a in ACCELERATORS}, "line": None, "written_utc": _utc(), "written_unix": time.time(), "pid": os.getpid()}


def emit(file=None) -> dict:
    """Print the ONE pass line (idempotent) and write the JSON record; returns the record."""
    with _LOCK:
        rec = record()
        line = format_line(_STATE["mode"], _STATE["route"], _STATE["settings"], rec["words"], rec["verdict"], info_tokens())
        rec["line"] = line; _STATE["line"] = line
        if not _STATE["printed"]:
            print(line, file=file or sys.stdout, flush=True); _STATE["printed"] = True
        if _STATE["json_path"]:
            try:
                os.makedirs(os.path.dirname(os.path.abspath(_STATE["json_path"])), exist_ok=True)
                tmp = _STATE["json_path"] + ".tmp"
                with open(tmp, "w") as fh:
                    json.dump(rec, fh, indent=1, default=str)
                os.replace(tmp, _STATE["json_path"])
            except OSError as e:
                print(f"[{TAG} {_STATE['mode']}] KERNELS-NOTE census record not written ({_STATE['json_path']}): {e}", file=sys.stderr, flush=True)
        return rec


def refuse(bad: List[Tuple[str, str, str]]) -> None:
    """The REQUIRE guard's refusal: name accelerator + word + expectation + route, write the record, exit EXIT_KERNELS (SystemExit: atexit — the
    pass line, the worker's own log dump — still runs)."""
    _STATE["refused"].extend({"accelerator": a, "word": w, "expected": e, "utc": _utc()} for a, w, e in bad)
    detail = " ".join(f"{a}={w} (expected {e})" for a, w, e in bad)
    print(f"[{TAG} {_STATE['mode']}] {REFUSED_MARK}{_STATE['route']} settings={_STATE['settings']} {detail}; exit {EXIT_KERNELS}", flush=True)
    print(f"[{TAG} {_STATE['mode']}] {REFUSED_MARK}{_STATE['route']} {detail}; exit {EXIT_KERNELS}", file=sys.stderr, flush=True)
    raise SystemExit(EXIT_KERNELS)


# ---------------------------------------------------------------- install (the two sites: stock_pred after its proof, worker_launch after the model import) ----------------------------------------------------------------
def install(route: str, expected: Dict[str, str], settings: str = "-", mode: Optional[str] = None, n_gpu: int = 1, rank: Optional[int] = None,
            json_path: Optional[str] = None, model_cls=None, do_probe: bool = True) -> bool:
    """Arm the census in this process: static facts, the signal trap, the exit-time pass line, and the wrap of ``Boltz2.predict_step`` whose FIRST
    call imports the library, places the counters, runs the probe, reads the live ``use_kernels`` and enforces REQUIRE before delegating
    (chained over whatever is installed — the PHASE wrapper — and idempotent). ``model_cls`` defaults to ``boltz.model.models.boltz2.Boltz2``."""
    if not ROUTE_RE.match(route or ""):
        raise ValueError(f"route {route!r}: the routes are stock|default|exact|fast|big|big_x<P>")
    with _LOCK:
        if _STATE["installed"]:
            return False
        _STATE.update(installed=True, route=route, expected=dict(expected), settings=settings or "-", mode=mode or ("stock" if route == "stock" else route.split("_x")[0]),
                      n_gpu=int(n_gpu or 1), rank=rank, json_path=json_path, t_install=_utc(), do_probe=bool(do_probe))
        _STATE["static"] = static_facts()
        _STATE["modules_at_install"] = tuple(m for m in sys.modules if m.startswith("cuequivariance"))
        _install_trap()
        atexit.register(_at_exit)
        if model_cls is None:
            model_cls = importlib.import_module(MODEL_MODULE).Boltz2
        cur = model_cls.predict_step
        if not getattr(cur, "_kernels_census", False):
            _STATE["orig_step"] = cur
            model_cls.predict_step = _wrap_predict_step(cur)
        return True


def install_after(module, **kw) -> None:
    """Import-hook form (worker_launch): install on the just-executed ``boltz.model.models.boltz2`` module's ``Boltz2``."""
    install(model_cls=module.Boltz2, **kw)


def _wrap_predict_step(orig):
    def predict_step(self, *args, **kwargs):
        if not _STATE["stepped"]:
            first_step(self)
        return orig(self, *args, **kwargs)

    predict_step._kernels_census = True; predict_step.__wrapped__ = orig; predict_step.__doc__ = orig.__doc__
    return predict_step


def device_facts() -> dict:
    """The CUDA device this process computes on (torch's current device): index, name, compute capability, torch / CUDA versions; cc None without one."""
    try:
        import torch
        if torch.cuda.is_available():
            i = torch.cuda.current_device(); cc = torch.cuda.get_device_capability(i)
            return {"index": i, "name": torch.cuda.get_device_name(i), "cc": [int(cc[0]), int(cc[1])], "torch": torch.__version__, "cuda": torch.version.cuda}
        return {"index": None, "name": None, "cc": None, "torch": torch.__version__, "cuda": torch.version.cuda}
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"[:200], "cc": None}


def read_record(path: Optional[str], since: Optional[float] = None) -> Optional[dict]:
    """The census record at `path` (None when absent / unreadable); with `since` (the pass's launch time, epoch s) a record written earlier is
    an earlier pass's — stale, returned as None (never deleted)."""
    if not path or not os.path.isfile(path):
        return None
    try:
        rec = json.load(open(path))
    except (OSError, ValueError):
        return None
    w = rec.get("written_unix")
    if since is not None and isinstance(w, (int, float)) and float(w) < float(since) - 1.0:
        return None
    return rec


def first_step(model=None) -> None:
    """Before the first item's timed work: import + counters, device facts, the probe (CUDA, an accelerator expected engaged), the live
    ``use_kernels``; then REQUIRE."""
    with _LOCK:
        if _STATE["stepped"]:
            return
        expects_on = [a for a in ACCELERATORS if word_kind(_STATE["expected"].get(a, "engaged")) == "engaged"]
        verdict: Dict[str, str] = {}
        if expects_on:                                   # a route that expects no accelerator (kernels off) imports nothing of the library, like upstream: a contradiction then shows as the ops module in sys.modules at exit
            verdict = install_counters(); _STATE["counted"] = True
        for acc, v in verdict.items():
            if v:
                _STATE["imports"][f"verdict:{acc}"] = v
        _STATE["device"] = device_facts()
        if expects_on:
            try:
                ext = sys.modules.get(f"{OPS_PACKAGE}._ext") or importlib.import_module(f"{OPS_PACKAGE}._ext")
                _STATE["has_sm100f"] = bool(ext.has_sm100f_support()) if hasattr(ext, "has_sm100f_support") else None
            except Exception:  # noqa: BLE001
                _STATE["has_sm100f"] = None
        _STATE["use_kernels"] = (bool(getattr(model, "use_kernels")) if model is not None and hasattr(model, "use_kernels") else None)
        wants = [a for a in ACCELERATORS if word_kind(_STATE["expected"].get(a, "engaged")) == "engaged" and not verdict.get(a)]
        if wants and _STATE.get("do_probe", True) and (_STATE["device"] or {}).get("cc") and _STATE["use_kernels"] is not False:
            pr = probe()
            _STATE["probe"] = {a: pr[a] for a in wants if a in pr}
        _STATE["stepped"] = True
        bad = judge(_STATE["expected"], words(final=False))
        if bad:
            refuse(bad)


def _at_exit() -> None:
    if not _STATE.get("installed"):
        return
    try:
        emit()
    except Exception as e:  # noqa: BLE001 — the exit hook never masks the process's own exit
        try:
            print(f"[{TAG} {_STATE.get('mode')}] KERNELS-NOTE exit census failed: {type(e).__name__}: {e}", file=sys.stderr, flush=True)
        except Exception:  # noqa: BLE001
            pass


def parse_cli_opts(argv: List[str]) -> Tuple[dict, List[str]]:
    """``--kernels-route R --kernels-expect SPEC [--kernels-settings S] [--kernels-json P] [--kernels-ngpu P] [--kernels-rank K] [--kernels-probe 0|1]``
    out of an argv (the two launch sites share the spelling); returns (install kwargs or {}, argv without those options)."""
    keys = {"--kernels-route": "route", "--kernels-expect": "expected", "--kernels-settings": "settings", "--kernels-json": "json_path",
            "--kernels-ngpu": "n_gpu", "--kernels-rank": "rank", "--kernels-probe": "do_probe", "--kernels-mode": "mode"}
    kw: dict = {}; rest: List[str] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        if a in keys and i + 1 < len(argv):
            kw[keys[a]] = argv[i + 1]; i += 2; continue
        rest.append(a); i += 1
    if not kw:
        return {}, rest
    if "route" not in kw or "expected" not in kw:
        raise ValueError("--kernels-route and --kernels-expect go together")
    kw["expected"] = parse_expect(kw["expected"])
    if "n_gpu" in kw:
        kw["n_gpu"] = int(kw["n_gpu"])
    if "rank" in kw:
        kw["rank"] = int(kw["rank"])
    if "do_probe" in kw:
        kw["do_probe"] = kw["do_probe"] not in ("0", "false", "no")
    return kw, rest


def main(argv: Optional[List[str]] = None) -> int:
    """``python -m boltz2_opt.kernels [--route R] [--expect SPEC] [--settings S] [--no-probe] [--json P]``: the census of this machine without a model
    (imports, counters, device, probe): prints the line as mode ``census`` and exits 5 when ``--expect`` is given and refused."""
    import argparse
    ap = argparse.ArgumentParser(prog="python -m boltz2_opt.kernels", description=main.__doc__)
    ap.add_argument("--route", default="stock"); ap.add_argument("--expect", default=format_expect({a: "engaged" for a in ACCELERATORS}))
    ap.add_argument("--settings", default="-"); ap.add_argument("--no-probe", action="store_true"); ap.add_argument("--json", default=None)
    a = ap.parse_args(argv)

    class _NoModel:                                    # the census without a model: predict_step is never wrapped, use_kernels stays '-'
        @staticmethod
        def predict_step(*_a, **_k):
            return None
    install(a.route, parse_expect(a.expect), settings=a.settings, mode="census", json_path=a.json, model_cls=_NoModel, do_probe=not a.no_probe)
    try:
        for o in OPS.values():                         # upstream's call-site modules, so the probe goes through the bound site functions
            try:
                importlib.import_module(o["site"][0])
            except Exception as e:  # noqa: BLE001
                _STATE["notes"].append({"source": "census", "message": f"{o['site'][0]}: {type(e).__name__}: {e}"[:300]})
        first_step(None)
    except SystemExit as e:
        emit()
        return int(e.code or 0)
    rec = emit()
    return 0 if rec["verdict"] in ("PASS",) else EXIT_KERNELS


if __name__ == "__main__":
    sys.exit(main())
