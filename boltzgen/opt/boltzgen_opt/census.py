"""The accelerator census — the per-process monitor of what upstream BoltzGen's two cuEquivariance call sites run on, with the PEAK
line beside it. Plumbing: armed by a payload, it counts calls and reads switches (arguments and results pass through untouched) and
prints its lines at interpreter exit; it applies nothing and decides nothing. What a route EXPECTS of the accelerators, what a kit mode
REFUSES, and the caller's account of a run's lines are the modes' policy over this census (kernels.py). ``design --mode off`` arms
nothing: the stock child runs upstream alone, and upstream's own ``Using kernels:`` line in the configure log is its record.

Upstream engages exactly two accelerators, both from NVIDIA cuEquivariance and both behind its ``use_kernels`` gate
(``boltzgen configure --use_kernels auto|true|false``; ``auto`` = on when the device's compute capability is >= 8, printed at configure
as ``Using kernels: <bool> [device capability: (M, m)]``; plumbed into every step config as ``override.use_kernels``): the triangle
attention (``model/layers/triangular_attention/primitives.py`` ``kernel_triangular_attn``) and the triangle multiplicative update
(``model/layers/triangular.py`` ``_kernel_triangular_mult``). Both call sites import the front-end
``cuequivariance_torch.primitives.triangle`` at CALL time, whose functions import ``cuequivariance_ops_torch.triangle_attention`` /
``.triangle_multiplicative_update`` from the package namespace at call time — so the objects bound at the call sites are the package
attributes of ``cuequivariance_ops_torch``, which is where this census counts.

The library routes some calls to its own PyTorch reference implementations BY ITS OWN RULES (``cuequivariance_ops_torch`` 0.11
``triangle_attention.py``: query length ``S_qo <= CUEQ_TRIATTN_FALLBACK_THRESHOLD`` (env, default 100, inclusive), a hidden dimension
its kernels do not take, an empty attention; ``triangle_multiplicative_update.py``: sequence length ``<= CUEQ_TRIMUL_FALLBACK_THRESHOLD``
(env, default 100) in eager mode). Those calls are ``byrule`` — admissible, named, counted. A reference call the rules do not explain is
a ``fallback``, named in the line for the reader of the run. "No warning" shows nothing: the library's ``FALLING BACK … SM100f``
UserWarning fires on every by-rule reference call on a card outside the Blackwell SM100f family (H100 included) whatever the build; the
served counter is the evidence.

The census lines of a model process, all on stderr with flush, all under the process's ``[boltzgen-opt <mode>]`` tag (``<mode>`` and
``item`` are the armed payload's), each behind a line feed so it starts at column 0 whatever upstream's progress bar left on the line
(``print_fresh``; the readers' grammars are anchored ``^…$``). The KERNELS line, once, at interpreter exit (``LINE_VERB``)::
    [boltzgen-opt <mode>] KERNELS route=<default|exact|fast|big> step=<name|all>
        cueq_triatt=<word> cueq_trimul=<word> cueq=<version>-<build> caps=sm100f:<0|1>,sm107f:<0|1>,sm120f:<0|1> cc=<M.m> thresholds=triatt:<n>,trimul:<n>
        compile=<off|engaged[graphs=<n>]|unknown:<reason>> torch=<torch.__version__> cuda=<torch.version.cuda|none> py=<X.Y.Z>
        tf32=<0|1>/<float32 matmul precision> cudnn_tf32=<0|1> alloc_conf=<PYTORCH_CUDA_ALLOC_CONF|unset> seed=<base seed|?>

with ``<word>`` one of ``engaged:<backend>@<version>-<build>[served=<k>,byrule=<m>(<reason>:<n>+…)]`` · ``off-by-route:<reason>`` ·
``absent:<reason>`` · ``fallback:<reason>[served=<k>,byrule=<m>,fallback=<f>(<reason>:<n>+…)]`` ·
``n/a-upstream:step=<step>(<reason>)`` (the process ran a
pipeline step whose network has no call site, ``KERNEL_FREE_STEPS``: the inverse-folding step builds no trunk; the REQUIRE guard accepts it there and
nowhere else). The words after ``thresholds=`` are the stack
words (``stack_words``, report-only, each read without raising — a probe
that fails prints ``<key>=unknown:<reason>``): ``compile`` is TorchDynamo's own count of the graphs it compiled in this process
(``torch._dynamo.utils.counters["stats"]["unique_graphs"]``; ``off`` when it compiled none — upstream engages ``torch.compile(dynamic=True,
fullgraph=False)`` on the pairformer / the diffusion token transformer only under ``--config design compile_pairformer=true`` /
``compile_structure=true``); ``tf32`` / ``cudnn_tf32`` are ``torch.backends.cuda.matmul.allow_tf32`` with
``torch.get_float32_matmul_precision()`` and ``torch.backends.cudnn.allow_tf32`` as the ``predict_step`` calls found them (the values
every call of the process saw, ``+``-joined when they differed; the exit-time values in a process that ran no call — upstream's design
step sets the precision itself, ``matmul_precision`` of the step config, and the in-process pipeline ``bg_inproc.py`` restores the
interpreter defaults after each step, so the exit-time value is not the value the batches ran under); ``seed`` is the run's base seed as
the caller armed it (payload ``seed=``; step i of a pipeline runs with seed + i).

The stack words are read as upstream's ``Boltz.predict_step`` calls found them: the method is wrapped once per process by a pass-through
observer (``install_call_observer``: at the census ``arm`` when upstream's model module is already imported, else by a post-import hook the
moment it is) that records the TF32 switches on entry and calls the original — no synchronisation, no clock, no line of its own; timing is not
this package's (a caller that wants per-call walls wraps the same method from outside). ``EXAMPLES`` holds one line of each kind as printed
on the pinned stack (``report.EXAMPLES`` the ACTIVE lines) for readers of the grammar.

What is observed and what is inferred. THAT a call took the library's reference path is observed — the reference implementations
themselves are wrapped and counted. WHY it did (and so whether the call is ``byrule`` or a ``fallback``) is read off the call's shapes
against the library's documented rule: the thresholds are the values the library module itself read (its ``_fallback_threshold()`` /
``CUEQ_TRIMUL_FALLBACK_THRESHOLD``), the triangle attention's hidden-dimension and
empty-tensor rules are mirrored from the library's
``triangle_attention.py`` — the one mirrored fact of this module; a reference call no mirrored rule explains is a ``fallback``, so a
library whose rules moved reads as named fallbacks, never as silence.

Forms. The census is armed by ``arm(payload)`` — in the runner child of `design` (a kit mode's one model process) by ``_autoload`` at
interpreter start (``BOLTZGEN_OPT_KERNELS=<payload>``, exported by design.kit_env; upstream's CPU steps and multiprocessing children stay
unarmed), and in an activated process by ``stack.activate`` (through the environment it exports, also in each of that process's model
children). The children arm EARLY (``arm(..., early=True)``: the install runs at the first import of upstream's own ``boltzgen`` package
— before any step can run, and without importing torch during interpreter start-up): torch and ``cuequivariance_ops_torch`` are imported
by the census itself, the two dispatch functions and their two reference functions get pass-through counters (arguments and results
untouched). In a kit mode's process an absent or broken library means the mode cannot activate: refused before anything of the step runs
(``kernels.refuse_if_absent``: the NOT ACTIVE line naming the escape, exit 3); a payload with ``mode=off`` (route ``default``) is never
refused — the line says ``absent``. The in-process form arms lazily (the finder wraps the library the moment the caller's interpreter
imports it) and prints no line from a process that never imported the library (an activated ``boltzgen run`` parent orchestrates step
processes, each of which prints its own): there the import order is the caller's, and an absent library reads ``absent`` in the exit
line. A process where the library was imported but never called prints ``fallback:no-call`` for a route that expects the kernels. The
caller (design.py) reads the lines back (``parse_lines``), computes what the route expects from the mode table and upstream's own
resolution line (``kernels.expectation``), and records the account (``kernels.verdict``) in ``opt_manifest.json``; the words themselves
are the evidence a reader of the run acts on — the kit gates nothing on them after the run.
"""
from __future__ import annotations

import atexit
import os
import platform
import re
import sys
import threading
from typing import Dict, Iterable, List, Optional, Tuple

from . import TAG, print_fresh


def is_oom(e) -> bool:
    """The core's classifier (opt_core.oom.is_oom), imported at call time: this module loads at interpreter start in every kit child and imports nothing but the stdlib until then."""
    from opt_core.oom import is_oom as _is_oom
    return _is_oom(e)

ENV_KERNELS = "BOLTZGEN_OPT_KERNELS"          # "<key>=<value>,..." (payload()): arms the census in a child at interpreter start (_autoload)
LINE_VERB = "KERNELS"
ACCELERATORS: Tuple[str, ...] = ("cueq_triatt", "cueq_trimul")
ROUTES: Tuple[str, ...] = ("default", "exact", "fast", "big")   # the `KERNELS route=` vocabulary: `default` = upstream alone (a payload with mode off), else the kit mode; this kit ships no multi-GPU form
WORD_KINDS: Tuple[str, ...] = ("engaged", "off-by-route", "absent", "fallback", "n/a-upstream")
KERNEL_FREE_STEPS: Tuple[str, ...] = ("inverse_folding",)   # pipeline steps whose network has no accelerator call site: the inverse-folding step configures `data.cfg.inverse_fold: true` (upstream's resources/config/inverse_fold.yaml), under which Boltz.forward builds no trunk and runs no pairformer — a process of such a step that reached the library zero times says `n/a-upstream:step=<step>(…)`, not `fallback:no-call`
LIB_PACKAGE = "cuequivariance_ops_torch"
LIB_TRIATT = "cuequivariance_ops_torch.triangle_attention"
LIB_TRIMUL = "cuequivariance_ops_torch.triangle_multiplicative_update"
FRONT_END = "cuequivariance_torch.primitives.triangle"   # what upstream's two call sites import at call time
LINE_RE = re.compile(r"^\[" + re.escape(TAG) + r" (?P<mode>\S+)\] " + LINE_VERB + r" (?P<body>route=.*)$")
PEAK_RE = re.compile(r"^\[" + re.escape(TAG) + r" (?P<mode>\S+)\] PEAK item=(?P<item>\S+) alloc_gib=(?P<alloc_gib>[0-9.]+) reserved_gib=(?P<reserved_gib>[0-9.]+)$")   # the readers' grammar: `PEAK item=(\S+) alloc_gib=([0-9.]+) reserved_gib=([0-9.]+)$`, nothing between or after
UPSTREAM_MODEL_MODULE = "boltzgen.model.models.boltz"   # `Boltz.predict_step` there is one diffusion batch of a design job (the stack-word observer's call)
WORD_RE = re.compile(r"^(?P<kind>engaged|off-by-route|absent|fallback|n/a-upstream)(?::(?P<detail>.*))?$")
MAX_REASONS = 6

_LOCK = threading.Lock()
_STATE: Dict[str, object] = {"armed": None, "installed": False, "printed": False, "facts": {}, "absent": {}, "print_if_unused": True,
                             "observer_wrapped": False, "stack_seen": {"tf32": [], "cudnn_tf32": []}}
_COUNTS: Dict[str, Dict[str, Dict[str, int]]] = {a: {"served": {}, "byrule": {}, "fallback": {}} for a in ACCELERATORS}
_TLS = threading.local()


# ----------------------------------------------------------------------------------------------------------------- payload
def route_of(mode: str) -> str:
    """The KERNELS line's ``route=`` word of a mode: ``default`` for ``off`` (upstream alone), else the mode's own name."""
    return "default" if mode == "off" else mode


def payload(mode: str, expect: str, step: Optional[str] = None, item: Optional[str] = None, seed: Optional[int] = None,
            route: Optional[str] = None) -> str:
    """The arming string: ``mode=<m>,route=<r>,expect=<on|unknown|off:<reason>>[,step=<name>][,item=<id>][,seed=<n>]`` (route: ``route_of(mode)``
    unless given; item: the design job's name, for the PEAK line; seed: the run's base seed, for the KERNELS line's ``seed=`` word —
    report-only, ``seed=none`` when the run is unseeded)."""
    route = route_of(mode) if route is None else route
    if route not in ROUTES:
        raise ValueError(f"route {route!r} is not one of {ROUTES}")
    parts = [f"mode={mode}", f"route={route}", f"expect={expect}"]
    if step:
        parts.append(f"step={step}")
    if item:
        parts.append(f"item={_clean(item)}")
    if seed is not None:
        parts.append(f"seed={int(seed)}")
    return ",".join(parts)


def parse_payload(text: str) -> Dict[str, str]:
    out = {}
    for part in (text or "").split(","):
        if "=" in part:
            k, v = part.split("=", 1)
            out[k.strip()] = v.strip()
    for k in ("mode", "route", "expect"):
        if k not in out:
            raise ValueError(f"{ENV_KERNELS} payload lacks {k!r}: {text!r}")
    return out


# ------------------------------------------------------------------------------------------------------------------ arming
UPSTREAM_PACKAGE = "boltzgen"                  # the early trigger: upstream's own package is imported before any step can run


def _others_spec(finder, fullname, path, target):
    """The spec the OTHER meta-path finders resolve for ``fullname`` (None when none does)."""
    for other in sys.meta_path:
        if other is finder:
            continue
        try:
            spec = other.find_spec(fullname, path, target)
        except Exception:
            spec = None
        if spec is not None:
            return spec
    return None


class _Hook:
    """Duck-typed meta-path finder, the post-import hook of one module: lets ``name`` import through the other finders, then calls
    ``after(module)`` once its body has executed (``_Loader``) and retires. The census hook of the library (``_Finder``) and the stack-word
    observer's hook of upstream's model module (``UPSTREAM_MODEL_MODULE`` → ``install_call_observer``) are the two of this process."""

    def __init__(self, name, after):
        self.name, self.after = name, after

    def find_spec(self, fullname, path=None, target=None):
        if fullname != self.name:
            return None
        spec = _others_spec(self, fullname, path, target)
        if spec is None or spec.loader is None:
            return None
        spec.loader = _Loader(spec.loader, self, self.after)
        return spec

    def retire(self):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass


class _Finder(_Hook):
    """The library's hook: lets ``cuequivariance_ops_torch`` import, then installs the census on it. With ``early`` it also
    installs the census (importing torch and the library itself) at the first import of upstream's own package and lets
    ``on_early`` refuse — before anything of a step runs, without importing torch during interpreter start-up."""

    def __init__(self, early: bool = False, on_early=None):
        super().__init__(LIB_PACKAGE, install)
        self.early, self.on_early, self._fired_early = early, on_early, False

    def find_spec(self, fullname, path=None, target=None):
        if self.early and not self._fired_early and fullname == UPSTREAM_PACKAGE and not _STATE["installed"]:
            self._fired_early = True
            install(None)                            # imports torch, then the library (through this same finder: wrapped on import)
            if self.on_early is not None:
                self.on_early()
            return None                              # upstream's package itself loads through the other finders
        return super().find_spec(fullname, path, target)


class _Loader:
    def __init__(self, inner, finder, after):
        self.inner, self.finder, self.after = inner, finder, after

    def create_module(self, spec):
        return self.inner.create_module(spec)

    def exec_module(self, module):
        self.finder.retire()
        module.__spec__.loader = self.inner
        module.__loader__ = self.inner
        self.inner.exec_module(module)
        self.after(module)

    def __getattr__(self, name):
        return getattr(self.inner, name)


def armed() -> Optional[Dict[str, str]]:
    return _STATE["armed"]                       # type: ignore[return-value]


def arm(text: str, eager: bool = False, early: bool = False, on_early=None, print_if_unused: bool = True) -> Dict[str, str]:
    """Arm the census with a payload (idempotent) and register the exit line. The install — importing torch and the library, wrapping
    it — happens: at once when the library is already imported or ``eager``; at the first import of upstream's package when ``early``
    (the children of `design`, armed at interpreter start: in a kit mode's process ``on_early`` refuses an absent library there, before
    any step, and nothing is imported during start-up itself); else at the
    library's own first import (the in-process form). ``print_if_unused`` False: no exit line from a process that never imported the
    library (an activated process that orchestrates steps without running a model itself). Every route also arms the stack-word observer
    here (``arm_call_observer``: upstream's ``Boltz.predict_step`` wrapped now or at its module's import)."""
    p = parse_payload(text)
    with _LOCK:
        if _STATE["armed"] is not None:
            return _STATE["armed"]            # type: ignore[return-value]
        _STATE["armed"] = p
        _STATE["print_if_unused"] = bool(print_if_unused)
    atexit.register(print_line)
    arm_call_observer()
    if LIB_PACKAGE in sys.modules:
        install(sys.modules[LIB_PACKAGE])
        if on_early is not None:
            on_early()
    elif eager:
        install(None)
    else:
        sys.meta_path.insert(0, _Finder(early=early, on_early=on_early))
    return p


def disarm_finder() -> None:
    """Remove a census finder that never fired."""
    for f in list(sys.meta_path):
        if isinstance(f, _Finder):
            f.retire()


# ----------------------------------------------------------------------------------------------------------------- install
def install(pkg=None) -> dict:
    """Wrap the library's two dispatch functions and their reference implementations with counters; record the static facts.
    ``pkg`` None imports the library here (torch first: the cu13 ``libcue_ops.so`` resolves libnvrtc/libcublas 13 through torch's own
    preloaded libraries). An import failure is recorded as ``absent`` for both accelerators, never raised."""
    with _LOCK:
        if _STATE["installed"]:
            return dict(_STATE["facts"])      # type: ignore[arg-type]
        _STATE["installed"] = True
    facts: Dict[str, object] = {}
    _peak_reset()                                # item START for the PEAK line: this process is one design job; the install precedes every step of it
    if pkg is None:
        try:
            import torch  # noqa: F401 — before the library: its extension links against torch's CUDA libraries
            import importlib
            pkg = importlib.import_module(LIB_PACKAGE)
        except Exception as e:                 # ImportError, OSError from the extension loader, …: the accelerator is `absent` — an out-of-memory is not that and is re-raised
            if is_oom(e):
                raise
            why = f"import {LIB_PACKAGE} failed: {type(e).__name__}: {str(e).splitlines()[0][:160] if str(e) else ''}"
            for a in ACCELERATORS:
                _STATE["absent"][a] = why      # type: ignore[index]
            _STATE["facts"] = facts
            return facts
    facts["version"] = str(getattr(pkg, "__version__", "?"))
    facts["build"] = build_tag()
    facts["caps"] = capabilities(pkg)
    facts["cc"] = device_cc()
    ta = sys.modules.get(LIB_TRIATT)
    tm = sys.modules.get(LIB_TRIMUL)
    # triangle attention: the package attribute upstream's front-end resolves at call time, and the module's reference implementation
    f = getattr(pkg, "triangle_attention", None)
    if ta is None or not callable(f) or not hasattr(ta, "_triangle_attention_torch"):
        _STATE["absent"]["cueq_triatt"] = f"{LIB_PACKAGE}.triangle_attention is not the library's dispatch function (module {LIB_TRIATT} {'absent' if ta is None else 'lacks _triangle_attention_torch'})"   # type: ignore[index]
    else:
        facts["triatt_threshold"] = int(getattr(ta, "CUEQ_TRIATTN_FALLBACK_THRESHOLD", -1))
        facts["triatt_bound"] = f"{getattr(f, '__module__', '?')}.{getattr(f, '__qualname__', getattr(f, '__name__', '?'))}"
        pkg.triangle_attention = _wrap_triatt(f, ta)
        if getattr(ta, "triangle_attention", None) is f:
            ta.triangle_attention = pkg.triangle_attention
        ta._triangle_attention_torch = _wrap_reference("cueq_triatt", ta._triangle_attention_torch)
    # triangle multiplication: bound to a `_raise_triton_import_error` stub in the package's __init__ when its triton components failed to import
    g = getattr(pkg, "triangle_multiplicative_update", None)
    if tm is None or not callable(g) or getattr(g, "__module__", None) != LIB_TRIMUL or not hasattr(tm, "_tri_mul_torch"):
        exc = getattr(pkg, "IMPORT_EXCEPTION", None)
        tail = (str(exc).strip().splitlines() or ["?"])[-1][:160] if exc else "module absent"
        _STATE["absent"]["cueq_trimul"] = f"{LIB_PACKAGE}.triangle_multiplicative_update is not the library's kernel dispatch ({tail})"   # type: ignore[index]
    else:
        facts["trimul_threshold"] = int(getattr(tm, "CUEQ_TRIMUL_FALLBACK_THRESHOLD", -1))
        facts["trimul_bound"] = f"{g.__module__}.{getattr(g, '__qualname__', g.__name__)}"
        pkg.triangle_multiplicative_update = _wrap_trimul(g, tm)
        if getattr(tm, "triangle_multiplicative_update", None) is g:
            tm.triangle_multiplicative_update = pkg.triangle_multiplicative_update
        tm._tri_mul_torch = _wrap_reference("cueq_trimul", tm._tri_mul_torch)
    fe = sys.modules.get(FRONT_END)                     # upstream's call sites import this module at call time; when already imported, its functions must resolve the package attributes (they do: `from cuequivariance_ops_torch import … as f` inside each call)
    facts["front_end"] = "imported" if fe is not None else "not-yet-imported"
    _STATE["facts"] = facts
    return facts


def build_tag() -> str:
    """``cu12`` / ``cu13`` / …: the suffix of the installed distribution that provides ``cuequivariance_ops_torch`` (the ops build's CUDA major)."""
    try:
        import importlib.metadata as md
        names = sorted({(d.metadata["Name"] or "") for d in md.distributions() if (d.metadata["Name"] or "").lower().replace("_", "-").startswith("cuequivariance-ops-torch")})
    except Exception:
        names = []
    if not names:
        return "unknown-build"
    tags = [n.lower().replace("_", "-")[len("cuequivariance-ops-torch"):].lstrip("-") or "unsuffixed" for n in names]
    return "+".join(tags)                        # two builds installed at once would read e.g. cu12+cu13 — itself a finding


def capabilities(pkg) -> Dict[str, Optional[bool]]:
    """The library's own compile-time kernel-family flags (``_ext.has_sm100f_support`` …; the C++ CUDA_HAS_* macros)."""
    out: Dict[str, Optional[bool]] = {}
    ext = getattr(pkg, "_ext", None) or sys.modules.get(LIB_PACKAGE + "._ext")
    for name in ("sm100f", "sm107f", "sm120f"):
        fn = getattr(ext, f"has_{name}_support", None) if ext is not None else None
        try:
            out[name] = bool(fn()) if callable(fn) else None
        except Exception:
            out[name] = None
    return out


def device_cc() -> Optional[Tuple[int, int]]:
    try:
        import torch
        if torch.cuda.is_available():
            return tuple(torch.cuda.get_device_capability(0))   # type: ignore[return-value]
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------------------------------------------- counting
def _bump(accel: str, kind: str, key: str) -> None:
    with _LOCK:
        d = _COUNTS[accel][kind]
        if key in d or len(d) < 64:
            d[key] = d.get(key, 0) + 1
        else:
            d["…"] = d.get("…", 0) + 1


def _wrap_reference(accel: str, ref):
    def reference(*args, **kwargs):
        _TLS.__dict__[accel] = getattr(_TLS, accel, 0) + 1
        return ref(*args, **kwargs)
    reference.__wrapped__ = ref                  # type: ignore[attr-defined]
    reference._boltzgen_opt_census = accel       # type: ignore[attr-defined]
    return reference


def _triatt_rule(q, ta) -> Optional[str]:
    """The library's own reason for a reference-path triangle attention call, or None when its rules say the call is kernel-eligible
    (mirrors ``triangle_attention.py`` of cuequivariance_ops_torch 0.11: the threshold on S_qo, the hidden-dimension rules)."""
    try:
        import torch
        s_qo, hidden = int(q.shape[3]), int(q.shape[-1])
        thr = int(ta._fallback_threshold()) if hasattr(ta, "_fallback_threshold") else int(getattr(ta, "CUEQ_TRIATTN_FALLBACK_THRESHOLD", 100))
        dtype = q.dtype
        if torch.is_autocast_enabled():
            try:
                dtype = torch.get_autocast_dtype("cuda")
            except Exception:
                pass
        if int(q.shape[3]) == 0 or (len(q.shape) > 3 and 0 in tuple(q.shape)):
            return "empty"
        if dtype in (torch.float16, torch.bfloat16):
            if hidden % 8 != 0 or hidden > 256:
                return f"hidden_dim={hidden}(16-bit rule)"
        elif dtype == torch.float32:
            if hidden > 32 or hidden % 4 != 0:
                return f"hidden_dim={hidden}(fp32 rule)"
        else:
            return f"dtype={dtype}"
        if hidden < 32:
            thr = max(thr, 200)
        if s_qo <= thr:
            return f"s_qo<={thr}"
        return None
    except Exception as e:                       # a shape the rule reader cannot parse is named, never guessed
        return f"unparsed({type(e).__name__})"


def _wrap_triatt(f, ta):
    def triangle_attention(q, *args, **kwargs):
        before = getattr(_TLS, "cueq_triatt", 0)
        out = f(q, *args, **kwargs)
        if getattr(_TLS, "cueq_triatt", 0) > before:
            why = _triatt_rule(q, ta)
            if why is None:
                _bump("cueq_triatt", "fallback", f"reference@s_qo={_dim(q, 3)}")
            else:
                _bump("cueq_triatt", "byrule", why)
        else:
            _bump("cueq_triatt", "served", _triatt_backend(ta, q, args, kwargs))
        return out
    triangle_attention.__wrapped__ = f           # type: ignore[attr-defined]
    triangle_attention._boltzgen_opt_census = "cueq_triatt"   # type: ignore[attr-defined]
    for k in ("__doc__", "__module__", "__qualname__"):
        try:
            setattr(triangle_attention, k, getattr(f, k))
        except Exception:
            pass
    return triangle_attention


def _triatt_backend(ta, q, args, kwargs) -> str:
    """The forward kernel family the library selects for a served call (its own ``_select_forward_backend``), lower-cased."""
    try:
        k = args[0] if args else kwargs.get("k")
        mask = args[3] if len(args) > 3 else kwargs.get("mask")
        sel = getattr(ta, "_select_forward_backend", None)
        if sel is None or k is None:
            return "kernel"
        backend, _cc = sel(q, k, mask, None)
        return str(getattr(backend, "name", backend)).lower()
    except Exception:
        return "kernel"


def _wrap_trimul(g, tm):
    def triangle_multiplicative_update(x, *args, **kwargs):
        before = getattr(_TLS, "cueq_trimul", 0)
        out = g(x, *args, **kwargs)
        if getattr(_TLS, "cueq_trimul", 0) > before:
            thr = int(getattr(tm, "CUEQ_TRIMUL_FALLBACK_THRESHOLD", 100))
            seq = _dim(x, -2)
            if isinstance(seq, int) and seq <= thr:
                _bump("cueq_trimul", "byrule", f"seq<={thr}")
            else:
                _bump("cueq_trimul", "fallback", f"reference@seq={seq}")
        else:
            _bump("cueq_trimul", "served", "triton")
        return out
    triangle_multiplicative_update.__wrapped__ = g   # type: ignore[attr-defined]
    triangle_multiplicative_update._boltzgen_opt_census = "cueq_trimul"   # type: ignore[attr-defined]
    for k in ("__doc__", "__module__", "__qualname__"):
        try:
            setattr(triangle_multiplicative_update, k, getattr(g, k))
        except Exception:
            pass
    return triangle_multiplicative_update


def _dim(t, i):
    try:
        return int(t.shape[i])
    except Exception:
        return "?"


# ------------------------------------------------------------------------------------------------- the stack-word observer
def arm_call_observer() -> bool:
    """Wrap upstream's ``Boltz.predict_step`` now when its module is imported, else hook the module's import (``_Hook`` →
    ``install_call_observer``); idempotent. True when the wrap is in place already."""
    m = sys.modules.get(UPSTREAM_MODEL_MODULE)
    if m is not None:
        return install_call_observer(m)
    if not _STATE["observer_wrapped"] and not any(isinstance(f, _Hook) and f.name == UPSTREAM_MODEL_MODULE for f in sys.meta_path):
        sys.meta_path.insert(0, _Hook(UPSTREAM_MODEL_MODULE, install_call_observer))
    return bool(_STATE["observer_wrapped"])


def install_call_observer(module=None) -> bool:
    """Wrap ``Boltz.predict_step`` of upstream's model module (``UPSTREAM_MODEL_MODULE``) with the stack-word observer — once per process: a
    class whose ``predict_step`` already is the census wrapper (``_boltzgen_opt_census == "observer"``), or a process that wrapped once,
    is left alone, whatever wrapped the method since (``bg_hook`` wraps it after the census in the kit routes). True when the wrap is in
    place; False when the module has no ``Boltz.predict_step``."""
    module = module if module is not None else sys.modules.get(UPSTREAM_MODEL_MODULE)
    cls = getattr(module, "Boltz", None) if module is not None else None
    f = getattr(cls, "predict_step", None) if cls is not None else None
    if not callable(f):
        return False
    with _LOCK:
        if _STATE["observer_wrapped"] or getattr(f, "_boltzgen_opt_census", None) == "observer":
            _STATE["observer_wrapped"] = True
            return True
        cls.predict_step = _wrap_predict_step(f)
        _STATE["observer_wrapped"] = True
    for h in list(sys.meta_path):
        if isinstance(h, _Hook) and h.name == UPSTREAM_MODEL_MODULE:
            h.retire()
    return True


def _wrap_predict_step(f):
    def predict_step(self, *args, **kwargs):
        _see_stack_state()                           # the TF32 switches this call runs under, for the KERNELS line; report-only, no device interaction
        return f(self, *args, **kwargs)              # arguments, result and any exception pass through unchanged
    predict_step.__wrapped__ = f                     # type: ignore[attr-defined]
    predict_step._boltzgen_opt_census = "observer"   # type: ignore[attr-defined]
    for k in ("__doc__", "__module__", "__qualname__", "__name__"):
        try:
            setattr(predict_step, k, getattr(f, k))
        except Exception:
            pass
    return predict_step


def _see_stack_state() -> None:
    """Record the TF32 state a ``predict_step`` call runs under (``stack_words`` prints what the calls saw): ``allow_tf32``/precision and cuDNN's flag."""
    t = sys.modules.get("torch")
    if t is None:
        return
    try:
        tf32 = f"{int(bool(t.backends.cuda.matmul.allow_tf32))}/{t.get_float32_matmul_precision()}"
        cudnn = str(int(bool(t.backends.cudnn.allow_tf32)))
    except Exception:                                # a torch without these switches: the exit-time probe names the reason
        return
    with _LOCK:
        seen = _STATE["stack_seen"]                  # type: ignore[assignment]
        if tf32 not in seen["tf32"]:
            seen["tf32"].append(tf32)
        if cudnn not in seen["cudnn_tf32"]:
            seen["cudnn_tf32"].append(cudnn)


# ------------------------------------------------------------------------------------------------------------------- words
def counts() -> Dict[str, Dict[str, Dict[str, int]]]:
    with _LOCK:
        return {a: {k: dict(v) for k, v in kinds.items()} for a, kinds in _COUNTS.items()}


def absent() -> Dict[str, str]:
    """``{accelerator: why}`` for every accelerator this process cannot provide (known once ``install`` ran: eager arming, or the library's import)."""
    with _LOCK:
        return dict(_STATE["absent"])                                   # type: ignore[arg-type]


def _reasons(d: Dict[str, int]) -> str:
    items = sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
    body = "+".join(f"{k}:{v}" for k, v in items[:MAX_REASONS])
    if len(items) > MAX_REASONS:
        body += f"+…:{sum(v for _, v in items[MAX_REASONS:])}"
    return body


def word(accel: str, expect: str, facts: dict, absent: Dict[str, str], census: Dict[str, Dict[str, int]], step: Optional[str] = None) -> str:
    """The one word of an accelerator in this process (grammar in the module docstring). ``step``: the pipeline step this process ran
    (``KERNEL_FREE_STEPS``: a step whose network has no call site and made no call is ``n/a-upstream``, never a fallback)."""
    if accel in absent:
        return f"absent:{_clean(absent[accel])}"
    served, byrule, fallback = census["served"], census["byrule"], census["fallback"]
    n_s, n_b, n_f = sum(served.values()), sum(byrule.values()), sum(fallback.values())
    ver = f"{facts.get('version', '?')}-{facts.get('build', '?')}"
    if expect.startswith("off"):
        if n_s + n_b + n_f == 0:
            return f"off-by-route:use_kernels=False({_clean(expect[4:])})"
        return f"fallback:calls-under-use_kernels=False[served={n_s},byrule={n_b},fallback={n_f}]"   # the gate said off and the library was called: named, refused
    if n_f:
        return f"fallback:{_clean(_reasons(fallback))}[served={n_s},byrule={n_b},fallback={n_f}]"
    if n_s + n_b == 0:
        if step in KERNEL_FREE_STEPS:
            return f"n/a-upstream:step={step}(inverse-fold network: no trunk, no call site)"
        return "fallback:no-call(the model never reached the library: use_kernels off in the step config, or no forward ran)"
    backend = "+".join(sorted(served)) if served else "none"
    cc = facts.get("cc")
    sm = f"-sm{cc[0]}{cc[1]}" if cc else ""
    tail = f"[served={n_s},byrule={n_b}" + (f"({_clean(_reasons(byrule))})" if n_b else "") + "]"
    return f"engaged:{backend}{sm}@{ver}{tail}"


def _clean(s: str) -> str:
    return re.sub(r"\s+", "_", str(s).strip())


def line(p: Optional[Dict[str, str]] = None) -> str:
    """The KERNELS line of this process."""
    p = p or armed() or {"mode": "?", "route": "?", "expect": "on"}
    facts, absent = dict(_STATE["facts"]), dict(_STATE["absent"])     # type: ignore[arg-type]
    if not _STATE["installed"] and not absent:                        # armed, never imported: no call reached the library
        absent = {}
    census = counts()
    step = p.get("step") or os.environ.get("BOLTZGEN_PIPELINE_STEP") or "all"
    words = " ".join(f"{a}={word(a, p.get('expect', 'on'), facts, absent, census[a], step=step)}" for a in ACCELERATORS)
    caps = facts.get("caps") or {}
    caps_s = ",".join(f"{k}:{'?' if v is None else int(v)}" for k, v in caps.items()) or "unknown"
    cc = facts.get("cc")
    thr = f"triatt:{facts.get('triatt_threshold', '?')},trimul:{facts.get('trimul_threshold', '?')}"
    return (f"[{TAG} {p.get('mode')}] {LINE_VERB} route={p.get('route')} step={step} {words} "
            f"cueq={facts.get('version', '?')}-{facts.get('build', '?')} caps={caps_s} cc={f'{cc[0]}.{cc[1]}' if cc else 'none'} thresholds={thr} "
            f"compile={compile_word()} {stack_words(p)}")


def compile_word() -> str:
    """``off`` · ``engaged[graphs=<n>]`` · ``unknown:<reason>`` — n is TorchDynamo's own count of the graphs it compiled in this process,
    ``torch._dynamo.utils.counters["stats"]["unique_graphs"]`` (the counter its ``convert_frame`` bumps once per compiled graph; a process
    that never imported ``torch._dynamo`` compiled nothing: ``torch.compile`` imports it)."""
    try:
        if "torch" not in sys.modules:
            return "unknown:torch-not-imported"
        utils = sys.modules.get("torch._dynamo.utils")
        if utils is None:
            return "off"
        n = int(utils.counters["stats"]["unique_graphs"])
        return f"engaged[graphs={n}]" if n > 0 else "off"
    except Exception as e:                           # a torch whose dynamo counters moved: named, never raised at exit
        return f"unknown:{type(e).__name__}"


def _probe(read) -> str:
    try:
        v = read()
    except Exception as e:                           # every stack word prints; a probe that fails says why
        return f"unknown:{type(e).__name__}"
    return _clean(v) if v not in (None, "") else "none"


def stack_words(p: Optional[Dict[str, str]] = None) -> str:
    """``torch=<v> cuda=<v|none> py=<X.Y.Z> tf32=<0|1>/<precision> cudnn_tf32=<0|1> alloc_conf=<PYTORCH_CUDA_ALLOC_CONF|unset> seed=<n|?>`` — the
    stack words of the KERNELS line (grammar and meaning in the module docstring; report-only, never raising)."""
    p = p or armed() or {}
    t = sys.modules.get("torch")
    with _LOCK:
        seen = {k: list(v) for k, v in _STATE["stack_seen"].items()}   # type: ignore[union-attr]
    absent = "unknown:torch-not-imported"
    torch_v = _probe(lambda: t.__version__) if t is not None else absent
    cuda_v = _probe(lambda: t.version.cuda) if t is not None else absent
    tf32 = "+".join(seen["tf32"]) if seen["tf32"] else (_probe(lambda: f"{int(bool(t.backends.cuda.matmul.allow_tf32))}/{t.get_float32_matmul_precision()}") if t is not None else absent)
    cudnn = "+".join(seen["cudnn_tf32"]) if seen["cudnn_tf32"] else (_probe(lambda: str(int(bool(t.backends.cudnn.allow_tf32)))) if t is not None else absent)
    alloc = _probe(lambda: os.environ.get("PYTORCH_CUDA_ALLOC_CONF") or "unset")
    seed = _probe(lambda: str(int(p["seed"])) if p.get("seed") not in (None, "") else "?")
    return f"torch={torch_v} cuda={cuda_v} py={_probe(platform.python_version)} tf32={tf32} cudnn_tf32={cudnn} alloc_conf={alloc} seed={seed}"


def _peak_reset() -> None:
    try:
        from opt_core.mem.allocator import reset_peak   # the core's per-pass peak counters (torch.cuda.reset_peak_memory_stats; False without torch / CUDA)
        reset_peak()
    except Exception:                            # no device / no context: nothing to reset, the PEAK line then reads what the process reached
        pass


def peak_line(p: Optional[Dict[str, str]] = None) -> Optional[str]:
    """``[boltzgen-opt <mode>] PEAK item=<id> alloc_gib=<torch.cuda.max_memory_allocated()/2**30:.2f> reserved_gib=<max_memory_reserved()/2**30:.2f>``
    (exactly the readers' grammar — route and step are on the KERNELS line beside it) — the allocator peak of this one-item model process, read through the core's
    ``opt_core.mem.allocator.counters`` (torch's own maxima: exact, reset at the census install = item start by ``reset_peak``; report-only).
    A device-side nvidia-smi sampler is the cross-check (device high-water, pools included: ``opt_core.mem.peak`` names the two).
    ``None`` from a process that never initialised CUDA."""
    p = p or armed() or {}
    t = sys.modules.get("torch")
    try:
        if t is None or not t.cuda.is_available() or not t.cuda.is_initialized():
            return None
        from opt_core.mem.allocator import counters
        c = counters()
        a, r = c["max_allocated_gib"], c["max_reserved_gib"]
        if a is None or r is None:
            return None
    except Exception:
        return None
    return f"[{TAG} {p.get('mode')}] PEAK item={p.get('item') or '?'} alloc_gib={a:.2f} reserved_gib={r:.2f}"


def parse_peak_lines(lines: Iterable[str]) -> List[dict]:
    """Every PEAK line among ``lines``: ``[{mode, item, alloc_gib, reserved_gib, raw}]``."""
    out = []
    for ln in lines:
        m = PEAK_RE.match(ln.rstrip("\n"))
        if m:
            d = m.groupdict()
            out.append(dict(d, alloc_gib=float(d["alloc_gib"]), reserved_gib=float(d["reserved_gib"]), raw=ln.rstrip("\n")))
    return out


def print_line() -> Optional[str]:
    """At interpreter exit (armed processes): the KERNELS line, once (nothing for a ``print_if_unused=False`` process that never imported the
    library), and the PEAK line beside it when this process used the GPU."""
    with _LOCK:
        if _STATE["printed"]:
            return None
        if not _STATE["print_if_unused"] and not _STATE["installed"] and not _STATE["absent"]:
            return None
        _STATE["printed"] = True
    try:
        text = line()
    except Exception as e:                       # never nothing: a census that cannot format itself says so
        text = f"[{TAG} ?] {LINE_VERB} route=? census-failed:{type(e).__name__}:{_clean(e)}"
    print_fresh(text)
    peak = peak_line()
    if peak:
        print_fresh(peak)
    return text


# ---------------------------------------------------------------------------------------------------------------- the readers
def parse_lines(lines: Iterable[str]) -> List[dict]:
    """Every KERNELS line among ``lines``: ``[{mode, route, step, words: {accel: word}, kinds: {accel: kind}, fields, raw}]``."""
    out = []
    for ln in lines:
        m = LINE_RE.match(ln.rstrip("\n"))
        if not m:
            continue
        fields: Dict[str, str] = {}
        for tok in m.group("body").split(" "):
            if "=" in tok:
                k, v = tok.split("=", 1)
                fields[k] = v
        words = {a: fields.get(a, "absent:not-in-line") for a in ACCELERATORS}
        kinds = {}
        for a, w in words.items():
            wm = WORD_RE.match(w)
            kinds[a] = wm.group("kind") if wm else "unparsed"
        out.append({"mode": m.group("mode"), "route": fields.get("route"), "step": fields.get("step"),
                    "words": words, "kinds": kinds, "fields": fields, "raw": ln.rstrip("\n")})
    return out


# ------------------------------------------------------------------------------------------------------------------ sample lines
EXAMPLES = {   # example census lines as model processes printed them on the pinned stack (one KERNELS line per route, one with upstream's torch.compile switched on through its own `--config design compile_*=true`, and a PEAK line): the kit's unit tests parse them with this module's grammar, and any other reader of these lines can test its own grammar on them
    'kernels_default': "[boltzgen-opt off] KERNELS route=default step=design cueq_triatt=engaged:generic-sm90@0.11.1-cu13[served=17920,byrule=0] cueq_trimul=engaged:triton-sm90@0.11.1-cu13[served=17920,byrule=0] cueq=0.11.1-cu13 caps=sm100f:1,sm107f:0,sm120f:1 cc=9.0 thresholds=triatt:100,trimul:100 compile=off torch=2.13.0+cu130 cuda=13.0 py=3.11.5 tf32=1/high cudnn_tf32=1 alloc_conf=unset seed=42",
    'kernels_exact': "[boltzgen-opt exact] KERNELS route=exact step=design cueq_triatt=engaged:generic-sm90@0.11.1-cu13[served=1120,byrule=0] cueq_trimul=engaged:triton-sm90@0.11.1-cu13[served=1120,byrule=0] cueq=0.11.1-cu13 caps=sm100f:1,sm107f:0,sm120f:1 cc=9.0 thresholds=triatt:100,trimul:100 compile=off torch=2.13.0+cu130 cuda=13.0 py=3.11.5 tf32=1/high cudnn_tf32=1 alloc_conf=unset seed=42",
    'kernels_big': "[boltzgen-opt big] KERNELS route=big step=design cueq_triatt=engaged:generic-sm90@0.11.1-cu13[served=1120,byrule=0] cueq_trimul=engaged:triton-sm90@0.11.1-cu13[served=1120,byrule=0] cueq=0.11.1-cu13 caps=sm100f:1,sm107f:0,sm120f:1 cc=9.0 thresholds=triatt:100,trimul:100 compile=off torch=2.13.0+cu130 cuda=13.0 py=3.11.5 tf32=1/high cudnn_tf32=1 alloc_conf=expandable_segments:True seed=42",
    'kernels_default_compiled': "[boltzgen-opt off] KERNELS route=default step=design cueq_triatt=engaged:generic-sm90@0.11.1-cu13[served=1120,byrule=0] cueq_trimul=engaged:triton-sm90@0.11.1-cu13[served=1120,byrule=0] cueq=0.11.1-cu13 caps=sm100f:1,sm107f:0,sm120f:1 cc=9.0 thresholds=triatt:100,trimul:100 compile=engaged[graphs=12] torch=2.13.0+cu130 cuda=13.0 py=3.11.5 tf32=1/high cudnn_tf32=1 alloc_conf=unset seed=42",
    'peak': "[boltzgen-opt off] PEAK item=stock_b16 alloc_gib=3.65 reserved_gib=4.28",
}
