"""opt_core.trimul — the one adapter a kit's triangle-multiplication lever composes over the carried TriMul kernels.

A triangle-multiplication module (the "TriMul" of every pairformer / Evoformer trunk: LayerNorm_in -> two gated projections a, b (+ mask)
-> the per-channel N x N contraction ab -> LayerNorm_out -> gated output projection (+ residual)) is served by one of the carried kernels
of :mod:`opt_core.kernels` or by the engine's own forward, and the choice is never silent.  This module owns the parts each adapter had
re-written around those kernels — the decision ladder, the weight vocabulary, the census and the activation-evidence line — and owns no
engine fact: which module class is patched, how its parameters are named, which kit mode routes which kernel, which fallbacks a mode
expects, are data the kit's thin adapter passes in.

Contract.
  * Pure-Python import surface (Python >= 3.9, stdlib only at module level).  ``torch`` and the carried kernels are imported inside the
    provider functions, i.e. only in a process that actually serves a TriMul call through them; a missing library is a NAMED refusal
    (``Refused('import:<module>')``), never an ImportError at ``import opt_core.trimul``.
  * The kernels are the carried ones, reached BY NAME (``import fpf_trimul_v4`` / ``import fpf_trimul``): the kit routes the name to the
    core copy (``opt_core.kernels.route(name)`` + ``exports`` + ``route_check`` BEFORE the first served call) or keeps its own copy on
    ``sys.path`` — this module imports whatever the name resolves to and reports where it resolved (``impl=<name>@<version> origin=<core|kit>``).
  * ONE decision ladder per call (:meth:`Lever.serve`): mode ``stock`` -> the engine's own forward (event ``fallback:mode_stock``);
    N below ``min_tokens`` -> the ``small`` provider when the kit gives one, else the engine's forward (``fallback:below_min_tokens``);
    the provider's ``eligible`` refuses (shape, dtype, a measured-off cell ``cell:<cls>_C<C>+off(not-measured)``, library) -> the engine's
    forward (``fallback:<reason>``); the kernel raises -> the engine's forward for THIS call (``error:<ExceptionType>``, the first two printed;
    ``strict`` re-raises); an exception that says the kernel CANNOT run in this process (``cannot_run``: :class:`CannotRun`, the carried
    fpf_trimul_v4's ``none:<Exc>`` / ``probe-failed``) is re-raised, never rerouted — the mode refuses by name; else ``served:<provider>``
    (an UNKNOWN (dtype, C) class a provider serves on its kernel's default tiles is named: ``cells=unverified(<cls>_C<C>):<n>`` on the line,
    ONE stderr line).  Every branch is counted per direction; nothing returns the stock path without an event.
  * FAIL-LOUD: :meth:`Lever.gate` is the fail-closed gate a kit folds into its exit verdict — refused when any ``error:*`` was counted, when a
    fallback reason outside the kit's ``expected`` list was counted, or when the mode routes a kernel and nothing was served (an ``OutOfMemoryError``
    from a provider is never counted or rerouted: it propagates to the caller).  :meth:`Lever.line`
    is the ONE activation-evidence line of the lever (strategy id ``F2.trimul``), composed from :mod:`opt_core.report` primitives:

        [<tag>] LEVER name=F2.trimul state=<on|off|skipped> [reason=<why>] impl=<kernel>@<version> origin=<core|kit|none> served=<n> fallback=<n> fallback_by=<reason:n,...|none> min_tokens=<n> shapes=<NxC/dtype/direction of the first served call|none> mode=<mode> errors=<Type:n,...|none> [cells=unverified(<cls>_C<C>):<n>,...] gate=<ok|refused>

    (``state``: on = the mode routes a provider in this process; off = mode ``stock``; skipped = selected by the kit and not applied, with
    the kit's reason. The bytes are composed with ``report.prefix`` + ``report.kv`` — the core's one grammar for per-lever lines.)

  * Exactness is never claimed here: ``exact``-class use of a provider on an engine is what that engine's equality leg proves; the providers
    only state which kernel construction they run (``tmk3`` = the TM-K3 line whose exact mode reproduces the cuEquivariance fused TriMul's
    floating-point operations in order; ``fpf_v4`` = the Tier-2 fused TriMul: same rounding points, different reduction trees).

Kit adapter (the whole engine-specific part; ``M`` is the engine's TriMul class, names are the engine's)::

    from opt_core import trimul as T

    def weights_of(m):                      # the ten canonical tensors (WEIGHT_KEYS); biases b_ag/b_ap/b_bg/b_bp/b_o/b_og when the module has them
        return dict(ln_in_w=m.layer_norm_in.weight, ln_in_b=m.layer_norm_in.bias, w_ag=m.linear_a_g.weight, w_ap=m.linear_a_p.weight,
                    w_bg=m.linear_b_g.weight, w_bp=m.linear_b_p.weight, ln_out_w=m.layer_norm_out.weight, ln_out_b=m.layer_norm_out.bias,
                    w_o=m.linear_z.weight, w_og=m.linear_g.weight)

    LEVER = T.Lever("acme-opt", mode, provider=T.fpf_v4(weights_of) if mode == "fast" else T.tmk3(weights_of, mode="exact"),
                    min_tokens=101, expected=("mode_stock", "below_min_tokens", "c=64"))
    _orig = M.forward
    def forward(self, z, mask=None):
        return LEVER.serve(T.Call(self, z, mask, "outgoing" if self.outgoing else "incoming", residual=False,
                                  orig=lambda: _orig(self, z, mask)))
    M.forward = forward
    LEVER.register_exit_line()              # or print LEVER.line() in the kit's own report; LEVER.gate() feeds the kit's exit verdict

Providers for engine-local kernels (a layout-only bmm path, a tiled transpose) are built with :func:`custom` so they share the ladder,
the census and the line; they name themselves, this module does not name them.
"""
from __future__ import annotations

import os
import sys
import threading
from typing import Any, Callable, Dict, Iterable, Mapping, Optional, Tuple

from .oom import is_oom
from .gates import Gate
from . import report as _report

STRATEGY = "F2.trimul"
MODES = ("stock", "exact", "fast")
GRID_REFUSED = "launch_grid_y>65535"          # tmk3: the TM-K3 stage-A/C launch would exceed CUDA's grid axis-1 limit at this N (kernels.launch_grid_y)
KERNEL_OF_PROVIDER = {"fpf_v4": "fpf_trimul_v4", "tmk3": "fpf_trimul"}      # provider -> the carried kernel NAME it imports (opt_core.kernels.names())
from .trimul_weights import WEIGHT_KEYS, BIAS_KEYS  # noqa: F401 — the shared weight vocabulary (re-exported: ``opt_core.trimul.WEIGHT_KEYS``)
DIRECTIONS = ("outgoing", "incoming")
_CACHE_ATTR = "_opt_core_trimul"          # per-module dict of packed weights, keyed (provider, cache_key, dtype): two providers on one module keep two packs
_PRINT_FIRST_ERRORS = 2


class Refused(Exception):
    """A named precondition refusal of a provider (``reason`` is a short token the census carries: ``c=64``, ``dtype:float32``,
    ``import:fpf_trimul_v4``, ``n<101``, ``cell:bf16_C64+off(not-measured)`` ...).  Raised by ``eligible``/``fn``; :meth:`Lever.serve` turns it
    into ``fallback:<reason>`` and the engine's own forward serves the call."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = str(reason)


class CannotRun(RuntimeError):
    """The provider's kernel CANNOT run in this process — its SAFE launch cell failed to build too, or its warm numerics probe refused the
    shape class (``fpf_trimul_v4``: ``none:<Exc>`` / ``probe-failed``).  ``cannot_run`` is True: :meth:`Lever.serve` re-raises it (never a
    per-call route to the engine's forward, never counted as a fallback) so the kit's forward sees the named signal and the MODE refuses by
    name — a mode is all of its levers.  The message names the provider, the kernel's reason and the escape (a mode without this lever)."""
    cannot_run = True

    def __init__(self, reason: str, provider: str = "", detail: str = ""):
        self.reason, self.provider = str(reason), str(provider)
        super().__init__("TriMul provider %s cannot run in this process (%s)%s — the mode that routes it refuses by name; run a mode without this "
                         "lever (e.g. the kit's stock TriMul mode)" % (provider or "?", reason, (": " + detail) if detail else ""))


def cannot_run(exc: BaseException) -> bool:
    """True when ``exc`` says its lever cannot run in this process (an exception carrying ``cannot_run = True``: :class:`CannotRun`, the carried
    ``fpf_trimul_v4`` (>= 4.3) ``TrimulUnsupported`` of reasons ``none:<Exc>`` / ``probe-failed``)."""
    return bool(getattr(exc, "cannot_run", False))


class Call:
    """One TriMul call as the kit's patched forward sees it.  ``z``: the pair tensor ``[N,N,C]`` or ``[B,N,N,C]``; ``mask``: None, ``[N,N]``
    or ``[B,N,N]``; ``direction``: ``outgoing`` | ``incoming``; ``residual``: True when the engine's forward returns ``z + update`` (the
    kernels fuse the add), False when it returns the update; ``orig``: a zero-argument callable running the engine's OWN forward for this
    call (the fallback target); ``module``: the patched module instance (weights are read and cached on it)."""

    __slots__ = ("module", "z", "mask", "direction", "residual", "orig", "extra")

    def __init__(self, module: Any, z: Any, mask: Any, direction: str, residual: bool, orig: Callable[[], Any], **extra: Any):
        if direction not in DIRECTIONS:
            raise ValueError(f"direction must be one of {DIRECTIONS}, not {direction!r}")
        self.module, self.z, self.mask, self.direction, self.residual, self.orig, self.extra = module, z, mask, direction, bool(residual), orig, extra

    @property
    def n_tokens(self) -> int:
        return int(self.z.shape[-2])

    @property
    def channels(self) -> int:
        return int(self.z.shape[-1])


class Provider:
    """A kernel behind the ladder: ``eligible(call) -> None`` (raises :class:`Refused`) and ``fn(call) -> output``.  ``kernel`` is the carried
    kernel name it imports (None for an engine-local provider); ``version`` is filled on first use from the imported module when not given.
    ``cells`` ({word: count}, the named cell facts of served calls) is OPTIONAL: a kit's own provider object (any object with ``name`` / ``fn`` /
    ``eligible``) need not carry it — the census reads it when present."""

    def __init__(self, name: str, fn: Callable[[Call], Any], eligible: Optional[Callable[[Call], None]] = None,
                 kernel: Optional[str] = None, version: Optional[str] = None):
        self.name, self.fn, self.eligible, self.kernel, self.version = name, fn, (eligible or (lambda call: None)), kernel, version
        self.resolved_from = None            # 'core' | 'kit' | None — where ``import <kernel>`` landed, filled on first use
        self.cells: Dict[str, int] = {}      # the named cell facts of served calls: 'unverified(<cls>_C<C>)' -> n (an UNKNOWN class served on the kernel's default tiles)
        self._lock = threading.Lock()

    def note_cell(self, word: str) -> int:
        """Count a named cell fact of a call this provider serves (``unverified(<key>)``: no certification record for the class — the kernel's
        default tiles serve it, named); returns the count (1 = first in the process: :meth:`Lever.serve` prints ONE line)."""
        with self._lock:
            self.cells[word] = self.cells.get(word, 0) + 1
            return self.cells[word]

    def describe(self) -> str:
        if self.kernel is None:
            return f"{self.name}@{self.version or 'local'}"
        return f"{self.kernel}@{self.version or '?'}"


# --------------------------------------------------------------------------------------------------------------------------- helpers
def _import(name: str):
    """Import a kernel module by NAME with a named refusal (never an ImportError out of the ladder)."""
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise Refused("import:%s(%s)" % (name, str(e).split("\n")[0][:60].replace(" ", "_")))


def _resolved_from(top_name: str) -> str:
    """'core' when ``top_name`` was imported from the core copy of :mod:`opt_core.kernels`, else 'kit' (the kit's own copy on sys.path)."""
    try:
        from . import kernels as _k
        mod = sys.modules.get(top_name)
        where = os.path.abspath(getattr(mod, "__file__", "") or "")
        core = os.path.abspath(_k.carried_path(top_name))
        if os.path.isdir(core):
            return "core" if where.startswith(core + os.sep) else "kit"
        return "core" if where == core else "kit"
    except Exception:  # noqa: BLE001 — a report field, never a failure
        return "kit"


def _weights(call: Call, weights_of: Callable[[Any], Mapping[str, Any]]) -> Dict[str, Any]:
    w = dict(weights_of(call.module))
    missing = [k for k in WEIGHT_KEYS if k not in w or w[k] is None]
    if missing:
        raise Refused("weights_missing:" + "+".join(missing))
    unknown = [k for k in w if k not in WEIGHT_KEYS and k not in BIAS_KEYS]
    if unknown:
        raise Refused("weights_unknown:" + "+".join(sorted(unknown)))
    return w


def _cache(module: Any) -> dict:
    c = getattr(module, _CACHE_ATTR, None)
    if c is None:
        c = {}
        try:
            setattr(module, _CACHE_ATTR, c)
        except Exception:  # noqa: BLE001 — a module that refuses attributes packs per call (slow, correct)
            pass
    return c


def _compute_input(call: Call, autocast_input: bool):
    """The tensor the kernel reads: under autocast, ``z`` cast once to the autocast dtype (what a fused stock kernel branch does); else ``z``."""
    import torch
    z = call.z
    if autocast_input and torch.is_autocast_enabled():
        dt = torch.get_autocast_dtype(z.device.type) if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()
        if z.dtype != dt:
            z = z.to(dt)
    return z


# ------------------------------------------------------------------------------------------------------------------------- providers
def fpf_v4(weights_of: Callable[[Any], Mapping[str, Any]], *, pad: int = 16, autocast_input: bool = True, cache_key: str = "F2.fpf_v4") -> Provider:
    """The Tier-2 fused TriMul (carried kernel ``fpf_trimul_v4``, entry ``fpf_trimul_v4.generic``): Triton LN_in + dual gated projection ->
    channel-major zero-padded planes, cuBLAS batched contraction, Triton LN_out + gated output projection (+ fused residual).  Served cell =
    what ``generic.supported`` accepts under the package's one cells table (``kernels/fpf_trimul_v4/table.json``; an uncertified capability
    serves the SAFE cell by name): C, D in {128, 256}, bf16 or fp32 z, N >= 101 (env FPF_TRIMUL_V4_NMIN; no upper size limit — the workspace,
    ``generic.workspace_bytes``, is allocated like any torch operation's); a structural miss (c, n, dtype, mask rank, no-cell) is
    ``Refused(<generic's reason>)`` — the engine's forward serves that call, counted.  A refusal after which the kernel CANNOT run in this
    process (``none:<Exc>``: its SAFE cell failed to build; ``probe-failed``: the warm numerics probe refused the class) is re-raised as is
    (``cannot_run``): the mode refuses by name, never a quiet subset.  Numerics: the cuEquivariance fused TriMul's rounding points with different
    reduction trees — fast-class unless an engine's equality leg proves otherwise."""

    def _packed(call: Call, G):
        z = call.extra.get("_z_in")
        if z is None:                                                      # ONE cast per call: eligible() and fn() both ask (the cast tensor rides on the call)
            z = _compute_input(call, autocast_input)
            call.extra["_z_in"] = z
        c = _cache(call.module)
        key = ("fpf_v4", cache_key, str(z.dtype))
        w = c.get(key)
        if w is None:
            t = _weights(call, weights_of)
            w = G.pack_weights(cache_owner=None, cache_key=cache_key, **{k: (v.detach() if hasattr(v, "detach") else v) for k, v in t.items()})
            c[key] = w
        return z, w

    def eligible(call: Call) -> None:
        G = _import("fpf_trimul_v4.generic")
        if prov.version is None:
            prov.version = str(getattr(G, "COUNTS", {}).get("version") or getattr(sys.modules.get("fpf_trimul_v4"), "__version__", None) or "?")
            prov.resolved_from = _resolved_from("fpf_trimul_v4")
        if not getattr(call.z, "is_cuda", False):
            raise Refused("device:%s" % getattr(getattr(call.z, "device", None), "type", "unknown"))
        if call.z.dim() not in (3, 4):
            raise Refused("rank:%d" % call.z.dim())
        z, w = _packed(call, G)
        ok, why = G.supported(z, call.mask, weights=w)
        if not ok:
            if _v4_cannot_run(G, reason=why):
                raise CannotRun(str(why), prov.name)
            raise Refused(str(why).replace(" ", "_")[:80])

    def fn(call: Call):
        G = sys.modules.get("fpf_trimul_v4.generic") or _import("fpf_trimul_v4.generic")
        z, w = _packed(call, G)
        try:
            return G.trimul(z, call.mask, direction=call.direction, weights=w, residual=call.residual, pad=pad)
        except G.TrimulUnsupported as e:                                   # the kernel's own refusal mid-call: cannot-run propagates (the mode refuses by name);
            if _v4_cannot_run(G, exc=e):                                     # a structural miss is this call's route to the engine's forward, counted by the kernel's word
                if cannot_run(e):
                    raise
                raise CannotRun(str(getattr(e, "reason", "")), prov.name, str(e)[:200]) from e
            raise Refused(str(getattr(e, "reason", e)).replace(" ", "_")[:80]) from None

    prov = Provider("fpf_v4", fn, eligible, kernel="fpf_trimul_v4")
    return prov


_V4_CANNOT_RUN = ("none", "probe-failed")     # fpf_trimul_v4 (>= 4.3) refusal reasons after which its lever is off for the process: 'none[:<Exc>]', 'probe-failed'


def _v4_cannot_run(G, *, exc: Optional[BaseException] = None, reason: Any = None) -> bool:
    """Whether an ``fpf_trimul_v4.generic`` refusal (the exception, or ``supported()``'s reason) means the kernel CANNOT run in this process: the
    exception's own ``cannot_run`` attribute when it carries one, else the package's ``cannot_run(reason)`` when it has one, else the reason's
    head (``none`` / ``probe-failed``)."""
    if exc is not None and getattr(exc, "cannot_run", None) is not None:
        return bool(exc.cannot_run)
    r = reason if reason is not None else getattr(exc, "reason", "")
    f = getattr(G, "cannot_run", None)
    if callable(f):
        try:
            return bool(f(r))
        except (TypeError, ValueError):
            pass
    head = str(r).split(":", 1)[0].strip()
    return head in _V4_CANNOT_RUN


def tmk3(weights_of: Callable[[Any], Mapping[str, Any]], *, mode: str = "exact", contract: Optional[str] = None, eps: float = 1e-5,
         autocast_input: bool = True, key: Optional[str] = None, n_min: int = 101, cache_key: str = "F2.tmk3") -> Provider:
    """The TM-K3 line (carried kernel ``fpf_trimul``; ``kernels.trimul_forward`` with the cfg of ``trimul._select_cfg``).  ``mode='exact'``: the
    construction that reproduces the cuEquivariance fused TriMul's floating-point operations in order (stock cuEq LayerNorm kernels, ONE dual gated
    GEMM, an unpadded cuBLAS contraction) — exact-class ONLY where the engine's stock TriMul IS that cuEq kernel and its equality leg proves it;
    ``mode='fast'``: the line's padded Tier-2 variant.  The package reads ``FPF_TRIMUL_MODE`` at import: the kit exports it (this provider refuses
    with ``package_mode:<m>`` when the imported package disagrees).  Served: CUDA z of rank 3/4, ``N >= n_min`` (below it the vendor TriMul runs its small-N torch algorithm), a (dtype, C)
    class by the package's cell verdict (``trimul.cell_verdict``: a VERIFIED class serves; a class certified engines run with the lever OFF —
    bf16 C=64, fp32 C=64 outside ``exact``, fp32 C=128 / C=256 — is ``Refused('cell:<cls>_C<C>+off(not-measured)')``, the engine's forward by name;
    any OTHER bf16 / fp32 class is UNKNOWN and SERVED on the package's default tiles, named ``cells=unverified(<cls>_C<C>)`` on the line and ONE
    stderr line; a dtype outside bf16 / fp32 is ``Refused('dtype:<cls>')``), and a launchable geometry — the line's stage-A/C kernels tile the N*N token rows on CUDA grid axis 1, so a call needs
    ``kernels.launch_grid_y(N, C, cfg) <= 65535``; the package schedules the stage tile height by N (``trimul._select_cfg``: the cell's tile
    through its own limit — N <= 2047 for the bf16 C=128 cell's 64-row tiles, N <= 2849 for the bf16 C=256 and fp32 C=384 cells' 128-row tiles —
    then 128 / 256 / 512-row tiles of the same arithmetic, narrower as they grow taller), so every served cell launches through N = 5632; above it
    ``Refused('launch_grid_y>65535')`` (a kernel geometry limit, not a proof range: `exact` is bit-exact to cuEquivariance's fused TriMul wherever
    it launches — checked through N = 4096 at C = 128, the scheduled tiles bitwise equal to the cell's own tiles wherever both launch).
    Anything else is ``Refused(<reason>)``.  Weights are packed once per (module, compute dtype): p = [w_ap; w_bp], g = [w_ag; w_bg]."""
    if mode not in ("exact", "fast"):
        raise ValueError("tmk3 mode must be 'exact' or 'fast'")

    def _mods():
        K = _import("fpf_trimul.kernels")
        Tm = _import("fpf_trimul.trimul")
        return K, Tm

    def _packed(call: Call, K, cdt):
        import torch
        c = _cache(call.module)
        ck = ("tmk3", cache_key, str(cdt))
        w = c.get(ck)
        if w is None:
            t = _weights(call, weights_of)
            if any(t.get(b) is not None for b in BIAS_KEYS):
                raise Refused("biases_unsupported_by_tmk3")

            def cc(x):
                return x.detach().to(cdt).contiguous()

            def n(x):
                return x.detach().contiguous()
            w = K.pack_weights(n(t["ln_in_w"]), n(t["ln_in_b"]), cc(torch.cat([t["w_ap"], t["w_bp"]], 0)), cc(torch.cat([t["w_ag"], t["w_bg"]], 0)),
                               n(t["ln_out_w"]), n(t["ln_out_b"]), cc(t["w_o"]), cc(t["w_og"]))
            c[ck] = w
        return w

    def _cdt(call: Call):
        import torch
        if autocast_input and torch.is_autocast_enabled():
            return torch.get_autocast_dtype(call.z.device.type) if hasattr(torch, "get_autocast_dtype") else torch.get_autocast_gpu_dtype()
        return _weights(call, weights_of)["w_ap"].dtype

    def eligible(call: Call) -> None:
        K, Tm = _mods()
        if prov.version is None:
            top = sys.modules.get("fpf_trimul")
            prov.version = "%s/%s" % (getattr(top, "__version__", None) or getattr(Tm, "VERSION", None) or "3.x", mode)
            prov.resolved_from = _resolved_from("fpf_trimul")
        pm = getattr(Tm, "_MODE", None) or getattr(Tm, "MODE", None)
        if pm is not None and str(pm) != mode:
            raise Refused("package_mode:%s" % pm)
        if not getattr(call.z, "is_cuda", False):
            raise Refused("device:%s" % getattr(getattr(call.z, "device", None), "type", "unknown"))
        if call.z.dim() not in (3, 4):
            raise Refused("rank:%d" % call.z.dim())
        N, C = call.n_tokens, call.channels
        if N < n_min:
            raise Refused("n<%d" % n_min)
        cdt = _cdt(call)
        cls = {"torch.bfloat16": "bf16", "torch.float32": "fp32", "torch.float16": "fp16"}.get(str(cdt), str(cdt))
        verdict = getattr(Tm, "cell_verdict", None)                                  # the package's ONE cell decision: verified | off (+off(not-measured)) | unverified | unsupported
        if verdict is not None:
            kind, word = verdict(cdt, C)
            if kind in ("off", "unsupported"):
                raise Refused(word)
            if kind == "unverified":
                call.extra["cell_word"] = word                                        # an UNKNOWN class: served on the default tiles, named (Lever.serve prints ONE line, cells= on the line)
        else:                                                                         # a routed copy that predates cell_verdict: its VERIFIED sets decide, as they always did
            isv = getattr(Tm, "is_verified", None)
            if isv is not None and not isv(cdt, C):
                raise Refused("cell:%s_C%d" % (cls, C))
        grid_y = getattr(K, "launch_grid_y", None)                                     # structural: stages A/C tile the N*N token rows on CUDA grid axis 1 (<= 65535 programs);
        if grid_y is not None and grid_y(N, C, Tm._select_cfg(cdt, N, C)) > K.CUDA_GRID_Y_MAX:   # a routed copy without the geometry function launches ungated (its own launch error is counted)
            raise Refused(GRID_REFUSED)
        if call.mask is not None and call.mask.dim() not in (2, 3):
            raise Refused("mask_rank:%d" % call.mask.dim())

    def fn(call: Call):
        import torch
        K, Tm = _mods()
        cdt = _cdt(call)
        w = _packed(call, K, cdt)
        z = call.z
        zs = z if z.dim() == 4 else z[None]
        m = call.mask
        ms = None if m is None else (m if m.dim() == 3 else m[None])
        N = call.n_tokens
        ctr = contract if contract is not None else getattr(Tm, "CONTRACT", "cublas")
        outs = []
        for b in range(zs.shape[0]):
            zb = zs[b].contiguous()
            mb = None if ms is None else ms[min(b, ms.shape[0] - 1)]
            outs.append(K.trimul_forward(zb, call.direction, mb, w, eps=eps, residual=call.residual,
                                         cfg=Tm._select_cfg(cdt, N, zb.shape[-1]), contract=ctr, key=key, cdt=cdt))
        return torch.stack(outs, 0) if z.dim() == 4 else outs[0]

    prov = Provider("tmk3.%s" % mode, fn, eligible, kernel="fpf_trimul")
    return prov


def custom(name: str, fn: Callable[[Call], Any], eligible: Optional[Callable[[Call], None]] = None, version: Optional[str] = None) -> Provider:
    """An engine-local TriMul path (e.g. a layout-only batched-matmul forward, a tiled-transpose operand builder) behind the same ladder, census and
    line.  ``eligible(call)`` raises :class:`Refused` for calls it does not serve; ``fn(call)`` returns the module's output."""
    return Provider(name, fn, eligible, kernel=None, version=version or "local")


def by_word(weights_of: Callable[[Any], Mapping[str, Any]], word: str, *, prefer: Optional[Tuple[str, ...]] = None, residency: Optional[str] = None,
            stock: Optional[Callable[..., Any]] = None, name: Optional[str] = None, pad: int = 16, contract: str = "bf16", cache_key: str = "F2.by_word") -> Provider:
    """A provider over :mod:`opt_core.kernels.trimul` — the ONE triangle-multiplication provider (every carried row + the measured cell table
    ``TRIMUL_CELLS.json``) — selected BY WORD behind this module's ladder, census and line.  ``word``: a row name (``v4``, ``tmk3_exact``,
    ``tmk3_fast``, ``tx_sm90a``, ``tx_sm90a_exact``, ``ef2_fused``, ``ef2_cueq_tiles``, ``cueq``, ``torch_math`` …: exactly that row — the
    default rule, what a kit binding that kernel today already runs) or a tier word (``fast`` | ``exact`` | ``big``: the measured winner of
    the call's cell (capability, precision incl. an fp32-resident z under autocast, c_z, c_hidden, N bucket, direction) on this process's stack,
    OPT-IN).  ``prefer`` narrows a tier to the rows the kit carries/routes, in its order; ``stock`` is the kit's own stock callable for the
    ``cueq`` / ``ef2_cueq_tiles`` rows.  ``eligible`` refuses BY NAME with the table's word (``<row>:<kind>``: ``tx_sm90a:cc:8.0!=9.0(...)``,
    ``tx_sm90a:no_prebuilt:<abi>``, ``v4:c_z:384`` …) — the engine's forward serves that call, counted; nothing is substituted here."""
    from .kernels import trimul as KT                                       # pure-stdlib import (torch only inside its serving functions)

    def _facts(call: Call):
        import torch
        z = call.z
        zs = z if z.dim() == 4 else z[None]
        prec, _cdt = KT.call_precision(zs)
        c = _cache(call.module)
        st = c.get(("by_word", "stack"))
        if st is None:
            st = c[("by_word", "stack")] = KT.stack_word(zs)
        bwd = bool(torch.is_grad_enabled() and z.requires_grad)
        res = "fp32" if prec.startswith("f32z_") else residency
        dt = prec.split("_")[-1] if prec.startswith("f32z_") else ("fp32" if prec == "tf32" else prec)
        return zs, KT._device_cc(zs), dt, res, prec == "tf32", bwd, st

    def eligible(call: Call) -> None:
        if not getattr(call.z, "is_cuda", False):
            raise Refused("device:%s" % getattr(getattr(call.z, "device", None), "type", "unknown"))
        if call.z.dim() not in (3, 4):
            raise Refused("rank:%d" % call.z.dim())
        zs, cc, dt, res, tf32, bwd, st = _facts(call)
        t = _weights(call, weights_of)
        D = int(t["w_ap"].shape[0])
        try:
            sel = KT.select(cc, dt, call.channels, D, call.n_tokens, call.direction, word=word, residency=res, backward=bwd, prefer=prefer, stack=st,
                            tf32=tf32, has_cueq=(True if stock is not None else None), batch=int(zs.shape[0]))   # the leading batch extent is a fact of the class: under the
            KT.admits(sel.row, cc, dt, call.channels, D, call.n_tokens, residency=res, backward=bwd, residual=call.residual, batch=int(zs.shape[0]),   # exact word: a B > 1 layout is its own vouch
                      abi=(KT.tx_abi_tag() if sel.row.startswith("tx_sm90a") else None))
        except KT.Refusal as e:
            raise Refused(("%s:%s" % (e.row, e.kind) if e.row else e.kind).replace(" ", "_")[:120]) from None
        call.extra["trimul_selection"] = sel
        if prov.version is None or prov.kernel != "trimul:%s" % sel.row:
            prov.kernel, prov.version = "trimul:%s" % sel.row, "%s/%s" % (word, sel.cell or "no-cell")
            prov.resolved_from = "core"

    def fn(call: Call):
        sel = call.extra.get("trimul_selection")
        c = _cache(call.module).setdefault(("by_word", cache_key), {})
        try:
            return KT.triangle_multiplication(call.z, call.mask, direction=call.direction, weights=_weights(call, weights_of), word=word, selection=sel,
                                            residual=call.residual, prefer=prefer, residency=residency, stock=stock, cache=c, pad=pad, contract=contract)
        except KT.Refusal as e:                                            # a refusal with the tensors in hand (the kernel's own supported()): this call's route to the engine's forward
            raise Refused(("%s:%s" % (e.row, e.kind) if e.row else e.kind).replace(" ", "_")[:120]) from None

    prov = Provider(name or ("trimul:%s" % word), fn, eligible, kernel="trimul")
    return prov


# ----------------------------------------------------------------------------------------------------------------------------- lever
class Lever:
    """One TriMul lever of a kit: a mode, the provider that mode routes (None for ``stock``), an optional ``small`` provider below ``min_tokens``,
    the fallback reasons the mode EXPECTS (anything else refuses the gate), the census and the line.  One instance per process per patched
    class family; thread-safe counting."""

    def __init__(self, tag: str, mode: str, *, provider: Optional[Provider] = None, small: Optional[Provider] = None, min_tokens: int = 0,
                 expected: Iterable[str] = ("mode_stock", "below_min_tokens"), strict: Optional[bool] = None, stream=None):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}")
        if mode != "stock" and provider is None:
            raise ValueError(f"mode {mode!r} routes a kernel: pass provider=")
        self.tag, self.mode, self.provider, self.small, self.min_tokens = tag, mode, provider, small, int(min_tokens)
        self.expected = tuple(expected)
        self.strict = (os.environ.get("OPT_CORE_TRIMUL_STRICT", "0") == "1") if strict is None else bool(strict)
        self.stream = stream
        self._lock = threading.Lock()
        self.counts = {d: {} for d in DIRECTIONS}          # direction -> event -> n
        self.first = None                                   # the first SERVED call: {N, C, dtype, direction, provider}
        self._printed_errors = 0

    # -- census
    def _count(self, direction: str, event: str) -> int:
        with self._lock:
            d = self.counts[direction]
            d[event] = d.get(event, 0) + 1
            return d[event]

    def census(self) -> dict:
        """``{mode, provider, kernel, origin, served, fallback{reason:n}, errors{type:n}, by_direction{...}, first}`` — plain data for a manifest."""
        with self._lock:
            served, fb, err = 0, {}, {}
            for d in DIRECTIONS:
                for ev, n in self.counts[d].items():
                    kind, _, rest = ev.partition(":")
                    if kind == "served":
                        served += n
                    elif kind == "fallback":
                        fb[rest] = fb.get(rest, 0) + n
                    elif kind == "error":
                        err[rest] = err.get(rest, 0) + n
            p = self.provider
            cells = {}
            for q in (self.provider, self.small):
                for wd, n in (getattr(q, "cells", None) or {}).items():      # ``cells`` is an OPTIONAL provider attribute ({word: count} of named cell facts); a provider without it has none
                    cells[wd] = cells.get(wd, 0) + n
            return {"strategy": STRATEGY, "mode": self.mode, "provider": None if p is None else p.name,
                    "kernel": None if p is None else p.describe(), "origin": None if p is None else (p.resolved_from or "none"),
                    "served": served, "fallback": fb, "errors": err, "cells": cells, "by_direction": {d: dict(v) for d, v in self.counts.items()},
                    "first": self.first, "expected": list(self.expected), "min_tokens": self.min_tokens}

    def _expected(self, reason: str) -> bool:
        return any(reason == e or reason.startswith(e + ":") or reason.startswith(e) for e in self.expected)

    def gate(self) -> Gate:
        """Fail-closed: refused on any kernel error, on any fallback reason the kit did not list in ``expected``, or when the mode routes a
        kernel and no call was served while calls arrived.  The kit folds this into its exit verdict (a refused gate = a partial activation)."""
        c = self.census()
        if c["errors"]:
            return Gate(name=STRATEGY, ok=False, reason="kernel errors: " + _report.kv(errors=dict(sorted(c["errors"].items()))), details=c)
        unexpected = {r: n for r, n in c["fallback"].items() if not self._expected(r)}
        if unexpected:
            return Gate(name=STRATEGY, ok=False, reason="unexpected fallback: " + _report.kv(fallback=dict(sorted(unexpected.items()))), details=c)
        n_calls = c["served"] + sum(c["fallback"].values())
        if self.mode != "stock" and n_calls > 0 and c["served"] == 0:
            from .counters import stock_by_cell_aside                          # every routed call ANSWERED with the cell's stock row BY NAME (stock:<row> / the library op / the
            aside = stock_by_cell_aside(c["fallback"], c["served"], c["errors"])   # caller's statement) and no launch fallback / error: a NAMED ASIDE, the gate passes; a real
            if aside:                                                          # zero-served lever (launch fallbacks, size gates, refusals by shape) keeps the violation
                return Gate(name=STRATEGY, ok=True, details=dict(c, aside=aside), words=("aside:" + aside,))
            return Gate(name=STRATEGY, ok=False, reason="mode %s routed %s but served 0 of %d calls" % (self.mode, c["kernel"], n_calls), details=c)
        return Gate(name=STRATEGY, ok=True, details=c)

    def line(self, state: Optional[str] = None, reason: Optional[str] = None) -> str:
        """The ONE activation-evidence line of this lever, in the core's per-lever grammar (see the module docstring): ``state`` = ``on``
        (the mode routes a provider in this process) | ``off`` (mode ``stock``: not in this mode's lever set) | ``skipped`` (the kit selected
        the lever and did not apply it — pass ``state="skipped", reason=<why>``; the reason feeds the kit's partial census)."""
        c = self.census()
        g = self.gate()
        st = state or ("off" if self.mode == "stock" else "on")
        if st not in ("on", "off", "skipped"):
            raise ValueError("state must be on|off|skipped")
        if st == "skipped" and not reason:
            raise ValueError("state=skipped needs reason=")
        first = "none" if not c["first"] else "%dx%d/%s/%s" % (c["first"]["N"], c["first"]["C"], c["first"]["dtype"], c["first"]["direction"])
        pairs = [("name", STRATEGY), ("state", st)]
        if reason or st != "on":
            pairs.append(("reason", reason or ("mode_stock" if st == "off" else None)))
        pairs += [("impl", c["kernel"] or "none"), ("origin", c["origin"] or "none"), ("served", c["served"]),
                  ("fallback", sum(c["fallback"].values())), ("fallback_by", dict(sorted(c["fallback"].items()))), ("min_tokens", self.min_tokens),
                  ("shapes", first), ("mode", self.mode), ("errors", dict(sorted(c["errors"].items())))]
        if c["cells"]:                                                            # named cell facts of served calls (cells=unverified(<cls>_C<C>):<n>): only when an UNKNOWN class was served
            pairs.append(("cells", ",".join("%s:%d" % kv for kv in sorted(c["cells"].items()))))
        if g.ok and (g.details or {}).get("aside"):                              # a passing gate's named aside (every call answered with the cell's stock row by name)
            pairs.append(("aside", g.details["aside"]))
        pairs.append(("gate", "ok" if g.ok else "refused"))
        return _report.prefix(self.tag) + " LEVER " + _report.kv(*pairs)

    def register_exit_line(self) -> bool:
        """Print :meth:`line` once at interpreter exit (through :func:`opt_core.report.register_exit_tally`, tag ``<tag>/F2.trimul``)."""
        return _report.register_exit_tally(self.tag + "/" + STRATEGY, self.line)

    # -- the ladder
    def _log(self, msg: str) -> None:
        print(_report.prefix(self.tag) + " " + STRATEGY + " " + msg, file=self.stream or sys.stderr, flush=True)

    def serve(self, call: Call):
        """Run ONE TriMul call through the ladder; returns the module output (the provider's or the engine's own)."""
        d = call.direction
        if self.mode == "stock" or self.provider is None:
            self._count(d, "fallback:mode_stock")
            return call.orig()
        prov = self.provider
        if call.n_tokens < self.min_tokens:
            if self.small is None:
                self._count(d, "fallback:below_min_tokens")
                return call.orig()
            prov = self.small
        try:
            prov.eligible(call)
        except Refused as r:
            self._count(d, "fallback:" + r.reason)
            return call.orig()
        try:
            out = prov.fn(call)
        except Refused as r:
            self._count(d, "fallback:" + r.reason)
            return call.orig()
        except Exception as e:  # noqa: BLE001 — counted, printed, gated; never silent
            if is_oom(e):
                raise                                     # an out-of-memory is the caller's to see: never rerouted to the engine's forward (which needs more memory, not less)
            if cannot_run(e):
                raise                                     # a cannot-run signal (the kernel's lever went off BY NAME) is the mode's refusal: never a quiet per-call subset
            n = self._count(d, "error:" + type(e).__name__)
            if self.strict:
                raise
            with self._lock:
                self._printed_errors += 1
                say = self._printed_errors <= _PRINT_FIRST_ERRORS
            if say:
                self._log("KERNEL ERROR provider=%s direction=%s n=%d -> engine forward for this call: %r" % (prov.name, d, n, e))
            return call.orig()
        if self._count(d, "served:" + prov.name) == 1 and self.first is None:
            with self._lock:
                if self.first is None:
                    self.first = {"N": call.n_tokens, "C": call.channels, "dtype": str(getattr(call.z, "dtype", "?")).replace("torch.", ""),
                                  "direction": d, "provider": prov.name}
        word = call.extra.get("cell_word")
        if word and prov.note_cell(word) == 1:            # an UNKNOWN cell class served on the kernel's default tiles: named ONCE per process (and cells= on the line)
            self._log("cells=%s: provider %s serves a (dtype, C) class with no certification record on its default tiles — named, not refused" % (word, prov.name))
        return out


# -------------------------------------------------------------------------------------------------------------------------- self-test
def _selftest() -> int:
    """The ladder, census, gate and line with fake providers (no torch): ``python -m opt_core.trimul``."""

    class Z:                                  # a stand-in tensor: shape, dim(), is_cuda, dtype
        def __init__(self, n, c):
            self.shape = (n, n, c)
            self.is_cuda, self.dtype = True, "bfloat16"

        def dim(self):
            return 3

    served = []

    def fn(call):
        served.append(call.n_tokens)
        return "kernel"

    def elig(call):
        if call.channels != 128:
            raise Refused("c=%d" % call.channels)

    def boom(call):
        raise RuntimeError("boom")

    lv = Lever("acme-opt", "fast", provider=custom("fake", fn, elig, "0"), min_tokens=101, expected=("mode_stock", "below_min_tokens", "c=64"))
    orig = lambda: "stock"  # noqa: E731
    assert lv.serve(Call(None, Z(300, 128), None, "outgoing", False, orig)) == "kernel"
    assert lv.serve(Call(None, Z(50, 128), None, "incoming", False, orig)) == "stock"         # below the gate
    assert lv.serve(Call(None, Z(300, 64), None, "incoming", False, orig)) == "stock"         # expected refusal
    assert lv.gate().ok, lv.gate().reason
    assert lv.serve(Call(None, Z(300, 256), None, "outgoing", False, orig)) == "stock"        # unexpected refusal -> gate refuses
    g = lv.gate()
    assert not g.ok and "c=256" in (g.reason or "") and "c=64" not in (g.reason or ""), g.reason
    ln = lv.line()
    assert ln == "[acme-opt] LEVER name=F2.trimul state=on impl=fake@0 origin=none served=1 fallback=3 fallback_by=below_min_tokens:1,c=256:1,c=64:1 min_tokens=101 shapes=300x128/bfloat16/outgoing mode=fast errors=none gate=refused", ln
    assert lv.line(state="skipped", reason="route_check").startswith("[acme-opt] LEVER name=F2.trimul state=skipped reason=route_check impl=fake@0"), lv.line(state="skipped", reason="route_check")
    lv2 = Lever("acme-opt", "exact", provider=custom("bad", boom), strict=False, stream=open(os.devnull, "w"))
    assert lv2.serve(Call(None, Z(300, 128), None, "outgoing", True, orig)) == "stock"
    assert not lv2.gate().ok and "RuntimeError" in lv2.gate().reason
    lv3 = Lever("acme-opt", "stock")
    assert lv3.serve(Call(None, Z(300, 128), None, "outgoing", False, orig)) == "stock" and lv3.gate().ok
    assert lv3.census()["fallback"] == {"mode_stock": 1}
    assert lv3.line().startswith("[acme-opt] LEVER name=F2.trimul state=off reason=mode_stock impl=none"), lv3.line()
    try:
        Call(None, Z(3, 3), None, "sideways", False, orig)
        raise AssertionError("direction not validated")
    except ValueError:
        pass
    print("opt_core.trimul selftest ok:", ln)
    return 0


if __name__ == "__main__":
    sys.exit(_selftest())
