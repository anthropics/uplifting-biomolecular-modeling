"""Served-or-named-fallback layer over the carried ``flash_triattn`` kernel (family F1, lever ``F1.flash_triattn``).

What it is. ``flash_triattn`` (the carried Triton kernel beside this file) is a drop-in for cuEquivariance's
``triangle_attention(q, k, v, bias, mask=None, scale=None)``: q/k/v ``[B, N, H, S, D]`` bf16/fp16 (D in 16/32/64/128; fp32 inputs are
computed at a ``tl.dot`` input precision — ``input_precision="tf32"|"tf32x3"|"ieee"`` — for an engine whose trunk runs fp32), pair bias
``[B, 1, H, S_q, S_k]`` fp32, key mask ``[B, N, 1, 1, S_k]`` bool; bf16 tensor-core products with fp32 accumulation and an fp32 online
softmax, no atomics, a launch configuration that is a pure function of (D, S, H) — run-to-run bit-exact, NOT bit-exact to cuEq or to a
materialised softmax (re-association) → class ``fast`` (Tier 2) unless the engine's equality tests prove otherwise.
This module adds what a kit adapter around it needs and nothing engine-specific:

* :func:`probe` / :func:`require` — is the kernel usable in this process (torch, triton, CUDA device with cc >= 8)? A ``dict`` that never
  raises / a :class:`Refusal` naming the reason (``torch_missing``, ``triton_missing``, ``no_cuda_device``, ``cc_lt_8``,
  ``kernel_import_failed``). Nothing heavy is imported until one of these is called.
* :func:`kernel_module` — the kernel module this process uses: the top-level ``flash_triattn`` when the kit routed it
  (``opt_core.kernels.route("flash_triattn")``) or imported its own copy, else the core copy ``opt_core.kernels.flash_triattn``;
  :func:`kernel_origin` says which (``core`` | ``kit``) and :func:`kernel_impl` gives ``flash_triattn@<sha8>`` for the evidence line.
* :class:`Ledger` — per-process served / fallback-by-reason counters and a census of served shapes; :meth:`Ledger.line` renders the ONE
  per-lever activation-evidence line (``[<tag>] LEVER name=F1.flash_triattn state=<on|skipped|off> reason=… impl=flash_triattn origin=…
  served=… fallback=… reasons=…``) from ``opt_core.report.prefix`` + ``kv``; :meth:`Ledger.partial` is True when the lever was asked and
  never served (the kit's exit rule turns that into EXIT_NOT_ACTIVE unless ``--allow-partial``, see ``opt_core.report.verdict``).
* :func:`triangle_attention` — serve-or-fallback in one call: below the kit's token gate → ``stock`` with event ``below_gate``; a shape /
  dtype / device the kernel does not serve → ``stock`` with event ``unsupported:<why>`` (the kernel's own ``flash_supported``); else the
  kernel, counted as served. A fallback is ALWAYS an event in the ledger — there is no silent stock path; with ``stock=None`` a call
  that cannot be served raises :class:`Refusal`.

Kit usage (the adapter lives in the kit; the engine's module patching is the kit's)::

    from opt_core.kernels import flash_triattn_serve as F1
    F1.require()                                              # or: gate on F1.probe()["ok"] and record the reason
    LEDGER = F1.ledger(min_tokens=300)                        # the kit's gate (a kit constant, cited in its mode table)
    def patched_core(q, k, v, bias, mask, scale):             # inside the kit's TriangleAttention adapter
        return F1.triangle_attention(q, k, v, bias, mask=mask, scale=scale, ledger=LEDGER, stock=stock_core)
    ... once per arm per process:  F1.emit_line(LEDGER, tag="my_kit");  LEDGER.fields() -> manifest;  LEDGER.gate() -> the kit's exit rule

Python floor: this file is generic core code (3.8-compatible, standard library at import); torch and triton are imported inside the
functions that need them and their absence is a named :class:`Refusal`, never an ImportError at ``import opt_core``.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import sys
import threading
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..counters import Ledger
from ..report import emit, lever_line

NAME = "flash_triattn"                       # the carried kernel this layer serves (META/flash_triattn.json) = the LEVER line's impl=
FAMILY = "F1"
LEVER = "F1.flash_triattn"                   # the LEVER line's name= (the strategy id of this lever)
CLASS_OF_RECORD = "fast"                     # Tier 2 wherever measured (re-associated bf16 products + online softmax)

# fallback event names (the ledger's reason keys; a kit may add its own names through Ledger.fallback(<name>))
BELOW_GATE = "below_gate"                    # S_q below the kit's min_tokens gate → stock path, counted
UNSUPPORTED = "unsupported"                  # the kernel's flash_supported() said no → "unsupported:<its reason>"
KERNEL_UNAVAILABLE = "kernel_unavailable"    # probe() not ok in this process → stock path, counted (only when the kit chose not to require())


class Refusal(RuntimeError):
    """A named refusal: ``kind`` is one of torch_missing · triton_missing · no_cuda_device · cc_lt_8 · kernel_import_failed ·
    no_stock_for_fallback · build_failed (the kernel's picked AND safe settings failed to build in this process); ``detail`` carries the
    underlying message."""

    def __init__(self, kind: str, detail: str = ""):
        self.kind = kind
        self.detail = detail
        super().__init__(f"{NAME}: {kind}" + (f" — {detail}" if detail else ""))


# ----------------------------------------------------------------------------------------------------------------- availability
_PROBE: Optional[Dict[str, Any]] = None
_LOCK = threading.Lock()


def probe(refresh: bool = False) -> Dict[str, Any]:
    """``{ok, kind, detail, torch, triton, device, cc}`` — never raises. ``kind`` is None when ok, else the Refusal kind."""
    global _PROBE
    if _PROBE is not None and not refresh:
        return dict(_PROBE)
    out: Dict[str, Any] = {"ok": False, "kind": None, "detail": "", "torch": None, "triton": None, "device": None, "cc": None}
    try:
        import torch  # noqa: WPS433 (lazy by contract)
        out["torch"] = getattr(torch, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        out.update(kind="torch_missing", detail=repr(e)[:200])
        _PROBE = out
        return dict(out)
    try:
        import triton  # noqa: F401
        out["triton"] = getattr(triton, "__version__", "?")
    except Exception as e:  # noqa: BLE001
        out.update(kind="triton_missing", detail=repr(e)[:200])
        _PROBE = out
        return dict(out)
    try:
        if not torch.cuda.is_available():
            out.update(kind="no_cuda_device", detail="torch.cuda.is_available() is False")
            _PROBE = out
            return dict(out)
        idx = torch.cuda.current_device()
        cc = torch.cuda.get_device_capability(idx)
        out["device"] = torch.cuda.get_device_name(idx)
        out["cc"] = f"{cc[0]}.{cc[1]}"
        if cc[0] < 8:
            out.update(kind="cc_lt_8", detail=f"compute capability {out['cc']} < 8.0")
            _PROBE = out
            return dict(out)
    except Exception as e:  # noqa: BLE001
        out.update(kind="no_cuda_device", detail=repr(e)[:200])
        _PROBE = out
        return dict(out)
    try:
        kernel_module()
    except Refusal as r:
        out.update(kind=r.kind, detail=r.detail)
        _PROBE = out
        return dict(out)
    out["ok"] = True
    _PROBE = out
    return dict(out)


def require() -> Dict[str, Any]:
    """The probe, or a :class:`Refusal` naming why the kernel cannot serve in this process."""
    p = probe()
    if not p["ok"]:
        raise Refusal(p["kind"] or "kernel_import_failed", p["detail"])
    return p


# --------------------------------------------------------------------------------------------------------------- kernel module
_KMOD = None


def _core_copy_path() -> str:
    return os.path.join(os.path.dirname(os.path.abspath(__file__)), NAME + ".py")


def kernel_module():
    """The ``flash_triattn`` module of this process: an already-imported / routed top-level ``flash_triattn`` first (the kit's decision),
    else the core copy imported as ``opt_core.kernels.flash_triattn``. Raises :class:`Refusal` (``kernel_import_failed``) on failure."""
    global _KMOD
    if _KMOD is not None:
        return _KMOD
    with _LOCK:
        if _KMOD is not None:
            return _KMOD
        mod = sys.modules.get(NAME)                                            # a kit that routed or imported the name already
        if mod is None:
            try:
                if NAME in _routed_names():
                    mod = importlib.import_module(NAME)                        # the kit routed the name → the core copy as top-level
                else:
                    mod = importlib.import_module("." + NAME, __package__)     # opt_core.kernels.flash_triattn (the core copy)
            except Exception as e:  # noqa: BLE001
                raise Refusal("kernel_import_failed", repr(e)[:300])
        for fn in ("flash_triangle_attention", "flash_supported"):
            if not hasattr(mod, fn):
                raise Refusal("kernel_import_failed", f"module {getattr(mod, '__file__', '?')} lacks {fn}")
        _KMOD = mod
        return mod


def _routed_names() -> List[str]:
    from . import routed                                                        # opt_core.kernels.routed (stdlib-only)
    return list(routed())


def kernel_origin() -> str:
    """``core`` when the module in use is the core copy (routed or imported as opt_core.kernels.flash_triattn), ``kit`` for a kit's
    own copy (its path is in the activation report's route record), ``unavailable:<kind>`` when it cannot be imported."""
    try:
        mod = kernel_module()
    except Refusal as r:
        return f"unavailable:{r.kind}"
    path = os.path.abspath(getattr(mod, "__file__", "") or "")
    return "core" if path == os.path.abspath(_core_copy_path()) else "kit"


def kernel_impl() -> str:
    """``flash_triattn@<sha256[:8] of the module file in use>`` — the LEVER line's ``impl=``; ``flash_triattn@unavailable`` otherwise."""
    try:
        path = getattr(kernel_module(), "__file__", None)
        with open(path, "rb") as fh:
            return f"{NAME}@{hashlib.sha256(fh.read()).hexdigest()[:8]}"
    except Exception:  # noqa: BLE001
        return f"{NAME}@unavailable"


# ----------------------------------------------------------------------------------------------------------------------- ledger
def ledger(min_tokens: int = 0, *, expected: Tuple[str, ...] = (BELOW_GATE,), impl: Optional[str] = None, origin: Optional[str] = None,
           **kw) -> Ledger:
    """The per-process ``opt_core.counters.Ledger`` of this lever: ``name=F1.flash_triattn``, ``impl`` (default the kernel name; pass
    :func:`kernel_impl` once the kernel is importable for ``flash_triattn@<sha8>``), ``origin`` (:func:`kernel_origin`), the kit's token gate
    ``min_tokens`` (0 = none) and the fallback reasons its mode DECLARES (``expected``; anything else refuses ``Ledger.gate()``).
    ``.serve(shape)`` / ``.fallback(reason)`` / ``.fields()`` / ``.partial`` / ``.state`` / ``.line(tag, ...)`` / ``.gate()`` are the core's."""
    return Ledger(LEVER, impl=impl if impl is not None else NAME, origin=origin, min_tokens=int(min_tokens), expected=expected, **kw)


def emit_line(ledger: Ledger, tag: str, **evidence) -> str:
    """Print this lever's ONE activation-evidence line (``report.emit(ledger.line(tag, ...))``) with ``impl``/``origin`` resolved from the
    kernel in use when the ledger does not carry them yet. Returns the text."""
    if ledger.origin is None:
        ledger.origin = kernel_origin()
    if ledger.impl in (None, NAME):
        ledger.impl = kernel_impl()
    if "cells" not in evidence:                     # the launch-configuration row serving this device: cells=<cc>|<triton mm> (a measured row) or
        key, note = cells_key()                     # cells=<cc>|* (the capability's default row); no fact when the kernel's default tables serve (cc 9.0)
        if key is not None:
            evidence["cells"] = key
        if note is not None and "cells_note" not in evidence:
            evidence["cells_note"] = note        # exception_row: a named-exception row (an exact cc|triton key) serves instead of the capability's default row;
                                                 # default:no_row: an UNKNOWN capability (no row, not cc 9.0) is served the default tables — engaged, named once
    if "settings" not in evidence:
        word = settings_word()
        if word is not None:
            evidence["settings"] = word          # safe:build_failed:<exc> — the kernel switched to its safe single-stage settings after a build failure
    return emit(ledger.line(tag, **evidence))


def settings_word() -> Optional[str]:
    """The kernel module's ``settings_word()`` (``safe:build_failed:<exception class>`` once it serves its safe settings), else None."""
    try:
        mod = kernel_module()
    except Refusal:
        return None
    f = getattr(mod, "settings_word", None)
    return f() if callable(f) else None


def cells_key() -> Tuple[Optional[str], Optional[str]]:
    """``(key, note)`` from the kernel module in use: its ``cells_key()`` / ``cells_note()`` (None, None when the module predates them or no
    CUDA device / cc 9.0 without a row; ``(None, "default:no_row")`` when an UNKNOWN capability is served the default tables)."""
    try:
        mod = kernel_module()
    except Refusal:
        return None, None
    f, g = getattr(mod, "cells_key", None), getattr(mod, "cells_note", None)
    return (f() if callable(f) else None), (g() if callable(g) else None)


# ------------------------------------------------------------------------------------------------------------ serve or fallback
def shape_key(q) -> str:
    """``N<rows>xS<q>xD<d>xH<h>`` from a ``[B, N, H, S, D]`` (or lower-rank) q tensor — the served-shape census key."""
    s = tuple(int(x) for x in q.shape)
    s = (1,) * (5 - len(s)) + s if len(s) < 5 else s[-5:]
    return f"N{s[1]}xS{s[3]}xD{s[4]}xH{s[2]}"


SERVE_DTYPES = ("bfloat16", "float16")             # the compute dtypes served by default (q's dtype, or the autocast dtype when autocast is on)


def triangle_attention(q, k, v, bias, mask=None, scale=None, *, ledger: Ledger, stock: Optional[Callable] = None,
                       min_tokens: Optional[int] = None, serve_dtypes: Tuple[str, ...] = SERVE_DTYPES, **kernel_kwargs):
    """cuEq-signature triangle attention, served by the flash kernel or by ``stock`` with a NAMED event in ``ledger``:

    * ``S_q < min_tokens`` (default: ``ledger.min_tokens``) → ``ledger.fallback("below_gate")``, ``stock(q, k, v, bias, mask, scale)``
    * compute dtype (q's, or the autocast dtype under autocast) not in ``serve_dtypes`` → ``ledger.fallback("unsupported:dtype_<name>")``,
      stock — the served dtypes are the ADAPTER's decision, independent of what the kernel file could accept (a kit serving fp32 says so:
      ``serve_dtypes=("bfloat16", "float16", "float32")`` plus the kernel's precision kwarg)
    * kernel unavailable in this process → ``ledger.fallback("kernel_unavailable:<kind>")``, stock
    * ``flash_supported`` false → ``ledger.fallback("unsupported:<why>")``, stock
    * else the kernel (``kernel_kwargs`` pass through: exact_exp, use_libdevice, bias16, config, out_dtype, input_precision (fp32 inputs:
      ``tf32`` | ``tf32x3`` | ``ieee``), …), ``ledger.serve(shape)``.

    ``stock=None`` makes every fallback a :class:`Refusal` (``no_stock_for_fallback``) — for adapters that gate before calling."""
    reason, mod = _gate(q, k, v, bias, mask, ledger=ledger, min_tokens=min_tokens, serve_dtypes=serve_dtypes)
    if reason is not None:
        ledger.fallback(reason)
        if stock is None:
            raise Refusal("no_stock_for_fallback", f"call not served ({reason}) and the adapter gave no stock path")
        return stock(q, k, v, bias, mask, scale)
    build_failed = getattr(mod, "BuildFailed", ())     # the kernel's "safe settings failed to build too" class (a module that predates it: nothing caught)
    try:
        out = mod.flash_triangle_attention(q, k, v, bias, mask=mask, scale=scale, **kernel_kwargs)
    except build_failed as e:                    # the picked AND the safe settings failed to build: the lever cannot run in this process — never a stock
        raise Refusal("build_failed", str(e)) from e  # reroute: the kit's hard error names its one-flag escape (--mode off / its explicit opt-out for this lever)
    ledger.serve(shape_key(q))
    return out


def _gate(q, k, v, bias, mask, *, ledger: Ledger, min_tokens: Optional[int], serve_dtypes: Tuple[str, ...]):
    """The documented gates of :func:`triangle_attention` in their order: ``(reason, kernel module)`` — ``reason`` None = serve with ``module``."""
    gate = int(ledger.min_tokens or 0) if min_tokens is None else int(min_tokens)
    sq = int(q.shape[-2])
    reason: Optional[str] = None
    mod = None
    if gate and sq < gate:
        reason = BELOW_GATE
    elif _compute_dtype_name(q) not in serve_dtypes:
        reason = f"{UNSUPPORTED}:dtype_{_compute_dtype_name(q)}"
    else:
        try:
            mod = kernel_module()
        except Refusal as r:
            reason = f"{KERNEL_UNAVAILABLE}:{r.kind}"
        if mod is not None:
            ok, why = mod.flash_supported(q, k, v, bias, mask) if _supports_mask_arg(mod) else mod.flash_supported(q, k, v, bias)
            if not ok:
                reason = f"{UNSUPPORTED}:{why}"
    return reason, mod


def query_blocks(sq: int, qblock: int) -> List[Tuple[int, int]]:
    """``[(j0, j1), ...]`` covering ``range(sq)`` in blocks of ``qblock`` queries (the last one ragged); one block when ``qblock`` >= ``sq`` or <= 0."""
    sq, qblock = int(sq), int(qblock)
    if qblock <= 0 or qblock >= sq:
        return [(0, sq)]
    return [(j0, min(sq, j0 + qblock)) for j0 in range(0, sq, qblock)]


INPLANE_INT32_LIM = 2 ** 31 - 2 ** 24                        # the kernel's bias-prep SOURCE offsets are int32: row*stride(-2) + col*stride(-1), cols padded to 16


def inplane_int32_safe(b) -> bool:
    """True when the kernel's bias-prep pass can read this block ``[.., rows, cols]`` through its own strides: its in-plane source offsets
    ``row * stride(-2) + col * stride(-1)`` (int32; ``col`` runs to ``ceil16(cols)``) stay below :data:`INPLANE_INT32_LIM`. A starting-node view of a
    channel-last ``[S, S, H]`` plane (strides ``S*H``, ``H``) is safe per query block at every S; its TRANSPOSED (ending-node) view (key stride ``S*H``)
    is not once ``S*S*H`` reaches 2**31 — that block is handed as a small fp32 contiguous copy instead (:func:`triangle_attention_qblocks`)."""
    rows, cols = int(b.shape[-2]), int(b.shape[-1])
    colsp = -(-cols // 16) * 16
    return max(rows - 1, 0) * abs(int(b.stride(-2))) + max(colsp - 1, 0) * abs(int(b.stride(-1))) < INPLANE_INT32_LIM


def triangle_attention_qblocks(q, k, v, bias, mask=None, scale=None, *, qblock: int, ledger: Ledger, stock: Optional[Callable] = None,
                               min_tokens: Optional[int] = None, serve_dtypes: Tuple[str, ...] = SERVE_DTYPES, **kernel_kwargs):
    """:func:`triangle_attention` with the kernel launched per QUERY block: ``out[..., j0:j1, :] = kernel(q[..., j0:j1, :], k, v, bias[..., j0:j1, :],
    mask, scale)`` over blocks of ``qblock`` queries (:func:`query_blocks`). For pair planes whose whole-plane bias copy the kernel makes per launch
    (``[B, 1, H, S_q, ceil16(S_k)]`` fp32, indexed ``row * pitch`` in int32) must stay below 2**31 elements and small: a block's copy is ``H * qblock *
    ceil16(S_k)`` elements (``qblock`` is clamped so that a block's plane stays inside that bound). ``bias`` is the caller's ``[B, 1, H, S_q, S_k]``
    tensor in ANY dtype / strides: a block whose strided view the kernel's prep can index in int32 (:func:`inplane_int32_safe` — the starting-node
    view at every S) is handed AS IS (no copy); any other (a transposed ending-node view above ``S*S*H >= 2**31``) as a small fp32 CONTIGUOUS copy
    (``H * qblock * S_k * 4`` bytes; no whole-plane copy is ever made here). Keys / values whole and every query row's
    online softmax over all ``S_k`` keys in the kernel's fixed key-tile order: per-query arithmetic identical to one launch over all queries (the
    kernel's q-tile programs are independent; measured bitwise-equal to the one-launch call). The gates, their
    events and ``stock`` (ONE call over all queries on a fallback), the served census (ONE ``serve`` per call) and the ``build_failed`` refusal are
    :func:`triangle_attention`'s exactly."""
    reason, mod = _gate(q, k, v, bias, mask, ledger=ledger, min_tokens=min_tokens, serve_dtypes=serve_dtypes)
    if reason is not None:
        ledger.fallback(reason)
        if stock is None:
            raise Refusal("no_stock_for_fallback", f"call not served ({reason}) and the adapter gave no stock path")
        return stock(q, k, v, bias, mask, scale)
    if int(bias.dim()) < 2 or int(bias.shape[-2]) != int(q.shape[-2]):
        raise ValueError(f"triangle_attention_qblocks: bias {tuple(bias.shape)} must carry the query dim S_q={int(q.shape[-2])} at -2")
    build_failed = getattr(mod, "BuildFailed", ())
    import torch  # noqa: WPS433 (present: q is a torch tensor)
    if q.stride(-1) != 1:                                  # the kernel's own last-dim rule, applied ONCE (it would copy k / v per block otherwise)
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    sq, sk = int(q.shape[-2]), int(k.shape[-2])
    qblock = max(1, min(int(qblock) if int(qblock) > 0 else sq, (2 ** 31 - 2 ** 24) // (-(-sk // 16) * 16)))   # a block's plane qblock*ceil16(S_k) stays inside the kernel's int32 row-pitch bound whatever the env says
    out = None
    for j0, j1 in query_blocks(sq, qblock):
        bv = bias[..., j0:j1, :]
        if inplane_int32_safe(bv):                                                                 # the kernel's prep pass reads the block's STRIDED view directly (its int32 in-plane source
            bj = bv                                                                                #  offsets row*stride(-2) + col*stride(-1) stay below 2**31): no copy — the starting-node view
        else:                                                                                      #  [.., H, S, S] of a channel-last [S, S, H] plane (strides S*H, H) at any S the shots reach
            bj = torch.empty(tuple(bv.shape), dtype=torch.float32, device=bv.device)              # else (an ending-node TRANSPOSED view: key stride S*H, (S-1)*S*H >= 2**31 exactly above the
            bj.copy_(bv)                                                                           #  bound) ONE fused cast+layout pass into a small fp32 contiguous block, H*qblock*S_k*4 bytes,
                                                                                                   #  freed per block. Same fp32 values reach the prep either way: outputs identical.
        try:                                                                                       #  view included); H*qblock*S_k*4 bytes, freed per block
            ob = mod.flash_triangle_attention(q[..., j0:j1, :], k, v, bj, mask=mask, scale=scale, **kernel_kwargs)
        except build_failed as e:
            raise Refusal("build_failed", str(e)) from e
        del bj
        if out is None:
            if j0 == 0 and j1 == sq:
                out = ob
                break
            out = ob.new_empty(tuple(ob.shape[:-2]) + (sq,) + tuple(ob.shape[-1:]))
        out[..., j0:j1, :].copy_(ob)
        del ob
    ledger.serve(shape_key(q))
    return out


def _compute_dtype_name(q) -> str:
    """``bfloat16`` / ``float16`` / ``float32`` …: q's dtype, or the CUDA autocast dtype when autocast is enabled (what the kernel computes in)."""
    name = str(getattr(q, "dtype", "unknown")).replace("torch.", "")
    try:
        import torch  # noqa: WPS433 (present whenever q is a torch tensor)
        if torch.is_autocast_enabled():
            name = str(torch.get_autocast_dtype("cuda")).replace("torch.", "")
    except Exception:  # noqa: BLE001
        pass
    return name


def _supports_mask_arg(mod) -> bool:
    fn = getattr(mod, "flash_supported", None)
    code = getattr(fn, "__code__", None)
    return bool(code is not None and "mask" in code.co_varnames[: code.co_argcount])


# ------------------------------------------------------------------------------------------- staged bias: prepare ONCE, launch per row window
PREPPED_NEEDS = ("_launch_prep", "_launch", "pick_config", "_launch_or_safe", "_maybe_to", "_ensure_dims", "_exp_mode", "_device_cc", "_triton_mm",
                 "_ONES", "_SUPPORTED_D", "_SUPPORTED_DTYPES", "_BIAS16_ENABLED", "_BIAS16_MIN_SK", "_IP_DEFAULT", "_EXACT_DEFAULT", "_LIBDEVICE_DEFAULT",
                 "_safe", "triton")             # the kernel module's names the prepped launch reads (a module that predates them: Refusal prepped_unavailable, by name)


class Prepped(object):
    """The flash kernel's bias operands of ONE plane (or one query block of it), prepared once: ``bias32`` fp32 ``[B, 1, H, S_q, ceil16(S_k)]``
    (unit key stride, 16-element row pitch), ``bias16`` its 16-bit copy in the q dtype + ``flags`` (per 32-query block: the fp32 -> 16-bit cast
    was lossless) when the launch settings want 16-bit bias tiles, else None -- exactly the tensors ``flash_triangle_attention`` builds with its
    ``_launch_prep`` pass on every call (:func:`prep_bias` runs that pass once), so every row window of the same plane launches the attention
    kernel alone (:func:`flash_prepped`). ``nbytes``: device bytes held."""
    __slots__ = ("bias32", "bias16", "flags", "n_flags", "has16", "B", "H", "SQ", "SK", "SKp", "D", "dtype", "device", "nbytes")

    def __init__(self, **kw):
        for k_ in self.__slots__:
            setattr(self, k_, kw.get(k_))

    def dims(self):
        return (int(self.B), int(self.H), int(self.SQ), int(self.SK))


def _prepped_module():
    """The kernel module in use when it carries what the prepped launch needs; Refusal(prepped_unavailable) by name otherwise."""
    mod = kernel_module()
    if callable(getattr(mod, "prep_bias", None)) and callable(getattr(mod, "flash_triangle_attention_prepped", None)):
        return mod                                        # a kernel module with its own staged entries: those serve
    missing = [n for n in PREPPED_NEEDS if not hasattr(mod, n)]
    if missing:
        raise Refusal("prepped_unavailable", "the flash_triattn module in use (%s) lacks %s" % (getattr(mod, "__file__", "?"), ",".join(missing[:4])))
    return mod


def prep_bias(bias, *, head_dim: int, dtype, device=None) -> Prepped:
    """Run the kernel's bias-preparation pass ONCE for ``bias`` ``[B, 1, H, S_q, S_k]`` (fp32 | bf16 | fp16, any strides whose in-plane source
    offsets fit int32 -- :func:`inplane_int32_safe`; a 16-bit bias is upcast with ``.float()`` first, as the kernel does per call) for launches
    whose q has ``head_dim`` / ``dtype``: the statements of ``flash_triangle_attention`` (lines 'bias preparation') with nothing else. Same
    source values -> the same ``bias32`` / ``bias16`` / ``flags`` bytes the per-call path builds, so :func:`flash_prepped` outputs are bitwise
    the per-call outputs. ``S_q * ceil16(S_k) >= 2**31`` raises ValueError as the kernel does (prep such planes per query block)."""
    import torch
    mod = _prepped_module()
    own = getattr(mod, "prep_bias", None)
    if callable(own):
        return own(bias, head_dim=int(head_dim), dtype=dtype, device=device)
    triton = mod.triton
    bias = mod._ensure_dims(bias, 5)
    B, one, H, SQ, SK = (int(x) for x in bias.shape)
    if one != 1:
        raise ValueError(f"prep_bias: bias must be [B, 1, H, S_q, S_k], got {tuple(bias.shape)}")
    D = int(head_dim)
    if dtype not in mod._SUPPORTED_DTYPES or D not in mod._SUPPORTED_D:
        raise ValueError(f"prep_bias: unsupported dtype/head_dim {dtype}/{D}")
    dev = bias.device if device is None else device
    SKp = -(-SK // 16) * 16
    if SQ * SKp >= 2 ** 31:
        raise ValueError("prep_bias: plane too large for the kernel's int32 row pitch (S_q * ceil16(S_k) >= 2**31): prep it per query block")
    if bias.dtype != torch.float32:
        bias = bias.float()                               # the kernel's own per-call upcast, once (freed when this returns)
    cfg = mod.pick_config(D, SQ, SK, H, dtype, dev)
    want16 = bool(cfg.get("BIAS16", mod._BIAS16_ENABLED and SK >= mod._BIAS16_MIN_SK))
    if dtype == torch.float32:
        want16 = False
    PB, PK = 32, 128
    nqb = triton.cdiv(SQ, PB)
    n_flags = B * H * nqb
    bias32 = torch.empty((B, 1, H, SQ, SKp), dtype=torch.float32, device=dev)
    if want16:
        bias16_t = torch.empty((B, 1, H, SQ, SKp), dtype=dtype, device=dev)
        flags = torch.empty((max(n_flags, 1),), dtype=torch.int32, device=dev)
    else:
        bias16_t = bias32
        flags = torch.ones((1,), dtype=torch.int32, device=dev) if mod._ONES.get(dev) is None else mod._ONES[dev]
        mod._ONES[dev] = flags
        n_flags = 1
    mod._launch_prep((nqb, B * H), (bias, bias32, bias16_t, flags),
                     (bias.stride(0), bias.stride(2), bias.stride(3), bias.stride(4), H, SQ, SK, SKp, nqb),
                     dict(PB=PB, PK=PK, MAKE16=want16))
    nbytes = bias32.numel() * 4 + ((bias16_t.numel() * bias16_t.element_size() + flags.numel() * 4) if want16 else 0)
    return Prepped(bias32=bias32, bias16=(bias16_t if want16 else None), flags=(flags if want16 else None), n_flags=int(n_flags), has16=bool(want16),
                   B=B, H=H, SQ=SQ, SK=SK, SKp=SKp, D=D, dtype=dtype, device=dev, nbytes=int(nbytes))


def flash_prepped(q, k, v, prep: Prepped, mask=None, scale=None, *, exact_exp=None, use_libdevice=None, config=None, out_dtype=None,
                  input_precision=None):
    """``flash_triangle_attention(q, k, v, <the plane prep was made from>, mask, scale, ...)`` WITHOUT the per-call bias preparation: the
    kernel wrapper's statements with ``prep``'s tensors in place of its ``_launch_prep`` outputs (shape / dtype checks against ``prep``;
    ValueError on a mismatch). Outputs bitwise those of the per-call entry on the same operands."""
    import math
    import torch
    mod = _prepped_module()
    own = getattr(mod, "flash_triangle_attention_prepped", None)
    if callable(own):
        return own(q, k, v, prep, mask=mask, scale=scale, exact_exp=exact_exp, use_libdevice=use_libdevice, config=config, out_dtype=out_dtype,
                   input_precision=input_precision)
    triton = mod.triton
    if exact_exp is None:
        exact_exp = mod._EXACT_DEFAULT
    if use_libdevice is None:
        use_libdevice = mod._LIBDEVICE_DEFAULT
    q = mod._ensure_dims(q, 5); k = mod._ensure_dims(k, 5); v = mod._ensure_dims(v, 5)
    if mask is not None:
        mask = mod._ensure_dims(mask, 5)
        if mask.dtype != torch.bool:
            mask = mask.to(torch.bool)
    B, N, H, SQ, D = (int(x) for x in q.shape)
    SK = int(k.shape[3])
    if tuple(k.shape) != (B, N, H, SK, D) or tuple(v.shape) != (B, N, H, SK, D):
        raise ValueError(f"flash_prepped: k/v must be (B, N, H, S_kv, D) = {(B, N, H, SK, D)}, got {tuple(k.shape)} / {tuple(v.shape)}")
    if prep.dims() != (B, H, SQ, SK) or int(prep.D) != D:
        raise ValueError(f"flash_prepped: prep is for (B, H, S_q, S_k, D) = {prep.dims() + (prep.D,)}, the call is {(B, H, SQ, SK, D)}")
    if mask is not None and tuple(mask.shape) != (B, N, 1, 1, SK):
        raise ValueError(f"flash_prepped: mask must be (B, N, 1, 1, S_kv), got {tuple(mask.shape)}")
    if torch.is_autocast_enabled():
        adt = torch.get_autocast_dtype("cuda")
        q = mod._maybe_to(q, adt); k = mod._maybe_to(k, adt); v = mod._maybe_to(v, adt)
    if k.dtype != q.dtype:
        k = k.to(q.dtype)
    if v.dtype != q.dtype:
        v = v.to(q.dtype)
    if q.dtype != prep.dtype:
        raise ValueError(f"flash_prepped: prep was made for q dtype {prep.dtype}, the call computes in {q.dtype}")
    if scale is None:
        scale = 1.0 / math.sqrt(D)
    odt = q.dtype if out_dtype is None else out_dtype
    out = torch.empty((B, N, H, SQ, D), dtype=odt, device=q.device)
    if out.numel() == 0:
        return out
    if SK == 0:
        return out.fill_(float("nan"))
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    if mask is not None:
        mask_u8 = mask.view(torch.uint8)
        smb, smi, smk = mask_u8.stride(0), mask_u8.stride(1), mask_u8.stride(4)
        has_mask = True
    else:
        mask_u8 = out
        smb = smi = smk = 0
        has_mask = False
    cfg = dict(config) if config is not None else mod.pick_config(D, SQ, SK, H, q.dtype, q.device)
    want16 = bool(cfg.get("BIAS16", mod._BIAS16_ENABLED and SK >= mod._BIAS16_MIN_SK)) and prep.has16      # the settings' wish AND the prepared copy (fp32 tiles otherwise: same results)
    if q.dtype == torch.float32:
        want16 = False
    SKp = int(prep.SKp)
    bias32 = prep.bias32
    if want16:
        bias16_t, flags, n_flags = prep.bias16, prep.flags, int(prep.n_flags)
    else:
        bias16_t = bias32
        flags = torch.ones((1,), dtype=torch.int32, device=q.device) if mod._ONES.get(q.device) is None else mod._ONES[q.device]
        mod._ONES[q.device] = flags
        n_flags = 1
    int_args = (q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                bias32.stride(0), bias32.stride(2), bias32.stride(3),
                smb, smi, smk,
                out.stride(0), out.stride(1), out.stride(2), out.stride(3),
                N, SQ, SK, H, n_flags)

    def _attn(c) -> None:
        bm, bn, rows, order = c["BLOCK_M"], c["BLOCK_N"], c["ROWS"], int(c.get("ORDER", 0))
        assert rows in (1, 2, 4), rows
        n_q, n_r = triton.cdiv(SQ, bm), triton.cdiv(N, rows)
        grid = (n_q, n_r, B * H) if order == 0 else (n_r, n_q, B * H)
        if int(bn) * max(int(k.stride(3)), int(v.stride(3))) + D >= 2 ** 31 or SQ * SKp >= 2 ** 31:
            raise ValueError("flash_triangle_attention: tensor too large for int32 tile offsets")
        const_kwargs = dict(HEAD_DIM=D, BLOCK_M=bm, BLOCK_N=bn, ROWS=rows, HAS_MASK=has_mask, EXP_MODE=mod._exp_mode(bool(exact_exp), bool(use_libdevice)),
                            BIAS16=1 if want16 else 0, ORDER=order, MSCAN=1024, NFLAG=max(1024, triton.next_power_of_2(n_flags)),
                            IP=(input_precision or mod._IP_DEFAULT))
        mod._launch(grid, (q, k, v, bias32, bias16_t, flags, mask_u8, out), int_args, float(scale), const_kwargs, c["num_warps"], c["num_stages"])

    mod._launch_or_safe(_attn, cfg, D, q.dtype, explicit=config is not None,
                        where=mod._safe.where_word(mod._device_cc(q.device) if q.is_cuda else None, mod._triton_mm()))
    return out


def triangle_attention_prepped(q, k, v, bias, prep: Prepped, mask=None, scale=None, *, ledger: Ledger, stock: Optional[Callable] = None,
                               min_tokens: Optional[int] = None, serve_dtypes: Tuple[str, ...] = SERVE_DTYPES, **kernel_kwargs):
    """:func:`triangle_attention` on a plane prepared ONCE (``prep = prep_bias(bias, ...)``): the same gates in the same order with the same
    events (``bias`` = the caller's plane, read by the gates only), ``stock(q, k, v, bias, mask, scale)`` on a fallback, ONE ``serve`` per call,
    the ``build_failed`` refusal; the kernel launched by :func:`flash_prepped` (no per-call bias pass)."""
    reason, mod = _gate(q, k, v, bias, mask, ledger=ledger, min_tokens=min_tokens, serve_dtypes=serve_dtypes)
    if reason is not None:
        ledger.fallback(reason)
        if stock is None:
            raise Refusal("no_stock_for_fallback", f"call not served ({reason}) and the adapter gave no stock path")
        return stock(q, k, v, bias, mask, scale)
    build_failed = getattr(mod, "BuildFailed", ())
    try:
        out = flash_prepped(q, k, v, prep, mask=mask, scale=scale, **kernel_kwargs)
    except build_failed as e:
        raise Refusal("build_failed", str(e)) from e
    ledger.serve(shape_key(q))
    return out


def prep_bias_qblocks(bias, *, qblock: int, head_dim: int, dtype) -> List[Tuple[int, int, Prepped]]:
    """:func:`prep_bias` per QUERY block of ``bias`` ``[B, 1, H, S_q, S_k]`` (:func:`query_blocks` of ``qblock`` queries, clamped so a block's
    plane stays inside the kernel's int32 row pitch as :func:`triangle_attention_qblocks` clamps it): each block from the caller's strided view when
    :func:`inplane_int32_safe`, else from a small fp32 contiguous copy (freed per block) -- the per-call path's source rule exactly. Holds
    ``(4 [+ 2]) * B * H * S_q * ceil16(S_k)`` bytes for the whole list."""
    import torch
    sq, sk = int(bias.shape[-2]), int(bias.shape[-1])
    qb = max(1, min(int(qblock) if int(qblock) > 0 else sq, (2 ** 31 - 2 ** 24) // (-(-sk // 16) * 16)))
    preps = []
    for j0, j1 in query_blocks(sq, qb):
        bv = bias[..., j0:j1, :]
        if inplane_int32_safe(bv):
            bj = bv
        else:
            bj = torch.empty(tuple(bv.shape), dtype=torch.float32, device=bv.device)
            bj.copy_(bv)
        preps.append((j0, j1, prep_bias(bj, head_dim=head_dim, dtype=dtype)))
        del bj
    return preps


def triangle_attention_prepped_qblocks(q, k, v, bias, preps: List[Tuple[int, int, Prepped]], mask=None, scale=None, *, ledger: Ledger,
                                       stock: Optional[Callable] = None, min_tokens: Optional[int] = None,
                                       serve_dtypes: Tuple[str, ...] = SERVE_DTYPES, **kernel_kwargs):
    """:func:`triangle_attention_qblocks` on query blocks prepared ONCE (``preps = prep_bias_qblocks(bias, ...)``): gates / events / stock / ONE
    serve per call exactly as there; each block launched by :func:`flash_prepped` on ``q[..., j0:j1, :]`` with keys / values / mask whole."""
    reason, mod = _gate(q, k, v, bias, mask, ledger=ledger, min_tokens=min_tokens, serve_dtypes=serve_dtypes)
    if reason is not None:
        ledger.fallback(reason)
        if stock is None:
            raise Refusal("no_stock_for_fallback", f"call not served ({reason}) and the adapter gave no stock path")
        return stock(q, k, v, bias, mask, scale)
    build_failed = getattr(mod, "BuildFailed", ())
    if q.stride(-1) != 1:
        q = q.contiguous()
    if k.stride(-1) != 1:
        k = k.contiguous()
    if v.stride(-1) != 1:
        v = v.contiguous()
    sq = int(q.shape[-2])
    out = None
    for j0, j1, pj in preps:
        try:
            ob = flash_prepped(q[..., j0:j1, :], k, v, pj, mask=mask, scale=scale, **kernel_kwargs)
        except build_failed as e:
            raise Refusal("build_failed", str(e)) from e
        if out is None:
            if j0 == 0 and j1 == sq:
                out = ob
                break
            out = ob.new_empty(tuple(ob.shape[:-2]) + (sq,) + tuple(ob.shape[-1:]))
        out[..., j0:j1, :].copy_(ob)
        del ob
    ledger.serve(shape_key(q))
    return out
