"""Host-resident input features under row sharding — the large, read-once-per-cycle features of a cofold engine (the raw MSA one-hot and
deletion features ``[S_msa, N, ·]``, an ``[N, N]`` token-bond matrix, ...) stay in (pinned) HOST memory and only what a cycle uses reaches the
device: the rows the engine's per-cycle MSA subsample selects (``<= 1,024`` of up to 16,384), a row slab of a pair matrix at its point of use.
At N = 31,140 with a 16,384-row MSA the raw MSA alone is 65 GB fp32 per rank on the device and an engine's ``cat([msa, has_deletion,
deletion_value])`` adds a 69 GB transient per cycle; parked, the device holds the selected rows only (4.3 GB). Values are untouched
(placement only): the EXACT class. The engine's subsample statements (its RNG calls, in its order, on its device) stay the adapter's — the
selection indices are applied to the host copies here (:meth:`HostTensor.select`), and :func:`opt_core.mem.rowpair.msa.guard_replicated` /
``draw_replicated`` prove / make them identical across ranks.

    host_mode(value=None)                 the placement words: ``ROWPAIR_MSA_HOST`` unset / ``0`` -> None (features stay where the engine put
                                          them); ``1`` / ``all`` -> ``"all"`` (every rank keeps its own host copy and gathers its rows itself);
                                          ``rank0`` -> ``"rank0"`` (only rank 0 keeps the host copy; ranks > 0 hold ZERO-ROW placeholders and
                                          receive the selected rows by a chunked device broadcast per cycle — no host RAM either)
    to_host(t, name, pin=True)            the host copy of a tensor: PINNED when CUDA is present and pinning succeeds, else pageable — which one
                                          is printed and recorded (``msa_host_pinned_gib`` / ``msa_host_pageable_gib``), never silent
    HostTensor(t, name, row_dim, pin)     holder of one parked feature: ``.host .pinned .row_dim .nbytes``, ``.select(index)`` (rows gathered ON
                                          THE HOST: ``index_select`` with the index moved to the host), ``.rows_to(index, device, dtype)``,
                                          ``.to(device)``, ``.placeholder()``, ``.facts()``
    host_placeholder(t, row_dim)          the zero-row placeholder of ``t`` (other dims kept) — what ranks > 0 hold in mode ``rank0``
    park_features(feats, keys, *, mode, row_dims, pin)
                                          in place on a feature dict: each key -> its host copy (idempotent; already-parked tensors are kept) or,
                                          in mode ``rank0`` on ranks > 0, its placeholder; returns the census facts
    is_parked(t)                          True for tensors produced by :func:`to_host` (or pinned host tensors)
    bcast_chunked_(t, src, chunk_gb)      in-place broadcast of a contiguous DEVICE tensor in ``<= ROWPAIR_BCAST_CHUNK_GB`` GiB pieces (default 2)
                                          through :func:`opt_core.mem.rowpair.dist.comm` — the only communication of this module
    rows_to_device(build_fn, *, shape, dtype, device, mode)
                                          the selected rows ON THE DEVICE on every rank: mode None / ``all`` -> ``build_fn()`` everywhere (each
                                          rank's own host gather + H2D); mode ``rank0`` at P > 1 -> rank 0 runs ``build_fn()``, ranks > 0 allocate
                                          ``empty(shape, dtype, device)`` and receive it by :func:`bcast_chunked_`
    sync_host_features_(feats, keys, device, src=0)
                                          mode ``all`` batch sync: every rank's HOST copies of ``keys`` made equal to rank ``src``'s (shape / dtype
                                          meta first, then ``<= ROWPAIR_BCAST_CHUNK_GB`` pieces staged through ``device``); no-op at P == 1 or in
                                          mode ``rank0`` (nothing to sync: ranks > 0 hold placeholders)
    residency_rows(feats) / residency_text(feats, tag)
                                          the census of a feature dict: every tensor's bytes, shape, dtype and where it lives (cuda | pinned | host)

P == 1: every function is the local statement (no communication; mode ``rank0`` degenerates to ``all``).
"""
from __future__ import annotations

import os
import time
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from . import RowpairRefused
from ._torch import torch
from .dist import broadcast_obj, is_dist, world
from .evidence import record_schedule

__all__ = ["ENV_MSA_HOST", "ENV_BCAST_CHUNK_GB", "BCAST_CHUNK_GB_DEFAULT", "host_mode", "to_host", "HostTensor", "host_placeholder", "host_placeholder_of", "place_skipped",
           "park_features", "is_parked", "is_pinned", "where", "parked_rows", "bcast_chunked_", "rows_to_device", "sync_host_features_", "residency_rows", "residency_text"]

ENV_MSA_HOST = "ROWPAIR_MSA_HOST"              # unset|0 -> None ; 1|all|true -> "all" ; rank0 -> "rank0"
ENV_BCAST_CHUNK_GB = "ROWPAIR_BCAST_CHUNK_GB"  # chunk size of the device broadcasts of this module (GiB)
BCAST_CHUNK_GB_DEFAULT = 2.0
_MODES = {"": None, "0": None, "false": None, "off": None, "1": "all", "all": "all", "true": "all", "on": "all", "rank0": "rank0"}
_HOST_ATTR = "_rowpair_host"
_PIN_FAILED_ATTR = "_rowpair_pin_failed"     # a host copy whose page-lock was asked and refused (pageable, named)
_PLACEHOLDER_ATTR = "_rowpair_placeholder"   # the zero-row stand-in of mode rank0 on ranks > 0
_ROWS_ATTR = "_rowpair_parked_rows"          # the row count a placeholder stands in for
_FACTS = {"pinned_bytes": 0, "pageable_bytes": 0, "tensors": 0}


def _log(msg: str) -> None:
    from ...report import emit
    emit(f"[rowpair] msa_host: {msg}")


def _comm():
    """The per-rank communicator of :mod:`opt_core.mem.rowpair.dist` (``comm()``): ``bcast_(t, src)``, ``rank``, ``world``."""
    from . import dist as _D
    factory = getattr(_D, "comm", None)
    if factory is None:
        raise RowpairRefused("msa_host: opt_core.mem.rowpair.dist.comm() is required for device broadcasts (the rowpair comm surface)")
    return factory()


def host_mode(value: Optional[str] = None) -> Optional[str]:
    """The placement mode: ``value`` (else ``ROWPAIR_MSA_HOST``) mapped to None | ``"all"`` | ``"rank0"``; an unknown word is refused by name."""
    v = (os.environ.get(ENV_MSA_HOST, "") if value is None else str(value)).strip().lower()
    if v not in _MODES:
        raise RowpairRefused(f"host_mode: {ENV_MSA_HOST}={v!r}: one of 0 | 1 | all | rank0 is required")
    return _MODES[v]


def _chunk_gb(chunk_gb: Optional[float]) -> float:
    if chunk_gb is not None:
        return float(chunk_gb)
    v = os.environ.get(ENV_BCAST_CHUNK_GB, "").strip()
    if not v:
        return BCAST_CHUNK_GB_DEFAULT
    try:
        return float(v)
    except ValueError:
        raise RowpairRefused(f"{ENV_BCAST_CHUNK_GB}={v!r}: a number of GiB is required") from None


# ----------------------------------------------------------------------------------------------------------------- host placement
def is_pinned(t) -> bool:
    """True for a page-locked host tensor (a CPU-only torch build answers False)."""
    try:
        return bool(t.is_pinned())
    except Exception:  # noqa: BLE001 — a CPU-only torch build answers False by raising
        return False


_is_pinned = is_pinned


def where(t) -> str:
    """The placement word of a feature after :func:`park_features` / :func:`to_host`, in the family's park vocabulary: ``host_pinned`` |
    ``host_pageable:pin_alloc_failed`` (a page-lock was asked and refused — said on the census line when it happened) | ``host`` (pageable, no
    page-lock asked or no CUDA) | ``device`` | ``placeholder`` (the zero-row stand-in of mode ``rank0`` on ranks > 0)."""
    if not torch.is_tensor(t):
        return "n/a"
    if t.is_cuda:
        return "device"
    if is_pinned(t):
        return "host_pinned"
    if getattr(t, _PIN_FAILED_ATTR, False):
        return "host_pageable:pin_alloc_failed"
    if getattr(t, _PLACEHOLDER_ATTR, False):
        return "placeholder"
    return "host"


def is_parked(t) -> bool:
    """True for a host tensor produced by :func:`to_host` (marked) or any pinned host tensor."""
    return bool(torch.is_tensor(t) and (not t.is_cuda) and (getattr(t, _HOST_ATTR, False) or _is_pinned(t)))


def to_host(t, name: str = "tensor", pin: bool = True, log: bool = True):
    """The host copy / keep of ``t``: PINNED when ``pin`` and CUDA is available and the pinned allocation succeeds (a MEMLOCK-capped host falls
    back to PAGEABLE — printed), else pageable (``t`` itself when it already is a contiguous CPU tensor). One census line per call (``log``);
    the bytes land in ``msa_host_pinned_gib`` / ``msa_host_pageable_gib`` of the schedule census. Non-tensors pass through."""
    if not torch.is_tensor(t):
        return t
    src = t.detach()
    nbytes = int(src.numel()) * int(src.element_size())
    out = None
    pinned = False
    pin_failed = False
    if pin and torch.cuda.is_available():
        try:
            out = torch.empty(src.shape, dtype=src.dtype, pin_memory=True)
            out.copy_(src)
            pinned = True
        except RuntimeError as e:
            if log:
                _log(f"pinning {nbytes / 2 ** 30:.2f} GiB for {name!r} failed ({str(e)[:60]}); pageable host memory")
            out = None
            pin_failed = True
    if out is None:
        out = src.cpu() if src.is_cuda else src.contiguous()
    try:
        setattr(out, _HOST_ATTR, True)
        if pin_failed:
            setattr(out, _PIN_FAILED_ATTR, True)
    except Exception:  # noqa: BLE001
        pass
    _FACTS["tensors"] += 1
    _FACTS["pinned_bytes" if pinned else "pageable_bytes"] += nbytes
    record_schedule(msa_host_pinned_gib=round(_FACTS["pinned_bytes"] / 2 ** 30, 2), msa_host_pageable_gib=round(_FACTS["pageable_bytes"] / 2 ** 30, 2))
    if log:
        _log(f"{name!r} {tuple(src.shape)} {str(src.dtype).replace('torch.', '')} {nbytes / 2 ** 30:.2f} GiB kept on {'PINNED' if pinned else 'pageable'} host")
    return out


def host_placeholder(t, row_dim: int):
    """The ZERO-ROW placeholder of ``t``: same dtype and dims, ``0`` on ``row_dim`` (host) — ranks > 0 in mode ``rank0`` hold this instead of
    the feature (row SELECTION is computed from other, replicated tensors; the selected rows arrive by broadcast). The placeholder carries the
    row count it stands in for (:func:`parked_rows`)."""
    return host_placeholder_of(tuple(t.shape), t.dtype, row_dim)


def host_placeholder_of(shape: Sequence[int], dtype, row_dim: int):
    """:func:`host_placeholder` from a tensor's ``shape`` / ``dtype`` alone (a rank that never held the tensor: the source rank's meta names them;
    ``dtype`` a torch dtype or its name)."""
    shp = [int(x) for x in shape]
    rd = row_dim % len(shp)
    n_rows = int(shp[rd])
    shp[rd] = 0
    ph = torch.empty(shp, dtype=getattr(torch, dtype) if isinstance(dtype, str) else dtype)
    setattr(ph, _PLACEHOLDER_ATTR, True)                       # a torch.Tensor carries python attributes
    setattr(ph, _ROWS_ATTR, n_rows)
    return ph


def place_skipped(feats: Dict[str, object], skipped: Mapping[str, Mapping], *, row_dims: Union[int, Mapping[str, int]] = 0) -> dict:
    """Mode ``rank0`` on a rank that RECEIVED the tree without the parked keys (:func:`rankdata.broadcast_features` ``skipped`` =
    ``{key: {"shape", "dtype", "dev"}}``): set ``feats[key]`` to the zero-row placeholder (:func:`host_placeholder_of`) exactly as
    :func:`park_features` leaves it on ranks > 0, and record the same facts. Returns ``{"placeholders": [keys], "where": {key: "placeholder"},
    "rows": {key: n}}``."""
    facts = {"placeholders": [], "where": {}, "rows": {}}
    for k, m in skipped.items():
        rd = row_dims if isinstance(row_dims, int) else int(dict(row_dims).get(k, 0))
        feats[k] = host_placeholder_of(m["shape"], m["dtype"], rd)
        facts["placeholders"].append(k)
        facts["where"][k] = "placeholder"
        facts["rows"][k] = parked_rows(feats[k], rd)
    if facts["placeholders"]:
        record_schedule(msa_host_mode="rank0", msa_host_where=",".join(f"{k}:placeholder" for k in facts["placeholders"]))
    return facts


def parked_rows(t, row_dim: int = 0) -> int:
    """The parked ROW COUNT of a feature on any rank: a placeholder (mode ``rank0``, ranks > 0) carries the row count of the tensor it stands in
    for (``S_total``), so an engine without a row mask needs no broadcast of the sampler's ``n`` from rank 0; a host copy / device tensor answers
    ``shape[row_dim]``."""
    n = getattr(t, _ROWS_ATTR, None)
    return int(n) if n is not None else int(t.shape[row_dim % t.dim()])


class HostTensor(object):
    """One parked feature: ``host`` (the CPU tensor, pinned when possible — :func:`to_host`), ``row_dim`` (the dim its per-cycle selection
    indexes), ``name``. ``select(index)`` gathers rows ON THE HOST (``index_select`` with ``index`` moved to the host tensor's device);
    ``rows_to(index, device, dtype)`` moves only those rows to ``device`` (``non_blocking`` from pinned memory; cast AFTER the move, as an
    engine's ``.to(device=, dtype=)`` does); ``to(device)`` moves the whole tensor; ``placeholder()`` is the zero-row stand-in."""

    def __init__(self, t, name: str = "tensor", row_dim: int = 0, pin: bool = True, log: bool = True):
        if not torch.is_tensor(t):
            raise RowpairRefused(f"HostTensor({name}): a tensor is required (got {type(t).__name__})")
        self.name = str(name)
        self.row_dim = int(row_dim) % t.dim()
        self.host = t if is_parked(t) else to_host(t, name=name, pin=pin, log=log)
        self.pinned = _is_pinned(self.host)

    @property
    def shape(self):
        return tuple(self.host.shape)

    @property
    def dtype(self):
        return self.host.dtype

    @property
    def nbytes(self) -> int:
        return int(self.host.numel()) * int(self.host.element_size())

    def n_rows(self) -> int:
        return int(self.host.shape[self.row_dim])

    def select(self, index):
        """``host.index_select(row_dim, index)`` with ``index`` (int64, any device) moved to the host — the selected rows, still on the host."""
        idx = index.to(device=self.host.device, dtype=torch.int64) if torch.is_tensor(index) else torch.as_tensor(index, dtype=torch.int64)
        return self.host.index_select(self.row_dim, idx)

    def rows_to(self, index, device, dtype=None, non_blocking: bool = False):
        """The selected rows on ``device`` (contiguous; cast to ``dtype`` after the move when given)."""
        rows = self.select(index)
        out = rows.to(device=device, non_blocking=bool(non_blocking and self.pinned))
        if dtype is not None and out.dtype != dtype:
            out = out.to(dtype=dtype)
        return out.contiguous()

    def to(self, device, non_blocking: bool = False):
        return self.host.to(device=device, non_blocking=bool(non_blocking and self.pinned))

    def placeholder(self):
        return host_placeholder(self.host, self.row_dim)

    def facts(self) -> dict:
        return {"name": self.name, "shape": self.shape, "dtype": str(self.dtype).replace("torch.", ""), "gib": round(self.nbytes / 2 ** 30, 3),
                "where": where(self.host), "row_dim": self.row_dim}

    def __repr__(self) -> str:
        return f"HostTensor({self.name!r}, shape={self.shape}, dtype={self.facts()['dtype']}, {'pinned' if self.pinned else 'pageable'}, row_dim={self.row_dim})"


def park_features(feats: Dict[str, object], keys: Iterable[str], *, mode: Optional[str] = None,
                  row_dims: Union[int, Mapping[str, int]] = 0, pin: bool = True, log: bool = True) -> dict:
    """Apply the placement to a feature dict IN PLACE (idempotent): for each key of ``keys`` present as a non-empty tensor — mode ``all`` (and
    mode ``rank0`` on rank 0): replace a device tensor / an unparked host tensor by its host copy (:func:`to_host`); mode ``rank0`` on ranks > 0:
    replace it by :func:`host_placeholder` (``row_dims``: the per-key selection dim, an int for all keys or a mapping). ``mode`` None reads
    :func:`host_mode` (None -> nothing is touched). Returns ``{"mode", "parked": [keys], "placeholders": [keys], "where": {key: word}, "rows": {key: n}}`` (``word`` = :func:`where`: ``host_pinned`` |
    ``host_pageable:pin_alloc_failed`` | ``host`` | ``placeholder``; ``n`` = :func:`parked_rows`, the parked row count — equal on every rank) and
    records ``msa_host_mode`` + ``msa_host_where=<key:word,...>``."""
    md = host_mode() if mode is None else host_mode(mode)
    facts = {"mode": md, "parked": [], "placeholders": [], "where": {}, "rows": {}}
    if md is None:
        return facts
    P, r = world() if is_dist() else (1, 0)
    for k in keys:
        v = feats.get(k)
        if not torch.is_tensor(v) or v.numel() == 0:
            continue
        rd = row_dims if isinstance(row_dims, int) else int(dict(row_dims).get(k, 0))
        if md == "rank0" and P > 1 and r != 0:
            feats[k] = host_placeholder(v, rd)
            facts["placeholders"].append(k)
            facts["where"][k] = "placeholder"
            facts["rows"][k] = parked_rows(feats[k], rd)
            continue
        if v.is_cuda or not is_parked(v):
            feats[k] = to_host(v, name=k, pin=pin, log=log)
        facts["parked"].append(k)
        facts["where"][k] = where(feats[k])
        facts["rows"][k] = parked_rows(feats[k], rd)
    record_schedule(msa_host_mode=md, msa_host_where=",".join(f"{k}:{w}" for k, w in facts["where"].items()) or "none")
    return facts


# ----------------------------------------------------------------------------------------------------------------- device broadcasts
def bcast_chunked_(t, src: int = 0, chunk_gb: Optional[float] = None):
    """In-place broadcast of a contiguous DEVICE tensor from rank ``src`` in pieces of at most ``chunk_gb`` GiB (``ROWPAIR_BCAST_CHUNK_GB``,
    default 2) along its flattened view — one ``comm().bcast_`` per piece; bool tensors travel as their uint8 view. No-op at P == 1."""
    if not is_dist():
        return t
    if not t.is_contiguous():
        raise RowpairRefused(f"bcast_chunked_: a contiguous tensor is required (shape {tuple(t.shape)}, strides {t.stride()})")
    comm = _comm()
    flat = t.view(-1)
    if flat.dtype == torch.bool:
        flat = flat.view(torch.uint8)
    n = int(flat.numel())
    step = max(1, int(_chunk_gb(chunk_gb) * 2 ** 30) // max(1, int(flat.element_size())))
    for s in range(0, n, step):
        comm.bcast_(flat[s: min(n, s + step)], src)
    return t


def rows_to_device(build_fn: Callable[[], object], *, shape: Sequence[int], dtype, device, mode: Optional[str] = None,
                   chunk_gb: Optional[float] = None, name: str = "msa_rows"):
    """The selected feature rows ON THE DEVICE on every rank. ``build_fn()`` is the engine statement that gathers the selected rows from the
    host copies and moves them (e.g. ``cat([msa.select(idx), has_deletion.select(idx)[..., None], ...], -1).to(device, dtype)``); it must
    return a contiguous ``device`` tensor of ``shape`` / ``dtype``. Mode None / ``all`` (or P == 1): every rank runs ``build_fn()``. Mode
    ``rank0`` at P > 1: rank 0 runs it, ranks > 0 allocate ``torch.empty(shape, dtype, device)`` and the rows arrive by :func:`bcast_chunked_`
    (``chunk_gb``). Records ``msa_host_rows`` (mode, rows shape, GiB, seconds)."""
    md = host_mode() if mode is None else host_mode(mode)
    t0 = time.time()
    shape = tuple(int(s) for s in shape)
    from_rank0 = (md == "rank0") and is_dist()
    P, r = world() if is_dist() else (1, 0)
    if not from_rank0 or r == 0:
        out = build_fn()
        if not torch.is_tensor(out) or tuple(int(s) for s in out.shape) != shape or out.dtype != dtype:
            raise RowpairRefused(f"rows_to_device({name}): build_fn returned {tuple(out.shape) if torch.is_tensor(out) else type(out).__name__} "
                                 f"{getattr(out, 'dtype', None)}; {shape} {dtype} was declared (every rank must agree on the rows' shape)")
        if torch.device(device).type != out.device.type:
            raise RowpairRefused(f"rows_to_device({name}): build_fn returned a {out.device} tensor; device {device} was declared")
        out = out.contiguous()
    else:
        out = torch.empty(shape, dtype=dtype, device=device)
    if from_rank0:
        bcast_chunked_(out, 0, chunk_gb)
    record_schedule(msa_host_rows=f"{md or 'device'}:{'x'.join(str(s) for s in shape)}:{out.numel() * out.element_size() / 2 ** 30:.2f}GiB:{time.time() - t0:.2f}s")
    return out


_DTYPE_NAMES = ("float32", "float64", "float16", "bfloat16", "int64", "int32", "int16", "int8", "uint8", "bool")


def sync_host_features_(feats: Dict[str, object], keys: Iterable[str], device, src: int = 0, chunk_gb: Optional[float] = None,
                        mode: Optional[str] = None) -> Dict[str, object]:
    """Mode ``all`` batch sync (an engine featurises on rank ``src`` only): make every rank's HOST copies of ``keys`` equal to rank ``src``'s,
    streaming through ``device`` in ``<= chunk_gb`` GiB pieces (the shape / dtype meta travels first, so a rank that starts with a
    differently-shaped or missing tensor ends with ``src``'s). No-op at P == 1 or in a mode other than ``all``. In place; returns ``feats``."""
    md = host_mode() if mode is None else host_mode(mode)
    if not is_dist() or md != "all":
        return feats
    keys = list(keys)
    comm = _comm()
    P, r = world()
    meta = None
    if r == src:
        meta = {}
        for k in keys:
            v = feats.get(k)
            if not torch.is_tensor(v):
                raise RowpairRefused(f"sync_host_features_: rank {src} holds no tensor {k!r} to broadcast")
            meta[k] = (tuple(int(s) for s in v.shape), str(v.dtype).replace("torch.", ""))
    meta = broadcast_obj(meta, src=src)
    step_bytes = max(1, int(_chunk_gb(chunk_gb) * 2 ** 30))
    for k in keys:
        shp, dt_name = meta[k]
        if dt_name not in _DTYPE_NAMES:
            raise RowpairRefused(f"sync_host_features_: dtype {dt_name} of {k!r} is outside {_DTYPE_NAMES}")
        dt = getattr(torch, dt_name)
        have = feats.get(k)
        if r != src and not (torch.is_tensor(have) and tuple(int(s) for s in have.shape) == tuple(shp) and have.dtype == dt and not have.is_cuda):
            feats[k] = torch.empty(tuple(shp), dtype=dt)
            try:
                setattr(feats[k], _HOST_ATTR, True)
            except Exception:  # noqa: BLE001
                pass
        h = feats[k]
        if not h.is_contiguous():
            if r == src:
                h = h.contiguous()
                feats[k] = h
            else:
                raise RowpairRefused(f"sync_host_features_: {k!r} on rank {r} is not contiguous")
        flat = h.view(-1)
        as_u8 = flat.dtype == torch.bool
        if as_u8:
            flat = flat.view(torch.uint8)
        n = int(flat.numel())
        step = max(1, step_bytes // max(1, int(flat.element_size())))
        buf = torch.empty(min(n, step), dtype=flat.dtype, device=device)
        for s in range(0, n, step):
            e = min(n, s + step)
            if r == src:
                buf[: e - s].copy_(flat[s:e])
            comm.bcast_(buf[: e - s], src)
            if r != src:
                flat[s:e].copy_(buf[: e - s])
        del buf
    return feats


# ----------------------------------------------------------------------------------------------------------------- residency census
def residency_rows(feats: Mapping[str, object]) -> List[Tuple[int, str, tuple, str, str]]:
    """``[(bytes, key, shape, dtype, where)]`` for every tensor of a (nested) feature dict, largest first; ``where`` is ``cuda`` | ``pinned`` |
    ``host``."""
    rows: List[Tuple[int, str, tuple, str, str]] = []

    def visit(k, v):
        if torch.is_tensor(v):
            rows.append((int(v.numel()) * int(v.element_size()), str(k), tuple(int(s) for s in v.shape), str(v.dtype).replace("torch.", ""),
                         "cuda" if v.is_cuda else ("pinned" if _is_pinned(v) else "host")))
        elif isinstance(v, Mapping):
            for kk, vv in v.items():
                visit(f"{k}.{kk}", vv)
        elif isinstance(v, (list, tuple)) and v and all(torch.is_tensor(x) for x in v):
            for i, vv in enumerate(v):
                visit(f"{k}[{i}]", vv)

    for k, v in feats.items():
        visit(k, v)
    rows.sort(reverse=True)
    return rows


def residency_text(feats: Mapping[str, object], tag: str, top: int = 24) -> str:
    """The residency census as text: totals on the device / host (+ ``torch.cuda`` allocated / peak when CUDA is present), then the ``top``
    largest tensors (one per line)."""
    rows = residency_rows(feats)
    dev_total = sum(b for b, _, _, _, w in rows if w == "cuda")
    host_total = sum(b for b, _, _, _, w in rows if w != "cuda")
    head = f"[residency] {tag}: feature tensors on device {dev_total / 2 ** 30:.2f} GiB, on host {host_total / 2 ** 30:.2f} GiB"
    if torch.cuda.is_available():
        head += f"; torch.cuda.memory_allocated {torch.cuda.memory_allocated() / 2 ** 30:.2f} GiB, peak {torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB"
    lines = [head]
    for b, k, shp, dt, where in rows[:top]:
        if b < 2 ** 20 and len(lines) > 8:
            break
        lines.append(f"[residency]   {b / 2 ** 30:9.3f} GiB  {where:6s}  {k:36s} {dt:8s} {shp}")
    return "\n".join(lines)
