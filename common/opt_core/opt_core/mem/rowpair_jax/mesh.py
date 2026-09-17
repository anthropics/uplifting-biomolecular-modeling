"""The device mesh of the row-sharded pair stack and its gates.

:func:`build` makes ONE 1-D mesh (axis ``row`` by default) over the FIRST ``n_gpu`` devices jax sees — ``n_gpu`` is explicit (an integer ≥ 1 the
user typed — :func:`opt_core.mem.ngpu.check_n_gpu`; the mechanism never sizes itself), and fewer visible devices than requested is
``refused: n_gpu=P visible=K`` (:class:`opt_core.mem.ngpu.NGpuRefused` — :mod:`opt_core.mem.ngpu` is the ONE producer of the ``--n_gpu`` token text
and refusal sentences; this module adds the jax mesh and device facts around them). The returned :class:`RowMesh` carries the jax ``Mesh``, the axis name, P, and
the facts of the devices as READ FROM THE RUNTIME (:func:`device_facts`: platform, device kind, ``bytes_limit`` / ``bytes_in_use`` from
``memory_stats()`` where the backend reports them) — no card name or memory size is assumed anywhere. :func:`refuse_mode` delegates the interface rule
(``--n_gpu>1`` only under ``big``; refused by name under ``exact`` / ``fast`` — row-sharded contractions reorder the pair GEMM accumulation, so the
run cannot be bit-exact to the single-device program) to :func:`opt_core.mem.ngpu.refuse_unless_big`. :func:`xla_memory_env` reads the XLA client-memory variables as found (recorded on the evidence line — a peak number
without them is meaningless) and :func:`mem_fraction_gate` refuses by name when the adapter's per-P ceiling on the client memory fraction is exceeded
— judged on the LIVE pool (``bytes_limit / device total``), not the environment string (NCCL communicators allocate OUTSIDE XLA's pool: a pool
sized to 0.95 of the card leaves them nothing at large P — the ceiling per P is the kit's measured value, passed in). ``n_gpu=1`` builds no mesh
(:data:`REFUSE_P1`): P=1 is the kit's single-device program, structurally. Standard library at import.
"""
from __future__ import annotations

import os
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .. import MemLeverRefused
from .. import ngpu as _ngpu
from . import AXIS, LEVER, _lazy

MEM_FRACTION_VARS = ("XLA_PYTHON_CLIENT_MEM_FRACTION", "XLA_CLIENT_MEM_FRACTION")   # both names jax reads (recorded as found; the live pool says which applied)
PREALLOCATE_VAR = "XLA_PYTHON_CLIENT_PREALLOCATE"
ALLOCATOR_VAR = "XLA_PYTHON_CLIENT_ALLOCATOR"


class RowMesh:
    """The mesh record: ``mesh`` (``jax.sharding.Mesh``), ``axis`` (its one axis name), ``n_gpu`` (P), ``devices`` (the P jax devices, in mesh
    order), ``visible`` (how many jax saw), ``facts`` (:func:`device_facts` of the P devices), ``lever`` (the name refusals carry)."""

    def __init__(self, mesh: Any, axis: str, n_gpu: int, devices: Sequence[Any], visible: int, facts: List[Dict[str, Any]], lever: str):
        self.mesh = mesh
        self.axis = str(axis)
        self.n_gpu = int(n_gpu)
        self.devices = list(devices)
        self.visible = int(visible)
        self.facts = list(facts)
        self.lever = str(lever)

    @property
    def size(self) -> int:
        return self.n_gpu

    def kinds(self) -> List[str]:
        """The distinct device kinds of the mesh, in first-seen order (one entry on a homogeneous host)."""
        out: List[str] = []
        for f in self.facts:
            k = str(f.get("kind", "unknown"))
            if k not in out:
                out.append(k)
        return out

    def describe(self) -> Dict[str, Any]:
        """Blank-free evidence values: ``n_gpu``, ``axis``, ``visible``, ``platform``, ``devices`` (``d<id>,…``), ``device_kind`` (``,``-joined, blanks → ``_``),
        ``bytes_limit`` (``d<id>:<bytes>,…`` or ``unavailable``)."""
        def tok(s: Any) -> str:
            return str(s).replace(" ", "_")
        limits = [f"d{f['id']}:{f['bytes_limit']}" for f in self.facts if f.get("bytes_limit") is not None]
        totals = [f"d{f['id']}:{f['total_bytes']}" for f in self.facts if f.get("total_bytes") is not None]
        sms = []
        for f in self.facts:
            v = str(f.get("sm") or "unknown")
            if v not in sms:
                sms.append(v)
        frac = self.pool_fraction()
        return {"n_gpu": self.n_gpu, "axis": self.axis, "visible": self.visible,
                "platform": tok(self.facts[0].get("platform", "unknown")) if self.facts else "unknown",
                "devices": ",".join(f"d{f['id']}" for f in self.facts) or "none",
                "device_kind": ",".join(tok(k) for k in self.kinds()) or "unknown",
                "sm": ",".join(sms) or "unknown",
                "xla_pool_limit": ",".join(limits) if limits else "unavailable",
                "device_total_bytes": ",".join(totals) if totals else "unavailable",
                "xla_pool_fraction": "unavailable" if frac is None else f"{frac:.3f}"}

    def pool_fraction(self) -> Optional[float]:
        """The LIVE client memory fraction: the largest ``bytes_limit / total_bytes`` over the mesh devices (the XLA pool ceiling the running client
        actually has, whatever the environment says), or None when either number is unavailable on this backend."""
        vals = [f["bytes_limit"] / float(f["total_bytes"]) for f in self.facts if f.get("bytes_limit") and f.get("total_bytes")]
        return max(vals) if vals else None


def explicit_n_gpu(n_gpu: Any) -> int:
    """``n_gpu`` as a positive integer — :func:`opt_core.mem.ngpu.check_n_gpu` (the one producer of the rule): ``None`` / ``auto`` / non-integers /
    < 1 are a usage error (ValueError); the mechanism never sizes itself to the visible devices."""
    return _ngpu.check_n_gpu(n_gpu)


def device_facts(devices: Sequence[Any]) -> List[Dict[str, Any]]:
    """Per device: ``{id, platform, kind, process_index, bytes_limit, bytes_in_use, peak_bytes_in_use, total_bytes, sm, total_source}`` read from the
    runtime — ``memory_stats()`` of the jax device (the XLA allocator's pool: ``bytes_limit`` = the pool ceiling the client memory fraction produced;
    keys absent on a backend read ``None``) and, for ``gpu`` devices, the card's total memory (:func:`opt_core.arch.device_memory`) and sm class
    (:func:`opt_core.jax_arch.jax_sm`) — the core's card readers; ``total_source`` names the source or why it is absent. Nothing is assumed about the card."""
    out: List[Dict[str, Any]] = []
    for pos, d in enumerate(devices):
        stats: Optional[Mapping[str, Any]] = None
        try:
            ms = d.memory_stats()
            stats = ms if isinstance(ms, Mapping) else None
        except Exception:  # noqa: BLE001 — a backend without memory stats
            stats = None
        def g(k: str):
            return None if stats is None else (int(stats[k]) if k in stats and stats[k] is not None else None)
        rec = {"id": int(getattr(d, "id", pos)), "platform": str(getattr(d, "platform", "unknown")),
               "kind": str(getattr(d, "device_kind", "unknown")), "process_index": int(getattr(d, "process_index", 0) or 0),
               "bytes_limit": g("bytes_limit"), "bytes_in_use": g("bytes_in_use"), "peak_bytes_in_use": g("peak_bytes_in_use"),
               "total_bytes": None, "sm": None, "total_source": "not_gpu"}
        if rec["platform"] == "gpu":
            from ... import arch, jax_arch  # noqa: PLC0415 — the core's card readers (sm per jax device; device total via the torch-free probe)
            local = getattr(d, "local_hardware_id", None)
            mem = arch.device_memory(int(local) if isinstance(local, int) else int(rec["id"]))
            sm, _info = jax_arch.jax_sm(d)
            rec.update(total_bytes=mem.get("total_bytes"), sm=sm, total_source=str(mem.get("source") or "unknown").replace(" ", "_"))
        out.append(rec)
    return out


REFUSE_P1 = "refused: n_gpu=1 installs nothing — the single-device program is the kit's P=1 path (no mesh)"
REFUSE_PLATFORM = "refused: n_gpu={P} platform={found} expected={platform}"


def build(n_gpu: Any, *, platform: str = "gpu", axis: str = AXIS, lever: str = LEVER, devices: Optional[Sequence[Any]] = None,
          require_local: bool = True, _allow_single_device_mesh: bool = False) -> RowMesh:
    """The 1-D mesh over the first ``n_gpu`` devices OF ``platform`` (default ``gpu``; the tests pass ``cpu`` for forced host devices — a jax that fell
    back to CPU because CUDA failed to initialise is thereby refused by name, never a silent CPU mesh). ``devices``: the candidate list (default
    ``jax.devices()`` — what this process sees, already bounded by the kit's ``CUDA_VISIBLE_DEVICES``); ``require_local``: every chosen device must
    belong to this process (single-process multi-device is the mechanism's process model). Refusals: a non-integer / < 1 P is a usage error
    (ValueError); ``n_gpu=1`` is :data:`REFUSE_P1` (P=1 installs nothing: the kit's single-device program runs as tested; the unit tests build a
    1-device mesh through ``_allow_single_device_mesh=True`` only); ``refused: n_gpu=P visible=K`` (:class:`opt_core.mem.ngpu.NGpuRefused`, lever ``n_gpu`` —
    the one producer of that sentence; K counts devices of ``platform``); ``refused: n_gpu=P platform=<found> expected=<platform>``; a non-local device."""
    n = explicit_n_gpu(n_gpu)
    if n == 1 and not _allow_single_device_mesh:
        raise _ngpu.NGpuRefused(REFUSE_P1)
    jax = _lazy.jax(lever)
    np = _lazy.np(lever)
    Mesh, _NamedSharding, _P = _lazy.sharding(lever)
    if devices is None:
        try:
            devs = list(jax.devices())
        except Exception as e:  # noqa: BLE001
            raise MemLeverRefused(lever, f"refused: n_gpu={n} — jax.devices() failed ({e!r})") from None
    else:
        devs = list(devices)
    found = sorted({str(getattr(d, "platform", "unknown")) for d in devs})
    of_platform = [d for d in devs if str(getattr(d, "platform", "unknown")) == str(platform)]
    if devs and not of_platform:
        raise _ngpu.NGpuRefused(REFUSE_PLATFORM.format(P=n, found=",".join(found), platform=platform))
    visible = len(of_platform)
    if visible < n:                                                    # P > 1: ngpu's sentence; P == 1 (test hook) with nothing visible: the same words
        _ngpu.refuse_unless_visible(n, visible)
        raise _ngpu.NGpuRefused(_ngpu.visible_refusal(n, visible))
    chosen = of_platform[:n]
    if require_local:
        pi = jax.process_index() if hasattr(jax, "process_index") else 0
        foreign = [str(d) for d in chosen if int(getattr(d, "process_index", pi) or 0) != int(pi)]
        if foreign:
            raise MemLeverRefused(lever, f"refused: n_gpu={n} — devices {','.join(foreign)} belong to another process (single-process multi-device only)")
    mesh = Mesh(np.array(chosen), (str(axis),))
    return RowMesh(mesh, str(axis), n, chosen, visible, device_facts(chosen), lever)


def refuse_mode(mode: str, n_gpu: Any) -> int:
    """The interface rule, delegated to :func:`opt_core.mem.ngpu.refuse_unless_big` (the one producer of the sentence): P when ``n_gpu == 1`` or
    ``mode == 'big'``; otherwise :class:`opt_core.mem.ngpu.NGpuRefused` (``refused: n_gpu>1 requires --mode big (sharded reductions are not bitwise)``)."""
    return _ngpu.refuse_unless_big(n_gpu, mode)


def xla_memory_env(environ: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """The XLA client-memory settings AS FOUND, as blank-free lever-line tokens — read through :func:`opt_core.mem.peak.xla_env` (the core's verbatim
    XLA/JAX env record): ``{"prealloc": <XLA_PYTHON_CLIENT_PREALLOCATE|unset>, "mem_fraction": <XLA_PYTHON_CLIENT_MEM_FRACTION|unset>, "client_mem_fraction":
    <XLA_CLIENT_MEM_FRACTION|unset>, "allocator": <XLA_PYTHON_CLIENT_ALLOCATOR|unset>, "xla_effective_fraction": <what jax resolves per peak's precedence
    table>}``. The mechanism never sets these — the kit's launcher owns the policy (and touches it only under n_gpu>1); the line records it."""
    from ..peak import xla_effective, xla_env  # noqa: PLC0415 — one env reader, one precedence table (mem.peak)
    env = xla_env(os.environ if environ is None else environ)
    def rd(k: str) -> str:
        v = env.get(k)
        return "unset" if v is None or str(v).strip() == "" else str(v).strip().replace(" ", "_")
    eff = xla_effective(env)
    frac = eff.get("mem_fraction")
    return {"prealloc": rd(PREALLOCATE_VAR), "mem_fraction": rd(MEM_FRACTION_VARS[0]), "client_mem_fraction": rd(MEM_FRACTION_VARS[1]), "allocator": rd(ALLOCATOR_VAR),
            "xla_effective_fraction": "unknown" if frac is None else f"{float(frac):g}"}


def mem_fraction_gate(rmesh: RowMesh, ceilings: Mapping[int, float], environ: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The NCCL-headroom precondition on the LIVE pool: ``ceilings`` maps a device count to the largest client memory fraction the kit tested at that
    count (e.g. ``{8: 0.85}`` — NCCL communicators allocate OUTSIDE XLA's pool). For the largest key <= ``rmesh.n_gpu`` the mesh's
    :meth:`RowMesh.pool_fraction` (``bytes_limit / device total``, read from the runtime — not the environment string, which may have been set after the
    backend initialised or under the other variable name) must not exceed the ceiling, else :class:`MemLeverRefused` by name; an unreadable live
    fraction under an applicable ceiling is refused too (fail-closed). Returns ``{"mem_fraction_ceiling": <c|'none'>, "xla_pool_fraction": <f|'unavailable'>}``
    for the evidence line; the environment AS FOUND is recorded separately (:func:`xla_memory_env`) — which variable wins when both are set is jax's
    rule, read off the live pool here rather than restated."""
    n = int(rmesh.n_gpu)
    keys = sorted(int(k) for k in ceilings if int(k) <= n)
    frac = rmesh.pool_fraction()
    rec: Dict[str, Any] = {"mem_fraction_ceiling": "none", "xla_pool_fraction": "unavailable" if frac is None else float(f"{frac:.4f}")}
    if not keys:
        return rec
    c = float(ceilings[keys[-1]])
    rec["mem_fraction_ceiling"] = c
    env = xla_memory_env(environ)
    if frac is None:
        raise MemLeverRefused(rmesh.lever, f"refused: n_gpu={n} needs the XLA pool fraction <= {c} and the live pool limit is unreadable on this backend "
                                           f"(env {MEM_FRACTION_VARS[0]}={env['mem_fraction']} {MEM_FRACTION_VARS[1]}={env['client_mem_fraction']})")
    if frac > c + 1e-6:
        raise MemLeverRefused(rmesh.lever, f"refused: n_gpu={n} needs the XLA pool fraction <= {c} (live {frac:.3f}; env {MEM_FRACTION_VARS[0]}={env['mem_fraction']} "
                                           f"{MEM_FRACTION_VARS[1]}={env['client_mem_fraction']}); NCCL communicators allocate outside XLA's pool")
    return rec
