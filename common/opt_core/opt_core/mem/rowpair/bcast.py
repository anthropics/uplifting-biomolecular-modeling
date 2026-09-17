"""Rank-0 featurisation + broadcast: an engine featurises ONCE on rank 0 (MSA parsing, template search, tokenisation — host work that must
not run P times and whose random draws must not differ across ranks) and broadcasts the resulting tensor dict; replicated inputs are then
PROVEN replicated (:func:`assert_replicated`, integer checksums) before the trunk runs. When every rank runs the engine's own data pipeline
instead, :func:`sync_tensordict_from_rank0` overwrites every rank's batch IN PLACE with rank 0's (shapes may differ across ranks — e.g. MSA
depth — so a shape header travels first) and checks the result. All traffic goes through the comm surface (:func:`opt_core.mem.rowpair.dist.comm`).

API:
    broadcast_tensordict(d, src=0, device=None, coalesce_bytes=64 MiB)   a nested container of tensors from ``src`` to every rank (small tensors
                                                                         coalesced into byte buffers; large ones one by one); non-tensor leaves
                                                                         travel pickled with the structure
    sync_tensordict_from_rank0(batch, skip_keys=(), strict=True, tag=)   IN PLACE: every tensor leaf (sorted-key order) becomes rank 0's (header
                                                                         ``(dtype, ndim, shape)`` first; receivers re-allocate on mismatch; bool as
                                                                         uint8); non-tensor leaves are hash-compared (mismatch = refusal unless
                                                                         ``strict=False``); then every synced tensor's fp64 ``(sum, sum_sq)`` is
                                                                         proven identical across ranks (mismatch = refusal). Leaves named in
                                                                         ``skip_keys`` or living off the comm device are SKIPPED BY NAME (returned
                                                                         census ``skipped``) — the caller owns them. Returns the census dict.
    tensordict_checksum(d) / assert_replicated(d, name)                  per-leaf :func:`opt_core.mem.rowpair.dist.checksum`; a mismatch across
                                                                         ranks is :class:`RowpairRefused` naming the leaf
    comm_device()                                                        the device collectives' payloads live on
"""
from __future__ import annotations

import hashlib
from typing import Any, Dict, Optional, Sequence

from . import RowpairRefused
from ._torch import torch
from .dist import broadcast_obj, checksum, comm, is_dist

__all__ = ["broadcast_tensordict", "sync_tensordict_from_rank0", "tensordict_checksum", "assert_replicated", "comm_device", "SYNC_DTYPES"]

SYNC_DTYPES = ("float32", "float64", "float16", "bfloat16", "int64", "int32", "int16", "int8", "uint8", "bool")   # header dtype codes (index)
_MAXDIM = 8


def _leaves(d: Any, prefix: str = ""):
    if torch.is_tensor(d):
        yield prefix or "tensor", d
    elif isinstance(d, dict):
        for k in sorted(d, key=str):
            for item in _leaves(d[k], f"{prefix}.{k}" if prefix else str(k)):
                yield item
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            for item in _leaves(v, f"{prefix}[{i}]"):
                yield item


def tensordict_checksum(d: Any, prefix: str = "") -> Dict[str, tuple]:
    """``{leaf path: checksum}`` over every tensor leaf of a nested dict / list / tuple."""
    return {k: checksum(t) for k, t in _leaves(d, prefix)}


def assert_replicated(d: Any, name: str = "tensordict") -> Dict[str, tuple]:
    """Every tensor leaf of ``d`` is bit-identical on all ranks (all-gather of the checksums); a differing leaf raises
    :class:`RowpairRefused` naming it. Returns the local checksums."""
    local = tensordict_checksum(d)
    if not is_dist():
        return local
    allv = comm().allgather_obj(local)
    for k in local:
        vals = [v.get(k) for v in allv]
        if not all(v == vals[0] for v in vals):
            raise RowpairRefused(f"{name}: leaf {k!r} differs across ranks: {vals}")
    return local


_TKEY = "__rowpair_tensor__"


def _flatten(obj, tensors, path=""):
    """Replace tensor leaves by placeholders {_TKEY: idx, shape, dtype, device_type}; collect tensors in order."""
    if isinstance(obj, torch.Tensor):
        idx = len(tensors)
        tensors.append(obj)
        return {_TKEY: idx, "shape": tuple(obj.shape), "dtype": str(obj.dtype).replace("torch.", ""), "dev": obj.device.type, "path": path}
    if isinstance(obj, dict):
        return {k: _flatten(v, tensors, f"{path}/{k}") for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_flatten(v, tensors, f"{path}/{i}") for i, v in enumerate(obj)]
        return seq if isinstance(obj, list) else ("__tuple__", seq)
    return obj


def _unflatten(meta, tensors):
    if isinstance(meta, dict):
        if _TKEY in meta:
            return tensors[meta[_TKEY]]
        return {k: _unflatten(v, tensors) for k, v in meta.items()}
    if isinstance(meta, tuple) and len(meta) == 2 and meta[0] == "__tuple__":
        return tuple(_unflatten(v, tensors) for v in meta[1])
    if isinstance(meta, list):
        return [_unflatten(v, tensors) for v in meta]
    return meta


def _iter_placeholders(meta):
    if isinstance(meta, dict):
        if _TKEY in meta:
            yield meta
        else:
            for v in meta.values():
                yield from _iter_placeholders(v)
    elif isinstance(meta, tuple) and len(meta) == 2 and meta[0] == "__tuple__":
        for v in meta[1]:
            yield from _iter_placeholders(v)
    elif isinstance(meta, list):
        for v in meta:
            yield from _iter_placeholders(v)


def comm_device():
    """Device tensors must live on for the comm's collectives (cuda for nccl, cpu for gloo / threaded / solo)."""
    return comm().device


def broadcast_tensordict(d: Any = None, src: int = 0, device=None, keep_device: bool = True, coalesce_bytes: int = 64 << 20) -> Any:
    """Broadcast a nested container of tensors from `src` to all ranks. Small tensors are coalesced into byte buffers of
    <= coalesce_bytes to avoid thousands of tiny collectives; large tensors go one by one. Returns the object on every rank."""
    if not is_dist():
        return d
    cm = comm()
    rank = cm.rank
    device = device or cm.device
    tensors: list = []
    meta = _flatten(d, tensors) if rank == src else None
    meta = broadcast_obj(meta, src=src)
    phs = sorted(_iter_placeholders(meta), key=lambda p: p[_TKEY])
    out_tensors: list = [None] * len(phs)
    # partition: big tensors individually, small ones coalesced (as uint8 views) in order
    small, small_bytes = [], 0

    def flush_small():
        nonlocal small, small_bytes
        if not small:
            return
        if rank == src:
            buf = torch.cat([tensors[p[_TKEY]].detach().contiguous().reshape(-1).view(torch.uint8).to(device)
                             if tensors[p[_TKEY]].numel() else torch.empty(0, dtype=torch.uint8, device=device) for p in small])
        else:
            total = sum(_nbytes(p) for p in small)
            buf = torch.empty(total, dtype=torch.uint8, device=device)
        if buf.numel():
            cm.bcast_(buf, src=src)
        off = 0
        for p in small:
            nb = _nbytes(p)
            if rank == src:
                out_tensors[p[_TKEY]] = tensors[p[_TKEY]]
            else:
                t = buf[off:off + nb].clone().view(_dtype(p["dtype"])).reshape(p["shape"]) if nb else torch.empty(p["shape"], dtype=_dtype(p["dtype"]), device=device)
                out_tensors[p[_TKEY]] = _place(t, p, device, keep_device)
            off += nb
        small, small_bytes = [], 0

    for p in phs:
        nb = _nbytes(p)
        if nb >= coalesce_bytes:
            flush_small()
            if rank == src:
                t = tensors[p[_TKEY]].detach().contiguous().to(device)
            else:
                t = torch.empty(p["shape"], dtype=_dtype(p["dtype"]), device=device)
            if p["dtype"] == "bool":
                cm.bcast_(t.view(torch.uint8), src=src)
            else:
                cm.bcast_(t, src=src)
            out_tensors[p[_TKEY]] = tensors[p[_TKEY]] if rank == src else _place(t, p, device, keep_device)
        else:
            small.append(p)
            small_bytes += nb
            if small_bytes >= coalesce_bytes:
                flush_small()
    flush_small()
    if rank == src:
        return d
    return _unflatten(meta, out_tensors)


def _dtype(name: str):
    return getattr(torch, name)


def _nbytes(p) -> int:
    n = 1
    for s in p["shape"]:
        n *= int(s)
    return n * torch.empty(0, dtype=_dtype(p["dtype"])).element_size()


def _place(t, p, device, keep_device: bool):
    if keep_device:
        return t
    if p["dev"] == "cpu" and t.device.type != "cpu":
        return t.cpu()
    return t


# ----------------------------------------------------------------------------------------------------------------- in-place sync
def _slots(obj, prefix: str = ""):
    """``(path, parent, key)`` for every leaf slot of nested dict / list structures (dict keys sorted by str) — writable positions."""
    if isinstance(obj, dict):
        for k in sorted(obj.keys(), key=str):
            yield from _slot(obj, k, f"{prefix}/{k}")
    elif isinstance(obj, list):
        for i in range(len(obj)):
            yield from _slot(obj, i, f"{prefix}[{i}]")
    elif isinstance(obj, tuple):
        for i in range(len(obj)):
            v = obj[i]
            if isinstance(v, (dict, list, tuple)):
                yield from _slots(v, f"{prefix}[{i}]")
            else:
                yield f"{prefix}[{i}]", None, i                              # tuple leaves are read-only positions


def _slot(parent, key, path):
    v = parent[key]
    if isinstance(v, (dict, list, tuple)):
        yield from _slots(v, path)
    else:
        yield path, parent, key


def _chunked_moments(v, max_bytes: int = 256 * 2 ** 20):
    """``[sum, sum of squares]`` in float64 over a tensor of any dtype / size, slice by slice (<= max_bytes per slice incl. the fp64 copy)."""
    flat = v.reshape(-1) if v.is_contiguous() else v.contiguous().view(-1)
    n = flat.numel()
    step = max(1, max_bytes // 16)
    acc = torch.zeros(2, dtype=torch.float64, device=v.device)
    for i in range(0, n, step):
        x = flat[i:i + step].to(torch.float64)
        acc[0] += x.sum()
        acc[1] += (x * x).sum()
    return acc


def sync_tensordict_from_rank0(batch: Any, *, skip_keys: Sequence[str] = (), strict: bool = True, tag: str = "batch",
                               verify: bool = True) -> dict:
    """Make every rank's ``batch`` (nested dict / list of tensors and scalars) IDENTICAL to rank 0's, in place. Returns the census
    ``{"tag", "synced", "realloc", "checked", "skipped": [paths], "meta_mismatch": [paths]}``; a no-op census at world size 1."""
    cm = comm()
    census = {"tag": tag, "synced": 0, "realloc": 0, "checked": 0, "skipped": [], "meta_mismatch": []}
    if cm.world <= 1:
        return census
    dev = cm.device
    skip = set(str(k) for k in skip_keys)
    synced_paths = []
    for path, parent, key in list(_slots(batch)):
        v = parent[key] if parent is not None else None
        if parent is None:
            census["skipped"].append(path)
            continue
        if isinstance(v, torch.Tensor):
            if str(key) in skip or v.device.type != dev.type:
                census["skipped"].append(path)                               # named: the caller owns host-resident / excluded features
                continue
            hdr = torch.zeros(2 + _MAXDIM, dtype=torch.int64, device=dev)
            if cm.rank == 0:
                if v.dim() > _MAXDIM:
                    raise RowpairRefused(f"sync_tensordict_from_rank0[{tag}]: {path}: ndim {v.dim()} > {_MAXDIM}")
                name = str(v.dtype).replace("torch.", "")
                hdr[0] = SYNC_DTYPES.index(name) if name in SYNC_DTYPES else -1
                hdr[1] = v.dim()
                for i, s in enumerate(v.shape):
                    hdr[2 + i] = int(s)
            cm.bcast_(hdr, 0)
            code, ndim = int(hdr[0].item()), int(hdr[1].item())
            if code < 0:
                raise RowpairRefused(f"sync_tensordict_from_rank0[{tag}]: unsupported dtype at {path}: {v.dtype}")
            shape = tuple(int(hdr[2 + i].item()) for i in range(ndim))
            dtype = _dtype(SYNC_DTYPES[code])
            if cm.rank != 0 and (tuple(v.shape) != shape or v.dtype != dtype):
                v = torch.empty(shape, dtype=dtype, device=v.device)
                census["realloc"] += 1
            src = v.contiguous() if cm.rank == 0 else (v if v.is_contiguous() else torch.empty(shape, dtype=dtype, device=v.device))
            if dtype == torch.bool:                                          # NCCL has no bool: ship as uint8
                tmp = src.to(torch.uint8) if cm.rank == 0 else torch.empty(shape, dtype=torch.uint8, device=src.device)
                cm.bcast_(tmp, 0)
                if cm.rank != 0:
                    src = tmp.to(torch.bool)
            else:
                cm.bcast_(src, 0)
            if cm.rank != 0:
                parent[key] = src
            elif src.data_ptr() != parent[key].data_ptr():                   # rank 0 held a non-contiguous leaf: keep its object, values unchanged
                pass
            census["synced"] += 1
            synced_paths.append((path, parent, key))
        elif isinstance(v, (str, int, float, bool)) or v is None:
            h = int(hashlib.sha256(repr(v).encode()).hexdigest()[:15], 16)
            t = torch.tensor([h, -h], dtype=torch.int64, device=dev)
            cm.allreduce_(t, op="max")                                       # max(h) and max(-h) = -min(h): equal iff all ranks agree
            if int(t[0].item()) != -int(t[1].item()):
                census["meta_mismatch"].append(path)
    if census["meta_mismatch"]:
        msg = f"sync_tensordict_from_rank0[{tag}]: non-tensor entries differ across ranks: {census['meta_mismatch'][:8]}"
        if strict:
            raise RowpairRefused(msg)
        cm.log(msg + " (strict=False: continuing)")
    if verify:
        bad = []
        for path, parent, key in synced_paths:
            v = parent[key]
            if v.numel() == 0:
                continue
            s = _chunked_moments(v)                                          # fp64 [sum, sum_sq] over <= 256 MiB slices: no full-size temporaries
            t = torch.cat([s, -s])
            cm.allreduce_(t, op="max")
            if not torch.equal(t[:2], -t[2:]):
                bad.append(path)
            census["checked"] += 1
        if bad:
            raise RowpairRefused(f"sync_tensordict_from_rank0[{tag}]: tensors still differ across ranks after broadcast: {bad[:8]}")
    cm.log(f"[bcast] {tag}: {census['synced']} tensors broadcast from rank 0 ({census['realloc']} re-allocated here), {census['checked']} verified "
           f"identical across {cm.world} ranks; skipped by name: {census['skipped']}")
    return census
