"""ptx_tp.bcast -- broadcast helpers (no protenix imports).

  broadcast_obj(obj, src=0)                      any picklable object (torch.distributed.broadcast_object_list)
  broadcast_tensordict(d, src=0, device=None)    nested dict/list/tuple of tensors + plain metadata: structure and non-tensor leaves
                                                 travel pickled; tensors travel by dist.broadcast on `device` (cuda for NCCL, cpu for
                                                 gloo). On src the ORIGINAL object is returned untouched (so the sender keeps stock
                                                 CPU tensors and its code path is unchanged); receivers get tensors on `device`
                                                 (or moved back to the sender's original device type when keep_device=False).
  tensordict_checksum(d) -> {path: (numel,sum,xor)}   bitwise fingerprints of every tensor leaf (debug equality asserts across ranks)
Works without a process group (identity)."""
from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.distributed as dist

from ptx_tp.dist import broadcast_obj, checksum, is_dist  # noqa: F401

_TKEY = "__ptx_tp_tensor__"


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


def comm_device() -> torch.device:
    """Device tensors must live on for the default group's collectives (cuda for nccl, cpu for gloo)."""
    if is_dist() and dist.get_backend() == "nccl":
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def broadcast_tensordict(d: Any = None, src: int = 0, device: Optional[torch.device] = None, keep_device: bool = True,
                         coalesce_bytes: int = 64 << 20) -> Any:
    """Broadcast a nested container of tensors from `src` to all ranks. Small tensors are coalesced into byte buffers of
    <= coalesce_bytes to avoid thousands of tiny collectives; large tensors go one by one. Returns the object on every rank."""
    if not is_dist():
        return d
    rank = dist.get_rank()
    device = device or comm_device()
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
            dist.broadcast(buf, src=src)
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
                tb = t.view(torch.uint8)
                dist.broadcast(tb, src=src)
            else:
                dist.broadcast(t, src=src)
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


def _dtype(name: str) -> torch.dtype:
    return getattr(torch, name)


def _nbytes(p) -> int:
    n = 1
    for s in p["shape"]:
        n *= int(s)
    return n * torch.empty(0, dtype=_dtype(p["dtype"])).element_size()


def _place(t: torch.Tensor, p, device, keep_device: bool) -> torch.Tensor:
    if keep_device:
        return t
    if p["dev"] == "cpu" and t.device.type != "cpu":
        return t.cpu()
    return t


def tensordict_checksum(d: Any, prefix: str = "") -> Dict[str, tuple]:
    """{path: (numel, sum, xor)} for every tensor leaf -- compare across ranks with dist.all_gather_object for a bitwise assert."""
    out = {}
    if isinstance(d, torch.Tensor):
        out[prefix or "/"] = checksum(d)
    elif isinstance(d, dict):
        for k, v in d.items():
            out.update(tensordict_checksum(v, f"{prefix}/{k}"))
    elif isinstance(d, (list, tuple)):
        for i, v in enumerate(d):
            out.update(tensordict_checksum(v, f"{prefix}/{i}"))
    return out


def assert_replicated(d: Any, name: str = "tensordict") -> Dict[str, tuple]:
    """All ranks compute leaf checksums; raise AssertionError listing the differing paths if any rank disagrees."""
    cs = tensordict_checksum(d)
    if not is_dist():
        return cs
    allcs = [None] * dist.get_world_size()
    dist.all_gather_object(allcs, cs)
    bad = sorted(k for k in cs if any(a.get(k) != cs[k] for a in allcs))
    missing = sorted(set().union(*[set(a) for a in allcs]) - set.intersection(*[set(a) for a in allcs]))
    if bad or missing:
        raise AssertionError(f"[ptx_tp.bcast] {name}: NOT replicated; differing leaves={bad[:20]} missing-on-some-rank={missing[:20]}")
    return cs
