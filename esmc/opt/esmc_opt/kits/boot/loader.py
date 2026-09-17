"""Device-side safetensors reader — the U1 (boot) lever's one mechanism.

A safetensors file is an 8-byte little-endian header length, a JSON header ({name: {dtype, shape, data_offsets}},
optional ``__metadata__``) and a flat data region. The stock loaders map the file (``torch.UntypedStorage.from_file``
behind ``safetensors.safe_open``) and materialise every tensor through page faults on the mapping plus one pageable
host-to-device copy per tensor. This reader instead

    parses the header (stdlib) -> plans every tensor's byte range -> reads the data region in large chunks with
    ``os.preadv`` from reader threads into PINNED host slots -> issues asynchronous H2D copies of each tensor's bytes
    into its own freshly allocated device tensor (contiguous, the tensor's dtype, the requested device).

With ``out_dtype`` (the dtype the caller will cast the model to — the SDK's ``model.to(dtype)`` after the load), every
floating tensor is delivered ALREADY in that dtype: its file bytes land in a per-tensor staging tensor of the file's dtype,
and the moment its last byte range arrives it is cast into its destination (``Tensor.copy_``, the same round-to-nearest-even
conversion ``.to(dtype)`` performs) and the staging tensor released — so the device never holds the whole checkpoint in the
file's dtype beside the cast copy (a 6B fp32 checkpoint: ~13 GB peak instead of ~38 GB), and the stock's later ``.to(dtype)``
finds nothing to do. Non-floating tensors and tensors already in ``out_dtype`` take the direct path.

The parameter BYTES are the file's bytes (or their cast, value for value): a pure copy, exact by construction. Anything the reader does not recognise
(dtype, shape/offset disagreement, overlapping or out-of-range ranges, a short read, a duplicate name across shards) is
refused with ``LayoutRefused`` — never a best-effort tensor.
"""
from __future__ import annotations

import collections
import json
import os
import queue
import struct
import threading
import time

#: safetensors dtype tag -> (torch dtype attribute, itemsize); anything else is refused
DTYPES = {"F64": ("float64", 8), "F32": ("float32", 4), "F16": ("float16", 2), "BF16": ("bfloat16", 2),
          "I64": ("int64", 8), "I32": ("int32", 4), "I16": ("int16", 2), "I8": ("int8", 1), "U8": ("uint8", 1),
          "BOOL": ("bool", 1)}
HEADER_MAX_BYTES = 100 * 1024 * 1024          # safetensors' own header ceiling
INDEX_NAME = "model.safetensors.index.json"
SINGLE_NAME = "model.safetensors"
CHUNK_BYTES = 32 << 20
N_SLOTS = 32                                 # the 6B readers ladder winner 16,32,32
N_READERS = 16

TensorSpec = collections.namedtuple("TensorSpec", "name dtype itemsize shape nbytes file start end")


class LayoutRefused(RuntimeError):
    """The file is not a safetensors layout this reader vouches for: refuse, never guess."""


def _prod(shape) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n


def read_header(path: str) -> dict:
    """Parse and validate one file's header. Returns {entries: [TensorSpec…] (ascending start), metadata, data_start,
    data_len, file_size, header_len, path}."""
    size = os.path.getsize(path)
    if size < 8:
        raise LayoutRefused(f"{path}: {size} bytes — no safetensors header")
    with open(path, "rb") as fh:
        (hlen,) = struct.unpack("<Q", fh.read(8))
        if hlen > HEADER_MAX_BYTES or 8 + hlen > size:
            raise LayoutRefused(f"{path}: header length {hlen} exceeds the file ({size} bytes) or the format ceiling")
        raw = fh.read(hlen)
    if len(raw) != hlen:
        raise LayoutRefused(f"{path}: short header read ({len(raw)} of {hlen} bytes)")
    try:
        header = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError) as e:
        raise LayoutRefused(f"{path}: header is not UTF-8 JSON ({e})") from e
    if not isinstance(header, dict):
        raise LayoutRefused(f"{path}: header is not a JSON object")
    data_start = 8 + hlen
    data_len = size - data_start
    metadata = header.get("__metadata__")
    if metadata is not None and not isinstance(metadata, dict):
        raise LayoutRefused(f"{path}: __metadata__ is not an object")
    entries = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(info, dict) or set(info) != {"dtype", "shape", "data_offsets"}:
            raise LayoutRefused(f"{path}: entry {name!r} is not {{dtype, shape, data_offsets}}: {info!r}"[:300])
        if info["dtype"] not in DTYPES:
            raise LayoutRefused(f"{path}: entry {name!r} dtype {info['dtype']!r} is not one this reader vouches for {sorted(DTYPES)}")
        tdtype, itemsize = DTYPES[info["dtype"]]
        shape = info["shape"]
        if not isinstance(shape, list) or any((not isinstance(s, int)) or isinstance(s, bool) or s < 0 for s in shape):
            raise LayoutRefused(f"{path}: entry {name!r} shape {shape!r} is not a list of non-negative ints")
        offs = info["data_offsets"]
        if not (isinstance(offs, list) and len(offs) == 2 and all(isinstance(o, int) and not isinstance(o, bool) for o in offs)):
            raise LayoutRefused(f"{path}: entry {name!r} data_offsets {offs!r} is not [start, end]")
        start, end = offs
        if not (0 <= start <= end <= data_len):
            raise LayoutRefused(f"{path}: entry {name!r} data_offsets {offs} outside the data region [0, {data_len})")
        nbytes = _prod(shape) * itemsize
        if end - start != nbytes:
            raise LayoutRefused(f"{path}: entry {name!r} spans {end - start} bytes but shape {shape} x {info['dtype']} needs {nbytes}")
        entries.append(TensorSpec(name, tdtype, itemsize, tuple(shape), nbytes, path, data_start + start, data_start + end))
    entries.sort(key=lambda e: (e.start, e.end, e.name))
    prev_end, prev_name = data_start, None
    for e in entries:
        if e.nbytes and e.start < prev_end:
            raise LayoutRefused(f"{path}: entries {prev_name!r} and {e.name!r} overlap")
        if e.nbytes:
            prev_end, prev_name = e.end, e.name
    return {"entries": entries, "metadata": metadata, "data_start": data_start, "data_len": data_len,
            "file_size": size, "header_len": hlen, "path": path}


def checkpoint_files(directory: str) -> list:
    """The stock's resolution (esm.models.hub.read_safetensors_dir): the index's shards (sorted, unique) when
    ``model.safetensors.index.json`` exists, else ``model.safetensors``; [] when neither exists (the caller hands the
    directory to the stock function so the stock's own FileNotFoundError is raised)."""
    index = os.path.join(directory, INDEX_NAME)
    if os.path.exists(index):
        with open(index) as fh:
            weight_map = json.load(fh)["weight_map"]
        shards = [os.path.join(directory, s) for s in sorted(set(weight_map.values()))]
        missing = [s for s in shards if not os.path.exists(s)]
        if missing:
            raise LayoutRefused(f"{index}: shards named by the index are missing: {missing[:5]}")
        return shards
    single = os.path.join(directory, SINGLE_NAME)
    return [single] if os.path.exists(single) else []


def plan(paths) -> dict:
    """Headers of every file + the global tensor list (names unique across files)."""
    headers = [read_header(p) for p in paths]
    seen, specs = {}, []
    for h in headers:
        for e in h["entries"]:
            if e.name in seen:
                raise LayoutRefused(f"tensor {e.name!r} appears in both {seen[e.name]} and {e.file}")
            seen[e.name] = e.file
            specs.append(e)
    return {"headers": headers, "specs": specs, "n_bytes": sum(e.nbytes for e in specs)}


def _chunks(headers, chunk_bytes: int) -> list:
    """[(file_index, abs_offset, n_bytes)] covering each file's tensor bytes (from the first entry's start to the last
    entry's end) in ``chunk_bytes`` pieces."""
    out = []
    for fi, h in enumerate(headers):
        ents = [e for e in h["entries"] if e.nbytes]
        if not ents:
            continue
        lo, hi = ents[0].start, max(e.end for e in ents)
        off = lo
        while off < hi:
            n = min(chunk_bytes, hi - off)
            out.append((fi, off, n))
            off += n
    return out


def _overlaps(headers, chunks) -> list:
    """Per chunk: [(spec, dst_lo, dst_hi, slot_lo, slot_hi)] — the tensor byte sub-ranges the chunk carries."""
    per_file = [[e for e in h["entries"] if e.nbytes] for h in headers]
    cursor = [0] * len(headers)
    out = []
    for fi, off, n in chunks:
        ents = per_file[fi]
        i = cursor[fi]
        while i < len(ents) and ents[i].end <= off:                # entries entirely before this chunk
            i += 1
        cursor[fi] = i
        items = []
        j = i
        while j < len(ents) and ents[j].start < off + n:
            e = ents[j]
            lo, hi = max(e.start, off), min(e.end, off + n)
            items.append((e, lo - e.start, hi - e.start, lo - off, hi - off))
            j += 1
        out.append(items)
    return out


def load_files(paths, device, *, out_dtype=None, chunk_bytes: int = CHUNK_BYTES, n_slots: int = N_SLOTS, n_readers: int = N_READERS,
               stats: dict | None = None) -> dict:
    """Every tensor of ``paths`` as a fresh contiguous tensor on ``device`` ({name: Tensor}); floating tensors cast to ``out_dtype``
    when given (per tensor, as its bytes complete — see the module text); ``stats`` (optional dict) receives the counts and clocks."""
    import torch
    if not paths:
        return {}
    if n_slots < 2 or n_readers < 1 or chunk_bytes < 1 << 16:
        raise ValueError("load_files: n_slots >= 2, n_readers >= 1, chunk_bytes >= 64 KiB")
    t0 = time.perf_counter()
    p = plan(paths)
    dev = torch.device(device)
    use_cuda = dev.type == "cuda"
    if use_cuda:
        if dev.index is None:
            dev = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(dev)
    chunks = _chunks(p["headers"], chunk_bytes)
    overlaps = _overlaps(p["headers"], chunks)
    dst, dst_u8 = {}, {}
    cast, remaining, staging, staging_u8 = set(), {}, {}, {}                # out_dtype: tensors delivered cast (their fp32 bytes staged per tensor, released at the cast)
    staging_bytes, staging_peak, n_cast = 0, 0, 0
    for e in p["specs"]:
        src_dtype = getattr(torch, e.dtype)
        if out_dtype is not None and e.nbytes and src_dtype.is_floating_point and src_dtype != out_dtype:
            dst[e.name] = torch.empty(e.shape, dtype=out_dtype, device=dev)
            dst_u8[e.name] = None
            cast.add(e.name)
            remaining[e.name] = e.nbytes
            continue
        t = torch.empty(e.shape, dtype=src_dtype, device=dev)
        dst[e.name] = t
        dst_u8[e.name] = t.reshape(-1).view(torch.uint8) if e.nbytes else None
    spec_of = {e.name: e for e in p["specs"]}
    t_alloc = time.perf_counter()
    fds = [os.open(h["path"], os.O_RDONLY) for h in p["headers"]]
    slots = [torch.empty(chunk_bytes, dtype=torch.uint8, device="cpu", pin_memory=use_cuda) for _ in range(n_slots)]
    views = [memoryview(s.numpy()) for s in slots]
    free, ready = queue.Queue(), queue.Queue()
    for i in range(n_slots):
        free.put(i)
    errors = []
    bytes_read = [0]
    lock = threading.Lock()

    def reader(indices):
        try:
            for ci in indices:
                si = free.get()
                if si is None:
                    return
                fi, off, n = chunks[ci]
                mv = views[si][:n]
                got = 0
                while got < n:
                    r = os.preadv(fds[fi], [mv[got:]], off + got)
                    if r <= 0:
                        raise LayoutRefused(f"{p['headers'][fi]['path']}: short read at offset {off + got} ({got} of {n} bytes)")
                    got += r
                with lock:
                    bytes_read[0] += n
                ready.put((ci, si))
        except BaseException as e:                                   # noqa: BLE001 — surfaced on the main thread
            errors.append(e)
            ready.put((None, None))

    n_readers = max(1, min(n_readers, len(chunks))) if chunks else 0
    threads = [threading.Thread(target=reader, args=(list(range(k, len(chunks), n_readers)),), daemon=True, name=f"boot-reader-{k}")
               for k in range(n_readers)]
    t_start = time.perf_counter()
    for th in threads:
        th.start()
    pending = collections.deque()                                    # (event, slot) of chunks whose H2D copies are in flight
    copies = 0
    try:
        for _ in range(len(chunks)):
            ci, si = ready.get()
            if ci is None:
                raise errors[0]
            for e, dlo, dhi, slo, shi in overlaps[ci]:
                if e.name in cast:                                          # staged in the file's dtype until complete, then cast into place
                    if e.name not in staging:
                        staging[e.name] = torch.empty(e.shape, dtype=getattr(torch, e.dtype), device=dev)
                        staging_u8[e.name] = staging[e.name].reshape(-1).view(torch.uint8)
                        staging_bytes += e.nbytes
                        staging_peak = max(staging_peak, staging_bytes)
                    staging_u8[e.name][dlo:dhi].copy_(slots[si][slo:shi], non_blocking=use_cuda)
                    remaining[e.name] -= dhi - dlo
                    if remaining[e.name] == 0:                              # the tensor's last bytes: the cast (stream-ordered after its copies), the staging released
                        dst[e.name].copy_(staging.pop(e.name))
                        staging_u8.pop(e.name)
                        staging_bytes -= spec_of[e.name].nbytes
                        n_cast += 1
                else:
                    dst_u8[e.name][dlo:dhi].copy_(slots[si][slo:shi], non_blocking=use_cuda)
                copies += 1
            if use_cuda:
                ev = torch.cuda.Event()
                ev.record()
                pending.append((ev, si))
                while pending and pending[0][0].query():              # completed copies hand their slot back
                    free.put(pending.popleft()[1])
                if len(pending) >= n_slots - 1:                        # readers would starve: wait for the oldest copy
                    ev0, s0 = pending.popleft()
                    ev0.synchronize()
                    free.put(s0)
            else:
                free.put(si)
        while pending:
            ev0, s0 = pending.popleft()
            ev0.synchronize()
            free.put(s0)
        if use_cuda:
            torch.cuda.current_stream(dev).synchronize()
    finally:
        for _ in threads:
            free.put(None)
        for th in threads:
            th.join(timeout=30)
        for fd in fds:
            os.close(fd)
    if errors:
        raise errors[0]
    t_end = time.perf_counter()
    if bytes_read[0] != sum(n for _, _, n in chunks):
        raise LayoutRefused(f"read {bytes_read[0]} bytes, planned {sum(n for _, _, n in chunks)}")
    if staging or any(remaining[n] for n in cast):
        raise LayoutRefused(f"{len(staging)} staged tensor(s) never completed: {sorted(staging)[:5]}")
    if stats is not None:
        stats.update({"n_files": len(paths), "n_tensors": len(p["specs"]), "n_bytes": p["n_bytes"], "n_chunks": len(chunks),
                      "n_copies": copies, "chunk_bytes": chunk_bytes, "n_slots": n_slots, "n_readers": n_readers,
                      "device": str(dev), "out_dtype": str(out_dtype) if out_dtype is not None else None, "n_cast": n_cast, "staging_peak_bytes": staging_peak,
                      "wall_s": round(t_end - t0, 4), "plan_alloc_s": round(t_alloc - t0, 4),
                      "pin_s": round(t_start - t_alloc, 4), "stream_s": round(t_end - t_start, 4),
                      "gb_per_s": round(p["n_bytes"] / max(t_end - t_start, 1e-9) / 1e9, 3)})
    return dst
