"""Parallel-deflate form of predict_to_bigwig.write_predictions_h5py (chrombpnet v1.0.1; h5py 3.11.0 / HDF5 1.14.2 / zlib 1.2.11 on the
pinned stack) that writes the same file bytes. The stock writes predictions/profs with create_dataset(data=profile, dtype=float,
compression='gzip') (auto chunks, level 4): HDF5 compresses every chunk serially through its deflate filter (zlib compress2 level 4 of the
full chunk buffer, edge chunks padded with the fill value 0). This writer keeps the STOCK's call sequence (groups, coords datasets by
h5py, profs, logcounts by h5py) and replaces only the profs write: the chunks are compressed in a thread pool (zlib releases the GIL)
and written raw (write_direct_chunk) in the order HDF5's chunk cache would have flushed them during the stock's single H5Dwrite
(write_order: H5D__chunk_lock / cache_prune / flush at the build's default cache parameters — rdcc 1 MiB / 521 slots / w0 0.75 on
HDF5 1.14.2), with the coords datasets' cached chunks flushed (their handles closed, after the logcounts dataset is created) between the
evicted chunks and the residue — the stock's close-time order (coords_chrom, coords_center, profs residue, logcounts). Below H5_FAST_MIN_N
rows the stock's own writer runs instead (the cache emulation is byte-exact only from there up). self_check() compares the stock's own
write with this writer on a synthetic array (file sha256) and raises on a difference; the job path never runs it."""
import os, tempfile, zlib, numpy as np, h5py
from ._oom import is_oom                                          # the kit's one out-of-memory classifier: the stock-writer source check below re-raises an OOM first
from concurrent.futures import ThreadPoolExecutor


def _log2_gen(n): return max(0, int(n).bit_length() - 1)
def _power2up(n):
    p = 1
    while p < n: p <<= 1
    return p


def hdf5_major():
    import h5py; return int(h5py.version.hdf5_version.split(".")[0])


def cache_params_of_build():
    """The running HDF5 build's DEFAULT chunk-cache parameters (what the stock's h5py.File(..., 'w') gets): (rdcc_nbytes, rdcc_nslots, rdcc_w0) read
        from a default file-access property list — HDF5 1.14.2: (1 MiB, 521, 0.75); HDF5 2.0.0: (8 MiB, 4099, 0.75); the same cache emulation reproduces
        either build's chunk address order at that build's own parameters."""
    import h5py, tempfile, os
    p = os.path.join(tempfile.mkdtemp(prefix="h5_fast_plist_"), "x.h5")
    with h5py.File(p, "w") as f: mdc, nslots, nbytes, w0 = f.id.get_access_plist().get_cache()
    os.remove(p); return int(nbytes), int(nslots), float(w0)


def write_order_for_build(shape, chunk, itemsize):
    """The chunk address order of the running HDF5 build = the chunk-cache emulation (write_order) at the BUILD's default cache parameters
        (read from the access plist, never assumed); self_check() confirms it on a given build and raises on a mismatch (no silent fallback)."""
    nbytes, nslots, w0 = cache_params_of_build()
    return write_order(shape, chunk, itemsize, nbytes_max=nbytes, nslots=nslots, w0=w0)


def write_order(shape, chunk, itemsize, nbytes_max=1048576, nslots=521, w0=0.75):
    """HDF5 1.14.2 chunk-cache order for a whole-dataset write of a chunked+filtered 2-D dataset; returns (order, n_evicted):
    order[:n_evicted] were flushed during the write (in that order), order[n_evicted:] flushed at close (cache head -> tail)."""
    nrow = -(-shape[0] // chunk[0]); ncol = -(-shape[1] // chunk[1]); csize = chunk[0] * chunk[1] * itemsize
    bits_col = _log2_gen(_power2up(ncol))
    def slot(r, c): return ((r << bits_col) ^ c) % nslots
    order = []; lst = []; slots = {}
    def nxt(x):
        if x is None or x not in lst: return None
        i = lst.index(x) + 1; return lst[i] if i < len(lst) else None
    def evict(ent):
        order.append(ent["idx"]); lst.remove(ent); slots.pop(ent["slot"], None)
    for r in range(nrow):
        for c in range(ncol):
            idx = r * ncol + c; s = slot(r, c)
            if s in slots: evict(slots[s])
            nused = len(lst); w = int(nused * w0); p = [lst[0] if lst else None, None]; nbytes_used = len(lst) * csize
            while (p[0] is not None or p[1] is not None) and nbytes_used + csize > nbytes_max:
                if w == 0: p[1] = lst[0] if lst else None
                n = [nxt(p[0]), nxt(p[1])]
                for i in range(2):
                    if not (nbytes_used + csize > nbytes_max): break
                    cur = None
                    if i == 0 and p[0] is not None and ((p[0]["rd"] == 0 and p[0]["wr"] == 0) or (p[0]["rd"] == 0 and p[0]["wr"] == csize) or (p[0]["rd"] == csize and p[0]["wr"] == 0)): cur = p[0]
                    elif i == 1 and p[1] is not None: cur = p[1]
                    if cur is not None:
                        for j in range(2):
                            if p[j] is cur: p[j] = None
                            if n[j] is cur: n[j] = nxt(cur)
                        evict(cur); nbytes_used -= csize
                p = n; w -= 1
            rows = min(chunk[0], shape[0] - r * chunk[0]); cols = min(chunk[1], shape[1] - c * chunk[1]); naccessed = rows * cols * itemsize
            ent = {"idx": idx, "rd": csize, "wr": csize - naccessed, "slot": s}; lst.append(ent); slots[s] = ent
    n_evicted = len(order)
    for ent in list(lst): order.append(ent["idx"])
    return order, n_evicted


def _guess_chunks(shape, dtype):
    """h5py's auto-chunk shape for (shape, dtype, gzip) — read off a throwaway dataset in a private temporary file (tempfile.mkstemp: made
    0600 with O_EXCL under a name no other process predicts, removed here), never a fixed name in the shared temporary directory"""
    fd, p = tempfile.mkstemp(prefix="fastkit_h5_probe_", suffix=".h5"); os.close(fd)
    try:
        with h5py.File(p, "w") as f: ch = f.create_dataset("x", shape=shape, dtype=dtype, chunks=True, compression="gzip").chunks
    finally:
        os.remove(p)
    return ch


def _chunk_blob(a, ch, r, c, level):
    buf = np.zeros(ch, dtype=a.dtype); blk = a[r * ch[0]:(r + 1) * ch[0], c * ch[1]:(c + 1) * ch[1]]; buf[:blk.shape[0], :blk.shape[1]] = blk
    return zlib.compress(buf.tobytes(), level)


def _write_profs(f, group, name, a, workers, level=4, between=None, blobs_by_idx=None):
    """the stock's pred_group.create_dataset('profs', data=profile, dtype=float, compression='gzip') as direct chunk writes in cache order;
    `between()` runs after the evicted chunks and before the residue — where the stock's close-time flushes of the OTHER datasets land"""
    ch = _guess_chunks(a.shape, a.dtype); ncol = -(-a.shape[1] // ch[1])
    d = group.create_dataset(name, shape=a.shape, dtype=a.dtype, chunks=ch, compression="gzip")
    order, n_ev = write_order_for_build(a.shape, ch, a.dtype.itemsize)
    with ThreadPoolExecutor(workers) as ex:                       # precomputed blobs (metrics_overlap) are the same function on the same rows; missing ones computed here
        blobs = list(ex.map(lambda idx: blobs_by_idx[idx] if (blobs_by_idx is not None and idx in blobs_by_idx) else _chunk_blob(a, ch, idx // ncol, idx % ncol, level), order))
    for k, idx in enumerate(order):
        if k == n_ev and between is not None: between()
        d.id.write_direct_chunk(((idx // ncol) * ch[0], (idx % ncol) * ch[1]), blobs[k], 0)
    if n_ev == len(order) and between is not None: between()
    return d, ch, n_ev


STOCK_H5_WRITER_SHA256_16 = "584f1236dc3ced22"      # sha256[:16] of the stock's `def write_predictions_h5py` source segment (chrombpnet 1.0.1 predict_to_bigwig.py)
_STOCK_H5_SRC = 'def write_predictions_h5py(output_prefix, profile, logcts, coords):\n    # open h5 file for writing predictions\n    output_h5_fname = "{}_predictions.h5".format(output_prefix)\n    h5_file = h5py.File(output_h5_fname, "w")\n    # create groups\n    coord_group = h5_file.create_group("coords")\n    pred_group = h5_file.create_group("predictions")\n\n    num_examples=len(coords)\n\n    coords_chrom_dset =  [str(coords[i][0]) for i in range(num_examples)]\n    coords_center_dset =  [int(coords[i][1]) for i in range(num_examples)]\n\n    dt = h5py.special_dtype(vlen=str)\n\n    # create the "coords" group datasets\n    coords_chrom_dset = coord_group.create_dataset(\n        "coords_chrom", data=np.array(coords_chrom_dset, dtype=dt),\n        dtype=dt, compression="gzip")\n    coords_start_dset = coord_group.create_dataset(\n        "coords_center", data=coords_center_dset, dtype=int, compression="gzip")\n\n    # create the "predictions" group datasets\n    profs_dset = pred_group.create_dataset(\n        "profs",\n        data=profile,\n        dtype=float, compression="gzip")\n    logcounts_dset = pred_group.create_dataset(\n        "logcounts", data=logcts,\n        dtype=float, compression="gzip")\n\n    # close hdf5 file\n    h5_file.close()'
_STOCK_WRITER = {"fn": None, "how": None}
def _stock_h5_writer():
    """The STOCK's write_predictions_h5py WITHOUT importing its module (predict_to_bigwig.py imports TensorFlow at module top). The vendored
        source is compared at call time with the installed stock file's own bytes (AST source segment == the vendored text AND its sha256[:16] ==
        STOCK_H5_WRITER_SHA256_16); a mismatch (another chrombpnet version) imports the stock module and uses ITS function, said on the line."""
    if _STOCK_WRITER["fn"] is not None: return _STOCK_WRITER["fn"], _STOCK_WRITER["how"]
    import importlib.util, ast, hashlib
    try:
        sp = importlib.util.find_spec("chrombpnet.evaluation.make_bigwigs.predict_to_bigwig"); src = open(sp.origin).read(); seg = None
        for n in ast.parse(src).body:
            if isinstance(n, ast.FunctionDef) and n.name == "write_predictions_h5py": seg = ast.get_source_segment(src, n)
        ok = seg is not None and seg == _STOCK_H5_SRC and hashlib.sha256(seg.encode()).hexdigest()[:16] == STOCK_H5_WRITER_SHA256_16
    except Exception as e:
        if is_oom(e): raise                                           # an out-of-memory error propagates; the stock-module route below is for every other failure
        ok = False
    if ok:
        ns = {"h5py": h5py, "np": np}; exec(compile(_STOCK_H5_SRC, "<stock write_predictions_h5py verbatim>", "exec"), ns)
        _STOCK_WRITER.update(fn=ns["write_predictions_h5py"], how=f"the stock's write_predictions_h5py VERBATIM (source segment sha256-16 {STOCK_H5_WRITER_SHA256_16}, equal to the installed stock file's; no TF import)")
    else:
        import chrombpnet.evaluation.make_bigwigs.predict_to_bigwig as p2b
        _STOCK_WRITER.update(fn=p2b.write_predictions_h5py, how="the stock module imported (its write_predictions_h5py source differs from the vendored 1.0.1 text -> the installed stock's own function; TF imported)")
    return _STOCK_WRITER["fn"], _STOCK_WRITER["how"]
H5_FAST_MIN_N = 20000        # the fast writer's chunk-cache emulation is byte-exact from here up; below, the stock's own writer
LAST_FORM = {"form": None}
def write_predictions_h5py_fast(output_prefix, profile, logcts, coords, workers=8, profs_blobs=None):
    from .hostres import cap; workers = cap(workers)          # the container's CPUs cap the pool
    output_h5_fname = "{}_predictions.h5".format(output_prefix)
    num_examples = len(coords)
    if 1 < num_examples < H5_FAST_MIN_N:
        # the chunk-cache emulation is byte-exact only in its regime (N >= H5_FAST_MIN_N); below it the STOCK's OWN writer runs (its bytes by
        # construction; little time to gain there). N = 1 keeps the fast path: the stock's writer cannot write a 0-d logcounts (the N = 1 scalar fix lives here).
        fn, how = _stock_h5_writer()
        LAST_FORM["form"] = f"stock writer (N = {num_examples} < H5_FAST_MIN_N = {H5_FAST_MIN_N}: below the fast writer's regime; {how})"
        return fn(output_prefix, profile, logcts, coords)
    LAST_FORM["form"] = f"fast writer (N = {num_examples}; its regime N >= {H5_FAST_MIN_N} or N = 1)"
    coords_chrom_dset = [str(coords[i][0]) for i in range(num_examples)]
    coords_center_dset = [int(coords[i][1]) for i in range(num_examples)]
    dt = h5py.special_dtype(vlen=str)
    prof64 = np.asarray(profile, dtype=np.float64)                 # dtype=float: h5py's conversion of the float32 softmax (exact)
    with h5py.File(output_h5_fname, "w") as h5_file:
        coord_group = h5_file.create_group("coords"); pred_group = h5_file.create_group("predictions")
        # h5py handle LIFETIMES are part of the file layout (they decide when cached chunks flush and where the logcounts object header
        # lands). The stock keeps every Dataset handle until h5_file.close():
        # the coords chunks (fitting their caches) and the logcounts chunks flush at close, in creation order, AFTER profs' evicted chunks
        # and with profs' cached residue between them; the logcounts object header is allocated at its creation, before those flushes.
        # Sequence reproduced: coords (cached) -> profs evicted chunks -> logcounts created (cached) -> coords_chrom, coords_center closed
        # (flushed) -> profs residue -> profs, logcounts closed (flushed) -> file.
        coords_chrom_dset = coord_group.create_dataset("coords_chrom", data=np.array(coords_chrom_dset, dtype=dt), dtype=dt, compression="gzip")
        coords_start_dset = coord_group.create_dataset("coords_center", data=coords_center_dset, dtype=int, compression="gzip")
        holder = {}
        def between():
            # N = 1: a 0-d logcounts refuses gzip ('Scalar datasets don't support chunk/filter options') -> shape (N,) always
            holder["logcounts"] = pred_group.create_dataset("logcounts", data=np.atleast_1d(np.asarray(logcts)), dtype=float, compression="gzip")
            coords_chrom_dset.id.close(); coords_start_dset.id.close()
        profs_dset, ch, n_ev = _write_profs(h5_file, pred_group, "profs", prof64, workers, between=between, blobs_by_idx=profs_blobs)
        profs_dset.id.close(); holder["logcounts"].id.close()
    return {"profs_chunks": ch, "profs_evicted": n_ev, "workers": workers, "h5py": h5py.__version__, "hdf5": h5py.version.hdf5_version, "zlib": zlib.ZLIB_RUNTIME_VERSION}


def self_check(n=20000, L=1000, workers=4):
    """stock write_predictions_h5py vs the fast writer on synthetic data of the real shape class (n=20,000 rows: profs evicts chunks, the
        coords/logcounts caches hold until close -> sensitive to the handle-lifetime layout); FILE sha256 equal or RuntimeError (no silent
        fallback). A self-test for a given stack; the job path never runs it."""
    import tempfile, hashlib
    rng = np.random.default_rng(0); x = rng.random((n, L)).astype(np.float32); x /= x.sum(axis=1, keepdims=True); lc = rng.random(n).astype(np.float32)
    coords = [["chr%d" % (1 + i % 23), 1000 + 2000 * i] for i in range(n)]; dt = h5py.special_dtype(vlen=str)
    d = tempfile.mkdtemp(prefix="h5fast_check_"); pa, pb = os.path.join(d, "stock"), os.path.join(d, "fast")
    with h5py.File(pa + "_predictions.h5", "w") as f:
        cg = f.create_group("coords"); pg = f.create_group("predictions")
        d1 = cg.create_dataset("coords_chrom", data=np.array([str(c[0]) for c in coords], dtype=dt), dtype=dt, compression="gzip")
        d2 = cg.create_dataset("coords_center", data=[int(c[1]) for c in coords], dtype=int, compression="gzip")
        d3 = pg.create_dataset("profs", data=x, dtype=float, compression="gzip"); d4 = pg.create_dataset("logcounts", data=lc, dtype=float, compression="gzip")
        del d1, d2, d3, d4                                                       # the stock's handle lifetimes (kept until file close)
    write_predictions_h5py_fast(pb, x, lc, coords, workers=workers)
    sa, sb = (hashlib.sha256(open(p + "_predictions.h5", "rb").read()).hexdigest() for p in (pa, pb))
    for p in (pa, pb): os.remove(p + "_predictions.h5")
    if sa != sb: raise RuntimeError(f"[h5_fast] self-check FAILED on this HDF5 ({h5py.version.hdf5_version}): stock {sa[:16]} != fast {sb[:16]} — refused (no silent fallback)")
    return sa
