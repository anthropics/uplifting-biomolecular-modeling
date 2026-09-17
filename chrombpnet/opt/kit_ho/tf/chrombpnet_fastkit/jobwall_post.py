"""chrombpnet_jobwall_post — exact (byte-identical) replacements for the host-side post/writer stages of `chrombpnet pred_bw`
(chrombpnet 1.0.1 @ eaa0fe58; the pinned stack: numpy 1.23.4, scipy 1.10.1, h5py 3.11.0 / HDF5 1.14.2, zlib 1.2.11).

Four drop-ins, each with the stock signature:

  write_predictions_h5py_fast(output_prefix, profile, logcts, coords)      <- predict_to_bigwig.write_predictions_h5py (l.18-50)
  profile_metrics_fast(true_counts, pred_probs, pseudocount=0.001)         <- training/metrics.profile_metrics (l.43-93)
  get_cts_fast(peaks_df, bw, width)                                        <- training/utils/data_utils.get_cts (l.21-35)
  write_stats_fast(all_entries, outstats_file)                             <- bigwig_helper.write_bigwig's -os block (l.115-126)

Exactness: every output file byte-identical to the stock writer's on the same host.
The h5 writer reproduces HDF5's file layout by emulating the chunk-cache eviction order of the stock's single H5Dwrite
(H5D__chunk_cache_prune: nbytes_max 1 MiB, w0 0.75, method 0 = fully-written chunks from the LRU head, method 1 = LRU) and the
stock's dataset-close order (coords_chrom, coords_center, profs, logcounts), while the per-chunk deflate (zlib level 4 = HDF5's
deflate filter) runs in a thread pool and lands through H5Dwrite_chunk (h5py write_direct_chunk).
Host-arithmetic note: the VALUES are unchanged (same numpy calls, same order); 'identical as printed' for metrics/stats text is a
property of one host's numerics stack (numpy / scipy versions and the libm underneath).
"""
import os
import zlib
from concurrent.futures import ThreadPoolExecutor

import numpy as np


# --------------------------------------------------------------------------------------------------------------- h5
def hdf5_chunk_cache_order(shape, chunks, itemsize, nbytes_max=1024 * 1024, w0=0.75):
    """Allocation order HDF5 1.14 produces for ONE sequential H5Dwrite over a whole chunked+filtered dataset with the default
    chunk cache (rdcc_nbytes 1 MiB, w0 0.75; no slot collisions at these geometries): returns (evicted_during_write, flushed_at_close).
    Emulates H5D__chunk_cache_prune: p[0] walks the LRU list from the head evicting fully-written chunks; p[1] starts at the head once
    w0*nused steps passed and evicts regardless (partial edge chunks are never 'fully written')."""
    n0, n1 = shape
    cr, cc = chunks
    nrow = (n0 + cr - 1) // cr
    ncol = (n1 + cc - 1) // cc
    csize = cr * cc * itemsize
    lru = []
    evicted = []

    def wr_count(idx):
        i, j = divmod(idx, ncol)
        return min(cr, n0 - i * cr) * min(cc, n1 - j * cc) * itemsize

    for idx in range(nrow * ncol):
        nused = len(lru)
        if nused * csize + csize > nbytes_max:
            w = [int(nused * w0)]
            p = [0 if lru else None, None]
            while (p[0] is not None or p[1] is not None) and (len(lru) * csize + csize) > nbytes_max:
                if w[0] == 0:
                    p[1] = 0 if lru else None
                nxt = [(p[i] + 1 if (p[i] is not None and p[i] + 1 < len(lru)) else None) for i in range(2)]
                for i in range(2):
                    if (len(lru) * csize + csize) <= nbytes_max:
                        break
                    cur = None
                    if i == 0 and p[0] is not None and lru[p[0]][1] == csize:
                        cur = p[0]
                    elif i == 1 and p[1] is not None:
                        cur = p[1]
                    if cur is not None:
                        ent = lru.pop(cur)
                        evicted.append(ent[0])
                        for j in range(2):
                            if p[j] == cur:
                                p[j] = None
                            elif p[j] is not None and p[j] > cur:
                                p[j] -= 1
                            if nxt[j] == cur:
                                nxt[j] = cur if cur < len(lru) else None
                            elif nxt[j] is not None and nxt[j] > cur:
                                nxt[j] -= 1
                p = nxt
                w[0] -= 1
        lru.append([idx, wr_count(idx)])
    return evicted, [e[0] for e in lru]


def write_predictions_h5py_fast(output_prefix, profile, logcts, coords, nthreads=None):
    """Byte-identical drop-in for predict_to_bigwig.write_predictions_h5py: same groups/datasets/dtypes/filters/order; the 'profs'
    chunks are deflated in parallel and written in the stock's allocation order."""
    import h5py
    from h5py._hl.filters import guess_chunk

    output_h5_fname = "{}_predictions.h5".format(output_prefix)
    h5_file = h5py.File(output_h5_fname, "w")
    coord_group = h5_file.create_group("coords")
    pred_group = h5_file.create_group("predictions")
    num_examples = len(coords)
    coords_chrom_dset = [str(coords[i][0]) for i in range(num_examples)]
    coords_center_dset = [int(coords[i][1]) for i in range(num_examples)]
    dt = h5py.special_dtype(vlen=str)
    # the stock keeps these two Dataset objects alive until h5_file.close(): their chunk caches flush at close, in creation order
    coords_chrom_dset = coord_group.create_dataset("coords_chrom", data=np.array(coords_chrom_dset, dtype=dt), dtype=dt, compression="gzip")
    coords_start_dset = coord_group.create_dataset("coords_center", data=coords_center_dset, dtype=int, compression="gzip")
    prof = np.asarray(profile)
    shape = prof.shape
    chunks = guess_chunk(shape, None, np.dtype(float).itemsize)            # the chunk shape h5py picks for the stock's data= call
    profs_dset = pred_group.create_dataset("profs", shape=shape, dtype=float, compression="gzip", chunks=chunks)
    level = profs_dset.compression_opts
    cr, cc = chunks
    n0, n1 = shape
    ncol = (n1 + cc - 1) // cc
    fill = np.zeros((cr, cc), dtype="<f8")

    def one(idx):
        i, j = divmod(idx, ncol)
        i0, j0 = i * cr, j * cc
        blk = fill.copy()
        a = prof[i0:i0 + cr, j0:j0 + cc]
        blk[:a.shape[0], :a.shape[1]] = a
        return idx, zlib.compress(blk.tobytes(), level)

    evicted, at_close = hdf5_chunk_cache_order(shape, chunks, 8)
    with ThreadPoolExecutor(max_workers=nthreads or os.cpu_count()) as ex:
        comp = dict(ex.map(one, range(len(evicted) + len(at_close))))
    for idx in evicted:                                                     # chunks the stock's write evicts from the cache, in that order
        i, j = divmod(idx, ncol)
        profs_dset.id.write_direct_chunk((i * cr, j * cc), comp[idx], filter_mask=0)
    logcounts_dset = pred_group.create_dataset("logcounts", data=logcts, dtype=float, compression="gzip")
    # stock close order: coords_chrom, coords_center (their cached chunks land here), then profs' cache remainder, then logcounts
    del coords_chrom_dset
    del coords_start_dset
    for idx in at_close:
        i, j = divmod(idx, ncol)
        profs_dset.id.write_direct_chunk((i * cr, j * cc), comp[idx], filter_mask=0)
    del profs_dset
    del logcounts_dset
    h5_file.close()


# --------------------------------------------------------------------------------------------------------------- metrics
def profile_metrics_fast(true_counts, pred_probs, pseudocount=0.001):
    """Bitwise drop-in for training/metrics.profile_metrics: the per-row jensenshannon calls become one axis=1 call (same per-row
    pairwise sums), the min/max bounds and the normalisation are vectorised from metrics_utils.jsd_min_max_bounds /
    get_min_max_normalized_value, and the shuffled-label draws keep the stock's RNG call order (np.random.permutation per row)."""
    from scipy.spatial.distance import jensenshannon

    tc = np.asarray(true_counts)
    pp = np.asarray(pred_probs)
    n, L = tc.shape
    p_true = tc / (pseudocount + np.nansum(tc, axis=1, keepdims=True))          # metrics.py:68
    jsd_pw = jensenshannon(p_true, pp, axis=1)                                  # metrics.py:68
    uniform_profile = np.ones(L) * (1.0 / L)                                    # metrics_utils.py:193
    profile_prob = tc / np.sum(tc, axis=1, keepdims=True)                       # metrics_utils.py:196
    max_jsd = jensenshannon(profile_prob, np.broadcast_to(uniform_profile, tc.shape), axis=1)   # metrics_utils.py:199
    min_jsd = 0.0                                                               # metrics_utils.py:202

    def norm(val):                                                              # metrics_utils.py:126-134
        r = (val - max_jsd) / (min_jsd - max_jsd)
        r = np.where(r < 0, 0.0, r)
        return np.where(r > 1, 1.0, r)

    jsd_norm = norm(jsd_pw)
    shuffled = np.empty_like(tc)
    for i in range(n):                                                          # metrics.py:76 — same draws, same order
        shuffled[i, :] = np.random.permutation(tc[i, :])
    shuffled_prob = shuffled / (pseudocount + np.nansum(shuffled, axis=1, keepdims=True))   # metrics.py:77
    jsd_rnd = jensenshannon(p_true, shuffled_prob, axis=1)                      # metrics.py:87
    jsd_rnd_norm = norm(jsd_rnd)                                                # metrics.py:90
    empty = np.array([])
    return empty, empty, jsd_pw, jsd_norm, jsd_rnd, jsd_rnd_norm, empty, empty


# --------------------------------------------------------------------------------------------------------------- observed counts
def get_cts_fast(peaks_df, bw, width):
    """Bitwise drop-in for data_utils.get_cts: the same bw.values(chr, start, end) calls in the same order, without the per-row
    pandas Series construction (iterrows) and with pyBigWig's numpy return (float32 values widened to float64 exactly, as the
    stock's list path does)."""
    ch = peaks_df["chr"].values
    st = (peaks_df["start"].values + peaks_df["summit"].values - width // 2)
    out = np.empty((len(peaks_df), width), dtype=np.float64)
    for i in range(len(peaks_df)):
        out[i] = np.nan_to_num(bw.values(str(ch[i]), int(st[i]), int(st[i]) + width, numpy=True))
    return out


# --------------------------------------------------------------------------------------------------------------- -os stats
STATS_LABELS = ("Min", ".1%", "1%", "50%", "99%", "99.9%", "99.95%", "99.99%", "Max")
STATS_Q = (0.001, 0.01, 0.5, 0.99, 0.999, 0.9995, 0.9999)


def write_stats_fast(all_entries, outstats_file):
    """Identical text to bigwig_helper.write_bigwig's -os block (l.116-126): the seven np.quantile calls (seven partitions of the
    written values) become one call with the seven probabilities (one partition, the same per-quantile interpolation)."""
    q = np.quantile(all_entries, STATS_Q)
    vals = [np.min(all_entries)] + list(q) + [np.max(all_entries)]
    with open(outstats_file, "w") as f:
        for lab, v in zip(STATS_LABELS, vals):
            f.write("{}\t{:.6f}\n".format(lab, v))
