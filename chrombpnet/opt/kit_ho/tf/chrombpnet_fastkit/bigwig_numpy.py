"""bigwig_numpy — a numpy reimplementation of libBigWig's bigWig WRITER (as bundled in pyBigWig 0.3.22, libBigWig bwWrite.c) for the
call pattern this kit uses: bw.addHeader(chrom_sizes, maxZooms=M); bw.addEntries(chrom, start, values=v, span=1, step=1)
in genome order; bw.close(). Goal: the same file bytes as pyBigWig writes for that call pattern, produced
incrementally so the zoom-level construction (libBigWig does it at close, single-threaded, re-reading the file) overlaps the GPU.

Reproduced from the C source including its quirks (each one load-bearing for byte identity):
  * pyBigWig appends contiguous span/step calls (same chrom, start == last end) to the open buffer -> data blocks follow RUNS of
    contiguous bases, chunked at 8185 values (bufSize 32768: flush when l+4 >= bufSize); an in-loop flush sets end = start + 8185,
    the last block of a run sets end = start + n + 6 (wb->l>>2 with the 24-byte header counted — libBigWig bug, kept).
  * header stats: minVal/maxVal with the `if (v < min) min = v; else if (v > max) max = v;` quirk; sumData/sumSquared in double.
  * zoom levels: zoom0 = max(10, 4 * (4 * floor(meanWidth))) then x4 per level up to the requested count while <= the longest
    chromosome; bins come from a per-base state machine: a bin's window is [s, s+Z) from the first base s it was created at (NOT
    aligned to Z); a base extends the last bin iff same tid and s <= x < s+Z, else a new bin starts at x; a buffer holds 1023 bins
    (l+32 >= 32768 is tested BEFORE the overlap test), so the base arriving at a full buffer always starts a new buffer AND a new
    bin — the 1023rd bin of every buffer holds exactly one base; sum/sumsq (double accumulators) are stored as float32 only when the
    NEXT bin is created, so the last bin of every buffer and the last bin of the level keep sum = sumsq = 0 (libBigWig bug, kept).
  * duplicate-level rule: a level whose number of extra buffers equals the previous level's ends the list (actualNLevels).
  * R-tree index (blockSize 64) built by addLeaves' ceil(toProcess/(blockSize-i)) distribution, written root first then level by level.
  * zlib `compress()` = deflate level 6, 15-bit window, memLevel 8 -> python zlib.compress(b, 6) (same zlib -> same bytes).
"""
import math, struct, zlib
import numpy as np

BIGWIG_MAGIC = 0x888FFC26; CIRTREE_MAGIC = 0x78CA8C91; IDX_MAGIC = 0x2468ACE0
BUF_SIZE = 32768; BLOCK_SIZE = 64; ITEMS_PER_SLOT_ZOOM = BUF_SIZE // 32          # 1024 slots
VALS_PER_BLOCK = (BUF_SIZE - 24) // 4 - 1                                          # 8185: flush when 24 + 4k + 4 >= 32768
IVALS_PER_BLOCK = (BUF_SIZE - 24) // 12                                            # 2728: flush when 24 + 12k + 12 > 32768 (type-1 blocks)
ZOOM_BINS_PER_BUFFER = ITEMS_PER_SLOT_ZOOM - 1                                     # 1023: l+32 >= m with 1023 entries
DBL_MAX = float(np.finfo(np.float64).max); DBL_MIN = float(np.finfo(np.float64).tiny)   # C's DBL_MIN = smallest positive normal
ZOOM_REC = np.dtype([("tid", "<u4"), ("start", "<u4"), ("end", "<u4"), ("count", "<u4"), ("min", "<f4"), ("max", "<f4"), ("sum", "<f4"), ("sumsq", "<f4")])


def _seq_add(acc, arr64):
    """acc + arr[0] + arr[1] + ... accumulated sequentially in double (the C loop)."""
    if len(arr64) == 0: return acc
    return float(np.cumsum(np.concatenate(([acc], arr64)))[-1])


def _seq_sums(v64, v2, starts_rel, counts):
    """Per-segment sequential double sums of v64 / v2 over contiguous segments [starts_rel[i], starts_rel[i]+counts[i])."""
    nb = len(counts); ids = np.repeat(np.arange(nb), counts)
    lo = starts_rel[0]; hi = starts_rel[-1] + counts[-1]
    return np.bincount(ids, weights=v64[lo:hi], minlength=nb), np.bincount(ids, weights=v2[lo:hi], minlength=nb)


class _ZoomLevel:
    """State machine of one zoom level, fed with closed runs of contiguous bases (tid, a, b, values) in genome order, a chunk of
    runs at a time (the per-base logic is reduced to a scalar scan per run + one vectorised expansion per chunk). Completed buffers
    (1023 bins) are compressed as soon as they are final, so close() only has the open buffer left."""

    def __init__(self, Z, compress=None):
        self.Z = int(Z); self.c = 0; self.nbins = 0; self.compress = compress or (lambda raw: zlib.compress(raw, 6))
        self.open = None                     # open bin: dict(tid, s, arrays, idx) -> the last bin created (extendable by later runs)
        self.pending = []                    # per-chunk arrays of bins not yet in a finalized buffer
        self.pending_base = 0                # global index of the first pending bin
        self.breaks_pending = []             # global bin indices where a new buffer starts, not yet flushed
        self.blocks_done = []                # (tid0, tid1, start0, end1, compressed) per finalized buffer
        self.n_buffers_extra = 0

    def feed_runs(self, runs, V, V64, V2, POS):
        """runs: list of (tid, a, b, off) with off = offset of base a in V (the chunk's concatenated values, file order);
        V64/V2 = V as double and its squares; POS = absolute base position per element of V."""
        Z = self.Z; FULL = ZOOM_BINS_PER_BUFFER
        c = self.c; op = self.open; o_tid = op["tid"] if op else None; o_s = op["s"] if op else None
        ext_first = 0                        # bases at the start of V extending the carried open bin (before any new bin of this chunk)
        ep_start = []; ep_n = []; ep_z = []; ep_end = []; ep_off = []; ep_tid = []      # epochs of new bins (arithmetic progressions)
        breaks = []; nnew = 0                # breaks: local indices (in this chunk's new bins) of the first bin of a new buffer
        for tid, a, b, off in runs:
            x = a
            if c == FULL:                    # full buffer: the first base opens a new buffer + a new bin
                breaks.append(nnew); c = 0; self.n_buffers_extra += 1
            elif o_s is not None and tid == o_tid and a < o_s + Z:        # extend the open bin (no new bin)
                e = min(b, o_s + Z)
                if nnew == 0: ext_first += e - a
                x = e
            while x < b:
                m = FULL - c; nreg = min(m - 1, -(-(b - x) // Z))
                if nreg > 0:
                    ep_start.append(x); ep_n.append(nreg); ep_z.append(Z); ep_end.append(b); ep_off.append(off + (x - a)); ep_tid.append(tid)
                    nnew += nreg; c += nreg; x = min(b, x + Z * nreg); o_tid = tid; o_s = x - Z if x < b else ep_start[-1] + Z * (nreg - 1)
                if c == FULL - 1 and x < b:  # the next bin fills the buffer: exactly one base
                    ep_start.append(x); ep_n.append(1); ep_z.append(1); ep_end.append(b); ep_off.append(off + (x - a)); ep_tid.append(tid)
                    nnew += 1; c += 1; o_tid = tid; o_s = x; x += 1
                    if x < b: breaks.append(nnew); c = 0; self.n_buffers_extra += 1
        # carried open bin: extension bases at the chunk start
        if ext_first:
            arr = op["arrays"]; i = op["idx"]; k = ext_first
            arr[3][i] += k; arr[4][i] = min(arr[4][i], float(V[:k].min())); arr[5][i] = max(arr[5][i], float(V[:k].max()))
            arr[6][i] = _seq_add(float(arr[6][i]), V64[:k]); arr[7][i] = _seq_add(float(arr[7][i]), V2[:k]); arr[2][i] = int(POS[k - 1]) + 1
        if nnew:
            n_e = np.array(ep_n, dtype=np.int64); st_e = np.array(ep_start, dtype=np.int64); z_e = np.array(ep_z, dtype=np.int64)
            end_e = np.array(ep_end, dtype=np.int64); off_e = np.array(ep_off, dtype=np.int64); tid_e = np.array(ep_tid, dtype=np.int64)
            cum = np.cumsum(n_e) - n_e; j = np.arange(nnew, dtype=np.int64) - np.repeat(cum, n_e)
            starts = np.repeat(st_e, n_e) + np.repeat(z_e, n_e) * j
            rel = np.repeat(off_e, n_e) + (starts - np.repeat(st_e, n_e))                 # offset of each bin's first base in V
            nxt = np.append(rel[1:], len(V)); cnt = nxt - rel                                 # a bin owns every base up to the next bin
            ends = POS[nxt - 1] + 1
            mins = np.minimum.reduceat(V, rel).astype(np.float64); maxs = np.maximum.reduceat(V, rel).astype(np.float64)
            ids = np.repeat(np.arange(nnew), cnt); base0 = rel[0]
            s64 = np.bincount(ids, weights=V64[base0:], minlength=nnew); q64 = np.bincount(ids, weights=V2[base0:], minlength=nnew)
            arrays = [np.repeat(tid_e, n_e), starts, ends, cnt, mins, maxs, s64, q64]
            base = self.nbins
            for bi in breaks: self.breaks_pending.append(base + bi)
            self.pending.append(arrays); self.nbins += nnew
            self.open = {"tid": int(arrays[0][-1]), "s": int(starts[-1]), "arrays": arrays, "idx": nnew - 1}
        elif breaks:                         # a break with no new bin cannot happen (a break is always followed by a new bin)
            raise AssertionError("zoom break without a new bin")
        self.c = c
        if self.breaks_pending: self._flush()

    def _records(self, parts, lo, hi, zero_last):
        cols = [np.concatenate([p[k] for p in parts])[lo:hi] if len(parts) > 1 else parts[0][k][lo:hi] for k in range(8)]
        rec = np.empty(hi - lo, dtype=ZOOM_REC)
        rec["tid"] = cols[0]; rec["start"] = cols[1]; rec["end"] = cols[2]; rec["count"] = cols[3]
        rec["min"] = cols[4].astype(np.float32); rec["max"] = cols[5].astype(np.float32)
        s = cols[6].copy(); q = cols[7].copy()
        if zero_last: s[-1] = 0.0; q[-1] = 0.0                               # never-closed bin keeps sum = sumsq = 0 (libBigWig)
        rec["sum"] = s.astype(np.float32); rec["sumsq"] = q.astype(np.float32)
        return int(cols[0][0]), int(cols[0][-1]), int(cols[1][0]), int(cols[2][-1]), rec.tobytes()

    def _flush(self):
        """Finalize every buffer that ends at a pending break; keep the bins after the last break pending."""
        last_break = self.breaks_pending[-1]; parts = self.pending; lo_abs = self.pending_base
        bounds = [lo_abs] + self.breaks_pending
        for i in range(len(bounds) - 1):
            t0, t1, s0, e1, raw = self._records(parts, bounds[i] - lo_abs, bounds[i + 1] - lo_abs, zero_last=True)
            self.blocks_done.append((t0, t1, s0, e1, self.compress(raw)))
        rem_lo = last_break - lo_abs; total = self.nbins - lo_abs
        if total > rem_lo:
            cols = [(np.concatenate([p[k] for p in parts]) if len(parts) > 1 else parts[0][k])[rem_lo:total].copy() for k in range(8)]
            self.pending = [cols]; self.open = {"tid": int(cols[0][-1]), "s": int(cols[1][-1]), "arrays": cols, "idx": len(cols[0]) - 1}
        else:
            self.pending = []
            if self.open is not None: self.open = {"tid": self.open["tid"], "s": self.open["s"], "arrays": None, "idx": -1}
        self.pending_base = last_break; self.breaks_pending = []

    def blocks(self):
        out = list(self.blocks_done)
        if self.nbins > self.pending_base:
            t0, t1, s0, e1, raw = self._records(self.pending, 0, self.nbins - self.pending_base, zero_last=True)
            out.append((t0, t1, s0, e1, self.compress(raw)))
        return out


class NumpyBigWig:
    def __init__(self, chrom_sizes, max_zooms=10, compress_workers=0):
        """chrom_sizes: list of (name, length) in addHeader order (defines tids)."""
        self.chroms = [(str(c), int(l)) for c, l in chrom_sizes]; self.tid_of = {c: i for i, (c, _) in enumerate(self.chroms)}
        self.max_zooms = 10 if (max_zooms < 0 or max_zooms > 65535) else int(max_zooms)
        self.blocks = []                       # data blocks (tid, start, end, compressed bytes)
        self.run = None                        # open run: [tid, start, [value arrays], nvals]
        self.nbases = 0; self.sum_data = 0.0; self.sum_sq = 0.0; self.min_val = DBL_MAX; self.max_val = DBL_MIN
        self.zooms = None; self.levels = None
        self._last_tid = None; self._last_end = None; self._pending_runs = []
        self.ltype = None                      # 3 = span/step blocks (add_entries), 1 = interval blocks (add_intervals: the stock CLI's layout)
        self.pool = None                       # optional multiprocessing pool: zlib.compress of independent blocks in parallel (bytes identical)
        if compress_workers:
            from .hostres import cap; compress_workers = cap(compress_workers)     # capped by the container's CPU quota
            import multiprocessing as mp
            self.pool = mp.get_context("fork").Pool(compress_workers)
        self.ibuf = None                       # open type-1 block: [tid, start_first, list of (starts, ends, values) arrays, n]
        self._ilast_tid = None; self._ilast_start = None

    # ---- pyBigWig.addEntries(chrom, start, values=v, span=1, step=1) ----
    def add_entries(self, chrom, start, values):
        v = np.ascontiguousarray(values, dtype=np.float32); n = len(v)
        if n == 0: return
        assert self.ltype in (None, 3), "cannot mix add_entries and add_intervals"
        self.ltype = 3; tid = self.tid_of[chrom]; start = int(start)
        if self.run is not None and tid == self._last_tid and start == self._last_end:   # canAppend -> bwAppendIntervalSpanSteps
            self.run[2].append(v); self.run[3] += n
        else:                                                                             # bwAddIntervalSpanSteps: flush + new block
            self._close_run(); self.run = [tid, start, [v], n]
        self._last_tid = tid; self._last_end = start + n
        self._stats(v)

    # ---- pyBigWig.addEntries([chrom]*n, starts, ends=ends, values=v): the stock CLI's layout (bwAddIntervals / bwAppendIntervals) ----
    def add_intervals(self, chrom, starts, ends, values):
        """Unit intervals only (end == start + 1), as the stock writes them; per-entry (start, end, value) blocks of 2728 entries.
        pyBigWig appends when the chromosome equals the last one and starts[0] >= the last end; otherwise bwAddIntervals (flush)."""
        st = np.ascontiguousarray(starts, dtype=np.int64); en = np.ascontiguousarray(ends, dtype=np.int64); v = np.ascontiguousarray(values, dtype=np.float32)
        n = len(v)
        if n == 0: return
        assert np.all(en - st == 1), "bigwig_numpy.add_intervals: only unit intervals are supported"
        assert self.ltype in (None, 1), "cannot mix add_entries and add_intervals"
        self.ltype = 1; tid = self.tid_of[chrom]
        append = self.ibuf is not None and tid == self._ilast_tid and int(st[0]) >= self._ilast_start
        if not append:
            # bwAddIntervals: flush if >= 2726 entries are buffered (l+36 > bufSize), then flush on a chromosome change
            if self.ibuf is not None and (self.ibuf[3] >= IVALS_PER_BLOCK - 2 or tid != self.ibuf[0]): self._iflush()
            if self.ibuf is None: self.ibuf = [tid, int(st[0]), [], 0]
        # split into blocks of 2728 entries (an in-loop flush when the buffer holds 2728 and another entry arrives)
        pos = 0
        while pos < n:
            room = IVALS_PER_BLOCK - self.ibuf[3]
            if room <= 0:
                self._iflush(); self.ibuf = [tid, int(st[pos]), [], 0]; room = IVALS_PER_BLOCK
            k = min(room, n - pos)
            self.ibuf[2].append((st[pos:pos + k], en[pos:pos + k], v[pos:pos + k])); self.ibuf[3] += k; pos += k
        self._ilast_tid = tid; self._ilast_start = int(en[-1])
        self._stats(v)
        # zoom runs: contiguous bases in file order (the same state machine as the span/step layout)
        self._register_bases(tid, st, v)

    def _iflush(self):
        if self.ibuf is None: return
        tid, start0, parts, k = self.ibuf
        st = np.concatenate([p[0] for p in parts]); en = np.concatenate([p[1] for p in parts]); v = np.concatenate([p[2] for p in parts])
        rec = np.empty(k, dtype=[("s", "<u4"), ("e", "<u4"), ("v", "<f4")]); rec["s"] = st; rec["e"] = en; rec["v"] = v
        hdr = struct.pack("<IIIIIBBH", tid, start0, int(en[-1]), 0, 0, 1, 0, k)
        self.blocks.append((tid, start0, int(en[-1]), self._compress(hdr + rec.tobytes()))); self.ibuf = None

    def _register_bases(self, tid, st, v):
        """Group unit intervals into contiguous runs for the zoom levels (a run continues across calls when contiguous)."""
        # split this call's bases at gaps
        cuts = np.flatnonzero(np.diff(st) != 1) + 1; b0 = 0
        for c in list(cuts) + [len(st)]:
            seg_s = int(st[b0]); seg_v = v[b0:c]; seg_n = c - b0
            if self.run is not None and tid == self.run[0] and seg_s == self.run[1] + self.run[3]:
                self.run[2].append(seg_v); self.run[3] += seg_n
            else:
                self._close_zoom_run(); self.run = [tid, seg_s, [seg_v], seg_n]
            b0 = c

    def _close_zoom_run(self):
        if self.run is None: return
        tid, start, parts, n = self.run; v = parts[0] if len(parts) == 1 else np.concatenate(parts)
        self._pending_runs.append((tid, start, start + n, v)); self.run = None

    def _compress(self, raw):
        return self.pool.apply_async(zlib.compress, (raw, 6)) if self.pool is not None else zlib.compress(raw, 6)

    def _stats(self, v):
        v64 = v.astype(np.float64)
        pm = np.minimum.accumulate(np.concatenate(([self.min_val], v64)))[:-1]           # running min before each value
        is_min = v64 < pm
        if is_min.any(): self.min_val = float(v64[is_min].min())
        cand = v64[~is_min]
        if len(cand): self.max_val = max(self.max_val, float(cand.max()))
        self.nbases += len(v)
        self.sum_data = _seq_add(self.sum_data, v64); self.sum_sq = _seq_add(self.sum_sq, v64 * v64)

    def _close_run(self):
        if self.run is None: return
        tid, start, parts, n = self.run; v = parts[0] if len(parts) == 1 else np.concatenate(parts)
        pos = 0; bstart = start
        while True:
            if n - pos > VALS_PER_BLOCK: k = VALS_PER_BLOCK; end = bstart + k; more = True
            else: k = n - pos; end = bstart + k + 6; more = False
            hdr = struct.pack("<IIIIIBBH", tid, bstart, end, 1, 1, 3, 0, k)
            self.blocks.append((tid, bstart, end, self._compress(hdr + v[pos:pos + k].tobytes())))
            pos += k; bstart = end
            if not more: break
        self._pending_runs.append((tid, start, start + n, v)); self.run = None

    # ---- zoom levels, fed incrementally with closed runs ----
    def _init_zooms(self):
        mean_bin = 1 * 4                                                         # (uint32)(runningWidthSum/nEntries) * 4: every entry has span 1
        zoom = 10
        if mean_bin * 4 > zoom: zoom = 4 * mean_bin
        max_zoom = max(l for _, l in self.chroms)
        if zoom > max_zoom: zoom = max_zoom
        levels = []
        for _ in range(self.max_zooms):
            if zoom > max_zoom: break
            levels.append(zoom)
            if (2 ** 32 - 1) // 4 < zoom: break
            zoom *= 4
        self.levels = levels; self.zooms = [_ZoomLevel(z, self._compress) for z in levels]

    def feed_zooms(self):
        """Feed every closed run to the zoom levels (safe after each chunk: closed runs are final)."""
        if self.max_zooms == 0 or not self._pending_runs: return
        if self.zooms is None: self._init_zooms()
        runs = []; vs = []; pos = []; off = 0
        for tid, a, b, v in self._pending_runs:
            runs.append((tid, a, b, off)); vs.append(v); pos.append(np.arange(a, b, dtype=np.int64)); off += b - a
        V = np.concatenate(vs) if len(vs) > 1 else vs[0]; POS = np.concatenate(pos) if len(pos) > 1 else pos[0]
        V64 = V.astype(np.float64); V2 = V64 * V64
        for zl in self.zooms: zl.feed_runs(runs, V, V64, V2, POS)
        self._pending_runs = []

    # ---- close(): assemble the file bytes ----
    def close(self, path=None):
        if self.ltype == 1: self._iflush(); self._close_zoom_run()
        else: self._close_run()
        if self.max_zooms and self.blocks: self.feed_zooms()
        out = bytearray(struct.pack("<IH", BIGWIG_MAGIC, 4) + bytes(58))
        out += bytes(24 * self.max_zooms)
        summary_off = len(out); out += bytes(40); struct.pack_into("<Q", out, 0x2C, summary_off)
        ct_off = len(out); out += self._chrom_tree(); struct.pack_into("<Q", out, 0x8, ct_off)
        data_off = len(out); struct.pack_into("<Q", out, 0x10, data_off); out += bytes(8)
        entries = []
        for tid, s, e, cb in self.blocks:
            cb = cb.get() if hasattr(cb, "get") else cb
            off = len(out); out += cb; entries.append((tid, tid, s, e, off, len(cb)))
        struct.pack_into("<Q", out, data_off, len(self.blocks))
        struct.pack_into("<I", out, 0x34, BUF_SIZE)
        struct.pack_into("<Qdddd", out, summary_off, self.nbases, self.min_val, self.max_val, self.sum_data, self.sum_sq)
        if self.blocks:
            struct.pack_into("<Q", out, 0x18, len(out)); out += _index_bytes(entries, 1, len(out), 0)[0]
        actual = 0
        if self.max_zooms and self.blocks:
            hdrs = []; prev_extra = None; carry = 0
            for i, zl in enumerate(self.zooms):
                if i and zl.n_buffers_extra == prev_extra: break
                prev_extra = zl.n_buffers_extra; actual += 1
                zblocks = zl.blocks(); d_off = len(out); out += struct.pack("<I", len(zblocks)); zentries = []
                for t0, t1, s0, e1, cb in zblocks:
                    cb = cb.get() if hasattr(cb, "get") else cb
                    off = len(out); out += cb; zentries.append((t0, t1, s0, e1, off, len(cb)))
                i_off = len(out); ib, carry = _index_bytes(zentries, ITEMS_PER_SLOT_ZOOM, len(out), carry); out += ib; hdrs.append((zl.Z, d_off, i_off))
            for i, (Z, d_off, i_off) in enumerate(hdrs): struct.pack_into("<IIQQ", out, 0x40 + 24 * i, Z, 0, d_off, i_off)
        struct.pack_into("<H", out, 0x6, actual)
        out += struct.pack("<I", BIGWIG_MAGIC)
        if path is not None:
            with open(path, "wb") as f: f.write(out)
        if self.pool is not None: self.pool.close(); self.pool.join(); self.pool = None
        return bytes(out)

    def _chrom_tree(self):
        n = len(self.chroms); nper = min(n, 0x7FFF); key = max(len(c) for c, _ in self.chroms)
        nblocks = n // nper + (1 if n % nper else 0)
        b = bytearray(struct.pack("<IIIIQQ", CIRTREE_MAGIC, nper, key, 8, n, 0))
        if nblocks > 1:
            b += struct.pack("<BBH", 0, 0, nblocks)
            non_leaf_end = len(b) + nper * (key + 8); leaf_size = nper * (key + 8) + 4
            for i in range(nblocks): b += self.chroms[i * nper][0].encode()[:key].ljust(key, b"\0") + struct.pack("<Q", non_leaf_end + i * leaf_size)
            for i in range(nblocks, nper): b += bytes(key) + struct.pack("<Q", 0)
        j = 0
        for i in range(nblocks):
            cnt = n - j if n - j < nper else nper
            b += struct.pack("<BBH", 1, 0, cnt)
            for k in range(nper):
                if j >= n: b += bytes(key) + struct.pack("<Q", 0)
                else: b += self.chroms[j][0].encode()[:key].ljust(key, b"\0") + struct.pack("<II", j, self.chroms[j][1]); j += 1
        return bytes(b)


class _Node:
    __slots__ = ("leaf", "ch", "cs", "bs", "ce", "be", "pos")


def _index_bytes(entries, items_per_slot, file_off, idx_size_in=0):
    """libBigWig writeIndex: entries = (tid0, tid1, start, end, offset, size) per block in order; file_off = absolute position of
    the index section (child pointers are absolute). idx_size_in: the running idxSize variable of writeZoomLevels (declared once
    for all zoom levels: a multi-leaf level ADDS to it, a single-leaf level assigns it) -> returns (bytes, idx_size_out)."""
    leaves = [entries[i:i + BLOCK_SIZE] for i in range(0, len(entries), BLOCK_SIZE)]
    idx_size = idx_size_in
    def make_leaf(ents):
        nd = _Node(); nd.leaf = True; nd.ch = ents; nd.cs, nd.bs = ents[0][0], ents[0][2]; nd.ce, nd.be = ents[-1][1], ents[-1][3]; return nd
    it = iter(leaves)
    def add_leaves(to_process):
        nonlocal idx_size
        nd = _Node(); nd.leaf = False; nd.ch = []
        if to_process <= BLOCK_SIZE:
            for _ in range(to_process):
                lf = make_leaf(next(it)); nd.ch.append(lf); idx_size += 4 + 32 * len(lf.ch)
        else:
            i = 0
            while i < BLOCK_SIZE:
                foo = math.ceil(to_process / (BLOCK_SIZE - i))
                try: nd.ch.append(add_leaves(foo))
                except StopIteration: break
                to_process -= foo; i += 1
                if to_process <= 0: break
        nd.cs, nd.bs = nd.ch[0].cs, nd.ch[0].bs; nd.ce, nd.be = nd.ch[-1].ce, nd.ch[-1].be
        idx_size += 4 + 24 * len(nd.ch); return nd
    if len(leaves) == 1:
        root = make_leaf(leaves[0]); idx_size = 4 + 24 * len(leaves[0])                    # as coded in writeIndex
    else:
        root = add_leaves(len(leaves))
    hdr = struct.pack("<IIQIIIIQII", IDX_MAGIC, BLOCK_SIZE, len(entries), root.cs, root.bs, root.ce, root.be, idx_size, items_per_slot, 0)
    # layout: root, then its children in order, then their children (level by level), each node = 4 + 24*n (non-leaf) / 4 + 32*n (leaf)
    pos = file_off + len(hdr); levels = [[root]]
    root.pos = pos; pos += 4 + (32 if root.leaf else 24) * len(root.ch)
    while not all(nd.leaf for nd in levels[-1]):
        nxt = [c for nd in levels[-1] if not nd.leaf for c in nd.ch]
        for c in nxt: c.pos = pos; pos += 4 + (32 if c.leaf else 24) * len(c.ch)
        levels.append(nxt)
    b = bytearray(hdr)
    for lvl in levels:
        for nd in lvl:
            b += struct.pack("<BBH", 1 if nd.leaf else 0, 0, len(nd.ch))
            if nd.leaf:
                for (t0, t1, s, e, off, sz) in nd.ch: b += struct.pack("<IIIIQQ", t0, s, t1, e, off, sz)
            else:
                for c in nd.ch: b += struct.pack("<IIIIQ", c.cs, c.bs, c.ce, c.be, c.pos)
    return bytes(b), idx_size


def write_bigwig_numpy(path, chrom_sizes, calls, max_zooms=10, feed_every=0):
    """Convenience: calls = iterable of (chrom, start, values) exactly as the pyBigWig addEntries sequence."""
    w = NumpyBigWig(chrom_sizes, max_zooms)
    for i, (c, s, v) in enumerate(calls):
        w.add_entries(c, s, v)
        if feed_every and (i + 1) % feed_every == 0: w.feed_zooms()
    return w.close(path)
