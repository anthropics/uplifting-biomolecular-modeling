"""numpy bigWig READER for data_utils.get_cts (the stock calls pyBigWig.values() once per region in a pandas iterrows loop). The reader parses
the header, the chromosome B+ tree and the full-resolution R-tree once (numpy struct arrays), inflates each data block that overlaps a
requested region ONCE, and fills the per-base values of all regions of a chromosome from the (region, interval) overlap pairs in one
vectorised pass (fill="pairs"; fill="searchsorted" = the per-base lookup form, same values; a threaded inflate is slower under the GIL, hence
inflate_workers=1) — the same semantics as libBigWig's bwGetValues (a base inside an interval [start, end) takes that interval's float32
value, an uncovered base is NaN; intervals in file order, a later overlapping interval wins) and the same float64 result as
np.array(bw.values(...)) (float32 -> float64 is exact). self_check=True compares 64 random regions with pyBigWig at open and raises on any
difference; the kit's metrics stage opens the reader with self_check=False."""
import struct, zlib, numpy as np

_HDR = struct.Struct("<IHHQQQHHQQIQ")            # magic, version, zoomLevels, chromTreeOffset, fullDataOffset, fullIndexOffset, fieldCount, definedFieldCount, autoSqlOffset, totalSummaryOffset, uncompressBufSize, reserved
_BIGWIG_MAGIC = 0x888FFC26; _CHROM_MAGIC = 0x78CA8C91; _RTREE_MAGIC = 0x2468ACE0
_RLEAF = np.dtype([("schrom", "<u4"), ("sbase", "<u4"), ("echrom", "<u4"), ("ebase", "<u4"), ("offset", "<u8"), ("size", "<u8")])
_RNODE = np.dtype([("schrom", "<u4"), ("sbase", "<u4"), ("echrom", "<u4"), ("ebase", "<u4"), ("child", "<u8")])
_BEDGRAPH = np.dtype([("start", "<u4"), ("end", "<u4"), ("value", "<f4")]); _VARSTEP = np.dtype([("start", "<u4"), ("value", "<f4")])


class NumpyBigWigReader:
    def __init__(self, path, self_check=True, inflate_workers=1, fill="pairs"):
        import threading
        self.f = open(path, "rb"); self._lock = threading.Lock(); self.inflate_workers = int(inflate_workers); self.fill = fill; h = _HDR.unpack(self.f.read(_HDR.size))
        if h[0] != _BIGWIG_MAGIC: raise ValueError("not a little-endian bigWig: %s" % path)
        self.compressed = h[10] > 0; self.chrom_id = {}; self.chrom_len = {}
        self._read_chrom_tree(h[3]); self.leaves = self._read_rtree(h[5])
        self._blocks = {}                                            # chromId -> (sorted leaf array, [parsed block cache])
        if self_check: self._self_check(path)

    # ---- chromosome B+ tree
    def _read_chrom_tree(self, off):
        self.f.seek(off); magic, block_size, key_size, val_size, item_count, _ = struct.unpack("<IIIIQQ", self.f.read(32))
        if magic != _CHROM_MAGIC: raise ValueError("bad chromosome tree magic")
        def node(pos):
            self.f.seek(pos); is_leaf, _, count = struct.unpack("<BBH", self.f.read(4))
            if is_leaf:
                raw = self.f.read(count * (key_size + 8))
                for i in range(count):
                    rec = raw[i * (key_size + 8):(i + 1) * (key_size + 8)]; name = rec[:key_size].split(b"\0", 1)[0].decode(); cid, clen = struct.unpack("<II", rec[key_size:])
                    self.chrom_id[name] = cid; self.chrom_len[name] = clen
            else:
                raw = self.f.read(count * (key_size + 8)); children = [struct.unpack("<Q", raw[i * (key_size + 8) + key_size:(i + 1) * (key_size + 8)])[0] for i in range(count)]
                for c in children: node(c)
        node(off + 32)

    # ---- full-resolution R-tree -> all leaf items
    def _read_rtree(self, off):
        self.f.seek(off); magic, block_size, item_count, *_ = struct.unpack("<IIQIIIIQII", self.f.read(48))
        if magic != _RTREE_MAGIC: raise ValueError("bad R-tree magic")
        leaves = []
        def node(pos):
            self.f.seek(pos); is_leaf, _, count = struct.unpack("<BBH", self.f.read(4))
            if is_leaf: leaves.append(np.frombuffer(self.f.read(count * _RLEAF.itemsize), dtype=_RLEAF))
            else:
                kids = np.frombuffer(self.f.read(count * _RNODE.itemsize), dtype=_RNODE)["child"].copy()
                for c in kids: node(int(c))
        node(off + 48); return np.concatenate(leaves) if leaves else np.zeros(0, dtype=_RLEAF)

    def _chrom_blocks(self, cid):
        if cid not in self._blocks:
            m = (self.leaves["schrom"] == cid) | (self.leaves["echrom"] == cid); L = self.leaves[m]
            L = L[np.argsort(L["offset"], kind="stable")]; self._blocks[cid] = (L, [None] * len(L))
        return self._blocks[cid]

    def _parse_block(self, cid, L, k, cache):
        if cache[k] is None:
            with self._lock:
                self.f.seek(int(L["offset"][k])); raw = self.f.read(int(L["size"][k]))
            if self.compressed: raw = zlib.decompress(raw)
            bchrom, bstart, bend, step, span, btype, _, n = struct.unpack("<IIIIIBBH", raw[:24]); body = raw[24:]
            if bchrom != cid: s = e = np.zeros(0, np.int64); v = np.zeros(0, np.float32)
            elif btype == 1: it = np.frombuffer(body, dtype=_BEDGRAPH, count=n); s, e, v = it["start"].astype(np.int64), it["end"].astype(np.int64), it["value"]
            elif btype == 2: it = np.frombuffer(body, dtype=_VARSTEP, count=n); s = it["start"].astype(np.int64); e = s + span; v = it["value"]
            elif btype == 3: v = np.frombuffer(body, dtype="<f4", count=n); s = bstart + np.arange(n, dtype=np.int64) * step; e = s + span
            else: raise ValueError("unknown block type %d" % btype)
            cache[k] = (s, e, v)
        return cache[k]

    def values_many(self, chrom, starts, width):
        """float64 (len(starts), width): values(chrom, s, s+width) for each s, NaN where uncovered (pyBigWig semantics)."""
        cid = self.chrom_id[chrom]; L, cache = self._chrom_blocks(cid); starts = np.asarray(starts, dtype=np.int64); ends = starts + width
        if starts.size and (starts.min() < 0 or ends.max() > self.chrom_len[chrom]): raise ValueError("region outside %s" % chrom)
        # blocks overlapping any region: block [sbase, ebase) vs region [s, e)
        bs = L["sbase"].astype(np.int64); be = L["ebase"].astype(np.int64); bs = np.where(L["schrom"] < cid, 0, bs); be = np.where(L["echrom"] > cid, self.chrom_len[chrom], be)
        need = np.zeros(len(L), dtype=bool)
        lo = np.searchsorted(be, starts, side="right"); hi = np.searchsorted(bs, ends, side="left")    # blocks with be > s and bs < e
        for a, b in zip(lo, hi): need[a:b] = True
        ks = np.where(need)[0]
        if self.inflate_workers > 1 and len(ks) > 64:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(self.inflate_workers) as ex: parts = list(ex.map(lambda k: self._parse_block(cid, L, int(k), cache), ks))
        else:
            parts = [self._parse_block(cid, L, int(k), cache) for k in ks]
        out = np.full((len(starts), width), np.nan)
        if not parts: return out
        S = np.concatenate([p[0] for p in parts]); E = np.concatenate([p[1] for p in parts]); V = np.concatenate([p[2] for p in parts]).astype(np.float64)
        if len(S) > 1 and not np.all(S[1:] >= E[:-1]):            # overlapping intervals: fall back to in-order fill (later wins), region by region
            for i, (s0, e0) in enumerate(zip(starts, ends)):
                for s, e, v in zip(S, E, V):
                    if e > s0 and s < e0: out[i, max(s, s0) - s0:min(e, e0) - s0] = v
            return out
        if self.fill == "pairs":
            # (region, interval) pairs: intervals with E > s and S < e, per region a contiguous index range [lo, hi)
            lo = np.searchsorted(E, starts, side="right"); hi = np.searchsorted(S, ends, side="left"); cnt = np.maximum(hi - lo, 0)
            P = int(cnt.sum())
            if P == 0: return out
            reg = np.repeat(np.arange(len(starts)), cnt); first = np.repeat(np.cumsum(cnt) - cnt, cnt); k = np.repeat(lo, cnt) + (np.arange(P) - first)
            cs = np.maximum(S[k], starts[reg]); ce = np.minimum(E[k], ends[reg]); ln = ce - cs; keep = ln > 0
            reg, k, cs, ln = reg[keep], k[keep], cs[keep], ln[keep]; T = int(ln.sum())
            if T == 0: return out
            off = reg * width + (cs - starts[reg])                                   # flat output offset of each run
            idx = np.repeat(off, ln) + (np.arange(T) - np.repeat(np.cumsum(ln) - ln, ln))
            flat = out.reshape(-1); flat[idx] = np.repeat(V[k], ln); return out
        pos = (starts[:, None] + np.arange(width, dtype=np.int64)[None, :]).ravel()
        k = np.searchsorted(S, pos, side="right") - 1; ok = (k >= 0)
        kk = np.where(ok, k, 0); ok &= (E[kk] > pos)
        vals = np.where(ok, V[kk], np.nan); return vals.reshape(len(starts), width)

    def values(self, chrom, start, end):
        return self.values_many(chrom, [start], end - start)[0]

    def _self_check(self, path, n=64, width=1000):
        import pyBigWig
        bw = pyBigWig.open(path); rng = np.random.default_rng(20260825); chroms = [c for c in self.chrom_id if self.chrom_len[c] > 10 * width]
        for i in range(n):
            c = chroms[rng.integers(len(chroms))]; s = int(rng.integers(0, self.chrom_len[c] - width))
            ref = np.array(bw.values(c, s, s + width)); mine = self.values(c, s, s + width)
            from compare.bit_equal import bit_equal   # bit_equal is not part of this package (the tree that provides `compare` must be on sys.path); ImportError = refusal, no fallback
            if not bit_equal(np.ascontiguousarray(ref, dtype=np.float64), np.ascontiguousarray(mine, dtype=np.float64)): raise RuntimeError(f"[bigwig_reader] self-check FAILED (compare.bit_equal) at {c}:{s} (refused, no silent fallback)")
        bw.close()


def get_cts_reader(peaks_df, reader, width):
    """data_utils.get_cts with the numpy reader: regions grouped per chromosome, values evaluated in one vectorised pass per chromosome,
    rows returned in the regions' file order; np.nan_to_num as the stock."""
    chrs = peaks_df["chr"].values.astype(str); starts = peaks_df["start"].values.astype(np.int64) + peaks_df["summit"].values.astype(np.int64) - width // 2
    out = np.empty((len(chrs), width), dtype=np.float64)
    for c in np.unique(chrs):
        idx = np.where(chrs == c)[0]; out[idx] = reader.values_many(c, starts[idx], width)
    return np.nan_to_num(out)
