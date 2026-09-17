"""THE METRICS STAGE OVERLAPPED WITH THE FORWARD — the same functions, the same per-row arithmetic and the same bytes as the serial stage.
The serial metrics stage runs AFTER the forward: get_cts, softmax, the h5 deflate, profile metrics, counts metrics, PNGs, json. Everything
in it is per-row work except (a) the RNG stream of the shuffled baseline (np.random.permutation per region IN FILE ORDER — it depends only on
the OBSERVED counts, so it can run ahead of the forward entirely) and (b) the HDF5 chunk layout (each chunk's deflate blob depends only on its
own rows; the write ORDER is h5_fast's cache emulation at assembly). So one worker thread does, during the forward:
  phase A (observed only, file order, blocks of `block` rows): get_cts rows -> P, profile_prob, max_jsd, the permutations (RNG in file order)
           -> jsd_rnd; keeps P (float64) for jsd_pw;
  phase B (arrivals: file rows + their logits, any order): probs rows = softmax(logits) (float32), prof64 rows, jsd_pw rows, and every
           HDF5 chunk whose rows are all present is deflated (h5_fast._chunk_blob on the float64 array) in a thread pool.
finish(): joins, assembles predictions.h5 from the blobs (h5_fast.write_predictions_h5py_fast with profs_blobs), then the stock's own
counts_metrics / plot_histogram (on the spawned png_workers helpers) / json. The per-row values are identical to the serial form because every
reduction is over a contiguous row. No in-job self-check, no fallback."""
import os, time, json, threading, queue, numpy as np
from concurrent.futures import ThreadPoolExecutor
def jensenshannon(*a, **k):
    """LAZY — scipy.spatial.distance may be importing on the preimport thread; resolved at first use, then rebound to the real function."""
    global jensenshannon
    from scipy.spatial.distance import jensenshannon as _js
    jensenshannon = _js; return _js(*a, **k)
from chrombpnet_fastkit import softmax
from . import h5_fast, png_workers, bigwig_reader



def _counts_png_worker(labels, preds, output_prefix, title, out_npy):
    """The stock's OWN counts_metrics (scipy spearman/pearson/mse + the density-scatter PNG via pyplot) run in a separate worker PROCESS —
        pyplot's state is process-local there; the main process draws its own histogram itself. Exact iff the PNG bytes reproduce. The three
        numbers return as float64 via .npy (no text round-trip). MetricsOverlap.finish uses the spawned png_workers helpers instead."""
    import numpy as _np, os as _os
    import chrombpnet.training.metrics as _metrics
    sp, pe, mse = _metrics.counts_metrics(labels, preds, output_prefix, title)
    _np.save(out_npy + ".tmp.npy", _np.array([sp, pe, mse], dtype=_np.float64)); _os.replace(out_npy + ".tmp.npy", out_npy)

class MetricsOverlap:
    def __init__(self, bigwig, regions_df, outputlen, output_prefix, h5_workers=8, block=4096, pseudocount=0.001, timings=None):
        from .hostres import cap, pool_form
        self.T = {} if timings is None else timings
        self.df = regions_df; self.N = len(regions_df); self.L = int(outputlen); self.prefix = output_prefix; self.block = int(block); self.pc = pseudocount
        self.h5_workers = cap(h5_workers); self.T["pool_form"] = pool_form(self.h5_workers); self.T["form"] = "overlap (metrics_overlap.MetricsOverlap)"
        self.rd = bigwig_reader.NumpyBigWigReader(bigwig, self_check=False)
        N, L = self.N, self.L
        self.obs = np.empty((N, L), dtype=np.float64); self.P = np.empty((N, L), dtype=np.float64)
        self.max_jsd = np.empty(N); self.jsd_rnd = np.empty(N); self.jsd_pw = np.full(N, np.nan)
        self.probs = np.empty((N, L), dtype=np.float32); self.prof64 = np.empty((N, L), dtype=np.float64)
        self.ch = h5_fast._guess_chunks((N, L), np.float64); self.ncol = -(-L // self.ch[1]); self.nrow = -(-N // self.ch[0])
        self.rows_done = np.zeros(self.nrow, dtype=np.int64); self.chunk_rows = np.array([min(self.ch[0], N - r * self.ch[0]) for r in range(self.nrow)])
        self.blobs = {}; self.futs = {}; self.pool = ThreadPoolExecutor(self.h5_workers)
        self.q = queue.Queue(); self.err = None; self.phaseA_done = threading.Event(); self.a_done = 0; self._pending_pw = []; self.t = {"obs_phase": 0.0, "arrivals": 0.0, "arrival_rows": 0, "chunks_deflated_during_forward": 0}
        self.thread = threading.Thread(target=self._run, daemon=True); self.t0 = time.time(); self.thread.start()

    # ---- producer side (called from predict_pipelined after each forward chunk; file rows + their logits, any order)
    def add(self, file_rows, logits):
        self.q.put((np.asarray(file_rows, dtype=np.int64), np.ascontiguousarray(logits, dtype=np.float32)))

    # ---- worker
    def _run(self):
        try:
            t0 = time.time(); uniform = np.ones(self.L) * (1.0 / self.L)
            for a in range(0, self.N, self.block):                                       # PHASE A: observed counts in FILE order (RNG in file order)
                b = min(a + self.block, self.N)
                obs = bigwig_reader.get_cts_reader(self.df.iloc[a:b], self.rd, self.L)       # the serial stage's get_cts call, on the file-order sub-frame
                self.obs[a:b] = obs
                denom = self.pc + np.nansum(obs, axis=1, keepdims=True); P = obs / denom; self.P[a:b] = P
                profile_prob = obs / np.sum(obs, axis=1, keepdims=True)
                self.max_jsd[a:b] = jensenshannon(profile_prob, np.broadcast_to(uniform, profile_prob.shape), axis=1)
                shuffled = np.empty_like(obs)
                for i in range(b - a): shuffled[i] = np.random.permutation(obs[i, :])       # the stock's RNG consumption, in the stock's (file) order
                S = shuffled / (self.pc + np.nansum(shuffled, axis=1, keepdims=True))
                self.jsd_rnd[a:b] = jensenshannon(P, S, axis=1); self.a_done = b
                self._drain(block=False)                                                   # interleave arrivals so the h5 chunks deflate early
            self.t["obs_phase"] = time.time() - t0; self.phaseA_done.set()
            pend = self._pending_pw; self._pending_pw = []
            for rows in pend: self.jsd_pw[rows] = jensenshannon(self.P[rows], self.probs[rows], axis=1)      # arrivals that preceded their phase-A rows
            while True:                                                                    # PHASE B: arrivals until the sentinel
                item = self.q.get()
                if item is None: break
                self._arrival(*item)
        except Exception as e:
            self.err = e

    def _drain(self, block):
        while True:
            try: item = self.q.get(block=False)
            except queue.Empty: return
            if item is None: self.q.put(None); return
            self._arrival(*item)

    def _arrival(self, rows, lg):
        t1 = time.time(); pr = softmax(lg)                                                 # float32 rows, per-row reductions over the contiguous row
        self.probs[rows] = pr; self.prof64[rows] = pr.astype(np.float64)
        ready = rows < self.a_done                                                         # P rows exist for the file rows phase A has passed
        if ready.all(): self.jsd_pw[rows] = jensenshannon(self.P[rows], pr, axis=1)
        else:
            if ready.any(): self.jsd_pw[rows[ready]] = jensenshannon(self.P[rows[ready]], pr[ready], axis=1)
            self._pending_pw.append(rows[~ready])
        r_idx = rows // self.ch[0]; cnt = np.bincount(r_idx, minlength=self.nrow); self.rows_done += cnt
        for r in np.unique(r_idx):
            if self.rows_done[r] == self.chunk_rows[r]:
                for c in range(self.ncol):
                    idx = r * self.ncol + c; self.futs[idx] = self.pool.submit(h5_fast._chunk_blob, self.prof64, self.ch, r, c, 4)
                self.t["chunks_deflated_during_forward"] += self.ncol
        self.t["arrivals"] += time.time() - t1; self.t["arrival_rows"] += int(len(rows))

    # ---- finish (after the forward): assemble the h5, the stock's counts metrics / PNGs / json
    def finish(self, pred_logcts_file, coordinates, used_mask=None):
        _t_entry = time.time()   # the finish stage's own clock (finish() entry -> return; the metrics import below is a finish-stage cost)
        import chrombpnet.training.metrics as metrics
        t = time.time(); self.q.put(None); self.thread.join()
        if self.err is not None: raise RuntimeError(f"[metrics_overlap] worker failed: {self.err!r}")
        if used_mask is not None and not bool(np.all(used_mask)): raise RuntimeError("[metrics_overlap] every region must be used (the frame is the full regions file) — refused")
        for rows in self._pending_pw: self.jsd_pw[rows] = jensenshannon(self.P[rows], self.probs[rows], axis=1)
        if int(self.rows_done.sum()) != self.N: raise RuntimeError(f"[metrics_overlap] rows arrived {int(self.rows_done.sum())} of {self.N}")   # every logits row arrived (a NaN JSD is a zero observed window's value, as in stock — not a missing row)
        for idx, f in self.futs.items(): self.blobs[idx] = f.result()
        self.pool.shutdown(); self.T["join_wait_s"] = round(time.time() - t, 3); self.T["worker"] = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in self.t.items()}; t = time.time()
        counts_sum_predictions = np.squeeze(np.asarray(pred_logcts_file, dtype=np.float32))
        # the finish stage as three concurrent tasks — (A) the counts-scatter PNG on its helper (the stock's own counts_metrics), (B) the
        # jsd-histogram PNG on its helper (the stock's own plot_histogram), (C) the h5 in this process
        # (the fast writer in its byte-exact regime, N >= H5_FAST_MIN_N; the stock's own writer below the regime or on a
        # mismatch). Every task is the serial finish's own call on the same arrays; the tasks touch disjoint files.
        true_counts_sum = np.log(np.sum(self.obs, axis=-1) + 1)
        _n1 = int(np.size(counts_sum_predictions)) < 2
        png_proc = jsd_proc = None; _png_npy = self.prefix + "_counts_metrics.fork.npy"; t_png0 = time.time()
        if _n1:
            spearman_cor = pearson_cor = mse = None; self.T["counts_metrics_png"] = 0.0; self.T["n1_form"] = "N < 2: the counts correlations are undefined (scipy pearsonr needs >= 2 rows) -> null with this reason; the counts-scatter and jsd-histogram PNGs skipped (v0.12.33)"
        else:
            # the two PNG tasks on the helpers png_workers SPAWNED at process start (fresh interpreters: never a fork of this threaded process —
            # a lock held by another thread at a fork stays locked in the child forever); the same stock calls on the same arrays
            _W = png_workers.require()
            _W["counts"].submit((true_counts_sum, counts_sum_predictions, self.prefix, "All regions provided"))
            _W["jsd"].submit((self.jsd_pw, self.jsd_rnd, self.prefix, "All regions provided"))
            png_proc = _W["counts"]; jsd_proc = _W["jsd"]
            self.T["counts_png_forked"] = "helper (spawned at process start)"; self.T["jsd_png_forked"] = "helper (spawned at process start)"; spearman_cor = pearson_cor = mse = None
        h5_fast.write_predictions_h5py_fast(self.prefix, self.probs, counts_sum_predictions, coordinates, workers=self.h5_workers, profs_blobs=self.blobs)
        self.T["h5_form"] = h5_fast.LAST_FORM.get("form")
        self.T["h5_assemble"] = round(time.time() - t, 3); t = time.time()
        min_jsd = 0.0
        def norm(val):
            ret = (val - self.max_jsd) / (min_jsd - self.max_jsd)
            return np.where(ret < 0, 0.0, np.where(ret > 1, 1.0, ret))
        jsd_norm = norm(self.jsd_pw)
        metrics_dictionary = {"counts_metrics": {"regions": {"spearmanr": spearman_cor, "pearsonr": pearson_cor, "mse": mse}},
                              "profile_metrics": {"regions": {"median_jsd": np.nanmedian(self.jsd_pw), "median_norm_jsd": np.nanmedian(jsd_norm)}}}
        if _n1: metrics_dictionary["n1_form"] = self.T["n1_form"]
        if jsd_proc is not None:
            jsd_proc.result(900)
            if not os.path.exists(self.prefix + ".profile_jsd.png"): raise RuntimeError("[metrics_overlap] jsd-PNG helper returned without the PNG — refused (no fallback)")
        self.T["jsd_histogram_png"] = round(time.time() - t, 3); t = time.time()
        if png_proc is not None:
            spearman_cor, pearson_cor, mse = [float(x) for x in png_proc.result(900)]
            if not os.path.exists(self.prefix + ".counts_pearsonr.png"): raise RuntimeError("[metrics_overlap] counts-PNG helper returned without the PNG — refused (no fallback)")
            metrics_dictionary["counts_metrics"]["regions"] = {"spearmanr": spearman_cor, "pearsonr": pearson_cor, "mse": mse}
            self.T["counts_metrics_png"] = 0.0; self.T["counts_png_join_wait"] = round(time.time() - t, 3); self.T["counts_png_worker_wall"] = round(time.time() - t_png0, 3); t = time.time()
        with open(self.prefix + '_metrics.json', 'w') as fp:
            json.dump(metrics_dictionary, fp, indent=4)
        self.T["json"] = round(time.time() - t, 3); self.T.update(_finish_stage_stamp(self.T, _t_entry, self.t0))
        return self.T


def _finish_stage_stamp(T, t_entry, t0):
    """The two clocks of the finish stage: finish_stage_s = finish() entry -> return (the named sub-stages: join_wait, h5_assemble, the counts PNG,
        the jsd PNG, json); overlap_lifetime_s = the overlap's construction -> the end of finish (it spans the whole of stage 1 — the load, the
        warm-up, the forward, the writer — plus the finish; the note string keeps its older name for readers of earlier records)."""
    now = time.time()
    return {"finish_stage_s": round(now - t_entry, 3), "overlap_lifetime_s": round(now - t0, 3),
            "finish_stage_note": "finish_stage_s = finish() entry -> return (= the named sub-stages); overlap_lifetime_s = construction -> the end of finish (the old 'finish_total', a misnomer)"}
