"""The per-variant HOST POST of ``borzoi_sad.py`` (untransform -> strand collapse -> write_snp_len statistics) computed per
COLUMN CHUNK in a thread pool — the stock functions applied to column subsets, every byte identical by construction — and, in the
pipelined form the entry uses, run BEHIND the next variant's one-hot and forward. The statistics writer is ``kitlib/writer.py``'s
(one writer, ``exact``: the stock's expressions, only the requested passes).

Why exact (borzoi_sad.py:273-297 + write_snp_len :408-523; baskerville dataset.py untransform_preds / untransform_preds1):
  * untransform is elementwise per column (scale / clip_soft / sum_stat are per-target columns) -> the stock function on the
    column subset ``preds[:, ilo:ihi]`` with ``targets_df.iloc[ilo:ihi]`` performs the identical operation on every element;
  * the strand collapse ``preds * strand_transform`` (dense (L, T) x CSR (T, Ts)) maps a CONTIGUOUS input-column range onto a
    contiguous output-column range (targets_prep_strand pairs adjacent rows; the transform's row->column map is monotone —
    asserted by ``chunk_plan``); the sub-product ``preds[:, ilo:ihi] * strand_transform[ilo:ihi, lo:hi]`` runs the same
    scipy kernel on the same nnz per output element (<= 2 terms, CSR column order preserved) and returns the same
    Fortran-ordered layout the stock gets, so
  * every statistic in write_snp_len (elementwise ops + per-column ``sum(axis=0)`` / percentile / argmax over the contiguous
    column) sees the identical column bytes in the identical layout -> identical reduction order -> identical float16 rows.

Why chunking cannot move a byte through numpy's SIMD loops: a numpy ufunc over a contiguous buffer of n elements runs its vector
kernel on the first n - (n mod W) elements and a scalar tail on the rest (W = the dispatched vector width, at most 8 float64 on
x86_64). Chunking changes n, so it could move elements between the two paths. That cannot change a byte here because (i) every
transcendental of the post (log2, the ** (4/3) pow) runs on a CONTIGUOUS buffer of L * w elements with L = 6144 = 16 * 384 -> n is a
multiple of every vector width for every chunk width w -> no element is ever in a tail, in the stock or the chunk (the post classes
refuse any L not divisible by 16 and the entry script then runs the stock path for that variant); (ii) every op with a broadcast
(1, w) operand (/ scale, * scale, > cs, the unclip add / square / where) and every abs / sqrt / square is IEEE correctly rounded per
element in both paths (numpy never fuses or reassociates elementwise loops), so the path does not matter; (iii) the per-column
reductions (sum(axis=0), percentile, argmax) act on a contiguous column of exactly L elements in both forms (the F layout the scipy
product returns), i.e. numpy's pairwise_sum over the same n with the same data. The stock's own transcendental bits depend on the
host CPU (numpy dispatches its kernels by instruction set); the kit's equal the stock's on the same host by (i)-(iii), and under the
``--det`` recipe (NPY_DISABLE_CPU_FEATURES pinned) both are byte-stable across hosts as well.

THE CHUNK WIDTH IS A PAGE-FAULT LEVER, INDEPENDENT OF THE POOL: every float64 temporary of the post (``preds / scale``, the unclip,
``* scale``, the collapse, log2, the squares) is a fresh numpy allocation; above glibc's dynamic mmap threshold
(DEFAULT_MMAP_THRESHOLD_MAX, 32 MiB on 64-bit) each is an mmap that the kernel zero-fills page by page and munmaps on free, so the
stock's single-thread post — one (L, T) float64 array per temporary, far above the threshold — re-faults every byte of every temporary
on every variant and spends a large share of its time in the kernel. The chunk count is therefore chosen from a TEMPORARY BUDGET
(``TEMP_BUDGET_BYTES``: w_max = budget / (8 * L) columns) so that each chunk's temporaries are served from the per-thread malloc
arenas, and is >= the pool size; the pool is sized from the cores the machine DELIVERS (``deliverable_cores`` = the scheduler affinity
count capped by the cgroup quota — a container runtime's cpu=8 request can advertise a larger cfs quota while 8 CPUs are schedulable —
refined by the throughput probe below). Exactness is unchanged by the width (the statement above holds for every w).
"""
from __future__ import annotations

import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

try:                                                   # the writer (kitlib/writer.py: exact — the stock's bits, only the requested passes)
    from . import writer as _writer
except ImportError:                                    # loaded as a bare module (the tests' spec_from_file_location): the sibling file by path
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("kitlib_writer_v16", os.path.join(os.path.dirname(os.path.abspath(__file__)), "writer.py"))
    _writer = _ilu.module_from_spec(_spec); _spec.loader.exec_module(_writer)

#: statistics write_snp_len can emit per column (everything but the full-length REF / ALT arrays, which stay on the stock path)
PER_COLUMN_STATS = ("SAD", "SADlog", "logSAD", "sqrtSAD", "SAX", "D1", "logD1", "sqrtD1", "D2", "logD2", "sqrtD2", "JS", "logJS")


def cgroup_cpu_quota() -> dict:
    """The process's CPU quota: cgroup v2 cpu.max -> v1 cfs quota -> sched affinity -> os.cpu_count()."""
    n_host = os.cpu_count() or 1
    try:
        v = open("/sys/fs/cgroup/cpu.max").read().split()
        if v and v[0] != "max":
            return {"cpus": max(1, -(-int(v[0]) // int(v[1]))), "source": "cgroup v2 cpu.max", "cpu_count": n_host}
    except (OSError, ValueError, IndexError):
        pass
    try:
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read()); per = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and per > 0:
            return {"cpus": max(1, -(-q // per)), "source": "cgroup v1 cfs quota", "cpu_count": n_host}
    except (OSError, ValueError):
        pass
    try:
        return {"cpus": len(os.sched_getaffinity(0)), "source": "sched_getaffinity", "cpu_count": n_host}
    except (AttributeError, OSError):
        return {"cpus": n_host, "source": "os.cpu_count", "cpu_count": n_host}


TEMP_BUDGET_BYTES = 6 << 20        # the largest float64 temporary a chunk allocates: under glibc's mmap threshold -> arena-served, no re-faulting


def deliverable_cores() -> dict:
    """The cores this process can actually run on: the scheduler affinity count (nproc) capped by the cgroup quota — the quota
    alone over-counts under a container runtime (cpu=8 -> cfs quota 24, affinity 8)."""
    q = cgroup_cpu_quota()
    try:
        aff = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        aff = os.cpu_count() or 1
    return {"cores": max(1, min(aff, q["cpus"])), "affinity": aff, "quota": q["cpus"], "quota_source": q["source"], "cpu_count": q["cpu_count"]}


PROBE_ELEMENTS = 2_000_000          # float64 elements per probe thread (well beyond any cache): the streaming regime of the post's temporaries
PROBE_REPS = 2                      # passes per rung: the probe's cost stays under what it overlaps (the import and model build)
PROBE_LADDER = (1, 2, 4, 8, 12, 16)  # the thread counts tried; cap = min(affinity, PROBE_CAP)
PROBE_CAP = 16

_PROBE_CODE = r"""
import json, os, sys, time
import numpy as np
from concurrent.futures import ThreadPoolExecutor
N, reps, cap = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
ladder = [n for n in (1, 2, 4, 8, 12, 16) if n <= cap]
if not ladder or ladder[-1] != cap:
    ladder.append(cap)
rng = np.random.default_rng(20260826)
arrs = [rng.uniform(1.0, 2.0, size=N) for _ in range(cap)]
outs = [np.ones_like(a) for a in arrs]                 # every page touched before any timing
def work(i):
    x, y = arrs[i], outs[i]
    for _ in range(reps):
        np.log2(x, out=y); np.multiply(y, 1.0001, out=y)
    return float(y[0])
for i in range(cap):
    work(i)
t1 = min((lambda: (lambda t0: (work(0), time.perf_counter() - t0)[1])(time.perf_counter()))() for _ in range(3))
table = {}
for n in ladder:
    best = None
    for _ in range(2):                                 # best of two: a scheduler hiccup never sizes the pool
        with ThreadPoolExecutor(n) as ex:
            t0 = time.perf_counter(); list(ex.map(work, range(n))); dt = time.perf_counter() - t0
        best = dt if best is None else min(best, dt)
    table[str(n)] = round(n * t1 / best, 2) if best > 0 else float(n)
print("PROBE_JSON " + json.dumps({"table": table, "t1_s": t1}))
"""


def plausibility(table: dict, deliverable: int) -> str | None:
    """The plausibility guard on the probe's table: None when plausible, else the refusal by name."""
    if not table:
        return "empty table"
    ks = sorted(int(k) for k in table)
    vals = [float(table[str(k)]) for k in ks]
    if len(vals) >= 2 and vals[1] < vals[0]:        # the 2-thread reading below the 1-thread one = refused
        return f"{ks[1]} threads read {vals[1]} effective cores, below the 1-thread reading {vals[0]}"
    for a, b, va, vb in zip(ks, ks[1:], vals, vals[1:]):
        if va > 0 and vb > 0 and (va / vb > 2.0 or vb / va > 2.0 * (b / a)):
            return f"non-monotone by > 2x between {a} and {b} threads ({va} -> {vb} effective cores)"
    knee = max(vals)
    if deliverable >= 4 and knee < 2.0:
        return f"knee {knee} < 2 on a host with {deliverable} deliverable cores"
    if knee > 1.5 * deliverable + 1:
        return f"knee {knee} above the deliverable cores {deliverable}"
    return None


def probe_effective_cores(max_threads: int = None) -> dict:
    """The cores the machine delivers to numpy, MEASURED in a SUBPROCESS — a fresh interpreter with numpy only (inside a process that
    has initialised TensorFlow on a GPU the same ladder reads implausibly low, so the probe never runs in-process). A float64 log2
    stream per thread (PROBE_ELEMENTS each, every page touched first, best of two per rung) at 1, 2, 4, 8, 12, 16 threads up to the
    affinity count; effective cores at n threads = n * t1 / t_n; the knee = the maximum. The PLAUSIBILITY GUARD refuses by name
    (adjacent rungs disagreeing by more than a factor of two; knee < 2 on >= 4 deliverable cores; knee far above the deliverable
    count) and the caller falls back to the OS witnesses with the refusal stamped — never a silent pool of 2. Runs once per process;
    the table is stamped. Never raises."""
    import subprocess, json as _json
    d = deliverable_cores()
    cap = int(max_threads) if max_threads else max(1, min(d["affinity"], PROBE_CAP))
    base = {"affinity": d["affinity"], "cpu_quota": d["quota"], "quota_source": d["quota_source"], "deliverable_os": d["cores"], "elements": PROBE_ELEMENTS, "reps": PROBE_REPS, "placement": "subprocess (numpy only, before the TensorFlow import)"}
    try:
        env = dict(os.environ); env.pop("OMP_NUM_THREADS", None)
        r = subprocess.run([sys.executable, "-c", _PROBE_CODE, str(PROBE_ELEMENTS), str(PROBE_REPS), str(cap)], capture_output=True, text=True, timeout=60, env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith("PROBE_JSON ")]
        if r.returncode != 0 or not line:
            return {**base, "effective_cores": float(d["cores"]), "table": {}, "probe_ok": False, "refusal": f"subprocess rc {r.returncode}: {r.stderr[-160:]}"}
        js = _json.loads(line[-1][11:])
        table = js["table"]; knee = max(float(v) for v in table.values())
        refusal = plausibility(table, d["cores"])
        if refusal:
            return {**base, "effective_cores": float(d["cores"]), "table": table, "t1_s": js["t1_s"], "probe_ok": False, "refusal": refusal}
        return {**base, "effective_cores": round(knee, 2), "table": table, "t1_s": js["t1_s"], "probe_ok": True, "refusal": None}
    except Exception as e:                                             # noqa: BLE001 — a stamp, never a gate
        return {**base, "effective_cores": float(d["cores"]), "table": {}, "probe_ok": False, "refusal": f"{type(e).__name__}: {str(e)[:120]}"}


def hard_count_skip(d: dict) -> str | None:
    """The ladder is skipped when the scheduler affinity count equals a CGROUP quota (a hard count both witnesses agree on);
    returns the stamped reason, or None (the witnesses disagree -> the capped ladder decides)."""
    if d["quota_source"].startswith("cgroup") and int(d["affinity"]) == int(d["quota"]):
        return f"ladder skipped: affinity == {d['quota_source']} == {d['quota']} (a hard count)"
    return None


def start_probe_async():
    """The SAME subprocess probe, started here (before the TensorFlow import) and NOT waited on — it runs under the import + model
    build the stock pays anyway and is joined at the first post, so its seconds never sit on the user's clock. Returns a handle for
    pool_size(handle=...); never raises."""
    import subprocess
    d = deliverable_cores()
    cap = max(1, min(d["affinity"], PROBE_CAP))
    t0 = time.perf_counter()
    skip = hard_count_skip(d)
    if skip is not None:
        # affinity == cgroup quota == a hard count both OS witnesses agree on -> the ladder is SKIPPED outright and the pool is
        # sized from the hard count: no probe process at all
        return {"proc": None, "t_start": t0, "cap": cap, "deliverable": d, "start_error": None, "skip": skip}
    try:
        env = dict(os.environ); env.pop("OMP_NUM_THREADS", None)
        proc = subprocess.Popen([sys.executable, "-c", _PROBE_CODE, str(PROBE_ELEMENTS), str(PROBE_REPS), str(cap)], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
    except Exception as e:                                             # noqa: BLE001 — a stamp, never a gate
        proc = None; err = f"{type(e).__name__}: {str(e)[:120]}"
    else:
        err = None
    return {"proc": proc, "t_start": t0, "cap": cap, "deliverable": d, "start_error": err}


def join_probe(handle, timeout_s: float = 60.0) -> dict:
    """Collect the async probe: waits only for what is still running (stamps the wait it actually cost the user's clock and the
    probe's own wall); the plausibility guard and the fallback exactly as probe_effective_cores."""
    import json as _json
    d = handle["deliverable"]
    base = {"affinity": d["affinity"], "cpu_quota": d["quota"], "quota_source": d["quota_source"], "deliverable_os": d["cores"], "elements": PROBE_ELEMENTS, "reps": PROBE_REPS,
            "placement": "subprocess started before the TensorFlow import, joined at the first post (async, v11)"}
    proc = handle.get("proc")
    if handle.get("skip"):
        return {**base, "effective_cores": float(d["cores"]), "table": {}, "probe_ok": True, "refusal": None, "skipped": handle["skip"], "probe_wall_s": 0.0, "join_wait_s": 0.0,
                "placement": "ladder skipped on a hard count (v14); pool = the hard count - margin"}
    if proc is None:
        return {**base, "effective_cores": float(d["cores"]), "table": {}, "probe_ok": False, "refusal": f"start failed: {handle.get('start_error')}", "probe_wall_s": None, "join_wait_s": 0.0}
    tj = time.perf_counter()
    try:
        out, err = proc.communicate(timeout=timeout_s)
    except Exception as e:                                             # noqa: BLE001
        try: proc.kill()
        except Exception: pass
        return {**base, "effective_cores": float(d["cores"]), "table": {}, "probe_ok": False, "refusal": f"join {type(e).__name__}", "probe_wall_s": time.perf_counter() - handle["t_start"], "join_wait_s": time.perf_counter() - tj}
    wait = time.perf_counter() - tj; wall = time.perf_counter() - handle["t_start"]
    line = [l for l in out.splitlines() if l.startswith("PROBE_JSON ")]
    if proc.returncode != 0 or not line:
        return {**base, "effective_cores": float(d["cores"]), "table": {}, "probe_ok": False, "refusal": f"subprocess rc {proc.returncode}: {err[-160:]}", "probe_wall_s": wall, "join_wait_s": wait}
    js = _json.loads(line[-1][11:]); table = js["table"]; knee = max(float(v) for v in table.values())
    refusal = plausibility(table, d["cores"])
    if refusal:
        return {**base, "effective_cores": float(d["cores"]), "table": table, "t1_s": js["t1_s"], "probe_ok": False, "refusal": refusal, "probe_wall_s": wall, "join_wait_s": wait}
    return {**base, "effective_cores": round(knee, 2), "table": table, "t1_s": js["t1_s"], "probe_ok": True, "refusal": None, "probe_wall_s": wall, "join_wait_s": wait}


def pool_size(margin: int = 2, floor: int = 2, handle=None) -> dict:
    """pool = max(floor, round(probe_effective_cores) - margin) when the subprocess probe passes the plausibility guard, else
    max(floor, min(affinity, quota) - margin) with the refusal stamped."""
    pr = join_probe(handle) if handle is not None else probe_effective_cores()
    n = max(floor, int(round(pr["effective_cores"])) - margin)
    form = ((f"max({floor}, hard_count - {margin}) [{pr['skipped']}]" if pr.get("skipped") else f"max({floor}, round(probe_effective_cores) - {margin})") if pr["probe_ok"] else f"max({floor}, min(affinity, quota) - {margin}) [probe refused: {pr['refusal']}]")
    return {"threads": n, "probe_effective_cores": pr["effective_cores"], "probe_table": pr["table"], "probe_ok": pr["probe_ok"], "probe_refusal": pr.get("refusal"),
            "probe_placement": pr["placement"], "affinity": pr["affinity"], "cpu_quota": pr["cpu_quota"], "quota_source": pr["quota_source"], "deliverable_os": pr["deliverable_os"],
            "form": form, "probe_wall_s": pr.get("probe_wall_s"), "join_wait_s": pr.get("join_wait_s"), "ladder_skipped": pr.get("skipped"),
            "fixed_cost_on_user_clock_s": pr.get("join_wait_s") if handle is not None else pr.get("probe_wall_s")}


def chunk_count(n_out: int, seq_len, threads: int, temp_budget_bytes: int = None) -> int:
    """>= threads chunks, each narrow enough that a (seq_len, width) float64 temporary stays under the budget (the module's
    TEMP_BUDGET_BYTES at call time when None)."""
    if not seq_len:
        return max(1, int(threads))
    if temp_budget_bytes is None:
        temp_budget_bytes = TEMP_BUDGET_BYTES
    w_max = max(32, int(temp_budget_bytes) // (8 * int(seq_len)))
    return max(int(threads), -(-int(n_out) // w_max))


def applies(options, sum_strand: bool, sum_length: bool) -> bool:
    """The option set the chunked post supports: a targets file, strand sums, the length-preserving write
    (write_snp_len) and per-column statistics only. Anything else runs the stock path in the entry script."""
    stats = options.sad_stats if isinstance(options.sad_stats, (list, tuple)) else str(options.sad_stats).split(",")
    return bool(options.targets_file is not None and sum_strand and not sum_length and all(s in PER_COLUMN_STATS for s in stats))


def chunk_plan(strand_transform, n_chunks: int) -> list:
    """[(ilo, ihi, lo, hi), ...]: output-column chunks [lo, hi) of the collapsed axis with their input-column ranges
    [ilo, ihi). Requires the transform's row -> column map to be monotone with contiguous input ranges (asserted)."""
    csr = strand_transform.tocsr()
    n_in, n_out = csr.shape
    col_of_row = np.full(n_in, -1, dtype=np.int64)
    for r in range(n_in):
        s, e = csr.indptr[r], csr.indptr[r + 1]
        if e - s != 1:
            raise ValueError(f"strand_transform row {r} has {e - s} entries (expected exactly 1)")
        col_of_row[r] = csr.indices[s]
    if np.any(np.diff(col_of_row) < 0):
        raise ValueError("strand_transform row->column map is not monotone; the chunked post does not apply")
    n_chunks = max(1, min(int(n_chunks), n_out))
    bounds = np.linspace(0, n_out, n_chunks + 1).astype(int)
    plan = []
    for c in range(n_chunks):
        lo, hi = int(bounds[c]), int(bounds[c + 1])
        if hi <= lo:
            continue
        rows = np.nonzero((col_of_row >= lo) & (col_of_row < hi))[0]
        ilo, ihi = int(rows[0]), int(rows[-1]) + 1
        if ihi - ilo != len(rows):
            raise ValueError("input columns of an output chunk are not contiguous; the chunked post does not apply")
        plan.append((ilo, ihi, lo, hi))
    assert plan[0][0] == 0 and plan[-1][1] == n_in and plan[0][2] == 0 and plan[-1][3] == n_out
    return plan


class ChunkedPost:
    """One instance per process: the plan, the pool and the stock callables. ``write_variant`` = the stock lines
    borzoi_sad.py:273-297 for one variant, chunked."""

    def __init__(self, targets_df, strand_transform, sad_stats, untransform, write_snp_len, threads: int, seq_len=None):
        self.targets_df = targets_df
        self.csr = strand_transform.tocsr()
        self.sad_stats = list(sad_stats)
        self.untransform = untransform            # callable(preds, targets_df_subset) -> preds, or None (no untransform)
        self.write_snp_len = _writer.make_writer(write_snp_len)    # the writer (exact: the stock's bits, only the requested passes; a statistic outside its set -> the entry script's own write_snp_len)
        print("KIT_POST " + _writer.line(self.write_snp_len.stamp), flush=True)   # the writer's line, once per process (the form + its tier, said once)
        self.threads = int(threads)
        self.seq_len = seq_len
        self.n_chunks = chunk_count(self.csr.shape[1], seq_len, self.threads)
        self.plan = chunk_plan(self.csr, self.n_chunks)
        self.subs = [self.csr[ilo:ihi, lo:hi] for (ilo, ihi, lo, hi) in self.plan]
        self.tdf = [targets_df.iloc[ilo:ihi] for (ilo, ihi, lo, hi) in self.plan]
        self.pool = ThreadPoolExecutor(self.threads)
        self.t_worker = 0.0
        self._t_lock = __import__('threading').Lock()
        self.n_out = self.csr.shape[1]
        self.n_variants = 0
        self.n_fallback = 0

    def _chunk(self, c, ref_preds, alt_preds):
        _t0 = time.perf_counter()
        try:
            return self._chunk_body(c, ref_preds, alt_preds)
        finally:
            with self._t_lock:
                self.t_worker += time.perf_counter() - _t0      # summed over threads: the post's CPU-seconds on the pool

    def _chunk_body(self, c, ref_preds, alt_preds):
        ilo, ihi, lo, hi = self.plan[c]
        r = ref_preds[:, ilo:ihi]
        a = alt_preds[:, ilo:ihi]
        if self.untransform is not None:
            r = self.untransform(r, self.tdf[c])
            a = self.untransform(a, self.tdf[c])
        r = r * self.subs[c]                       # the stock's `ref_preds * strand_transform` on the sub-block (scipy)
        a = a * self.subs[c]
        out = {s: np.empty((1, hi - lo), dtype="float16") for s in self.sad_stats}
        self.write_snp_len(r, a, out, 0, self.sad_stats)       # the selected writer (the stock's statistics), row 0 of a per-chunk buffer
        return c, out

    def write_variant(self, ref_preds, alt_preds, sad_out, si) -> bool:
        """One variant; returns False (nothing written) when the arrays are outside the supported shape class, so the entry
        script runs the stock path for it."""
        L = ref_preds.shape[0]
        if ref_preds.ndim != 2 or alt_preds.shape != ref_preds.shape or L % 16 != 0 or ref_preds.shape[1] != self.csr.shape[0]:
            self.n_fallback += 1
            return False
        rows = {s: np.empty(self.n_out, dtype="float16") for s in self.sad_stats}
        for c, out in self.pool.map(lambda c: self._chunk(c, ref_preds, alt_preds), range(len(self.plan))):
            lo, hi = self.plan[c][2], self.plan[c][3]
            for s in self.sad_stats:
                rows[s][lo:hi] = out[s][0]
        for s in self.sad_stats:                   # the stock's per-stat row write (write_snp_len: sad_out[stat][si] = ...)
            sad_out[s][si] = rows[s]
        self.n_variants += 1
        return True

    def submit(self, ref_preds, alt_preds, sad_out, si) -> bool:
        """The synchronous form of the submit/flush protocol (the base of the pipelined post; the tests drive it directly)."""
        return self.write_variant(ref_preds, alt_preds, sad_out, si)

    def flush(self):
        return None

    @staticmethod
    def mmap_facts(seq_len, max_width, n_cols_in) -> dict:
        """glibc exposes no getter for M_MMAP_THRESHOLD; the facts: the dynamic rule (128 KiB at start, raised to the size
        of each freed mmapped chunk up to DEFAULT_MMAP_THRESHOLD_MAX = 32 MiB on 64-bit; fixed when MALLOC_MMAP_THRESHOLD_ /
        mallopt set it), the env overrides if set, and the float64 temporaries' sizes: the stock's (seq_len x n_cols_in) and the
        kit's widest chunk (seq_len x max_width) — the crossing of the 32 MiB maximum is the page-fault mechanism."""
        L = int(seq_len or 0)
        return {"glibc_mmap_threshold_rule": "dynamic: 128 KiB initial, raised by frees up to DEFAULT_MMAP_THRESHOLD_MAX = 32 MiB (64-bit); no read API",
                "env_MALLOC_MMAP_THRESHOLD_": os.environ.get("MALLOC_MMAP_THRESHOLD_"), "env_MALLOC_ARENA_MAX": os.environ.get("MALLOC_ARENA_MAX"),
                "stock_temporary_bytes_f64": 8 * L * int(n_cols_in), "kit_max_chunk_temporary_bytes_f64": 8 * L * int(max_width),
                "kit_chunk_temporaries_under_32MiB": (8 * L * int(max_width)) <= (32 << 20), "stock_temporary_over_32MiB": (8 * L * int(n_cols_in)) > (32 << 20)}

    def stamp(self) -> dict:
        return {"kit_post": "column_chunked_threaded", "writer": self.write_snp_len.stamp, "threads": self.threads, "n_chunks": len(self.plan), "plan": self.plan, "worker_post_cpu_s": self.t_worker,
                "mmap": self.mmap_facts(self.seq_len, max(hi - lo for (_, _, lo, hi) in self.plan), self.csr.shape[0]),
                "seq_len": self.seq_len, "temp_budget_bytes": TEMP_BUDGET_BYTES,
                "max_chunk_width": max(hi - lo for (_, _, lo, hi) in self.plan),
                "stats": self.sad_stats, "untransform": getattr(self.untransform, "__name__", str(self.untransform)),
                "n_variants": self.n_variants, "n_fallback_stock_path": self.n_fallback,
                "host_pin_env": os.environ.get("NPY_DISABLE_CPU_FEATURES")}


class PipelinedChunkedPost(ChunkedPost):
    """The post of variant k runs on the pool WHILE the main thread runs variant k+1's one-hot + forward (the stock loop leaves the
    GPU idle for the whole host post, which is most of a variant's wall). The per-variant computation is ChunkedPost's (``_chunk``
    per column block; bytes identical); only WHEN it runs changes. The h5 rows are written from the main thread only (h5py is not
    thread-safe) at the next ``submit`` / ``flush``, each at its own index ``si``, so the file's contents do not depend on timing.
    ``max_in_flight`` variants may be pending (1 = the next forward overlaps one post; memory = one extra (ref, alt) float32 pair +
    its float64 temporaries)."""

    def __init__(self, targets_df, strand_transform, sad_stats, untransform, write_snp_len, threads: int, seq_len=None, max_in_flight: int = 1):
        super().__init__(targets_df, strand_transform, sad_stats, untransform, write_snp_len, threads, seq_len=seq_len)
        self.max_in_flight = max(1, int(max_in_flight))
        self.pending = []                      # [(si, sad_out, [future per chunk]), ...] in submission order
        self.t_wait = 0.0                      # main-thread seconds spent waiting for a pending post (the un-overlapped remainder)

    def _write(self, si, sad_out, futs):
        rows = {s: np.empty(self.n_out, dtype="float16") for s in self.sad_stats}
        for f in futs:
            c, out = f.result()
            lo, hi = self.plan[c][2], self.plan[c][3]
            for s in self.sad_stats:
                rows[s][lo:hi] = out[s][0]
        for s in self.sad_stats:
            sad_out[s][si] = rows[s]
        self.n_variants += 1

    def _drain(self, keep: int):
        while len(self.pending) > keep:
            si, sad_out, futs = self.pending.pop(0)
            t0 = time.perf_counter()
            self._write(si, sad_out, futs)
            self.t_wait += time.perf_counter() - t0

    def submit(self, ref_preds, alt_preds, sad_out, si) -> bool:
        """Queue variant si's post; returns False (nothing queued) outside the supported shape class -> the entry script runs
        the stock path for it (its row is written directly; pending rows are other indices)."""
        L = ref_preds.shape[0]
        if ref_preds.ndim != 2 or alt_preds.shape != ref_preds.shape or L % 16 != 0 or ref_preds.shape[1] != self.csr.shape[0]:
            self.n_fallback += 1
            return False
        self._drain(keep=self.max_in_flight - 1)
        futs = [self.pool.submit(self._chunk, c, ref_preds, alt_preds) for c in range(len(self.plan))]
        self.pending.append((si, sad_out, futs))
        return True

    def flush(self):
        self._drain(keep=0)

    def stamp(self) -> dict:
        d = super().stamp()
        d.update({"kit_post": "column_chunked_threaded_pipelined", "max_in_flight": self.max_in_flight, "pending_at_stamp": len(self.pending),
                  "main_thread_wait_s": self.t_wait})
        return d
