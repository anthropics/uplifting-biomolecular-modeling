# SPDX-License-Identifier: Apache-2.0
# odde_offload.py -- ODDE_OFFLOAD: host-resident pair tensors for OpenDDE 1.1.1 (single card; the memory mode's offload unit: trades time for peak memory).
#
# The pair tensor of a PairformerStack (z of the trunk / z_struct of the structural-token refiner / the confidence-head pair
# stack) lives in (pinned) host RAM as an [n, n, c] fp32 tensor and every pair op streams row (or column) blocks through the
# GPU using the STOCK module weights and, wherever the op is row-local, the STOCK sub-module code:
#   * TriangleMultiplication (out/in): gated projections a, b built on the GPU from streamed row blocks (stock torch
#     `_inference_forward` blocking: 256-row projection chunks, 256-column output chunks); "full" mode holds a, b for all hidden
#     channels (GEMM shapes identical to the stock torch path), "chunked" mode holds cc channels at a time and parks the product
#     x in a host buffer laid out chunk-major so that every transfer is a contiguous block.
#   * TriangleAttention (starting / ending node): pass 1 builds the [H, n, n] triangle bias from streamed LayerNorm'd rows
#     WITHOUT holding a LayerNorm'd copy of the pair tensor (the same chunked prologue as levers/XL tri_ln); pass 2 runs the stock
#     Attention module (stock kernel choice, stock chunk rows) per row block.  The ending
#     node reads/writes column blocks of the host tensor exactly where stock transposes.
#   * PairTransition / AttentionPairBias pair bias: stock modules per row block (APB streaming mirrors upstream's own foldcp
#     row-chunk streaming helper).
#   * Structural expander: upstream's chunked row builders write z_struct row blocks straight to the host; relative-position
#     one-hots are materialised lazily per row block (upstream LazyRelativePositionEncodingFeatures) -- never as an
#     n_struct^2 x 139 tensor.
#   * Diffusion pair conditioning: computed ONCE per row block from host z_struct into a GPU-resident [n_s, n_s, 128] cache
#     (== upstream --enable_cache semantics), so the 200-step sampler never touches z_struct.
# Numerics: fp32 everywhere, no algorithmic substitution, no dtype change; per-element arithmetic is the stock arithmetic up to
# GEMM/LayerNorm launch-shape effects (row-blocked launches) -> tier 2 (bit-exactness to stock is not claimed).
# CUDA graphs: none (the CLI path has none and this module never captures).
from opt_core.oom import is_oom                      # an out-of-memory error is re-raised before any reroute below (the core's one classifier)
import os, sys, time, math, json, resource
from typing import Optional, Any
import torch
import torch.nn.functional as F

_T0 = time.time()
CFG = {
    "stages": set(),            # subset of {'struct','trunk','conf'}
    "rows": int(os.environ.get("ODDE_OFFLOAD_ROWS", "256")),          # row block for projections / transitions / expander / bias passes
    "cc": os.environ.get("ODDE_OFFLOAD_CC", "auto"),                  # TriMul hidden-channel chunk ('auto' or int)
    "memfrac": float(os.environ.get("ODDE_OFFLOAD_MEMFRAC", "0.80")),  # fraction of currently-free GPU memory usable for TriMul operands
    "xbuf": os.environ.get("ODDE_OFFLOAD_XBUF", "host"),              # 'host' | 'disk:<dir>'  (chunked-TriMul product buffer)
    "ckpt_dir": os.environ.get("ODDE_OFFLOAD_CKPT_DIR", ""),
    "resume": os.environ.get("ODDE_OFFLOAD_RESUME", ""),              # '' | 'trunk' | 'refiner'
    "ckpt_max_gb": float(os.environ.get("ODDE_OFFLOAD_CKPT_MAX_GB", "80")),
    "pin": os.environ.get("ODDE_OFFLOAD_PIN", "1") == "1",           # 1: pinned host buffers (a refused pinned allocation is an error naming this switch); 0: pageable host buffers
    "log": os.environ.get("ODDE_OFFLOAD_LOG", ""),
    "diffz": os.environ.get("ODDE_OFFLOAD_DIFFZ", "1") == "1",   # diffusion: no per-step pair_z clone + LayerNorm(pair_z) memoised across steps
    "free_templ": os.environ.get("ODDE_OFFLOAD_FREE_TEMPL", "1") == "1",     # drop the (dummy) template pair features from the feature dicts after the trunk (only the trunk template embedder reads them; ~176 B x T x N^2 on the GPU)
    "diffz_perm": os.environ.get("ODDE_OFFLOAD_DIFFZ_PERM", "1") == "1",     # memoise the permuted LN image (saves one pair-sized tensor per diffusion step)
    "diffz_release": os.environ.get("ODDE_OFFLOAD_DIFFZ_RELEASE", "1") == "1",   # free the GPU pair_z once its LN image exists
}


def refuse_retired(name, accepted=("", "0"), reason="retired:superseded_by_upstream_bound"):
    """A RETIRED env switch set to anything but its inert values refuses BY NAME at import (never a knob that silently does nothing).
    The shim turns the refusal into `NOT ACTIVE` + exit 3 (ODDE_OFFLOAD_STRICT=1) or one stderr line (STRICT=0)."""
    v = os.environ.get(name, "").strip()
    if v not in accepted:
        raise ValueError(f"{name}={v!r}: NOT ACTIVE: reason={reason}")


# ODDE_OFFLOAD_DCHUNK / ODDE_OFFLOAD_FORCE_CHUNK are RETIRED: OpenDDE 1.1.1 bounds the dynamic pairformer chunk size itself
# (OpenDDE._bound_pairformer_chunk_size after _get_dynamic_chunk_size, at the trunk and at the 2N structural-refiner call site),
# which post-clamps any floor these switches could set.  "0" / unset (off) is accepted; an active value refuses by name.
refuse_retired("ODDE_OFFLOAD_DCHUNK")
refuse_retired("ODDE_OFFLOAD_FORCE_CHUNK")
_v = os.environ.get("ODDE_OFFLOAD", "")
if _v:
    for tok in _v.split(","):
        tok = tok.strip().lower()
        if tok in ("1", "all", "on", "true"):
            CFG["stages"] |= {"struct", "trunk", "conf"}
        elif tok in ("struct", "trunk", "conf"):
            CFG["stages"].add(tok)
        elif tok:
            raise ValueError(f"ODDE_OFFLOAD: unknown token {tok!r} (use all|struct|trunk|conf)")

STATS = {"h2d_bytes": 0, "d2h_bytes": 0, "strided_bytes": 0, "stages": [], "fallbacks": {},
         # RAN-OR-REFUSE call counters, one flat integer key per lever entry point (0 until the offloaded code path actually runs in this
         # process; the kit's ran.py sums STATS[key]):
         #   calls_trunk      off_get_pairformer_output (pair_offload_trunk)        calls_conf   off_confidence_head (pair_offload_conf)
         #   calls_disto      off_contact_probs (pair_offload_conf)                 calls_struct the structural expander+refiner stage (pair_offload_struct)
         #   calls_diff_cache off_diffusion_prepare_pair_cache (pair_offload_struct) calls_diffz  DiffusionConditioning.forward without the per-step clone (diffz), per call
         #   calls_free_templ template pair features dropped after the trunk (free_templ), per forward
         # bigln_guard keeps its own record in _BIGLN (installed / calls / hits).  "stages" holds the per-stage telemetry records (timers that
         # also wrap a STOCK trunk/confidence stage), so it is not evidence that an offloaded body ran; the calls_* keys are.
         "calls_trunk": 0, "calls_conf": 0, "calls_disto": 0, "calls_struct": 0, "calls_diff_cache": 0, "calls_diffz": 0, "calls_free_templ": 0}


def fallback(reason):
    """Every runtime fallback path counts itself here (reason -> n); the kit reads it."""
    STATS["fallbacks"][reason] = STATS["fallbacks"].get(reason, 0) + 1


class PinnedHostMemoryRefused(RuntimeError):
    """A pinned host allocation the driver refused (cudaHostAlloc: out of memory / the memlock limit) with ODDE_OFFLOAD_PIN=1: no pageable
    fallback is applied — the run ends naming the switch that selects pageable host buffers.  Raised `from` the driver's error, so
    `opt_core.oom.is_oom` classifies it as the out-of-memory it is."""


def pinned_refused(what, nbytes, exc):
    return PinnedHostMemoryRefused(f"[odde_offload] pinned host memory refused for {what} ({nbytes/2**30:.1f} GiB; cudaHostAlloc: out of memory / memlock limit: "
                                   f"{str(exc)[:120]}); no fallback applied; run with ODDE_OFFLOAD_PIN=0 (big: OPENDDE_BIG_PAIR_OFFLOAD_PIN=0) to use pageable host buffers")


def count(key, n=1):
    """RAN-OR-REFUSE: a lever's entry point counts itself here (STATS["calls_<key>"]); a lever that installed but never ran reads 0."""
    STATS["calls_" + key] = STATS.get("calls_" + key, 0) + n


def host_rss_gib():
    try:
        with open("/proc/self/status") as f:
            d = {l.split(":")[0]: l.split(":")[1].strip() for l in f if l.startswith(("VmRSS", "VmHWM"))}
        return float(d["VmRSS"].split()[0]) / 2**20, float(d["VmHWM"].split()[0]) / 2**20
    except Exception:
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
        return r, r


def log(msg):
    """Quiet unless ODDE_OFFLOAD_LOG is set: "-" prints to stdout; a path appends to that file and prints. The per-block lines this emits
    scale with the number of row/column blocks (~N^2/rows^2 per stage): the stage records in STATS["stages"] carry the telemetry."""
    if not CFG["log"]:
        return
    line = f"[odde_offload] {time.time()-_T0:8.1f}s | {msg}"
    print(line, flush=True)
    if CFG["log"] != "-":
        try:
            with open(CFG["log"], "a") as f:
                f.write(line + "\n")
        except Exception:
            pass


def gpu_mem_str():
    if not torch.cuda.is_available():
        return "cpu"
    a = torch.cuda.max_memory_allocated() / 2**30; r = torch.cuda.max_memory_reserved() / 2**30
    fr, tot = torch.cuda.mem_get_info()
    return f"peak_alloc={a:.2f}GiB peak_reserved={r:.2f}GiB free_now={fr/2**30:.1f}/{tot/2**30:.1f}GiB"


class stage_timer:
    def __init__(self, name, **kw):
        self.name = name; self.kw = kw
    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        self.t = time.time(); return self
    def __exit__(self, *a):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        w = time.time() - self.t; rss, hwm = host_rss_gib()
        rec = {"stage": self.name, "wall_s": round(w, 2), "host_rss_gib": round(rss, 2), "host_hwm_gib": round(hwm, 2), **self.kw}
        if torch.cuda.is_available():
            rec.update(peak_alloc_gib=round(torch.cuda.max_memory_allocated() / 2**30, 3), peak_reserved_gib=round(torch.cuda.max_memory_reserved() / 2**30, 3))
        STATS["stages"].append(rec)
        log(f"STAGE {self.name} wall={w:.1f}s {gpu_mem_str()} host_rss={rss:.1f}GiB hwm={hwm:.1f}GiB "
            f"h2d={STATS['h2d_bytes']/2**30:.0f}GiB d2h={STATS['d2h_bytes']/2**30:.0f}GiB strided={STATS['strided_bytes']/2**30:.0f}GiB {self.kw if self.kw else ''}")


def blocks(n, b):
    return [(i, min(i + b, n)) for i in range(0, n, b)]


# ---- pitched host<->device copies for column blocks (cudaMemcpy2D: a pitched column-block copy runs at the rate of a contiguous one and is
#      bit-identical; torch's strided copy_ between host and device is far slower, hence the staging path only as the fallback) -----------
_CUDART = None
def _cudart():
    global _CUDART
    if _CUDART is None:
        import ctypes, glob, sys as _sys
        cands = []
        for p in _sys.path:
            cands += glob.glob(os.path.join(p, "nvidia", "cuda_runtime", "lib", "libcudart.so*"))
        cands += glob.glob(os.path.join(os.path.dirname(torch.__file__), "lib", "libcudart*.so*"))
        cands += ["libcudart.so", "libcudart.so.12", "libcudart.so.13", "libcudart.so.11.0"]
        for c in cands:
            try:
                lib = ctypes.CDLL(c)
                f = lib.cudaMemcpy2D
                f.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_size_t, ctypes.c_int]
                f.restype = ctypes.c_int
                _CUDART = lib; break
            except (OSError, AttributeError):
                continue
        if _CUDART is None:
            _CUDART = False; fallback("libcudart unavailable: column blocks via pinned staging + CPU strided copy")
            log("cudaMemcpy2D unavailable (libcudart not found) -> column blocks via pinned staging + CPU strided copy")
    return _CUDART


def memcpy2d_cols_h2d(host_t, c0, c1, dst):
    """dst[n, cb, c] (cuda, contiguous) <- host_t[:, c0:c1, :] (host [n, n, c] contiguous) in ONE pitched copy."""
    n, _, c = host_t.shape; cb = c1 - c0; es = host_t.element_size()
    lib = _cudart(); assert lib
    torch.cuda.current_stream().synchronize()
    rc = lib.cudaMemcpy2D(dst.data_ptr(), cb * c * es, host_t.data_ptr() + c0 * c * es, n * c * es, cb * c * es, n, 1)
    if rc != 0: raise RuntimeError(f"cudaMemcpy2D H2D failed rc={rc}")


def memcpy2d_cols_d2h(host_t, c0, c1, src):
    n, _, c = host_t.shape; cb = c1 - c0; es = host_t.element_size()
    lib = _cudart(); assert lib
    torch.cuda.current_stream().synchronize()
    rc = lib.cudaMemcpy2D(host_t.data_ptr() + c0 * c * es, n * c * es, src.data_ptr(), cb * c * es, cb * c * es, n, 2)
    if rc != 0: raise RuntimeError(f"cudaMemcpy2D D2H failed rc={rc}")


# --------------------------------------------------------------------------------------------------------------------------
# Host-resident pair tensor
# --------------------------------------------------------------------------------------------------------------------------
class HostPair:
    """[n, n, c] pair tensor resident in host RAM (pinned when possible; or a disk memmap).  Row blocks are contiguous (plain
    DMA); column blocks go through a pinned staging buffer with a multi-threaded strided CPU copy."""

    def __init__(self, n: int, c: int, dtype=torch.float32, device="cuda", pin=None, disk_dir: str = ""):
        self.n, self.c, self.dtype, self.device = int(n), int(c), dtype, torch.device(device)
        pin = CFG["pin"] if pin is None else pin
        nbytes = self.n * self.n * self.c * (torch.finfo(dtype).bits // 8)
        self.kind = "host"; self.pinned = False
        if disk_dir:
            import numpy as np
            os.makedirs(disk_dir, exist_ok=True)
            self._path = os.path.join(disk_dir, f"hostpair_{id(self)}.bin")
            mm = np.memmap(self._path, dtype=np.float32, mode="w+", shape=(self.n, self.n, self.c))
            self.t = torch.from_numpy(mm); self.kind = "disk"
        else:
            t = None
            if pin and self.device.type == "cuda" and torch.cuda.is_available():
                try:
                    t = torch.empty((self.n, self.n, self.c), dtype=dtype, pin_memory=True); self.pinned = True
                except Exception as e:
                    if is_oom(e): raise pinned_refused("the host pair tensor", nbytes, e) from e
                    fallback("HostPair pinned alloc refused: pageable memory"); log(f"HostPair: pinned alloc of {nbytes/2**30:.1f} GiB refused ({str(e)[:80]}); using pageable memory")
            if t is None:
                t = torch.empty((self.n, self.n, self.c), dtype=dtype)
            self.t = t
        self._cbuf = None
        log(f"HostPair n={self.n} c={self.c} {nbytes/2**30:.2f} GiB kind={self.kind} pinned={self.pinned}")

    def rows(self, r0, r1) -> torch.Tensor:
        x = self.t[r0:r1]
        STATS["h2d_bytes"] += x.numel() * x.element_size()
        return x.to(self.device, non_blocking=False)

    def put_rows(self, r0, r1, x: torch.Tensor):
        STATS["d2h_bytes"] += x.numel() * x.element_size()
        self.t[r0:r1].copy_(x)

    def _stage(self, cb):
        if self._cbuf is None or self._cbuf.shape[1] < cb:
            try:
                self._cbuf = torch.empty((self.n, cb, self.c), dtype=self.dtype, pin_memory=(CFG["pin"] and self.device.type == "cuda" and torch.cuda.is_available()))
            except Exception as e:
                if is_oom(e): raise pinned_refused("the column staging buffer", self.n * cb * self.c * (torch.finfo(self.dtype).bits // 8), e) from e
                fallback("HostPair staging buffer pinned alloc failed: pageable staging buffer"); self._cbuf = torch.empty((self.n, cb, self.c), dtype=self.dtype)
        return self._cbuf[:, :cb]

    def _pitched_ok(self):
        if getattr(self, "_pitched", None) is None:
            ok = self.kind == "host" and self.device.type == "cuda" and torch.cuda.is_available() and bool(_cudart()) and os.environ.get("ODDE_OFFLOAD_PITCHED", "1") == "1"
            if ok:   # one-time consistency check against the torch path
                try:
                    c1 = min(self.n, 3); ref = self.t[:, 0:c1].contiguous().to(self.device)
                    got = torch.empty((self.n, c1, self.c), dtype=self.dtype, device=self.device); memcpy2d_cols_h2d(self.t, 0, c1, got)
                    ok = torch.equal(ref, got)
                    if not ok: fallback("pitched column copy consistency check mismatch: staging path"); log("pitched column copy consistency check MISMATCH -> disabled")
                except Exception as e:
                    if is_oom(e): raise
                    fallback("pitched column copy unavailable: staging path"); log(f"pitched column copy unavailable ({e}) -> staging path"); ok = False
            self._pitched = ok
            log(f"HostPair column transfers: {'cudaMemcpy2D (pitched)' if ok else 'pinned staging + CPU strided copy'}")
        return self._pitched

    def cols(self, c0, c1) -> torch.Tensor:
        """z[:, c0:c1] as a contiguous [n, cb, c] GPU tensor."""
        cb = c1 - c0; nb = self.n * cb * self.c * self.t.element_size()
        if self._pitched_ok():
            dst = torch.empty((self.n, cb, self.c), dtype=self.dtype, device=self.device)
            memcpy2d_cols_h2d(self.t, c0, c1, dst)
            STATS["h2d_bytes"] += nb
            return dst
        st = self._stage(cb)
        st.copy_(self.t[:, c0:c1])
        STATS["strided_bytes"] += nb; STATS["h2d_bytes"] += nb
        return st.to(self.device, non_blocking=False)

    def put_cols(self, c0, c1, x: torch.Tensor):
        cb = c1 - c0; nb = self.n * cb * self.c * self.t.element_size()
        if self._pitched_ok():
            memcpy2d_cols_d2h(self.t, c0, c1, x.contiguous())
            STATS["d2h_bytes"] += nb
            return
        st = self._stage(cb)
        st.copy_(x)
        self.t[:, c0:c1].copy_(st)
        STATS["strided_bytes"] += nb; STATS["d2h_bytes"] += nb

    def rows_T(self, r0, r1) -> torch.Tensor:
        """rows r0:r1 of z^T (= columns of z) as a contiguous [r1-r0, n, c] GPU tensor."""
        return self.cols(r0, r1).transpose(0, 1).contiguous()

    def put_rows_T(self, r0, r1, x: torch.Tensor):
        self.put_cols(r0, r1, x.transpose(0, 1).contiguous())

    @classmethod
    def from_tensor(cls, z: torch.Tensor, rows=256, **kw):
        n, c = z.shape[-3], z.shape[-1]
        hp = cls(n, c, dtype=z.dtype, device=(z.device if z.is_cuda else ("cuda" if torch.cuda.is_available() else "cpu")), **kw)
        for r0, r1 in blocks(n, rows):
            hp.put_rows(r0, r1, z[r0:r1])
        return hp

    def to_tensor(self, device=None) -> torch.Tensor:
        return self.t.to(self.device if device is None else device)

    def free(self):
        self.t = None; self._cbuf = None
        if self.kind == "disk":
            try: os.remove(self._path)
            except Exception: pass


# --------------------------------------------------------------------------------------------------------------------------
# Streamed pair ops (stock modules, stock weights)
# --------------------------------------------------------------------------------------------------------------------------
def _free_bytes():
    if not torch.cuda.is_available():
        return 1 << 62
    fr, tot = torch.cuda.mem_get_info()
    return fr + (torch.cuda.memory_reserved() - torch.cuda.memory_allocated())


def pick_cc(n, c_hidden, extra_bytes=0):
    """TriMul operand channel chunk: largest cc (== c_hidden, else a multiple of 32) with 2*cc*n*n*4 B inside the budget."""
    if CFG["cc"] != "auto":
        return max(1, min(int(CFG["cc"]), c_hidden))
    budget = _free_bytes() * CFG["memfrac"] - extra_bytes
    cc = int(budget // (2 * n * n * 4))
    if cc >= c_hidden:
        return c_hidden
    # round down to a divisor of c_hidden (>= 16) so every chunk is full width (contiguous host blocks, equal GEMM shapes)
    return max([d for d in range(16, c_hidden) if c_hidden % d == 0 and d <= cc] or [16])


_PIN_POOL = {}
def pooled_pinned(shape, dtype=torch.float32):
    """Reusable pinned host buffers keyed by (numel, dtype): torch's caching host allocator never returns freed pinned blocks to the
    OS, so allocating a fresh multi-GB buffer per call inflates host RSS without bound; one buffer per size instead."""
    numel = 1
    for s in shape: numel *= int(s)
    key = (numel, dtype)
    t = _PIN_POOL.get(key)
    if t is None:
        try:
            t = torch.empty(numel, dtype=dtype, pin_memory=(CFG["pin"] and torch.cuda.is_available()))
        except Exception as e:
            if is_oom(e): raise pinned_refused("a pooled host buffer", numel * torch.empty(0, dtype=dtype).element_size(), e) from e
            fallback("pinned pool alloc failed: pageable buffer"); t = torch.empty(numel, dtype=dtype)
        _PIN_POOL[key] = t
        tot = sum(v.numel() * v.element_size() for v in _PIN_POOL.values()) / 2**30
        log(f"pinned pool: new buffer {tuple(shape)} {numel*t.element_size()/2**30:.1f} GiB (pool total {tot:.1f} GiB)")
    return t.view(*shape)


class XBuf:
    """product buffer of the chunked TriMul: chunk-major layout [n_chunk][n(j: output column), n(i), cc] so that every
    transfer is one contiguous block.  get(j0, j1) returns x[:, j0:j1, :] laid out as the stock epilogue expects."""
    def __init__(self, n, c_hidden, cc, device, dtype=torch.float32):
        self.n, self.ch, self.cc, self.device = n, c_hidden, cc, device
        self.nchunk = (c_hidden + cc - 1) // cc
        disk = CFG["xbuf"].startswith("disk:")
        if disk:
            import numpy as np
            d = CFG["xbuf"][5:]; os.makedirs(d, exist_ok=True); self._path = os.path.join(d, f"xbuf_{id(self)}.bin")
            self.t = torch.from_numpy(np.memmap(self._path, dtype=np.float32, mode="w+", shape=(self.nchunk, n, n, cc)))
        else:
            self._path = None
            self.t = pooled_pinned((self.nchunk, n, n, cc), dtype)
        log(f"XBuf {self.nchunk} x [{n},{n},{cc}] {self.t.numel()*4/2**30:.1f} GiB ({'disk' if disk else 'host (pooled)'})")
    def put(self, ci, j0, j1, x_cnj):      # x_cnj: [w, n(i), |J|] on GPU  ->  stored as [|J|, n(i), w]
        blk = x_cnj.permute(2, 1, 0).contiguous()
        STATS["d2h_bytes"] += blk.numel() * 4
        self.t[ci, j0:j1, :, :blk.shape[-1]].copy_(blk)
    def get(self, j0, j1):                  # -> [ch, n(i), |J|] on GPU (the layout torch.matmul produced in full mode)
        parts = []
        for ci in range(self.nchunk):
            w = min(self.cc, self.ch - ci * self.cc)
            p = self.t[ci, j0:j1, :, :w]
            STATS["h2d_bytes"] += p.numel() * 4
            parts.append(p.to(self.device))
        x = torch.cat(parts, dim=-1)          # [|J|, n(i), ch]
        return x.permute(2, 1, 0)             # [ch, n(i), |J|] (view)
    def free(self):
        self.t = None
        if self._path:
            try: os.remove(self._path)
            except Exception: pass


@torch.no_grad()
def off_trimul(mod, hp: HostPair, rows=None, col_chunk=256):
    """z += TriangleMultiplication{Outgoing,Incoming}(z), z host-resident.  Mirrors the stock torch `_inference_forward`
    (with_add=True, mask == 1): projections in `rows`-row chunks, products in `col_chunk` output-column chunks, epilogue
    LN_out -> linear_z -> * sigmoid(linear_g(LN_in(z[:, J]))) -> z[:, J] += x."""
    from opendde.model.utils import permute_final_dims
    rows = rows or CFG["rows"]
    n, dev = hp.n, hp.device
    ch = mod.c_hidden; outgoing = bool(mod._outgoing)
    par = next(mod.parameters())
    cc = pick_cc(n, ch, extra_bytes=6 * rows * n * max(ch, hp.c) * 4 + 4 * n * col_chunk * max(ch, hp.c) * 4)
    full = cc >= ch
    t0 = time.time()

    def helper(zr, a: bool, csel):
        # == stock compute_projection_helper: p = sigmoid(linear_g(LN(z))) * linear_p(LN(z)) * mask ; -> [c, B, n]
        linear_g, linear_p = (mod.linear_a_g, mod.linear_a_p) if a else (mod.linear_b_g, mod.linear_b_p)
        ln = mod.layer_norm_in(zr)
        p = linear_g(ln); p.sigmoid_(); p *= linear_p(ln)
        p *= zr.new_ones(zr.shape[:-1]).unsqueeze(-1)
        p = permute_final_dims(p, (2, 0, 1))
        return p if csel is None else p[csel]

    def fill(A, Bm, csel):
        for r0, r1 in blocks(n, rows):
            zr = hp.rows(r0, r1)
            pa = helper(zr, True, csel)                  # [c, B, n]: pa[c, r, k] = a[r0+r, k, c]
            if outgoing:
                A[:, r0:r1, :] = pa                      # A[c, i, k] = a[i, k, c]
            else:
                A[:, :, r0:r1] = pa.transpose(-1, -2)    # A[c, x, y] = a[y, x, c]
            del pa
            Bm[:, r0:r1, :] = helper(zr, False, csel)    # Bm[c, r, k] = b[r, k, c]
            del zr

    def bsel(Bm, j0, j1):
        if outgoing:   # [c, k, j] = b[j, k, c]  (stock: transposed view of a row-chunk projection)
            return Bm[:, j0:j1, :].transpose(-1, -2)
        else:          # [c, k, j] = b[k, j, c]  (stock: contiguous column-chunk projection)
            return Bm[:, :, j0:j1].contiguous()

    def epilogue(j0, j1, x_c):                           # x_c: [ch, n, |J|]
        x = permute_final_dims(x_c, (1, 2, 0))           # [n, |J|, ch]
        x = mod.layer_norm_out(x)
        x = mod.linear_z(x)
        zc = hp.cols(j0, j1)                             # original z[:, J]
        g = mod.linear_g(mod.layer_norm_in(zc)); g.sigmoid_()
        x *= g; del g
        zc += x
        hp.put_cols(j0, j1, zc)
        del zc, x

    if full:
        A = torch.empty((ch, n, n), dtype=par.dtype, device=dev); Bm = torch.empty((ch, n, n), dtype=par.dtype, device=dev)
        fill(A, Bm, None)
        for j0, j1 in blocks(n, col_chunk):
            x_c = torch.matmul(A, bsel(Bm, j0, j1))
            epilogue(j0, j1, x_c); del x_c
        del A, Bm
    else:
        xb = XBuf(n, ch, cc, dev)
        A = torch.empty((cc, n, n), dtype=par.dtype, device=dev); Bm = torch.empty((cc, n, n), dtype=par.dtype, device=dev)
        for ci, c0 in enumerate(range(0, ch, cc)):
            c1 = min(c0 + cc, ch); w = c1 - c0
            Av, Bv = A[:w], Bm[:w]
            fill(Av, Bv, slice(c0, c1))
            for j0, j1 in blocks(n, col_chunk):
                xb.put(ci, j0, j1, torch.matmul(Av, bsel(Bv, j0, j1)))
        del A, Bm, Av, Bv
        for j0, j1 in blocks(n, col_chunk):
            epilogue(j0, j1, xb.get(j0, j1))
        xb.free()
    if torch.cuda.is_available(): torch.cuda.synchronize()
    log(f"trimul {'out' if outgoing else 'in '} n={n} cc={cc} ({'full' if full else 'chunked'}) rows={rows} colchunk={col_chunk} {time.time()-t0:.1f}s")


@torch.no_grad()
def off_triatt(mod, hp: HostPair, transposed: bool, triangle_attention: str, chunk_size: Optional[int], inplace_safe: bool):
    """z += TriangleAttention(x) with x = z (starting node) or x = z^T (`transposed`: PairformerBlock's tri_att_end call, whose
    module is also a starting-node TriangleAttention applied to the transposed tensor).  Pass 1: triangle bias from streamed LN
    rows; pass 2: stock `mod.mha` per row block with the inputs chunk_layer would feed it."""
    from opendde.model.utils import permute_final_dims
    n, dev = hp.n, hp.device
    assert getattr(mod, "starting", True), "offload expects starting-node modules (PairformerBlock transposes for the ending node)"
    get = hp.rows_T if transposed else hp.rows
    put = hp.put_rows_T if transposed else hp.put_rows
    rows_bias = CFG["rows"]
    rows_att = chunk_size if chunk_size is not None else min(n, 4 * CFG["rows"])
    H = mod.linear.weight.shape[0]
    t0 = time.time()
    tb = torch.empty((H, n, n), dtype=torch.float32, device=dev)
    for r0, r1 in blocks(n, rows_bias):
        xr = mod.layer_norm(get(r0, r1))
        tb[:, r0:r1, :] = permute_final_dims(mod.linear(xr), (2, 0, 1))
        del xr
    tb = tb.unsqueeze(0)                                 # [1, H, I, J]
    for r0, r1 in blocks(n, rows_att):
        xr_raw = get(r0, r1)
        xr = mod.layer_norm(xr_raw)
        mask_bias = xr.new_zeros((r1 - r0, 1, 1, n))    # == inf * (mask - 1), mask == 1
        # biases exactly as chunk_layer feeds them: mask bias sliced to the row block, triangle bias kept [1, H, n, n] (size-1 batch
        # dims are never expanded by chunk_layer; the cuEq kernel requires bias (B, 1, H, S, S))
        o = mod.mha(q_x=xr, kv_x=xr, biases=[mask_bias, tb], triangle_attention=triangle_attention)
        xr_raw += o
        put(r0, r1, xr_raw)
        del xr, xr_raw, o, mask_bias
    del tb
    if torch.cuda.is_available(): torch.cuda.synchronize()
    log(f"triatt {'end(T)' if transposed else 'start '} n={n} H={H} rows_bias={rows_bias} rows_att={rows_att} kernel={triangle_attention} {time.time()-t0:.1f}s")


@torch.no_grad()
def off_transition(mod, hp: HostPair, rows=None):
    rows = rows or CFG["rows"]
    for r0, r1 in blocks(hp.n, rows):
        zr = hp.rows(r0, r1)
        zr += mod(zr)
        hp.put_rows(r0, r1, zr); del zr


@torch.no_grad()
def off_apb(apb, a, hp: HostPair, extra_attn_bias=None, rows=None):
    """== AttentionPairBias.forward(a=a, s=None, z=z, extra_attn_bias=...) for the has_s == False configuration used by
    PairformerBlock: pair bias linear(LN(z rows)) (+ extra bias rows) streamed from host row blocks, attention evaluated per
    query-row block with the stock `_attention` primitive, then the stock transpose + `_wrap_up` (gating + output projection)."""
    from opendde.model.utils import permute_final_dims
    from opendde.model.modules.primitives import _attention
    assert not apb.has_s and not apb.cross_attention_mode
    rows = rows or CFG["rows"]
    an = apb.layernorm_a(a)
    att = apb.attention
    q, k, v = att._prep_qkv(q_x=an, kv_x=an, apply_scale=True)          # [..., H, n, d]
    outs = []
    for r0, r1 in blocks(hp.n, rows):
        zr = hp.rows(r0, r1)
        bias = permute_final_dims(apb.linear_nobias_z(apb.layernorm_z(zr)), [2, 0, 1])   # [H, rows, n]
        bias = apb._add_extra_attn_bias_to_chunk(bias, extra_attn_bias, r0, r1)
        bias = apb._align_bias_to_query(bias, an, n_pair_dims=2)
        if len(bias.shape) != len(q.shape):
            bias = bias.unsqueeze(dim=-3)
        o = _attention(q=q[..., r0:r1, :], k=k, v=v, attn_bias=bias, use_efficient_implementation=att.use_efficient_implementation, inplace_safe=False)
        outs.append(o); del zr, bias
    o = torch.cat(outs, dim=-2).transpose(-2, -3)                         # [..., n, H, d]
    return att._wrap_up(o, an)


@torch.no_grad()
def off_pairformer_block(block, s, hp: HostPair, triangle_multiplicative, triangle_attention, inplace_safe, chunk_size, extra_attn_bias=None):
    """== PairformerBlock.forward(s, z, pair_mask=None, ...) with z host-resident (dropout = equality at inference)."""
    off_trimul(block.tri_mul_out, hp)
    off_trimul(block.tri_mul_in, hp)
    off_triatt(block.tri_att_start, hp, transposed=False, triangle_attention=triangle_attention, chunk_size=chunk_size, inplace_safe=inplace_safe)
    off_triatt(block.tri_att_end, hp, transposed=True, triangle_attention=triangle_attention, chunk_size=chunk_size, inplace_safe=inplace_safe)
    off_transition(block.pair_transition, hp)
    if block.c_s > 0 and s is not None:
        s = s + off_apb(block.attention_pair_bias, s, hp, extra_attn_bias)
        s = s + block.single_transition(s)
    return s


@torch.no_grad()
def off_pairformer_stack(stack, s, hp: HostPair, triangle_multiplicative="torch", triangle_attention="torch", inplace_safe=True, chunk_size=None,
                         extra_attn_bias=None, tag="stack", every=1, on_block_done=None):
    nb = len(stack.blocks)
    with base_autocast(hp):                                                # the blocks run in the precision the pair was produced in (bf16 host pair: bf16 autocast, as the stock stack runs it)
        for bi, block in enumerate(stack.blocks):
            t = time.time()
            s = off_pairformer_block(block, s, hp, triangle_multiplicative, triangle_attention, inplace_safe, chunk_size, extra_attn_bias)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            if bi % every == 0 or bi == nb - 1:
                log(f"{tag} block {bi+1}/{nb} {time.time()-t:.1f}s {gpu_mem_str()} rss={host_rss_gib()[0]:.1f}GiB")
            if on_block_done is not None:
                on_block_done(bi, s)
    return s


def base_autocast(hp):
    """The autocast region a host-resident pair stack runs its blocks in: a bf16 ``HostPair`` was produced by the model under bf16 autocast
    (upstream ``--dtype bf16``) and its blocks run under ``torch.autocast('cuda', bfloat16)`` like the stock stack in that model call — whatever
    the autocast state of the frame that reads the rows back (the confidence stage reads them after upstream's autocast-disabled diffusion
    stage); an fp32 ``HostPair`` leaves the caller's state untouched (a null context: the fp32 base's bytes)."""
    import contextlib
    if getattr(hp, "dtype", None) == torch.bfloat16 and torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


# --------------------------------------------------------------------------------------------------------------------------
# Structural-token stage
# --------------------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def off_structural_expander(expander, input_feature_dict, s_inputs_res, s_res, z_res, rows=None):
    """== StructuralTokenExpander.forward(input_feature_dict, s_inputs_res, s_res, z_res) (1.1.1, chunked pair path =
    `_make_structural_pair_activations_chunked`) with each z_struct row chunk written to a HostPair instead of torch.cat.
    z_res may be a GPU tensor or a HostPair (then parent rows are gathered on the host, parent columns on the GPU)."""
    self = expander
    parent = input_feature_dict["parent_residue_idx"].long()
    role = input_feature_dict["subtoken_role_id"].long()
    s_inputs_struct = self._gather_parent_single(s_inputs_res, parent) + self.single_input_role_embedding(role).to(dtype=s_inputs_res.dtype)
    s_parent = self._gather_parent_single(s_res, parent)
    s_struct = s_parent + self.single_split_mlp(s_parent) + self.single_role_embedding(role).to(dtype=s_parent.dtype)
    n_struct = role.shape[-1]
    chunk_size = min((rows or self.pair_chunk_size or CFG["rows"]), n_struct)
    context = self._build_structural_pair_context(input_feature_dict=input_feature_dict, role=role, parent=parent)
    zres_is_host = isinstance(z_res, HostPair)
    c = z_res.c if zres_is_host else int(z_res.shape[-1])
    dev = s_res.device
    hp = HostPair(n_struct, c, dtype=s_res.dtype, device=dev)
    parent_cpu = parent.cpu() if zres_is_host else None
    attn_bias_chunks = []
    for start in range(0, n_struct, chunk_size):
        end = min(start + chunk_size, n_struct)
        row_index = torch.arange(start, end, device=parent.device)
        pair_features = self._build_structural_pair_features_for_rows(context=context, row_index=row_index)
        if zres_is_host:
            g = z_res.t.index_select(0, parent_cpu[start:end])
            STATS["h2d_bytes"] += g.numel() * 4
            z_chunk = g.to(dev).index_select(-2, parent)
        else:
            z_chunk = self._gather_parent_pair_rows(z=z_res, parent=parent, row_index=row_index)
        delta = self._pair_project_by_role(z=z_chunk, role=role, pair_features=pair_features, row_index=row_index)
        if delta is not None:
            z_chunk = z_chunk + delta
        z_chunk = z_chunk + self._make_pair_init_bias(pair_features, dtype=z_chunk.dtype)
        hp.put_rows(start, end, z_chunk)
        attn_bias_chunks.append(self._make_attention_bias(pair_features, dtype=z_chunk.dtype))
        del z_chunk, delta, pair_features
    return s_inputs_struct, s_struct, hp, {"structural_pair_attn_bias": torch.cat(attn_bias_chunks, dim=0)}


@torch.no_grad()
def off_diffusion_prepare_pair_cache(dc, relp, hp: HostPair, rows=None):
    """== DiffusionConditioning.prepare_cache(relp_feature, z_trunk) with z_trunk host-resident; relp rows materialised
    lazily per block.  Returns the GPU-resident pair_z [n, n, c_z_pair_diffusion] after the two (stock, in-place) transitions."""
    rows = rows or CFG["rows"]
    n, dev = hp.n, hp.device
    cpd = dc.c_z_pair_diffusion
    pair_z = torch.empty((n, n, cpd), dtype=torch.float32, device=dev)
    for r0, r1 in blocks(n, rows):
        zr = hp.rows(r0, r1)
        zt = dc._project_z_trunk(zr) if dc.compress_pair_z else zr
        rel = relp.materialize(slice(r0, r1), slice(None)) if hasattr(relp, "materialize") else relp[..., r0:r1, :, :]
        rel = dc.relpe.linear_no_bias(rel)
        pair_z[r0:r1] = dc._project_pair_z(torch.cat([zt, rel], dim=-1))
        del zr, zt, rel
    return dc._apply_pair_z_transitions(pair_z, inplace_safe=False)


# --------------------------------------------------------------------------------------------------------------------------
# checkpoints
# --------------------------------------------------------------------------------------------------------------------------
def ckpt_save(name, obj):
    d = CFG["ckpt_dir"]
    if not d:
        return
    try:
        os.makedirs(d, mode=0o755, exist_ok=True); p = os.path.join(d, name + ".pt"); t = time.time(); tmp = p + ".tmp"
        _ckpt_write(obj, tmp, p)                        # the file + its sha256 record (end of this file)
        log(f"CKPT saved {name} -> {os.path.getsize(p)/2**30:.2f} GiB in {time.time()-t:.0f}s")
    except Exception as e:
        if is_oom(e): raise
        fallback(f"CKPT save {name} failed"); log(f"CKPT save {name} FAILED: {e}")


def ckpt_load(name):
    p = os.path.join(CFG["ckpt_dir"], name + ".pt")
    t = time.time(); obj = _ckpt_read(p)            # None: refused by name (end of this file) = the absent checkpoint
    log(f"CKPT loaded {name} ({os.path.getsize(p)/2**30:.2f} GiB) in {time.time()-t:.0f}s") if obj is not None else None
    return obj


# --------------------------------------------------------------------------------------------------------------------------
# diffusion pair lever ("diffz"): under the pair cache, pair_z is constant across the 200 steps.  Stock (1.1.1) nevertheless (a) clones
# pair_z at every step inside DiffusionConditioning.forward when inplace_safe and (b) recomputes LayerNorm(pair_z) (DiffusionModule.normalize)
# at every step -> two extra [n_s, n_s, 128] fp32 tensors per step.  Nothing downstream writes into pair_z or its
# LayerNorm (they are read by linear layers only), so skipping the clone and memoising the LayerNorm output is the same arithmetic on the
# same inputs (same kernels, same shapes) -> bit-preserving by design (not separately claimed exact: the unit is tier 2).
# --------------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------------
# BIGLN guard.  torch 2.7.1's CUDA layer_norm returns WRONG values for rows whose linear element index is >= 2^32
# ([6000,6000,128] fp32: rows >= 5,593 off by O(1); Linear / advanced indexing / index_select /
# elementwise / transpose are correct).  On one device the diffusion pair-conditioning tensor [n_s, n_s, 128] crosses 2^32
# elements at n_s >= 5,793 (N >~ 3,000 residues) -> the tail chains of such an input receive garbage pair conditioning and
# collapse.  Stock never reaches this size (OOM first); multi-rank runs shard the tensor.
# Fix = the same LayerNorm applied per leading-dim block (per-row arithmetic is unchanged: mean/var are per row; small-N check in
# blocked == unblocked bitwise below the limit).  Installed as a wrapper around torch.nn.functional.layer_norm so
# EVERY call site (DiffusionModule.normalize incl. the diffz memo, OpenDDE's LayerNorm class, torch.nn.LayerNorm) is covered.
_BIGLN = {"limit": int(os.environ.get("ODDE_OFFLOAD_BIGLN_LIMIT", str(1 << 31))), "hits": 0, "calls": 0, "installed": False}
_stock_layer_norm = torch.nn.functional.layer_norm

def _blocked_layer_norm(input, normalized_shape, weight=None, bias=None, eps=1e-5):
    _BIGLN["calls"] += 1                      # RAN-OR-REFUSE: layer_norm calls routed through the guard (hits = the blocked ones)
    if input.numel() < _BIGLN["limit"] or input.dim() < 2 or not input.is_cuda:
        return _stock_layer_norm(input, normalized_shape, weight, bias, eps)
    inner = 1
    for d in input.shape[1:]:
        inner *= int(d)
    rows = max(1, (_BIGLN["limit"] // 2) // max(inner, 1))
    out = torch.empty_like(input)
    for i in range(0, int(input.shape[0]), rows):
        out[i:i + rows] = _stock_layer_norm(input[i:i + rows], normalized_shape, weight, bias, eps)
    _BIGLN["hits"] += 1
    if _BIGLN["hits"] <= 3:
        log(f"BIGLN: layer_norm on {tuple(input.shape)} ({input.numel()/2**30:.1f} Gi elements) computed in leading-dim blocks of {rows}")
    return out

def _install_bigln():
    if _BIGLN["installed"]:
        return
    torch.nn.functional.layer_norm = _blocked_layer_norm
    try:
        import torch.nn.modules.normalization as _N
        if getattr(_N, "F", None) is not None and _N.F is torch.nn.functional:
            pass   # nn.LayerNorm.forward calls F.layer_norm via the module attribute -> already covered
    except Exception:
        pass
    _BIGLN["installed"] = True
    log(f"BIGLN guard installed: F.layer_norm inputs >= {_BIGLN['limit']/2**30:.1f} Gi elements are normalised per leading-dim block")


class _MemoLayerNorm(torch.nn.Module):
    """DiffusionModule.normalize(pair_z) memoised across the diffusion steps (pair_z is constant under the pair cache).
    For a big 3-D input the memo builds P = permute(LN(pair_z), [2,0,1]) DIRECTLY, row block by row block (peak = P + one block; the
    per-row LayerNorm arithmetic is unchanged -> identical bits to the blocked/unblocked LN), returns the view P.permute(1,2,0) so that stock's
    permute_final_dims(., [2,0,1]).contiguous() is a no-copy no-op, and (ODDE_OFFLOAD_DIFFZ_RELEASE=1, default) releases the GPU storage of pair_z
    after P exists: pair_z has no other consumer during sampling (DiffusionConditioning returns it untouched; f_forward only normalises it), so the
    diffusion steady state holds ONE pair-sized tensor (P) instead of three (pair_z + LN image + per-step permuted copy)."""
    def __init__(self, inner):
        super().__init__(); self.inner = inner; self._key = None; self._val = None; self._released = False
    def forward(self, x):
        minbytes = int(os.environ.get("ODDE_OFFLOAD_DIFFZ_MINBYTES", str(1 << 28)))
        if self._released and x.numel() == 0 and self._val is not None:      # pair_z storage released after the image was built (steps 2..200)
            return self._val.permute(1, 2, 0)
        big3 = x.dim() == 3 and x.numel() * x.element_size() >= minbytes and CFG.get("diffz_perm", True)
        if big3 and self._val is not None and self._key == (x.data_ptr(), tuple(x.shape), x.dtype, x.device, x._version):
            return self._val.permute(1, 2, 0)
        key = (x.data_ptr(), tuple(x.shape), x.dtype, x.device, x._version)
        if not big3:
            if self._key == key and self._val is not None:
                return self._val
            self._val = None; self._key = None
            out = self.inner(x)
            if x.numel() * x.element_size() >= minbytes:
                self._key, self._val = key, out
            return out
        # big 3-D pair input: build the permuted image block-wise
        self._val = None; self._key = None
        n0, n1, C = x.shape
        P = torch.empty((C, n0, n1), dtype=x.dtype, device=x.device)
        rows = max(1, int(os.environ.get("ODDE_OFFLOAD_DIFFZ_ROWS", "0")) or max(1, (1 << 28) // max(1, n1 * C)))   # ~256 Mi elements per block
        for r0 in range(0, n0, rows):
            r1 = min(n0, r0 + rows)
            P[:, r0:r1, :].copy_(self.inner(x[r0:r1]).permute(2, 0, 1))
        self._key, self._val = key, P
        STATS["diffz_images"] = STATS.get("diffz_images", 0) + 1
        log(f"diffz: permuted LayerNorm image of pair_z {tuple(x.shape)} built block-wise ({P.numel()*x.element_size()/2**30:.1f} GiB, {rows} rows/block)")
        if CFG.get("diffz_release", True):
            try:
                x.data = torch.empty(0, dtype=x.dtype, device=x.device); self._released = True
                if x.is_cuda: torch.cuda.empty_cache()
                log("diffz: released the GPU storage of pair_z after building its LayerNorm image (pair_z has no other consumer during sampling)")
            except Exception as e:
                if is_oom(e): raise
                fallback("diffz: pair_z release skipped"); log(f"diffz: pair_z release skipped ({type(e).__name__}: {e})")
        return P.permute(1, 2, 0)


def _install_diffz():
    from opendde.model.modules import diffusion as D
    DC, DM = D.DiffusionConditioning, D.DiffusionModule
    if getattr(DC, "_oo_diffz", False):
        return
    stock_dc_forward = DC.forward
    def dc_forward(self, t_hat_noise_level, relp_feature, s_inputs, s_trunk, z_trunk, pair_z, inplace_safe=False, use_conditioning=True):
        if pair_z is not None and inplace_safe:
            # == stock with the defensive per-step clone skipped (pair_z is read-only downstream); single path identical to stock inplace branch
            count("diffz")                                                 # RAN-OR-REFUSE: diffz engaged (once per DiffusionConditioning call)
            single_s = torch.cat(tensors=[s_trunk, s_inputs], dim=-1)
            single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
            noise_ratio = (t_hat_noise_level / self.sigma_data).clamp(min=1e-10)
            noise_n = self.fourier_embedding(t_hat_noise_level=torch.log(input=noise_ratio) / 4).to(single_s.dtype)
            single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(self.layernorm_n(noise_n)).unsqueeze(dim=-2)
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
            return single_s, pair_z
        fallback("diffz not applied: pair_z is None or not inplace_safe (stock per-step clone)")
        return stock_dc_forward(self, t_hat_noise_level, relp_feature, s_inputs, s_trunk, z_trunk, pair_z, inplace_safe=inplace_safe, use_conditioning=use_conditioning)
    DC.forward = dc_forward
    stock_f_forward = DM.f_forward
    def f_forward(self, *a, **kw):
        if hasattr(self, "normalize") and not isinstance(self.normalize, _MemoLayerNorm):
            self.normalize = _MemoLayerNorm(self.normalize)
        return stock_f_forward(self, *a, **kw)
    DM.f_forward = f_forward
    DC._oo_diffz = True
    log("diffz installed: DiffusionConditioning per-step pair_z clone skipped; DiffusionModule.normalize(pair_z) memoised across steps")


# --------------------------------------------------------------------------------------------------------------------------
# Model patches
# --------------------------------------------------------------------------------------------------------------------------
_INSTALLED = False


def _stock_structural_feature_dict_pre(input_feature_dict):
    """verbatim: OpenDDE.expand_to_structural_tokens (1.1.1, opendde.py L456-471) -- the part BEFORE the expander call."""
    structural_feature_dict = dict(input_feature_dict)
    for residue_feature in ["token_index", "asym_id", "residue_index", "entity_id", "sym_id", "atom_to_token_idx", "atom_to_tokatom_idx",
                            "has_frame", "frame_atom_index", "pae_rep_atom_mask", "distogram_rep_atom_mask"]:
        structural_feature_dict[f"residue_level_{residue_feature}"] = input_feature_dict[residue_feature]
    return structural_feature_dict


def _stock_structural_feature_dict_post(input_feature_dict, structural_feature_dict, parent, structural_pair_features):
    """verbatim: OpenDDE.expand_to_structural_tokens (1.1.1, opendde.py L490-516) -- the part AFTER the expander call (relp generated by the caller)."""
    structural_feature_dict["token_index"] = input_feature_dict["structural_token_index"].long()
    structural_feature_dict["atom_to_token_idx"] = input_feature_dict["atom_to_structural_token_idx"].long()
    structural_feature_dict["atom_to_tokatom_idx"] = input_feature_dict["atom_to_structural_tokatom_idx"].long()
    for token_feature in ["asym_id", "residue_index", "entity_id", "sym_id"]:
        structural_feature_dict[token_feature] = input_feature_dict[token_feature].index_select(dim=-1, index=parent)
    structural_feature_dict["has_frame"] = input_feature_dict["structural_has_frame"]
    structural_feature_dict["frame_atom_index"] = input_feature_dict["structural_frame_atom_index"]
    structural_feature_dict["pae_rep_atom_mask"] = input_feature_dict["structural_pae_rep_atom_mask"].long()
    structural_feature_dict["distogram_rep_atom_mask"] = input_feature_dict["structural_distogram_rep_atom_mask"].long()
    for feature_name, feature_value in structural_pair_features.items():
        structural_feature_dict[feature_name] = feature_value
    return structural_feature_dict


def install():
    """Monkey-patch OpenDDE 1.1.1 so that the stages named in ODDE_OFFLOAD run with host-resident pair tensors."""
    global _INSTALLED
    if _INSTALLED or not CFG["stages"]:
        return
    from opendde.model import opendde as M
    OpenDDE = M.OpenDDE
    stages = CFG["stages"]
    log(f"install: stages={sorted(stages)} cfg={ {k: v for k, v in CFG.items() if k != 'stages'} }")

    if "struct" in stages:
        def expand_to_structural_tokens(self, input_feature_dict, s_inputs, s, z, inplace_safe=False, chunk_size=None, lazy_relp=False):
            """== OpenDDE.expand_to_structural_tokens (opendde.py L422-538) with z_res / z_struct host-resident.  The structural relp is
            generated lazy whatever ``lazy_relp`` says (_main_inference_loop passes True): the row-block consumers materialise relp rows,
            never the [n_s, n_s, 139] one-hot this stage exists to avoid.  ``chunk_size`` is the caller's (upstream's N^2-bounded value
            for the 2N structural tokens)."""
            if not self.enable_structural_token_expansion:
                return input_feature_dict, s_inputs, s, z
            assert self._maybe_foldcp_mesh() is None, "ODDE_OFFLOAD is single-process; foldcp mesh not supported"
            count("struct")                                                # RAN-OR-REFUSE: pair_offload_struct engaged (one per forward)
            required = ["parent_residue_idx", "subtoken_role_id", "structural_token_index", "atom_to_structural_token_idx",
                        "atom_to_structural_tokatom_idx", "structural_distogram_rep_atom_mask", "structural_pae_rep_atom_mask",
                        "structural_has_frame", "structural_frame_atom_index"]
            missing = [k for k in required if k not in input_feature_dict]
            if missing:
                raise KeyError("Structural token expansion is enabled, but input_feature_dict is missing required structural feature(s): " + ", ".join(missing))
            parent = input_feature_dict["parent_residue_idx"].long()
            if CFG["free_templ"]:
                freed = 0
                for k in [k for k in list(input_feature_dict.keys()) if k.startswith("template_") and torch.is_tensor(input_feature_dict[k]) and input_feature_dict[k].dim() >= 3]:
                    freed += input_feature_dict[k].numel() * input_feature_dict[k].element_size(); del input_feature_dict[k]
                if freed:
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    count("free_templ"); STATS["free_templ_bytes"] = STATS.get("free_templ_bytes", 0) + freed   # RAN-OR-REFUSE: free_templ engaged
                    log(f"free_templ: released {freed/2**30:.1f} GiB of template pair features after the trunk")
            sfd = _stock_structural_feature_dict_pre(input_feature_dict)
            resumed = False
            if CFG["resume"] == "refiner" and CFG["ckpt_dir"] and os.path.exists(os.path.join(CFG["ckpt_dir"], "refiner.pt")) and (ck := ckpt_load("refiner")) is not None:
                dev = s.device
                hp = HostPair(ck["z_struct"].shape[0], ck["z_struct"].shape[-1], dtype=ck["z_struct"].dtype, device=dev)
                hp.t.copy_(ck["z_struct"]); s_inputs_s, s_s = ck["s_inputs"].to(dev), ck["s"].to(dev)
                spf = {"structural_pair_attn_bias": ck["structural_pair_attn_bias"].to(dev)}; resumed = True
            else:
                with stage_timer("structural_expander"):
                    s_inputs_s, s_s, hp, spf = off_structural_expander(self.structural_token_expander, input_feature_dict, s_inputs, s, z)
                    log(f"structural tokens n_struct={hp.n} (N={int(parent.shape[-1])})")
            sfd = _stock_structural_feature_dict_post(input_feature_dict, sfd, parent, spf)
            sfd = self.relative_position_encoding.generate_relp(sfd, lazy=True)
            if self.enable_structural_token_refiner and not resumed:
                with stage_timer("structural_refiner", n_struct=hp.n):
                    s_s = off_pairformer_stack(self.structural_token_refiner, s_s, hp,
                                               triangle_multiplicative=self.configs.triangle_multiplicative, triangle_attention=self.configs.triangle_attention,
                                               inplace_safe=inplace_safe, chunk_size=chunk_size, extra_attn_bias=sfd.get("structural_pair_attn_bias", None), tag="refiner")
                if CFG["ckpt_dir"] and hp.n * hp.n * hp.c * 4 / 2**30 <= CFG["ckpt_max_gb"]:
                    ckpt_save("refiner", {"s_inputs": s_inputs_s.cpu(), "s": s_s.cpu(), "z_struct": hp.t, "structural_pair_attn_bias": sfd["structural_pair_attn_bias"].cpu()})
            self.drop_residue_only_features_for_structural_branch(sfd)
            return sfd, s_inputs_s, s_s, hp
        OpenDDE.expand_to_structural_tokens = expand_to_structural_tokens

        stock_prepare = OpenDDE.prepare_diffusion_cache_for_sampling
        def prepare_diffusion_cache_for_sampling(self, *, input_feature_dict, z, foldcp_mesh=None, diffusion_z_spec=None, **extra):
            """== OpenDDE.prepare_diffusion_cache_for_sampling (1.1.1, opendde.py L1347-1454) when z (structural pair tensor) is host-resident:
            the diffusion shared-vars cache (== --enable_cache true) is REQUIRED (the sampler can not read a host tensor at every step) and
            pair_z is built row-blocked from the host z; p_lm/c_l exactly as stock from the cached pair_z."""
            if not isinstance(z, HostPair):
                return stock_prepare(self, input_feature_dict=input_feature_dict, z=z, foldcp_mesh=foldcp_mesh, diffusion_z_spec=diffusion_z_spec, **extra)
            assert foldcp_mesh is None, "ODDE_OFFLOAD is single-process; foldcp mesh not supported"
            from opendde.utils.torch_utils import autocasting_disable_decorator
            if not self.enable_diffusion_shared_vars_cache:
                log("note: offload forces the diffusion shared-vars cache (identical to --enable_cache true)")
            cache = {"pair_z_spec": None, "atom_window_spec": None, "diffusion_attn_bias": None}   # the stock single-device cache keys
            count("diff_cache")                                            # RAN-OR-REFUSE: the diffusion pair cache was built from the host z
            if torch.cuda.is_available():
                torch.cuda.empty_cache(); fr, tot = torch.cuda.mem_get_info()
                log(f"before diffusion pair cache: allocated={torch.cuda.memory_allocated()/2**30:.1f} GiB reserved={torch.cuda.memory_reserved()/2**30:.1f} GiB free={fr/2**30:.1f} GiB; pair_z needs {z.n*z.n*128*4/2**30:.1f} GiB (+ LayerNorm image in sampling)")
            with stage_timer("diffusion_pair_cache", n_struct=z.n):
                relp = input_feature_dict["relp"]
                dc = self.diffusion_module.diffusion_conditioning
                cache["pair_z"] = autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
                    lambda relp_, z_, inplace: off_diffusion_prepare_pair_cache(dc, relp_, z_))(relp, z, False)
                cache["p_lm/c_l"] = autocasting_disable_decorator(self.configs.skip_amp.sample_diffusion)(
                    self.diffusion_module.atom_attention_encoder.prepare_cache)(
                    ref_pos=input_feature_dict["ref_pos"], ref_charge=input_feature_dict["ref_charge"], ref_mask=input_feature_dict["ref_mask"],
                    ref_element=input_feature_dict["ref_element"], ref_atom_name_chars=input_feature_dict["ref_atom_name_chars"],
                    atom_to_token_idx=input_feature_dict["atom_to_token_idx"], d_lm=input_feature_dict["d_lm"], v_lm=input_feature_dict["v_lm"],
                    pad_info=input_feature_dict["pad_info"], r_l=True, z=cache["pair_z"], inplace_safe=False)
                log(f"diffusion pair cache pair_z {tuple(cache['pair_z'].shape)} {cache['pair_z'].numel()*4/2**30:.1f} GiB on GPU; {gpu_mem_str()}")
            return cache
        OpenDDE.prepare_diffusion_cache_for_sampling = prepare_diffusion_cache_for_sampling

        stock_run_sample = OpenDDE.run_sample_diffusion_stage
        def run_sample_diffusion_stage(self, **kw):
            if isinstance(kw.get("z"), HostPair):
                assert kw["cache"]["pair_z"] is not None, "offload: diffusion pair cache missing"
                kw = dict(kw); kw["z"] = None
            with stage_timer("diffusion_sampling"):
                return stock_run_sample(self, **kw)
        OpenDDE.run_sample_diffusion_stage = run_sample_diffusion_stage

    if CFG["diffz"]:
        _install_bigln()
        _install_diffz()

    if "trunk" in stages or "conf" in stages:
        try:
            from odde_offload_trunk import install_trunk
            install_trunk(stages)
        except ImportError as e:
            fallback("trunk/conf offload not importable: trunk + confidence run stock"); log(f"WARNING: trunk/conf offload requested but odde_offload_trunk is not importable ({e}); trunk + confidence run STOCK")

    # stage timers (+ trunk checkpoint / resume) around the stock trunk and confidence stages
    stock_gpo = OpenDDE.get_pairformer_output
    def get_pairformer_output(self, *a, **kw):
        if CFG["resume"] in ("trunk", "refiner") and CFG["ckpt_dir"] and os.path.exists(os.path.join(CFG["ckpt_dir"], "trunk.pt")) and (ck := ckpt_load("trunk")) is not None:
            dev = next(self.parameters()).device
            z = ck["z"]
            z = HostPair.from_tensor(z) if ck.get("z_is_host") else z.to(dev)
            return ck["s_inputs"].to(dev), ck["s"].to(dev), z
        with stage_timer("trunk"):
            s_inputs, s, z = stock_gpo(self, *a, **kw)
        zbytes = (z.t.numel() if isinstance(z, HostPair) else z.numel()) * 4 / 2**30
        if CFG["ckpt_dir"] and zbytes <= CFG["ckpt_max_gb"]:
            ckpt_save("trunk", {"s_inputs": s_inputs.cpu(), "s": s.cpu(), "z": (z.t if isinstance(z, HostPair) else z.cpu()), "z_is_host": isinstance(z, HostPair)})
        return s_inputs, s, z
    OpenDDE.get_pairformer_output = get_pairformer_output
    stock_conf = OpenDDE.run_confidence_head_stage
    def run_confidence_head_stage(self, *a, **kw):
        with stage_timer("confidence"):
            return stock_conf(self, *a, **kw)
    OpenDDE.run_confidence_head_stage = run_confidence_head_stage

    import atexit
    atexit.register(lambda: log("SUMMARY " + json.dumps({"h2d_GiB": round(STATS['h2d_bytes']/2**30, 1), "d2h_GiB": round(STATS['d2h_bytes']/2**30, 1),
                                                          "strided_GiB": round(STATS['strided_bytes']/2**30, 1), "calls": {k: v for k, v in STATS.items() if k.startswith("calls_")},
                                                          "bigln": {k: _BIGLN[k] for k in ("installed", "calls", "hits")},
                                                          "diffz_images": STATS.get("diffz_images", 0), "fallbacks": STATS["fallbacks"], "stages": STATS["stages"]})))
    _INSTALLED = True


# --------------------------------------------------------------------------------------------------------------------------
# checkpoint files on disk. torch.load unpickles, so a checkpoint is read back only when its bytes are the ones ckpt_save wrote and no
# other account could have written it: ckpt_save records the file's sha256 beside it (<name>.pt.sha256, hashed while torch.save streams);
# ckpt_load refuses a file whose sha256 differs or is missing, and a file or directory writable by group or other or — unless this process
# is uid 0 (a container's root reading a bind-mounted host directory) — owned by neither this uid nor root. A refusal is one stderr line
# (what, why, the fix) and then the absent checkpoint: the stage runs. The sha256 catches truncation and corruption, not a writer of the
# directory — hence the owner / mode rule. (Kept at the end of the file: the registry names lines above.)
# --------------------------------------------------------------------------------------------------------------------------
class _Sha256Writer(object):
    """The file object torch.save streams into: sha256 of the bytes as they are written (a checkpoint is never re-read to hash it); mode 0644
    whatever the umask."""

    def __init__(self, path):
        import hashlib
        self.f, self.h = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), "wb"), hashlib.sha256()

    def write(self, b):
        self.h.update(b)
        return self.f.write(b)

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.close()


def _ckpt_write(obj, tmp, p):
    w = _Sha256Writer(tmp)
    try:
        torch.save(obj, w)
    finally:
        w.close()
    with os.fdopen(os.open(p + ".sha256.tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), "w") as f:
        f.write(w.h.hexdigest() + "\n")
    os.replace(p + ".sha256.tmp", p + ".sha256"); os.replace(tmp, p)   # record first: a crash between the two leaves a mismatch (refused, the stage runs), never a checkpoint read unchecked


def _ckpt_refusal(p):
    """Why the checkpoint ``p`` must not be unpickled (reason + fix), or None."""
    import hashlib
    uid, d = os.geteuid(), os.path.dirname(os.path.abspath(p))
    for what, q in (("directory", d), ("file", p)):
        st = os.stat(q)
        if st.st_mode & 0o022:
            return f"{what} {q} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {q}"
        if uid != 0 and st.st_uid not in (uid, 0):
            return f"{what} {q} belongs to uid {st.st_uid}, not to this process (uid {uid}) or root; fix: name an ODDE_OFFLOAD_CKPT_DIR of your own, or chown {q}"
    try:
        with open(p + ".sha256") as f:
            want = f.read().strip()
    except OSError:
        want = ""
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != want:
        return f"its sha256 is not the one recorded in {p}.sha256 (truncated, corrupted, or written without a record); fix: none needed, the stage runs and the checkpoint is written again"
    return None


def _ckpt_read(p):
    why = _ckpt_refusal(p)
    if why is not None:
        print(f"[odde_offload] REFUSED checkpoint {p}: {why} — not unpickled, treated as absent", file=sys.stderr, flush=True)
        return None
    return torch.load(p, map_location="cpu", weights_only=False)
