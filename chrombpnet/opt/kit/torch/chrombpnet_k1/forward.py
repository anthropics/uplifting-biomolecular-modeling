"""ChromBPNet forward on the K1 Triton kernels: the bias + accessibility composition (K1ChromBPNet), a fixed-shape batch predictor (predict),
and apply(), which builds the model from stock's two .h5 model files and makes the shipped Triton cache available to the process (or says why
the kernels compile in-process instead). cache_witness() reports afterwards whether the shipped cache served every kernel."""
import os, time, hashlib, json, shutil, tempfile
import numpy as np, pandas as pd, torch
from .kernels import K1BPNetV3, logsumexp_tf
from ._te_guard import TE_GUARD
from ._oom import is_oom                                            # the kit's one out-of-memory classifier: a handler below that reroutes re-raises an OOM first

NARROWPEAK = ["chr", "start", "end", "name", "score", "strand", "signalValue", "pValue", "qValue", "summit"]
INPUT_LEN = 2114; OUTPUT_LEN = 1000
# The Triton kernels index the batch as n*C*L in int32, so n*C*L must stay < 2^31 (at C=512, L=2114: <= 1984 units per call). predict() splits any
# batch above K1_MAX_BATCH (1024) and prints the cap once; a direct forward past the bound raises ValueError (check_batch_bound) instead of faulting
# inside a kernel. A 64-bit index would change kernels.py, whose sha256 keys every shipped cache.
K1_MAX_BATCH = 1024; _INT32_ELEMENTS = 2 ** 31; _CAP_PRINTED = [False]
PRECISIONS = ("ieee", "tf32")   # the conv stack's arithmetic: "ieee" = fp32 FFMA chains in stock's deterministic summation order (kernels.py); "tf32" = tensor-core dots (kernels_tc.py), the precision class stock runs in by default
def _tc():
    from . import kernels_tc; return kernels_tc
def check_batch_bound(x):
    """x: (N, C, L). Raise ValueError when n*C*L — or n*512*L, the widest layer of the stack — reaches 2^31, the kernels' int32 index bound."""
    n, c, l = int(x.shape[0]), int(x.shape[1]), int(x.shape[2])
    if n * 512 * l >= _INT32_ELEMENTS or n * c * l >= _INT32_ELEMENTS:
        raise ValueError(f"[chrombpnet_k1] batch {n} x (C 512, L {l}) exceeds the kernels' int32 index bound (n*C*L < 2^31 => <= {_INT32_ELEMENTS // (512 * l)} units per call); predict() splits batches at K1_MAX_BATCH={K1_MAX_BATCH}")


class K1ChromBPNet(torch.nn.Module):
    """bias + accessibility sub-models on the K1 kernels; profile = acc profile + centre-cropped bias profile; logcounts = logsumexp of the two counts
    heads in TF's reduce_logsumexp form (logsumexp_tf)."""
    def __init__(self, chrombpnet_port, counts_mode="exact", tile=None, head="serial", tile_selected_by=None, counts_form=None, precision="ieee"):
        super().__init__()
        if precision not in PRECISIONS: raise ValueError(f"[chrombpnet_k1] precision {precision!r} — one of {PRECISIONS}")
        self.precision = precision              # "ieee": the fp32 FFMA chains (kernels.py; bitwise to stock's deterministic run) | "tf32": the tensor-core convs (kernels_tc.py; stock's own shipped precision class)
        if counts_form is None: counts_form, self.counts_line = select_counts_form()   # EVERY constructor path resolves this arch's counts order, so a model built directly (not through apply()) gets the same order apply() would pass
        self.counts_form = counts_form            # the per-arch dense summation order of the counts head (None = the H100 order)
        if tile is None: tile, self.tile_selected_by = select_tile(None)          # by GPU arch (arch_tiles.json); an unlisted arch gets a tile sized by its shared memory (select_tile)
        else: self.tile_selected_by = tile_selected_by or "explicit"             # apply() resolves the tile itself and passes HOW it was chosen (arch map / explicit) — "explicit" only when the caller named a tile
        bm, bn, bk, nw, ns = (int(v) for v in tile.split("x"))
        os.environ.update({"K1_BLOCK_M": str(bm), "K1_BLOCK_N": str(bn), "K1_BLOCK_K": str(bk), "K1_NUM_WARPS": str(nw), "K1_NUM_STAGES": str(ns)})
        from . import kernels; kernels._BM, kernels._BN, kernels._BK, kernels._NW, kernels._NS = bm, bn, bk, nw, ns
        Sub = K1BPNetV3 if precision == "ieee" else _tc().K1BPNetV3TC
        self.acc = Sub(chrombpnet_port.accessibility, counts_mode=counts_mode, head=head, counts_form=counts_form); self.bias = Sub(chrombpnet_port.bias, counts_mode=counts_mode, head=head, counts_form=counts_form)
        self.tile = tile; self.counts_mode = counts_mode; self.head = head
        # The tile is chosen PER CALL from the batch size (this arch's by_batch regimes in arch_tiles.json; an explicit tile= disables the regimes);
        # every tile keeps the same per-output FMA chain, so outputs are bitwise equal across tiles — only the CTA tiling changes. Calls are counted per tile; the policy is printed once.
        self.by_batch = [] if self.tile_selected_by == "explicit" else batch_regime_tiles(); self.tile_calls = {}; self._regime_printed = False
        self.regime_policy = ("explicit tile (no regimes)" if self.tile_selected_by == "explicit" else (("tile by batch: " + ", ".join(f"B<={r['max_B']} {r['tile']}" for r in self.by_batch) + f", else {tile}") if self.by_batch else f"one tile at every B ({tile}; no regimes measured for this arch)"))
        st = os.environ.get("CHROMBPNET_K1_STREAMS", "1")            # "1" (default): the bias sub-model runs on a second CUDA stream | "0": both sub-models on the current stream; anything else is refused
        if st not in ("0", "1"): raise RuntimeError(f"[chrombpnet_k1] CHROMBPNET_K1_STREAMS={st!r} — '0' or '1' only (refused)")
        self.streams = st == "1"; self._s2 = torch.cuda.Stream() if self.streams else None

    def _set_tile(self, t):
        bm, bn, bk, nw, ns = (int(v) for v in t.split("x")); from . import kernels; kernels._BM, kernels._BN, kernels._BK, kernels._NW, kernels._NS = bm, bn, bk, nw, ns

    def forward(self, x):                       # x: (N, 4, 2114) float32 one-hot
        check_batch_bound(x)                    # ValueError past the int32 index bound, never a kernel fault
        t_ = regime_tile(int(x.shape[0]), self.tile, self.by_batch); self.tile_calls[t_] = self.tile_calls.get(t_, 0) + 1
        if t_ != getattr(self, "_tile_now", None): self._set_tile(t_); self._tile_now = t_
        if not self._regime_printed: print(f"[chrombpnet_k1] {self.regime_policy} | this call B={int(x.shape[0])} -> {t_}", flush=True); self._regime_printed = True
        if self.streams:                        # the bias sub-model on a SECOND CUDA stream, concurrent with the 512-wide accessibility model — independent kernels,
            cur = torch.cuda.current_stream()   # hence the same bytes as the serial form; the events order fork and join, record_stream keeps pb/cb valid on the current stream
            ev0 = torch.cuda.Event(); ev0.record(cur)
            with torch.cuda.stream(self._s2):
                self._s2.wait_event(ev0); pb, cb = self.bias(x); ev1 = torch.cuda.Event(); ev1.record(self._s2)
            pa, ca = self.acc(x); cur.wait_event(ev1); pb.record_stream(cur); cb.record_stream(cur)
        else:
            pa, ca = self.acc(x); pb, cb = self.bias(x)
        w = (pb.shape[1] - pa.shape[1]) // 2
        if w > 0: pb = pb[:, w:-w]
        return pa + pb, logsumexp_tf(ca, cb)


def featurize(df, fasta, width=INPUT_LEN):
    """One-hot (N, width, 4) int8 of the windows centred on start+summit, read straight from the FASTA bytes through a memmap + byte LUT using the .fai
    line geometry (bases other than ACGT/acgt -> all zeros); the same array stock's featuriser produces."""
    fai = pd.read_csv(fasta + ".fai", sep="\t", header=None, names=["name", "length", "offset", "linebases", "linewidth"], usecols=[0, 1, 2, 3, 4]).set_index("name")
    mm = np.memmap(fasta, dtype=np.uint8, mode="r"); lut = np.full(256, 4, dtype=np.uint8)
    for ch, v in (("A", 0), ("C", 1), ("G", 2), ("T", 3)): lut[ord(ch)] = v; lut[ord(ch.lower())] = v
    eye = np.identity(5, dtype=np.int8)[:, :4]
    centers = (df["start"].values + df["summit"].values).astype(np.int64); starts = centers - width // 2; chroms = df["chr"].values
    X = np.empty((len(df), width, 4), dtype=np.int8); ar = np.arange(width, dtype=np.int64)
    for chrom in pd.unique(chroms):
        sel = np.where(chroms == chrom)[0]; off, lb, lw = (int(fai.loc[chrom, c]) for c in ("offset", "linebases", "linewidth"))
        pos = starts[sel][:, None] + ar[None, :]; X[sel] = eye[lut[mm[off + (pos // lb) * lw + (pos % lb)]]]
    return X


_DEV_PRINTED = [False]
def _resident(model, X):
    """True when X is a torch tensor already on the model's device. Such an input is read IN PLACE — no H2D copy, no host-side cast; the same
    permute/float kernels as the host path run on it, so the outputs are the same bytes. A numpy array or a CPU tensor takes the host path
    (never a refusal)."""
    if not torch.is_tensor(X): return False
    if X.is_cuda: return True
    try: return X.device == next(model.parameters()).device
    except Exception as e:
        if is_oom(e): raise                                             # an out-of-memory error propagates; the not-resident (copy) route is for every other failure
        return False

def predict(model, X, batch_size=64, pad_to_batch=True, pinned=False, stamp_md5=False):
    """Forward all N units through model in fixed-shape batches of batch_size (the last batch padded with zeros, the padding dropped) and
    return logits (N, 1000), logcounts (N, 1) and a stamp dict describing the call. A batch_size above K1_MAX_BATCH is lowered to it (printed once
    per process). A tensor already on the model's device is read in place (_resident), in (N, L, 4) or (N, 4, L) layout; pinned is then ignored.
    pinned=True: the two result blocks are PINNED host tensors, every batch's heads are copied straight into them (copy_ from the device, no
    pageable staging) and the returned numpy arrays are zero-copy views of those blocks; the input batch is staged through one pinned buffer too.
    Only the copies differ from the default path, not the values. stamp_md5=True adds md5s of both result arrays; otherwise predict() does no
    hashing and no file I/O."""
    N = len(X); calls = 0; padded = 0; batch_requested = batch_size; batch_cap = None
    resident = _resident(model, X); layout = None
    if resident:                                                    # a device-resident tensor: read in place (the contract is stated on the printed line)
        if X.dim() != 3 or (X.shape[2] != 4 and X.shape[1] != 4): raise ValueError(f"[chrombpnet_k1.predict] a device-resident input must be (N, L, 4) [the featurizer layout] or (N, 4, L) [the model layout] with 0/1 one-hot values; got {tuple(X.shape)}")
        layout = "NL4" if X.shape[2] == 4 else "N4L"; pinned = False      # nothing to stage through a pinned host buffer
        if not _DEV_PRINTED[0]: print(f"[chrombpnet_k1.predict] device input: {str(X.dtype).replace('torch.', '')} {tuple(X.shape)} on {X.device} read IN PLACE (no H2D copy, no host cast; CONTRACT v2) — layout {layout} -> (N, 4, L) float32 on the device by the same permute/float kernels as the host path (bitwise by construction); the values must be a 0/1 one-hot", flush=True); _DEV_PRINTED[0] = True
    if batch_size > K1_MAX_BATCH:
        batch_size = K1_MAX_BATCH; batch_cap = f"batch {batch_requested} -> {K1_MAX_BATCH} (K1 int32 index bound n*C*L < 2^31: <= {_INT32_ELEMENTS // (512 * INPUT_LEN)} units per call at C=512, L={INPUT_LEN}; {K1_MAX_BATCH} = the saturation plateau, said once)"
        if not _CAP_PRINTED[0]: print("[chrombpnet_k1.predict] " + batch_cap, flush=True); _CAP_PRINTED[0] = True
    if pinned:
        lt = torch.empty((N, OUTPUT_LEN), dtype=torch.float32, pin_memory=True); ct = torch.empty((N, 1), dtype=torch.float32, pin_memory=True)
        logits = lt.numpy(); logcts = ct.numpy()
        xin = torch.empty((batch_size,) + tuple(X.shape[1:]), dtype=(X.dtype if torch.is_tensor(X) else torch.from_numpy(X[:1]).dtype), pin_memory=True)
    else:
        logits = np.empty((N, OUTPUT_LEN), dtype=np.float32); logcts = np.empty((N, 1), dtype=np.float32)
    torch.cuda.synchronize(); t0 = time.time()
    with torch.no_grad():
        for b0 in range(0, N, batch_size):
            xb = X[b0:b0 + batch_size]; nb = len(xb)
            if pad_to_batch and nb < batch_size:
                if torch.is_tensor(xb): xb = torch.cat([xb, torch.zeros((batch_size - nb,) + tuple(xb.shape[1:]), dtype=xb.dtype, device=xb.device)])
                else: xb = np.concatenate([xb, np.zeros((batch_size - nb,) + xb.shape[1:], dtype=xb.dtype)])
                padded += batch_size - nb
            if resident:                                                # in place — the batch is a device slice; the same permute/float as the host path below
                x = (xb.permute(0, 2, 1) if layout == "NL4" else xb).float().contiguous()
            elif pinned:
                torch.cuda.current_stream().synchronize()      # the previous batch's H2D from this pinned buffer must be complete before the host rewrites it
                xin[:len(xb)].copy_(xb if torch.is_tensor(xb) else torch.from_numpy(xb)); x = xin[:len(xb)].cuda(non_blocking=True).permute(0, 2, 1).float().contiguous()
            else:
                x = (xb if torch.is_tensor(xb) else torch.from_numpy(xb)).cuda().permute(0, 2, 1).float().contiguous()
            p, c = model(x); calls += 1
            if pinned:
                lt[b0:b0 + nb].copy_(p[:nb], non_blocking=True); ct[b0:b0 + nb].copy_(c[:nb], non_blocking=True)
            else:
                logits[b0:b0 + nb] = p[:nb].cpu().numpy(); logcts[b0:b0 + nb] = c[:nb].cpu().numpy()
    torch.cuda.synchronize(); dt = time.time() - t0
    if _FIRST_FORWARD.get('wall_s') is None: _FIRST_FORWARD.update(wall_s=round(dt, 3), n=int(N), batch=int(batch_size))   # the wall time of this process's FIRST predict() (a cold JIT compile lands here); reported by cache_witness()
    # NO file I/O and NO whole-array hashing inside predict(): the Triton cache is recorded once at apply() (cache_snapshot_at_apply) and read again only
    # by cache_witness(); the md5s of the two result arrays are computed only on request (stamp_md5=True).
    md5 = (lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()) if stamp_md5 else (lambda a: None)
    stamp = {"kit": "chrombpnet_k1", "units_in": N, "units_forwarded": N, "forward_calls": calls, "padded_rows": padded, "batch_size": batch_size, "batch_size_requested": batch_requested, "batch_cap": batch_cap, "pad_to_batch": pad_to_batch,
             "input": {"kind": ("device-resident tensor (in place; CONTRACT v2)" if resident else ("host tensor" if torch.is_tensor(X) else "host array")), "layout": (layout or "NL4"), "dtype": str(getattr(X, "dtype", None))},
             "tile": model.tile, "tile_calls": dict(getattr(model, "tile_calls", {})), "regime_policy": getattr(model, "regime_policy", None), "head": model.head, "counts_mode": model.counts_mode, "counts_form": (getattr(model, "counts_form", None) or {"form": "h100_lanes"}), "streams": bool(getattr(model, "streams", False)), "forward_s": dt, "ms_per_unit": dt / N * 1e3, "md5_logits": md5(logits), "md5_logcts": md5(logcts),
             "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32, "tf32_cudnn": torch.backends.cudnn.allow_tf32,
             "device": torch.cuda.get_device_name(0), "triton": _triton_version(),
             "md5_semantics": "md5 of np.ascontiguousarray(arr).tobytes(): logits float32 C-order (N, L), logcts float32 C-order (N, 1)",
             "triton_cache": "not read per call (k1r3): see apply()'s cache_snapshot_at_apply / cache_witness()", "triton_cache_dir": os.environ.get("TRITON_CACHE_DIR"), "pinned_host_buffers": bool(pinned), "stamp_md5": bool(stamp_md5)}
    return logits, logcts, stamp


def _triton_version():
    try:
        import triton; return triton.__version__
    except Exception as e: return f"unavailable: {e!r}"


def triton_cache_record(cache_dir):
    """The Triton cache keys under cache_dir (one dir per compiled kernel variant), each with the sha256 of every cached file (cubin / ptx /
    json / so); None when cache_dir is unset or not a dir. For a dir populated by one process this is exactly the set of kernels that process
    compiled; build_cache_record() stores it as the "entries" of JIT_IDENTITY.json."""
    if not cache_dir or not os.path.isdir(cache_dir): return None
    rec = []
    for key in sorted(os.listdir(cache_dir)):
        d = os.path.join(cache_dir, key)
        if not os.path.isdir(d): continue
        files = {}
        for f in sorted(os.listdir(d)):
            fp = os.path.join(d, f)
            if os.path.isfile(fp): files[f] = hashlib.sha256(open(fp, "rb").read()).hexdigest()
        rec.append({"key": key, "files": files})
    return rec


KERNELS_SHA256_16 = None
def _triton_identity():
    """What Triton 3.0 keys its cache on besides the kernel source: triton_key (a hash of the Triton install's own files), the backend hash
    (ptxas version + capability), the target, and the TRITON_* environment (Triton 3.0 has no get_cache_invalidating_env_vars, so the environ
    is recorded as is, TRITON_CACHE_DIR excluded). READS THE DRIVER (the target), which initialises CUDA: call it only AFTER the shipped entries
    are installed. Any failure other than out-of-memory comes back as {"triton_identity_error": ...} (the shipped cache is then not applicable
    -> cold JIT)."""
    try:
        import triton
        from triton.compiler.compiler import triton_key, make_backend
        target = triton.runtime.driver.active.get_current_target(); be = make_backend(target)
        return {"triton_key": triton_key(), "backend_hash": be.hash(), "target": f"{target.backend}-{target.arch}-{target.warp_size}", "triton_env": {k: v for k, v in os.environ.items() if k.startswith("TRITON_") and k != "TRITON_CACHE_DIR"}}
    except Exception as e:
        if is_oom(e): raise                                             # an out-of-memory error propagates; the identity-error record (-> shipped cache refused, cold JIT) is for every other failure
        return {"triton_identity_error": f"{type(e).__name__}: {str(e)[:100]}"}


KERNELS_JIT_SHA256_16 = None
def _kernels_jit_sha():
    """sha256[:16] over the @triton.jit function sources of kernels.py (the text Triton itself hashes), computed once. Printed beside the whole-file
    pin (_kernels_sha) so a pin mismatch can be read as "the kernels changed" vs "only the Python helpers changed"; not itself compared as a pin."""
    global KERNELS_JIT_SHA256_16
    if KERNELS_JIT_SHA256_16 is None:
        from .cache_identity import kernels_jit_sha256_16
        KERNELS_JIT_SHA256_16 = kernels_jit_sha256_16(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py"))
    return KERNELS_JIT_SHA256_16
def _kernels_sha():
    global KERNELS_SHA256_16
    if KERNELS_SHA256_16 is None: KERNELS_SHA256_16 = hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "kernels.py"), "rb").read()).hexdigest()[:16]
    return KERNELS_SHA256_16


def _cache_entries(cache_dir):
    if not cache_dir or not os.path.isdir(cache_dir): return {}
    out = {}
    for key in sorted(os.listdir(cache_dir)):
        d = os.path.join(cache_dir, key)
        if os.path.isdir(d): out[key] = sorted(f for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))
    return out


_COMPILES = {"make_ttir": 0, "make_cubin": 0}
_JIT_RUNS = {"calls": [], "installed": False, "n0": 0}
_FIRST_FORWARD = {'wall_s': None, 'n': None, 'batch': None}   # the first predict() wall time in this process (a cold JIT shows here); reported by cache_witness()
_COMPILES["bound_on"] = None
def _install_compile_counter():
    """Count make_ttir / make_cubin calls on the CUDABackend class that compile() actually uses — triton.backends.backends['nvidia'].compiler.
    Triton 3.0 loads each backend's compiler.py through spec_from_file_location into a module that is never registered in sys.modules, so
    `from triton.backends.nvidia.compiler import CUDABackend` yields a SECOND class object that compile() never calls (counters placed there stay
    0 through a cold compile). _COMPILES['bound_on'] records which object was wrapped. The counts are informational: cache_witness() decides
    HIT/MISS from the cache-content snapshot and the JIT run meter, and only reports these."""
    try:
        from triton.backends import backends as _bk
        obj = _bk["nvidia"].compiler; CUDABackend = obj if isinstance(obj, type) else obj.CUDABackend   # Triton 3.0: Backend.compiler IS the class; the .CUDABackend attribute form covers a module-valued .compiler
        import sys as _sys; mod = _sys.modules.get(CUDABackend.__module__) or type("m", (), {"__name__": CUDABackend.__module__, "__file__": "?"})
    except Exception as e:
        _COMPILES["bound_on"] = f"UNBOUND ({type(e).__name__})"; return False
    if getattr(CUDABackend, "_k1_counted", False): return True
    for name in ("make_ttir", "make_cubin"):
        raw = CUDABackend.__dict__.get(name); orig = getattr(CUDABackend, name)
        if isinstance(raw, staticmethod):
            fn = raw.__func__
            def wrap(*a, _fn=fn, _name=name, **k):
                _COMPILES[_name] += 1; return _fn(*a, **k)
            setattr(CUDABackend, name, staticmethod(wrap))
        else:
            def wrap(self, *a, _fn=orig, _name=name, **k):
                _COMPILES[_name] += 1; return _fn(self, *a, **k)
            setattr(CUDABackend, name, wrap)
    CUDABackend._k1_counted = True; _COMPILES["bound_on"] = f"{getattr(mod, '__name__', '?')} ({getattr(mod, '__file__', '?')}) via triton.backends.backends['nvidia'].compiler.CUDABackend"; return True


def _install_jit_meter():
    """Time every triton JITFunction.run call by wrapping the class method (once per process; launches under 5 ms are not recorded). A call
    that has to compile takes >= 0.15 s per K1 kernel while a load from the cache takes milliseconds and a warm launch well under 1 ms, so
    cache_witness() counts calls >= 0.15 s since apply() as compiles."""
    if _JIT_RUNS["installed"]: return True
    try:
        import triton.runtime.jit as tj
    except Exception:
        return False
    orig = tj.JITFunction.run
    def run(self, *a, **k):
        t = time.time(); r = orig(self, *a, **k); dt = time.time() - t
        if dt >= 0.005: _JIT_RUNS["calls"].append((getattr(self, "__name__", "?"), round(dt, 4)))    # launches under 5 ms are not recorded (bulk)
        return r
    tj.JITFunction.run = run; _JIT_RUNS["installed"] = True; return True


def jit_meter_since(n0):
    calls = _JIT_RUNS["calls"][n0:]; slow = [c for c in calls if c[1] >= 0.15]
    return {"calls_over_5ms": len(calls), "compile_class_calls_over_150ms": len(slow), "slow": slow[:24], "loads_5_150ms": [c for c in calls if c[1] < 0.15][:24], "threshold_s": 0.15}


def own_keys_of(rec, member="k1"):
    """The cache keys this kit's loader may install = the entries the identity record lists for it: for the kit's own shipped cache,
    rec["entries"]; for a union dir (one cache shared by several kits, written by a separate tool and read-only here) the member's list under
    union.members[member].entries, else the keys whose union.entry_owners value names the member. None = the record lists nothing usable
    (apply() then leaves the shipped cache aside: cold JIT, said on the line)."""
    if not isinstance(rec, dict): return None
    u = rec.get("union")
    if isinstance(u, dict):
        m = (u.get("members") or {}).get(member) or {}
        ents = m.get("entries")
        if ents is None:
            owners = u.get("entry_owners") or {}; ents = [k for k, o in owners.items() if member in str(o).split("+")]
        return set(ents)
    ents = rec.get("entries")
    if isinstance(ents, list): return set(e["key"] if isinstance(e, dict) else str(e) for e in ents)
    if isinstance(ents, dict): return set(ents.keys())
    return None


SUMS_PREFIX = "[chrombpnet-opt]"                                    # the package's stderr line prefix (chrombpnet_opt.report.PREFIX)
CACHE_RECORD_FILES = ("JIT_IDENTITY.json", "SHA256SUMS", "DONE", "SERVES")   # the record files at the top of a shipped cache dir: read here, never installed


def shipped_cache_sums_refusal(cache_dir):
    """None when every file of the shipped cache dir that an install copies (all but the top-level record files) is a `<sha256>  <relative path>`
    line of the SHA256SUMS at its top and its bytes re-hash to that line. Otherwise the reason, naming the first file (in sorted order) that is
    not listed, unreadable or differs — or the missing SHA256SUMS — after one ``[chrombpnet-opt] SHA256SUMS: refused <file>: <reason>`` line on
    stderr. apply() then installs nothing of that cache and the kernels JIT-compile in the process (the cold path, said on its line): a file
    that is not the shipped one never reaches the Triton cache the process loads from."""
    import sys
    name = os.path.basename(os.path.normpath(cache_dir)); sums_path = os.path.join(cache_dir, "SHA256SUMS")
    def refuse(rel, reason):
        print(f"{SUMS_PREFIX} SHA256SUMS: refused {name}/{rel}: {reason}", file=sys.stderr, flush=True); return f"{name}/{rel}: {reason}"
    if not os.path.isfile(sums_path): return refuse("SHA256SUMS", "absent — the cache's files are held to nothing")
    listed = {}
    with open(sums_path, encoding="utf-8") as fh:
        for ln in fh:
            parts = ln.split(None, 1) if ln.strip() and not ln.startswith("#") else []
            if len(parts) == 2: r = parts[1].strip().lstrip("*"); listed[r[2:] if r.startswith("./") else r] = parts[0].lower()
    for root, dirs, fs in os.walk(cache_dir):
        dirs.sort(); rel_dir = os.path.relpath(root, cache_dir)
        for f in sorted(fs):
            if rel_dir == "." and f in CACHE_RECORD_FILES: continue
            rel = f if rel_dir == "." else os.path.join(rel_dir, f).replace(os.sep, "/")
            want = listed.get(rel)
            if want is None: return refuse(rel, "not listed in its SHA256SUMS")
            try:
                with open(os.path.join(root, f), "rb") as fh: have = hashlib.sha256(fh.read()).hexdigest()
            except OSError as e: return refuse(rel, f"unreadable ({type(e).__name__})")
            if have != want: return refuse(rel, f"sha256 {have[:16]} != SHA256SUMS {want[:16]}")
    return None


def hold_shipped_cache(cache_dir):
    """apply()'s step before any file of a shipped cache is installed: (cache_dir, None) when the cache passes shipped_cache_sums_refusal, else
    (None, the line's note) — the same outcome as an inapplicable cache: nothing installed, cold JIT in the process, said."""
    why = shipped_cache_sums_refusal(cache_dir)
    if why is None: return cache_dir, None
    return None, f"shipped cache REFUSED by its SHA256SUMS ({why}) -> nothing of it installed; cold JIT in the process cache dir: EVERY FRESH PROCESS PAYS THE JIT (said)"


def _install_entries(src, dst, own_keys=None, strict_own=True, existing_policy="refuse"):
    """Copy the shipped cache entries from src into dst. Only the keys in own_keys are installed (own_keys=None: all); a key outside own_keys
    raises when src is the kit's OWN shipped cache (strict_own=True) and is skipped and counted in a union dir (strict_own=False). A file that
    already exists in dst is never overwritten: existing_policy="keep" leaves it as is (the user's working entries in an effective cache dir),
    "refuse" raises unless the bytes are identical (a loaded .so must never be rewritten). The record files at the top of src (JIT_IDENTITY.json,
    SHA256SUMS, DONE, SERVES) are not copied. Returns the number of files copied; the details of the call (own/foreign/kept counts,
    copied_entries) are left in _install_entries.last."""
    n = 0; own_seen = set(); foreign = []; kept = set(); kept_files = 0; copied_entries = set()
    top = [e for e in sorted(os.listdir(src)) if os.path.isdir(os.path.join(src, e))]
    for e in top:
        if own_keys is not None and e not in own_keys:
            foreign.append(e)
            if strict_own: raise RuntimeError(f"[chrombpnet_k1.apply] R-CACHE-OWN: the shipped cache {src} carries a FOREIGN entry {e[:16]}… (not in this kit's JIT_IDENTITY) — refused by name")
            continue
        own_seen.add(e)
    for root, _, fs in os.walk(src):
        rel = os.path.relpath(root, src)
        if rel != "." and rel.split(os.sep)[0] in foreign: continue
        d = dst if rel == "." else os.path.join(dst, rel); os.makedirs(d, exist_ok=True)
        for f in fs:
            if rel == "." and f in CACHE_RECORD_FILES: continue
            a = os.path.join(root, f); b = os.path.join(d, f)
            if os.path.exists(b):
                if existing_policy == "keep":
                    kept.add(rel.split(os.sep)[0]); kept_files += 1; continue          # same-key files already present in the effective dir are KEPT (the user's working entries), never overwritten

                if open(a, "rb").read() != open(b, "rb").read(): raise RuntimeError(f"[chrombpnet_k1.apply] cache install refused: {os.path.join(rel, f)} exists in {dst} with different bytes (a loaded .so must never be overwritten)")
                continue
            shutil.copy2(a, b); n += 1
            if rel != ".": copied_entries.add(rel.split(os.sep)[0])
    _install_entries.last = {"files": n, "own_installed": len(own_seen), "own_total": (len(own_keys) if own_keys is not None else len(own_seen)), "foreign_skipped": len(foreign), "foreign_keys": [k[:16] for k in foreign][:8],
                             "kept_existing_entries": len(kept), "kept_existing_files": kept_files, "copied_entries": sorted(copied_entries)}
    return n


def cache_snapshot(cache_dir):
    """{relative path: (sha256[:16], mtime_ns, size)} for EVERY file under cache_dir ({} when absent). Content rather than an entry count:
    a cache whose __grp__ paths were not relocated recompiles every kernel into the SAME key dirs, which only a content comparison detects
    (cache_witness() compares the snapshot taken at apply() with a fresh one)."""
    snap = {}
    if not cache_dir or not os.path.isdir(cache_dir): return snap
    for root, _, fs in os.walk(cache_dir):
        for f in fs:
            q = os.path.join(root, f); st = os.stat(q)
            snap[os.path.relpath(q, cache_dir)] = (hashlib.sha256(open(q, "rb").read()).hexdigest()[:16], st.st_mtime_ns, st.st_size)
    return snap


def relocate_cache_groups(cache_dir, only_entries=None):
    """Triton 3.0's FileCacheManager.put_group writes __grp__<kernel>.json with the ABSOLUTE child paths of the populating dir, and get_group
    keeps only the paths that EXIST — so a cache copied elsewhere loads no group metadata and every kernel silently recompiles into the same key
    dir. Rewrite every group's child paths to THIS dir (only_entries: just the entries this install copied); a listed member that is absent is a
    broken entry -> RuntimeError (no silent JIT). Returns the number of group files rewritten."""
    n = 0; cache_dir = os.path.abspath(cache_dir)                    # Triton writes ABSOLUTE child paths; so does this
    for entry in sorted(os.listdir(cache_dir)):
        d = os.path.join(cache_dir, entry)
        if not os.path.isdir(d): continue
        if only_entries is not None and entry not in only_entries: continue   # only the entries THIS install copied (never foreign ones / the user's own)
        for f in sorted(os.listdir(d)):
            if f.startswith("__grp__") and f.endswith(".json"):
                q = os.path.join(d, f); g = json.load(open(q)); cps = g.get("child_paths")
                if cps is None: raise RuntimeError(f"[chrombpnet_k1.apply] shipped cache entry {entry}: {f} carries no child_paths — refused")
                # Triton 3.0's put_group form: child_paths = {member_name: absolute_path} (a dict; get_group iterates .items()) — keep the form, relocate the values
                items = list(cps.items()) if isinstance(cps, dict) else [(os.path.basename(c), c) for c in cps]
                new = {}
                for name, c in items:
                    m = os.path.join(d, os.path.basename(c))
                    if not os.path.isfile(m): raise RuntimeError(f"[chrombpnet_k1.apply] shipped cache entry {entry}: group member {os.path.basename(c)} is absent — refused (no silent JIT)")
                    new[name] = m
                g["child_paths"] = new; json.dump(g, open(q, "w")); n += 1
    return n


def cache_witness(cache_dir, snapshot_at_apply, compiles_at_apply=None, jit_meter_n0=0):
    """Compare cache_dir now with the snapshot taken at apply(). HIT = no file changed or added since apply (sha256 + mtime_ns + size) AND no
    JITFunction.run call of compile class (>= 0.15 s) since apply; the make_ttir/make_cubin stage counts are reported beside the verdict but do
    not enter it. When the shipped cache was not applied in this process (no dir, or an empty snapshot) the result says so and never reads HIT.
    A snapshot in the older {key: [files]} form gets an entry-count comparison labelled VOID (it cannot see an in-place recompile)."""
    before = snapshot_at_apply or {}
    if before and not isinstance(next(iter(before.values())), (tuple, list)):
        now = _cache_entries(cache_dir); new = [k for k in now if k not in before or now[k] != before[k]]
        return {"cache_dir": cache_dir, "entries_at_apply": len(before), "entries_now": len(now), "new_or_modified": len(new), "witness": "count witness (VOID: blind to in-place recompiles — the k1r3 content witness supersedes)",
                "verdict": ("count witness VOID: " + ("no new entries" if not new else f"{len(new)} entries written after apply"))}
    now = cache_snapshot(cache_dir)
    changed = sorted(k for k in now if k in before and tuple(now[k]) != tuple(before[k])); added = sorted(k for k in now if k not in before)
    meter = jit_meter_since(jit_meter_n0); c0 = compiles_at_apply or {}; stage = {k: _COMPILES[k] - (c0.get(k, 0) or 0) for k in ("make_ttir", "make_cubin")}
    stage["bound_on"] = _COMPILES.get("bound_on"); stage["admissible"] = bool(stage["bound_on"] and "triton.backends.backends" in str(stage["bound_on"]))
    hit = not changed and not added and meter["compile_class_calls_over_150ms"] == 0
    applied = bool(cache_dir) and bool(before)          # a verdict may read HIT only when the SHIPPED cache was applied in this process
    if not applied:
        served = os.environ.get("TRITON_CACHE_DIR") or os.path.expanduser("~/.triton/cache")
        return {"cache_dir": cache_dir, "applied": False, "files_at_apply": len(before), "files_now": len(now), "jit_meter": meter, "stage_counter": stage, "first_forward": dict(_FIRST_FORWARD),
                "witness": "NOT A SHIPPED-CACHE WITNESS: the shipped cache was not applied in this process",
                "verdict": (f"served by the process/box Triton cache ({served}); shipped cache not applicable ({_CACHE_SKIP.get('note') or 'no cache dir given'}); compile-class JIT calls {meter['compile_class_calls_over_150ms']}"
                            + (" = a COLD JIT in this process" if meter["compile_class_calls_over_150ms"] else " = warm from the box's own dir (a count-based verdict here is VOID as a shipped-cache witness)") + f"; first forward {_FIRST_FORWARD.get('wall_s')} s at n={_FIRST_FORWARD.get('n')} b={_FIRST_FORWARD.get('batch')}")}
    return {"cache_dir": cache_dir, "applied": True, "files_at_apply": len(before), "files_now": len(now), "n_changed": len(changed), "changed": changed[:12], "n_added": len(added), "added": added[:12],
            "jit_meter": meter, "stage_counter": stage,
            "witness": "content (sha256+mtime_ns+size of every cache file) + the JIT run meter (a compile-class call >= 0.15 s; a cached load ~7 ms)",
            "first_forward": dict(_FIRST_FORWARD),
            "verdict": (f"shipped cache APPLIED — HIT (0 files changed/new, 0 compile-class JIT calls: the shipped cache served every kernel; first forward {_FIRST_FORWARD.get('wall_s')} s at n={_FIRST_FORWARD.get('n')} b={_FIRST_FORWARD.get('batch')} in this process)" if hit else f"MISS/PARTIAL ({len(changed)} changed, {len(added)} new, {meter['compile_class_calls_over_150ms']} compile-class calls; first forward {_FIRST_FORWARD.get('wall_s')} s in this process) -> cold apply")}


CACHE_OF_RECORD_LIST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "triton_cache_of_record.json")
def resolve_cache_of_record(list_path=None):
    """Pick the shipped Triton cache for this GPU from torch/triton_cache_of_record.json = {"entries": [{"arch": "cuda-90-32", "device_match":
    "<substring of the device name, optional>", "stack": {"triton": ..., "torch": ...}, "dir": "<relative to torch/>", "sha256sums_sha256_16": ...},
    ...]}. An entry matches on arch (from torch's device capability — no Triton driver use before the install) and on every stack version it
    names; an entry whose device_match occurs in the device name is preferred over an arch-only one; the chosen entry's SHA256SUMS is re-hashed
    and must equal the listed pin. No list, no matching entry or a pin mismatch = cold JIT with the reason in the note — never a refusal.
    Returns (dir_or_None, note). A cache_dir passed to apply() bypasses this list."""
    list_path = list_path or CACHE_OF_RECORD_LIST
    if not os.path.isfile(list_path): return None, f"no cache-of-record list at {list_path}: cold JIT"
    lst = json.load(open(list_path)); cap = torch.cuda.get_device_capability(0); arch = f"cuda-{cap[0]}{cap[1]}-32"; name = torch.cuda.get_device_name(0)
    stack = {"triton": _triton_version(), "torch": torch.__version__}
    cands = [e for e in lst.get("entries", []) if e.get("arch") == arch and not any(e.get("stack", {}).get(k) not in (None, v) for k, v in stack.items())]
    # an entry that names this device class (device_match: a substring of torch's device name, e.g. "H200") is preferred over the arch-only entry
    named = [e for e in cands if e.get("device_match") and e["device_match"] in name]; generic = [e for e in cands if not e.get("device_match")]
    for e in named + generic:
        d = os.path.join(os.path.dirname(list_path), e["dir"]); sums = os.path.join(d, "SHA256SUMS")
        if not os.path.isfile(sums): return None, f"cache-of-record entry {e['dir']} for {arch} has no SHA256SUMS: cold JIT (said)"
        h = hashlib.sha256(open(sums, "rb").read()).hexdigest()[:16]
        if h != e.get("sha256sums_sha256_16"): return None, f"shipped cache entry {e['dir']} for {arch}: SHA256SUMS sha {h} != the list's pin {e.get('sha256sums_sha256_16')} (the entry's bytes are not the frozen ones) -> cold JIT (said; never a refusal — k1r4.7)"
        note = f"shipped cache for {arch} ({name}; stack {stack}): {e['dir']} (pin {h})"
        # an entry that serves several device classes (served_classes_note) names the class being served on the line
        for cls, txt in (e.get("served_classes_note") or {}).items():
            if cls in name: note += f"; {txt}"
        return d, note
    return None, f"no shipped cache for {arch} ({name}; stack {stack}) in the list: cold JIT (said on the line)"


def _effective_triton_cache_dir():
    """The process's EFFECTIVE Triton cache dir and where it came from: TRITON_CACHE_DIR if the user set it, else Triton's own default
    (~/.triton/cache via triton.runtime.cache.default_cache_dir(); the same path spelled out if that helper is absent)."""
    d = os.environ.get("TRITON_CACHE_DIR")
    if d: return os.path.abspath(d), "TRITON_CACHE_DIR (set by the user)"
    try:
        from triton.runtime.cache import default_cache_dir
        return os.path.abspath(default_cache_dir()), "Triton's default cache dir"
    except Exception as e:
        if is_oom(e): raise                                             # an out-of-memory error propagates; the default-path route below is for every other failure
        return os.path.join(os.path.expanduser("~"), ".triton", "cache"), "~/.triton/cache (Triton's default path)"


ARCH_TILES_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "arch_tiles.json")
def batch_regime_tiles():
    """This arch's by_batch regime tiles [{max_B, tile}, ...] from arch_tiles.json, sorted by max_B; [] when the arch has none (or the file /
    device is unreadable): one tile at every batch size."""
    try:
        cap = torch.cuda.get_device_capability(0); arch = f"cuda-{cap[0]}{cap[1]}"; e = (json.load(open(ARCH_TILES_PATH)).get("entries") or {}).get(arch) or {}
        return sorted((e.get("by_batch") or []), key=lambda r: int(r["max_B"]))
    except Exception as e:
        if is_oom(e): raise                                             # an out-of-memory error propagates; the default-tile route ([]) is for every other failure
        return []
def regime_tile(B, default_tile, by_batch):
    for r in by_batch:
        if B <= int(r["max_B"]): return r["tile"]
    return default_tile
def recipe_mode(mode=None):
    """The mode of the call: 'det' when TF_DETERMINISTIC_OPS=1 is set in the environment (stock's deterministic recipe), else 'prod'; a caller
    that knows the recipe passes mode explicitly. Any other value raises ValueError."""
    if mode is None: mode = "det" if os.environ.get("TF_DETERMINISTIC_OPS", "0") == "1" else "prod"
    if mode not in ("prod", "det"): raise ValueError(f"[chrombpnet_k1] mode {mode!r}: 'prod' or 'det' only")
    return mode

def default_route(arch=None, mode=None):
    """The kit's route for this arch under this mode, from arch_tiles.json entries[arch].default_route — either a per-mode dict
    {'prod': {route, basis}, 'det': {route, basis}} or a single {route, basis} that holds for both modes ('det' = TF_DETERMINISTIC_OPS=1, see
    recipe_mode). Returns (route, basis) with route one of 'k1' (this package's Triton forward), 'tf' (the kit's TensorFlow route), 'stock'
    (stock's own path; also the answer when the mode has no entry or the file cannot be read) or 'unlisted' (no entry for this arch: the
    caller's device-class table decides). The mode is validated before anything is read: a bad mode raises, it is never swallowed by the
    fallback."""
    mode = recipe_mode(mode)                                   # validated BEFORE the guarded read: a bad mode is refused, never swallowed into the fallback
    try:
        if arch is None: cap = torch.cuda.get_device_capability(0); arch = f"cuda-{cap[0]}{cap[1]}"
        m = json.load(open(ARCH_TILES_PATH)); e = (m.get("entries") or {}).get(arch)
        if not e: return "unlisted", f"arch {arch} is not in arch_tiles.json (listed: {sorted((m.get('entries') or {}).keys())}): no row for this arch — the kit's class table decides, and a class it does not name engages K1 with a cold JIT and the tile chosen by the device's shared memory (said on the line)"
        r = e.get("default_route") or {"route": "k1", "basis": "listed arch with a tile row; no default_route field"}
        if "route" not in r: r = r.get(mode) or {"route": "stock", "basis": f"no {mode} row for {arch}: the stock path by default"}
        return r["route"], f"{arch}/{mode}: {r['basis']}"
    except Exception as ex:
        if is_oom(ex): raise                                            # an out-of-memory error propagates; the stock route below is for every other failure
        return "stock", f"default_route unreadable ({ex!r}): the stock path"
def select_counts_form(arch=None):
    """The counts head's dense summation ORDER for this GPU arch from arch_tiles.json entries[arch].counts_form -> ({"form", "group"}, line).
    The form names an accumulation order implemented by kernels.exact_counts (e.g. "groups_prod_bfly_seq" with group 128 on cuda-89) that
    reproduces the order stock's own dense layer accumulates in on that arch, so logcounts stay bitwise equal to stock there. An arch without
    an entry returns (None, line) = the H100 order (dense_exact lanes)."""
    if arch is None: cap = torch.cuda.get_device_capability(0); arch = f"cuda-{cap[0]}{cap[1]}"
    m = json.load(open(ARCH_TILES_PATH)); e = (m.get("entries") or {}).get(arch) or {}
    cf = e.get("counts_form")
    if not cf: return None, f"counts head: the H100 order (dense_exact lanes; arch {arch} has no counts_form entry) — bitwise with the H100 lock DET"
    return {"form": cf["form"], "group": int(cf.get("group", 128))}, f"counts head: {cf['form']} (group {cf.get('group', 128)}) = the {arch} stock's OWN dense order — basis: {str(cf.get('basis'))[:160]}"

def select_tile(tile=None):
    """The K1 conv tile "BLOCK_MxBLOCK_NxBLOCK_Kxnum_warpsxnum_stages" for this GPU arch from arch_tiles.json (cuda-90 / cuda-80 / cuda-89);
    an explicit tile= wins (how = "explicit"). An arch that is not listed is not refused: the tile is chosen by the device's opt-in shared
    memory per block (the cuda-90 tile at >= 133,120 B, else the smaller cuda-89 tile — a tile that does not fit the device's shared memory
    would fail at launch with OutOfResources) and `how` says the arch is unlisted. Returns (tile, how)."""
    if tile: return tile, "explicit"
    cap = torch.cuda.get_device_capability(0); arch = f"cuda-{cap[0]}{cap[1]}"; m = json.load(open(ARCH_TILES_PATH)); e = (m.get("entries") or {}).get(arch)
    if not e:
        # an UNLISTED arch is not a refusal — the tile is chosen by the device's opt-in shared memory (the cuda-90 tile at >= 133,120 B, else the cuda-89 tile) and the line says the arch is unlisted
        try: smem = int(getattr(torch.cuda.get_device_properties(0), "shared_memory_per_block_optin", 0) or 0)
        except Exception as e:
            if is_oom(e): raise                                         # an out-of-memory error propagates; the small-tile route (smem 0) is for every other failure
            smem = 0
        tile = "64x512x16x16x3" if smem >= 133120 else "64x256x16x8x3"
        return tile, f"arch {arch} ({torch.cuda.get_device_name(0)}) is NOT in arch_tiles.json (listed: {sorted((m.get('entries') or {}).keys())}): tile {tile} chosen by the device's opt-in smem {smem} B — an unlisted class (cold JIT; the tile rows are per listed class)"
    return e["tile"], f"arch_tiles.json[{arch}] ({torch.cuda.get_device_name(0)}; shared {e.get('shared_B')} B; certificate {str(e.get('certificate'))[:60]})"


def apply(bias_h5, nobias_h5, cache_dir=None, require_device=None, counts_mode="exact", tile=None, head="serial", install_form="effective", precision="ieee", tc_tile=None):
    """Build the K1 model from stock's bias and no-bias .h5 model files and make the shipped Triton cache available to this process; returns
    (model, info) where info records the pins, the cache decision and the install details (info["r8_line"] is the one-line summary).

    The shipped cache is used only when it matches this install, checked in this order: (1) without touching the driver — JIT_IDENTITY.json
    must record this package's kernels.py sha256, this triton and torch version (and triton_key when recorded); (2) the kit's own entries are
    held file by file to the cache's SHA256SUMS (an unlisted or differing file leaves the whole cache aside), installed and their __grp__ paths relocated; (3) only then is the driver read — target, backend_hash and the IR/PTX identity key of the
    installed entries must equal the record's. Whichever check fails, the outcome is the same and never a refusal: the kernels JIT-compile in
    the process's cache dir on first launch and the printed line says why. The device name and cuDNN version are printed for information,
    never compared; require_device likewise only prints. No forward pass, no subprocess and no kernel build happens here — the kernels load
    (or compile) inside the first forward, and cache_witness() tells afterwards which it was.

    cache_dir: None / "auto" = resolve_cache_of_record(); "cold" / "none" = no shipped cache; a path = that cache dir. install_form
    "effective": install ADD-ONLY into the process's effective Triton cache dir (TRITON_CACHE_DIR if set, else ~/.triton/cache) — same-key
    files already there are KEPT, groups are relocated only for the entries this install copied, no env var is set, so the entries serve this
    process and the user's later ones; if that dir is not writable, or with install_form="process", a fresh temp dir is populated instead and
    TRITON_CACHE_DIR points at it for the rest of the process (never restored), said on the line. precision "ieee" = kernels.py, "tf32" =
    kernels_tc.py (tc_tile sets its tile); the conv tile and the counts order come from arch_tiles.json for this arch (select_tile,
    select_counts_form)."""
    _install_compile_counter(); _install_jit_meter(); n_groups = 0; n_files = 0; cache_note = "override (cache_dir given)"
    if cache_dir is None or cache_dir == "auto":
        cache_dir, cache_note = resolve_cache_of_record()
    elif cache_dir in ("cold", "none"):
        cache_dir, cache_note = None, "cold requested (cache_dir='cold'): the process TRITON_CACHE_DIR / JIT"
    """RULE 165 apply(): PINS (kernels.py bytes vs the shipped JIT-identity record, device class, triton version) + the PATCH (build the K1 model
    from the stock's own h5 weights) — NO forward, NO subprocess, NO build of kernels. The kernel LOAD from the shipped Triton cache
    happens inside the first clocked batch (stamped first_batch_s by the caller); a cache MISS is witnessed by cache_witness() and labelled cold."""
    import shutil, tempfile, json
    t0 = time.time(); rec = None; inst = None
    if cache_dir:
        rec_path = os.path.join(cache_dir, "JIT_IDENTITY.json")
        if not os.path.isfile(rec_path): raise RuntimeError(f"[chrombpnet_k1.apply] shipped cache {cache_dir} carries no JIT_IDENTITY.json — refused (no silent JIT)")
        rec = json.load(open(rec_path))
        # STEP 1 (no driver use): the pins that need no CUDA context — kernels.py bytes, triton/torch versions, triton_key (a hash of Triton's own files)
        mine = {"triton": _triton_version(), "torch": torch.__version__, "kernels_sha256_16": _kernels_sha()}
        if "triton_key" in rec:
            from triton.compiler.compiler import triton_key as _tk
            mine["triton_key"] = _tk()
        bad = {k: (rec.get(k), v) for k, v in mine.items() if rec.get(k) != v}
        if bad:
            # an inapplicable shipped cache is NEVER a refusal — the JIT path, with the reason printed
            # the jit-source sha says whether the @triton.jit kernels differ or only the Python helpers around them (the pin is the whole file either way)
            jit_datum = f"; @triton.jit sources {'EQUAL' if rec.get('kernels_jit_sha256_16') == _kernels_jit_sha() else 'differ' if rec.get('kernels_jit_sha256_16') else 'not recorded'} (record {rec.get('kernels_jit_sha256_16')}, package {_kernels_jit_sha()})" if "kernels_sha256_16" in bad else ""
            cache_note = f"shipped cache NOT APPLICABLE (stage-1 pins differ: {bad}{jit_datum}) -> cold JIT in the process cache dir: EVERY FRESH PROCESS PAYS THE JIT (said; the compile-class count on the R8-close line)"
            print(f"[chrombpnet_k1.apply] {cache_note}", flush=True); _CACHE_SKIP["note"] = cache_note; cache_dir = None; rec = None
    if cache_dir:
        # before STEP 2: every file the install would copy is held to its line of the cache's SHA256SUMS; one unlisted or differing file leaves the whole cache uninstalled (cold JIT, said)
        cache_dir, held_note = hold_shipped_cache(cache_dir)
        if held_note:
            cache_note = held_note; print(f"[chrombpnet_k1.apply] {cache_note}", flush=True); _CACHE_SKIP["note"] = cache_note; rec = None
    if cache_dir:
        # STEP 2: INSTALL the kit's own entries (add-only into the effective cache dir, or into a fresh temp dir that TRITON_CACHE_DIR then points at; never overwriting a differing file) + the __grp__ relocation
        own = own_keys_of(rec, "k1"); is_union = isinstance(rec.get("union"), dict)
        if own is None:
            cache_note = "shipped cache NOT APPLICABLE (R-CACHE-OWN: the record lists no entries for this kit) -> cold JIT (said)"; print(f"[chrombpnet_k1.apply] {cache_note}", flush=True); _CACHE_SKIP["note"] = cache_note; cache_dir = None; rec = None; own = []
    if cache_dir:
        form_used = None
        if install_form == "effective":
            eff, eff_why = _effective_triton_cache_dir()
            try:
                os.makedirs(eff, exist_ok=True); probe = os.path.join(eff, ".k1_write_probe_%d" % os.getpid()); open(probe, "w").close(); os.remove(probe); writable = True
            except Exception as e:
                if is_oom(e): raise                                     # an out-of-memory error propagates; the temp-dir fallback below is for every other failure
                writable = False; eff_err = repr(e)[:80]
            if writable:
                n_files = _install_entries(cache_dir, eff, own_keys=own, strict_own=not is_union, existing_policy="keep"); inst = dict(_install_entries.last)
                n_groups = relocate_cache_groups(eff, only_entries=set(inst["copied_entries"])); work = eff
                form_used = f"(2) effective dir add-only: {eff} [{eff_why}]; no env var set; {inst['kept_existing_entries']} entries already present kept"
                if inst["kept_existing_entries"]:
                    # the kept (pre-existing) entries vs the shipped ones — identity means equal IR/PTX/json; cubin/.so differences are reported on the line, nothing more
                    cmpk = cache_identity_compare(cache_dir, eff); inst["kept_identity"] = cmpk
                    form_used += (f"; kept entries: identical IR/PTX ({cmpk['n_ir_compared']} files)" if cmpk["identity_equal"] else f"; kept entries: IR/PTX DIFFER in {len(cmpk['ir_differing'])} files (a different build under the same key — kept, never overwritten; said)") + \
                                 (f", different SASS/.so in {len(cmpk['artefacts_differing'])} of {cmpk['n_artefacts_compared']} build artefacts (datum: ptxas is not box-deterministic)" if cmpk["artefacts_differing"] else ", identical SASS/.so")
            else:
                form_used = f"(2)-fallback: the effective dir {eff} is not writable ({eff_err}) -> a temp dir + TRITON_CACHE_DIR kept FOR THE PROCESS (not restored)"
        if form_used is None or form_used.startswith("(2)-fallback"):
            work = tempfile.mkdtemp(prefix="triton_cache_k1_"); n_files = _install_entries(cache_dir, work, own_keys=own, strict_own=not is_union); inst = dict(_install_entries.last); n_groups = relocate_cache_groups(work); os.environ["TRITON_CACHE_DIR"] = work
            if form_used is None: form_used = "process: a fresh temp dir + TRITON_CACHE_DIR kept FOR THE PROCESS (not restored; install_form='process')"
        cache_dir = work; inst["install_form"] = form_used; inst["env_var_set"] = (os.environ.get("TRITON_CACHE_DIR") == work)
        inst["source"] = "union dir (read-only for this loader; written by the union tool only)" if is_union else "the kit's own shipped cache"
        missing = sorted(k[:16] for k in own if not os.path.isdir(os.path.join(work, k)))
        if missing:
            inst["missing_own_entries"] = missing; print(f"[chrombpnet_k1.apply] R-CACHE-OWN: {len(missing)} of this kit's own entries are absent after the install: {missing[:4]} — those kernels JIT-compile into the process dir (cold, said; never a refusal)", flush=True)
        # STEP 3: only now the driver/target read — the identity = target (arch) + backend_hash (the compiler key) + the IR/PTX identity key of the
        # installed own entries; the DEVICE NAME and the cuDNN version are printed on the line, never part of the key (two device names of one
        # arch, e.g. H100 and H200, are served by the same cache)
        ti = _triton_identity(); mine2 = {k: v for k, v in ti.items() if k in rec and k in ("backend_hash", "target")}
        bad = {k: (rec.get(k), v) for k, v in mine2.items() if rec.get(k) != v}
        ident = None
        if not bad and rec.get("identity_key"):
            try:
                ident = cache_identity_key(work, rec, only_entries=set(own)); bad2 = {} if ident["identity_key"] == rec["identity_key"] else {"identity_key": (rec["identity_key"], ident["identity_key"])}
            except Exception as e:
                if is_oom(e): raise                                     # an out-of-memory error propagates; the cache-not-applicable (cold JIT) route below is for every other failure
                bad2 = {"identity_key": ("computed", f"error {type(e).__name__}: {str(e)[:60]}")}
            bad.update(bad2)
        dev_datum = f"record built on {rec.get('device')} (cudnn {rec.get('cudnn')}), running on {torch.cuda.get_device_name(0)} (cudnn {torch.backends.cudnn.version()}): {'key equal, served' if not bad else 'KEY DIFFERS'}"
        inst["device_datum"] = dev_datum; inst["identity_after_install"] = {"target": ti.get("target"), "backend_hash": ti.get("backend_hash"), "identity_key": (ident or {}).get("identity_key"), "record_identity_key": rec.get("identity_key"), "equal": not bad}
        if bad:
            cache_note = f"shipped cache NOT APPLICABLE after install ({bad}; {dev_datum}) -> the process JIT path (cold, said; the installed entries are inert under other keys)"
            print(f"[chrombpnet_k1.apply] {cache_note}", flush=True); _CACHE_SKIP["note"] = cache_note
        else:
            print(f"[chrombpnet_k1.apply] identity after install: arch {ti.get('target')} + IR/PTX key {rec.get('identity_key')} equal; {dev_datum}", flush=True)
        _JIT_RUNS["n0"] = len(_JIT_RUNS["calls"])
    else:
        cache_dir = os.environ.get("TRITON_CACHE_DIR")          # no shipped cache: cache_witness() watches the process cache dir (a fresh dir = every kernel JIT-compiles in the first batch)
    if require_device and require_device not in torch.cuda.get_device_name(0):
        # a device expectation is printed on the line, never a refusal (the caller's device-class table decides K1 vs stock)
        print(f"[chrombpnet_k1.apply] device datum: running on {torch.cuda.get_device_name(0)}, the caller expected {require_device} (the tile rows are per class; this row is labelled by its class)", flush=True); _CACHE_SKIP["device_expectation"] = f"expected {require_device}, running on {torch.cuda.get_device_name(0)}"
    from bpnetlite.chrombpnet import ChromBPNet
    tile, tile_how = select_tile(tile)
    counts_form, counts_line = select_counts_form(); print(f"[chrombpnet_k1.apply] {counts_line}", flush=True)
    if precision == "tf32" and tc_tile: _tc().set_tile(tc_tile)
    port = ChromBPNet.from_chrombpnet(bias_h5, nobias_h5).cuda().eval(); model = K1ChromBPNet(port, counts_mode=counts_mode, tile=tile, head=head, tile_selected_by=tile_how, counts_form=counts_form, precision=precision); model.port = port; model.counts_line = counts_line
    torch.cuda.synchronize()
    return model, {"apply_s": round(time.time() - t0, 3), "pins": {"kernels_sha256_16": _kernels_sha(), "kernels_jit_sha256_16": _kernels_jit_sha(), "triton": _triton_version(), "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0)},
                   "cache_dir": cache_dir, "cache_identity_record": rec, "cache_entries_at_apply": _cache_entries(cache_dir), "cache_snapshot_at_apply": cache_snapshot(cache_dir) if cache_dir else {},
                   "compiles_at_apply": {k: _COMPILES[k] for k in ("make_ttir", "make_cubin")}, "cache_groups_relocated": n_groups, "cache_files_installed": n_files, "jit_meter_n0": _JIT_RUNS.get("n0", 0),
                   "install_order": "triton_key -> install+relocate -> driver/target read", "cache_of_record": cache_note, "forwards_in_apply": 0, "subprocesses_in_apply": 0,
                   "r_cache_own": (inst if rec else {"files": 0, "own_installed": 0, "own_total": 0, "foreign_skipped": 0, "source": "no shipped cache (cold)"}),
                   "install_form": (inst.get("install_form") if rec else "cold: no shipped cache (the process TRITON_CACHE_DIR / Triton's default)"),
                   "tile": tile, "tile_selected_by": tile_how, "precision": precision, "tc_tile": (_tc()._TILE if precision == "tf32" and not tc_tile else tc_tile), "default_route": dict(zip(("route", "basis"), default_route())),
                   "cache_identity": ({"identity_key": rec.get("identity_key"), "identity_form": rec.get("identity_form"), "build_artefacts_recorded": len(rec.get("build_artefacts") or {})} if rec else None),
                   "batch_bound": f"batch bound: <= {K1_MAX_BATCH} units per forward call (the K1 tile indexes n*C*L in int32: < 2^31 => <= {_INT32_ELEMENTS // (512 * INPUT_LEN)} units at C=512, L={INPUT_LEN}); predict() CHUNKS larger batches through the class's tile (the cap printed once); a direct forward past the bound REFUSES by name (ValueError), the process alive (CHK-CBP-48)",
                   "typing_extensions_guard": dict(TE_GUARD),
                   "r8_line": (f"typing_extensions {TE_GUARD.get('action')}; batch bound <= {K1_MAX_BATCH}/call (int32 index; larger batches chunked; a direct forward past 2^31 elements refuses by name); " + (f"installed {inst['own_installed']}/{inst['own_total']} own entries; foreign {inst['foreign_skipped']} (from {inst['source']}); form {inst['install_form']};" if rec else f"shipped cache NOT APPLIED ({_CACHE_SKIP.get('note')}) — this process JITs (CHK-CBP-54: no HIT wording on this path);") + f" kernels.py {_kernels_sha()} (jit sources {_kernels_jit_sha()}); cache identity = IR/PTX key {rec.get('identity_key', 'n/a (pre-rule record)')} (cubin/.so = build artefacts, datum); {inst.get('device_datum', '')}; class route: {default_route()[0]} ({default_route()[1][:160]})" if rec else f"installed 0/0 own entries; foreign 0 (cold: no shipped cache{'; ' + _CACHE_SKIP['note'] if _CACHE_SKIP.get('note') else ''})") + (f"; {_CACHE_SKIP['device_expectation']}" if _CACHE_SKIP.get("device_expectation") else ""),
                   "not_applicable": _CACHE_SKIP.get("note"), "device_expectation": _CACHE_SKIP.get("device_expectation")}


def build_cache_record(cache_dir, out=None):
    """Write JIT_IDENTITY.json for a populated Triton cache dir — run in the populating process, after its forwards: triton/torch/cuDNN
    versions, device, the kernels.py shas, the Triton identity (_triton_identity), the entry list and the IR/PTX identity key. This is the
    record apply() compares a shipped cache against. Returns the record."""
    import json
    rec = {"triton": _triton_version(), "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0), "kernels_sha256_16": _kernels_sha(), "kernels_jit_sha256_16": _kernels_jit_sha(), **_triton_identity(), "build_dir": os.path.abspath(cache_dir), "populate_path": os.path.abspath(__import__("chrombpnet_k1.kernels", fromlist=["x"]).__file__), "canonical_populate_path": "/opt/cbp_build/chrombpnet_k1/kernels.py", "record_version": 2, "entries": triton_cache_record(cache_dir)}
    rec.update(cache_identity_key(cache_dir, rec))
    json.dump(rec, open(out or os.path.join(cache_dir, "JIT_IDENTITY.json"), "w"), indent=1); return rec


IR_SUFFIXES = (".ttir", ".ttgir", ".llir", ".ptx", ".json")           # the identity-bearing files
BUILD_ARTEFACT_SUFFIXES = (".cubin", ".so")                            # ptxas / host-gcc outputs: not byte-reproducible across machines of one device class -> reported, never part of the key
_CACHE_SKIP = {}   # why a shipped cache was NOT applied (printed on the line; returned by apply() as not_applicable)


def cache_identity_key(cache_dir, ident=None, only_entries=None):
    """The identity key of a Triton cache dir: sha256[:16] over {the IR/PTX/json shas per entry (ttir/ttgir/llir/ptx/json; __grp__ jsons
    excluded because they carry paths), triton_key, backend_hash, target}. cubin/.so shas are returned separately as build artefacts and never
    enter the key: ptxas and the host compiler do not produce byte-identical files across machines of one device class, whereas the IR/PTX are
    identical. only_entries restricts the key to this kit's own entries (an effective cache dir may also hold the user's)."""
    import json as _json
    ident = ident or _triton_identity(); ir = {}; art = {}
    for entry in sorted(os.listdir(cache_dir)):
        d = os.path.join(cache_dir, entry)
        if not os.path.isdir(d): continue
        if only_entries is not None and entry not in only_entries: continue      # the key over THIS kit's own entries only (an effective dir may hold foreign / the user's entries)
        for f in sorted(os.listdir(d)):
            q = os.path.join(d, f)
            if not os.path.isfile(q) or f.startswith("__grp__"): continue
            h = hashlib.sha256(open(q, "rb").read()).hexdigest()[:16]
            if f.endswith(IR_SUFFIXES): ir[f"{entry[:16]}/{f}"] = h
            elif f.endswith(BUILD_ARTEFACT_SUFFIXES): art[f"{entry[:16]}/{f}"] = h
    basis = _json.dumps({"ir": ir, "triton_key": ident.get("triton_key"), "backend_hash": ident.get("backend_hash"), "target": ident.get("target")}, sort_keys=True)
    return {"identity_key": hashlib.sha256(basis.encode()).hexdigest()[:16], "identity_form": "sha256-16 over {IR/PTX/json shas per entry (ttir/ttgir/llir/ptx/json; __grp__ excluded) + triton_key + backend_hash + target}; cubin/.so excluded (build artefacts)",
            "ir_files": len(ir), "build_artefacts": art, "build_artefact_note": "cubin/.so shas are a datum (ptxas/host gcc are not deterministic across boxes of one class); a difference is printed on the R8 line, never a refusal"}


def cache_identity_compare(shipped_dir, other_dir):
    """Compare the files two cache dirs have in common (__grp__ jsons excluded): identity_equal = every common IR/PTX/json file has the same
    sha256; differing cubin/.so files are listed separately as information. Returns {identity_equal, ir_differing, n_ir_compared,
    artefacts_differing, n_artefacts_compared, n_common_files}."""
    def files(d):
        out = {}
        for entry in sorted(os.listdir(d)):
            e = os.path.join(d, entry)
            if not os.path.isdir(e): continue
            for f in sorted(os.listdir(e)):
                q = os.path.join(e, f)
                if os.path.isfile(q) and not f.startswith("__grp__"): out[f"{entry}/{f}"] = hashlib.sha256(open(q, "rb").read()).hexdigest()[:16]
        return out
    a, b = files(shipped_dir), files(other_dir); common = sorted(set(a) & set(b))
    ir_diff = [k for k in common if k.endswith(IR_SUFFIXES) and a[k] != b[k]]; art_diff = [k for k in common if k.endswith(BUILD_ARTEFACT_SUFFIXES) and a[k] != b[k]]
    return {"identity_equal": not ir_diff, "ir_differing": ir_diff[:8], "n_ir_compared": sum(k.endswith(IR_SUFFIXES) for k in common), "artefacts_differing": art_diff[:8], "n_artefacts_compared": sum(k.endswith(BUILD_ARTEFACT_SUFFIXES) for k in common), "n_common_files": len(common)}
