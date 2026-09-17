"""chrombpnet_fastkit — host-side levers for `chrombpnet pred_bw` (chrombpnet 1.0.1, TF 2.8.0).

The forward is the stock Keras graph (chrombpnet's load_model_wrapper wrapped in tf.function, or the stock's own model.predict
on the file-order route) unless an external forward callable is passed in; this package changes the host side of the job:
  jit_cache_boot()              install a shipped CUDA ComputeCache tarball add-only into the effective cache dir and export
                                CUDA_CACHE_PATH / CUDA_CACHE_MAXSIZE — must run before any CUDA context exists
  Genome / featurize()          memmap + lookup-table one-hot featurisation with the stock get_seq window rule (int8, the same
                                bytes as chrombpnet's dna_to_one_hot)
  write_bigwig*()               numpy bigWig writer with bigwig_helper.write_bigwig's sort order and overlap rule (stock block
                                layout; span/step blocks as an option)
  install_native_dilation()     dilated Conv1D through tf.nn.conv2d(dilations=) instead of TF's SpaceToBatch route
  predict_pipelined()           the pred_bw job: sorted-order chunked featurise -> forward -> softmax*exp(logcounts) -> bigWig,
                                the writer running in one pre-spawned child interpreter, featurisation of the next chunk
                                prefetched on a thread, stock-shaped tail or pad-to-batch handling of the remainder batch
  write_outputs_from_heads()    stage 2 (preds.bed, bigWig, -os stats) from precomputed heads
No silent fallback: every unit is counted through the kit path; the kit_stamp dict is returned with each job and a count
mismatch raises. Dropped units (window outside the chromosome = the stock get_seq rule) are returned by name. Once a writer
is registered, an uncaught exception releases the writer children, removes their partial outputs and exits 1 — never a hang.
"""
import os, json, hashlib, sys, time, json, hashlib, threading, tarfile, multiprocessing as mp

# Fail-fast hook: writers are registered when they are bound to a job; an uncaught exception anywhere after that releases them
# (the abort sentinel, join(5), terminate), unlinks their partial outputs and exits 1 with the reason on the kit's line — never a hang.
_LIVE_WRITERS = []
_PARTIAL_OUTPUTS = []   # the temp-named (.partial) outputs of the live writers; unlinked by the fail-fast release
ABORT_SENTINEL = "__chrombpnet_fastkit_abort__"   # the release's sentinel — the writer closes WITHOUT the rename and unlinks its partial
_FINAL_OUTPUTS_NEW = []   # final names that did NOT exist when the writer was bound: unlinked by the release if the writer renamed anyway
def _register_partial(final_path):
    """Register a writer's outputs for the fail-fast release: the partial name always; the final name only if it does not exist yet (this process would be its creator)."""
    _PARTIAL_OUTPUTS.append(final_path + ".partial")
    if not os.path.exists(final_path): _FINAL_OUTPUTS_NEW.append(final_path)
_FAIL_FAST = {"installed": False, "orig": None}
def _release_writers(reason):
    n = 0
    for q, wp in list(_LIVE_WRITERS):
        try: q.put(ABORT_SENTINEL)   # abort, never the success sentinel (a plain None would make the writer rename its partial onto the final name)
        except Exception: pass
        try:
            if hasattr(wp, "join"): wp.join(5)
            if getattr(wp, "is_alive", lambda: False)() and hasattr(wp, "terminate"): wp.terminate(); wp.join(5)
        except Exception: pass
        n += 1
    for p in list(_PARTIAL_OUTPUTS) + list(_FINAL_OUTPUTS_NEW):
        try:
            if os.path.exists(p): os.unlink(p)
        except Exception: pass
    _PARTIAL_OUTPUTS.clear(); _FINAL_OUTPUTS_NEW.clear(); _LIVE_WRITERS.clear(); return n
def _install_fail_fast():
    if _FAIL_FAST["installed"]: return
    _FAIL_FAST["orig"] = sys.excepthook
    def hook(exc_type, exc, tb):
        n = _release_writers("uncaught " + exc_type.__name__)
        try: (_FAIL_FAST["orig"] or sys.__excepthook__)(exc_type, exc, tb)
        except Exception: pass
        print("[chrombpnet_fastkit] FAIL-FAST: uncaught %s: %s — %d writer child(ren) released (sentinel + join/terminate); exit 1 (never a hang: the stock's own error class surfaces fast under the kit's line)" % (exc_type.__name__, str(exc)[:200], n), file=sys.stderr, flush=True)
        try: sys.stdout.flush()
        except Exception: pass
        os._exit(1)
    sys.excepthook = hook; _FAIL_FAST["installed"] = True
import numpy as np, pandas as pd
from ._oom import is_oom                                          # the kit's one out-of-memory classifier: a handler below that reroutes re-raises an OOM first

def softmax(x, temp=1):                                          # chrombpnet.evaluation.make_bigwigs.predict_to_bigwig.softmax, verbatim
    norm_x = x - np.mean(x, axis=1, keepdims=True)
    return np.exp(temp * norm_x) / np.sum(np.exp(temp * norm_x), axis=1, keepdims=True)


__version__ = "0.12.73+ho2"
KIT_LEVERS = ["L5_jit_cache", "L2_memmap_lut_featurise", "L1_numpy_bigwig_writer", "L3_batch", "L4_tf_function_forward", "L7_sorted_overlapped_writer", "L10_numpy_bigwig_backend", "L8_maxzooms", "L9_pad_to_batch", "F2_native_dilated_conv", "L7c_featurise_prefetch"]
NARROWPEAK_SCHEMA = ["chr", "start", "end", "1", "2", "3", "4", "5", "6", "summit"]


def md5_array(a):
    return hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


# ---------------------------------------------------------------- L5: JIT cache
JIT_CACHE_STAMP = {}   # jit_cache_boot()'s install stamp (the entry script folds it into its run record)

INDEX_MEMBERS = ("index",)   # the ComputeCache's index file(s) — the driver's bookkeeping, rewritten on use; never counted as a cache rewrite
MARKER_PREFIX = ".chrombpnet_fastkit_cache_installed_"   # the kit's own one-time install marker files inside the effective cache dir (never cache members)
def cuda_libs_fingerprint():
    """First 16 hex digits of the sha256 of the CUDA libraries TensorFlow 2.8 resolves (libcudnn.so.8, libcublas.so.11, libcudart.so.11.0), located
        through `ldconfig -p` — the image key of a shipped driver cache (the same GPU and driver under different CUDA libraries JIT-compiles different
        PTX modules). Best effort: {} when unresolvable; an out-of-memory error propagates."""
    out = {}
    try:
        import subprocess as _sp, re as _re2
        lp = _sp.run(["/sbin/ldconfig", "-p"], capture_output=True, text=True, timeout=20).stdout
        for lib in ("libcudnn.so.8", "libcublas.so.11", "libcudart.so.11.0"):
            m = _re2.search(r"^\s*" + _re2.escape(lib) + r" \(.*?\) => (\S+)$", lp, _re2.M)
            if m and os.path.exists(m.group(1)):
                h = hashlib.sha256()
                with open(m.group(1), "rb") as fh:
                    for chunk in iter(lambda: fh.read(1 << 22), b""): h.update(chunk)
                out[lib] = h.hexdigest()[:16]
    except Exception as e:
        if is_oom(e): raise                                           # an out-of-memory error propagates; a lib left unfingerprinted (-> the stock JIT) is for every other failure
    return out
def image_key_status(want, here):
    """Compare this host's CUDA-libs fingerprint `here` with the images a shipped driver cache was captured on (the sidecar's
        captures[].cuda_libs_sha16 plus its own cuda_libs_sha16). Fingerprint in the list: witnessed_on_this_image=True; a foreign fingerprint: False with
        a note (the shipped members may not cover this image's kernels; cache_content_witness() after the first forward tells); no fingerprint on either
        side: None. Never a refusal."""
    known = []
    for c in (want.get("captures") or []):
        if isinstance(c, dict) and c.get("cuda_libs_sha16"): known.append((c.get("image"), c["cuda_libs_sha16"]))
    if want.get("cuda_libs_sha16") and want["cuda_libs_sha16"] not in [k[1] for k in known]: known.append((str(want.get("image"))[:80], want["cuda_libs_sha16"]))
    if not known: return {"witnessed_on_this_image": None, "known_images": 0, "here": here or None, "note": "the sidecar carries no CUDA-libs fingerprint (captured before v0.12.64): the image key is unknown"}
    if not here: return {"witnessed_on_this_image": None, "known_images": len(known), "here": None, "note": "this box's CUDA-libs fingerprint is unresolvable (ldconfig): the image key is unknown here"}
    for img, fp in known:
        if fp == here: return {"witnessed_on_this_image": True, "known_images": len(known), "matched_image": img, "here": here}
    return {"witnessed_on_this_image": False, "known_images": len(known), "here": here, "note": f"a foreign image (this box's CUDA libs match none of the {len(known)} witnessed images): the shipped members may not cover this image's kernels — the content witness after the first forward is the proof (stated, never a refusal)"}
def _shipped_members(tarball):
    """The shipped members {relpath: sha256} from the MEMBERS.sha256 sidecar beside the tarball ({} when absent or unreadable) — the reference for cache_content_witness(), loaded on every loader path."""
    mpath = str(tarball) + ".MEMBERS.sha256"
    try: return {l.split("  ", 1)[1].strip(): l.split("  ", 1)[0].strip() for l in open(mpath) if "  " in l} if os.path.exists(mpath) else {}
    except Exception: return {}
def jit_cache_boot(tarball, local_dir=None, maxsize=4294967296):
    # the target = the process's EFFECTIVE ComputeCache dir: CUDA_CACHE_PATH if the user set it, else ~/.nv/ComputeCache
    if local_dir is None: local_dir = os.environ.get("CUDA_CACHE_PATH") or os.path.join(os.path.expanduser("~"), ".nv", "ComputeCache")
    """Untar the persisted CUDA JIT cache to local disk and export CUDA_CACHE_PATH / CUDA_CACHE_MAXSIZE.
    MUST run before the CUDA context exists (before TensorFlow touches the GPU). Boxes never write the shared cache.
    The cache is keyed on the driver version + the TF 2.8.0/CUDA 11.2/cuDNN 8.1 PTX; a driver bump re-JITs once. Returns seconds."""
    t0 = time.time()
    parent = os.path.dirname(local_dir.rstrip("/")); os.makedirs(parent, exist_ok=True)
    top = os.path.basename(local_dir.rstrip("/"))
    # A .tar.zst cache tarball is read through a zstandard stream (no silent fallback: a .zst without the module raises); on the staged path the
    # top-level-dir check and the extraction run in ONE streaming pass for both tar forms (a streamed tar cannot be listed then extracted).
    # The identity sidecar (<tarball>.identity.json, else <tarball>.key.json) is compared with nvidia-smi's answer (a subprocess, BEFORE any CUDA
    # context); a missing sidecar = unkeyed (stated in the stamp); a mismatch is named in the stamp, never a refusal (see below).
    key_path = tarball + ".key.json"; key = None; gpu = None
    ident_path = tarball + ".identity.json"; ident = json.load(open(ident_path)) if os.path.exists(ident_path) else None
    try:
        import subprocess
        from .fastdefault import nvsmi_query as _nvq   # ONE nvidia-smi per process (memoized with the class probe)
        _q = _nvq()
        if "error" in _q: raise RuntimeError(_q["error"])
        gpu = {"name": _q["name"], "driver": _q["driver"], "cc": _q["cc"]}
    except Exception as e:
        if is_oom(e): raise                                           # an out-of-memory error propagates; every other failure leaves the GPU unidentified (named in the stamp) and the install proceeds
        gpu = {"error": f"{type(e).__name__}: {str(e)[:80]}"}
    if os.path.exists(key_path): key = json.load(open(key_path))
    # the identity stamp is NAMED, never a gate: a tarball captured on another driver / device name is installed add-only all the same (CUDA keys its
    # ComputeCache entries by driver and device: members that do not apply are simply not hit and the driver JIT-compiles what it does not find — a cost, said)
    want = ident or ({"driver": key.get("driver"), "gpu_class": key.get("gpu_class")} if key else None)
    if want and "driver" in gpu:
        mism = []
        if want.get("driver") and want["driver"] != gpu["driver"]: mism.append(f"driver {want['driver']} vs {gpu['driver']}")
        if want.get("gpu_class") and str(want["gpu_class"]).upper() not in gpu["name"].upper(): mism.append(f"gpu {want['gpu_class']} vs {gpu['name']}")
        if want.get("gpu_name") and want["gpu_name"] != gpu["name"]: mism.append(f"gpu {want['gpu_name']} vs {gpu['name']}")
        if want.get("cc") and gpu.get("cc") and str(want["cc"]) != str(gpu["cc"]): mism.append(f"cc {want['cc']} vs {gpu['cc']}")
        # the CUDA-libs fingerprint — a differing image is STATED (the shipped members may not cover this image's kernels), never a mismatch
        JIT_CACHE_STAMP["image_key"] = image_key_status(want, cuda_libs_fingerprint() if (want.get("cuda_libs_sha16") or want.get("captures")) else {})   # against the LIST of images the cache was captured on
        if mism:
            JIT_CACHE_STAMP["identity_note"] = "captured on " + "; ".join(mism) + " here — installed add-only; the driver JIT-compiles what it does not find in it (a one-time cost on this box)"
    JIT_CACHE_STAMP["identity"] = want; JIT_CACHE_STAMP["box_gpu"] = gpu
    if tarball.endswith(".zst"):
        import zstandard                               # no silent fallback: a .zst without the module raises here
        _fh = open(tarball, "rb"); _rd = zstandard.ZstdDecompressor().stream_reader(_fh); _tf = tarfile.open(fileobj=_rd, mode="r|")
    else:
        _fh = None; _rd = None; _tf = tarfile.open(tarball, mode="r|")
    # Loader rules: (1) no driver/context use before the install — this loader runs before TensorFlow touches the GPU and reads the GPU identity
    # through an nvidia-smi SUBPROCESS only (touching the driver first has crashed the process); (2) NEVER overwrite an existing cache file with
    # different bytes: into a non-empty dir the tarball is extracted to a STAGING dir, an existing identical file is skipped, an existing DIFFERENT
    # file is kept and named in the stamp, everything else is moved in add-only.
    import shutil
    # the install MARKER (one-time install semantics), keyed by the shipped tar's sha16 (the identity sidecar's tar_sha256; else the tar's size+mtime)
    _tkey = (ident or {}).get("tar_sha256") if ident else None
    _tkey = (_tkey[:16] if _tkey else "sz%d_mt%d" % (os.path.getsize(tarball), int(os.path.getmtime(tarball))))
    _marker = os.path.join(local_dir, ".chrombpnet_fastkit_cache_installed_%s.json" % _tkey)
    if os.path.exists(_marker):
        try: _mk = json.load(open(_marker))
        except Exception: _mk = {}
        os.environ["CUDA_CACHE_PATH"] = local_dir; os.environ["CUDA_CACHE_MAXSIZE"] = str(maxsize)
        _present = sum(1 for _r, _, _fs in os.walk(local_dir) for _f in _fs if not _f.startswith(".chrombpnet_fastkit_cache_installed_"))
        JIT_CACHE_STAMP.update({"tarball": tarball, "local_dir": local_dir, "untar_s": round(time.time() - t0, 3), "entries_after_untar": _present, "gpu": gpu, "identity": ident,
                                "install_marker": {"path": _marker, "installed_utc": _mk.get("utc"), "members": _mk.get("members"), "installed": _mk.get("installed"), "kept_different": _mk.get("kept_different")},
                                "loader_rule": "form (2), one-time install: the marker %s present (installed %s: %s members) -> no staging, no compare (O(1)); the effective dir %s holds %d files; identity by nvidia-smi subprocess before any CUDA context" % (os.path.basename(_marker), _mk.get("utc"), _mk.get("members"), local_dir, _present)})
        JIT_CACHE_STAMP["members"] = _shipped_members(tarball)   # the reference for cache_content_witness() on the marker-HIT path too
        return time.time() - t0
    # An EMPTY effective dir takes the ONE-PASS install — the tar binary when present (else tarfile.extractall), the tarball's one top component
    # renamed onto the effective dir; no per-file walk, no compare (nothing to compare). The staged add-only content-compare below runs only
    # when the dir already holds files. The marker-HIT path (process 2+) is above.
    _existing = [f for _r, _, _fs in os.walk(local_dir) for f in _fs if not f.startswith(".chrombpnet_fastkit_cache_installed_")] if os.path.isdir(local_dir) else []
    _staged = bool(_existing)
    if not _staged:
        _t_ins = time.time(); _one = local_dir.rstrip("/") + ".onepass"; shutil.rmtree(_one, ignore_errors=True); os.makedirs(_one, exist_ok=True)
        try: _tf.close()
        except Exception: pass
        if _rd is not None: _rd.close()
        if _fh is not None: _fh.close()
        _tar_bin = shutil.which("tar"); _how = None
        if _tar_bin and os.path.isfile(tarball):
            import subprocess as _sp
            _p = _sp.run([_tar_bin, "-xf", tarball, "-C", _one], capture_output=True, text=True)
            if _p.returncode == 0: _how = "tar binary (%s)" % _tar_bin
        if _how is None:
            with tarfile.open(tarball) as _tf2: _tf2.extractall(_one); _how = "tarfile.extractall"
        _tops = os.listdir(_one)
        if len(_tops) != 1: raise RuntimeError(f"[jit_cache_boot] REFUSED: the tarball must carry ONE top-level dir (found {_tops!r}) — repack it")
        _tar_top = _tops[0]; n_members = sum(len(_fs) for _r, _, _fs in os.walk(os.path.join(_one, _tar_top)))
        if n_members == 0: raise RuntimeError("[jit_cache_boot] REFUSED: the tarball is empty")
        if os.path.isdir(local_dir): shutil.rmtree(local_dir, ignore_errors=True)   # an empty dir (or the marker-less remains of one)
        os.makedirs(os.path.dirname(local_dir.rstrip("/")) or ".", exist_ok=True); os.replace(os.path.join(_one, _tar_top), local_dir); shutil.rmtree(_one, ignore_errors=True)
        conflicts = []; skipped_identical = 0; moved = n_members; kept_different = 0
        JIT_CACHE_STAMP.update({"loader_rule": "form (2), one-pass: the effective dir %s was EMPTY -> the tarball's %d members installed in one pass (%s, the top component renamed into place; no compare) in %.2f s; identity by nvidia-smi subprocess before any CUDA context" % (local_dir, n_members, _how, time.time() - _t_ins), "install_form": "one-pass (empty dir)", "install_s": round(time.time() - _t_ins, 3)})
    staging = local_dir.rstrip("/") + ".staging"; shutil.rmtree(staging, ignore_errors=True); os.makedirs(staging, exist_ok=True)
    if _staged: _tar_top = None
    if _staged: n_members = 0
    with (_tf if _staged else open(os.devnull, "rb")) as tf_:
      if _staged:
          for m in tf_:
              # the tarball's ONE top-level component is stripped whatever the target dir's name (the shipped tar's top dir is nv_compute_cache/,
              # the effective dir is usually ComputeCache/)
              _tc = m.name.split("/")[0]
              if _tar_top is None: _tar_top = _tc
              if _tc != _tar_top:
                  raise RuntimeError(f"[jit_cache_boot] REFUSED: the tarball must carry ONE top-level dir (found {_tar_top!r} and {_tc!r}) — repack it")
              tf_.extract(m, staging); n_members += 1
    if _rd is not None: _rd.close()
    if _fh is not None: _fh.close()
    if _staged and n_members == 0:
        raise RuntimeError("[jit_cache_boot] REFUSED: the tarball is empty")
    if _staged: conflicts = []; skipped_identical = 0; moved = 0
    for r, _, fs in (os.walk(os.path.join(staging, _tar_top)) if _staged else []):
        for f in fs:
            src = os.path.join(r, f); rel = os.path.relpath(src, os.path.join(staging, _tar_top)); dst = os.path.join(local_dir, rel)
            if os.path.exists(dst):
                if hashlib.sha256(open(dst, "rb").read()).hexdigest() == hashlib.sha256(open(src, "rb").read()).hexdigest(): skipped_identical += 1
                else: conflicts.append(rel)
    # existing files with DIFFERENT bytes are KEPT (never overwritten) and the rest is installed ADD-ONLY — no fallback to an empty dir (that would
    # mean the full PTX JIT on this host); the kept count is a value in the stamp
    if _staged: kept_different = len(conflicts)
    for r, _, fs in (os.walk(os.path.join(staging, _tar_top)) if _staged else []):
        for f in fs:
            src = os.path.join(r, f); rel = os.path.relpath(src, os.path.join(staging, _tar_top)); dst = os.path.join(local_dir, rel)
            if os.path.exists(dst): continue
            os.makedirs(os.path.dirname(dst), exist_ok=True); shutil.move(src, dst); moved += 1
    shutil.rmtree(staging, ignore_errors=True)
    if _staged: JIT_CACHE_STAMP.update({"loader_rule": "form (2): add-only into the effective ComputeCache dir %s; existing identical skipped %d, existing DIFFERENT kept %d (never overwritten; reported), installed %d; identity by nvidia-smi subprocess before any CUDA context" % (local_dir, skipped_identical, kept_different, moved), "kept_different": conflicts[:20]})
    try:
        json.dump({"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "tar_key": _tkey, "tarball": tarball, "members": n_members, "installed": moved, "skipped_identical": skipped_identical, "kept_different": kept_different, "kit": "chrombpnet_fastkit v0.12.39 (one-time install marker; add-only bookkeeping)"}, open(_marker, "w"), indent=1)
        JIT_CACHE_STAMP["install_marker"] = {"path": _marker, "written": True}
    except Exception as _e:
        JIT_CACHE_STAMP["install_marker"] = {"path": _marker, "written": False, "error": f"{type(_e).__name__}: {str(_e)[:80]}"}
    os.environ["CUDA_CACHE_PATH"] = local_dir
    os.environ["CUDA_CACHE_MAXSIZE"] = str(maxsize)
    ents = [os.path.join(r, f) for r, _, fs in os.walk(local_dir) for f in fs]
    # the shipped members' shas (MEMBERS.sha256 beside the tarball when shipped; else computed now) = the reference for cache_content_witness()
    members = _shipped_members(tarball) or {os.path.relpath(e, local_dir): hashlib.sha256(open(e, "rb").read()).hexdigest() for e in ents if not os.path.basename(e).startswith(MARKER_PREFIX)}
    JIT_CACHE_STAMP["members"] = members
    JIT_CACHE_STAMP.update({"tarball": tarball, "tarball_sha256_16": hashlib.sha256(open(tarball, "rb").read()).hexdigest()[:16], "key": key, "box_gpu": gpu,
                            "entries_after_untar": len(ents), "bytes_after_untar": sum(os.path.getsize(e) for e in ents), "untar_s": round(time.time() - t0, 3), "cuda_cache_path": local_dir, "cuda_cache_maxsize": maxsize})
    return time.time() - t0


# ---------------------------------------------------------------- L2: featurisation
def cache_content_witness(cache_dir, members):
    """Content comparison of the effective cache dir after the first forward: sha256 of every file under cache_dir vs the shipped
        members {relpath: sha256} (install markers and the driver's index file excluded). HIT = 0 rewritten AND 0 new; else MISS with the counts
        (a rewrite = the JIT ran under the same key; a new file = a kernel the shipped members do not cover)."""
    if not cache_dir or not os.path.isdir(cache_dir): return {"witness": "content", "verdict": "n/a (no cache dir)", "rewritten": None, "new": None, "missing": None}
    now = {}; markers = 0; index_rewritten = []
    for r, _, fs in os.walk(cache_dir):
        for f in fs:
            if f.startswith(MARKER_PREFIX): markers += 1; continue   # the kit's own install marker is never a cache member
            p = os.path.join(r, f); now[os.path.relpath(p, cache_dir)] = hashlib.sha256(open(p, "rb").read()).hexdigest()
    members = members or {}
    rewritten = sorted(k for k in now if k in members and now[k] != members[k]); new = sorted(k for k in now if k not in members); missing = sorted(k for k in members if k not in now)
    index_rewritten = [k for k in rewritten if os.path.basename(k) in INDEX_MEMBERS]; rewritten = [k for k in rewritten if k not in index_rewritten]   # the driver's index is rewritten on use — named, never counted as a rewrite
    verdict = "HIT" if (members and not rewritten and not new) else ("MISS (%d rewritten, %d new, %d missing of %d shipped)" % (len(rewritten), len(new), len(missing), len(members)) if members else "MISS (no shipped-member list on this path; %d entries in the dir — a witness without a reference, stated)" % len(now))
    return {"witness": "content", "verdict": verdict, "rewritten": rewritten[:20], "new": new[:20], "missing": missing[:20], "entries_now": len(now), "shipped": len(members), "markers": markers, "index_rewritten": index_rewritten[:5]}
_LUT = np.full(256, 4, dtype=np.uint8)
for _ch, _v in (("A", 0), ("C", 1), ("G", 2), ("T", 3)):
    _LUT[ord(_ch)] = _v; _LUT[ord(_ch.lower())] = _v
_EYE = np.identity(5, dtype=np.int8)[:, :4]


class Genome:
    """Random access to a fasta through its .fai (fixed line width per record) via a memmap; no pyfaidx string slicing."""
    def __init__(self, fasta):
        if not os.path.exists(fasta + ".fai"):
            raise FileNotFoundError(f"{fasta}.fai missing — build it with pyfaidx.Faidx(fasta) (as the stock does) before calling the kit")
        self.fai = pd.read_csv(fasta + ".fai", sep="\t", header=None, names=["name", "length", "offset", "linebases", "linewidth"], usecols=[0, 1, 2, 3, 4]).set_index("name")
        self.mm = np.memmap(fasta, dtype=np.uint8, mode="r")

    def windows_ok(self, chroms, starts, width):
        ok = np.zeros(len(chroms), dtype=bool)
        for chrom in pd.unique(chroms):
            sel = np.where(chroms == chrom)[0]
            if chrom not in self.fai.index:
                continue
            L = int(self.fai.loc[chrom, "length"]); st = starts[sel]
            ok[sel] = (st >= 0) & (st + width <= L)
        return ok

    def one_hot(self, chroms, starts, width):
        """int8 one-hot (n, width, 4); identical bytes to chrombpnet.training.utils.one_hot.dna_to_one_hot on the same windows."""
        out = np.empty((len(chroms), width, 4), dtype=np.int8); ar = np.arange(width, dtype=np.int64)
        for chrom in pd.unique(chroms):
            sel = np.where(chroms == chrom)[0]
            off, lb, lw = (int(self.fai.loc[chrom, c]) for c in ("offset", "linebases", "linewidth"))
            pos = starts[sel][:, None] + ar[None, :]
            out[sel] = _EYE[_LUT[self.mm[off + (pos // lb) * lw + (pos % lb)]]]
        return out


def featurize(regions_df, fasta, inputlen):
    """Stock get_seq semantics: window = [start+summit-inputlen//2, +inputlen); windows that do not fit the chromosome are dropped.
    Returns (one_hot int8 (n_used, inputlen, 4), used bool mask over regions_df rows)."""
    g = Genome(fasta)
    centers = (regions_df["start"].values + regions_df["summit"].values).astype(np.int64); starts = centers - inputlen // 2
    chroms = regions_df["chr"].values
    used = g.windows_ok(chroms, starts, inputlen)
    return g.one_hot(chroms[used], starts[used], inputlen), used


# ---------------------------------------------------------------- L1 (+L7): writer
def read_chrom_sizes(fname):
    with open(fname) as f:
        gs = [x.strip().split("\t") for x in f]
    return [(x[0], int(x[1])) for x in gs if len(x) == 2]


def _write_regions(bw, regions, prof, i0, state, N, entries=None):
    """bigwig_helper.write_bigwig's overlap resolution, same semantics: regions sorted by (chrom order, start);
    an overlapping next region truncates this one at the midpoint of the two summits."""
    cur_chr, cur_end = state
    for k in range(prof.shape[0]):
        i = i0 + k; i_chr, i_start, i_end, i_mid = regions[i]
        if i_chr != cur_chr:
            cur_chr, cur_end = i_chr, 0
        if cur_end < i_start:
            cur_end = i_start
        next_end = i_end
        if i + 1 != N:
            next_chr, next_start, _, next_mid = regions[i + 1]
            if next_chr == i_chr and next_start < i_end:
                next_end = (i_mid + next_mid) // 2
        vals = prof[k][cur_end - i_start:next_end - i_start]
        if entries is not None: entries.append(vals)              # bigwig_helper.write_bigwig's all_entries (same slices, same order) for the -os stats file
        if len(vals):
            bw.addEntries(i_chr, int(cur_end), values=vals, span=1, step=1)
        cur_end = next_end
    return (cur_chr, cur_end)


def _regions_arrays(regions):
    """The regions list as arrays once per writer: chromosome ids in order of first appearance (the ids only compare equality), starts, ends, mids."""
    chr_ids = {}; cid = np.fromiter((chr_ids.setdefault(r[0], len(chr_ids)) for r in regions), dtype=np.int64, count=len(regions))
    st = np.fromiter((r[1] for r in regions), dtype=np.int64, count=len(regions)); en = np.fromiter((r[2] for r in regions), dtype=np.int64, count=len(regions))
    md = np.fromiter((r[3] for r in regions), dtype=np.int64, count=len(regions))
    names = [None] * len(chr_ids)
    for k, v in chr_ids.items(): names[v] = k
    return {"cid": cid, "start": st, "end": en, "mid": md, "names": names}


def _write_regions_vec(bw, R, regions, prof, i0, state, N, entries=None):
    """VECTORISED form of _write_regions (same semantics, same bytes): the overlap resolution per region depends only on its neighbours
    (next_end_i on region i+1; cur_end_i on next_end_{i-1} and start_i), so the chunk's windows are computed in numpy; the windows are
    concatenated in file order and handed to add_intervals ONCE PER CHROMOSOME SEGMENT of the chunk — equivalent to the per-region calls
    because consecutive sorted regions on one chromosome always satisfy pyBigWig's append rule (start >= the last end), while a chromosome
    change is exactly where the per-region form flushes. `entries` (the -os stats file's all_entries) keeps the per-region slices."""
    n = prof.shape[0]; L = prof.shape[1]; i1 = i0 + n
    cid = R["cid"][i0:i1]; st = R["start"][i0:i1]; en = R["end"][i0:i1]; md = R["mid"][i0:i1]
    # next_end_i: truncated at the midpoint of the two summits when region i+1 (same chromosome) starts before end_i
    nxt_end = en.copy()
    if i1 < N:
        ncid = R["cid"][i0 + 1:i1 + 1]; nst = R["start"][i0 + 1:i1 + 1]; nmd = R["mid"][i0 + 1:i1 + 1]
    else:
        ncid = np.append(R["cid"][i0 + 1:i1], -1); nst = np.append(R["start"][i0 + 1:i1], 0); nmd = np.append(R["mid"][i0 + 1:i1], 0)
    ov = (ncid == cid) & (nst < en); nxt_end[ov] = (md[ov] + nmd[ov]) // 2
    # cur_end_i = max(next_end_{i-1}, start_i) on the same chromosome, else start_i (a chromosome change resets cur_end to 0)
    cur_chr, cur_end = state
    prev_end = np.concatenate(([cur_end if (n and R["names"][cid[0]] == cur_chr) else 0], nxt_end[:-1]))
    same_prev = np.concatenate(([n > 0 and R["names"][cid[0]] == cur_chr], cid[1:] == cid[:-1]))
    prev_end = np.where(same_prev, prev_end, 0)
    cur = np.maximum(prev_end, st)
    lo = cur - st; hi = nxt_end - st; lo = np.minimum(np.maximum(lo, 0), L); hi = np.minimum(np.maximum(hi, lo), L)     # the scalar form's slices clip the same way
    cnt = hi - lo
    if entries is not None:
        for k in range(n): entries.append(prof[k][lo[k]:hi[k]])
    # windows in file order: a mask over the (n, L) chunk
    col = np.arange(L)
    mask = (col[None, :] >= lo[:, None]) & (col[None, :] < hi[:, None])
    vals = prof[mask]
    starts = np.repeat(cur, cnt) + (np.arange(int(cnt.sum()), dtype=np.int64) - np.repeat(np.cumsum(cnt) - cnt, cnt))
    # one call per chromosome segment (in order), skipping empty segments like the scalar form skips empty windows
    row_cid = np.repeat(cid, cnt)
    if len(vals):
        bounds = np.flatnonzero(np.diff(row_cid) != 0) + 1; segs = np.split(np.arange(len(vals)), bounds)
        for seg in segs:
            if len(seg) == 0: continue
            a, b = int(seg[0]), int(seg[-1]) + 1
            bw.w.add_intervals(R["names"][int(row_cid[a])], starts[a:b], starts[a:b] + 1, vals[a:b])       # == the shim's addEntries(chrom, start, values) per region, concatenated
    # state after the chunk: the last region's (chr, next_end)
    return (R["names"][int(cid[-1])], int(nxt_end[-1])) if n else state


class _NumpyBW:
    """pyBigWig-shaped shim over bigwig_numpy.NumpyBigWig. layout='stock': the stock CLI's block layout (bigwig_helper.write_bigwig:
        addEntries([chrom]*n, starts, ends=, values=) -> type-1 interval blocks; file byte-identical to the stock CLI's);
        layout='fixedstep': span/step blocks (type 3; same values, smaller file; the bytes pyBigWig writes for that call pattern)."""
    def __init__(self, gs, max_zooms, layout="stock", compress_workers=3):
        from .bigwig_numpy import NumpyBigWig
        self.w = NumpyBigWig(gs, max_zooms, compress_workers=compress_workers); self.layout = layout
    def addEntries(self, chrom, start, values, span=1, step=1):
        assert span == 1 and step == 1
        if self.layout == "stock":
            n = len(values); self.w.add_intervals(chrom, np.arange(start, start + n), np.arange(start + 1, start + n + 1), values)
        else:
            self.w.add_entries(chrom, start, values)


def _writer_proc(q, bw_path, gs, regions, max_zooms, N, statfile, backend="numpy", layout="stock", stats_file=None):
    """The writer loop (child or thread side): consumes (i0, prof) chunks from `q` until the None sentinel (ABORT_SENTINEL = close without
        the rename and drop the partial), writes <bw_path>.partial and renames it on success, then dumps its timings to `statfile`.
        backend='pybigwig': the stock library (zoom levels built at close, serial). backend='numpy': the bigwig_numpy reimplementation — the
        same file bytes as pyBigWig 0.3.22 for the same calls, zoom levels built incrementally per chunk so the close is short."""
    t_zoom = 0.0; t_ready = time.time()                        # the child's ready time (the main process stamps its start; the spawned child imports numpy + this module first)
    def _proc_witness():                                         # what the child carries — page faults (minor/major), RSS, VmSize, threads, from /proc/self
        try:
            st = open("/proc/self/stat").read().split(")")[-1].split(); status = open("/proc/self/status").read()
            g = lambda k: int(status.split(k + ":")[1].split()[0]) if (k + ":") in status else None
            return {"minflt": int(st[7]), "majflt": int(st[9]), "vm_rss_kb": g("VmRSS"), "vm_size_kb": g("VmSize"), "vm_hwm_kb": g("VmHWM"), "threads": g("Threads"), "pid": os.getpid(), "ppid": os.getppid()}
        except Exception as e: return {"error": repr(e)[:80]}
    w_start = _proc_witness()
    if backend == "numpy":
        # the writer's block-compress pool size (a pool size touches no byte): sized from the quota, never a constant — min(max(1, quota - 2), 8)
        from .hostres import effective_cpus, cgroup_cpu_quota
        _cq = cgroup_cpu_quota(); eff = effective_cpus()
        cw_rec = {"value": int(min(max(1, eff - 2), 8)), "rule": "min(max(1, quota - 2), 8)", "quota": eff, "quota_source": ("cgroup quota" if _cq else "NO cgroup quota readable — effective_cpus() fell back to affinity/host count (stated miss; cut 9 supersedes)")}
        bw = _NumpyBW(gs, max_zooms, layout=layout, compress_workers=cw_rec["value"])
    else:
        import pyBigWig
        _bw = pyBigWig.open(bw_path + ".partial", "w"); _bw.addHeader(gs, maxZooms=max_zooms)   # a temp name, renamed on success
        class _PyBW:                                         # pyBigWig in the requested layout (stock = bigwig_helper's list-call form)
            def __init__(self, b): self.b = b
            def addEntries(self, chrom, start, values, span=1, step=1):
                n = len(values)
                if layout == "stock": self.b.addEntries([chrom] * n, list(range(start, start + n)), ends=list(range(start + 1, start + n + 1)), values=values)
                else: self.b.addEntries(chrom, start, values=values, span=1, step=1)
            def close(self): self.b.close()
        bw = _PyBW(_bw)
    _VEC = backend == "numpy" and layout == "stock"     # the vectorised per-chunk form (stock layout only; the same bytes as the scalar form by construction)
    R_arr = _regions_arrays(regions) if _VEC else None
    state = ("", 0); nreg = 0; t_write = 0.0; entries = [] if stats_file else None
    while True:
        item = q.get()
        if item is None:
            break
        if isinstance(item, str) and item == ABORT_SENTINEL:   # the fail-fast release — close without the rename, drop the partial
            try:
                if backend == "numpy": bw.w.close(bw_path + ".partial")
                else: bw.close()
            except Exception: pass
            try: os.unlink(bw_path + ".partial")
            except Exception: pass
            return
        i0, prof = item; tw = time.time()
        if _VEC: state = _write_regions_vec(bw, R_arr, regions, prof, i0, state, N, entries)
        else: state = _write_regions(bw, regions, prof, i0, state, N, entries)
        nreg += prof.shape[0]; t_write += time.time() - tw
        if backend == "numpy":
            tz = time.time(); bw.w.feed_zooms(); t_zoom += time.time() - tz
    tc = time.time()
    if backend == "numpy": bw.w.close(bw_path + ".partial")
    else: bw.close()
    os.replace(bw_path + ".partial", bw_path)               # the final name exists only after a successful close (a failed job leaves 0 files, like the stock)
    if stats_file:                                          # the stock's -os / --output-prefix-stats file: bigwig_helper.write_bigwig's stats block on the same all_entries
        all_entries = np.hstack(entries)
        from .jobwall_post import write_stats_fast        # the stats drop-in (the stock's quantile lines), inside the kit package: the kit is import-closed
        write_stats_fast(all_entries, stats_file)
    json.dump({"regions": nreg, "t_write": t_write, "t_zoom": t_zoom, "t_close": time.time() - tc, "backend": backend, "layout": layout, "stats_file": stats_file,
               "writer_vec": bool(_VEC), "t_ready_epoch": t_ready, "witness_at_start": w_start, "witness_at_close": _proc_witness(), "compress_workers": (cw_rec if backend == "numpy" else None)}, open(statfile, "w"))


def write_bigwig(profile, regions_df_used, chrom_sizes, bw_path, outputlen=1000, max_zooms=10, backend="numpy", layout="stock"):
    """Standalone writer (no overlap with the forward): profile (n, outputlen) float32 in regions_df_used row order. Default = the numpy
    writer in the STOCK layout (file byte-identical to the stock CLI's / bigwig_helper.write_bigwig); backend='pybigwig' writes the
    same stock layout through pyBigWig's list-call form (bigwig_helper's own calls); layout='fixedstep' = span/step blocks (values
    identical, file not byte-identical to the stock's)."""
    gs = read_chrom_sizes(chrom_sizes); chr_to_idx = {x[0]: i for i, x in enumerate(gs)}
    centers = (regions_df_used["start"].values + regions_df_used["summit"].values).astype(np.int64)
    key_chr = np.array([chr_to_idx.get(c, 10 ** 6) for c in regions_df_used["chr"].values])
    order = np.lexsort((centers, key_chr))
    regions = [(regions_df_used["chr"].values[i], int(centers[i] - outputlen // 2), int(centers[i] + outputlen // 2), int(centers[i])) for i in order]
    P = np.ascontiguousarray(profile[order], dtype=np.float32)
    if backend == "numpy":
        bw = _NumpyBW(gs, max_zooms, layout=layout); _write_regions(bw, regions, P, 0, ("", 0), len(regions)); bw.w.feed_zooms(); bw.w.close(bw_path)
    else:
        import pyBigWig
        class _StockCalls:                                   # bigwig_helper.write_bigwig's call form: addEntries([chrom]*n, starts, ends=, values=)
            def __init__(self, bw): self.bw = bw
            def addEntries(self, chrom, start, values, span=1, step=1):
                n = len(values)
                if layout == "stock": self.bw.addEntries([chrom] * n, list(range(start, start + n)), ends=list(range(start + 1, start + n + 1)), values=values)
                else: self.bw.addEntries(chrom, start, values=values, span=1, step=1)
        bw = pyBigWig.open(bw_path, "w"); bw.addHeader(gs, maxZooms=max_zooms); _write_regions(_StockCalls(bw), regions, P, 0, ("", 0), len(regions)); bw.close()


def write_bigwig_with_stats(data, regions, gs, bw_path, max_zooms=10, stats_file=None):
    """bigwig_helper.write_bigwig(data, regions, gs, bw_out, outstats_file=) with the numpy writer: `regions` = get_regions(...) rows
    (chrom, start, end, summit) in regions-file order, `data` (n, outputlen) float32 in the same order; the stock sorts by (chrom order,
    start) and resolves overlaps at summit midpoints — same here (_write_regions) — and the -os stats are the same np.hstack(all_entries)."""
    chr_to_idx = {x[0]: i for i, x in enumerate(gs)}
    order = sorted(range(len(regions)), key=lambda x: (chr_to_idx[regions[x][0]], regions[x][1]))
    regs = [regions[i] for i in order]; P = np.ascontiguousarray(np.asarray(data, dtype=np.float32)[order])
    bw = _NumpyBW(gs, max_zooms, layout="stock"); entries = [] if stats_file else None
    _write_regions(bw, regs, P, 0, ("", 0), len(regs), entries); bw.w.feed_zooms(); bw.w.close(bw_path)
    if stats_file:
        all_entries = np.hstack(entries)
        with open(stats_file, 'w') as f:
            f.write("Min\t{:.6f}\n".format(np.min(all_entries)))
            f.write(".1%\t{:.6f}\n".format(np.quantile(all_entries, 0.001)))
            f.write("1%\t{:.6f}\n".format(np.quantile(all_entries, 0.01)))
            f.write("50%\t{:.6f}\n".format(np.quantile(all_entries, 0.5)))
            f.write("99%\t{:.6f}\n".format(np.quantile(all_entries, 0.99)))
            f.write("99.9%\t{:.6f}\n".format(np.quantile(all_entries, 0.999)))
            f.write("99.95%\t{:.6f}\n".format(np.quantile(all_entries, 0.9995)))
            f.write("99.99%\t{:.6f}\n".format(np.quantile(all_entries, 0.9999)))
            f.write("Max\t{:.6f}\n".format(np.max(all_entries)))


# ---------------------------------------------------------------- F2: native dilated convolution
def install_native_dilation():
    """Route every dilated Keras Conv1D through tf.nn.conv2d(dilations=[1,1,d,1]) on the [N,1,L,C] view (cuDNN's native dilated conv)
    instead of nn_ops.convolution_internal's SpaceToBatchND -> conv -> BatchToSpaceND route (TF 2.8.0 and 2.15.1 both take that route for
    dilation > 1 on GPU). Same weights, same graph otherwise. Must be called before the model is loaded. Idempotent."""
    import tensorflow as tf
    try:
        from keras.layers.convolutional import Conv as _KConv                      # keras 2.8
    except Exception:
        from keras.src.layers.convolutional.base_conv import Conv as _KConv       # keras 2.15
    if getattr(_KConv, "_fastkit_native_dilation", False):
        return _KConv.__module__
    _orig = _KConv.convolution_op
    def _native_conv_op(self, inputs, kernel):
        if self.rank == 1 and self.dilation_rate[0] > 1 and tuple(self.strides) == (1,) and self.data_format == "channels_last":
            d = int(self.dilation_rate[0])
            y = tf.nn.conv2d(tf.expand_dims(inputs, 1), tf.expand_dims(kernel, 0), strides=[1, 1, 1, 1], padding=self.padding.upper(), dilations=[1, 1, d, 1])
            return tf.squeeze(y, 1)
        return _orig(self, inputs, kernel)
    _KConv.convolution_op = _native_conv_op; _KConv._fastkit_native_dilation = True; _KConv._fastkit_orig_convolution_op = _orig
    return _KConv.__module__

def uninstall_native_dilation():
    """Restore the Keras Conv class's convolution_op after the job — nothing process-global stays changed at rest. True if it was restored."""
    try:
        from keras.layers.convolutional import Conv as _KConv
    except Exception:
        try:
            from keras.src.layers.convolutional.base_conv import Conv as _KConv
        except Exception:
            return False
    if getattr(_KConv, "_fastkit_native_dilation", False) and hasattr(_KConv, "_fastkit_orig_convolution_op"):
        _KConv.convolution_op = _KConv._fastkit_orig_convolution_op; _KConv._fastkit_native_dilation = False; return True
    return False


# ---------------------------------------------------------------- L7: pipelined job
def stock_tail_plan(N, batch, chunk, pos_in_file_order):
    """Stock-shaped tail plan: the stock's batch shapes are file-order batches of `batch`; its tail = the last N % batch rows in FILE
        order, forwarded at their true size. tail_sorted = those rows' sorted positions (forwarded through the model's original Keras predict at
        size r); every other row goes in chunks that are multiples of `batch` (full stock-shaped batches, NO padding; a full batch's per-row
        outputs do not depend on which rows fill it). Returns (tail_sorted, chunks)."""
    assert chunk % batch == 0, (chunk, batch)
    r = int(N % batch)
    tail_sorted = np.sort(np.asarray(pos_in_file_order[N - r:], dtype=np.int64)) if r else np.zeros(0, dtype=np.int64)
    keep = np.ones(N, dtype=bool); keep[tail_sorted] = False; nontail = np.nonzero(keep)[0]
    assert len(nontail) % batch == 0
    chunks = [nontail[c0:c0 + chunk] for c0 in range(0, len(nontail), chunk)]
    assert all(len(c) % batch == 0 for c in chunks)
    return tail_sorted, chunks


class _PipeQueue:
    """Child side of the spawned writer: length-prefixed pickle frames read from a RAW dup of fd 0 (no BufferedReader: the writer's own
    compression Pool forks workers, and a buffered stdin locked by this reader thread at fork time deadlocks them in _close_stdin) by a reader
    thread into a 32-deep in-memory queue. None = the sentinel; EOF before it = the main process died (raised, never silent)."""
    def __init__(self, raw):
        import pickle, struct, queue as _queue
        self.q = _queue.Queue(maxsize=32); self.raw = raw
        def _readn(n):
            buf = bytearray(n); mv = memoryview(buf); got = 0
            while got < n:
                k = raw.readinto(mv[got:])
                if not k: raise EOFError
                got += k
            return bytes(buf)
        def _reader():
            try:
                while True:
                    try: (n,) = struct.unpack("<Q", _readn(8)); obj = pickle.loads(_readn(n))
                    except EOFError: self.q.put(RuntimeError("[spawn writer] pipe closed before the sentinel")); return
                    self.q.put(obj)
                    if obj is None: return
            except BaseException as e: self.q.put(e)
        self.t = threading.Thread(target=_reader, daemon=True); self.t.start()
    def get(self):
        obj = self.q.get()
        if isinstance(obj, BaseException): raise obj
        return obj


_PRESPAWNED = {"writer": None}


def _writer_child_main_resident(fd_rep):
    """The PRE-SPAWNED writer's loop (a fresh interpreter started once per process, before the main process's TF/torch runtime exists): runs
        `_writer_proc` once per item — a config frame opens an item (the cfg tuple = `_writer_proc`'s arguments after the queue), the chunk frames and
        the None sentinel end it, then a ("done", bw_path) frame (("aborted", bw_path) after the abort sentinel) goes up the reply pipe and the loop
        waits for the next config; a None (or the abort sentinel) at the config position ends the process. Every frame is length-prefixed (the
        raw-fd reader of _PipeQueue)."""
    import io, sys, struct, pickle
    raw = io.FileIO(os.dup(0), "rb"); sys.stdin.close(); sys.stdin = open(os.devnull, "r")
    def reply(obj):
        data = pickle.dumps(obj, protocol=4); view = memoryview(struct.pack("<Q", len(data)) + data); n = 0
        while n < len(view): n += os.write(fd_rep, view[n:])
    reply(("ready", os.getpid()))
    while True:
        pq = _PipeQueue(raw); cfg = pq.get()
        if cfg is None or (isinstance(cfg, str) and cfg == ABORT_SENTINEL): break
        try: os.nice(-5)                                                  # back to the main process's priority for the item (a non-root child cannot raise it: then the nice(5) stays)
        except Exception: pass
        _writer_proc(pq, *cfg)
        reply(("done" if os.path.exists(cfg[0]) else "aborted", cfg[0]))      # the final name exists only after a successful close; an abort left none
    sys.stderr.flush(); os._exit(0)


class _PrespawnedWriter:
    """Parent side of the pre-spawned writer: one child interpreter for the whole process (its imports overlap the model load), bound to one
    item at a time by bind(cfg); put() frames chunks to its stdin (None ends the item without closing the pipe); join() waits for the item's
    ("done", bw_path) reply (bounded) instead of the process exit; stop() sends the end-of-process None. is_alive()/terminate() serve the
    kit's fail-fast release. Never a fork of the main process."""
    def __init__(self):
        import pickle, struct, subprocess, sys, select
        self._pickle, self._struct, self._select = pickle, struct, select
        env = dict(os.environ); pkg_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        env["PYTHONPATH"] = pkg_root + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
        r_rep, w_rep = os.pipe()
        # the child lowers its own CPU priority before its imports (they overlap the main process's model load, which wins the contention); the priority is
        # restored to the main process's before the first item (the writer's own compression pool inherits it)
        self.proc = subprocess.Popen([sys.executable, "-c", "import os; os.nice(5); import chrombpnet_fastkit as _k; _k._writer_child_main_resident(%d)" % w_rep], stdin=subprocess.PIPE, env=env, pass_fds=(w_rep,))
        os.close(w_rep); self.fd_rep = r_rep; self.t_spawn = time.time(); self.ready = None; self.bound = None; self.items_done = 0
    def _recv(self, timeout):
        if not self._select.select([self.fd_rep], [], [], timeout)[0]:
            raise RuntimeError(f"[prespawned writer] no reply within {timeout} s (alive={self.is_alive()}) — refused (no fallback)")
        def readn(n):
            chunks = []; got = 0
            while got < n:
                b = os.read(self.fd_rep, n - got)
                if not b: raise RuntimeError(f"[prespawned writer] child closed its reply pipe (exit {self.proc.poll()}) — refused (no fallback)")
                chunks.append(b); got += len(b)
            return b"".join(chunks)
        (n,) = self._struct.unpack("<Q", readn(8)); return self._pickle.loads(readn(n))
    def wait_ready(self, timeout=600):
        if self.ready is None:
            status, payload = self._recv(timeout)
            if status != "ready": raise RuntimeError(f"[prespawned writer] the child reported {status!r} before ready — refused")
            self.ready = {"pid": payload, "ready_after_s": round(time.time() - self.t_spawn, 3)}
        return self.ready
    def bind(self, *cfg):
        if self.bound is not None: raise RuntimeError("[prespawned writer] bind() while an item is open — refused")
        self.wait_ready(); self.bound = cfg[0]; self.put(cfg); return self, self
    def put(self, obj):
        data = self._pickle.dumps(obj, protocol=4)
        try:
            self.proc.stdin.write(self._struct.pack("<Q", len(data))); self.proc.stdin.write(data); self.proc.stdin.flush()
        except BrokenPipeError:
            raise RuntimeError(f"[prespawned writer] child died (exit {self.proc.poll()}) before consuming the stream — refused (no silent fallback)")
    def join(self, timeout=1800):
        if self.bound is None: return
        status, payload = self._recv(timeout)
        if status != "done" or payload != self.bound: raise RuntimeError(f"[prespawned writer] expected done for {self.bound}, got {status!r} {str(payload)[:80]!r} — refused")
        self.bound = None; self.items_done += 1
    def is_alive(self): return self.proc.poll() is None
    def terminate(self):
        try: self.proc.kill()
        except Exception: pass
    def stop(self, timeout=10):
        try:
            if self.bound is None: self.put(None)
            self.proc.stdin.close()
        except Exception: pass
        try: self.proc.wait(timeout)
        except Exception: self.terminate()
        try: os.close(self.fd_rep)
        except Exception: pass


def prespawn_writer(restart=False):
    """Start the process's one pre-spawned writer now (idempotent); the entry script calls this at process start, before the TF/torch runtime.
    restart=True replaces a dead one (a writer that died mid-item has already raised in the main process)."""
    if _PRESPAWNED["writer"] is None or (restart and not _PRESPAWNED["writer"].is_alive()): _PRESPAWNED["writer"] = _PrespawnedWriter()
    return _PRESPAWNED["writer"]


def stop_prespawned_writer():
    w = _PRESPAWNED["writer"]
    if w is not None: w.stop(); _PRESPAWNED["writer"] = None


def predict_pipelined(model_h5, regions_bed, genome_fa, chrom_sizes, bw_path, batch=64, chunk=4096, max_zooms=10, writer="process",
                      save_heads=None, log=print, pad_to_batch=True, native_dilation=False, prefetch=True, writer_backend="numpy", bigwig_layout="stock", stats_file=None,
                      forward="tf_function", forward_fn=None, overlap=None, warmup=True, fastdefault=None, caches=None):
    """The pred_bw job: regions -> writer (sorted) order -> chunked featurise + forward -> softmax*exp(logcounts) -> bigWig, with the
        writer in the process's pre-spawned child interpreter (writer='process') or a thread. forward='tf_function': the stock graph wrapped in
        tf.function (stock-shaped tail or pad-to-batch per CHROMBPNET_FASTKIT_TAIL); 'keras_predict' / 'keras_predict_fileorder': the stock's own
        model.predict per chunk (file order gives the stock's batch membership); forward_fn: an external forward callable (numpy (n, L, 4) int8 ->
        (logits, logcounts)) that batches itself. Returns (kit_stamp, logits, logcounts, used_mask) with the heads in regions-file order; raises
        if the kit path did not cover every unit."""
    # forward="keras_predict": the stock's own model.predict([x], batch_size=batch) per chunk with NO padding (chunk % batch == 0 enforced) so the
    # batch shapes are the stock's {batch, N % batch}; nothing of the forward is touched. forward="tf_function": the tf.function-wrapped graph
    # (stock-shaped tail by default, else pad-to-batch).
    # forward="keras_predict_fileorder": the stock's own model.predict per chunk over the units in INPUT (file) order with chunk % batch == 0, so
    # every batch has the STOCK's membership (the stock's one predict call splits the file-order units into consecutive batches; chunk boundaries
    # at multiples of batch coincide with them) — including the remainder batch = the last N % batch input units. The writer still receives
    # sorted-order chunks (emitted as the contiguous sorted prefix becomes available), so the bigWig path is unchanged.
    if forward_fn is not None: forward = "callable"                   # an external forward (the K1 route) — numpy (n, L, 4) int8 -> (logits, logcounts); chunks in writer (sorted) order, no padding here
    if forward not in ("tf_function", "keras_predict", "keras_predict_fileorder", "callable"):
        raise ValueError(f"forward must be 'tf_function', 'keras_predict' or 'keras_predict_fileorder', got {forward!r}")
    if forward in ("keras_predict", "keras_predict_fileorder"):
        if chunk % batch:
            raise ValueError(f"host-only mode: chunk ({chunk}) must be a multiple of batch ({batch}) so the batch shapes are the stock's")
        pad_to_batch = False                                  # the stock does not pad: its remainder batch is N % batch
    T0 = time.time()
    stamp = {"kit": "chrombpnet_fastkit", "version": __version__, "levers": KIT_LEVERS, "batch": batch, "chunk": chunk, "max_zooms": max_zooms, "writer": writer,
             "pad_to": batch if pad_to_batch else None, "native_dilation": bool(native_dilation), "prefetch": bool(prefetch), "writer_backend": writer_backend, "bigwig_layout": bigwig_layout, "forward": forward}
    import h5py
    with h5py.File(model_h5, "r") as h:
        cfg = json.loads(h.attrs["model_config"])
    inputlen = int(cfg["config"]["layers"][0]["config"]["batch_input_shape"][1]); outputlen = 1000
    gs = read_chrom_sizes(chrom_sizes); chr_to_idx = {x[0]: i for i, x in enumerate(gs)}
    df = pd.read_csv(regions_bed, sep="\t", names=NARROWPEAK_SCHEMA); N0 = len(df)
    center = (df["start"].values + df["summit"].values).astype(np.int64)
    key_chr = np.array([chr_to_idx.get(c, 10 ** 6) for c in df["chr"].values])
    order = np.lexsort((center, key_chr)); dfs = df.iloc[order].reset_index(drop=True); cen_s = center[order]
    g = Genome(genome_fa); starts = cen_s - inputlen // 2; chroms = dfs["chr"].values
    used_sorted = g.windows_ok(chroms, starts, inputlen)
    dropped = [(str(dfs["chr"].values[i]), int(dfs["start"].values[i]), int(dfs["summit"].values[i])) for i in np.where(~used_sorted)[0]]
    order = order[used_sorted]; cen_s = cen_s[used_sorted]; chroms = chroms[used_sorted]; starts = starts[used_sorted]; N = len(order)
    stamp.update({"units_in": int(N0), "units_dropped_named": int(N0 - N), "dropped_units": dropped[:1000]})
    regions = [(c, int(m - outputlen // 2), int(m + outputlen // 2), int(m)) for c, m in zip(chroms, cen_s)]
    statfile = bw_path + ".writer_stats.json"
    if writer == "process":
        # the bigWig writer = the process's ONE pre-spawned writer interpreter (prespawn_writer(): started at process start by the entry script, before
        # the TF/torch runtime; started here if it was not), bound to this item — never a fork of this process after the runtimes exist (fork() in a
        # threaded parent copies only the forking thread: a lock another thread holds at that instant stays locked in the child), never a spawn per item.
        w = _PRESPAWNED["writer"]
        if w is None or not w.is_alive(): w = prespawn_writer(restart=True)
        stamp["writer_start_method"] = "spawn"; stamp["writer"] = "prespawned"; t_wstart = time.time(); stamp["writer_start_epoch"] = t_wstart
        q, wp = w.bind(bw_path, gs, regions, max_zooms, N, statfile, writer_backend, bigwig_layout, stats_file)
        stamp["writer_ready"] = w.wait_ready(); stamp["writer_items_before"] = w.items_done
        _LIVE_WRITERS.append((q, wp)); _register_partial(bw_path); _install_fail_fast()
    else:
        import queue as _queue
        q = _queue.Queue(maxsize=64)
        wp = threading.Thread(target=_writer_proc, args=(q, bw_path, gs, regions, max_zooms, N, statfile, writer_backend, bigwig_layout), daemon=True); wp.start()
        _LIVE_WRITERS.append((q, wp)); _register_partial(bw_path); _install_fail_fast()   # a daemon thread cannot hang the exit; registered for the uniform release
    warm_stamp = None
    if forward_fn is None:
        import tensorflow as tf
        import chrombpnet.evaluation.make_bigwigs.predict_to_bigwig as p2b   # the stock's loader (imports TensorFlow) only for the TF forwards; the callable route imports no TensorFlow here
        if native_dilation:
            stamp["native_dilation_patched_module"] = install_native_dilation()
        t0 = time.time(); model = p2b.load_model_wrapper(model_h5); t_load = time.time() - t0
        assert inputlen == int(model.input_shape[1]) and outputlen == int(model.output_shape[0][1])

        @tf.function(input_signature=[tf.TensorSpec([None, inputlen, 4], tf.int8)])
        def fwd(x):
            return model(tf.cast(x, tf.float32), training=False)
    else:
        t_load = 0.0; pad_to_batch = False                     # the callable batches itself (the K1 route pads to its own batch and drops the pad rows)

    logits_all = np.empty((N, outputlen), dtype=np.float32); logcts_all = np.empty((N, 1), dtype=np.float32)
    t_feat = t_fwd = t_post = 0.0; forward_calls = 0; units_forwarded = 0; padded_rows = 0
    tail_form = os.environ.get("CHROMBPNET_FASTKIT_TAIL", "stock") if (forward == "tf_function" and forward_fn is None) else "n/a"
    computed = np.zeros(N, dtype=bool); emitted = 0
    def _feat(idx):
        return g.one_hot(chroms[idx], starts[idx], inputlen)
    if forward == "keras_predict_fileorder":
        pos_in_file_order = np.argsort(order, kind="stable")           # sorted positions listed in increasing input-file order (used units)
        chunks = [pos_in_file_order[c0:c0 + chunk] for c0 in range(0, N, chunk)]
        stamp["fastdefault"] = fastdefault; stamp["caches"] = caches
        if fastdefault is not None:   # the kit's status line on THIS route too (the tf_function and padded routes print it below; the K1 route prints it at apply): every job says its class, cc and route once
            from chrombpnet_fastkit import fastdefault as _fdm; log(_fdm.r8_line(fastdefault, caches, stamp.get("warmup")))
    elif tail_form == "stock":
        # STOCK-SHAPED TAIL: the stock's file-order tail rows (N % batch) through the model's ORIGINAL Keras predict at their true size, FIRST;
        # every other row in full stock-shaped batches (chunks are multiples of `batch`; no padding). The padded form is the tail table's other value.
        pos_in_file_order = np.argsort(order, kind="stable")
        tail_sorted, chunks = stock_tail_plan(N, batch, chunk, pos_in_file_order); pad_to_batch = False
        # WARM-UP AT LOAD, SIZED BY THE JOB: the regions file names the job's whole shape set - a full batch (through the tf.function forward) only if
        # the plan has one, the tail (through the stock tail path model.predict) only if it exists; exactly the traces the stock pays at its first batch
        # and at its end, moved to load; one zero-input forward per shape beyond that.
        if warmup:
            _tw = time.time(); _shapes = ([int(batch)] if len(chunks) else []) + ([int(len(tail_sorted))] if len(tail_sorted) else []); _per = {}
            for _s in _shapes:
                _t1 = time.time()
                if _s == int(batch) and len(chunks): fwd(tf.zeros([_s, inputlen, 4], tf.int8))
                else: model.predict([np.zeros((_s, inputlen, 4), dtype=np.int8)], batch_size=_s, verbose=0)
                _per[str(_s)] = round(time.time() - _t1, 3)
            warm_stamp = {"shapes": _shapes, "inputlen": int(inputlen), "s": round(time.time() - _tw, 3), "per_shape_s": _per, "deferred": "none (the job's shape set is known from the regions file)"}; stamp["warmup"] = warm_stamp
        stamp["fastdefault"] = fastdefault; stamp["caches"] = caches
        if fastdefault is not None:
            from chrombpnet_fastkit import fastdefault as _fdm; log(_fdm.r8_line(fastdefault, caches, warm_stamp))
        t1 = time.time()
        if len(tail_sorted):
            Xt = _feat(tail_sorted); t_feat += time.time() - t1; t1 = time.time()
            lgt, lct = model.predict([Xt], batch_size=int(len(tail_sorted)), verbose=0)
            logits_all[tail_sorted] = np.asarray(lgt, dtype=np.float32); logcts_all[tail_sorted] = np.asarray(lct, dtype=np.float32); computed[tail_sorted] = True
            if overlap is not None: overlap.add(order[tail_sorted], logits_all[tail_sorted])          # the metrics stage consumes rows as they land (file rows = order[sorted])
            forward_calls += 1; units_forwarded += int(len(tail_sorted))
        t_tail = time.time() - t1
        stamp["tail"] = {"tail_form": "stock", "tail_rows": int(len(tail_sorted)), "tail_file_positions": [int(N - len(tail_sorted)), int(N)], "tail_predict": f"model.predict([X], batch_size={int(len(tail_sorted))}) (the model's original Keras predict at the true size)" if len(tail_sorted) else "none (N % batch == 0)",
                         "full_batches": {"batch": int(batch), "chunks": len(chunks), "rows": int(N - len(tail_sorted)), "path": "tf.function (full stock-shaped batches, no padding)"}, "t_tail_s": round(t_tail, 3)}
    else:
        chunks = [np.arange(c0, min(c0 + chunk, N)) for c0 in range(0, N, chunk)]
        stamp["tail"] = {"tail_form": tail_form, "tail_form_source": os.environ.get("CHROMBPNET_FASTKIT_TAIL_SOURCE", "CHROMBPNET_FASTKIT_TAIL (the entry script sets both from the tail table)"), "pad_to_batch": bool(pad_to_batch)}
        stamp["fastdefault"] = fastdefault; stamp["caches"] = caches
        if fastdefault is not None:
            try:
                from chrombpnet_fastkit import fastdefault as _fdm
                _pw = {"shapes": [int(batch)], "inputlen": int(locals().get("inputlen") or 0) or None, "s": 0.0, "per_shape_s": {}, "deferred": "the padded form's one bs-%d trace is paid in the first batch (no separate warm-up)" % int(batch)}
                stamp["warmup"] = _pw; log(_fdm.r8_line(fastdefault, caches, _pw))
            except Exception as _e:   # the status line is a print: it must never kill the job
                log(f"[chrombpnet_fastkit] R8 line could not be composed: {_e!r}")
    try:
        from concurrent.futures import ThreadPoolExecutor
        pool = ThreadPoolExecutor(max_workers=1) if prefetch else None
        nxt = pool.submit(_feat, chunks[0]) if (prefetch and chunks) else None
        for ci, idx in enumerate(chunks):
            t1 = time.time()
            if prefetch:
                X = nxt.result(); nxt = pool.submit(_feat, chunks[ci + 1]) if ci + 1 < len(chunks) else None
            else:
                X = _feat(idx)
            t_feat += time.time() - t1
            t1 = time.time(); outs = []
            if forward in ("keras_predict", "keras_predict_fileorder"):   # the stock's OWN call (predict_to_bigwig.main: model.predict([seqs], batch_size=bs)) for EVERY
                # chunk — file order + chunk % batch == 0 give the stock's batch membership, and the remainder batch (N % batch, down to N=1) runs through Keras'
                # predict path exactly as the stock's (a tf.function call can differ from model.predict on a 1-row final batch)
                lg, lc = model.predict([X], batch_size=batch, verbose=0)
                lg = np.asarray(lg, dtype=np.float32); lc = np.asarray(lc, dtype=np.float32); n_calls = -(-len(idx) // batch)
            elif forward_fn is not None:
                lg, lc = forward_fn(X); lg = np.asarray(lg, dtype=np.float32); lc = np.asarray(lc, dtype=np.float32).reshape(len(idx), -1); n_calls = -(-len(idx) // batch)
            else:
                for b in range(0, len(idx), batch):
                    xb = X[b:b + batch]
                    if pad_to_batch and xb.shape[0] < batch:      # L9: keep every cuDNN autotune key at the full batch shape
                        padded_rows += batch - xb.shape[0]
                        xb = np.concatenate([xb, np.zeros((batch - xb.shape[0],) + xb.shape[1:], dtype=xb.dtype)])
                    outs.append(fwd(xb))
                lg = np.concatenate([o[0].numpy() for o in outs])[:len(idx)]; lc = np.concatenate([o[1].numpy() for o in outs])[:len(idx)]; n_calls = len(outs)
            t_fwd += time.time() - t1
            forward_calls += n_calls; units_forwarded += len(idx)
            logits_all[idx] = lg; logcts_all[idx] = lc
            if overlap is not None: overlap.add(order[idx], lg)
            if forward == "keras_predict_fileorder" or tail_form == "stock":
                computed[idx] = True                                          # emit the contiguous SORTED prefix that is fully computed
                p1 = emitted
                while p1 < N and computed[p1]: p1 += 1
                for e0 in range(emitted, p1, chunk):
                    e1 = min(e0 + chunk, p1); t1 = time.time(); prof = np.ascontiguousarray(softmax(logits_all[e0:e1]) * np.expand_dims(np.exp(logcts_all[e0:e1])[:, 0], axis=1), dtype=np.float32); t_post += time.time() - t1
                    q.put((int(e0), prof))
                emitted = p1
            else:
                t1 = time.time(); prof = np.ascontiguousarray(softmax(lg) * np.expand_dims(np.exp(lc)[:, 0], axis=1), dtype=np.float32); t_post += time.time() - t1
                q.put((int(idx[0]), prof))
        if (forward == "keras_predict_fileorder" or tail_form == "stock") and emitted < N:
            # with N < batch the plan has no full chunk: the loop body never runs and the tail rows computed before it are not yet emitted.
            # Emit the fully computed sorted prefix now; a no-op when the loop already emitted through N.
            p1 = emitted
            while p1 < N and computed[p1]: p1 += 1
            for e0 in range(emitted, p1, chunk):
                e1 = min(e0 + chunk, p1); t1 = time.time(); prof = np.ascontiguousarray(softmax(logits_all[e0:e1]) * np.expand_dims(np.exp(logcts_all[e0:e1])[:, 0], axis=1), dtype=np.float32); t_post += time.time() - t1
                q.put((int(e0), prof))
            emitted = p1
        if (forward == "keras_predict_fileorder" or tail_form == "stock") and emitted != N:
            raise RuntimeError(f"file-order forward: emitted {emitted} of {N} sorted units to the writer")
        q.put(None); wp.join()
        try: _LIVE_WRITERS.remove((q, wp))   # the normal end deregisters the writer
        except ValueError: pass
        try: _PARTIAL_OUTPUTS.remove(bw_path + ".partial")
        except ValueError: pass
    except BaseException as _exc:
        # never hang on failure — release the writer (a None sentinel; then terminate a process writer), drop the partial output and re-raise
        try: q.put(None)
        except Exception: pass
        try:
            if hasattr(wp, 'terminate'): wp.join(5); (wp.terminate() if getattr(wp, 'is_alive', lambda: False)() else None)
        except Exception: pass
        try: _LIVE_WRITERS.remove((q, wp))
        except ValueError: pass
        try:
            if os.path.exists(bw_path + ".partial"): os.unlink(bw_path + ".partial")   # a failed job leaves no partial file
            _PARTIAL_OUTPUTS.remove(bw_path + ".partial")
        except Exception: pass
        raise
    if pool: pool.shutdown()
    ws = json.load(open(statfile))
    try: os.unlink(statfile)                                                           # the writer's handshake file: folded into the stamp below, never left beside the outputs
    except OSError: pass
    stamp.update({"units_forwarded": int(units_forwarded), "forward_calls": int(forward_calls), "writer_regions": int(ws["regions"]), "padded_rows": int(padded_rows),
                  "timings_s": {"load_model": round(t_load, 3), "featurise": round(t_feat, 3), "forward": round(t_fwd, 3), "postprocess": round(t_post, 3),
                                "writer_cpu": round(ws["t_write"], 3), "writer_zoom_cpu": round(ws.get("t_zoom", 0.0), 3), "bigwig_close": round(ws["t_close"], 3), "writer_start_to_ready_s": (round(ws["t_ready_epoch"] - stamp["writer_start_epoch"], 3) if ws.get("t_ready_epoch") and stamp.get("writer_start_epoch") else None), "writer_witness": {"at_start": ws.get("witness_at_start"), "at_close": ws.get("witness_at_close")}, "writer_compress_workers": ws.get("compress_workers"), "writer_vec": ws.get("writer_vec"), "job_wall": round(time.time() - T0, 3)}})
    if not (units_forwarded == N and ws["regions"] == N):
        raise RuntimeError(f"kit path did not cover every unit: {stamp}")
    inv = np.full(N0, -1, dtype=np.int64); inv[order] = np.arange(N); file_pos = inv[inv >= 0]
    lg_file = logits_all[file_pos]; lc_file = logcts_all[file_pos]; used_mask = inv >= 0
    stamp.update({"md5_logits_fileorder": md5_array(lg_file), "md5_logcts_fileorder": md5_array(lc_file)})
    if save_heads:
        np.save(save_heads + "_logits.npy", lg_file); np.save(save_heads + "_logcts.npy", lc_file)
    log(f"[chrombpnet_fastkit] job {stamp['timings_s']['job_wall']:.1f}s units {N}/{N0} md5 {stamp['md5_logits_fileorder'][:8]}/{stamp['md5_logcts_fileorder'][:8]}")
    stamp["native_dilation_restored_after_job"] = uninstall_native_dilation() if native_dilation else None    # no residue at rest
    return stamp, lg_file, lc_file, used_mask


def write_outputs_from_heads(regions_bed, genome_fa, chrom_sizes, bw_path, logits, logcts, max_zooms=10, writer_backend="numpy", bigwig_layout="stock", stats_file=None, save_heads=None):
    """Stage 2 from precomputed heads (the K1 route): the stock's preds.bed, the stock's profile expression softmax(logits)*exp(logcts) and the
        bigWig with the -os stats — writer_backend='stock' calls bigwig_helper.write_bigwig itself, anything else goes through write_bigwig_with_stats
        (numpy writer, stock layout). Returns a timing/md5 stamp."""
    import time, pandas as pd
    import chrombpnet.evaluation.make_bigwigs.bigwig_helper as bigwig_helper
    t0 = time.time(); logits = np.asarray(logits, dtype=np.float32); logcts = np.asarray(logcts, dtype=np.float32)
    if logcts.ndim == 1: logcts = logcts[:, None]
    regions_df = pd.read_csv(regions_bed, sep="\t", names=NARROWPEAK_SCHEMA); N0 = len(regions_df); used = np.ones(N0, dtype=bool)
    assert logits.shape[0] == N0, (logits.shape, N0)
    outputlen = int(logits.shape[1]); prefix = bw_path[:-3] if bw_path.endswith(".bw") else bw_path
    regions_df[used].to_csv(prefix + "_preds.bed", sep="\t", header=False, index=False)
    gs = bigwig_helper.read_chrom_sizes(chrom_sizes); regions = bigwig_helper.get_regions(regions_bed, outputlen, used)
    prof = softmax(logits) * (np.expand_dims(np.exp(logcts)[:, 0], axis=1))
    if writer_backend == "stock": bigwig_helper.write_bigwig(prof, regions, gs, bw_path, outstats_file=stats_file, debug_chr=None, use_tqdm=False)
    else:
        assert bigwig_layout == "stock", "the stock layout is the parity form"
        write_bigwig_with_stats(prof, regions, gs, bw_path, max_zooms=max_zooms, stats_file=stats_file)
    if save_heads: np.save(save_heads + "_logits.npy", logits); np.save(save_heads + "_logcts.npy", logcts)
    return {"stage2_from_heads_s": round(time.time() - t0, 3), "writer_backend": writer_backend, "bigwig_layout": bigwig_layout, "units": int(N0), "md5_logits": md5_array(logits), "md5_logcts": md5_array(logcts), "outputlen": outputlen}
