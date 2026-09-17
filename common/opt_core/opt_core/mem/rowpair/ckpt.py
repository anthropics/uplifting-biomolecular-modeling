"""Per-cycle trunk checkpoints of a sharded run: each rank writes its pair-row shard, rank 0 writes the replicated tensors + the RNG state, under a
manifest written LAST (sha256 per file) — a tag is complete iff its manifest exists; resume restores every rank's shard and the RNG stream so a
resumed run continues the same arithmetic. Mechanism only — WHICH tensors form a cycle's state is the engine adapter's list. No collectives in
the writer (every rank drops a json fragment next to its shard; rank 0 polls for the fragments before writing the manifest), so the whole write
may run from a background thread.

Layout::

    <root>/<tag>/{rank<r>.pt, rank<r>.json, replicated.pt, manifest.json}      tag = cycle_<kkk> | trunk_final | any caller name
    manifest.json = {tag, P, step, meta, files: {name: {sha256, bytes, features}}, utc, format: FORMAT}
                    meta = {kind, cycle, N, P, parts, align, B, policy, query, precision: {word, z_dtype, s_dtype, ...}, features_rank0}
                    files["rank<r>.pt"].features = rank r's feature digest (feature_hash of the adapter's input object) or null

API:
    save_tag(root, tag, *, rank, P, shards, replicated=None, meta=None, step=None, rng=True)   every rank: ``rank<r>.pt`` (+ fragment); rank 0:
                                                      ``replicated.pt`` (tensors + RNG + meta) and, once all fragments exist, ``manifest.json``
    load(root, tag, *, rank, P=None, map_location="cpu", verify=True) -> {shards, replicated, rng, meta, manifest}   (sha256 re-checked, always,
                                                      BEFORE ``torch.load`` unpickles a byte; a tag whose directory or files are writable by group or
                                                      other, or — unless this process is uid 0 — owned by neither this uid nor root, is REFUSED by name
                                                      on stderr and is an absent tag: ``load`` raises FileNotFoundError, ``find_resume`` passes it over.
                                                      The sha256 catches truncation and corruption, not a writer of the directory — hence that rule)
    list_complete(root) -> [(tag, manifest)] oldest first / latest(root, prefix) -> tag | None
    save_cycle(dir, cycle, rank, shards, replicated, rng_state) -> manifest path      the ``cycle_<kkk>`` tag of a cycle (P from the comm)
    load_cycle(dir, cycle, rank) -> (shards, replicated, rng_state)
    latest_complete_cycle(dir, world) -> int | None   (a cycle counts only when its manifest names every rank's file and P == world)
    rng_state() / set_rng_state(st) / sha256_file(path) / feature_hash(obj)
    TrunkCheckpointer(query_id, seed, N, root=, every=, resume=, keep=, log=)   the per-query policy object a sharded trunk loop drives:
                                                      ``maybe_save_cycle(cycle, layout=, s=, z_loc=)``, ``save_trunk_final(layout=, s_input=, s=,
                                                      z_loc=, cycle=)``, ``find_resume(layout)``, ``try_resume(layout=, device=)``, ``finalize()``,
                                                      ``phase(name)``. A tag is a CANDIDATE iff format, P, N and the row partition (``layout.parts``)
                                                      match (a complete tag passed over is said in the log: ``[ckpt] <tag> passed over: <why>``); the
                                                      chosen candidate must then carry THIS rank's feature digest (``features=``; else
                                                      ``resume_refused:features_differ`` — ``ROWPAIR_RESUME_ALLOW_FEATS=1`` allows it, named) and this
                                                      line's precision word (``precision=``; else ``resume_refused:precision_differs``), and the loaded
                                                      shard has ``dtype=`` when given (else ``resume_refused:dtype_differs``) — refusals by name, never
                                                      a silent fresh start. Log words: ``[ckpt] wrote <tag> ranks <P> bytes <B> in <s>s`` /
                                                      ``[ckpt] resume from <tag> (cycle <k>) digest ok``; census words ``ckpt_wrote`` / ``ckpt_resume``.
                                                      A failed write is NON-FATAL by default (costs resumability, never the run;
                                                      ``ROWPAIR_CKPT_FATAL=1`` raises).

Environment (policy defaults; explicit constructor arguments win): ``ROWPAIR_CKPT_DIR`` (unset = disabled), ``ROWPAIR_CKPT_EVERY`` (cycles between
tags; default 1 when N >= 8000 else 0 = ``trunk_final`` only), ``ROWPAIR_CKPT_KEEP`` (newest complete cycle tags kept, default 1),
``ROWPAIR_CKPT_ASYNC`` (1: device->pinned-host copy synchronously, file write + hash in a thread), ``ROWPAIR_CKPT_FATAL``, ``ROWPAIR_CKPT_READ_DIR``
(resume SOURCE directory; default = the write root), ``ROWPAIR_RESUME`` (1 = default: trunk_final if complete+compatible else the newest compatible
cycle; 0 = never; ``trunk_final`` = REQUIRE it), ``ROWPAIR_RESUME_TAG`` (an explicit tag), ``ROWPAIR_RESUME_RNG`` (1 = restore the RNG state, default),
``ROWPAIR_RESUME_ALLOW_FEATS`` (1 = a resume whose feature digest differs is allowed and named; default 0 = refused by name).
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import sys
import threading
import time
from typing import Any, Dict, List, Optional, Tuple

from . import RowpairRefused
from ._torch import torch
from .dist import comm, env_int, world

__all__ = ["FORMAT", "save_tag", "load", "list_complete", "latest", "save_cycle", "load_cycle", "latest_complete_cycle", "rng_state", "set_rng_state",
           "sha256_file", "feature_hash", "cycle_tag", "TrunkCheckpointer", "precision_tag", "ENV_RESUME_ALLOW_FEATS", "REFUSED_FEATURES",
           "REFUSED_PRECISION", "REFUSED_DTYPE", "HASH_CHUNK_BYTES"]

FORMAT = "rowpair.ckpt/v2"
TRUNK_FINAL = "trunk_final"
ENV_RESUME_ALLOW_FEATS = "ROWPAIR_RESUME_ALLOW_FEATS"     # 1: a resume whose per-rank feature digest differs from this run's is ALLOWED (named), not refused
REFUSED_FEATURES = "resume_refused:features_differ"      # the refusal word of a digest mismatch (RowpairRefused.reason starts with it)
REFUSED_PRECISION = "resume_refused:precision_differs"   # the refusal word of a precision-tag mismatch
REFUSED_DTYPE = "resume_refused:dtype_differs"           # the loaded shard's dtype is not the dtype this line's trunk produces


def cycle_tag(cycle: int) -> str:
    return f"cycle_{int(cycle):03d}"


def _record(**facts) -> None:
    """Checkpoint facts of this process into the schedule census (``ckpt_wrote=<tag>``, ``ckpt_resume=<tag>|none|resume_refused:<why>:<tag>``)."""
    from .evidence import record_schedule
    record_schedule(**facts)


# ----------------------------------------------------------------------------------------------------------------- bytes on disk
def sha256_file(path: str, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(block)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _to_cpu(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().to("cpu").contiguous()
    if isinstance(obj, dict):
        return {k: _to_cpu(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        seq = [_to_cpu(v) for v in obj]
        return seq if isinstance(obj, list) else tuple(seq)
    return obj


HASH_CHUNK_BYTES = 64 << 20                                   # feature_hash feeds tensor bytes to sha256 in slices of this size (no whole-tensor copy)


def _hash_tensor_bytes(h, t) -> None:
    """Feed ``t``'s bytes (CPU, contiguous) to ``h`` through a zero-copy memoryview in :data:`HASH_CHUNK_BYTES` slices — no ``tobytes()`` copy, so
    hashing an ``[N, N]`` feature costs no second host copy."""
    if t.numel() == 0:
        return                                                 # no bytes (a zero-row placeholder): the header already carries dtype + shape
    if t.dtype == torch.bool:
        t = t.view(torch.uint8)                                # same itemsize: a view, not a copy
    elif t.dtype == torch.bfloat16:
        t = t.view(torch.int16)                                # numpy has no bfloat16: hash the same bytes as int16
    mv = memoryview(t.numpy().reshape(-1)).cast("B")           # the 1-D view of a contiguous array (no copy); casts for any shape
    n = len(mv)
    step = max(1, int(HASH_CHUNK_BYTES))
    for i in range(0, n, step):
        h.update(mv[i: i + step])


def hash_tensor_leaf(h, path: str, o) -> None:
    """Feed one tensor leaf to ``h``: the header ``path|dtype|shape|`` then the bytes (:func:`_hash_tensor_bytes`; a CUDA or non-contiguous
    tensor is copied to a contiguous host tensor once). The one tensor-leaf rule of both feature walkers (this module's :func:`feature_hash`,
    :func:`rankdata.digest_features`)."""
    t = o.detach()
    if t.is_cuda or not t.is_contiguous():
        t = t.to("cpu").contiguous()
    h.update(f"{path}|{t.dtype}|{tuple(t.shape)}|".encode())
    _hash_tensor_bytes(h, t)


def feature_hash(obj) -> str:
    """sha256 over tensor bytes (dtype + shape + data) and ``repr()`` of other leaves, keys sorted — an equality for an input / feature dict
    WITHIN ONE PROCESS FORM (the checkpointer's resume witness). Cross-RANK input identity is :func:`rankdata.digest_features` (non-tensor
    leaves canonical there, never ``repr()``).
    Tensor bytes are hashed through a zero-copy memoryview in chunks (:func:`_hash_tensor_bytes`); a CUDA tensor is copied to the host once
    (its bytes must reach the hash), a CPU tensor is hashed in place."""
    h = hashlib.sha256()

    def rec(o, path):
        if isinstance(o, torch.Tensor):
            hash_tensor_leaf(h, path, o)
        elif isinstance(o, dict):
            for k in sorted(o.keys(), key=str):
                rec(o[k], f"{path}/{k}")
        elif isinstance(o, (list, tuple)):
            for i, v in enumerate(o):
                rec(v, f"{path}/{i}")
        else:
            h.update(f"{path}|{o!r}".encode())

    rec(obj, "")
    return h.hexdigest()


def rng_state() -> Dict[str, Any]:
    """The python / numpy / torch CPU / torch CUDA RNG states (what a resumed run must restore to continue the same draws)."""
    st = {"python": random.getstate(), "torch_cpu": torch.get_rng_state()}
    try:
        import numpy as np
        st["numpy"] = np.random.get_state()
    except Exception:  # noqa: BLE001 — numpy absent: nothing to capture
        pass
    if torch.cuda.is_available():
        st["torch_cuda"] = torch.cuda.get_rng_state()
    return st


def set_rng_state(st: Dict[str, Any]) -> None:
    if "python" in st:
        random.setstate(st["python"])
    if "numpy" in st:
        try:
            import numpy as np
            np.random.set_state(st["numpy"])
        except Exception:  # noqa: BLE001
            pass
    if "torch_cpu" in st:
        torch.set_rng_state(st["torch_cpu"].cpu() if isinstance(st["torch_cpu"], torch.Tensor) else st["torch_cpu"])
    if "torch_cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state(st["torch_cuda"].cpu() if isinstance(st["torch_cuda"], torch.Tensor) else st["torch_cuda"])


class _HashingWriter(object):
    """A file object that sha256-hashes while ``torch.save`` streams into it (a shard is never re-read to hash it)."""

    def __init__(self, path):
        self.f = os.fdopen(os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), "wb")   # never group/other-writable, whatever the umask (_refusal)
        self.h = hashlib.sha256()
        self.n = 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b)

    def flush(self):
        self.f.flush()

    def close(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()


def _save_hashed(obj, path: str) -> Tuple[str, int]:
    w = _HashingWriter(path + ".tmp")
    try:
        torch.save(obj, w)
    finally:
        w.close()
    os.replace(path + ".tmp", path)
    return w.h.hexdigest(), w.n


def _write_json_atomic(obj, path: str) -> None:
    with os.fdopen(os.open(path + ".tmp", os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644), "w") as f:
        json.dump(obj, f, indent=1, default=str)
        f.flush()
        os.fsync(f.fileno())
    os.replace(path + ".tmp", path)


# ----------------------------------------------------------------------------------------------------------------- save / load
def precision_tag(word: Optional[str] = None, **tensors) -> Dict[str, Any]:
    """The precision tag of a checkpoint: ``{"word": <the line's precision word, or None>, "<name>_dtype": str(dtype) per given tensor}`` —
    what a resumed run must reproduce to continue the same arithmetic (a bf16 shard resumed into an fp32 line is refused by name)."""
    tag: Dict[str, Any] = {"word": (str(word) if word is not None else None)}
    for k, v in tensors.items():
        if isinstance(v, torch.Tensor):
            tag[f"{k}_dtype"] = str(v.dtype).replace("torch.", "")
    return tag


def save_tag(root: str, tag: str, *, rank: int, P: int, shards: Optional[dict], replicated: Optional[dict] = None, meta: Optional[dict] = None,
             step=None, rng: bool = True, rng_state_override: Optional[dict] = None, poll_s: float = 0.5, timeout_s: float = 3600.0, log=None,
             features: Optional[str] = None) -> Optional[dict]:
    """Every rank: ``rank<r>.pt`` (+ the ``rank<r>.json`` fragment ``{sha256, bytes, features}`` — ``features`` = THIS rank's feature digest,
    :func:`feature_hash`, or None); rank 0: ``replicated.pt`` (tensors + RNG + meta) and, once all P fragments exist, ``manifest.json`` (written
    LAST; ``meta`` carries the precision tag when the caller gives one). Files go to ``.tmp`` and are ``os.replace``'d (a crash never leaves a
    torn file under its final name). No collectives. Returns the manifest on rank 0, None elsewhere. Log line (rank 0):
    ``[ckpt] wrote <tag> ranks <P> bytes <B> in <s>s``."""
    d = os.path.join(root, tag)
    os.makedirs(d, mode=0o755, exist_ok=True)
    t0 = time.time()
    mine = os.path.join(d, f"rank{rank}.pt")
    sha, nb = _save_hashed(_to_cpu(shards or {}), mine)
    _write_json_atomic({f"rank{rank}.pt": {"sha256": sha, "bytes": nb, "features": (str(features) if features is not None else None)}},
                       os.path.join(d, f"rank{rank}.json"))
    if int(rank) != 0:
        return None
    rp = os.path.join(d, "replicated.pt")
    rst = rng_state_override if rng_state_override is not None else (rng_state() if rng else None)
    rep = {"tensors": _to_cpu(replicated or {}), "rng": rst, "meta": meta or {}}
    rsha, rnb = _save_hashed(rep, rp)
    files = {"replicated.pt": {"sha256": rsha, "bytes": rnb}}
    deadline = time.time() + float(timeout_s)
    missing = list(range(int(P)))
    while True:
        still = []
        for r in missing:
            fp = os.path.join(d, f"rank{r}.json")
            if os.path.exists(fp):
                try:
                    with open(fp) as f:
                        files.update(json.load(f))
                except Exception:  # noqa: BLE001 — a fragment mid-rename: poll again
                    still.append(r)
            else:
                still.append(r)
        missing = still
        if not missing:
            break
        if time.time() > deadline:
            raise TimeoutError(f"checkpoint {tag}: fragments from ranks {missing} did not appear within {timeout_s}s")
        time.sleep(poll_s)
    manifest = {"tag": tag, "P": int(P), "step": step, "meta": meta or {}, "files": files,
                "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "format": FORMAT}
    _write_json_atomic(manifest, os.path.join(d, "manifest.json"))
    if log:
        tot = sum(v["bytes"] for v in files.values())
        log(f"[ckpt] wrote {tag} ranks {P} bytes {tot} in {time.time() - t0:.1f}s (files={len(files)}, {tot / 2 ** 30:.2f} GiB)")
    return manifest


def _refusal(*paths: str) -> Optional[str]:
    """Why the checkpoint files ``paths`` (one tag directory) must not be loaded (reason + fix), or None: neither they nor their directory
    may be writable by group or other, and all must belong to this uid or to root — uid 0 accepts any owner (a container's root reading a
    bind-mounted host directory). Said once per reason on stderr."""
    uid = os.geteuid()
    for p in (os.path.dirname(os.path.abspath(paths[0])),) + paths:
        st = os.stat(p)
        why = None
        if st.st_mode & 0o022:
            why = f"{p} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {p}"
        elif uid != 0 and st.st_uid not in (uid, 0):
            why = f"{p} belongs to uid {st.st_uid}, not to this process (uid {uid}) or root; fix: name a ROWPAIR_CKPT_DIR of your own, or chown {p}"
        if why is not None:
            if why not in _REFUSALS_SAID:
                _REFUSALS_SAID.add(why)
                print(f"[opt_core.rowpair.ckpt] REFUSED checkpoint {os.path.dirname(os.path.abspath(paths[0]))}: {why} — not loaded, treated as absent",
                      file=sys.stderr, flush=True)
            return why
    return None


_REFUSALS_SAID: set = set()


def list_complete(root: str) -> List[Tuple[str, dict]]:
    """``[(tag, manifest)]`` of every tag under ``root`` whose manifest exists (= complete), oldest first (by step, then mtime)."""
    out = []
    if not root or not os.path.isdir(root):
        return out
    for tag in os.listdir(root):
        mp = os.path.join(root, tag, "manifest.json")
        if os.path.exists(mp):
            try:
                with open(mp) as f:
                    out.append((tag, json.load(f)))
            except Exception:  # noqa: BLE001 — a manifest mid-rename is not complete yet
                continue
    out.sort(key=lambda tm: ((tm[1].get("step") if isinstance(tm[1].get("step"), (int, float)) else -1),
                            os.path.getmtime(os.path.join(root, tm[0], "manifest.json"))))
    return out


def latest(root: str, prefix: str = "") -> Optional[str]:
    c = [t for t, _ in list_complete(root) if t.startswith(prefix)]
    return c[-1] if c else None


def load(root: str, tag: str, *, rank: int, P: Optional[int] = None, map_location="cpu", verify: bool = True) -> dict:
    """Load this rank's shard + the replicated dict of a COMPLETE tag. ``verify`` re-hashes the two files this rank reads against the manifest;
    ``P`` (if given) must equal the manifest's P (re-sharding across a different P is refused by name)."""
    d = os.path.join(root, tag)
    mp = os.path.join(d, "manifest.json")
    if not os.path.exists(mp):
        raise FileNotFoundError(f"checkpoint {tag} incomplete (no manifest.json) in {root}")
    with open(mp) as f:
        manifest = json.load(f)
    if manifest.get("format") != FORMAT:
        raise RowpairRefused(f"checkpoint {tag}: format {manifest.get('format')!r} is not {FORMAT!r}")
    if P is not None and int(manifest.get("P", -1)) != int(P):
        raise RowpairRefused(f"checkpoint {tag} was written with P={manifest.get('P')}, cannot load with P={P}")
    mine = os.path.join(d, f"rank{rank}.pt")
    rp = os.path.join(d, "replicated.pt")
    why = _refusal(mp, mine, rp)
    if why is not None:
        raise FileNotFoundError(f"checkpoint {tag} in {root} refused, treated as absent: {why}")
    for p in (mine, rp):                                             # always, before torch.load unpickles a byte (``verify`` stays for call compatibility)
        exp = manifest["files"][os.path.basename(p)]["sha256"]
        got = sha256_file(p)
        if exp != got:
            raise IOError(f"sha256 mismatch for {p}: manifest {exp[:16]} != file {got[:16]}")
    shards = torch.load(mine, map_location=map_location, weights_only=False)
    rep = torch.load(rp, map_location=map_location, weights_only=False)
    return {"shards": shards, "replicated": rep.get("tensors", {}), "rng": rep.get("rng"), "meta": rep.get("meta", {}), "manifest": manifest}


def save_cycle(directory: str, cycle: int, rank: int, shards: dict, replicated: dict, rng_state_: Optional[dict] = None, *, meta: Optional[dict] = None,
               P: Optional[int] = None) -> str:
    """Write the ``cycle_<kkk>`` tag of ``cycle`` (:func:`save_tag`; ``P`` from the comm when None; ``rng_state_`` None = capture now). Returns the
    manifest path (it exists once rank 0 has seen every rank's fragment)."""
    P = world()[0] if P is None else int(P)
    tag = cycle_tag(cycle)
    m = dict(meta or {})
    m.setdefault("cycle", int(cycle))
    m.setdefault("kind", "cycle")
    save_tag(directory, tag, rank=int(rank), P=P, shards=shards, replicated=replicated, meta=m, step=int(cycle), rng_state_override=rng_state_)
    return os.path.join(directory, tag, "manifest.json")


def load_cycle(directory: str, cycle: int, rank: int, *, P: Optional[int] = None, map_location="cpu"):
    """``(shards, replicated, rng_state)`` of the complete ``cycle_<kkk>`` tag for this rank (sha256 checked)."""
    st = load(directory, cycle_tag(cycle), rank=int(rank), P=P, map_location=map_location, verify=True)
    return st["shards"], st["replicated"], st["rng"]


def latest_complete_cycle(directory: str, world_: int) -> Optional[int]:
    """The newest cycle whose tag is complete for ``world_`` ranks (manifest present, ``P == world_``, every ``rank<r>.pt`` named), else None."""
    best = None
    for tag, man in list_complete(directory):
        if not tag.startswith("cycle_") or int(man.get("P", -1)) != int(world_):
            continue
        files = man.get("files", {})
        if not all(f"rank{r}.pt" in files for r in range(int(world_))):
            continue
        try:
            k = int(tag[len("cycle_"):])
        except ValueError:
            continue
        best = k if best is None else max(best, k)
    return best


def _owned(t):
    """``t`` itself when torch owns its storage (resizable), else a clone: a ``torch.load``'ed CPU tensor sits on a non-resizable storage, and the
    trunk's statements release / re-grow / park the shard they are handed (:func:`.shard.storage_resizable`)."""
    from .shard import storage_resizable
    return t if storage_resizable(t) else t.clone()


def _pinned_cpu_copy(t):
    if not isinstance(t, torch.Tensor):
        return t
    if t.is_cuda:
        h = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        h.copy_(t)
        return h
    return t.detach().clone()


# ----------------------------------------------------------------------------------------------------------------- policy object
class TrunkCheckpointer(object):
    """The per-query checkpoint policy a sharded trunk loop drives (see the module docstring for the environment defaults)."""

    def __init__(self, query_id: str, seed, N: int, root: Optional[str] = None, every: Optional[int] = None, resume=None, keep: Optional[int] = None,
                 log=None, phase_log=None, *, features=None, precision: Optional[str] = None, dtype=None, allow_features: Optional[bool] = None):
        """``features``: EITHER this rank's input-feature object (a batch dict / any nest of tensors and plain values — hashed with
        :func:`feature_hash`) OR a precomputed 64-hex digest string. The object is hashed LAZILY: never when checkpointing is disabled
        (``ROWPAIR_CKPT_DIR`` unset: the object is not even looked at), else ONCE, at the first :meth:`try_resume` / save / :meth:`digest_features`
        call — adapters call :meth:`try_resume` at trunk entry, so the writing run and the resuming run hash the same state — after which the
        object reference is dropped. The digest is written per rank into every tag and REQUIRED to match on resume
        (``resume_refused:features_differ``; ``ROWPAIR_RESUME_ALLOW_FEATS=1`` / ``allow_features=True`` turns the refusal into a named allowance);
        None records ``null`` (matches only a tag that recorded ``null``). ``precision``: the line's
        precision word (``bf16`` / ``fp32`` / ``tf32`` …) — written into every tag with the saved tensors' dtypes and REQUIRED to match on resume
        (``resume_refused:precision_differs``). ``dtype``: the pair shard's dtype this line produces; a resumed shard of another dtype is refused
        (``resume_refused:dtype_differs``)."""
        self.root_base = root if root is not None else os.environ.get("ROWPAIR_CKPT_DIR", "")
        self.enabled = bool(self.root_base)
        self.query = f"{query_id}_{seed}"
        self.root = os.path.join(self.root_base, self.query) if self.enabled else ""
        self.N = int(N)
        dflt_every = 1 if self.N >= 8000 else 0
        self.every = env_int("ROWPAIR_CKPT_EVERY", dflt_every) if every is None else int(every)
        rmode = os.environ.get("ROWPAIR_RESUME", "1") if resume is None else (resume if isinstance(resume, str) else ("1" if resume else "0"))
        self.resume = rmode not in ("0", "", "false", "no")
        self.require_final = rmode == TRUNK_FINAL
        self.restore_rng = os.environ.get("ROWPAIR_RESUME_RNG", "1") == "1"
        self.keep = env_int("ROWPAIR_CKPT_KEEP", 1) if keep is None else int(keep)
        self.async_ = os.environ.get("ROWPAIR_CKPT_ASYNC", "0") == "1"
        self.fatal = os.environ.get("ROWPAIR_CKPT_FATAL", "0") == "1"
        self.log = log if log is not None else comm().log
        self.phase_log = phase_log
        self._thread = None
        self._thread_err = None
        self._features_obj, self._features_digest, self._features_done = None, None, True
        if isinstance(features, str) and len(features) == 64:
            self._features_digest = features.lower()
        elif features is not None:
            if self.enabled:
                self._features_obj, self._features_done = features, False     # hashed at the first try_resume / save (digest_features), never here
            # disabled: the object is neither kept nor hashed — a default run pays nothing for passing its batch
        self.precision = str(precision) if precision is not None else None
        self.dtype = str(dtype).replace("torch.", "") if dtype is not None else None
        self.allow_features = (os.environ.get(ENV_RESUME_ALLOW_FEATS, "0").strip() not in ("", "0", "false", "no", "off")) if allow_features is None \
            else bool(allow_features)
        self.last_written = None                   # the newest tag this rank wrote (census word ckpt_wrote)
        self.resumed_from = None                   # the tag this rank resumed from (census word ckpt_resume)

    # ---- phases (memory peaks per phase go to the rank log)
    @property
    def features(self) -> Optional[str]:
        """This rank's feature digest (64 hex) or None — hashing the object given at construction on first access (:meth:`digest_features`)."""
        return self.digest_features()

    def digest_features(self) -> Optional[str]:
        """Hash the features object NOW (idempotent; a no-op when a digest string was given, when ``features`` was None, or when checkpointing is
        disabled). Returns the digest or None. Called by :meth:`try_resume` and by every save; an adapter may call it to pin the moment."""
        if not self._features_done:
            t0 = time.time()
            self._features_digest = feature_hash(self._features_obj)
            self._features_obj, self._features_done = None, True
            self.log(f"[ckpt] feature digest {self._features_digest[:12]} ({time.time() - t0:.1f}s)")
        return self._features_digest

    def phase(self, name: str) -> None:
        if self.phase_log is not None:
            try:
                self.phase_log(name)
            except Exception:  # noqa: BLE001 — a phase log is advisory
                pass
        elif torch.cuda.is_available():
            self.log(f"phase {name}: max_alloc={torch.cuda.max_memory_allocated() / 2 ** 30:.2f} GiB reserved={torch.cuda.memory_reserved() / 2 ** 30:.2f} GiB")

    # ---- save
    def _meta(self, layout, cycle: int, kind: str, tensors: Optional[dict] = None) -> dict:
        return {"kind": kind, "cycle": int(cycle), "N": int(layout.N), "P": int(layout.P), "parts": [list(p) for p in layout.parts],
                "align": layout.align, "B": int(layout.B), "policy": layout.policy, "query": self.query,
                "precision": precision_tag(self.precision, **(tensors or {})), "features_rank0": self.features}

    def _join(self) -> None:
        if self._thread is not None:
            self._thread.join()
            self._thread = None
            if self._thread_err is not None:      # a failed checkpoint write is NON-FATAL (costs resumability, never the run) unless ROWPAIR_CKPT_FATAL=1
                e = self._thread_err
                self._thread_err = None
                if self.fatal:
                    raise RuntimeError(f"async checkpoint write failed: {e}")
                self.log(f"[ckpt] WRITE FAILED (non-fatal; tag incomplete, retried at the next boundary): {type(e).__name__}: {e}")

    def _do_save(self, tag, layout, shards, replicated, meta, step, rank, rng_st):
        # `rank` and the RNG state are captured by the caller in the rank's own thread — the writer thread never touches the comm
        man = save_tag(self.root, tag, rank=rank, P=layout.P, shards=shards, replicated=replicated, meta=meta, step=step, rng_state_override=rng_st,
                       log=self.log, features=self.features)
        self.last_written = tag
        _record(ckpt_wrote=tag)
        if rank == 0 and self.keep > 0 and tag.startswith("cycle_"):
            self._prune()
        return man

    def _save(self, tag, layout, shards, replicated, meta, step):
        self._join()
        rank = comm().rank
        rng_st = rng_state()                       # captured NOW (synchronously, at the statement boundary) even if the file write is deferred
        if self.async_:
            hs = {k: _pinned_cpu_copy(v) for k, v in shards.items()}
            hr = {k: _pinned_cpu_copy(v) for k, v in replicated.items()}

            def body():
                try:
                    self._do_save(tag, layout, hs, hr, meta, step, rank, rng_st)
                except BaseException as e:  # noqa: BLE001 — reported at the next join
                    self._thread_err = e
            self._thread = threading.Thread(target=body, name=f"rowpair-ckpt-{tag}-r{rank}", daemon=True)
            self._thread.start()
            return None
        try:
            return self._do_save(tag, layout, shards, replicated, meta, step, rank, rng_st)
        except Exception as e:  # noqa: BLE001 — non-fatal by policy (named in the log), fatal under ROWPAIR_CKPT_FATAL=1
            if self.fatal:
                raise
            self.log(f"[ckpt] WRITE FAILED (non-fatal; tag incomplete): {type(e).__name__}: {e}")
            return None

    def maybe_save_cycle(self, cycle: int, *, layout, s, z_loc):
        """Write ``cycle_<kkk>`` when enabled and ``(cycle + 1) % every == 0``; shards = ``{"z_loc"}``, replicated = ``{"s"}``."""
        if not self.enabled or self.every <= 0 or (int(cycle) + 1) % self.every != 0:
            return None
        return self._save(cycle_tag(cycle), layout, {"z_loc": z_loc}, {"s": s}, self._meta(layout, cycle, "cycle", {"z": z_loc, "s": s}), int(cycle))

    def save_trunk_final(self, *, layout, s_input, s, z_loc, cycle: int):
        """Write ``trunk_final`` (shards = ``{"z_loc"}``, replicated = ``{"s", "s_input"}``) when enabled."""
        if not self.enabled:
            return None
        return self._save(TRUNK_FINAL, layout, {"z_loc": z_loc}, {"s": s, "s_input": s_input},
                          self._meta(layout, cycle, TRUNK_FINAL, {"z": z_loc, "s": s, "s_input": s_input}), int(cycle))

    def finalize(self) -> None:
        """Join a pending async write (call before process exit / before relying on the files)."""
        self._join()

    def _prune(self) -> None:
        tags = [t for t, m in list_complete(self.root) if t.startswith("cycle_")]
        for t in tags[:-self.keep]:
            shutil.rmtree(os.path.join(self.root, t), ignore_errors=True)

    # ---- resume
    def compatible(self, man: dict, layout) -> bool:
        """A tag is a CANDIDATE for this run iff format, P, N and the row partition match (an older format, another P / N / partition is passed
        over, said in the log by :meth:`find_resume`). Equality — features, precision — is :meth:`check_identity`'s refusal, not a skip."""
        m = man.get("meta", {})
        return (man.get("format") == FORMAT and int(man.get("P", -1)) == layout.P and int(m.get("N", -1)) == layout.N
                and [list(p) for p in m.get("parts", [])] == [list(p) for p in layout.parts])

    def why_incompatible(self, man: dict, layout) -> str:
        m = man.get("meta", {})
        if man.get("format") != FORMAT:
            return f"format {man.get('format')} != {FORMAT}"
        if int(man.get("P", -1)) != layout.P:
            return f"P {man.get('P')} != {layout.P}"
        if int(m.get("N", -1)) != layout.N:
            return f"N {m.get('N')} != {layout.N}"
        return "row partition differs"

    def check_identity(self, tag: str, man: dict, rank: int) -> str:
        """REFUSE BY NAME a candidate tag whose equality differs from this run's: the per-rank feature digest (``resume_refused:features_differ``
        — an allowance under ``ROWPAIR_RESUME_ALLOW_FEATS=1``, named) and the precision word (``resume_refused:precision_differs``). Returns the
        digest word for the resume line: ``digest ok`` | ``digest unrecorded`` (both sides None) | ``digest DIFFERS (allowed: ROWPAIR_RESUME_ALLOW_FEATS=1)``."""
        theirs = (man.get("files", {}).get(f"rank{rank}.pt") or {}).get("features")
        mine = self.features
        if theirs != mine:
            detail = (f"{tag}: rank {rank} feature digest {str(theirs)[:12] if theirs else 'unrecorded'} (checkpoint) != "
                      f"{str(mine)[:12] if mine else 'unrecorded'} (this run) — the checkpoint was written for other inputs")
            if not self.allow_features:
                _record(ckpt_resume=f"{REFUSED_FEATURES}:{tag}")
                raise RowpairRefused(f"{REFUSED_FEATURES}: {detail}; {ENV_RESUME_ALLOW_FEATS}=1 resumes anyway (named)")
            self.log(f"[ckpt] resume from {tag}: features DIFFER — ALLOWED by {ENV_RESUME_ALLOW_FEATS}=1 ({detail})")
            word = f"digest DIFFERS (allowed: {ENV_RESUME_ALLOW_FEATS}=1)"
        else:
            word = "digest ok" if mine is not None else "digest unrecorded"
        prec = (man.get("meta", {}).get("precision") or {}).get("word")
        if prec != self.precision:
            _record(ckpt_resume=f"{REFUSED_PRECISION}:{tag}")
            raise RowpairRefused(f"{REFUSED_PRECISION}: {tag} was written by a '{prec}' line; this line is '{self.precision}' "
                                 "(a shard resumed into another precision continues different arithmetic)")
        return word

    @property
    def read_root(self) -> str:
        """``ROWPAIR_CKPT_READ_DIR`` (default = the write root): the resume SOURCE directory — lets a fork read ``cycle_k`` of a live run while writing
        its own tags elsewhere. The run-key subdirectory is mirrored under the read dir unless it already points at one."""
        rd = os.environ.get("ROWPAIR_CKPT_READ_DIR")
        if not rd:
            return self.root
        key = os.path.basename(os.path.normpath(self.root))
        cand = os.path.join(rd, key)
        return cand if os.path.isdir(cand) else rd

    def find_resume(self, layout):
        """``(tag, manifest)``: ``ROWPAIR_RESUME_TAG`` if set (must be complete + compatible, else refused by name), else ``trunk_final`` if complete +
        compatible, else the newest compatible ``cycle_<k>``, else ``(None, None)``."""
        if not (self.enabled and self.resume) or not os.path.isdir(self.read_root):
            return None, None
        complete = dict(list_complete(self.read_root))
        for t in sorted(complete):                                       # a tag load() would refuse is an absent tag here too (said by _refusal)
            try:
                why = _refusal(*(os.path.join(self.read_root, t, n) for n in ("manifest.json", f"rank{comm().rank}.pt", "replicated.pt")))
            except OSError:                                              # a file load() needs is missing: load() says so, as before
                why = None
            if why is not None:
                del complete[t]
        for t in sorted(complete):                                       # a complete tag passed over is SAID (never a silent fresh start)
            if not self.compatible(complete[t], layout):
                self.log(f"[ckpt] {t} passed over: {self.why_incompatible(complete[t], layout)}")
        want = os.environ.get("ROWPAIR_RESUME_TAG", "").strip()
        if want:
            if want in complete and self.compatible(complete[want], layout):
                return want, complete[want]
            raise RowpairRefused(f"ROWPAIR_RESUME_TAG={want} not complete/compatible under {self.read_root} (complete: {sorted(complete)})")
        if TRUNK_FINAL in complete and self.compatible(complete[TRUNK_FINAL], layout):
            return TRUNK_FINAL, complete[TRUNK_FINAL]
        cyc = sorted(t for t in complete if t.startswith("cycle_") and self.compatible(complete[t], layout))
        if cyc:
            return cyc[-1], complete[cyc[-1]]
        return None, None

    def try_resume(self, *, layout, device, num_cycles: Optional[int] = None):
        """Load the resume tag for this rank: ``{"tag", "cycle", "z_loc", "s" (+ "s_input" for trunk_final)}`` on ``device`` with the RNG state
        restored (``ROWPAIR_RESUME_RNG``), or None when nothing resumes. ``ROWPAIR_RESUME=trunk_final`` without a compatible ``trunk_final`` is
        refused by name. The returned tensors sit on torch-OWNED (resizable) storage on every device (a ``torch.load``'ed CPU tensor is
        cloned), so the trunk's release / re-grow / park statements accept the resumed shard."""
        if self.enabled:
            self.digest_features()                                       # trunk entry: the state the writing run and the resuming run share
        tag, man = self.find_resume(layout)
        if self.require_final and tag != TRUNK_FINAL:
            raise RowpairRefused(f"ROWPAIR_RESUME=trunk_final but no complete/compatible trunk_final checkpoint under {self.read_root} (N={layout.N}, P={layout.P})")
        if tag is None:
            _record(ckpt_resume="none")
            return None
        rank = comm().rank
        digest_word = self.check_identity(tag, man, rank)               # refuses by name BEFORE any byte is loaded
        st = load(self.read_root, tag, rank=rank, P=layout.P, map_location="cpu", verify=True)
        z_loc = _owned(st["shards"]["z_loc"].to(device))                    # a torch.load'ed CPU storage is not resizable: the trunk gets a torch-OWNED shard
        if self.dtype is not None and str(z_loc.dtype).replace("torch.", "") != self.dtype:
            _record(ckpt_resume=f"{REFUSED_DTYPE}:{tag}")
            raise RowpairRefused(f"{REFUSED_DTYPE}: {tag} holds a {z_loc.dtype} pair shard; this line's trunk produces {self.dtype}")
        rep = st["replicated"]
        if st.get("rng") is not None and self.restore_rng:
            set_rng_state(st["rng"])
        out = {"tag": tag, "cycle": int(st["meta"]["cycle"]), "z_loc": z_loc, "s": _owned(rep["s"].to(device))}
        if tag == TRUNK_FINAL:
            out["s_input"] = _owned(rep["s_input"].to(device))
        if int(z_loc.shape[-3]) != layout.n_loc or int(z_loc.shape[-2]) != layout.N:
            raise RowpairRefused(f"resume {tag}: shard shape {tuple(z_loc.shape)} does not match layout rows {layout.r0}:{layout.r1} of N={layout.N}")
        self.resumed_from = tag
        _record(ckpt_resume=tag)
        prec = (st.get("manifest", {}).get("meta", {}).get("precision") or man.get("meta", {}).get("precision") or {}).get("word")
        self.log(f"[ckpt] resume from {tag} (cycle {out['cycle']}) {digest_word} (sha256 verified; precision {prec}; "
                 f"read_root={'own' if self.read_root == self.root else self.read_root})")
        return out
