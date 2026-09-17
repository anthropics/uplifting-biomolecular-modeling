"""`opt_manifest.json` — written beside the outputs by every `pred` and `warm` run: what ran, on what, with which levers.

Schema `opendde_opt.manifest/1`: the activation report (the same dict `enable()` returned, refreshed with the kits' own install records
after the run), upstream's flags as stated and as in force (stock_flags), the deterministic recipe, the inputs (path, sha256, items), the outputs index, the versions of
the stack in the process, the GPU, the JIT cache key, the weights (root, checkpoint path, size, sha256 when hashed), the stock proof
(mode off) or the served-levers report (kit modes), timings and the exit code.
"""
from __future__ import annotations

import json
import errno
import os
import platform
import sys
import time

from opt_core import manifest as _core_manifest
from opt_core.mem.rowpair import launch as _launch

from . import digest_memo

SCHEMA = "opendde_opt.manifest/1"
NAME = "opt_manifest.json"


def versions() -> dict:
    """Versions from package metadata (no import of torch when it is not loaded); torch/cuda/triton details when torch is loaded."""
    import importlib.metadata as md
    out = {"python": platform.python_version(), "executable": sys.executable}
    for name in ("opendde", "torch", "triton", "numpy", "scipy", "gemmi", "cuequivariance", "cuequivariance-torch", "cuequivariance-ops-torch-cu12",
                 "cuequivariance-ops-cu12", "rdkit", "biotite", "biopython", "pandas"):
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            out[name] = None
    t = sys.modules.get("torch")
    if t is not None:
        out["torch_full"] = getattr(t, "__version__", None)
        out["cuda"] = getattr(getattr(t, "version", None), "cuda", None)
        try:
            out["cudnn"] = t.backends.cudnn.version()
        except Exception:  # noqa: BLE001
            pass
    return out


def pinned_checkpoint(tree: str | None) -> dict:
    """stock/PINS.json "checkpoint": the equality as shipped (bytes, sha256) — a note, never a gate."""
    if not tree:
        return {}
    try:
        with open(os.path.join(tree, "stock", "PINS.json")) as fh:
            c = json.load(fh).get("checkpoint") or {}
        return {"bytes": int(c["bytes"]), "sha256": str(c["sha256"])} if "bytes" in c and "sha256" in c else {}
    except (OSError, ValueError, KeyError):
        return {}


DIGEST_DIR_ENV = "MODEL_OPT_WEIGHTS_DIGEST_DIR"                     # the weights digest memo's directory; default: beside the JIT caches of the stack key
DIGEST_RELAY_ENV = "MODEL_OPT_WEIGHTS_SHA256"                      # the launcher's digest handed to its rank processes through their inherited environment:
                                                                   # "<stat key json>\t<sha256>\t<utc|fresh>" — read by a rank only when the memo has no entry


def digest_memo_dir() -> str:
    """The directory of the on-disk weights digest memo (``digest_memo.MEMO_NAME`` inside it): ``$MODEL_OPT_WEIGHTS_DIGEST_DIR``, else
    ``<parent of $TRITON_CACHE_DIR>/weights_digest`` (configs/<gpu>.env under MODEL_OPT_JIT_ROOT exports the first as <root>/weights and lays the caches out as
    <root>/<stack key>/{triton,torch_ext}),
    else ``~/.cache/opendde_opt/weights_digest``."""
    d = os.environ.get(DIGEST_DIR_ENV)
    if d:
        return d
    tc = os.environ.get("TRITON_CACHE_DIR")
    if tc:
        return os.path.join(os.path.dirname(tc.rstrip("/")) or "/", "weights_digest")
    return os.path.join(os.path.expanduser("~"), ".cache", "opendde_opt", "weights_digest")


MEMO_READONLY_ERRNOS = (errno.EROFS, errno.EACCES, errno.EPERM)                          # a read-only / permission-denied memo directory: benign, worded without exception text


def memo_readonly(e: OSError) -> bool:
    """True for the benign memo failures: the memo directory is read-only or not writable by this user (``MEMO_READONLY_ERRNOS``)."""
    return isinstance(e, OSError) and getattr(e, "errno", None) in MEMO_READONLY_ERRNOS


def _checkpoint_digest(ck: str, refresh: bool) -> dict:
    """The checkpoint's sha256 through the digest memo: this process hashes (a miss, or ``refresh``) or reads the memoised digest of the same
    (realpath, size, mtime_ns, inode); a rank process of a ``--n_gpu P>1`` run reads the entry its launcher wrote and never hashes (a miss there
    is an error by name). A memo on a read-only or permission-denied directory (``MEMO_READONLY_ERRNOS``: a cache volume mounted read-only) is a
    benign condition: the digest is taken afresh and the WEIGHTS line says `digest memo read-only — hashed afresh`, with no exception text;
    any other OSError on the memo is named on the WEIGHTS line with its exception text and the digest is still taken afresh."""
    memo_dir = digest_memo_dir()
    out = {"digest_memo": os.path.join(memo_dir, digest_memo.MEMO_NAME), "digest_cached_utc": None, "digest_memo_error": None, "digest_memo_readonly": False}
    if os.environ.get(_launch.ENV_RANK) is not None:                                  # a rank process: the launcher digested before the ranks started — never hash here
        try:
            out["checkpoint_sha256"], out["digest_cached_utc"] = digest_memo.rank_digest(ck, memo_dir)
            out["digest_source"] = "rank:launcher_memo"
        except RuntimeError:                                                            # no memo entry (the launcher could not write it): its relayed digest, or the miss by name
            relay = (os.environ.get(DIGEST_RELAY_ENV) or "").split("\t")
            if len(relay) != 3 or relay[0] != digest_memo.stat_key(ck) or len(relay[1]) != 64:
                raise
            out["checkpoint_sha256"], out["digest_cached_utc"] = relay[1], (None if relay[2] == "fresh" else relay[2])
            out["digest_source"] = "rank:launcher_env"
        return out
    try:
        out["checkpoint_sha256"], out["digest_cached_utc"] = digest_memo.digest(ck, memo_dir, refresh=refresh)
        out["digest_source"] = "hashed" if out["digest_cached_utc"] is None else "memo"
    except OSError as e:                                                                # the memo file could not be written/read: hash afresh, name it
        out["checkpoint_sha256"] = digest_memo.sha256_file(os.path.realpath(ck))
        if memo_readonly(e):                                                            # benign: a read-only / permission-denied memo directory — no exception text on the line
            out["digest_source"] = "hashed:memo_readonly"; out["digest_memo_readonly"] = True
        else:
            out["digest_source"] = "hashed:memo_unwritable"; out["digest_memo_error"] = f"{type(e).__name__}: {e}"[:200]
    os.environ[DIGEST_RELAY_ENV] = "\t".join((digest_memo.stat_key(ck), out["checkpoint_sha256"], out["digest_cached_utc"] or "fresh"))   # inherited by the rank processes this process launches
    return out


def weights(root: str | None, tree: str | None = None, refresh: bool = False) -> dict:
    """The weights facts + the checkpoint's equality, decided by its sha256 DIGEST against the pinned digest in stock/PINS.json "checkpoint":
    `pinned`, `unknown` (any other digest — a file of the recorded size included —: a WARNING by name; every route proceeds), `absent` (no
    file; `pred`'s frozen-weights gate refuses that by name), `unchecked` (no tree given). The digest comes through the on-disk memo
    (``digest_memo``): size / mtime / inode select the memo entry, they never decide equality; ``refresh=True`` (the `check` verb) hashes afresh
    and rewrites the entry. The digest is always taken."""
    root = root or os.environ.get("OPENDDE_ROOT_DIR")
    ck = os.path.join(root, "checkpoint", "opendde.pt") if root else None
    present = bool(ck and os.path.isfile(ck))
    w = {"root": root, "checkpoint": ck, "checkpoint_present": present,
         "checkpoint_bytes": os.path.getsize(ck) if present else None, "checkpoint_sha256": None,
         "files_hashed": ["checkpoint/opendde.pt"] if present else [], "digest_memo": None, "digest_cached_utc": None, "digest_source": None, "digest_memo_error": None,
         "digest_memo_readonly": False}
    if present:
        w.update(_checkpoint_digest(ck, refresh))
    common = os.path.join(root, "common") if root else None
    w["common_present"] = bool(common and os.path.isdir(common))
    rec = pinned_checkpoint(tree)
    w["pinned"] = rec or None
    if not present:
        w["checkpoint_match"] = "absent"
    elif not rec:
        w["checkpoint_match"] = "unchecked"
    elif w["checkpoint_sha256"] == rec["sha256"]:
        w["checkpoint_match"] = "pinned"
    else:
        w["checkpoint_match"] = "unknown"
    return w


def weights_line(w: dict, prefix: str = "[opendde-opt]") -> str:
    """One WEIGHTS line per process, the digest first: `WEIGHTS sha256=<12> (pinned)`, or the WARNING by name for any other digest — the
    run proceeds (never a refusal); `(cached digest <utc>)` follows the equality word when the memo served the digest; `absent` / `unchecked`
    say so."""
    d = str(w.get("checkpoint_sha256") or "")[:12]
    ident = w.get("checkpoint_match"); cached = w.get("digest_cached_utc")
    tail = f" checkpoint={w.get('checkpoint')} bytes={w.get('checkpoint_bytes')} memo={w.get('digest_memo')}"
    if w.get("digest_memo_readonly"):
        tail += " (digest memo read-only — hashed afresh)"
    elif w.get("digest_memo_error"):
        tail += f" (digest memo unwritable, hashed afresh: {w['digest_memo_error']})"
    if ident == "pinned":
        return f"{prefix} WEIGHTS sha256={d} {digest_memo.word('(pinned)', cached)}{tail}"
    if ident == "unknown":
        rec = w.get("pinned") or {}
        return (f"{prefix} WEIGHTS sha256={d} {digest_memo.word('UNKNOWN', cached)} — not the pinned checkpoint "
                f"(sha256 {str(rec.get('sha256'))[:12]}, {rec.get('bytes')} bytes); proceeding —{tail}")
    if ident == "absent":
        return f"{prefix} WEIGHTS checkpoint absent under root={w.get('root')} (pred's frozen-weights gate names it)"
    return f"{prefix} WEIGHTS sha256={d} {digest_memo.word('unchecked', cached)} (no tree: no pinned digest to compare){tail}"


def build(*, command: str, mode: str | None, line: str | None, activation: dict, tree: str, stock_flags: dict | None, det: dict | None,
          inputs: dict | None, outputs: dict | None, gpu: dict | None, stack_key: str | None, root: str | None, extra: dict | None = None,
          started: float | None = None, exit_code: int | None = None) -> dict:
    now = time.time()
    m = {"schema": SCHEMA, "model": "opendde", "command": command, "mode": mode, "line": line, "activation": activation,
         "stock_flags": stock_flags, "det": det, "inputs": inputs, "outputs": outputs, "versions": versions(), "gpu": gpu, "stack_key": stack_key, "core": _core_manifest.core_block(),
         "weights": weights(root, tree=tree), "tree": tree,
         "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started or now)),
         "finished_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(now)),
         "wall_s": round(now - started, 3) if started else None, "exit_code": exit_code, "pid": os.getpid(), "argv": list(sys.argv)}
    if extra:
        m.update(extra)
    return m


def write(out_dir: str, m: dict) -> str:
    os.makedirs(out_dir, exist_ok=True)
    p = os.path.join(out_dir, NAME)
    with open(p, "w") as fh:
        json.dump(m, fh, indent=1, default=str)
        fh.write("\n")
    return p
