"""The checkpoint's digest against the pin, said once per command that touches weights (``pred``, ``check``; ``warm`` goes through ``pred``):

    [rosettafold3-opt] WEIGHTS checkpoint=pinned sha256=<12 hex> bytes=<n> file=<name> (the pinned checkpoint: stock/PINS.json weights rf3)
    [rosettafold3-opt] WEIGHTS checkpoint=pinned (cached digest <utc>) sha256=<12 hex> ...          (the digest served from the on-disk memo)
    [rosettafold3-opt] WEIGHTS checkpoint=unknown sha256=<12 hex> bytes=<n> file=<name> (not the pinned checkpoint <12 hex>: ...); proceeding

Whether a file is the pinned checkpoint is decided by its sha256 DIGEST against the pin (``stock/PINS.json`` "weights"."rf3"."sha256") —
never by size, name or mtime. A checkpoint that is not the pinned one is a NAMED WARNING and the command proceeds in every mode, never a
refusal; the line carries the pin beside the file's own digest.

The digest goes through the on-disk memo ``digest_memo`` (``<memo dir>/weights_digests.json``): the file's (realpath, size, mtime_ns,
inode) SELECT a memo entry — they never decide the answer — and a hit serves the memoised digest with the line saying
``(cached digest <utc>)``; ``check`` hashes afresh and rewrites the entry (``REFRESH_VERBS``); ``pred`` reads the memo. The memo dir
is ``ROSETTAFOLD3_OPT_DIGEST_DIR`` when set, else ``<jit cache root>/weights`` beside the deployment's JIT caches
(``TRITON_CACHE_DIR`` = ``<root>/<stack key>/triton`` in configs/h100.env → ``<root>/weights``), else ``~/.cache/rosettafold3_opt/weights``.
One process digests: the ``pred`` driver before its fold subprocesses and row-sharded ranks start, which never hash. A memo dir the
process cannot write (a read-only mount, a foreign owner) is NAMED on the line and never a refusal: an entry already there is still read,
otherwise the file is hashed afresh and not memoised (``memo=read_only:<errno>`` / ``memo=unwritable:<errno>``).

The install step's weights route (``run.sh install --weights DIR`` = ``python -m rosettafold3_opt.weights DIR``, :func:`fetch`): the pinned
checkpoint present in ``DIR`` with the pinned digest — absent, upstream's own downloader fetches it (``foundry install rf3 --checkpoint-dir
DIR``: ``foundry_cli.download_checkpoints`` of the pinned rc-foundry, run on this interpreter); present or fetched, its sha256 must be the
pin's, else exit 1 with the file left in place and named::

    [rosettafold3-opt install] WEIGHTS OK: <DIR>/<file> has the pinned digest sha256=<12 hex> bytes=<n> (<s> s hashing) — export ROSETTAFOLD3_OPT_CKPT=<DIR>/<file>
"""
import errno
import os
import subprocess
import sys
import tempfile
import time
from typing import Callable, List, Optional

from . import digest_memo as _memo
from . import manifest as _manifest
from . import report as _report

ENV_DIGEST_DIR = "ROSETTAFOLD3_OPT_DIGEST_DIR"           # the weights digest memo directory (optional; default beside the JIT caches)
REFRESH_VERBS = ("check",)                              # the verbs that hash afresh and rewrite the memo entry; every other verb reads the memo


def memo_dir(environ=None) -> str:
    """The memo directory: ROSETTAFOLD3_OPT_DIGEST_DIR, else <jit cache root>/weights (TRITON_CACHE_DIR is <root>/<stack key>/triton), else
    ~/.cache/rosettafold3_opt/weights."""
    env = os.environ if environ is None else environ
    if env.get(ENV_DIGEST_DIR):
        return env[ENV_DIGEST_DIR]
    tc = env.get("TRITON_CACHE_DIR")
    if tc:
        return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(tc))), "weights")
    return os.path.join(os.path.expanduser("~"), ".cache", "rosettafold3_opt", "weights")


def memo_write_error(mdir: str):
    """None when this process can create the memo dir and write a file in it; else the errno name (EROFS, EACCES, ...) of the refusal."""
    try:
        os.makedirs(mdir, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=".w-", dir=mdir)
        os.close(fd); os.unlink(tmp)
        return None
    except OSError as e:
        return errno.errorcode.get(e.errno, type(e).__name__)


def digest(path: str, verb: str = "pred", environ=None):
    """(sha256, cached_utc, memo_state): through the memo when the memo dir is writable (``memo_state`` "ok"); on a memo dir this process cannot
    write, an entry already there is read (``read_only:<errno>``; not for ``REFRESH_VERBS``) and otherwise the file is hashed afresh and not
    memoised (``unwritable:<errno>``) — named, never a refusal. An unreadable CHECKPOINT still raises: that is the caller's error to see."""
    mdir = memo_dir(environ); refresh = verb in REFRESH_VERBS
    os.stat(path)                                                                # the checkpoint itself must exist: not the memo's condition to name
    werr = memo_write_error(mdir)
    if werr is None:
        sha, cached_utc = _memo.digest(path, mdir, refresh=refresh)
        return sha, cached_utc, "ok"
    if not refresh:
        try:
            sha, cached_utc = _memo.rank_digest(path, mdir)                     # the read-only lookup of an entry another process wrote
            return sha, cached_utc, f"read_only:{werr}"
        except RuntimeError:
            pass
    return _memo.sha256_file(os.path.realpath(path)), None, f"unwritable:{werr}"


def pin_status(path: str, pins: dict | None, verb: str = "pred", environ=None) -> dict:
    """``manifest.checkpoint_info`` with the file's digest (through the memo; afresh for ``REFRESH_VERBS``) and ``status`` =
    ``pinned`` | ``unknown``; ``status_word`` carries the memo note, ``digest_cached_utc`` the memo hit's time (None when hashed now),
    ``digest_memo_state`` ok | read_only:<errno> | unwritable:<errno>."""
    sha, cached_utc, memo_state = digest(path, verb=verb, environ=environ)
    info = _manifest.checkpoint_info(path, pins, sha=sha)
    info["digest_memo_state"] = memo_state
    info["status"] = "pinned" if info.get("sha256_matches_pin") else "unknown"
    info["status_word"] = _memo.word(info["status"], cached_utc)
    info["digest_cached_utc"] = cached_utc
    info["digest_memo"] = os.path.join(memo_dir(environ), _memo.MEMO_NAME)
    return info


def line(info: dict) -> str:
    sha = (info.get("sha256") or "")[:12]; pin = (info.get("pinned_sha256") or "")[:12]
    head = f"WEIGHTS checkpoint={info.get('status_word') or info['status']} sha256={sha} bytes={info.get('bytes')} file={os.path.basename(info.get('path') or '')}"
    st = info.get("digest_memo_state") or "ok"
    if st != "ok":                                                               # a memo dir this process cannot write: named, hashed afresh or read only, never a refusal
        head += f" memo={st}:{info.get('digest_memo')}" + ("" if st.startswith("read_only") else " (hashed afresh, not memoised)")
    if info["status"] == "pinned":
        return f"{head} (the pinned checkpoint: stock/PINS.json weights rf3)"
    return (f"{head} (not the pinned checkpoint {pin or 'none pinned'}: this run's outputs are this checkpoint's, "
            f"not the pinned one's); proceeding")


def announce(path: str, pins: dict | None, verb: str = "pred", file=None, environ=None) -> dict:
    """Identify ``path`` against the pin (``verb`` decides afresh vs memo), print the WEIGHTS line, return the info (path, size, digest,
    the pin's). Never raises on an unknown checkpoint."""
    info = pin_status(path, pins, verb=verb, environ=environ)
    print(f"{_report.PREFIX} {line(info)}", file=file or sys.stderr, flush=True)
    return info


# ----------------------------------------------------------------------------------------------------- the install step's weights route
PREFIX_INSTALL = _report.PREFIX + " install"             # the install step's lines ([rosettafold3-opt install] …), as run.sh install prints them
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
UPSTREAM_MODEL = "rf3"                                    # the registry name of the pinned checkpoint (stock/PINS.json "weights" key; `foundry install rf3`)


def upstream_fetch(directory: str, python: Optional[str] = None) -> int:
    """Upstream's own downloader — ``foundry install rf3 --checkpoint-dir DIR`` (the ``foundry`` console script of rc-foundry =
    ``foundry_cli.download_checkpoints``), run as a module on ``python`` (default: this interpreter, which carries the pinned rc-foundry) so no
    PATH lookup decides which install fetches. Its exit code."""
    cmd = [python or sys.executable, "-m", "foundry_cli.download_checkpoints", "install", UPSTREAM_MODEL, "--checkpoint-dir", directory]
    return subprocess.run(cmd).returncode


def fetch(directory: str, *, pins: Optional[dict] = None, route: Optional[Callable[[str], int]] = None, out=None) -> int:
    """``run.sh install --weights DIR``: the pinned checkpoint (stock/PINS.json "weights"."rf3": filename, url, sha256, bytes) in ``directory``
    with the pinned digest. A file already there is kept and checked; an absent one is fetched by ``route`` (default :func:`upstream_fetch`)
    and then checked. The digest is the file's sha256 hashed now (no memo: this is the one time the bytes arrive). Returns EXIT_OK, or
    EXIT_FAIL with the reason printed — a fetch that failed, a file that did not appear, a digest that is not the pin's (left in place, named)."""
    out = out or sys.stderr
    if pins is None:
        from . import stack as _stack
        pins = _stack.pins()
    w = ((pins or {}).get("weights") or {}).get(UPSTREAM_MODEL) or {}
    name, want = w.get("filename"), (w.get("sha256") or "").lower()
    if not name or not want:
        print(f"{PREFIX_INSTALL} FAILED: stock/PINS.json \"weights\".\"{UPSTREAM_MODEL}\" names no filename / sha256 to check against", file=out, flush=True)
        return EXIT_FAIL
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    target = os.path.join(directory, name)
    if os.path.isfile(target):
        print(f"{PREFIX_INSTALL} {name} is already under {directory}: kept, its digest is checked", file=out, flush=True)
    else:
        print(f"{PREFIX_INSTALL} fetching {name} ({w.get('bytes', '?')} bytes, {w.get('url', 'the registry URL')}) into {directory} with upstream's own downloader "
              f"(foundry install {UPSTREAM_MODEL} --checkpoint-dir {directory})", file=out, flush=True)
        rc = (route or upstream_fetch)(directory)
        if rc != 0:
            print(f"{PREFIX_INSTALL} FAILED: upstream's downloader exited {rc}; whatever it left under {directory} is in place", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.isfile(target):
            print(f"{PREFIX_INSTALL} FAILED: {name} is not at {target} after upstream's fetch", file=out, flush=True)
            return EXIT_FAIL
    t0 = time.time()
    sha = _memo.sha256_file(os.path.realpath(target))
    secs = time.time() - t0
    size = os.path.getsize(target)
    if sha != want:
        print(f"{PREFIX_INSTALL} REFUSED: {target} sha256 {sha[:16]}… is not the pin's {want[:16]}… (stock/PINS.json weights {UPSTREAM_MODEL}, {w.get('bytes', '?')} bytes; "
              f"this file has {size}) — left in place; remove it and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX_INSTALL} WEIGHTS OK: {target} has the pinned digest sha256={sha[:12]} bytes={size} ({secs:.0f} s hashing) — export ROSETTAFOLD3_OPT_CKPT={target}",
          file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m rosettafold3_opt.weights DIR   (run.sh install --weights DIR: the pinned checkpoint fetched into DIR and digest-checked)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
