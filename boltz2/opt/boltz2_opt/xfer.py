"""boltz2_opt.xfer — the featurized batch's process-to-process transport for LARGE CPU storages: a disk file, not ``/dev/shm``.

Why. The persistent featurizer's helper (``boltz2_opt.prefetch``) hands the featurized batch to the predicting process over a
``multiprocessing`` connection, and the first served input's bit-compare reference (and any named fallback) comes from a stock ``DataLoader``
worker over its result queue. Both pickle with ``multiprocessing.reduction.ForkingPickler``, for which ``torch.multiprocessing`` registers
``reduce_storage``: every CPU storage is MOVED into a POSIX shared-memory segment (``storage._share_fd_cpu_()`` → ``shm_open`` under
``/dev/shm``) and travels as a file descriptor. Boltz-2's featurized batch grows with the square of the token count — ``disto_target``
``[N, N, 1, 64]`` float32 (256 B/token²), the four one-hot atom↔token maps int64 (~250 B/token² at ~8 atoms per token),
``contact_conditioning`` one-hot int64 (40 B/token²), the ``[N, N]`` bond / threshold planes: ~560 B/token², i.e. ~9 GB at 4,000 tokens,
~45 GB at 9,000, ~80 GB at 12,000 — and ``/dev/shm`` is often a fixed-size tmpfs far smaller than that (a container's, for
instance). At 9,000 tokens the helper's batch (in shm, mapped by the predicting process) plus the reference worker's copy no longer fit: the
worker's queue feeder thread raises ``RuntimeError: unable to allocate shared memory(shm) … No space left on device (28)`` INSIDE
``multiprocessing.queues._feed`` — which prints the traceback and drops the payload — and the predicting process waits for a batch that never
comes (torch's loader loop has no deadline when ``timeout=0``): a silent hang with idle GPUs.

What. :func:`install` registers :func:`reduce_storage` for ``torch.UntypedStorage`` on ``ForkingPickler`` in the calling process (the
predicting process — its forked loader workers inherit the registration — and the helper, which installs it itself). Per storage:

  * not a CPU storage, an empty one, or — under the default word ``auto`` — one smaller than :data:`MIN_BYTES_DEFAULT` (4 GiB;
    ``BOLTZ_OPT_XFER_MIN_MIB``): torch's own ``reduce_storage``, verbatim (the shared-memory descriptor transport, unchanged —
    an input under ~4,000 tokens carries no storage that large, so its hand-over is torch's own path exactly);
  * otherwise the storage's bytes are written to a fresh file under :func:`xfer_dir` (``BOLTZ_OPT_XFER_DIR``, else ``<kit workdir>/_xfer`` — the
    launch's kit directory on the box's local disk — else a private per-user directory in the temp dir, this user's alone: made 0700 and
    refused by name when it is a symbolic link, another account's, or writable by group or other — the received bytes are mapped back as
    tensors, so nothing another account could write is read), the page cache told to let go of them (``posix_fadvise DONTNEED``), and
    the pickle carries ``(rebuild_storage_file, (path, nbytes))``; the receiving process maps the file privately
    (``torch.UntypedStorage.from_file(shared=False)``) and unlinks it at once — the disk blocks leave with the last mapping, storage by storage
    as the receiver consumes them. Tensor metadata (dtype, storage offset, size, stride) ride torch's own ``rebuild_tensor`` /
    ``rebuild_typed_storage`` unchanged: the received tensors are the sent tensors byte for byte, strides included. (Two distinct tensor
    objects viewing one storage arrive as two storages — values identical, aliasing not kept; a featurized batch reads its entries, never
    writes through one into another.)

Words (``BOLTZ_OPT_XFER``, read at install): ``auto`` (default) | ``shm`` (nothing registered: torch's transport for every storage,
by construction) | ``file`` (every non-empty CPU storage through a file — the tests' and the bitwise proof's word). A directory
without the free bytes a storage needs is refused BY NAME (``[boltz2-opt xfer] REFUSED …``), never a partial file: in the helper that is the
input's named fallback, in a loader worker torch's loader deadline names it (prefetch: ``BOLTZ_OPT_LOADER_TIMEOUT_S``).
Facts: :func:`stats` (files / bytes / seconds written and read in this process) — the ``[boltz2-opt] XFER lever=prefetch …`` line (prefetch
``xfer_line``, report ``xfer_lines``: a line of its own, no existing LEVER / ACTIVE / EXIT line changes format) and ``prefetch_report.xfer``.
"""
from __future__ import annotations

import os
import shutil
import stat
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Optional

TAG = "boltz2-opt"
ENV_WORD = "BOLTZ_OPT_XFER"                 # auto | shm | file
ENV_MIN_MIB = "BOLTZ_OPT_XFER_MIN_MIB"       # auto's per-storage floor in MiB (a storage of at least this many bytes travels as a file)
ENV_DIR = "BOLTZ_OPT_XFER_DIR"               # the directory the files are written in (default: <BOLTZ_OPT_WORKDIR>/_xfer, else the private per-user tempfile.gettempdir()/bz2xfer_<uid>)
WORKDIR_ENV = "BOLTZ_OPT_WORKDIR"            # worker_launch.WORKDIR_ENV: the launch's kit directory (<out_dir>/_kit), on the machine's local disk
WORDS = ("auto", "shm", "file")
DEFAULT_WORD = "auto"
MIN_BYTES_DEFAULT = 4 << 30                  # 4 GiB: disto_target [N, N, 1, 64] f32 crosses it at N = 4,096; the int64 atom<->token maps at ~8,300 tokens; nothing below ~4,000 tokens does
SUBDIR = "_xfer"
TMP_PREFIX = "bz2xfer_"                       # the per-user directory's name in the temp dir (bz2xfer_<uid>), and the prefix of the private one made when that is refused
WRITE_CHUNK = 1 << 30                        # os.write per call (Linux caps a single write at 0x7ffff000 bytes)
FREE_MARGIN = 64 << 20                       # bytes that must stay free in the directory beyond the storage being written

_LOCK = threading.Lock()
_STATE: Dict[str, Any] = {"installed": False, "word": None, "min_bytes": MIN_BYTES_DEFAULT, "dir": None, "orig": None, "count": 0,
                          "files_written": 0, "bytes_written": 0, "write_s": 0.0, "files_read": 0, "bytes_read": 0, "read_s": 0.0,
                          "delegated": 0, "largest_file_bytes": 0, "errors": []}


class Refused(RuntimeError):
    """A storage this transport cannot carry, by name (no space in the directory, an unknown word)."""


def _say(*words) -> None:
    sys.stderr.write(f"[{TAG} xfer] " + " ".join(str(w) for w in words) + "\n"); sys.stderr.flush()


def word(environ=None) -> str:
    """``BOLTZ_OPT_XFER``'s word (``auto`` when unset/empty); an unknown word raises :class:`Refused`."""
    env = os.environ if environ is None else environ
    w = (env.get(ENV_WORD) or DEFAULT_WORD).strip().lower() or DEFAULT_WORD
    if w not in WORDS:
        raise Refused(f"refused: {ENV_WORD}={env.get(ENV_WORD)!r}: one of {' | '.join(WORDS)} (or unset = {DEFAULT_WORD})")
    return w


def min_bytes(environ=None) -> int:
    """``auto``'s per-storage floor: ``BOLTZ_OPT_XFER_MIN_MIB`` MiB (default :data:`MIN_BYTES_DEFAULT`); a non-integer raises :class:`Refused`."""
    env = os.environ if environ is None else environ
    v = (env.get(ENV_MIN_MIB) or "").strip()
    if not v:
        return MIN_BYTES_DEFAULT
    try:
        return max(0, int(v)) << 20
    except ValueError:
        raise Refused(f"refused: {ENV_MIN_MIB}={v!r} is not an integer (MiB)") from None


def private_dir_refusal(path: str) -> Optional[str]:
    """Why the per-user temp-dir default ``path`` cannot carry this user's storages — None when it is a real directory (not a symbolic link)
    owned by this process's uid with no group / other write bit; made here (mode 0700) when absent. The files written under it are mapped
    back as tensor bytes by the receiving process, so a directory another account made, owns or can write is nothing to write through or
    read from: mkdir first, then lstat what is there (whoever made it first owns it)."""
    uid = os.geteuid()
    try:
        os.mkdir(path, 0o700)
    except FileExistsError:
        pass
    except OSError as e:
        return f"{path} cannot be made ({e.strerror or type(e).__name__}); fix: set {ENV_DIR} to a directory of your own"
    try:
        st = os.lstat(path)
    except OSError as e:
        return f"{path} cannot be read ({e.strerror or type(e).__name__}); fix: set {ENV_DIR} to a directory of your own"
    if stat.S_ISLNK(st.st_mode) or not stat.S_ISDIR(st.st_mode):
        return f"{path} is a symbolic link or not a directory; fix: remove it, or set {ENV_DIR} to a directory of your own"
    if st.st_uid != uid:
        return f"{path} belongs to uid {st.st_uid}, not to this process (uid {uid}); fix: remove it, or set {ENV_DIR} to a directory of your own"
    if st.st_mode & 0o022:
        return f"{path} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {path}"
    return None


def xfer_dir(environ=None) -> str:
    """Where the files go: ``BOLTZ_OPT_XFER_DIR``; else ``<BOLTZ_OPT_WORKDIR>/_xfer`` (the launch's kit directory on local disk,
    removed with :func:`cleanup`; created on first use, :func:`_written_path`); else the per-user ``<tempdir>/bz2xfer_<uid>`` — made 0700
    and used only when it is this user's own private directory (:func:`private_dir_refusal`): refused, it is named once on a ``REFUSED``
    line and a fresh private directory made for this process stands in (tempfile.mkdtemp, mode 0700; removed with :func:`cleanup`)."""
    env = os.environ if environ is None else environ
    d = (env.get(ENV_DIR) or "").strip()
    if d:
        return os.path.abspath(d)
    wd = (env.get(WORKDIR_ENV) or "").strip()
    if wd and os.path.isdir(wd):
        return os.path.join(os.path.abspath(wd), SUBDIR)
    d = os.path.join(tempfile.gettempdir(), f"{TMP_PREFIX}{os.geteuid()}")
    why = private_dir_refusal(d)
    if why is None:
        return os.path.abspath(d)
    private = os.path.abspath(tempfile.mkdtemp(prefix=TMP_PREFIX))
    _say(f"REFUSED: {why} — nothing under it is read or written; this process hands its storages over in {private} (private, this run only)")
    return private


def install(environ=None) -> bool:
    """Register :func:`reduce_storage` for ``torch.UntypedStorage`` on ``ForkingPickler`` in THIS process (idempotent; children forked after
    this inherit it). Under the word ``shm`` nothing is registered (torch's transport for every storage) and False is returned. torch's own
    reducer is kept as the delegate for the storages that stay on its path."""
    with _LOCK:
        if _STATE["installed"]:
            return _STATE["word"] != "shm"
        w = word(environ)
        _STATE.update(word=w, min_bytes=(0 if w == "file" else min_bytes(environ)), dir=xfer_dir(environ))
        import torch
        import torch.multiprocessing  # noqa: F401  (init_reductions: torch's reducers registered before ours replaces the UntypedStorage entry)
        from multiprocessing.reduction import ForkingPickler
        from torch.multiprocessing import reductions as TR
        _STATE["orig"] = ForkingPickler._extra_reducers.get(torch.UntypedStorage) or TR.reduce_storage
        _STATE["installed"] = True
        if w == "shm":
            return False
        ForkingPickler.register(torch.UntypedStorage, reduce_storage)
        return True


def installed() -> bool:
    return bool(_STATE["installed"]) and _STATE["word"] != "shm"


def describe() -> Dict[str, Any]:
    """The transport in force in this process: ``{word, min_mib, dir, installed}``."""
    return {"word": _STATE["word"] or None, "min_mib": (_STATE["min_bytes"] >> 20) if _STATE["word"] else None, "dir": _STATE["dir"], "installed": installed()}


def goes_by_file(nbytes: int, device_type: str = "cpu") -> bool:
    """Would a CPU storage of ``nbytes`` travel as a file under the installed word? (False before :func:`install` / under ``shm``.)"""
    if not installed() or device_type != "cpu" or int(nbytes) <= 0:
        return False
    return _STATE["word"] == "file" or int(nbytes) >= int(_STATE["min_bytes"])


# ----------------------------------------------------------------------------------------------------------------- sending side
def reduce_storage(storage):
    """``ForkingPickler``'s reducer for ``torch.UntypedStorage`` once :func:`install` ran: torch's own for the storages that stay on the
    shared-memory path (:func:`goes_by_file` False), else the storage's bytes in a file and ``(rebuild_storage_file, (path, nbytes))``."""
    nbytes = int(storage.nbytes())
    if not goes_by_file(nbytes, storage.device.type):
        with _LOCK:
            _STATE["delegated"] += 1
        return _STATE["orig"](storage)
    path = _write_file(storage, nbytes)
    return (rebuild_storage_file, (path, nbytes))


def _written_path() -> str:
    d = _STATE["dir"] or xfer_dir()
    os.makedirs(d, exist_ok=True)
    with _LOCK:
        _STATE["count"] += 1
        n = _STATE["count"]
    return os.path.join(d, f"st_{os.getpid()}_{n}_{os.urandom(4).hex()}.bin")


def _write_file(storage, nbytes: int) -> str:
    """The storage's bytes into a fresh file (O_EXCL, 0600) under :func:`xfer_dir`; refused by name when the directory lacks the room; the
    written pages handed back to the kernel (``posix_fadvise DONTNEED``: the receiver reads them from the page cache or the disk, they do
    not sit in this process's memory). Returns the path."""
    import torch
    path = _written_path()
    d = os.path.dirname(path)
    try:
        free = shutil.disk_usage(d).free
    except OSError:
        free = None
    if free is not None and free < nbytes + FREE_MARGIN:
        msg = (f"REFUSED: {d} has {free / 2 ** 30:.2f} GiB free, the featurized storage to hand over needs {nbytes / 2 ** 30:.2f} GiB "
               f"({ENV_DIR}=<a directory on a disk with room> moves the transport; {ENV_WORD}=shm restores torch's /dev/shm transport)")
        with _LOCK:
            _STATE["errors"].append(msg)
        _say(msg)
        raise Refused(f"[{TAG} xfer] {msg}")
    view = torch.empty(0, dtype=torch.uint8)
    view.set_(storage)                                        # [nbytes] uint8 over the very storage (no copy)
    mv = memoryview(view.numpy()).cast("B")
    t0 = time.perf_counter()
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        off = 0
        while off < nbytes:
            n = os.write(fd, mv[off:off + WRITE_CHUNK])
            if n <= 0:
                raise OSError(f"write returned {n} at offset {off} of {nbytes}")
            off += n
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)   # start writeback, drop what is clean: the bytes leave this process's page share
        except (AttributeError, OSError):
            pass
    except BaseException:
        os.close(fd); fd = None
        try:
            os.unlink(path)
        except OSError:
            pass
        raise
    finally:
        if fd is not None:
            os.close(fd)
        del mv, view
    dt = time.perf_counter() - t0
    with _LOCK:
        _STATE["files_written"] += 1; _STATE["bytes_written"] += nbytes; _STATE["write_s"] += dt
        _STATE["largest_file_bytes"] = max(_STATE["largest_file_bytes"], nbytes)
    return path


# ----------------------------------------------------------------------------------------------------------------- receiving side
def rebuild_storage_file(path: str, nbytes: int):
    """The receiving process's constructor (named in the pickle): the file mapped privately as an ``UntypedStorage`` of ``nbytes`` and
    unlinked at once (its blocks go when the storage does). A missing file is a named error (the sender's words are on its stderr)."""
    import torch
    t0 = time.perf_counter()
    if not os.path.isfile(path):
        raise Refused(f"[{TAG} xfer] the handed-over storage file {path} ({nbytes} bytes) is absent on the receiving side (the sending process's REFUSED / error line names why)")
    got = os.path.getsize(path)
    if got != int(nbytes):
        raise Refused(f"[{TAG} xfer] {path}: {got} bytes on disk, {nbytes} announced (a torn hand-over)")
    storage = torch.UntypedStorage.from_file(path, shared=False, nbytes=int(nbytes)) if int(nbytes) > 0 else torch.UntypedStorage(0)
    try:
        os.unlink(path)
    except OSError:
        pass
    with _LOCK:
        _STATE["files_read"] += 1; _STATE["bytes_read"] += int(nbytes); _STATE["read_s"] += time.perf_counter() - t0
    return storage


# ----------------------------------------------------------------------------------------------------------------- facts, cleanup
def stats() -> Dict[str, Any]:
    with _LOCK:
        st = dict(_STATE)
    return {"word": st["word"], "min_mib": (st["min_bytes"] >> 20) if st["word"] else None, "dir": st["dir"], "installed": bool(st["installed"]) and st["word"] != "shm",
            "files_written": st["files_written"], "bytes_written": st["bytes_written"], "gib_written": round(st["bytes_written"] / 2 ** 30, 3), "write_s": round(st["write_s"], 2),
            "files_read": st["files_read"], "bytes_read": st["bytes_read"], "gib_read": round(st["bytes_read"] / 2 ** 30, 3), "read_s": round(st["read_s"], 2),
            "largest_file_gib": round(st["largest_file_bytes"] / 2 ** 30, 3), "delegated": st["delegated"], "errors": list(st["errors"])}


def cleanup() -> Optional[str]:
    """Remove this launch's transport directory when it is the kit-owned default (``<workdir>/_xfer`` or the tempdir one) — called by the
    predicting process at exit, after the helper was told to stop; a caller-named ``BOLTZ_OPT_XFER_DIR`` is left in place (only stray
    ``st_<pid>_*`` files of this process's own launches would be in it, each unlinked by its reader). Returns the directory removed, or None."""
    d = _STATE["dir"]
    if not d or (os.environ.get(ENV_DIR) or "").strip():
        return None
    if os.path.basename(d) != SUBDIR and not os.path.basename(d).startswith(TMP_PREFIX):
        return None
    shutil.rmtree(d, ignore_errors=True)
    return d


def reset_for_tests() -> None:
    """Unregister (torch's reducer back on ``ForkingPickler``) and clear the process state."""
    with _LOCK:
        orig = _STATE["orig"]
        if _STATE["installed"] and orig is not None:
            try:
                import torch
                from multiprocessing.reduction import ForkingPickler
                ForkingPickler.register(torch.UntypedStorage, orig)
            except Exception:  # noqa: BLE001
                pass
        _STATE.update(installed=False, word=None, min_bytes=MIN_BYTES_DEFAULT, dir=None, orig=None, count=0, files_written=0, bytes_written=0, write_s=0.0,
                      files_read=0, bytes_read=0, read_s=0.0, delegated=0, largest_file_bytes=0, errors=[])
