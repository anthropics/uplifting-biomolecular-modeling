"""Side processes for the non-forward levers (``prefetch``: the persistent featurizer; ``awrite``: the asynchronous writer).

One rule decides where they are born: a side process is FORKED from the fold process at ``BaseInferenceEngine._construct_pipeline``
— the engine has built its Transform pipeline and has NOT yet built its trainer / model (``foundry/inference_engines/base.py``:
``initialize()`` calls ``_construct_pipeline`` before ``_construct_trainer``), so no CUDA context exists in the process yet
(``torch.cuda.is_initialized()`` is recorded at the fork and printed: ``forked_before_cuda=<True|False>``). A fork (not a spawn) is the
point: the child inherits the parent's hash secret, its imported modules, the kit's wrappers and the constructed pipeline object as
they are — the statements it runs later are the parent's statements on the parent's objects — and it never initialises CUDA (it holds
no device memory and issues no kernel). The fold process itself never forks again per item for these levers.

Transport is two one-way ``multiprocessing`` pipes per side process (parent→child, child→parent); messages are pickled by this
module's own pickler: CPU tensors and C-contiguous ndarrays of 64 KiB or more travel as named POSIX shared-memory blocks (no
file-descriptor passing and no sockets — torch.multiprocessing's storage sharing needs a unix-socket listener that restricted sandboxes
refuse), smaller leaves by value; without /dev/shm every leaf travels by value (``transport=bytes``, named). Tensor views keep their
storage offset / size / stride exactly. A child may run its
loop with a reader thread (the writer does: its inbox drains while it writes, so the parent's hand-off never waits on a write in
progress). The child lowers its own priority (``os.nice``) so the fold process's launch thread wins the scheduler when the box has few
cores; it ignores SIGINT (the parent owns the terminal) and ends when its pipe closes or the parent is gone. The parent stops every side
process at exit (``atexit``, registered after ``multiprocessing``'s own handler so it runs before it).
"""
from __future__ import annotations

import atexit
import os
import pickle
import signal
import sys
import threading
import time
import traceback
from typing import Any, Callable, Optional

NICE = 10                                                   # the side process's niceness increment (its pools / subprocesses inherit it)
STOP = ("stop",)
_LIVE: list = []                                            # every Sidecar started in this process (atexit stops them; a child closes the copies it inherits)


class SidecarError(RuntimeError):
    """The side process is gone or did not answer (the lever falls back to the stock statement by name)."""


# ---------------------------------------------------------------- transport: pickling with array payloads through POSIX shared memory
# No file-descriptor passing, no sockets (torch.multiprocessing's tensor reductions share storages through a unix-socket listener —
# resource_sharer — which socket-restricted sandboxes refuse; this transport never uses it): a CPU tensor / C-contiguous ndarray of
# SHM_MIN bytes or more travels as one named POSIX shared-memory block (one memcpy by the sender; the receiver maps it, wraps it without a
# copy and unlinks the name at once), everything else — and every array when shared memory is unavailable in this sandbox (probed once,
# ``transport=bytes`` then, by name) — by value inside the pickle. Tensor identity is kept exactly: dtype, storage bytes, storage offset,
# size, stride (a view arrives as the same view of an equal storage); ndarrays keep dtype / shape / C order.
SHM_MIN = 1 << 16
TRANSPORT: dict = {"mode": None, "reason": None, "shm_blocks": 0, "shm_bytes": 0, "inline_bytes": 0}
_KEEP: list = []                                            # received blocks whose buffers back live arrays (swept when their exports are gone)


def _shm_mode() -> str:
    """'shm' when a POSIX shared-memory block can be created here (probed once), else 'bytes' with the reason."""
    if TRANSPORT["mode"] is None:
        try:
            from multiprocessing import shared_memory
            blk = shared_memory.SharedMemory(create=True, size=4096)
            blk.close(); blk.unlink()                       # create registered the name with the resource tracker once, unlink() unregisters it once (no _untrack here: a second unregister is a KeyError traceback in the tracker process)
            TRANSPORT["mode"] = "shm"
        except Exception as e:                              # noqa: BLE001 — no /dev/shm (or refused): arrays travel inside the pickle
            TRANSPORT["mode"], TRANSPORT["reason"] = "bytes", f"{type(e).__name__}: {e}"
    return TRANSPORT["mode"]


def _untrack(name: str) -> None:
    """Keep Python's resource tracker out of it: sender and receiver manage the block's lifetime explicitly (the receiver unlinks)."""
    try:
        from multiprocessing import resource_tracker
        resource_tracker.unregister("/" + name if not name.startswith("/") else name, "shared_memory")
    except Exception:                                       # noqa: BLE001
        pass


def _put(mv) -> tuple:
    """A contiguous byte view → ("shm", name, nbytes) or ("bytes", bytes)."""
    n = mv.nbytes if hasattr(mv, "nbytes") else len(mv)
    if n >= SHM_MIN and _shm_mode() == "shm":
        from multiprocessing import shared_memory
        try:
            blk = shared_memory.SharedMemory(create=True, size=n)
            _untrack(blk.name)
            blk.buf[:n] = mv
            name = blk.name
            blk.close()
            TRANSPORT["shm_blocks"] += 1; TRANSPORT["shm_bytes"] += n
            return ("shm", name, n)
        except Exception as e:                              # noqa: BLE001 — shm exhausted or refused now: this payload inline
            TRANSPORT["reason"] = f"{type(e).__name__}: {e}"
    TRANSPORT["inline_bytes"] += n
    return ("bytes", bytes(mv), n)


def _get(payload):
    """("shm", name, n) → a uint8 ndarray over the mapped block (name unlinked at once; the mapping lives while arrays reference it);
    ("bytes", b, n) → a uint8 ndarray over a private copy."""
    import numpy as np
    if payload[0] == "shm":
        from multiprocessing import shared_memory
        blk = shared_memory.SharedMemory(name=payload[1])
        try:
            blk.unlink()                                    # the name goes now (the mapping lives on); unlink() also drops the tracker's record once
        except FileNotFoundError:
            pass
        _KEEP.append(blk)
        return np.ndarray((payload[2],), dtype=np.uint8, buffer=blk.buf)
    return np.frombuffer(bytearray(payload[1]), dtype=np.uint8) if payload[2] else np.zeros((0,), dtype=np.uint8)


def sweep() -> int:
    """Unmap received blocks no array references any more (a block still exported raises BufferError on close: kept)."""
    kept, n = [], 0
    for blk in _KEEP:
        try:
            blk.close(); n += 1
        except BufferError:
            kept.append(blk)
        except Exception:                                   # noqa: BLE001
            pass
    _KEEP[:] = kept
    return n


def _rebuild_ndarray(payload, dtype, shape):
    import numpy as np
    raw = _get(payload)
    return raw.view(np.dtype(dtype)).reshape(shape)


def _rebuild_tensor(payload, dtype, offset, size, stride, requires_grad):
    import torch
    raw = _get(payload)
    if not raw.flags.writeable:
        raw = raw.copy()
    base = torch.from_numpy(raw)                            # uint8 over the mapped block (zero-copy); the storage keeps the ndarray (and so the block) alive
    t = torch.empty((0,), dtype=getattr(torch, dtype)).set_(base.untyped_storage(), offset, tuple(size), tuple(stride))
    if requires_grad:
        t.requires_grad_(True)
    return t


def _reduce_tensor(t):
    import torch
    st = t.untyped_storage()
    n = st.nbytes()
    u8 = torch.empty((0,), dtype=torch.uint8).set_(st, 0, (n,), (1,))
    payload = _put(memoryview(u8.numpy())) if n else ("bytes", b"", 0)
    return (_rebuild_tensor, (payload, str(t.dtype).replace("torch.", ""), t.storage_offset(), tuple(t.size()), tuple(t.stride()), False))


def _reduce_ndarray(a):
    payload = _put(memoryview(a).cast("B"))
    return (_rebuild_ndarray, (payload, a.dtype.str, a.shape))


class _Pickler(pickle.Pickler):
    """pickle with the two array reductions above (reducer_override); protocol 5."""

    def __init__(self, f):
        super().__init__(f, protocol=5)
        self._torch = sys.modules.get("torch")
        import numpy as np
        self._np = np

    def reducer_override(self, obj):                        # noqa: D401 — pickle's hook
        np = self._np
        if isinstance(obj, np.ndarray):
            if obj.dtype.hasobject or not obj.flags.c_contiguous or obj.nbytes < SHM_MIN:
                return NotImplemented                       # small / object / strided arrays: numpy's own pickling (by value)
            return _reduce_ndarray(obj)
        t = self._torch
        if t is not None and isinstance(obj, t.Tensor):
            if type(obj) not in (t.Tensor, t.nn.Parameter) or obj.device.type != "cpu" or obj.is_sparse or obj.is_quantized:
                return NotImplemented                       # subclasses / non-CPU / sparse: torch's own by-value pickling
            return _reduce_tensor(obj.detach())
        return NotImplemented


def dumps(obj) -> bytes:
    import io
    f = io.BytesIO()
    _Pickler(f).dump(obj)
    return f.getvalue()


def loads(buf: bytes):
    return pickle.loads(buf)


def transport_word() -> str:
    m = _shm_mode()
    return m if m == "shm" else f"bytes({TRANSPORT['reason']})"


def send(tx, obj) -> int:
    """A child's out-of-turn reply (e.g. an answer sent before more work): the same pickling the loop's replies use."""
    buf = dumps(obj)
    tx.send_bytes(buf)
    return len(buf)


def cuda_initialized() -> Optional[bool]:
    t = sys.modules.get("torch")
    if t is None:
        return False
    try:
        return bool(t.cuda.is_initialized())
    except Exception:                                       # noqa: BLE001
        return None


def _child_main(rx, tx, parent_ends: tuple, name: str, target: Callable, args: tuple, nice: int) -> None:
    for c in parent_ends:
        try:
            c.close()
        except Exception:                                   # noqa: BLE001
            pass
    for sc in list(_LIVE):                                  # the parent's ends of OTHER side processes, inherited by this fork: not ours to hold
        for c in (sc.wconn, sc.rconn):
            try:
                c.close()
            except Exception:                               # noqa: BLE001
                pass
    del _LIVE[:]
    try:
        os.nice(int(nice))
    except Exception:                                       # noqa: BLE001 — a platform without nice: the child runs at the parent's priority
        pass
    try:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
    except Exception:                                       # noqa: BLE001
        pass
    try:
        target(rx, tx, *args)
    except (EOFError, BrokenPipeError, KeyboardInterrupt):
        pass
    except BaseException:                                   # noqa: BLE001 — the child's last word; the parent sees the pipe close and names it
        sys.stderr.write(f"[rosettafold3-opt] SIDECAR {name} pid={os.getpid()} died:\n{traceback.format_exc()}\n")
        sys.stderr.flush()
    finally:
        for c in (rx, tx):
            try:
                c.close()
            except Exception:                               # noqa: BLE001
                pass


class Sidecar:
    """One forked side process serving ``target(rx, tx, *args)``; ``send`` / ``recv`` on the parent's ends, byte counts recorded."""

    def __init__(self, name: str, target: Callable, args: tuple = (), nice: int = NICE):
        self.name, self.target, self.args, self.nice = name, target, tuple(args), int(nice)
        self.proc = None
        self.wconn = None                                   # parent -> child
        self.rconn = None                                   # child -> parent
        self.forked_before_cuda: Optional[bool] = None
        self.threads_at_fork: Optional[int] = None
        self.bytes_out = 0
        self.bytes_in = 0
        self.stopped = False

    def start(self) -> "Sidecar":
        import multiprocessing as mp
        ctx = mp.get_context("fork")
        child_rx, parent_tx = ctx.Pipe(duplex=False)       # (reader, writer)
        parent_rx, child_tx = ctx.Pipe(duplex=False)
        self.forked_before_cuda = (cuda_initialized() is False)
        self.threads_at_fork = threading.active_count()
        self.proc = ctx.Process(target=_child_main, args=(child_rx, child_tx, (parent_tx, parent_rx), self.name, self.target, self.args, self.nice),
                                name=f"rosettafold3_opt-{self.name}", daemon=False)   # not daemonic: the featurizer's own pools must be allowed
        self.proc.start()
        child_rx.close()
        child_tx.close()
        self.wconn, self.rconn = parent_tx, parent_rx
        _LIVE.append(self)
        if len(_LIVE) == 1:
            atexit.register(stop_all)
        return self

    @property
    def pid(self) -> Optional[int]:
        return getattr(self.proc, "pid", None)

    def alive(self) -> bool:
        return self.proc is not None and not self.stopped and self.proc.is_alive()

    def send(self, msg) -> int:
        buf = dumps(msg)
        try:
            self.wconn.send_bytes(buf)
        except (OSError, ValueError, EOFError) as e:
            raise SidecarError(f"{self.name}: send failed ({type(e).__name__}: {e})") from e
        self.bytes_out += len(buf)
        return len(buf)

    def recv(self, timeout: Optional[float] = None, tick: float = 5.0):
        """``(message, n_bytes)``; raises SidecarError when the child is gone (or ``timeout`` seconds pass without a message)."""
        t0 = time.perf_counter()
        while True:
            try:
                ready = self.rconn.poll(tick if timeout is None else min(tick, max(0.0, timeout - (time.perf_counter() - t0))))
            except (OSError, ValueError, EOFError) as e:
                raise SidecarError(f"{self.name}: pipe closed ({type(e).__name__})") from e
            if ready:
                sweep()                                     # blocks of earlier messages no array references any more: unmapped
                try:
                    buf = self.rconn.recv_bytes()
                except (OSError, EOFError) as e:
                    raise SidecarError(f"{self.name}: pipe closed while reading ({type(e).__name__})") from e
                self.bytes_in += len(buf)
                return loads(buf), len(buf)
            if not self.alive():
                raise SidecarError(f"{self.name}: side process gone (exitcode={getattr(self.proc, 'exitcode', None)})")
            if timeout is not None and time.perf_counter() - t0 >= timeout:
                raise SidecarError(f"{self.name}: no answer in {timeout:.0f} s")

    def poll(self) -> bool:
        try:
            return bool(self.rconn.poll(0))
        except (OSError, ValueError, EOFError):
            return False

    def stop(self, timeout: float = 10.0) -> Optional[int]:
        """Ask the child to end, wait, terminate if it lingers; returns its exit code."""
        if self.proc is None or self.stopped:
            return getattr(self.proc, "exitcode", None)
        self.stopped = True
        try:
            self.wconn.send_bytes(dumps(STOP))
        except Exception:                                   # noqa: BLE001
            pass
        self.proc.join(timeout)
        if self.proc.is_alive():
            self.proc.terminate()
            self.proc.join(5.0)
        for c in (self.wconn, self.rconn):
            try:
                c.close()
            except Exception:                               # noqa: BLE001
                pass
        return self.proc.exitcode


def stop_all() -> None:
    for sc in list(_LIVE):
        try:
            sc.stop()
        except Exception:                                   # noqa: BLE001
            pass


def parent_gone() -> bool:
    """In a child: True once the fold process has exited (re-parented to init)."""
    return os.getppid() == 1


def serve_loop(rx, tx, handle: Callable[[tuple], Any], idle: Optional[Callable[[], bool]] = None, tick: float = 2.0,
               threaded: bool = False) -> None:
    """A child's message loop: ``handle(msg)`` per message in arrival order (its return value, when not None, goes back to the parent);
    ``idle()`` runs whenever no message is waiting (speculative work; True = it did something). ``threaded``: a reader thread drains the
    inbox into a queue while ``handle`` runs (a sender never waits on work in progress). Ends on ``stop``, a closed pipe or a dead parent."""
    send_lock = threading.Lock()

    def reply(out):
        if out is not None:
            with send_lock:
                tx.send_bytes(dumps(out))

    if not threaded:
        while True:
            if idle is not None and not rx.poll(0):
                if idle():
                    continue
            if not rx.poll(tick):
                if parent_gone():
                    return
                continue
            try:
                buf = rx.recv_bytes()
            except (EOFError, OSError):
                return
            msg = loads(buf)
            if not isinstance(msg, tuple) or not msg:
                continue
            if msg[0] == STOP[0]:
                return
            reply(handle(msg))
        return

    import queue
    q: "queue.Queue" = queue.Queue()
    END = object()

    def reader():
        while True:
            try:
                if not rx.poll(tick):
                    if parent_gone():
                        break
                    continue
                buf = rx.recv_bytes()
            except (EOFError, OSError):
                break
            try:
                msg = loads(buf)
            except Exception:                               # noqa: BLE001
                continue
            if isinstance(msg, tuple) and msg and msg[0] == STOP[0]:
                break
            q.put(msg)
        q.put(END)

    th = threading.Thread(target=reader, name="rosettafold3_opt-sidecar-reader", daemon=True)
    th.start()
    while True:
        try:
            msg = q.get(timeout=tick)
        except queue.Empty:
            if idle is not None:
                idle()
            continue
        if msg is END:
            return
        if not isinstance(msg, tuple) or not msg:
            continue
        reply(handle(msg))
