"""The launcher: a kit process spawns its own P workers, one per GPU of the node, joined into one process group; users never type
``torchrun``. Two forms share one refusal / environment / fail-fast layer (and ONE ``PYTHONHASHSEED`` for every rank of a launch —
:mod:`.rankdata`: inherited from the launching process when it names a seed, else ``0``):

    run_sharded(n_gpu, entry, *args, mode=, **kwargs)        P == 1: ``return entry(*args, **kwargs)`` in THIS process — no group, no worker,
                                                            no environment change (the engine's single-GPU path, byte for byte).
                                                            P > 1: P workers by ``torch.multiprocessing`` (start method ``spawn``); worker r
                                                            gets :func:`rank_env` (the NAMESPACED ``ROWPAIR_{RANK,WORLD,LOCAL_RANK,ADDR,PORT}`` —
                                                            torchrun's ``RANK/WORLD_SIZE/...`` are NOT exported unless ``torchrun_names=True``,
                                                            because frameworks such as pytorch-lightning enter DDP and shard the QUERY set when
                                                            they see them), binds ``cuda:r`` (``torch.cuda.set_device``), initialises the
                                                            group (:func:`opt_core.mem.rowpair.dist.init_from_env`, NCCL when CUDA is used else
                                                            gloo, bounded ``nccl_timeout_s``), runs ``entry(*args, **kwargs)``, destroys the group;
                                                            rank 0's return value (picklable) is returned to the caller. ``entry`` must be
                                                            importable by the workers (a module-level function), as ``spawn`` requires.
    run_rank_processes(n_gpu, argv, mode=, env=, log_dir=)  one COMMAND LINE per rank (the kit's own model entry point, e.g. ``python -m
                                                            <kit> pred ...``): rank r runs ``argv`` (or ``argv_of(r)``) under :func:`rank_env`;
                                                            transcripts go to ``<log_dir>/rank<r>.log``, rank 0's also to this process's
                                                            stderr (``on_line``); returns the per-rank records of :func:`opt_core.process.run_logged`.

Fail-fast (both forms): the parent polls every rank; the FIRST rank that exits non-zero (or raises) is the named event — every other
rank is killed (SIGTERM, then SIGKILL after :data:`KILL_GRACE_S`), no worker outlives the call, and the caller gets
:class:`RankFailed` (``rank``, ``exitcode``, the failing rank's traceback / log tail) — the kit prints
``evidence.rank_failed_line`` and exits non-zero. Under ``run_rank_processes`` the kill is immediate for a rank that died on a signal;
a rank that EXITED with a non-zero status leaves its peers ``fail_grace_s`` (default :data:`FAIL_GRACE_S` = 60 s) first — when an item
failed on every rank the peers reach the same exit on their own and rank 0 writes its last outputs; a peer stuck in a collective is
killed after the grace — announced by one ``[<tag>] RANKEXIT rank=<r> exitcode=<c> grace_s=<g>`` line. Timeouts (same names in both
forms): ``nccl_timeout_s`` = the group's collective timeout; ``run_timeout_s`` = the whole call's wall clock; ``finish_grace_s`` = the time
every other rank has once the first rank finished — exceeding either is the event ``rank_timeout`` (``rank_failed`` naming the exited
rank when one had exited non-zero). Per-rank records (``launch.last_run()``: rank, ok, exitcode, wall_s, log, error) exist
after every P > 1 call, success or failure. NCCL's own collective timeout is ``nccl_timeout_s`` (``init_process_group(timeout=)``), so a
peer that died mid-collective surfaces as an error on the survivors instead of a hang.

Output ownership: rank 0 owns the run's outputs (it writes the engine's files; ranks > 0 compute and write nothing the user reads —
:func:`is_output_rank`). Stdout discipline: under ``run_sharded`` ranks > 0 have ``sys.stdout`` / ``sys.stderr`` redirected to
``<log_dir>/rank<r>.log`` before ``entry`` runs, so the arm's captured stream carries rank 0's evidence lines once; the parent names the
log directory on its ``ROWPAIR launch`` line and prints the failing rank's log tail on failure (never nothing).

Device binding is card-generic: worker r uses the r-th VISIBLE device (``CUDA_VISIBLE_DEVICES`` order); nothing assumes a device
name, count or memory size. ``devices=[...]`` picks explicit visible indices (length P).
"""
from __future__ import annotations

import os
import signal
import socket
import sys
import tempfile
import time
import traceback
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

from . import MEMORY_MODE, RowpairRefused, SHARDING, check_n_gpu, refuse_unless_big, refuse_unless_visible
from .rankdata import HASHSEED_ENV, hash_seed, rankenv_line, ranks_env


def _tag(env=None) -> str:
    """The line tag of a launch / rank: ``ROWPAIR_TAG`` in ``env`` (default ``os.environ``), else ``rowpair``."""
    return (os.environ if env is None else env).get(ENV_TAG) or "rowpair"


def rankexit_line(tag: str, rank: int, exitcode: int, grace_s: Optional[float]) -> str:
    """The launcher's line when a rank of :func:`run_rank_processes` EXITED with a non-zero status while peers still run:
    ``[<tag>] RANKEXIT rank=<r> exitcode=<c> grace_s=<g|none>`` — the peers run on for ``grace_s`` (then are torn down)."""
    g = "none" if grace_s is None else f"{float(grace_s):g}"
    return f"[{tag}] RANKEXIT rank={int(rank)} exitcode={int(exitcode)} grace_s={g}"

__all__ = ["ENV_RANK", "ENV_WORLD", "ENV_LOCAL_RANK", "ENV_ADDR", "ENV_PORT", "ENV_LOG_DIR", "KILL_GRACE_S", "FAIL_GRACE_S", "RankFailed", "free_port",
           "rank_env", "new_store_path", "rank", "world_size", "local_rank", "rendezvous", "init_method", "is_output_rank", "device_census", "last_run", "run_sharded",
           "run_rank_processes", "rankexit_line", "ENV_NCCL_TIMEOUT", "ENV_STORE"]

ENV_RANK = "ROWPAIR_RANK"                # set in every worker (both forms); absent in a P == 1 run
ENV_WORLD = "ROWPAIR_WORLD"
ENV_LOCAL_RANK = "ROWPAIR_LOCAL_RANK"    # the visible-device index this rank binds
ENV_ADDR = "ROWPAIR_ADDR"                # rendezvous (127.0.0.1: one node)
ENV_PORT = "ROWPAIR_PORT"
ENV_LOG_DIR = "ROWPAIR_LOG_DIR"
ENV_TAG = "ROWPAIR_TAG"                  # the kit's log tag for the per-rank SCHEDULE line (default "rowpair")
ENV_NCCL_TIMEOUT = "ROWPAIR_NCCL_TIMEOUT_S"   # the collective timeout a rank process's init_from_env applies (argv form)
ENV_STORE = "ROWPAIR_STORE"              # path of the file:// rendezvous store of this run (one node: no port to race for)
TORCHRUN_NAMES = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
KILL_GRACE_S = 10.0                      # SIGTERM -> SIGKILL grace when tearing ranks down
FAIL_GRACE_S = 60.0                      # run_rank_processes: how long the peers of a rank that EXITED non-zero run on before they are torn down
POLL_S = 0.5
MIXED_MEM_TOL = 0.10                     # devices of one run may differ in total memory by at most this fraction (else refused by name)


class RankFailed(RuntimeError):
    """A rank ended the sharded run: ``event`` is ``rank_failed`` (non-zero exit or an exception in ``entry``), ``rank_timeout`` or
    ``rank_killed``; ``rank`` the first offending rank, ``exitcode`` its exit code (None when it raised inside ``entry`` and was joined),
    ``detail`` the traceback text or the log tail. The launcher has already torn every other rank down when this is raised."""

    def __init__(self, event: str, rank: int, exitcode: Optional[int], detail: str = "", log: Optional[str] = None):
        self.event, self.rank, self.exitcode, self.detail, self.log = str(event), int(rank), exitcode, str(detail or ""), log
        super().__init__(f"rowpair {self.event}: rank={self.rank} exitcode={self.exitcode}"
                         + (f" log={self.log}" if log else "") + (f"\n{self.detail}" if self.detail else ""))


# ----------------------------------------------------------------------------------------------------------------- rank equality
def _env_first(names, default: int) -> int:
    for name in names:
        v = os.environ.get(name, "")
        if v.strip():
            return int(v)
    return int(default)


def rank() -> int:
    """This process's rank: :data:`ENV_RANK`, else 0 (a P == 1 run). torchrun's ``RANK`` is never read: a foreign ``RANK`` in a cluster
    job's environment is not a rowpair world."""
    return _env_first((ENV_RANK,), 0)


def world_size() -> int:
    """This process's world size: :data:`ENV_WORLD`, else 1."""
    return _env_first((ENV_WORLD,), 1)


def local_rank() -> int:
    """The visible-device index this rank binds: :data:`ENV_LOCAL_RANK`, else :func:`rank`."""
    return _env_first((ENV_LOCAL_RANK,), rank())


def rendezvous() -> tuple:
    """``(addr, port)`` of the group: ``ROWPAIR_ADDR`` / ``ROWPAIR_PORT`` (default ``("127.0.0.1", 29500)``); used when no ``ROWPAIR_STORE``."""
    addr = os.environ.get(ENV_ADDR) or "127.0.0.1"
    port = _env_first((ENV_PORT,), 29500)
    return addr, port


def init_method() -> str:
    """The rendezvous of the group: ``file://$ROWPAIR_STORE`` when the launcher set a store path (one node: nothing listens on a port, so
    back-to-back runs cannot collide on one), else ``tcp://addr:port`` from :func:`rendezvous`."""
    store = os.environ.get(ENV_STORE, "").strip()
    if store:
        return "file://" + os.path.abspath(store)
    addr, port = rendezvous()
    return f"tcp://{addr}:{int(port)}"


def is_output_rank() -> bool:
    """True on the rank that owns the run's outputs (rank 0) and in every P == 1 run."""
    return rank() == 0


def new_store_path() -> str:
    """A fresh node-local path for this run's ``file://`` rendezvous store (``tempfile``'s directory — never the log dir, which may be a
    network or FUSE mount where the store's file locking is unreliable). The file must not exist yet; the directory is created here."""
    d = tempfile.mkdtemp(prefix="rowpair_store_")
    return os.path.abspath(os.path.join(d, "c10d_store"))


def free_port(host: str = "127.0.0.1") -> int:
    """A TCP port the kernel hands out as free right now (bind to 0, read, close) — the group's ``MASTER_PORT``."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.bind((host, 0))
        return int(s.getsockname()[1])
    finally:
        s.close()


def rank_env(r: int, world: int, port: int, *, addr: str = "127.0.0.1", log_dir: Optional[str] = None,
             base: Optional[Mapping[str, str]] = None, isolate_device: Optional[int] = None, local: Optional[int] = None,
             torchrun_names: bool = False, store: Optional[str] = None, cpu_threads=None) -> Dict[str, str]:
    """The environment of rank ``r``: ``base`` (default: a copy of ``os.environ``) plus the NAMESPACED ``ROWPAIR_RANK / ROWPAIR_WORLD /
    ROWPAIR_LOCAL_RANK / ROWPAIR_ADDR / ROWPAIR_PORT`` (+ ``ROWPAIR_LOG_DIR``). ``torchrun_names=True`` additionally exports ``RANK WORLD_SIZE
    LOCAL_RANK LOCAL_WORLD_SIZE MASTER_ADDR MASTER_PORT`` for an engine that reads them — OFF by default: frameworks that see them (e.g.
    pytorch-lightning) enter DDP and shard the query set across ranks. ``isolate_device=i``: ``CUDA_VISIBLE_DEVICES=<i-th visible entry>`` and
    local rank 0 (one visible device per rank process — the form for an engine entry point that always uses ``cuda:0``); default: every device
    stays visible and the local rank is ``local`` (default ``r``). ``TORCH_NCCL_ASYNC_ERROR_HANDLING=1`` is set when absent (a peer's death
    surfaces as an error on the survivors). Every rank of the launch starts with ONE ``PYTHONHASHSEED`` (:func:`rankdata.ranks_env`: the
    base's own value when it names a seed, else ``0``), so hash-ordered featurisation cannot differ between the rank processes.
    ``cpu_threads`` (the per-rank CPU-thread cap): ``None`` = the lever ``ROWPAIR_RANK_THREADS`` of ``base`` (else of this process)
    decides, and an unset lever changes nothing (today's environment, byte for byte); ``inherit`` | ``auto`` (``cpus // world``) | ``<n>``
    lay ``OMP_NUM_THREADS = MKL_NUM_THREADS = OPENBLAS_NUM_THREADS = n`` and the resolved ``ROWPAIR_RANK_THREADS = n`` over the rank's
    environment (:func:`dist.rank_thread_env`; the :func:`run_sharded` worker also applies ``torch.set_num_threads(n)`` and names
    ``rank_threads=n`` in its schedule census)."""
    env, _hashseed_word = ranks_env(os.environ if base is None else base, world)
    loc = int(r if local is None else local)
    if isolate_device is not None:
        vis = [d for d in env.get("CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")).split(",") if d.strip()]
        env["CUDA_VISIBLE_DEVICES"] = vis[int(isolate_device)] if vis else str(int(isolate_device))
        loc = 0
    env.update({ENV_RANK: str(r), ENV_WORLD: str(world), ENV_LOCAL_RANK: str(loc), ENV_ADDR: str(addr), ENV_PORT: str(port)})
    if store:
        env[ENV_STORE] = str(store)                                          # file:// rendezvous (a fresh node-local path per run)
    if torchrun_names:
        env.update({"RANK": str(r), "WORLD_SIZE": str(world), "LOCAL_RANK": str(loc), "LOCAL_WORLD_SIZE": str(world),
                    "MASTER_ADDR": str(addr), "MASTER_PORT": str(port)})
    env.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")
    if log_dir:
        env[ENV_LOG_DIR] = str(log_dir)
    from .dist import ENV_RANK_THREADS, rank_thread_env
    spec = cpu_threads
    if spec is None:                                                     # the launching environment's lever (``base``, else this process); unset = nothing added
        spec = (os.environ if base is None else base).get(ENV_RANK_THREADS, os.environ.get(ENV_RANK_THREADS))
    delta = rank_thread_env(spec, world)
    if delta:
        env.update(delta)
    elif cpu_threads is not None and ENV_RANK_THREADS in env:
        env[ENV_RANK_THREADS] = "inherit"                                # an explicit inherit wins over a lever word the base carries
    return env


def device_census(devices: Sequence[int], allow_mixed: bool = False) -> List[dict]:
    """``[{index, name, capability, total_gib}]`` of the visible devices a run will bind (``torch.cuda.get_device_properties``; no CUDA
    context on them is created by the parent beyond the property query). Devices of one run must be one class: differing compute capability,
    or total memory differing by more than :data:`MIXED_MEM_TOL`, is refused by name unless ``allow_mixed`` (then the caller sizes to the
    smallest card and says so in its line)."""
    from ._torch import torch
    out = []
    for i in devices:
        p = torch.cuda.get_device_properties(int(i))
        out.append({"index": int(i), "name": str(p.name), "capability": f"sm{p.major}{p.minor}", "total_gib": round(p.total_memory / 2 ** 30, 1)})
    if out and not allow_mixed:
        caps = {d["capability"] for d in out}
        mems = [d["total_gib"] for d in out]
        if len(caps) > 1 or (max(mems) - min(mems)) > MIXED_MEM_TOL * max(mems):
            desc = ",".join(f"{d['index']}:{d['name'].replace(' ', '_')}:{d['capability']}:{d['total_gib']}GiB" for d in out)
            raise RowpairRefused(f"refused: heterogeneous devices for one n_gpu run ({desc})")
    return out


def _log_dir(log_dir: Optional[str]) -> str:
    d = log_dir or os.environ.get(ENV_LOG_DIR) or tempfile.mkdtemp(prefix="rowpair_ranks_")
    os.makedirs(d, exist_ok=True)
    return d


def _tail(path: Optional[str], lines: int = 40) -> str:
    if not path or not os.path.isfile(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return "".join(fh.readlines()[-lines:])
    except OSError:
        return ""


# ----------------------------------------------------------------------------------------------------------------- in-process form
def _exit_with_parent(ppid: int, poll_s: float = 1.0) -> None:
    """A daemon thread that ends this worker (``os._exit(70)``) when its parent is gone — a killed launcher never leaves ranks behind."""
    import threading

    def watch():
        while True:
            if os.getppid() != ppid:
                os._exit(70)
            time.sleep(poll_s)

    threading.Thread(target=watch, name="rowpair-parent-watch", daemon=True).start()


def _pdeathsig():
    """``preexec_fn`` for rank subprocesses on Linux: SIGTERM when the launcher dies (no orphans); a no-op elsewhere."""
    try:
        import ctypes
        import signal
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, int(signal.SIGTERM), 0, 0, 0)
    except Exception:  # noqa: BLE001
        pass


class _TensorBytes:
    """A tensor serialised BY VALUE (``torch.save`` bytes) for the result queue: no shared-memory / file-descriptor storage handle whose
    lifetime depends on the sending process."""
    __slots__ = ("b",)

    def __init__(self, b: bytes):
        self.b = b


def _hostify(obj, lever: str = "rowpair"):
    """Rank 0's return value crosses a process boundary: tensors travel BY VALUE (moved to CPU and serialised into :class:`_TensorBytes`;
    an fd-shared CPU storage or a CUDA IPC handle would point into a process about to exit); containers are walked (dict / list / tuple);
    other objects must pickle. :func:`_unhostify` rebuilds the tensors in the parent."""
    from ._torch import real_torch
    torch = real_torch()
    if isinstance(obj, torch.Tensor):
        import io
        buf = io.BytesIO()
        torch.save(obj.detach().cpu().contiguous(), buf)
        return _TensorBytes(buf.getvalue())
    if isinstance(obj, dict):
        return {k: _hostify(v, lever) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_hostify(v, lever) for v in obj)
    return obj


def _unhostify(obj):
    """Inverse of :func:`_hostify` in the parent: :class:`_TensorBytes` -> CPU tensor; containers walked."""
    if isinstance(obj, _TensorBytes):
        import io
        from ._torch import real_torch
        return real_torch().load(io.BytesIO(obj.b), map_location="cpu", weights_only=True)
    if isinstance(obj, dict):
        return {k: _unhostify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_unhostify(v) for v in obj)
    return obj


def _worker(r: int, world: int, env: Dict[str, str], entry: Callable, args: tuple, kwargs: dict, backend: Optional[str],
            nccl_timeout_s: float, device_index: int, result_q, log_dir: str) -> None:
    """Body of rank ``r`` under ``run_sharded`` (runs in a fresh ``spawn`` interpreter)."""
    os.environ.update(env)
    _exit_with_parent(os.getppid())
    from .dist import ENV_RANK_THREADS
    if str(env.get(ENV_RANK_THREADS, "")).strip():                    # the per-rank CPU-thread cap rank_env resolved; absent = untouched
        from .dist import apply_rank_threads
        apply_rank_threads(env[ENV_RANK_THREADS], local_world=world)
    log_path = os.path.join(log_dir, f"rank{r}.log")
    if r != 0:                                                        # ranks > 0: EVERYTHING (Python and C-level fd 1/2: NCCL WARN, CUDA
        fh = open(log_path, "a", buffering=1, encoding="utf-8")     # driver messages, faulthandler dumps) goes to the per-rank log
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(fh.fileno(), 1)
        os.dup2(fh.fileno(), 2)
        sys.stdout = fh
        sys.stderr = fh
    from .dist import destroy, exit_code, init_from_env
    t0 = time.monotonic()
    try:
        init_from_env(backend=backend, timeout_s=nccl_timeout_s, device_index=device_index)
        out = entry(*args, **kwargs)
        from .evidence import memstats_line, schedule_line               # the as-run schedule (+ the allocator census on CUDA) of THIS rank,
        print(schedule_line(_tag(), r, world), flush=True)   # once, after the entry returns
        ms = memstats_line(_tag(), r, world)
        if ms:
            print(ms, flush=True)
        if r == 0:
            result_q.put(("result", 0, (_hostify(out), round(time.monotonic() - t0, 3))))
        else:
            result_q.put(("done", r, round(time.monotonic() - t0, 3)))
        exit_code(0)                                                   # a registered success: the leave below is the library's own, unbounded; the except arm registers 1 and its leave is the bounded one
    except BaseException as exc:  # noqa: BLE001 — every failure is reported to the parent, then re-raised for the exit code
        exit_code(1)
        tb = traceback.format_exc()
        try:
            result_q.put(("error", r, f"{type(exc).__name__}: {exc}\n{tb}"))
        except Exception:  # noqa: BLE001
            pass
        sys.stderr.write(tb)
        sys.stderr.flush()
        raise
    finally:
        try:
            destroy()
        except Exception:  # noqa: BLE001
            pass


_LAST_RUN: List[dict] = []


def last_run() -> List[dict]:
    """The per-rank records of the most recent P > 1 :func:`run_sharded` / :func:`run_rank_processes` call in this process:
    ``[{rank, ok, exitcode, wall_s, log, error}]`` — the kit's manifest states ``ranks_ok=k/P`` from it (every rank ok or failed with a
    named reason). Empty for a P == 1 call (no ranks exist)."""
    return [dict(r) for r in _LAST_RUN]


def run_sharded(n_gpu, entry: Callable[..., Any], *args, mode: Optional[str] = MEMORY_MODE, backend: Optional[str] = None,
                nccl_timeout_s: float = 1800.0, run_timeout_s: Optional[float] = None, finish_grace_s: Optional[float] = 600.0,
                devices: Optional[Sequence[int]] = None, log_dir: Optional[str] = None, visible: Optional[int] = None, cpu_ok: bool = False,
                allow_mixed: bool = False, torchrun_names: bool = False, return_records: bool = False, **kwargs) -> Any:
    """Run ``entry(*args, **kwargs)`` on ``n_gpu`` ranks and return rank 0's value (``(value, records)`` with ``return_records``; the records
    are also :func:`last_run`). ``mode`` is the kit's mode name (``n_gpu > 1`` outside ``big`` is refused by name); ``visible`` overrides the
    visible-GPU probe (tests); ``cpu_ok=True`` allows a P > 1 run without CUDA (gloo; the CPU logic tests) — otherwise fewer visible GPUs than
    P is refused by name and the bound devices must be one class (:func:`device_census`, ``allow_mixed``). ``backend`` None = ``nccl`` with
    CUDA else ``gloo``. Timeouts, all named events: ``nccl_timeout_s`` = the collective timeout of the group; ``run_timeout_s`` = the whole
    call's wall clock (the kit passes its watchdog budget; None = unbounded); ``finish_grace_s`` = once the first rank has finished, every
    other rank must finish within it (a rank hung outside a collective is ``rank_timeout``, not an endless wait). ``torchrun_names``: see
    :func:`rank_env`. Rank 0's value must be host data: tensors in it are moved to CPU (:func:`_hostify`). Raises :class:`RankFailed` after
    tearing every rank down when any rank fails or times out.

    Every worker starts with the launch's ONE ``PYTHONHASHSEED`` (:mod:`.rankdata`: this process's own value when it names a seed, else ``0``):
    it is exported into this process's environment while the workers are started (a ``spawn`` interpreter takes its str-hash seed from the
    environment it starts with) and this process's previous value is restored right after; the line ``[<tag>] RANKENV hashseed=<v>
    source=default|inherited ranks=<P>`` is printed to stderr once.
    """
    P = refuse_unless_big(n_gpu, mode)
    if P == 1:
        del _LAST_RUN[:]
        out = entry(*args, **kwargs)
        return (out, []) if return_records else out
    if not cpu_ok:
        refuse_unless_visible(P, visible)
    if devices is not None:
        devices = [int(d) for d in devices]
        if len(devices) != P:
            raise RowpairRefused(f"devices={devices}: {P} entries required for n_gpu={P}")
    else:
        devices = list(range(P))
    if not cpu_ok:
        device_census(devices, allow_mixed=allow_mixed)
    from ._torch import torch
    mp = torch.multiprocessing.get_context("spawn")
    sys.stderr.write(rankenv_line(_tag(), os.environ, P) + "\n")           # `[<tag>] RANKENV hashseed=<v> source=default|inherited ranks=P`, once per launch
    sys.stderr.flush()
    seed, had_seed = hash_seed(os.environ)[0], os.environ.get(HASHSEED_ENV)
    os.environ[HASHSEED_ENV] = seed                                          # the launch's ONE hash seed (rankdata): a ``spawn`` worker takes it from THIS process's environment at start-up; restored below once every worker has started
    port = free_port()
    store = new_store_path()
    ldir = _log_dir(log_dir)
    result_q = mp.SimpleQueue()
    procs = []
    t0 = time.monotonic()
    try:
        for r in range(P):
            env = rank_env(r, P, port, log_dir=ldir, base={HASHSEED_ENV: seed}, local=devices[r], torchrun_names=torchrun_names, store=store)   # the delta (os.environ is inherited at spawn); it names the seed the parent exported above, so the in-worker update agrees with the interpreter's actual seed
            p = mp.Process(target=_worker, args=(r, P, env, entry, tuple(args), dict(kwargs), backend, float(nccl_timeout_s), devices[r],
                                                 result_q, ldir), daemon=False, name=f"rowpair-rank{r}")
            p.start()
            procs.append(p)
    finally:                                                                 # the parent's environment as it was: the export above exists for the workers' start-up only
        if had_seed is None:
            os.environ.pop(HASHSEED_ENV, None)
        else:
            os.environ[HASHSEED_ENV] = had_seed
    result = None
    have_result = False
    first_error = None
    walls: Dict[int, float] = {}
    first_done_t = None

    def drain():
        nonlocal result, have_result, first_error, first_done_t
        while not result_q.empty():
            kind, r, payload = result_q.get()
            if kind == "result":
                (result, walls[0]), have_result = (_unhostify(payload[0]), payload[1]), True
            elif kind == "done":
                walls[int(r)] = payload
            elif kind == "error" and first_error is None:
                first_error = RankFailed("rank_failed", int(r), None, payload, os.path.join(ldir, f"rank{r}.log"))
            if kind in ("result", "done") and first_done_t is None:
                first_done_t = time.monotonic()

    try:
        while True:
            drain()
            for r, p in enumerate(procs):
                if p.exitcode not in (None, 0) and first_error is None:
                    first_error = RankFailed("rank_failed", r, p.exitcode, _tail(os.path.join(ldir, f"rank{r}.log")),
                                             os.path.join(ldir, f"rank{r}.log"))
            if first_error is not None:
                break
            if all(p.exitcode is not None for p in procs):
                break
            now = time.monotonic()
            live = [r for r, p in enumerate(procs) if p.exitcode is None]
            if run_timeout_s is not None and now - t0 > float(run_timeout_s):
                first_error = RankFailed("rank_timeout", live[0] if live else 0, None, f"run_timeout_s={float(run_timeout_s):g} exceeded",
                                         os.path.join(ldir, f"rank{live[0] if live else 0}.log"))
                break
            if finish_grace_s is not None and first_done_t is not None and now - first_done_t > float(finish_grace_s):
                lag = [r for r in live if r not in walls] or live
                first_error = RankFailed("rank_timeout", lag[0] if lag else 0, None,
                                         f"finish_grace_s={float(finish_grace_s):g} exceeded after the first rank finished",
                                         os.path.join(ldir, f"rank{lag[0] if lag else 0}.log"))
                break
            time.sleep(POLL_S)
    finally:
        _teardown(procs)
    drain()                                                                  # a result posted right before exit
    wall = round(time.monotonic() - t0, 3)
    records = []
    for r, p in enumerate(procs):
        rec = {"rank": r, "ok": p.exitcode == 0 and (r in walls), "exitcode": p.exitcode, "wall_s": walls.get(r, wall),
               "log": os.path.join(ldir, f"rank{r}.log"), "error": None}
        if first_error is not None and first_error.rank == r:
            rec["error"] = first_error.event
        elif p.exitcode != 0:
            rec["error"] = "rank_failed" if p.exitcode not in (None, -signal.SIGTERM, -signal.SIGKILL) else "torn_down"
        records.append(rec)
    _LAST_RUN[:] = records
    if first_error is not None:
        raise first_error
    bad = [(r, p.exitcode) for r, p in enumerate(procs) if p.exitcode != 0]
    if bad:
        r, code = bad[0]
        raise RankFailed("rank_failed", r, code, _tail(os.path.join(ldir, f"rank{r}.log")), os.path.join(ldir, f"rank{r}.log"))
    if not have_result:
        raise RankFailed("rank_failed", 0, procs[0].exitcode, "rank 0 exited without returning a result", os.path.join(ldir, "rank0.log"))
    return (result, records) if return_records else result


def _teardown(procs) -> None:
    """No worker outlives the call: terminate the live ones, kill what ignores SIGTERM after the grace, join all."""
    live = [p for p in procs if p.is_alive()]
    for p in live:
        try:
            p.terminate()
        except Exception:  # noqa: BLE001
            pass
    t0 = time.monotonic()
    for p in procs:
        rem = max(0.0, KILL_GRACE_S - (time.monotonic() - t0))
        p.join(rem)
    for p in procs:
        if p.is_alive():
            try:
                p.kill()
            except Exception:  # noqa: BLE001
                pass
            p.join(KILL_GRACE_S)


# ----------------------------------------------------------------------------------------------------------------- command-line form
def run_rank_processes(n_gpu, argv: Optional[Sequence[str]] = None, *, argv_of: Optional[Callable[[int], Sequence[str]]] = None,
                       mode: Optional[str] = MEMORY_MODE, env: Optional[Mapping[str, str]] = None, log_dir: Optional[str] = None,
                       isolate_devices: bool = False, devices: Optional[Sequence[int]] = None, run_timeout_s: Optional[float] = None,
                       finish_grace_s: Optional[float] = 600.0, fail_grace_s: Optional[float] = FAIL_GRACE_S, nccl_timeout_s: Optional[float] = None,
                       on_line: Optional[Callable[[str], None]] = None, visible: Optional[int] = None, cpu_ok: bool = False,
                       allow_mixed: bool = False, torchrun_names: bool = False, what: str = "rowpair rank") -> List[dict]:
    """Run one command line per rank (module contract) and return ``[{rank, argv, rc, wall_s, log, ok[, error, reason]}...]``.
    ``argv`` (the same for every rank) or ``argv_of(r)``; ``env`` is the base environment (default ``os.environ``) each rank's
    :func:`rank_env` is layered on; ``isolate_devices``: ``CUDA_VISIBLE_DEVICES=<devices[r]>`` per rank (``devices`` default ``range(P)``,
    read as indices into the CURRENT visible list). ``on_line`` receives rank 0's transcript lines as they arrive (the arm's stream); every
    rank's transcript lands in ``<log_dir>/rank<r>.log``. P == 1 runs the one command with no rank environment at all. Teardown: a rank
    that dies on a signal (exit code < 0, or > 128 from a wrapper reporting its child's signal) tears the rest down at once; a rank that
    EXITS with a non-zero status is announced (``[<tag>] RANKEXIT rank=<r> exitcode=<c> grace_s=<g>`` on stderr) and its peers run on for
    ``fail_grace_s`` seconds (``None``: ``finish_grace_s``; both ``None``: until they exit or ``run_timeout_s``) — when the item failed on
    every rank the peers reach the same exit on their own and the output rank writes its last outputs; otherwise they are torn down after
    the grace. The records name every rank's rc (``-9``/``-15`` for the killed), and :class:`RankFailed` naming the first rank that exited
    non-zero (else the killed rank) is raised after all are down."""
    import signal
    import subprocess
    import threading
    P = refuse_unless_big(n_gpu, mode)
    if argv is None and argv_of is None:
        raise ValueError("run_rank_processes: argv or argv_of is required")
    base = os.environ if env is None else env
    ldir = _log_dir(log_dir)
    if P == 1:
        from ...process import run_logged
        cmd = list(argv if argv is not None else argv_of(0))
        log = os.path.join(ldir, "rank0.log")
        r = run_logged(cmd, timeout_s=run_timeout_s, on_line=on_line, log_path=log, what=what, env=dict(base))
        rec = {"rank": 0, "argv": cmd, "rc": r.rc, "wall_s": round(r.wall_s, 3), "log": log, "ok": r.ok}
        if r.error:
            rec.update(error=r.error, reason=r.reason)
        del _LAST_RUN[:]
        if not r.ok:
            raise RankFailed("rank_timeout" if r.timed_out else "rank_failed", 0, r.rc, _tail(log), log)
        return [rec]
    if not cpu_ok:
        refuse_unless_visible(P, visible)
    devs = [int(d) for d in devices] if devices is not None else list(range(P))
    if len(devs) != P:
        raise RowpairRefused(f"devices={devs}: {P} entries required for n_gpu={P}")
    if not cpu_ok:
        device_census(devs, allow_mixed=allow_mixed)
    sys.stderr.write(rankenv_line(_tag(base), base, P) + "\n")            # `[<tag>] RANKENV hashseed=<v> source=… ranks=P`, once per launch (rank_env layers every rank on that seed)
    sys.stderr.flush()
    port = free_port()
    store = new_store_path()
    procs, logs, fhs = [], [], []
    t0 = time.monotonic()
    for r in range(P):
        cmd = list(argv if argv is not None else argv_of(r))
        renv = rank_env(r, P, port, log_dir=ldir, base=base, isolate_device=(devs[r] if isolate_devices else None),
                        local=(None if isolate_devices else devs[r]), torchrun_names=torchrun_names, store=store)
        if nccl_timeout_s is not None:
            renv[ENV_NCCL_TIMEOUT] = str(float(nccl_timeout_s))               # read by dist.init_from_env in the rank process
        log = os.path.join(ldir, f"rank{r}.log")
        fh = open(log, "w", encoding="utf-8")
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE if r == 0 else fh, stderr=subprocess.STDOUT, text=True, errors="replace",
                             bufsize=1, start_new_session=True, env=renv, preexec_fn=_pdeathsig)
        procs.append((r, cmd, p))
        logs.append(log)
        fhs.append(fh)

    def pump():                                                              # rank 0's transcript: to its log AND the arm's stream
        r0 = procs[0][2]
        for ln in r0.stdout:
            fhs[0].write(ln)
            fhs[0].flush()
            if on_line is not None:
                on_line(ln)

    reader = threading.Thread(target=pump, name="rowpair-rank0-transcript", daemon=True)
    reader.start()
    first = None                                                             # the named event: (event, rank, exitcode)
    failed = None                                                            # the first rank that EXITED with a non-zero status (not killed): (event, rank, exitcode)
    failed_at = None
    finished: Dict[int, float] = {}
    try:
        while True:
            codes = [p.poll() for _, _, p in procs]
            now = time.monotonic()
            for (r, _, _), c in zip(procs, codes):
                if c is not None and r not in finished:
                    finished[r] = now
            exited = sorted((finished[r], r, c) for (r, _, _), c in zip(procs, codes) if c is not None and 0 < c <= 128)
            if exited and failed is None:                                    # exited with an error status (the earliest, then the lowest rank): its peers run on for
                failed_at, r, c = exited[0]                                  # the failure grace — the output rank writes its last item's outputs — then are torn down
                failed = ("rank_failed", r, c)
                grace = fail_grace_s if fail_grace_s is not None else finish_grace_s
                sys.stderr.write(rankexit_line(_tag(base), r, c, grace) + "\n")
                sys.stderr.flush()
            killed = [(r, c) for (r, _, _), c in zip(procs, codes) if c is not None and (c < 0 or c > 128)]
            if killed and first is None:                                     # died on a signal (or a wrapper reporting 128+signal): nothing more comes — tear down now
                first = failed or ("rank_failed",) + killed[0]
            if first is not None or all(c is not None for c in codes):
                break
            live = [r for (r, _, _), c in zip(procs, codes) if c is None]
            if run_timeout_s is not None and now - t0 > float(run_timeout_s):
                first = failed or ("rank_timeout", live[0] if live else 0, None)
                break
            if failed is not None:
                grace = fail_grace_s if fail_grace_s is not None else finish_grace_s
                if grace is not None and now - failed_at > float(grace):
                    first = failed
                    break
            elif finish_grace_s is not None and finished and now - min(finished.values()) > float(finish_grace_s):
                first = ("rank_timeout", live[0] if live else 0, None)
                break
            time.sleep(POLL_S)
        if first is None and failed is not None:
            first = failed
    finally:
        for _, _, p in procs:
            if p.poll() is None:
                try:
                    os.killpg(p.pid, signal.SIGTERM)
                except (ProcessLookupError, OSError):
                    pass
        t1 = time.monotonic()
        for _, _, p in procs:
            try:
                p.wait(max(0.0, KILL_GRACE_S - (time.monotonic() - t1)))
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(p.pid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    pass
                p.wait()
        reader.join(KILL_GRACE_S)
        for fh in fhs:
            fh.close()
    wall = round(time.monotonic() - t0, 3)
    recs = [{"rank": r, "argv": cmd, "rc": p.returncode, "wall_s": round(finished.get(r, time.monotonic()) - t0, 3), "log": logs[r],
             "ok": p.returncode == 0, "error": None} for r, cmd, p in procs]
    for rec in recs:
        if first is not None and rec["rank"] == first[1]:
            rec["error"] = first[0]
        elif rec["rc"] != 0:
            rec["error"] = "rank_failed" if rec["rc"] not in (None, -signal.SIGTERM, -signal.SIGKILL) else "torn_down"
    _LAST_RUN[:] = [{k: v for k, v in rec.items() if k != "argv"} for rec in recs]
    if first is not None:
        ev, r, c = first
        raise RankFailed(ev, r, c, _tail(logs[r]), logs[r])
    bad = [rec for rec in recs if not rec["ok"]]
    if bad:
        raise RankFailed("rank_failed", bad[0]["rank"], bad[0]["rc"], _tail(bad[0]["log"]), bad[0]["log"])
    return recs
