"""boltz2_opt.prefetch — the persistent featurizer of the pipelined worker (registry: ``prefetch``; switch ``BOLTZ_PREFETCH=persistent``).

What it replaces. The pipelined worker (``bz_worker_lev*.py --pipeline 1``) featurizes input k+1 while input k predicts: before every
prediction it builds the next input's ``Boltz2InferenceDataModule`` and iterates its ``predict_dataloader()`` — a stock
``DataLoader(num_workers=1, pin_memory=True)`` whose iterator FORKS a worker process from the predicting process itself, i.e. from a process
that holds the CUDA context, the model and every kit hook. After such a fork the parent's address space is copy-on-write shared with the
child; every page the parent then writes is copied on first touch, and the parent writes a great many pages while it issues a forward
(Python heap, allocator metadata, driver command buffers). On a launch-bound item (200-400 tokens) that shows as a 10-25 % slower host for the
whole following forward (the input embedder and the sampler's first step 3-5x slower) — paid again on EVERY input, because every input forks anew.

What it does instead. ONE helper process serves every input's featurization, and it descends from the kit's parsing ZYGOTE
(``boltz2_opt.prep``: forked by worker_launch at launch — before CUDA is initialised, before any hook, before the model exists): at attach
time the zygote forks a fresh child which detaches the helper (``spawn``), so the helper's lineage never held a CUDA context and shares no
page with the predicting process; the predicting process itself never forks again. ``Boltz2InferenceDataModule.predict_dataloader`` (stock
class, patched at runtime; the stock CLI path and its ``--num_workers`` are untouched — this module is only ever imported inside the kit
worker) then returns a :class:`PrefetchLoader` whose iterator

  * draws ``base_seed`` from the default CPU generator with the very statement ``torch.utils.data.dataloader._BaseDataLoaderIter`` uses
    (``torch.empty((), dtype=torch.int64).random_().item()``), at the same point of the worker's sequence (``iter(dl)``), so the parent's
    generator state after ``iter()`` is the state the stock iterator leaves (the worker snapshots it right there and restores it at
    ``predict_step``; the digest of that state per input rides :func:`report`);
  * hands (the dataset's constructor arguments, ``base_seed``, ``num_workers``) to the helper, which replays ``_worker_loop``'s preamble for
    worker id 0 — ``torch.set_num_threads(1)``, ``random.seed(base_seed)``, ``torch.manual_seed(base_seed)``,
    ``numpy.random.seed(_generate_state(base_seed, 0))``, the ``WorkerInfo`` global — builds the stock ``PredictionDataset`` from the same
    arguments the DataModule holds, and fetches index 0 through torch's own map-dataset fetcher with the stock ``collate`` (= what worker 0 of
    the stock loader computes, statement for statement);
  * receives the batch over torch's shared-memory tensor transport (the reductions the stock loader's worker queue uses, across a unix
    socket; the helper and the worker carry the same multiprocessing authkey — the zygote inherited it at launch) and pins it in the calling
    thread as the stock loader's pin thread would (``torch.utils.data._utils.pin_memory.pin_memory``; ``BOLTZ_PREFETCH=persistent,nopin``
    leaves it pageable — a transfer-property word, the bytes are the same either way). The templ guard's featurizer-side census
    (``boltz2_opt.templates``) is installed in the helper when ``templ`` is attached, so its TEMPLATE lines keep printing.

The FIRST input the helper serves in a process is also featurized by the stock loader (one stock ``DataLoader`` iteration from the same
generator state, once) and the two batches are bit-compared tensor by tensor: unequal bytes refuse the lever by name for the rest of the
process (``execution=refused:featurized_differ``, gate refused) — a future featurizer whose bytes depend on the process it runs in cannot
ride this path unnoticed. The dataset is ALSO constructed in the parent exactly where stock constructs it (``predict_dataloader``), so any parent-side effect of its
constructor is kept; only the fork moves. Featurization output is byte-identical tensor by tensor (tests/test_prefetch.py bit-compares it
against the stock loader on real inputs); predictions are bit-identical (the lever is in the exact row).

Failure is named, never silent: a helper that died or raised makes THAT input's loader the stock ``DataLoader`` (``fallback_by=<reason>`` on
the LEVER line, one stderr line per event); the worker's ``--pipeline 0`` route (Lightning's ``Trainer.predict``) is left stock by name
(``execution=skipped_by_name:pipeline0``).

Transport and deadlines (0.3.36, ``boltz2_opt.xfer``). The batch's CPU storages travel by torch's shared-memory descriptor transport as
before, EXCEPT a storage of 4 GiB or more (``BOLTZ_OPT_XFER=auto``, the default; nothing under ~4,000 tokens carries one), which travels as a
file on the box's disk (``<kit workdir>/_xfer``): the helper's reply AND the stock loader workers this module forks (the first input's
bit-compare reference, a named fallback) — a featurized batch is ~560 B/token² (45 GB at 9,000 tokens) and a container's ``/dev/shm`` is a
fixed 80 GiB, so from ~9,000 tokens the reference worker's queue feeder raised ``No space left on device`` inside ``multiprocessing``'s feeder
thread, which drops the payload, and the predicting process waited for ever. Every wait of this module now has a deadline that FAILS BY NAME:
the helper's reply (``BOLTZ_OPT_HELPER_TIMEOUT_S``, default 3600 s after the predicting process asks — a hung helper is that input's named
fallback ``helper_timeout``; a dead one was already ``helper_died``), and the stock loaders this module builds (torch's ``DataLoader(timeout=)``,
``BOLTZ_OPT_LOADER_TIMEOUT_S``, default 1800 s: the reference that does not come refuses the lever by name, a fallback loader that does not
deliver raises ``[boltz2-opt prefetch] FAILED …`` out of the worker's ``next()`` — the pass exits non-zero with those words, it never sleeps).

LEVER line (stderr at exit, and ``prefetch_report`` in the worker log)::

    [boltz2-opt] LEVER name=prefetch state=on impl=boltz2_opt.prefetch origin=kit strategy=LOCAL.boltz2.persistent_featurizer served=<n>
                 fallback=<n> [fallback_by=<reason:n,…>] forks_avoided=<served-ref_forks> ref_forks=1 helper_pid=<pid> helper_lineage=zygote forked_before_cuda=True
                 pin=parent|none featurize_s_total=<s> handoff_ms_max=<ms> base_seed_draws=<n>
  (unchanged since 0.3.2x — the harness's activation tables parse it) and, since 0.3.36, the hand-over's census on its own line:
  ``[boltz2-opt] XFER lever=prefetch word=auto|shm|file [min_mib=<m>] feats_gib_max=<g> files_read=<n> gib_read=<g> read_s=<s> helper=<word>
                 helper_timeout_s=<s> loader_timeout_s=<s> dir=<path>``

Contract (worker_launch): ``LEVERS``; ``apply()`` -> levers installed; ``dispositions()``; ``report()``.
"""
from __future__ import annotations

import hashlib
import os
import sys
import threading
import time
import traceback
from typing import Any, Dict, List, Optional

TAG = "boltz2-opt"
SWITCH = "BOLTZ_PREFETCH"
WORD = "persistent"
NOPIN = "nopin"
HELPER_TIMEOUT_ENV = "BOLTZ_OPT_HELPER_TIMEOUT_S"   # the predicting process waits this long for the helper's reply once it asks (Helper.result), then the input is a named fallback (helper_timeout)
HELPER_TIMEOUT_S = 3600.0
LOADER_TIMEOUT_ENV = "BOLTZ_OPT_LOADER_TIMEOUT_S"   # torch's DataLoader(timeout=) on the stock loaders THIS module builds (the first input's bit-compare reference, a named fallback): a worker
LOADER_TIMEOUT_S = 1800.0                         # that never delivers (its queue feeder dropped the payload) is a named failure after this long, never a sleeping predicting process


def _seconds(name: str, default: float) -> float:
    """A deadline from the environment (seconds, > 0), else ``default``; a value that is not a positive number is the default, said once."""
    v = (os.environ.get(name) or "").strip()
    if not v:
        return float(default)
    try:
        s = float(v)
        if s > 0:
            return s
    except ValueError:
        pass
    sys.stderr.write(f"[{TAG} prefetch] {name}={v!r} is not a positive number of seconds: the default {default:g} s stands\n")
    return float(default)


def helper_timeout_s() -> float:
    return _seconds(HELPER_TIMEOUT_ENV, HELPER_TIMEOUT_S)


def loader_timeout_s() -> float:
    return _seconds(LOADER_TIMEOUT_ENV, LOADER_TIMEOUT_S)


def batch_bytes(batch) -> int:
    """Bytes of the distinct CPU storages under a featurized batch (dict / list / tensor tree) — what crosses the transport for it."""
    import torch
    seen, total = set(), 0
    stack = [batch]
    while stack:
        x = stack.pop()
        if isinstance(x, torch.Tensor):
            try:
                u = x.untyped_storage()
                key = (u.data_ptr(), int(u.nbytes()))
            except Exception:  # noqa: BLE001
                continue
            if key not in seen:
                seen.add(key); total += key[1]
        elif isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
    return total
LEVERS = ("prefetch",)
STRATEGY = "LOCAL.boltz2.persistent_featurizer"
IMPL = "boltz2_opt.prefetch"
DM_MODULE = "boltz.data.module.inferencev2"

STATE: Dict[str, Any] = {
    "installed": False, "disposition": None, "helper": None, "pin": True,
    "served": 0, "fallback": 0, "fallback_by": {}, "base_seed_draws": 0,
    "featurize_s_total": 0.0, "handoff_ms_max": 0.0, "wait_ms_total": 0.0, "pin_ms_total": 0.0, "per_item": [],
    "feats_bytes_max": 0,         # the largest featurized batch handed over in this process (distinct CPU storage bytes: batch_bytes)
    "helper_xfer": None,          # the helper's transport word as it installed it (boltz2_opt.xfer; the hello carries it)
    "forked_before_cuda": None, "helper_pid": None, "orig_predict_dataloader": None,
    "helper_cuda_initialized": None,
    "ref_forks": 0,               # stock loader workers forked from the predicting process BY THIS MODULE (the first-input bit-compare: 1)
    "bit_compare": None,          # the first served input: helper batch vs the stock loader's, tensor by tensor — "equal:<n_tensors>" | "differ:<where>" | "not_run:<why>"
    "refused": None,              # set when the bit-compare differs: every later input goes through the stock loader, the gate refuses
}
_LOCK = threading.Lock()


def _say(*words) -> None:
    sys.stderr.write(f"[{TAG} prefetch] " + " ".join(str(w) for w in words) + "\n"); sys.stderr.flush()


def _env(environ=None):
    return os.environ if environ is None else environ


def words(environ=None) -> List[str]:
    return [w.strip() for w in _env(environ).get(SWITCH, "").split(",") if w.strip()]


def requested(environ=None) -> bool:
    return WORD in words(environ)


def levers_of(environ=None) -> List[str]:
    return list(LEVERS) if requested(environ) else []


def problems(environ=None) -> List[str]:
    """Words after the switch this module does not know, and a transport word ``boltz2_opt.xfer`` refuses (refused by name at apply)."""
    out = [f"{SWITCH}: unknown word {w!r} (known: {WORD}, {NOPIN})" for w in words(environ) if w not in (WORD, NOPIN)]
    from . import xfer
    for probe in (xfer.word, xfer.min_bytes):
        try:
            probe(environ)
        except xfer.Refused as e:
            out.append(str(e))
    return out


def worker_pipeline(argv=None) -> Optional[int]:
    """The worker script's ``--pipeline`` word from this process's argv (worker_launch runs the script in this interpreter), None when absent."""
    argv = sys.argv if argv is None else argv
    if "--pipeline" in argv:
        try:
            return int(argv[argv.index("--pipeline") + 1])
        except (ValueError, IndexError):
            return None
    return None


def worker_attaches(argv=None) -> List[str]:
    """The launcher's ``--attach`` names (worker_launch's argv is this process's ``sys.orig_argv``; the script's argv no longer carries them)."""
    argv = list(getattr(sys, "orig_argv", None) or []) if argv is None else list(argv)
    if "--attach" in argv:
        try:
            return [n for n in argv[argv.index("--attach") + 1].split(",") if n]
        except IndexError:
            return []
    return []


def rng_digest() -> str:
    """sha256 (16 hex) of the default CPU generator's state — the state the worker snapshots right after ``iter(dl)``."""
    import torch
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------------------- the helper process
def _worker_preamble(base_seed: int, worker_id: int, num_workers: int, dataset) -> int:
    """``torch.utils.data._utils.worker._worker_loop``'s per-worker preamble, statement for statement (thread count, the three seeds, the
    WorkerInfo global). Returns the worker seed."""
    import random
    import torch
    from torch.utils.data._utils import worker as _w
    torch.set_num_threads(1)
    seed = base_seed + worker_id
    random.seed(seed)
    torch.manual_seed(seed)
    if _w.HAS_NUMPY:
        np_seed = _w._generate_state(base_seed, worker_id)
        import numpy as np
        np.random.seed(np_seed)
    _w._worker_info = _w.WorkerInfo(id=worker_id, num_workers=num_workers, seed=seed, dataset=dataset)
    return seed


def featurize(dataset_kwargs: dict, base_seed: int, num_workers: int, dataset=None, collate_fn=None):
    """What worker 0 of the stock loader computes for a one-record inference dataset: the preamble, then index batch ``[0]`` through torch's
    map-dataset fetcher with the stock ``collate``. ``dataset`` / ``collate_fn`` given = use those (tests); else ``PredictionDataset(**kwargs)``
    and ``inferencev2.collate``."""
    from torch.utils.data import _DatasetKind
    if dataset is None or collate_fn is None:
        import importlib
        inf = importlib.import_module(DM_MODULE)
        ds = inf.PredictionDataset(**dataset_kwargs) if dataset is None else dataset
        collate_fn = inf.collate if collate_fn is None else collate_fn
    else:
        ds = dataset
    _worker_preamble(base_seed, 0, num_workers, ds)
    fetcher = _DatasetKind.create_fetcher(_DatasetKind.Map, ds, True, collate_fn, False)   # auto_collation=True (batch_size=1), drop_last=False
    return fetcher.fetch([0])


SPAWN = "boltz2_opt.prefetch:spawn"          # the zygote target (prep.Zygote.run): detaches the helper from a fresh zygote child


def _helper_main(address: str, attaches, ready_fd: int) -> None:
    """The helper's life: bind the listener, accept the worker's ONE connection, serve jobs ``(seq, dataset_kwargs, base_seed, num_workers)``
    -> ``("ok", seq, batch, featurize_s)`` | ``("err", seq, traceback)`` until ``None`` / EOF. Runs in the zygote's grandchild: no CUDA, no
    model, no hook but the templ guard's featurizer census (installed here when ``templ`` is attached, as a stock loader worker inherits it)."""
    import multiprocessing
    from multiprocessing.connection import Listener
    import torch
    import torch.multiprocessing  # noqa: F401  (tensor reductions for the replies)
    from . import xfer
    try:                                                   # the reply's large CPU storages travel as files on the box's disk, not /dev/shm segments (xfer: BOLTZ_OPT_XFER, auto by default)
        xfer.install()
        xfer_word = xfer.describe()["word"]
    except Exception as e:  # noqa: BLE001                 # an unknown word is refused in the predicting process before the spawn (problems); here it is said and torch's transport stands
        xfer_word = f"shm:refused:{type(e).__name__}"
        _say(f"helper: transport not installed ({type(e).__name__}: {e}); torch's shared-memory transport stands")
    if "templ" in attaches:
        try:
            import boltz.data.feature.featurizerv2 as FZ
            from . import templates as _T
            _T._wrap_featurizer(FZ)
        except Exception as e:  # noqa: BLE001
            _say(f"helper: templ featurizer census not installed: {type(e).__name__}: {e}")
    try:                                                   # the process-wide RDKit pickling option boltz's predict() sets before any loader worker exists
        from rdkit import Chem                             # (main.py; the kit worker mirrors it): a stock worker inherits it by fork, the helper sets it here
        Chem.SetDefaultPickleProperties(Chem.PropertyPickleOptions.AllProps)
    except Exception as e:  # noqa: BLE001
        _say(f"helper: RDKit pickle option not set: {type(e).__name__}: {e}")
    key = multiprocessing.current_process().authkey
    with Listener(address, family="AF_UNIX", backlog=1, authkey=key) as listener:
        sys.stdout.write(f"ready pid={os.getpid()} cuda_initialized={torch.cuda.is_initialized()}\n"); sys.stdout.flush()
        os.close(ready_fd)                                 # the spawning child returns to the zygote now
        conn = listener.accept()
    conn.send(("hello", os.getpid(), bool(torch.cuda.is_initialized()), torch.__version__, xfer_word))
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            break
        if job is None:
            break
        seq = job[0]
        try:
            _, kwargs, base_seed, num_workers = job
            t = time.perf_counter()
            batch = featurize(kwargs, base_seed, num_workers)
            dt = time.perf_counter() - t
            facts = {"feats_bytes": batch_bytes(batch)}
            conn.send(("ok", seq, batch, dt, facts))     # ForkingPickler: each CPU storage by torch's descriptor transport or, from xfer's floor up, as a file (boltz2_opt.xfer)
            del batch
        except Exception:  # noqa: BLE001
            try:
                conn.send(("err", seq, traceback.format_exc()))
            except Exception:  # noqa: BLE001
                break
    try:
        conn.close()
    except Exception:  # noqa: BLE001
        pass


def spawn(address: str, attaches=()) -> None:
    """Zygote target (``prep.zygote().run(SPAWN, address=…, attaches=[…])``): runs in a FRESH child of the zygote and detaches the helper —
    double fork, new session, stdout/stderr back on the worker's real descriptors — then returns once the helper's listener is bound
    (prints ``helper pid=<pid>``). The helper is thus a descendant of the pre-CUDA zygote, never of the predicting process."""
    r, w = os.pipe()
    sys.stdout.flush(); sys.stderr.flush()
    pid = os.fork()
    if pid == 0:                                           # intermediate -> helper
        code = 0
        try:
            os.close(r)
            os.setsid()
            sys.stdout, sys.stderr = sys.__stdout__, sys.__stderr__   # the zygote child captured them into StringIO; the helper speaks on the real fds
            _helper_main(address, tuple(attaches), w)
        except BaseException:
            code = 1
            try:
                sys.__stderr__.write(f"[{TAG} prefetch] helper died:\n{traceback.format_exc()}"); sys.__stderr__.flush()
            except Exception:  # noqa: BLE001
                pass
        finally:
            os._exit(code)
    os.close(w)
    with os.fdopen(r) as fh:                               # blocks until the helper bound its listener (or died: EOF either way)
        fh.read()
    print(f"helper pid={pid}")


class Helper:
    """The worker's handle on the helper: spawned through the zygote, reached over one unix-socket connection. Not thread-safe (the worker's
    main thread is the only caller)."""

    def __init__(self, attaches=()):
        import multiprocessing
        import tempfile
        from multiprocessing.connection import Client
        import torch.multiprocessing  # noqa: F401  (tensor reductions for the replies this side rebuilds)
        from . import prep
        z = prep.zygote()
        if z is None or not hasattr(z, "run"):
            raise RuntimeError("no_zygote")
        d = z.describe() if hasattr(z, "describe") else {}
        if "zygote_pid" in d and d.get("zygote_pid") is None:
            z.start()
            d = z.describe()
        self.zygote_cuda_initialized = d.get("cuda_initialized_at_fork")
        self.dir = tempfile.mkdtemp(prefix="bz2pf_")
        self.address = os.path.join(self.dir, "s")
        res = z.run(SPAWN, address=self.address, attaches=list(attaches))
        if res.rc != 0 or "helper pid=" not in res.out:
            raise RuntimeError(f"spawn_failed:rc={res.rc}:{(res.err or res.out).strip().splitlines()[-1][:120] if (res.err or res.out).strip() else '-'}")
        self.pid = int(res.out.split("helper pid=")[1].split()[0])
        self.conn = Client(self.address, family="AF_UNIX", authkey=multiprocessing.current_process().authkey)
        if not self.conn.poll(60.0):
            raise RuntimeError("spawn_failed:no_hello")
        hello = self.conn.recv()
        if not (isinstance(hello, tuple) and hello[:1] == ("hello",)):
            raise RuntimeError("spawn_failed:bad_hello")
        self.helper_cuda_initialized = bool(hello[2])          # the helper's own torch.cuda.is_initialized() — False by lineage; carried on the LEVER line
        self.helper_xfer = str(hello[4]) if len(hello) > 4 else None   # the transport word the helper installed (boltz2_opt.xfer; None from a helper older than 0.3.35)
        self.seq = 0
        self.pending = None

    def alive(self) -> bool:
        try:
            os.kill(self.pid, 0)
        except OSError:
            return False
        return not self.conn.closed

    def submit(self, dataset_kwargs: dict, base_seed: int, num_workers: int) -> int:
        self.seq += 1
        self.conn.send((self.seq, dataset_kwargs, int(base_seed), int(num_workers)))
        self.pending = self.seq
        return self.seq

    def result(self, seq: int, poll_s: float = 1.0, timeout_s: Optional[float] = None):
        """Block for job ``seq``'s reply; raises RuntimeError (named) when the helper dies first (``helper_died:…``) or has not answered
        ``timeout_s`` seconds (default :func:`helper_timeout_s`) after this call (``helper_timeout:<s>s`` — a hung helper is a named
        fallback of this input, never a predicting process that waits for ever)."""
        deadline = time.monotonic() + float(helper_timeout_s() if timeout_s is None else timeout_s)
        while True:
            if time.monotonic() > deadline:
                raise RuntimeError(f"helper_timeout:{int(helper_timeout_s() if timeout_s is None else timeout_s)}s")
            try:
                ready = self.conn.poll(poll_s)
            except (EOFError, OSError) as e:
                raise RuntimeError(f"helper_died:{type(e).__name__}") from None
            if ready:
                try:
                    msg = self.conn.recv()
                except (EOFError, OSError) as e:
                    raise RuntimeError(f"helper_died:{type(e).__name__}") from None
                if msg[1] != seq:                      # a stale reply (an earlier job nobody collected): drop it
                    continue
                self.pending = None
                return msg
            if not self.alive():
                raise RuntimeError("helper_died:gone")

    def close(self) -> None:
        try:
            self.conn.send(None)
        except Exception:  # noqa: BLE001
            pass
        try:
            self.conn.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            import shutil
            shutil.rmtree(self.dir, ignore_errors=True)
        except Exception:  # noqa: BLE001
            pass


# ----------------------------------------------------------------------------------------------------------------- the loader the DataModule returns
def dataset_kwargs_of(dm) -> dict:
    """The ``PredictionDataset`` constructor arguments stock's ``predict_dataloader`` passes, read from the DataModule instance."""
    return dict(manifest=dm.manifest, target_dir=dm.target_dir, msa_dir=dm.msa_dir, mol_dir=dm.mol_dir, constraints_dir=dm.constraints_dir,
                template_dir=dm.template_dir, extra_mols_dir=dm.extra_mols_dir, override_method=dm.override_method, affinity=dm.affinity)


def draw_base_seed() -> int:
    """``_BaseDataLoaderIter.__init__``'s draw for a loader with no ``generator``: one int64 from the default CPU generator."""
    import torch
    return torch.empty((), dtype=torch.int64).random_().item()


class _Iter:
    def __init__(self, loader: "PrefetchLoader"):
        import torch
        self.loader = loader
        self.done = False
        self.rng_before = torch.get_rng_state()               # the stock loader (first-input bit-compare, or a named fallback) re-draws base_seed from HERE: same value, same end state
        self.base_seed = draw_base_seed()                     # the parent's ONE generator draw, where iter(stock loader) makes it
        self.digest = rng_digest()
        with _LOCK:
            STATE["base_seed_draws"] += 1
        self.seq = None
        self.fallback_iter = None
        self.fallback_reason = None
        self.fallback_timeout = 0
        if STATE["refused"]:
            self._fallback(STATE["refused"])
            return
        h: Optional[Helper] = STATE["helper"]
        if h is None or not h.alive():
            self._fallback("helper_not_alive")
            return
        try:
            self.seq = h.submit(loader.dataset_kwargs, self.base_seed, loader.num_workers)
            self.t_submit = time.perf_counter()
        except Exception as e:  # noqa: BLE001
            self._fallback(f"submit:{type(e).__name__}")

    def _fallback(self, reason: str):
        """This input through the stock loader — the exact stock statement — counted by name. The default CPU generator is put back to its
        state before this iterator's draw, so the stock iterator draws the SAME base_seed at the same position and leaves the same state
        (worker 0 then seeds exactly as it would have without the lever)."""
        import torch
        from torch.utils.data import DataLoader
        _say(f"FALLBACK input through the stock DataLoader: {reason}")
        with _LOCK:
            STATE["fallback"] += 1
            STATE["fallback_by"][reason] = STATE["fallback_by"].get(reason, 0) + 1
        torch.set_rng_state(self.rng_before)
        self.fallback_reason = reason
        self.fallback_timeout = loader_timeout_s() if int(self.loader.num_workers) > 0 else 0    # torch's own deadline on the worker's delivery (a worker whose queue feeder dropped the payload never delivers): named in _fallback_next
        dl = DataLoader(self.loader.dataset, batch_size=1, num_workers=self.loader.num_workers, pin_memory=True, shuffle=False,
                        collate_fn=self.loader.collate_fn, timeout=self.fallback_timeout)
        self.fallback_iter = iter(dl)

    def _fallback_next(self):
        """The fallback loader's batch; a worker that delivers nothing within the deadline, or dies, FAILS BY NAME (RuntimeError out of the
        worker script's ``next()``: the pass exits non-zero with these words) — the predicting process never waits for ever."""
        try:
            return next(self.fallback_iter)
        except StopIteration:
            raise
        except Exception as e:  # noqa: BLE001
            raise RuntimeError(f"[{TAG} prefetch] FAILED by name: the stock DataLoader this input fell back to (fallback_by={self.fallback_reason}) delivered no batch — "
                               f"{type(e).__name__}: {str(e).strip()[:400]} (deadline {LOADER_TIMEOUT_ENV}={self.fallback_timeout:g}s; a loader worker whose queue feeder "
                               f"raised — e.g. 'unable to allocate shared memory' — printed its traceback above and dropped the batch)") from e

    def __iter__(self):
        return self

    def __next__(self):
        if self.fallback_iter is not None:
            return self._fallback_next()
        if self.done:
            raise StopIteration
        h: Helper = STATE["helper"]
        from . import xfer
        xf0 = xfer.stats()                                     # the reply's file-borne storages are rebuilt INSIDE conn.recv (Helper.result): bracket it
        t0 = time.perf_counter()
        try:
            msg = h.result(self.seq)
        except Exception as e:  # noqa: BLE001
            self._fallback(f"{e}" if str(e).startswith(("helper_died", "helper_timeout")) else f"result:{type(e).__name__}")
            return self._fallback_next()
        wait_ms = (time.perf_counter() - t0) * 1e3
        if msg[0] != "ok":
            _say("helper raised while featurizing:\n" + str(msg[2])[-2000:])
            self._fallback("helper_raised")
            return self._fallback_next()
        batch, feat_s = msg[2], msg[3]
        facts = msg[4] if len(msg) > 4 and isinstance(msg[4], dict) else {}
        feats_bytes = int(facts.get("feats_bytes") or batch_bytes(batch))
        if STATE["bit_compare"] is None:                       # the FIRST served input of this process: the helper's bytes against the stock loader's, once
            try:
                ref = _stock_reference_batch(self.loader, self.rng_before)
                diffs = compare_batches(ref, batch)
                n = sum(1 for v in batch.values() if hasattr(v, "dtype")) if isinstance(batch, dict) else 0
                STATE["bit_compare"] = f"equal:{n}" if not diffs else f"differ:{diffs[0]}"
                del ref
            except Exception as e:  # noqa: BLE001
                STATE["bit_compare"] = f"not_run:{type(e).__name__}"
                _say(f"first input: the stock loader's reference batch did not come — {type(e).__name__}: {str(e).strip()[:400]} (deadline {LOADER_TIMEOUT_ENV}={loader_timeout_s():g}s)")
            if not STATE["bit_compare"].startswith("equal:"):
                STATE["refused"] = f"refused:featurized_{STATE['bit_compare'].replace(':', '_', 1).split(':')[0]}"
                _say(f"REFUSED by name: the helper's featurized batch is not the stock loader's ({STATE['bit_compare']}); this and every later input through the stock DataLoader")
                del batch
                self._fallback(STATE["refused"])
                return self._fallback_next()
            _say(f"first input: helper batch bit-compared against the stock loader's — {STATE['bit_compare']} tensors equal")
        t1 = time.perf_counter()
        if self.loader.pin:
            from torch.utils.data._utils.pin_memory import pin_memory
            batch = pin_memory(batch, None)
        pin_ms = (time.perf_counter() - t1) * 1e3
        self.done = True
        with _LOCK:
            STATE["served"] += 1
            STATE["featurize_s_total"] += float(feat_s)
            STATE["wait_ms_total"] += wait_ms
            STATE["pin_ms_total"] += pin_ms
            STATE["handoff_ms_max"] = max(STATE["handoff_ms_max"], pin_ms)
            STATE["feats_bytes_max"] = max(int(STATE["feats_bytes_max"]), feats_bytes)
            xf1 = xfer.stats()
            STATE["per_item"].append({"seq": self.seq, "base_seed": int(self.base_seed), "rng_after_iter": self.digest, "featurize_s": round(float(feat_s), 3),
                                      "wait_ms": round(wait_ms, 1), "pin_ms": round(pin_ms, 1), "feats_gib": round(feats_bytes / 2 ** 30, 3),
                                      "xfer_files": int(xf1["files_read"]) - int(xf0["files_read"]), "xfer_gib": round(float(xf1["gib_read"]) - float(xf0["gib_read"]), 3)})
        return batch

    def __len__(self):
        return 1


class PrefetchLoader:
    """What the patched ``predict_dataloader`` returns: iterable once per ``iter()``, one batch, like the stock loader over a one-record dataset."""

    def __init__(self, dataset, collate_fn, dataset_kwargs: dict, num_workers: int, pin: bool):
        self.dataset, self.collate_fn, self.dataset_kwargs, self.num_workers, self.pin = dataset, collate_fn, dataset_kwargs, num_workers, pin

    def __iter__(self):
        return _Iter(self)

    def __len__(self):
        return 1


def _patched_predict_dataloader(self):
    """Stock's body (the dataset constructed here, in the parent, with the same arguments) returning the persistent-helper loader."""
    import importlib
    inf = importlib.import_module(DM_MODULE)
    kwargs = dataset_kwargs_of(self)
    dataset = inf.PredictionDataset(**kwargs)
    if len(dataset) != 1:                                      # the worker's DataModules carry one record each; anything else is stock's loader by name
        _say(f"FALLBACK stock DataLoader: dataset has {len(dataset)} records (the persistent path serves one-record datasets)")
        with _LOCK:
            STATE["fallback"] += 1; STATE["fallback_by"]["multi_record"] = STATE["fallback_by"].get("multi_record", 0) + 1
        return STATE["orig_predict_dataloader"](self)
    return PrefetchLoader(dataset, inf.collate, kwargs, int(self.num_workers), STATE["pin"])


# ----------------------------------------------------------------------------------------------------------------- first-input bit-compare
def _tensor_bytes(t):
    import torch
    t = t.detach().cpu().contiguous()
    return t.view(torch.uint8) if t.dtype != torch.bool else t.to(torch.uint8)


def compare_batches(x, y, path="batch") -> List[str]:
    """Every difference between two featurized batches (dicts / lists / tensors / scalars), tensors compared dtype + shape + bytes."""
    import torch
    out: List[str] = []
    if isinstance(x, dict):
        if not isinstance(y, dict) or set(x) != set(y):
            return [f"{path}:keys"]
        for k in sorted(x):
            out += compare_batches(x[k], y[k], f"{path}.{k}")
    elif isinstance(x, torch.Tensor):
        if not isinstance(y, torch.Tensor) or x.dtype != y.dtype or tuple(x.shape) != tuple(y.shape):
            out.append(f"{path}:dtype_or_shape")
        elif not bool(torch.equal(_tensor_bytes(x), _tensor_bytes(y))):
            out.append(f"{path}:bytes")
    elif isinstance(x, (list, tuple)):
        if not isinstance(y, (list, tuple)) or len(x) != len(y):
            return [f"{path}:len"]
        for i, (a, b) in enumerate(zip(x, y)):
            out += compare_batches(a, b, f"{path}[{i}]")
    elif x != y:
        out.append(f"{path}:value")
    return out


def _stock_reference_batch(loader: "PrefetchLoader", rng_before_draw):
    """The stock loader's batch for this input, ONCE (the first served input of the process): the stock ``DataLoader(num_workers, pin_memory
    off, stock collate)`` iterated from the very CPU-generator state this input's draw started from, so its worker seeds equal the helper's;
    the generator is left where the persistent path left it (one draw)."""
    import torch
    from torch.utils.data import DataLoader
    after = torch.get_rng_state()
    with _LOCK:
        STATE["ref_forks"] += 1
    try:
        torch.set_rng_state(rng_before_draw)
        dl = DataLoader(loader.dataset, batch_size=1, num_workers=max(int(loader.num_workers), 1), pin_memory=False, shuffle=False, collate_fn=loader.collate_fn,
                        timeout=loader_timeout_s())                       # a reference worker that never delivers is `not_run:RuntimeError` (torch: "DataLoader timed out after …"), the lever refused by name — not a wait without end
        it = iter(dl)
        ref = next(it)
        del it, dl
    finally:
        torch.set_rng_state(after)
    return ref


# ----------------------------------------------------------------------------------------------------------------- contract
def dispositions() -> Dict[str, str]:
    d = STATE.get("disposition")
    return {"prefetch": d} if d else {}


def apply(spec=None) -> List[str]:
    """Fork the helper (once) and patch the DataModule. Returns the levers installed ([] when the switch is absent or a disposition applies)."""
    if STATE["installed"]:
        return list(LEVERS)
    if not requested():
        return []
    bad = problems()
    if bad:
        raise RuntimeError("; ".join(bad))
    pipe = worker_pipeline()
    if pipe != 1:
        STATE["disposition"] = "skipped_by_name:pipeline0"
        _say(f"not installed: the worker runs --pipeline {pipe} (Lightning's own loader route); stock DataLoader by name")
        return []
    import importlib
    inf = importlib.import_module(DM_MODULE)
    STATE["pin"] = NOPIN not in words()
    from . import xfer
    xfer.install()                                              # this process REBUILDS what the helper sends and FORKS the stock loader workers (bit-compare reference, fallbacks): their large storages travel as files too
    try:
        h = Helper(attaches=worker_attaches())
    except Exception as e:  # noqa: BLE001
        STATE["disposition"] = f"skipped_by_name:{str(e).split(':')[0] if str(e).startswith(('no_zygote', 'spawn_failed')) else 'helper_' + type(e).__name__}"
        _say(f"not installed: the helper could not be spawned through the zygote ({type(e).__name__}: {e}); stock DataLoader by name")
        return []
    STATE.update(helper=h, helper_pid=h.pid, forked_before_cuda=(h.zygote_cuda_initialized is False and h.helper_cuda_initialized is False),
                 helper_cuda_initialized=h.helper_cuda_initialized, helper_xfer=h.helper_xfer)
    STATE["orig_predict_dataloader"] = inf.Boltz2InferenceDataModule.predict_dataloader
    inf.Boltz2InferenceDataModule.predict_dataloader = _patched_predict_dataloader
    STATE["installed"] = True
    import atexit
    atexit.register(_at_exit)
    _say(f"helper pid={h.pid} spawned through the zygote (zygote cuda_initialized_at_fork={h.zygote_cuda_initialized}); Boltz2InferenceDataModule.predict_dataloader -> persistent featurizer; pin={'parent' if STATE['pin'] else 'none'}; "
         f"transport {xfer.describe()['word']} (helper: {h.helper_xfer}; a CPU storage of {xfer.describe()['min_mib']} MiB or more travels as a file under {xfer.describe()['dir']}, smaller ones by torch's shared memory); "
         f"deadlines helper {helper_timeout_s():g}s loader {loader_timeout_s():g}s")
    return list(LEVERS)


def line() -> str:
    from opt_core.report import lever_line
    st = STATE
    if not st["installed"]:
        reason = st.get("disposition") or "not_requested"
        return lever_line(TAG, "prefetch", "skipped", reason=reason.replace("skipped_by_name:", ""), impl=IMPL, origin="kit", strategy=STRATEGY,
                          execution=st.get("disposition") or "not_requested")
    fields = dict(served=st["served"], fallback=st["fallback"])
    if st["fallback_by"]:
        fields["fallback_by"] = ",".join(f"{k}:{v}" for k, v in sorted(st["fallback_by"].items()))
    fields.update(forks_avoided=st["served"] - st["ref_forks"], ref_forks=st["ref_forks"], helper_pid=st["helper_pid"], helper_lineage="zygote", forked_before_cuda=st["forked_before_cuda"],
                  helper_cuda_initialized=st["helper_cuda_initialized"], pin="parent" if st["pin"] else "none",
                  featurize_s_total=round(st["featurize_s_total"], 2), wait_ms_total=round(st["wait_ms_total"], 1), pin_ms_total=round(st["pin_ms_total"], 1),
                  handoff_ms_max=round(st["handoff_ms_max"], 1), base_seed_draws=st["base_seed_draws"], bit_compare=st["bit_compare"] or "not_run:no_input",
                  execution=(st["refused"] or f"executed:{st['served']}"))
    return lever_line(TAG, "prefetch", "on", impl=IMPL, origin="kit", strategy=STRATEGY, **fields)


def xfer_line() -> Optional[str]:
    """``[boltz2-opt] XFER lever=prefetch word=<auto|shm|file> [min_mib=<m>] feats_gib_max=<g> files_read=<n> gib_read=<g> read_s=<s> helper=<word>
    helper_timeout_s=<s> loader_timeout_s=<s> dir=<path>`` — the hand-over's census on its OWN line (0.3.36; the LEVER line's token set is
    unchanged, the harness's activation tables parse it). None when the lever is not installed."""
    st = STATE
    if not st["installed"]:
        return None
    try:
        from . import xfer
        x = xfer.stats()
    except Exception as e:  # noqa: BLE001
        return f"[{TAG}] XFER lever=prefetch word=unknown:{type(e).__name__}"
    words = [f"word={x['word'] or 'not_installed'}"] + ([f"min_mib={x['min_mib']}"] if x["word"] == "auto" else [])
    words += [f"feats_gib_max={int(st['feats_bytes_max']) / 2 ** 30:.2f}", f"files_read={x['files_read']}", f"gib_read={x['gib_read']}", f"read_s={x['read_s']}",
              f"helper={st.get('helper_xfer') or 'unknown'}", f"helper_timeout_s={helper_timeout_s():g}", f"loader_timeout_s={loader_timeout_s():g}", f"dir={x['dir'] or '-'}"]
    if x.get("errors"):
        words.append(f"errors={len(x['errors'])}")
    return f"[{TAG}] XFER lever=prefetch " + " ".join(words)


def _xfer_report() -> Dict[str, Any]:
    try:
        from . import xfer
        out = xfer.stats()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    out["helper_word"] = STATE.get("helper_xfer")
    out["helper_timeout_s"], out["loader_timeout_s"] = helper_timeout_s(), loader_timeout_s()
    return out


def gate() -> Dict[str, Any]:
    """Fail-closed: installed, every loader this module handed out was served by the helper (no fallback), the helper forked before CUDA."""
    st = STATE
    compared = bool(st["bit_compare"] and st["bit_compare"].startswith("equal:"))
    ok = bool(st["installed"] and st["fallback"] == 0 and st["forked_before_cuda"] and st["served"] == st["base_seed_draws"] and compared and not st["refused"])
    why = [] if ok else [w for w, c in (("not_installed", not st["installed"]), ("fallbacks", st["fallback"] > 0), ("forked_after_cuda", st["forked_before_cuda"] is False),
                                         ("unserved_iterators", st["served"] != st["base_seed_draws"]), ("bit_compare_" + (st["bit_compare"] or "not_run"), not compared),
                                         ("refused", bool(st["refused"]))) if c]
    return {"ok": ok, "why": why}


def report() -> Dict[str, Any]:
    st = STATE
    return {"installed": st["installed"], "applied": list(LEVERS) if st["installed"] else [], "disposition": st["disposition"], "switch": os.environ.get(SWITCH),
            "helper_pid": st["helper_pid"], "forked_before_cuda": st["forked_before_cuda"], "helper_cuda_initialized": st["helper_cuda_initialized"], "pin": "parent" if st["pin"] else "none",
            "served": st["served"], "fallback": st["fallback"], "fallback_by": dict(st["fallback_by"]), "forks_avoided": st["served"] - st["ref_forks"], "ref_forks": st["ref_forks"],
            "base_seed_draws": st["base_seed_draws"], "featurize_s_total": round(st["featurize_s_total"], 3), "wait_ms_total": round(st["wait_ms_total"], 1),
            "pin_ms_total": round(st["pin_ms_total"], 1), "handoff_ms_max": round(st["handoff_ms_max"], 1), "per_item": list(st["per_item"]),
            "bit_compare": st["bit_compare"], "refused": st["refused"], "feats_gib_max": round(int(st["feats_bytes_max"]) / 2 ** 30, 3), "xfer": _xfer_report(),
            "gate": gate(), "line": line()}


def _at_exit() -> None:
    try:
        sys.stderr.write(line() + "\n"); sys.stderr.flush()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[{TAG} prefetch] LEVER line not written: {type(e).__name__}: {e}\n")
    try:
        xl = xfer_line()                                        # the hand-over's census on its own line (the LEVER line's token set stays as the activation tables know it)
        if xl:
            sys.stderr.write(xl + "\n"); sys.stderr.flush()
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[{TAG} prefetch] XFER line not written: {type(e).__name__}: {e}\n")
    h = STATE.get("helper")
    if h is not None:
        h.close()
    try:
        from . import xfer
        xfer.cleanup()                                          # the launch's <workdir>/_xfer: every file in it was unlinked by its reader; strays of a failed hand-over leave with the directory
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"[{TAG} prefetch] transport directory not removed: {type(e).__name__}: {e}\n")
