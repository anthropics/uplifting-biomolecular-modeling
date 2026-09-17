"""The stock ``--gpu-ids`` target-parallel workers under a kit mode (``atlasfold.cli.multigpu``: the parent distributes the targets and
``torch.multiprocessing.spawn``s one process per GPU id — ``--gpu-ids 0`` on a one-GPU box included — each running the stock
``worker_entry``: load the model, fold its share).  A spawned worker is a fresh interpreter: the parent's activation (the levers patched
into ITS process) does not reach it; left alone, a worker would fold with stock code while the parent printed ``ACTIVE`` and every
lever read ``skipped reason=no_calls``.  Here ``cli.cmd_pred`` rebinds ``multigpu.worker_entry`` (in the parent, after activation) to a
picklable partial of :func:`worker_entry` carrying the mode, ``--det`` and the partial-activation policy; in the worker it activates the
SAME mode first (``ACTIVE … trigger=worker:gpu<id>``; ``MODEL_OPT_LEVERS_OFF`` and the ``AFO_*`` words arrive through the inherited
environment), installs the per-item PHASE timing, runs the stock worker entry unchanged, and ends the worker with exit 3 when one of its
lever gates refuses (``WORKER gpu=<id> … gates=refused:…``; the parent's stock main then raises — never a silent stock run under a kit
word).  Each worker prints its own LEVER / EXIT lines at exit; the parent folds no item itself (its LEVER lines read ``skipped
reason=no_calls``) and says so: ``WORKERS gpu_ids=<ids> activation=per-worker``.  ``--mode off`` is untouched (the stock subprocess spawns
stock workers)."""
import functools
import os
import sys

SITE = "atlasfold.cli.multigpu"
ATTR = "worker_entry"


def _inputs_of(args) -> dict:
    """The run's inputs hint from the stock worker's args namespace (its --input-fasta), {} when absent."""
    try:
        from . import inputs_hint
        p = getattr(args, "input_fasta", None)
        return inputs_hint.from_argv(["--input-fasta", str(p)]) if p else {}
    except Exception:  # noqa: BLE001
        return {}


def worker_entry(worker_index, args, assignments, gpu_ids, model_type, *, mode: str, det: int = 0, allow_partial: bool = False):
    """Runs IN the spawned worker: activate ``mode`` exactly as the parent did, then the stock worker entry; exit 3 on a refused gate."""
    from . import TAG, enable
    gpu = gpu_ids[worker_index]
    enable(mode, det=det, n_gpu=1, allow_partial=allow_partial, trigger=f"worker:gpu{gpu}", inputs=_inputs_of(args))
    from . import phase_timing
    phase_timing.install()
    import importlib
    mg = importlib.import_module(SITE)
    stock = getattr(getattr(mg, ATTR), "__wrapped_stock__", None) or getattr(mg, ATTR)   # the worker's module is fresh: the stock entry
    try:
        stock(worker_index, args, assignments, gpu_ids, model_type)
    finally:
        from .stack import gates_verdict
        v = gates_verdict()
        word = "ok" if v["ok"] else "refused:" + ";".join(f"{l}:{r}" for l, r in v["refused"])
        sys.stderr.write(f"[{TAG}] WORKER gpu={gpu} pid={os.getpid()} mode={mode} det={det} gates={word}\n"); sys.stderr.flush()
        if not v["ok"]:
            sys.exit(3)


def install(mode: str, det: int = 0, allow_partial: bool = False) -> dict:
    """Rebind ``atlasfold.cli.multigpu.worker_entry`` (idempotent) so spawned workers activate ``mode``; {installed, reason}."""
    import importlib
    try:
        mg = importlib.import_module(SITE)
        stock = getattr(mg, ATTR)
    except Exception as e:  # noqa: BLE001 — a stock tree without the multi-GPU module: nothing to propagate into, named
        return {"installed": False, "reason": f"{type(e).__name__}:{str(e)[:60]}"}
    if getattr(stock, "__wrapped_stock__", None) is not None:
        return {"installed": True, "reason": "already"}
    wrapped = functools.partial(worker_entry, mode=mode, det=int(det or 0), allow_partial=bool(allow_partial))
    wrapped.__wrapped_stock__ = stock                     # pickled by reference with the partial; the worker resolves it to its own stock entry
    wrapped.__qualname__ = f"{ATTR}[atlasfold_opt:workers]"
    setattr(mg, ATTR, wrapped)
    return {"installed": True, "reason": None}


def gpu_ids_of(stock_argv) -> list:
    """The ``--gpu-ids`` values of a stock argv ([] when absent) — for the parent's WORKERS line only; the stock parser stays the authority."""
    out, take = [], False
    for w in stock_argv:
        if w == "--gpu-ids":
            take = True; continue
        if take:
            if w.startswith("-"):
                break
            out.append(w)
    return out
