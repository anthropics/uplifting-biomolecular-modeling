"""Lever output_overlap (class exact, LOCAL.atlasfold.output_overlap) — the multimer runner's per-batch HOST work (``_make_outputs``: SASA
(numba) + Protein objects + confidence dicts per sample, the ranking statement, the writer's mmCIF/PDB text + confidence dict, and — when the
CLI gave ``--out-dir`` — the record's files themselves) done by ONE host-only worker PROCESS while the caller's thread runs the NEXT batch's
forward, instead of serialising forward -> outputs -> write -> forward.  The model forward never leaves the caller's (main) thread: featurize +
``model_run`` are the stock statements issued where stock issues them, and the kit process creates NO thread (a worker thread would contend with
the launch-bound LM / trunk loops for the interpreter; a separate process does not).

Site: ``atlasfold.runner_multimer.MultimerFoldingRunner.fold_iter_batch`` (runner_multimer.py:241-325), re-stated.  The stock generator
does, per bucketed batch: ``_make_batch_features`` -> per seed ``model_run`` (the forward; returns HOST numpy arrays) -> per complex
``_make_outputs`` -> rank -> ``yield`` (the caller — ``atlasfold.cli.multimer.run`` — writes the files before resuming the generator).  Here:

  caller's thread (the generator body): the stock preamble, then per batch the stock ``_make_batch_features`` and ``model_run(batch, seed=…)``
    statements in the stock batch / seed order; the batch's host arrays + its ``ProteinMultimer`` inputs are pickled (this thread, between two
    forwards) into the worker's scratch directory and announced on a pipe; every batch the worker has FINISHED is collected (unpickled on this
    thread), in the stock order, and yielded — so the CLI loop reaches batch k after forward k+1 instead of before it;
  worker (ONE ``multiprocessing`` spawn-context process ``atlasfold_opt.output_overlap``, started at install so its interpreter boot hides
    behind the model load; ``CUDA_VISIBLE_DEVICES=""`` and no kit word in its environment: a stock, device-less host process; nice +10): per
    batch, for every seed in the stock order, the same distogram slice and the same STOCK ``MultimerFoldingRunner._make_outputs(complex_input=…,
    out=…, batch_idx=…, num_samples=…, seed=…)`` calls into the same per-complex dicts, the same ranking statement, the same
    ``MultimerFoldingOutput(...)`` constructor (``post_batch`` below = runner_multimer.py:277-324 on host operands); then per sample the
    writer's stock ``to_mmcif(model="multimer")`` (or ``to_pdb``, the CLI's ``--format``) and ``confidence_scores``, kept ON the sample object
    (which travels back whole); then, when the CLI gave ``--out-dir``, every record's files written at once with the CLI's OWN nested
    ``write_outputs`` (cli/multimer.py ``run``, lifted from its code object: the same code, the same arguments) — ``early_write`` below;
  the CLI loop's writer: ``run``'s nested ``write_outputs`` code constant is swapped for a dispatcher (``guard_cli_loop`` -> ``loop_write``) so
    the loop SKIPS a record the worker already wrote (every record is written exactly once; the loop receives the return value the worker's call
    produced) and writes every other record with the stock statements (the run's last batch, stepped-aside batches, a run without
    ``--out-dir``), its ``to_mmcif`` / ``to_pdb`` / ``confidence_scores`` answered from the sample's precomputed text where present (the memo
    below).  When the constant cannot be swapped (``loop_write=stock:<reason>``) the loop re-writes worker-written records with the same bytes.

Why the bytes cannot change: every value-producing statement is the stock statement on the stock operands in the stock per-batch / per-seed
order; the forward and everything it reads stay on the caller's thread exactly as stock; ``_make_outputs`` / ranking / ``to_mmcif`` / the
writer are pure host functions of the forward's host arrays (numpy + numba SASA + gemmi; no RNG draw, no device), so running them in another
process with the same installed packages on a pickled copy of those arrays yields the same objects, the same text and the same files.  Numerics
class: exact — every output file byte-identical to stock under ``--det 1`` (``--det 1`` itself needs nothing here: host-only work, no draw).

Yield timing (the one observable difference besides speed): batch k is yielded after forward k+1 has been issued and returned (at most
``MAX_INFLIGHT`` batches are handed over and not yet collected; beyond that the caller blocks on the oldest).  Two policies keep the forward's
clock clean: (1) the run's LAST batch is never handed over — no forward follows it, so the caller computes it at once with the stock statements
(``last=1``) while the worker finishes the previous one; a single-record run (one batch: nothing to overlap) is therefore the stock body plus an
idle worker; (2) a hand-over waits for the worker's ``ready`` (``boot_wait_s``, normally 0: the boot hides behind the model load and the first
forward), so the worker's interpreter boot never runs beside a later forward.  The one statement that moves: batch k+1's
``ProteinMultimer.get_empty`` inputs (``_iter_batch``, a pure host constructor) are built one batch early (the look-ahead that tells the last
batch).  A forward that raises first yields every batch already computed (stock had yielded them before the failing forward began), then
re-raises.  No interaction with denoiser_graph / alloc_expandable (the worker has no device and allocates nothing on it; the caller's CUDA graphs
and allocator are untouched).

Steps aside BY NAME (counted on the LEVER line, stock body runs): ``AFO_OUTPUT_OVERLAP=0`` (``disabled``; no worker is started); the monomer
runner ``atlasfold.runner.FoldingRunner.fold_iter_batch`` (``runner:monomer`` — its generator yields a different structure and is not
re-stated); MODEL_OPT_LEVERS_OFF=output_overlap removes the lever before install (ablation.py).  Install refuses by name when the stock text's
digest is not the one re-stated here (``source:<digest>``) or when fold_iter_batch is already wrapped (``not_innermost:<qualname>``).
NOT expected (the lever's gate refuses at exit and the run exits non-zero; the files are still complete because the caller recomputes the batch
with the stock statements): ``worker:unavailable:<Type>`` (the process could not be started), ``worker:died[:exitcode=<n>]`` /
``worker:pipe:<Type>`` / ``worker:init:<summary>`` (the process became unusable), ``worker:submit:<Type>`` (a batch could not be handed over);
a worker-side exception in the stock statements is re-raised in the caller (``error:<Type>`` counted) with the worker's traceback attached —
never a silent pass.

LEVER line facts: served = batches (shape ``B<complexes>x<seeds>seed``), worker=process, worker_start, queued (batches handed to the worker),
last (the run's last batch, computed on the caller by design), inline (batches computed on the caller by a named step-aside), wait_s (caller
blocked on the worker = output work NOT hidden), boot_wait_s (of which: waiting for the worker's boot), early_by / early_written /
early_write_s / early_failed (who writes a record's files first — ``worker`` or ``loop:no_out_dir`` —, records the worker wrote with the CLI's
writer the moment their outputs existed, the seconds that took, the exception type if an early write failed), loop_write / written_once /
loop_written (the loop guard's state ``guarded`` | ``stock:<reason>``; records the loop skipped as already written; records it wrote itself),
post_s (worker compute seconds: outputs_s = _make_outputs + ranking, memo_s = the writer's text), overlapped_s (post_s - wait_s, >= 0: output
work hidden behind forwards), xfer_s (caller-side pickle + unpickle seconds), worker_io_s (worker-side unpickle + pickle), max_inflight,
memo_served (writer calls answered from the precomputed text), worker_boot_s / worker_rss_mib / worker_pss_mib / worker_uss_mib (the worker's
interpreter boot and peak resident / proportional / private memory; -1 where the kernel does not serve smaps_rollup), worker_dead /
worker_detail (why the worker became unusable, when it did), threads=0.  Memory: host only (<= MAX_INFLIGHT+1 batches of output arrays + the
worker process, whose resident set is mostly the file-backed pages of ``import torch`` it shares with the kit process); device peak unchanged
(one forward at a time, as stock; the worker holds no device)."""
from __future__ import annotations

import atexit
import contextlib
import itertools
import os
import pickle
import shutil
import sys
import tempfile
import time
import traceback
from collections import deque
from typing import Dict, Optional

from opt_core.counters import Ledger

from . import Installed, rebind

LEVER = "output_overlap"
NAME = "LOCAL.atlasfold.output_overlap"
ENV_SWITCH = "AFO_OUTPUT_OVERLAP"          # =0: the stock generator body, counted `disabled` (no worker process is started)
MAX_INFLIGHT = 2                           # batches handed to the worker and not yet collected before the caller blocks on the oldest (host memory bound)
WORKER_NICE = 10                           # the worker's niceness increment: the caller's launch thread wins any core it wants
WORKER_NAME = "atlasfold_opt.output_overlap"
MEMO_ATTR = "_atlasfold_opt_output_memo"   # (kind, model, text) kept on a ProteinMultimerOutput by the worker; read by the to_mmcif / to_pdb memo below
T_MULTI = "atlasfold.runner_multimer"
T_MONO = "atlasfold.runner"
SOURCE_SHA256 = {                          # opt_core.diffusion_loop.source_guard.source_sha256 of the re-stated text (atlasfold v1.0.0 as carried in stock/src)
    "MultimerFoldingRunner.fold_iter_batch": ("3366ba5f96b16a756feb7442e0b61378d69757eeb29e072130b8a1064b20ad6a",),
}
_WORKER: "Optional[_Worker]" = None       # the ONE worker of this interpreter (started at install, stopped at exit)
_JOB_IDS = itertools.count(1)              # job ids are unique per interpreter (a batch abandoned by an early close can never be mistaken for a later call's)


def enabled(env=None) -> bool:
    env = os.environ if env is None else env
    return str(env.get(ENV_SWITCH, "1")).strip().lower() not in ("0", "off", "no", "false")


def changed_sources() -> Dict[str, str]:
    """{fn: 'observed digest'} for every re-stated text whose digest is NOT in SOURCE_SHA256 (empty = all match)."""
    import importlib
    from opt_core.diffusion_loop.source_guard import source_sha256
    RM = importlib.import_module(T_MULTI)
    funcs = {"MultimerFoldingRunner.fold_iter_batch": RM.MultimerFoldingRunner.fold_iter_batch}
    bad = {}
    for fn, f in funcs.items():
        f = getattr(f, "__wrapped_stock__", f)
        try:
            d = source_sha256(f)
        except Exception as e:  # noqa: BLE001
            d = f"unreadable:{type(e).__name__}"
        if d not in SOURCE_SHA256[fn]:
            bad[fn] = d
    return bad


def _writer_format(argv=None) -> str:
    """The CLI writer's structure format (``atlasfold multimer --format cif|pdb``, default cif) — a HINT for the memo: a miss costs nothing but the stock call."""
    argv = list(sys.argv if argv is None else argv)
    for i, a in enumerate(argv):
        if a == "--format" and i + 1 < len(argv):
            return str(argv[i + 1])
        if isinstance(a, str) and a.startswith("--format="):
            return a.split("=", 1)[1]
    return "cif"


def write_opts(argv=None) -> dict:
    """The CLI writer's arguments from the argv of ``atlasfold multimer`` (``--out-dir`` — required by the CLI —, ``--format``,
    ``--save-confidence-arrays``, ``--save-distogram``): what the outputs worker needs to write a record's files the moment that record's
    outputs are ready.  {} without ``--out-dir`` (a programmatic ``fold_iter_batch``: the caller owns the writing; nothing is written early)."""
    argv = list(sys.argv if argv is None else argv)
    out = None
    for i, a in enumerate(argv):
        if a == "--out-dir" and i + 1 < len(argv):
            out = argv[i + 1]
        elif isinstance(a, str) and a.startswith("--out-dir="):
            out = a.split("=", 1)[1]
    if not out:
        return {}
    return {"out_dir": str(out), "format": _writer_format(argv), "save_confidence_arrays": "--save-confidence-arrays" in argv,
            "save_distogram": "--save-distogram" in argv, "writer": "stock"}


_WRITER_CACHE: Dict[str, object] = {}


def stock_writer():
    """The CLI's own ``write_outputs`` (cli/multimer.py, nested in ``run`` with no free variables): the SAME code object the CLI loop calls, bound
    as a plain function — so a record written here carries exactly the bytes the loop's own call would write.  Raises when the stock function is
    not liftable (renamed / gained free variables): the early write then steps aside by name and the loop writes as before."""
    import importlib
    import types
    M = importlib.import_module("atlasfold.cli.multimer")
    code = _GUARD.get("stock_code")                                  # the original, saved before the loop guard swapped run's constant (parent); None in the worker (pristine module)
    if code is None:
        for c in M.run.__code__.co_consts:
            if isinstance(c, types.CodeType) and c.co_name == "write_outputs" and "_afo_output_overlap_write" not in c.co_names:
                code = c
                break
    if code is None:
        raise LookupError("cli.multimer.run has no nested write_outputs")
    if code.co_freevars:
        raise ValueError(f"write_outputs closes over {code.co_freevars}")
    names = code.co_varnames[: code.co_argcount]
    if tuple(names[:3]) != ("out_dir", "output", "format"):
        raise ValueError(f"write_outputs signature {names}")
    fn = types.FunctionType(code, vars(M), "write_outputs", tuple(False for _ in names[3:]))   # trailing flags default False, as in the source
    if code.co_kwonlyargcount:
        fn.__kwdefaults__ = {k: False for k in code.co_varnames[code.co_argcount: code.co_argcount + code.co_kwonlyargcount]}
    return fn


def resolve_writer(spec):
    """``stock`` (default) -> :func:`stock_writer`; ``module:function`` -> that callable (scripted writers in tests travel by name)."""
    key = spec or "stock"
    fn = _WRITER_CACHE.get(key)
    if fn is None:
        if key == "stock":
            fn = stock_writer()
        else:
            import importlib
            mod, _, name = key.partition(":")
            fn = getattr(importlib.import_module(mod), name)
        _WRITER_CACHE[key] = fn
    return fn


def early_write(batch_outputs, write: dict) -> dict:
    """Write every record of the batch with the CLI's writer and arguments NOW (in the outputs worker, the moment the batch's outputs exist):
    ``<out_dir>/<record>/…`` exactly as the CLI loop writes them when the batch is yielded to it; the loop's guarded writer then skips these
    records (or, under ``loop_write=stock:<reason>``, re-writes the same bytes).  Never raises: {early_n, early_s, early_at, early_err, written}."""
    st = {"early_n": 0, "early_s": 0.0, "early_err": None, "written": {}}
    if not write or not write.get("out_dir"):
        return st
    import pathlib
    t = time.perf_counter()
    try:
        writer = resolve_writer(write.get("writer"))
        root = pathlib.Path(write["out_dir"])
        for fo in batch_outputs:
            rec = writer(root / fo.name, fo, write.get("format", "cif"), save_confidence_arrays=bool(write.get("save_confidence_arrays")),
                         save_distogram=bool(write.get("save_distogram")))
            st["early_n"] += 1
            st["written"][os.path.abspath(str(root / fo.name))] = rec        # the loop's guard skips this record (written once) and hands its caller this return value
    except Exception as e:  # noqa: BLE001 — the loop's own write follows regardless: named, never fatal
        st["early_err"] = type(e).__name__
    st["early_s"] = round(time.perf_counter() - t, 4)
    st["early_at"] = round(time.time(), 3)
    return st


_GUARD: Dict[str, object] = {"stock_code": None, "state": "stock:not_installed", "ledger": None}
WRITTEN: Dict[str, object] = {}                    # abspath(<out_dir>/<record>) -> the writer's return value, for records the worker already wrote (popped by the loop's guard)


def _cli_write_outputs(out_dir, output, format, save_confidence_arrays=False, save_distogram=False):   # noqa: A002 — the CLI's own parameter names
    # Runs AS the CLI loop's nested `write_outputs` (its code object replaces the stock constant in `run`): no free variables; the one global it
    # reads is injected into the CLI module's namespace by guard_cli_loop().
    return _afo_output_overlap_write(out_dir, output, format, save_confidence_arrays, save_distogram)  # noqa: F821


def loop_write(out_dir, output, format, save_confidence_arrays=False, save_distogram=False):   # noqa: A002
    """The CLI loop's write of one yielded record: SKIPPED when the outputs worker already wrote it (the same function and arguments, earlier —
    each record is written exactly once; the loop receives the writer's return value the worker recorded), the stock statements otherwise (the
    run's last batch, a batch the worker stepped aside from, a run without the worker)."""
    key = os.path.abspath(str(out_dir))
    L = _GUARD.get("ledger")
    if key in WRITTEN:
        rec = WRITTEN.pop(key)
        if L is not None:
            L.set("written_once", int(L.get("written_once", 0)) + 1)
        return rec
    if L is not None:
        L.set("loop_written", int(L.get("loop_written", 0)) + 1)
    return resolve_writer(write_opts().get("writer"))(out_dir, output, format, save_confidence_arrays=save_confidence_arrays, save_distogram=save_distogram)


def guard_cli_loop() -> str:
    """Make the CLI loop's nested `write_outputs` (atlasfold.cli.multimer.run) dispatch through :func:`loop_write`: the code-object constant of
    `run` is replaced by :func:`_cli_write_outputs`'s (same parameters and defaults, no free variables, the stock name and qualified name) and
    the dispatcher is injected into the module namespace.  Idempotent.  Returns the census word: ``guarded`` | ``stock:<reason>`` (the stock
    constant left in place: the loop then re-writes worker-written records with the same bytes, as before)."""
    import importlib
    import types
    try:
        M = importlib.import_module("atlasfold.cli.multimer")
        consts = list(M.run.__code__.co_consts)
        for c in consts:
            if isinstance(c, types.CodeType) and c.co_name == "write_outputs" and "_afo_output_overlap_write" in c.co_names:
                vars(M)["_afo_output_overlap_write"] = loop_write
                _GUARD["state"] = "guarded"; return "guarded"                                  # already swapped in this process
        idx = [i for i, c in enumerate(consts) if isinstance(c, types.CodeType) and c.co_name == "write_outputs"]
        if len(idx) != 1:
            raise LookupError(f"write_outputs constants in run: {len(idx)}")
        stock = consts[idx[0]]
        if stock.co_freevars:
            raise ValueError(f"write_outputs closes over {stock.co_freevars}")
        if tuple(stock.co_varnames[: stock.co_argcount]) != ("out_dir", "output", "format", "save_confidence_arrays", "save_distogram") or stock.co_kwonlyargcount:
            raise ValueError(f"write_outputs signature {stock.co_varnames[: stock.co_argcount]}")
        new = _cli_write_outputs.__code__
        kw = {"co_name": stock.co_name, "co_filename": stock.co_filename, "co_firstlineno": stock.co_firstlineno}
        if hasattr(stock, "co_qualname"):
            kw["co_qualname"] = stock.co_qualname
        new = new.replace(**kw)
        consts[idx[0]] = new
        vars(M)["_afo_output_overlap_write"] = loop_write
        _GUARD["stock_code"] = stock
        M.run.__code__ = M.run.__code__.replace(co_consts=tuple(consts))
        _GUARD["state"] = "guarded"
    except Exception as e:  # noqa: BLE001 — the stock constant stays: the loop re-writes the same bytes (named on the LEVER line)
        _GUARD["state"] = f"stock:{type(e).__name__}"
    return str(_GUARD["state"])


def _rss_mib() -> float:
    try:
        import resource
        return round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)   # Linux: KiB
    except Exception:  # noqa: BLE001
        return -1.0


def _mem_facts() -> dict:
    """This process's peak RSS and, where the kernel serves /proc/self/smaps_rollup, its proportional / private set (MiB): the worker's RSS is
    dominated by the file-backed pages of `import torch` it SHARES with the kit process; Pss / private (USS) is what it adds to the host."""
    out = {"rss_mib": _rss_mib(), "pss_mib": -1.0, "uss_mib": -1.0}
    try:
        priv = 0
        with open("/proc/self/smaps_rollup") as f:
            for line in f:
                k = line.split(":")[0]
                if k == "Pss":
                    out["pss_mib"] = round(int(line.split()[1]) / 1024.0, 1)
                elif k in ("Private_Clean", "Private_Dirty", "Private_Hugetlb"):
                    priv += int(line.split()[1])
        out["uss_mib"] = round(priv / 1024.0, 1)
    except Exception:  # noqa: BLE001
        pass
    return out


# ----------------------------------------------------------------------------------------------------------------- the batch's host statements
def post_batch(*, complexes, outs, num_samples, return_distogram, post_fn=None, fmt="cif", memo=True, write=None):
    """runner_multimer.py:277-324 for ONE bucketed batch given its forwards' host outputs ``outs = [(seed, out), ...]`` in the stock seed order:
    the stock ``_make_outputs`` per seed per complex into the per-complex dicts, the stock ranking, the stock ``MultimerFoldingOutput``; then
    (``memo``) per sample the writer's stock ``to_mmcif``/``to_pdb`` text and ``confidence_scores``, kept on the sample. Runs in the worker
    (or on the caller when the worker stepped aside, ``memo=False`` = exactly the stock body). Returns (batch_outputs, stats)."""
    import importlib
    RM = importlib.import_module(T_MULTI)
    MultimerFoldingOutput = RM.MultimerFoldingOutput
    if post_fn is None:
        post_fn = RM.MultimerFoldingRunner._make_outputs
    t0 = time.perf_counter()
    batch_size = len(complexes)
    model_outputs = [
        {} for _ in range(batch_size)
    ]
    batch_distogram_logits = [
        {} for _ in range(batch_size)
    ]
    distogram_boundaries = None
    for seed_value, out in outs:
        if return_distogram:
            distogram_boundaries = out["distogram.boundaries"]
        for batch_i, complex_input in enumerate(complexes):
            if return_distogram:
                length = complex_input.num_residues
                batch_distogram_logits[batch_i][seed_value] = out[
                    "distogram.logits"
                ][batch_i, :length, :length]
            model_outputs[batch_i].update(
                post_fn(
                    complex_input=complex_input,
                    out=out,
                    batch_idx=batch_i,
                    num_samples=num_samples,
                    seed=seed_value,
                )
            )
    batch_outputs = []
    for batch_i, outputs in enumerate(model_outputs):
        ranking = sorted(
            outputs.keys(), key=lambda k: outputs[k].ranking_score, reverse=True
        )
        batch_outputs.append(
            MultimerFoldingOutput(
                outputs,
                ranking,
                distogram_logits=batch_distogram_logits[batch_i],
                distogram_boundaries=distogram_boundaries,
            )
        )
    t1 = time.perf_counter()
    n_memo = 0
    if memo:
        for fo in batch_outputs:
            for sample in fo.outputs.values():
                n_memo += _precompute(sample, fmt)
    t2 = time.perf_counter()
    stats = {"outputs_s": round(t1 - t0, 4), "memo_s": round(t2 - t1, 4), "memo_n": n_memo, "pid": os.getpid()}
    stats.update(early_write(batch_outputs, write))                         # the record's files NOW (worker: while the caller forwards the next batch), not when the loop reaches the yield
    return batch_outputs, stats


def _precompute(sample, fmt: str) -> int:
    """The CLI writer's two per-sample computations (cli/multimer.py write_outputs: ``to_mmcif(model="multimer")`` | ``to_pdb(...)`` and
    ``confidence_scores``), issued here once with the writer's arguments and kept on the sample (MEMO_ATTR; cached_property __dict__ slot)."""
    kind = "mmcif" if fmt == "cif" else ("pdb" if fmt == "pdb" else None)
    n = 0
    if kind is not None:
        meth = getattr(type(sample), "to_" + kind, None)
        if meth is not None:
            meth = getattr(meth, "__wrapped_stock__", meth)          # on the caller the class carries the memo wrapper: the stock text either way
            sample.__dict__[MEMO_ATTR] = (kind, "multimer", meth(sample, model="multimer"))
            n = 1
    if hasattr(type(sample), "confidence_scores"):
        try:
            getattr(sample, "confidence_scores")                       # functools.cached_property: lands in sample.__dict__ and travels with it
        except Exception:  # noqa: BLE001
            pass
    return n


def make_memo_method(stock, kind: str, ledger: Ledger):
    """ProteinMultimerOutput.to_mmcif / to_pdb: the text the worker computed with these very arguments when present, else the stock call."""
    def method(self, model=None):
        m = self.__dict__.get(MEMO_ATTR)
        if m is not None and m[0] == kind and m[1] == model:
            ledger.count("memo_served")
            return m[2]
        return stock(self, model=model)
    method.__name__ = "to_" + kind
    method.__qualname__ = f"ProteinMultimerOutput.to_{kind}[atlasfold_opt:output_overlap:memo]"
    return method


# ----------------------------------------------------------------------------------------------------------------- the worker process (child side)
def _worker_main(conn, workdir: str, opts: dict) -> None:
    """Runs in the spawned interpreter. Host-only by construction: no device is visible (CUDA_VISIBLE_DEVICES=""), no kit word is armed
    (a stock atlasfold import), niced. Protocol (pipe messages are small; arrays travel through pickles in ``workdir``):
    <- ("job", id, path) | ("stop",)   -> ("ready", facts) | ("done", id, out_path, stats) | ("error", id, summary, traceback, exc|None) | ("init_error", summary, traceback)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("ATLASFOLD_OPT", None)
    try:
        import signal
        signal.signal(signal.SIGINT, signal.SIG_IGN)                    # the caller owns this process's lifetime (stop / terminate); a terminal ^C is the caller's to handle
    except Exception:  # noqa: BLE001
        pass
    sys.meta_path[:] = [f for f in sys.meta_path if not (type(f).__name__ == "_Finder" and (type(f).__module__ or "").startswith("atlasfold_opt"))]
    try:
        err = open(os.path.join(workdir, "worker.stderr"), "a", buffering=1)
        os.dup2(err.fileno(), 2)
        sys.stderr = err
        import faulthandler
        faulthandler.enable(err)
    except Exception:  # noqa: BLE001
        pass
    try:
        os.nice(int(opts.get("nice", 0)))
    except Exception:  # noqa: BLE001
        pass
    t0 = time.perf_counter()
    try:
        import importlib
        import warnings
        warnings.simplefilter("ignore")
        importlib.import_module("numpy")
        importlib.import_module(T_MULTI)                                # torch + the stock host code; nothing here touches a device
        conn.send(("ready", dict(_mem_facts(), pid=os.getpid(), boot_s=round(time.perf_counter() - t0, 2),
                                 cuda_visible=os.environ.get("CUDA_VISIBLE_DEVICES"), nice=opts.get("nice", 0))))
    except BaseException as e:  # noqa: BLE001
        try:
            conn.send(("init_error", f"{type(e).__name__}: {e}", traceback.format_exc()))
        finally:
            return
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            return
        if not msg or msg[0] == "stop":
            return
        if msg[0] != "job":
            continue
        _, job_id, path = msg[:3]
        t1 = time.perf_counter()
        try:
            with open(path, "rb") as f:
                job = pickle.load(f)
            t2 = time.perf_counter()
            batch_outputs, stats = post_batch(**job)
            t3 = time.perf_counter()
            out_path = path + ".out"
            with open(out_path, "wb") as f:
                pickle.dump(batch_outputs, f, protocol=pickle.HIGHEST_PROTOCOL)
            t4 = time.perf_counter()
            try:
                os.unlink(path)
            except OSError:
                pass
            stats.update(_mem_facts(), load_s=round(t2 - t1, 4), post_s=round(t3 - t2, 4), dump_s=round(t4 - t3, 4), wall_end=time.time())
            conn.send(("done", job_id, out_path, stats))
        except BaseException as e:  # noqa: BLE001 — reported to the caller, which re-raises it there
            tb = traceback.format_exc()
            try:
                exc = pickle.loads(pickle.dumps(e))
            except Exception:  # noqa: BLE001
                exc = None
            try:
                conn.send(("error", job_id, f"{type(e).__name__}: {e}", tb, exc))
            except Exception:  # noqa: BLE001
                return


@contextlib.contextmanager
def _spawn_env():
    """The worker interpreter's start conditions, set for the instant of the spawn on this (single, activating) thread and restored at once:
    no visible device, no kit word (the .pth autoload arms on ATLASFOLD_OPT at interpreter start), and — the worker needs nothing from
    ``__main__`` — multiprocessing kept from re-running an unguarded driver script as ``__mp_main__`` in the child."""
    keep = {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "ATLASFOLD_OPT")}
    main = sys.modules.get("__main__")
    spec, had_file, file_ = getattr(main, "__spec__", None), hasattr(main, "__file__"), getattr(main, "__file__", None)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("ATLASFOLD_OPT", None)
    try:
        if main is not None:
            try:
                main.__spec__ = None
                if had_file:
                    del main.__file__
            except Exception:  # noqa: BLE001
                pass
        yield
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        if main is not None:
            try:
                main.__spec__ = spec
                if had_file:
                    main.__file__ = file_
            except Exception:  # noqa: BLE001
                pass


# ----------------------------------------------------------------------------------------------------------------- the worker process (caller side)
class _Worker:
    """The caller-side handle of the ONE worker process: start / submit (pickle on the calling thread + a pipe message) / result (messages
    pumped on the calling thread only — no reader thread) / stop. Every wait is bounded by the worker's liveness."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.proc = None
        self.conn = None
        self.dir = None
        self.ready: Optional[dict] = None
        self.dead: Optional[str] = None            # reason word once the worker is unusable
        self.results: Dict[int, tuple] = {}
        self.cancelled = set()                     # job ids an early close abandoned: their results are dropped (and their files removed) on arrival

    def start(self) -> None:
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        self.dir = tempfile.mkdtemp(prefix="afo_output_overlap_")
        parent_conn, child_conn = ctx.Pipe(duplex=True)
        self.proc = ctx.Process(target=_worker_main, args=(child_conn, self.dir, {"nice": WORKER_NICE}), name=WORKER_NAME, daemon=True)
        with _spawn_env():
            self.proc.start()
        child_conn.close()
        self.conn = parent_conn
        atexit.register(self.stop)

    # -- messages (calling thread only)
    def _pump(self) -> None:
        if self.dead is not None or self.conn is None:
            return
        try:
            while self.conn.poll(0):
                msg = self.conn.recv()
                kind = msg[0]
                if kind == "ready":
                    self.ready = dict(msg[1]); self._mem(self.ready)
                    self.ledger.set("worker_boot_s", self.ready.get("boot_s"))
                elif kind in ("done", "error"):
                    if int(msg[1]) in self.cancelled:
                        self.cancelled.discard(int(msg[1]))
                        if kind == "done":
                            try:
                                os.unlink(msg[2])
                            except OSError:
                                pass
                        continue
                    self.results[int(msg[1])] = msg
                    if kind == "done":
                        self._mem(msg[3] or {})
                elif kind == "init_error":
                    self._die("init:" + str(msg[1]).split(":", 1)[0].strip()[:40], detail=msg[2])
                    return
        except (EOFError, OSError) as e:
            self._die(f"pipe:{type(e).__name__}")

    def _mem(self, facts: dict) -> None:
        """worker_rss_mib / worker_pss_mib / worker_uss_mib = the worker's peaks so far (-1: not served by this kernel)."""
        for k in ("rss_mib", "pss_mib", "uss_mib"):
            try:
                v = float(facts.get(k, -1.0))
            except Exception:  # noqa: BLE001
                v = -1.0
            self.ledger.peak("worker_" + k, v)

    def _die(self, reason: str, detail: str = "") -> None:
        if self.dead is None:
            code = None
            try:
                if self.proc is not None and not self.proc.is_alive():
                    code = self.proc.exitcode
            except Exception:  # noqa: BLE001
                pass
            self.dead = reason if code is None else f"{reason}:exitcode={code}"
            self.ledger.set("worker_dead", self.dead)
            if detail:
                self.ledger.set("worker_detail", detail.strip().splitlines()[-1][:160] if detail.strip() else "")

    def usable(self) -> bool:
        self._pump()
        if self.dead is None and (self.proc is None or not self.proc.is_alive()):
            self._die("died")
        return self.dead is None

    def wait_ready(self) -> bool:
        """Block until the worker's interpreter boot is done (its ``ready`` message) or it died: a hand-over never lets the boot's page-fault /
        import storm run beside a forward. Counted as ``boot_wait_s`` (normally 0: the boot hides behind the model load and the first forward)."""
        t = time.perf_counter()
        try:
            while self.usable() and self.ready is None:
                try:
                    self.conn.poll(0.5)
                except (EOFError, OSError) as e:
                    self._die(f"pipe:{type(e).__name__}")
            return self.dead is None
        finally:
            waited = time.perf_counter() - t
            if waited > 0.001:
                self.ledger.set("boot_wait_s", round(float(self.ledger.get("boot_wait_s", 0.0)) + waited, 3))

    def submit(self, job_id: int, payload: dict) -> str:
        path = os.path.join(self.dir, f"job{job_id:05d}.pkl")
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
        self.conn.send(("job", int(job_id), path))
        return path

    def result(self, job_id: int, block: bool):
        """("done", id, out_path, stats) | ("error", id, summary, tb, exc) | ("dead", id, reason) | None (not ready, block=False)."""
        waited = 0.0
        try:
            while True:
                self._pump()
                if job_id in self.results:
                    return self.results.pop(job_id)
                if self.dead is not None:
                    return ("dead", job_id, self.dead)
                if not block:
                    return None
                t = time.perf_counter()
                try:
                    got = self.conn.poll(1.0)
                except (EOFError, OSError) as e:
                    self._die(f"pipe:{type(e).__name__}"); got = False
                waited += time.perf_counter() - t
                if not got and self.dead is None and not self.proc.is_alive():
                    self._pump()                                       # a last message may still sit in the pipe
                    if job_id in self.results:
                        return self.results.pop(job_id)
                    self._die("died")
        finally:
            if waited:
                self.ledger.set("wait_s", round(float(self.ledger.get("wait_s", 0.0)) + waited, 3))

    def worker_stderr_tail(self, n: int = 12) -> str:
        try:
            with open(os.path.join(self.dir, "worker.stderr"), "r", errors="replace") as f:
                return "".join(f.readlines()[-n:])
        except OSError:
            return ""

    def stop(self) -> None:
        proc, conn, self.proc, self.conn = self.proc, self.conn, None, None
        try:
            if conn is not None:
                try:
                    conn.send(("stop",))
                except Exception:  # noqa: BLE001
                    pass
            if proc is not None:
                proc.join(timeout=5.0)
                if proc.is_alive():
                    proc.terminate(); proc.join(timeout=5.0)
        except Exception:  # noqa: BLE001
            pass
        finally:
            if conn is not None:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            if self.dir and os.path.isdir(self.dir):
                shutil.rmtree(self.dir, ignore_errors=True)


def start_worker(ledger: Ledger) -> "Optional[_Worker]":
    """Start (or return) this interpreter's worker. A failure to start is recorded by name on the ledger and returns None."""
    global _WORKER
    if _WORKER is not None and _WORKER.dead is None and _WORKER.proc is not None and _WORKER.proc.is_alive():
        _WORKER.ledger = ledger
        return _WORKER
    w = _Worker(ledger)
    try:
        w.start()
    except Exception as e:  # noqa: BLE001
        ledger.set("worker_dead", f"unavailable:{type(e).__name__}")
        w.dead = f"unavailable:{type(e).__name__}"
        try:
            w.stop()
        except Exception:  # noqa: BLE001
            pass
        _WORKER = w
        return None
    _WORKER = w
    return w


def stop_worker() -> None:
    global _WORKER
    if _WORKER is not None:
        _WORKER.stop()
    _WORKER = None


# ----------------------------------------------------------------------------------------------------------------- the pipeline (caller side)
class _Pipeline:
    """One fold_iter_batch call's book-keeping on the caller's thread: batches handed to the worker (or marked for the caller), collected in order."""

    def __init__(self, worker: "Optional[_Worker]", ledger: Ledger, *, post_fn, fmt: str, num_samples: int, return_distogram: bool, write: dict = None):
        self.worker, self.ledger, self.post_fn, self.fmt = worker, ledger, post_fn, fmt
        self.write = dict(write or {})                                      # the CLI writer's arguments: batches handed to the worker are written there as soon as their outputs exist
        self.num_samples, self.return_distogram = int(num_samples), bool(return_distogram)
        self.pending = deque()
        self.n = 0
    def _acc(self, key: str, seconds: float) -> None:
        self.ledger.set(key, round(float(self.ledger.get(key, 0.0)) + float(seconds), 3))

    def submit(self, complexes, outs, *, last: bool = False) -> None:
        self.n = next(_JOB_IDS)
        entry = {"id": self.n, "complexes": complexes, "outs": outs, "where": "caller", "names": "+".join(getattr(c, "name", "?") for c in complexes)}
        w = self.worker
        if last:                                                        # the run's last batch: no forward follows it, so the caller computes it now
            entry["result"] = self._inline(entry, word="last")          # (the stock statements, this thread) while the worker finishes the previous one
            self.pending.append(entry)
            return
        if w is not None and w.usable() and w.wait_ready():
            t = time.perf_counter()
            try:
                w.submit(self.n, dict(complexes=complexes, outs=outs, num_samples=self.num_samples, return_distogram=self.return_distogram,
                                       post_fn=self.post_fn, fmt=self.fmt, memo=True, write=self.write or None))
                entry["where"] = "worker"
                self.ledger.count("queued")
            except Exception as e:  # noqa: BLE001 — an unpicklable operand / a broken pipe: this batch on the caller, by name
                self.ledger.fallback(f"worker:submit:{type(e).__name__}")
            self._acc("xfer_s", time.perf_counter() - t)
        elif w is None:
            self.ledger.fallback(f"worker:{_WORKER.dead}" if _WORKER is not None and _WORKER.dead else "worker:unavailable")
        else:
            self.ledger.fallback(f"worker:{w.dead or 'died'}")
        self.pending.append(entry)
        self.ledger.peak("max_inflight", sum(1 for e in self.pending if e["where"] == "worker"))

    def _inline(self, entry, word: str = "inline"):
        """The batch's stock host statements on THIS (the caller's) thread — the stock body for that batch. ``word``: ``last`` (the run's last
        batch, by design) or ``inline`` (a worker that stepped aside, by name)."""
        t = time.perf_counter()
        batch_outputs, _stats = post_batch(complexes=entry["complexes"], outs=entry["outs"], num_samples=self.num_samples,
                                            return_distogram=self.return_distogram, post_fn=self.post_fn, fmt=self.fmt, memo=False)
        self._acc(word + "_s", time.perf_counter() - t)
        self.ledger.count(word)
        return batch_outputs

    def _serve(self, entry):
        self.ledger.serve(f"B{len(entry['complexes'])}x{len(entry['outs'])}seed")

    def drain(self, *, block: bool):
        """Yield every finished batch in the submitted order. block=False: stop at the first batch not finished yet unless more than
        MAX_INFLIGHT are in flight (then wait for the oldest); block=True: wait for all."""
        while self.pending:
            head = self.pending[0]
            if head["where"] == "caller":
                self.pending.popleft()
                out = head.pop("result", None)
                if out is None:
                    out = self._inline(head)
                self._serve(head)
                yield out
                continue
            inflight = sum(1 for e in self.pending if e["where"] == "worker")
            r = self.worker.result(head["id"], block=block or inflight > MAX_INFLIGHT)
            if r is None:
                return
            self.pending.popleft()
            kind = r[0]
            if kind == "done":
                t = time.perf_counter()
                with open(r[2], "rb") as f:
                    batch_outputs = pickle.load(f)
                try:
                    os.unlink(r[2])
                except OSError:
                    pass
                self._acc("xfer_s", time.perf_counter() - t)
                stats = r[3] or {}
                self._acc("post_s", float(stats.get("post_s", 0.0)))
                self._acc("outputs_s", float(stats.get("outputs_s", 0.0)))      # of which: _make_outputs + ranking (the stock generator's host statements) ...
                self._acc("memo_s", float(stats.get("memo_s", 0.0)))            # ... and the writer's per-sample text + confidence dict
                self._acc("worker_io_s", float(stats.get("load_s", 0.0)) + float(stats.get("dump_s", 0.0)))
                self._acc("early_write_s", float(stats.get("early_s", 0.0)))    # the records' files written in the worker the moment their outputs existed
                self.ledger.set("early_written", int(self.ledger.get("early_written", 0)) + int(stats.get("early_n", 0) or 0))
                WRITTEN.update(stats.get("written") or {})                     # the loop's guard skips these records (written once, by the worker)
                if stats.get("early_err"):
                    self.ledger.set("early_failed", str(stats["early_err"]))
                self._serve(head)
                yield batch_outputs
            elif kind == "error":
                self.ledger.error(r[2].split(":", 1)[0] if isinstance(r[2], str) else "worker")
                exc = r[4] if len(r) > 4 and isinstance(r[4], BaseException) else RuntimeError(f"output_overlap worker: {r[2]}")
                note = f"[atlasfold_opt output_overlap] raised in the output worker process on batch {head['names']}; worker traceback:\n{r[3]}"
                if hasattr(exc, "add_note"):
                    exc.add_note(note)
                else:
                    exc = RuntimeError(f"{r[2]}\n{note}")
                raise exc
            else:                                                          # the worker died with this batch in hand: the stock statements here, named
                self.ledger.fallback(f"worker:{r[2]}")
                out = self._inline(head)
                self._serve(head)
                yield out

    def settle(self) -> None:
        """overlapped_s = worker compute that did not block the caller (post_s - wait_s, floored at 0)."""
        post, wait = float(self.ledger.get("post_s", 0.0)), float(self.ledger.get("wait_s", 0.0))
        self.ledger.set("overlapped_s", round(max(0.0, post - wait), 3))

    def cancel(self) -> None:
        """The caller went away (close / an error it did not survive): forget the pending batches; results that still arrive are dropped by id."""
        for e in list(self.pending):
            if e["where"] == "worker" and self.worker is not None:
                r = self.worker.results.pop(e["id"], None)
                if r is None:
                    self.worker.cancelled.add(e["id"])
                elif r[0] == "done":
                    try:
                        os.unlink(r[2])
                    except OSError:
                        pass
        self.pending.clear()


def _note_batch(complexes, bucket_length) -> None:
    """Tell the kit's per-item PHASE timer (phase_timing, when installed) which batch the next forwards belong to."""
    try:
        from .. import phase_timing as PT
        PT.note_batch(complexes, bucket_length)
    except Exception:  # noqa: BLE001 — timing is report-only: never in the way of the run
        pass


def make_fold_iter_batch(stock, ledger: Ledger, RM):
    """The re-stated MultimerFoldingRunner.fold_iter_batch (runner_multimer.py:241-325): forwards on the caller's thread in the stock order, the
    batch's host statements in the worker process, yields in the stock order. `stock` is the innermost stock generator function."""
    SamplingConfig = RM.SamplingConfig

    def fold_iter_batch(self, inputs, *, num_samples: int = 5, seeds=1, num_recycles: int = 10, sampling_config=None, mlm_prob: float = 0.20,
                        length_buckets=None, max_tokens_per_batch: int = 1024, return_distogram: bool = False):
        if not enabled():
            ledger.fallback("disabled")
            yield from stock(self, inputs, num_samples=num_samples, seeds=seeds, num_recycles=num_recycles, sampling_config=sampling_config,
                             mlm_prob=mlm_prob, length_buckets=length_buckets, max_tokens_per_batch=max_tokens_per_batch, return_distogram=return_distogram)
            return
        # ---- the stock preamble, verbatim (runner_multimer.py:254-270)
        seeds = [seeds] if isinstance(seeds, int) else list(seeds)

        if len(inputs) == 0:
            raise ValueError("No inputs provided.")
        if len(seeds) == 0:
            raise ValueError("No seeds provided.")
        if num_samples <= 0:
            raise ValueError(f"num_samples must be positive, got {num_samples}.")
        if max_tokens_per_batch <= 0:
            raise ValueError(
                f"max_tokens_per_batch must be positive, got {max_tokens_per_batch}."
            )
        if sampling_config is None:
            sampling_config = SamplingConfig(num_steps=200)

        normalized_inputs = self._normalize_inputs(inputs)
        bucketed_inputs = self._bucket_inputs(normalized_inputs, length_buckets)

        # ---- the pipeline: forwards HERE (the caller's thread), the batch's host statements in the worker process, yields in the stock order
        ledger.set("worker", "process"); ledger.set("threads", 0)
        worker = start_worker(ledger)
        wo = write_opts()
        ledger.set("early_by", "worker" if wo else "loop:no_out_dir")          # who writes a record's files first: the worker at outputs-ready, or (no CLI --out-dir) the caller's loop as before
        pipe = _Pipeline(worker, ledger, post_fn=self._make_outputs, fmt=_writer_format(), num_samples=num_samples, return_distogram=return_distogram, write=wo)
        try:
            batches = iter(self._iter_batch(
                bucketed_inputs, max_tokens_per_batch
            ))
            nxt = next(batches, None)
            while nxt is not None:
                (bucket_length, complexes), nxt = nxt, next(batches, None)     # one batch of look-ahead: the run's last batch is known when it comes
                _note_batch(complexes, bucket_length)                           # the PHASE / PEAK label = THIS batch (the look-ahead just advanced _iter_batch to the next one)
                batch = self._make_batch_features(complexes, bucket_length)
                outs = []
                for seed_value in seeds:
                    out = self.model_run(
                        batch,
                        seed=seed_value,
                        num_samples=num_samples,
                        num_recycles=num_recycles,
                        mlm_prob=mlm_prob,
                        sampling_config=sampling_config,
                        return_distogram=return_distogram,
                    )
                    outs.append((seed_value, out))
                pipe.submit(complexes, outs, last=nxt is None)
                for batch_outputs in pipe.drain(block=False):
                    yield batch_outputs
            for batch_outputs in pipe.drain(block=True):
                yield batch_outputs
        except GeneratorExit:
            pipe.cancel()
            raise
        except Exception:
            # a forward (or featurization) raised: the caller first receives every batch already computed, in order (stock had yielded them
            # before the failing forward began), then the exception
            try:
                for batch_outputs in pipe.drain(block=True):
                    yield batch_outputs
            except GeneratorExit:
                pipe.cancel()
                raise
            except Exception as e2:  # noqa: BLE001
                ledger.set("drain_after_error", type(e2).__name__)
            raise
        finally:
            pipe.cancel()
            pipe.settle()
            if worker is not None:
                worker._pump()                                          # the worker's boot / memory facts reach the LEVER line even when nothing was handed over
    fold_iter_batch.__qualname__ = "MultimerFoldingRunner.fold_iter_batch[atlasfold_opt:output_overlap]"
    return fold_iter_batch


def make_monomer_passthrough(stock, ledger: Ledger):
    """FoldingRunner.fold_iter_batch: the stock generator, counted `runner:monomer` (only the multimer runner is re-stated)."""
    def fold_iter_batch(self, *a, **k):
        ledger.fallback("runner:monomer")
        return stock(self, *a, **k)
    fold_iter_batch.__qualname__ = "FoldingRunner.fold_iter_batch[atlasfold_opt:output_overlap:stock_by_name]"
    return fold_iter_batch


def uninstall(stop: bool = True) -> None:
    """Put the stock attributes back (and stop the worker unless ``stop=False``); used by the kit's unit tests."""
    import importlib
    RM = importlib.import_module(T_MULTI)
    R = importlib.import_module(T_MONO)
    for owner, attr in ((RM.MultimerFoldingRunner, "fold_iter_batch"), (R.FoldingRunner, "fold_iter_batch")):
        f = getattr(owner, attr, None)
        if f is not None and hasattr(f, "__wrapped_stock__"):
            setattr(owner, attr, f.__wrapped_stock__)
    for attr in ("to_mmcif", "to_pdb"):
        if attr in vars(RM.ProteinMultimerOutput) and hasattr(vars(RM.ProteinMultimerOutput)[attr], "__wrapped_stock__"):
            delattr(RM.ProteinMultimerOutput, attr)
    if stop:
        stop_worker()


def install(mode: str, tag: str, ctx: dict) -> Installed:
    import importlib
    try:
        RM = importlib.import_module(T_MULTI)
        R = importlib.import_module(T_MONO)
    except Exception as e:  # noqa: BLE001
        return Installed(LEVER, False, reason=f"{type(e).__name__}: {e}")
    cls = RM.MultimerFoldingRunner
    stock = cls.fold_iter_batch
    if hasattr(stock, "__wrapped_stock__") or "[atlasfold_opt:" in getattr(stock, "__qualname__", ""):
        return Installed(LEVER, False, reason=f"not_innermost:{getattr(stock, '__qualname__', '?')}")
    bad = changed_sources()
    if bad:
        return Installed(LEVER, False, reason="source:" + ",".join(f"{k}={v[:12]}" for k, v in bad.items()))
    from ..registry import LEVERS
    words = tuple(LEVERS[LEVER]["expected"]) if LEVER in LEVERS else ("disabled", "runner:monomer")
    ledger = Ledger(NAME, impl=f"pipeline:caller-forward+worker-process-outputs(inflight<={MAX_INFLIGHT})", origin="kit", expected=words, max_shapes=8)
    for k in ("queued", "last", "inline", "wait_s", "boot_wait_s", "post_s", "xfer_s", "overlapped_s", "max_inflight", "early_written", "early_write_s", "written_once", "loop_written"):
        ledger.set(k, 0)
    _GUARD["ledger"] = ledger
    ledger.set("loop_write", guard_cli_loop())                             # the CLI loop skips records the worker wrote (each record written once): guarded | stock:<reason>
    ledger.set("worker", "process"); ledger.set("threads", 0)
    rebind(cls, "fold_iter_batch", make_fold_iter_batch(stock, ledger, RM), stock)
    mono_stock = R.FoldingRunner.fold_iter_batch
    if not hasattr(mono_stock, "__wrapped_stock__"):
        rebind(R.FoldingRunner, "fold_iter_batch", make_monomer_passthrough(mono_stock, ledger), mono_stock)
    PMO = RM.ProteinMultimerOutput                                    # the writer's to_mmcif / to_pdb on the SUBCLASS: the worker's text when it computed it with these arguments
    for kind in ("mmcif", "pdb"):
        attr = "to_" + kind
        if attr not in vars(PMO):
            inherited = getattr(PMO, attr)
            rebind(PMO, attr, make_memo_method(inherited, kind, ledger), inherited)
    started = "not_started:disabled"
    if enabled():                                                     # the worker boots now (torch + atlasfold host imports, seconds) — behind the model load, not behind the first item
        w = start_worker(ledger)
        started = "started" if w is not None else f"unavailable:{_WORKER.dead if _WORKER is not None else '?'}"
    ledger.set("worker_start", started)

    def gate():
        return ledger.gate(require_served=False)       # a monomer-only or disabled run is a legitimate stock run of this statement
    return Installed(LEVER, True, lines=[lambda: ledger.line(tag)], gates=[gate],
                     facts={"ledger": ledger, "max_inflight": MAX_INFLIGHT, "enabled": enabled(), "worker": (lambda: _WORKER)})
