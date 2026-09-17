"""writer_overlap — the tree's own lever on upstream's result dump (exact class: the same bytes, written off the critical path).

Stock OpenDDE (the pin) writes every item's outputs INSIDE the item loop, between one item's forward and the next item's featurization:
``runner/inference.py:1790-1802`` calls ``runner.dumper.DataDumper.dump`` (``runner/dumper.py:93-123`` -> ``dump_predictions`` :137-204: per
sample one CIF through ``opendde.data.utils.save_structure_cif``, one ``summary_confidence`` JSON and — ``--need_atom_confidence true``,
the pin's default — one ``full_data`` JSON of the token x token PAE / contact maps through ``opendde.utils.file_io.save_json``, i.e. the
standard library's pure-Python ``json.dump`` encoder over ``tolist()``-ed arrays). That is host work (a device->host copy per confidence leaf,
``numpy.round``, JSON encoding, CIF formatting) of seconds per item that the GPU waits out — a material fraction of the item's forward,
growing with the token count.

The lever: ``DataDumper.dump`` becomes :func:`dump` — in the caller's thread it takes ONE owned host copy of the prediction tree
(``opt_core.host.outputs.to_host``: every CUDA tensor of ``pred_dict`` copied to a pageable host tensor in one pass with one stream
synchronisation; ``atom_array.copy()``; ``dict(entity_poly_type)``) and hands ``(dumper, names, host tree)`` to ONE background writer thread
(``opt_core.host.outputs.AsyncWriter``, mode ``thread``, ``workers=1``, ``max_pending=2``: at most two items' host trees outstanding, the
producer blocks in ``submit`` beyond that — host memory is bounded by the constant, not by the query), which runs upstream's OWN ``dump`` on
them: the same function on the same values, the tensors merely already on the host (every conversion on the path — ``.float()`` of a bf16
leaf, ``.cpu().numpy()``, ``numpy.round``, ``tolist()``, the ranking ``argsort`` on a CPU tensor — is device-independent), so every output
file is byte-identical to the stock dump BY CONSTRUCTION; item k's files are written while item k+1 featurizes and runs. Ordering and
publication are upstream's (one item at a time, in submission order, through upstream's staging directory + atomic rename). The writer is
DRAINED — every pending write waited for — in ``InferenceRunner.close`` (``runner/inference.py:1190``, which upstream's ``main`` runs in its
``finally`` before the process can exit and before the kit indexes the outputs, ``cli.py`` ``outputs.index_cli``), so no reader ever sees a
directory the writer has not finished; a write that raised is printed at once from the writer thread (``WRITER FAILED item=…``), reported in
upstream's own per-item error file (``<out>/ERR/<item>.txt``, as the stock loop does for a dump failure, ``inference.py:1826-1844``),
counted (LEVER row ``failed=``, ``levers_fallback`` -> PARTIAL), and re-raised from ``close`` as upstream's own end-of-run ``RuntimeError`` —
never silent, never a zero exit; the core's exit guard (``opt_core.host.outputs``: an undrained writer at interpreter exit drains itself and
forces exit status 70 on a lost write) stands behind a process that skips ``close``.

What is NOT moved: the device->host copy (it stays in the caller's thread: one pass instead of upstream's per-leaf copies, the same bytes),
the model, the allocator. The payload holds no CUDA tensor (checked; an item whose tree cannot be copied is written synchronously by the
stock path, by name: ``aside=<reason>``) and the host copies are pageable, not pinned, so the writer thread never issues a CUDA call — it
composes with a graph-capturing sampler in the main thread (``stepgraph`` captures ``thread_local`` anyway). GIL: the writer's work is
Python (``json``'s encoder, biotite's CIF writer); the forward's launches release the GIL inside every ATen call, so the contention is the
interpreter's 5 ms switch interval against a launch queue that runs ahead. Every line carries
the lever, the row-sharded ``--n_gpu P>1`` line included (every rank's writer binds it; rank 0 writes the run's documents).
STATS: installed, items (dump calls), submitted, written, failed, sync (items written synchronously by a named aside), d2h_s, submit_s
(incl. back-pressure waits), write_s (background), drain_s, bytes.
"""
from __future__ import annotations

import functools
import os
import sys
import threading
import time

RUNNER_MODULE = "runner.inference"
DUMPER_MODULE = "runner.dumper"
DUMP_ATTR = ("DataDumper", "dump")                                   # runner/dumper.py:93 — the one output call of the item loop (inference.py:1792)
CLOSE_ATTR = ("InferenceRunner", "close")                            # runner/inference.py:1190 — upstream's main() runs it in `finally`: the drain point
ERROR_REPORT = "_append_error_report"                                # runner/inference.py:172 — upstream's per-item error file writer (<error_dir>/<item>.txt)
DUMP_PARAMS = ("group_name", "pdb_id", "seed", "pred_dict", "atom_array", "entity_poly_type")   # dumper.py:93-101: the pin's signature (a drift is an aside by name)
MAX_PENDING = 2                                                      # items whose host trees may be outstanding (the producer blocks beyond): host RAM bound
MODE = "thread"                                                      # opt_core.host.outputs.AsyncWriter mode: one thread in this process (no pickling, no child interpreter)
MARK = "_opendde_opt_writer_overlap"
STATS = {"installed": False, "armed": False, "items": 0, "submitted": 0, "written": 0, "failed": 0, "sync": 0,
         "d2h_s": 0.0, "submit_s": 0.0, "write_s": 0.0, "queued_s": 0.0, "drain_s": 0.0, "drains": 0, "bytes": 0, "leaves": 0,
         "aside": {}, "failures": [], "missing": [], "errors": 0}
_ST = {"orig_dump": None, "orig_close": None, "dumper_cls": None, "runner_cls": None, "writer": None, "raised": False, "module": None}
_OURS: list = []                                                     # the wrapper functions this module installed (identity, not a copied attribute: functools.wraps
_LOCK = threading.Lock()                                             # copies __dict__ marks onto any later wrapper of ours, so a mark alone cannot say who is underneath)


def _ours(fn) -> bool:
    """Is one of this module's wrappers ``fn`` or anywhere down its ``__wrapped__`` chain (another unit may have wrapped on top)?"""
    for _ in range(16):
        if fn is None:
            return False
        if any(fn is w for w in _OURS):
            return True
        fn = getattr(fn, "__wrapped__", None)
    return False


class ActivationError(RuntimeError):
    """The core's host-outputs primitive is not importable: the kit's activation fails by name."""


def _emit(line: str, err: bool = False) -> None:
    if os.environ.get("RANK", "0") in ("0", ""):
        print(line, file=(sys.stderr if err else sys.stdout), flush=True)


def _prefix() -> str:
    try:
        from .report import PREFIX
        return PREFIX
    except Exception:  # noqa: BLE001
        return "[opendde-opt]"


# ------------------------------------------------------------------------------------------------------------------ the host copy
def _cuda_leaves(tree) -> int:
    """Number of CUDA tensors left in a tree (the payload's post-condition: zero)."""
    torch = sys.modules.get("torch")
    if torch is None:
        return 0
    n = 0
    stack = [tree]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            stack.extend(x.values())
        elif isinstance(x, (list, tuple)):
            stack.extend(x)
        elif isinstance(x, torch.Tensor) and x.device.type != "cpu":
            n += 1
    return n


def host_tree(pred_dict):
    """(host copy of ``pred_dict``, bytes copied, device leaves) — one pass, one synchronisation, pageable host tensors (the core's to_host,
    no pinned pool: the writer thread must never own a CUDA resource)."""
    from opt_core.host.outputs import to_host
    census = {}
    out = to_host(pred_dict, stats=census)
    return out, int(census.get("bytes") or 0), int(census.get("leaves") or 0)


# ------------------------------------------------------------------------------------------------------------------ the writer
def _write(item) -> None:
    """The background thread's body: upstream's own dump on the host tree."""
    orig, dumper, args, t_sub = item
    t0 = time.perf_counter()
    name, seed = args[1], args[2]
    try:
        orig(dumper, *args)
    except BaseException as e:
        with _LOCK:
            STATS["failed"] += 1
            STATS["failures"].append((str(name), seed, f"{type(e).__name__}: {e}"))
        _emit(f"{_prefix()} WRITER FAILED item={name} seed={seed} {type(e).__name__}: {e}", err=True)
        raise
    finally:
        dt = time.perf_counter() - t0
        with _LOCK:
            STATS["write_s"] += dt
            STATS["queued_s"] += max(0.0, t0 - t_sub)
    with _LOCK:
        STATS["written"] += 1
    _emit(f"{_prefix()} WRITER-DONE item={name} seed={seed} write_s={dt:.3f} queued_s={max(0.0, t0 - t_sub):.3f}")


def _writer():
    """The process's one AsyncWriter (created at the first dump; the core's exit guard registered with it)."""
    w = _ST["writer"]
    if w is None:
        from opt_core.host.outputs import AsyncWriter
        w = AsyncWriter(_write, workers=1, max_pending=MAX_PENDING, name="writer_overlap", copy_on_submit=False, mode=MODE)
        try:
            w.register_exit_tally(_prefix().strip("[]"))
        except Exception:  # noqa: BLE001 — the exit tally is evidence, not function
            pass
        _ST["writer"] = w
    return w


def _aside(reason: str, name) -> None:
    with _LOCK:
        STATS["sync"] += 1
        STATS["aside"][reason] = STATS["aside"].get(reason, 0) + 1
    _emit(f"{_prefix()} WRITER item={name} written synchronously by the stock path: aside={reason}")


def _wrap_dump(orig):
    @functools.wraps(orig)
    def dump(self, *args, **kwargs):
        """DataDumper.dump under lever writer_overlap: host copy here, upstream's dump on it in the writer thread."""
        with _LOCK:
            STATS["items"] += 1
        rest = DUMP_PARAMS[len(args):] if len(args) <= len(DUMP_PARAMS) else None
        if rest is None or set(kwargs) != set(rest):                 # the pin's signature moved: this call is upstream's, synchronously, by name
            _aside("signature", kwargs.get("pdb_id", args[1] if len(args) > 1 else "?"))
            return orig(self, *args, **kwargs)
        args = tuple(args) + tuple(kwargs[k] for k in rest)
        group_name, name, seed, pred_dict, atom_array, entity_poly_type = args
        t0 = time.perf_counter()
        try:
            host_pred, nbytes, leaves = host_tree(pred_dict)
            aa = atom_array.copy() if hasattr(atom_array, "copy") else atom_array
            ept = dict(entity_poly_type) if entity_poly_type is not None else entity_poly_type
        except Exception as e:  # noqa: BLE001 — a tree that cannot be copied is written by the stock path, named
            _aside(f"host_copy:{type(e).__name__}", name)
            return orig(self, *args)
        if _cuda_leaves(host_pred):                                  # post-condition: the writer thread owns no CUDA tensor
            _aside("cuda_leaf", name)
            return orig(self, *args)
        t1 = time.perf_counter()
        w = _writer()
        w.submit((orig, self, (group_name, name, seed, host_pred, aa, ept), t1))
        t2 = time.perf_counter()
        with _LOCK:
            STATS["submitted"] += 1
            STATS["d2h_s"] += t1 - t0
            STATS["submit_s"] += t2 - t1
            STATS["bytes"] += nbytes
            STATS["leaves"] += leaves
        _emit(f"{_prefix()} WRITER item={name} seed={seed} d2h_s={t1 - t0:.3f} submit_s={t2 - t1:.3f} host_mib={nbytes / 2**20:.1f} "
              f"leaves={leaves} pending={w.pending()}")
        return None
    setattr(dump, MARK, True)
    _OURS.append(dump)
    return dump


def drain(where: str = "drain") -> list:
    """Wait for every submitted write; returns the failures so far [(item, seed, 'Type: msg'), ...]. Idempotent, cheap when idle."""
    w = _ST["writer"]
    if w is None:
        return list(STATS["failures"])
    t0 = time.perf_counter()
    try:
        w.drain()
    except RuntimeError:                                             # the failures are in STATS (counted in _write); raised by the close wrapper in upstream's words
        pass
    except Exception:  # noqa: BLE001
        with _LOCK:
            STATS["errors"] += 1
    dt = time.perf_counter() - t0
    with _LOCK:
        STATS["drain_s"] += dt
        STATS["drains"] += 1
        t = (STATS["submitted"], STATS["written"], STATS["failed"])
    _emit(f"{_prefix()} WRITER DRAIN at={where} waited_s={dt:.3f} submitted={t[0]} written={t[1]} failed={t[2]}")
    return list(STATS["failures"])


def _report_failures(runner, fails: list) -> None:
    """Upstream's own per-item error file for every failed write (inference.py:1836-1841 does this for a synchronous dump failure)."""
    mod = _ST["module"] or sys.modules.get(RUNNER_MODULE)
    rep = getattr(mod, ERROR_REPORT, None) if mod is not None else None
    err_dir = getattr(runner, "error_dir", None)
    if rep is None or not err_dir:
        return
    for name, seed, msg in fails:
        try:
            rep(err_dir, f"{name}.txt", f"[writer_overlap] output stage failed for {name} [seed:{seed}]: {msg}")
        except Exception:  # noqa: BLE001 — the report is best effort; the raise below is the guarantee
            pass


def _wrap_close(orig):
    @functools.wraps(orig)
    def close(self, *args, **kwargs):
        """InferenceRunner.close under lever writer_overlap: drain the writer first (every file on disk before teardown / any reader), then
        upstream's close; failed writes are raised in upstream's end-of-run form."""
        fails = drain("close")
        try:
            return orig(self, *args, **kwargs)
        finally:
            if fails and not _ST["raised"]:
                _ST["raised"] = True
                _report_failures(self, fails)
                raise RuntimeError(f"{len(fails)} inference sample(s) failed at the output stage (writer_overlap). "
                                   f"First error:\n{fails[0][0]} [seed:{fails[0][1]}]: {fails[0][2]}")
    setattr(close, MARK, True)
    _OURS.append(close)
    return close


# ------------------------------------------------------------------------------------------------------------------ install
def _bind(module) -> None:
    """Subscriber of phase.on_runner_module: wrap DataDumper.dump and InferenceRunner.close on the imported upstream (idempotent)."""
    _ST["module"] = module
    dm = sys.modules.get(DUMPER_MODULE)                              # upstream's runner.inference imports runner.dumper in its body (inference.py:72): present here on the pin;
    if dm is None:                                                   # a runner module that does not import it (a stub, another upstream) leaves the lever armed, not installed
        return                                                       # (inert by name in the report: no item could reach the dump)
    cls = getattr(dm, DUMP_ATTR[0], None)
    fn = getattr(cls, DUMP_ATTR[1], None) if cls is not None else None
    if fn is None:                                                   # the module is there but the pin's dump site moved: named (every item written by the stock path)
        if f"{DUMPER_MODULE}.{'.'.join(DUMP_ATTR)}" not in STATS["missing"]:
            STATS["missing"].append(f"{DUMPER_MODULE}.{'.'.join(DUMP_ATTR)}")
        return
    if not _ours(fn):
        _ST["orig_dump"], _ST["dumper_cls"] = fn, cls
        setattr(cls, DUMP_ATTR[1], _wrap_dump(fn))
    rcls = getattr(module, CLOSE_ATTR[0], None)
    cfn = getattr(rcls, CLOSE_ATTR[1], None) if rcls is not None else None
    if cfn is None:
        if f"{RUNNER_MODULE}.{'.'.join(CLOSE_ATTR)}" not in STATS["missing"]:
            STATS["missing"].append(f"{RUNNER_MODULE}.{'.'.join(CLOSE_ATTR)}")   # no drain point on this upstream: the exit guard drains (named)
    elif not _ours(cfn):
        _ST["orig_close"], _ST["runner_cls"] = cfn, rcls
        setattr(rcls, CLOSE_ATTR[1], _wrap_close(cfn))
    STATS["installed"] = True


def install() -> str:
    """Arm the lever: bind now when ``runner.inference`` is imported, else at its import (phase's one meta-path hook). Idempotent."""
    if STATS["installed"]:
        return "installed"
    try:
        import opt_core.host.outputs  # noqa: F401 — the core primitive this adapter drives; absent = the activation fails by name
    except Exception as e:  # noqa: BLE001
        raise ActivationError(f"writer_overlap: opt_core.host.outputs is not importable ({e!r})") from None
    from . import phase
    STATS["armed"] = True
    return phase.on_runner_module(_bind)


def uninstall() -> None:
    w = _ST["writer"]
    if w is not None:
        try:
            w.close()
        except Exception:  # noqa: BLE001
            pass
        _ST["writer"] = None
    if _ST["dumper_cls"] is not None and _ST["orig_dump"] is not None:
        if _ours(getattr(_ST["dumper_cls"], DUMP_ATTR[1], None)):
            setattr(_ST["dumper_cls"], DUMP_ATTR[1], _ST["orig_dump"])
    if _ST["runner_cls"] is not None and _ST["orig_close"] is not None:
        if _ours(getattr(_ST["runner_cls"], CLOSE_ATTR[1], None)):
            setattr(_ST["runner_cls"], CLOSE_ATTR[1], _ST["orig_close"])
    _OURS.clear()
    try:
        from . import phase
        if _bind in phase._SUBSCRIBERS:
            phase._SUBSCRIBERS.remove(_bind)
    except Exception:  # noqa: BLE001
        pass
    for k in ("orig_dump", "orig_close", "dumper_cls", "runner_cls", "module"):
        _ST[k] = None
    _ST["raised"] = False
    STATS["installed"] = False
    STATS["armed"] = False


def _reset() -> None:
    """Test hook: the wraps removed, the writer closed, counters cleared (both levers of the unit)."""
    _reset_json()
    uninstall()
    for k, v in list(STATS.items()):
        if isinstance(v, bool):
            STATS[k] = False
        elif isinstance(v, (int, float)):
            STATS[k] = 0 if isinstance(v, int) else 0.0
        elif isinstance(v, dict):
            v.clear()
        elif isinstance(v, list):
            v.clear()



# ------------------------------------------------------------------------------------------------------------------------------------------
# json_oneshot — the tree's own lever on upstream's JSON writer (exact class: identical bytes, one C-encoder pass instead of the pure-Python
# streaming encoder). ``runner/dumper.py`` writes every confidence document through ``opendde.utils.file_io.save_json`` (file_io.py:211-218:
# ``json.dump(data_json, f[, indent=4])``). ``json.dump`` streams the document through the standard library's pure-Python ``_make_iterencode``
# (the C encoder serves ``json.dumps`` only) in thousands of small ``write`` calls: for the ``full_data`` document — the token x token PAE /
# contact maps as nested lists of Python floats, ~3 x N_token^2 numbers per sample, 5 samples — that is nearly the whole result-dump wall.
# The lever binds the name ``runner.dumper.save_json`` (the module-level name
# ``dump_predictions`` / ``_save_confidence`` call, dumper.py:18) to :func:`save_json_oneshot`: upstream's statement with the document encoded
# ONCE by ``json.dumps`` (``indent=None``: the C encoder; ``indent=4``: the same Python encoder ``json.dump`` uses) and written with one
# ``write``. ``json.dumps(o) == ''.join(JSONEncoder().iterencode(o))`` is the standard library's own contract (same float repr, separators, key
# order, NaN words, ASCII escaping), so every file is byte-identical to the stock writer's. Wherever the dump runs (the loop, or writer_overlap's
# background thread) the bound name is what it calls. Counters: files, mib, encode_s, write_s; a missing site is named (PARTIAL).
JSON_SITE = ("runner.dumper", "save_json")                           # dumper.py:18 `from opendde.utils.file_io import save_json` — the name the dump calls
FILE_IO = "opendde.utils.file_io"                                    # map_values_to_list (file_io.py:186) — upstream's own list conversion, called as save_json calls it
JMARK = "_opendde_opt_json_oneshot"
JSTATS = {"installed": False, "armed": False, "files": 0, "bytes": 0, "encode_s": 0.0, "write_s": 0.0, "indent_files": 0, "missing": [], "errors": 0}
_JST = {"orig": None, "module": None, "map_values_to_list": None}


def save_json_oneshot(data, output_fpath, indent=4):
    """= opendde/utils/file_io.py:211-218 ``save_json`` with the document encoded once (``json.dumps``) and written once — identical bytes."""
    import json
    t0 = time.perf_counter()
    data_json = data.copy()
    data_json = _JST["map_values_to_list"](data_json)
    text = json.dumps(data_json, indent=indent) if indent is not None else json.dumps(data_json)
    t1 = time.perf_counter()
    with open(output_fpath, "w") as f:
        f.write(text)
    t2 = time.perf_counter()
    with _LOCK:
        JSTATS["files"] += 1; JSTATS["bytes"] += len(text); JSTATS["encode_s"] += t1 - t0; JSTATS["write_s"] += t2 - t1
        if indent is not None:
            JSTATS["indent_files"] += 1


setattr(save_json_oneshot, JMARK, True)


def _bind_json(module) -> None:
    """Subscriber of phase.on_runner_module: bind runner.dumper.save_json (idempotent); upstream's list conversion resolved from file_io."""
    _JST["module"] = module
    dm = sys.modules.get(JSON_SITE[0])
    if dm is None:                                                   # a runner module that does not import runner.dumper: armed, not installed (inert by name)
        return
    cur = getattr(dm, JSON_SITE[1], None)
    fio = sys.modules.get(FILE_IO)
    mvl = getattr(fio, "map_values_to_list", None) if fio is not None else None
    if cur is None or mvl is None:                                   # the pin's site moved: named — every document written by the stock writer
        name = f"{JSON_SITE[0]}.{JSON_SITE[1]}" if cur is None else f"{FILE_IO}.map_values_to_list"
        if name not in JSTATS["missing"]:
            JSTATS["missing"].append(name)
        return
    if getattr(cur, JMARK, False):
        JSTATS["installed"] = True
        return
    _JST["orig"], _JST["map_values_to_list"] = cur, mvl
    setattr(dm, JSON_SITE[1], save_json_oneshot)
    JSTATS["installed"] = True


def install_json() -> str:
    """Arm json_oneshot (idempotent): bind now when ``runner.inference`` is imported, else at its import through phase's hook."""
    if JSTATS["installed"]:
        return "installed"
    from . import phase
    JSTATS["armed"] = True
    return phase.on_runner_module(_bind_json)


def uninstall_json() -> None:
    dm = sys.modules.get(JSON_SITE[0])
    if dm is not None and _JST["orig"] is not None and getattr(getattr(dm, JSON_SITE[1], None), JMARK, False):
        setattr(dm, JSON_SITE[1], _JST["orig"])
    try:
        from . import phase
        if _bind_json in phase._SUBSCRIBERS:
            phase._SUBSCRIBERS.remove(_bind_json)
    except Exception:  # noqa: BLE001
        pass
    _JST.update(orig=None, module=None, map_values_to_list=None)
    JSTATS["installed"] = False; JSTATS["armed"] = False


def _reset_json() -> None:
    uninstall_json()
    JSTATS.update(installed=False, armed=False, files=0, bytes=0, encode_s=0.0, write_s=0.0, indent_files=0, errors=0)
    JSTATS["missing"].clear()


def kit_stats_json() -> dict:
    with _LOCK:
        out = dict(JSTATS); out["missing"] = list(JSTATS["missing"])
    out["mib"] = round(out.pop("bytes", 0) / 2**20, 1)
    out["encode_s"], out["write_s"] = round(float(out["encode_s"]), 3), round(float(out["write_s"]), 3)
    return out


def fallbacks_json(planned) -> list:
    if "json_oneshot" not in (planned or ()):
        return []
    out = [f"json_oneshot: {m} not found on the installed upstream (every document written by the stock writer)" for m in JSTATS["missing"]]
    dm = sys.modules.get(JSON_SITE[0])
    if JSTATS["installed"] and dm is not None and not getattr(getattr(dm, JSON_SITE[1], None), JMARK, False):
        out.append("json_oneshot: runner.dumper.save_json was re-bound after the lever without relaying to it")
    return out


def kit_stats() -> dict:
    with _LOCK:
        out = {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in STATS.items()}
    out["bytes_mib"] = round(out.pop("bytes", 0) / 2**20, 1)
    out["mode"], out["max_pending"] = MODE, MAX_PENDING
    for k in ("d2h_s", "submit_s", "write_s", "queued_s", "drain_s"):
        out[k] = round(float(out[k]), 3)
    return out


def fallbacks(planned) -> list:
    """The lever's named events at exit: a write that failed, the dump site absent on the installed upstream, items written synchronously
    by an aside, the wrap re-bound away, bookkeeping errors."""
    if "writer_overlap" not in (planned or ()):
        return []
    out = []
    if STATS["failures"]:
        n, f = len(STATS["failures"]), STATS["failures"][0]
        out.append(f"writer_overlap: {n} write(s) failed (first: {f[0]} seed {f[1]}: {f[2]})")
    for m in STATS["missing"]:
        out.append(f"writer_overlap: {m} not found on the installed upstream (every item written by the stock path)")
    if STATS["sync"]:
        out.append("writer_overlap: %d item(s) written synchronously by the stock path (aside=%s)" %
                   (STATS["sync"], ",".join(f"{k}x{v}" for k, v in sorted(STATS["aside"].items()))))
    cls = _ST["dumper_cls"]
    if STATS["installed"] and cls is not None and not _ours(getattr(cls, DUMP_ATTR[1], None)):
        out.append("writer_overlap: runner.dumper.DataDumper.dump was re-bound after the lever without relaying to it")
    w = _ST["writer"]
    if w is not None and w.pending():
        out.append(f"writer_overlap: {w.pending()} write(s) still pending at the report (the drain point was not reached)")
    if STATS["errors"]:
        out.append(f"writer_overlap: {STATS['errors']} drain error(s)")
    return out
