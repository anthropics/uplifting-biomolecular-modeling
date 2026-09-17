"""Item-boundary census — where an item's wall goes OUTSIDE the model forward, one line per item on every kit line (rank 0):

    [opendde-opt] ITEM item=<name> n=<k> seed_s= feat_s= prep_s= predict_s= dump_s= cleanup_s= other_s= loop_s= outside_s=

Timing only (no argument, return value or tensor is touched), the boundaries are the pinned upstream's own item-loop callables in
``runner/inference.py`` ``_infer_predict_impl`` (:1583-1897): ``seed_s`` = ``seed_everything`` (the per-item RNG reset, :1683-1690; the per-seed
one, :1644-1651, accrues to the first item); ``feat_s`` = ``_next_inference_batch_synchronized`` (:217 — the DataLoader's ``next``: with upstream's
default ``num_workers=0`` the item is FEATURISED HERE, serially, in the loop: JSON entities, CCD / RDKit conformers, tokenisation, MSA
features); ``prep_s`` = ``_prepare_inference_batch`` + ``_prepare_prediction_batch`` (:197, :1519 — configs update); ``predict_s`` =
``InferenceRunner.predict`` (the PHASE line's total: H2D feature transfer + the forward); ``dump_s`` = ``DataDumper.dump`` as the loop sees it
(the stock writer's whole wall, or under ``writer_overlap`` the host copy + hand-off); ``cleanup_s`` = ``_cleanup_batch_synchronized`` (:764 —
``gc.collect`` + ``torch.cuda.empty_cache`` after every item) + the per-seed ``cleanup_device_memory`` calls (:1653, :1870); ``loop_s`` = wall
from the previous item's cleanup end (item 1: from the dataloader iterator's creation) to this item's cleanup end; ``other_s`` = ``loop_s``
minus the named parts (the Fold-CP no-op synchronisations, logging); ``outside_s`` = ``loop_s - predict_s``: the seconds of this item the
GPU had no forward to run. The census is the input of the non-forward levers' decisions (writer_overlap; the featurisation prefetch below,
since ``feat_s`` is material against ``predict_s``) and reads the same on both arms of an
ablation. Installed through phase's one runner-module hook; standard library only.

Lever ``prefetch`` (exact class; this module's second half): featurisation costs seconds per item (of the order of a tenth of the fast
forward at mid sizes) — serial, in the loop, because upstream's DataLoader runs with ``num_workers=0``
(``opendde/config/inference_defaults.py:22``; the ``opendde pred`` CLI has no option for it). The lever binds
``runner.inference._create_inference_dataloader_synchronized`` (inference.py:291): when the configs carry upstream's default ``num_workers == 0``
on a single-GPU run it sets ``configs.num_workers = 1`` before the DataLoader is built (``opendde/data/inference/infer_dataloader.py:143-149``), so
ONE DataLoader worker featurises item k+1 (and k+2: torch's ``prefetch_factor=2``) while item k runs; the features are the dataset's own
``__getitem__`` output handed over torch's worker queue (CPU tensors; the worker never touches the device), the loop's ``next()`` returns at
once, and nothing of the forward changes. A caller's own ``num_workers`` (a JSON/config override) stands (aside by name); a row-sharded rank
never carries the lever (modes.BIG_TP_DROP; tp.py refuses DataLoader workers on ranks) and a process inside an initialised process group of
more than one rank steps aside by name; a one-item query steps aside (nothing to featurise ahead); and so does a run of two or more seed
passes over a query with a ligand the featuriser embeds through RDKit (its process-global conformer generator continues across seeds in
the stock loop and would restart in a worker forked per seed: ``PREFETCH aside reason=rdkit_rng_multiseed``). Byte identity: the featuriser
DOES draw random numbers after the loop's per-item RNG reset (the reference conformers' rigid transform), so the worker replays that reset
(upstream's own ``seed_everything``) right before it featurises each item — the second half below.
"""
from __future__ import annotations

import functools
import os
import sys
import time

RUNNER_MODULE = "runner.inference"
SITES = {                                                               # field -> module-level names of runner.inference whose calls accrue to it
    "seed": ("seed_everything",),
    "feat": ("_next_inference_batch_synchronized",),
    "prep": ("_prepare_inference_batch", "_prepare_prediction_batch"),
    "cleanup": ("_cleanup_batch_synchronized", "cleanup_device_memory"),
    "iter": ("_create_dataloader_iterator_synchronized",),
}
MARK = "_opendde_opt_item_census"
STATS = {"installed": False, "armed": False, "items": 0, "missing": [], "totals": {}, "lines": 0}
_ACC = {"seed": 0.0, "feat": 0.0, "prep": 0.0, "cleanup": 0.0, "dump": 0.0, "predict": 0.0}
_ST = {"t_prev": None, "item": None, "n": 0, "pending": False, "t_end_cleanup": None, "orig": {}}


def _emit(line: str) -> None:
    if os.environ.get("RANK", "0") in ("0", ""):
        print(line, flush=True)


def _prefix() -> str:
    try:
        from .report import PREFIX
        return PREFIX
    except Exception:  # noqa: BLE001
        return "[opendde-opt]"


def _flush_line() -> None:
    """Print the pending item's line (called at the point the loop moves on: the next featurisation, the iterator's end, or close)."""
    if not _ST["pending"]:
        return
    now = time.perf_counter()
    t_prev = _ST["t_prev"] if _ST["t_prev"] is not None else now
    loop = max(0.0, _ST["t_end_cleanup"] - t_prev) if _ST.get("t_end_cleanup") else max(0.0, now - t_prev)
    named = sum(_ACC[k] for k in ("seed", "feat", "prep", "cleanup", "dump", "predict"))
    other = max(0.0, loop - named)
    outside = max(0.0, loop - _ACC["predict"])
    _ST["n"] += 1
    STATS["lines"] += 1
    tot = STATS["totals"]
    for k, v in _ACC.items():
        tot[k] = tot.get(k, 0.0) + v
    tot["loop"] = tot.get("loop", 0.0) + loop
    tot["outside"] = tot.get("outside", 0.0) + outside
    _emit(f"{_prefix()} ITEM item={_ST['item']} n={_ST['n']} seed_s={_ACC['seed']:.3f} feat_s={_ACC['feat']:.3f} prep_s={_ACC['prep']:.3f} "
          f"predict_s={_ACC['predict']:.3f} dump_s={_ACC['dump']:.3f} cleanup_s={_ACC['cleanup']:.3f} other_s={other:.3f} loop_s={loop:.3f} outside_s={outside:.3f}")
    _ST["t_prev"] = _ST.get("t_end_cleanup") or now
    _ST["pending"] = False
    _ST["item"] = None
    _ST["t_end_cleanup"] = None
    for k in _ACC:
        _ACC[k] = 0.0


_DEPTH = {k: 0 for k in SITES}


def _timed(fn, field: str):
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        if field in ("seed", "feat", "iter"):                        # the loop moved on (next item's RNG reset / fetch, next seed): the previous item's line is complete
            _flush_line()
        outer = _DEPTH[field] == 0                                   # nested calls of one field (cleanup_device_memory inside _cleanup_batch_synchronized) count once
        t0 = time.perf_counter()
        if _ST["t_prev"] is None:
            _ST["t_prev"] = t0                                       # item 1's loop starts at the first loop callable of the run
        _DEPTH[field] += 1
        try:
            return fn(*args, **kwargs)
        finally:
            _DEPTH[field] -= 1
            t1 = time.perf_counter()
            if outer and field != "iter":
                _ACC[field] += t1 - t0
                if field == "cleanup" and _ST["pending"]:
                    _ST["t_end_cleanup"] = t1
    setattr(wrapper, MARK, field)
    return wrapper


def _wrap_predict(fn):
    @functools.wraps(fn)                                             # carries phase's mark when phase wrapped first (no double PHASE wrap either way)
    def predict(self, data, *args, **kwargs):
        try:
            _ST["item"] = str(data.get("sample_name", "unknown"))
        except Exception:  # noqa: BLE001
            _ST["item"] = "unknown"
        _ST["pending"] = True
        STATS["items"] += 1
        t0 = time.perf_counter()
        try:
            return fn(self, data, *args, **kwargs)
        finally:
            _ACC["predict"] += time.perf_counter() - t0
    predict.__dict__[MARK] = "predict"
    return predict


def _wrap_dump(fn):
    @functools.wraps(fn)                                             # carries writer_overlap's mark when it wrapped first (its bind sees the mark and does not re-wrap)
    def dump(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return fn(self, *args, **kwargs)
        finally:
            _ACC["dump"] += time.perf_counter() - t0
    dump.__dict__[MARK] = "dump"
    return dump


def _wrap_close(fn):
    @functools.wraps(fn)
    def close(self, *args, **kwargs):
        _flush_line()                                                # the last item's line (its cleanup ran; nothing follows it in the loop)
        tot = STATS["totals"]
        if STATS["lines"]:
            _emit(f"{_prefix()} ITEMS n={STATS['lines']} " + " ".join(f"{k}_s={tot.get(k, 0.0):.3f}" for k in ("seed", "feat", "prep", "predict", "dump", "cleanup", "loop", "outside")))
        return fn(self, *args, **kwargs)
    close.__dict__[MARK] = "close"
    return close


def _bind(module) -> None:
    for field, names in SITES.items():
        for name in names:
            cur = getattr(module, name, None)
            if cur is None:
                if name not in STATS["missing"]:
                    STATS["missing"].append(name)
                continue
            if getattr(cur, MARK, None) == field:
                continue
            _ST["orig"][name] = cur
            setattr(module, name, _timed(cur, field))
    cls = getattr(module, "InferenceRunner", None)
    if cls is not None:
        for attr, wrap, tag in (("predict", _wrap_predict, "predict"), ("close", _wrap_close, "close")):
            cur = getattr(cls, attr, None)
            if cur is not None and cur.__dict__.get(MARK) != tag:
                _ST["orig"][f"InferenceRunner.{attr}"] = cur
                setattr(cls, attr, wrap(cur))
    dm = sys.modules.get("runner.dumper")
    dcls = getattr(dm, "DataDumper", None) if dm is not None else None
    dfn = getattr(dcls, "dump", None) if dcls is not None else None
    if dfn is not None and dfn.__dict__.get(MARK) != "dump":
        _ST["orig"]["DataDumper.dump"] = dfn
        setattr(dcls, "dump", _wrap_dump(dfn))
    STATS["installed"] = True


def install() -> str:
    """Arm the census (idempotent): bind now when ``runner.inference`` is imported, else at its import through phase's hook."""
    if STATS["installed"]:
        return "installed"
    from . import phase
    STATS["armed"] = True
    return phase.on_runner_module(_bind)


def _reset() -> None:
    """Test hook: unbind, clear (the census and the prefetch lever)."""
    _reset_prefetch()
    mod = sys.modules.get(RUNNER_MODULE)
    for name, orig in list(_ST["orig"].items()):
        try:
            if name.startswith("InferenceRunner.") and mod is not None:
                setattr(mod.InferenceRunner, name.split(".", 1)[1], orig)
            elif name == "DataDumper.dump":
                dm = sys.modules.get("runner.dumper")
                if dm is not None:
                    dm.DataDumper.dump = orig
            elif mod is not None:
                setattr(mod, name, orig)
        except Exception:  # noqa: BLE001
            pass
    _ST["orig"].clear()
    try:
        from . import phase
        if _bind in phase._SUBSCRIBERS:
            phase._SUBSCRIBERS.remove(_bind)
    except Exception:  # noqa: BLE001
        pass
    STATS.update(installed=False, armed=False, items=0, lines=0)
    STATS["missing"].clear(); STATS["totals"].clear()
    _ST.update(t_prev=None, item=None, n=0, pending=False, t_end_cleanup=None)
    for k in _ACC:
        _ACC[k] = 0.0
    for k in _DEPTH:
        _DEPTH[k] = 0


def kit_stats() -> dict:
    return {"installed": STATS["installed"], "items": STATS["items"], "lines": STATS["lines"], "missing": list(STATS["missing"]),
            "totals": {k: round(v, 3) for k, v in STATS["totals"].items()}}


# ------------------------------------------------------------------------------------------------------------------------------------------
# lever `prefetch`: one DataLoader worker featurises the next item while this one runs (upstream's own DataLoader, num_workers 0 -> 1)
LEVER = "prefetch"
CREATE_SITE = "_create_inference_dataloader_synchronized"              # runner/inference.py:291 — builds the DataLoader from configs (infer_dataloader.py:143-149 reads configs.num_workers)
WORKERS = 1                                                             # one worker: items are featurised in order, one at a time, ahead of the loop
PMARK = "_opendde_opt_prefetch"
SEED_SITE = "seed_everything"                                           # runner/inference.py:63 imports it (opendde/utils/seed.py:10); the loop's per-item RNG reset (:1683-1690)
DATASET = ("opendde.data.inference.infer_dataloader", "InferenceDataset")   # __getitem__ (:338) featurises one item — in the worker process under this lever
PSTATS = {"installed": False, "armed": False, "calls": 0, "engaged": 0, "workers": 0, "aside": {}, "missing": [], "errors": 0, "reseeds": 0, "relayed": 0}
_PST = {"orig": None, "module": None, "orig_seed": None, "orig_getitem": None, "dataset_cls": None}
_SEED = {"seed": None, "deterministic": False}                          # the loop's last RNG reset (recorded in the parent; a forked worker inherits it)
NEXT_SITE = "_next_inference_batch_synchronized"                        # runner/inference.py:217 — the loop's next(): the parent folds the worker's relayed counters here
RELAY = (("opendde_opt.bondmask", "dropped"), ("opendde_opt.bondmask", "bytes_dropped"), ("opendde_opt.bondmask", "missing"))
# ^ the house counters that tick INSIDE featurisation (drop_bond_mask's, featurizer.py get_mask_features): under this lever they tick in the worker
#   process, so the worker adds its per-item deltas to a shared-memory array created before the fork and the parent folds them into its own
#   books at every next() — the ran-or-refuse accounting (ran.COUNTERS) reads the parent's books as on the stock loader.
_RELAY = {"shm": None, "folded": None, "orig_next": None}


def _p_emit(line: str) -> None:
    print(f"[opendde-opt] {line}", flush=True)


def _p_aside(reason: str) -> None:
    PSTATS["aside"][reason] = PSTATS["aside"].get(reason, 0) + 1
    _p_emit(f"PREFETCH aside reason={reason} (upstream's DataLoader as configured: num_workers unchanged)")


def _world() -> int:
    try:
        dist = sys.modules.get("torch.distributed")
        if dist is not None and dist.is_available() and dist.is_initialized():
            return int(dist.get_world_size())
    except Exception:  # noqa: BLE001
        pass
    return 1


def _record_seed(orig):
    """runner.inference.seed_everything, recording the loop's (seed, deterministic) before running it: the worker replays exactly this reset."""
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        try:
            seed = kwargs["seed"] if "seed" in kwargs else (args[0] if args else None)
            det = kwargs["deterministic"] if "deterministic" in kwargs else (args[1] if len(args) > 1 else False)
            _SEED["seed"], _SEED["deterministic"] = (int(seed) if seed is not None else None), bool(det)
        except Exception:  # noqa: BLE001
            pass
        return orig(*args, **kwargs)
    setattr(wrapper, PMARK, True)
    return wrapper


def _reseed(seed: int, deterministic: bool) -> None:
    """The loop's RNG reset, in this (worker) process: upstream's own seed_everything when importable, else its three host seeds."""
    fn = getattr(sys.modules.get("opendde.utils.seed"), "seed_everything", None)
    if fn is not None:
        try:
            fn(seed=seed, deterministic=deterministic)
            return
        except Exception:  # noqa: BLE001 — a CUDA switch refusing in a forked child: the host RNGs are what featurisation reads
            pass
    import random
    random.seed(seed)
    np = sys.modules.get("numpy")
    if np is not None:
        np.random.seed(seed)
    torch = sys.modules.get("torch")
    if torch is not None:
        torch.random.manual_seed(seed)


def _relay_read(i: int):
    mod, key = RELAY[i]
    m = sys.modules.get(mod)
    st = getattr(m, "STATS", None) if m is not None else None
    try:
        return int(st.get(key) or 0) if isinstance(st, dict) else None
    except Exception:  # noqa: BLE001
        return None


def _relay_fold() -> int:
    """Parent side: add the worker's counter deltas accumulated in shared memory since the last fold into this process's books."""
    shm, folded = _RELAY["shm"], _RELAY["folded"]
    if shm is None or folded is None:
        return 0
    moved = 0
    try:
        with shm.get_lock():
            vals = [int(v) for v in shm[:]]
        for i, v in enumerate(vals):
            d = v - folded[i]
            if d:
                mod, key = RELAY[i]
                m = sys.modules.get(mod)
                st = getattr(m, "STATS", None) if m is not None else None
                if isinstance(st, dict):
                    st[key] = int(st.get(key) or 0) + d
                    folded[i] = v; moved += d
    except Exception:  # noqa: BLE001
        PSTATS["errors"] += 1
    PSTATS["relayed"] += moved
    return moved


def _wrap_next(orig):
    @functools.wraps(orig)
    def wrapper(*args, **kwargs):
        try:
            return orig(*args, **kwargs)
        finally:
            _relay_fold()
    setattr(wrapper, PMARK, True)
    return wrapper


def _getitem_replayed(orig):
    """InferenceDataset.__getitem__: in a DataLoader worker, the loop's per-item RNG reset is replayed right before the item is featurised —
    the stock loop featurises inside next() immediately after seed_everything(seed), so the featuriser's random draws (the reference
    conformers' random rigid transform, featurizer.py:369; covalent leaving groups) read the same RNG state here as there: the same features."""
    @functools.wraps(orig)
    def __getitem__(self, index):
        try:
            import torch.utils.data as _tud
            wi = _tud.get_worker_info()
        except Exception:  # noqa: BLE001
            wi = None
        if wi is None:
            return orig(self, index)
        if _SEED["seed"] is not None:
            _reseed(_SEED["seed"], _SEED["deterministic"])
            PSTATS["reseeds"] += 1                                   # counted in the worker's copy (the parent reads its own 0: by design)
        before = [_relay_read(i) for i in range(len(RELAY))]
        try:
            return orig(self, index)
        finally:
            shm = _RELAY["shm"]
            if shm is not None:
                try:
                    after = [_relay_read(i) for i in range(len(RELAY))]
                    with shm.get_lock():
                        for i, (b, a) in enumerate(zip(before, after)):
                            if a is not None and b is not None and a > b:
                                shm[i] += a - b
                except Exception:  # noqa: BLE001
                    pass
    setattr(__getitem__, PMARK, True)
    return __getitem__


RDKIT_ASIDE = "rdkit_rng_multiseed"
# ^ upstream builds ONE DataLoader per process (runner/inference.py:1622) and a fresh iterator — under this lever a freshly forked worker — for
#   every seed of the run (:1642-1672). A ligand given as a SMILES string is embedded by RDKit (json_parser.py:510-559, AllChem.EmbedMolecule with
#   no seed: RDKit's process-global generator, which upstream's seed_everything does not touch), so in the stock loop the second seed's
#   conformers continue the generator's sequence where the first seed's left it, while a worker forked from the parent (which never featurises)
#   would start the sequence again. The lever therefore steps aside by name when the run makes two or more seed passes AND a job carries a
#   ligand the featuriser does not build from CCD atoms: upstream's in-loop featurisation, identical bytes. One seed pass (any ligands) and
#   CCD-only queries (any seeds) are unaffected: one worker featurises the items in the loop's order from the parent's generator state.


def _seed_passes(configs, items) -> int:
    """How many per-seed passes upstream's loop makes over the DataLoader (runner/inference.py:1606-1615: the union of the jobs' seeds — the
    command line's ``--seeds`` for every job, else each job's ``modelSeeds``, else one drawn default per job, :870-892) — counted here WITHOUT
    drawing anything (a job without seeds counts as one distinct pass)."""
    cli = getattr(configs, "seeds", None)
    try:
        cli = [int(x) for x in cli] if cli else []
    except Exception:  # noqa: BLE001
        cli = list(cli or [])
    if cli:
        return len(dict.fromkeys(cli))
    union, drawn = set(), 0
    for job in (items or []):
        ms = job.get("modelSeeds") if isinstance(job, dict) else None
        if ms:
            try:
                union.update(int(x) for x in ms)
            except Exception:  # noqa: BLE001
                drawn += 1
        else:
            drawn += 1
    return len(union) + drawn


def _rdkit_ligand_jobs(items) -> int:
    """Jobs with a ligand entity the featuriser builds through RDKit rather than from CCD atoms (json_parser.py:600-628: a ``ligand`` string
    that does not start with ``CCD_`` — a SMILES, embedded on RDKit's process-global generator, :531; ``FILE_`` ligands are counted too,
    conservatively). Ions, polymers and their CCD modifications draw nothing from that generator."""
    n = 0
    for job in (items or []):
        seqs = job.get("sequences") if isinstance(job, dict) else None
        for ent in (seqs or []):
            info = ent.get("ligand") if isinstance(ent, dict) else None
            lig = info.get("ligand") if isinstance(info, dict) else None
            if isinstance(lig, str) and not lig.startswith("CCD_"):
                n += 1
                break
    return n


def _wrap_create(orig):
    @functools.wraps(orig)
    def wrapper(configs, *args, **kwargs):
        PSTATS["calls"] += 1
        try:
            nw = getattr(configs, "num_workers", None)
            items = args[0] if args else kwargs.get("inputs")
            n_items = len(items) if hasattr(items, "__len__") else None
            if nw is None:
                _p_aside("configs_without_num_workers")
            elif n_items is not None and n_items < 2:
                _p_aside(f"single_item_query")                       # one item: nothing to featurise ahead of it — upstream's in-loop featurisation, no worker process
            elif int(nw) != 0:
                _p_aside(f"num_workers={int(nw)}_set_by_the_caller")
            elif _world() > 1:
                _p_aside(f"process_group_of_{_world()}_ranks")
            elif _seed_passes(configs, items) >= 2 and _rdkit_ligand_jobs(items):
                _p_aside(RDKIT_ASIDE)                                # two or more seed passes AND a ligand embedded by RDKit: a worker re-forked per seed would restart
                                                                     # RDKit's conformer sequence the stock loop continues across seeds — upstream's in-loop featurisation
            elif _PST["orig_seed"] is None or _PST["orig_getitem"] is None:
                _p_aside("rng_replay_unbound")                       # without the worker-side RNG replay the features would differ from the loop's: upstream's loader stands
            else:
                configs.num_workers = WORKERS
                if int(getattr(configs, "num_workers", 0)) != WORKERS:
                    _p_aside("configs_immutable")
                else:
                    if _RELAY["shm"] is None:
                        import multiprocessing as _mp
                        _RELAY["shm"], _RELAY["folded"] = _mp.Array("q", len(RELAY), lock=True), [0] * len(RELAY)
                    PSTATS["engaged"] += 1; PSTATS["workers"] = WORKERS
                    _p_emit(f"PREFETCH dataloader num_workers=0->{WORKERS} (one worker process featurises the next item while this one runs; torch prefetch_factor=2)")
        except Exception as e:  # noqa: BLE001 — the lever's own bookkeeping never breaks the run: named, counted, upstream's path
            PSTATS["errors"] += 1
            _p_emit(f"PREFETCH aside reason=error:{type(e).__name__}:{e}")
        return orig(configs, *args, **kwargs)
    setattr(wrapper, PMARK, True)
    return wrapper


def _bind_prefetch(module) -> None:
    _PST["module"] = module
    cur = getattr(module, CREATE_SITE, None)
    if cur is None:
        if f"runner.inference.{CREATE_SITE}" not in PSTATS["missing"]:
            PSTATS["missing"].append(f"runner.inference.{CREATE_SITE}")
        return
    if getattr(cur, PMARK, False):
        PSTATS["installed"] = True
        return
    se = getattr(module, SEED_SITE, None)
    if se is not None and not getattr(se, PMARK, False):
        _PST["orig_seed"] = se
        setattr(module, SEED_SITE, _record_seed(se))
    elif se is None and f"runner.inference.{SEED_SITE}" not in PSTATS["missing"]:
        PSTATS["missing"].append(f"runner.inference.{SEED_SITE}")
    dmod = sys.modules.get(DATASET[0])
    dcls = getattr(dmod, DATASET[1], None) if dmod is not None else None
    gi = getattr(dcls, "__getitem__", None) if dcls is not None else None
    if gi is not None and not getattr(gi, PMARK, False):
        _PST["orig_getitem"], _PST["dataset_cls"] = gi, dcls
        dcls.__getitem__ = _getitem_replayed(gi)
    elif gi is None and f"{DATASET[0]}.{DATASET[1]}.__getitem__" not in PSTATS["missing"]:
        PSTATS["missing"].append(f"{DATASET[0]}.{DATASET[1]}.__getitem__")
    nx = getattr(module, NEXT_SITE, None)
    if nx is not None and not getattr(nx, PMARK, False):
        _RELAY["orig_next"] = nx
        setattr(module, NEXT_SITE, _wrap_next(nx))
    _PST["orig"] = cur
    setattr(module, CREATE_SITE, _wrap_create(cur))
    PSTATS["installed"] = True


def install_prefetch() -> str:
    """Arm the prefetch lever (idempotent): bind now when ``runner.inference`` is imported, else at its import through phase's hook."""
    if PSTATS["installed"]:
        return "installed"
    from . import phase
    PSTATS["armed"] = True
    return phase.on_runner_module(_bind_prefetch)


def uninstall_prefetch() -> None:
    mod = _PST["module"]
    if mod is not None and _PST["orig"] is not None and getattr(getattr(mod, CREATE_SITE, None), PMARK, False):
        setattr(mod, CREATE_SITE, _PST["orig"])
    if mod is not None and _PST["orig_seed"] is not None and getattr(getattr(mod, SEED_SITE, None), PMARK, False):
        setattr(mod, SEED_SITE, _PST["orig_seed"])
    if _PST["dataset_cls"] is not None and _PST["orig_getitem"] is not None and getattr(getattr(_PST["dataset_cls"], "__getitem__", None), PMARK, False):
        _PST["dataset_cls"].__getitem__ = _PST["orig_getitem"]
    if mod is not None and _RELAY["orig_next"] is not None and getattr(getattr(mod, NEXT_SITE, None), PMARK, False):
        setattr(mod, NEXT_SITE, _RELAY["orig_next"])
    _RELAY.update(shm=None, folded=None, orig_next=None)
    try:
        from . import phase
        if _bind_prefetch in phase._SUBSCRIBERS:
            phase._SUBSCRIBERS.remove(_bind_prefetch)
    except Exception:  # noqa: BLE001
        pass
    _PST.update(orig=None, module=None, orig_seed=None, orig_getitem=None, dataset_cls=None)
    PSTATS["installed"] = False; PSTATS["armed"] = False


def _reset_prefetch() -> None:
    uninstall_prefetch()
    PSTATS.update(installed=False, armed=False, calls=0, engaged=0, workers=0, errors=0, reseeds=0, relayed=0)
    _SEED.update(seed=None, deterministic=False)
    PSTATS["aside"].clear(); PSTATS["missing"].clear()


def kit_stats_prefetch() -> dict:
    _relay_fold()
    return {k: (dict(v) if isinstance(v, dict) else list(v) if isinstance(v, list) else v) for k, v in PSTATS.items()}


def fallbacks_prefetch(planned) -> list:
    if LEVER not in (planned or ()):
        return []
    _relay_fold()
    out = [f"prefetch: {m} not found on the installed upstream (items featurised in the loop)" for m in PSTATS["missing"]]
    if PSTATS["errors"]:
        out.append(f"prefetch: {PSTATS['errors']} bookkeeping error(s) (items featurised in the loop)")
    mod = _PST["module"]
    if PSTATS["installed"] and mod is not None and not getattr(getattr(mod, CREATE_SITE, None), PMARK, False):
        out.append(f"prefetch: runner.inference.{CREATE_SITE} was re-bound after the lever without relaying to it")
    return out
