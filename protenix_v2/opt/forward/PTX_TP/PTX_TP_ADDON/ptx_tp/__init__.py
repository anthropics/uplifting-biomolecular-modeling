"""ptx_tp -- tensor-parallel (row-sharded pair representation) inference add-on for STOCK Protenix 2.x (model 'protenix-v2').

Driver layer: launcher, runner hooks (rank-0 featurize + broadcast, rank-0 outputs), trunk re-orchestration that keeps
z ROW-SHARDED and calls the fixed seam names, the per-rank phase log.
Seam modules (pairformer / trimul / triatt / rowlocal, msa, diffusion, confidence / summary): ptx_tp.<seam>, the row-sharded
implementation of each stock module; ``impl(seam)`` imports it (an absent seam module is an ImportError by name, never a fallback).

Environment (read by apply_from_env / impl()):
  PTX_TP=P                         enable TP hooks (P must equal torchrun WORLD_SIZE; PTX_TP<=1 or unset -> no hooks = stock behaviour)
  PTX_TP_FEAT=bcast|all            rank-0 featurize + broadcast (default) | every rank featurizes and asserts equality by checksum
  PTX_DET=1                        the deterministic recipe (det_recipe()): allreduce_checksum asserts on the replicated tensors at the
                                   phase boundaries and per diffusion step (rank-identical under deterministic kernels; a mismatch is refused)
  PTX_TP_ZINIT_RECOMPUTE=1         rebuild z_init rows each recycling cycle instead of storing them (tp trunk statements only)
  PTX_TP_RELP=lazy|full            lazy (the default once applied): never materialise the dense relp one-hot [N,N,139] fp32 (the seams
                                   read rows) | full: keep stock's dense tensor
  PTX_TP_PHASE_LOG=<path>          per-rank phase markers + torch.cuda max_memory_allocated/reserved (jsonl), see ptx_tp/mirror.py
"""
from __future__ import annotations

import importlib
import os
import sys

__version__ = "0.1.13"
SEAMS = ("pairformer", "msa", "diffusion", "confidence", "summary", "trimul", "triatt", "rowlocal", "trunk")
_IMPL_CACHE: dict = {}
_LEDGER_PRINTED = False


def log(msg: str) -> None:
    rank = os.environ.get("RANK", "0")
    print(f"[ptx_tp r{rank}] {msg}", file=sys.stderr, flush=True)


def det_recipe() -> bool:
    """Whether the deterministic recipe is on in this process (PTX_DET=1): the replicated-tensor checksum guards run under it only."""
    return os.environ.get("PTX_DET", "0") == "1"


def tp_size_from_env() -> int:
    try:
        return int(os.environ.get("PTX_TP", "0") or 0)
    except ValueError:
        return 0


def impl(seam: str):
    """The module implementing a seam: ``ptx_tp.<seam>`` (``summary`` is the confidence head's finisher, ``trunk`` the row-local z_init /
    recycle / distogram statements in ptx_tp.trunk). An absent or broken seam module raises ImportError naming it — never a fallback."""
    seam = seam.lower()
    if seam not in _IMPL_CACHE:
        try:
            _IMPL_CACHE[seam] = importlib.import_module(f"ptx_tp.{seam}")
        except Exception as e:
            raise ImportError(f"seam {seam!r}: ptx_tp.{seam} is not importable: {e!r}")
    return _IMPL_CACHE[seam]


def ledger() -> str:
    """The seams record of the APPLIED / effective lines: ``<seam>=tp`` for every seam, each module imported first (an absent one raises by
    name before the line is printed), so the line is the statement that every seam of the run is its row-sharded implementation."""
    for s in SEAMS:
        impl(s)
    return " ".join(f"{s}=tp" for s in SEAMS)


def apply_from_env(force: bool = False) -> bool:
    """Install the TP runner hooks when PTX_TP>1 (and torchrun WORLD_SIZE matches). Returns True if installed.
    Prints one ledger line: '[ptx_tp r<rank>] APPLIED P=.. feat=.. relp=.. templ=.. zinit_recompute=.. extras=[..] seams: pairformer=tp msa=tp ...'."""
    global _LEDGER_PRINTED
    P = tp_size_from_env()
    ws = int(os.environ.get("WORLD_SIZE", "1"))
    if P <= 1 and not force:
        if not _LEDGER_PRINTED:
            log(f"NOT applied (PTX_TP={os.environ.get('PTX_TP')!r}); stock behaviour")
            _LEDGER_PRINTED = True
        return False
    if ws != P:
        raise RuntimeError(f"PTX_TP={P} but torchrun WORLD_SIZE={ws}; launch with: python -m ptx_tp.launch --nproc {P} -- <protenix args>")
    # defaults: lazy relp (never the [N,N,139] fp32 one-hot; a single-card LazyRelp already in place takes precedence),
    # O(N_atom) atom local-attention masks (atom_local, exact by construction), optional SDPA backend pin (diffusion).
    os.environ.setdefault("PTX_TP_RELP", "lazy")
    from ptx_tp import runner_hooks
    runner_hooks.install()
    extras = []
    try:
        from ptx_tp import atom_local
        if atom_local.apply_from_env():
            extras.append("atom_local")
    except Exception as e:
        log(f"atom_local not applied: {e!r}")
    try:
        from ptx_tp import diffusion as _dif
        if os.environ.get("PTX_TP_SDPA_BACKEND", "").strip():
            extras.append(f"sdpa={_dif.apply_sdpa_backend_from_env()}")
    except Exception as e:
        log(f"diffusion env hooks not applied: {e!r}")
    if not _LEDGER_PRINTED:
        if os.environ.get("PTX_TP_TRIMUL_BCACHE", "") == "host":
            try:
                import time as _time
                from ptx_tp.trimul import reserve_pinned_pool
                _t0 = _time.time()
                reserve_pinned_pool()
                log(f"pinned host pool reserved for the TriMul b-stash ({os.environ.get('PTX_TP_PINNED_POOL_GB', '?')} GB) in {_time.time() - _t0:.0f}s")
            except Exception as _e:
                log(f"reserve_pinned_pool failed: {_e!r}")
        log(f"APPLIED P={P} feat={os.environ.get('PTX_TP_FEAT', 'bcast')} relp={os.environ.get('PTX_TP_RELP')} "
            f"zinit_recompute={os.environ.get('PTX_TP_ZINIT_RECOMPUTE', '0')} extras={extras} seams: {ledger()}")
        _LEDGER_PRINTED = True
    return True


def runmeta_path():
    """<PTX_TP_PHASE_LOG or out>/runmeta_<PTX_TP_JOB_LABEL>.json -- per-run metadata (mc-dropout draw + mask mode, seams, N, P ...)."""
    d = os.environ.get("PTX_TP_PHASE_LOG", "out")
    if d.endswith(".jsonl"):
        d = os.path.dirname(d) or "."
    os.makedirs(d, exist_ok=True)
    return os.path.join(d, f"runmeta_{os.environ.get('PTX_TP_JOB_LABEL', 'job')}.json")


def runmeta_update(**kv):
    """Merge keys into the run metadata json (rank 0 only; list-valued 'predictions' entries are appended)."""
    if os.environ.get("RANK", "0") != "0":
        return
    import json
    p = runmeta_path()
    try:
        cur = json.load(open(p)) if os.path.exists(p) else {}
    except Exception:
        cur = {}
    preds = kv.pop("prediction", None)
    cur.update(kv)
    if preds is not None:
        cur.setdefault("predictions", []).append(preds)
    with open(p + ".tmp", "w") as f:
        json.dump(cur, f, indent=1, default=str)
    os.replace(p + ".tmp", p)
