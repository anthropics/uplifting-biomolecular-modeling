"""pairstack.py — the driver-side adapter of the memory line's pair-stack lever (imported ONLY inside the driver process, by patch 06's hook
``-trimul_chunk <rows>:<min residues>`` of ``predict_pdb.py``; never by the wrapper — the wrapper composes the flag, `big.py`).

``trimul_chunk`` (registry ``TRIMUL``): ``alphafold.model.modules.TriangleMultiplication`` is replaced (``opt_core.mem.rowpair_jax.haiku.PatchSet``)
by the tree's ONE row-chunked subclass ``opt_core.mem.rowpair_jax.rowchunk.chunked_class`` under this kit's size rule — chunk at compiled lengths
``N >= <min residues>`` (row chunks of the projections / einsum: [rows, N, c] intermediates instead of [N, N, c]); shorter lengths run the stock
body untouched (``disengaged``, counted, never silent). Same class name scope and parameter names: the stock parameters load unchanged.

Evidence: :func:`census` is the record the driver emits as its ``pairstack`` timer line per compiled length and once at exit (``final=True``);
:func:`lines` is the lever's LEVER line on stderr (``opt_core.report.lever_line``). A core without the producer is a RuntimeError naming the
module (the flag was asked for; no silent stock run) — the wrapper checks the same names before it starts the driver (``stack.producer_gate``),
so this is the second wall.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import TAG
PRODUCER_ROWCHUNK = "opt_core.mem.rowpair_jax.rowchunk"          # the ONE row-chunk body
PRODUCER_HAIKU = "opt_core.mem.rowpair_jax.haiku"                # the tree's haiku class-patch helper (PatchSet)
_STATE: Dict[str, object] = {"installed": False, "trimul": None, "patches": None,
                             "trimul_engaged": 0, "trimul_disengaged": 0, "trimul_shapes": {}, "trimul_ledger": None, "trimul_shapes_list": None}


def parse_trimul(spec: Optional[str]) -> Optional[Tuple[int, int]]:
    """``"<rows>:<min residues>"`` → (rows, min_residues); ``None``/empty → None (the lever off). Anything else is a usage error (ValueError)."""
    if spec is None or not str(spec).strip():
        return None
    try:
        rows_s, min_s = str(spec).split(":", 1)
        rows, min_res = int(rows_s), int(min_s)
    except ValueError:
        raise ValueError(f"-trimul_chunk {spec!r}: expected <rows>:<min residues>, two positive integers") from None
    if rows < 1 or min_res < 1:
        raise ValueError(f"-trimul_chunk {spec!r}: expected <rows>:<min residues>, two positive integers")
    return rows, min_res


def producers(trimul: Optional[Tuple[int, int]]) -> List[str]:
    """The core modules an install with this switch imports (what ``stack.producer_gate`` checks by name before the driver starts)."""
    return [PRODUCER_HAIKU, PRODUCER_ROWCHUNK] if trimul is not None else []


def _import(name: str):
    import importlib
    try:
        return importlib.import_module(name)
    except ImportError as e:
        raise RuntimeError(f"pairstack: {name} is not importable from this opt_core ({e}); install the tree's core at the kit's pin "
                           f"(opt/pyproject.toml [tool.opt_core]) or drop the flag") from None


def install(trimul: Optional[Tuple[int, int]] = None) -> dict:
    """Install the lever ONCE, before the model runner is built (patch 06 calls this right after argument parsing). Returns :func:`census`.
    A refusal of the shared producer ends the process with ONE named line (``[af2ig-opt] pairstack refused: <reason>``, exit 3) — never a
    stock-body run under the lever's flag."""
    from ._core_gate import gate
    gate(__file__, tag=TAG)                                      # the driver process's own core pin check before any opt_core import (a driver started by hand is an entry too): NOT ACTIVE reason=… -> exit 3
    from opt_core.oom import is_oom
    try:
        return _install(trimul)
    except Exception as e:                                       # MemLeverRefused and its kin carry .reason; anything else is named by its type — except an out-of-memory, which propagates as itself
        if is_oom(e): raise
        import sys
        reason = getattr(e, "reason", None) or f"{type(e).__name__}: {e}"
        print(f"[{TAG}] pairstack refused: {reason}", file=sys.stderr, flush=True)
        raise SystemExit(3) from e


def _install(trimul: Optional[Tuple[int, int]]) -> dict:
    if _STATE["installed"]:
        raise RuntimeError("pairstack.install called twice in one process")
    _STATE.update(trimul=trimul)
    if trimul is None:
        _STATE["installed"] = True
        return census()
    rp_hk = _import(PRODUCER_HAIKU)
    from alphafold.model import modules                          # the pinned checkout's monomer modules (the driver's own import path)
    patches = rp_hk.PatchSet("pairstack")
    _STATE["patches"] = patches
    rows, min_res = trimul                                       # replace the CLASS before anything is traced (the Evoformer / template stack look it up at call time)
    rowchunk = _import(PRODUCER_ROWCHUNK)
    from opt_core.mem import Ledger
    ledger = Ledger(); shapes: List[dict] = []
    _STATE["trimul_ledger"], _STATE["trimul_shapes_list"] = ledger, shapes
    stock_cls = modules.TriangleMultiplication
    chunked_cls = rowchunk.chunked_class(stock_cls, rows, ledger, shapes, lever="trimul_chunk")

    class TriangleMultiplication(chunked_cls):              # the kit's size rule over the producer: chunk only at compiled lengths >= <min residues> (every ladder length keeps the stock body)
        def __call__(self, act, mask, is_training=True):
            n = int(act.shape[-3])
            if n < min_res:
                _STATE["trimul_disengaged"] = int(_STATE["trimul_disengaged"]) + 1
                return stock_cls.__call__(self, act, mask, is_training=is_training)
            _STATE["trimul_engaged"] = int(_STATE["trimul_engaged"]) + 1     # per TRACE (one per distinct compiled program × call site)
            key = f"N{n}xR{rows}"
            _STATE["trimul_shapes"][key] = _STATE["trimul_shapes"].get(key, 0) + 1
            return chunked_cls.__call__(self, act, mask, is_training=is_training)

    patches.replace(modules, "TriangleMultiplication", TriangleMultiplication)
    _STATE["installed"] = True
    return census()


def census() -> dict:
    """The ``pairstack`` record: the trimul_chunk switch and its trace tally."""
    out = {"installed": bool(_STATE["installed"])}
    trimul = _STATE["trimul"]
    out["trimul_chunk"] = None if trimul is None else {"rows": trimul[0], "min_residues": trimul[1], "engaged_traces": int(_STATE["trimul_engaged"]),
                                                       "disengaged_traces": int(_STATE["trimul_disengaged"]), "shapes": dict(_STATE["trimul_shapes"]), "producer": PRODUCER_ROWCHUNK}
    return out


def lines() -> List[str]:
    """The LEVER line of this install (stderr at the driver's exit): ``trimul_chunk`` via ``opt_core.report.lever_line``."""
    from opt_core import report as _core_report
    out: List[str] = []
    trimul = _STATE["trimul"]
    if trimul is not None:
        eng, dis = int(_STATE["trimul_engaged"]), int(_STATE["trimul_disengaged"])
        shapes = ",".join(f"{k}:{v}" for k, v in sorted(_STATE["trimul_shapes"].items())) or "none"
        if eng:
            out.append(_core_report.lever_line(TAG, "trimul_chunk", "on", impl=PRODUCER_ROWCHUNK, origin="core", strategy="F7.chunked_eval",
                                               rows=trimul[0], min_residues=trimul[1], engaged_traces=eng, disengaged_traces=dis, shapes=shapes))
        else:
            out.append(_core_report.lever_line(TAG, "trimul_chunk", "skipped", reason=f"disengaged:no_compiled_length_reached_{trimul[1]}", impl=PRODUCER_ROWCHUNK,
                                               origin="core", strategy="F7.chunked_eval", rows=trimul[0], min_residues=trimul[1], engaged_traces=0, disengaged_traces=dis))
    return out


def print_lines(stream=None) -> List[str]:
    import sys
    out = lines()
    for line in out:
        print(line, file=stream or sys.stderr, flush=True)
    return out
