"""opt_core.seq — sequence-model strategy modules: the shared levers of the kits whose engines read or write biological sequences
(precision policies, attention-backend selection, capture and warm-up witnesses, variable-length batching plans, resident lookup
tables, host-side input/output adapters, allocator and host-environment settings, memory budgets, deterministic torch recipes).

Contract (the core's, restated for this sub-package):
  * standard library only at module top level; framework code (torch, triton, numpy, ...) is imported inside the function that
    needs it, so importing any module here costs no framework import and a kit that never calls a lever never loads its framework;
  * a framework a module needs but cannot import is a named refusal raised at the call, never a silent fallback;
  * no engine knowledge: every engine-specific value (shapes, tables, thresholds, switch names) is data the kit passes in, and a
    kit wires a module through a thin adapter inside its own package;
  * the activation-evidence hook of every module is `line_fields(...) -> dict` (str -> str): the fields the kit puts on its one
    activation line through `opt_core.report.kv(**fields)`, so a run record proves the lever was wired and active; no other name;
  * Python floor, as `tests/test_selfcontained.py` enforces it: this file and the modules an engine on Python 3.8 imports — `lut`,
    `numerics`, `det_torch`, `batch_plan`, `alloc_env` — parse and run on 3.8 (like the core's generic modules); every other module
    here holds 3.9.

Sub-modules resolve lazily on attribute access (PEP 562) and `dir()` lists the ones present on disk; nothing is imported here.
"""
from __future__ import annotations

__all__ = []


from .. import lazy_getattr

__getattr__ = lazy_getattr(__name__)          # every sub-module by name, on first access


def __dir__():
    return sorted({n for n in globals() if n.startswith("__")} | set(_present()))
