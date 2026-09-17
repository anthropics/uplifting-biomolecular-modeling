"""Host-side pipeline primitives: the work around the network that a pipeline repeats — writing outputs off the critical path,
skipping parameter initialisation a checkpoint load overwrites, and running many items in one resident process with every item
accounted for.

Contract. Nothing here imports an engine, torch, jax or numpy: the kit passes its own objects in (the writer function, the modules
whose initialisers are suppressed, the per-item run and reseed callables) and gets engine-free behaviour plus the record of what
happened out. No primitive prints: each returns its evidence as ``key=value`` fields (``fields()``) that the kit appends to its own
ACTIVE line through :func:`opt_core.report.kv` — one line per arm per process stays the kit's. Every primitive is fail-loud: a write
that did not land, an initialiser skipped over a parameter the checkpoint did not restore, an item that raised — each is a named entry
in the returned record, never a silent branch.

    bg_writer     :class:`~opt_core.host_cache.bg_writer.BackgroundWriter` — output files written by worker threads / processes while
                  the device computes the next item; mandatory ``join()``; census ``written/submitted``
    fast_init     :func:`~opt_core.host_cache.fast_init.suppressed` — named initialiser functions are no-ops while a checkpoint load
                  runs; :func:`~opt_core.host_cache.fast_init.check_loaded` refuses a load that left parameters unrestored
    resident      :func:`~opt_core.host_cache.resident.run_items` — the item loop of a resident process: per-item hook (reseed),
                  journal, census ``ok/requested`` with every failure's reason, the ``incomplete`` clause for :func:`opt_core.report.verdict`
"""
from .. import lazy_getattr

__all__ = ["bg_writer", "fast_init", "resident"]
__getattr__ = lazy_getattr(__name__)          # every sub-module by name, on first access (PEP 562): importing the package imports nothing
