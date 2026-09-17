"""Host-side primitives: the work a kit does around the model call, engine-free.

Contract. Four mechanisms, one module each, all importable on a box with no GPU framework installed (``import opt_core.host``
touches the standard library only; a function that needs torch or jax imports it inside the call and answers a missing or too-old
framework with a NAMED refusal, never a silent stock path):

``memo``       :class:`~opt_core.host.memo.Memo` — a content-addressed memo of an expensive host-side build (a featurisation, an MSA
               parse, a conformer set, a language-model embedding): the key is the sha256 of every input that determines the value
               (bytes, seed, builder version, settings — the adapter names them), the value is whatever the build returns; bounded
               in memory, optionally mirrored to a directory, with a check-every-N recomputation that fails CLOSED on disagreement.
``workers``    :class:`~opt_core.host.workers.Prefetcher` — builds item k+1 on host threads (or processes) while the consumer works on
               item k; delivery in input order, bounded look-ahead, a failed build delivered WITH its item (raised at consumption or
               yielded as a record — the consumer accounts for it), never dropped.
``outputs``    :func:`~opt_core.host.outputs.to_host` (one bulk device→host transfer of a result tree instead of one lazy transfer per
               leaf), :class:`~opt_core.host.outputs.PinnedPool` (re-used pinned staging buffers for non-blocking copies) and
               :class:`~opt_core.host.outputs.AsyncWriter` (file writing off the critical path in thread or process workers, the item owned at submit, drained with total accounting, an exit guard for a writer left undrained).
``coldstart``  :func:`~opt_core.host.coldstart.no_init` / :func:`~opt_core.host.coldstart.meta_init` /
               :func:`~opt_core.host.coldstart.mmap_state_dict` — construct a model without the random initialisation a checkpoint
               load overwrites, and read the checkpoint without a second copy; each form proves the load was TOTAL (no parameter or
               buffer left as constructed) or refuses by name.

Evidence. Every primitive describes itself for the kit's ONE activation-evidence line per lever (``evidence()`` → the fields,
``active_line(tag, name)`` → the core's LEVER form ``[<tag>] LEVER name=<F6.strategy> state=on impl=<lever> origin=core k=v ...``; a refusal prints
``state=skipped reason=..`` through ``exc.event.lever_line(tag, name)``) and censuses itself at exit (``tally()`` / ``tally_line(tag)`` →
``[<tag>] HOST <lever> TALLY ...``; :func:`opt_core.report.register_exit_tally` prints it once). A degraded
path is an event with a name (:class:`~opt_core.host.events.HostEvent`: ``REFUSED`` · ``BYPASS`` · ``FALLBACK`` · ``MISMATCH``),
counted in the tally and, where the primitive cannot continue correctly, raised as :class:`~opt_core.host.events.HostRefused` — the
adapter logs the event's line and decides; nothing here falls back to a stock path on its own.

Numerics. ``memo``, ``workers`` and ``outputs`` move no arithmetic: a kit line that adds them stays in its class (exact stays exact)
PROVIDED the memo key names every input of the build and the build draws its randomness from the item (a seed IN the key), not from
process-global state consumed in build order — the adapter states both in the kit's CHANGES entry, and the engine's equality tests are
the proof. ``coldstart`` changes no weight that the checkpoint covers; its totality check is what makes that sentence true for the engine.

The engine-specific glue (which function builds features, which object is the model, where the outputs are written) lives in the
kit's adapter, never here; this package names no engine.
"""
from __future__ import annotations

from .events import HostEvent, HostRefused

__all__ = ["HostEvent", "HostRefused", "memo", "workers", "outputs", "coldstart"]
