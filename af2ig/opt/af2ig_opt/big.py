"""big.py — the engine adapter of af2ig's memory mode: the one place the memory line's values live.

The line = ``fast``'s composition (the parallel precompile refused) + the memory levers of
``registry.MEMORY_DEFAULT``: ``XLA_PYTHON_CLIENT_MEM_FRACTION=<MEM_FRACTION>`` in the driver's environment (the reach lever); ``registry.MEMORY_OPT_IN``:
``-trimul_chunk <TRIMUL_ROWS>:<TRIMUL_MIN_RESIDUES>`` on the driver's argv only when ``AF2IG_OPT_TRIMUL_CHUNK`` opts it in (its body installed in the
driver process by ``af2ig_opt.pairstack`` over the tree's row-chunk producer; it costs memory and time at every size, so no shipped mode composes it)
(:func:`child_env`). One GPU. The flags themselves are ``modes.resolve``'s (nothing here is a second source of them).
"""
from __future__ import annotations

from . import modes

TRIMUL_ROWS, TRIMUL_MIN_RESIDUES = modes.TRIMUL_ROWS, modes.TRIMUL_MIN_RESIDUES        # the mode table's values (modes.py is the one place they live)
ENV_MEM_FRACTION, MEM_FRACTION = modes.ENV_MEM_FRACTION, modes.MEM_FRACTION
trimul_flag_value = modes.trimul_flag_value


def child_env(levers=None) -> dict:
    """The memory line's environment for the driver child: the XLA pool fraction (registry ``MEMF``; recorded by the driver in ``proc_start.env``) — empty when the
    lever set as run (``levers``) does not carry ``mem_fraction`` (``MODEL_OPT_LEVERS_OFF=mem_fraction``)."""
    from . import registry
    return {ENV_MEM_FRACTION: MEM_FRACTION} if levers is None or registry.MEMF in levers else {}
