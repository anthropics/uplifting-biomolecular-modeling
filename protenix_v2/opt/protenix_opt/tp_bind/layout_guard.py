"""The rank-side guard of the unit's row layout (``ptx_tp.trunk.make_layout``): the layout a rank builds must SHARD (never the unit's
replicated regime), give every rank rows, and use a supported grid (``tp.BLOCKS_SUPPORTED``); anything else raises the core's
``RowpairRefused`` by name in the rank instead of running. The line's launcher refuses such inputs before any rank starts
(``tp.block_for``); this guard makes the same rule hold inside the rank process whatever environment reached it.
``tp_route.install`` binds it over the unit's name (the unit's trunk and runner hooks call ``make_layout`` through the module global)."""
from __future__ import annotations

import functools

from opt_core.mem.rowpair import RowpairRefused

__all__ = ["SITE", "guard_factory", "check"]

SITE = ("ptx_tp.trunk", "make_layout")


def check(layout, supported) -> None:
    if layout.P > 1 and layout.replicated:
        raise RowpairRefused(f"make_layout: N={layout.N} < P*B={layout.P}*{layout.B}: the row grid cannot shard — the replicated regime is not a "
                             f"setting of the multi-GPU line (use fewer GPUs for this input)", "protenix_v2.tp")
    empty = [q for q in range(layout.P) if layout.nrows(q) == 0]
    if empty:
        raise RowpairRefused(f"make_layout: N={layout.N} P={layout.P} B={layout.B}: ranks {empty} own zero rows", "protenix_v2.tp")
    if layout.P > 1 and int(layout.B) not in tuple(supported):
        raise RowpairRefused(f"make_layout: the B={layout.B} layout grid is not supported (supported: {'/'.join(map(str, supported))})", "protenix_v2.tp")


def guard_factory(original):
    from ..tp import BLOCKS_SUPPORTED

    @functools.wraps(original)
    def make_layout(N):
        layout = original(N)
        check(layout, BLOCKS_SUPPORTED)
        return layout
    return make_layout
