"""The single-GPU row-blocked outer-product-mean: ``msa.opm_rows_budgeted(..., allow_unsharded=True)`` / ``transition.opm_rows(...,
allow_unsharded=True)`` on an UNSHARDED layout ``Layout(N, 1, 0)`` walk all N rows block by block and equal the dense statement bit-exact
(given rows, budget-chosen rows, in-place accumulation into z); without the flag the unsharded layout is refused by name; ``a_local`` is
refused under the flag; the schedule census gains ``opm_layout=unsharded``."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import evidence as EV, msa as M, transition as T  # noqa: E402
from opt_core.mem.rowpair.dist import Layout, RowpairRefused  # noqa: E402

S, N, c, CZ = 5, 37, 4, 16


def _operands():
    g = torch.Generator().manual_seed(0)
    a, b, W = torch.randn(S, N, c, generator=g), torch.randn(S, N, c, generator=g), torch.randn(c * c, CZ, generator=g)

    def outer(a_blk, b_):                                                   # [S, r, c] x [S, N, c] -> [r, N, CZ]  (mean over S, projection)
        o = torch.einsum("sic,sjd->ijcd", a_blk, b_).reshape(a_blk.shape[1], N, c * c) / S
        return o @ W
    return a, b, outer, outer(a, b)


def test_unsharded_layout_refused_without_the_flag():
    a, b, outer, _ = _operands()
    for fn in (lambda: T.opm_rows(a, b, Layout(N, 1, 0), outer, 8), lambda: M.opm_rows_budgeted(a, b, Layout(N, 1, 0), outer, 8, C_z=CZ)):
        with pytest.raises(RowpairRefused, match="opm_rows"):
            fn()


@pytest.mark.parametrize("rows", [1, 8, 37, None])
def test_unsharded_opm_rows_equal_dense(rows):
    a, b, outer, dense = _operands()
    EV.reset_schedule() if hasattr(EV, "reset_schedule") else None
    kw = dict(C_z=CZ, allow_unsharded=True)
    if rows is None:
        kw["budget_bytes"] = 3 * N * 2 * CZ * 4                                # the budget form: a few rows per block
    got = M.opm_rows_budgeted(a, b, Layout(N, 1, 0), outer, rows, **kw)
    assert torch.equal(got, dense)
    facts = dict(EV.schedule_fields())
    assert facts.get("opm_layout") == "unsharded" and "opm_rows" in facts and "opm_rows_source" in facts, facts


def test_unsharded_opm_inplace_add_into_z():
    a, b, outer, dense = _operands()
    z = torch.randn(N, N, CZ, generator=torch.Generator().manual_seed(1))
    z0 = z.clone()
    out = M.opm_rows_budgeted(a, b, Layout(N, 1, 0), outer, 8, C_z=CZ, out=z, add=True, allow_unsharded=True)
    assert out.data_ptr() == z.data_ptr() and torch.equal(z, z0 + dense)


def test_unsharded_refuses_a_local():
    a, b, outer, _ = _operands()
    with pytest.raises(RowpairRefused, match="a_local"):
        M.opm_rows_budgeted(a, b, Layout(N, 1, 0), outer, 8, C_z=CZ, allow_unsharded=True, a_local=True)
