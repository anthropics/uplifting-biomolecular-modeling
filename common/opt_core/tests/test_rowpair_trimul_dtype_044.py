"""A COMMUNICATED trimul operand must carry the pair schedule dtype: a ``TriMulFns.proj`` that returns another dtype (an autocast projection)
is refused BY NAME by ``trimul_update_`` (the b slabs enter the ring; a rank owning no sub-block at some ring step would post the schedule-dtype
empty slab while producers post theirs — transfers of different byte counts), at P in {2, 3} on rank-threads with uneven sub-block counts, and
by the 2-process gloo transport; ``out`` / ``gate`` returning another dtype stay ALLOWED (local epilogue). ``ring_pass`` refuses an own block
whose dtype differs from the named schedule dtype."""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused                 # noqa: E402
from opt_core.mem.rowpair import trimul as TM, ring             # noqa: E402
from opt_core.mem.rowpair.dist import Layout                    # noqa: E402
from opt_core.testing import run_ranks                          # noqa: E402

N, C, C_H = 40, 8, 6


class _Fns(object):
    """proj = a linear map of the block; ``bad`` casts the b projection (or out) to another dtype, as an autocast region would."""

    def __init__(self, bad: str, g):
        self.W = torch.randn(C, C_H, generator=g) * 0.3
        self.Wo = torch.randn(C_H, C, generator=g) * 0.3
        self.bad = bad

    def proj(self, zblk, mblk, is_a):
        p = (zblk @ self.W) * mblk
        return p.to(torch.bfloat16) if (self.bad == "proj_b" and not is_a) or self.bad == "proj_both" else p

    def out(self, x):
        y = x @ self.Wo
        return y.to(torch.bfloat16) if self.bad == "out" else y

    def fns(self):
        return TM.TriMulFns(self.proj, self.out, None, C_H)


def _rank(rank, P, bad, outgoing):
    g = torch.Generator().manual_seed(0)
    z = torch.randn(N, N, C, generator=g)
    f = _Fns(bad, g)
    lay = Layout(N, P, rank, 8)                                   # grid B=8: 5 blocks over P ranks -> uneven sub-block counts at P in {2, 3}
    zs = z[lay.r0:lay.r1].contiguous()
    try:
        TM.trimul_update_(f.fns(), zs, None, lay, outgoing=outgoing, inplace_chunk=8)
        return "ran"
    except RowpairRefused as e:
        return str(e)


@pytest.mark.parametrize("P", [2, 3])
@pytest.mark.parametrize("outgoing", [True, False])
@pytest.mark.parametrize("bad", ["proj_b", "proj_both"])
def test_proj_in_another_dtype_is_refused_by_name(P, outgoing, bad):
    res = run_ranks(P, _rank, bad, outgoing)
    assert all("pair schedule is torch.float32" in r and "proj must return" in r for r in res), res


@pytest.mark.parametrize("P", [2, 3])
def test_out_in_another_dtype_stays_allowed(P):
    res = run_ranks(P, _rank, "out", True)
    assert all(r == "ran" for r in res), res


@pytest.mark.parametrize("P", [2])
def test_clean_fns_run(P):
    res = run_ranks(P, _rank, "none", False)
    assert all(r == "ran" for r in res), res


def _ring_pass_rank(rank, P):
    lay = Layout(N, P, rank)
    own = torch.zeros(3, 4, dtype=torch.bfloat16)
    try:
        for _ in ring.ring_pass(own, lay, [[3, 4]] * P, dtype=torch.float32, device="cpu"):
            pass
        return "ran"
    except RowpairRefused as e:
        return str(e)


def test_ring_pass_refuses_own_block_in_another_dtype():
    res = run_ranks(2, _ring_pass_rank)
    assert all("ring_pass: own block is torch.bfloat16" in r for r in res), res


def _mp_entry():
    from opt_core.mem.rowpair.dist import default_ctx
    lay = default_ctx(N)
    return {"rank": lay.rank, "res": _rank(lay.rank, lay.P, "proj_b", True) if False else _mp_body(lay)}


def _mp_body(lay):
    g = torch.Generator().manual_seed(0)
    z = torch.randn(N, N, C, generator=g)
    f = _Fns("proj_b", g)
    try:
        TM.trimul_update_(f.fns(), z[lay.r0:lay.r1].contiguous(), None, lay, outgoing=True, inplace_chunk=8)
        return "ran"
    except RowpairRefused as e:
        return str(e)


def test_mp_refusal_gloo_processes():
    from opt_core.mem.rowpair import launch
    got = launch.run_sharded(2, _mp_entry, cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert got["rank"] == 0 and "proj must return" in got["res"], got


def test_the_rule_is_spelled_once():
    """SSOT: ``dist.require_schedule_dtype`` is the ONE statement of 'a communicated operand carries the schedule dtype/device'; trimul's b
    slabs, dist.alltoall_window and ring.ring_pass CALL it — no inline re-spelling of the refusal in those modules."""
    import glob
    import re
    rp = os.path.join(os.path.dirname(HERE), "opt_core", "mem", "rowpair")
    src = {os.path.basename(p): open(p, encoding="utf-8").read() for p in glob.glob(os.path.join(rp, "*.py"))}
    defs = [n for n, s in src.items() if re.search(r"^def require_schedule_dtype\(", s, re.M)]
    assert defs == ["dist.py"], defs
    spell = [n for n, s in src.items() if "but the pair schedule is" in s or "schedule names" in s]
    assert spell == ["dist.py"], spell
    for n in ("trimul.py", "ring.py"):
        assert "require_schedule_dtype(" in src[n], n
    assert src["dist.py"].count("require_schedule_dtype(") >= 2          # the definition + the alltoall_window call
