"""opt_core.mem.rowpair.dist.gather_rows_to_rank0_host — a row-sharded ``[R, K, *rest]`` output assembled into a HOST tensor on rank 0 in
column blocks (bounded rank-0 device transient), bit-exact equal to the dense tensor; ranks > 0 get None. Threaded ranks (P in {2, 3}, grid and
aligned layouts, several block budgets incl. one column per block and one block for everything, a rank-1 ``[R]`` shard, a caller-supplied
``out``) and one 2-process gloo run through the kit launcher (the production transport on CPU)."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import dist as D  # noqa: E402
from opt_core.mem.rowpair import evidence  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402


def _x(N=40, K=33, rest=(5,), seed=0):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((N, K) + tuple(rest), generator=g, dtype=torch.float32)


def _worker(rank, P, x, block_bytes, align, use_out):
    lay = D.Layout(int(x.shape[0]), P, rank, 8, align=align) if align else D.Layout(int(x.shape[0]), P, rank, 8)
    xs = x[lay.r0:lay.r1].clone().contiguous()
    out = torch.empty_like(x) if (use_out and rank == 0) else None
    evidence.reset_schedule()
    got = D.gather_rows_to_rank0_host(xs, lay, block_bytes=block_bytes, out=out)
    sched = dict(evidence.SCHEDULE) if hasattr(evidence, "SCHEDULE") else {}
    if rank == 0:
        assert got is not None and got.device.type == "cpu"
        if use_out:
            assert got is out
        return bool(torch.equal(got, x)), sched
    assert got is None
    return None, sched


@pytest.mark.parametrize("P", [2, 3])
@pytest.mark.parametrize("align", [0, 4])
@pytest.mark.parametrize("block_bytes", [1, 40 * 5 * 4 * 7, 1 << 30])
def test_host_gather_equals_dense_threaded(P, align, block_bytes):
    x = _x()
    res = run_ranks(P, _worker, x, block_bytes, align, False)
    assert res[0][0] is True, res
    assert all(r[0] is None for r in res[1:])


def test_host_gather_rank1_shard_and_out_buffer():
    x1 = _x(K=1, rest=())[:, 0].contiguous()                                  # [N]
    res = run_ranks(2, _worker, x1, 64, 0, False)
    assert res[0][0] is True
    x = _x()
    res = run_ranks(3, _worker, x, 4096, 4, True)                             # caller-supplied host out on rank 0
    assert res[0][0] is True


def test_host_gather_p1_and_block_width():
    x = _x()
    lay = D.Layout(40, 1, 0, 8)
    got = D.gather_rows_to_rank0_host(x, lay, block_bytes=4000)
    assert got.device.type == "cpu" and torch.equal(got, x)
    assert D.host_gather_cols((40, 33, 5), 4, 4000) == 5                      # 4000 // (40*5*4) = 5 columns per block
    assert D.host_gather_cols((40, 33, 5), 4, 1) == 1 and D.host_gather_cols((40, 33, 5), 4, 1 << 40) == 33
    assert D.host_gather_cols((40,), 4, 1) == 0                               # rank-1: one block
    with pytest.raises(RowpairRefused):
        D.gather_rows_to_rank0_host(x[:7], lay)                               # P=1: the whole tensor is expected (rows != N refused by name)


def _mp_entry():
    """Each of 2 gloo processes (launch.run_sharded): rank 0 returns the verdict."""
    from opt_core.mem.rowpair import dist as DD
    P, r = DD.world()
    assert DD.comm().backend == "gloo" and P == 2
    x = _x(N=48, K=20, rest=(3,), seed=1)                                     # identical on both ranks (seeded)
    lay = DD.Layout(48, P, r, 8)
    got = DD.gather_rows_to_rank0_host(x[lay.r0:lay.r1].contiguous(), lay, block_bytes=48 * 3 * 4 * 6)   # 6 columns per block -> 4 blocks
    DD.barrier()
    if r == 0:
        return {"rank": r, "equal": bool(torch.equal(got, x)), "device": got.device.type, "cols": DD.host_gather_cols((48, 20, 3), 4, 48 * 3 * 4 * 6)}
    return {"rank": r, "none": got is None}


def test_mp_host_gather_gloo():
    from opt_core.mem.rowpair import launch
    res = launch.run_sharded(2, _mp_entry, cpu_ok=True, backend="gloo", run_timeout_s=300)
    assert res["rank"] == 0 and res["equal"] is True and res["device"] == "cpu" and res["cols"] == 6, res
