"""The block-streamed (z + zᵀ) readers on RAGGED row shards.

``dist.transpose_blocks`` walks ``max_q ceil(R_q / step)`` all-to-all rounds — a COLLECTIVE schedule — so ``step`` must be one value on every
rank. A ``rowpair._distogram_rows`` that clamps the ROWPAIR_ROWBLK_MB staging block to the rank's OWN row count (``n_max=R``):
below the budget (N ≲ 1.4 K at P = 2, C = 128 fp32) a ragged layout — N = 199 → rows 100 | 99 — gives rank 0 step 100 (one round) and rank 1
step 99 (two rounds): one all-to-all more on the short rank at the trunk's distogram, every later collective mis-paired, the NCCL watchdog
later in the confidence pair stack.
The step is clamped to the layout's WIDEST shard (``lay.n_max``) — ``rowpair._sym_zT_step`` — as the PDE twin
(``rowpair_heads._zzT_row_blocks``) always was. Pinned here: (1) the step, hence the round count, is rank-independent on ragged (N, P) cells and
byte-for-byte today's value on even ones (pure logic, no comm); (2) the distogram rows seam RUNS in ``blocks`` mode on threaded ranks (native
all_to_all → the blocks path engages, unlike the gloo process tests of test_rowpair_seams) at ragged N: every rank issues the same number of
all-to-alls and its rows equal the dense DistogramModule rows."""
import math
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
torch = pytest.importorskip("torch")
pytest.importorskip("opt_core.mem.rowpair.dist", reason="opt_core (opt_core.mem.rowpair) is required")

from boltz2_opt import rowpair  # noqa: E402

C_Z, ELT = 128, 4                                                       # boltz2's pair width, fp32 shards (the trunk's z rows)
RAGGED = [(199, 2), (201, 2), (481, 2), (41, 2), (199, 4), (203, 8)]    # N not a multiple of P: the last rank is one row short (align 1)
EVEN = [(612, 2), (956, 2), (4048, 2), (1400, 2), (10120, 4)]           # even splits


def _lays(N, P):
    from opt_core.mem.rowpair import dist as D
    return [D.Layout(N, P, q, B=1, align=1) for q in range(P)]         # the kit's programme grid: balanced row parts, align 1 (boltz pins no chunk grid)


def _rounds(lays, step):
    return max(math.ceil(l.R / step) for l in lays)                     # dist.transpose_blocks' round count, as seen by a rank whose step is `step`


@pytest.mark.parametrize("N,P", RAGGED + EVEN)
def test_sym_zT_step_is_one_value_on_every_rank(N, P, monkeypatch):
    from opt_core.mem.rowpair import shard as SHD
    monkeypatch.delenv("ROWPAIR_ROWBLK_MB", raising=False)
    lays = _lays(N, P)
    assert sum(l.R for l in lays) == N and max(l.R for l in lays) - min(l.R for l in lays) <= 1
    steps = [rowpair._sym_zT_step(l, N, C_Z, ELT) for l in lays]
    assert len(set(steps)) == 1, (N, P, steps)                          # ONE step on every rank ...
    assert len({_rounds(lays, s) for s in steps}) == 1                 # ... hence one round count: the ranks meet in the same all-to-alls
    step = steps[0]
    assert 1 <= step <= max(l.R for l in lays)                          # never longer than the widest shard (staging stays [N, step, C], never N x N)
    budget, _ = SHD.choose_block_rows(N=N, C=C_Z, elem_bytes=ELT, n_max=None)
    assert step == min(int(budget), max(l.R for l in lays))            # = the ROWBLK budget clamped to Rmax: on every EVEN cell and above the budget byte for byte the per-rank clamp's value
    if (N, P) in RAGGED and budget >= max(l.R for l in lays):           # the defect class, documented: a per-rank clamp gives the ranks different steps AND round counts here
        local = [max(1, min(int(budget), l.R)) for l in lays]
        assert len(set(local)) > 1 and len({_rounds(lays, s) for s in local}) > 1, (N, P, local)


def _disto_entry(rank, P, N: int, dm, z):
    """One threaded rank: the distogram rows seam in blocks mode on its shard of the shared dense z; returns its rows, the all-to-all count
    of its comm through the seam and the census the seam recorded."""
    from opt_core.mem.rowpair import dist as D, evidence as EV
    torch.set_num_threads(1)
    P_, r = D.world()
    assert (P_, r) == (P, rank)
    lay = D.Layout.checked(N, P, r, B=1, align=1)
    assert rowpair._sym_blocks_streamed() is True                      # threaded comm: native all_to_all -> the blocks path (the one NCCL runs)
    z_loc = z[:, lay.r0:lay.r1].contiguous()
    cm = D.comm()
    c0 = int(cm.stats.get("comm_calls", 0))
    with torch.no_grad():
        out = rowpair._distogram_rows(dm, z_loc, lay)
    calls = int(cm.stats.get("comm_calls", 0)) - c0
    sch = dict(EV.schedule() or {})
    return {"rank": r, "rows": (lay.r0, lay.r1), "out": out.clone(), "comm_calls": calls,
            "disto_zT": sch.get("disto_zT"), "disto_zT_rows": sch.get("disto_zT_rows")}


@pytest.mark.parametrize("N,P", [(41, 2), (43, 4), (40, 2)])           # 41 | 43: ragged (21|20, 11|11|11|10); 40: even control
def test_distogram_rows_blocks_on_threaded_ranks_ragged(N, P, monkeypatch):
    pytest.importorskip("boltz.model.modules.trunkv2", reason="the pure-python boltz 2.2.1 model tree is required (pip install --no-deps boltz==2.2.1 einops)")
    from boltz.model.modules.trunkv2 import DistogramModule
    from opt_core import testing
    monkeypatch.delenv(rowpair.SYM_ZT_ENV, raising=False)              # the ×P line's default: blocks
    monkeypatch.setenv("ROWPAIR_ROWBLK_MB", "1")                        # 1 MiB / (N * 32 * 4 B) rows >= every shard here: the below-budget regime of the defect (step = the clamp)
    C = 32
    torch.manual_seed(3)
    dm = DistogramModule(C, 10).eval()
    z = torch.randn(1, N, N, C)
    with torch.no_grad():
        ref = dm(z)                                                     # the dense stock statement: distogram(z + zᵀ)
    rowpair._STATE.setdefault("calls", {}).setdefault("transposes", 0)
    rowpair._STATE["calls"].setdefault("distogram_rows", 0)
    outs = testing.run_ranks(P, _disto_entry, N, dm, z, timeout_s=60)  # a rank-dependent round count would deadlock here (named timeout), as it wedged NCCL
    assert [o["rank"] for o in outs] == list(range(P))
    assert len({o["comm_calls"] for o in outs}) == 1, [o["comm_calls"] for o in outs]   # every rank issued the same number of all-to-alls
    assert all(o["disto_zT"] == "blocks" for o in outs), [o["disto_zT"] for o in outs]
    for o in outs:
        r0, r1 = o["rows"]
        d = (o["out"] - ref[:, r0:r1]).abs().max().item() / max(ref.abs().max().item(), 1e-30)
        assert d <= 1e-5, (o["rank"], d)
    assert sum(o["rows"][1] - o["rows"][0] for o in outs) == N
