"""The n_gpu>1 MC-dropout statement of the recycling projection (tp.make_mc_dropout_rows).

`stock_mask` (N <= PROTENIX_V1_TP_MC_DROPOUT_FULL_MAX, default 2048): every rank draws the stock statement's ONE keep-mask over the whole
[N, N, c_z] plane through the GLOBAL generator — exactly the draw `F.dropout(x, p, training=True)` makes — and applies its rows; so on every rank
(a) the generator ends where the full-plane F.dropout leaves it (cycle after cycle), and (b) the rows this rank produces, assembled over the
ranks, ARE the full-plane F.dropout's output. `rank_streams` (above the ceiling): rank-forked private generators,
the global generator untouched. CPU, fp32, P ranks simulated in turn from one saved generator state (what the ×P trunk guard proves on the
box: identical replicated generator state on every rank at the statement). On CUDA F.dropout(training=True) IS torch.native_dropout (the fused
Philox kernel); with this form a ×2 process's Philox offset equals a ×1 process's at every item."""
import os

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("opt_core.mem.rowpair.dist")
from opt_core.mem.rowpair import dist as D          # noqa: E402

from protenix_v1_opt import tp                      # noqa: E402

F = torch.nn.functional
P_DROP = 0.4
C = 16


def _blocks(r0, r1, rows):
    """The consecutive row blocks trunk.recycle_shard_ visits on a rank: [r0, r1) in order, `rows` per block, the last ragged."""
    g = r0
    while g < r1:
        yield g, min(g + rows, r1)
        g = min(g + rows, r1)


def _run_rank(N, P, rank, B, x_full, state, cycles, rows, form="stock_mask", environ=None):
    """One rank's statement over `cycles` recycles from the saved global state; returns (rows out per cycle, final generator state)."""
    lay = D.Layout(N=N, P=P, rank=rank, B=B)
    torch.set_rng_state(state)
    drop = tp.make_mc_dropout_rows(lay, P_DROP, torch.device("cpu"), form)
    outs = []
    for _cycle in range(cycles):
        parts = [drop(x_full[g0:g1]) for g0, g1 in _blocks(lay.r0, lay.r1, rows or (lay.r1 - lay.r0))]
        outs.append(torch.cat(parts, dim=0) if parts else x_full[0:0])
    return (lay.r0, lay.r1), outs, torch.get_rng_state()


@pytest.mark.parametrize("N,P,B,rows", [(24, 2, 8, None), (24, 2, 8, 8), (40, 3, 8, 8), (40, 3, 8, 5), (17, 2, 8, 3)])
def test_stock_mask_rows_are_the_full_plane_dropout_and_the_generator_ends_where_F_dropout_leaves_it(N, P, B, rows):
    torch.manual_seed(1234 + N)
    x = torch.randn(N, N, C)
    torch.manual_seed(99)
    S = torch.get_rng_state()
    # the stock statement, twice (two recycles): F.dropout on the whole plane from the global generator
    ref, ref_states = [], []
    torch.set_rng_state(S)
    for _ in range(2):
        ref.append(F.dropout(x, P_DROP, training=True)); ref_states.append(torch.get_rng_state())
    assert not torch.equal(ref_states[0], S) and not torch.equal(ref_states[1], ref_states[0])      # the statement consumes the generator, per cycle
    covered = torch.zeros(N, dtype=torch.bool)
    for rank in range(P):
        (r0, r1), outs, final = _run_rank(N, P, rank, B, x, S, cycles=2, rows=rows)
        assert torch.equal(final, ref_states[1]), f"rank {rank}: generator not where two full-plane F.dropout calls leave it"
        for cyc in range(2):
            assert torch.equal(outs[cyc], ref[cyc][r0:r1]), f"rank {rank} cycle {cyc}: rows [{r0},{r1}) != the full-plane F.dropout rows"
        covered[r0:r1] = True
    assert bool(covered.all())


def test_stock_mask_generator_advance_after_ONE_cycle_equals_one_F_dropout_on_every_rank():
    N, P, B = 24, 2, 8
    x = torch.randn(N, N, C)
    torch.manual_seed(5); S = torch.get_rng_state()
    torch.set_rng_state(S); F.dropout(x, P_DROP, training=True); after_one = torch.get_rng_state()
    finals = [_run_rank(N, P, q, B, x, S, cycles=1, rows=8)[2] for q in range(P)]
    assert all(torch.equal(f, after_one) for f in finals)                                          # identical on every rank (the trunk guard's premise) and == the stock's


def test_rank_streams_leaves_the_global_generator_untouched_and_is_a_dropout():
    N, P, B = 24, 2, 8
    x = torch.ones(N, N, C)
    torch.manual_seed(5); S = torch.get_rng_state()
    for rank in range(P):
        (r0, r1), outs, final = _run_rank(N, P, rank, B, x, S, cycles=2, rows=8, form="rank_streams")
        assert torch.equal(final, S), "rank_streams must not consume the global generator"
        vals = torch.unique(outs[0])
        assert set(round(float(v), 4) for v in vals) <= {0.0, round(1.0 / (1.0 - P_DROP), 4)}       # kept values scaled by 1/(1-p), dropped zero
        frac = float((outs[0] == 0).float().mean())
        assert 0.3 < frac < 0.5                                                                     # ~p of the elements dropped


def test_form_by_token_ceiling_default_2048_and_the_env_word():
    assert tp.MC_DROPOUT_FULL_MAX_DEFAULT == 2048 and tp.MC_DROPOUT_FULL_MAX_ENV == "PROTENIX_V1_TP_MC_DROPOUT_FULL_MAX"
    assert tp.mc_dropout_form(2048, environ={}) == "stock_mask" and tp.mc_dropout_form(2049, environ={}) == "rank_streams"
    assert tp.mc_dropout_form(1340, environ={}) == "stock_mask" and tp.mc_dropout_form(3036, environ={}) == "rank_streams"   # either side of the default ceiling
    assert tp.mc_dropout_form(1340, environ={tp.MC_DROPOUT_FULL_MAX_ENV: "0"}) == "rank_streams"          # 0 = rank_streams at every N
    assert tp.mc_dropout_form(6000, environ={tp.MC_DROPOUT_FULL_MAX_ENV: "6000"}) == "stock_mask"
    assert tp.mc_dropout_form(24, environ={tp.MC_DROPOUT_FULL_MAX_ENV: " "}) == "stock_mask"              # blank = default
    for bad in ("-1", "abc", "4k"):
        with pytest.raises(tp.TPRefused):
            tp.mc_dropout_form(24, environ={tp.MC_DROPOUT_FULL_MAX_ENV: bad})
    with pytest.raises(tp.TPRefused):
        tp.make_mc_dropout_rows(D.Layout(N=24, P=2, rank=0, B=8), P_DROP, torch.device("cpu"), "whole")


def test_the_n_gpu_1_statement_is_untouched():
    """×1 inertness: big.py's get_pairformer_output keeps the stock F.dropout statement verbatim; the new names live in tp.py only."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    big_src = open(os.path.join(here, "big.py")).read()
    assert "F.dropout(self.linear_no_bias_z_cycle(self.layernorm_z_cycle(z)), p=self.configs.mc_dropout_rate)" in big_src
    assert "make_mc_dropout_rows" not in big_src and "MC_DROPOUT_FULL_MAX" not in big_src
