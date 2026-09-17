"""opt_core.mem.rowpair.rng — rows of a stock CUDA Philox draw (trunc_normal_ / fused dropout), 0.4.3.

CPU (always): the launch-geometry arithmetic (grid, counter offset, the 32-bit split of a 1-D iterator) against independent statements
written here; refusals by name (CPU device, offset not a multiple of 4, batch != 1, non-saturating dropout size, the log-pdf branch);
``agree_rounds`` = equality without a group and MAX over ranks under gloo (P = 2, 3). GPU (skipped without CUDA — the replica re-issues CUDA
Philox launches and CPU generators have no offset): ``trunc_normal_rows`` over a row partition is BIT-EXACT the stock ``nn.init.trunc_normal_``
tensor's rows and ``finish_generator(agreed max rounds)`` lands on the stock generator offset (also with narrow bounds forcing many rejection
rounds, and with a bf16 ``out_dtype``); ``dropout_rows`` is BIT-EXACT ``F.dropout(training=True)`` rows + mask with the stock offset advance;
``selfcheck`` reports equal and restores the generator. On a torch whose ``trunc_normal_`` is the inverse-CDF form the trunc-normal cases
assert the refusal by name instead.

Run: ``python -m pytest tests/test_rowpair_rng_043.py -q -rfE -s`` (GPU cases need one CUDA device).
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pytest  # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
    HAVE_CUDA = bool(torch.cuda.is_available())
except Exception:  # noqa: BLE001
    HAVE_TORCH = False
    HAVE_CUDA = False

needs_torch = pytest.mark.skipif(not HAVE_TORCH, reason="torch not importable")
needs_cuda = pytest.mark.skipif(not HAVE_CUDA, reason="CUDA Philox rows replica: needs a CUDA device (CPU generators have no offset)")


# ================================================================================================================ CPU: geometry + refusals
def _ref_split(numel, itemsize):
    """Independent recursive statement of the 32-bit split: halve (first half = floor) until every piece fits, address order."""
    INT32_MAX = 2 ** 31 - 1

    def fits(n):
        return n <= INT32_MAX and 1 + (n - 1) * itemsize <= INT32_MAX

    def rec(s, n):
        if fits(n):
            return [(s, n)]
        first = n // 2
        return rec(s, first) + rec(s + first, n - first)
    return rec(0, numel)


@needs_torch
def test_split_until_32bit_matches_recursive_halving():
    from opt_core.mem.rowpair import rng as R
    for numel, isz in [(1000, 4), (2 ** 29, 4), (2 ** 29 + 1, 4), (2 ** 29 + 4, 4), (3 * 2 ** 29 + 17, 4), (2 ** 31 + 12, 4), (2 ** 30 + 3, 8),
                       (2 ** 33 + 5, 2)]:
        got = R._split_until_32bit(numel, isz)
        assert got == _ref_split(numel, isz), (numel, isz, got[:4])
        assert sum(n for _, n in got) == numel and got[0][0] == 0
        assert all(got[i][0] + got[i][1] == got[i + 1][0] for i in range(len(got) - 1))          # contiguous, address order
        assert all(R._fits_32bit(n, isz) for _, n in got)
    assert R._split_until_32bit(2 ** 29, 4) == [(0, 2 ** 29)]                                    # exactly fits: (n-1)*4+1 = 2^31-3
    assert len(R._split_until_32bit(2 ** 29 + 1, 4)) == 2


@needs_torch
def test_launch_plan_arithmetic_injected_device():
    """grid = min(ceil(N/256), SM * (MTP // 256)); counter = ((N-1) // (256*grid*unroll) + 1) * 4; split pieces get their own offsets after the
    unused whole-tensor state. Device facts injected (132 SMs x 2048 threads: 1056 resident blocks)."""
    from opt_core.mem.rowpair import rng as R
    sm, mtp = 132, 2048
    gmax = sm * (mtp // 256)
    assert R._grid(1000, sm, mtp) == 4 and R._grid(10 ** 9, sm, mtp) == gmax == 1056
    assert R._counter_offset(1000, sm, mtp) == 4                                                   # one curand4 call per thread
    n = 256 * gmax * 4 * 10 + 1                                                                    # 10 full sweeps + 1 element -> 11 calls
    assert R._counter_offset(n, sm, mtp) == 44
    pl = R._launch_plan_for(10 ** 6, 4, sm, mtp)
    assert pl.launches == ((0, 10 ** 6, 0),) and pl.round_increment == R._counter_offset(10 ** 6, sm, mtp)
    big = 2 ** 29 + 4                                                                              # fp32 byte extent > 2^31-1 -> split in two
    pl2 = R._launch_plan_for(big, 4, sm, mtp)
    first = R._counter_offset(big, sm, mtp)
    halves = _ref_split(big, 4)
    assert [(s, n) for s, n, _ in pl2.launches] == halves
    assert pl2.launches[0][2] == first                                                             # the whole-tensor state is consumed, unused
    assert pl2.launches[1][2] == first + R._counter_offset(halves[0][1], sm, mtp)
    assert pl2.round_increment == first + sum(R._counter_offset(n, sm, mtp) for _, n in halves)
    assert pl2.round_increment % 4 == 0 and "launches=2" in pl2.describe()
    # dropout plan: saturation rule and period
    dp = R._dropout_plan_for(10 ** 7, sm, mtp)
    assert dp.grid == gmax and dp.period == 4 * 256 * gmax and dp.min_saturating == (gmax - 1) * 256 + 1
    assert dp.round_increment == ((10 ** 7 - 1) // (256 * gmax * 4) + 1) * 4
    with pytest.raises(R.RowpairRefused) as e:
        R._dropout_plan_for(dp.min_saturating - 1, sm, mtp)
    assert "does not saturate the grid" in e.value.reason
    assert R._dropout_plan_for(dp.min_saturating, sm, mtp).grid == gmax


@needs_torch
def test_refusals_by_name_on_cpu():
    import torch
    from opt_core.mem.rowpair import rng as R
    from opt_core.mem.rowpair import RowpairRefused
    with pytest.raises(RowpairRefused) as e1:
        R.cuda_default_generator("cpu")
    assert "not CUDA" in e1.value.reason
    with pytest.raises(RowpairRefused) as e2:
        R.set_offset(None, 6)
    assert "multiple of 4" in e2.value.reason
    with pytest.raises(RowpairRefused):
        R.get_offset(torch.Generator())                                                            # CPU generator: no offset
    with pytest.raises(RowpairRefused) as e3:
        R.trunc_normal_rows(8, 4, 0, 8, std=0.1, B=2)
    assert "batch of 1" in e3.value.reason
    algo = R.trunc_normal_algorithm()
    assert algo in ("rejection", "inverse_cdf", "unknown")
    with pytest.raises(RowpairRefused) as e4:
        R.trunc_normal_rows(8, 4, 0, 8, std=1.0, a=-0.1, b=0.1, device="cpu")                       # p <= 0.3 (or the algorithm / device refusal)
    assert any(k in e4.value.reason for k in ("log-pdf branch", "form", "not CUDA")), e4.value.reason
    with pytest.raises(RowpairRefused) as e5:
        R.dropout_rows(None, 5, 3, 0, 5, p=0.1, device="cpu")                                      # 75 elements: not a multiple of 4
    assert "multiple of 4" in e5.value.reason or "not CUDA" in e5.value.reason
    from opt_core.mem.rowpair.dist import Layout
    with pytest.raises(RowpairRefused) as e6:
        R.trunc_normal_shard(Layout(64, 1, 0), 8, std=0.1)
    assert "refused at n_gpu=1" in e6.value.reason
    assert R.agree_rounds(3) == 3                                                                   # equality without a group


def _entry_agree():
    import torch  # noqa: F401
    from opt_core.mem.rowpair import dist as D
    from opt_core.mem.rowpair import rng as R
    P, r = D.world()
    got = R.agree_rounds(2 * r + 1, device="cpu")
    return {"P": P, "got": got, "want": 2 * (P - 1) + 1, "ok": got == 2 * (P - 1) + 1}


@needs_torch
@pytest.mark.parametrize("P", [2, 3])
def test_mp_agree_rounds_is_max_over_ranks(P):
    from opt_core.mem.rowpair import launch
    res = launch.run_sharded(P, _entry_agree, mode="big", backend="gloo", cpu_ok=True, nccl_timeout_s=120, run_timeout_s=300)
    print("RESULT " + json.dumps(dict(res, case="agree_rounds")))
    assert res["ok"], res


# ================================================================================================================ GPU: bit-exact vs the stock calls
def _report(d):
    print("RESULT " + json.dumps(d, sort_keys=True, default=str))


@needs_cuda
@pytest.mark.parametrize("case", ["esm_bounds", "narrow_bounds_many_rounds", "bf16_out"])
def test_gpu_trunc_normal_rows_bitwise_vs_stock(case):
    import torch
    from opt_core.mem.rowpair import rng as R
    from opt_core.mem.rowpair import RowpairRefused
    dev = torch.device("cuda", torch.cuda.current_device())
    L, C = 61, 32
    std = math.sqrt(2.0 / (5.0 * C))
    k = 1.0 if case == "narrow_bounds_many_rounds" else 3.0                                        # p = 0.68 -> several rejection rounds
    out_dtype = torch.bfloat16 if case == "bf16_out" else torch.float32
    gen = R.cuda_default_generator(dev)
    torch.cuda.manual_seed(1234)
    base = R.get_offset(gen)
    t = torch.empty((1, L, L, C), dtype=torch.float32, device=dev)
    if R.trunc_normal_algorithm() != "rejection":
        with pytest.raises(RowpairRefused) as e:
            R.trunc_normal_rows(L, C, 0, L, std=std, a=-k * std, b=k * std, device=dev)
        assert "form" in e.value.reason
        _report({"case": f"trunc_normal/{case}", "refused": e.value.reason, "torch": torch.__version__})
        return
    torch.nn.init.trunc_normal_(t, mean=0.0, std=std, a=-k * std, b=k * std)
    off_stock = R.get_offset(gen)
    t = t.to(out_dtype)
    parts = [(0, 1), (1, 16), (16, 40), (40, 60), (60, 61)]
    rounds, incr, eq, maxabs = [], None, [], 0.0
    for g0, g1 in parts:
        rows, info = R.trunc_normal_rows(L, C, g0, g1, std=std, mean=0.0, a=-k * std, b=k * std, device=dev, generator=gen, base_offset=base,
                                         out_dtype=out_dtype, elems_budget=7 * L * C)               # 7-row super-blocks inside each part
        eq.append(bool(torch.equal(rows, t[:, g0:g1])))
        maxabs = max(maxabs, float((rows.float() - t[:, g0:g1].float()).abs().max()))
        rounds.append(info["rounds"]); incr = info["round_increment"]
    final = R.finish_generator(gen, base, incr, max(rounds))
    res = {"case": f"trunc_normal/{case}", "torch": torch.__version__, "device": torch.cuda.get_device_name(dev), "L": L, "C": C,
           "equal_per_part": eq, "maxabs": maxabs, "rounds_per_part": rounds, "global_rounds": max(rounds),
           "stock_offset_advance": off_stock - base, "replica_offset_advance": final - base, "offset_equal": final == off_stock,
           "launches_per_draw": info["launches_per_draw"]}
    res["ok"] = all(eq) and res["offset_equal"]
    if case == "narrow_bounds_many_rounds":
        res["ok"] = res["ok"] and max(rounds) >= 1
    _report(res)
    assert res["ok"], res


@needs_cuda
@pytest.mark.parametrize("with_input", [True, False])
def test_gpu_dropout_rows_bitwise_vs_stock(with_input):
    import torch
    from opt_core.mem.rowpair import rng as R
    dev = torch.device("cuda", torch.cuda.current_device())
    sm, mtp = R._mp_props(dev)
    C = 32
    period = 4 * 256 * sm * (mtp // 256)                                                           # elements per curand_uniform4 sweep of the grid
    L = int(math.ceil(math.sqrt(3.2 * period / C))) + 1                                            # >= 3 periods: parts start mid-period, F0 > 0
    p = 0.25
    g = torch.Generator(device=dev); g.manual_seed(5)
    x = torch.randn((1, L, L, C), generator=g, device=dev) if with_input else torch.ones((1, L, L, C), device=dev)
    gen = R.cuda_default_generator(dev)
    torch.cuda.manual_seed(99)
    base = R.get_offset(gen)
    o = torch.nn.functional.dropout(x, p=p, training=True)
    off_stock = R.get_offset(gen)
    parts = [(0, 1), (1, L // 3), (L // 3, L - 2), (L - 2, L)]
    assert L * L * C >= 3 * period and (L // 3) * L * C % period != 0                             # multi-period draw, unaligned part starts
    eq_o, eq_m, final, launches = [], [], None, 0
    for g0, g1 in parts:
        xr = x[:, g0:g1].contiguous() if with_input else None
        orow, mrow, info = R.dropout_rows(xr, L, C, g0, g1, p=p, generator=gen, base_offset=base, device=dev, finish=False,
                                          elems_budget=max(L * C, period // 3))                   # several staging launches per part
        launches += info["launches"]
        eq_o.append(bool(torch.equal(orow, o[:, g0:g1])))
        eq_m.append(bool(torch.equal(mrow, o[:, g0:g1] != 0)))                                     # randn / ones have no exact zeros
        final = info["final_offset"]
    R.set_offset(gen, final)
    res = {"case": f"dropout/{'input' if with_input else 'mask_only'}", "torch": torch.__version__, "device": torch.cuda.get_device_name(dev),
           "L": L, "C": C, "periods": round(L * L * C / period, 2), "staging_launches": launches, "equal_out_per_part": eq_o,
           "equal_mask_per_part": eq_m, "stock_offset_advance": off_stock - base, "replica_offset_advance": final - base,
           "offset_equal": final == off_stock, "plan": info["plan"]}
    res["ok"] = all(eq_o) and all(eq_m) and res["offset_equal"]
    _report(res)
    assert res["ok"], res


@needs_cuda
def test_gpu_selfcheck_reports_equal_and_restores_offset():
    import torch
    from opt_core.mem.rowpair import rng as R
    dev = torch.device("cuda", torch.cuda.current_device())
    gen = R.cuda_default_generator(dev)
    torch.cuda.manual_seed(3)
    before = R.get_offset(gen)
    rep = R.selfcheck(dev)
    after = R.get_offset(gen)
    _report({"case": "selfcheck", **{k: v for k, v in rep.items()}})
    assert after == before
    assert rep["dropout"].get("equal") is True and rep["dropout"].get("offset_equal") is True, rep
    if R.trunc_normal_algorithm() == "rejection":
        assert rep["trunc_normal"].get("equal") is True and rep["trunc_normal"].get("offset_equal") is True, rep
    else:
        assert "refused" in rep["trunc_normal"], rep


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-rfE", "-s"] + sys.argv[1:]))
