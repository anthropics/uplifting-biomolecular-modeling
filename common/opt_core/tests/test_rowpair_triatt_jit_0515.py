"""opt_core.mem.rowpair.triatt — the SHARDED-BIAS triangle-attention update (``ROWPAIR_TRIATT_BIAS=jit``: ``triatt_update_jit_`` +
``BiasBlockExchange``) versus the gather update (``triatt_update_``) on the threaded test hub (``opt_core.testing.run_ranks``): BITWISE equal for
both bias orientations (``bias_transposed`` False = plane rows per query block, per-owner broadcasts; True = plane columns, column-slab
all-gather), P in {2, 3}, uneven R, N not a multiple of the query block, grid and chunk-aligned layouts (both all-gather forms), masked rows,
``group`` 1 and 3, ``stream`` on / off, the raw core and the core door's torch path; the default word binds the gather update (nothing recorded);
the refusals by name (missing TriAttFns fields, P = 1, malformed lever words, ranks that disagree on the schedule facts).

Run: ``python -m pytest tests/test_rowpair_triatt_jit_0515.py -q -rfE`` or ``python tests/test_rowpair_triatt_jit_0515.py``.
"""
from __future__ import annotations

import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

import pytest  # noqa: E402

torch = pytest.importorskip("torch")
torch.set_num_threads(1)

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import evidence as EV  # noqa: E402
from opt_core.mem.rowpair import pairstack as PS  # noqa: E402
from opt_core.mem.rowpair import triatt as TA  # noqa: E402
from opt_core.mem.rowpair.dist import Layout, all_gather_rows  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402

INF = 1e9
RESULTS = []


def _report(name, **kw):
    line = "RESULT " + json.dumps(dict(case=name, **kw), sort_keys=True, default=str)
    RESULTS.append(line)
    print(line)


def _core4(q, k, v, biases):
    """Attention core o[.., j, :] = softmax_k(q_j . k + sum(bias[.., j, k])) v computed in QUERY CHUNKS OF 4 from the start of ITS q: a launch over
    queries [j0, j1) with j0 % 4 == 0 walks the same absolute chunks as the whole-row launch -> per-query results are launch-independent by
    construction (the property the flash kernel has on a GPU), so jit == gather can be asserted BITWISE on the CPU."""
    S = int(q.shape[-2])
    outs = []
    kT = k.transpose(-1, -2)
    for c0 in range(0, S, 4):
        c1 = min(S, c0 + 4)
        logits = torch.matmul(q[..., c0:c1, :], kT)
        for b in biases:
            if b is None:
                continue
            logits = logits + (b if int(b.shape[-2]) == 1 else b[..., c0:c1, :])
        outs.append(torch.matmul(torch.softmax(logits, dim=-1), v))
    return torch.cat(outs, dim=-2)


class TriAttJ(torch.nn.Module):
    """Synthetic starting-node triangle attention whose ``attend`` IS ``wrap(core(*proj(x), [mask_bias(m), view(tb_full)]), x)`` — the pieces the jit
    schedule takes. ``transposed``: the bias plane enters the kernel operand transposed (``bias[h, j, k] = plane[k, j, h]``: an engine that projects
    the ending node's bias in z's own frame); ``door``: ``core`` is the core door's torch path (``attention_core(kernel='torch')``, query sub-blocks of 8)
    instead of the raw core."""

    def __init__(self, C, H, d, g, transposed=False, door=False):
        super().__init__()
        L = lambda i, o, b=False: torch.nn.Linear(i, o, bias=b)
        self.ln, self.b = torch.nn.LayerNorm(C), L(C, H)
        self.q, self.k, self.v, self.gt, self.o = L(C, H * d), L(C, H * d), L(C, H * d), L(C, H * d, True), L(H * d, C, True)
        self.H, self.d, self.transposed = H, d, bool(transposed)
        for p in self.parameters():
            torch.nn.init.normal_(p, std=0.3, generator=g)
        self.core = TA.attention_core(_core4, kernel="torch", layout="bnhsd", mask_from="bias0", tri_bias="bias1", stock_qblock=8) if door else _core4
        self.calls = {"attend": 0, "proj": 0, "wrap": 0}

    def _heads(self, t):                                       # [I, J, H*d] -> [I, H, J, d]
        return t.unflatten(-1, (self.H, self.d)).transpose(-2, -3)

    def proj(self, x):
        self.calls["proj"] += 1
        return self._heads(self.q(x)) / math.sqrt(self.d), self._heads(self.k(x)), self._heads(self.v(x))

    def mask_bias(self, mask_rows):
        return (INF * (mask_rows - 1.0))[..., :, None, None, :]                          # [I, 1, 1, J]

    def view(self, tb_full):                                    # [N, N, H] -> [1, H, Nq, Nk]
        return (tb_full.permute(2, 1, 0) if self.transposed else tb_full.permute(2, 0, 1)).unsqueeze(0)

    def wrap(self, o, x):                                       # o [I, H, J, d]
        self.calls["wrap"] += 1
        return self.o(o.transpose(-2, -3).flatten(-2) * torch.sigmoid(self.gt(x)))

    def attend(self, x, mask_rows, tb_full, blk=None):
        self.calls["attend"] += 1
        q, k, v = self.proj(x)
        return self.wrap(self.core(q, k, v, [self.mask_bias(mask_rows), self.view(tb_full)]), x)

    def dense(self, y, mask):                                   # y + attention(y) with the plane b(LN(y)) in this module's orientation
        x = self.ln(y)
        return y + self.attend(x, mask, self.b(x))

    def fns(self, fields=True):
        if not fields:
            return TA.TriAttFns(self.ln, self.b, self.attend)
        return TA.TriAttFns(self.ln, self.b, self.attend, proj=self.proj, mask_bias=self.mask_bias, wrap=self.wrap, core=self.core, bias_transposed=self.transposed)


def _inputs(N, C, seed, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    y = torch.randn(N, N, C, generator=g, dtype=dtype)
    mask = (torch.rand(N, N, generator=g) > 0.15).to(dtype)
    mask[N // 3] = 0.0                                          # a fully masked row and column
    mask[:, N // 5] = 0.0
    return y, mask


LAYOUTS = [(2, 80, 16, None), (3, 104, 16, None), (3, 104, 16, 16), (2, 71, 16, None)]   # (P, N, B, align): grid (padded all-gather) x2, chunk-aligned (region broadcasts), odd N


def _entry_update(rank, P, N, B, align, mod, y, mask, which, kw):
    torch.set_grad_enabled(False)                               # grad mode is thread-local: the rank threads run inference
    lay = Layout(N, P, rank, B, align=align)
    ys = y[lay.r0:lay.r1].clone().contiguous()
    ms = mask[lay.r0:lay.r1].contiguous()
    EV.reset_schedule()
    fn = TA.triatt_update_jit_ if which == "jit" else TA.triatt_update_
    out = fn(mod.fns(), ys, ms, lay, **kw)
    assert out is ys
    return ys, EV.schedule(), bool(lay.padded_is_global)


@pytest.mark.parametrize("P,N,B,align", LAYOUTS)
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("group,rows,stream,door", [(1, 16, True, False), (3, 16, True, True), (2, 12, False, False)])
def test_jit_equals_gather_bitwise(P, N, B, align, transposed, group, rows, stream, door):
    g = torch.Generator().manual_seed(7)
    mod = TriAttJ(8, 2, 4, g, transposed=transposed, door=door)
    y, mask = _inputs(N, 8, 11)
    qb = 32
    with torch.no_grad():
        dense = mod.dense(y, mask)
        base_kw = dict(rows=rows, stream=stream, tb_rows=8)
        res_g = run_ranks(P, _entry_update, N, B, align, mod, y, mask, "gather", base_kw)
        res_j = run_ranks(P, _entry_update, N, B, align, mod, y, mask, "jit", dict(base_kw, qblock=qb, group=group, prefetch=True))
    got_g = torch.cat([r[0] for r in res_g], dim=0)
    got_j = torch.cat([r[0] for r in res_j], dim=0)
    sched = res_j[0][1]
    nb = len(TA.jit_qblocks(N, qb))
    m_dense = float((got_j.double() - dense.double()).abs().max())
    bit = bool(torch.equal(got_j, got_g))
    _report("jit_vs_gather", P=P, N=N, align=align, transposed=transposed, group=group, rows=rows, stream=stream, door=door, bitwise=bit,
            maxabs_vs_dense=m_dense, padded=res_j[0][2], census={k: v for k, v in sched.items() if k.startswith("triatt_")})
    assert bit, "jit != gather bitwise"
    assert m_dense <= 1e-4, m_dense
    assert sched["triatt_bias"] == f"jit:qb{qb}:{nb}", sched
    assert sched["triatt_jit_group"] == group and sched["triatt_jit_prefetch"] == 0          # CPU: fetched in line
    lay0 = Layout(N, P, 0, B, align=align)
    n_groups = -(-(-(-lay0.Rmax // rows)) // group)
    assert sched["triatt_jit_fetches"] == n_groups * nb, (sched["triatt_jit_fetches"], n_groups, nb)
    assert "triatt_jit_gib" in sched and "triatt_jit_wire_gib" in sched


def _entry_exchange(rank, P, N, B, align, tb_plane, qb, transposed, rounds):
    lay = Layout(N, P, rank, B, align=align)
    shard = tb_plane[lay.r0:lay.r1].contiguous()
    full = all_gather_rows(shard, lay)
    assert torch.equal(full, tb_plane)
    nbk = len(TA.jit_qblocks(N, qb))
    ex = TA.BiasBlockExchange(shard, lay, qb, total=rounds * nbk, transposed=transposed, prefetch=True)
    ok = []
    for f in range(rounds * nbk):
        (j0, j1), view = ex.get(f)
        ref = (tb_plane[:, j0:j1].permute(2, 1, 0) if transposed else tb_plane[j0:j1].permute(2, 0, 1)).unsqueeze(0)
        ok.append(bool(torch.equal(view, ref)) and tuple(view.shape) == (1, int(tb_plane.shape[2]), j1 - j0, N))
        ex.done(f)
    stats = ex.close()
    return ok, stats


@pytest.mark.parametrize("P,N,B,align", LAYOUTS)
@pytest.mark.parametrize("transposed", [False, True])
def test_exchange_blocks_equal_the_gathered_plane(P, N, B, align, transposed):
    g = torch.Generator().manual_seed(5)
    tb = torch.randn(N, N, 3, generator=g).to(torch.bfloat16)
    for qb in (32, 48, 4096):                                    # ragged last block; one block (qb >= N)
        res = run_ranks(P, _entry_exchange, N, B, align, tb, qb, transposed, 2)
        for r, (ok, stats) in enumerate(res):
            assert all(ok), (r, ok)
            assert stats["fetches"] == 2 * len(TA.jit_qblocks(N, qb))
        _report("exchange", P=P, N=N, align=align, transposed=transposed, qb=qb, stats=res[0][1])


def test_grid_and_words(monkeypatch):
    assert TA.jit_qblocks(80, 32) == [(0, 32), (32, 64), (64, 80)]
    assert TA.jit_qblocks(64, 32) == [(0, 32), (32, 64)]
    assert TA.jit_qblocks(20, 32) == [(0, 20)] and TA.jit_qblocks(20, 0) == [(0, 20)]
    monkeypatch.delenv(TA.ENV_TRIATT_BIAS, raising=False)
    assert TA.triatt_bias_word() == "gather" and TA.triatt_bias_word("JIT") == "jit"
    monkeypatch.setenv(TA.ENV_TRIATT_BIAS, "jit")
    assert TA.triatt_bias_word() == "jit"
    monkeypatch.setenv(TA.ENV_TRIATT_BIAS, "ring")
    with pytest.raises(RowpairRefused):
        TA.triatt_bias_word()
    monkeypatch.delenv(TA.ENV_TRIATT_JIT_QBLOCK, raising=False)
    assert TA.jit_qblock() == TA.JIT_QBLOCK_DEFAULT == 2048 and TA.jit_qblock(64) == 64
    monkeypatch.setenv(TA.ENV_TRIATT_JIT_QBLOCK, "100")
    with pytest.raises(RowpairRefused):
        TA.jit_qblock()
    with pytest.raises(RowpairRefused):
        TA.jit_qblock(24)
    monkeypatch.delenv(TA.ENV_TRIATT_JIT_GROUP, raising=False)
    assert TA.jit_group() == TA.JIT_GROUP_DEFAULT == 4 and TA.jit_group(2) == 2
    monkeypatch.setenv(TA.ENV_TRIATT_JIT_GROUP, "0")
    with pytest.raises(RowpairRefused):
        TA.jit_group()
    monkeypatch.delenv(TA.ENV_TRIATT_JIT_PREFETCH, raising=False)
    assert TA.jit_prefetch() is True and TA.jit_prefetch(False) is False
    monkeypatch.setenv(TA.ENV_TRIATT_JIT_PREFETCH, "0")
    assert TA.jit_prefetch() is False
    g = torch.Generator().manual_seed(1)
    m = TriAttJ(8, 2, 4, g)
    assert TA.jit_fields_missing(m.fns()) == [] and TA.jit_fields_missing(m.fns(fields=False)) == list(TA.JIT_FIELDS)
    f3 = TA.TriAttFns(m.ln, m.b, m.attend)                       # today's positional construction: unchanged, optional fields None / False
    assert (f3.proj, f3.mask_bias, f3.wrap, f3.core, f3.bias_transposed) == (None, None, None, None, False)


def _entry_bind(rank, P, N, B, mod_s, mod_e, y, mask, fields):
    torch.set_grad_enabled(False)                               # grad mode is thread-local: the rank threads run inference
    lay = Layout(N, P, rank, B)
    ys = y[lay.r0:lay.r1].clone().contiguous()
    ms = mask[lay.r0:lay.r1].contiguous()
    EV.reset_schedule()
    b = PS.bind(trimul_out=None, trimul_in=None, triatt_start=mod_s.fns(fields), triatt_end=mod_e.fns(fields), transition=None, chunk=16, tb_rows=8,
                trimul_kw=dict(inplace_chunk=8))
    out = b.triatt_start(ys, ms, lay)
    return out, EV.schedule()


def test_bind_routes_by_word_and_fields(monkeypatch):
    P, N, B = 2, 80, 16
    g = torch.Generator().manual_seed(3)
    ms_, me_ = TriAttJ(8, 2, 4, g), TriAttJ(8, 2, 4, g, transposed=True)
    y, mask = _inputs(N, 8, 13)
    seen = {"jit": 0, "gather": 0}
    real_jit, real_gather = TA.triatt_update_jit_, TA.triatt_update_

    def spy_jit(*a, **k):
        seen["jit"] += 1
        return real_jit(*a, **k)

    def spy_gather(*a, **k):
        seen["gather"] += 1
        return real_gather(*a, **k)

    monkeypatch.setattr(TA, "triatt_update_jit_", spy_jit)
    monkeypatch.setattr(TA, "triatt_update_", spy_gather)
    with torch.no_grad():
        monkeypatch.delenv(TA.ENV_TRIATT_BIAS, raising=False)               # default: gather, nothing recorded
        res = run_ranks(P, _entry_bind, N, B, ms_, me_, y, mask, True)
        assert seen == {"jit": 0, "gather": P} and all("triatt_bias" not in r[1] for r in res), (seen, res[0][1])
        ref = torch.cat([r[0] for r in res], dim=0)
        monkeypatch.setenv(TA.ENV_TRIATT_BIAS, "jit")                       # jit + fields: the jit schedule, bitwise to the default
        monkeypatch.setenv(TA.ENV_TRIATT_JIT_QBLOCK, "32")
        res = run_ranks(P, _entry_bind, N, B, ms_, me_, y, mask, True)
        assert seen["jit"] == P and all(r[1].get("triatt_bias") == "jit:qb32:3" for r in res), (seen, res[0][1])
        assert torch.equal(torch.cat([r[0] for r in res], dim=0), ref)
        res = run_ranks(P, _entry_bind, N, B, ms_, me_, y, mask, False)     # jit without the fields: gather BY NAME
        assert seen["jit"] == P and seen["gather"] == 2 * P
        assert all(r[1].get("triatt_bias") == "gather:fields_missing:core+mask_bias+proj+wrap" for r in res), res[0][1]
        assert torch.equal(torch.cat([r[0] for r in res], dim=0), ref)
        monkeypatch.setenv(TA.ENV_TRIATT_BIAS, "sideways")                  # a malformed word: refused by name at bind
        with pytest.raises(RuntimeError, match="ROWPAIR_TRIATT_BIAS"):
            run_ranks(P, _entry_bind, N, B, ms_, me_, y, mask, True)
    _report("bind_routes", seen=seen)


def _entry_refusals(rank, P, N, mod, y, mask, case):
    torch.set_grad_enabled(False)                               # grad mode is thread-local: the rank threads run inference
    lay = Layout(N, P, rank, 16)
    ys = y[lay.r0:lay.r1].clone().contiguous()
    ms = mask[lay.r0:lay.r1].contiguous()
    if case == "fields":
        fns = mod.fns(fields=False)
        kw = dict(rows=16, qblock=32)
    elif case == "disagree":
        fns = mod.fns()
        kw = dict(rows=16, qblock=32 if rank == 0 else 48)
    else:
        fns = mod.fns()
        kw = dict(rows=16, qblock=32)
    try:
        TA.triatt_update_jit_(fns, ys, ms, lay, **kw)
    except RowpairRefused as e:
        return str(e)
    return "served"


def test_refusals_by_name():
    g = torch.Generator().manual_seed(9)
    mod = TriAttJ(8, 2, 4, g)
    y, mask = _inputs(48, 8, 17)
    with torch.no_grad():
        r1 = run_ranks(1, _entry_refusals, 48, mod, y, mask, "p1")[0]
        assert "n_gpu=1" in r1, r1
        rf = run_ranks(2, _entry_refusals, 48, mod, y, mask, "fields")
        assert all("lacks" in r and "proj" in r for r in rf), rf
        rd = run_ranks(2, _entry_refusals, 48, mod, y, mask, "disagree")
        assert all("disagree" in r for r in rd), rd                          # refused on EVERY rank (no rank left in a collective)
        ok = run_ranks(2, _entry_refusals, 48, mod, y, mask, "ok")
        assert ok == ["served", "served"], ok
    _report("refusals", p1=r1[:60], fields=rf[0][:60], disagree=rd[0][:60])


def test_stage_once_is_ignored_under_jit_by_name(monkeypatch):
    """ROWPAIR_TRIATT_STAGE=once (stage-once of the GATHERED plane) does not apply to the exchange's refilled block buffers: the jit loop runs
    under a per_call span by name (census ``triatt_stage=per_call:jit``), results unchanged (bitwise to gather)."""
    P, N, B = 2, 80, 16
    g = torch.Generator().manual_seed(21)
    mod = TriAttJ(8, 2, 4, g, door=True)
    y, mask = _inputs(N, 8, 23)
    with torch.no_grad():
        monkeypatch.delenv(TA.ENV_TRIATT_STAGE, raising=False)
        ref = run_ranks(P, _entry_update, N, B, None, mod, y, mask, "gather", dict(rows=16, stream=True, tb_rows=8))
        monkeypatch.setenv(TA.ENV_TRIATT_STAGE, "once")
        res = run_ranks(P, _entry_update, N, B, None, mod, y, mask, "jit", dict(rows=16, stream=True, tb_rows=8, qblock=32, group=2))
    assert torch.equal(torch.cat([r[0] for r in res], 0), torch.cat([r[0] for r in ref], 0))
    assert all(r[1].get("triatt_stage") == "per_call:jit" for r in res), res[0][1]
    monkeypatch.setenv(TA.ENV_TRIATT_STAGE, "sideways")                      # a malformed stage word is refused by name under jit too (same resolver)
    with torch.no_grad():
        out = run_ranks(P, _entry_refusals, N, mod, y, mask, "ok")
    assert all("ROWPAIR_TRIATT_STAGE" in r for r in out), out
    _report("stage_once_ignored", census=res[0][1].get("triatt_stage"))


# - the tile's kernel operand prepared once
def test_jit_prep_words_and_size_law(monkeypatch):
    monkeypatch.delenv(TA.ENV_TRIATT_JIT_PREP, raising=False)
    assert TA.jit_prep_word() == TA.JIT_PREP_DEFAULT == "tile" and TA.jit_prep_word("LAUNCH") == "launch"
    monkeypatch.setenv(TA.ENV_TRIATT_JIT_PREP, "launch")
    assert TA.jit_prep_word() == "launch"
    monkeypatch.setenv(TA.ENV_TRIATT_JIT_PREP, "plane")
    with pytest.raises(RowpairRefused, match="ROWPAIR_TRIATT_JIT_PREP"):
        TA.jit_prep_word()
    H, S, QB = 4, 70320, 2048                                    # ONE tile [1, 1, H, QB, S]: fp32 [.., QB, ceil16(S)] + bf16 copy + flags + a fp32 source copy (transient)
    assert TA.stage_need_bytes("flash_tile", H, S, qblock=QB) == 6 * H * QB * S + 4 * H * (QB // 32) + 4 * H * QB * S     # S % 16 == 0 here
    assert TA.stage_need_bytes("flash_tile", H, 100, qblock=QB) == 6 * H * 100 * 112 + 4 * H * 4 + 4 * H * 100 * 100       # qblock clipped to S; ceil16(100) = 112
    assert TA.stage_need_bytes("flash_tile", H, S, qblock=QB) < TA.stage_need_bytes("flash_qblocks", H, S, qblock=QB) // 20  # one tile (5.4 GiB at 70,320), never the plane (113 GiB)


@pytest.mark.parametrize("prep", ["tile", "launch"])
def test_jit_prep_census_and_result_on_the_hub(monkeypatch, prep):
    """Under the CPU cores of this suite no kernel operand exists (the flash tile kind is CUDA-only: the GPU unit asserts tile == launch bitwise per
    window); here: the lever word routes and is recorded (tile: triatt_jit_prep=tile + triatt_jit_prep_tiles=0 on CPU; launch: triatt_jit_prep=launch),
    the schedule's result is unchanged (bitwise to gather) and a malformed word is refused by name on every rank."""
    P, N, B = 2, 87, 16
    g = torch.Generator().manual_seed(31)
    mod = TriAttJ(8, 2, 4, g, door=True)
    y, mask = _inputs(N, 8, 33)
    with torch.no_grad():
        ref = run_ranks(P, _entry_update, N, B, None, mod, y, mask, "gather", dict(rows=16, stream=True, tb_rows=8))
        monkeypatch.setenv(TA.ENV_TRIATT_JIT_PREP, prep)
        res = run_ranks(P, _entry_update, N, B, None, mod, y, mask, "jit", dict(rows=16, stream=True, tb_rows=8, qblock=32, group=2))
    assert torch.equal(torch.cat([r[0] for r in res], 0), torch.cat([r[0] for r in ref], 0))
    for _, sched, _ in res:
        assert sched.get("triatt_jit_prep") == prep, sched
        assert sched.get("triatt_stage") == "per_call", sched
        if prep == "tile":
            assert sched.get("triatt_jit_prep_tiles") == 0 and sched.get("triatt_jit_prep_served") == 0, sched
        else:
            assert "triatt_jit_prep_tiles" not in sched, sched
    monkeypatch.setenv(TA.ENV_TRIATT_JIT_PREP, "sideways")
    with torch.no_grad():
        out = run_ranks(P, _entry_refusals, N, mod, y, mask, "ok")
    assert all("ROWPAIR_TRIATT_JIT_PREP" in r for r in out), out
    _report("jit_prep_census", prep=prep, census={k: v for k, v in res[0][1].items() if "prep" in k or k == "triatt_stage"})


def test_jit_prep_tile_no_room_steps_aside_by_name(monkeypatch):
    """The room check of ONE tile (stage_need_bytes('flash_tile') + stage_room_margin) against a (fake) free reading: without room _room_or_aside raises
    _NoRoom, records need / free GiB under the jit words, the slot remembers the tile's key (its other windows do not re-measure) and the aside is
    census'd triatt_jit_prep=launch:no_room; a release (the next tile) clears the memory so the next tile asks again; with room nothing is recorded."""
    EV.reset_schedule()
    slot = TA.StageSlot()
    tb = torch.zeros(3, 40, 2).permute(2, 0, 1).unsqueeze(0).unsqueeze(0)          # a [1, 1, H=2, QB=3, S=40] tile view
    dev = torch.device("cpu")
    with slot.tiles():
        assert slot.tile() and not slot.once()
        assert EV.schedule().get("triatt_jit_prep") == "tile" and EV.schedule().get("triatt_stage") == "per_call"
        sb = slot.get("flash_tile", tb, 3, 2)
        need = TA.stage_need_bytes("flash_tile", 2, 40, qblock=3)
        monkeypatch.setattr(TA, "_free_device_bytes", lambda device: need + TA.stage_room_margin(need) - 1)      # one byte short
        with pytest.raises(TA._NoRoom):
            TA._room_or_aside(slot, sb, "flash_tile", 2, 40, dev, qblock=3)
        assert slot.no_room == sb.key
        sch = EV.schedule()
        assert "triatt_jit_prep_need_gib" in sch and "triatt_jit_prep_free_gib" in sch and "triatt_stage_need_gib" not in sch, sch
        slot.aside("no_room")
        assert EV.schedule().get("triatt_jit_prep") == "launch:no_room", EV.schedule()
        slot.release()                                                               # the next tile: measured again
        assert slot.no_room is None and slot.staged is None
        monkeypatch.setattr(TA, "_free_device_bytes", lambda device: need + TA.stage_room_margin(need))              # exactly enough
        sb2 = slot.get("flash_tile", tb, 3, 2)
        TA._room_or_aside(slot, sb2, "flash_tile", 2, 40, dev, qblock=3)             # no raise
        monkeypatch.setattr(TA, "_free_device_bytes", lambda device: None)           # no reading (CPU): no check
        TA._room_or_aside(slot, sb2, "flash_tile", 2, 40, dev, qblock=3)
        monkeypatch.setenv(TA.ENV_TRIATT_STAGE_ROOM, "0")                            # the engineering switch skips the check
        monkeypatch.setattr(TA, "_free_device_bytes", lambda device: 0)
        TA._room_or_aside(slot, sb2, "flash_tile", 2, 40, dev, qblock=3)
    sch = EV.schedule()
    assert sch.get("triatt_jit_prep") == "launch:no_room" and sch.get("triatt_jit_prep_tiles") == 0, sch   # all tiles of the span stepped aside: the aside word stands
    assert not slot.tile() and not slot.armed
    with slot.plane("once"):                                                          # a plane span after a tiles span: the stage-once words, tile off
        assert slot.once() and not slot.tile()
    _report("jit_prep_no_room", census={k: v for k, v in sch.items() if "prep" in k})


if __name__ == "__main__":
    rc = pytest.main([__file__, "-q", "-rfE", "-x"])
    print("SUMMARY %d RESULT lines, rc=%s" % (len(RESULTS), rc))
    sys.exit(int(rc))
