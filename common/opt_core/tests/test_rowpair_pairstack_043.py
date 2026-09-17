"""opt_core.mem.rowpair 0.4.3 — the streamed triangle-multiplication schedule (``trimul.trimul_update_``), the triangle-attention update
(``triatt.triatt_update_`` + query blocks) and the pair-block / pair-stack driver (``pairstack``) on row shards versus DENSE synthetic AF3-family
modules (random weights): P in {2, 3, 4} rank-threads (``opt_core.testing.run_ranks``), uneven N, fp32 (max|diff| <= 1e-5, torch.equal
REPORTED) and fp64 (<= 1e-9); P = 1 refuses by name (the structural n_gpu=1 rule).

Run: ``python -m pytest tests/test_rowpair_pairstack_043.py -q -rfE`` or ``python tests/test_rowpair_pairstack_043.py`` (RESULT lines + SUMMARY).
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
from opt_core.mem.rowpair import dist as D  # noqa: E402
from opt_core.mem.rowpair import pairstack as PS  # noqa: E402
from opt_core.mem.rowpair import triatt as TA  # noqa: E402
from opt_core.mem.rowpair import trimul as TM  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402

INF = 1e9
RESULTS = []


def _report(name, **kw):
    line = "RESULT " + json.dumps(dict(case=name, **kw), sort_keys=True, default=str)
    RESULTS.append(line)
    print(line)


def _cmp(got, ref):
    return {"bitwise": bool(torch.equal(got, ref)), "maxabs": float((got.double() - ref.double()).abs().max().item())}


def _tol(dtype):
    return 1e-9 if dtype == torch.float64 else 1e-5


# ================================================================================================ synthetic AF3-family modules (dense statements)
class TriMul(torch.nn.Module):
    def __init__(self, C, C_h, g):
        super().__init__()
        L = lambda i, o: torch.nn.Linear(i, o, bias=True)
        self.ln_in, self.ln_out = torch.nn.LayerNorm(C), torch.nn.LayerNorm(C_h)
        self.a_p, self.a_g, self.b_p, self.b_g = L(C, C_h), L(C, C_h), L(C, C_h), L(C, C_h)
        self.g, self.z = L(C, C), L(C_h, C)
        self.C_h = C_h
        for p in self.parameters():
            torch.nn.init.normal_(p, std=0.3, generator=g)

    def proj(self, zblk, mblk, is_a):
        zl = self.ln_in(zblk)
        p, gg = (self.a_p, self.a_g) if is_a else (self.b_p, self.b_g)
        return torch.sigmoid(gg(zl)) * p(zl) * mblk

    def out(self, x):
        return self.z(self.ln_out(x))

    def gate(self, zblk):
        return torch.sigmoid(self.g(self.ln_in(zblk)))

    def dense(self, z, mask, outgoing):                      # z + update (the in-place inference statement's values)
        m = mask[..., None]
        a, b = self.proj(z, m, True), self.proj(z, m, False)
        x = torch.einsum("ikc,jkc->ijc", a, b) if outgoing else torch.einsum("kic,kjc->ijc", a, b)
        return z + self.out(x) * self.gate(z)

    def fns(self):
        return TM.TriMulFns(self.proj, self.out, self.gate, self.C_h)


def _attention_core(q, k, v, biases):                          # q [.., H, Q, d]; softmax over keys; every bias broadcastable to [.., H, Q, K]
    logits = torch.matmul(q, k.transpose(-1, -2))
    for b in biases:
        logits = logits + b
    return torch.matmul(torch.softmax(logits, dim=-1), v)


class TriAtt(torch.nn.Module):
    def __init__(self, C, H, d, g):
        super().__init__()
        L = lambda i, o, b=False: torch.nn.Linear(i, o, bias=b)
        self.ln, self.b = torch.nn.LayerNorm(C), L(C, H)
        self.q, self.k, self.v, self.gt, self.o = L(C, H * d), L(C, H * d), L(C, H * d), L(C, H * d, True), L(H * d, C, True)
        self.H, self.d = H, d
        self.qblock = None
        for p in self.parameters():
            torch.nn.init.normal_(p, std=0.3, generator=g)

    def _heads(self, t):                                       # [I, J, H*d] -> [I, H, J, d]
        return t.unflatten(-1, (self.H, self.d)).transpose(-2, -3)

    def attend(self, x, mask_rows, tb_full, blk=None):         # x = LN'd rows [I, J, C]; tb_full [N, N, H]
        mask_bias = (INF * (mask_rows - 1.0))[..., :, None, None, :]                    # [I, 1, 1, J]
        tb = tb_full.permute(2, 0, 1).unsqueeze(-4)                                    # [1, H, Jq, Jk]
        q, k, v = self._heads(self.q(x)) / math.sqrt(self.d), self._heads(self.k(x)), self._heads(self.v(x))
        o = TA.attend_query_blocks(_attention_core, q, k, v, [mask_bias, tb], self.qblock)   # [I, H, J, d]
        o = o.transpose(-2, -3).flatten(-2)                                            # [I, J, H*d]
        return self.o(o * torch.sigmoid(self.gt(x)))

    def dense(self, z, mask):                                  # z + starting attention(z)
        x = self.ln(z)
        return z + self.attend(x, mask, self.b(x))

    def fns(self):
        return TA.TriAttFns(self.ln, self.b, self.attend)


class Transition(torch.nn.Module):
    def __init__(self, C, g, n=2):
        super().__init__()
        self.ln, self.a, self.bb, self.o = torch.nn.LayerNorm(C), torch.nn.Linear(C, n * C), torch.nn.Linear(C, n * C), torch.nn.Linear(n * C, C)
        for p in self.parameters():
            torch.nn.init.normal_(p, std=0.3, generator=g)

    def delta(self, x, mask_u):
        xl = self.ln(x)
        return self.o(torch.nn.functional.silu(self.a(xl)) * self.bb(xl)) * mask_u

    def dense(self, z, mask):
        return z + self.delta(z, mask[..., None])


class APB(torch.nn.Module):                                    # attention-pair-bias of the single track + its transition
    def __init__(self, C, c_s, H, d, g):
        super().__init__()
        L = lambda i, o, b=False: torch.nn.Linear(i, o, bias=b)
        self.ln_s, self.ln_z, self.pb = torch.nn.LayerNorm(c_s), torch.nn.LayerNorm(C), L(C, H)
        self.q, self.k, self.v, self.gt, self.o = L(c_s, H * d, True), L(c_s, H * d), L(c_s, H * d), L(c_s, H * d, True), L(H * d, c_s, True)
        self.st = Transition(c_s, g)
        self.H, self.d = H, d
        for p in list(self.parameters()):
            if p.dim() > 0 and p not in set(self.st.parameters()):
                torch.nn.init.normal_(p, std=0.3, generator=g)

    def _heads(self, t):                                       # [N, H*d] -> [H, N, d]
        return t.unflatten(-1, (self.H, self.d)).transpose(0, 1)

    def rows(self, s, z_rows, i0, i1):                          # attention outputs o[i, H*d] for query rows i0:i1 (GLOBAL) from pair rows z[i0:i1]
        a = self.ln_s(s)                                                                 # projections on the FULL replicated s (M = N)
        q, k, v = self._heads(self.q(a)) / math.sqrt(self.d), self._heads(self.k(a)), self._heads(self.v(a))
        b = self.pb(self.ln_z(z_rows)).permute(2, 0, 1)                                # [H, rows, N]
        o = _attention_core(q[:, i0:i1], k, v, [b])                                      # [H, rows, d]
        return o.transpose(0, 1).flatten(-2)                                             # [rows, H*d]

    def finish(self, s, o_full):
        a = self.ln_s(s)
        return s + self.o(o_full * torch.sigmoid(self.gt(a)))

    def dense(self, s, z):
        N = s.shape[0]
        s = self.finish(s, self.rows(s, z, 0, N))
        return s + self.st.delta(s, 1.0)

    def apb_sharded(self, s, z_shard, layout):                 # local query rows read local pair rows; o rows all-gathered; wrap-up on the full s
        o_loc = self.rows(s, z_shard, layout.r0, layout.r1)
        return self.finish(s, D.all_gather_rows(o_loc.contiguous(), layout))

    def single_transition(self, s):
        return s + self.st.delta(s, 1.0)


class Block(object):
    """One Pairformer block of synthetic modules (dense forward + its PairBlockFns)."""

    def __init__(self, C, C_h, H, d, c_s, g, with_single=True):
        self.tmo, self.tmi = TriMul(C, C_h, g), TriMul(C, C_h, g)
        self.tas, self.tae = TriAtt(C, H, d, g), TriAtt(C, H, d, g)
        self.tr = Transition(C, g)
        self.apb = APB(C, c_s, H, d, g) if with_single else None

    def to(self, dtype):
        for m in (self.tmo, self.tmi, self.tas, self.tae, self.tr, self.apb):
            if m is not None:
                m.to(dtype)
        return self

    def dense(self, z, mask, s=None):
        z = self.tmo.dense(z, mask, True)
        z = self.tmi.dense(z, mask, False)
        z = self.tas.dense(z, mask)
        zT = z.transpose(0, 1).contiguous()
        zT = self.tae.dense(zT, mask.transpose(0, 1).contiguous())
        z = zT.transpose(0, 1).contiguous()
        z = self.tr.dense(z, mask)
        if self.apb is not None:
            s = self.apb.dense(s, z)
        return z, s

    def fns(self, chunk, **kw):
        kw.setdefault("trimul_kw", {}).setdefault("inplace_chunk", 256)          # the synthetic engine's in-place tri-mult column chunk
        return PS.bind(trimul_out=self.tmo.fns(), trimul_in=self.tmi.fns(), triatt_start=self.tas.fns(), triatt_end=self.tae.fns(),
                       transition=self.tr.delta, chunk=chunk, apb=(self.apb.apb_sharded if self.apb else None),
                       single_transition=(self.apb.single_transition if self.apb else None), **kw)


def _inputs(N, C, c_s, seed, dtype):
    g = torch.Generator().manual_seed(seed)
    z = torch.randn((N, N, C), generator=g, dtype=torch.float64).to(dtype)
    s = torch.randn((N, c_s), generator=g, dtype=torch.float64).to(dtype)
    mask = (torch.rand((N, N), generator=g) > 0.15).to(dtype)
    return z, s, mask


# ================================================================================================ pure checks (no ranks)
def test_slab_grid_partitions_and_uniform_grid_is_042s():
    for N, P, B in ((200, 4, 16), (104, 3, 16), (300, 2, 128), (1000, 3, 128)):
        lays = [Layout(N, P, q, B) for q in range(P)]
        for RB in (7, 24, 64, 256):
            for cb in (None, TM.inplace_chunk_bounds(N, 48), TM.inplace_chunk_bounds(N, 256)):
                grid = TM.slab_grid(lays[0], RB, cb)
                assert all(TM.slab_grid(l, RB, cb) == grid for l in lays)                 # identical on every rank
                for q, g in enumerate(grid):
                    R = lays[q].nrows(q) if hasattr(lays[q], "nrows") else lays[0].nrows(q)
                    flat = [i for (j0, j1) in g for i in range(j0, j1)]
                    assert flat == list(range(lays[0].nrows(q))), (N, P, B, RB, q, g)
                    assert all(0 < j1 - j0 <= RB for j0, j1 in g)
                    if cb:
                        q0 = lays[0].bounds[q][0]
                        for j0, j1 in g:
                            assert not any(q0 + j0 < b < q0 + j1 for b in cb), (g, cb)
                if cb is None:                                                             # 0.4.2's uniform slabs
                    for q, g in enumerate(grid):
                        Rq = lays[0].nrows(q)
                        assert g == [(t * RB, min(Rq, (t + 1) * RB)) for t in range(-(-Rq // RB))]
                    assert max(len(g) for g in grid) == -(-lays[0].Rmax // RB)
                # deferral peak: zero iff every rank has one own sub-block; bounded by the A operand
                pk = [TM.deferred_peak_bytes(lays[q], grid, 8, 4, rank=q) for q in range(P)]
                assert all(p >= 0 for p in pk)
                assert all((p == 0) == (len(grid[q]) <= 1) for q, p in enumerate(pk)), (pk, grid)
    assert TM.inplace_chunk_bounds(1000, 256) == [0, 256, 500, 756, 1000]
    assert TM.inplace_chunk_bounds(512, 256) == [0, 256, 512]
    assert TM.grid_bounds(1000, "rank")[0] is None and TM.grid_bounds(1000, "stock", 256)[0] == [0, 256, 500, 756, 1000]
    with pytest.raises(RowpairRefused):
        TM.grid_bounds(10, "diagonal")


def test_block_policies():
    assert TA.blocks4(10, None) == [(0, 10)] and TA.blocks4(10, 12) == [(0, 10)]
    assert TA.blocks4(10, 4) == [(0, 4), (4, 10)]                                          # tail merged into the previous block
    assert TA.blocks4(16, 4) == [(0, 4), (4, 8), (8, 12), (12, 16)]
    with pytest.raises(RowpairRefused):
        TA.blocks4(10, 6)
    assert TM.mm_row_pieces(1000, 64, 8, None) == [(0, 1000)]
    assert TM.mm_row_pieces(1000, 64, 8, "auto") == [(0, 1000)] and TM.mm_row_pieces(1000, 64, 8, 0) == [(0, 1000)]
    assert TM.mm_row_pieces(10, 64, 8, 4) == [(0, 4), (4, 8), (8, 10)]
    big = TM.mm_row_pieces(70000, 40000, 128, "auto")                                     # 128*70000*40000 > 2^31 -> pieces of 256-multiples
    assert len(big) > 1 and all((p1 - p0) % 256 == 0 for p0, p1 in big[:-1]) and 128 * (big[0][1] - big[0][0]) * 40000 <= TM.INT32_MAX
    os.environ["ROWPAIR_TRIATT_QBLOCK"] = "8"
    assert TA.lever_rows("TRIATT_QBLOCK") == 8
    os.environ["ROWPAIR_TRIATT_QBLOCK"] = "6"
    with pytest.raises(RowpairRefused):
        TA.lever_rows("TRIATT_QBLOCK")
    del os.environ["ROWPAIR_TRIATT_QBLOCK"]
    assert TA.lever_rows("TRIATT_QBLOCK") is None


def test_query_blocks_equal_one_call():
    g = torch.Generator().manual_seed(3)
    q, k, v = (torch.randn((3, 2, 20, 5), generator=g) for _ in range(3))
    b1 = torch.randn((3, 1, 1, 20), generator=g)                                          # broadcast over queries
    b2 = torch.randn((1, 2, 20, 20), generator=g)                                          # per query
    one = TA.attend_query_blocks(_attention_core, q, k, v, [b1, b2], None)
    blk = TA.attend_query_blocks(_attention_core, q, k, v, [b1, b2], 8)
    m = _cmp(blk, one)
    _report("query_blocks", **m)
    assert m["maxabs"] <= 1e-6


# ================================================================================================ rank entries (threaded comm)
def _entry_p1_refusals(rank, P, N):
    lay = Layout(N, P, rank, 16)
    g = torch.Generator().manual_seed(0)
    blk = Block(8, 8, 2, 4, 12, g)
    z, s, mask = _inputs(N, 8, 12, 1, torch.float32)
    names = {}
    for what, call in (("trimul_update_", lambda: TM.trimul_update_(blk.tmo.fns(), z.clone(), mask, lay, outgoing=True, inplace_chunk=256)),
                       ("triatt_update_", lambda: TA.triatt_update_(blk.tas.fns(), z.clone(), mask, lay, rows=16)),
                       ("pair_block_", lambda: PS.pair_block_(blk.fns(16), z.clone(), mask, lay, s=s)),
                       ("pair_stack_", lambda: PS.pair_stack_([blk.fns(16)], z.clone(), mask, lay, s=s))):
        try:
            call()
            names[what] = "RAN"
        except RowpairRefused as e:
            names[what] = str(e)
    return names


def test_p1_refuses_by_name():
    names = run_ranks(1, _entry_p1_refusals, 64)[0]
    _report("p1_refusals", **names)
    for what, msg in names.items():
        assert msg != "RAN" and "n_gpu=1" in msg and what in msg, (what, msg)


def _entry_trimul(rank, P, N, B, mod, z, mask, outgoing, kw):
    lay = Layout(N, P, rank, B)
    zs = z[lay.r0:lay.r1].clone().contiguous()
    st = TM.ContractStats()
    kw = dict({"inplace_chunk": 256}, **kw)
    out = TM.trimul_update_(mod.fns(), zs, mask[lay.r0:lay.r1].contiguous(), lay, outgoing=outgoing, stats=st, **kw)
    assert out is zs
    return zs, st.facts(), dict(st.__dict__)


@pytest.mark.parametrize("P,N,B", [(2, 80, 16), (3, 104, 16), (4, 200, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mp_trimul_update_equals_dense(P, N, B, dtype):
    g = torch.Generator().manual_seed(11)
    C, C_h = 8, 6
    mod = TriMul(C, C_h, g).to(dtype)
    z, _, mask = _inputs(N, C, 12, 5, dtype)
    with torch.no_grad():
        for outgoing in (True, False):
            dense = mod.dense(z, mask, outgoing)
            for tag, kw in (("stock_grid", dict(RB=24, grid="stock", inplace_chunk=48)),
                            ("rank_grid_mmrows", dict(RB=40, grid="rank", mm_rows=16)),
                            ("one_subblock_per_rank", dict(RB=4096, grid="rank"))):
                res = run_ranks(P, _entry_trimul, N, B, mod, z, mask, outgoing, kw)
                got = torch.cat([r[0] for r in res], dim=0)
                m = _cmp(got, dense)
                facts = [r[1] for r in res]
                _report(f"trimul_{'out' if outgoing else 'in'}_{tag}", P=P, N=N, dtype=str(dtype), facts_rank0=facts[0], **m)
                assert m["maxabs"] <= _tol(dtype), (tag, outgoing, m)
                # schedule facts: one pass, every rank the same slab / ring-step counts, deferral only outgoing
                assert all(f["passes"] == 1 for f in facts)
                assert len({(f["slabs"], f["ring_steps"]) for f in facts}) == 1, facts
                assert all(f["ring_steps"] == f["slabs"] * P for f in facts)
                if outgoing and tag != "one_subblock_per_rank":
                    assert any(r[2]["deferred_tiles"] > 0 for r in res), [r[2]["deferred_tiles"] for r in res]
                if not outgoing or tag == "one_subblock_per_rank":
                    assert all(r[2]["deferred_tiles"] == 0 for r in res)
                assert all(f["grid"] == ("bounds" if kw["grid"] == "stock" else "rank") for f in facts)


def _entry_triatt(rank, P, N, B, mod, z, mask, kw, qblock):
    lay = Layout(N, P, rank, B)
    mod.qblock = qblock                                   # read inside attend (shared module object; every rank sets the same value)
    zs = z[lay.r0:lay.r1].clone().contiguous()
    out = TA.triatt_update_(mod.fns(), zs, mask[lay.r0:lay.r1].contiguous(), lay, **kw)
    assert out is zs
    return zs


@pytest.mark.parametrize("P,N,B", [(2, 80, 16), (3, 104, 16), (4, 200, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mp_triatt_update_equals_dense(P, N, B, dtype):
    g = torch.Generator().manual_seed(12)
    mod = TriAtt(8, 2, 4, g).to(dtype)
    z, _, mask = _inputs(N, 8, 12, 6, dtype)
    with torch.no_grad():
        mod.qblock = None
        dense = mod.dense(z, mask)
        outs = {}
        for tag, kw, qb in (("stream", dict(rows=16, stream=True), None), ("stream_qblock8_tbrows8", dict(rows=16, stream=True, tb_rows=8), 8),
                            ("ln_once", dict(rows=16, stream=False), None), ("ln_once_qblock8", dict(rows=32, stream=False, tb_rows=16), 8)):
            res = run_ranks(P, _entry_triatt, N, B, mod, z, mask, kw, qb)
            got = torch.cat(res, dim=0)
            m = _cmp(got, dense)
            outs[tag] = got
            _report(f"triatt_start_{tag}", P=P, N=N, dtype=str(dtype), **m)
            assert m["maxabs"] <= _tol(dtype), (tag, m)
        _report("triatt_stream_vs_ln_once", P=P, dtype=str(dtype), **_cmp(outs["stream"], outs["ln_once"]))
        _report("triatt_qblock_on_vs_off", P=P, dtype=str(dtype), **_cmp(outs["stream_qblock8_tbrows8"], outs["stream"]))
        mod.qblock = None


def _entry_transposes(rank, P, N, B, z, mask):
    from opt_core.mem.rowpair.ring import transpose_shard, transpose_shard_inplace_
    lay = Layout(N, P, rank, B)
    zs = z[lay.r0:lay.r1].clone().contiguous()
    zT = transpose_shard(zs, lay)
    back = transpose_shard(zT, lay)
    zi = zs.clone()
    transpose_shard_inplace_(zi, lay)
    mT = PS.mask_transposed(mask[lay.r0:lay.r1].contiguous(), lay)
    return {"roundtrip_bitwise": bool(torch.equal(back, zs)), "zT_bitwise": bool(torch.equal(zT, z.transpose(0, 1)[lay.r0:lay.r1])),
            "inplace_bitwise": bool(torch.equal(zi, zT)), "maskT_bitwise": bool(torch.equal(mT, mask.transpose(0, 1)[lay.r0:lay.r1]))}


@pytest.mark.parametrize("P,N,B", [(2, 80, 16), (3, 104, 16), (4, 200, 16)])
def test_mp_transposes_bitwise(P, N, B):
    z, _, mask = _inputs(N, 8, 12, 7, torch.float32)
    res = run_ranks(P, _entry_transposes, N, B, z, mask)
    _report("transposes", P=P, N=N, ranks=res)
    assert all(all(r.values()) for r in res), res


def _entry_block(rank, P, N, B, blk, z, s, mask, chunk, block_kw, bind_kw):
    lay = Layout(N, P, rank, B)
    zs = z[lay.r0:lay.r1].clone().contiguous()
    stats = {}
    z_out, s_out = PS.pair_block_(blk.fns(chunk, stats=stats, **bind_kw), zs, mask[lay.r0:lay.r1].contiguous(), lay, s=s.clone(), **block_kw)
    facts = {k: v.facts() for k, v in stats.items()}
    return z_out, s_out, bool(z_out is zs), facts


@pytest.mark.parametrize("P,N,B", [(2, 80, 16), (3, 104, 16), (4, 200, 16)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mp_pair_block_driver_equals_dense_block(P, N, B, dtype):
    g = torch.Generator().manual_seed(13)
    blk = Block(8, 8, 2, 4, 12, g).to(dtype)
    z, s, mask = _inputs(N, 8, 12, 8, dtype)
    with torch.no_grad():
        zd, sd = blk.dense(z, mask, s)
        variants = (("default_end_free", dict(), dict(trimul_kw=dict(RB=24, inplace_chunk=48))),
                    ("transpose_inplace", dict(transpose_inplace=True), dict(trimul_kw=dict(RB=24, grid="rank"))),
                    ("fresh_output_ln_once", dict(reuse_storage=False, end_free=False), dict(stream=False, trimul_kw=dict(RB=4096))))
        for tag, block_kw, bind_kw in variants:
            res = run_ranks(P, _entry_block, N, B, blk, z, s, mask, 16, block_kw, bind_kw)
            zg = torch.cat([r[0] for r in res], dim=0)
            mz, ms = _cmp(zg, zd), _cmp(res[0][1], sd)
            s_replicated = all(torch.equal(r[1], res[0][1]) for r in res)
            _report(f"pair_block_{tag}", P=P, N=N, dtype=str(dtype), z=mz, s=ms, s_replicated=s_replicated, same_object=[r[2] for r in res],
                    facts_rank0=res[0][3])
            assert mz["maxabs"] <= _tol(dtype) and ms["maxabs"] <= _tol(dtype), (tag, mz, ms)
            assert s_replicated
            assert all(r[2] == (block_kw.get("reuse_storage", True) and not block_kw.get("transpose_inplace", False) or block_kw.get("transpose_inplace", False)) for r in res)


def _entry_stack(rank, P, N, B, blocks, z, s, mask, chunk):
    lay = Layout(N, P, rank, B)
    zs = z[lay.r0:lay.r1].clone().contiguous()
    seen = []
    z_out, s_out = PS.pair_stack_([b.fns(chunk, trimul_kw=dict(RB=24)) for b in blocks], zs, mask[lay.r0:lay.r1].contiguous(), lay, s=s.clone(),
                                  block_callback=lambda i, s_, z_: seen.append(i))
    return z_out, s_out, seen


@pytest.mark.parametrize("P,N,B,depth", [(2, 64, 16, 48), (4, 64, 16, 48), (3, 104, 16, 8)])
def test_mp_pair_stack_driver_equals_dense_stack(P, N, B, depth):
    dtype = torch.float64
    g = torch.Generator().manual_seed(14)
    blocks = [Block(8, 8, 2, 4, 12, g).to(dtype) for _ in range(depth)]
    for b in blocks:                                              # keep 48 residual blocks numerically tame: scale the output projections down
        with torch.no_grad():
            for m in (b.tmo.z, b.tmi.z, b.tas.o, b.tae.o, b.tr.o, b.apb.o, b.apb.st.o):
                m.weight.mul_(0.05)
    z, s, mask = _inputs(N, 8, 12, 9, dtype)
    with torch.no_grad():
        zd, sd = z, s
        for b in blocks:
            zd, sd = b.dense(zd, mask, sd)
        assert torch.isfinite(zd).all() and torch.isfinite(sd).all()
        res = run_ranks(P, _entry_stack, N, B, blocks, z, s, mask, 16)
    zg = torch.cat([r[0] for r in res], dim=0)
    mz, ms = _cmp(zg, zd), _cmp(res[0][1], sd)
    _report(f"pair_stack_depth{depth}", P=P, N=N, z=mz, s=ms, callbacks=res[0][2][:3] + ["..."], ref_scale=float(zd.abs().max()))
    assert mz["maxabs"] <= 1e-8 * max(1.0, float(zd.abs().max())) and ms["maxabs"] <= 1e-8 * max(1.0, float(sd.abs().max())), (mz, ms)
    assert all(r[2] == list(range(depth)) for r in res)


if __name__ == "__main__":
    rc = pytest.main([__file__, "-q", "-rfE", "-p", "no:cacheprovider"])
    print("SUMMARY " + json.dumps({"rc": int(rc), "results": len(RESULTS)}))
    sys.exit(int(rc))



def test_attention_core_accepts_the_tier_word_door_and_refuses_malformed_words(monkeypatch):
    """kernel='tier:<word>': kernels.triattn's row for a tier word per row window (attn.pair_fused.core_attention), the flash path by name where
    it refuses; a malformed word is refused at construction; the engineering override takes tier words too; on a CPU tensor the lever refuses
    exactly as the flash kernel does (it must be able to step aside to it)."""
    import pytest, torch
    from opt_core.mem.rowpair import triatt as T
    stock = lambda q, k, v, biases: q
    c = T.attention_core(stock, kernel="tier:fast")
    assert c.kernel == "tier:fast" and c.ledger.impl == "kernels.triattn:tier:fast"
    with pytest.raises(T.RowpairRefused):
        T.attention_core(stock, kernel="tier:")
    with pytest.raises(T.RowpairRefused):
        T.attention_core(stock, kernel="native")
    monkeypatch.setenv(T.ENV_TRIATT_CORE, "tier:exact")
    c2 = T.attention_core(stock, kernel="flash_triattn")
    assert c2.kernel == "tier:exact"
    monkeypatch.delenv(T.ENV_TRIATT_CORE)
    q = torch.zeros(1, 4, 2, 8, 16, dtype=torch.bfloat16)
    with pytest.raises(T.RowpairRefused) as r:
        c(q, q, q, [torch.zeros(1, 4, 1, 1, 8), torch.zeros(1, 1, 2, 8, 8)])
    assert "not a CUDA device" in str(r.value)
