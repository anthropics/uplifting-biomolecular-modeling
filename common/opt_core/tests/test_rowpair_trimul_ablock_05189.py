"""opt_core.mem.rowpair 0.5.18.9 — the A-BLOCK schedule of ``trimul.trimul_update_`` (``RA`` rows of A resident per pass, ``ROWPAIR_TRIMUL_ROWS_A`` /
``RA=``; more than one pass = z read-only + output row block + host row mirror) is BITWISE the one-pass schedule for both directions, every grid,
ragged layouts, RA not dividing Rmax, RA > Rmax; the opt-out ``ROWPAIR_TRIMUL_ABLOCK=0``; the census words; the mirror's lifecycle. And the
exact-size chunked host copies of the parks (``torch_hostpair.HostChunks`` / ``ROWPAIR_PARK_CHUNK_GIB``): chunk arithmetic, park -> rows -> restore
bitwise for awkward shapes and dtypes, the one-allocation opt-out. CPU rank-threads (``opt_core.testing.run_ranks``); the GPU legs are the kits'."""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import trimul as TM  # noqa: E402
from opt_core.mem.rowpair import trunk as T  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402
from opt_core.mem.rowpair.evidence import schedule as SCHED  # noqa: E402
from opt_core.mem import torch_hostpair as HP  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_rowpair_pairstack_043 import TriMul, _inputs  # noqa: E402


# ================================================================================================ K.T20: exact-size chunked host copies
def test_chunk_sizes_arithmetic():
    GiB, MiB = 1 << 30, 1 << 20
    n = int(147.54 * GiB)
    sizes = HP.chunk_sizes(n, 8 * GiB)
    assert sizes[:18] == [8 * GiB] * 18 and sum(sizes) >= n and sum(sizes) - n < 64 * MiB
    assert all(HP.next_pow2(x) == x for x in sizes), sizes                       # every piece a power of two: the caching host allocator wastes nothing
    assert sum(HP.next_pow2(x) for x in sizes) - n < 64 * MiB
    assert HP.chunk_sizes(n, 0) == [n]                                            # ROWPAIR_PARK_CHUNK_GIB=0: one allocation (today: next_pow2 -> 256 GiB)
    assert HP.next_pow2(n) == 256 * GiB
    assert HP.chunk_sizes(0, 8 * GiB) == []
    assert HP.chunk_sizes(5, 8 * GiB) == [5]
    assert HP.chunk_sizes(64 * MiB + 1, 8 * GiB) == [64 * MiB, 64 * MiB]
    assert HP.chunk_bytes("0") == 0 and HP.chunk_bytes("0.5") == GiB // 2 and HP.chunk_bytes(None) == 8 * GiB
    with pytest.raises(HP.MemLeverRefused):
        HP.chunk_bytes("-1")


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("chunk", [4096, 0, 1 << 20])
def test_host_chunks_round_trip_bitwise(dtype, chunk, monkeypatch):
    monkeypatch.setattr(HP, "CHUNK_TAIL_MIN", 512)                                   # small tensors: let the tail split like a large park's
    g = torch.Generator().manual_seed(5)
    t = torch.randn((1, 37, 50, 6), generator=g).to(dtype)                        # 11,100 elements: chunk=4096 B -> several chunks + a ragged tail
    pool = HP.PinPool(None, pin=False, pageable=True, lever="test")
    hc = HP.HostChunks(t.numel() * t.element_size(), pool, pin=False, chunk=chunk, tag="t")
    assert hc.where == "host" and hc.fallback is None and hc.pinned is False
    assert hc.n_chunks == len(HP.chunk_sizes(hc.nbytes, chunk)) and hc.alloc_bytes >= hc.nbytes and (hc.n_chunks == 1) == (chunk == 0)
    hc.store(t)
    back = torch.empty_like(t)
    assert torch.equal(hc.load(back), t)
    rowb = 50 * 6 * t.element_size()
    for i0, i1 in [(0, 37), (0, 5), (5, 36), (36, 37), (10, 10), (17, 18)]:
        stage = torch.empty((1, i1 - i0, 50, 6), dtype=dtype)
        assert torch.equal(hc.load_range(stage, i0 * rowb), t[..., i0:i1, :, :])
        assert torch.equal(T.load_rows(hc, t.shape, t.element_size(), i0, i1, torch.empty_like(stage)), t[..., i0:i1, :, :])
    patch = torch.randn((1, 3, 50, 6), generator=g).to(dtype)
    hc.store_range(patch, 20 * rowb)
    t2 = t.clone()
    t2[..., 20:23, :, :] = patch
    assert torch.equal(hc.load(torch.empty_like(t)), t2)
    with pytest.raises(HP.MemLeverRefused):
        hc.load_range(torch.empty((2, 50, 6), dtype=dtype), 36 * rowb)            # past the end
    assert pool.bytes_now == hc.alloc_bytes
    hc.release()
    assert pool.bytes_now == 0 and hc.released
    with pytest.raises(HP.MemLeverRefused):
        hc.load(back)


def test_host_chunks_lead_dims(monkeypatch):
    monkeypatch.setattr(HP, "CHUNK_TAIL_MIN", 16)
    t = torch.arange(2 * 5 * 4 * 3, dtype=torch.float32).reshape(2, 5, 4, 3)     # leading dim 2: rows i0:i1 are two byte ranges
    pool = HP.PinPool(None, pin=False, pageable=True, lever="test")
    hc = HP.HostChunks(t.numel() * 4, pool, pin=False, chunk=64, tag="t")
    hc.store(t)
    for i0, i1 in [(0, 5), (1, 4), (4, 5)]:
        stage = torch.empty((2, i1 - i0, 4, 3))
        assert torch.equal(T.load_rows(hc, t.shape, 4, i0, i1, stage), t[:, i0:i1])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_park_chunked_round_trip_bitwise(monkeypatch, dtype):
    monkeypatch.setenv("ROWPAIR_PARK_CHUNK_GIB", str(4096 / 2 ** 30))             # 4 KiB chunks: a 37x50x6 shard spans several + a tail
    monkeypatch.setattr(HP, "CHUNK_TAIL_MIN", 512)
    g = torch.Generator().manual_seed(3)
    z = torch.randn((1, 37, 50, 6), generator=g).to(dtype)
    lines = []
    p = T.ShardPark(z.clone(), park=True, force_host=True, name="z_init", log=lines.append)
    r = p.record()
    assert r["parked"] and r["where"] == "host" and r["chunks"] == len(HP.chunk_sizes(p.nbytes, 4096)) and r["chunks"] > 2, r
    assert any("chunks=" in ln and "pinned_gib=" in ln for ln in lines), lines
    w = p.census_words("park_z_init")
    assert w["park_z_init_chunks"] == r["chunks"] and "park_z_init_pinned_gib" in w
    for i0, i1 in [(0, 37), (0, 5), (5, 36), (36, 37), (10, 10)]:
        assert torch.equal(p.block(i0, i1), z[..., i0:i1, :, :])
    assert torch.equal(p.full(), z)
    p.release()
    assert p.chunks == 0
    monkeypatch.setenv("ROWPAIR_PARK_CHUNK_GIB", "0")                              # opt-out: one allocation
    q = T.ShardPark(z.clone(), park=True, force_host=True, name="z_init")
    assert q.record()["chunks"] == 1 and torch.equal(q.full(), z)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_parked_storage_chunked_round_trip_bitwise(monkeypatch, dtype):
    from opt_core.mem.rowpair.template import ParkedStorage
    monkeypatch.setenv("ROWPAIR_PARK_CHUNK_GIB", str(4096 / 2 ** 30))
    monkeypatch.setattr(HP, "CHUNK_TAIL_MIN", 512)
    monkeypatch.delenv("ROWPAIR_PARK_PIN_MAX_GB", raising=False)
    g = torch.Generator().manual_seed(4)
    z = torch.randn((1, 29, 61, 5), generator=g).to(dtype)
    ref = z.clone()
    ps = ParkedStorage(z, name="z_shard (test)")
    assert ps.ok and ps.where == "host" and ps.chunks > 2 and z.untyped_storage().nbytes() == 0
    for i0, i1 in [(0, 29), (3, 17), (28, 29)]:
        assert torch.equal(ps.block(i0, i1), ref[..., i0:i1, :, :])
    out = ps.restore()
    assert out is z and torch.equal(z, ref) and ps.chunks == 0 and ps.where == "device"


# ================================================================================================ K.T23: the A-block schedule
def test_ablock_rows_words(monkeypatch):
    monkeypatch.delenv("ROWPAIR_TRIMUL_ROWS_A", raising=False)
    monkeypatch.delenv("ROWPAIR_TRIMUL_ABLOCK", raising=False)
    assert TM.ablock_rows(8832) == (8832, 1, "whole")
    assert TM.ablock_rows(8832, 1536) == (1472, 6, "fixed")                      # balanced: 6 x 1472 == 8832
    assert TM.ablock_rows(8448, 1536) == (1408, 6, "fixed")
    assert TM.ablock_rows(7552, 1536) == (1520, 5, "fixed")                      # ceil(7552/5)=1511 -> x16 = 1520 (<= the cap)
    assert TM.ablock_rows(100, 4096) == (100, 1, "fixed")                        # cap > Rmax: one pass
    assert TM.ablock_rows(48, 7) == (7, 7, "fixed")                              # x16 would exceed the cap: exact ceil
    assert TM.ablock_rows(48, 47) == (32, 2, "fixed")                            # no 47 + 1 sliver
    monkeypatch.setenv("ROWPAIR_TRIMUL_ROWS_A", "1024")
    assert TM.ablock_rows(8832) == (992, 9, "env")
    assert TM.ablock_rows(8832, 2048) == (1776, 5, "fixed")                      # the argument wins
    monkeypatch.setenv("ROWPAIR_TRIMUL_ABLOCK", "0")
    assert TM.ablock_rows(8832) == (8832, 1, "off") and TM.ablock_rows(8832, 2048) == (8832, 1, "off")
    monkeypatch.setenv("ROWPAIR_TRIMUL_ABLOCK", "1")
    monkeypatch.setenv("ROWPAIR_TRIMUL_ROWS_A", "many")
    with pytest.raises(TM.RowpairRefused):
        TM.ablock_rows(8832)


def test_pass_windows():
    lay = Layout(200, 4, 0, 16)
    grid = TM.slab_grid(lay, 24, None)
    for t in range(max(len(g) for g in grid)):
        assert TM.pass_windows(lay, grid, t, 10 ** 6, 0) == TM.grid_windows(lay, grid, t)        # one pass == today's windows
    wins = [TM.pass_windows(lay, grid, t, 16, 1) for t in range(3)]                           # pass 1 = local rows 16:32: sub-blocks 0 (0:24) and 1 (24:48)
    assert wins[0][0] != (0, 0) and wins[1][0] != (0, 0) and wins[2][0] == (0, 0)


class ExactTriMul(object):
    """TriMulFns whose every statement is EXACT in fp32 (small integers: no LayerNorm, 0/1 gates, +-1/0 weights), so any summation order gives
    the same bits: one pass vs many passes must then agree BITWISE whatever paths the BLAS takes for the differing row extents — a test of the
    schedule's LOGIC (coverage, original-row reads, the output block / mirror plumbing) independent of tile-extent invariance."""

    def __init__(self, C, C_h, g):
        self.C, self.C_h = C, C_h
        self.W = torch.randint(-1, 2, (C_h, C), generator=g).float()

    def proj(self, zblk, mblk, is_a):
        x = zblk[..., : self.C_h] if is_a else zblk[..., self.C_h: 2 * self.C_h]
        return x * mblk

    def out(self, x):
        return x @ self.W.to(x.dtype)

    def gate(self, zblk):
        return (zblk[..., :1] > 0).to(zblk.dtype)

    def dense(self, z, mask, outgoing):
        m = mask[..., None]
        a, b = self.proj(z, m, True), self.proj(z, m, False)
        x = torch.einsum("ikc,jkc->ijc", a, b) if outgoing else torch.einsum("kic,kjc->ijc", a, b)
        return z + self.out(x) * self.gate(z)

    def fns(self):
        return TM.TriMulFns(self.proj, self.out, self.gate, self.C_h)

    def to(self, dtype):
        self.W = self.W.to(dtype)
        return self


def _exact_inputs(N, C, g, dtype):
    z = torch.randint(-3, 4, (N, N, C), generator=g).to(dtype)
    mask = (torch.rand((N, N), generator=g) > 0.15).to(dtype)
    return z, mask


def _entry_update(rank, P, N, B, mod, z, mask, outgoing, kw):
    lay = Layout(N, P, rank, B)
    zs = z[lay.r0:lay.r1].clone().contiguous()
    st = TM.ContractStats()
    out = TM.trimul_update_(mod.fns(), zs, mask[lay.r0:lay.r1].contiguous(), lay, outgoing=outgoing, stats=st, **dict({"inplace_chunk": 48}, **kw))
    assert out is zs
    mirror = TM._MIRRORS.get((str(zs.device), rank, P))
    return zs, st.facts(), (id(mirror) if mirror is not None else None), dict(SCHED())


LAYOUTS = [(2, 80, 16), (3, 104, 16), (3, 120, 16), (4, 200, 16), (2, 50, 16)]     # rows 48|32, 48|48|8, 48|48|24, 64|64|64|8, 32|18
GRIDS = (("stock_grid", dict(RB=24, grid="stock")), ("rank_grid_mmrows", dict(RB=40, grid="rank", mm_rows=16)), ("one_subblock_per_rank", dict(RB=4096, grid="rank")))


def _ra_cases(Rmax):
    return (1, 7, 12, 16, Rmax - 1, Rmax, 4096)


@pytest.mark.parametrize("P,N,B", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mp_ablock_equals_one_pass_bitwise_exact_arithmetic(P, N, B, dtype, monkeypatch):
    """Exact statements: many passes == one pass == the dense statement BITWISE, every layout (ragged, a rank whose last pass is one row, RA
    not dividing Rmax, RA >= Rmax), every grid, both directions; the schedule facts and census words say what ran."""
    monkeypatch.delenv("ROWPAIR_TRIMUL_ROWS_A", raising=False)
    monkeypatch.delenv("ROWPAIR_TRIMUL_ABLOCK", raising=False)
    g = torch.Generator().manual_seed(21)
    mod = ExactTriMul(8, 4, g).to(dtype)
    z, mask = _exact_inputs(N, 8, g, dtype)
    Rmax = Layout(N, P, 0, B).Rmax
    for outgoing in (True, False):
        dense = mod.dense(z, mask, outgoing)
        for tag, kw in GRIDS:
            one = run_ranks(P, _entry_update, N, B, mod, z, mask, outgoing, kw)
            ref = torch.cat([r[0] for r in one], dim=0)
            assert all(f["passes"] == 1 for _, f, _, _ in one) and torch.equal(ref, dense), (tag, outgoing)
            for RA in _ra_cases(Rmax):
                res = run_ranks(P, _entry_update, N, B, mod, z, mask, outgoing, dict(kw, RA=RA))
                got = torch.cat([r[0] for r in res], dim=0)
                ra_eff, npass, src = TM.ablock_rows(Rmax, RA)
                assert npass == -(-Rmax // min(RA, Rmax)) and ra_eff <= min(RA, Rmax) and src == "fixed"
                assert all(f["passes"] == npass for _, f, _, _ in res), (RA, [f["passes"] for _, f, _, _ in res])
                assert torch.equal(got, ref), (tag, outgoing, RA, float((got.double() - ref.double()).abs().max()))
                sched = res[0][3]
                assert sched["trimul_ablock_ra"] == ra_eff and sched["trimul_ablock_passes"] == npass and sched["trimul_ablock_source"] == "fixed"
                if npass > 1:
                    assert all(m is not None for _, _, m, _ in res) and sched.get("trimul_ablock_host") == "host"
                    assert all(f["deferred_tiles"] == 0 for _, f, _, _ in res)                    # nothing deferred: z is read-only through the call


@pytest.mark.parametrize("P,N,B", LAYOUTS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_mp_ablock_module_statements_within_rounding(P, N, B, dtype, monkeypatch):
    """The AF3-family statements (LayerNorms, sigmoid gates): many passes vs one pass agree to the last bits CPU BLAS keeps across row extents
    (|d| <= 1e-5 fp32 / 1e-12 fp64 — an M=1 or M=7 product may take another BLAS path than M=48; on the GPU line the kits' legs measure
    bitwise), and both are the dense statement within the family's tolerance."""
    monkeypatch.delenv("ROWPAIR_TRIMUL_ROWS_A", raising=False)
    monkeypatch.delenv("ROWPAIR_TRIMUL_ABLOCK", raising=False)
    g = torch.Generator().manual_seed(11)
    mod = TriMul(8, 6, g).to(dtype)
    z, _, mask = _inputs(N, 8, 12, 5, dtype)
    Rmax = Layout(N, P, 0, B).Rmax
    tol = 1e-12 if dtype == torch.float64 else 1e-5
    n_bitwise = n = 0
    with torch.no_grad():
        for outgoing in (True, False):
            dense = mod.dense(z, mask, outgoing)
            for tag, kw in GRIDS:
                ref = torch.cat([r[0] for r in run_ranks(P, _entry_update, N, B, mod, z, mask, outgoing, kw)], dim=0)
                assert float((ref.double() - dense.double()).abs().max()) <= (1e-9 if dtype == torch.float64 else 1e-5)
                for RA in _ra_cases(Rmax):
                    got = torch.cat([r[0] for r in run_ranks(P, _entry_update, N, B, mod, z, mask, outgoing, dict(kw, RA=RA))], dim=0)
                    d = float((got.double() - ref.double()).abs().max())
                    assert d <= tol, (tag, outgoing, RA, d)
                    n += 1
                    n_bitwise += int(d == 0.0)
    print(f"RESULT ablock_module P={P} N={N} dtype={dtype}: {n_bitwise}/{n} cells bitwise, all within {tol}")


def test_mp_ablock_env_optout_and_mirror_reuse(monkeypatch):
    P, N, B, dtype = 3, 120, 16, torch.float32
    g = torch.Generator().manual_seed(12)
    mod = ExactTriMul(8, 4, g).to(dtype)
    z, mask = _exact_inputs(N, 8, g, dtype)
    Rmax = Layout(N, P, 0, B).Rmax
    TM.release_host_mirror()
    monkeypatch.delenv("ROWPAIR_TRIMUL_ROWS_A", raising=False)
    monkeypatch.delenv("ROWPAIR_TRIMUL_ABLOCK", raising=False)
    ref = torch.cat([r[0] for r in run_ranks(P, _entry_update, N, B, mod, z, mask, True, dict(RB=24))], dim=0)
    monkeypatch.setenv("ROWPAIR_TRIMUL_ROWS_A", "10")
    res1 = run_ranks(P, _entry_update, N, B, mod, z, mask, True, dict(RB=24))
    assert all(f["passes"] == -(-Rmax // 10) for _, f, _, _ in res1) and res1[0][3]["trimul_ablock_source"] == "env"
    assert torch.equal(torch.cat([r[0] for r in res1], dim=0), ref)
    res2 = run_ranks(P, _entry_update, N, B, mod, z, mask, False, dict(RB=24))
    assert [m for _, _, m, _ in res2] == [m for _, _, m, _ in res1]                 # the mirror is allocated ONCE per rank and re-used by later calls
    monkeypatch.setenv("ROWPAIR_TRIMUL_ABLOCK", "0")                                  # opt-out by name: one pass whatever ROWS_A / RA= say
    res3 = run_ranks(P, _entry_update, N, B, mod, z, mask, True, dict(RB=24, RA=10))
    assert all(f["passes"] == 1 for _, f, _, _ in res3) and res3[0][3]["trimul_ablock_source"] == "off"
    assert torch.equal(torch.cat([r[0] for r in res3], dim=0), ref)
    d = TM.describe_ablock()
    assert d["calls"] > 0 and d["words"].startswith("ablock=on ablock_ra=") and " ablock_passes=1 " in d["words"] and "ablock_host=host " in d["words"], d
    assert TM.release_host_mirror() == P and not TM._MIRRORS


def test_ablock_ranks_differ_is_a_named_refusal():
    """A per-rank RA (a rank-local row count handed to RA=) gives the ranks different pass counts: refused by name before any collective."""
    lay0, lay1 = Layout(4133, 2, 0, 16), Layout(4133, 2, 1, 16)                     # ragged: R = 2080 | 2053 (the N=4133 cell shape class)
    per_rank = {0: TM.ablock_rows(lay0.Rmax, lay0.R), 1: TM.ablock_rows(lay1.Rmax, lay1.R)}   # RA = R_local: rank 0 one pass, rank 1 two
    assert per_rank[0][1] == 1 and per_rank[1][1] == 2
    world = lambda me: (lambda obj: [obj if r == me else (r,) + per_rank[r] for r in (0, 1)])  # noqa: E731 — an all-gather stand-in
    for lay in (lay0, lay1):
        ra, npass, src = per_rank[lay.rank]
        with pytest.raises(TM.RowpairRefused, match=r"trimul_ablock_ranks_differ: rank 0 ra=%d passes=1 .*rank 1 ra=%d passes=2" % (per_rank[0][0], per_rank[1][0])):
            TM.require_ablock_agreement(lay, ra, npass, src, gather=world(lay.rank))
    same = TM.ablock_rows(lay0.Rmax, 512)
    TM.require_ablock_agreement(lay0, same[0], same[1], same[2], gather=lambda obj: [obj, (1,) + same])       # equal schedules: silent
    TM.require_ablock_agreement(lay0, same[0], same[1], same[2])                                              # no group: silent
