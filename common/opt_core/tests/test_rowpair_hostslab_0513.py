"""opt_core.mem.rowpair 0.5.213 — W2 lever (d) "host parks 1x": the LEASED host slab (``ROWPAIR_HOST_SLAB=lease``: trunk.HostSlabLease / LeasedHost /
park_host; one u-sized host copy per rank handed to ONE holder at a time — ShardPark, template.ParkedStorage, heads.ZTrunkPlan, the trimul row mirror),
the COPY-FREE z_init (``ROWPAIR_PARK_ZINIT=recompute``: ShardPark.recompute re-runs the init statement on the init grid), ZTrunkPlan.retire (the park
leaves before the last pass's pair stack), and DEFAULTS UNCHANGED (the words unset: the same host allocations in the same order as 0.5.212, no lease
object anywhere). CPU (plain host buffers stand in for pinned ones; rank-threads via opt_core.testing.run_ranks); the GPU legs are the kits'."""
from __future__ import annotations

import os
import sys

import pytest

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import heads  # noqa: E402
from opt_core.mem.rowpair import trimul as TM  # noqa: E402
from opt_core.mem.rowpair import trunk as T  # noqa: E402
from opt_core.mem.rowpair.dist import Layout  # noqa: E402
from opt_core.mem.rowpair.evidence import schedule as SCHED  # noqa: E402
from opt_core.mem.rowpair.template import ParkedStorage  # noqa: E402
from opt_core.mem import torch_hostpair as HP  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402

sys.path.insert(0, os.path.dirname(__file__))
from test_rowpair_trunk_043 import Synth  # noqa: E402
from test_rowpair_trimul_ablock_05189 import ExactTriMul, _exact_inputs  # noqa: E402

WORDS = ("ROWPAIR_HOST_SLAB", "ROWPAIR_HOST_SLAB_STRICT", "ROWPAIR_PARK_ZINIT", "ROWPAIR_PARK_ZINIT_HOLD", "ROWPAIR_PARK_CHUNK_GIB", "ROWPAIR_TRIMUL_ROWS_A",
         "ROWPAIR_TRIMUL_ABLOCK", "ROWPAIR_FREE_ZTRUNK", "ROWPAIR_CONF_PARK_ZTRUNK", "ROWPAIR_POOL_SHRINK", "ROWPAIR_INIT_ROWS", "ROWPAIR_RECYCLE_ROWS")


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    for w in WORDS:
        monkeypatch.delenv(w, raising=False)
    T.release_host_slabs()
    TM.release_host_mirror()
    yield
    T.release_host_slabs()
    TM.release_host_mirror()


def _z(R=6, N=10, C=4, seed=3, dtype=torch.float32):
    g = torch.Generator().manual_seed(seed)
    return torch.randn((R, N, C), generator=g).to(dtype)


class _Ev(object):
    """A hand-over event double: records construction order and wait() calls into a shared transcript."""

    def __init__(self, log, n):
        self.log, self.n = log, n
        log.append(("record", n))

    def wait(self):
        self.log.append(("wait", self.n))


def _factory(log):
    box = {"n": 0}

    def make():
        box["n"] += 1
        return _Ev(log, box["n"])
    return make


# ============================================================================================================ words
def test_mode_words(monkeypatch):
    assert T.host_slab_mode() == "private" and T.zinit_park_mode() == "resident"
    for w, m in (("lease", "lease"), ("1", "lease"), ("private", "private"), ("0", "private"), ("", "private")):
        monkeypatch.setenv("ROWPAIR_HOST_SLAB", w)
        assert T.host_slab_mode() == m, w
    monkeypatch.setenv("ROWPAIR_HOST_SLAB", "pool")
    with pytest.raises(RowpairRefused, match="ROWPAIR_HOST_SLAB='pool'"):
        T.host_slab_mode()
    monkeypatch.delenv("ROWPAIR_HOST_SLAB")
    for w, m in (("recompute", "recompute"), ("1", "parked"), ("true", "parked"), ("0", "resident")):
        monkeypatch.setenv("ROWPAIR_PARK_ZINIT", w)
        assert T.zinit_park_mode() == m, w
    assert T.zinit_park_mode(True) == "parked" and T.zinit_park_mode(False) == "resident" and T.zinit_park_mode("recompute") == "recompute"
    with pytest.raises(RowpairRefused, match="z_init park word"):
        T.zinit_park_mode("sometimes")


# ============================================================================================================ the lease itself
def test_lease_handover_order_grow_and_refusals_after_release():
    log = []
    lease = T.HostSlabLease(pin=False, event_factory=_factory(log), name="slab")
    a, b, big = _z(), _z(seed=4), _z(R=9, seed=5)
    ha = lease.acquire_for(a, "A")
    assert isinstance(ha, T.LeasedHost) and ha.leased and lease.holder == "A" and not lease.free and ha.nbytes == a.numel() * 4
    ha.store(a)
    back = torch.empty_like(a)
    assert torch.equal(ha.load(back), a) and torch.equal(ha.load_range(torch.empty((2, 10, 4)), 4 * 10 * 4 * 1), a[1:3])
    ha.release()
    assert lease.free and log == [("record", 1)]                                  # the hand-over event is recorded AT release
    with pytest.raises(RowpairRefused, match="after release"):
        ha.store(a)
    ha.release()                                                                   # idempotent
    hb = lease.acquire_for(b, "B")
    assert log == [("record", 1), ("wait", 1)]                                     # B WAITED on A's event before it got the bytes
    assert hb.slab is ha.slab and lease.n_grow == 0                                # the same slab (same size): no re-allocation
    with pytest.raises(RowpairRefused, match="outside its lease"):
        hb.load_range(torch.empty((7, 10, 4)), 0)
    hb.release()
    hbig = lease.acquire_for(big, "BIG")                                           # a larger holder: the slab grows (old chunks released first)
    assert lease.n_grow == 1 and lease.nbytes == big.numel() * 4 and log[-2:] == [("wait", 2), ("record", 2)][::-1]
    hbig.store(big)
    assert torch.equal(hbig.load(torch.empty_like(big)), big)
    hbig.release()
    hsmall = lease.acquire_for(a, "A2")                                            # a smaller holder after a larger: a PREFIX of the slab, no re-allocation
    assert lease.n_grow == 1 and hsmall.nbytes == a.numel() * 4 and lease.nbytes == big.numel() * 4
    hsmall.release()
    kinds = [e[0] for e in lease.events]
    assert kinds == ["lease", "release", "wait", "lease", "release", "wait", "grow", "lease", "release", "wait", "lease", "release"], kinds
    assert lease.drop() == lease.events[-1][2] or lease.slab is None               # a free slab drops; the registry's drop returns its bytes
    assert lease.drop() == 0 and lease.record()["leases"] == 4


def test_second_holder_named_private_and_strict_refusal(monkeypatch):
    SCHED().clear() if hasattr(SCHED(), "clear") else None
    said = []
    lease = T.HostSlabLease(pin=False, name="slab")
    a, b = _z(), _z(R=3, seed=8)
    ha = lease.acquire_for(a, "z_trunk park", say=said.append)
    hb = lease.acquire_for(b, "trimul row mirror", say=said.append)                # the slab is HELD: a private buffer, NAMED
    assert isinstance(hb, HP.HostChunks) and not getattr(hb, "leased", False) and lease.holder == "z_trunk park" and lease.n_second == 1
    sched = dict(SCHED())
    assert sched["host_slab_second"] == "z_trunk_park+trimul_row_mirror" and sched["host_slab_seconds"] == 1   # census values: single tokens (0.5.217.1)
    assert abs(sched["host_slab_second_gib"] - b.numel() * 4 / 2 ** 30) < 1e-3 or sched["host_slab_second_gib"] == round(b.numel() * 4 / 2 ** 30, 3)
    assert any("PRIVATE" in m and "host_slab_second=z_trunk park+trimul row mirror" in m for m in said), said
    hb.store(b)
    assert torch.equal(hb.load(torch.empty_like(b)), b)
    hb.release()
    assert lease.holder == "z_trunk park"                                          # the second's release does not touch the lease
    ha.release()
    assert lease.free
    # strict: refused by name, naming both holders and the word
    strict = T.HostSlabLease(pin=False, strict_second=True, name="slab")
    h1 = strict.acquire_for(a, "template z park")
    with pytest.raises(RowpairRefused, match=r"host_slab_second: trimul row mirror .* while template z park holds it; ROWPAIR_HOST_SLAB_STRICT=1"):
        strict.acquire_for(b, "trimul row mirror")
    h1.release()
    monkeypatch.setenv("ROWPAIR_HOST_SLAB_STRICT", "1")                             # the env word reaches a lease built without the argument
    assert T.HostSlabLease(pin=False).strict_second is True


def test_registry_per_rank_and_host_alloc_on_a_lease(monkeypatch):
    monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
    z = _z()

    def entry(rank, P):
        lease = T.park_host(z)
        assert isinstance(lease, T.HostSlabLease) and lease is T.host_slab_lease(z.device, pin=False)
        host, where, fallback, pinned = T.host_alloc(lease, z, f"park r{rank}")
        assert isinstance(host, T.LeasedHost) and where == "host" and fallback is None and pinned is False
        host.store(z)
        out = host.load(torch.empty_like(z))
        host.release()
        return id(lease), torch.equal(out, z), dict(SCHED()).get("host_slab")

    res = run_ranks(3, entry)
    assert len({r[0] for r in res}) == 3 and all(r[1] for r in res) and all(r[2] == "lease" for r in res)   # one lease per rank-thread
    assert len(T.host_slab_leases()) == 3 and T.host_slab_bytes()[0] == 3 and T.host_slab_bytes()[1] == 0
    assert T.release_free_host_slabs() > 0 and T.host_slab_bytes() == (0, 0, 0)
    assert T.release_host_slabs() == 3 and T.host_slab_leases() == {}
    monkeypatch.delenv("ROWPAIR_HOST_SLAB")
    assert isinstance(T.park_host(z), HP.PinPool) and dict(SCHED()).get("host_slab") == "private"   # unset: today's private pool


# ============================================================================================================ the parks on the lease
def test_parks_hand_the_slab_over_in_call_order(monkeypatch):
    """z_init park -> template z park -> row mirror -> z_trunk park: each takes THE slab after the previous released it; bytes bitwise."""
    monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
    z = _z(R=8, N=12, C=4)
    ref = z.clone()
    # z_init (ShardPark, force_host on CPU)
    p = T.ShardPark(z.clone(), park=True, force_host=True, name="z_init")
    lease = T.host_slab_lease(z.device, pin=False)
    assert isinstance(p.host, T.LeasedHost) and lease.holder == "ShardPark(z_init)" and p.where == "host" and p.mode == "parked"
    assert torch.equal(p.block(2, 7), ref[2:7]) and torch.equal(p.full(), ref)
    p.release()
    assert lease.free and lease.slab is not None                                   # handed back, the slab stays allocated
    # template z park (ParkedStorage, pool=None -> the lease)
    t = z.clone()
    ps = ParkedStorage(t, name="z_shard (template pair stack)")
    assert isinstance(ps.host, T.LeasedHost) and lease.holder == "ParkedStorage(z_shard (template pair stack))" and t.untyped_storage().nbytes() == 0
    assert torch.equal(ps.block(0, 3), ref[0:3])
    ps.restore()
    assert torch.equal(t, ref) and lease.free
    # z_trunk (ZTrunkPlan.park_now, pool=None -> the lease), 2 passes; retire on the last pass hands the slab back BEFORE end
    zt = z.clone()
    plan = heads.ZTrunkPlan(zt, passes=2, free=True, park=True)
    assert plan.park_now().startswith("parked:host") and isinstance(plan.parked.host, T.LeasedHost) and lease.holder == "ParkedStorage(z_trunk (confidence))"
    src, inplace = plan.begin(0)
    assert src is plan.parked and not inplace and torch.equal(src.zrows(1, 5), ref[1:5])
    assert plan.retire(0) == "kept:not_last" and plan.parked is not None and not lease.free   # an earlier pass keeps the park
    # a mirror-class holder while the park holds the slab: second (named), private
    h2 = T.host_alloc(lease, z, "trimul row mirror")[0]
    assert isinstance(h2, HP.HostChunks) and dict(SCHED())["host_slab_second"] == "ParkedStorage:z_trunk.confidence+trimul_row_mirror"
    assert dict(SCHED())["host_slab_holder"] == "ParkedStorage:z_trunk.confidence"                                    # the log lines keep the readable names
    h2.release()
    plan.end(0)
    src1, _ = plan.begin(1)
    assert src1 is src and torch.equal(src1.zrows(0, 8), ref)
    assert plan.retire(1) == "pass1@embed" and plan.parked is None and plan.consumed and lease.free
    assert dict(SCHED())["ztrunk_retired"] == "pass1@embed" and dict(SCHED())["conf_ztrunk_retired"] == "pass1@embed"
    assert plan.retire(1) == "pass1@embed"                                         # idempotent
    h3 = T.host_alloc(lease, z, "trimul row mirror")[0]                            # the last pass's mirror now LEASES the slab (1u, not 2u)
    assert isinstance(h3, T.LeasedHost) and lease.holder == "trimul row mirror"
    h3.release()
    with pytest.raises(RowpairRefused, match="RETIRED"):
        plan.end(1, device_needed_next=True)
    plan.end(1)
    assert plan.closed and plan.consumed and zt.untyped_storage().nbytes() == 0
    kinds = [e[0] for e in lease.events]
    assert kinds.count("lease") == 4 and kinds.count("second") == 1 and lease.n_grow == 0, kinds


def test_retire_words_resident_and_inplace():
    z = _z()
    res = heads.ZTrunkPlan(z.clone(), passes=1, free=False, park=False)
    with pytest.raises(RowpairRefused, match="not begun"):
        res.retire(0)
    res.begin(0)
    assert res.retire(0) == "nothing_parked" and dict(SCHED())["ztrunk_retired"] == "nothing_parked"
    res.end(0)
    zi = z.clone().contiguous()
    inp = heads.ZTrunkPlan(zi, passes=1, free=True, park=True)
    src, inplace = inp.begin(0)
    assert inplace and inp.words[0] == "inplace" and inp.retire(0) == "nothing_parked"
    inp.end(0)
    with pytest.raises(RowpairRefused, match="retire"):
        inp.retire(3)


# ============================================================================================================ recompute z_init
@pytest.mark.parametrize("P,N,B", [(2, 40, 8), (3, 52, 4), (2, 30, 2)])
@pytest.mark.parametrize("like_dtype", [torch.float32, torch.bfloat16])
def test_recompute_park_equals_parked_park_bitwise(P, N, B, like_dtype, monkeypatch):
    """ShardPark.recompute serves, per row block, exactly the bytes the parked copy of init_pair_shard's shard serves (any block range: inside one
    init block, spanning several, ragged ends, the whole shard), in the shard's dtype (rows_fn returns fp32; a bf16 `like` exercises the cast), and its
    grid IS init_pair_shard's (the rows_fn calls init_pair_shard makes == the recompute grid)."""
    C = 4
    sy = Synth(N, C)

    def entry(rank, P):
        torch.set_num_threads(1)
        lay = Layout(N, P, rank, B)
        like = sy.a_i.to(like_dtype)
        calls = []

        def rows_fn(g0, g1):
            calls.append((g0, g1))
            return sy.rows_fn(g0, g1)                                           # fp32 statement; the shard is like.dtype
        out = {}
        for rows in (5, 7, None):
            calls.clear()
            zd = T.init_pair_shard(lay, rows_fn, like, rows=rows)
            init_calls = list(calls)
            parked = T.ShardPark(zd.clone().contiguous(), park=True, force_host=True, name="z_init")
            rp = T.ShardPark.recompute(lay, rows_fn, like, rows=rows, name="z_init")
            assert rp.mode == "recompute" and rp.where == "recompute" and rp.host is None and rp.dev is None and rp.shape == tuple(zd.shape)
            assert [(g0, g1) for _, _, g0, g1 in rp._grid] == init_calls, (rows, rp._grid, init_calls)   # THE init grid
            R = lay.R
            ranges = [(0, R), (0, 1), (R - 1, R), (1, min(R, 6)), (2, min(R, 13)), (0, min(R, 5)), (min(R, 5), min(R, 10))]
            eq = all(torch.equal(rp.block(i0, i1), parked.block(i0, i1)) for i0, i1 in ranges if i0 < i1)
            eq = eq and torch.equal(rp.full(), parked.full()) and rp.full().dtype == like_dtype
            # the recycle on either park: bitwise
            g = torch.Generator().manual_seed(rank + 1)
            z_prev = torch.randn(zd.shape, generator=g).to(like_dtype)
            a = T.recycle_shard_(z_prev.clone().contiguous(), parked, lambda x: sy.update(x.float()).to(like_dtype), lay, rows=6)
            b = T.recycle_shard_(z_prev.clone().contiguous(), rp, lambda x: sy.update(x.float()).to(like_dtype), lay, rows=6)
            a0 = T.recycle_shard_(None, parked, lambda x: sy.update(x.float()).to(like_dtype), lay, rows=None)
            b0 = T.recycle_shard_(None, rp, lambda x: sy.update(x.float()).to(like_dtype), lay, rows=None)
            out[str(rows)] = bool(eq and torch.equal(a, b) and torch.equal(a0, b0))
            words = rp.census_words("park_z_init")
            assert words["park_z_init"] == "recompute" and words["park_z_init_release"] == "not_materialized"
            n = rp.n_recompute
            parked.release(); rp.release()
            assert n > 0 and dict(SCHED())["zinit_recompute_calls"] == n
            with pytest.raises(RowpairRefused, match="released"):
                rp.block(0, 1)
        return out

    res = run_ranks(P, entry)
    assert all(all(r.values()) for r in res), res


@pytest.mark.parametrize("P", [2, 3])
def test_driver_recompute_equals_parked_and_resident_bitwise(P, monkeypatch):
    """run_trunk_sharded under ROWPAIR_PARK_ZINIT=recompute == =1 (parked; CPU: resident by contract) == unset, 3 cycles, bitwise; the census says
    zinit_park=recompute and no z_init host bytes exist; init_pair_shard never runs under recompute (cycle 0's recycle re-runs the statement)."""
    N, C, B = 40, 4, 8
    sy = Synth(N, C)

    def entry(rank, P, word):
        torch.set_num_threads(1)
        if word is None:
            os.environ.pop("ROWPAIR_PARK_ZINIT", None)
        else:
            os.environ["ROWPAIR_PARK_ZINIT"] = word
        lay = Layout(N, P, rank, B)
        torch.manual_seed(7)
        kw = dict(n_cycles=3, init_rows_fn=sy.rows_fn, init_like=sy.a_i, recycle_update_fn=sy.update, s_init=sy.s_input,
                  single_recycle_fn=lambda s, c: sy.single(s), template_fn=lambda z, c: z + sy.template(z),
                  pairstack_fn=lambda s, z, c: sy.pairstack_rows(s, z), init_rows=5, recycle_rows=6, guard="off", gather="none")
        s, z, out = T.run_trunk_sharded(lay, **kw)
        sched = dict(SCHED())
        return z.clone(), s.clone(), out.record["park"], sched.get("zinit_park"), sched.get("park_z_init"), out.record["timings_s"]["init"]

    ref = run_ranks(P, entry, None)
    one = run_ranks(P, entry, "1")
    rec = run_ranks(P, entry, "recompute")
    for r in range(P):
        assert torch.equal(ref[r][0], one[r][0]) and torch.equal(ref[r][0], rec[r][0]), r
        assert torch.equal(ref[r][1], rec[r][1])
        assert ref[r][3] == "resident" and one[r][3] == "resident" and rec[r][3] == "recompute"       # CPU: '1' without force_host stays resident
        assert rec[r][2]["where"] == "recompute" and rec[r][2]["parked"] is False and rec[r][4] == "recompute" and rec[r][5] < ref[r][5] + 1.0
    os.environ.pop("ROWPAIR_PARK_ZINIT", None)


# ============================================================================================================ trimul row mirror on the lease
def _entry_mirror(rank, P, N, B, mod, z, mask, outgoing, RA, hold_slab):
    torch.set_num_threads(1)
    lay = Layout(N, P, rank, B)
    zs = z[lay.r0:lay.r1].clone().contiguous()
    holder = None
    if hold_slab:                                                                  # a z park holds the slab through the call: the mirror is the SECOND holder
        holder = T.host_alloc(T.host_slab_lease(zs.device, pin=False), zs, "z_trunk park")[0]
    st = TM.ContractStats()
    out = TM.trimul_update_(mod.fns(), zs, mask[lay.r0:lay.r1].contiguous(), lay, outgoing=outgoing, stats=st, inplace_chunk=48, RB=24, grid="stock", RA=RA)
    lease = T.host_slab_leases().get((str(zs.device), rank))
    facts = {"passes": st.facts()["passes"], "mirrors_held": len(TM._MIRRORS), "lease_free": (lease.free if lease is not None else None),
             "events": ([e[0] for e in lease.events] if lease is not None else []), "holder_now": (lease.holder if lease is not None else None),
             "ablock_host_slab": TM.ABLOCK.get("host_slab"), "sched_second": dict(SCHED()).get("host_slab_second")}
    if holder is not None:
        holder.release()
    return out, facts


@pytest.mark.parametrize("P,N,B", [(2, 80, 16), (3, 104, 16)])
def test_mirror_leases_per_call_bitwise_and_second_when_held(P, N, B, monkeypatch):
    g = torch.Generator().manual_seed(21)
    mod = ExactTriMul(8, 4, g)
    z, mask = _exact_inputs(N, 8, g, torch.float32)
    for outgoing in (True, False):
        dense = mod.dense(z, mask, outgoing)
        monkeypatch.delenv("ROWPAIR_HOST_SLAB", raising=False)
        T.release_host_slabs(); TM.release_host_mirror()
        base = run_ranks(P, _entry_mirror, N, B, mod, z, mask, outgoing, 12, False)          # today: process-held mirror
        got0 = torch.cat([r[0] for r in base], dim=0)
        assert torch.equal(got0, dense) and all(f["passes"] > 1 for _, f in base) and base[0][1]["mirrors_held"] == P and base[0][1]["events"] == []
        TM.release_host_mirror()
        monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
        leased = run_ranks(P, _entry_mirror, N, B, mod, z, mask, outgoing, 12, False)
        got1 = torch.cat([r[0] for r in leased], dim=0)
        assert torch.equal(got1, dense), float((got1 - dense).abs().max())
        f = leased[0][1]
        assert f["mirrors_held"] == 0 and f["lease_free"] is True and f["events"][:2] == ["lease", "release"] and f["ablock_host_slab"] == "lease"
        T.release_host_slabs()
        held = run_ranks(P, _entry_mirror, N, B, mod, z, mask, outgoing, 12, True)          # the slab is held by a park: second, named, still bitwise
        got2 = torch.cat([r[0] for r in held], dim=0)
        assert torch.equal(got2, dense)
        f = held[0][1]
        assert f["mirrors_held"] == 0 and f["events"].count("second") >= 1 and f["holder_now"] == "z_trunk park" and f["ablock_host_slab"] == "second"
        assert f["sched_second"] == "z_trunk_park+trimul_row_mirror"
        T.release_host_slabs()


# ============================================================================================================ defaults unchanged
class _RecPool(HP.PinPool):
    """A plain-host PinPool recording every allocation (tag, bytes) in order."""

    LOG: list = []

    def __init__(self, *a, **kw):
        kw["pin"] = False
        super().__init__(*a, **kw)

    def alloc_counted(self, shape, dtype, tag="buf", pin=None):
        t, kind = super().alloc_counted(shape, dtype, tag, pin)
        _RecPool.LOG.append((str(tag), int(t.numel()) * int(t.element_size())))
        return t, kind


def _family_sequence(z):
    """The family's four host-copy statements in kit call order on ONE rank (P=1 objects suffice: placement only)."""
    p = T.ShardPark(z.clone(), park=True, force_host=True, name="z_init")
    p.block(0, 2); p.release()
    t = z.clone()
    ps = ParkedStorage(t, name="z_shard (template pair stack)")
    ps.restore()
    plan = heads.ZTrunkPlan(z.clone(), passes=1, free=False, park=True)
    plan.park_now(); plan.begin(0); plan.end(0)


def test_defaults_unchanged_same_allocations_no_lease(monkeypatch):
    """ROWPAIR_HOST_SLAB unset: every park builds its PRIVATE pool through park_pool exactly as 0.5.212 did (same tags, same bytes, same order), no
    HostSlabLease exists, census host_slab=private; =lease: ONE slab allocation serves all three parks."""
    z = _z(R=8, N=12, C=4)
    n = z.numel() * 4
    made = []

    def fake_park_pool(*, pin=True, strict=None, pin_max_bytes=None):
        made.append((bool(pin), strict, pin_max_bytes))
        return _RecPool(lever=T.LEVER)
    monkeypatch.setattr(T, "park_pool", fake_park_pool)
    _RecPool.LOG = []
    _family_sequence(z)
    tags = [t for t, _ in _RecPool.LOG]
    assert tags == ["ShardPark(z_init)[0]", "ParkedStorage(z_shard (template pair stack))[0]", "ParkedStorage(z_trunk (confidence))[0]"], tags   # one chunk each at this size
    assert all(b == n for _, b in _RecPool.LOG) and len(made) == 3 and T.host_slab_leases() == {} and dict(SCHED()).get("host_slab") == "private"
    # the same sequence under the lease: one slab, three leases, zero private pools for the parks
    monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
    _RecPool.LOG, made[:] = [], []
    _family_sequence(z)
    assert [t for t, _ in _RecPool.LOG] == ["host_slab[0]"] and len(made) == 1, _RecPool.LOG     # the lease's slab pool is the only park_pool call
    (lease,) = T.host_slab_leases().values()
    assert lease.n_leases == 3 and lease.n_second == 0 and lease.free and dict(SCHED()).get("host_slab") == "lease"


def test_pinned_shrink_drops_a_free_slab_keeps_a_held_one(monkeypatch):
    monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
    z = _z(R=8, N=12, C=4)
    plan = heads.ZTrunkPlan(z.clone(), passes=1, free=False, park=True)
    plan.park_now()
    lease = T.host_slab_lease(z.device, pin=False)
    said = []
    rec = heads.pinned_shrink(said.append, plan=plan)
    assert "slab_gib" not in rec and lease.slab is not None and not lease.free                     # held by the park: kept
    plan.begin(0); plan.retire(0)
    rec2 = heads.pinned_shrink(said.append, plan=plan)
    assert "slab_gib" in rec2 and lease.slab is None and "leased host slab (free) returned" in said[-1]
    plan.end(0)
    monkeypatch.delenv("ROWPAIR_HOST_SLAB")
    said2 = []
    heads.pinned_shrink(said2.append)
    assert "leased host slab" not in said2[-1]                                                     # the default line is unchanged


# ============================================================================================================ census hygiene (0.5.217.1)
def test_census_values_are_single_tokens(monkeypatch):
    """Every rowpair census VALUE recorded on a synthetic P=2 run through the paths (recompute z_init + leased slab in the trunk driver,
    template z park, tri-mult multi-pass mirror leasing / second-while-held, z_trunk park -> retire -> mirror leases, pinned_shrink's slab drop)
    and every other word those runs reach on CPU is ONE token: ``re.fullmatch(r'\\S+', str(value))`` — a kit's LEVER-line writer refuses
    values with blanks by contract (the holder names 'ParkedStorage(z_trunk (confidence))' broke it; census_token now maps them)."""
    import re
    from opt_core.mem.rowpair.evidence import reset_schedule
    monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
    monkeypatch.setenv("ROWPAIR_PARK_ZINIT", "recompute")
    N, C, B = 40, 4, 8
    sy = Synth(N, C)
    g = torch.Generator().manual_seed(5)
    mod = ExactTriMul(8, 4, g)
    zz, mask = _exact_inputs(24, 8, g, torch.float32)

    def entry(rank, P):
        torch.set_num_threads(1)
        reset_schedule()
        lay = Layout(N, P, rank, B)
        torch.manual_seed(7)
        T.run_trunk_sharded(lay, n_cycles=2, init_rows_fn=sy.rows_fn, init_like=sy.a_i, recycle_update_fn=sy.update, s_init=sy.s_input,
                            single_recycle_fn=lambda s, c: sy.single(s), template_fn=lambda z, c: z + sy.template(z),
                            pairstack_fn=lambda s, z, c: sy.pairstack_rows(s, z), init_rows=5, recycle_rows=6, guard="off", gather="none")
        # template-stage z park on the lease + a mirror-class second holder while it is held, then the tri-mult multi-pass mirror on the lease
        lay2 = Layout(24, P, rank, 4)
        zs = zz[lay2.r0:lay2.r1].clone().contiguous()
        park = ParkedStorage(zs.clone(), name="z_shard (template pair stack)")
        second = T.host_alloc(T.host_slab_lease(zs.device, pin=False), zs, "trimul row mirror")[0]
        second.release(); park.restore()
        st = TM.ContractStats()
        TM.trimul_update_(mod.fns(), zs, mask[lay2.r0:lay2.r1].contiguous(), lay2, outgoing=True, stats=st, inplace_chunk=48, RB=24, grid="stock", RA=5)
        # confidence: z_trunk park -> retire on the last pass -> the seam shrink drops the free slab
        plan = heads.ZTrunkPlan(zs.clone(), passes=1, free=False, park=True)
        plan.park_now(); plan.begin(0); plan.retire(0)
        heads.pinned_shrink(lambda m: None, plan=plan)
        plan.end(0)
        return dict(SCHED())

    scheds = run_ranks(2, entry)
    seen = set()
    for r, sched in enumerate(scheds):
        assert sched.get("host_slab") == "lease" and sched.get("zinit_park") == "recompute" and sched.get("ztrunk_retired") == "pass0@embed", (r, sched)
        assert "host_slab_second" in sched and "host_slab_holder" in sched and "trimul_ablock_host_slab" in sched, sorted(sched)
        bad = {k: v for k, v in sched.items() if not re.fullmatch(r"\S+", str(v))}
        assert not bad, f"rank {r}: census values with blanks: {bad}"
        seen.update(sched)
    assert {"host_slab_gib", "host_slab_leases", "zinit_recompute_calls", "pool_shrink", "park_z_init"} <= seen, sorted(seen)
    assert T.census_token("ParkedStorage(z_trunk (confidence))") == "ParkedStorage:z_trunk.confidence" and T.census_token("trimul row mirror") == "trimul_row_mirror"
