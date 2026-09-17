"""opt_core.mem.rowpair.heads.ZTrunkPlan — the placement of the trunk pair shard across the confidence stage's passes (``inplace`` |
``parked`` | ``resident``), the ONE statement adapters bind. Held here (CPU, threaded ranks P in {2, 3} — ``opt_core.testing.run_ranks`` — and
in-process checks):

* numerics are placement only: the pair input ``embed_rows`` builds is bit-identical under the three forms, at S == 1 and S == 2 samples per
  pass, one pass and a 3-pass sample loop (``torch.equal`` against the resident form AND against the dense statement on rank 0's rows);
* the per-pass decision words: ``inplace`` chosen only at (one sample-equivalent, last use, contiguous); declined BY NAME at S > 1
  (``free_declined:samples``), before the last pass (``free_declined:not_last``), on a non-contiguous shard; ``parked`` declined by name on a shard
  that does not own its storage (``not_parked:shared_storage``); ``resident`` when both levers are off; explicit arguments win over
  ``ROWPAIR_FREE_ZTRUNK`` / ``ROWPAIR_CONF_PARK_ZTRUNK``; the words land in the schedule census (``conf_ztrunk*``);
* the parked form: the device storage is released INSIDE ``begin`` — before ``embed_rows`` allocates the pass's output (``untyped_storage().nbytes()
  == 0`` when ``fn`` first runs); ``zrows`` serves rows equal to the resident rows; ``end(i, device_needed_next=True)`` re-grows the SAME storage
  (the caller's views valid, contents equal); a park stays live across passes that do not need the device tensor; after the last pass the host
  copy is dropped and the storage stays released (``consumed``); a pageable answer is NAMED (``parked:host_pageable:<kind>``); a strict pool's
  refusal is a refusal by name;
* ``template.ParkedStorage`` generalised in place: ``zrows`` / ``shape`` / ``dtype`` / ``device`` / ``dim()`` / ``where`` / ``drop()`` / ``pool=``,
  the default path unchanged (``test_rowpair_template_043.py`` holds the template driver).
"""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import dist as D  # noqa: E402
from opt_core.mem.rowpair import evidence, heads  # noqa: E402
from opt_core.mem.rowpair.shard import shard_rows  # noqa: E402
from opt_core.mem.rowpair.template import ParkedStorage  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402

C_Z, C_S, NB = 8, 5, 7


# ================================================================================================================ the synthetic statement
def _synthetic(N: int, S: int, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    rn = lambda *shape, scale=1.0: (torch.randn(shape, generator=g) * scale).float()  # noqa: E731
    return {"z": rn(1, N, N, C_Z, scale=0.5), "s": rn(N, C_S), "x": rn(S, N, 3, scale=6.0), "W_i": rn(C_S, C_Z, scale=0.3),
            "W_j": rn(C_S, C_Z, scale=0.3), "W_d": rn(NB, C_Z, scale=0.3)}


def _bins(device):
    b = torch.linspace(3.0, 21.0, NB, device=device)
    sq = b ** 2
    return sq, torch.cat([sq[1:], sq.new_tensor([1e9])])


def _embed_fn(d, samples):
    """The confidence pair input for GLOBAL rows g0:g1 of the block (an AF3-family ``embed_zij``): zblk [1, w, N, C] -> [k, w, N, C] for the
    samples ``samples`` (a slice of S). ``calls`` counts invocations; ``probe`` runs at the first call (the storage-released assertion)."""
    state = {"calls": 0, "probe": None}

    def fn(zblk, g0, g1):
        if state["calls"] == 0 and state["probe"] is not None:
            state["probe"]()
        state["calls"] += 1
        s, x = d["s"], d["x"][samples]
        zij = zblk + (s @ d["W_i"])[None, g0:g1, None, :] + (s @ d["W_j"])[None, None, :, :]
        sq, upper = _bins(zblk.device)
        d2 = torch.sum((x[:, g0:g1, None, :] - x[:, None, :, :]) ** 2, dim=-1, keepdim=True)
        oh = ((d2 > sq) * (d2 < upper)).to(x.dtype)
        return zij + oh @ d["W_d"]
    fn.state = state
    return fn


def _dense(d, samples):
    """The dense statement on the whole z (rank 0's reference)."""
    N = int(d["z"].shape[-2])
    return _embed_fn(d, samples)(d["z"], 0, N)


# ================================================================================================================ the sharded entry (every rank)
def _entry(rank, P, N, S, k, free, park, B, rows, needed_between):
    """One confidence stage of ceil(S / k) passes on this rank's rows under (free, park); returns per-pass outputs + facts."""
    lay = D.Layout.auto(N, P, rank) if not B else D.Layout.checked(N, P, rank, B)
    d = _synthetic(N, S)
    zt = shard_rows(d["z"], lay, dim=-3).clone()[0]                       # [R, N, C] with its OWN storage (the adapter's z2)
    ref_rows = zt.clone()
    view = zt[2:5]                                                          # a view the "engine" holds: valid again after restore
    chunks = [slice(a, min(S, a + k)) for a in range(0, S, k)]
    plan = heads.ZTrunkPlan(zt, passes=len(chunks), free=free, park=park, name="z_trunk")
    out = {"rank": rank, "P": P, "R": lay.R, "r0": lay.r0, "words": None, "zc": [], "checks": {}, "decide": []}
    for i, ch in enumerate(chunks):
        kk = ch.stop - ch.start
        if i == 0 or needed_between:
            # a statement that reads the device shard whole before the pass (diffusion of pass i): it must see the trunk rows
            out["checks"][f"pre_pass{i}_rows_intact"] = bool(zt.untyped_storage().nbytes() == ref_rows.numel() * 4 and torch.equal(zt, ref_rows))
        word = plan.decide(i, (kk,))
        out["decide"].append(word)
        fn = _embed_fn(d, ch)
        released_at_fn = {}

        def probe(zt=zt, rec=released_at_fn):
            rec["nbytes"] = int(zt.untyped_storage().nbytes())
        fn.state["probe"] = probe
        src, inplace = plan.begin(i, (kk,))
        out["checks"][f"pass{i}_begin_released"] = (int(zt.untyped_storage().nbytes()) == 0) == word.startswith("parked")
        out["checks"][f"pass{i}_src_kind"] = (src is zt) == (not word.startswith("parked"))
        zc = heads.embed_rows(fn, src, lay, rows=rows, lead_out=(kk,), inplace=inplace, bins=NB)
        out["checks"][f"pass{i}_released_before_alloc"] = (released_at_fn.get("nbytes") == 0) == word.startswith("parked")
        shares = zt.untyped_storage().nbytes() > 0 and zc.untyped_storage().data_ptr() == zt.untyped_storage().data_ptr()
        out["checks"][f"pass{i}_inplace_storage"] = bool(shares) == (word == "inplace")      # (a released storage's address may be handed to the new output: compare live storages only)
        out["checks"][f"pass{i}_shape"] = tuple(zc.shape) == (kk, lay.R, N, C_Z)
        out["zc"].append(zc.clone())
        del zc
        last = i == len(chunks) - 1
        plan.end(i, device_needed_next=(not last) and needed_between)
        if not last and needed_between:
            # the SAME storage object re-grown (its address may move): the tensor and every view of it read the trunk rows again
            out["checks"][f"pass{i}_restored_same_storage"] = bool(zt.untyped_storage().nbytes() == ref_rows.numel() * 4 and torch.equal(zt, ref_rows)
                                                                  and view.untyped_storage().data_ptr() == zt.untyped_storage().data_ptr()
                                                                  and torch.equal(view, ref_rows[2:5]))
    out["words"] = list(plan.words)
    out["consumed"] = bool(plan.consumed)
    out["closed"] = bool(plan.closed)
    out["host_gib"] = float(plan.host_gib)
    out["final_nbytes"] = int(zt.untyped_storage().nbytes())
    return out


FORMS = [(False, False), (True, False), (False, True), (True, True)]      # (free, park)


def _run(P, N, S, k, B, rows, needed_between):
    """Every (free, park) form on P threaded ranks; the schedule census (process-global, identical words on every rank) is read on the main
    thread after the ranks joined and attached to each rank's facts as ``sched``."""
    results = {}
    for fp in FORMS:
        evidence.reset_schedule()
        res = run_ranks(P, _entry, N, S, k, fp[0], fp[1], B, rows, needed_between)
        sched = {k_: v for k_, v in evidence.schedule().items() if str(k_).startswith("conf_ztrunk")}
        for o in res:
            o["sched"] = dict(sched)
        results[fp] = res
    return results


def _assert_identical(results, N, S, k):
    """Every form's per-pass pair input equals the resident form's bit for bit on every rank, and rank r's rows equal the dense statement's."""
    d = _synthetic(N, S)
    base = results[(False, False)]
    chunks = [slice(a, min(S, a + k)) for a in range(0, S, k)]
    for fp, res in results.items():
        for r, o in enumerate(res):
            assert all(o["checks"].values()), (fp, r, {c: v for c, v in o["checks"].items() if not v}, o["words"])
            for i, ch in enumerate(chunks):
                assert torch.equal(o["zc"][i], base[r]["zc"][i]), (fp, r, i, o["words"])
                dense = _dense(d, ch)[:, o["r0"]:o["r0"] + o["R"]]
                assert torch.equal(o["zc"][i], dense), (fp, r, i, "vs dense")


# ================================================================================================================ threaded ranks: numerics + words
@pytest.mark.parametrize("P,N,B,rows", [(2, 64, 16, 8), (3, 61, 8, 8), (2, 48, 0, None)])
def test_one_pass_S1_three_forms_identical_and_words(P, N, B, rows):
    res = _run(P, N, 1, 1, B, rows, needed_between=False)
    _assert_identical(res, N, 1, 1)
    for r in range(P):
        assert res[(False, False)][r]["words"] == ["resident"] and not res[(False, False)][r]["consumed"]
        assert res[(True, False)][r]["words"] == ["inplace"] and res[(True, False)][r]["consumed"]
        assert res[(False, True)][r]["words"] == ["parked:host"] and res[(False, True)][r]["consumed"] and res[(False, True)][r]["final_nbytes"] == 0
        assert res[(True, True)][r]["words"] == ["inplace"]                              # both set, one sample, last use: in place wins (no copies)
        assert res[(False, True)][r]["host_gib"] > 0 and res[(True, False)][r]["host_gib"] == 0.0
        assert all(o["closed"] for o in (res[fp][r] for fp in FORMS))


@pytest.mark.parametrize("P,N,B,rows", [(2, 64, 16, 8), (3, 61, 8, None)])
def test_one_pass_S2_inplace_declined_by_name_park_engages(P, N, B, rows):
    res = _run(P, N, 2, 2, B, rows, needed_between=False)
    _assert_identical(res, N, 2, 2)
    for r in range(P):
        assert res[(True, False)][r]["words"] == ["resident:free_declined:samples"] and not res[(True, False)][r]["consumed"]
        assert res[(False, True)][r]["words"] == ["parked:host"]
        assert res[(True, True)][r]["words"] == ["parked:host"]                           # S > 1: in place impossible, the park keeps ONE shard-equivalent off the device
        assert res[(True, True)][r]["final_nbytes"] == 0 and res[(True, True)][r]["consumed"]


@pytest.mark.parametrize("S", [2, 3])
@pytest.mark.parametrize("needed_between", [True, False])
def test_sample_loop_per_sample_passes(S, needed_between):
    # k = 1: S passes of one sample each; between passes a statement reads the device shard whole iff needed_between
    P, N, k = 2, 64, 1
    res = _run(P, N, S, k, 16, 8, needed_between)
    _assert_identical(res, N, S, k)
    pre = S - 1                                                             # the passes before the last
    for r in range(P):
        assert res[(True, False)][r]["words"] == ["resident:free_declined:not_last"] * pre + ["inplace"]
        assert res[(True, False)][r]["decide"] == res[(True, False)][r]["words"]
        if needed_between:
            # restored after every pass but the last (a new copy out per pass); the device tensor HOLDS the rows again at the last pass:
            # in place when free (no live park), parked otherwise
            assert res[(True, True)][r]["words"] == ["parked:host"] * pre + ["inplace"]
            assert res[(False, True)][r]["words"] == ["parked:host"] * S and res[(False, True)][r]["final_nbytes"] == 0
        else:
            # the park stays live across the passes (ONE copy out): the last pass stays 'parked' even with free set — a fresh [k, R, N, C]
            # allocation served from the host copy, never restore-then-inplace (an H2D of a whole shard for zero device benefit)
            assert res[(True, True)][r]["words"] == ["parked:host"] * S and res[(True, True)][r]["consumed"]
            assert res[(True, True)][r]["decide"] == ["parked"] * S
            assert res[(False, True)][r]["final_nbytes"] == 0 and res[(True, True)][r]["final_nbytes"] == 0
        sched = res[(True, True)][r]["sched"]
        assert sched["conf_ztrunk"] == ",".join(res[(True, True)][r]["words"]) and sched["conf_ztrunk_passes"] == S
        assert sched["conf_ztrunk_free"] == 1 and sched["conf_ztrunk_park"] == 1 and sched["conf_ztrunk_host_gib"] >= 0.0


# ================================================================================================================ in-process: decision words, env, refusals
def _zt(R=6, N=10):
    return torch.randn(R, N, C_Z)


def test_decide_words_and_env(monkeypatch):
    monkeypatch.delenv("ROWPAIR_FREE_ZTRUNK", raising=False)
    monkeypatch.delenv("ROWPAIR_CONF_PARK_ZTRUNK", raising=False)
    z = _zt()
    p = heads.ZTrunkPlan(z, passes=2)
    assert (p.free, p.park) == (False, False) and p.decide(0) == "resident" and p.decide(1, (2,)) == "resident"
    monkeypatch.setenv("ROWPAIR_FREE_ZTRUNK", "1")
    monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", "0")
    p = heads.ZTrunkPlan(z, passes=2)
    assert (p.free, p.park) == (True, False)
    assert p.decide(0) == "resident:free_declined:not_last" and p.decide(1) == "inplace" and p.decide(1, (1,)) == "inplace"
    assert p.decide(1, (2,)) == "resident:free_declined:samples" and p.decide(0, last_use=True) == "inplace" and p.decide(1, last_use=False) == "resident:free_declined:not_last"
    zt = _zt().transpose(-1, -2).contiguous().transpose(-1, -2)               # non-contiguous, same shape
    assert heads.ZTrunkPlan(zt, passes=1).decide(0) == "resident:free_declined:not_contiguous"
    monkeypatch.setenv("ROWPAIR_CONF_PARK_ZTRUNK", "true")
    p = heads.ZTrunkPlan(z, passes=2)
    assert (p.free, p.park) == (True, True) and p.decide(0) == "parked" and p.decide(1) == "inplace" and p.decide(1, (3,)) == "parked"
    big = torch.randn(2, 6, 10, C_Z)
    assert heads.ZTrunkPlan(big[1], passes=1, free=False).decide(0) == "resident:not_parked:shared_storage"       # a view into a larger storage
    assert heads.ZTrunkPlan(big[1], passes=1).decide(0, (2,)) == "resident:free_declined:samples+not_parked:shared_storage"
    assert heads.ZTrunkPlan(z, passes=1, free=False, park=False).decide(0) == "resident"                            # explicit arguments win over env
    sched = evidence.schedule()
    assert sched["conf_ztrunk"] == "-" and sched["conf_ztrunk_free"] == 0 and sched["conf_ztrunk_park"] == 0 and sched["conf_ztrunk_passes"] == 1


def test_parked_pass_lifecycle_in_process():
    z = _zt()
    ref, view = z.clone(), z[1:3]
    logs = []
    p = heads.ZTrunkPlan(z, passes=3, free=False, park=True, log=logs.append)
    src, inplace = p.begin(0)
    assert isinstance(src, ParkedStorage) and src is p.parked and not inplace and z.untyped_storage().nbytes() == 0 and p.consumed
    assert src.where == "host" and p.words[0] == "parked:host" and tuple(src.shape) == tuple(z.shape) and src.dtype == z.dtype and src.dim() == 3
    assert torch.equal(src.zrows(2, 5), ref[2:5]) and torch.equal(src.zrows(0, 6), ref)
    p.end(0)                                                                # not needed on the device: the park stays live
    assert p.parked is src and z.untyped_storage().nbytes() == 0
    src1, _ = p.begin(1)
    assert src1 is src and p.words[1] == "parked:host"                     # no second copy out
    p.end(1, device_needed_next=True)                                       # restored into the SAME storage
    assert p.parked is None and not p.consumed and z.untyped_storage().nbytes() == ref.numel() * 4
    assert torch.equal(z, ref) and torch.equal(view, ref[1:3]) and view.untyped_storage().data_ptr() == z.untyped_storage().data_ptr()
    src2, _ = p.begin(2)
    assert src2 is not src and isinstance(src2, ParkedStorage) and z.untyped_storage().nbytes() == 0
    p.end(2)                                                                # last pass: host dropped, storage stays released
    assert p.closed and p.consumed and p.parked is None and z.untyped_storage().nbytes() == 0 and src2.dropped and src2.host is None
    with pytest.raises(RowpairRefused, match="dropped"):
        src2.zrows(0, 1)
    with pytest.raises(RowpairRefused, match="closed"):
        p.begin(0)
    rec = p.record()
    assert rec["words"] == ["parked:host"] * 3 and rec["forms"] == ["parked"] * 3 and rec["consumed"] and rec["host_bytes"] == ref.numel() * 4
    assert any("storage released" in m for m in logs) and any("host copy dropped" in m for m in logs)


def test_refusals_by_name():
    z = _zt()
    with pytest.raises(RowpairRefused, match="passes=0"):
        heads.ZTrunkPlan(z, passes=0)
    with pytest.raises(RowpairRefused, match="TENSOR"):
        heads.ZTrunkPlan(ParkedStorage(_zt()), passes=1)
    with pytest.raises(RowpairRefused, match="TENSOR"):
        heads.ZTrunkPlan(torch.randn(4, 4), passes=1)
    p = heads.ZTrunkPlan(z, passes=2, free=True, park=False)
    with pytest.raises(RowpairRefused, match="outside"):
        p.decide(2)
    with pytest.raises(RowpairRefused, match="not begun"):
        p.end(0)
    p.begin(0)
    with pytest.raises(RowpairRefused, match="already begun"):
        p.begin(0)
    p.end(0)
    src, inplace = p.begin(1)
    assert inplace and src is z and p.consumed
    with pytest.raises(RowpairRefused, match="IN PLACE"):
        p.end(1, device_needed_next=True)                                   # the rows were overwritten: cannot be needed afterwards
    p.end(1)
    assert p.closed
    # a last pass that keeps the rows (last_use=False) may restore-or-keep: resident here, no refusal
    q = heads.ZTrunkPlan(_zt(), passes=1, free=True, park=False)
    src, inplace = q.begin(0, last_use=False)
    assert not inplace and q.words == ["resident:free_declined:not_last"]
    q.end(0, device_needed_next=True)
    assert not q.consumed


class _FakePool(object):
    """A PinPool double answering every request PAGEABLE with a named kind (the MEMLOCK-capped box), counting releases."""

    def __init__(self, kind="pin_alloc_failed"):
        self.kind, self.released, self.allocs = kind, 0, 0

    def alloc_counted(self, shape, dtype, tag="buf", pin=None):
        self.allocs += 1
        return torch.empty(tuple(shape), dtype=dtype), self.kind

    def release(self, t):
        self.released += 1


def test_pageable_answer_is_named_and_pool_release():
    pool = _FakePool()
    z = _zt()
    ref = z.clone()
    p = heads.ZTrunkPlan(z, passes=1, free=False, park=True, pool=pool)
    src, _ = p.begin(0)
    assert p.words == ["parked:host_pageable:pin_alloc_failed"] and src.fallback == "pin_alloc_failed" and src.where == "host_pageable:pin_alloc_failed"
    assert evidence.schedule()["conf_ztrunk"] == "parked:host_pageable:pin_alloc_failed"
    assert torch.equal(src.zrows(1, 4), ref[1:4])
    p.end(0, device_needed_next=True)                                       # restore returns the pool's bytes
    assert pool.allocs == 1 and pool.released == 1 and torch.equal(z, ref) and not p.consumed


def test_strict_pool_refusal_is_by_name():
    from opt_core.mem.torch_hostpair import PinPool
    z = _zt()
    strict = PinPool(max_bytes=16, pin=False, pageable=False, lever="test")   # budget below the shard: refused, not pageable
    p = heads.ZTrunkPlan(z, passes=1, free=False, park=True, pool=strict)
    with pytest.raises(RowpairRefused, match="pin_budget_exceeded"):
        p.begin(0)
    assert z.untyped_storage().nbytes() == z.numel() * 4                    # nothing released on a refused park
    lenient = PinPool(max_bytes=16, pin=False, pageable=True, lever="test")  # the same budget, pageable answer: named
    q = heads.ZTrunkPlan(_zt(), passes=1, free=False, park=True, pool=lenient)
    src, _ = q.begin(0)
    assert q.words == ["parked:host_pageable:pin_budget_exceeded"] and src.fallback == "pin_budget_exceeded"
    q.end(0)


def test_park_pool_is_the_one_constructor(monkeypatch):
    from opt_core.mem.rowpair import trunk
    from opt_core.mem.torch_hostpair import GiB, PinPool
    monkeypatch.setenv("ROWPAIR_PARK_PIN_MAX_GB", "0.5")
    monkeypatch.setenv("ROWPAIR_PARK_STRICT", "1")
    pool = trunk.park_pool(pin=False)
    assert isinstance(pool, PinPool) and pool.max_bytes == int(0.5 * GiB) and pool.pageable is False and trunk.park_strict() is True
    monkeypatch.setenv("ROWPAIR_PARK_STRICT", "0")
    assert trunk.park_pool(pin=False).pageable is True and trunk.park_pool(pin=False, strict=True).pageable is False
    assert trunk.park_pool(pin=False, pin_max_bytes=7).max_bytes == 7


# ================================================================================================================ ParkedStorage generalised in place
def test_parked_storage_passthroughs_and_drop():
    z = torch.randn(1, 12, 9, 4)
    ref = z.clone()
    p = ParkedStorage(z, name="z")
    assert p.ok and p.where == "host" and p.fallback is None and not p.pinned and tuple(p.shape) == (1, 12, 9, 4) and p.dtype == z.dtype
    assert p.device == z.device and p.dim() == 4 and torch.equal(p.zrows(3, 7), ref[:, 3:7]) and p.how == "parked pageable"
    p.drop()
    assert p.dropped and p.host is None and z.untyped_storage().nbytes() == 0 and p.where == "dropped"
    p.drop()                                                                # idempotent
    with pytest.raises(RowpairRefused, match="dropped"):
        p.restore()
    q = ParkedStorage(ref.clone()[:, 1:], name="view")                     # not parked: passthroughs still read the live tensor
    assert not q.ok and q.where == "device" and tuple(q.shape) == (1, 11, 9, 4) and torch.equal(q.zrows(0, 2), ref[:, 1:3])
    q.drop()                                                                # no-op when not parked
    assert not q.dropped
    pool = _FakePool(kind="host_ram_insufficient")
    w = ref.clone()
    r = ParkedStorage(w, name="w", pool=pool)
    assert r.ok and r.where == "host_pageable:host_ram_insufficient" and r.fallback == "host_ram_insufficient"
    back = r.restore()
    assert back is w and torch.equal(w, ref) and pool.released == 1 and r.where == "device"


def test_embed_rows_accepts_a_park_as_the_shard_and_refuses_inplace_on_it():
    # in-process on the single-device opt-in (allow_unsharded): a park stands in for z_shard; inplace on a park is refused by name
    N, S = 12, 2
    d = _synthetic(N, S)
    lay = D.Layout(N, 1, 0, B=1)
    z = d["z"].clone()[0]                                                    # [N, N, C]
    dense = _dense(d, slice(0, S))
    park = ParkedStorage(z, name="z")
    got = heads.embed_rows(_embed_fn(d, slice(0, S)), park, lay, rows=5, lead_out=(S,), allow_unsharded=True, bins=NB)
    assert torch.equal(got, dense)
    with pytest.raises(RowpairRefused, match="parked"):
        heads.embed_rows(_embed_fn(d, slice(0, S)), park, lay, rows=5, lead_out=(S,), inplace=True, allow_unsharded=True, bins=NB)
    park.restore()
    assert torch.equal(z, d["z"][0])


def test_out_dtype_rule_inplace_needs_the_shard_dtype():
    # engines whose trunk shard is bf16 under autocast while the confidence pair input is fp32 (a stock .float()): in place is declined BY NAME
    zb = _zt().to(torch.bfloat16)
    p = heads.ZTrunkPlan(zb, passes=1, free=True, park=False)
    assert p.decide(0) == "inplace" and p.decide(0, out_dtype=torch.bfloat16) == "inplace"
    assert p.decide(0, out_dtype=torch.float32) == "resident:free_declined:dtype"
    q = heads.ZTrunkPlan(_zt().to(torch.bfloat16), passes=1, free=True, park=True)
    assert q.decide(0, out_dtype=torch.float32) == "parked"                    # falls to the park when that lever is set
    src, inplace = p.begin(0, out_dtype=torch.float32)
    assert src is zb and not inplace and p.words == ["resident:free_declined:dtype"] and not p.consumed
    p.end(0)
    # embed_rows holds the same rule on its own (inplace + another out_dtype refused by name; out_dtype sizes the new tensor otherwise)
    N, S = 12, 1
    d = _synthetic(N, S)
    lay = D.Layout(N, 1, 0, B=1)
    z = d["z"].clone()[0].to(torch.bfloat16)
    fn32 = lambda zblk, g0, g1: _embed_fn(d, slice(0, S))(zblk.float(), g0, g1)   # noqa: E731 — the engine's .float() statement
    with pytest.raises(RowpairRefused, match="out_dtype"):
        heads.embed_rows(fn32, z, lay, rows=5, lead_out=(S,), inplace=True, out_dtype=torch.float32, allow_unsharded=True, bins=NB)
    got = heads.embed_rows(fn32, z, lay, rows=5, lead_out=(S,), out_dtype=torch.float32, allow_unsharded=True, bins=NB)
    assert got.dtype == torch.float32 and tuple(got.shape) == (1, N, N, C_Z)
    assert torch.equal(got, _embed_fn(d, slice(0, S))(z.float()[None], 0, N))
    with pytest.raises(RowpairRefused, match="expected"):
        heads.embed_rows(fn32, z, lay, rows=5, lead_out=(S,), out=torch.empty(1, N, N, C_Z, dtype=torch.bfloat16), out_dtype=torch.float32,
                         allow_unsharded=True, bins=NB)


def _cond_entry(rank, P, N, parked, rows, B):
    from opt_core.mem.rowpair import diffusion as DF
    lay = D.Layout.auto(N, P, rank) if not B else D.Layout.checked(N, P, rank, B)
    d = _synthetic(N, 1)
    zt = shard_rows(d["z"], lay, dim=-3).clone()                             # [1, R, N, C], own storage
    ref = zt.clone()
    W = d["W_i"].t()[:, :5].contiguous()                                     # [C, 5]: a per-(i, j) linear = the conditioning embedder's shape
    embed = lambda zr, g0, g1: zr @ W + float(g0)                            # noqa: E731 — reads the block's global rows too
    trans = (lambda x, g0, g1: x * 0.5,)
    src = ParkedStorage(zt, name="z_trunk (rollout)") if parked else zt
    if parked:
        assert src.ok and zt.untyped_storage().nbytes() == 0
    out = DF.pair_cond_rows(embed, src, lay, c_out=5, rows=rows, transitions=trans)
    if parked:
        assert zt.untyped_storage().nbytes() == 0                            # released through the whole statement
        src.restore()
        assert torch.equal(zt, ref)
    return out


@pytest.mark.parametrize("P,N,rows,B", [(2, 48, 8, 0), (3, 61, None, 8)])
def test_pair_cond_rows_accepts_a_parked_shard_bitwise(P, N, rows, B):
    # the diffusion conditioning reads the trunk shard through .zrows when it is parked (an adapter parks z_trunk at roll-out entry): same bits
    dense = run_ranks(P, _cond_entry, N, False, rows, B)
    parked = run_ranks(P, _cond_entry, N, True, rows, B)
    for r in range(P):
        assert torch.equal(dense[r], parked[r]) and dense[r].dtype == parked[r].dtype and tuple(dense[r].shape) == tuple(parked[r].shape)


def test_foreign_storage_is_neither_parked_nor_embedded_in_place():
    # a hybrid line's rank 0: the trunk shard is a zero-copy DLPack view of a JAX buffer — its storage is NOT resizable although the
    # ownership test (contiguous, offset 0, nbytes) passes; torch.from_numpy is the CPU stand-in
    import numpy as np
    from opt_core.mem.rowpair.shard import owns_whole_storage, release_storage_, storage_resizable
    zf = torch.from_numpy(np.random.RandomState(0).randn(6, 10, C_Z).astype(np.float32))
    ref = zf.clone()
    assert owns_whole_storage(zf) and not storage_resizable(zf) and storage_resizable(_zt())
    p = heads.ZTrunkPlan(zf, passes=1, free=True, park=True)
    assert p.decide(0) == "resident:free_declined:foreign_storage+not_parked:storage_not_resizable"
    assert heads.ZTrunkPlan(zf, passes=1, free=False, park=True).decide(0) == "resident:not_parked:storage_not_resizable"
    src, inplace = p.begin(0)
    assert src is zf and not inplace and zf.untyped_storage().nbytes() == ref.numel() * 4 and torch.equal(zf, ref) and not p.consumed
    p.end(0)
    ps = ParkedStorage(zf, name="foreign")
    assert not ps.ok and ps.why == "storage_not_resizable" and "not resizable" in ps.how and torch.equal(ps.zrows(0, 2), ref[0:2])
    assert ParkedStorage(ref.clone()[1:], name="v").why == "shared_storage"
    with pytest.raises(RowpairRefused, match="not resizable"):
        release_storage_(zf)
    lay = D.Layout(10, 1, 0, B=1)
    with pytest.raises(RowpairRefused, match="foreign storage"):
        heads.embed_rows(lambda zb, g0, g1: zb, torch.from_numpy(np.zeros((10, 10, C_Z), np.float32)), lay, rows=5, inplace=True, allow_unsharded=True)


def test_shardpark_census_words():
    from opt_core.mem.rowpair.trunk import ShardPark
    z = torch.randn(1, 6, 10, C_Z)
    sp = ShardPark(z, park=False, name="z_init")
    assert sp.census_words("park_z_init") == {"park_z_init": "device", "park_z_init_gib": round(z.numel() * 4 / 2 ** 30, 3), "park_z_init_release": "resident"}
    hp = ShardPark(z.clone(), park=True, name="z_init", force_host=True)             # a CPU shard parked by force: host copy, device copy kept
    w = hp.census_words("park_z_init")
    assert w["park_z_init"] in ("host", "host_pinned") and w["park_z_init_release"] == "kept"
    assert set(w) == {"park_z_init", "park_z_init_gib", "park_z_init_release", "park_z_init_chunks", "park_z_init_pinned_gib"}   # 0.5.18.9: the host copy's chunk count / page-locked GiB
    assert w["park_z_init_chunks"] == 1 and w["park_z_init_pinned_gib"] == 0.0                                                 # one small chunk, not page-locked (CPU)


def test_embed_rows_refuses_a_narrowing_store_by_name_and_names_a_widening():
    N, S = 12, 1
    d = _synthetic(N, S)
    lay = D.Layout(N, 1, 0, B=1)
    zb = d["z"].clone()[0].to(torch.bfloat16)
    fn32 = lambda zblk, g0, g1: _embed_fn(d, slice(0, S))(zblk.float(), g0, g1)   # noqa: E731 — fp32 rows for a bf16 shard
    with pytest.raises(RowpairRefused, match="out_dtype"):                     # out defaults to the shard's bf16: storing fp32 rows would round them
        heads.embed_rows(fn32, zb, lay, rows=5, lead_out=(S,), allow_unsharded=True, bins=NB)
    z32 = d["z"].clone()[0]
    fnb = lambda zblk, g0, g1: _embed_fn(d, slice(0, S))(zblk, g0, g1).to(torch.bfloat16)   # noqa: E731 — bf16 rows into an fp32 pair input: exact
    evidence.reset_schedule()
    got = heads.embed_rows(fnb, z32, lay, rows=5, lead_out=(S,), allow_unsharded=True, bins=NB)
    assert got.dtype == torch.float32 and torch.equal(got, _embed_fn(d, slice(0, S))(d["z"], 0, N).to(torch.bfloat16).float())
    assert evidence.schedule()["conf_embed_dtype"] == "bfloat16->float32"


def test_logit_heads_refuse_a_parked_source_by_name():
    lay = D.Layout(10, 1, 0, B=1)
    park = ParkedStorage(torch.randn(10, 10, C_Z), name="z")
    fn = lambda zb: zb[..., :3]                                              # noqa: E731
    with pytest.raises(RowpairRefused, match="rows only"):
        list(heads.logit_rows(fn, park, lay, rows=4, allow_unsharded=True))
    with pytest.raises(RowpairRefused, match="rows only"):
        list(heads.sym_logit_rows(fn, park, lay, rows=4, allow_unsharded=True))
    park.restore()


def _rollout_entry(rank, P, N, S, park, B):
    """The reference adapter's sample-loop flow: per pass, park at ROLL-OUT ENTRY, the diffusion conditioning reads plan.source(), then the confidence pass."""
    from opt_core.mem.rowpair import diffusion as DF
    lay = D.Layout.checked(N, P, rank, B)
    d = _synthetic(N, S)
    zt = shard_rows(d["z"], lay, dim=-3).clone()[0]                          # [R, N, C]
    ref = zt.clone()
    Wc = d["W_j"].t()[:, :5].contiguous()
    embed_c = lambda zr, g0, g1: zr @ Wc + float(g0)                        # noqa: E731
    plan = heads.ZTrunkPlan(zt, passes=S, free=True, park=park)
    out = {"cond": [], "zc": [], "entry": [], "words": None, "nbytes_in_rollout": []}
    for i in range(S):
        out["entry"].append(plan.park_now())
        src = plan.source()
        out["nbytes_in_rollout"].append(int(zt.untyped_storage().nbytes()))
        out["cond"].append(DF.pair_cond_rows(embed_c, src, lay, c_out=5, rows=8).clone())
        s2, inplace = plan.begin(i, (1,))
        out["zc"].append(heads.embed_rows(_embed_fn(d, slice(i, i + 1)), s2, lay, rows=8, lead_out=(1,), inplace=inplace, bins=NB).clone())
        plan.end(i, device_needed_next=False)
    out["words"] = list(plan.words)
    out["final_nbytes"] = int(zt.untyped_storage().nbytes())
    out["ref_nbytes"] = ref.numel() * 4
    return out


@pytest.mark.parametrize("S", [1, 2])
def test_rollout_entry_park_composes_with_the_passes(S):
    P, N, B = 2, 64, 16
    res_r = run_ranks(P, _rollout_entry, N, S, False, B)                    # free only: resident roll-outs, the last pass in place
    res_p = run_ranks(P, _rollout_entry, N, S, True, B)                     # park: parked at the FIRST roll-out entry, stays parked, never restored
    for r in range(P):
        a, b = res_r[r], res_p[r]
        for i in range(S):
            assert torch.equal(a["cond"][i], b["cond"][i]) and torch.equal(a["zc"][i], b["zc"][i])
        assert a["entry"] == ["resident"] * S and a["words"] == ["resident:free_declined:not_last"] * (S - 1) + ["inplace"]
        assert a["nbytes_in_rollout"] == [a["ref_nbytes"]] * S
        assert b["entry"] == ["parked:host"] * S and b["words"] == ["parked:host"] * S       # a live park keeps serving: no restore-then-inplace
        assert b["nbytes_in_rollout"] == [0] * S and b["final_nbytes"] == 0                   # the trunk shard is OFF the device for every roll-out


def _pair_block_entry(rank, P, N, foreign):
    import numpy as np
    from opt_core.mem.rowpair import pairstack as PS
    lay = D.Layout.checked(N, P, rank, 8)
    g = torch.Generator().manual_seed(3)
    z = torch.randn(1, N, N, C_Z, generator=g)
    zl = shard_rows(z, lay, dim=-3)[0].clone()                              # [R, N, C], owning its whole (torch) storage
    if foreign:
        zl = torch.from_numpy(zl.numpy().copy())                           # a non-resizable storage holding the same rows (a torch.load'ed CPU shard's class)
    W = torch.randn(C_Z, C_Z, generator=g) / 8

    # elementwise / row-local stand-ins: the block's plumbing is under test, not the maths
    fns = PS.PairBlockFns(trimul_out=lambda z, m, lay_: z.add_(torch.tanh(z @ W)), trimul_in=lambda z, m, lay_: z.mul_(0.5),
                          triatt_start=lambda z, m, lay_: z.add_(0.25), triatt_end=lambda zT, mT, lay_: zT.add_(torch.sigmoid(zT @ W)),
                          transition=lambda z, m, lay_: z.sub_(0.125))                      # in place on the shard, as the trunk's updates are
    out, _s = PS.pair_block_(fns, zl, None, lay, end_free=True, transpose_inplace=False)
    return {"out": out.clone()}


def test_pair_block_end_free_names_a_foreign_storage_and_keeps_the_values():
    P, N = 2, 32
    evidence.reset_schedule()
    own = run_ranks(P, _pair_block_entry, N, False)
    freed_own = evidence.schedule().get("pairstack_end_free")               # the census is process-global: read on the main thread after the ranks joined
    evidence.reset_schedule()
    frn = run_ranks(P, _pair_block_entry, N, True)
    freed_frn = evidence.schedule().get("pairstack_end_free")
    for r in range(P):
        assert torch.equal(own[r]["out"], frn[r]["out"])
    assert freed_own == "applied" and freed_frn == "skipped:storage_not_resizable"


def test_import_is_light():
    """Importing the module imports no torch (the package contract): a fresh interpreter with torch masked imports it and reads the words."""
    import subprocess
    code = ("import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('masked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
            f"sys.path.insert(0, {os.path.dirname(HERE)!r})\n"
            "import opt_core.mem.rowpair.heads as H\n"
            "assert 'ZTrunkPlan' in H.__all__ and 'torch' not in sys.modules\n"
            "assert H.ENV_FREE_ZTRUNK == 'ROWPAIR_FREE_ZTRUNK' and H.ENV_CONF_PARK_ZTRUNK == 'ROWPAIR_CONF_PARK_ZTRUNK'\n"
            "assert H.ZTRUNK_FORMS == ('inplace', 'parked', 'resident')\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
