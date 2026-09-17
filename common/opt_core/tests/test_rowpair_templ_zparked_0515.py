"""The template stage's Z-PARKED order and the template->MSA seam host trim (CPU, threaded ranks).

* ``templ_order`` words: unset = classic; ``ROWPAIR_TEMPL_ORDER=zparked``; an unknown word refused by name; the argument wins over the env.
* ``ParkedStorage.write_block`` / ``read_block`` round trips (``[R, N, C]`` and ``[2, R, N, C]`` leads), refusals by name (shape / dtype / dropped /
  restored), the not-parked pass-through.
* ``template_embed_rows_zparked`` == the classic ``template_embed_rows`` BITWISE on the synthetic template stack of test_rowpair_template_0511 (P in
  2/3; de-duplicated and per-slot groups; parked / resident u slabs; add / out forms; private pools and ``ROWPAIR_HOST_SLAB=lease``); z IS off its
  storage while the unit rows and the pair stacks run; census ``templ_order=zparked``.
* ``host_trim``: never without the lease word (no call, no census word), once under ``ROWPAIR_HOST_SLAB=lease``, ``ROWPAIR_HOST_TRIM=0|1`` overrides;
  ``run_trunk_sharded`` calls it once per template stage and never without a template stage; ``DevicePeak`` answers ``kind=cpu`` on CPU.
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import dist as D  # noqa: E402
from opt_core.mem.rowpair import evidence  # noqa: E402
from opt_core.mem.rowpair import template as TM  # noqa: E402
from opt_core.mem.rowpair import trunk as T  # noqa: E402
from opt_core.mem.rowpair.shard import shard_rows  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402
from test_rowpair_template_0511 import _precursors, _dense_products, _grp_for  # noqa: E402
from test_rowpair_trunk_043 import Synth  # noqa: E402


LEVER_WORDS = ("ROWPAIR_TEMPL_ORDER", "ROWPAIR_HOST_SLAB", "ROWPAIR_HOST_TRIM", "ROWPAIR_TEMPL_PARK_U", "ROWPAIR_TEMPL_PARK_Z", "ROWPAIR_HOST_SLAB_STRICT")


@pytest.fixture(autouse=True)
def _clean():
    """Every lever word this file exercises is scrubbed BEFORE the test and put back exactly as found AFTER it (0.5.220.6 gate fix). The words are
    set ONLY through ``monkeypatch.setenv`` in the test body — the rank threads of ``run_ranks`` share the process environment, so nothing is
    written inside an entry. (0.5.220.0 wrote ``os.environ`` inside the entries behind an autouse ``monkeypatch.delenv``: a write the fixture had
    not recorded was never undone and the NEXT test's delenv restored it at teardown, so ``ROWPAIR_TEMPL_ORDER=zparked`` / ``ROWPAIR_HOST_SLAB=lease``
    leaked into every later module of the session — the gate's template_043 / trimul_ablock failures.)"""
    saved = {k: os.environ.get(k) for k in LEVER_WORDS}
    for k in LEVER_WORDS:
        os.environ.pop(k, None)
    T.release_host_slabs()
    evidence.reset_schedule()
    try:
        yield
    finally:
        T.release_host_slabs()
        leaked = {k: os.environ.get(k) for k in LEVER_WORDS if os.environ.get(k) is not None}
        for k in LEVER_WORDS:
            os.environ.pop(k, None)
            if saved[k] is not None:
                os.environ[k] = saved[k]
        assert not leaked, f"lever words left in os.environ by the test body (set them with monkeypatch.setenv): {leaked}"


# ================================================================================================================ words
def test_templ_order_words(monkeypatch):
    assert TM.templ_order() == "classic" and TM.TEMPL_ORDERS == ("classic", "zparked")
    monkeypatch.setenv("ROWPAIR_TEMPL_ORDER", "zparked")
    assert TM.templ_order() == "zparked" and TM.templ_order("classic") == "classic"       # the argument wins
    monkeypatch.setenv("ROWPAIR_TEMPL_ORDER", " Classic ")
    assert TM.templ_order() == "classic"
    monkeypatch.setenv("ROWPAIR_TEMPL_ORDER", "parkfirst")
    with pytest.raises(RowpairRefused, match="ROWPAIR_TEMPL_ORDER='parkfirst'"):
        TM.templ_order()
    with pytest.raises(RowpairRefused, match="template order='sideways'"):
        TM.templ_order("sideways")
    assert T.ENV_TEMPL_ORDER_NAME == TM.ENV_TEMPL_ORDER


# ================================================================================================================ write_block / read_block
@pytest.mark.parametrize("lead", [(), (2,)])
def test_write_block_read_block_round_trip(lead):
    g = torch.Generator().manual_seed(1)
    R, N, C = 7, 5, 3
    t = torch.randn(*lead, R, N, C, generator=g)
    ref = t.clone()
    ps = TM.ParkedStorage(t, name="z (test)")
    assert ps.ok and t.untyped_storage().nbytes() == 0 and ps.where == "host"
    # read_block == the rows
    assert torch.equal(ps.read_block(2, 5), ref[..., 2:5, :, :]) and torch.equal(ps.block(0, R), ref)
    # write_block lands exactly those rows; the others are untouched
    new = torch.randn(*lead, 3, N, C, generator=g)
    ps.write_block(2, 5, new)
    assert torch.equal(ps.read_block(2, 5), new) and torch.equal(ps.read_block(0, 2), ref[..., 0:2, :, :]) and torch.equal(ps.read_block(5, R), ref[..., 5:, :, :])
    ps.write_block(0, 1, ref[..., 6:7, :, :].contiguous())
    ps.write_block(R - 1, R, ref[..., 0:1, :, :] * 0 + 3.5)
    # refusals by name
    with pytest.raises(RowpairRefused, match="write_block: src"):
        ps.write_block(2, 5, new.double())
    with pytest.raises(RowpairRefused, match="write_block: src"):
        ps.write_block(2, 4, new)
    with pytest.raises(RowpairRefused, match="outside"):
        ps.write_block(5, R + 1, new)
    out = ps.restore()
    assert out is t
    want = ref.clone()
    want[..., 2:5, :, :] = new
    want[..., 0:1, :, :] = ref[..., 6:7, :, :]
    want[..., R - 1:R, :, :] = 3.5
    assert torch.equal(t, want)
    with pytest.raises(RowpairRefused, match="restored"):
        ps.write_block(0, 1, new[..., :1, :, :])
    # a dropped park refuses; a not-parked tensor (shared storage) writes through to the live tensor
    ps2 = TM.ParkedStorage(t, name="z (test 2)")
    ps2.drop()
    with pytest.raises(RowpairRefused, match="dropped"):
        ps2.write_block(0, 1, new[..., :1, :, :])
    base = torch.zeros(*lead, R + 2, N, C)
    view = base[..., 1:R + 1, :, :]
    ps3 = TM.ParkedStorage(view, name="a view")
    assert not ps3.ok and ps3.why == "shared_storage"
    ps3.write_block(0, 3, new)
    assert torch.equal(base[..., 1:4, :, :], new) and torch.equal(ps3.read_block(0, 3), new)


def test_store_rows_refuses_wrong_rows():
    t = torch.randn(4, 3, 2)
    ps = TM.ParkedStorage(t, name="z")
    with pytest.raises(RowpairRefused, match="store_rows"):
        T.store_rows(ps.host, (4, 3, 2), 4, 0, 2, torch.randn(3, 3, 2))
    ps.restore()


# ================================================================================================================ zparked == classic, bitwise
def _order_entry(rank, P, N, B, dedupe, with_grp, park_u, add, lease):
    torch.set_num_threads(1)
    assert (os.environ.get("ROWPAIR_HOST_SLAB") == "lease") == bool(lease)                 # set by the test body (monkeypatch), shared by the rank threads
    lay = D.Layout.checked(N, P, rank, B)
    P_ = _precursors(N=N, T=3, seed=4, nan_rows=(3, 30), chains=(N // 2, N - N // 2))
    Tn = P_["T"]
    grp = _grp_for(P_) if with_grp else None
    g = torch.Generator().manual_seed(9)
    C_z, c_t = 8, 6
    z = torch.randn(1, N, N, C_z, generator=g)
    Wz = torch.randn(C_z, c_t, generator=g) / 4
    Wd = torch.randn(39, c_t, generator=g) / 4
    Wu = torch.randn(3, c_t, generator=g) / 4
    Wt = torch.randn(c_t, C_z, generator=g) / 4
    mask = torch.ones(1, N, N)
    acc = TM.real_template_rows(lay, Tn, pb_coords=P_["pb"], pb_mask=P_["pbm"], frames=P_["frames"], frame_mask=P_["bfm"], asym_id=P_["asym"],
                                edges=P_["edges"], grp=grp, lead=(1,))
    seen = {"unit_parked": [], "stack_parked": [], "unit_rows": []}
    zl_holder = {}

    def unit_rows(z_rows, t, rows):
        g0, g1 = rows
        i0, i1 = g0 - lay.r0, g1 - lay.r0
        seen["unit_parked"].append(int(zl_holder["z"].untyped_storage().nbytes()) == 0)
        seen["unit_rows"].append((t, g0, g1, bool(z_rows.is_contiguous())))
        return (z_rows @ Wz)[:, None] + acc.rows("template_distogram", [t], i0, i1) @ Wd + acc.rows("template_unit_vector", [t], i0, i1) @ Wu

    def stack(u, m):
        seen["stack_parked"].append(int(zl_holder["z"].untyped_storage().nbytes()) == 0)
        return torch.tanh(u) * 0.5 + u

    finish = lambda t: torch.relu(t.mean(dim=-4)) @ Wt                                   # noqa: E731
    keys = [P_["pbm"][0].bool(), P_["bfm"][0].bool(), P_["pb"][0], P_["frames"][0]] + ([grp[0]] if grp is not None else [])
    groups = TM.template_slot_groups(keys, Tn, lay, dedupe=dedupe, device=z.device)
    out = {}
    for order in ("classic", "zparked"):
        evidence.reset_schedule() if rank == 0 else None
        zl = shard_rows(z, lay, dim=-3).contiguous().clone()
        zl_holder["z"] = zl
        for k in seen:
            seen[k] = []
        tgt = None if add else torch.empty_like(zl)
        res = TM.template_embed_rows(zl, lay, n_templ=Tn, c_t=c_t, unit_rows_fn=unit_rows, pair_stack_fn=stack, finish_fn=finish,
                                    mask_loc=shard_rows(mask, lay, dim=-2), slot_groups=groups, rows=8, add=add, out=tgt, park_u=park_u,
                                    feat_channels=acc.transient_channels, order=order)
        if add:
            assert res is zl
        else:
            assert res is tgt and torch.equal(zl, shard_rows(z, lay, dim=-3))              # the shard itself is untouched (restored bit for bit)
        out[order] = {"z": res.clone(), "zl_nbytes": int(zl.untyped_storage().nbytes()), "unit_parked": list(seen["unit_parked"]),
                      "stack_parked": list(seen["stack_parked"]), "unit_rows": list(seen["unit_rows"]), "sched": dict(evidence.schedule())}
    T.release_host_slabs()
    return out


@pytest.mark.parametrize("P,N,B", [(2, 40, 8), (3, 61, 8)])
@pytest.mark.parametrize("dedupe,with_grp,park_u,add,lease", [
    (True, False, None, True, False),          # de-duplicated groups, this order's default park_u (on), in place, private pools
    (False, False, False, True, True),         # every slot its own group, slabs resident, in place, the LEASE (z park = the leased slab)
    (True, True, True, False, False),          # inter-chain groups, slabs parked, out= form
    (False, True, None, False, True),          # lease + out= form + 3 groups (u parks are lease-second private buffers, named)
])
def test_zparked_equals_classic_bitwise(monkeypatch, P, N, B, dedupe, with_grp, park_u, add, lease):
    if lease:
        monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
    res = run_ranks(P, _order_entry, N, B, dedupe, with_grp, park_u, add, lease)
    for r in range(P):
        c, zp = res[r]["classic"], res[r]["zparked"]
        assert torch.equal(c["z"], zp["z"]), (r, float((c["z"] - zp["z"]).abs().max()))
        assert zp["zl_nbytes"] > 0 and c["zl_nbytes"] > 0                                 # restored into the same storage
        assert zp["unit_parked"] and all(zp["unit_parked"]) and all(zp["stack_parked"])   # z off its storage through the builds AND the stacks
        assert not any(c["unit_parked"]) and not any(c["stack_parked"])                   # classic: z on device through the builds; parked only if PARK_Z
        assert [u[:3] for u in zp["unit_rows"]] == [u[:3] for u in c["unit_rows"]]        # same slots, same row blocks, same order
        assert all(u[3] for u in zp["unit_rows"])
    s0 = res[0]["zparked"]["sched"]
    assert s0["templ_order"] == "zparked" and s0["templ_park_z"] == "parked" and "templ_zparked_peak_gib" in s0 and s0["templ_zparked_h2d_gib"] >= 0
    assert "templ_order" not in res[0]["classic"]["sched"]                                # the classic order records no new word (default census unchanged)
    n_groups = len(s0["templ_groups"].split("/"))
    want_u = "off" if (park_u is False or n_groups == 1) else f"{n_groups - 1}of{n_groups}"
    assert s0["templ_park_u"] == want_u, (s0["templ_park_u"], want_u)
    if lease:
        assert s0.get("host_slab") == "lease"


def test_zparked_env_dispatch_and_not_parked_fallback(monkeypatch):
    """ROWPAIR_TEMPL_ORDER=zparked reaches template_embed_rows without the argument; a z shard that does not own its storage runs the classic order
    by name (templ_order=zparked:not_parked(shared_storage)) with the same result."""
    N, P, B = 24, 2, 8

    def entry(rank, P, env_word):
        torch.set_num_threads(1)
        assert os.environ.get("ROWPAIR_TEMPL_ORDER") == env_word                            # set by the test body (monkeypatch)
        lay = D.Layout.checked(N, P, rank, B)
        g = torch.Generator().manual_seed(3)
        z = torch.randn(1, N, N, 4, generator=g)
        W = torch.randn(4, 2, generator=g)
        Wt = torch.randn(2, 4, generator=g)
        unit = lambda zr, t, rows: (zr @ W)[:, None] * (1 + t)                            # noqa: E731
        stack = lambda u, m: u * 2                                                        # noqa: E731
        finish = lambda t: t.sum(dim=-4) @ Wt                                             # noqa: E731
        big = torch.zeros(1, lay.R + 1, N, 4)
        view = big[:, :lay.R]
        view.copy_(shard_rows(z, lay, dim=-3))
        owned = shard_rows(z, lay, dim=-3).contiguous().clone()
        evidence.reset_schedule() if rank == 0 else None
        a = TM.template_embed_rows(view, lay, n_templ=2, c_t=2, unit_rows_fn=unit, pair_stack_fn=stack, finish_fn=finish, rows=8)
        w1 = dict(evidence.schedule()).get("templ_order")
        b = TM.template_embed_rows(owned, lay, n_templ=2, c_t=2, unit_rows_fn=unit, pair_stack_fn=stack, finish_fn=finish, rows=8)
        w2 = dict(evidence.schedule()).get("templ_order")
        ref = TM.template_embed_rows(shard_rows(z, lay, dim=-3).contiguous().clone(), lay, n_templ=2, c_t=2, unit_rows_fn=unit, pair_stack_fn=stack,
                                     finish_fn=finish, rows=8, order="classic")           # the classic order on the same row blocks (the dense statement's
        return bool(torch.equal(a, ref)), bool(torch.equal(b, ref)), w1, w2               # one GEMM over all rows is another launch shape: not the yardstick)

    monkeypatch.setenv("ROWPAIR_TEMPL_ORDER", "zparked")
    res = run_ranks(P, entry, "zparked")
    for r in range(P):
        assert res[r][0] and res[r][1]
    assert res[0][2] == "zparked:not_parked(shared_storage)" and res[0][3] == "zparked"


# ================================================================================================================ host trim (c2)
def _spy_trim(monkeypatch):
    import opt_core.mem.torch_hostpair as HP
    calls = []

    def fake(lever="pool"):
        calls.append(lever)
        return {"call": "fake_host_emptyCache", "available": True, "ok": True, "error": None, "reserved_before": 3 * 2 ** 30, "reserved_after": 2 ** 30,
                "rss_before": 10.0, "rss_after": 8.0}
    monkeypatch.setattr(HP, "host_cache_trim", fake)
    return calls


def test_host_trim_gated_on_the_lease_word(monkeypatch):
    calls = _spy_trim(monkeypatch)
    T.HOST_TRIM_STATS.update({"calls": 0, "returned_bytes": 0, "seconds": 0.0, "skipped": 0})
    lines = []
    # unset lease word: never (no call, no census word)
    assert T.host_trim("templ.cycle0", lines.append) is None and calls == [] and "host_trim" not in evidence.schedule() and lines == []
    assert T.host_trim_mode() == "auto"
    # lease: once per call, census + one line
    monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
    rec = T.host_trim("templ.cycle0", lines.append)
    assert rec["word"] == "templ.cycle0:2.000" and len(calls) == 1 and evidence.schedule()["host_trim"] == "templ.cycle0:2.000"
    assert evidence.schedule()["host_trim_calls"] == 1 and len(lines) == 1 and "[park] host trim at templ.cycle0" in lines[0] and "2.00 GiB" in lines[0]
    # ROWPAIR_HOST_TRIM=0 wins over the lease; =1 wins without it; a bad word is refused by name
    monkeypatch.setenv("ROWPAIR_HOST_TRIM", "0")
    assert T.host_trim("x") is None and len(calls) == 1
    monkeypatch.delenv("ROWPAIR_HOST_SLAB")
    monkeypatch.setenv("ROWPAIR_HOST_TRIM", "1")
    assert T.host_trim("y")["word"] == "y:2.000" and len(calls) == 2
    monkeypatch.setenv("ROWPAIR_HOST_TRIM", "sometimes")
    with pytest.raises(RowpairRefused, match="ROWPAIR_HOST_TRIM='sometimes'"):
        T.host_trim("z")
    monkeypatch.delenv("ROWPAIR_HOST_TRIM")
    assert T.host_trim("f", force=True)["word"] == "f:2.000" and T.host_trim("g", force=False) is None and len(calls) == 3
    # a torch without the binding is SAID (never a silent no-op)
    import opt_core.mem.torch_hostpair as HP
    monkeypatch.setattr(HP, "host_cache_trim", lambda lever="pool": {"call": None, "available": False, "ok": False, "error": None,
                                                                  "reserved_before": None, "reserved_after": None, "rss_before": 1.0, "rss_after": 1.0})
    assert T.host_trim("h", force=True)["word"] == "h:unavailable"


@pytest.mark.parametrize("lease", [False, True])
def test_run_trunk_sharded_trims_once_per_template_stage(monkeypatch, lease):
    """The driver calls host_trim at the template->MSA seam of every cycle that HAS a template stage: under the lease word one trim per cycle; without
    it none (and no census word); a trunk without template_fn never trims. Outputs are identical either way (the trim touches no tensor)."""
    calls = _spy_trim(monkeypatch)
    N, C, B, P = 40, 4, 8, 2
    sy = Synth(N, C)

    def entry(rank, P, with_template):
        torch.set_num_threads(1)
        assert (os.environ.get("ROWPAIR_HOST_SLAB") == "lease") == bool(lease)             # set by the test body (monkeypatch)
        lay = D.Layout(N, P, rank, B)
        torch.manual_seed(7)
        kw = dict(n_cycles=3, init_rows_fn=sy.rows_fn, init_like=sy.a_i, recycle_update_fn=sy.update, s_init=sy.s_input,
                  single_recycle_fn=lambda s, c: sy.single(s), template_fn=(lambda z, c: z + sy.template(z)) if with_template else None,
                  pairstack_fn=lambda s, z, c: sy.pairstack_rows(s, z), init_rows=5, recycle_rows=6, guard="off", gather="none")
        s, z, out = T.run_trunk_sharded(lay, **kw)
        w = dict(evidence.schedule()).get("host_trim")
        T.release_host_slabs()
        return z.clone(), w

    if lease:
        monkeypatch.setenv("ROWPAIR_HOST_SLAB", "lease")
    n0 = len(calls)
    res = run_ranks(P, entry, True)
    n_templ = len(calls) - n0
    res2 = run_ranks(P, entry, False)
    n_none = len(calls) - n0 - n_templ
    assert n_none == 0
    if lease:
        assert n_templ == 3 * P and res[0][1] == "templ.cycle2:2.000"                     # one per cycle per rank (threads share the spy)
    else:
        assert n_templ == 0 and res[0][1] is None


def test_device_peak_cpu_and_templ_stage_words(monkeypatch):
    pk = T.DevicePeak(torch.device("cpu"))
    pk.mark("a", 10)
    rec = pk.close()
    assert rec["kind"] == "cpu" and rec["peak_gib"] == 0.0
    # the driver brackets the template stage only when ROWPAIR_TEMPL_ORDER is SET (unset: no word)
    N, C, B, P = 40, 4, 8, 2
    sy = Synth(N, C)

    def entry(rank, P, word):
        torch.set_num_threads(1)
        assert os.environ.get("ROWPAIR_TEMPL_ORDER", "") == word                            # set by the test body (monkeypatch)
        lay = D.Layout(N, P, rank, B)
        torch.manual_seed(7)
        evidence.reset_schedule() if rank == 0 else None
        s, z, out = T.run_trunk_sharded(lay, n_cycles=1, init_rows_fn=sy.rows_fn, init_like=sy.a_i, recycle_update_fn=sy.update,
                                        template_fn=lambda z, c: z + sy.template(z), init_rows=5, recycle_rows=6, guard="off", gather="none")
        return z.clone(), dict(evidence.schedule())

    a = run_ranks(P, entry, "")
    monkeypatch.setenv("ROWPAIR_TEMPL_ORDER", "classic")
    b = run_ranks(P, entry, "classic")
    for r in range(P):
        assert torch.equal(a[r][0], b[r][0])
    assert "templ_stage_peak_gib" not in a[0][1] and b[0][1]["templ_stage_peak_kind"] == "cpu"
