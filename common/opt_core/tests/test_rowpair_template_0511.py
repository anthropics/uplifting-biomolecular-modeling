"""opt_core.mem.rowpair.template — the COMPUTED (lazy REAL) template pair rows and their words (CPU; threaded ranks via
``opt_core.testing.run_ranks`` where rows are compared across a sharded layout):

* ``unit_vector_rows``: OpenFold's frame geometry restated component by component == a per-pair scalar reference loop of the same formulas,
  bit-exact (fp32 and fp64; degenerate / collinear / zero frames finite through eps); row slices compose; refusals by name;
* ``real_template_rows``: distogram / unit-vector rows x pair mask x chain mask == the dense products sliced, bit-exact, for slot subsets, row
  ranges with NaN rows and a chain boundary inside the block, with and without inter-chain groups; float64 drift refused by name; the
  feature-device env word; ``template_embed_rows`` over the computed rows == ``template_embed_dense`` over the dense products (threaded P in
  {2, 3}, park_u / nodedupe variants);
* ``slot_census(..., asym_id=)``: tokens / chains words and the consumption line; the positional contract unchanged;
* the inter-chain spec: both formats agree, the consistency rule masks a group by name, ``group_feature`` == brute force, ``interchain_words``
  counts == ``pairs_unmasked``, malformed specs refused, env readers;
* ``templ_block_rows(feat_dims=)`` shrinks the row block monotonically; ``template_embed_rows(feat_channels=)`` records it.
"""
import json
import math
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

torch = pytest.importorskip("torch")

from opt_core.mem.rowpair import RowpairRefused  # noqa: E402
from opt_core.mem.rowpair import dist as D  # noqa: E402
from opt_core.mem.rowpair import evidence  # noqa: E402
from opt_core.mem.rowpair import template as TM  # noqa: E402
from opt_core.mem.rowpair.shard import shard_rows  # noqa: E402
from opt_core.testing import run_ranks  # noqa: E402


# ================================================================================================================ fixtures
def _precursors(N=20, T=3, seed=0, nan_rows=(3, 11), chains=(8, 12)):
    """Per-token template precursors of a 2-chain query: pb coords float64 (NaN rows = absent atoms), masks, backbone frames fp32 (some
    degenerate), the featurizer's asym ids, squared float64 bin edges (39 bins, inf appended)."""
    g = torch.Generator().manual_seed(seed)
    pb = (torch.randn(1, T, N, 3, generator=g, dtype=torch.float64) * 6.0)
    pbm = torch.ones(1, T, N)
    for r in nan_rows:
        if r < N:
            pb[0, 0, r] = float("nan")
            pbm[0, 0, r] = 0.0
    if T > 2:
        pbm[0, 2] = 0.0                                                      # slot 2 is a dummy (all masks zero)
        pb[0, 2] = float("nan")
    frames = torch.randn(1, T, N, 3, 3, generator=g) * 3.0
    frames[0, 1, 5] = 0.0                                                    # a zero frame (nan_to_num of an absent residue)
    frames[0, 1, 7] = float("nan")                                           # a RAW absent residue: NaN coordinates, mask 0 (real_template_rows' nan_to_num)
    frames[0, 1, 6, 2] = frames[0, 1, 6, 1] + 2.0 * (frames[0, 1, 6, 0] - frames[0, 1, 6, 1])   # collinear N-CA-C
    bfm = torch.ones(1, T, N)
    bfm[0, 1, 5] = 0.0
    bfm[0, 1, 7] = 0.0
    if T > 2:
        frames[0, 2] = 0.0
        bfm[0, 2] = 0.0
    c0 = min(chains[0], N - 1)
    asym = torch.cat([torch.full((c0,), 1), torch.full((N - c0,), 2)]).to(torch.int32)[None]                # [1, N]
    n_bins, dmin, dmax = 39, 3.25, 50.75
    lower = torch.linspace(dmin, dmax, n_bins, dtype=torch.float64) ** 2
    upper = torch.cat([lower[1:], torch.tensor([1e8], dtype=torch.float64)])
    edges = torch.stack([lower, upper])
    return {"pb": pb, "pbm": pbm, "frames": frames, "bfm": bfm, "asym": asym, "edges": edges, "N": N, "T": T}


def _uv_reference(frames, eps=1e-6):
    """Per pair, python floats through torch 0-d tensors of the frames' dtype: the OpenFold formulas in their operation order."""
    T, N = frames.shape[-4], frames.shape[-3]
    f = frames.reshape(-1, T, N, 3, 3)[0]
    out = torch.empty(T, N, N, 3, dtype=frames.dtype)
    dt = frames.dtype

    def norm3(x, y, z):
        n2 = x * x + y * y + z * z
        n2 = torch.clamp(n2, min=eps ** 2)
        n = torch.sqrt(n2)
        return x / n, y / n, z / n
    for t in range(T):
        R, TI, TT = [], [], []
        for i in range(N):
            a, b, c = f[t, i, 0], f[t, i, 1], f[t, i, 2]
            e0 = norm3(c[0] - b[0], c[1] - b[1], c[2] - b[2])
            v = (a[0] - b[0], a[1] - b[1], a[2] - b[2])
            cc = v[0] * e0[0] + v[1] * e0[1] + v[2] * e0[2]
            e1 = norm3(v[0] - cc * e0[0], v[1] - cc * e0[1], v[2] - cc * e0[2])
            e2 = (e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0])
            nt = (b[0] * -1, b[1] * -1, b[2] * -1)
            ti = (e0[0] * nt[0] + e0[1] * nt[1] + e0[2] * nt[2], e1[0] * nt[0] + e1[1] * nt[1] + e1[2] * nt[2], e2[0] * nt[0] + e2[1] * nt[1] + e2[2] * nt[2])
            R.append((e0, e1, e2)); TI.append(ti); TT.append((b[0], b[1], b[2]))
        for i in range(N):
            (e0, e1, e2), ti = R[i], TI[i]
            for j in range(N):
                p = TT[j]
                ux = e0[0] * p[0] + e0[1] * p[1] + e0[2] * p[2] + ti[0]
                uy = e1[0] * p[0] + e1[1] * p[1] + e1[2] * p[2] + ti[1]
                uz = e2[0] * p[0] + e2[1] * p[1] + e2[2] * p[2] + ti[2]
                ux, uy, uz = norm3(ux, uy, uz)
                out[t, i, j, 0], out[t, i, j, 1], out[t, i, j, 2] = ux, uy, uz
    assert out.dtype == dt
    return out.reshape(tuple(frames.shape[:-4]) + (T, N, N, 3))


def _dense_products(P_, grp=None, fd=torch.float32, uv_reference=True):
    """The dense statements: float64 one-hot distogram x pb pair mask x chain mask; the reference unit vectors (or the module's statement over all
    rows at once, ``uv_reference=False``) x bb pair mask x chain mask."""
    pb, pbm, fr, bfm, asym, edges = P_["pb"], P_["pbm"], P_["frames"], P_["bfm"], P_["asym"], P_["edges"]
    T, N = P_["T"], P_["N"]
    diff = pb[..., :, :, None, :] - pb[..., :, None, :, :]                   # [1, T, N, N, 3]
    sq = diff ** 2
    d2 = ((sq[..., 0] + sq[..., 1]) + sq[..., 2])[..., None]
    onehot = ((d2 > edges[0]) * (d2 < edges[1])).to(torch.float64).to(fd)  # [1, T, N, N, 39]
    pm = (pbm[..., :, None] * pbm[..., None, :])[..., None].to(fd)
    if grp is None:
        chain = (asym[..., :, None] == asym[..., None, :])[..., None, :, :, None].to(fd)      # [1, 1, N, N, 1]
    else:
        chain = TM.same_or_group_dense(asym, grp).to(fd)                                     # [1, T, N, N, 1]
    dist = onehot * pm * chain
    fr = torch.nan_to_num(fr, nan=0.0)                                       # the featurizer's first statement
    uv = (_uv_reference(fr) if uv_reference else TM.unit_vector_rows(fr, list(range(T)), 0, N)).to(fd)
    bm = (bfm[..., :, None] * bfm[..., None, :])[..., None].to(fd)
    uvm = uv * bm * chain
    return dist, uvm


# ================================================================================================================ C1
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_unit_vector_rows_matches_reference_loop(dtype):
    P_ = _precursors(N=14, T=2, seed=1)
    fr = torch.nan_to_num(P_["frames"], nan=0.0).to(dtype)
    ref = _uv_reference(fr)
    got = TM.unit_vector_rows(fr, [0, 1], 0, 14)
    assert got.dtype == dtype and tuple(got.shape) == (1, 2, 14, 14, 3)
    assert torch.isfinite(got).all()                                         # zero / collinear frames finite through eps
    assert torch.equal(got, ref), (got - ref).abs().max()
    assert torch.equal(TM.unit_vector_rows(fr, [1], 3, 9), ref[:, [1], 3:9])
    assert TM.unit_vector_rows(fr, [1, 0], 2, 5, out_dtype=torch.float16).dtype == torch.float16


def test_unit_vector_rows_slices_compose():
    P_ = _precursors(N=17, T=3, seed=2)
    fr = torch.nan_to_num(P_["frames"], nan=0.0)
    full = TM.unit_vector_rows(fr, [0, 1, 2], 0, 17)
    for slots in ([0], [2, 0], [1, 2], [0, 1, 2]):
        for g0, g1 in ((0, 17), (0, 5), (5, 11), (11, 17), (16, 17), (4, 4)):
            blk = TM.unit_vector_rows(fr, slots, g0, g1)
            assert torch.equal(blk, full[:, slots, g0:g1]), (slots, g0, g1)


def test_unit_vector_rows_refusals():
    fr = _precursors(N=9, T=2)["frames"]
    with pytest.raises(RowpairRefused, match="frames"):
        TM.unit_vector_rows(fr[..., :2], [0], 0, 3)
    with pytest.raises(RowpairRefused, match="slots"):
        TM.unit_vector_rows(fr, [2], 0, 3)
    with pytest.raises(RowpairRefused, match="rows"):
        TM.unit_vector_rows(fr, [0], 5, 10)


# ================================================================================================================ C2
def _grp_for(P_):
    """Slot 1 un-masks the two chains as one group (both chains' row 1 in group 0); slots 0 / 2 none."""
    T, N = P_["T"], P_["N"]
    grp = torch.full((1, T, N), -1, dtype=torch.int32)
    grp[0, 1, :] = 0
    return grp


@pytest.mark.parametrize("with_grp", [False, True])
def test_real_template_rows_equal_dense_products_sliced(with_grp):
    P_ = _precursors(N=20, T=3, seed=3)
    N, T = P_["N"], P_["T"]
    grp = _grp_for(P_) if with_grp else None
    dist, uvm = _dense_products(P_, grp)
    lay = D.Layout(N, 1, 0, B=1)
    evidence.reset_schedule()
    acc = TM.real_template_rows(lay, T, pb_coords=P_["pb"], pb_mask=P_["pbm"], frames=P_["frames"], frame_mask=P_["bfm"], asym_id=P_["asym"],
                                edges=P_["edges"], grp=grp, feature_dtype=torch.float32, lead=(1,))
    assert acc.mode == "computed" and acc.keys() == ["template_distogram", "template_unit_vector"] and acc.feat_device == "cpu"
    assert acc.transient_channels == TM.COMPUTED_TRANSIENT_CHANNELS                      # cpu features on a cpu pair device: the transients live in the block
    sched = evidence.schedule()
    assert sched["templ_mode"] == "computed" and sched["templ_feat_device"] == "cpu" and sched["templ_interchain_grp"] == int(with_grp)
    assert sched["templ_feat_keys"] == "template_distogram+template_unit_vector"
    for slots in ([0, 1, 2], [0], [0, 2], [2, 1, 0], [1]):
        for i0, i1 in ((0, N), (0, 7), (7, 13), (2, 4), (10, 12), (19, 20)):        # (7, 13) holds the chain boundary 8 and NaN row 11
            d = acc.rows("template_distogram", slots, i0, i1)
            u = acc.rows("template_unit_vector", slots, i0, i1)
            assert d.dtype == torch.float32 and tuple(d.shape) == (1, len(slots), i1 - i0, N, 39)
            assert torch.equal(d, dist[:, slots, i0:i1]), ("distogram", slots, i0, i1)
            assert torch.equal(u, uvm[:, slots, i0:i1]), ("unit_vector", slots, i0, i1, (u - uvm[:, slots, i0:i1]).abs().max())
    assert float(dist[:, 2].abs().sum()) == 0.0 and float(uvm[:, 2].abs().sum()) == 0.0      # the dummy slot's rows are +0.0
    whole = acc.rows("template_unit_vector", [0, 1, 2], 0, N)
    assert torch.isfinite(whole).all() and bool(torch.isnan(P_["frames"][0, 1, 7]).all())    # RAW NaN frames at a masked residue: finite (zero) rows
    assert float(whole[0, 1, 7].abs().sum()) == 0.0 and float(whole[0, 1, :, 7].abs().sum()) == 0.0
    if with_grp:
        assert int(dist[0, 1, 0, N - 1].sum()) >= 0 and bool((uvm[0, 1, 0, N - 1] != 0).any())   # inter-chain pair un-masked on slot 1 only
        assert not bool((uvm[0, 0, 0, N - 1] != 0).any())


def test_real_template_rows_dtype_drift_refused():
    P_ = _precursors(N=10, T=2)
    lay = D.Layout(10, 1, 0, B=1)
    kw = dict(pb_mask=P_["pbm"], frames=P_["frames"], frame_mask=P_["bfm"], asym_id=P_["asym"], edges=P_["edges"])
    with pytest.raises(RowpairRefused, match="float64"):
        TM.real_template_rows(lay, 2, pb_coords=P_["pb"].float(), **kw)
    with pytest.raises(RowpairRefused, match="frames"):
        TM.real_template_rows(lay, 2, pb_coords=P_["pb"], pb_mask=P_["pbm"], frames=P_["frames"][..., 0, :], frame_mask=P_["bfm"], asym_id=P_["asym"],
                              edges=P_["edges"])
    with pytest.raises(RowpairRefused, match="edges"):
        TM.real_template_rows(lay, 2, pb_coords=P_["pb"], pb_mask=P_["pbm"], frames=P_["frames"], frame_mask=P_["bfm"], asym_id=P_["asym"],
                              edges=P_["edges"][0])
    with pytest.raises(RowpairRefused, match="keys"):
        TM.real_template_rows(lay, 2, pb_coords=P_["pb"], keys=("template_backbone",), **kw)


def test_real_template_rows_feat_device_env(monkeypatch):
    monkeypatch.delenv("ROWPAIR_TEMPL_FEAT_DEVICE", raising=False)
    assert TM.feat_device() == "cpu" and TM.feat_device("CUDA") == "cuda"
    monkeypatch.setenv("ROWPAIR_TEMPL_FEAT_DEVICE", "tpu")
    with pytest.raises(RowpairRefused, match="cpu | cuda"):
        TM.feat_device()
    monkeypatch.setenv("ROWPAIR_TEMPL_FEAT_DEVICE", "cuda")
    P_ = _precursors(N=10, T=2)
    lay = D.Layout(10, 1, 0, B=1)
    kw = dict(pb_coords=P_["pb"], pb_mask=P_["pbm"], frames=P_["frames"], frame_mask=P_["bfm"], asym_id=P_["asym"], edges=P_["edges"])
    if not torch.cuda.is_available():
        with pytest.raises(RowpairRefused, match="pair device"):              # cuda features on a cpu pair device: refused by name
            TM.real_template_rows(lay, 2, **kw)
    monkeypatch.setenv("ROWPAIR_TEMPL_FEAT_DEVICE", "cpu")
    acc = TM.real_template_rows(lay, 2, **kw)
    assert acc.feat_device == "cpu" and "feat_device=cpu" in acc.describe()


# ---- template_embed_rows over computed rows == template_embed_dense over the dense products (threaded ranks)
def _embed_entry(rank, P, N, B, park_u, dedupe, with_grp):
    lay = D.Layout.checked(N, P, rank, B)
    P_ = _precursors(N=N, T=3, seed=4, nan_rows=(3, 30), chains=(N // 2, N - N // 2))
    T = P_["T"]
    grp = _grp_for(P_) if with_grp else None
    g = torch.Generator().manual_seed(9)
    C_z, c_t = 8, 6
    z = torch.randn(1, N, N, C_z, generator=g)
    Wz = torch.randn(C_z, c_t, generator=g) / 4
    Wd = torch.randn(39, c_t, generator=g) / 4
    Wu = torch.randn(3, c_t, generator=g) / 4
    Wt = torch.randn(c_t, C_z, generator=g) / 4
    mask = torch.ones(1, N, N)
    dist, uvm = _dense_products(P_, grp, uv_reference=False)                # the statement's own dense form (its equality to the reference loop is held above)

    def unit_dense(zz, t, rows):                                             # [1, 1, r, N, c_t] from the DENSE features
        g0, g1 = rows
        return (zz @ Wz)[:, None] + dist[:, [t], g0:g1] @ Wd + uvm[:, [t], g0:g1] @ Wu
    stack = lambda u, m: torch.tanh(u) * 0.5 + u                            # noqa: E731 — an elementwise stand-in for the per-slot pair stack
    finish = lambda t: torch.relu(t.mean(dim=-4)) @ Wt                       # noqa: E731 — [..., T, r, N, c_t] -> [..., r, N, C_z]
    ref = TM.template_embed_dense(z, n_templ=T, unit_fn=unit_dense, pair_stack_fn=stack, finish_fn=finish, mask=mask)
    # the row form over the COMPUTED accessor
    zl = shard_rows(z, lay, dim=-3).contiguous()
    acc = TM.real_template_rows(lay, T, pb_coords=P_["pb"], pb_mask=P_["pbm"], frames=P_["frames"], frame_mask=P_["bfm"], asym_id=P_["asym"],
                                edges=P_["edges"], grp=grp, lead=(1,))

    def unit_rows(z_rows, t, rows):
        g0, g1 = rows
        i0, i1 = g0 - lay.r0, g1 - lay.r0
        return (z_rows @ Wz)[:, None] + acc.rows("template_distogram", [t], i0, i1) @ Wd + acc.rows("template_unit_vector", [t], i0, i1) @ Wu
    keys = [P_["pbm"][0].bool(), P_["bfm"][0].bool(), P_["pb"][0], P_["frames"][0]] + ([grp[0]] if grp is not None else [])
    groups = TM.template_slot_groups(keys, T, lay, dedupe=dedupe, device=zl.device)
    out = TM.template_embed_rows(zl, lay, n_templ=T, c_t=c_t, unit_rows_fn=unit_rows, pair_stack_fn=stack, finish_fn=finish,
                                mask_loc=shard_rows(mask, lay, dim=-2), slot_groups=groups, rows=8, add=True, park_u=park_u,
                                feat_channels=acc.transient_channels)
    return {"eq": bool(torch.equal(out, shard_rows(ref, lay, dim=-3))), "maxdiff": float((out - shard_rows(ref, lay, dim=-3)).abs().max()),
            "groups": TM.groups_word(groups)}


@pytest.mark.parametrize("P,N,B", [(2, 40, 8), (3, 61, 8)])
@pytest.mark.parametrize("park_u,dedupe,with_grp", [(False, True, False), (True, False, False), (False, True, True)])
def test_template_embed_rows_over_computed_rows_equals_dense(P, N, B, park_u, dedupe, with_grp):
    evidence.reset_schedule()
    res = run_ranks(P, _embed_entry, N, B, park_u, dedupe, with_grp)
    sched = evidence.schedule()                                              # process-global: read on the main thread after the ranks joined
    assert sched["templ_feat_channels"] == TM.COMPUTED_TRANSIENT_CHANNELS and sched["templ_mode"] == "computed"
    for r in range(P):
        assert res[r]["eq"], (r, res[r]["maxdiff"])
        assert res[r]["groups"] == ("0/1/2" if not dedupe else res[0]["groups"])


# ================================================================================================================ C3
def test_slot_census_tokens_chains_words():
    T, N = 3, 10
    pbm = torch.zeros(1, T, N)
    bfm = torch.zeros(1, T, N)
    pbm[0, 0, :4] = 1                                                        # slot 0: tokens 0-3 (chain 1)
    bfm[0, 0, 6] = 1                                                         # + token 6 (chain 2) through the other mask
    pbm[0, 1, 5:] = 1                                                        # slot 1: tokens 5-9 (chain 2 only)
    asym = torch.tensor([[1, 1, 1, 1, 1, 2, 2, 2, 2, 2]])
    evidence.reset_schedule()
    c = TM.slot_census(pbm, bfm, slot_dim=-2, asym_id=asym)
    assert c.real == (0, 1) and c.dummy == (2,) and c.tokens == (5, 5, 0) and c.n_tokens == N and c.chains == (2, 1, 0) and c.n_chains == 2
    w = c.words()
    assert w == {"templ_slots": 3, "templ_real": 2, "templ_dummy": 1, "templ_tokens": "5+5", "templ_chains": "2+1", "templ_n_chains": 2}
    assert evidence.schedule()["templ_tokens"] == "5+5"
    assert c.line("cycle 0") == "[templ] cycle 0 REAL templates: slots 2/3 real (t0: tokens 5/10 chains 2/2; t1: tokens 5/10 chains 1/2) dummy=[2]"
    d = TM.slot_census(torch.zeros(1, 2, N), asym_id=asym)
    assert d.all_dummy and d.line() == "[templ] templates: all 2 slots dummy" and d.words()["templ_tokens"] == "none"


def test_slot_census_backcompat():
    T, N = 4, 6
    pbm = torch.zeros(2, T, N)
    pbm[1, 2, 3] = 1.0
    c = TM.slot_census(pbm, torch.zeros(2, T, N))
    assert c.real == (2,) and c.words() == {"templ_slots": 4, "templ_real": 1, "templ_dummy": 3}          # no asym_id: the three words only
    assert c.tokens == (0, 0, 1, 0) and c.chains == () and c.line() == "[templ] REAL templates: slots 1/4 real (t2: tokens 1/6) dummy=[0, 1, 3]"
    e = TM.SlotCensus(3, [1])                                                # the positional contract
    assert e.real == (1,) and e.dummy == (0, 2) and e.tokens == () and e.words() == {"templ_slots": 3, "templ_real": 1, "templ_dummy": 2}


# ================================================================================================================ C4
PREP = {"item": "abc", "variant": "1", "chains": ["A", "B", "C"],
        "groups": [{"group_id": "g0", "query_to_template_chain": {"A": "X", "B": "Y"}}, {"group_id": "g1", "query_chains_ring_order": ["B", "C"]}],
        "per_chain_templates": {"A": [{"group": "g0"}, None], "B": [{"group": "g0"}, "g1"], "C": [None, {"group": "g1"}]}}
SLOTS = {"slots": [{"name": "s0", "groups": [["A", "B"]]}, {"name": "s1", "groups": [["B", "C"]]}]}


def test_interchain_spec_formats_agree():
    a = TM.parse_interchain_spec(PREP, source="prep.json")
    b = TM.parse_interchain_spec(SLOTS, source="slots.json")
    assert a.fmt == "prep" and b.fmt == "slots" and a.invalid == {} and b.invalid == {}
    assert a.item == "abc" and a.variant == "1" and b.item is None
    norm = lambda sp: {c: tuple(None if g is None else sp.group_ids.index(g) for g in rows) for c, rows in sp.row_group.items()}   # noqa: E731
    assert norm(a) == norm(b) == {"A": (0, None), "B": (0, 1), "C": (None, 1)}
    assert TM.parse_interchain_spec(SLOTS["slots"]).row_group == b.row_group          # a bare slot list
    assert b.names == {"0": "s0", "1": "s1"} and b.members["s0#0"] == ("A", "B")


def test_interchain_spec_consistency_rule_masks_group():
    bad = json.loads(json.dumps(PREP))
    bad["per_chain_templates"]["B"] = ["g1", {"group": "g0"}]               # B's g0 row moved to slot 1: rows differ across A (0) and B (1)
    sp = TM.parse_interchain_spec(bad)
    assert set(sp.invalid) == {"g0", "g1"}                                   # g1 too: B carries it on row 0, C on row 1
    assert "rows differ across chains" in sp.invalid["g0"] and all(g is None for rows in sp.row_group.values() for g in rows)
    grp, events = TM.group_feature(sp, ["A"] * 3 + ["B"] * 3 + ["C"] * 2, 2, 10)
    assert int((grp >= 0).sum()) == 0 and any(e.startswith("templ_interchain_masked=g0:rows differ") for e in events)
    two = json.loads(json.dumps(PREP))
    two["per_chain_templates"]["A"] = [{"group": "g0"}, {"group": "g0"}]    # A carries g0 on two rows
    sp2 = TM.parse_interchain_spec(two)
    assert "several rows" in sp2.invalid["g0"] and "g1" not in sp2.invalid


def test_group_feature_vs_bruteforce():
    sp = TM.parse_interchain_spec(PREP)
    chain_ids = ["A"] * 4 + ["B"] * 3 + ["C"] * 2 + ["D"] * 2               # D absent from the spec; 2 padding tokens beyond
    T, N = 3, 13
    evidence.reset_schedule()
    grp, events = TM.group_feature(sp, chain_ids, T, N)
    assert grp.dtype == torch.int32 and tuple(grp.shape) == (T, N)
    brute = torch.full((T, N), -1, dtype=torch.int32)
    for i, c in enumerate(chain_ids):
        rows = sp.row_group.get(c, ())
        for t in range(T):
            if t < len(rows) and rows[t] is not None:
                brute[t, i] = sp.group_ids.index(rows[t])
    assert torch.equal(grp, brute) and events == [] and int((grp[2] >= 0).sum()) == 0 and int((grp[:, 11:] >= 0).sum()) == 0
    g2, ev2 = TM.group_feature(sp, ["A"] * 5, T, 5)                          # B and C absent from the query: named
    assert ev2 == ["templ_interchain_absent_chains=B+C"] and int((g2 >= 0).sum()) == 5
    g3, ev3 = TM.group_feature(None, chain_ids, T, N)
    assert int((g3 >= 0).sum()) == 0 and ev3 == []
    with pytest.raises(RowpairRefused, match="chain ids"):
        TM.group_feature(sp, ["A"] * 6, T, 5)


def test_interchain_words_counts_equal_pairs_unmasked(monkeypatch):
    monkeypatch.delenv("ROWPAIR_TEMPL_INTERCHAIN_SCOPE", raising=False)
    sp = TM.parse_interchain_spec(PREP)
    chain_ids = ["A"] * 4 + ["B"] * 3 + ["C"] * 2
    T, N = 2, 9
    grp, _ = TM.group_feature(sp, chain_ids, T, N)
    asym = torch.tensor([1] * 4 + [2] * 3 + [3] * 2)
    per = TM.pairs_unmasked(asym, grp)
    assert per == [2 * 4 * 3, 2 * 3 * 2]                                     # ordered pairs across A-B (slot 0) and B-C (slot 1)
    dense = TM.same_or_group_dense(asym, grp)[..., 0]                        # [T, N, N]
    same = (asym[:, None] == asym[None, :])
    assert [int((dense[t] & ~same).sum()) for t in range(T)] == per
    evidence.reset_schedule()
    words = TM.interchain_words(asym, grp, sp)
    assert words == "groups=2 pairs=36 per_slot=24+12 scope=all INTERCHAIN_UNMASK" and evidence.schedule()["templ_interchain"] == "groups:2,pairs:36,scope:all"
    none = torch.full((T, N), -1, dtype=torch.int32)
    assert TM.interchain_words(asym, none, scope="features") == "groups=0 pairs=0 per_slot=0+0 scope=features"


def test_interchain_spec_malformed_refused(tmp_path):
    with pytest.raises(RowpairRefused, match="neither"):
        TM.parse_interchain_spec({"foo": 1}, source="x.json")
    with pytest.raises(RowpairRefused, match="group_id"):
        TM.parse_interchain_spec({"per_chain_templates": {}, "groups": [{"name": "g"}]})
    with pytest.raises(RowpairRefused, match="two groups"):
        TM.parse_interchain_spec({"slots": [{"name": "s", "groups": [["A", "B"], ["B", "C"]]}]})
    with pytest.raises(RowpairRefused, match="chain ids"):
        TM.parse_interchain_spec([{"groups": ["AB"]}])
    p = tmp_path / "bad.json"
    p.write_text("{not json")
    with pytest.raises(RowpairRefused, match="unreadable"):
        TM.load_interchain_spec(str(p))
    with pytest.raises(RowpairRefused, match="unreadable"):
        TM.load_interchain_spec(str(tmp_path / "absent.json"))


def test_interchain_env_readers(monkeypatch, tmp_path):
    monkeypatch.delenv("ROWPAIR_TEMPL_INTERCHAIN", raising=False)
    monkeypatch.delenv("ROWPAIR_TEMPL_INTERCHAIN_SCOPE", raising=False)
    assert TM.load_interchain_spec() is None and TM.interchain_scope() == "all"
    monkeypatch.setenv("ROWPAIR_TEMPL_INTERCHAIN_SCOPE", "Features")
    assert TM.interchain_scope() == "features"
    monkeypatch.setenv("ROWPAIR_TEMPL_INTERCHAIN_SCOPE", "model")
    with pytest.raises(RowpairRefused, match="all | features"):
        TM.interchain_scope()
    p = tmp_path / "abc_var1.groups.json"
    p.write_text(json.dumps(PREP))
    monkeypatch.setenv("ROWPAIR_TEMPL_INTERCHAIN", str(p))
    sp = TM.load_interchain_spec()
    assert sp is not None and sp.fmt == "prep" and sp.source == str(p) and TM.load_interchain_spec() is sp       # cached per (path, mtime)
    assert TM.ENV_TEMPL_INTERCHAIN == "ROWPAIR_TEMPL_INTERCHAIN" and TM.ENV_TEMPL_FEAT_DEVICE == "ROWPAIR_TEMPL_FEAT_DEVICE"


# ================================================================================================================ C5
def test_templ_block_rows_feat_dims_words(monkeypatch):
    monkeypatch.delenv("ROWPAIR_ROWBLK_MB", raising=False)
    evidence.reset_schedule()
    prev = None
    for fdim in (16, 64, TM.COMPUTED_TRANSIENT_CHANNELS, 512, 4096):
        rows, source = TM.templ_block_rows(20000, 64, 4, 128, feat_dims=fdim)
        assert source == "default" and rows >= 1
        assert prev is None or rows <= prev
        prev = rows
    assert evidence.schedule()["templ_rows_source"] == "default"
    assert TM.templ_block_rows(20000, 64, 4, 128, rows=12, feat_dims=4096) == (12, "given")


def test_import_is_light():
    import subprocess
    code = ("import sys\n"
            "class _Block:\n"
            "    def find_spec(self, name, path=None, target=None):\n"
            "        if name == 'torch' or name.startswith('torch.'):\n"
            "            raise ImportError('masked')\n"
            "        return None\n"
            "sys.meta_path.insert(0, _Block())\n"
            f"sys.path.insert(0, {os.path.dirname(HERE)!r})\n"
            "import opt_core.mem.rowpair.template as T\n"
            "assert 'real_template_rows' in T.__all__ and 'parse_interchain_spec' in T.__all__ and 'torch' not in sys.modules\n"
            "sp = T.parse_interchain_spec({'slots': [{'name': 's', 'groups': [['A', 'B']]}]})\n"
            "assert sp.row_group == {'A': ('s#0',), 'B': ('s#0',)}\n")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-2000:]
