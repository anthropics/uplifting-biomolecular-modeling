"""The `rank0_bcast` data form of `--mode big --n_gpu P` (P > 1; tp._rank0_item): rank 0 featurises each query item and the core broadcasts
the tree; ranks > 0 build only their own rows of the template pair features from the broadcast precursors (tp_feats.rows_for_rank); an error
item on rank 0 is the same error item everywhere; a rank-0 failure outside upstream's per-item guard refuses every rank by one name."""
import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("opt_core.mem.rowpair.rankdata", reason="the pinned core is older than the rank-data surface (broadcast_features)")

from opendde_opt import tp, tp_feats
from opendde_opt.tests.test_tp_support import _TF, _rank_env, _replicated_feats, _templates


def _precursor_fd(templ) -> dict:
    """The feature dict's template precursors as the featurising path leaves them (upstream dict_to_tensor: float32 / int64 tensors)."""
    from opendde.utils.torch_utils import dict_to_tensor
    return dict_to_tensor(dict(templ.as_data_dict()))


@pytest.mark.parametrize("geom", [(37, 2, 1), (64, 2, 0), (100, 3, 2), (90, 3, 1)], ids=["N37_P2_r1", "N64_P2_r0", "N100_P3_r2", "N90_P3_r1"])
def test_rows_built_from_received_precursors_equal_the_rows_a_featurising_rank_bears(monkeypatch, geom):
    """rows_for_rank(fd) on rank r == what rank r's own featurizer bears (tp_feats._make_as_opendde_dict_rows → dict_to_tensor): the four
    [T, R_r, N(, F)] row features and the [r0, R, N] marker, element for element and dtype for dtype — from precursors that went through
    upstream's tensor conversion (int32 → int64, float → float32) exactly as a received tree carries them."""
    from opendde.utils.torch_utils import dict_to_tensor
    N, P, r = geom
    templ = _templates(N)
    _rank_env(monkeypatch, P, r)
    born = dict_to_tensor(tp_feats._make_as_opendde_dict_rows(_TF().Templates.as_opendde_dict)(templ))   # the featurising rank's dict
    fd = _precursor_fd(templ)
    assert not any(k in fd for k in tp_feats.PAIR_KEYS)
    out = tp_feats.rows_for_rank(fd)
    assert out is fd
    for k in tp_feats.PAIR_KEYS + (tp_feats.ROWS_KEY,):
        a, b = fd[k], born[k]
        assert torch.is_tensor(a) and a.dtype == b.dtype and tuple(a.shape) == tuple(b.shape), (k, a.dtype, b.dtype, a.shape, b.shape)
        assert torch.equal(a, b), (k, float((a.double() - b.double()).abs().max()))
    lay = tp.layout_of(N, P, r)
    assert fd[tp_feats.ROWS_KEY].tolist() == [lay.r0, lay.R, N]


def test_rows_for_rank_refuses_by_the_receive_side_word_without_precursors(monkeypatch):
    from opt_core.mem.rowpair import RowpairRefused
    _rank_env(monkeypatch, 2, 1)
    with pytest.raises(RowpairRefused, match=r"refused: feats_bcast_malformed: template_aatype \(the received features carry no template precursors \['template_aatype', 'template_atom_positions', 'template_atom_mask'\]"):
        tp_feats.rows_for_rank({"msa": torch.zeros(2, 3)})


def test_every_receive_side_refusal_carries_the_cores_malformed_words():
    """The kit's receive-side checks and the core's receipt check share one text: `refused: feats_bcast_malformed: <key> (…)`."""
    from opt_core.mem.rowpair import rankdata, RowpairRefused
    assert tp.BCAST_MALFORMED == "refused: feats_bcast_malformed:" == f"refused: {tp.FEATS_WHAT}_bcast_malformed:"
    e = tp._malformed("sample_index", "why")
    assert isinstance(e, RowpairRefused) and f"{tp.BCAST_MALFORMED} sample_index (why)" in str(e), str(e)
    with pytest.raises(RowpairRefused, match=r"refused: feats_bcast_malformed: b \(") as ei:                    # the core's own words for a top-level violation: same class, same text shape
        rankdata.check_received({"a": 1}, {"keys": ["a", "b"], "tensors": {}})
    core, kit = str(ei.value), str(e)
    assert core[:core.index(tp.BCAST_MALFORMED)] == kit[:kit.index(tp.BCAST_MALFORMED)], (core, kit)          # identical lead-in (the lever prefix the exception class adds)


def test_the_words_are_the_cores_and_named_once():
    from opt_core.mem.rowpair import rankdata
    assert tp.DATA_FORM in rankdata.DATA_FORMS and tp.data_form_fields() == rankdata.data_form_word(tp.DATA_FORM) == "data_form=rank0_bcast"
    assert tp.ERROR_ITEM.split()[0] not in ("ok", "failed:") and not tp.ERROR_ITEM.startswith("failed:")     # a control word to the core: nothing travels
    assert tp.ATOM_ARRAY_KEY not in tp.PER_RANK_FEATS and tp.ATOM_ARRAY_KEY.startswith("_rowpair_")
    assert tp_feats.PRECURSOR_KEYS == ("template_aatype", "template_atom_positions", "template_atom_mask")
    assert {"feats_bcast_calls", "feats_error_items", "feats_bcast_gib", "feats_wait_s", "feats_bcast_s", "feats_rng"} <= set(tp.STATS)
    assert tp.BCAST_MALFORMED.startswith("refused: ") and tp.FEATS_WHAT in tp.BCAST_MALFORMED


GROUP_SITES = {("tp", "_rank0_item"), ("tp", "_feats_census"), ("tp", "_layout"), ("tp_diffusion", "struct_layout")}   # the functions of opendde_opt/tp*.py that may create the group


def test_the_default_group_is_created_only_where_the_dataloader_already_exists():
    """Upstream branches on torch.distributed.is_initialized() when it builds the inference dataloader (data-parallel job sharding), its control
    groups and its seed schedule; the adapter creates the default group only inside a dataset item (rank0_bcast), at the feature move, at a trunk
    layout, or at the structural layout — all after the dataloader exists. Pinned on the source of EVERY opendde_opt/tp*.py: the call sites of
    tp._group() are exactly GROUP_SITES, and nothing but tp._group() reaches the core's init_from_env."""
    import glob, os, re
    pkg = os.path.dirname(tp.__file__)
    callers, initers = set(), set()
    for path in sorted(glob.glob(os.path.join(pkg, "tp*.py"))):
        mod = os.path.basename(path)[:-3]
        src = open(path).read()
        for m in re.finditer(r"^def (\w+)\(.*?(?=^def |^class |\Z)", src, re.S | re.M):
            body = m.group(0)
            if re.search(r"\b_group\(\)", body) and m.group(1) != "_group":
                callers.add((mod, m.group(1)))
            if "init_from_env" in body:
                initers.add((mod, m.group(1)))
    assert callers == GROUP_SITES, sorted(callers ^ GROUP_SITES)
    assert initers == {("tp", "_group")}, initers


def test_building_the_dataloader_and_its_iterator_creates_no_group(monkeypatch):
    """The behavioural side of the pin: in a rank process (P=2, rank 1 by environment) the patched __getitem__ is bound, a DataLoader is built over
    the dataset and its iterator created — no item read yet — and no process group exists (tp.STATS['group'] falsy, torch.distributed not
    initialised): upstream's dataloader construction sees no group, exactly as at P=1."""
    import torch.distributed
    import torch.utils.data as tud
    _rank_env(monkeypatch, 2, 1)
    assert tp.in_rank_process() and not tp.STATS["group"] and not (torch.distributed.is_available() and torch.distributed.is_initialized())
    class DS(tud.Dataset):
        inputs = [{"name": "item0"}, {"name": "item1"}]
        def __len__(self): return len(self.inputs)
        def _stock_getitem(self, index): raise AssertionError("no item is read while the loader is built")
        __getitem__ = tp._make_getitem_rank0(_stock_getitem)
    ds = DS()
    dl = tud.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda b: b, sampler=tud.SequentialSampler(ds))
    it = iter(dl)
    assert not tp.STATS["group"] and not (torch.distributed.is_available() and torch.distributed.is_initialized())
    assert len(dl) == 2 and it is not None


def test_at_one_gpu_the_patched_getitem_is_the_stock_method(monkeypatch):
    """P = 1 (no rank world in the environment): the wrapper calls the stock __getitem__ and touches nothing of the adapter."""
    from opt_core.mem.rowpair import launch
    monkeypatch.delenv(launch.ENV_WORLD, raising=False); monkeypatch.delenv(launch.ENV_RANK, raising=False)
    before = dict(tp.STATS)
    calls = []
    def stock(self, index):
        calls.append(index); return {"sample_name": "x", "sample_index": index}, "atoms", ""
    getitem = tp._make_getitem_rank0(stock)
    assert getitem(object(), 3) == ({"sample_name": "x", "sample_index": 3}, "atoms", "") and calls == [3]
    assert {k: tp.STATS[k] for k in before} == before


# ------------------------------------------------------------------ two CPU ranks (gloo): the item handed from rank 0
class _StubDataset:
    """What tp._rank0_item reads of upstream's InferenceDataset: `.inputs` (the parsed query items)."""
    def __init__(self, n=2):
        self.inputs = [{"name": f"item{i}", "sequences": []} for i in range(n)]


def _stock_item(N: int, index: int):
    """A featurising rank's item as upstream's __getitem__ returns it: (data, atom_array, "") with this rank's template rows born in the dict."""
    from opendde.utils.torch_utils import dict_to_tensor
    templ = _templates(N, seed=index)
    fd = dict(_replicated_feats(seed=index))
    fd.update(dict_to_tensor(tp_feats._make_as_opendde_dict_rows(_TF().Templates.as_opendde_dict)(templ)))   # precursors + this rank's rows + marker
    random.random(); np.random.random(); torch.rand(1)                                        # featurisation consumes host RNG (rank 0 only under rank0_bcast)
    data = {"input_feature_dict": fd, "N_token": torch.tensor([N]), "entity_poly_type": {"1": "polypeptide(L)"}, "sample_name": f"item{index}", "sample_index": index}
    return data, {"atom_array_stand_in": np.arange(7)}, ""


def _item_rank_entry(N: int, case: str):
    """Rank body: seed identically (runseed), hand item 0 through tp._rank0_item with a stock stand-in that featurises on rank 0 ONLY (a call on
    another rank fails the test), then compare: the received replicated features, this rank's rebuilt rows vs the rows its own featurizer would
    bear, the carried RNG, the census verdict; `case` = ok | erroritem | raises."""
    import contextlib, io
    from opendde_opt import tp as _tp, tp_feats as _tf
    from opt_core.mem.rowpair import RowpairRefused, launch, rankdata
    r = launch.rank()
    random.seed(11); np.random.seed(11); torch.manual_seed(11)                                # every rank runs the same seed (settings.run_seed)
    calls = []
    def orig(ds, index):
        calls.append(index)
        assert r == 0, "a rank other than 0 featurised"
        if case == "raises":
            raise ValueError("query item unreadable")
        if case == "erroritem":
            return {"sample_name": ds.inputs[index]["name"], "sample_index": index}, None, "boom:\nTraceback (most recent call last): …"
        return _stock_item(N, index)
    ds = _StubDataset()
    if case == "raises":
        with pytest.raises((RowpairRefused, ValueError)) as ei:
            _tp._rank0_item(ds, 0, orig)
        mine_facts = {"rank": r, "refused": str(ei.value)}
        return {**mine_facts, "all": sorted(_gather(mine_facts), key=lambda d: d["rank"])}
    data, atom_array, err = _tp._rank0_item(ds, 0, orig)
    assert calls == ([0] if r == 0 else []), calls
    if case == "erroritem":
        assert atom_array is None and err and data["sample_name"] == "item0" and data["sample_index"] == 0 and "input_feature_dict" not in data
        assert _tp.STATS["feats_error_items"] == 1 and _tp.STATS["feats_bcast_calls"] == 1
        mine_facts = {"rank": r, "err": err}
        return {**mine_facts, "all": sorted(_gather(mine_facts), key=lambda d: d["rank"])}
    assert err == "" and data["sample_index"] == 0 and data["sample_name"] == "item0" and data["entity_poly_type"] == {"1": "polypeptide(L)"}
    assert isinstance(atom_array, dict) and np.array_equal(atom_array["atom_array_stand_in"], np.arange(7)) and _tp.ATOM_ARRAY_KEY not in data
    fd = data["input_feature_dict"]
    mine = _stock_item(N, 0)[0]["input_feature_dict"]                                         # what THIS rank's featurizer would have borne (its rows)
    for k, v in mine.items():
        assert k in fd, k
        if torch.is_tensor(v):
            assert fd[k].dtype == v.dtype and tuple(fd[k].shape) == tuple(v.shape) and torch.equal(fd[k].cpu(), v), k
    assert set(fd) == set(mine), sorted(set(fd) ^ set(mine))
    lay = _tp.layout_of(N, launch.world_size(), r)
    assert fd[_tf.ROWS_KEY].tolist() == [lay.r0, lay.R, N] and tuple(fd["template_distogram"].shape) == (3, lay.R, N, 39)
    nxt = (random.random(), float(np.random.random()), float(torch.rand(1)))                    # rank 0's post-featurisation host RNG state carried: the same next draws on every rank
    digest = _tp.feats_digest(fd)
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        _tp._feats_census(fd)                                                                   # equal by construction: the census line, no refusal
    line = [ln for ln in buf.getvalue().splitlines() if "[feats]" in ln]
    assert len(line) == 1 and "feats_ranks_equal=yes" in line[0], buf.getvalue()
    ev = dict(_tp.evidence_pairs())
    assert ev["data_form"] == "rank0_bcast" and ev["feats_bcast_calls"] == 1 and ev["feats_error_items"] == 0 and ev["feats_rng"] == "carried"
    assert (ev["feats_bcast_gib"] > 0) and ev["host_peak_gib"] > 0
    mine_facts = {"rank": r, "digest": digest, "nxt": nxt, "rows": fd[_tf.ROWS_KEY].tolist()}
    allv = sorted(_gather(mine_facts), key=lambda d: d["rank"])                               # every rank's facts, on every rank
    assert len(allv) == 2 and allv[0]["digest"] == allv[1]["digest"] and allv[0]["nxt"] == allv[1]["nxt"], allv   # one tree, one RNG stream
    assert allv[1]["rows"][0] == allv[0]["rows"][0] + allv[0]["rows"][1] and allv[0]["rows"][0] == 0          # rank 1's rows start where rank 0's end
    return {**mine_facts, "gib": ev["feats_bcast_gib"], "line": line[0], "all": allv}


def _gather(obj):
    from opt_core.mem.rowpair import dist as _dist
    return list(_dist.comm().allgather_obj(obj))


def _both(N, case):
    pytest.importorskip("opt_core.mem.rowpair.msa", reason="the pinned core is older than the rowpair surface these ranks use")
    from opendde_opt.tests.conftest import run_sharded_or_skip
    return run_sharded_or_skip(2, _item_rank_entry, N, case, mode="big", backend="gloo", cpu_ok=True, run_timeout_s=300)   # rank 0's value; rank 1 asserts in its own process


def test_rank_0_featurises_and_every_rank_holds_its_own_complete_item():
    """Item 0 on two gloo ranks: rank 1 never featurises; its replicated features equal rank 0's (one digest on both ranks, census yes), its template
    pair rows are ITS rows (marker [R0, R1, N]) and equal what its own featurizer bears, the AtomArray and the item's scalars arrive, and the next
    host RNG draws are the same on both ranks (rank 0's state carried)."""
    r0 = _both(37, "ok")
    assert r0["rank"] == 0 and "feats_ranks_equal=yes" in r0["line"] and r0["gib"] > 0 and len(r0["all"]) == 2


def test_an_error_item_on_rank_0_is_the_same_error_item_on_every_rank():
    r0 = _both(37, "erroritem")
    assert r0["err"].startswith("boom:") and [d["err"].split(":")[0] for d in r0["all"]] == ["boom", "rank 0"], r0["all"]


def test_a_rank_0_failure_outside_the_item_guard_refuses_every_rank_by_one_name():
    r0 = _both(37, "raises")
    assert all("feats_rank0_failed: ValueError: query item unreadable" in d["refused"] for d in r0["all"]), r0["all"]


def _loader_rank_entry(N: int):
    """Rank body: a dataset of three items — item 1 is an error item on rank 0 — behind the PATCHED __getitem__, read through a DataLoader to
    exhaustion (upstream's shape: batch_size 1, identity collate, sequential sampler, num_workers 0). Featurisation happens on rank 0 only; every
    rank sees the same three items in the same order (the error item as an error item), one broadcast per item (three calls, keys feats/0..2 in
    step on both ranks — a rank out of step would block or refuse), and the exhausted iterator raises StopIteration on both ranks with no fourth
    broadcast."""
    import torch.utils.data as tud
    from opendde_opt import tp as _tp
    from opt_core.mem.rowpair import launch
    r = launch.rank()
    random.seed(5); np.random.seed(5); torch.manual_seed(5)
    featurised = []
    class DS(tud.Dataset):
        inputs = [{"name": "item0"}, {"name": "item1"}, {"name": "item2"}]
        def __len__(self): return len(self.inputs)
        def _stock_getitem(self, index):
            featurised.append(index)
            assert r == 0, "a rank other than 0 featurised"
            if index == 1:
                return {"sample_name": "item1", "sample_index": 1}, None, "boom: item 1 cannot be parsed"
            data, atoms, err = _stock_item(N, index)
            data["sample_name"], data["sample_index"] = f"item{index}", index
            return data, atoms, err
        __getitem__ = _tp._make_getitem_rank0(_stock_getitem)
    ds = DS()
    it = iter(tud.DataLoader(ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda b: b, sampler=tud.SequentialSampler(ds)))
    seen = []
    for _ in range(3):
        (data, atoms, err), = next(it)
        seen.append((data["sample_index"], data["sample_name"], bool(err), atoms is None))
    with pytest.raises(StopIteration):
        next(it)                                                                               # exhausted on every rank: no item read, no broadcast
    assert seen == [(0, "item0", False, False), (1, "item1", True, True), (2, "item2", False, False)], seen
    assert featurised == ([0, 1, 2] if r == 0 else []), featurised
    assert _tp.STATS["feats_bcast_calls"] == 3 and _tp.STATS["feats_error_items"] == 1 and _tp.STATS["feats_rng"] == "carried"   # sticky: the error item (nothing carried) did not mask it
    facts = {"rank": r, "calls": _tp.STATS["feats_bcast_calls"], "errors": _tp.STATS["feats_error_items"], "seen": seen}
    allv = sorted(_gather(facts), key=lambda d: d["rank"])
    assert allv[0]["calls"] == allv[1]["calls"] == 3 and allv[0]["seen"] == allv[1]["seen"], allv
    return {**facts, "all": allv}


def test_three_items_through_the_loader_stay_in_step_on_two_ranks_and_stop_together():
    pytest.importorskip("opt_core.mem.rowpair.msa", reason="the pinned core is older than the rowpair surface these ranks use")
    from opendde_opt.tests.conftest import run_sharded_or_skip
    r0 = run_sharded_or_skip(2, _loader_rank_entry, 37, mode="big", backend="gloo", cpu_ok=True, run_timeout_s=300)
    assert r0["rank"] == 0 and r0["calls"] == 3 and r0["errors"] == 1 and len(r0["all"]) == 2
