"""`--mode big --n_gpu P>1`: rank 0 featurises, every rank computes on its item (rowpair.item_of_rank0 / replicate_inputs over the core's
`rankdata.broadcast_features`, data form `rank0_bcast`). Proven on 2 threaded CPU ranks of one process (opt_core.testing.run_ranks: the
thread comm carries the store rendezvous, the object and tensor broadcasts and the digest gate exactly as the NCCL group does):

  * rank 0 alone calls the featuriser; rank 1 returns rank 0's item — tensors byte-equal, sizes and names equal, its `atom_array` None;
    under ROWPAIR_MSA_HOST=rank0 the raw MSA features never travel (rank 0 keeps its own tensors, rank 1 holds none, both know the shapes);
  * an item whose featurisation stock caught (error message, no features) reaches both ranks as the error triple, in step, and is counted
    failed through the activation's callback on both;
  * a featuriser that raises on rank 0 is `refused: feats_rank0_failed` on BOTH ranks — no rank left waiting; a rank whose digest differs
    makes BOTH ranks print the core's `[feats] … feats_ranks_equal=no digests=…` line and refuse (`feats_ranks_differ`);
  * a sequence ok / error item / ok in one run stays in step (one rendezvous key per call, the featuriser runs 3 times on rank 0, never on
    rank 1) and both good items reach `predict` on both ranks;
  * a refusal inside the dataset's `__getitem__` ends the rank with SystemExit(EXIT_NOT_ACTIVE) and an ITEM failed record even though
    upstream's loop swallows every Exception around the loader (runner/batch_inference.py:545-549) — the exit-code contract;
  * the pairing guards refuse by name before any collective (a loader worker; a stock sampler world other than 1), same exit;
  * `predict` admits only the item the rank replicated last (`rowpair_item_unreplicated` otherwise);
  * P == 1 is the stock statement: the featuriser's triple, untouched.
No GPU, no protenix model; the item is a small synthetic feature dict shaped like upstream's (input_feature_dict + sizes + names).
"""
import pytest

torch = pytest.importorskip("torch")
import importlib.util                                                        # noqa: E402
needs_protenix = pytest.mark.skipif(importlib.util.find_spec("protenix") is None,      # the four tests below drive upstream's own loop shape / distributed
                                    reason="protenix is not importable in this interpreter: the test drives upstream's loop / "
                                           "protenix.utils.distributed (rowpair.DIST_MODULE) — run it in the kit's stack")   # wrapper: named, skipped where protenix is absent

from opt_core.mem.rowpair import RowpairRefused
from opt_core.mem.rowpair import rankdata as RD
from opt_core.testing import run_ranks

from protenix_v1_opt import report as R
from protenix_v1_opt import rowpair
from protenix_v1_opt import tp

ATOM_ARRAY = ("atom_array_of_rank0",)                      # stands for the biotite AtomArray: rank 0's own, never broadcast
NAME = "q1"


def _item(n_tok: int = 6, s_msa: int = 5) -> dict:
    g = torch.Generator().manual_seed(0)
    ifd = {"msa": torch.randint(0, 32, (s_msa, n_tok), generator=g), "has_deletion": torch.zeros(s_msa, n_tok),
           "deletion_value": torch.rand(s_msa, n_tok, generator=g), "restype": torch.rand(n_tok, 32, generator=g),
           "token_bonds": torch.zeros(n_tok, n_tok), "asym_id": torch.zeros(n_tok, dtype=torch.long), "ref_pos": torch.rand(n_tok * 4, 3, generator=g)}
    return {"input_feature_dict": ifd, "N_token": torch.tensor([n_tok]), "N_atom": torch.tensor([n_tok * 4]), "N_msa": torch.tensor([s_msa]),
            "N_asym": torch.tensor([1]), "entity_poly_type": {"1": "polypeptide(L)"}, "sample_name": NAME, "sample_index": 0}


def _entry(rank, P, mode):
    calls = {"n": 0}

    def featurise():
        calls["n"] += 1
        if mode == "raise":
            raise RuntimeError("featuriser died: no CCD entry")
        if mode == "error_item":                                # what the stock __getitem__ returns when process_one raised (infer_dataloader.py:288-292)
            return {"sample_name": NAME, "sample_index": 0}, None, "KeyError: 'ZZZ':\nTraceback (most recent call last): ..."
        return _item(), ATOM_ARRAY, ""

    try:
        data, atom_array, error_message = rowpair.replicate_inputs(featurise, NAME, 0)
    except RowpairRefused as e:
        return "refused", str(e), calls["n"]
    ifd = data.get("input_feature_dict", {})
    return "ok", {"data": data, "ifd": ifd, "atom_array": atom_array, "error": error_message, "calls": calls["n"],
                  "hold": (tp.STATE.get("msa_hold") or {}).get(rank)}, calls["n"]


class _SeqDataset:
    """The stock dataset's shape item_of_rank0 reads: `inputs` (the per-rank JSON parse) and, for the fake featuriser, a mode per item."""
    def __init__(self, modes):
        self.modes = list(modes)
        self.inputs = [{"name": f"q{i}"} for i in range(len(self.modes))]


def _stock_getitem_of(calls):
    """Stands for the stock `InferenceDataset.__getitem__`: featurises item i (or returns the caught-error triple); records every call."""
    def stock(ds, i):
        calls.append(i)
        mode = ds.modes[i]
        if mode == "raise":                                      # an exception escaping the stock method (outside process_one's own try)
            raise RuntimeError("featuriser died: no CCD entry")
        if mode == "error_item":
            return {"sample_name": ds.inputs[i]["name"], "sample_index": i}, None, "KeyError: 'ZZZ':\nTraceback (most recent call last): ..."
        d = _item()
        d["sample_name"], d["sample_index"] = ds.inputs[i]["name"], i
        return d, ATOM_ARRAY, ""
    return stock


def _upstream(ds, getitem):
    """Upstream's shape around the item fetch: the loader loop OUTSIDE the per-item try (runner/inference.py:453-468: an error item is skipped,
    a good one predicted) inside the per-JSON handler that swallows every Exception (runner/batch_inference.py:545-549)."""
    reached = []
    try:
        for i in range(len(ds.inputs)):                          # for batch in dataloader:
            data, atom_array, error_message = getitem(ds, i)
            try:
                if len(error_message) > 0:
                    continue                                     # the stock per-item skip
                reached.append(data["sample_name"])              # runner.predict(data) runs here
            except Exception:
                continue
    except Exception as exc:                                     # logger.warning("Run inference failed: …"); the run goes on and returns normally
        return "swallowed", reached, str(exc)
    return "returned", reached, None


def _entry_loop(rank, P, modes, flip=False):
    """item_of_rank0 driven through upstream's loop shape on this rank; `flip` perturbs rank 1's digest (a rank holding different bytes)."""
    calls = []
    if flip and rank == 1:
        real = RD.digest_features
        rowpair.RKD.digest_features = lambda feats, exclude=(): (lambda d: d._replace(digest=("1" if d.digest[0] == "0" else "0") + d.digest[1:]))(real(feats, exclude)) if tp.world()[1] == 1 else real(feats, exclude)
    try:
        try:
            how, reached, exc = _upstream(_SeqDataset(modes), lambda ds, i: rowpair.item_of_rank0(ds, i, _stock_getitem_of(calls)))
        except SystemExit as e:                                  # the verb: rc = e.code, the EXIT line
            return "exit", e.code, calls, None
        return how, exc, calls, reached
    finally:
        if flip and rank == 1:
            rowpair.RKD.digest_features = real


@pytest.fixture
def ranks(monkeypatch):
    """Two threaded ranks over an entry (default: rowpair.replicate_inputs on one item); the featurisation-failure callback records
    (rank, item, message)."""
    seen = []
    monkeypatch.setitem(rowpair.STATE, "on_item_error", lambda data, msg: seen.append((tp.world()[1], data.get("sample_name"), msg)))
    monkeypatch.setitem(rowpair.STATE, "calls", {})
    monkeypatch.setitem(rowpair.STATE, "last_item", {})
    monkeypatch.setitem(rowpair.STATE, "items", 0)                      # the process-level item count, restored after the threaded ranks bumped it
    monkeypatch.setitem(rowpair.STATE, "device", None)
    monkeypatch.setitem(rowpair.STATE, "P", 2)                          # item_of_rank0's own P (a rank process records it at init_rank)
    monkeypatch.setitem(tp.STATE, "msa_hold", {})

    def go(mode, msa_host=None, entry=_entry, **kw):
        if msa_host is None:
            monkeypatch.delenv("ROWPAIR_MSA_HOST", raising=False)
        else:
            monkeypatch.setenv("ROWPAIR_MSA_HOST", msa_host)
        return run_ranks(2, entry, mode, timeout_s=120, **kw), seen
    return go


def _feats_lines(capfd):
    """The `ROWPAIR event=feats` / `event=msa_host_hold` lines both rank threads printed (split at the kit prefix: two threads' lines may share
    one physical line)."""
    err = capfd.readouterr().err
    chunks = [" ".join(c.split()) for c in err.split(R.PREFIX)]
    return [c for c in chunks if c.startswith("ROWPAIR event=feats ")], [c for c in chunks if c.startswith("ROWPAIR event=msa_host_hold ")]


def _refused_and_census(capfd):
    """The kit's `ROWPAIR event=refused` chunks and the core's `[feats] rank <r> digest …` census lines of both rank threads."""
    import re
    err = capfd.readouterr().err
    chunks = [" ".join(c.split()) for c in err.split(R.PREFIX)]
    return [c for c in chunks if c.startswith("ROWPAIR event=refused ")], re.findall(r"\[feats\] rank \d digest [0-9a-f]+ feats_ranks_equal=\w+", err)


@pytest.mark.parametrize("msa_host", [None, "rank0"])
def test_rank0_featurises_and_every_rank_holds_its_item(ranks, capfd, msa_host):
    res, seen = ranks("ok", msa_host)
    assert [r[0] for r in res] == ["ok", "ok"] and [r[2] for r in res] == [1, 0], res      # the featuriser ran once, on rank 0 only
    a, b = res[0][1], res[1][1]
    assert a["atom_array"] is ATOM_ARRAY and b["atom_array"] is None and a["error"] == b["error"] == "" and seen == []
    assert a["data"]["sample_name"] == b["data"]["sample_name"] == NAME and int(b["data"]["N_token"]) == 6 and b["data"]["entity_poly_type"] == {"1": "polypeptide(L)"}
    held = set(tp.MSA_HOST_KEYS) if msa_host else set()
    assert set(a["ifd"]) == set(_item()["input_feature_dict"]) and set(b["ifd"]) == set(a["ifd"]) - held        # rank 0 keeps its raw MSA tensors; under rank0 mode they never travel
    for k in b["ifd"]:
        assert torch.equal(a["ifd"][k], b["ifd"][k]) and b["ifd"][k].device.type == "cpu", k
    shared = lambda d: {k: v for k, v in d.items() if k != "input_feature_dict"} | {"ifd": {k: v for k, v in d["input_feature_dict"].items() if k not in held}}
    assert RD.feature_digest(shared(a["data"])) == RD.feature_digest(shared(b["data"]))
    if msa_host:
        assert a["hold"]["mode"] == b["hold"]["mode"] == "rank0" and set(a["hold"]["meta"]) == set(b["hold"]["meta"]) == held
        assert all(torch.equal(a["hold"]["mine"][k], _item()["input_feature_dict"][k]) for k in held) and b["hold"]["mine"] == {}
    else:
        assert a["hold"] is None and b["hold"] is None
    feats, holds = _feats_lines(capfd)
    assert len(feats) == 2 and all(RD.data_form_word(rowpair.DATA_FORM) in ln and RD.agree_word("feats", True) in ln and "rng=carried" in ln and f"item={NAME}" in ln for ln in feats), feats
    assert len({ln.split(" digest=")[1].split()[0] for ln in feats}) == 1                                         # one digest on both ranks
    assert len(holds) == (2 if msa_host else 0)


def test_an_item_whose_featurisation_failed_reaches_every_rank_in_step(ranks, capfd):
    res, seen = ranks("error_item")
    assert [r[0] for r in res] == ["ok", "ok"] and [r[2] for r in res] == [1, 0], res
    a, b = res[0][1], res[1][1]
    assert a["error"].startswith("KeyError") and NAME in b["error"] and "rank 0" in b["error"] and "KeyError: 'ZZZ':" in b["error"] and "Traceback" not in b["error"]   # rank 0's first line travels in the control word
    assert a["data"]["sample_name"] == b["data"]["sample_name"] == NAME and a["atom_array"] is None and b["atom_array"] is None
    assert sorted(seen) == [(0, NAME, a["error"]), (1, NAME, b["error"])]                                            # counted failed on every rank
    feats, _ = _feats_lines(capfd)
    assert len(feats) == 2 and all(f"feats_status={rowpair.ITEM_ERROR}" in ln for ln in feats), feats


def test_a_featuriser_that_raises_on_rank0_refuses_every_rank(ranks):
    res, seen = ranks("raise")
    assert [r[0] for r in res] == ["refused", "refused"] and [r[2] for r in res] == [1, 0], res
    assert all("feats_rank0_failed" in r[1] and "featuriser died" in r[1] for r in res), res
    assert seen == []


@needs_protenix
def test_a_rank_holding_different_bytes_refuses_every_rank_after_naming_the_digests(ranks, capfd):
    res, seen = ranks(["ok"], entry=_entry_loop, flip=True)
    assert [r[:2] for r in res] == [("exit", R.EXIT_NOT_ACTIVE), ("exit", R.EXIT_NOT_ACTIVE)] and [r[2] for r in res] == [[0], []], res
    assert sorted((r, n) for r, n, _ in seen) == [(0, "q0"), (1, "q0")] and all("feats_ranks_differ" in m for _, _, m in seen), seen
    refused, census = _refused_and_census(capfd)
    assert len(refused) == 2 and all("item=q0" in c and "feats_ranks_differ" in c for c in refused), refused
    assert len(census) == 2 and all(c.endswith("feats_ranks_equal=no") for c in census), census      # every rank named its digest before refusing


@needs_protenix
def test_ok_error_ok_in_one_run_stays_in_step(ranks, capfd):
    res, seen = ranks(["ok", "error_item", "ok"], entry=_entry_loop)
    assert [r[:2] for r in res] == [("returned", None), ("returned", None)], res
    assert [r[2] for r in res] == [[0, 1, 2], []]                                                     # the featuriser: three calls on rank 0, none on rank 1
    assert [r[3] for r in res] == [["q0", "q2"], ["q0", "q2"]]                                        # both good items reach predict on both ranks; the error item is skipped on both
    assert sorted((r, n) for r, n, _ in seen) == [(0, "q1"), (1, "q1")]
    assert rowpair.STATE["calls"] == {0: 3, 1: 3}                                                     # one rendezvous key per call, equal across ranks
    feats, _ = _feats_lines(capfd)
    assert len(feats) == 6 and sum(f"feats_status={rowpair.ITEM_ERROR}" in c for c in feats) == 2, feats


@needs_protenix
@pytest.mark.parametrize("modes, word", [(["raise", "ok"], "feats_rank0_failed")])
def test_a_refusal_inside_getitem_ends_the_rank_nonzero_though_upstream_swallows_exceptions(ranks, capfd, modes, word):
    res, seen = ranks(modes, entry=_entry_loop)
    assert [r[:2] for r in res] == [("exit", R.EXIT_NOT_ACTIVE), ("exit", R.EXIT_NOT_ACTIVE)], res     # not "returned"/"swallowed": the rank ends non-zero
    assert [r[2] for r in res] == [[0], []]                                                            # the item after the refused one is never fetched
    assert sorted((r, n) for r, n, _ in seen) == [(0, "q0"), (1, "q0")] and all(word in m for _, _, m in seen), seen   # an ITEM failed record on every rank
    refused, _ = _refused_and_census(capfd)
    assert len(refused) == 2 and all(f"item=q0" in c and word in c for c in refused), refused


class _Dataset:
    inputs = [{"name": NAME}]


@needs_protenix
def test_the_pairing_guards_refuse_by_name_before_any_collective(monkeypatch, capfd):
    import importlib
    import torch.utils.data as tud
    seen = []
    monkeypatch.setitem(rowpair.STATE, "P", 2)
    monkeypatch.setitem(rowpair.STATE, "on_item_error", lambda data, msg: seen.append((data.get("sample_name"), msg)))
    stock_calls = []
    stock = lambda ds, i: stock_calls.append(i)
    dw = importlib.import_module(rowpair.DIST_MODULE).DIST_WRAPPER
    monkeypatch.setattr(dw, "world_size", 2)
    with pytest.raises(SystemExit) as ei:
        rowpair.item_of_rank0(_Dataset(), 0, stock)
    assert ei.value.code == R.EXIT_NOT_ACTIVE and "sampler's world is 2" in seen[-1][1] and seen[-1][0] == NAME
    monkeypatch.setattr(dw, "world_size", 1)
    monkeypatch.setattr(tud, "get_worker_info", lambda: object())
    with pytest.raises(SystemExit) as ei:
        rowpair.item_of_rank0(_Dataset(), 0, stock)
    assert ei.value.code == R.EXIT_NOT_ACTIVE and "DataLoader worker" in seen[-1][1]
    assert stock_calls == [] and len(seen) == 2
    refused, _ = _refused_and_census(capfd)
    assert len(refused) == 2 and all(f"item={NAME}" in c for c in refused), refused


def test_predict_admits_only_the_item_the_rank_replicated_last(monkeypatch):
    monkeypatch.setitem(rowpair.STATE, "P", 2)
    monkeypatch.setitem(rowpair.STATE, "last_item", {})
    monkeypatch.setattr(tp, "host_side_inputs", lambda feats, configs=None: {"stub": True})
    item = _item()
    with pytest.raises(RowpairRefused, match="rowpair_item_unreplicated"):                            # nothing replicated on this rank: the item came around the line's __getitem__
        rowpair.host_side_inputs(item)
    monkeypatch.setitem(rowpair.STATE, "last_item", {0: (NAME, 0)})                                  # what replicate_inputs stamps (rank 0 of no group)
    assert rowpair.host_side_inputs(item) is item and rowpair.STATE["host_facts"] == {"stub": True}
    other = dict(item, sample_name="q9")
    with pytest.raises(RowpairRefused, match="rowpair_item_unreplicated.*q9#0.*q1#0"):
        rowpair.host_side_inputs(other)


def test_p1_is_the_stock_statement(monkeypatch):
    monkeypatch.setitem(rowpair.STATE, "P", 1)
    triple = (_item(), ATOM_ARRAY, "")
    assert rowpair.item_of_rank0(_Dataset(), 0, lambda ds, i: triple) is triple
    assert rowpair.replicate_inputs(lambda: triple, NAME, 0) is triple                    # no group: the featuriser's triple, untouched
