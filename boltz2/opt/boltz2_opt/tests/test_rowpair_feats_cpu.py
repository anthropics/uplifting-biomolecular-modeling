"""The ×P line's input path (boltz2_opt.rowpair_msa + boltz2_opt.prep, n_gpu > 1, ``data_form=rank0_bcast``): rank 0 parses and featurizes, every
other rank receives — asserted on two gloo ranks (the core's launcher, one process per rank), and solo (n_gpu = 1: everything is stock's).

  source   ``Boltz2InferenceDataModule.predict_dataloader`` rebound: rank 0 iterates stock's DataLoader and every batch goes to the other ranks
           (the core's rankdata.broadcast_features); a rank > 0 builds no DataLoader and calls no featurizer; both ranks end after the manifest's
           records; a featurizer exception on rank 0 is ``refused: feats_rank0_failed`` on EVERY rank; the ranks' CPU generator states stay equal.
  census   the H2D hook digests the batch on every rank (rankdata.digest_features; ``record`` is not a feature): one
           ``[boltz2-opt] [feats] rank <r> digest <hex16> feats_ranks_equal=yes|no ranks=<P> digests=…`` line per batch per rank, the LEVER line's
           ``entry_inputs_equal=<equal>/<checked>``, and digests that differ are ``refused: feats_ranks_differ`` on every rank.
  parse    prep: rank 0's parse result and processed directory go to the other ranks; a rank > 0 links its ``<out_dir>/processed`` to rank 0's
           and returns rank 0's result verbatim; a directory a rank cannot read is ``refused: rank0_processed_dir_not_visible`` on every rank;
           under the launcher's rank environment a rank > 0 gets a Receiver (no zygote is forked), rank 0 a Zygote.

Skipped by name without torch or without the core's rank-data contract (``opt_core.mem.rowpair.rankdata.broadcast_features``)."""
import json
import os
import re
import types

import pytest


@pytest.fixture(autouse=True)
def _cpu_ranks_only(monkeypatch):
    """The spawned rank processes see no CUDA device (gloo on the CPU whatever the box carries)."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    yield


torch = pytest.importorskip("torch")
rankdata = pytest.importorskip("opt_core.mem.rowpair.rankdata", reason="the core's rank-data module (opt_core.mem.rowpair.rankdata) is required")
if not hasattr(rankdata, "broadcast_features"):
    pytest.skip("the core's rank-data contract (rankdata.broadcast_features) is required", allow_module_level=True)

from opt_core.mem.rowpair import msa_host  # noqa: E402

from boltz2_opt import prep, rowpair, rowpair_msa  # noqa: E402

HOST_ENV = msa_host.ENV_MSA_HOST                                       # the placement word of the raw MSA features (ROWPAIR_MSA_HOST)
N, S, A = 24, 6, 90                                                     # tokens, MSA rows, atoms: small — the digest is over bytes, sizes do not matter


def make_batch(record_word: str = "r", shift: float = 0.0) -> dict:
    """A featurizer batch in boltz's collated form: stacked tensors ``[1, …]``, list-valued entries (``all_coords``: a list of tensors;
    ``record``: the manifest records, not features), identical bytes on every call (``shift`` moves one element of ``ref_pos``)."""
    g = torch.Generator().manual_seed(7)
    b = {"msa": torch.randint(0, 33, (1, S, N), generator=g), "has_deletion": (torch.rand(1, S, N, generator=g) < 0.1).float(),
         "deletion_value": torch.rand(1, S, N, generator=g), "msa_paired": (torch.rand(1, S, N, generator=g) < 0.5).float(),
         "msa_mask": torch.ones(1, S, N), "token_pad_mask": torch.ones(1, N), "token_index": torch.arange(N)[None],
         "ref_pos": torch.randn(1, A, 3, generator=g), "atom_pad_mask": torch.ones(1, A), "atom_resolved_mask": torch.ones(1, A, dtype=torch.bool),
         "all_coords": [torch.randn(A, 3, generator=g)], "affinity_mw": [412.5], "record": [f"{record_word}-record"]}
    b["ref_pos"][0, 0, 0] += shift
    return b


class FakeDataset(torch.utils.data.Dataset):
    """One record's features, as boltz's PredictionDataset hands them (``shift`` makes this rank's differ; ``boom`` makes the featurizer raise);
    counts its calls in a file so the parent can see a rank that never featurized."""
    def __init__(self, shift, boom, calls_file):
        self.shift, self.boom, self.calls_file = shift, boom, calls_file

    def __len__(self):
        return 1

    def __getitem__(self, i):
        open(self.calls_file, "a").write("x")
        if self.boom:
            raise ValueError("featurizer boom")
        b = make_batch(shift=self.shift)
        return {k: (v[0] if torch.is_tensor(v) else v) for k, v in b.items()}


def _collate(items):
    (it,) = items
    return {k: (v[None] if torch.is_tensor(v) else v) for k, v in it.items()}


class FakeDM:
    """The shape of boltz's ``Boltz2InferenceDataModule`` (inferencev2.py:314-433): ``manifest.records``, ``predict_dataloader`` (a DataLoader,
    batch 1, no shuffle), ``transfer_batch_to_device`` (tensors move, the listed keys stay)."""
    def __init__(self, shift=0.0, boom=False, calls_file=os.devnull):
        self.manifest = types.SimpleNamespace(records=["rec"])
        self.dataset = FakeDataset(shift, boom, calls_file)

    def predict_dataloader(self):
        return torch.utils.data.DataLoader(self.dataset, batch_size=1, num_workers=0, shuffle=False, collate_fn=_collate)

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        for key in batch:
            if key not in ["all_coords", "all_resolved_mask", "crop_to_all_atom_map", "chain_symmetries", "amino_acids_symmetries", "ligand_symmetries", "record", "affinity_mw"]:
                batch[key] = batch[key].to(device)
        return batch


def _bind():
    fake = types.ModuleType("fake_inferencev2"); fake.Boltz2InferenceDataModule = FakeDM
    rowpair_msa._bind(fake)


def _rng_digest():
    import hashlib
    return hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest()[:16]


# ----------------------------------------------------------------------------------------------------- rank bodies (rank 0's return value is the run's; every body all-gathers)
def _source_entry(tmp, boom):
    """One item through the worker's statements: dl = dm.predict_dataloader(); it_ = iter(dl); batch = next(it_); batch = H2D(batch); next(it_) ends.
    Rank 1's dataset would featurize DIFFERENT bytes (shift) — it is never called; rank 1 holds rank 0's batch."""
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    torch.set_num_threads(1)
    cm = D.comm()
    _bind()
    calls = os.path.join(tmp, f"featurizer_calls.rank{cm.rank}")
    dm = FakeDM(shift=float(cm.rank), boom=boom, calls_file=calls)
    torch.manual_seed(11)
    dl = dm.predict_dataloader(); it_ = iter(dl)
    rng_after_iter = _rng_digest()
    mine = {"rank": cm.rank, "source_type": type(dl).__name__, "len": len(dl), "rng_after_iter": rng_after_iter}
    try:
        batch = next(it_)
        batch = dm.transfer_batch_to_device(batch, torch.device("cpu"), 0)
        mine.update(digest=rankdata.feature_digest(batch, exclude=rowpair_msa.NON_FEATURE_KEYS), refused=None,
                    ended=(next(it_, "END") == "END"), feats=rowpair_msa.report()["feats"], source=rowpair_msa.report()["source"], guards=dict(rowpair.report()["guards"]))
    except RowpairRefused as e:
        mine.update(refused=str(e), cause=repr(e.__context__) if e.__context__ is not None else None)
    mine["featurizer_calls"] = len(open(calls).read()) if os.path.exists(calls) else 0
    return cm.allgather_obj(mine)


def _census_entry(host_word_unused, flip):
    """One batch straight into the bound H2D on each rank (no source): rank 1's differs by one element when ``flip``."""
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    torch.set_num_threads(1)
    cm = D.comm()
    _bind()
    batch = make_batch(record_word=f"rank{cm.rank}", shift=1.0 if (flip and cm.rank == 1) else 0.0)
    expect = rankdata.feature_digest(batch, exclude=rowpair_msa.NON_FEATURE_KEYS)
    mine = {"rank": cm.rank, "expect": expect}
    try:
        out = FakeDM().transfer_batch_to_device(batch, torch.device("cpu"), 0)
        mine.update(refused=None, parked=rowpair_msa.report()["parked"], msa_rows_on_rank=int(out["msa"].shape[1]))
    except RowpairRefused as e:
        mine.update(refused=str(e))
    mine.update(facts=rowpair_msa.report()["feats"], guards=dict(rowpair.report()["guards"]))
    return cm.allgather_obj(mine)


def _parse_entry(tmp, case):
    """prep's two sides: rank 0 shares a parse result + processed dir (a real one written here, a missing one, or a failed parse); rank 1 receives."""
    from opt_core.mem.rowpair import RowpairRefused, dist as D
    cm = D.comm()
    out0, out1 = os.path.join(tmp, "rank0", "boltz_results_x"), os.path.join(tmp, "rank1", "boltz_results_x")
    mine = {"rank": cm.rank}
    try:
        if cm.rank == 0:
            pdir = os.path.join(out0, "processed")
            if case == "ok":
                os.makedirs(os.path.join(pdir, "structures")); json.dump({"records": [{"id": "x"}]}, open(os.path.join(pdir, "manifest.json"), "w"))
            res = prep.share_parse(prep.Result(0 if case != "failed" else 1, "stock words\n", "" if case != "failed" else "Traceback…\n"), out0)
            mine.update(rc=res.rc, out=res.out, err=res.err)
        else:
            os.makedirs(out1)
            rx = prep._new()
            mine.update(kind=type(rx).__name__, zygote_pid=rx.describe()["zygote_pid"])
            res = rx.parse("x.yaml", out1, "ccd.pkl", "mols")
            link = os.path.join(out1, "processed")
            mine.update(rc=res.rc, out=res.out, err=res.err, islink=os.path.islink(link), target=os.readlink(link) if os.path.islink(link) else None,
                        manifest_seen=os.path.isfile(os.path.join(link, "manifest.json")), describe=rx.describe())
        mine["refused"] = None
    except RowpairRefused as e:
        mine["refused"] = str(e)
    return cm.allgather_obj(mine)


def _run(entry, *args, tmp_path):
    from opt_core.mem.rowpair import launch
    return launch.run_sharded(2, entry, *args, mode=launch.MEMORY_MODE, backend="gloo", cpu_ok=True, run_timeout_s=600, log_dir=str(tmp_path))


# ----------------------------------------------------------------------------------------------------- the source: rank 0 featurizes, rank 1 receives
def test_rank0_featurizes_and_the_other_rank_receives_its_batch(tmp_path, monkeypatch):
    monkeypatch.delenv(HOST_ENV, raising=False)
    per_rank = _run(_source_entry, str(tmp_path), False, tmp_path=tmp_path)
    r0, r1 = per_rank
    assert [r0["rank"], r1["rank"]] == [0, 1] and r0["refused"] is None and r1["refused"] is None, per_rank
    assert r0["source_type"] == "_Rank0Batches" and r1["source_type"] == "_ReceivedBatches" and r0["len"] == r1["len"] == 1
    assert r0["featurizer_calls"] == 1 and r1["featurizer_calls"] == 0                        # rank 1 never featurized
    assert r0["digest"] == r1["digest"] == rankdata.feature_digest(make_batch(), exclude=rowpair_msa.NON_FEATURE_KEYS)   # rank 1 holds rank 0's bytes, not its own featurizer's (shift 1.0)
    assert r0["ended"] is True and r1["ended"] is True                                          # both sources end after the manifest's one record
    assert r0["rng_after_iter"] == r1["rng_after_iter"]                                       # iter() drew base_seed on rank 0; the receiver drew the same
    assert r0["source"]["role"] == "featurize" and r1["source"]["role"] == "receive" and r0["source"]["form"] == r1["source"]["form"] == "rank0_bcast" == rowpair_msa.FEATS_FORM
    assert r0["source"]["status"] == r1["source"]["status"] == "ok" and r1["source"]["tensors"] == r0["source"]["tensors"] >= 10 and r1["source"]["bytes"] == r0["source"]["bytes"] > 0
    assert set(r0["source"]["msa_keys"]) == {"msa", "has_deletion", "deletion_value", "msa_paired", "msa_mask"} and r0["source"]["msa_keys"]["msa"]["shape"] == [1, S, N]
    for m in per_rank:                                                                          # the census over the received batch: equal, counted, real
        assert m["feats"]["equal"] is True and m["feats"]["form"] == "rank0_bcast" and m["guards"]["inputs_checked"] == 1 == m["guards"]["inputs_equal"]
    log1 = open(os.path.join(str(tmp_path), "rank1.log"), errors="replace").read()
    want1 = f"[boltz2-opt] [feats] rank 1 digest {r1['digest'][:16]} feats_ranks_equal=yes ranks=2 digests={r0['digest'][:16]},{r1['digest'][:16]}"
    assert want1 in log1.splitlines(), log1[-2000:]
    from boltz2_opt import worker                                                              # pred relays the census line to the caller's transcript (worker.RELAY_LINES)
    assert worker.relay_lines(f"x\n{want1}\ny\n", []) == [want1]


def test_a_featurizer_error_on_rank0_fails_every_rank_by_name(tmp_path, monkeypatch):
    monkeypatch.delenv(HOST_ENV, raising=False)
    r0, r1 = _run(_source_entry, str(tmp_path), True, tmp_path=tmp_path)
    for m in (r0, r1):
        assert m["refused"] is not None and "refused: feats_rank0_failed:" in m["refused"] and "ValueError" in m["refused"] and "featurizer boom" in m["refused"], m
    assert "featurizer boom" in (r0["cause"] or "")                                            # rank 0's own exception rides under the refusal
    assert r0["featurizer_calls"] == 1 and r1["featurizer_calls"] == 0


# ----------------------------------------------------------------------------------------------------- the census at the H2D: equal digests counted, different digests refused on every rank
@pytest.mark.parametrize("host_word", ["", "rank0"])
@pytest.mark.parametrize("flip", [False, True])
def test_the_h2d_digests_every_ranks_batch_and_refuses_a_difference(tmp_path, monkeypatch, host_word, flip):
    if host_word:
        monkeypatch.setenv(HOST_ENV, host_word)
    else:
        monkeypatch.delenv(HOST_ENV, raising=False)
    per_rank = _run(_census_entry, host_word, flip, tmp_path=tmp_path)
    d0, d1 = per_rank[0]["expect"], per_rank[1]["expect"]
    assert re.fullmatch(r"[0-9a-f]{64}", d0) and (d0 != d1) is flip                       # `record` differs on every rank and is outside the digest; one element of ref_pos is inside it
    for m in per_rank:
        f = m["facts"]
        assert f["digest"] == m["expect"] and f["equal"] is (not flip) and f["excluded"] == ["record"] and f["unhashed"] == [] and f["tensor_leaves"] >= 10
        assert m["guards"]["inputs_checked"] == 1 and m["guards"]["inputs_equal"] == (0 if flip else 1)          # the LEVER line's entry_inputs_equal=<equal>/<checked>
        if flip:                                                                               # refused on EVERY rank, by the core's word
            assert m["refused"] is not None and "refused: feats_ranks_differ:" in m["refused"], m
        else:
            assert m["refused"] is None
            if host_word == "rank0":                                                            # the census read the batch BEFORE the placement: rank 1 holds zero-row placeholders after it
                assert m["parked"]["mode"] == "rank0" and m["msa_rows_on_rank"] == (S if m["rank"] == 0 else 0)
            else:                                                                              # no host word: stock's H2D after the census, every tensor moved whole
                assert m["parked"] is None and m["msa_rows_on_rank"] == S
    log1 = open(os.path.join(str(tmp_path), "rank1.log"), errors="replace").read()
    want1 = f"[boltz2-opt] [feats] rank 1 digest {d1[:16]} feats_ranks_equal={'no' if flip else 'yes'} ranks=2 digests={d0[:16]},{d1[:16]}"
    assert want1 in log1.splitlines(), log1[-2000:]


# ----------------------------------------------------------------------------------------------------- the parse: rank 0 shares, rank 1 receives and links
@pytest.mark.parametrize("case", ["ok", "missing", "failed"])
def test_rank0s_parse_result_and_processed_dir_reach_the_other_rank(tmp_path, monkeypatch, case):
    r0, r1 = _run(_parse_entry, str(tmp_path), case, tmp_path=tmp_path)
    pdir0 = os.path.join(str(tmp_path), "rank0", "boltz_results_x", "processed")
    if case == "missing":                                                                       # rank 1 cannot read rank 0's directory: refused on BOTH ranks, by name
        for m in (r0, r1):
            assert m["refused"] is not None and f"refused: {prep.NOT_VISIBLE}: {pdir0}" in m["refused"] and "rank(s) [0, 1]" in m["refused"], m
        return
    assert r0["refused"] is None and r1["refused"] is None, (r0, r1)
    assert r1["kind"] == "Receiver" and r1["zygote_pid"] is None                              # rank 1 forked no zygote
    assert (r1["rc"], r1["out"], r1["err"]) == (r0["rc"], r0["out"], r0["err"])                 # rank 0's result verbatim (stock's words, a failed parse's traceback)
    if case == "ok":
        assert r1["rc"] == 0 and r1["islink"] and r1["target"] == pdir0 and r1["manifest_seen"]
        assert r1["describe"]["last"]["processed"] == pdir0 and r1["describe"]["data_form"] == "rank0_bcast" and r1["describe"]["calls"] == 1
    else:                                                                                      # a failed parse: the rc travels, nothing is linked; the worker raises on every rank alike
        assert r1["rc"] == 1 and not r1["islink"]


def test_the_launchers_rank_environment_picks_receiver_or_zygote(monkeypatch):
    from opt_core.mem.rowpair import launch
    monkeypatch.setenv(launch.ENV_WORLD, "2"); monkeypatch.setenv(launch.ENV_RANK, "1")
    rx = prep._new()
    assert isinstance(rx, prep.Receiver) and rx.describe()["zygote_pid"] is None and (rx.P, rx.rank) == (2, 1)
    monkeypatch.setenv(launch.ENV_RANK, "0")
    assert isinstance(prep._new(), prep.Zygote)                                              # constructed, not started: no fork here
    monkeypatch.delenv(launch.ENV_WORLD); monkeypatch.delenv(launch.ENV_RANK)
    assert isinstance(prep._new(), prep.Zygote)                                              # n_gpu = 1


# ----------------------------------------------------------------------------------------------------- n_gpu = 1: stock's source, stock's H2D, nothing computed
def test_solo_computes_nothing_and_the_hooks_are_stocks(monkeypatch):
    monkeypatch.delenv(HOST_ENV, raising=False)
    rowpair.reset_for_tests()
    _bind()
    dm = FakeDM()
    dl = dm.predict_dataloader()
    assert isinstance(dl, torch.utils.data.DataLoader)                                      # no group: stock's DataLoader, unwrapped
    batch = next(iter(dl))
    assert rowpair_msa.feats_census(batch) is None
    out = dm.transfer_batch_to_device(batch, torch.device("cpu"), 0)
    assert rowpair_msa.report()["feats"] is None and rowpair_msa.report()["source"] is None
    assert rowpair.report()["guards"]["inputs_checked"] == 0 == rowpair.report()["guards"]["inputs_equal"]
    assert torch.is_tensor(out["msa"]) and isinstance(out["all_coords"], list) and out["record"] == ["r-record"]
    res = prep.share_parse(prep.Result(0, "o", "e"), "/nonexistent")                        # no group: a pass-through, nothing sent
    assert res == prep.Result(0, "o", "e")
    rowpair.reset_for_tests()
    assert FakeDM.predict_dataloader is not rowpair_msa._predict_dataloader and FakeDM.transfer_batch_to_device is not rowpair_msa._transfer_batch_to_device


def test_the_kits_exclusion_leaves_every_feature_in_the_digest():
    """``NON_FEATURE_KEYS`` excludes the manifest records only: ``record`` → the same digest; one value of any feature (a tensor inside a list
    included) → another digest (the digest rule itself is the core's, rankdata.digest_features)."""
    dig = lambda b: rankdata.feature_digest(b, exclude=rowpair_msa.NON_FEATURE_KEYS)   # noqa: E731
    d = dig(make_batch())
    assert d == dig(make_batch()) and re.fullmatch(r"[0-9a-f]{64}", d) and dig(make_batch(record_word="other")) == d
    b = make_batch(); b["deletion_value"][0, 1, 2] += 0.5
    assert dig(b) != d
    b = make_batch(); b["all_coords"][0][3, 1] += 1.0
    assert dig(b) != d
    fd = rankdata.digest_features(make_batch(), exclude=rowpair_msa.NON_FEATURE_KEYS)
    assert fd.unhashed == () and "/affinity_mw/0" in fd.nontensor_leaves                  # the plain values enter canonically; nothing is dropped unnamed


def test_the_census_line_is_relayed_to_the_callers_transcript():
    """pred relays every rank's census line verbatim (worker.RELAY_LINES), rank 0's from the worker transcript, the others' from their logs;
    the relayed form is the core's line under this kit's tag and nothing else."""
    from boltz2_opt import worker
    line = f"{rowpair.TAG} [{rowpair_msa.FEATS_WHAT}] rank 3 digest {'ab' * 8} feats_ranks_equal=no ranks=4 digests={'cd' * 8},{'cd' * 8},{'cd' * 8},{'ab' * 8}"
    assert line.startswith("[boltz2-opt] [feats] rank 3 digest abababab")
    assert worker.relay_lines(f"x\n{line}\n[boltz2-opt r3] {line[len(rowpair.TAG) + 1:]}\ny\n", []) == [line]      # the core comm's own rank-prefixed form is not the kit's line


def test_a_source_object_serves_one_pass_and_refuses_a_second_iter():
    """The rank0_bcast batch source serves the worker's in-process predict loop (--pipeline 1): one iter() per source object; a second iter()
    (a caller that re-iterates the loader — its second pass would draw a second base_seed on rank 0 only) is refused by name on both kinds,
    before any collective."""
    from opt_core.mem.rowpair import RowpairRefused
    r0 = rowpair_msa._Rank0Batches(torch.utils.data.DataLoader(FakeDataset(0.0, False, os.devnull), batch_size=1, collate_fn=_collate))
    iter(r0)
    with pytest.raises(RowpairRefused, match="refused: feats source iterated twice"):
        iter(r0)
    rx = rowpair_msa._ReceivedBatches(1)
    torch.manual_seed(3); before = _rng_digest()
    iter(rx)
    assert _rng_digest() != before                                                          # the one base_seed draw
    with pytest.raises(RowpairRefused, match="refused: feats source iterated twice"):
        iter(rx)
    assert len(r0) == 1 == len(rx)


def test_prep_reset_restarts_the_store_keys_and_a_self_link_is_left_alone(tmp_path):
    prep._CALLS["parse"] = 5
    prep.reset_for_tests()
    assert prep._parse_key() == "parse/1" and prep._parse_key() == "parse/2"
    prep.reset_for_tests()
    out0 = tmp_path / "boltz_results_x"; pdir = out0 / "processed"; pdir.mkdir(parents=True)
    assert prep._link_processed(str(out0), str(pdir)) == (None, None) and pdir.is_dir() and not pdir.is_symlink()     # a rank on rank 0's out_dir: nothing renamed, nothing linked
    out1 = tmp_path / "rank1" / "boltz_results_x"; (out1 / "processed").mkdir(parents=True)                            # a real directory of this rank's own: set aside by name, then linked
    link, aside = prep._link_processed(str(out1), str(pdir))
    assert link == str(out1 / "processed") and os.readlink(link) == str(pdir) and aside == f"{out1 / 'processed'}.rank_local.0" and os.path.isdir(aside)
    link2, aside2 = prep._link_processed(str(out1), str(pdir))                                                          # a symlink already there: re-pointed, nothing set aside
    assert link2 == link and aside2 is None
