"""Seam 4's raw-MSA placement (boltz2_opt.rowpair_msa on opt_core.mem.rowpair.msa_host): the host placement of the MSA module's input
statements is PLACEMENT ONLY — the module's fp32 input rows and mask rows are bit-identical to the stock device placement, with and without the
per-cycle subsample, at P = 1 (mode ``all``) and at P = 2 (mode ``rank0``: rank 1 holds zero-row placeholders and receives the selected rows by
broadcast; threaded ranks, the core's test hub); the batch hook keeps exactly ``HOST_KEYS`` on the host and moves everything else; the census
words are recorded; a host word with the raw features on the device is refused by name. CPU, fp32; the boltz 2.2.1 model tree and opt_core's
``msa_host`` are required (skipped by name otherwise). The end-to-end leg (the born-sharded trunk under ``rank0`` / ``all`` equal to the dense
trunk) is ``test_rowpair_seams.py::test_born_sharded_trunk_equals_dense_and_census[...-rank0|all]``.
"""
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("boltz.model.modules.trunkv2", reason="the pure-python boltz 2.2.1 model tree is required (pip install --no-deps boltz==2.2.1 einops)")
import opt_core.mem.rowpair.msa_host  # noqa: E402,F401  (the pinned core; an older core FAILS here, never skips)

from boltz2_opt import rowpair, rowpair_msa  # noqa: E402
from boltz2_opt.tests.test_rowpair_seams import build, DIMS  # noqa: E402


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    monkeypatch.delenv("ROWPAIR_MSA_HOST", raising=False)
    monkeypatch.delenv("ROWPAIR_MSA_M_LAYOUT", raising=False)
    from opt_core.mem.rowpair import evidence
    evidence.reset_schedule()
    yield
    rowpair_msa.reset_for_tests()
    rowpair.reset_for_tests()
    evidence.reset_schedule()


def _reference(model, feats, seed: int):
    """The device placement (stock order: one-hot / cat over all rows, then the draw and the selection)."""
    torch.manual_seed(seed)
    return rowpair_msa.msa_input(model.msa_module, dict(feats), torch.device("cpu"), {})


@pytest.mark.parametrize("subsample", [True, False])
def test_host_placement_equals_device_placement_P1(monkeypatch, subsample):
    from opt_core.mem.rowpair import evidence, msa_host as MH
    model, feats = build(24, subsample=subsample)
    m_ref, mask_ref = _reference(model, feats, 5)
    S, n = DIMS["S"], (DIMS["S_sub"] if subsample else DIMS["S"])
    assert tuple(m_ref.shape) == (1, n, 24, 33 + 3) and m_ref.dtype == torch.float32 and tuple(mask_ref.shape) == (1, n, 24)
    sched = evidence.schedule()
    assert sched["msa_host"] == "off" and sched["msa_rows"] == f"{n}/{S}" and sched["msa_subsample"] == int(subsample)
    monkeypatch.setenv("ROWPAIR_MSA_HOST", "all")
    hf = dict(feats)
    facts = MH.park_features(hf, rowpair_msa.HOST_KEYS, row_dims=rowpair_msa.ROW_DIM)      # what the batch hook does (P = 1: every key a host copy)
    assert facts["mode"] == "all" and sorted(facts["parked"]) == sorted(rowpair_msa.HOST_KEYS) and all(MH.is_parked(hf[k]) for k in rowpair_msa.HOST_KEYS)
    torch.manual_seed(5)
    guards = {}
    m, mask = rowpair_msa.msa_input(model.msa_module, hf, torch.device("cpu"), guards)
    assert torch.equal(m, m_ref) and torch.equal(mask, mask_ref)                          # placement only: bit-identical input rows and mask rows
    assert guards.get("replicated_checked", 0) == (2 if subsample else 1)                 # the msa_mask guard (once per batch) + the draw's
    sched = evidence.schedule()
    assert sched["msa_host"] == "all" and sched["msa_rows"] == f"{n}/{S}" and sched["msa_subsample"] == int(subsample) and sched["msa_host_cycles"] == 1
    assert sched["msa_host_cycle_gib"] >= 0 and str(sched["msa_host_rows"]).startswith("all:1x")   # the core's per-tensor word + this seam's per-cycle GiB
    rep = rowpair_msa.report()
    assert rep["mode"] == "all" and rep["cycles"] == 1 and rep["host_keys"] == list(rowpair_msa.HOST_KEYS)


def test_host_word_with_device_features_refused_by_name(monkeypatch):
    model, feats = build(16, subsample=True)
    monkeypatch.setenv("ROWPAIR_MSA_HOST", "rank0")

    class _OnDevice:                                                                        # a stand-in tensor that says it stands on a CUDA device
        is_cuda = True
        device = "cuda:0"
    hf = dict(feats); hf["msa"] = _OnDevice()
    monkeypatch.setattr(torch, "is_tensor", lambda t, _o=torch.is_tensor: True if isinstance(t, _OnDevice) else _o(t))
    with pytest.raises(rowpair_msa.Refused, match="ROWPAIR_MSA_HOST=rank0 but feats\\['msa'\\] stands on cuda:0"):
        rowpair_msa.msa_input(model.msa_module, hf, torch.device("cpu"), {})


def test_batch_hook_keeps_host_keys_and_moves_the_rest(monkeypatch):
    from opt_core.mem.rowpair import msa_host as MH
    moved = []

    class FakeDM:                                                                           # Boltz2InferenceDataModule's two hooks (inferencev2.py:369-433): the batch source and the H2D
        def predict_dataloader(self):
            raise AssertionError("not iterated here")

        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            for key in batch:
                if key not in ["record"]:
                    moved.append(key); batch[key] = batch[key].to(device)
            return batch
    fake = types.ModuleType("fake_inferencev2"); fake.Boltz2InferenceDataModule = FakeDM
    rowpair_msa._bind(fake)
    assert FakeDM.transfer_batch_to_device is rowpair_msa._transfer_batch_to_device and rowpair_msa.report()["hook_bound"]
    _, feats = build(16)
    batch = dict(feats); batch["record"] = object()
    out = FakeDM().transfer_batch_to_device(batch, torch.device("cpu"), 0)                 # no host word: the stock hook verbatim
    assert set(moved) == set(feats) and rowpair_msa.report()["parked"] is None
    moved.clear()
    monkeypatch.setenv("ROWPAIR_MSA_HOST", "all")
    batch = dict(feats); batch["record"] = object()
    out = FakeDM().transfer_batch_to_device(batch, torch.device("cpu"), 0)
    assert set(moved) == set(feats) - set(rowpair_msa.HOST_KEYS)                           # the raw MSA keys never went through .to(device)
    assert all(k in out and MH.is_parked(out[k]) for k in rowpair_msa.HOST_KEYS) and "msa_mask" in moved and "record" in out
    parked = rowpair_msa.report()["parked"]
    assert parked["mode"] == "all" and sorted(parked["parked"]) == sorted(rowpair_msa.HOST_KEYS) and parked["placeholders"] == []
    rowpair_msa.reset_for_tests()
    assert FakeDM.transfer_batch_to_device is not rowpair_msa._transfer_batch_to_device    # the original restored


def _rank0_entry(rank: int, P: int, N: int, m_ref, mask_ref):
    """One threaded rank under ROWPAIR_MSA_HOST=rank0 (the module's subsample off: the threaded ranks share the process's CPU generator, so the
    draw is exercised by the process ranks of test_rowpair_seams): the batch parking (rank 1: zero-row placeholders) and the host placement —
    rows equal to the device-placement reference ``m_ref`` (computed by the caller, no word set) on every rank, received by broadcast on rank 1."""
    from opt_core.mem.rowpair import msa_host as MH
    torch.set_num_threads(1)
    model, feats = build(N, subsample=False)
    hf = dict(feats)
    facts = MH.park_features(hf, rowpair_msa.HOST_KEYS, mode="rank0", row_dims=rowpair_msa.ROW_DIM)
    if rank == 0:
        assert sorted(facts["parked"]) == sorted(rowpair_msa.HOST_KEYS) and facts["placeholders"] == []
    else:
        assert sorted(facts["placeholders"]) == sorted(rowpair_msa.HOST_KEYS) and all(tuple(hf[k].shape) == (1, 0, N) and hf[k].dtype == feats[k].dtype for k in rowpair_msa.HOST_KEYS)
    guards = {}
    m, mask = rowpair_msa.msa_input(model.msa_module, hf, torch.device("cpu"), guards)
    return {"rank": rank, "equal_m": bool(torch.equal(m, m_ref)), "equal_mask": bool(torch.equal(mask, mask_ref)), "shape": tuple(m.shape),
            "guards": dict(guards), "placeholders": list(facts["placeholders"])}


def test_rank0_placement_two_ranks_broadcast_equals_device(monkeypatch):
    from opt_core import testing
    from opt_core.mem.rowpair import evidence
    N = 20
    model, feats = build(N, subsample=False)
    m_ref, mask_ref = rowpair_msa.msa_input(model.msa_module, dict(feats), torch.device("cpu"), {})   # the device placement (no word set)
    assert evidence.schedule()["msa_host"] == "off"
    monkeypatch.setenv("ROWPAIR_MSA_HOST", "rank0")
    outs = testing.run_ranks(2, _rank0_entry, N, m_ref, mask_ref, timeout_s=120)
    for o in outs:
        assert o["equal_m"] and o["equal_mask"] and o["shape"] == (1, DIMS["S"], N, 36), o
        assert o["guards"].get("replicated_checked") == 1, o                             # the msa_mask guard, once per batch on every rank
    assert outs[0]["placeholders"] == [] and sorted(outs[1]["placeholders"]) == sorted(rowpair_msa.HOST_KEYS)
    assert evidence.schedule()["msa_host"] == "rank0" and str(evidence.schedule()["msa_host_rows"]).startswith("rank0:")


def test_the_line_exports_rank0_and_the_report_names_it():
    from boltz2_opt import modes, stack
    assert modes.TP_EXPORTS["ROWPAIR_MSA_HOST"] == "rank0"
    assert stack.child_env("big", {}, n_gpu=2)["ROWPAIR_MSA_HOST"] == "rank0" and stack.child_env("big", {"ROWPAIR_MSA_HOST": "0"}, n_gpu=2)["ROWPAIR_MSA_HOST"] == "rank0"   # the row's alone: a caller's copy is stripped
    assert "ROWPAIR_MSA_HOST" not in stack.child_env("big", {}, n_gpu=1)
    assert set(rowpair.report()["tp_exports"]) == set(modes.TP_EXPORTS) and rowpair.report()["msa_host"]["mode"] == "off"


def test_m_layout_word(monkeypatch):
    """``ROWPAIR_MSA_M_LAYOUT``: token_sharded when unset (the default) or set; replicated when set; any other word refused by name."""
    assert rowpair_msa.m_layout() == "token_sharded"
    monkeypatch.setenv("ROWPAIR_MSA_M_LAYOUT", "replicated")
    assert rowpair_msa.m_layout() == "replicated"
    monkeypatch.setenv("ROWPAIR_MSA_M_LAYOUT", "token_sharded")
    assert rowpair_msa.m_layout() == "token_sharded"
    for bad in ("0", "1", "sharded"):
        monkeypatch.setenv("ROWPAIR_MSA_M_LAYOUT", bad)
        with pytest.raises(rowpair_msa.Refused, match="ROWPAIR_MSA_M_LAYOUT"):
            rowpair_msa.m_layout()


@pytest.mark.parametrize("subsample", [True, False])
@pytest.mark.parametrize("host", ["", "all"])
def test_column_block_equals_columns_of_whole_rows(monkeypatch, subsample, host):
    """The token-sharded layout's input statement (``msa_input(cols=(c0, c1))``): the column block's rows are bit-identical to columns ``c0:c1``
    of the whole rows under the same draw (column selection commutes with the one-hot / cat and the row selection), on the device placement and
    on the host placement; the mask rows stay WHOLE (the outer-product mean's pair count reads every column); the census names the columns."""
    from opt_core.mem.rowpair import evidence, msa_host as MH
    N, (c0, c1) = 24, (8, 20)
    model, feats = build(N, subsample=subsample)
    m_ref, mask_ref = _reference(model, feats, 7)
    hf = dict(feats)
    if host:
        monkeypatch.setenv("ROWPAIR_MSA_HOST", host)
        MH.park_features(hf, rowpair_msa.HOST_KEYS, row_dims=rowpair_msa.ROW_DIM)
    torch.manual_seed(7)
    m, mask = rowpair_msa.msa_input(model.msa_module, hf, torch.device("cpu"), {}, cols=(c0, c1))
    assert tuple(m.shape) == (1, m_ref.shape[1], c1 - c0, m_ref.shape[3]) and torch.equal(m, m_ref[:, :, c0:c1]) and torch.equal(mask, mask_ref)
    sched = evidence.schedule()
    assert sched["msa_cols"] == f"{c0}:{c1}/{N}" and sched["msa_host"] == (host or "off")
    torch.manual_seed(7)
    m_all, _ = rowpair_msa.msa_input(model.msa_module, hf, torch.device("cpu"), {})
    assert torch.equal(m_all, m_ref) and evidence.schedule()["msa_cols"] == "all"
    with pytest.raises(rowpair_msa.Refused, match="cols"):
        rowpair_msa.msa_input(model.msa_module, hf, torch.device("cpu"), {}, cols=(20, 8))
