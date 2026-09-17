"""The tp line's data form (rank0_bcast | per_rank): the variable's parsing and refusal, the light-record dataset a rank > 0 registers
under rank0_bcast (upstream's own record fields, no featurisation; the registry restored on uninstall; the class resolvable by name for
DataLoader workers of any start method), and the launcher's `data_form=` word / refusal before any rank starts."""
import os
import pickle

import pytest

from openfold3_ob0_opt.tp_rowpair import model as TPM


def test_data_form_default_per_rank_and_refusal():
    from opt_core.mem.rowpair import RowpairRefused
    assert TPM.data_form({}) == "rank0_bcast" == TPM.DATA_FORM_DEFAULT
    assert TPM.data_form({"OF3TP_DATA_FORM": "per_rank"}) == "per_rank" and TPM.data_form({"OF3TP_DATA_FORM": " rank0_bcast "}) == "rank0_bcast"
    with pytest.raises(RowpairRefused) as ei:
        TPM.data_form({"OF3TP_DATA_FORM": "row_born"})
    assert "data_form='row_born' is not one of rank0_bcast|per_rank" in ei.value.reason and "OF3TP_DATA_FORM='row_born'" in ei.value.reason


def test_rank_stub_record_is_upstreams_light_record():
    torch = pytest.importorskip("torch")
    pd = pytest.importorskip("pandas")
    from openfold3_ob0_opt.tp_rowpair import data as DATA

    class _DS:
        datapoint_cache = pd.DataFrame([{"query_id": "q7", "seed": 101, "repeated_sample": False}])

    rec = DATA.rank_stub_record(_DS(), 0)
    assert set(rec) == {"query_id", "seed", "repeated_sample", "valid_sample", DATA.RANK_STUB_FLAG}
    assert rec["query_id"] == "q7" and rec["seed"].tolist() == [101] and rec["repeated_sample"].dtype == torch.bool and rec["valid_sample"].tolist() == [True]


def test_rank_stub_dataset_registered_and_restored():
    pytest.importorskip("openfold3.core.data.framework.single_datasets.inference")
    from openfold3.core.data.framework.single_datasets.abstract_single import DATASET_REGISTRY
    from openfold3.core.data.framework.single_datasets.inference import InferenceDataset
    from openfold3_ob0_opt.tp_rowpair import data as DATA
    assert DATASET_REGISTRY["InferenceDataset"] is InferenceDataset
    try:
        assert DATA.install_rank_stub() is True and DATA.install_rank_stub() is False           # idempotent
        cls = DATASET_REGISTRY["InferenceDataset"]
        assert cls is DATA.RankStubInferenceDataset and issubclass(cls, InferenceDataset) and cls.__getitem__ is DATA.rank_stub_record
        assert pickle.loads(pickle.dumps(cls)) is cls                                           # resolvable by module + name (forkserver / spawn workers)
    finally:
        assert DATA.uninstall_rank_stub() is True
    assert DATASET_REGISTRY["InferenceDataset"] is InferenceDataset


# ------------------------------------------------------------------------------------------ the transfer hook under rank0_bcast, two ranks (threads)
def _batch6(n=256):
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    return {"token_mask": torch.ones(1, n), "ref_pos": torch.randn(1, n, 3, generator=g), "query_id": ["q1"], "seed": torch.tensor([[5]]),
            "valid_sample": torch.tensor([[True]]), "repeated_sample": torch.tensor([[False]]),
            "msa": torch.randint(0, 20, (1, 40, n, 32)), "has_deletion": torch.zeros(1, 40, n), "deletion_value": torch.zeros(1, 40, n),
            "token_bonds": torch.zeros(1, n, n)}


def _transfer_entry(rank, P, fail):
    import torch
    from opt_core.mem.rowpair import RowpairRefused
    from openfold3_ob0_opt.tp_rowpair import data as DATA, model as M
    lines = []
    import opt_core.mem.rowpair.dist as DD
    DD.comm().log = lines.append
    if rank == 0:
        batch = _batch6()
    else:                                                                    # the light record this rank's dataset hands the DataLoader (collated)
        batch = {"query_id": ["q1"], "seed": torch.tensor([[5]]), "valid_sample": torch.tensor([[True]]), "repeated_sample": torch.tensor([[False]]),
                 DATA.RANK_STUB_FLAG: torch.tensor([[1]])}
    try:
        out = M.transfer_rank0_bcast(object(), batch, "cpu", 0)
        return ("ok", {k: (v.clone() if torch.is_tensor(v) else v) for k, v in out.items()}, lines)
    except RowpairRefused as e:
        return ("refused", e.reason, lines)
    except ValueError as e:
        return ("valueerror", str(e), lines)


def _with_fakes(monkeypatch, fail):
    from openfold3_ob0_opt.tp_rowpair import model as M
    monkeypatch.setenv("OF3TP_WORLD", "2")
    monkeypatch.setattr(M, "ensure_group", lambda: None)                    # the threaded hub is the group

    def fake_transfer(self, batch, device, idx):                            # upstream's transfer: identity on the host; or rank 0's featurised batch fails to move
        if fail:
            raise ValueError("boom in upstream transfer")
        return dict(batch)
    monkeypatch.setattr(M.PATCHES, "original", lambda owner, name: fake_transfer)


def test_rank0_bcast_transfer_two_ranks_receive_rank0s_batch(monkeypatch):
    torch = pytest.importorskip("torch")
    from opt_core.testing import run_ranks
    from openfold3_ob0_opt.tp_rowpair import msa as MSA
    _with_fakes(monkeypatch, fail=False)
    res = run_ranks(2, _transfer_entry, False, timeout_s=120)
    assert [r[0] for r in res] == ["ok", "ok"], res
    b0, b1 = res[0][1], res[1][1]
    assert set(b0) == set(b1), (sorted(b0), sorted(b1))
    for k in ("token_mask", "ref_pos", "seed", "valid_sample", "token_bonds"):
        assert torch.equal(b0[k], b1[k]), k
    assert b0["query_id"] == b1["query_id"] == ["q1"]
    skipped = MSA.HOST_KEYS_MSA if MSA.mode() == "rank0" else ()
    for k in skipped:                                                        # ranks > 0 hold zero-row placeholders of rank 0's parked MSA keys (same dtype, 0 rows)
        assert b1[k].shape[MSA.ROW_DIMS_MSA[k]] == 0 and b1[k].dtype == b0[k].dtype, (k, b1[k].shape)
    l1 = [l for l in res[1][2] if l.startswith("[feats] data_form=rank0_bcast rank 1 feats_status=ok")]
    assert l1 and "key=feats/" in l1[0], res[1][2]


def test_rank0_failure_before_the_broadcast_is_signalled_not_hung(monkeypatch):
    pytest.importorskip("torch")
    from opt_core.testing import run_ranks
    _with_fakes(monkeypatch, fail=True)
    res = run_ranks(2, _transfer_entry, True, timeout_s=120)
    assert res[0][0] == "valueerror" and "boom in upstream transfer" in res[0][1], res[0]           # rank 0 re-raises its own error
    assert res[1][0] == "refused" and res[1][1].startswith("refused: feats_rank0_failed: ValueError: boom in upstream transfer"), res[1]   # rank 1 is refused by name, at once
