"""Lever hostlean (hostlean.py): the symmetry resolutions inside validation_step are skipped exactly when no configured metric reads
ground truth; a metric that does (or whose inputs cannot be read) keeps them, by name. CPU: stand-in trainer / loss modules."""
import types

import pytest

from .. import hostlean as hl


class _MetricBase:                                   # the shape of foundry.metrics.metric.Metric the rule reads
    kwargs_to_compute_args = None
    def compute_from_kwargs(self, **kw): return {}


def _metric(table, own_cfk=False):
    ns = {"kwargs_to_compute_args": property(lambda self: table)}
    if own_cfk:
        ns["compute_from_kwargs"] = lambda self, **kw: {}
    return type("M", (_MetricBase,), ns)()


@pytest.fixture()
def upstream(monkeypatch):
    """Stand-ins for rf3.trainers.rf3 and rf3.loss.af3_losses with stock's call shape; foundry's Metric base is _MetricBase here."""
    import sys
    metric_mod = types.ModuleType("foundry.metrics.metric"); metric_mod.Metric = _MetricBase
    for name, mod in (("foundry", types.ModuleType("foundry")), ("foundry.metrics", types.ModuleType("foundry.metrics")), ("foundry.metrics.metric", metric_mod)):
        monkeypatch.setitem(sys.modules, name, mod)
    calls = []

    class Sub:
        def forward(self, network_output, loss_input, symm_input):
            calls.append("sub"); loss_input = dict(loss_input); loss_input["X_gt_L"] = "permuted"; return loss_input
        __call__ = lambda self, *a, **k: type(self).forward(self, *a, **k)

    class Res:
        def forward(self, network_output, loss_input, automorphs):
            calls.append("res"); return loss_input
        __call__ = lambda self, *a, **k: type(self).forward(self, *a, **k)

    class RF3Trainer:
        def __init__(self, metrics):
            self.metrics = metrics; self.subunit_symm_resolve = Sub(); self.residue_symm_resolve = Res()
        def validation_step(self, batch, batch_idx=0, compute_metrics=True):
            extra = {"X_gt_L": "native"}
            if compute_metrics and self.metrics is not None:
                extra = self.subunit_symm_resolve({"X_L": 0}, extra, {})
                extra = self.residue_symm_resolve({"X_L": 0}, extra, [])
            return {"metrics_output": {}, "network_output": {"X_gt_seen": extra["X_gt_L"]}}

    class RF3TrainerWithConfidence(RF3Trainer):
        def validation_step(self, batch, batch_idx=0, compute_metrics=True):
            return RF3Trainer.validation_step(self, batch, batch_idx, compute_metrics)

    trainers = types.SimpleNamespace(RF3Trainer=RF3Trainer, RF3TrainerWithConfidence=RF3TrainerWithConfidence)
    losses = types.SimpleNamespace(SubunitSymmetryResolution=Sub, ResidueSymmetryResolution=Res)
    hl.CENSUS.clear(); hl._CONSUMERS.clear(); hl.STATE.update({"on": False, "installed": False, "reason": None})
    yield trainers, losses, calls
    hl.disable(trainers, losses); hl.CENSUS.clear(); hl._CONSUMERS.clear()


def _manager(**metrics):
    return types.SimpleNamespace(metrics=dict(metrics))


def test_reads_ground_truth_by_declared_inputs():
    assert hl.reads_ground_truth(_metric({"pae": ("network_output", "pae"), "asym_id": ("network_input", "f", "asym_id")})) is None      # ptm / iptm
    assert hl.reads_ground_truth(_metric({"X_L": ("network_output", "X_L"), "predicted_atom_array_stack": ("predicted_atom_array_stack",)})) is None  # clashing chains
    assert hl.reads_ground_truth(_metric({"gt": ("ground_truth_atom_array_stack",)})) == "reads ground_truth_atom_array_stack"      # lddt / distogram / chirality
    assert hl.reads_ground_truth(_metric({"gt": "ground_truth_atom_array_stack"})) == "reads ground_truth_atom_array_stack"         # a bare-string key
    assert hl.reads_ground_truth(_metric({"x": ("extra_info", "X_gt_L")})) == "reads extra_info.X_gt_L"
    assert hl.reads_ground_truth(_metric({"m": ("extra_info", "crd_mask_L")})) == "reads extra_info.crd_mask_L"
    assert hl.reads_ground_truth(_metric({"e": ("extra_info",)})) == "reads extra_info (whole)"
    assert hl.reads_ground_truth(_metric({"e": ("extra_info", "is_real_atom")})) is None                                             # extra_info keys the resolutions do not write
    assert hl.reads_ground_truth(_metric(None)) == "no input table (receives every argument)"
    assert hl.gt_consumers(None) == [] and hl.gt_consumers(types.SimpleNamespace()) == ["<manager SimpleNamespace: no metrics table>"]


def test_own_compute_from_kwargs_counts_as_a_reader(upstream):
    assert hl.reads_ground_truth(_metric({"pae": ("network_output", "pae")}, own_cfk=True)) == "own compute_from_kwargs"   # its inputs are its own business: kept


def test_inference_metric_set_skips_both_resolutions_output_unchanged(upstream):
    trainers, losses, calls = upstream
    ptm = _metric({"pae": ("network_output", "pae")}); clash = _metric({"X_L": ("network_output", "X_L")})
    tr = trainers.RF3TrainerWithConfidence(_manager(ptm=ptm, iptm=ptm, count_clashing_chains=clash))
    assert tr.validation_step({})["network_output"]["X_gt_seen"] == "permuted" and calls == ["sub", "res"]   # stock: both run
    d = hl.enable(trainers, losses)
    assert d["on"] and d["installed"] and hl.STATE["reason"] is None
    calls.clear()
    out = tr.validation_step({})
    assert calls == [] and out["network_output"]["X_gt_seen"] == "native"                                    # skipped: the natives pass through untouched
    assert hl.census()["skipped"] == 2 and hl.census()["resolved"] == 0 and hl.census()["steps_skip"] == 1 and hl.census()["kept_reasons"] == []
    assert hl.lever_evidence() == [("skipped", 2), ("resolved", 0), ("steps_kept", 0)]
    hl.enable(trainers, losses)                                                                                # idempotent: no double wrap
    tr.validation_step({}); assert hl.census()["skipped"] == 4 and calls == []


def test_a_ground_truth_metric_keeps_the_resolutions_by_name(upstream):
    trainers, losses, calls = upstream
    lddt = _metric({"gt": ("ground_truth_atom_array_stack",), "pred": ("predicted_atom_array_stack",)})
    tr = trainers.RF3Trainer(_manager(ptm=_metric({"pae": ("network_output", "pae")}), all_atom_lddt=lddt))
    hl.enable(trainers, losses)
    out = tr.validation_step({})
    assert calls == ["sub", "res"] and out["network_output"]["X_gt_seen"] == "permuted"                       # stock statements ran
    c = hl.census()
    assert c["skipped"] == 0 and c["resolved"] == 2 and c["steps_kept"] == 1 and c["kept_reasons"] == ["all_atom_lddt(reads ground_truth_atom_array_stack)"]
    assert ("kept", "all_atom_lddt(reads ground_truth_atom_array_stack)") in hl.lever_evidence()


def test_resolutions_outside_validation_step_run_as_stock(upstream):
    """The training step's loss path calls the same modules: never skipped (the flag is armed only inside validation_step)."""
    trainers, losses, calls = upstream
    hl.enable(trainers, losses)
    sub = losses.SubunitSymmetryResolution()
    assert sub({"X_L": 0}, {"X_gt_L": "native"}, {})["X_gt_L"] == "permuted" and calls == ["sub"] and hl.census()["resolved"] == 1


def test_refused_by_name_when_upstream_is_reshaped(upstream):
    trainers, losses, _ = upstream
    with pytest.raises(hl.HostleanRefused):
        hl.enable(types.SimpleNamespace(RF3Trainer=trainers.RF3Trainer), losses)                               # RF3TrainerWithConfidence absent
    assert hl.STATE["on"] is False and "RF3TrainerWithConfidence" in hl.STATE["reason"]


def test_upstream_call_shape_is_the_one_the_lever_wraps():
    """The pinned upstream: validation_step on both trainer classes calls self.subunit_symm_resolve / residue_symm_resolve with
    (network_output, metrics_extra_info, …) and the resolutions write only X_gt_L / crd_mask_L (stock/ tarball, read as text)."""
    import io, os, tarfile
    from .. import stack
    tb = os.path.join(stack.kit_root(), "stock", "foundry-4010e3e2e.tar.gz") if hasattr(stack, "kit_root") else None
    if not tb or not os.path.exists(tb):
        here = os.path.dirname(os.path.abspath(__file__))
        tb = os.path.normpath(os.path.join(here, "..", "..", "..", "stock", "foundry-4010e3e2e.tar.gz"))
    if not os.path.exists(tb):
        pytest.skip("stock tarball not in this tree")
    with tarfile.open(tb) as tf:
        tr = tf.extractfile("foundry-4010e3e2e/models/rf3/src/rf3/trainers/rf3.py").read().decode()
        lo = tf.extractfile("foundry-4010e3e2e/models/rf3/src/rf3/loss/af3_losses.py").read().decode()
    assert tr.count("metrics_extra_info = self.subunit_symm_resolve(") == 2 and tr.count("metrics_extra_info = self.residue_symm_resolve(") == 2
    assert "class SubunitSymmetryResolution(nn.Module)" in lo and "class ResidueSymmetryResolution(nn.Module)" in lo
    for cls_src in lo.split("class ")[1:]:
        if cls_src.startswith(("SubunitSymmetryResolution", "ResidueSymmetryResolution")):
            fwd = cls_src.split("def forward")[1]
            writes = {ln.strip().split("[")[1].split("]")[0] for ln in fwd.splitlines() if ln.strip().startswith("loss_input[")}
            assert writes == {'"X_gt_L"', '"crd_mask_L"'}, writes                                             # the only writes: the natives and their mask
            assert "network_output[" not in fwd.replace('network_output["X_L"]', "")                          # the prediction is read, never written
