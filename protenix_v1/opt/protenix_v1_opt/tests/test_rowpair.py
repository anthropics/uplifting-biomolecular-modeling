"""The process half of the tensor-parallel mode (rowpair.py) on the CPU: nothing is installed at n_gpu=1 (refused by name, rowpair and
tp alike), the rank environment decides parent vs rank, the rank command line is this package's `pred` with the parent's arguments, the
in-launch admission rule, the LEVER line grammar at n_gpu=1, the layout alignment rule. The sharded statements run in test_tp_cpu.py
(threaded ranks) and on P GPUs (the dev lanes and the multi-GPU comparison runs)."""
import sys

import pytest

from opt_core.mem import rowpair as RP
from opt_core.mem.rowpair import launch as L

from protenix_v1_opt import ngpu, rowpair, tp
from protenix_v1_opt import report as R


@pytest.fixture(autouse=True)
def no_rank_env(monkeypatch):
    for k in list(__import__("os").environ):
        if k.startswith("ROWPAIR_") or k in ("RANK", "WORLD_SIZE", "LOCAL_RANK"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(rowpair, "STATE", dict(rowpair.STATE, P=1, rank=0, installed=False, items=0, sites=[]))
    yield


def test_nothing_is_installed_at_n_gpu_1():
    """The structural rule: outside a rank process of a P>1 launch the adapter wraps nothing (refused by name)."""
    assert rowpair.is_rank_process() is False and rowpair.world_size() == 1 and rowpair.is_output_rank() is True
    with pytest.raises(RP.RowpairRefused) as ei:
        rowpair.install()
    assert "refused at n_gpu=1" in str(getattr(ei.value, "reason", ei.value))
    with pytest.raises(tp.TPRefused) as ei:                                  # the statement half alike: tp.install at P=1 is refused by name
        tp.install(1, 0)
    assert "n_gpu=1" in str(ei.value) and tp.STATE["installed"] is False
    pm = sys.modules.get("protenix.model.protenix")
    if pm is not None:                                                       # protenix importable here: the class method is the stock's
        assert pm.Protenix.get_pairformer_output is not tp.get_pairformer_output_tp


def test_replicate_inputs_is_identity_at_n_gpu_1():
    triple = ({"input_feature_dict": {"x": 1}, "sample_name": "s"}, None, "")
    assert rowpair.replicate_inputs(lambda: triple, "s", 0) is triple and rowpair.STATE["items"] == 0     # no group: the featuriser's own triple, no count


def test_rank_environment_decides_parent_vs_rank(monkeypatch):
    monkeypatch.setenv(L.ENV_WORLD, "2"); monkeypatch.setenv(L.ENV_RANK, "1")
    assert rowpair.is_rank_process() is True and rowpair.world_size() == 2 and rowpair.rank() == 1 and rowpair.is_output_rank() is False
    monkeypatch.setenv(L.ENV_RANK, "0")
    assert rowpair.is_output_rank() is True


def test_rank_argv_is_this_packages_pred_with_the_parents_arguments():
    argv = rowpair.rank_argv(["--mode", "big", "--n_gpu", "2", "--input", "x.json", "--out_dir", "o"])
    assert argv[:4] == [sys.executable, "-m", "protenix_v1_opt", "pred"] and argv[4:] == ["--mode", "big", "--n_gpu", "2", "--input", "x.json", "--out_dir", "o"]


def test_in_launch_admission_names_the_launch_P(monkeypatch):
    monkeypatch.setenv(L.ENV_WORLD, "2"); monkeypatch.setenv(L.ENV_RANK, "0")
    assert ngpu.admit(2, "big") == 2                                        # a rank of a world-2 launch sees one device: no visible rule, the launch's P
    with pytest.raises(ngpu.NG.NGpuRefused) as ei:
        ngpu.admit(4, "big")
    assert str(ei.value.reason) == "refused: n_gpu=4 inside a launch of world 2"


def test_lever_line_at_n_gpu_1_is_off_by_name():
    line = rowpair.lever_line()
    assert line.startswith(f"{R.PREFIX} LEVER name=rowpair state=off reason=n_gpu_1 ") and " n_gpu=1 sharding=none" in line
    assert f"strategy={RP.LEVER}" in line and "origin=core" in line


def test_layout_alignment_rule():
    """Row boundaries follow the pinned chunk grid when every rank can own a chunk; below that the largest power of two >= 4; refused by name
    when N < 4*P (never a silent replicated run)."""
    # each N below is a distinct number of halvings from the 128 chunk (0, 1, 2, then the a=4 floor); N=600 dropped (R5): same
    # one-halving-step branch as N=1000, no added coverage.
    assert tp.choose_align(2956, 8, 128) == 128 and tp.choose_align(1000, 8, 128) == 64
    assert tp.choose_align(300, 8, 128) == 32 and tp.choose_align(40, 8, 128) == 4
    with pytest.raises(tp.TPRefused):
        tp.choose_align(30, 8, 128)


def test_no_size_gate_in_the_sharded_path():
    """R-TP-3: the statement half keys nothing on a token count — every seam is sharded at every admitted N (the only N rule is the layout's
    refusal). Source census: no comparison of N / n_token against a literal threshold in tp.py outside the stock's own >2000 cache-release /
    fp32-one-hot placements it reproduces."""
    import re
    src = open(tp.__file__).read()
    hits = [l.strip() for l in src.splitlines() if re.search(r"\bN\s*[<>]=?\s*\d{3,}|n_token\w*\s*[<>]=?\s*\d{3,}", l) and "2000" not in l]
    assert hits == [], hits
