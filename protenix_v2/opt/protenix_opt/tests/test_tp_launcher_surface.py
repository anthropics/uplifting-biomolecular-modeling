"""The multi-GPU line's launcher (opt/forward/PTX_TP/PTX_TP_ADDON/ptx_tp/launch.py) takes exactly the arguments the line passes
(tp.command: --nproc --lazy-relp --lift-guard --label --out -- <stock arguments>) and builds the rank environment of the line from them:
no reference-leg, checkpoint, mirror, template, sampler-arm, seam or extra-environment switches."""
import json
import os
import sys

import pytest

from protenix_opt import tp

UNIT_DIR = tp.unit_dir()                                                     # opt/forward/PTX_TP/PTX_TP_ADDON: holds the ptx_tp package


@pytest.fixture()
def launch(monkeypatch):
    if not os.path.isfile(os.path.join(UNIT_DIR, "ptx_tp", "launch.py")):
        pytest.skip(f"carried unit absent: {UNIT_DIR}")
    monkeypatch.syspath_prepend(UNIT_DIR)
    sys.modules.pop("ptx_tp.launch", None)
    import ptx_tp.launch as L
    return L


def _flags(launch):
    import argparse
    ap_flags = set()
    real = argparse.ArgumentParser.add_argument

    def spy(self, *names, **kw):
        ap_flags.update(n for n in names if n.startswith("--"))
        return real(self, *names, **kw)
    argparse.ArgumentParser.add_argument = spy
    try:
        launch.parse(["--nproc", "2", "--", "pred"])
    finally:
        argparse.ArgumentParser.add_argument = real
    return ap_flags - {"--help"}


def test_the_launcher_takes_exactly_the_line_arguments(launch):
    assert _flags(launch) == {"--nproc", "--label", "--lazy-relp", "--lift-guard", "--out"}
    line = tp.command(2, ["-i", "in.json", "-o", "OUT"], "OUT", "L")
    assert line[1:3] == ["-m", tp.LAUNCHER_MODULE]
    a, rest = launch.parse(line[3:])
    assert (a.nproc, a.label, a.lazy_relp, a.lift_guard, a.out) == (2, "L", True, True, os.path.join("OUT", tp.PHASE_LOG_DIR)) and rest == ["pred", "-i", "in.json", "-o", "OUT"]
    for gone in (["--ref"], ["--det"], ["--dap"], ["--mirror", "m"], ["--ckpt", "c"], ["--resume"], ["--templates"], ["--mc-dropout-off"],
                 ["--sampler-arms", "off,B"], ["--env", "K=V"], ["--impl", "tp"], ["--seam", "pairformer=tp"], ["--feat", "all"], ["--debug-checksum"], ["--dry-run"]):
        with pytest.raises(SystemExit):
            launch.parse(["--nproc", "2"] + gone + ["--", "pred"])
    with pytest.raises(SystemExit):
        launch.parse(["--", "pred"])                                                     # --nproc is required: no PTX_TP environment default


def test_build_is_the_line_rank_environment(launch, tmp_path, monkeypatch):
    inp = tmp_path / "in.json"
    inp.write_text(json.dumps([{"name": "x", "sequences": [{"proteinChain": {"sequence": "A" * 700, "count": 2}}]}]))
    for k in [k for k in os.environ if k.startswith(("PTX_", "XL_", "ROWPAIR_"))]:
        monkeypatch.delenv(k)
    a, rest = launch.parse(["--nproc", "2", *tp.LINE_ARGS, "--label", "L", "--out", str(tmp_path / "tp"), "--", "pred", "-i", str(inp), "-o", str(tmp_path / "o")])
    cmd, env, desc = launch.build(a, rest)
    assert env["PTX_TP"] == "2" and env["PTX_TP_RELP"] == "lazy" and env["XL_LIFT_GUARD"] == "1" and env["PTX_TP_JOB_LABEL"] == "L"
    assert env["PTX_TP_PHASE_LOG"] == str(tmp_path / "tp")
    for k, v in (("PTX_TP_DIFF_REPLICATE_BELOW", "3841"), ("PTX_TP_DIFF_ATTN", "rowsplit"), ("PTX_TP_SDPA_QPAD", "1"), ("PTX_TP_DIFFCACHE_GB", "8"),
                 ("PTX_TP_BIAS_CHUNK", "128"), ("PTX_TP_F2_GATHER_BELOW", "3841"), ("PTX_TP_RNG_SYNC", "1"), ("PTX_TP_ATOM_LOCAL", "1"),
                 ("PTX_TP_TRIMUL_BCACHE", "device"), ("PTX_TP_TRIMUL_ROWS_A", "1024"), ("PTX_TP_TRIMUL_ROWS_B", "256"), ("PTX_TP_TRIMUL_ZCOPY", "lazy")):
        assert env[k] == v, k
    assert env["PTX_TP_TILING_DESC"].startswith("N~1400 P=2 shard=")
    i = cmd.index("--")
    assert cmd[i + 1:] == rest + list(tp.STOCK_ARG_DELTA)                                # the stock arguments verbatim + the line's one delta
    assert [cmd[j + 1] for j, t in enumerate(cmd[:i]) if t == "--hook"] == ["ptx_tp.runner_hooks:install_guard_lift", "ptx_tp:apply_from_env"]
    assert desc == "TP P=2 (torchrun; hooks: ['ptx_tp.runner_hooks:install_guard_lift', 'ptx_tp:apply_from_env'])"
    monkeypatch.setenv("PTX_TP_DIFF_REPLICATE_BELOW", "0")                            # the line's forced gates arrive pre-set and are kept
    _cmd, env2, _ = launch.build(*launch.parse(["--nproc", "2", *tp.LINE_ARGS, "--", "pred", "-i", str(inp), "--trimul_kernel", "cuequivariance"]))
    assert env2["PTX_TP_DIFF_REPLICATE_BELOW"] == "0" and _cmd[_cmd.index("--") + 1:] == ["pred", "-i", str(inp), "--trimul_kernel", "cuequivariance"]


def test_rank_init_passes_the_collective_timeout_explicitly(monkeypatch):
    """launch_core.rank_init hands the process group's collective timeout to init_from_env as an argument (NCCL_TIMEOUT_S = tp.PG_TIMEOUT_S =
    1800 s); nothing is read from the environment for it."""
    if not os.path.isfile(os.path.join(UNIT_DIR, "ptx_tp", "launch_core.py")):
        pytest.skip(f"carried unit absent: {UNIT_DIR}")
    monkeypatch.syspath_prepend(UNIT_DIR)
    for m in ("ptx_tp.launch_core", "ptx_tp.dist"):
        sys.modules.pop(m, None)
    import types
    seen = {}
    fake_dist = types.ModuleType("ptx_tp.dist")
    fake_dist.init_from_env = lambda backend=None, timeout_s=None: (seen.update(backend=backend, timeout_s=timeout_s), (1, 0, "cpu"))[1]
    monkeypatch.setitem(sys.modules, "ptx_tp.dist", fake_dist)
    import ptx_tp.launch_core as LC
    assert LC.rank_init(verbose=False) == (1, 0, "cpu")
    assert seen == {"backend": None, "timeout_s": 1800} and LC.NCCL_TIMEOUT_S == tp.PG_TIMEOUT_S == 1800
