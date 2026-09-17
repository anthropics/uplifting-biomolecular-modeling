"""check-time CUDA-graph probe + exit-code folding (CPU; no model, no GPU)."""
import types
import pytest

from atlasfold_opt import cli
from atlasfold_opt.hooks import denoiser_graph as DG


def test_probe_without_cuda_says_no_cuda(monkeypatch):
    monkeypatch.setattr(DG.torch.cuda, "is_available", lambda: False)
    row = DG.probe()
    assert row == {"kernel": "cuda_graph", "ok": None, "routed": None, "resolved": None, "reason": "no_cuda"}


def _stub_check_env(monkeypatch, graph_row):
    """cmd_check with every probe stubbed: the core gate, GPU / stock facts, the Triton KERNEL rows, sdpa_fused, and the cuda_graph row given."""
    from atlasfold_opt import _core, stack
    monkeypatch.setattr(_core, "gate", lambda: {"installed": {"version": "x"}, "pinned": {"version": "x", "path": "p"}})
    monkeypatch.setattr(stack, "_gpu_probe", lambda: {"name": "stub", "cc": "9.0", "torch": "2.7.1"})
    monkeypatch.setattr(stack, "_stock_facts", lambda: {"importable": True, "dist_version": None, "cuequivariance_torch": None})
    monkeypatch.setattr(cli, "kernel_probe", lambda: [])
    monkeypatch.setattr(DG, "probe", lambda: dict(graph_row))
    monkeypatch.delenv("ATLASFOLD_WEIGHTS_DIR", raising=False)


@pytest.mark.parametrize("mode,ok,rc", [("fast", False, 3), ("fast", True, 0), ("fast", None, 0), ("exact", False, 0), ("off", False, 0)])
def test_check_refuses_a_box_that_cannot_capture_only_when_the_row_carries_the_lever(monkeypatch, capsys, mode, ok, rc):
    _stub_check_env(monkeypatch, {"kernel": "cuda_graph", "ok": ok, "routed": ok, "resolved": "torch.cuda.graphs@x", "reason": None if ok else "capture_failed:RuntimeError: boom"})
    a = types.SimpleNamespace(mode=mode, weights=None)
    assert cli.cmd_check(a, []) == rc
    err = capsys.readouterr().err
    assert "KERNEL cuda_graph ok=" in err
    assert ("CHECK refused: cuda_graph capture_failed:RuntimeError" in err) == (rc == 3)


def test_pred_folds_a_raised_run_into_a_named_line_and_the_gate_exit_code(monkeypatch, capsys, tmp_path):
    """The run raising (a lever ending an item by name) is `RUN raised <Exc>: …` + rc 1, and a refused gate reads FINAL exit=3 — never a bare traceback."""
    import sys
    from atlasfold_opt import stack
    fake_cli = types.ModuleType("atlasfold.cli")
    from atlasfold_opt.hooks import LeverAborted
    def main(argv=None): raise LeverAborted("denoiser_graph", "capture_failed:RuntimeError", "denoiser_graph: the CUDA-graph capture of DiffusionModule.forward failed (X: y)", hint="max_tokens=1024; AFO_DENOISER_GRAPH_MAX_TOKENS=0 runs the denoiser eagerly")
    fake_cli.main = main
    monkeypatch.setitem(sys.modules, "atlasfold.cli", fake_cli)
    import atlasfold_opt
    monkeypatch.setattr(atlasfold_opt, "enable", lambda *a, **k: True)
    monkeypatch.setattr(stack, "gates_verdict", lambda: {"ok": False, "refused": [("denoiser_graph", "capture_failed:RuntimeError")]})
    out = tmp_path / "o"
    monkeypatch.delenv("ATLASFOLD_WEIGHTS_DIR", raising=False)
    a = types.SimpleNamespace(mode="fast", det=0, n_gpu=1, allow_partial=False, weights=None)
    stock = ["multimer", "--input-fasta", str(tmp_path / "x.fasta"), "--out-dir", str(out), "--seed", "1", "--num-samples", "1"]
    (tmp_path / "x.fasta").write_text(">a\nMKV\n")
    code = cli.cmd_pred(a, stock)
    err = capsys.readouterr().err
    assert "REFUSED: denoiser_graph capture_failed:RuntimeError (max_tokens=1024; AFO_DENOISER_GRAPH_MAX_TOKENS=0 runs the denoiser eagerly)" in err, err
    assert code == 3 and "FINAL mode=fast rc=1 exit=3" in err and "gates=refused:denoiser_graph:capture_failed:RuntimeError" in err, err


def test_pred_folds_any_other_raised_run_into_a_named_line(monkeypatch, capsys, tmp_path):
    """A run raising something that is not a lever's refusal (the stock code) is `RUN raised <Exc>: …` + rc 1 (exit 1 when no gate is refused)."""
    import sys
    from atlasfold_opt import stack
    import atlasfold_opt
    fake_cli = types.ModuleType("atlasfold.cli")
    def main(argv=None): raise ValueError("FASTA target contains an empty chain.")
    fake_cli.main = main
    monkeypatch.setitem(sys.modules, "atlasfold.cli", fake_cli)
    monkeypatch.setattr(atlasfold_opt, "enable", lambda *a, **k: True)
    monkeypatch.setattr(stack, "gates_verdict", lambda: {"ok": True, "refused": []})
    monkeypatch.delenv("ATLASFOLD_WEIGHTS_DIR", raising=False)
    (tmp_path / "x.fasta").write_text(">a\nMKV\n")
    a = types.SimpleNamespace(mode="fast", det=0, n_gpu=1, allow_partial=False, weights=None)
    code = cli.cmd_pred(a, ["multimer", "--input-fasta", str(tmp_path / "x.fasta"), "--out-dir", str(tmp_path / "o"), "--seed", "1"])
    err = capsys.readouterr().err
    assert code == 1 and "RUN raised ValueError: FASTA target contains an empty chain." in err and "FINAL mode=fast rc=1 exit=1" in err, err
