"""The numerics a kit line must deliver, judged from the driver's READBACK, never from its flag: an fp32 line (exact) whose record reads TF32
matmuls live is failed by name (exit 1, `NUMERICS … verdict=FAIL`), the tf32 line (fast) whose record reads them off likewise, a record without a
readback likewise; L13's evidence is the readback; L8's evidence is a filled hoist with no anomaly, L11's a finished request (judged,
design.JUDGES); TORCH_ALLOW_TF32_* never reaches a kit child. The matmul precision is the mode's own (exact fp32, fast tf32): no selector.

No GPU, no torch, no upstream throughout."""

import json
import os

from genie3_opt import design, modes, stack
from genie3_opt.tests import _stubs


def test_precision_is_the_modes_own(tmp_path, monkeypatch, capsys):
    assert modes.precision_of(modes.resolve("fast")) == "tf32" and modes.precision_of(modes.resolve("exact")) == "fp32" and modes.precision_of(modes.resolve("off")) == "fp32"
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), str(tmp_path / "t"), "fast")
    assert rc == 0 and man["precision"] == "tf32" and "--tf32" in man["driver_pass"]["cmd"] and man["levers_planned"] == ["L1", "L2", "L4", "L8", "L9", "L11", "L17", "L18", "L19", "L12", "L13", "L7", "L16"] and man["trimul"] == "fpf"
    err = capsys.readouterr().err
    assert "[genie3-opt] RUN mode=fast problems=1 designs=2 seed=unseeded batch=1 precision=tf32 out=" in err          # --det 0, no seed in the request: unseeded, said so; batch: upstream's default
    active = [l for l in err.splitlines() if l.startswith("[genie3-opt] ACTIVE ")]
    assert len(active) == 1 and "levers=L1,L2,L4,L8,L9,L11,L17,L18,L19,L12,L13,L7,L16 " in active[0] and active[0].endswith("--pt-chunk design --tf32 --trimul fpf --compile")
    T = json.load(open(os.path.join(str(tmp_path / "t"), "timings.json")))
    assert T["tf32"] is True and T["pt_chunk"]["chunk"] == "design"
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2, seed=0, name="e.yaml"), str(tmp_path / "e"), "exact")
    assert rc == 0 and man["precision"] == "fp32" and "--tf32" not in man["driver_pass"]["cmd"]
    assert "[genie3-opt] RUN mode=exact problems=1 designs=2 seed=0 batch=1 precision=fp32 out=" in capsys.readouterr().err
    stack.reset_for_tests()


# ===== NUMERICS: judged from the driver's readback, never from the flag that requested it =====
def _run(tmp_path, monkeypatch, mode, name, fail=None, **kw):
    stack.reset_for_tests()
    if fail:
        monkeypatch.setenv("STUB_FAIL", fail)
    else:
        monkeypatch.delenv("STUB_FAIL", raising=False)
    return design.run(_stubs.request(str(tmp_path), n_sample=2, name=name + ".yaml"), str(tmp_path / name), mode, **kw)


def test_exact_reads_fp32_and_says_so(tmp_path, monkeypatch, capsys):
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = _run(tmp_path, monkeypatch, "exact", "e")
    assert rc == 0 and man["status"] == "ok", man
    assert man["driver_pass"]["numerics"] == {"line": "fp32", "live": {"matmul_tf32": False, "matmul": "highest", "cudnn_tf32": True}, "verdict": "ok"}
    err = capsys.readouterr().err
    assert "[genie3-opt] NUMERICS mode=exact line=fp32 matmul_tf32=False matmul=highest cudnn_tf32=True batch=1/1 verdict=ok" in err


def test_tf32_live_on_the_identity_line_is_failed_by_readback(tmp_path, monkeypatch, capsys):
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = _run(tmp_path, monkeypatch, "exact", "leak", fail="tf32leak")      # the stub's record reads TF32 live although the line never asked for it
    assert rc == 1 and man["status"] == "failed", man
    assert man["numerics_note"] == "numerics readback matmul_tf32=True matmul=high on a fp32 line"
    assert man["driver_pass"]["evidence"]["missing"] == []                      # the levers' own evidence is all there — the readback is what fails the pass
    err = capsys.readouterr().err
    assert "NUMERICS mode=exact line=fp32 matmul_tf32=True matmul=high" in err and "verdict=FAIL: numerics readback matmul_tf32=True matmul=high on a fp32 line" in err


def test_l13_evidence_is_the_readback(tmp_path, monkeypatch, capsys):
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = _run(tmp_path, monkeypatch, "fast", "f")
    assert rc == 0 and "L13" in man["driver_pass"]["evidence"]["applied"] and man["driver_pass"]["numerics"]["line"] == "tf32" and man["driver_pass"]["numerics"]["verdict"] == "ok"
    assert "NUMERICS mode=fast line=tf32 matmul_tf32=True matmul=high" in capsys.readouterr().err
    rc, man = _run(tmp_path, monkeypatch, "fast", "off13", fail="tf32off")        # --tf32 passed, the record reads TF32 off: L13 missing AND the verdict fails
    assert rc != 0 and "L13" in man["driver_pass"]["evidence"]["missing"] and man["driver_pass"]["numerics"]["verdict"].startswith("numerics readback matmul_tf32=False")
    rc, man = _run(tmp_path, monkeypatch, "exact", "norb", fail="noreadback")     # a record with no readback at all: failed, never assumed fp32
    assert rc == 1 and man["numerics_note"] == "no numerics readback in the driver's record"


def test_hoist_and_batches_are_judged_not_flags(tmp_path, monkeypatch, capsys):
    _stubs.box(str(tmp_path), monkeypatch)
    rc, man = _run(tmp_path, monkeypatch, "exact", "stale", fail="hoiststale")   # a hoist anomaly counted by the driver: L8's evidence is missing
    assert rc == 1 and man["status"] == "failed" and man["levers_broken"] == ["L8"] and man["driver_pass"]["evidence"]["missing"] == ["L8"], man   # a broken lever (judged predicate), not a stand-down: failed, 1
    rc, man = _run(tmp_path, monkeypatch, "exact", "unf", fail="unfinished")     # not every batch sampled: L11 missing
    assert rc == 1 and man["status"] == "failed" and "L11" in man["levers_broken"]        # the batch table is a judged predicate: a driver that did not sample every batch is broken, not stood down


def test_kit_children_never_inherit_torch_tf32_override(tmp_path, monkeypatch):
    _stubs.box(str(tmp_path), monkeypatch)
    monkeypatch.setenv("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE", "1")
    monkeypatch.setenv("NVIDIA_TF32_OVERRIDE", "0")
    env, dropped = stack.child_environment()
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" not in env and "NVIDIA_TF32_OVERRIDE" not in env
    assert "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in dropped and "NVIDIA_TF32_OVERRIDE" in dropped
    stack.reset_for_tests()
    rc, man = design.run(_stubs.request(str(tmp_path), n_sample=2), str(tmp_path / "x"), "exact")
    assert rc == 0 and "TORCH_ALLOW_TF32_CUBLAS_OVERRIDE" in man["env_dropped"] and man["driver_pass"]["numerics"]["verdict"] == "ok"
