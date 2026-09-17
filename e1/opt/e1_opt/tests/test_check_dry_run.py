"""`e1-opt check`: a dry run that resolves and gates the mode on this box and applies nothing — in a subprocess on the mocked box
(the fake nvidia-smi on PATH, the stub upstream on PYTHONPATH) and in process with the placeholders accepted."""
import json
import os
import subprocess
import sys

from e1_opt import cli, report, stack
from e1_opt.tests import _stubs


def test_check_subprocess_with_a_mocked_gpu(monkeypatch, tmp_path):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    env = _stubs.child_env(box)
    p = subprocess.run([sys.executable, "-m", "e1_opt", "check", "--variant", box["variant"]], env=env,
                       capture_output=True, text=True, timeout=120)
    lines = p.stdout.splitlines()
    m = report.RE_DRY_RUN.match(lines[0])
    assert m and m.group("mode") == "exact" and m.group("variant") == box["variant"], p.stdout + p.stderr
    assert f"gpu={box['gpu'][0]} " in lines[0] and " card=h100 " in lines[0] and f" mib={box['gpu'][1]} " in lines[0]
    # the placeholder kernel file is the one pin a real box holds and a stub cannot: NAMED (notes=), never a refusal
    assert m.group("would_refuse") == "none" and p.returncode == 0, lines[0]
    assert ' pins=drift ' in lines[0] and ' notes="hub kernel snapshot ' in lines[0] and "layer_norm.py does not hash to the pin" in lines[0], lines[0]
    assert len(lines) == 1, lines                                                      # the DRY-RUN line is the whole report
    assert " stack=torch" in lines[0] and "-sm90 " in lines[0], lines[0]


def test_check_in_process_green_and_another_target(monkeypatch, tmp_path, capsys):
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    rc = cli.main(["check", "--variant", box["variant"]])
    line = capsys.readouterr().out.splitlines()[0]
    assert rc == 0 and report.RE_DRY_RUN.match(line).group("would_refuse") == "none", line
    assert stack.status()["active"] is False and "has not run" in stack.status()["reason"]      # a dry run activates nothing
    rc = cli.main(["check", "--variant", box["variant"], "--mode", "off"])
    line = capsys.readouterr().out.splitlines()[0]
    assert rc == 0 and "mode=off" in line and line.endswith("would_refuse=none")
    assert " notes=" not in line and " pins=ok " in line                                  # a tested box: nothing to name
    monkeypatch.setenv(stack.ENV_TARGET_GPU, "H200")                                            # a config targeting another card class: named, never refused
    rc = cli.main(["check", "--variant", box["variant"]])
    line = capsys.readouterr().out.splitlines()[0]
    assert rc == 0 and line.endswith("would_refuse=none") and ' notes="the config targets h200 (MODEL_OPT_TARGET_GPU), this card is class h100' in line, line


def test_api_check_and_enable_report_shape(monkeypatch, tmp_path):
    import e1_opt
    box = _stubs.setup_box(monkeypatch, tmp_path)
    _stubs.patch_pins_for_stub(monkeypatch, box["pins"], box["variant"], box["caches"])
    rep = e1_opt.check("exact", box["variant"])
    for k in ("active", "mode", "variant", "kit", "kit_mode", "gpu", "stack_key", "package_version", "reason", "would_refuse", "dry_run"):
        assert k in rep
    assert rep["would_refuse"] is None and rep["package_version"] == e1_opt.__version__ and rep["kit_mode"] == "eager"
    rep = e1_opt.enable("off", box["variant"])
    assert rep["active"] is False and rep["reason"] == report.OFF_REASON and e1_opt.status()["mode"] == "off"
    rep2 = e1_opt.enable("exact", box["variant"])
    assert rep2["active"] is False and "already activated" in rep2["reason"]
