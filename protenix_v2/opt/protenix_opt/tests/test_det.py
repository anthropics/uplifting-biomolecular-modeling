"""--det 1 = the kit's own deterministic recipe (det.py): its data names files the tree carries; the detref copy is placed
before enable() / before the stock child and restored when the run ends; every precondition failure is a named refusal; the stock
child's proof allows exactly the carve-out and nothing beyond it; check --det 1 prints DET lines."""
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys

import pytest

import protenix_opt
from protenix_opt import cli, det, kits, stack, stock_pred
from protenix_opt.tests import _stock_stub
from protenix_opt.tests.conftest import needs_durable_install
from protenix_opt.tests.test_cli_passthrough import CoreStub, core, fake_stock   # noqa: F401  (fixtures)

KIT = stack.kit_home()
STOCK_SCATTER = b"# stock scatter_utils (stand-in)\n"


# ------------------------------------------------------------------------------------------------------- conformance ----
def test_det_data_names_files_the_tree_carries():
    """The stock arm's one path entry is the unit's src/ (its sitecustomize keys on PTX_DET); the detref files are on disk; the replaced file is one of them."""
    assert det.REPLACED_FILE == "scatter_utils.py" and det.REPLACED_FILE in det.DETREF_FILES
    assert [f for f in det.DETREF_FILES if f != det.REPLACED_FILE] == ["det_segment_reduce.py"], "the second file is removed, not restored"
    assert det.stock_pythonpath() == os.path.join(KIT, "src") and os.path.isfile(os.path.join(det.stock_pythonpath(), "sitecustomize.py"))
    sc = open(os.path.join(KIT, "src", "sitecustomize.py"), encoding="utf-8").read()
    assert 'os.environ.get("PTX_DET", "0") == "1"' in sc and "torch.use_deterministic_algorithms(True, warn_only=True)" in sc
    for f in det.DETREF_FILES:
        assert os.path.isfile(os.path.join(det.detref_dir(), f)), f


# ------------------------------------------------------------------------------------------------------ copy / restore ----
@pytest.fixture
def fake_pkg(tmp_path, monkeypatch):
    """A stand-in installed protenix package (utils/scatter_utils.py = stock bytes) the recipe patches; det.protenix_package_dir points at it."""
    pkg = tmp_path / "site" / "protenix"
    (pkg / "utils").mkdir(parents=True)
    (pkg / "__init__.py").write_text("")
    (pkg / "utils" / "scatter_utils.py").write_bytes(STOCK_SCATTER)
    monkeypatch.setattr(det, "protenix_package_dir", lambda: str(pkg))
    saved = {k: os.environ.pop(k, None) for k in det.DET_ENV}            # det.apply_env() exports into os.environ: restored by hand (an absent
    yield pkg                                                             # name is not recorded by monkeypatch.delenv)
    for k, v in saved.items():
        os.environ.pop(k, None)
        if v is not None:
            os.environ[k] = v


def _target(pkg, f):
    return pkg / "utils" / f


def test_plan_apply_restore(fake_pkg):
    p = det.plan()
    assert p["problems"] == [] and p["level"] == 1 and p["env"] == det.DET_ENV and p["package_dir"] == str(fake_pkg)
    assert [os.path.basename(e["source"]) for e in p["patch"]] == list(det.DETREF_FILES)
    assert p["patch"][0]["replaces"] is True and p["patch"][0]["backup"] == str(_target(fake_pkg, "scatter_utils.py")) + det.BACKUP_SUFFIX
    assert p["patch"][1]["replaces"] is False and p["patch"][1]["backup"] is None and p["patch"][1]["sha256_target_before"] is None
    assert p["patch"][0]["sha256_target_before"] == stack.sha256_file(str(_target(fake_pkg, "scatter_utils.py")))
    for e in p["patch"]:
        assert e["sha256_source"] == stack.sha256_file(e["source"])
    patch = det.Patch(p).apply()
    assert patch.applied and not patch.restored
    assert _target(fake_pkg, "scatter_utils.py").read_bytes() == open(os.path.join(det.detref_dir(), "scatter_utils.py"), "rb").read()
    assert _target(fake_pkg, "det_segment_reduce.py").read_bytes() == open(os.path.join(det.detref_dir(), "det_segment_reduce.py"), "rb").read()
    assert (fake_pkg / "utils" / ("scatter_utils.py" + det.BACKUP_SUFFIX)).read_bytes() == STOCK_SCATTER
    rec = patch.record()
    assert rec["level"] == 1 and rec["env"] == det.DET_ENV and [r["restored"] for r in rec["patch"]] == [False, False]
    assert rec["patch"][0]["sha256_before"] == p["patch"][0]["sha256_target_before"]
    assert rec["patch"][0]["sha256_after"] == p["patch"][0]["sha256_source"] and rec["patch"][1]["sha256_before"] is None
    patch.restore()
    assert patch.restored and patch.restore_problems == []
    assert _target(fake_pkg, "scatter_utils.py").read_bytes() == STOCK_SCATTER
    assert not _target(fake_pkg, "det_segment_reduce.py").exists() and not (fake_pkg / "utils" / ("scatter_utils.py" + det.BACKUP_SUFFIX)).exists()
    rec = patch.record(stock_exception={"x": 1})
    assert [r["restored"] for r in rec["patch"]] == [True, True] and rec["stock_exception"] == {"x": 1} and rec["restore_problems"] == []
    patch.restore()                                                       # idempotent
    assert _target(fake_pkg, "scatter_utils.py").read_bytes() == STOCK_SCATTER
    assert det.plan()["problems"] == [], "the package is back to stock: a new run plans clean"


def test_refusals_are_named(fake_pkg, monkeypatch):
    backup = fake_pkg / "utils" / ("scatter_utils.py" + det.BACKUP_SUFFIX)
    backup.write_bytes(STOCK_SCATTER)
    assert [p for p in det.plan()["problems"] if "a backup from an earlier run is present (never overwritten)" in p and str(backup) in p]
    with pytest.raises(det.DetError, match="--det 1 refused: .*a backup from an earlier run"):
        det.Patch(det.plan())
    backup.unlink()
    shutil.copy(os.path.join(det.detref_dir(), "scatter_utils.py"), _target(fake_pkg, "scatter_utils.py"))
    assert [p for p in det.plan()["problems"] if "already carries the detref bytes" in p]
    _target(fake_pkg, "scatter_utils.py").write_bytes(STOCK_SCATTER)
    _target(fake_pkg, "det_segment_reduce.py").write_text("stale\n")
    assert [p for p in det.plan()["problems"] if "a stale det file from an earlier run" in p and "det_segment_reduce.py" in p]
    _target(fake_pkg, "det_segment_reduce.py").unlink()
    _target(fake_pkg, "scatter_utils.py").unlink()
    assert [p for p in det.plan()["problems"] if "the stock file the copy replaces is missing" in p]
    _target(fake_pkg, "scatter_utils.py").write_bytes(STOCK_SCATTER)
    monkeypatch.setattr(det, "dir_writable", lambda path: False)
    assert [p for p in det.plan()["problems"] if "is not writable by this process" in p]
    monkeypatch.setattr(det, "dir_writable", lambda path: True)
    shutil.rmtree(fake_pkg / "utils")
    assert [p for p in det.plan()["problems"] if "has no utils/ directory" in p]
    monkeypatch.setattr(det, "protenix_package_dir", lambda: None)
    assert det.plan()["problems"] == ["protenix package not installed: no target for the detref copy (<protenix package>/utils/)"]
    rel = kits.kit_rel("flashpairformer", f"{det.DETREF_RELDIR}/scatter_utils.py")
    monkeypatch.setattr(kits, "missing_kit_files", lambda rels: [f"{rel}: missing"])
    assert f"detref {rel}: missing" in det.plan()["problems"]


def test_apply_env_exports_the_recipe_and_records_what_it_replaced(monkeypatch):
    env = {"PTX_DET": "0", "HOME": "/h"}
    assert det.apply_env(env) == {"CUBLAS_WORKSPACE_CONFIG": None, "PTX_DET": "0"}
    assert env == {"PTX_DET": "1", "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "HOME": "/h"}
    exc = det.stock_exception()
    assert exc == {"env": det.DET_ENV, "pythonpath": [os.path.join(KIT, "src")], "sitecustomize": os.path.join(KIT, "src", "sitecustomize.py"),
                   "modules": ["sitecustomize"]}


# ---------------------------------------------------------------------------------------------- the stock child's proof ----
def _proof(environ, path, modules, det_exception):
    return stock_pred.env_proof(stack.DEFAULT_STOCK_ENV_ABSENT, stack.kit_class_dirs(), environ=environ, modules=modules, path=path, meta_path=[],
                                det_exception=det_exception)


def test_env_proof_allows_exactly_the_det_exception():
    src = os.path.join(KIT, "src")
    exc = det.stock_exception()

    class M:                                                              # a module stand-in with a __file__
        def __init__(self, f): self.__file__ = f
    sc = M(os.path.join(src, "sitecustomize.py"))
    base_env = {"PATH": "/usr/bin", "PROTENIX_ROOT_DIR": "/zoo", **det.DET_ENV}
    base_mods = {"sitecustomize": sc, "torch": M("/site/torch/__init__.py"), "os": os}
    base_path = ["", src, "/site"]
    p = _proof(base_env, base_path, base_mods, exc)
    assert p["ok"] is True and p["det_exception"]["deviations"] == [] and p["forbidden_present"] == ["PTX_DET"], "CUBLAS_WORKSPACE_CONFIG is not a must-be-absent name"
    assert p["det_exception"]["torch_loaded_by_sitecustomize"] is True and p["det_exception"]["modules"] == ["sitecustomize"]
    # every deviation from the carve-out refuses by name
    p = _proof({**base_env, "PTX_BLK": "2"}, base_path, base_mods, exc)
    assert p["ok"] is False and any("forbidden names present" in d and "PTX_BLK" in d for d in p["det_exception"]["deviations"])
    p = _proof({**base_env, "PROTENIX_OPT": "exact"}, base_path, base_mods, exc)
    assert p["ok"] is False and any("PROTENIX_OPT" in d for d in p["det_exception"]["deviations"])
    p = _proof({**base_env, "PTX_DET": "0"}, base_path, base_mods, exc)
    assert p["ok"] is False and any(d.startswith("PTX_DET='0' != the recipe's '1'") for d in p["det_exception"]["deviations"])
    p = _proof({k: v for k, v in base_env.items() if k != "CUBLAS_WORKSPACE_CONFIG"}, base_path, base_mods, exc)
    assert p["ok"] is False and p["det_exception"]["deviations"] == ["CUBLAS_WORKSPACE_CONFIG=None != the recipe's ':4096:8'"]
    p = _proof(base_env, base_path + [os.path.join(KIT, "third_party")], base_mods, exc)
    assert p["ok"] is False and any("kit directories on sys.path" in d for d in p["det_exception"]["deviations"])
    p = _proof(base_env, ["", "/site"], {"os": os}, exc)
    assert p["ok"] is False and any("kit directories on sys.path [] != the recipe's" in d for d in p["det_exception"]["deviations"]) \
        and any("the process's sitecustomize is none" in d for d in p["det_exception"]["deviations"])
    p = _proof(base_env, base_path, {**base_mods, "ptx_trunk2_levers": M(os.path.join(src, "ptx_trunk2_levers.py"))}, exc)
    assert p["ok"] is False and any("kit modules loaded ['ptx_trunk2_levers', 'sitecustomize'] != ['sitecustomize']" in d for d in p["det_exception"]["deviations"])
    p = _proof(base_env, base_path, {**base_mods, "sitecustomize": M("/site/sitecustomize.py")}, exc)
    assert p["ok"] is False and any("the process's sitecustomize is /site/sitecustomize.py" in d for d in p["det_exception"]["deviations"])
    # without the exception the same environment is NOT stock (the names, the dir, the sitecustomize, torch)
    p = _proof(base_env, base_path, base_mods, None)
    assert p["ok"] is False and p["det_exception"] is None and p["forbidden_present"] == ["PTX_DET"] and p["kit_sitecustomize"] and p["torch_loaded_before_proof"]
    after = stock_pred.after_call_check(stack.kit_class_dirs(), exc, modules=base_mods)
    assert after == {"kit_modules_loaded_after": ["sitecustomize"], "lever_module_imported": False, "after_ok": True}
    after = stock_pred.after_call_check(stack.kit_class_dirs(), exc, modules={**base_mods, "ptx_trunk2_levers": M(os.path.join(src, "x.py"))})
    assert after["after_ok"] is False and after["lever_module_imported"] is True
    assert stock_pred.after_call_check(stack.kit_class_dirs(), None, modules=base_mods)["after_ok"] is False, "no exception: sitecustomize from the kit is a kit module"


# --------------------------------------------------------------------------------------------------------- the CLI ----
def test_pred_det_1_on_the_kit_arms(core, fake_stock, fake_pkg, tmp_path, monkeypatch):     # noqa: F811
    """exact|fast --det 1: the env is exported and the detref copy is in place BEFORE enable() runs, the stock CLI runs, the package dir is
    restored after it returns (the patch record: the level, the env, the two files with sha256 before/after and restored=True)."""
    seen = {}
    real_enable = protenix_opt.enable

    def enable(mode):
        seen["target_at_enable"] = _target(fake_pkg, "scatter_utils.py").read_bytes()
        seen["env_at_enable"] = {k: os.environ.get(k) for k in det.DET_ENV}
        seen["segment_reduce_present"] = _target(fake_pkg, "det_segment_reduce.py").exists()
        return real_enable(mode)
    monkeypatch.setattr(protenix_opt, "enable", enable)
    patches = []
    real_patch = det.Patch

    class RecordingPatch(real_patch):                                        # the same object the CLI applies and restores; kept for inspection
        def __init__(self, *a, **k):
            super().__init__(*a, **k); patches.append(self)
    monkeypatch.setattr(det, "Patch", RecordingPatch)
    detref_scatter = open(os.path.join(det.detref_dir(), "scatter_utils.py"), "rb").read()
    for mode in ("exact", "fast"):
        out = tmp_path / mode
        rc = cli.main(["pred", "--mode", mode, "--det", "1", "--input", "in.json", "--out_dir", str(out)])
        assert rc == 0 and core.calls[-1] == (mode, False) and fake_stock["main_args"][-1] == ["pred", "--input", "in.json", "--out_dir", str(out)]
        assert seen["target_at_enable"] == detref_scatter and seen["segment_reduce_present"] and seen["env_at_enable"] == det.DET_ENV
        assert _target(fake_pkg, "scatter_utils.py").read_bytes() == STOCK_SCATTER and not _target(fake_pkg, "det_segment_reduce.py").exists()
        d = patches[-1].record()
        assert d["level"] == 1 and d["env"] == det.DET_ENV and d["package_dir"] == str(fake_pkg) and d["stock_exception"] is None and d["restore_problems"] == []
        assert [os.path.basename(p["target"]) for p in d["patch"]] == list(det.DETREF_FILES) and all(p["restored"] for p in d["patch"])
        assert d["patch"][0]["sha256_before"] == stack.sha256_file(str(_target(fake_pkg, "scatter_utils.py"))) and d["patch"][0]["sha256_after"] == stack.sha256_file(os.path.join(det.detref_dir(), "scatter_utils.py"))
        assert d["patch"][0]["backup"].endswith(det.BACKUP_SUFFIX) and d["patch"][1]["backup"] is None and d["patch"][1]["sha256_before"] is None
        assert set(d["patch"][0]) == {"source", "target", "backup", "sha256_before", "sha256_after", "restored"}
        for k in det.DET_ENV:
            os.environ.pop(k, None)                                       # not monkeypatch.delenv: it would record "1" as the value to restore
    # a refused activation still restores
    core.active = False
    rc = cli.main(["pred", "--mode", "exact", "--det", "1", "--input", "in.json", "--out_dir", str(tmp_path / "r")])
    assert rc == cli.EXIT_NOT_ACTIVE and _target(fake_pkg, "scatter_utils.py").read_bytes() == STOCK_SCATTER and not _target(fake_pkg, "det_segment_reduce.py").exists()


@needs_durable_install
def test_pred_det_1_on_the_stock_arm(core, fake_pkg, tmp_path, monkeypatch, capsys):     # noqa: F811
    """off --det 1: the stock child gets exactly the recipe (the det names, $FPF_HOME/src alone among the kit dirs on PYTHONPATH — the kit's
    sitecustomize sets deterministic algorithms), proves the carve-out and nothing beyond it, the copy is in place while it runs and
    restored after it exits; the child's proof (in the call's temporary directory) carries the det_exception it ran under."""
    root = _stock_stub.write_stub_runner(str(tmp_path / "stub"))
    for m in [m for m in sys.modules if m == "runner" or m.startswith("runner.")]:
        monkeypatch.delitem(sys.modules, m)
    monkeypatch.syspath_prepend(root)
    monkeypatch.setenv("PYTHONPATH", root)
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    monkeypatch.setenv("PTX_BLK", "2")                                   # pollution the stock child must not see
    monkeypatch.setattr(stack, "_REPORT", None)
    monkeypatch.setattr(stack, "_kit_sitecustomize_present", lambda: None)
    monkeypatch.setattr(cli.tempfile, "tempdir", str(tmp_path))
    out = tmp_path / "o"
    rc = cli.main(["pred", "--mode", "off", "--det", "1", "--input", "x", "--out_dir", str(out)])
    err = capsys.readouterr().err
    assert rc == 0, rc
    rec = json.loads((out / _stock_stub.RUN_RECORD).read_text())
    assert rec["env_kit_names"] == ["PTX_DET"] and rec["sitecustomize"] == os.path.join(KIT, "src", "sitecustomize.py")
    assert rec["torch_loaded"] is (importlib.util.find_spec("torch") is not None)   # the sitecustomize's PTX_DET block imports torch where torch is installed
    assert rec["pythonpath"].split(os.pathsep) == [os.path.join(KIT, "src"), root] and rec["modules_kit"] == []
    import glob
    proof = json.load(open(sorted(glob.glob(os.path.join(str(tmp_path), "protenix_opt_stock_*", cli.STOCK_PROOF_NAME)))[-1], encoding="utf-8"))
    assert _target(fake_pkg, "scatter_utils.py").read_bytes() == STOCK_SCATTER and not _target(fake_pkg, "det_segment_reduce.py").exists(), "restored after the child exits"
    assert proof["ok"] is True and proof["det_exception"]["deviations"] == [] and proof["det_exception"]["env"] == det.DET_ENV
    assert proof["det_exception"]["pythonpath"] == [os.path.join(KIT, "src")] and proof["det_exception"]["modules"] == ["sitecustomize"]
    assert proof["forbidden_present"] == ["PTX_DET"] and proof["kit_dirs_on_path"] == [os.path.join(KIT, "src")]
    assert proof["kit_modules_loaded_after"] == ["sitecustomize"] and proof["lever_module_imported"] is False and proof["after_ok"] is True
    sub = [l for l in err.splitlines() if l.startswith("[protenix-opt] stock subprocess:")][-1]
    assert "env stripped of " in sub and "PTX_BLK" in sub.split("env stripped of ")[1].split(";")[0] and "PTX_DET" not in sub.split("env stripped of ")[1].split(";")[0]
    assert "det exception: CUBLAS_WORKSPACE_CONFIG=:4096:8 PTX_DET=1 PYTHONPATH+=" + os.path.join(KIT, "src") in sub
    assert _target(fake_pkg, "scatter_utils.py").read_bytes() == STOCK_SCATTER and not _target(fake_pkg, "det_segment_reduce.py").exists()
    # the child refuses when the carve-out is exceeded: a lever key beside the det names
    r = subprocess.run([sys.executable, "-s", "-m", "protenix_opt.stock_pred", "--proof-json", str(tmp_path / "p.json"), "--env-absent", ",".join(stack.DEFAULT_STOCK_ENV_ABSENT),
                        "--kit-dirs", os.pathsep.join(stack.kit_class_dirs()), "--det-env", "CUBLAS_WORKSPACE_CONFIG=:4096:8,PTX_DET=1", "--det-path", os.path.join(KIT, "src"),
                        "--", "pred", "--input", "x", "--out_dir", str(tmp_path / "o2")],
                       env={k: v for k, v in os.environ.items() if not k.startswith(stack.DEFAULT_STOCK_ENV_ABSENT)} | {"PYTHONPATH": f"{KIT}/src:{root}", **det.DET_ENV, "PTX_BLK": "2"},
                       capture_output=True, text=True)
    assert r.returncode == stock_pred.EXIT_NOT_STOCK and "NOT STOCK (det exception not met): forbidden names present" in r.stderr and "PTX_BLK" in r.stderr
    assert json.load(open(tmp_path / "p.json"))["ok"] is False


def test_check_det_1_prints_the_det_lines_and_refuses_on_problems(core, fake_pkg, monkeypatch, capsys):     # noqa: F811
    from protenix_opt import report
    monkeypatch.setattr(stack, "activate", lambda mode, strict=False, trigger=None, dry_run=False, det=None:
                        (lambda rep: (report.log_activation(rep), rep)[1])({"active": False, "dry_run": True, "mode": mode, "env": {}, "reason": "dry run"}))
    rc = cli.main(["check", "--mode", "exact", "--det", "1", "--json"])
    cap = capsys.readouterr()
    det_lines = [l for l in cap.err.splitlines() if l.startswith("[protenix-opt] DET ")]
    assert rc == 0 and det_lines[-1].startswith("[protenix-opt] DET ok:") and len(det_lines) == 5
    assert cap.err.index("[protenix-opt] DET ") < cap.err.index("[protenix-opt] DRY-RUN")
    rep = json.loads(cap.out)
    assert rep["det"]["problems"] == [] and rep["det"]["package_dir"] == str(fake_pkg)
    assert cli.main(["check", "--mode", "exact"]) == 0 and "[protenix-opt] DET " not in capsys.readouterr().err, "--det 0 (the default): no DET lines"
    (fake_pkg / "utils" / ("scatter_utils.py" + det.BACKUP_SUFFIX)).write_bytes(STOCK_SCATTER)
    rc = cli.main(["check", "--mode", "off", "--det", "1"])
    err = capsys.readouterr().err
    assert rc == cli.EXIT_FAIL and "[protenix-opt] DET PROBLEMS 1" in err and "a backup from an earlier run is present" in err
    with pytest.raises(SystemExit) as e:                                  # argparse: an invalid level is a usage error
        cli.main(["check", "--det", "2"])
    assert e.value.code == cli.EXIT_USAGE


