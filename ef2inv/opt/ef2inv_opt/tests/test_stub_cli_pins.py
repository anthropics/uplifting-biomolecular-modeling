"""The kit's `modal` stand-in leaves the cookbook's classes unchanged and is registered without a path entry, and the design kit's own `modal`
stub is genuinely absent from the carried tree with no stray top-level `modal` package anywhere under opt/; check_pins refuses a wrong install; the
CLI's check and its refusals; the launcher's argv per arm; the output-set names."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile

import pytest

from .. import launch, modes, outputs
from . import _paths as P

IGNORED_TREE_DIRS = {"__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".ipynb_checkpoints", ".venv", ".idea", ".vscode", "do_not_commit"}


def test_sdk_standin_is_a_no_op_and_registered_by_the_arm(monkeypatch):
    """ef2inv_opt/_absent_sdk_stub.py (the kit's stub bytes, carried once here) leaves decorated objects unchanged; stock_design.prepare_arm_process registers
    it as sys.modules["modal"] without touching sys.path, refuses a foreign `modal`, and import_cookbook refuses to run before it."""
    from .. import _absent_sdk_stub as modal, stock_design as SD
    assert os.path.abspath(modal.__file__) == os.path.abspath(P.SDK_STANDIN) == os.path.abspath(modes.SDK_STANDIN)
    app = modal.App(name="x", image=modal.Image.debian_slim(), volumes={"/m": modal.Volume.from_name("v", create_if_missing=True)})

    class C:
        def load(self):
            return 1
    assert app.cls(gpu="H100", timeout=1)(C) is C and modal.enter()(C.load) is C.load and modal.method()(C.load) is C.load
    monkeypatch.delitem(sys.modules, "modal", raising=False)
    path0 = list(sys.path)
    SD.prepare_arm_process(modes.MODES["off"])
    assert sys.modules["modal"] is modal and sys.path == path0
    SD.prepare_arm_process(modes.MODES["off"])                                                        # idempotent
    monkeypatch.setitem(sys.modules, "modal", type(sys)("modal"))
    with pytest.raises(RuntimeError):
        SD.prepare_arm_process(modes.MODES["off"])                                                    # a foreign modal already imported: refused by name
    with pytest.raises(RuntimeError):
        SD.import_cookbook(P.STOCK_FILE, "_never_")                                                   # the cookbook never runs on a foreign modal
    monkeypatch.delitem(sys.modules, "modal")
    with pytest.raises(ValueError):
        SD.prepare_arm_process(modes.MODES["fast"], None)                                             # a kit arm names its k/ dir
def test_check_pins_refuses_wrong_file(tmp_path):
    cp = os.path.join(P.ROOT, "stock", "check_pins.py")
    wrong = tmp_path / "binder_design.py"; wrong.write_text("x = 1\n")
    r = subprocess.run([sys.executable, "-I", cp, "--json", "--cookbook", str(wrong), "--hf-home", str(tmp_path)], capture_output=True, text=True)
    assert r.returncode == 3
    rep = json.loads(r.stdout)
    assert any("sha256" in f and "!=" in f for f in rep["fails"]) or any("upstream file" in f for f in rep["fails"])
    assert any(f.startswith("weights ") for f in rep["fails"])


def test_check_pins_accepts_the_extract_as_the_file(tmp_path):
    cp = os.path.join(P.ROOT, "stock", "check_pins.py")
    r = subprocess.run([sys.executable, "-I", cp, "--json", "--cookbook", P.STOCK_FILE, "--hf-home", str(tmp_path)], capture_output=True, text=True)
    rep = json.loads(r.stdout)
    assert rep["facts"]["cookbook_sha256"] == hashlib.sha256(open(P.STOCK_FILE, "rb").read()).hexdigest()
    assert not any("upstream file" in f for f in rep["fails"])


def test_kit_presence_reports_presence():
    """modes.kit_presence() on the real tree: the design kit's root and its entry module (k/ + KIT_PATH_MARKER) are both there (no hash of
    any kind -- the kit's bytes are identified by the git commit that carries them, STOCK.md \u00a7Kit)."""
    ki = modes.kit_presence(P.ROOT)
    assert ki["root"] == P.DK and ki["root_present"] is True
    assert ki["entry"] == os.path.join(P.KDIR, modes.KIT_PATH_MARKER) and ki["entry_present"] is True
    assert set(ki) == {"root", "root_present", "entry", "entry_present"}


def test_kit_presence_missing_entry_refuses_a_fastkit_mode_by_name(tmp_path, monkeypatch):
    """A half-copied design kit (root present, entry module missing) is refused by name from check/design pre-flight (_check_facts):
    it does not reach the kit's own bytes at all. A root that does not exist at all is refused by a different named reason. Neither
    check fires for `off` (the stock arm never touches the design kit)."""
    from .. import cli as C
    broken_root = tmp_path / "ESMFOLD2_DESIGN_FAST_INFERENCE_KIT_v1"
    (broken_root / "k").mkdir(parents=True)                                            # k/ exists, but not the entry module
    real_kp = modes.kit_paths(P.ROOT)
    broken_kp = dict(real_kp, design_kit=str(broken_root), k_dir=str(broken_root / "k"))
    monkeypatch.setattr(C.MD, "kit_paths", lambda model_opt=None: broken_kp)
    rep = C._check_facts(modes.MODES["fast"], P.ROOT, P.PINS, cb=P.STOCK_FILE, require_gpu=False, weights=False)
    assert rep["kit_presence"]["root_present"] is True and rep["kit_presence"]["entry_present"] is False
    assert any(f"{modes.KIT_PATH_MARKER} missing" in p or str(broken_root / "k") in p for p in rep["problems"])

    missing_kp = dict(real_kp, design_kit=str(tmp_path / "nope"), k_dir=str(tmp_path / "nope" / "k"))
    monkeypatch.setattr(C.MD, "kit_paths", lambda model_opt=None: missing_kp)
    rep2 = C._check_facts(modes.MODES["fast"], P.ROOT, P.PINS, cb=P.STOCK_FILE, require_gpu=False, weights=False)
    assert rep2["kit_presence"]["root_present"] is False
    assert any(str(tmp_path / "nope") in p for p in rep2["problems"])

    rep3 = C._check_facts(modes.MODES["off"], P.ROOT, P.PINS, cb=P.STOCK_FILE, require_gpu=False, weights=False)   # off never needs the design kit
    assert not [p for p in rep3["problems"] if "design kit" in p]


def test_cli_check_json_and_refusal():
    core = os.path.normpath(os.path.join(P.ROOT, "..", "common", "opt_core"))          # the install: the pinned core beside the tree, then the package
    env = dict(os.environ, MODEL_OPT=P.ROOT, PYTHONPATH=os.pathsep.join([P.OPT, core]))
    r = subprocess.run([sys.executable, "-m", "ef2inv_opt", "check", "--mode", "exact", "--json"], capture_output=True, text=True, env=env, cwd=P.OPT)
    rep = json.loads(r.stdout)
    assert rep["mode"] == "exact" and rep["kit_switch"] == "exact" and "fastkit_cookbook" not in rep
    assert rep["jit_cache"]["form"] in ("unset", "shared", "per-box") and not [p for p in rep["problems"] if "not keyed" in p]
    r = subprocess.run([sys.executable, "-m", "ef2inv_opt", "check", "--mode", "nosuch"], capture_output=True, text=True, env=env, cwd=P.OPT)
    assert r.returncode == 3 and r.stdout.startswith("REFUSED")
    r = subprocess.run([sys.executable, "-m", "ef2inv_opt", "design", "--mode", "fast", "--levers", "x", "--target-name", "t", "--target-sequence", "ACDEFGHIK", "--binder-len", "80", "--seed", "0", "--out", "o"], capture_output=True, text=True, env=env,
                       cwd=P.OPT)
    assert r.returncode == 2 and "unrecognized arguments: --levers" in r.stderr                       # no lever axis: argparse rejects it (modes are off | exact | fast)


def test_launcher_argv_and_env():
    for name in ("off", "exact", "fast", "big"):
        m = modes.MODES[name]
        argv = launch.argv_for(m, P.ROOT, cookbook_stock="/x/binder_design.py", target_name="c", target_sequence="ACDEFGHIK", binder_len=80, seed=3, out="/o", python="py")
        assert argv[:4] == ["py", "-I", "-m", "ef2inv_opt.stock_design"] and "--stubs-dir" not in argv
        assert ("--kit-dir" in argv) == m.is_kit
        assert argv[argv.index("--cookbook") + 1] == "/x/binder_design.py"                  # every mode runs the one stock file
    env = launch.arm_env(modes.MODES["off"], P.PINS["stock_environment"]["must_be_absent_prefixes"], base={"EF2_FAST_KIT": "agk3", "ESMFOLD2_ANY": "1", "CUBLAS_WORKSPACE_CONFIG": ":16:8", "MODEL_OPT": "/t", "PATH": "/bin"})
    assert "EF2_FAST_KIT" not in env and "ESMFOLD2_ANY" not in env and "CUBLAS_WORKSPACE_CONFIG" not in env and env["MODEL_OPT"] == "/t" and env["PYTHONDONTWRITEBYTECODE"] == "1"


def test_output_set_names():
    assert outputs.NAMES[:2] == ("design.pdb", "design.cif") and "trajectory.jsonl" in outputs.NAMES and "opt_manifest.json" in outputs.NAMES


def test_run_sh_gate_and_usage():
    src = open(os.path.join(P.ROOT, "run.sh")).read()
    assert "hardware.py" in src and "nvidia-smi" not in src and 'case "$CMD" in design|check|warm|install)' in src      # the gate lives in the package (one source); the four verbs


# ---- weights: warn-and-run (stock/check_pins.py weights_gate) — missing is refused by name; present files are digested and pinned | unpinned
def _check_pins_module():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_pins_under_test", os.path.join(P.ROOT, "stock", "check_pins.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


def _weights_tree(tmp_path, files):
    """A one-snapshot HF layout under tmp_path with the given {rel: bytes}; returns (pins-like weights table pinned to THESE bytes, hf_home)."""
    import hashlib
    repo, commit = "biohub/Tiny", "0123456789abcdef0123456789abcdef01234567"
    d = tmp_path / "hub" / "models--biohub--Tiny" / "snapshots" / commit; d.mkdir(parents=True)
    recs = {}
    for rel, b in files.items():
        (d / rel).write_bytes(b); recs[rel] = {"sha256": hashlib.sha256(b).hexdigest(), "size_bytes": len(b)}
    return {repo: {"snapshot_commit": commit, "hf_home": str(tmp_path), "files": recs}}, str(tmp_path), d


def _run_main_weights_only(cp, monkeypatch, capsys, pins_path, hf, extra=()):
    """check_pins.main with the code checks stubbed to hold (they need the stack's install), so the exit code is the weights gate's."""
    real = cp.check
    def weights_only(pins, weights=True, hf_home=None, cookbook_path=None, **kw):
        fails, facts = real({"weights": pins["weights"], "transformers_fork": {"commit": "", "installed_files_sha256": {}}}, weights, hf_home, cookbook_path, **kw)
        return [f for f in fails if f.startswith("weights")], facts
    monkeypatch.setattr(cp, "check", weights_only)
    rc = cp.main(["--pins", str(pins_path), "--hf-home", hf, *extra])
    return rc, capsys.readouterr()


def test_weights_gate_unknown_digest_runs_unpinned(tmp_path, monkeypatch, capsys):
    cp = _check_pins_module()
    table, hf, d = _weights_tree(tmp_path, {"config.json": b'{"a": 1}\n', "model.safetensors": b"\x00" * 64})
    (d / "model.safetensors").write_bytes(b"\x01" * 64)                                            # same size, other bytes: a user-supplied checkpoint
    pins_path = tmp_path / "PINS.json"; pins_path.write_text(json.dumps({"weights": table}))
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=True)
    assert fails == [] and census["biohub/Tiny"]["word"] == "unpinned" and len(census["biohub/Tiny"]["sha256"]) == 64
    assert len(lines) == 1 and "NOT PINNED — the speed and exactness figures apply to the pinned weights only" in lines[0] and lines[0].startswith("weights=biohub/Tiny@0123456789 sha256=")
    rc, io = _run_main_weights_only(cp, monkeypatch, capsys, pins_path, hf)
    assert rc == 0 and io.err.count("NOT PINNED") == 1                                               # exit 0: it runs, with ONE line naming it
    (d / "model.safetensors").write_bytes(b"\x01" * 65)                                             # a size mismatch on a PRESENT file is unpinned too, not missing
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=False)
    assert fails == [] and census["biohub/Tiny"]["word"] == "unpinned" and "size 65 != 64" in status["biohub/Tiny"]


def test_weights_gate_known_digest_is_pinned(tmp_path, monkeypatch, capsys):
    cp = _check_pins_module()
    table, hf, d = _weights_tree(tmp_path, {"config.json": b'{"a": 1}\n', "model.safetensors": b"\x00" * 64})
    pins_path = tmp_path / "PINS.json"; pins_path.write_text(json.dumps({"weights": table}))
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=True)
    assert fails == [] and census["biohub/Tiny"] == {"word": "pinned", "sha256": census["biohub/Tiny"]["sha256"], "checked": "sha256", "snapshot_commit": table["biohub/Tiny"]["snapshot_commit"],
                                                   "files": 2, "ok": 2, "differs": [], "missing": []}
    assert lines == [f"weights=biohub/Tiny@0123456789 sha256={census['biohub/Tiny']['sha256'][:12]} (pinned)"] and status["biohub/Tiny"].startswith("2/2 files ok (pinned)")
    rc, io = _run_main_weights_only(cp, monkeypatch, capsys, pins_path, hf)
    assert rc == 0 and "(pinned)" in io.err and "NOT PINNED" not in io.err
    rc, io = _run_main_weights_only(cp, monkeypatch, capsys, pins_path, hf, extra=("--json",))
    rep = json.loads(io.out); assert rc == 0 and rep["ok"] and rep["facts"]["weights_census"]["biohub/Tiny"]["word"] == "pinned"


def test_weights_gate_missing_file_is_refused_by_name(tmp_path, monkeypatch, capsys):
    cp = _check_pins_module()
    table, hf, d = _weights_tree(tmp_path, {"config.json": b'{"a": 1}\n', "model.safetensors": b"\x00" * 64})
    (d / "model.safetensors").unlink()
    pins_path = tmp_path / "PINS.json"; pins_path.write_text(json.dumps({"weights": table}))
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=True)
    assert fails == ["weights biohub/Tiny/model.safetensors: missing"] and census["biohub/Tiny"]["word"] == "missing" and census["biohub/Tiny"]["missing"] == ["model.safetensors"]
    rc, io = _run_main_weights_only(cp, monkeypatch, capsys, pins_path, hf)
    assert rc == 3 and "model.safetensors: missing" in io.err
    fails, *_ = cp.weights_gate(table, str(tmp_path / "elsewhere"), hash_files=False)                # the snapshot dir itself absent: refused by name too
    assert len(fails) == 1 and "snapshot 0123456789abcdef0123456789abcdef01234567 missing under" in fails[0]


def test_weights_gate_other_revision_runs_unpinned_and_no_snapshot_is_refused(tmp_path, monkeypatch, capsys):
    """An unknown checkpoint installed as ANOTHER HF revision (the pinned snapshot dir absent, refs/main naming the present one) runs — worded
    NOT PINNED with `revision <found> != pin <pinned>`, exit 0; a repo with no snapshot at all is the one refusal by name."""
    cp = _check_pins_module()
    table, hf, d = _weights_tree(tmp_path, {"config.json": b'{"a": 1}\n', "model.safetensors": b"\x00" * 64})
    other = "fedcba9876543210fedcba9876543210fedcba98"
    d.rename(d.parent / other)                                                                          # the same files under another revision
    refs = d.parent.parent / "refs"; refs.mkdir(); (refs / "main").write_text(other + "\n")
    pins_path = tmp_path / "PINS.json"; pins_path.write_text(json.dumps({"weights": table}))
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=False)
    assert fails == [] and census["biohub/Tiny"]["word"] == "unpinned" and census["biohub/Tiny"]["missing"] == []
    assert census["biohub/Tiny"]["differs"] == ["revision fedcba9876 != pin 0123456789"] and census["biohub/Tiny"]["ok"] == 2
    assert len(lines) == 1 and lines[0].startswith("weights=biohub/Tiny@fedcba9876 ") and "NOT PINNED — the speed and exactness figures apply to the pinned weights only" in lines[0]
    rc, io = _run_main_weights_only(cp, monkeypatch, capsys, pins_path, hf, extra=("--no-weights",))
    assert rc == 0 and io.err.count("NOT PINNED") == 1 and "revision fedcba9876 != pin 0123456789" in io.err     # exit 0: it runs, named on ONE line
    (refs / "main").unlink()                                                                            # no refs/main: the newest snapshot dir runs all the same
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=True)
    assert fails == [] and census["biohub/Tiny"]["word"] == "unpinned" and lines[0].startswith("weights=biohub/Tiny@fedcba9876 sha256=")
    import shutil; shutil.rmtree(d.parent)                                                              # no snapshot of the repo at all: weights missing — refused by name
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=False)
    assert len(fails) == 1 and "snapshot 0123456789abcdef0123456789abcdef01234567 missing under" in fails[0] and census["biohub/Tiny"]["word"] == "missing"
    rc, io = _run_main_weights_only(cp, monkeypatch, capsys, pins_path, hf, extra=("--no-weights",))
    assert rc == 3 and "missing under" in io.err


def test_launch_digests_weights_same_size_other_bytes_is_unpinned_and_the_memo_hashes_once(tmp_path, monkeypatch, capsys):
    """An unknown checkpoint of the PINNED byte count (AF-class weights share one size) is detected only by the digest — the design's launch
    digests (sha256 through the (path, size, mtime_ns) memo): worded NOT PINNED by name, exit 0; the memo hashes an unchanged file once and re-hashes
    on an mtime change; presence-and-size alone would have called it pinned."""
    cp = _check_pins_module()
    table, hf, d = _weights_tree(tmp_path, {"config.json": b'{"a": 1}\n', "model.safetensors": b"\x00" * 64})
    (d / "model.safetensors").write_bytes(b"\x01" * 64)                                              # same size, other bytes
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=False)
    assert fails == [] and census["biohub/Tiny"]["word"] == "pinned"                                 # size alone cannot see it (why the launch digests)
    memo = tmp_path / "memo" / "weights_sha256.json"; calls = []
    real = cp.sha256; monkeypatch.setattr(cp, "sha256", lambda p, *a, **k: (calls.append(p), real(p, *a, **k))[1])
    fails, status, census, lines = cp.weights_gate(table, hf, hash_files=True, cache=str(memo))
    assert fails == [] and census["biohub/Tiny"]["word"] == "unpinned" and census["biohub/Tiny"]["differs"] == ["model.safetensors: sha256 " + cp.sha256(str(d / "model.safetensors"))[:12] + " != " + table["biohub/Tiny"]["files"]["model.safetensors"]["sha256"][:12]]
    n = len(calls); assert n >= 2 and memo.is_file()
    cp.weights_gate(table, hf, hash_files=True, cache=str(memo)); assert len(calls) == n               # unchanged files: memo hits, nothing re-hashed
    p = d / "config.json"; st = os.stat(p); os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))
    cp.weights_gate(table, hf, hash_files=True, cache=str(memo)); assert len(calls) == n + 1           # an mtime change re-hashes that file
    pins_path = tmp_path / "PINS.json"; pins_path.write_text(json.dumps({"weights": table}))
    rc, io = _run_main_weights_only(cp, monkeypatch, capsys, pins_path, hf, extra=("--weights-cache", str(memo)))
    assert rc == 0 and io.err.count("NOT PINNED") == 1 and "weights=biohub/Tiny@0123456789 sha256=" in io.err               # the launch's form: digested, named, runs


def test_mode_vs_environment_disagreement_is_a_usage_error_exit_2(monkeypatch, capsys):
    """`--mode M` with `EF2INV_OPT` set to another mode is a USAGE error — exit 2 with the disagreement named — in run.sh (before any python runs) and in
    the package's own resolver; 3 stays the refusal / NOT ACTIVE code."""
    import subprocess, sys
    from ef2inv_opt import cli
    env = {"PATH": os.environ.get("PATH", "/usr/bin:/bin"), "EF2INV_OPT": "fast"}
    r = subprocess.run(["bash", os.path.join(P.ROOT, "run.sh"), "check", "--mode", "exact"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "run.sh: --mode exact disagrees with EF2INV_OPT=fast" in r.stderr, (r.returncode, r.stderr[-300:])
    r = subprocess.run(["bash", os.path.join(P.ROOT, "run.sh"), "nonsense"], env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 2 and "unknown command nonsense" in r.stderr                              # the sibling usage branches keep 2
    monkeypatch.setenv("EF2INV_OPT", "fast")
    with pytest.raises(SystemExit) as ex:
        cli._mode_from("exact")
    assert ex.value.code == 2 and "ef2inv-opt: --mode exact disagrees with EF2INV_OPT=fast" in capsys.readouterr().err
    assert cli._mode_from("fast") is cli.MD.MODES["fast"] and cli._mode_from(None).name == "fast" and cli.MD.alias_word(None) is None   # agreement and env-only resolve; `fast` is its own row


DK_SDK_STUB = "cookbook/stubs/modal/__init__.py"   # the design kit's own `modal` stub: not in the tree


def test_design_kit_stub_absent_and_no_stray_sdk_package():
    """The design kit's own `modal` stub is not in the carried tree, and no directory under opt/ holds a top-level `modal` package (the
    stand-in lives once, as opt/ef2inv_opt/_absent_sdk_stub.py)."""
    assert not os.path.exists(os.path.join(P.DK, DK_SDK_STUB)), DK_SDK_STUB
    for r, dirs, fns in os.walk(P.OPT):                                                                          # no directory of the tree holds a `modal` package
        assert not (os.path.basename(r) == "modal" and "__init__.py" in fns), r


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def test_no_bytecode_under_forward():
    for r, dirs, fns in os.walk(P.FWD):
        assert "__pycache__" not in dirs, r
        assert not any(f.endswith(".pyc") for f in fns), r
