"""The gates name what is missing; the activation line; the model-process environment (the stock proof) and the caches."""
import json
import os
import re
import subprocess

from af3_torch_opt import cli, modes, stack

from .conftest import archive_or_skip
from af3_torch_opt.report import activation_line

from .conftest import HOME


def test_home_and_kit_dirs():
    assert stack.home() == HOME
    assert os.path.isdir(stack.kit_home()) and os.path.isdir(stack.dtk_home())
    assert stack.kit_home().endswith(os.path.join("opt", "forward", "af3t")) and stack.dtk_home().endswith(os.path.join("opt", "forward", "dtk"))


def test_gates_name_every_missing_thing(monkeypatch, tmp_path):
    """Interpreters and the fork checkout unset (each named: no default), then named but absent (paths that do not exist / a checkout
    without run_alphafold.py), the parameters dir unset, then set to a dir without the checkpoint."""
    for k in ("AF3_TORCH_PY", "AF3_TORCH_JAX_PY", "AF3_TORCH_JAX_REPO", "AF3_TORCH_PARAMS_DIR"):
        monkeypatch.delenv(k, raising=False)
    why = stack.gates()
    for k in ("AF3_TORCH_PY", "AF3_TORCH_JAX_PY", "AF3_TORCH_JAX_REPO"):
        assert any(w.startswith(f"{k} is not set — {stack.REQUIRED[k]}") for w in why), (k, why)
    monkeypatch.setenv("AF3_TORCH_PY", str(tmp_path / "no-torch-python")); monkeypatch.setenv("AF3_TORCH_JAX_PY", str(tmp_path / "no-jax-python"))
    monkeypatch.setenv("AF3_TORCH_JAX_REPO", str(tmp_path))
    why = stack.gates()
    assert any(w.startswith(f"AF3_TORCH_JAX_REPO={tmp_path} holds no run_alphafold.py") for w in why), why
    assert any(w.startswith(f"AF3_TORCH_PY={tmp_path / 'no-torch-python'} is not an executable interpreter") for w in why), why
    assert any(w.startswith(f"AF3_TORCH_JAX_PY={tmp_path / 'no-jax-python'} is not an executable interpreter") for w in why), why
    assert any("AF3_TORCH_PARAMS_DIR is not set" in w for w in why), why
    monkeypatch.setenv("AF3_TORCH_PARAMS_DIR", str(tmp_path))
    assert any("holds no parameters file (*.bin.zst | *.bin; the pinned OpenFold3-preview2 checkpoint is of3_ported_weights.bin.zst)" in w for w in stack.gates())
    assert stack.gates(need_params=False) and not any("PARAMS" in w for w in stack.gates(need_params=False))


def test_activation_on_a_box(box, capsys):
    rep = stack.activate("fast")
    assert rep["active"] and rep["reason"] is None and rep["lever_set"] == "fastest" and rep["dtk"] is True
    assert rep["torch_python"] == box["torch_py"] and rep["params_dir"] == box["params"] and rep["jax_repo"] == box["jax_repo"] and "image" not in rep
    ln = activation_line(rep)
    assert ln.startswith("[af3-torch-opt] ACTIVE mode=fast lever_set=fastest levers=bf16w+") and " dtk=1 " in ln
    err = capsys.readouterr().err.strip().splitlines()
    assert err[0] == ln and len(err) == 2 and err[1].startswith("[af3-torch-opt] weights sha256=") and err[1].endswith("UNPINNED — the kit's tests and timings cover the pinned checkpoint only")   # the stub checkpoint is not the pinned bytes: it runs, named
    assert f" weights={stack.checkpoint()} sha256={rep['weights']['sha256'][:12]} (unpinned) " in ln
    assert stack.status() is rep
    off = stack.check("off")
    assert off["levers"] == [] and off["levers_label"] == "none" and "levers=none dtk=0" in activation_line(off)


def test_not_active_line(monkeypatch):
    monkeypatch.setenv("AF3_TORCH_PY", "/nonexistent/python")
    rep = stack.check("fast")
    assert not rep["active"] and "NOT ACTIVE mode=fast n_gpu=1 reason=AF3_TORCH_PY=/nonexistent/python" in activation_line(rep)


def test_model_process_env_is_clean(box, monkeypatch):
    monkeypatch.setenv("AF3_TORCH_OPT", "fast"); monkeypatch.setenv("AF3_TORCH_OPT_HOME", HOME); monkeypatch.setenv("AF3_TORCH_OPT_KIT", stack.kit_home())
    env = stack.model_process_env()
    assert not [k for k in env if k.startswith("AF3_TORCH_OPT")] and stack.proof(env)["ok"]
    assert env["AF3_TORCH_PY"] == box["torch_py"] and env["AF3_TORCH_PARAMS_DIR"] == box["params"]      # deployment variables stay
    root = stack.cache_root()
    assert env["TRITON_CACHE_DIR"] == os.path.join(root, "triton") and env["TORCHINDUCTOR_CACHE_DIR"] == os.path.join(root, "inductor") and env["JAX_CACHE_DIR"] == os.path.join(root, "jax")
    assert env["PYTHONDONTWRITEBYTECODE"] == "1" and "JAX_PLATFORMS" not in env
    jenv = stack.model_process_env(jax=True)
    assert jenv["JAX_PLATFORMS"] == "cpu" and jenv["XLA_FLAGS"] == stack.pins()["image"]["env"]["XLA_FLAGS"]
    assert not stack.proof({"AF3_TORCH_OPT_LEVERS": "x"})["ok"]


def test_sys_path_recipe_is_the_kits(box):
    kit = stack.kit_home()
    assert stack.kit_sys_path() == [os.path.join(kit, "af3_torch"), os.path.join(kit, "kernels"), os.path.join(kit, "kernels", "third_party"), stack.dtk_home()]
    doc = open(os.path.join(kit, modes.API_RELPATH), encoding="utf-8").read()
    assert 'sys.path += ["<af3t>/af3_torch", "<af3t>/kernels", "<af3t>/kernels/third_party"]' in doc


def test_missing_kit_or_tree_is_not_active_not_an_exception(box, monkeypatch, capsys):
    """F1: a missing kit dir / wrong tree root degrades to NOT ACTIVE with the reason (rc 3 through the CLI), never an exception."""
    from af3_torch_opt import cli
    monkeypatch.setenv("AF3_TORCH_OPT_KIT", "/nonexistent/kit")
    rep = stack.check("fast")
    assert not rep["active"] and "kit dir /nonexistent/kit is missing" in rep["reason"] and "af3_torch_api.py is missing" in rep["reason"]
    assert cli.main(["check", "--mode", "fast"]) == 3
    monkeypatch.delenv("AF3_TORCH_OPT_KIT")
    monkeypatch.setenv("AF3_TORCH_OPT_HOME", str(box["tmp"] / "nowhere"))
    rep = stack.check("off")
    assert not rep["active"] and "PINS.json is missing" in rep["reason"]
    assert cli.main(["check", "--mode", "off"]) == 3
    capsys.readouterr()


def test_model_opt_env_names_the_tree(monkeypatch):
    monkeypatch.delenv("AF3_TORCH_OPT_HOME", raising=False); monkeypatch.setenv("MODEL_OPT", "/some/tree")
    assert stack.home() == "/some/tree"
    monkeypatch.setenv("AF3_TORCH_OPT_HOME", "/other/tree")
    assert stack.home() == "/other/tree"


def test_gpu_target_and_config_exports(box, monkeypatch):
    monkeypatch.setenv("AF3_TORCH_GPU", "H100")
    rep = stack.check("fast")
    assert rep["gpu_target"] == "H100" and "gpu_target=H100" in activation_line(rep)
    assert list(stack.REQUIRED) == ["AF3_TORCH_PY", "AF3_TORCH_JAX_PY", "AF3_TORCH_JAX_REPO"] and set(stack.REQUIRED) <= set(stack.DEPLOYMENT)
    lines = stack.config_exports().splitlines()                         # one presence check per required variable, in REQUIRED's order; no value, no default
    assert len(lines) == len(stack.REQUIRED) and all(ln.startswith(f'[ -n "${{{var}:-}}" ] || {{ echo "af3_torch_opt: {var} is not set — ') and "return 3" in ln for ln, var in zip(lines, stack.REQUIRED)), lines
    assert set(stack.pins()["image"]) == {"tested_on", "env", "env_note"}   # the pins carry no interpreter / checkout default: one documentation line + the JAX env
    sourced = lambda env: subprocess.run(["bash", "-c", 'eval "$1"; echo sourced-ok', "_", stack.config_exports()], env=env, capture_output=True, text=True)
    full = {k: os.environ[k] for k in ("PATH", "AF3_TORCH_PY", "AF3_TORCH_JAX_PY", "AF3_TORCH_JAX_REPO")}
    r = sourced(full); assert r.returncode == 0 and r.stdout.strip() == "sourced-ok" and r.stderr == "", (r.returncode, r.stdout, r.stderr)
    for var in stack.REQUIRED:                                            # each unset variable alone: named on stderr, rc 3, nothing after the checks runs
        r = sourced({k: v for k, v in full.items() if k != var})
        assert r.returncode == 3 and r.stderr.startswith(f"af3_torch_opt: {var} is not set — {stack.REQUIRED[var]} (README Variables)") and "sourced-ok" not in r.stdout, (var, r.returncode, r.stdout, r.stderr)
    assert stack.proof(stack.model_process_env())["deployment"] == ["AF3_TORCH_CACHE_ROOT", "AF3_TORCH_GPU", "AF3_TORCH_JAX_PY", "AF3_TORCH_JAX_REPO", "AF3_TORCH_PARAMS_DIR", "AF3_TORCH_PY", "AF3_TORCH_STOCK_PY"]
    assert stack.jax_repo() == box["jax_repo"] == os.environ["AF3_TORCH_JAX_REPO"]
    monkeypatch.delenv("AF3_TORCH_JAX_REPO"); rep = stack.check("fast")
    assert stack.jax_repo() is None and rep["active"] is False and rep["reason"].startswith("AF3_TORCH_JAX_REPO is not set — "), rep["reason"]


def test_other_parameter_files_beside_the_pinned_name(box, monkeypatch):
    """Other *.bin / *.bin.zst beside the PINNED name: the pinned name is the checkpoint (no refusal, no guess); the CLI is handed that file."""
    import pathlib
    stray = pathlib.Path(box["params"]) / "af3.bin.zst"; stray.write_bytes(b"")
    stray_bin = pathlib.Path(box["params"]) / "af3.bin"; stray_bin.write_bytes(b"")
    assert not [w for w in stack.gates() if "PARAMS" in w], stack.gates()
    assert stack.checkpoint_path() == os.path.join(box["params"], stack.checkpoint())
    assert stack.checkpoint() == "of3_ported_weights.bin.zst" == stack.pins()["variants"]["p2"]["converted"]["file"]   # one spelling: the pins


# ---- the weights this process runs: warn-and-run (any checkpoint file runs; the pinned digest is `pinned`, any other digest prints ONE
#      UNPINNED line and runs; several files run the pinned name else the first by name; a missing checkpoint stays a refusal by name) --------------------------------------------------

import hashlib as _hashlib
import importlib.util as _ilu
import re as _re
import subprocess as _subprocess
import sys as _sys

_UNPINNED_RX = _re.compile(r"^\[af3-torch-opt\] weights sha256=([0-9a-f]{12}) UNPINNED — the kit's tests and timings cover the pinned checkpoint only$", _re.M)


def _pin_the_stub(monkeypatch, path):
    """Make the box's stub checkpoint THE pinned bytes: PINS variants.p2.converted.{sha256,bytes} := the file's (a copy of the pins dict, patched)."""
    import copy
    P = copy.deepcopy(stack.pins()); c = P["variants"]["p2"]["converted"]
    c["sha256"] = _hashlib.sha256(open(path, "rb").read()).hexdigest(); c["bytes"] = os.path.getsize(path)
    monkeypatch.setattr(stack, "pins", lambda: P); stack._WEIGHTS.clear()
    return c["sha256"]


def test_unknown_weights_run_unpinned_with_one_line(box, capsys, tmp_path):
    """A checkpoint whose digest is not the pin's — under the pinned name or ANY *.bin.zst | *.bin name alone in the dir — is accepted: ACTIVE
    carries `weights=<file> sha256=<12> (unpinned)`, ONE UNPINNED line follows, pred runs to rc 0 and the run record says so."""
    stack._WEIGHTS.clear()
    rep = stack.activate("fast")
    err = capsys.readouterr().err
    assert rep["active"] and rep["weights"]["pinned"] is False and rep["weights"]["file"] == os.path.join(box["params"], stack.checkpoint())
    sha12 = rep["weights"]["sha256"][:12]
    assert f" weights={stack.checkpoint()} sha256={sha12} (unpinned) " in err and _UNPINNED_RX.findall(err) == [sha12], err
    # any name: the dir's ONE parameters file is the checkpoint
    os.rename(os.path.join(box["params"], stack.checkpoint()), os.path.join(box["params"], "my_finetune.bin"))
    open(os.path.join(box["params"], "my_finetune.bin"), "wb").write(b"not the pinned bytes")
    stack._WEIGHTS.clear()
    from .test_cli_manifest import _inputs
    out = tmp_path / "out"
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(out)]) == 0
    err = capsys.readouterr().err
    sha = _hashlib.sha256(b"not the pinned bytes").hexdigest()
    assert f" weights=my_finetune.bin sha256={sha[:12]} (unpinned) " in err and _UNPINNED_RX.findall(err) == [sha[:12]], err
    man = cli.last_run()
    assert man["activation"]["weights"] == {"file": os.path.join(box["params"], "my_finetune.bin"), "name": "my_finetune.bin", "sha256": sha, "bytes": 20, "pinned": False, "variant": None, "cached_utc": None, "memo": "ok"}
    calls = [json.loads(l) for l in open(box["log"])]
    fwd = next(c for c in calls if any(str(x).endswith("/forward.py") for x in c))
    assert fwd[fwd.index("--params") + 1] == os.path.join(box["params"], "my_finetune.bin") and fwd[fwd.index("--weights-sha256") + 1] == sha and fwd[fwd.index("--weights-pinned") + 1] == "0"


def test_pinned_weights_print_no_warning(box, capsys, monkeypatch):
    """The pinned digest: `weights=OpenFold3-preview2 sha256=<12> (pinned)` on the ACTIVE line, no UNPINNED line; a second parameters file
    beside the pinned name does not make the directory ambiguous (the pinned name wins)."""
    path = os.path.join(box["params"], stack.checkpoint()); open(path, "wb").write(b"the pinned bytes (stub)")
    sha = _pin_the_stub(monkeypatch, path)
    open(os.path.join(box["params"], "someone_elses.bin.zst"), "wb").write(b"x")
    rep = stack.activate("fast")
    err = capsys.readouterr().err
    assert rep["active"] and rep["weights"] == {"file": path, "name": "OpenFold3-preview2", "sha256": sha, "bytes": 23, "pinned": True, "variant": "p2", "cached_utc": None, "memo": "ok"}
    assert f" weights=OpenFold3-preview2 sha256={sha[:12]} (pinned) " in err and not _UNPINNED_RX.findall(err), err
    assert stack.weights_record(path) is stack.weights_record(path)                   # digested once per (path, size, mtime) within the process


def test_missing_weights_are_a_refusal_by_name_and_several_files_run_the_first(box, capsys, tmp_path):
    """No parameters file: NOT ACTIVE by name (rc 3). Several files none of which is the pinned name: the first in name order runs (as
    xfold's own loader picks) and the weights line names it UNPINNED — never a refusal stock would not make."""
    os.remove(os.path.join(box["params"], stack.checkpoint()))
    rep = stack.activate("fast")
    assert not rep["active"] and "holds no parameters file" in rep["reason"] and rep["weights"] is None
    from .test_cli_manifest import _inputs
    assert cli.main(["pred", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o")]) == 3
    assert "NOT ACTIVE" in capsys.readouterr().err
    for f in ("a.bin", "b.bin.zst"):
        open(os.path.join(box["params"], f), "wb").write(f.encode())
    rep = stack.activate("fast")
    err = capsys.readouterr().err
    assert rep["active"] and rep["weights"]["file"] == os.path.join(box["params"], "a.bin") and rep["weights"]["pinned"] is False, rep.get("reason")
    assert _UNPINNED_RX.findall(err), err


def _check_pins_module():
    spec = _ilu.spec_from_file_location("af3t_check_pins", os.path.join(stack.home(), "stock", "check_pins.py"))
    m = _ilu.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_the_checkpoint_rule_has_one_producer(tmp_path):
    """stock/check_pins.py (standard library only) IS the resolver the package uses (stack.pins_tool): pinned name > the first parameters file
    in name order > refusal (none: missing)."""
    cp = stack.pins_tool()
    assert cp.__file__ == os.path.join(stack.home(), "stock", "check_pins.py") and cp.resolve_checkpoint is stack.pins_tool().resolve_checkpoint
    ck = stack.checkpoint()
    layouts = {"empty": ([], None, "missing"), "pinned": ([ck], ck, None), "one_other": (["mine.bin"], "mine.bin", None), "pinned_plus": ([ck, "mine.bin"], ck, None),
               "two_others": (["a.bin", "b.bin.zst"], "a.bin", None), "not_weights": (["notes.txt"], None, "missing")}
    for name, (files, want, verdict) in layouts.items():
        d = tmp_path / name; d.mkdir()
        for f in files:
            (d / f).write_bytes(b"w")
        assert cp.resolve_checkpoint(str(d), ck) == ((str(d / want) if want else None), verdict), name
        path, why = stack.resolve_checkpoint(str(d))
        assert path == (str(d / want) if want else None) and (why is None) == (want is not None), (name, path, why)
        assert why is None or "holds no parameters file" in why, (name, why)
    assert cp.resolve_checkpoint(str(tmp_path / "absent"), ck) == (None, "missing") and "is not a directory" in stack.resolve_checkpoint(str(tmp_path / "absent"))[1]


def test_the_pins_tool_digest_verdicts(box, monkeypatch):
    """check_pins.py --digest: an unknown digest is the UNPINNED verdict and PASSES (exit 0, the one UNPINNED line); the pinned digest is
    pinned; a missing checkpoint is NOT MET (exit 3). The stub interpreters answer its version queries with the pinned versions."""
    tool = os.path.join(stack.home(), "stock", "check_pins.py")
    stubs = {}
    for which in ("torch_python", "jax_python"):                              # two stub interpreters answering the version query with their pinned set
        stub = box["tmp"] / f"pins_stub_{which}"; stubs[which] = str(stub)
        stub.write_text("#!/usr/bin/env python3\nimport json, os, sys\nP = json.load(open(os.path.join(%r, 'stock', 'PINS.json')))\n"
                        "print(json.dumps({'python': '3.12.0', **P['check_packages'][%r]}))\n" % (stack.home(), which), encoding="utf-8")
        stub.chmod(stub.stat().st_mode | 0o111)
    def run(*extra):
        r = _subprocess.run([_sys.executable, "-I", tool, "--digest", "--torch-py", stubs["torch_python"], "--jax-py", stubs["jax_python"], *extra], capture_output=True, text=True, env=dict(os.environ))
        return r.returncode, r.stdout
    rc, out = run()
    sha = _hashlib.sha256(open(os.path.join(box["params"], stack.checkpoint()), "rb").read()).hexdigest()
    assert rc == 0 and f"digest=UNPINNED:{sha[:12]}" in out and f"weights sha256={sha[:12]} UNPINNED — the kit's tests and timings cover the pinned checkpoint only" in out, out
    rc, out = run("--json")
    d = json.loads(out)["digest"]
    assert rc == 0 and d["verdict"] == "unpinned" and d["variants"]["p2"]["pinned"] is False and d["ok"] is True
    rc, out = run("--quiet")                                                   # --quiet silences the PINS chatter, never the weights line
    assert rc == 0 and out.strip().splitlines() == [f"[af3-torch-opt] weights sha256={sha[:12]} UNPINNED — the kit's tests and timings cover the pinned checkpoint only"], out
    os.remove(os.path.join(box["params"], stack.checkpoint()))
    rc, out = run()
    assert rc != 0 and "digest=MISSING" in out and "PINS NOT MET" in out, out


def test_quiet_never_silences_the_unpinned_line(box, capsys, tmp_path):
    """`pred --quiet` with an unknown digest: rc 0, no ACTIVE line, and EXACTLY ONE UNPINNED line (it prints under --quiet,
    once per process); `stack.check` (the quiet activation) prints it too, once."""
    stack._WEIGHTS.clear(); stack._WARNED.clear()
    from .test_cli_manifest import _inputs
    assert cli.main(["pred", "--quiet", "--mode", "fast", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err
    assert "] ACTIVE " not in err and len(_UNPINNED_RX.findall(err)) == 1, err
    stack._WARNED.clear()
    stack.check("fast"); stack.check("fast")
    assert len(_UNPINNED_RX.findall(capsys.readouterr().err)) == 1


def test_stock_route_names_the_weights_and_hands_the_resolved_file(box, tmp_path, capsys):
    """run.sh stock: the STOCK-CLI line carries `weights=…`, the one UNPINNED line prints once, and the CLI's --model_dir is the RESOLVED file."""
    archive_or_skip()                                                                     # the stock route reads xfold's CLI out of the pinned archive
    os.rename(os.path.join(box["params"], stack.checkpoint()), os.path.join(box["params"], "custom.bin.zst")); stack._WEIGHTS.clear()
    from .test_cli_manifest import _inputs
    assert cli.main(["stock", "--json_path", _inputs(box["tmp"], ("a",))[0], "--output_dir", str(tmp_path / "o")]) == 0
    err = capsys.readouterr().err
    assert " weights=custom.bin.zst sha256=" in err and "(unpinned) " in err and len(_UNPINNED_RX.findall(err)) == 1, err
    calls = [json.loads(l) for l in open(box["log"])]
    sc_ = [c for c in calls if c[0] == box["stock_py"]]
    assert sc_ and sc_[0][sc_[0].index("--model_dir") + 1] == os.path.join(box["params"], "custom.bin.zst")


def test_pred_reads_the_digest_memo_and_check_hashes_afresh(box, monkeypatch):
    """The checkpoint's sha256 goes through the kit-local memo (`<cache root>/weights_digests.json`): the first activation hashes and writes the
    entry; a new process (an empty in-process table) with the same (realpath, size, mtime, inode) READS it — no hash, the word names the cached
    time; `check` (refresh) hashes afresh and rewrites the entry even when the stat key is unchanged; the digest still decides (a memo entry
    whose digest is not a pin is unpinned whatever the stat fields say)."""
    from af3_torch_opt import digest_memo
    path = os.path.join(box["params"], stack.checkpoint()); open(path, "wb").write(b"some checkpoint bytes")
    calls = []
    real = stack.sha256
    monkeypatch.setattr(stack, "sha256", lambda p: (calls.append(p), real(p))[1])
    monkeypatch.setattr(stack, "_WEIGHTS", {})
    memo = os.path.join(stack.weights_memo_dir(), digest_memo.MEMO_NAME)
    assert stack.weights_memo_dir() == stack.cache_root() and not os.path.exists(memo)
    r1 = stack.weights_record(path); r2 = stack.weights_record(path)
    assert r1 is r2 and len(calls) == 1 and r1["cached_utc"] is None and os.path.isfile(memo)          # hashed once, the entry written after the full hash
    assert stack.weights_word(r1) == f"of3_ported_weights.bin.zst sha256={r1['sha256'][:12]} (unpinned)" and stack.weights_digest_token(r1) == "fresh"
    monkeypatch.setattr(stack, "_WEIGHTS", {})                                                          # a new process: pred reads the memo entry
    r3 = stack.weights_record(path)
    assert len(calls) == 1 and r3["sha256"] == r1["sha256"] and r3["cached_utc"] and r3["pinned"] is False
    assert stack.weights_word(r3) == stack.weights_word(r1) and stack.weights_digest_token(r3) == f"cached@{r3['cached_utc']}" and " " not in stack.weights_digest_token(r3)   # the ACTIVE token keeps its grammar
    assert stack.weights_line(r3) == f"[af3-torch-opt] weights sha256={r1['sha256'][:12]} UNPINNED — {stack.UNPINNED_WORDS} (cached digest {r3['cached_utc']})"
    r4 = stack.weights_record(path, refresh=True)                                                       # check: hashed afresh, the entry rewritten, no cached word
    assert len(calls) == 2 and r4["cached_utc"] is None and r4["sha256"] == r1["sha256"]
    table = json.load(open(memo)); assert len(table) == 1 and list(table.values())[0]["sha256"] == r1["sha256"]


def test_check_passes_refresh_and_pred_reads_the_memo(box, monkeypatch):
    """`check` digests the checkpoint with refresh=True (afresh); `pred`'s activation and the stock route with refresh=False (the memo)."""
    from af3_torch_opt import cli, digest_memo
    path = os.path.join(box["params"], stack.checkpoint()); open(path, "wb").write(b"weights")
    seen = []
    def fake_digest(p, memo_dir, refresh=False, hasher=None):
        seen.append((os.path.basename(p), memo_dir, refresh)); return "ab" * 32, None
    monkeypatch.setattr(digest_memo, "digest", fake_digest)
    monkeypatch.setattr(stack, "_WEIGHTS", {})
    stack.check("fast")                                                            # check: refresh=True
    monkeypatch.setattr(stack, "_WEIGHTS", {}); stack.activate("fast", quiet=True)     # pred's activation in a fresh process: refresh=False
    stack.weights_record(path)                                                     # the stock route's call in the same process: the in-process table, no digest call
    assert seen == [("of3_ported_weights.bin.zst", stack.cache_root(), True), ("of3_ported_weights.bin.zst", stack.cache_root(), False)], seen
    monkeypatch.setattr(stack, "_WEIGHTS", {}); seen.clear()
    assert cli.main(["check", "--mode", "fast"]) == 0 and seen and all(r for _, _, r in seen), seen


ACTIVE_TAIL_RX = re.compile(r" params=\S+ weights=\S+ sha256=[0-9a-f]{12} \((?:pinned|unpinned)\) torch_py=\S+ jax_py=\S+ cache=\S+ gpu_target=\S+ package=\S+ padding=\S+ compile=(?:on|off:user|off:mode)( overridden=1)?$")


def test_active_line_grammar_holds_with_a_warm_digest_memo(box, capsys, monkeypatch):
    """The ACTIVE line is the same whitespace-free key=value tokens whether the checkpoint was hashed in this process or read from the memo:
    `weights=<name> sha256=<12> (pinned|unpinned)` keeps its shape and no token is added; the memo state is the report's `weights_digest`
    (`fresh` | `cached@<utc>`) and the words `(cached digest <utc>)` appear on the weights line only."""
    path = os.path.join(box["params"], stack.checkpoint()); open(path, "wb").write(b"the pinned bytes (stub)")
    sha = _pin_the_stub(monkeypatch, path)
    monkeypatch.setattr(stack, "_WEIGHTS", {})
    rep1 = stack.activate("fast"); err1 = capsys.readouterr().err
    line1 = next(l for l in err1.splitlines() if " ACTIVE " in l)
    assert ACTIVE_TAIL_RX.search(line1) and rep1["weights_digest"] == "fresh" and "cached" not in err1, line1
    assert f" weights=OpenFold3-preview2 sha256={sha[:12]} (pinned) torch_py=" in line1
    monkeypatch.setattr(stack, "_WEIGHTS", {}); monkeypatch.setattr(stack, "_WARNED", set())          # a new process on the warm memo
    rep2 = stack.activate("fast"); err2 = capsys.readouterr().err
    line2 = next(l for l in err2.splitlines() if " ACTIVE " in l)
    assert ACTIVE_TAIL_RX.search(line2) and "cached" not in line2, line2                                  # the token grammar, unchanged on a hit
    assert line2 == line1                                                                                  # byte for byte the cold line
    utc = rep2["weights"]["cached_utc"]; assert utc and rep2["weights_digest"] == f"cached@{utc}" and " " not in rep2["weights_digest"]
    assert f"[af3-torch-opt] weights sha256={sha[:12]} pinned (cached digest {utc})" in err2          # the human words: the weights line only
    rep3 = stack.check("fast")                                                                             # check: afresh again
    assert rep3["weights_digest"] == "fresh" and rep3["weights"]["cached_utc"] is None


def _unwritable_dirs(tmp):
    """Two cache roots nothing can write a memo into: a 0o555 directory (for a non-root user) and a path beneath a regular file (for anyone)."""
    ro = os.path.join(tmp, "ro_cache"); os.makedirs(ro); os.chmod(ro, 0o555)
    blocker = os.path.join(tmp, "a_file"); open(blocker, "w").write("x")
    return [ro, os.path.join(blocker, "cache")]


def test_an_unwritable_memo_dir_is_named_and_hashed_afresh_never_a_refusal(box, capsys, monkeypatch):
    """A read-only cache root (the memo cannot be written): `check` and `pred`'s activation name it on ONE `weights digest memo UNWRITABLE …`
    line, hash the checkpoint afresh in the process (once — the digest computed inside the memo call is kept), decide which checkpoint it is by that digest,
    and proceed (rc 0); the report says memo=unwritable:<errno>."""
    from af3_torch_opt import cli
    path = os.path.join(box["params"], stack.checkpoint()); open(path, "wb").write(b"the pinned bytes (stub)")
    sha = _pin_the_stub(monkeypatch, path)
    for ro in _unwritable_dirs(str(box["tmp"])):
        if os.geteuid() == 0 and not ro.endswith(os.sep + "cache"):
            continue                                                                    # root writes through 0o555; the beneath-a-file root holds for root too
        monkeypatch.setenv("AF3_TORCH_CACHE_ROOT", ro)
        monkeypatch.setattr(stack, "_WEIGHTS", {}); monkeypatch.setattr(stack, "_MEMO_WARNED", set())
        calls = []; real = stack.sha256
        monkeypatch.setattr(stack, "sha256", lambda p, _r=real: (calls.append(p) if os.path.abspath(p) == os.path.abspath(path) else None, _r(p))[1])   # count the CHECKPOINT hashes only
        rep = stack.check("fast")                                                       # refresh=True: must write — cannot — named, afresh
        err = capsys.readouterr().err
        assert rep["active"] and rep["weights"]["sha256"] == sha and rep["weights"]["pinned"] is True and rep["weights"]["cached_utc"] is None, rep["weights"]
        assert rep["weights"]["memo"].startswith("unwritable:") and rep["weights_digest"] == "fresh" and len(calls) == 1, (rep["weights"], calls)
        assert err.count("weights digest memo UNWRITABLE dir=" + ro) == 1 and "hashed afresh, nothing memoised" in err, err
        monkeypatch.setattr(stack, "_WEIGHTS", {})
        rep2 = stack.activate("fast"); err2 = capsys.readouterr().err                  # pred's activation: a miss it cannot write — the same, the line once per process
        assert rep2["active"] and rep2["weights"]["memo"].startswith("unwritable:") and "UNWRITABLE" not in err2 and len(calls) == 2
        monkeypatch.setattr(stack, "_WEIGHTS", {}); monkeypatch.setattr(stack, "_MEMO_WARNED", set())
        assert cli.main(["check", "--mode", "fast"]) == 0 and "UNWRITABLE" in capsys.readouterr().err
        monkeypatch.setattr(stack, "sha256", real)
