import hashlib
"""Every carried kit file is present, the stock pin check accepts the pinned tree and
refuses another, the config and run.sh carry no lever switch and refuse a mode disagreement."""
import importlib.util
import os
import re
import shutil
import subprocess
import sys

import pytest

from .. import modes, stack
from . import _stubs


def test_every_carried_kit_file_is_present():
    problems, shas = stack.check_kit_files()
    assert problems == [] and len(shas) == len(stack.read_sums())
    assert len(shas) >= 38, "the trunk / hoist add-ons plus the lever sources under opt/forward"
    for rel in ("forward/trunk_levers/boltz_trunk_levers.py", "forward/trunk_levers/boltz_flash_triattn_patch.py", "forward/trunk_levers/src/bz_worker_lev.py", "forward/trunk_levers/src/bz_worker_levf2.py",
                "forward/dit_hoist/src/boltz_dit_hoist.py", "forward/dit_hoist/src/boltz_graph_patch.py", "forward/dit_hoist/src/make_worker_variant.py"):
        assert rel in shas


def test_kit_files_checked_before_staging(tmp_path, monkeypatch):
    """A kit file `stage()` needs but does not find on disk refuses the stage and the gate by name, before anything is copied."""
    target = os.path.join(stack.opt_dir(), "forward/trunk_levers/boltz_trunk_levers.py")
    real_isfile = os.path.isfile
    monkeypatch.setattr(stack.os.path, "isfile", lambda p: False if p == target else real_isfile(p))
    with pytest.raises(RuntimeError) as e:
        stack.stage("exact", str(tmp_path / "w"))
    assert "missing" in str(e.value) and "boltz_trunk_levers.py" in str(e.value)
    plan = stack.gate("exact", need_gpu=False, need_cache=False, check_pins=False)
    assert plan["kit_ok"] is False and any("boltz_trunk_levers.py" in r for r in plan["reasons"])


def test_stage_derives_the_variant_with_the_kit_script(tmp_path):
    st = stack.stage("exact", str(tmp_path / "w"))
    assert os.path.isfile(st["worker"]) and "worker variant written" in st["variant_line"] and "BZ2DIT blocks" in st["variant_line"]
    txt = open(st["worker"]).read()
    assert "import boltz_trunk_levers as _LEV" in txt and "import boltz_dit_hoist as _DH" in txt and "import boltz_graph_patch as _BGP" in txt
    assert set(st["files"]) >= {"boltz_trunk_levers.py", "boltz_graph_patch.py", "boltz_dit_hoist.py", "make_worker_variant.py", "bz_worker_lev.py"}
    assert not {"bz_score.py", "m__tools__score_interface.py", "bz_b1_compare.py"} & set(st["files"]), "no scoring / comparison library is carried or staged: rows keep upstream's own confidences"
    assert "boltz_flash_triattn_patch.py" not in st["files"]
    st2 = stack.stage("fast", str(tmp_path / "w2"))
    assert "boltz_flash_triattn_patch.py" in st2["files"] and "import boltz_flash_triattn_patch as F2P" in open(st2["worker"]).read()
    assert stack.worker_name("exact") == "bz_worker_lev.py+bz2dit" and stack.worker_name("fast") == "bz_worker_levf2.py+bz2dit"


def _check_pins_module():
    p = os.path.join(stack.tree_dir(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("cp", p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_check_pins_accepts_the_pinned_tree_and_refuses_a_modified_one(tmp_path):
    site = _stubs.unpack_wheel(str(tmp_path / "site"))
    code = ("import sys, importlib.util; sys.path.insert(0, %r); spec = importlib.util.spec_from_file_location('cp', %r); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); "
            "bad, d = m.check(m.pins()); print(bad, d['pinned'], d['n_py_files'])")
    cp = os.path.join(stack.tree_dir(), "stock", "check_pins.py")
    r = subprocess.run([sys.executable, "-c", code % (site, cp)], capture_output=True, text=True)
    assert r.stdout.strip() == "[] True 107", r.stdout + r.stderr
    with open(os.path.join(site, "boltz", "main.py"), "a") as fh:
        fh.write("\n# modified\n")
    r = subprocess.run([sys.executable, "-c", code % (site, cp)], capture_output=True, text=True)
    assert "want 107 files" in r.stdout and "False" in r.stdout
    r = subprocess.run([sys.executable, cp, "--quiet"], capture_output=True, text=True, env={"PATH": os.environ["PATH"], "PYTHONPATH": site})
    assert r.returncode == 3 and "want 107 files" in r.stderr


def test_check_pins_refuses_another_distribution_version(tmp_path):
    """The version branch: the pinned tree under a boltz-2.2.0 distribution is 'boltz 2.2.0: want 2.2.1' (tree digest still the pin's)."""
    site = _stubs.unpack_wheel(str(tmp_path / "site"))
    info = os.path.join(site, "boltz-2.2.1.dist-info"); meta = open(os.path.join(info, "METADATA")).read()
    assert "Version: 2.2.1" in meta
    open(os.path.join(info, "METADATA"), "w").write(meta.replace("Version: 2.2.1", "Version: 2.2.0"))
    os.rename(info, os.path.join(site, "boltz-2.2.0.dist-info"))
    cp = os.path.join(stack.tree_dir(), "stock", "check_pins.py")
    r = subprocess.run([sys.executable, cp], capture_output=True, text=True, env={"PATH": os.environ["PATH"], "PYTHONPATH": site})
    assert r.returncode == 3 and "check_pins: boltz 2.2.0: want 2.2.1" in r.stderr and "want 107 files" not in r.stderr, r.stdout + r.stderr
    m = _check_pins_module()
    bad, d = m.check(m.pins(), environ={})
    assert isinstance(bad, list) and "pinned" in d


def _python_shim(tmp_path, site):
    """A `python` on PATH that runs this interpreter on a modified boltz tree: `-I` dropped, PYTHONPATH set — what run.sh's pin gate sees on a
    box whose installed boltz is not the pin."""
    d = tmp_path / "bin"; d.mkdir()
    shim = d / "python"
    shim.write_text("#!/bin/bash\nARGS=(); for a in \"$@\"; do [ \"$a\" == -I ] || ARGS+=(\"$a\"); done\n"
                    f"PYTHONPATH={site} exec {sys.executable} -s \"${{ARGS[@]}}\"\n")
    shim.chmod(0o755)
    return str(d)


def test_run_sh_pin_gate_refuses_a_box_whose_boltz_is_not_the_pin(tmp_path):
    """run.sh:40 refuses every route with rc 3 before the package runs when stock/check_pins.py fails — exercised here on a modified tree
    even when this box's boltz is at the pin."""
    site = _stubs.unpack_wheel(str(tmp_path / "site"))
    with open(os.path.join(site, "boltz", "main.py"), "a") as fh:
        fh.write("\n# modified\n")
    env = {k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}
    env["PATH"] = _python_shim(tmp_path, site) + os.pathsep + env["PATH"]
    run = os.path.join(stack.tree_dir(), "run.sh")
    for args in (["check", "--mode", "exact"], ["pred", "--mode", "off", "--input", "x.yaml", "--out_dir", str(tmp_path / "o")], ["warm", "--mode", "fast", "--out", str(tmp_path / "v")]):
        r = subprocess.run(["bash", run] + args, capture_output=True, text=True, env=env)
        assert r.returncode == 3 and "run.sh: boltz is not installed at the pin" in r.stderr and "check_pins: boltz tree" in r.stderr, (args, r.stdout, r.stderr)
        assert "[boltz2-opt]" not in r.stdout, "nothing of the package ran"


def test_run_sh_refuses_mode_disagreement_and_off_without_pred(tmp_path):
    run = os.path.join(stack.tree_dir(), "run.sh")
    env = dict(os.environ, BOLTZ2_OPT="fast")
    r = subprocess.run(["bash", run, "check", "--mode", "exact"], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "disagrees with BOLTZ2_OPT=fast" in r.stderr
    r = subprocess.run(["bash", run, "nonsense"], capture_output=True, text=True)
    assert r.returncode == 2
    env = {k: v for k, v in os.environ.items() if k != "BOLTZ2_OPT"}
    pinned = subprocess.run([sys.executable, "-I", os.path.join(stack.tree_dir(), "stock", "check_pins.py"), "--quiet"], capture_output=True).returncode == 0
    if not pinned:                                                    # boltz not at the pin -> every route refuses with rc 3 before the package runs
        r = subprocess.run(["bash", run, "check", "--mode", "off"], capture_output=True, text=True, env=env)
        assert r.returncode == 3 and "not installed at the pin" in r.stderr
        pytest.skip("boltz 2.2.1 is not installed at the pin on this box (pip install --no-deps stock/boltz-2.2.1-py3-none-any.whl): the pinned-route checks need it")
    r = subprocess.run(["bash", run, "warm", "--mode", "off", "--out", str(tmp_path)], capture_output=True, text=True, env=env)
    assert r.returncode == 2 and "--mode off has one command, pred" in r.stderr, r.stderr
    r = subprocess.run(["bash", run, "check", "--mode", "exact"], capture_output=True, text=True, env=env)
    assert r.returncode in (0, 3) and "[boltz2-opt] DRY-RUN mode=exact route=worker" in r.stdout, r.stdout + r.stderr


def test_config_fills_only_unset_deployment_parameters(tmp_path):
    """configs/h100.env fills the non-path settings, carries no path default (BOLTZ_CACHE is the caller's; TRITON_CACHE_DIR passes through or
    is keyed under MODEL_OPT_JIT_ROOT when that is set) and derives MODEL_OPT_STACK_KEY."""
    cfg = os.path.join(stack.tree_dir(), "configs", "h100.env")
    base = {k: v for k, v in os.environ.items() if k not in ("BOLTZ_CACHE", "TRITON_CACHE_DIR", "MODEL_OPT_JIT_ROOT", "MODEL_OPT_STACK_KEY", "MODEL_OPT_TARGET_GPU")}
    echo = f"source {cfg} && echo \"$MODEL_OPT|$MODEL_OPT_TARGET_GPU|${{BOLTZ_CACHE:-}}|$MODEL_OPT_STACK_KEY|${{TRITON_CACHE_DIR:-}}\""
    r = subprocess.run(["bash", "-c", echo], capture_output=True, text=True, env=base)                       # nothing pre-set: no path appears
    parts = r.stdout.strip().split("|")
    assert r.returncode == 0 and parts[0] == stack.tree_dir() and parts[1] == "H100" and parts[2] == "" and parts[3] and parts[4] == "", r.stdout + r.stderr
    key = parts[3]
    r = subprocess.run(["bash", "-c", "export BOLTZ_CACHE=/mine MODEL_OPT_JIT_ROOT=/jc; " + echo], capture_output=True, text=True, env=base)   # the root keys the Triton cache by the stack
    parts = r.stdout.strip().split("|")
    assert r.returncode == 0 and parts[2] == "/mine" and parts[4] == f"/jc/{key}/triton", r.stdout + r.stderr
    r = subprocess.run(["bash", "-c", "export TRITON_CACHE_DIR=/pre/set MODEL_OPT_JIT_ROOT=/jc; " + echo], capture_output=True, text=True, env=base)   # a pre-set Triton cache passes through
    assert r.returncode == 0 and r.stdout.strip().split("|")[4] == "/pre/set", r.stdout + r.stderr


def test_unknown_checkpoint_is_named_and_proceeds(tmp_path, monkeypatch, capsys):
    """Weights: a checkpoint whose sha256 is not the pin's is a named WARNING on the WEIGHTS line and the gate proceeds (no reason added,
    so every route runs, rc unaffected); the pinned checkpoint reads `pinned … the pinned checkpoint`; the status rides the plan."""
    cache = tmp_path / "cache"; cache.mkdir()
    for f in stack.CACHE_FILES:
        (cache / f).mkdir() if f == "mols" else (cache / f).write_bytes(b"x")
    monkeypatch.setenv("BOLTZ_CACHE", str(cache)); monkeypatch.delenv("TRITON_CACHE_DIR", raising=False); monkeypatch.delenv(stack.WEIGHTS_DIGEST_DIR_ENV, raising=False)   # the digest memo lands under the cache dir here
    w = stack.weights_status()
    assert w["status"] == "unknown" and w["cached_utc"] is None and w["sha256"] == hashlib.sha256(b"x").hexdigest() and w["pinned_sha256"] == stack.load_pins()["weights"]["files"][stack.WEIGHTS_FILE]["sha256"]
    line = stack.weights_line(w)
    assert line.startswith("[boltz2-opt] WEIGHTS unknown sha256=") and "not the pinned checkpoint" in line and "WARNING: proceeding" in line
    plan = stack.gate("exact", need_gpu=False, need_cache=True, check_pins=False)
    assert plan["weights"]["status"] == "unknown" and not any("weights" in r.lower() or "cache" in r.lower() for r in plan["reasons"]), plan["reasons"]
    out = capsys.readouterr().out
    assert "[boltz2-opt] WEIGHTS unknown (cached digest " in out and "WARNING: proceeding" in out, "the second sight of the same file is a memo hit, named on the line; status still by digest"
    assert stack.weights_status(pinned=hashlib.sha256(b"x").hexdigest())["status"] == "pinned" and "the pinned checkpoint" in stack.weights_line(stack.weights_status(pinned=hashlib.sha256(b"x").hexdigest()))
    (cache / stack.WEIGHTS_FILE).unlink()
    assert stack.weights_status()["status"] == "absent" and stack.cache_check(), "an absent checkpoint is the frozen-weights refusal (no download), not a digest question"


def test_every_switch_a_mode_row_sets_is_stripped_from_the_stock_process():
    """stock/PINS.json stock_environment.must_be_absent_prefixes covers every variable any mode row exports (modes.env_row) and the pair-track
    adapters' switches, their _IMPL development switches included — the stock route's env proof and stripped_env() start from one list."""
    from .. import pairblock, transition, trimul
    prefixes = tuple(stack.load_pins()["stock_environment"]["must_be_absent_prefixes"])
    names = set()
    for m in modes.MODE_NAMES:
        names |= set(modes.env_row(m))
    names |= {pairblock.SWITCH, transition.SWITCH, trimul.SWITCH, pairblock.IMPL_ENV, "BOLTZ_TRANSITION_IMPL", "BOLTZ_PAIRBLOCK_MIN_TOKENS"}   # BOLTZ_TRANSITION_IMPL: the transition adapter's former A/B switch (gone at 0.3.30) stays stripped from stock processes
    names.discard("PYTORCH_CUDA_ALLOC_CONF")                                  # the allocator lever's variable is torch's own, set by the big row on purpose
    uncovered = sorted(n for n in names if not n.startswith(prefixes))
    assert not uncovered, uncovered
    assert stack.stripped_env({"BOLTZ_PAIRBLOCK_IMPL": "lnl", "BOLTZ_TRANSITION_IMPL": "torch", "BOLTZ_PAIRBLOCK": "k2b", "BOLTZ_CACHE": "/c"}) == {"BOLTZ_CACHE": "/c"}





def test_check_hashes_afresh_and_the_run_verbs_take_a_memo_hit(tmp_path, monkeypatch, capsys):
    """The checkpoint digest goes through digest_memo.digest(path, memo_dir, refresh): `check` passes refresh=True (hash afresh, rewrite the
    entry), the run verbs' gate passes refresh=False (a memo hit serves it and the WEIGHTS line says `(cached digest <utc>)`); the memo lives
    under the parent of TRITON_CACHE_DIR when set, else under BOLTZ_CACHE; status stays by digest."""
    from .. import cli, digest_memo
    cache = tmp_path / "cache"; cache.mkdir()
    for f in stack.CACHE_FILES:
        (cache / f).mkdir() if f == "mols" else (cache / f).write_bytes(b"x")
    monkeypatch.setenv("BOLTZ_CACHE", str(cache)); monkeypatch.delenv("TRITON_CACHE_DIR", raising=False); monkeypatch.delenv(stack.WEIGHTS_DIGEST_DIR_ENV, raising=False)
    assert stack.weights_memo_dir() == str(cache)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "jit" / "key" / "triton"))
    assert stack.weights_memo_dir() == str(tmp_path / "jit" / "key")
    monkeypatch.setenv(stack.WEIGHTS_DIGEST_DIR_ENV, str(tmp_path / "elsewhere"))          # the deployment override is read first
    assert stack.WEIGHTS_DIGEST_DIR_ENV == "MODEL_OPT_WEIGHTS_DIGEST_DIR" and stack.weights_memo_dir() == str(tmp_path / "elsewhere")
    w0 = stack.weights_status(); assert w0["memo"] == os.path.join(str(tmp_path / "elsewhere"), digest_memo.MEMO_NAME) and os.path.isfile(w0["memo"])
    monkeypatch.delenv(stack.WEIGHTS_DIGEST_DIR_ENV)
    calls = []
    real = digest_memo.digest
    def spy(path, memo_dir, refresh=False, hasher=digest_memo.sha256_file):
        calls.append((os.path.basename(path), memo_dir, refresh)); return real(path, memo_dir, refresh=refresh, hasher=hasher)
    monkeypatch.setattr(digest_memo, "digest", spy)
    w1 = stack.weights_status()                                            # first sight: hashed, entry written under the memo dir
    assert w1["cached_utc"] is None and w1["sha256"] == hashlib.sha256(b"x").hexdigest() and os.path.isfile(os.path.join(str(tmp_path / "jit" / "key"), digest_memo.MEMO_NAME))
    w2 = stack.weights_status()                                            # second sight: a memo hit, named on the line
    assert w2["cached_utc"] is not None and w2["sha256"] == w1["sha256"] and "(cached digest " in stack.weights_line(w2)
    stack.gate("exact", need_gpu=False, need_cache=True, check_pins=False)                          # the run verbs' gate: refresh=False
    stack.gate("exact", need_gpu=False, need_cache=True, check_pins=False, refresh_weights=True)    # check's gate: refresh=True
    assert [c[2] for c in calls] == [False, False, False, True] and all(c[1] == str(tmp_path / "jit" / "key") for c in calls)
    seen = {}
    monkeypatch.setattr(stack, "gate", lambda mode, **kw: seen.setdefault("kw", kw) and {"gpu": None, "n_gpu": None, "reasons": ["test box"], "cache": str(cache)})
    cli.main(["check", "--mode", "exact"])
    assert seen["kw"].get("refresh_weights") is True, seen


def test_an_unwritable_memo_dir_is_named_and_the_digest_computed_afresh_stands(tmp_path, monkeypatch, capsys):
    """A read-only memo directory (e.g. a read-only cache mount) never refuses and never dies: the checkpoint is
    hashed once, status decided by that digest, and the WEIGHTS line names the memo that was not written; `check` (refresh=True) likewise."""
    import errno
    from .. import digest_memo
    cache = tmp_path / "cache"; cache.mkdir()
    for f in stack.CACHE_FILES:
        (cache / f).mkdir() if f == "mols" else (cache / f).write_bytes(b"x")
    ro = tmp_path / "ro" / "key"; ro.mkdir(parents=True); os.chmod(ro, 0o555)
    monkeypatch.setenv("BOLTZ_CACHE", str(cache)); monkeypatch.setenv("TRITON_CACHE_DIR", str(ro / "triton")); monkeypatch.delenv(stack.WEIGHTS_DIGEST_DIR_ENV, raising=False)
    if os.access(ro, os.W_OK):                                              # root writes through 0o555: model the read-only mount at the store
        def erofs(memo_path, table):
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), memo_path)
        monkeypatch.setattr(digest_memo, "_store", erofs)
    hashed = []
    real = digest_memo.sha256_file
    monkeypatch.setattr(digest_memo, "sha256_file", lambda p, chunk=digest_memo.CHUNK: hashed.append(p) or real(p, chunk))
    for refresh in (False, True):
        w = stack.weights_status(refresh=refresh)
        assert w["status"] == "unknown" and w["sha256"] == hashlib.sha256(b"x").hexdigest() and w["cached_utc"] is None
        assert w["memo_note"].startswith("digest memo not written (") and "weights_digests.json" in w["memo_note"] and w["memo_note"].endswith("hashed afresh")
        line = stack.weights_line(w)
        assert line.startswith("[boltz2-opt] WEIGHTS unknown sha256=") and "; digest memo not written (" in line
    assert len(hashed) == 2, "one full sha256 per call — the unwritable memo costs no second pass"
    plan = stack.gate("exact", need_gpu=False, need_cache=True, check_pins=False, refresh_weights=True)      # check's gate on the read-only memo: proceeds
    assert plan["weights"]["memo_note"] and not any("memo" in r or "weights" in r.lower() for r in plan["reasons"]), plan["reasons"]
    assert "digest memo not written" in capsys.readouterr().out
    os.chmod(ro, 0o755)


def test_the_config_names_the_memo_override_and_exports_nothing_under_the_policed_prefix():
    """configs/h100.env documents MODEL_OPT_WEIGHTS_DIGEST_DIR (optional, passes through) and exports no name the package's own gate would
    refuse (`_autoload.undeclared_names`: anything under the BOLTZ2_OPT prefix but the variable itself) and none the stock route strips."""
    from .. import _autoload
    cfg = open(os.path.join(stack.tree_dir(), "configs", "h100.env")).read()
    names = set(re.findall(r"^\s*export\s+([A-Z0-9_]+)=", cfg, re.M))
    assert {"MODEL_OPT", "MODEL_OPT_TARGET_GPU", "TRITON_CACHE_DIR", "MODEL_OPT_STACK_KEY"} <= names and "BOLTZ_CACHE" not in names, names   # BOLTZ_CACHE is the caller's: no default here
    assert _autoload.undeclared_names({n: "x" for n in names}) == [], "no exported name falls under the policed BOLTZ2_OPT prefix"
    assert stack.WEIGHTS_DIGEST_DIR_ENV in cfg and not any(stack.WEIGHTS_DIGEST_DIR_ENV.startswith(p) for p in stack.load_pins()["stock_environment"]["must_be_absent_prefixes"])


def test_the_config_probe_passes_the_imports_own_refusal_through(tmp_path):
    """configs/h100.env probes `python -c "import boltz2_opt"`: only the package itself missing (ModuleNotFoundError: No module named
    'boltz2_opt') is the config's own `not installed` refusal (rc 2); any other import-time failure or refusal — the package's undeclared-name
    gate (NOT ACTIVE, rc 3), an error inside the package — reaches the caller with the import's own words and exit code, never masked."""
    fake = tmp_path / "bin"; fake.mkdir()
    py = fake / "python"
    py.write_text("#!/bin/bash\n"
                  "case \"$FAKE\" in\n"
                  "  absent)  echo \"Traceback (most recent call last):\" >&2; echo \"ModuleNotFoundError: No module named 'boltz2_opt'\" >&2; exit 1 ;;\n"
                  "  refuse)  echo \"[boltz2-opt] NOT ACTIVE: undeclared BOLTZ2_OPT_MODE in the environment\" >&2; exit 3 ;;\n"
                  "  broken)  echo \"Traceback (most recent call last):\" >&2; echo \"ModuleNotFoundError: No module named 'opt_core'\" >&2; exit 1 ;;\n"
                  "  ok)      exit 0 ;;\n"
                  "esac\n")
    py.chmod(0o755)
    cfg = os.path.join(stack.tree_dir(), "configs", "h100.env")
    def run(mode):
        env = {"PATH": f"{fake}:{os.environ['PATH']}", "FAKE": mode, "MODEL_OPT_STACK_KEY": "torch2.12.0-cu130-sm90", "HOME": str(tmp_path)}
        return subprocess.run(["bash", "-c", f"source {cfg}; echo SOURCED rc=$?"], capture_output=True, text=True, env=env)
    r = run("absent");  assert "SOURCED rc=2" in r.stdout and "NOT ACTIVE: boltz2_opt is not installed" in r.stderr, (r.stdout, r.stderr)
    r = run("refuse");  assert "SOURCED rc=3" in r.stdout and "undeclared BOLTZ2_OPT_MODE" in r.stderr and "not installed" not in r.stderr, (r.stdout, r.stderr)
    r = run("broken");  assert "SOURCED rc=1" in r.stdout and "No module named 'opt_core'" in r.stderr and "not installed" not in r.stderr, (r.stdout, r.stderr)
    r = run("ok");      assert "SOURCED rc=0" in r.stdout and r.stderr == "", (r.stdout, r.stderr)
    rs = open(os.path.join(stack.tree_dir(), "run.sh")).read()
    assert 'if [ "$PROBE_RC" = 5 ]; then echo "run.sh: boltz2_opt is not installed' in rs and 'exit "$PROBE_RC"' in rs, "run.sh's own probe: absent -> not installed, anything else passes through"

