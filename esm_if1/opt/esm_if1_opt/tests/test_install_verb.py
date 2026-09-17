"""`run.sh install [--weights DIR]` (CPU, no network): the verb's steps in order through a stub interpreter on PATH — the installed-from-here probe,
`pip install -e ../common/opt_core -e opt` unless the probe says installed, `python -I stock/check_pins.py`, then `python -m esm_if1_opt.weights DIR`
when --weights is given — with each step's failure ending the verb under its own exit code; the weights module's digest gate over an injected
transfer; the pin card held equal to the package's constants and to upstream's own URL; the pin checker over the carried archive; and the
environment recipe naming the same interpreter, lock, archive and image tag the other files name."""
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import tarfile

import pytest

from esm_if1_opt import stack, weights

TREE = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))     # esm_if1/
RUN_SH = os.path.join(TREE, "run.sh")
PINS = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))

STUB = r"""#!/bin/bash
sig=""
for a in "$@"; do case "$a" in *import*) sig="$sig <code>" ;; *) sig="$sig $a" ;; esac; done
echo "${sig# }" >> "$STUB_LOG"
case "$sig" in
  *"-I -c <code>"*) exit "${STUB_PROBE_RC:-1}" ;;
  *"-m pip install"*) exit "${STUB_PIP_RC:-0}" ;;
  *check_pins.py*) exit "${STUB_PINS_RC:-0}" ;;
  *"-m esm_if1_opt.weights"*) exit "${STUB_WEIGHTS_RC:-0}" ;;
esac
exit 0
"""


@pytest.fixture
def stub(tmp_path):
    bindir = tmp_path / "bin"; bindir.mkdir()
    p = bindir / "python"; p.write_text(STUB); p.chmod(0o755)
    log = tmp_path / "calls.log"

    def run(*args, **rc):
        env = dict(os.environ, PATH=f"{bindir}{os.pathsep}{os.environ['PATH']}", STUB_LOG=str(log))
        env.update({k: str(v) for k, v in rc.items()})
        if log.exists():
            log.unlink()
        r = subprocess.run(["bash", RUN_SH, "install", *args], env=env, capture_output=True, text=True, timeout=60)
        calls = log.read_text().splitlines() if log.exists() else []
        return r.returncode, calls, r.stdout, r.stderr
    return run


PROBE = f"-I -c <code> {TREE}/opt {TREE}/../common/opt_core"
PIP = f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt"
PINCHECK = f"-I {TREE}/stock/check_pins.py"


def test_install_runs_pip_then_the_pin_check(stub):
    rc, calls, out, err = stub()
    assert rc == 0 and calls == [PROBE, PIP, PINCHECK], (rc, calls, err)


def test_weights_dir_adds_the_weights_step_last(stub, tmp_path):
    rc, calls, out, err = stub("--weights", str(tmp_path / "w"))
    assert rc == 0 and calls == [PROBE, PIP, PINCHECK, f"-m esm_if1_opt.weights {tmp_path / 'w'}"], (rc, calls, err)
    rc, calls, _, _ = stub(f"--weights={tmp_path / 'w2'}")
    assert rc == 0 and calls[-1] == f"-m esm_if1_opt.weights {tmp_path / 'w2'}"


def test_a_tree_already_installed_from_here_skips_pip_and_says_so(stub):
    rc, calls, out, err = stub(STUB_PROBE_RC=0)
    assert rc == 0 and calls == [PROBE, PINCHECK], calls
    assert re.search(r"^\[esm_if1-opt\] install: esm_if1_opt and opt_core are already installed from this tree \(.+/opt, .+/common/opt_core\) — the pip step is skipped$", out, re.M), out


def test_each_step_failure_ends_the_verb_under_its_own_code(stub, tmp_path):
    rc, calls, out, err = stub("--weights", str(tmp_path), STUB_PIP_RC=1)
    assert rc == 1 and calls == [PROBE, PIP] and "[esm_if1-opt] install FAILED: pip install -e ../common/opt_core -e opt (exit 1)" in err
    rc, calls, out, err = stub("--weights", str(tmp_path), STUB_PINS_RC=3)
    assert rc == 3 and calls == [PROBE, PIP, PINCHECK]
    rc, calls, out, err = stub("--weights", str(tmp_path), STUB_WEIGHTS_RC=1)
    assert rc == 1 and calls[-1] == f"-m esm_if1_opt.weights {tmp_path}"


@pytest.mark.parametrize("args", [["extra"], ["--weights"], ["--weights", ""], ["--bogus", "x"], ["--weights", "/w", "more"]])
def test_usage_errors_exit_2_before_any_step(stub, args):
    rc, calls, out, err = stub(*args)
    assert rc == 2 and calls == [] and "run.sh install [--weights DIR]" in err


# ---- the weights step: python -m esm_if1_opt.weights DIR (transfer injected; no network, no torch)

def _want(payload):
    return {"file": stack.WEIGHTS_FILE, "source_url": "https://example.invalid/x.pt", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}


def test_fetch_transfers_then_checks_and_names_the_export(tmp_path, capsys):
    payload = b"esm-if1 weights stand-in"
    seen = []

    def transfer(url, dst):
        seen.append((url, dst)); open(dst, "wb").write(payload)
    assert weights.fetch(str(tmp_path / "w"), want=_want(payload), transfer=transfer) == 0
    dst = str(tmp_path / "w" / stack.WEIGHTS_FILE)
    assert seen == [("https://example.invalid/x.pt", dst)]
    out = capsys.readouterr().out
    assert f"[esm_if1-opt] WEIGHTS OK: 1/1 files in {tmp_path / 'w'} ({stack.WEIGHTS_FILE}: {len(payload)} bytes, sha256 = the pin)" in out
    assert f"export ESM_IF1_WEIGHTS={dst}" in out
    seen.clear()                                                              # a file already there is checked, never fetched again
    assert weights.fetch(str(tmp_path / "w"), want=_want(payload), transfer=transfer) == 0 and seen == []


def test_a_file_off_its_pin_fails_by_name_and_stays_in_place(tmp_path, capsys):
    payload = b"the right bytes"
    d = tmp_path / "w"; d.mkdir()
    (d / stack.WEIGHTS_FILE).write_bytes(b"the wrong bytes")               # same length, other digest
    assert weights.fetch(str(d), want=_want(payload), transfer=lambda u, p: None) == 1
    err = capsys.readouterr().err
    assert re.search(r"\[esm_if1-opt\] WEIGHTS FAILED: esm_if1_gvp4_t16_142M_UR50\.pt: .+ sha256 [0-9a-f]{16}… is not the pin [0-9a-f]{16}… \(stock/PINS\.json weights\.sha256\) — the file is left in place", err), err
    assert (d / stack.WEIGHTS_FILE).read_bytes() == b"the wrong bytes"
    (d / stack.WEIGHTS_FILE).write_bytes(b"short")                          # the byte count is checked first
    assert weights.fetch(str(d), want=_want(payload), transfer=lambda u, p: None) == 1
    assert "is 5 bytes; the pin (stock/PINS.json weights.bytes) is 15" in capsys.readouterr().err


def test_an_interrupted_transfer_is_a_named_failure(tmp_path, capsys):
    def transfer(url, dst):
        raise ConnectionError("network unreachable")
    assert weights.fetch(str(tmp_path), want=_want(b"x"), transfer=transfer) == 1
    assert re.search(r"WEIGHTS FAILED: the download of esm_if1_gvp4_t16_142M_UR50\.pt into .+ did not complete \(ConnectionError: network unreachable\); run the step again", capsys.readouterr().err)


def test_weights_usage():
    assert weights.main([]) == 2 and weights.main(["a", "b"]) == 2 and weights.main(["--dir"]) == 2


# ---- the pin card: one set of values across PINS.json, the package constants and upstream's own source

def test_pin_card_equals_the_package_constants_and_upstreams_url():
    w = PINS["weights"]
    assert (w["file"], w["bytes"], w["sha256"]) == (stack.WEIGHTS_FILE, stack.WEIGHTS_BYTES, stack.WEIGHTS_SHA256)
    assert weights.pin(TREE) == w
    src = open(os.path.join(TREE, "stock", "src", "esm", "pretrained.py"), encoding="utf-8").read()     # upstream's downloader composes this URL
    m = re.search(r'url = f"(https://dl\.fbaipublicfiles\.com/fair-esm/models/)\{model_name\}\.pt"', src)
    assert m and w["source_url"] == m.group(1) + stack.MODEL_NAME + ".pt"
    up = PINS["upstream"]["fair-esm"]
    assert (up["version"], up["commit"], up["archive"]) == (stack.FAIR_ESM_VERSION, stack.UPSTREAM_COMMIT, stack.ARCHIVE)


# ---- stock/check_pins.py over the carried archive (a stand-in distribution object; no fair-esm install needed)

def _check_pins_module():
    spec = importlib.util.spec_from_file_location("check_pins", os.path.join(TREE, "stock", "check_pins.py"))
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


class _Dist:
    def __init__(self, version, base):
        self.version, self._base = version, base

    def locate_file(self, rel):
        return os.path.join(self._base, str(rel))


def test_check_pins_accepts_the_archives_files_and_refuses_others(tmp_path):
    cp = _check_pins_module()
    up = PINS["upstream"]["fair-esm"]
    archive = os.path.join(TREE, up["archive"])
    files = cp.archive_files(archive)
    assert len(files) >= 20 and "esm/pretrained.py" in files and "esm/inverse_folding/gvp_transformer.py" in files and all(k.startswith("esm/") for k in files)
    with tarfile.open(archive, "r:gz") as tar:                                # a site-packages stand-in holding exactly the archive's esm/ files
        tar.extractall(str(tmp_path), members=[m for m in tar.getmembers() if m.name in files])
    bad, d = cp.check(up, archive, _Dist("2.0.1", str(tmp_path)))
    assert bad == [] and d["files_checked"] == len(files) and not d["files_absent"] and not d["files_differ"]
    bad, _ = cp.check(up, archive, _Dist("2.0.0", str(tmp_path)))            # PyPI's 2.0.0 is not the pin
    assert len(bad) == 1 and "want fair-esm==2.0.1" in bad[0]
    (tmp_path / "esm" / "pretrained.py").write_text("# edited\n")             # an edited / other-commit file is refused by name
    bad, d = cp.check(up, archive, _Dist("2.0.1", str(tmp_path)))
    assert d["files_differ"] == ["esm/pretrained.py"] and len(bad) == 1 and "differ from stock/fair-esm-2b369911.tar.gz" in bad[0]
    assert cp.main(["--anything"]) == 2


@pytest.mark.skipif(importlib.util.find_spec("esm") is not None, reason="fair-esm is installed here: the absent case cannot be shown")
def test_check_pins_names_an_absent_fair_esm():
    r = subprocess.run([sys.executable, "-I", os.path.join(TREE, "stock", "check_pins.py")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 3 and re.match(r"check_pins: fair-esm is not installed on \S+; want fair-esm==2\.0\.1 \(pip install --no-deps stock/fair-esm-2b369911\.tar\.gz; STOCK\.md\)", r.stderr), r.stderr


# ---- environment/: the recipe names the same interpreter, lock, archive and tag as the pin card and the other two files

def test_environment_recipe_agrees_with_the_pin_card():
    env = os.path.join(TREE, "environment")
    dockerfile = open(os.path.join(env, "Dockerfile"), encoding="utf-8").read()
    raw = [l.strip() for l in open(os.path.join(env, "requirements.lock"), encoding="utf-8") if l.strip() and not l.startswith("#")]
    options, lock = [l for l in raw if l.startswith("--")], [l for l in raw if not l.startswith("--")]
    assert options == []                                                       # no index / find-links lines: every non-PyPI file is named by URL and digest
    adef = open(os.path.join(env, "apptainer.def"), encoding="utf-8").read()
    assert re.search(r"^FROM nvidia/cuda:12\.1\.1-runtime-ubuntu22\.04$", dockerfile, re.M)
    assert "cpython-3.11.5+20230826-x86_64-unknown-linux-gnu-install_only.tar.gz" in dockerfile
    up = PINS["upstream"]["fair-esm"]
    assert f"fair-esm @ git+{up['repo']}.git@{up['commit']}" in lock
    urls = {l.split(" @ ")[0]: l.split(" @ ")[1] for l in lock if " @ https://" in l}
    assert sorted(urls) == ["setuptools", "torch", "torch_cluster", "torch_scatter", "torch_sparse"], sorted(urls)
    assert urls["torch"].startswith("https://download.pytorch.org/whl/cu121/torch-2.4.0%2Bcu121-cp311-cp311-linux_x86_64.whl#sha256=")
    assert all(urls[k].startswith("https://data.pyg.org/whl/torch-2.4.0%2Bcu121/" + k + "-") and "%2Bpt24cu121-cp311-cp311-linux_x86_64.whl#sha256=" in urls[k] for k in ("torch_cluster", "torch_scatter", "torch_sparse"))
    assert all(re.match(r"^[A-Za-z0-9_.\-]+(==[^ ]+| @ git\+https://\S+@[0-9a-f]{40}| @ https://\S+\.whl#sha256=[0-9a-f]{64})$", l) for l in lock), [l for l in lock if " " in l and " @ " not in l]
    arc = os.path.basename(up["archive"])
    assert f"COPY esm_if1/{up['archive']} /tmp/stock/{arc}" in dockerfile and f"pip install --no-cache-dir --no-deps /tmp/stock/{arc}" in dockerfile
    assert "COPY esm_if1/environment/requirements.lock /tmp/requirements.lock" in dockerfile and "RUN bash run.sh install" in dockerfile
    assert "grep -v -E '^(#|fair-esm @)' /tmp/requirements.lock > /tmp/stack.txt" in dockerfile and "pip install --no-cache-dir --no-deps -r /tmp/stack.txt" in dockerfile
    assert "--index-url" not in dockerfile and "--find-links" not in dockerfile and "--extra-index-url" not in dockerfile      # the lock names its sources; the recipe adds none
    assert re.search(r"^Bootstrap: docker-daemon\nFrom: esm_if1-kit:dev$", adef, re.M) and "docker build -f esm_if1/environment/Dockerfile -t esm_if1-kit:dev ." in dockerfile
