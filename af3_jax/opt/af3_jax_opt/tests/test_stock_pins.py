"""stock/: the freeze file and the patches are present as PINS.json says, the upstream archive when this tree carries it; the nine stock files
are present under stock/src; the one variant; check_pins.py reports; unpack_src.sh lays the pinned source out."""
import json
import os
import importlib.util
import subprocess
import sys

import pytest

from .conftest import TREE

STOCK = os.path.join(TREE, "stock")


def _archive_or_skip(pins):
    """The upstream source archive under stock/, or a skip BY NAME: a tree distributed without it lays the source out by cloning upstream.repo
    at upstream.commit (stock/unpack_src.sh) — nothing of the archive to check there, and the tests never touch the network."""
    archive = os.path.join(STOCK, pins["upstream"]["archive"]["file"])
    if not os.path.isfile(archive):
        pytest.skip(f"stock/{pins['upstream']['archive']['file']} is not in this tree: the install clones upstream.repo at upstream.commit instead (stock/unpack_src.sh)")
    return archive


def test_pins_names_the_present_files():
    pins = json.load(open(os.path.join(STOCK, "PINS.json")))
    assert os.path.isfile(os.path.join(STOCK, "unpack_src.sh")) and os.path.isfile(os.path.join(STOCK, "check_pins.py"))
    assert os.path.isfile(os.path.join(os.path.dirname(STOCK), pins["freeze"]["file"]))            # the pinned stack's one list, relative to the kit tree: environment/requirements.lock
    assert pins["upstream"]["commit"] == "bc32b22ff5902e3daffd5d1f7203d7f2ab6cb997"
    for rel in pins["stock_files"]:
        assert os.path.isfile(os.path.join(STOCK, "src", rel)), rel
    assert list(pins["variants"]) == ["p2"]
    spec = pins["variants"]["p2"]
    assert set(spec["checkpoint"]) >= {"file", "url", "sha256", "bytes"} and len(spec["checkpoint"]["sha256"]) == 64
    assert spec["converted"]["file"] == "of3_ported_weights.bin.zst" and len(spec["converted"]["sha256"]) == 64
    assert pins["stock_proof"]["must_be_absent_prefixes"] == ["AF3_JAX_", "AF3P_", "AF3_FLASHPAIRFORMER", "AF3_DIFFUSION_HOIST"]
    assert set(pins["stock_proof"]["prefix_owners"]) == set(pins["stock_proof"]["must_be_absent_prefixes"])


def test_src_tree_is_the_archive_minus_test_data():
    assert not os.path.exists(os.path.join(STOCK, "src", "src", "alphafold3", "test_data"))
    assert os.path.isfile(os.path.join(STOCK, "src", "run_alphafold.py")) and os.path.isfile(os.path.join(STOCK, "src", "uv.lock"))


def test_check_pins_reports_on_a_box_without_the_stack():
    p = subprocess.run([sys.executable, "-I", os.path.join(STOCK, "check_pins.py"), "--py", "/nonexistent/python", "--json"],
                       capture_output=True, text=True, timeout=120)
    assert p.returncode == 3, p.stderr
    rep = json.loads(p.stdout)
    assert rep["files"]["ok"] is True and rep["interpreter"]["ok"] is False and rep["ok"] is False


def _pins_env(**overrides):
    pins = json.load(open(os.path.join(STOCK, "PINS.json")))
    env = dict(os.environ)
    env.update(pins["image"]["env"])                                     # start at the pinned values, then apply the scenario's overrides
    env.update(overrides)
    return env


def _check_pins(env, *extra):
    return subprocess.run([sys.executable, "-I", os.path.join(STOCK, "check_pins.py"), "--py", "/nonexistent/python", *extra],
                          capture_output=True, text=True, timeout=120, env=env)


def test_env_pins_note_or_drift_never_refuse():
    """XLA_PYTHON_CLIENT_PREALLOCATE and XLA_CLIENT_MEM_FRACTION size JAX's device-memory pool — allocator behaviour, not model outputs: a value other
    than the pinned image's is a NOTE (one line on stderr, never silent, even under --quiet). XLA_FLAGS at another value, or any pinned variable
    UNSET, is DRIFT: named on one `PINS drift …` line each, never a refusal (exit 0 as far as the environment goes). Nothing outside the pinned
    variables is read."""
    rep = json.loads(_check_pins(_pins_env(XLA_PYTHON_CLIENT_PREALLOCATE="false"), "--json").stdout)
    assert rep["env"]["met"] is True and rep["env"]["drift"] == []
    var = rep["env"]["vars"]["XLA_PYTHON_CLIENT_PREALLOCATE"]
    assert (var["expected"], var["actual"], var["met"]) == ("true", "false", True) and var["note"].startswith("XLA_PYTHON_CLIENT_PREALLOCATE=false ")
    assert rep["env"]["notes"] == [var["note"]]

    p = _check_pins(_pins_env(XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_CLIENT_MEM_FRACTION="0.5"), "--quiet")
    notes = [ln for ln in p.stderr.splitlines() if ln.startswith("[af3-jax-opt] NOTE XLA_")]                 # on stderr explicitly, never on a stdout a caller json-parses
    assert len(notes) == 2 and "XLA_PYTHON_CLIENT_PREALLOCATE=false" in notes[0] and "XLA_CLIENT_MEM_FRACTION=0.5" in notes[1]
    assert "NOTE XLA_" not in p.stdout and "PINS drift" not in p.stderr

    rep = json.loads(_check_pins(_pins_env(), "--json").stdout)                                                # the pinned values: met, no note, no drift
    assert rep["env"]["met"] is True and rep["env"]["notes"] == [] and rep["env"]["drift"] == [] and all("note" not in v for v in rep["env"]["vars"].values())

    rep = json.loads(_check_pins(_pins_env(XLA_FLAGS="--xla_gpu_enable_triton_gemm=true"), "--json").stdout)   # XLA_FLAGS selects code paths: another value is drift, named
    assert rep["env"]["met"] is False and rep["env"]["vars"]["XLA_FLAGS"]["met"] is False and rep["env"]["notes"] == []
    assert rep["env"]["drift"] == ["XLA_FLAGS='--xla_gpu_enable_triton_gemm=true' (pinned '--xla_gpu_enable_triton_gemm=false')"] and rep["drift"] == rep["env"]["drift"]
    p = _check_pins(_pins_env(XLA_FLAGS="--xla_gpu_enable_triton_gemm=true"), "--quiet")
    drift = [ln for ln in p.stderr.splitlines() if ln.startswith("[af3-jax-opt] PINS drift ")]
    assert drift == ["[af3-jax-opt] PINS drift XLA_FLAGS='--xla_gpu_enable_triton_gemm=true' (pinned '--xla_gpu_enable_triton_gemm=false') — named, not refused: the modes run on this stack and their lines name it"], p.stderr
    assert "env=drift" in p.stdout                                                                             # (the text report prints and rc is 3 here only because this test box has no stock install: install=missing)

    env_unset = _pins_env(); env_unset.pop("XLA_CLIENT_MEM_FRACTION")                                          # a pool variable UNSET: drift, named (the stack states all three)
    rep = json.loads(_check_pins(env_unset, "--json").stdout)
    assert rep["env"]["met"] is False and rep["env"]["vars"]["XLA_CLIENT_MEM_FRACTION"] == {"expected": "0.95", "actual": None, "met": False, "drift": "XLA_CLIENT_MEM_FRACTION=unset (pinned '0.95')"}

def test_stock_patches_are_present_and_named_in_the_declared_exception():
    """stock/patches: each patch file is present and named in PINS upstream.patches, touching the files it says (every touched file is
    one of the nine stock_files); the pinned stack carries one `tested_on` documentation line and no id anything compares against."""
    pins = json.load(open(os.path.join(STOCK, "PINS.json")))
    patches = pins["upstream"]["patches"]
    assert [p["file"] for p in patches] == ["patches/04_of3_empty_template_restype_gap.diff"]
    for p in patches:
        assert os.path.isfile(os.path.join(STOCK, p["file"])), p["file"]
        body = open(os.path.join(STOCK, p["file"]), encoding="utf-8").read()
        for rel in p["touches"]:
            assert f"--- a/{rel}" in body and f"+++ b/{rel}" in body, rel
            assert rel in pins["stock_files"], rel
    assert "id" not in pins["image"] and "base" not in pins["image"] and isinstance(pins["image"]["tested_on"], str) and "docker/Dockerfile" in pins["image"]["tested_on"]


def test_stock_src_differs_from_upstream_in_exactly_the_declared_exception(tmp_path):
    """stock/src is upstream at the pin, byte for byte, with exactly ONE declared, named exception (stock/PINS.json upstream.patches[0]:
    the OF3-weights port-correctness fix, applied stock-side — not a kit-runtime lever, so every arm, off included, carries it). A full
    tree walk (not just the one file): if a second file ever silently diverges from upstream, this fails by naming it."""
    import tarfile
    pins = json.load(open(os.path.join(STOCK, "PINS.json")))
    archive = _archive_or_skip(pins)
    prefix = "alphafold3-bc32b22f/"  # pins["upstream"]["archive"]["recipe"] names this git-archive --prefix
    excludes = tuple(pins["upstream"]["src_dir"]["excludes"])
    with tarfile.open(archive) as tf:
        tf.extractall(tmp_path, filter="data")
    pristine_root = os.path.join(tmp_path, prefix.rstrip("/"), "src")
    stock_root = os.path.join(STOCK, "src", "src")

    pristine_files = {}
    for dirpath, _dirs, files in os.walk(pristine_root):
        rel_dir = os.path.relpath(dirpath, pristine_root)
        for fn in files:
            rel = os.path.normpath(os.path.join(rel_dir, fn))
            if any(rel.replace(os.sep, "/").startswith(ex[len("src/"):].rstrip("/")) for ex in excludes):
                continue
            pristine_files[rel] = os.path.join(dirpath, fn)

    differing = []
    missing = []
    for rel, pristine_path in pristine_files.items():
        stock_path = os.path.join(stock_root, rel)
        if not os.path.isfile(stock_path):
            missing.append(rel)
            continue
        if open(pristine_path, "rb").read() != open(stock_path, "rb").read():
            differing.append(rel.replace(os.sep, "/"))
    assert not missing, f"present upstream, missing from stock/src: {missing}"

    declared = {p["touches"][0] for p in pins["upstream"]["patches"]}
    for rel in pins["upstream"]["patches"][0]["touches"]:
        assert rel.startswith("src/")
    declared_rel = {d[len("src/"):] for d in declared}
    assert set(differing) == declared_rel, f"stock/src diverges from upstream outside the declared exception: {set(differing) - declared_rel}"

    # the ONE declared file differs in exactly the declared way: pristine + the patch, byte for byte, is what stock/src actually carries
    rel = next(iter(declared_rel))
    pristine_bytes = open(pristine_files[rel], "rb").read()
    patched_copy = tmp_path / "patched_copy.py"
    patched_copy.write_bytes(pristine_bytes)
    patch_path = os.path.join(STOCK, pins["upstream"]["patches"][0]["file"])
    r = subprocess.run(["patch", "-p0", str(patched_copy)], stdin=open(patch_path, "rb"), capture_output=True, timeout=30)
    assert r.returncode == 0, r.stderr.decode(errors="replace")
    assert patched_copy.read_bytes() == open(os.path.join(stock_root, rel), "rb").read()


def _load_check_pins():
    import importlib.util
    spec = importlib.util.spec_from_file_location("check_pins_under_test", os.path.join(STOCK, "check_pins.py"))
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def _interpreter_shim(tmp_path, version, packages):
    """An executable standing in for the fork's interpreter: whatever it is asked (`-I -c <code> <names>`), it answers the version probe's
    JSON line with the given python version and package versions."""
    shim = tmp_path / f"python-{version}"
    shim.write_text("#!/bin/sh\ncat <<'EOF'\n" + json.dumps({"python": version, "packages": packages}) + "\nEOF\n")
    shim.chmod(0o755)
    return str(shim)


def test_python_pin_accepts_patch_level_differences_only(tmp_path):
    """The interpreter's python against PINS image.python_version (3.12.1, the pinned value, unchanged): 3.12.1 is met silently; 3.12.3
    (upstream's own docker/Dockerfile on Ubuntu 24.04) is met with ONE `[af3-jax-opt] NOTE ... patch level only — proceeding` line on stderr;
    3.11.x and 3.13.x are DRIFT — named (`python=3.13.0 (pinned 3.12.1)`, `interpreter=drift`), never refused; a package off the freeze is
    drift likewise at any python. The interpreter report's `ok` is False only for an interpreter that is absent or cannot answer."""
    cp = _load_check_pins()
    pins = json.load(open(os.path.join(STOCK, "PINS.json")))
    assert pins["image"]["python_version"] == "3.12.1"
    want = cp.freeze_versions(pins)
    good = {n: want[n.lower().replace("_", "-")] for n in pins["check_packages"]}
    assert all(good.values())

    rep = cp.interpreter_report(_interpreter_shim(tmp_path, "3.12.1", good), pins)          # the pin itself: met, silent
    assert rep["ok"] is True and rep["met"] is True and rep["drift"] == [] and "python_note" not in rep and "python_version_expected" not in rep

    rep = cp.interpreter_report(_interpreter_shim(tmp_path, "3.12.3", good), pins)          # patch level: met, one NOTE
    assert rep["ok"] is True and rep["met"] is True and rep["python_version_expected"] == "3.12.1"
    assert rep["python_note"] == "python 3.12.3 differs from the pinned 3.12.1 in patch level only — proceeding"
    assert all(d["met"] for d in rep["packages"].values())

    for other in ("3.11.9", "3.13.0"):                                                       # minor level: drift, named — never a refusal
        rep = cp.interpreter_report(_interpreter_shim(tmp_path, other, good), pins)
        assert rep["ok"] is True and rep["met"] is False and rep["drift"] == [f"python={other} (pinned 3.12.1)"] and "python_note" not in rep, other

    bad = dict(good); bad[pins["check_packages"][0]] = "0.0.0"                               # a package off the freeze: drift, named, at any python
    rep = cp.interpreter_report(_interpreter_shim(tmp_path, "3.12.3", bad), pins)
    n0 = pins["check_packages"][0]
    assert rep["ok"] is True and rep["met"] is False and rep["packages"][n0]["met"] is False and rep["drift"] == [f"{n0}=0.0.0 (pinned {good[n0]})"]

    # the script's own words: ONE NOTE line on stderr for 3.12.3 (text mode, --quiet), none for 3.12.1; the refusal line for 3.13.0 is the existing one
    def run(version, *extra):
        return subprocess.run([sys.executable, "-I", os.path.join(STOCK, "check_pins.py"), "--py", _interpreter_shim(tmp_path, version, good), *extra],
                              capture_output=True, text=True, timeout=120, env=_pins_env())
    p = run("3.12.3", "--quiet")
    notes = [l for l in p.stderr.splitlines() if l.startswith("[af3-jax-opt] NOTE")]
    assert notes == ["[af3-jax-opt] NOTE python 3.12.3 differs from the pinned 3.12.1 in patch level only — proceeding"], p.stderr
    assert "NOTE" not in p.stdout and "interpreter=ok" in p.stdout                             # (rc is 3 here only because this test box has no stock install: install=missing)
    p = run("3.12.1", "--quiet")
    assert "NOTE" not in p.stderr and "interpreter=ok" in p.stdout
    p = run("3.13.0", "--quiet")
    assert "NOTE" not in p.stderr and "[af3-jax-opt] PINS drift python=3.13.0 (pinned 3.12.1) — named, not refused" in p.stderr and "interpreter=drift" in p.stdout
    rep = json.loads(run("3.12.3", "--json").stdout)                                          # --json: stdout stays one JSON object; the note rides inside it
    assert rep["interpreter"]["ok"] is True and rep["interpreter"]["python_note"].startswith("python 3.12.3 differs")


def test_torch_is_held_to_the_locks_release_not_its_build():
    """torch serves the weights converter only (`run.sh install`); a CUDA build of the pinned release (2.7.1+cu126) is accepted against the lock's CPU
    build (2.7.1+cpu) with ONE NOTE naming both — the same release, another build. Another release, an absent torch, and any other package's
    local tag are drift (named on a PINS drift line, never refused)."""
    spec = importlib.util.spec_from_file_location("check_pins", os.path.join(STOCK, "check_pins.py"))
    cp = importlib.util.module_from_spec(spec); spec.loader.exec_module(cp)
    e = cp.package_entry("torch", "2.7.1+cpu", "2.7.1+cu126")
    assert e["met"] is True and "drift" not in e and e["note"].startswith("torch=2.7.1+cu126 (the lock's is 2.7.1+cpu: the same release")
    assert cp.package_entry("torch", "2.7.1+cpu", "2.7.1+cpu") == {"expected": "2.7.1+cpu", "actual": "2.7.1+cpu", "met": True}     # the pinned build: met, no note
    assert cp.package_entry("torch", "2.7.1+cpu", "2.7.1")["met"] is True                                                            # PyPI's build of the release: the same release
    assert cp.package_entry("torch", "2.7.1+cpu", "2.8.0+cpu") == {"expected": "2.7.1+cpu", "actual": "2.8.0+cpu", "met": False, "drift": "torch=2.8.0+cpu (pinned 2.7.1+cpu)"}   # another release: drift, named
    assert cp.package_entry("torch", "2.7.1+cpu", None) == {"expected": "2.7.1+cpu", "actual": None, "met": False, "drift": "torch=absent (pinned 2.7.1+cpu)"}                # absent: drift, named
    assert cp.package_entry("jax", "0.10.2", "0.10.2+local")["drift"] == "jax=0.10.2+local (pinned 0.10.2)"                          # the local-tag rule is torch's alone
    assert cp.LOCAL_TAG_FREE == ("torch",)


def test_unpack_src_lays_out_the_archive_with_the_patches_applied(tmp_path):
    """stock/unpack_src.sh DEST on a tree that carries the archive: the archive route, the pinned source at DEST, every upstream.patches entry
    applied (the ONE declared exception is then present in the laid-out file), one STOCK SOURCE line per step; a non-empty DEST is refused."""
    pins = json.load(open(os.path.join(STOCK, "PINS.json")))
    _archive_or_skip(pins)
    dest = tmp_path / "af3src"
    r = subprocess.run(["bash", os.path.join(STOCK, "unpack_src.sh"), str(dest)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    lines = [l for l in r.stdout.splitlines() if l.startswith("[af3-jax-opt] STOCK SOURCE")]
    assert lines[0].startswith("[af3-jax-opt] STOCK SOURCE route=archive file=stock/" + pins["upstream"]["archive"]["file"]) and lines[1] == f"[af3-jax-opt] STOCK SOURCE patches={len(pins['upstream']['patches'])} applied (stock/PINS.json upstream.patches)"
    assert (dest / "run_alphafold.py").is_file() and (dest / "convert_of3_weights.py").is_file()
    for p in pins["upstream"]["patches"]:
        for rel in p["touches"]:
            assert (dest / rel).read_bytes() == open(os.path.join(STOCK, "src", rel), "rb").read(), rel   # laid out + patched == stock/src, the patched pin
    again = subprocess.run(["bash", os.path.join(STOCK, "unpack_src.sh"), str(dest)], capture_output=True, text=True)
    assert again.returncode == 2 and "not empty" in again.stderr
