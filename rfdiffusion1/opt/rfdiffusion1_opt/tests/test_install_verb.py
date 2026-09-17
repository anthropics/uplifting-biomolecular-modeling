"""`run.sh install [--weights DIR]` — the verb's argument handling and call sequence (a stub `python` on PATH records every invocation; nothing is
installed), the weights step's digest gate (rfdiffusion1_opt.weights.fetch with an injected transfer and pin table; no network), and the
environment recipe (environment/): the lock, the Dockerfile and the Apptainer definition install the stack stock/PINS.json pins. CPU only."""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.normpath(os.path.join(HERE, "..", "..", ".."))          # the kit tree: run.sh, stock/, opt/, environment/
RUN_SH = os.path.join(TREE, "run.sh")
PINS = json.load(open(os.path.join(TREE, "stock", "PINS.json"), encoding="utf-8"))

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PROBE / STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m rfdiffusion1_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


# --------------------------------------------------------------------------------------------------------------- run.sh install: the call sequence
@pytest.fixture
def stub(tmp_path):
    bindir = tmp_path / "bin"; bindir.mkdir()
    py = bindir / "python"; py.write_text(STUB); py.chmod(py.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "calls.log"

    def run_sh(args, **rc):
        if log.exists(): log.unlink()
        env = {"PATH": str(bindir) + os.pathsep + "/usr/bin:/bin", "HOME": str(tmp_path), "STUB_LOG": str(log)}
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=str(tmp_path))
        lines = log.read_text().splitlines() if log.exists() else []
        probes = [c for c in lines if c.startswith("-I -c import os,sys")]             # the installed-from-this-tree probe (one per install call)
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith("-I -c import os,sys")], probes
    return run_sh


def test_install_runs_pip_then_the_pin_check(stub):
    rc, out, calls, probes = stub(["install"])
    assert rc == 0, out
    assert calls == [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py"], calls
    assert len(probes) == 1, probes


def test_weights_dir_adds_the_weights_step_last(stub):
    for args in (["install", "--weights", "/data/rfd_weights"], ["install", "--weights=/data/rfd_weights"]):
        rc, out, calls, _ = stub(args)
        assert rc == 0, out
        assert calls[2:] == ["-m rfdiffusion1_opt.weights /data/rfd_weights"] and len(calls) == 3, calls


def test_usage_errors_call_nothing(stub):
    for args in (["install", "--weights"], ["install", "--weights", "--bogus"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                 ["install", "--mode", "exact"], ["install", "--config", "h100"]):
        rc, out, calls, probes = stub(args)
        assert rc == 2, (args, out)
        assert calls == [] and probes == [], (args, calls, probes)
        assert "run.sh:" in out


def test_a_failed_step_stops_the_sequence_with_its_code(stub):
    assert stub(["install", "--weights", "/w"], pip=1)[::2] == (1, [f"-m pip install -e {TREE}/../common/opt_core -e {TREE}/opt"])
    rc, out, calls, _ = stub(["install", "--weights", "/w"], pins=3); assert (rc, len(calls)) == (3, 2), (out, calls)
    rc, out, calls, _ = stub(["install", "--weights", "/w"], weights=1); assert (rc, len(calls)) == (1, 3), (out, calls)


def test_a_tree_already_installed_skips_pip_by_name(stub):
    """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
    --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
    rc, out, calls, probes = stub(["install", "--weights", "/w"], probe=0)
    assert rc == 0, out
    assert "installed from this tree already" in out
    assert calls == [f"-I {TREE}/stock/check_pins.py", "-m rfdiffusion1_opt.weights /w"], calls   # pin check, weights — no pip
    assert len(probes) == 1, probes


def test_install_is_reached_past_the_routes_and_the_routes_never_reach_it():
    """run.sh: the routes (design | check | warm) run under `if [ "$CMD" != install ]` and end in exec; the install step is the block after that `fi` —
    so the two gate probes stay the file's first python invocations (test_core_gate_routes) and no route runs pip or the pin check."""
    code = [ln.split("#")[0].rstrip() for ln in open(RUN_SH, encoding="utf-8").read().splitlines() if ln.strip() and not ln.lstrip().startswith("#")]
    guard = code.index('if [ "$CMD" != install ]; then'); exec_at = next(i for i, ln in enumerate(code) if ln.startswith("exec python -m rfdiffusion1_opt "))
    assert code[exec_at + 1] == "fi" and guard < exec_at
    for needle in ("-m pip install", "check_pins.py", "-m rfdiffusion1_opt.weights"):
        at = [i for i, ln in enumerate(code) if needle in ln]
        assert at and all(i > exec_at for i in at), (needle, at, exec_at)
    assert code[-1] == "exit 0"


# ------------------------------------------------------------------------------------------------------------- the weights step's digest gate
PAYLOAD = {"Complex_base_ckpt.pt": b"complex-base-bytes", "Base_ckpt.pt": b"base-bytes"}


@pytest.fixture
def gate(tmp_path):
    from rfdiffusion1_opt import weights as W
    d = tmp_path / "w"
    files = [{"name": k, "url": f"https://upstream.example/pub/{k}", "sha256": hashlib.sha256(v).hexdigest(), "size_bytes": len(v)} for k, v in PAYLOAD.items()]
    calls = []

    def fetcher(url, dest):
        calls.append((url, dest))
        with open(dest, "wb") as fh: fh.write(PAYLOAD[os.path.basename(dest)])

    def fetch(**kw):
        out = io.StringIO()
        rc = W.fetch(str(d), files=kw.pop("files", files), fetcher=kw.pop("fetcher", fetcher),
                     hasher=kw.pop("hasher", lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()), out=out, **kw)
        return rc, out.getvalue()
    return {"W": W, "dir": d, "files": files, "calls": calls, "fetch": fetch, "fetcher": fetcher}


def test_all_fetched_and_pinned(gate):
    rc, out = gate["fetch"]()
    assert rc == 0, out
    assert "WEIGHTS OK: 2/2" in out and f"export WEIGHTS={gate['dir']}" in out
    assert gate["calls"] == [(f["url"], str(gate["dir"] / f["name"])) for f in gate["files"]]
    assert sorted(os.listdir(gate["dir"])) == sorted(PAYLOAD)


def test_a_present_file_is_kept_and_checked(gate):
    gate["dir"].mkdir(); (gate["dir"] / "Base_ckpt.pt").write_bytes(PAYLOAD["Base_ckpt.pt"])
    rc, out = gate["fetch"]()
    assert rc == 0, out
    assert "Base_ckpt.pt: present" in out and [os.path.basename(dst) for _, dst in gate["calls"]] == ["Complex_base_ckpt.pt"]


def test_a_digest_off_the_pin_is_refused_by_name_and_left_in_place(gate):
    gate["dir"].mkdir(); p = gate["dir"] / "Complex_base_ckpt.pt"; p.write_bytes(b"x" * len(PAYLOAD["Complex_base_ckpt.pt"]))   # the pinned size, other bytes
    rc, out = gate["fetch"]()
    assert rc == 1, out
    assert "REFUSED: 1 of 2" in out and "Complex_base_ckpt.pt (sha256 " in out.splitlines()[-1]
    assert p.read_bytes() == b"x" * len(PAYLOAD["Complex_base_ckpt.pt"])                     # left in place, never deleted


def test_a_size_off_the_pin_is_refused_before_hashing(gate):
    gate["dir"].mkdir(); (gate["dir"] / "Base_ckpt.pt").write_bytes(b"short")
    rc, out = gate["fetch"](hasher=lambda p: pytest.fail("a file of the wrong size is not hashed") if p.endswith("Base_ckpt.pt") else hashlib.sha256(open(p, "rb").read()).hexdigest())
    assert rc == 1 and "Base_ckpt.pt (5 bytes ≠ pin " in out, out


def test_a_file_without_a_url_is_refused(gate):
    files = [dict(f, url=None) if f["name"] == "Base_ckpt.pt" else f for f in gate["files"]]
    rc, out = gate["fetch"](files=files)
    assert rc == 1, out
    assert "FAILED Base_ckpt.pt: absent" in out and "records no URL" in out


def test_a_fetch_error_is_relayed_and_the_partial_file_named(gate):
    def broken(url, dest):
        with open(dest + gate["W"].PART_SUFFIX, "wb") as fh: fh.write(b"half")
        raise ConnectionError("network unreachable")
    rc, out = gate["fetch"](fetcher=broken)
    assert rc == 1, out
    assert "FAILED fetching Complex_base_ckpt.pt: ConnectionError: network unreachable — the partial file " in out
    assert os.listdir(gate["dir"]) == ["Complex_base_ckpt.pt.part"]                            # left in place, nothing deleted


def test_transfer_writes_through_a_part_file(gate, monkeypatch):
    """weights.transfer streams the URL into <dest>.part and renames it onto <dest> when complete (urllib; no network here: urlopen is replaced)."""
    W = gate["W"]; gate["dir"].mkdir(); dest = str(gate["dir"] / "Base_ckpt.pt"); seen = {}

    class Resp(io.BytesIO):
        def __enter__(self): return self
        def __exit__(self, *a): return False

    def urlopen(url, timeout=None):
        seen["url"] = url; assert not os.path.exists(dest); return Resp(PAYLOAD["Base_ckpt.pt"])
    monkeypatch.setattr(W.urllib.request, "urlopen", urlopen)
    W.transfer("https://upstream.example/pub/Base_ckpt.pt", dest, chunk=4)
    assert open(dest, "rb").read() == PAYLOAD["Base_ckpt.pt"] and not os.path.exists(dest + W.PART_SUFFIX) and seen["url"].endswith("/Base_ckpt.pt")


def test_pinned_files_are_the_pins_weights_entries_with_a_digest(gate):
    files = gate["W"].pinned_files(PINS)
    assert [f["name"] for f in files] == ["Complex_base_ckpt.pt", "Base_ckpt.pt"] and files[0]["name"] == PINS["weights"]["checkpoint"]
    for f in files:
        e = PINS["weights"][f["name"]]
        assert (f["sha256"], f["size_bytes"], f["url"]) == (e["sha256"], e["size_bytes"], e["url_kit_installer"]) and f["url"].startswith("https://")


def test_pins_urls_are_upstreams_download_list():
    """stock/PINS.json weights URLs are upstream's own: scripts/download_models.sh of the pinned archive lists each file's URL over plain
    http; `url_kit_installer` — the one the weights step transfers from — is the same host and path over https."""
    with tarfile.open(os.path.join(TREE, PINS["upstream"]["rfdiffusion"]["archive"])) as tf:
        member = next(m for m in tf.getmembers() if m.name.endswith("/scripts/download_models.sh"))
        script = tf.extractfile(member).read().decode()
    listed = {u.rsplit("/", 1)[1]: u for u in re.findall(r"http://files\.ipd\.uw\.edu/pub/RFdiffusion/[0-9a-f]+/\w+\.pt", script)}
    for name, e in PINS["weights"].items():
        if not isinstance(e, dict): continue
        assert e["url_kit_installer"] == "https://" + listed[name][len("http://"):], name


# ---------------------------------------------------------------------------------------- environment/: the recipe installs the pinned stack
def _lock():
    """{dist: version} from the lock's `name==version` lines and `name @ <url of name-version-…whl>#sha256=…` lines (a build off PyPI, by URL)."""
    lock, urls = {}, {}
    for line in open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#"): continue
        if " @ " in line:
            k, url = (x.strip() for x in line.split(" @ ", 1)); k = k.lower().replace("-", "_"); urls[k] = url
            lock[k] = re.search(r"/" + k + r"-([^-]+)-", url.replace("%2B", "+")).group(1)
        else:
            k, v = line.split("==", 1); lock[k.lower().replace("-", "_")] = v
    return lock, urls


def test_the_lock_pins_the_stack_pins_names():
    """environment/requirements.lock is the stack's one package list (the image and README route C install it): every distribution stock/PINS.json
    `pinned_stack` names is locked at that version (torch by its PyPI dist version — torch.__version__ carries +cu121, PINS `cuda`); dgl, the one
    build not on PyPI, by its wheel's URL on DGL's wheel list with a sha256; no index option lines; stock, the kit, the core and the
    interpreter's own pip / setuptools are not lines of it (stock installs from stock/, the kit by `run.sh install`)."""
    st = PINS["pinned_stack"]; lock, urls = _lock()
    for k in ("torch", "dgl", "triton", "e3nn", "numpy", "scipy", "opt_einsum", "hydra-core", "omegaconf", "pyrsistent"):
        assert lock[k.lower().replace("-", "_")] == st[k], k
    assert (lock["torch"], lock["dgl"]) == (st["stack_check"]["torch"], st["stack_check"]["dgl"])
    assert st["cuda"] == "12.1" and lock["nvidia_cuda_runtime_cu12"].startswith(st["cuda"] + ".")
    assert list(urls) == ["dgl"] and re.match(r"https://data\.dgl\.ai/wheels/torch-2\.4/cu124/dgl-2\.4\.0%2Bcu124-cp311-cp311-\w+\.whl#sha256=[0-9a-f]{64}$", urls["dgl"])
    text = open(os.path.join(TREE, "environment", "requirements.lock"), encoding="utf-8").read()
    assert not re.search(r"(?m)^-", text)                                                        # no option lines: every line is a pin
    assert not {"rfdiffusion", "se3_transformer", "rfdiffusion1_opt", "opt_core", "pip", "setuptools"} & set(lock)
    assert st["python"].startswith("3.11.")


def test_the_dockerfile_builds_the_pinned_stack():
    """environment/Dockerfile: the CUDA runtime base of the stack's toolkit generation (PINS `nvcc` 12.4 — DGL's cu124 build), the python-build-standalone
    interpreter at PINS `python`, the lock and the stock archive PINS names as the files it installs (SE3Transformer, then the checkout editable —
    upstream's two steps), `run.sh install` as its kit step; environment/apptainer.def is made from that image."""
    st = PINS["pinned_stack"]; up = PINS["upstream"]["rfdiffusion"]
    df = open(os.path.join(TREE, "environment", "Dockerfile"), encoding="utf-8").read()
    m = re.search(r"^FROM (\S+)$", df, re.M); assert m
    assert m.group(1).startswith(f"nvidia/cuda:{st['nvcc']}.") and "-runtime-ubuntu22.04" in m.group(1), m.group(1)
    assert f"cpython-{st['python']}+" in df
    assert "COPY rfdiffusion1/environment/requirements.lock " in df and f"COPY rfdiffusion1/{up['archive']} " in df
    assert df.index("--no-deps /opt/rfd/" + up["vendored"]["se3-transformer"]["path"]) < df.index("--no-deps -e /opt/rfd")
    assert re.search(r"(?m)^RUN (PIP_CONSTRAINT=\S+ )?bash run\.sh install$", df) and "ENV PYTHONHASHSEED=0" in df
    assert not re.search(r"(?m)^ENV.*\bWEIGHTS=", df) and "RFD_WEIGHTS=" not in df                  # the weights directory is named at run time, never baked
    assert re.search(r"(?m)^From: rfdiffusion1-kit:", open(os.path.join(TREE, "environment", "apptainer.def"), encoding="utf-8").read())
