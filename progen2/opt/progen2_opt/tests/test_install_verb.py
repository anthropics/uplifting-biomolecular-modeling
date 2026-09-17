"""`run.sh install [--weights DIR [--model M]]` — the verb's argument handling and call sequence (a stub interpreter named by PROGEN2_PYTHON
records every invocation; nothing is installed), and the weights step (progen2_opt.weights.fetch with an injected transfer and checker: the
archive unpacked to upstream's layout, a bad archive digest or a failed pin check refused by name, present files kept; no network). CPU only."""
import hashlib
import io
import json
import os
import re
import stat
import subprocess
import tarfile

import pytest

from .conftest import TREE

RUN_SH = os.path.join(TREE, "run.sh")

STUB = """#!/bin/bash
# stub interpreter: one line per invocation in $STUB_LOG; exit codes chosen per call kind by STUB_RC_PIP / STUB_RC_PINS / STUB_RC_WEIGHTS / STUB_RC_PROBE
printf '%s\\n' "$*" >> "$STUB_LOG"
case "$*" in
  *"-I -c import os,sys"*) exit "${STUB_RC_PROBE:-1}" ;;
  *"-m pip install"*) exit "${STUB_RC_PIP:-0}" ;;
  *"check_pins.py"*) exit "${STUB_RC_PINS:-0}" ;;
  *"-m progen2_opt.weights"*) exit "${STUB_RC_WEIGHTS:-0}" ;;
  *) exit 0 ;;
esac
"""


@pytest.fixture
def stub(tmp_path):
    """A stub `python`: reachable both as PROGEN2_PYTHON=<path> and as `python` on PATH; returns (run, log) — run(args, via, **rc) -> (rc, out, calls, probes)."""
    bindir = tmp_path / "bin"; bindir.mkdir()
    exe = bindir / "python"; exe.write_text(STUB); exe.chmod(exe.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "calls.log"

    def run(args, via="env", **rc):
        env = {"HOME": str(tmp_path), "STUB_LOG": str(log)}
        if via == "env":                                                   # PROGEN2_PYTHON names the interpreter (README 'Install')
            env["PATH"] = "/usr/bin:/bin"; env["PROGEN2_PYTHON"] = str(exe)
        else:                                                              # default: `python` on PATH
            env["PATH"] = str(bindir) + os.pathsep + "/usr/bin:/bin"
        env.update({f"STUB_RC_{k.upper()}": str(v) for k, v in rc.items()})
        if log.exists(): log.unlink()
        r = subprocess.run(["bash", RUN_SH] + args, capture_output=True, text=True, env=env, cwd=str(tmp_path))
        lines = log.read_text().splitlines() if log.exists() else []
        probes = [c for c in lines if c.startswith("-I -c import os,sys")]
        return r.returncode, r.stdout + r.stderr, [c for c in lines if not c.startswith("-I -c import os,sys")], probes
    return run


@pytest.mark.parametrize("via", ["env", "path"])
def test_install_runs_pip_then_the_pin_check(stub, via):
    rc, out, calls, probes = stub(["install"], via)
    assert rc == 0, out
    assert calls == [f"-m pip install -e {TREE}/opt", f"-I {TREE}/stock/check_pins.py"], calls
    assert len(probes) == 1, probes


def test_weights_dir_adds_the_weights_step_last(stub):
    for args, tail in ((["install", "--weights", "/data/w"], "-m progen2_opt.weights /data/w"),
                       (["install", "--weights=/data/w"], "-m progen2_opt.weights /data/w"),
                       (["install", "--weights", "/data/w", "--model", "progen2-xlarge"], "-m progen2_opt.weights /data/w --model progen2-xlarge"),
                       (["install", "--model=progen2-small", "--weights", "/data/w"], "-m progen2_opt.weights /data/w --model progen2-small")):
        rc, out, calls, _ = stub(args)
        assert rc == 0, (args, out)
        assert len(calls) == 3 and calls[2] == tail, (args, calls)


def test_usage_errors_call_nothing(stub):
    for args in (["install", "--weights"], ["install", "--weights", "--model"], ["install", "--weights="], ["install", "--bogus"], ["install", "extra"],
                 ["install", "--mode", "exact"], ["install", "--config", "h100"], ["install", "--model", "progen2-xlarge"],
                 ["install", "--weights", "/data/w", "--variant", "xlarge"]):
        rc, out, calls, probes = stub(args)
        assert rc == 2, (args, out)
        assert calls == [] and probes == [], (args, calls, probes)
        assert "run.sh:" in out


def test_a_failed_step_stops_the_sequence_with_its_code(stub):
    assert stub(["install", "--weights", "/w"], pip=1)[0:1] + (len(stub(["install", "--weights", "/w"], pip=1)[2]),) == (1, 1)
    rc, out, calls, _ = stub(["install", "--weights", "/w"], pins=3)
    assert (rc, len(calls)) == (3, 2), (out, calls)
    rc, out, calls, _ = stub(["install", "--weights", "/w"], weights=1)
    assert (rc, len(calls)) == (1, 3), (out, calls)


def test_a_missing_interpreter_is_not_active_by_name(tmp_path):
    env = {"HOME": str(tmp_path), "PATH": "/usr/bin:/bin", "PROGEN2_PYTHON": str(tmp_path / "no" / "such" / "python")}
    r = subprocess.run(["bash", RUN_SH, "install"], capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 3 and "no such interpreter" in r.stderr, r.stderr


def test_a_tree_already_installed_skips_pip_by_name(stub):
    """The container image ships the kit installed (editable, from /kit): `install` there names the skip and goes on to the pin check (and
    --weights) — a read-only image cannot re-run pip. The stub answers the find_spec probe with STUB_RC_PROBE."""
    rc, out, calls, probes = stub(["install", "--weights", "/w"], probe=0)
    assert rc == 0, out
    assert "installed from this tree already" in out
    assert calls == [f"-I {TREE}/stock/check_pins.py", "-m progen2_opt.weights /w"], calls
    assert len(probes) == 1, probes


# ---- the weights step: progen2_opt.weights.fetch with an injected transfer (no network) and checker (no 30 GB) ----

def _archive_bytes(members: dict) -> bytes:
    """A .tar.gz holding `members` {name: bytes} the way upstream's archives do (./config.json, ./pytorch_model.bin)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, data in members.items():
            ti = tarfile.TarInfo("./" + name); ti.size = len(data); tf.addfile(ti, io.BytesIO(data))
    return buf.getvalue()


@pytest.fixture
def fake(tmp_path):
    """Two fake sizes with archives, a pin table in stock/PINS.json's shape, a transfer that writes the archive bytes, and a checker that hashes like check_pins."""
    from progen2_opt import weights as W
    payload = {"progen2-small": {"config.json": b'{"n_layer": 12}', "pytorch_model.bin": b"small-weights"},
               "progen2-xlarge": {"config.json": b'{"n_layer": 32}', "pytorch_model.bin": b"xlarge-weights"}}
    archives = {s: _archive_bytes(m) for s, m in payload.items()}
    pins = {"weights": {s: {"url": f"https://upstream.example/checkpoints/{s}.tar.gz",
                            "files": {"pytorch_model.bin": {"sha256": hashlib.sha256(m["pytorch_model.bin"]).hexdigest()},
                                      "config.json": {"sha256": hashlib.sha256(m["config.json"]).hexdigest()} if s == "progen2-small" else {}},
                            "tarball": {"sha256": hashlib.sha256(archives[s]).hexdigest(), "size_bytes": len(archives[s])} if s == "progen2-small" else {}}
                        for s, m in payload.items()}}
    calls = []

    def transfer(url, dest):
        calls.append(url); size = os.path.basename(url)[:-len(".tar.gz")]
        with open(dest, "wb") as fh: fh.write(archives[size])
        return hashlib.sha256(archives[size]).hexdigest()

    def checker(wdir, sizes):                                              # check_pins.check_weights' rule: the bytes against the pins, missing files named
        bad = []
        for s in sizes:
            for f, pin in pins["weights"][s]["files"].items():
                if not pin.get("sha256"): continue
                p = os.path.join(wdir, s, f)
                got = hashlib.sha256(open(p, "rb").read()).hexdigest() if os.path.isfile(p) else "missing"
                if got != pin["sha256"]: bad.append(f"check_pins: {s}: {f} sha256 {got} (bytes) != pin {pin['sha256']}")
        return (3 if bad else 0), "\n".join(bad or [f"weights at {wdir}: pinned"])

    def fetch(sizes=("progen2-small", "progen2-xlarge"), **kw):
        out = io.StringIO()
        rc = W.fetch(str(tmp_path / "w"), list(sizes), pins=pins, out=out, transfer=kw.pop("transfer", transfer), checker=kw.pop("checker", checker))
        return rc, out.getvalue()
    return type("Fake", (), {"dir": tmp_path / "w", "pins": pins, "payload": payload, "archives": archives, "calls": calls, "fetch": staticmethod(fetch),
                             "transfer": staticmethod(transfer), "checker": staticmethod(checker), "W": W})


def test_all_fetched_unpacked_checked_and_summed(fake):
    rc, out = fake.fetch()
    assert rc == 0, out
    assert "WEIGHTS OK: 3/3 pinned files at their pins, 4 files present" in out, out
    assert fake.calls == [fake.pins["weights"][s]["url"] for s in ("progen2-small", "progen2-xlarge")]
    for s, members in fake.payload.items():
        d = fake.dir / s
        assert sorted(os.listdir(d)) == ["SHA256SUMS", "config.json", "pytorch_model.bin"], os.listdir(d)   # upstream's layout; the archive removed
        for name, data in members.items(): assert (d / name).read_bytes() == data
        sums = (d / "SHA256SUMS").read_text().splitlines()
        assert sums == [f"{p['sha256']}  {n}" for n, p in fake.pins["weights"][s]["files"].items() if p.get("sha256")], sums


def test_a_present_size_is_kept_and_only_checked(fake):
    d = fake.dir / "progen2-small"; d.mkdir(parents=True)
    for name, data in fake.payload["progen2-small"].items(): (d / name).write_bytes(data)
    rc, out = fake.fetch()
    assert rc == 0, out
    assert "progen2-small: pytorch_model.bin, config.json present — kept" in out, out
    assert fake.calls == [fake.pins["weights"]["progen2-xlarge"]["url"]], fake.calls          # nothing fetched for the present size


def test_an_archive_off_its_pin_is_refused_and_kept(fake):
    def tampered(url, dest):
        with open(dest, "wb") as fh: fh.write(b"not the archive")
        return hashlib.sha256(b"not the archive").hexdigest()
    rc, out = fake.fetch(sizes=("progen2-small",), transfer=tampered)
    assert rc == 1, out
    assert "REFUSED: 1 of 1" in out and "is not the pin's" in out, out
    assert (fake.dir / "progen2-small" / "progen2-small.tar.gz").is_file()                    # left in place for inspection, nothing unpacked
    assert not (fake.dir / "progen2-small" / "pytorch_model.bin").exists()


def test_a_transfer_error_is_relayed_by_name(fake):
    def broken(url, dest): raise ConnectionError("network unreachable")
    rc, out = fake.fetch(transfer=broken)
    assert rc == 1, out
    assert "FAILED progen2-small: transfer of https://upstream.example/checkpoints/progen2-small.tar.gz failed: ConnectionError: network unreachable" in out, out
    assert "REFUSED: 2 of 2" in out


def test_a_file_off_its_pin_fails_the_check_and_stays(fake):
    d = fake.dir / "progen2-xlarge"; d.mkdir(parents=True)
    (d / "pytorch_model.bin").write_bytes(b"other bytes"); (d / "config.json").write_bytes(b"{}")
    rc, out = fake.fetch(sizes=("progen2-xlarge",))
    assert rc == 1, out
    assert "check_pins: progen2-xlarge: pytorch_model.bin sha256" in out and "REFUSED: stock/check_pins.py exit 3" in out, out
    assert (d / "pytorch_model.bin").read_bytes() == b"other bytes" and not (d / "SHA256SUMS").exists()


def test_sizes_for_and_the_urls_are_upstreams(tree):
    """sizes_for: one variant -> its upstream name, none -> all seven in the stock order; stock/PINS.json's URL per size is the one upstream's
    README fetches (`https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz`, read from the pinned README as text)."""
    from progen2_opt import modes, stack, weights as W
    assert W.sizes_for("bfd90") == ["progen2-BFD90"] and W.sizes_for(None) == [modes.UPSTREAM_NAME[v] for v in modes.VARIANTS]
    readme = open(os.path.join(tree, "stock", "src", "progen2", "README.md"), encoding="utf-8").read()
    pattern = re.search(r"(https://\S+/checkpoints/)\$\{model\}\.tar\.gz", readme)
    assert pattern, "upstream README's wget line"
    pins = stack.pins()
    for v in modes.VARIANTS:
        name = modes.UPSTREAM_NAME[v]
        assert pins["weights"][name]["url"] == f"{pattern.group(1)}{name}.tar.gz", name
        assert pins["weights"][name]["files"]["pytorch_model.bin"].get("sha256"), name       # every size's checkpoint is pinned
        w = pins["weights"][name]                                                              # and so are its config.json and the archive they come in: the archive is
        for entry in (w["files"]["pytorch_model.bin"], w["files"]["config.json"], w["tarball"]):   # compared with its pin before anything is unpacked (weights.fetch)
            assert re.fullmatch(r"[0-9a-f]{64}", entry.get("sha256") or ""), (name, entry)
            assert isinstance(entry.get("size_bytes"), int) and entry["size_bytes"] > 0, (name, entry)
        assert w["tarball"]["members"] == ["./config.json", "./pytorch_model.bin"], name


def test_pins_check_is_the_kits_checker_on_the_weights_alone(monkeypatch, tmp_path):
    """pins_check runs stock/check_pins.py under stack.python() with the software checks skipped, the bytes hashed and the sizes named."""
    from progen2_opt import stack, weights as W
    seen = {}

    class R: returncode, stdout, stderr = 0, "weights at /w: pinned\n", ""
    monkeypatch.setattr(W.subprocess, "run", lambda cmd, **kw: seen.setdefault("cmd", cmd) and R())
    monkeypatch.setenv("PROGEN2_PYTHON", "/opt/venv/bin/python")
    rc, text = W.pins_check("/w", ["progen2-small", "progen2-xlarge"])
    assert (rc, text) == (0, "weights at /w: pinned\n")
    assert seen["cmd"] == ["/opt/venv/bin/python", "-I", os.path.join(stack.tree_home(), "stock", "check_pins.py"), "--skip", "interpreter", "--skip", "packages",
                           "--skip", "stock-files", "--weights-dir", "/w", "--hash-weights", "--size", "progen2-small", "--size", "progen2-xlarge"], seen["cmd"]


def test_main_model_resolution(monkeypatch, tmp_path):
    from progen2_opt import weights as W
    got = []
    monkeypatch.setattr(W, "fetch", lambda d, sizes, **kw: got.append((d, list(sizes))) or 0)
    assert W.main([str(tmp_path), "--model", "progen2-xlarge"]) == 0 and got[-1][1] == ["progen2-xlarge"]
    assert W.main([str(tmp_path), "--model", "progen2-bfd90"]) == 0 and got[-1][1] == ["progen2-BFD90"]        # the lowercase alias
    monkeypatch.setenv("PROGEN2_VARIANT", "small")                                                              # no environment variable names a size: all seven
    assert W.main([str(tmp_path)]) == 0 and len(got[-1][1]) == 7
    assert W.main([str(tmp_path), "--model", "huge"]) == 2
    with pytest.raises(SystemExit):
        W.main([str(tmp_path), "--variant", "xlarge"])                                                          # the retired flag is unknown to argparse
