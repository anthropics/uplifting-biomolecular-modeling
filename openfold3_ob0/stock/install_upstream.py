#!/usr/bin/env python3
"""Install stock `openfold3` at its pin — the step of `run.sh install` that makes upstream present, whichever way this tree arrived.
usage: python -I stock/install_upstream.py [--wheel FILE] [--src TARBALL] [--wheel-only]
  1. the pinned wheel `stock/openfold3-0.5.0-py3-none-any.whl` (stock/PINS.json "wheel"): present, or fetched from PyPI (`pip download
     openfold3==0.5.0`), or taken from `--wheel FILE` (a copy fetched beforehand, for a machine without network access) — each judged against
     the wheel's sha256 and size in stock/PINS.json before it is placed; any other digest is refused by name and nothing is written;
  2. the distribution: when `openfold3` is not installed in this interpreter's environment, `pip install --no-deps` of that wheel (the rest
     of the stack comes from environment/requirements.lock; check_pins.py then proves the install file for file);
  3. upstream's source tree `stock/src/` (stock/PINS.json "source", "upstream": the tagged tree, `examples/` included): present, or fetched as
     the repository's archive of the pinned commit (`<repo>/archive/<commit>.tar.gz`), or taken from `--src TARBALL` — extracted beside, checked
     (every packaged `openfold3/` file byte-identical to the pinned wheel: check_pins.py --wheel-vs-source), then moved into place.
     `--wheel-only` stops after step 2 (an image's stack layer).
Exit 0 with one `WHEEL OK` / `INSTALLED` / `SOURCE OK` line per step on stderr; exit 1 on a failed fetch or a refused file (the line names
the remedy). Standard library plus `pip`; no torch, no openfold3 import. check_pins.py (this directory) is the one reader of the pin table
and the one hasher; this file only fetches, judges and places.
"""
import argparse
import importlib.metadata as md
import importlib.util
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location("check_pins", os.path.join(HERE, "check_pins.py"))   # THIS directory's check_pins.py (read_pins, sha256_file, check_wheel_vs_source, PREFIX), bound by path
CP = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(CP)

PREFIX = f"{CP.PREFIX[:-1]} install]"                                   # "[openfold3_ob0-opt install]" — the install step's lines (openfold3_ob0_opt.weights prints the same prefix)
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def wheel_path(pins):
    return os.path.join(HERE, os.path.basename(pins["wheel"]["file"]))


def judge_wheel(path, pins):
    """(is_pinned, words): a file against stock/PINS.json wheel sha256 + bytes."""
    want, size = pins["wheel"]["sha256"], pins["wheel"]["bytes"]
    got, n = CP.sha256_file(path), os.path.getsize(path)
    return (got == want and n == size), f"sha256 {got[:12]}…, {n} bytes"


def pip_download(requirement, dest):
    """The PyPI route: `pip download --no-deps --only-binary :all: <requirement>` into `dest`; returns the one wheel it wrote."""
    r = subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary", ":all:", "--no-cache-dir", "-q", "-d", dest, requirement],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pip download {requirement} exited {r.returncode}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")
    got = [f for f in os.listdir(dest) if f.endswith(".whl")]
    if len(got) != 1:
        raise RuntimeError(f"pip download {requirement} wrote {sorted(os.listdir(dest))}, not one wheel")
    return os.path.join(dest, got[0])


def wheel(pins, source=None, download=pip_download, out=sys.stderr):
    """Step 1: the pinned wheel into stock/. Returns an exit code."""
    path, rel = wheel_path(pins), pins["wheel"]["file"]
    pinned = f"sha256 {pins['wheel']['sha256'][:12]}…, {pins['wheel']['bytes']} bytes; stock/PINS.json wheel"
    if os.path.isfile(path):
        ok, seen = judge_wheel(path, pins)
        if ok:
            print(f"{PREFIX} WHEEL OK: {rel} is the pinned wheel ({seen}) — present", file=out); return EXIT_OK
        print(f"{PREFIX} REFUSED: {rel} is present but is not the pinned wheel ({seen}; pinned {pinned}) — the file is left in place; remove it and re-run", file=out)
        return EXIT_FAIL
    tmp = tempfile.mkdtemp(prefix=".wheel-", dir=HERE)
    try:
        if source:
            if not os.path.isfile(source):
                print(f"{PREFIX} REFUSED: --wheel {source} is not a file — nothing placed", file=out); return EXIT_FAIL
            cand, how = source, f"placed from --wheel {source} (no fetch)"
        else:
            req = pins["wheel"]["pypi"]
            print(f"{PREFIX} fetching {rel} from PyPI ({req}) …", file=out); out.flush()
            try:
                cand = download(req, tmp)
            except Exception as e:                                      # noqa: BLE001 — any fetch failure is relayed by name with the offline remedy
                print(f"{PREFIX} FAILED: the PyPI fetch of {req} raised {type(e).__name__}: {e} — on a machine without network access fetch the wheel "
                      f"beforehand (PyPI {req}, {pinned}) and pass it: run.sh install --wheel FILE", file=out)
                return EXIT_FAIL
            how = f"fetched from PyPI ({req})"
        ok, seen = judge_wheel(cand, pins)
        if not ok:
            print(f"{PREFIX} REFUSED: {os.path.basename(cand)} ({how.split(' (')[0]}) is {seen}, not the pinned wheel ({pinned}) — nothing placed", file=out)
            return EXIT_FAIL
        staged = os.path.join(tmp, os.path.basename(path) + ".staged")
        shutil.copyfile(cand, staged); os.replace(staged, path)         # placed whole or not at all
        print(f"{PREFIX} WHEEL OK: {rel} is the pinned wheel ({seen}) — {how}", file=out)
        return EXIT_OK
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def pip_install(path):
    r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-deps", "--no-cache-dir", "-q", path], capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"pip install --no-deps {os.path.basename(path)} exited {r.returncode}: {(r.stderr or r.stdout).strip().splitlines()[-1:]}")


def installed_version():
    try:
        return md.distribution("openfold3").version
    except md.PackageNotFoundError:
        return None


def distribution(pins, install=pip_install, version=installed_version, out=sys.stderr):
    """Step 2: `openfold3` installed in this environment — from the pinned wheel when absent. An install that is present is left as it is:
    check_pins.py judges it file for file right after (a different version or an edited file is refused there by name)."""
    have = version()
    if have is not None:
        print(f"{PREFIX} INSTALLED: openfold3 {have} is installed in this environment — left as is (the pin check follows)", file=out); return EXIT_OK
    path = wheel_path(pins)
    print(f"{PREFIX} installing {pins['wheel']['file']} into this environment (pip install --no-deps) …", file=out); out.flush()
    try:
        install(path)
    except Exception as e:                                              # noqa: BLE001
        print(f"{PREFIX} FAILED: {e} — the environment must have pip and be writable (README.md 'Install')", file=out); return EXIT_FAIL
    have = version()
    if have != pins["openfold3_version"]:
        print(f"{PREFIX} FAILED: pip returned but openfold3 {have!r} is what this interpreter now finds (pinned {pins['openfold3_version']})", file=out); return EXIT_FAIL
    print(f"{PREFIX} INSTALLED: openfold3 {have} from {pins['wheel']['file']}", file=out)
    return EXIT_OK


def archive_url(pins):
    """The repository's archive of the pinned commit (stock/PINS.json upstream repo + commit)."""
    return f"{pins['upstream']['repo'].rstrip('/')}/archive/{pins['upstream']['commit']}.tar.gz"


def url_download(url, dest):
    path = os.path.join(dest, "src.tar.gz")
    with urllib.request.urlopen(url, timeout=120) as r, open(path, "wb") as fh:
        shutil.copyfileobj(r, fh, 1 << 20)
    return path


def extract_tree(tarball, dest):
    """A repository archive (one top directory) extracted so that `dest` IS that top directory. Refuses members that would land outside it."""
    with tarfile.open(tarball, "r:*") as t:
        members = t.getmembers()
        tops = {m.name.split("/", 1)[0] for m in members if m.name not in ("", ".", "pax_global_header")}
        if len(tops) != 1:
            raise RuntimeError(f"{os.path.basename(tarball)} has {len(tops)} top-level entries ({sorted(tops)[:3]}), not one source tree")
        top = tops.pop()
        root = os.path.realpath(os.path.join(dest, ".."))
        for m in members:
            if m.name == "pax_global_header": continue
            target = os.path.realpath(os.path.join(root, m.name))
            if not (target == os.path.realpath(os.path.join(root, top)) or target.startswith(os.path.realpath(os.path.join(root, top)) + os.sep)):
                raise RuntimeError(f"{os.path.basename(tarball)} member {m.name!r} would extract outside the tree")
        kw = {"filter": "data"} if hasattr(tarfile, "data_filter") else {}
        t.extractall(root, members=[m for m in members if m.name != "pax_global_header"], **kw)
    os.replace(os.path.join(root, top), dest)


def count_files(d):
    return sum(len(f) for _, _, f in os.walk(d))


def source(pins, tarball=None, download=url_download, out=sys.stderr):
    """Step 3: upstream's source tree into stock/src, checked against the pinned wheel. Returns an exit code."""
    src, rel = os.path.join(HERE, "src"), pins["source"]["dir"]
    tag, commit = pins["upstream"]["tag"], pins["upstream"]["commit"]
    def checked():
        """(is_the_tagged_tree, words) of stock/src as it stands: check_pins.py --wheel-vs-source plus the file count against stock/PINS.json source.files."""
        bad, d = CP.check_wheel_vs_source(pins)
        n = count_files(src)
        files = f"{n} files" + ("" if n == pins["source"].get("files") else f" (stock/PINS.json source.files says {pins['source'].get('files')})")
        for b in bad: print(f"{PREFIX} NOT STOCK: {b}", file=out)
        return (not bad), f"{d['identical']}/{d['files']} packaged files byte-identical to the pinned wheel, {files}"
    if os.path.isdir(src):
        ok, seen = checked()
        if ok:
            print(f"{PREFIX} SOURCE OK: {rel} is upstream's source tree at tag {tag} (commit {commit[:8]}; {seen}) — present", file=out); return EXIT_OK
        print(f"{PREFIX} REFUSED: {rel} is present but is not the tagged tree ({seen}) — the directory is left in place; remove it and re-run", file=out)
        return EXIT_FAIL
    tmp = tempfile.mkdtemp(prefix=".src-", dir=HERE)
    try:
        if tarball:
            if not os.path.isfile(tarball):
                print(f"{PREFIX} REFUSED: --src {tarball} is not a file — nothing placed", file=out); return EXIT_FAIL
            cand, how = tarball, f"extracted from --src {tarball} (no fetch)"
        else:
            url = archive_url(pins)
            print(f"{PREFIX} fetching {rel} from {url} …", file=out); out.flush()
            try:
                cand = download(url, tmp)
            except Exception as e:                                      # noqa: BLE001
                print(f"{PREFIX} FAILED: the fetch of {url} raised {type(e).__name__}: {e} — on a machine without network access fetch that archive "
                      f"beforehand and pass it: run.sh install --src TARBALL", file=out)
                return EXIT_FAIL
            how = f"fetched from {url}"
        staged = os.path.join(tmp, "src")
        try:
            extract_tree(cand, staged)
        except Exception as e:                                          # noqa: BLE001
            print(f"{PREFIX} FAILED: extracting {os.path.basename(cand)} raised {type(e).__name__}: {e} — nothing placed", file=out); return EXIT_FAIL
        os.replace(staged, src)                                         # the tree appears whole; judged in place, moved back out when it is not the tagged tree
        ok, seen = checked()
        if not ok:
            os.replace(src, staged)
            print(f"{PREFIX} REFUSED: the tree {how.split(' ', 1)[0]} is not upstream's tree at tag {tag} ({seen}) — nothing placed", file=out); return EXIT_FAIL
        print(f"{PREFIX} SOURCE OK: {rel} is upstream's source tree at tag {tag} (commit {commit[:8]}; {seen}) — {how}", file=out)
        return EXIT_OK
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main(argv=None, out=sys.stderr):
    ap = argparse.ArgumentParser(prog="install_upstream.py", description=__doc__.splitlines()[0])
    ap.add_argument("--wheel", default=None, metavar="FILE", help="a copy of the pinned PyPI wheel fetched beforehand (no network fetch)")
    ap.add_argument("--src", default=None, metavar="TARBALL", help="a copy of the repository archive of the pinned commit fetched beforehand (no network fetch)")
    ap.add_argument("--wheel-only", action="store_true", help="steps 1-2 only: the wheel and the distribution, not the source tree")
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return EXIT_USAGE if e.code else EXIT_OK
    pins = CP.read_pins()
    for step in (lambda: wheel(pins, source=a.wheel, out=out), lambda: distribution(pins, out=out),
                 None if a.wheel_only else (lambda: source(pins, tarball=a.src, out=out))):
        if step is None: continue
        rc = step()
        if rc != EXIT_OK: return rc
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
