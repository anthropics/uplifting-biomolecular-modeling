#!/usr/bin/env python3
"""check_pins.py — is the fair-esm this interpreter imports the pinned one?  Standard library only: `python -I stock/check_pins.py` (`run.sh install` runs it).

Reads `PINS.json` beside this file: `upstream."fair-esm"` {version, commit, archive}. The installed distribution `fair-esm` must be at that version
(PyPI's `fair-esm` 2.0.0 is not this pin) and every `esm/**.py` file of the carried archive (`git archive` of the pinned commit) must be the file on
disk under the distribution's location with the same sha256 — so a checkout of another commit, or an edited copy, is refused by name. The weights
file is not this script's to check (`run.sh install --weights DIR` and `run.sh check` do that).
Prints one line, `fair-esm <version> (<commit8>): pinned — <N> files match stock/<archive> at <location>`; exit 0 pinned · 3 not pinned or not
installed (each refusal on stderr as `check_pins: …`) · 2 usage.
"""
import hashlib
import json
import os
import sys
import tarfile
from importlib import metadata

HERE = os.path.dirname(os.path.abspath(__file__))          # esm_if1/stock/
TREE = os.path.dirname(HERE)                                # esm_if1/
DIST = "fair-esm"


def read_pins():
    with open(os.path.join(HERE, "PINS.json"), encoding="utf-8") as fh:
        return json.load(fh)


def archive_files(archive_path):
    """{member: sha256} for every `esm/**.py` member of the carried archive — the files the pinned install puts under site-packages/esm/."""
    out = {}
    with tarfile.open(archive_path, "r:gz") as tar:
        for m in tar.getmembers():
            if m.isfile() and m.name.startswith("esm/") and m.name.endswith(".py"):
                out[m.name] = hashlib.sha256(tar.extractfile(m).read()).hexdigest()
    return out


def _sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(pin, archive_path, dist):
    """The refusals (an empty list = pinned) and the detail {version, location, files_checked, files_absent, files_differ} for one installed distribution."""
    bad = []
    base = str(dist.locate_file(""))
    detail = {"version": dist.version, "location": base, "files_checked": 0, "files_absent": [], "files_differ": []}
    if dist.version != pin["version"]:
        bad.append(f"{DIST} {dist.version} is installed at {base}; want {DIST}=={pin['version']} (commit {pin['commit'][:8]}: pip install --no-deps {pin['archive']})")
    if not os.path.isfile(archive_path):
        bad.append(f"the pinned archive is not on disk: {archive_path} (PINS.json upstream.\"{DIST}\".archive)")
        return bad, detail
    want = archive_files(archive_path)
    if not want:
        bad.append(f"the pinned archive {archive_path} lists no esm/*.py files")
    for rel, digest in sorted(want.items()):
        detail["files_checked"] += 1
        p = os.path.join(base, rel)
        if not os.path.isfile(p):
            detail["files_absent"].append(rel)
        elif _sha(p) != digest:
            detail["files_differ"].append(rel)
    if detail["files_absent"]:
        bad.append(f"{len(detail['files_absent'])} file(s) of {pin['archive']} are not under {base} ({detail['files_absent'][:3]}…): not the pinned install")
    if detail["files_differ"]:
        bad.append(f"{len(detail['files_differ'])} file(s) under {base} differ from {pin['archive']} ({detail['files_differ'][:3]}…): not commit {pin['commit'][:8]}")
    return bad, detail


def main(argv):
    if argv:
        print("usage: python -I stock/check_pins.py", file=sys.stderr)
        return 2
    pin = read_pins()["upstream"][DIST]
    archive_path = os.path.join(TREE, pin["archive"])
    try:
        dist = metadata.distribution(DIST)
    except metadata.PackageNotFoundError:
        print(f"check_pins: {DIST} is not installed on {sys.executable}; want {DIST}=={pin['version']} (pip install --no-deps {pin['archive']}; STOCK.md)", file=sys.stderr)
        return 3
    bad, d = check(pin, archive_path, dist)
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        return 3
    print(f"{DIST} {d['version']} ({pin['commit'][:8]}): pinned — {d['files_checked']} files match {pin['archive']} at {d['location']}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
