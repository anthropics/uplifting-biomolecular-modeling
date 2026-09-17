#!/usr/bin/env python3
"""Refuse unless the pinned upstream is installed: opendde at the pinned version with the pinned wheel's bytes (stock/PINS.json "wheel").

usage: python -I stock/check_pins.py [--quiet] [--json]      exit 0 = the installed opendde is the pinned wheel's bytes; 3 = not (one line on stderr)

The pin is judged on bytes, not on the install route: every package file the wheel ships (`opendde/`, `runner/`; its RECORD) is hashed
where the installed distribution locates it and compared with the wheel's own RECORD hash. An install from the wheel, from PyPI, or from
a source directory at the pinned commit (whose files are the wheel's bytes) all pass;
another version, a patched file, a missing file or an extra `.py` under the two packages is refused. Standard library only (zipfile for
the wheel's RECORD, no torch import), so run.sh, the configs and the package can all call it; this file is the one place the pin check lives.
"""
import base64
import hashlib
import importlib.metadata as md
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))


def pins():
    with open(os.path.join(HERE, "PINS.json")) as fh:
        return json.load(fh)


def wheel_record(path):
    """{package file path: sha256 hex} from the wheel's RECORD (dist-info entries excluded)."""
    with zipfile.ZipFile(path) as z:
        rec = [n for n in z.namelist() if n.endswith(".dist-info/RECORD")][0]
        out = {}
        for line in z.read(rec).decode().splitlines():
            if not line.strip():
                continue
            name, digest, _size = line.rsplit(",", 2)
            if name.endswith(".dist-info/RECORD") or ".dist-info/" in name or not digest.startswith("sha256="):
                continue
            out[name] = base64.urlsafe_b64decode(digest[len("sha256="):] + "==").hex()
        return out


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check(quiet=False):
    p = pins()
    want_v = p["upstream"]["version"]
    wheel = os.path.join(HERE, os.path.basename(p["wheel"]["file"]))
    if not os.path.isfile(wheel):
        return 3, f"{wheel} is missing from this tree (stock/PINS.json \"wheel\")", {}
    try:
        dist = md.distribution("opendde")
    except md.PackageNotFoundError:
        return 3, f"opendde is not installed (want {want_v} from {p['wheel']['file']})", {}
    have_v = dist.version
    detail = {"version": have_v, "pinned_version": want_v, "route": None, "files": 0, "mismatch": [], "missing": []}
    du = dist.read_text("direct_url.json")
    if du:
        d = json.loads(du)
        detail["route"] = d.get("url") + (" (directory)" if "dir_info" in d else " (vcs)" if "vcs_info" in d else " (archive)")
    else:
        detail["route"] = "index (no direct_url.json)"
    if have_v != want_v:
        return 3, f"opendde {have_v} is installed; the pin is {want_v} ({p['wheel']['file']})", detail
    want = wheel_record(wheel)
    detail["files"] = len(want)
    for name, sha in want.items():
        loc = dist.locate_file(name)
        if not os.path.isfile(loc):
            detail["missing"].append(name)
        elif sha256_file(loc) != sha:
            detail["mismatch"].append(name)
    # extra .py files under the two packages (a patched tree would carry them)
    extra = []
    for pkg in ("opendde", "runner"):
        root = dist.locate_file(pkg)
        if os.path.isdir(root):
            for dp, dn, fn in os.walk(root):
                dn[:] = [x for x in dn if x != "__pycache__"]
                for f in fn:
                    if f.endswith(".py"):
                        rel = os.path.relpath(os.path.join(dp, f), os.path.dirname(str(root))).replace(os.sep, "/")
                        if rel not in want:
                            extra.append(rel)
    detail["extra"] = extra
    if detail["missing"] or detail["mismatch"] or extra:
        return 3, (f"opendde {have_v} is installed ({detail['route']}) but its files are not the pinned wheel's bytes: "
                   f"{len(detail['mismatch'])} differ, {len(detail['missing'])} missing, {len(extra)} extra — e.g. "
                   f"{(detail['mismatch'] + detail['missing'] + extra)[:3]}"), detail
    if not quiet:
        print(f"opendde {have_v} == pin {want_v}; {len(want)}/{len(want)} package files are the pinned wheel's bytes; route {detail['route']}")
    return 0, None, detail


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    rc, msg, detail = check(quiet="--quiet" in argv)
    if msg:
        print(f"[opendde-opt stock] {msg}", file=sys.stderr)
    if "--json" in argv:
        print(json.dumps({"rc": rc, "message": msg, **detail}, indent=1))
    return rc


if __name__ == "__main__":
    sys.exit(main())
