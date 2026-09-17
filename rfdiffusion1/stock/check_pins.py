#!/usr/bin/env python3
"""Refuse unless the installed RFdiffusion checkout is the pinned commit, byte for byte, on the pinned stack (stock/PINS.json).

usage: python -I stock/check_pins.py [--quiet] [--no-stack] [--rfd-root DIR]
       exit 0 = the checkout equals the pinned archive and the stack matches; 3 = not (one line per finding on stderr)

Upstream ships no wheel: "installed" means an editable checkout (`pip install --no-deps -e .` at $RFD_ROOT, PINS.json "install"), whose
pip metadata carries no commit. The check therefore reads the checkout itself: every member of the pinned archive
(stock/rfdiffusion-86507b65.tar.gz, tracked in this repo) must be present in the checkout with equal bytes — a checkout at the
pin with a source patch applied (the kit's P01/P02/U01/U02 diffs, or anything else) is refused by name; a git HEAD other than the pin is
reported beside the byte verdict. The stack check compares the installed torch / dgl distribution versions with PINS.json
"pinned_stack" "stack_check" (`--no-stack` skips it: the stock route on another stack is a labelled run, not the stock).
Standard library only (no torch import), so run.sh, the configs and the package all call this one file.
The checkout is found from --rfd-root, else $RFD_ROOT, else the `rfdiffusion` distribution's editable location (PEP 610 direct_url.json).
"""
import hashlib
import importlib.metadata as md
import io
import json
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = "rfdiffusion"


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for part in iter(lambda: fh.read(chunk), b""):
            h.update(part)
    return h.hexdigest()


def editable_location(dist=DIST):
    """The checkout pip recorded for an editable install (direct_url.json url file://<dir>), or None."""
    try:
        d = md.distribution(dist)
    except md.PackageNotFoundError:
        return None, None
    raw = d.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    url = info.get("url") or ""
    root = url[len("file://"):] if url.startswith("file://") else None
    return root, {"version": d.version, "direct_url": info}


def find_root(arg=None):
    """--rfd-root, else $RFD_ROOT, else pip's editable location. Returns (root or None, how)."""
    if arg:
        return os.path.abspath(arg), "--rfd-root"
    env = os.environ.get("RFD_ROOT")
    if env:
        return os.path.abspath(env), "RFD_ROOT"
    root, _ = editable_location()
    return (os.path.abspath(root) if root else None), "pip direct_url.json"


def archive_members(archive):
    """{path relative to the checkout: sha256} for every file member of the pinned archive (the top directory stripped)."""
    out = {}
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf:
            if not m.isfile():
                continue
            rel = m.name.split("/", 1)[1] if "/" in m.name else m.name
            fh = tf.extractfile(m)
            out[rel] = hashlib.sha256(fh.read()).hexdigest()
    return out


def git_head(root):
    try:
        r = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=20)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def check_checkout(pins, root):
    """The checkout against the pinned archive. Returns (bad, detail)."""
    up = pins["upstream"][DIST]
    archive = os.path.join(HERE, os.path.basename(up["archive"]))
    detail = {"root": root, "commit_pinned": up["commit"], "archive": os.path.basename(archive), "git_head": None, "files_checked": 0,
              "files_differ": [], "files_missing": [], "pinned": False}
    bad = []
    if root is None or not os.path.isdir(root):
        bad.append(f"{DIST}: no checkout found (set RFD_ROOT to the editable checkout at {up['repo']} @ {up['commit']}, PINS.json \"install\")")
        return bad, detail
    if not os.path.isfile(archive):
        bad.append(f"{DIST}: the pinned archive {archive} is missing from stock/")
        return bad, detail
    members = archive_members(archive)
    for rel, sha in sorted(members.items()):
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            detail["files_missing"].append(rel)
        elif sha256_file(p) != sha:
            detail["files_differ"].append(rel)
        detail["files_checked"] += 1
    detail["git_head"] = git_head(root)
    detail["pinned"] = not detail["files_differ"] and not detail["files_missing"]
    if detail["files_missing"]:
        bad.append(f"{DIST}: {len(detail['files_missing'])} pinned file(s) missing from {root}: " + ", ".join(detail["files_missing"][:6]))
    if detail["files_differ"]:
        bad.append(f"{DIST}: {len(detail['files_differ'])} file(s) differ from the pinned commit {up['commit'][:8]} in {root} (a patched checkout is not stock): "
                   + ", ".join(detail["files_differ"][:6]))
    if detail["git_head"] and detail["git_head"] != up["commit"]:
        bad.append(f"{DIST}: git HEAD {detail['git_head'][:8]} at {root} is not the pin {up['commit'][:8]}")
    return bad, detail


def base_version(v):
    """A version without its local tag (torch's pip metadata reads 2.4.0 for the 2.4.0+cu121 wheel; dgl's keeps +cu124)."""
    return v.split("+", 1)[0] if isinstance(v, str) else v


def check_stack(pins):
    """Installed torch / dgl distribution versions against pinned_stack.stack_check, compared without local tags. Returns (bad, detail)."""
    want = {k: v for k, v in (pins.get("pinned_stack") or {}).get("stack_check", {}).items() if k != "note"}
    bad, detail = [], {}
    for name, ver in want.items():
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            have = None
        ok = base_version(have) == base_version(ver)
        detail[name] = {"installed": have, "pinned": ver, "ok": ok}
        if not ok:
            bad.append(f"stack: {name} {have} != pinned stack {ver} (PINS.json pinned_stack)")
    return bad, detail


def check(pins, root=None, stack=True):
    """Everything: the checkout (bytes, HEAD) and, with ``stack``, torch / dgl. Returns (bad, detail)."""
    bad, detail = check_checkout(pins, root)
    detail = {"checkout": detail}
    if stack:
        b2, d2 = check_stack(pins)
        bad += b2
        detail["stack"] = d2
    return bad, detail


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet, stack = "--quiet" in argv, "--no-stack" not in argv
    root = None
    if "--rfd-root" in argv:
        root = argv[argv.index("--rfd-root") + 1]
    pins = json.load(open(os.path.join(HERE, "PINS.json"), encoding="utf-8"))
    found, how = find_root(root)
    bad, detail = check(pins, found, stack)
    if not quiet:
        c = detail["checkout"]
        if c["pinned"]:
            print(f"{DIST} {pins['upstream'][DIST]['version']}: pinned (checkout {c['root']} via {how}; {c['files_checked']} files == {c['archive']}; git HEAD {c['git_head'] or 'n/a'})")
        for name, d in (detail.get("stack") or {}).items():
            if d["ok"]:
                print(f"{name} {d['installed']}: pinned stack")
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
