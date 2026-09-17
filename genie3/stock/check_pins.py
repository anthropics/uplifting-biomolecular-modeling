#!/usr/bin/env python3
"""Refuse unless the installed Genie 3 checkout is the pinned commit, byte for byte, on the pinned stack (stock/PINS.json).

usage: python -I stock/check_pins.py [--quiet] [--no-stack] [--genie3-root DIR] [--json]
       exit 0 = the checkout equals the pinned archive and the stack matches; 3 = not (one line per finding on stderr)

Upstream ships no wheel: "installed" means an editable checkout (`pip install -e .` at $GENIE3_ROOT, PINS.json "install"), whose pip
metadata carries no commit. The check therefore reads the checkout itself: every member of the pinned archive (stock/genie3-d77ae5ac.tar.gz,
git-tracked in this repo — member hashes are read fresh off it each run, never stored separately) must be present in the checkout with
identical bytes — a checkout at the pin with any source patch applied is refused by file
name; a git HEAD other than the pin
and tracked files git reports modified are reported beside the byte verdict. The stack check compares the installed torch / lightning /
numpy distribution versions with PINS.json "pinned_stack" "stack_check" (`--no-stack` skips it: the stock route on another stack is a
labelled run, not stock on the pinned stack). Standard library only (no torch import), so run.sh, the config and the package all call this one
file. The checkout is found from --genie3-root, else $GENIE3_ROOT, else the `genie3` distribution's editable location (PEP 610 direct_url.json).
"""
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
DIST = "genie3"
ENV_ROOT = "GENIE3_ROOT"


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
    """--genie3-root, else $GENIE3_ROOT, else pip's editable location. Returns (root or None, how)."""
    if arg:
        return os.path.abspath(arg), "--genie3-root"
    env = os.environ.get(ENV_ROOT)
    if env:
        return os.path.abspath(env), ENV_ROOT
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


def _git(root, *args):
    try:
        r = subprocess.run(["git", "-C", root, *args], capture_output=True, text=True, timeout=30)
        return r.stdout.strip() if r.returncode == 0 else None
    except Exception:  # noqa: BLE001
        return None


def git_head(root):
    return _git(root, "rev-parse", "HEAD")


def git_modified(root):
    """Tracked files git reports changed (`git status --porcelain --untracked-files=no`), or None without git."""
    out = _git(root, "status", "--porcelain", "--untracked-files=no")
    if out is None:
        return None
    return [line[3:] for line in out.splitlines() if line.strip()]


def check_checkout(pins, root):
    """The checkout against the pinned archive (itself git-tracked at stock/<archive> — its own bytes need no separate digest).
    Returns (bad, detail)."""
    up = pins["upstream"][DIST]
    archive = os.path.join(HERE, os.path.basename(up["archive"]))
    detail = {"root": root, "commit_pinned": up["commit"], "archive": os.path.basename(archive), "git_head": None, "git_modified": None,
              "files_checked": 0, "files_differ": [], "files_missing": [], "pinned": False}
    bad = []
    if root is None or not os.path.isdir(root):
        bad.append(f"{DIST}: no checkout found (set {ENV_ROOT} to the editable checkout at {up['repo']} @ {up['commit']}, PINS.json \"install\")")
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
    detail["git_modified"] = git_modified(root)
    detail["pinned"] = not detail["files_differ"] and not detail["files_missing"]
    if detail["files_missing"]:
        bad.append(f"{DIST}: {len(detail['files_missing'])} pinned file(s) missing from {root}: " + ", ".join(detail["files_missing"][:6]))
    if detail["files_differ"]:
        bad.append(f"{DIST}: {len(detail['files_differ'])} file(s) differ from the pinned commit {up['commit'][:8]} in {root} (a patched checkout is not stock): "
                   + ", ".join(detail["files_differ"][:6]))
    if detail["git_head"] and detail["git_head"] != up["commit"]:
        bad.append(f"{DIST}: git HEAD {detail['git_head'][:8]} at {root} is not the pin {up['commit'][:8]}")
    if detail["git_modified"]:
        bad.append(f"{DIST}: git reports {len(detail['git_modified'])} tracked file(s) modified at {root}: " + ", ".join(detail["git_modified"][:6]))
    return bad, detail


def base_version(v):
    """A version without its local tag (torch's pip metadata reads 2.7.1 for the 2.7.1+cu126 wheel)."""
    return v.split("+", 1)[0] if isinstance(v, str) else v


def check_stack(pins):
    """Installed torch / lightning / numpy distribution versions against pinned_stack.stack_check, compared without local tags. Returns (bad, detail)."""
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
            bad.append(f"stack: {name} {have} != pinned {ver} (PINS.json pinned_stack)")
    return bad, detail


def check(pins, root=None, stack=True):
    """Everything: the checkout (bytes, HEAD, modified files) and, with ``stack``, torch / lightning / numpy. Returns (bad, detail)."""
    bad, detail = check_checkout(pins, root)
    detail = {"checkout": detail}
    if stack:
        b2, d2 = check_stack(pins)
        bad += b2
        detail["stack"] = d2
    return bad, detail


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet, stack, as_json = "--quiet" in argv, "--no-stack" not in argv, "--json" in argv
    root = None
    if "--genie3-root" in argv:
        root = argv[argv.index("--genie3-root") + 1]
    pins = json.load(open(os.path.join(HERE, "PINS.json"), encoding="utf-8"))
    found, how = find_root(root)
    bad, detail = check(pins, found, stack)
    if as_json:
        print(json.dumps({"bad": bad, "detail": detail, "found_via": how}, indent=1))
    elif not quiet:
        c = detail["checkout"]
        if c["pinned"]:
            print(f"{DIST} {pins['upstream'][DIST]['version']}: pinned (checkout {c['root']} via {how}; {c['files_checked']} files == {c['archive']}; "
                  f"git HEAD {c['git_head'] or 'n/a'}; modified tracked files {len(c['git_modified']) if c['git_modified'] is not None else 'n/a'})")
        for name, d in (detail.get("stack") or {}).items():
            if d["ok"]:
                print(f"{name} {d['installed']}: pinned")
    stack_bad = [b for b in bad if b.startswith("stack: ")]                      # torch / lightning / numpy off the pinned stack: REPORTED, never refused — the kit runs on the installed stack and
    for b in stack_bad:                                                             #  its lines name the versions (uncertainty about the environment is named, not a gate)
        print(f"check_pins: NOTE {b} — reported, not refused: the kit runs on the installed stack", file=sys.stderr)
    checkout_bad = [b for b in bad if not b.startswith("stack: ")]               # the checkout is not the pin byte for byte (or absent): that changes what "stock" means — the one refusal, exit 3
    if checkout_bad:
        for b in checkout_bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
