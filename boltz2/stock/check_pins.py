#!/usr/bin/env python3
"""Refuse unless the pinned upstream is installed: boltz 2.2.1 whose installed `boltz/` tree is byte-identical to the pin (stock/PINS.json "upstream").

usage: python -I stock/check_pins.py [--quiet] [--ckpt]     exit 0 = pinned; 3 = not (one line per problem on stderr)

The pin is the installed tree itself, not the install route: the sha256 over the concatenated bytes of every `boltz/**/*.py` file
(sorted paths) must equal PINS.json "installed_tree.sha256_all_py_concat" with the pinned file count, and the distribution version must
be 2.2.1. The PyPI wheel, the wheel in stock/, the source archive in stock/ and an sdist install all produce that tree; an editable
checkout of another commit, a patched site-packages tree or another version are refused. `--ckpt` also checks the sha256 of
$BOLTZ_CACHE/boltz2_conf.ckpt against PINS.json "weights" (reads ~3 GB; a few seconds). Standard library only, no torch import,
so run.sh, the configs and the package can all call it; this file is the one place the pin check lives.
"""
import glob
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def pins(path=None):
    return json.load(open(path or os.path.join(HERE, "PINS.json")))


def installed_tree_digest(root):
    """(n_py_files, sha256) of the installed boltz tree at `root`, computed as the pin recipe states."""
    files = sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True))
    h = hashlib.sha256()   # private on purpose: stock-side tool, runs in the stock environment where opt_core is absent by design
    for f in files:
        h.update(open(f, "rb").read())
    return len(files), h.hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def check(p, ckpt=False, environ=None):
    """Returns ``(bad, detail)``: ``bad`` = one line per problem (empty = pinned); ``detail`` = what was found."""
    environ = os.environ if environ is None else environ
    bad, detail = [], {"version": None, "root": None, "n_py_files": None, "sha256_all_py_concat": None, "pinned": False}
    want_v = p["boltz_version"]; tree = p["upstream"]["installed_tree"]
    try:
        detail["version"] = md.version("boltz")
    except md.PackageNotFoundError:
        bad.append(f"boltz: not installed (want {want_v}: {p['upstream']['install']})")
        return bad, detail
    spec = importlib.util.find_spec("boltz")
    root = os.path.dirname(spec.origin) if spec and spec.origin else None
    detail["root"] = root
    if detail["version"] != want_v:
        bad.append(f"boltz {detail['version']}: want {want_v}")
    if root is None:
        bad.append("boltz: distribution present but the package is not importable (no boltz/__init__.py found)")
        return bad, detail
    n, sha = installed_tree_digest(root)
    detail["n_py_files"], detail["sha256_all_py_concat"] = n, sha
    if not (n == tree["n_py_files"] and sha == tree["sha256_all_py_concat"]):
        bad.append(f"boltz tree at {root}: {n} .py files, sha256 {sha[:16]}…; want {tree['n_py_files']} files, {tree['sha256_all_py_concat'][:16]}… (tag {p['upstream']['tag']}, unmodified)")
    detail["pinned"] = not bad
    if ckpt:
        cache = environ.get("BOLTZ_CACHE") or os.path.expanduser("~/.boltz")
        want = p["weights"]["files"]["boltz2_conf.ckpt"]["sha256"]
        f = os.path.join(cache, "boltz2_conf.ckpt")
        if not os.path.isfile(f):
            bad.append(f"checkpoint: {f} not found (BOLTZ_CACHE={environ.get('BOLTZ_CACHE')})")
        else:
            got = file_sha256(f); detail["ckpt_sha256"] = got
            if got != want:
                bad.append(f"checkpoint {f}: sha256 {got[:16]}…, want {want[:16]}…")
        detail["pinned"] = not bad
    return bad, detail


def main():
    quiet = "--quiet" in sys.argv[1:]
    bad, detail = check(pins(), ckpt="--ckpt" in sys.argv[1:])
    if not quiet and detail["pinned"]:
        print(f"boltz {detail['version']}: pinned (tree {detail['n_py_files']} files, sha256 {detail['sha256_all_py_concat'][:16]}… at {detail['root']})"
              + (f"; boltz2_conf.ckpt sha256 {detail['ckpt_sha256'][:16]}…" if "ckpt_sha256" in detail else ""))
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
