#!/usr/bin/env python3
"""Refuse unless the pinned upstream is installed.

    python -I stock/check_pins.py        # exit 0 = borzoi and baskerville installed at their pins, 3 = not (one line per refusal on stderr)

The pins are stock/PINS.json (beside this file). Bytes are the identity, not version strings, so a git install at the pinned
commit, an install of the archive in stock/ and an editable checkout of the commit all pass:

* ``baskerville`` — the distribution is installed and every ``.py`` of its importable package equals the same file in
  ``stock/baskerville-<commit>.tar.gz`` (``upstream.baskerville.archive``): the library the stock script and the kit call.
* ``borzoi`` — the distribution is installed and its package files equal ``stock/borzoi-<commit>.tar.gz``'s; the stock entry
  ``borzoi_sad.py``, located by the rule the kit applies when it runs (``$BORZOI_DIR/src/scripts/borzoi_sad.py``, else the
  ``borzoi_sad.py`` on PATH), has the sha256 of ``stock_scripts``.

The stack — python and the ``pins`` table (tensorflow, keras, numpy, h5py, …) — is a record, not a gate: on another version this
prints ``STACK not pinned: <name> <found> (pinned <version>), …`` on stderr and the check passes. Standard library only; the
interpreter that runs this file is the one inspected (``-I`` keeps PYTHONPATH and the current directory out of the answer, so an
importable package is an installed one).
"""
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import shutil
import sys
import tarfile

HERE = os.path.dirname(os.path.abspath(__file__))
EXIT_NOT_PINNED = 3


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def sha256_file(path, bufsize=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def dist_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def package_dir(package):
    """Directory of the importable top-level package, or None (not importable by this interpreter)."""
    try:
        spec = importlib.util.find_spec(package)
    except (ImportError, ValueError):
        return None
    if spec is None or not spec.origin or not spec.origin.endswith("__init__.py"):
        return None
    return os.path.dirname(os.path.abspath(spec.origin))


def archive_py_files(archive, subdir):
    """{path relative to <top>/<subdir>: sha256} for every .py under <top>/<subdir>/ of a `git archive` tarball in stock/."""
    out = {}
    with tarfile.open(archive, "r:gz") as tf:
        for m in tf.getmembers():
            if not (m.isfile() and m.name.endswith(".py")):
                continue
            parts = m.name.split("/", 1)
            if len(parts) < 2 or not parts[1].startswith(subdir.rstrip("/") + "/"):
                continue
            out[parts[1][len(subdir.rstrip("/")) + 1:]] = sha256_bytes(tf.extractfile(m).read())
    return out


def compare_package(package, archive, subdir):
    """(installed dir or None, files compared, [relative paths whose installed bytes differ or are missing])."""
    pdir = package_dir(package)
    if pdir is None:
        return None, 0, []
    want = archive_py_files(archive, subdir)
    bad = []
    for rel, digest in sorted(want.items()):
        p = os.path.join(pdir, rel)
        if not os.path.isfile(p) or sha256_file(p) != digest:
            bad.append(rel)
    return pdir, len(want), bad


def stock_entry(environ, script):
    """The stock entry the kit runs: $BORZOI_DIR/src/scripts/<script> when BORZOI_DIR is set, else <script> on PATH; None when neither exists."""
    root = environ.get("BORZOI_DIR")
    if root:
        p = os.path.join(root, "src", "scripts", script)
        return p if os.path.isfile(p) else None
    return shutil.which(script, path=environ.get("PATH"))


def check(pins, root=HERE, environ=None):
    """(ok, report lines, refusal lines, stack note or None) for this interpreter against the pin table."""
    environ = os.environ if environ is None else environ
    lines, refusals = [], []
    for name in ("baskerville", "borzoi"):
        up = pins["upstream"][name]
        archive = os.path.join(root, os.path.basename(up["archive"]))
        version = dist_version(name)
        if version is None:
            refusals.append(f"{name} is not installed (pinned: {up['repo']} @ {up['commit'][:7]}; STOCK.md 'Install' — e.g. pip install --no-deps {up['archive']})")
            continue
        if not os.path.isfile(archive):
            refusals.append(f"{archive} is missing from the tree (the pinned source of {name})")
            continue
        pdir, n, bad = compare_package(name, archive, f"src/{name}")
        if pdir is None:
            refusals.append(f"{name} {version} is installed but the package `{name}` is not importable by {sys.executable}")
        elif bad:
            refusals.append(f"{name} {version} at {pdir} is NOT the pinned source {up['repo']} @ {up['commit'][:7]}: {len(bad)} of {n} files differ from {os.path.basename(archive)} ({', '.join(bad[:6])}{' …' if len(bad) > 6 else ''})")
        else:
            lines.append(f"{name} {version}: pinned ({n} package files match {os.path.basename(archive)}) at {pdir}")
    for script, pin in sorted(pins["stock_scripts"].items()):
        entry = stock_entry(environ, script)
        if entry is None:
            where = f"$BORZOI_DIR/src/scripts/{script} (BORZOI_DIR={environ.get('BORZOI_DIR')})" if environ.get("BORZOI_DIR") else f"{script} on PATH (BORZOI_DIR unset)"
            refusals.append(f"stock entry not found: {where} — BORZOI_DIR names the calico/borzoi checkout (STOCK.md 'Install')")
            continue
        digest = sha256_file(entry)
        if digest != pin["sha256"]:
            refusals.append(f"stock entry {entry} sha256 {digest[:12]}… is NOT the pinned {script} ({pin['sha256'][:12]}…, {pin['bytes']} B)")
        else:
            lines.append(f"stock entry {entry}: pinned (sha256 {digest[:12]}…)")
    found_py = ".".join(map(str, sys.version_info[:3]))
    off = [] if found_py == pins["python"] else [f"python {found_py} (pinned {pins['python']})"]
    for name, want in pins["pins"].items():
        have = dist_version(name)
        if have != want:
            off.append(f"{name} {have or 'absent'} (pinned {want})")
    note = ("STACK not pinned: " + ", ".join(off) + " — stock/PINS.json `pins` is the tested stack; the run proceeds on this one") if off else None
    if not off:
        lines.append(f"stack: python {found_py} and the {len(pins['pins'])} pinned distributions (tensorflow {pins['pins'].get('tensorflow')}, …) at their pins")
    return not refusals, lines, refusals, note


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if argv:
        print("usage: python -I stock/check_pins.py", file=sys.stderr)
        return 2
    with open(os.path.join(HERE, "PINS.json")) as fh:
        pins = json.load(fh)
    ok, lines, refusals, note = check(pins)
    for line in lines:
        print(line)
    if note:
        print(note, file=sys.stderr)
    for r in refusals:
        print("REFUSED: " + r, file=sys.stderr)
    return 0 if ok else EXIT_NOT_PINNED


if __name__ == "__main__":
    sys.exit(main())
