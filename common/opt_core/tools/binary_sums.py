#!/usr/bin/env python
"""binary_sums.py — the SHA256SUMS lines of every compiled binary the core package ships (extension modules, cubins, compressed cubins).

Every such file under ``opt_core/`` is a ``<sha256>  <relpath>`` line of a SHA256SUMS in its own directory or in a directory above it
inside the package — the nearest one that lists it (``opt_core.gates.binary_sums_for``); the loaders re-hash the file against that line
before mapping it and refuse it by name otherwise (``opt_core.gates.binary_refusal``).  A sealed payload keeps its own SHA256SUMS at its
root; a binary inside a directory whose bytes are pinned elsewhere is listed from a SHA256SUMS above that directory.  After adding or
rebuilding a binary, run ``--write`` (new binaries get a SHA256SUMS beside them; a listed one has its line refreshed) and commit the file.

  python tools/binary_sums.py            # check: every binary is a line of the SHA256SUMS covering it, digest equal; exit 1 otherwise
  python tools/binary_sums.py --write    # refresh the line of every listed binary; write <dir>/SHA256SUMS beside every unlisted one
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
from typing import Dict, List, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.join(os.path.dirname(HERE), "opt_core")
SUMS = "SHA256SUMS"
BINARY_SUFFIXES = (".so", ".cubin", ".xz", ".ptx", ".fatbin")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def binaries(package: str = PACKAGE) -> List[str]:
    """Every compiled binary under the package, sorted."""
    out = []
    for root, dirs, files in os.walk(package):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        out += [os.path.join(root, f) for f in sorted(files) if f.endswith(BINARY_SUFFIXES)]
    return out


def sums_files(package: str = PACKAGE) -> List[str]:
    out = []
    for root, dirs, files in os.walk(package):
        dirs[:] = sorted(d for d in dirs if d != "__pycache__")
        if SUMS in files:
            out.append(os.path.join(root, SUMS))
    return out


def read_sums(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    with open(path, encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln or ln.startswith("#"):
                continue
            digest, rel = ln.split(None, 1)
            out[rel.strip()] = digest.lower()
    return out


def covering(path: str, package: str = PACKAGE):
    """(SHA256SUMS path, key) of the nearest SHA256SUMS at or above the file's directory (up to the package) that lists the file;
    (nearest SHA256SUMS, None) when none lists it; None when there is no SHA256SUMS at all."""
    p = os.path.abspath(path)
    d = os.path.dirname(p)
    nearest = None
    while True:
        cand = os.path.join(d, SUMS)
        if os.path.isfile(cand):
            key = os.path.relpath(p, d).replace(os.sep, "/")
            if key in read_sums(cand):
                return cand, key
            nearest = nearest or cand
        if d == os.path.abspath(package) or os.path.dirname(d) == d:
            return (nearest, None) if nearest else None
        d = os.path.dirname(d)


def check(package: str = PACKAGE) -> List[str]:
    """The problems, one line each ([] = every binary under the package is a line of the SHA256SUMS covering it, digest equal).  A
    sealed payload's own SHA256SUMS may list files the tree does not carry or carries restated; those lines are left alone — only the
    binaries present are checked."""
    problems = []
    for p in binaries(package):
        rel = os.path.relpath(p, package)
        cov = covering(p, package)
        if cov is None:
            problems.append(f"{rel}: no {SUMS} in or above its directory")
            continue
        sums_path, key = cov
        want = read_sums(sums_path).get(key) if key is not None else None
        if want is None:
            problems.append(f"{rel}: not listed in {os.path.relpath(sums_path, package)}")
        elif sha256_file(p) != want:
            problems.append(f"{rel}: sha256 differs from {os.path.relpath(sums_path, package)}")
    return problems


def write(package: str = PACKAGE) -> List[Tuple[str, int]]:
    """For each binary a SHA256SUMS above its directory lists: refresh that line when the digest moved.  For every other binary:
    write <dir>/SHA256SUMS beside it (every binary of that directory, sorted).  Returns [(SHA256SUMS path, entries)] written."""
    todo, refresh = set(), {}
    for p in binaries(package):
        cov = covering(p, package)
        if cov is not None and cov[1] is not None and os.path.dirname(cov[0]) != os.path.dirname(p):
            if read_sums(cov[0])[cov[1]] != sha256_file(p):    # a SHA256SUMS above it lists it: that line is kept, its digest refreshed
                refresh.setdefault(cov[0], {})[cov[1]] = sha256_file(p)
            continue
        todo.add(os.path.dirname(p))
    written = []
    for sums_path, moved in sorted(refresh.items()):
        lines = []
        with open(sums_path, encoding="utf-8") as fh:
            for ln in fh:
                parts = ln.split(None, 1)
                key = parts[1].strip() if len(parts) == 2 else None
                lines.append(f"{moved[key]}  {key}\n" if key in moved else ln)
        with open(sums_path, "w", encoding="utf-8") as fh:
            fh.write("".join(lines))
        written.append((sums_path, len(moved)))
    for d in sorted(todo):
        names = sorted(f for f in os.listdir(d) if f.endswith(BINARY_SUFFIXES) and os.path.isfile(os.path.join(d, f)))
        text = "".join(f"{sha256_file(os.path.join(d, f))}  {f}\n" for f in names)
        with open(os.path.join(d, SUMS), "w", encoding="utf-8") as fh:
            fh.write(text)
        written.append((os.path.join(d, SUMS), len(names)))
    return written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--write", action="store_true", help="write / refresh the SHA256SUMS files, then check")
    ap.add_argument("--package", default=PACKAGE, help="the opt_core package directory (default: the one beside this tool)")
    args = ap.parse_args(argv)
    package = os.path.abspath(args.package)
    if args.write:
        for path, n in write(package):
            print(f"wrote {os.path.relpath(path, package)} ({n} entries)")
    problems = check(package)
    for line in problems:
        print(line)
    n_bin, n_sums = len(binaries(package)), len(sums_files(package))
    print(f"{n_bin} binaries, {n_sums} {SUMS} files, {len(problems)} problems")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
