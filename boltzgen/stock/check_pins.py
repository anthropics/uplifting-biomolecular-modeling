#!/usr/bin/env python3
"""Refuse unless the pinned upstream is installed: boltzgen 0.3.2 with the files of the pinned wheel (stock/PINS.json ``boltzgen_version``, ``wheel``).

usage: python -I stock/check_pins.py [--quiet]      exit 0 = pinned; 3 = not (one line per refusal on stderr)

Reads only package metadata and file bytes (no torch and no boltzgen import — either would load the CUDA stack): the installed ``boltzgen``
distribution's version must be the pin, and every ``boltzgen/*.py`` file the wheel's RECORD lists must be on disk with the RECORD's sha256 — so an
install of the wheel in stock/, a package-index install of the same version and a git install at the tag all pass (same bytes), and an editable
checkout at another commit, a patched site-packages or another version is refused by name.

The torch / CUDA stack is a RECORD here, not a gate: the pinned stack is PINS ``stack_of_record`` (torch 2.13.0+cu130, CUDA 13.0). A torch that is
that build prints nothing about it; any other torch (or none) prints ONE line to stderr, ``STACK not pinned: torch <version> (pinned: torch
2.13.0+cu130)``, and the check still passes — the package names the stack it runs on again at activation (its KERNELS census line). Software pins
only: the weights are pinned by digest in PINS ``weights`` and checked where they are fetched (``run.sh install --weights DIR``). Standard library
only, so run.sh can call it on the interpreter as installed, before anything of the kit is imported.
"""
import base64
import hashlib
import importlib.metadata as md
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = "boltzgen"


def read_pins(path=None):
    with open(path or os.path.join(HERE, "PINS.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def wheel_record(wheel_path):
    """{relative path: sha256 hex} for the package's files in the wheel (RECORD carries urlsafe-b64 sha256)."""
    out = {}
    with zipfile.ZipFile(wheel_path) as z:
        rec = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
        for line in z.read(rec).decode().splitlines():
            parts = line.split(",")
            if len(parts) < 2 or not parts[1].startswith("sha256=") or not parts[0].startswith(PACKAGE + "/"):
                continue
            b64 = parts[1][len("sha256="):]
            out[parts[0]] = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).hex()
    return out


def _sha(path):
    h = hashlib.sha256()                                     # its own digest loop by design: this checker runs before the kit or its core is importable
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_build():
    """torch's version as its installed distribution states it, WITHOUT importing torch, completed with the local tag of ``torch/version.py``
    (``__version__``) when the metadata carries none (2.13.0 + version.py 2.13.0+cu130 -> 2.13.0+cu130); None when torch is not installed."""
    try:
        dist = md.distribution("torch")
    except md.PackageNotFoundError:
        return None
    version = dist.version
    try:
        src = dist.locate_file("torch/version.py").read_text(encoding="utf-8")
    except (OSError, AttributeError, ValueError):
        return version
    m = re.search(r"^__version__\b[^=\n]*=\s*['\"]([^'\"]+)['\"]", src, re.M)
    full = m.group(1) if m else None
    if full and "+" not in version and full.split("+")[0] == version:
        version = full
    if "+" not in version:                                                   # a build with no local tag anywhere: name its CUDA from version.py so the line says what it is
        m = re.search(r"^cuda\b[^=\n]*=\s*['\"]([^'\"]*)['\"]", src, re.M)
        cuda = (m.group(1) or None) if m else None
        version = f"{version}+cu{cuda.replace('.', '')}" if cuda else f"{version} (CUDA build unknown)"
    return version


def stack_record(pins):
    """The stack record: ``{"torch", "pinned_torch", "pinned", "line"}`` — ``line`` is None when the installed torch is the pinned build
    (same version, same local tag), else the one ``STACK not pinned: …`` line."""
    want = str(pins["stack_of_record"]["torch"])
    found = torch_build()
    pinned = found is not None and found == want
    line = None if pinned else f"STACK not pinned: {'torch ' + found if found else 'torch not installed'} (pinned: torch {want})"
    return {"torch": found, "pinned_torch": want, "pinned": pinned, "line": line}


def check(pins=None, wheel_path=None):
    """(bad, detail): bad = refusal lines (empty = pinned); detail = what was found (``detail["stack"]`` is the stack record, never a refusal)."""
    pins = pins or read_pins()
    want = pins["boltzgen_version"]
    wheel_path = wheel_path or os.path.join(HERE, os.path.basename(pins["wheel"]["file"]))
    detail = {"python": sys.version.split()[0], "stack": stack_record(pins)}
    try:
        dist = md.distribution(PACKAGE)
    except md.PackageNotFoundError:
        detail[PACKAGE] = None
        return [f"{PACKAGE} is not installed; want {PACKAGE}=={want} (pip install --no-deps {pins['wheel']['file']})"], detail
    bad = []
    d = {"version": dist.version, "location": str(dist.locate_file("")), "pinned": False, "files_checked": 0, "files_differ": []}
    if dist.version != want:
        bad.append(f"{PACKAGE} {dist.version} installed; want {want}")
    if os.path.isfile(wheel_path):
        base = dist.locate_file("")
        differ = []
        for rel, digest in wheel_record(wheel_path).items():
            if not rel.endswith(".py"):
                continue
            p = os.path.join(str(base), rel)
            d["files_checked"] += 1
            if not os.path.isfile(p) or _sha(p) != digest:
                differ.append(rel)
        d["files_differ"] = differ
        if differ:
            bad.append(f"{PACKAGE} {dist.version} at {base}: {len(differ)} file(s) differ from the pinned wheel ({differ[:3]}...)")
        if d["files_checked"] == 0:
            bad.append(f"the pinned wheel {wheel_path} lists no {PACKAGE} .py files")
    else:
        bad.append(f"pinned wheel not found: {wheel_path} (stock/PINS.json wheel.file)")
    d["pinned"] = not bad
    detail[PACKAGE] = d
    return bad, detail


def main():
    quiet = "--quiet" in sys.argv[1:]
    bad, detail = check(read_pins())
    s = detail["stack"]
    if not quiet:
        b = detail.get(PACKAGE) or {}
        if b.get("pinned"):
            print(f"{PACKAGE} {b['version']}: pinned ({b['files_checked']} files match the wheel) at {b['location']}")
        print(f"torch {s['torch'] or 'not installed'}: {'the pinned stack' if s['pinned'] else 'not the pinned stack'} (stock/PINS.json stack_of_record: torch {s['pinned_torch']}); python {detail['python']}")
    if s["line"]:                                                            # the record, on stderr whatever --quiet says; never a refusal
        print(f"{s['line']} — proceeding (the package names the stack it runs on at activation)", file=sys.stderr)
    if bad:
        for line in bad:
            print(f"check_pins: {line}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
