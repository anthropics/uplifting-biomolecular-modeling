#!/usr/bin/env python3
"""Refuse unless the pinned upstream is installed: protenix 2.0.0 with the files of the pinned wheel (stock/PINS.json "protenix_version", "wheel").

usage: python -I stock/check_pins.py [--quiet]      exit 0 = pinned; 3 = not (one line per refusal on stderr)

Reads only package metadata and file bytes (no torch, no protenix import; standard library only, so it runs on the bare stack before the kit
is importable and inside an image build with no GPU): the installed distribution's version must be the pin and every ``.py`` file the wheel's
RECORD lists must be on disk with the RECORD's sha256 — a PyPI install, an install of the wheel in stock/ and a source install at the tag all
pass (same bytes); another version, a patched site-packages or an editable checkout at another commit is refused by name.

The software stack is a RECORD here, not a gate: PINS "pinned_stack" names the interpreter, torch build, triton and cuEquivariance versions the
kit's lever cells are configured for (STOCK.md 'Install'). When every one is installed as pinned nothing is printed; otherwise ONE line goes to
stderr, ``STACK not pinned: <name> <found> (pinned <want>); … — proceeding``, and the check passes: on another stack `check` / `pred` report
the levers they refuse by name. Weights are not this file's concern (`run.sh install --weights DIR`, and the WEIGHTS line of `check` / `pred`).
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
PACKAGE = "protenix"
STACK_DISTS = ("torch", "triton", "cuequivariance-torch", "cuequivariance-ops-torch-cu13")   # the pinned_stack entries that are distributions


def read_pins(path=None):
    with open(path or os.path.join(HERE, "PINS.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def wheel_record(wheel_path):
    """{relative path: sha256 hex} for the ``.py`` files the wheel's RECORD lists (RECORD carries urlsafe-b64 sha256)."""
    out = {}
    with zipfile.ZipFile(wheel_path) as z:
        rec = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
        for line in z.read(rec).decode().splitlines():
            parts = line.split(",")
            if len(parts) < 2 or not parts[1].startswith("sha256=") or not parts[0].endswith(".py") or ".dist-info/" in parts[0]:
                continue
            b64 = parts[1][len("sha256="):]
            out[parts[0]] = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).hex()
    return out


def _sha(path):
    h = hashlib.sha256()                                     # its own digest loop by design: this checker runs where opt_core may be absent
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dist_version(name):
    try:
        return md.distribution(name).version
    except md.PackageNotFoundError:
        return None


def torch_build():
    """torch's version WITH its local tag, without importing torch: the distribution's version, completed from ``torch/version.py``
    ``__version__`` when the metadata carries no tag (PyPI's 2.13.0 is the +cu130 build; the PyTorch index names it 2.13.0+cu130). None when absent."""
    v = dist_version("torch")
    if v is None or "+" in v:
        return v
    try:
        src = md.distribution("torch").locate_file("torch/version.py").read_text(encoding="utf-8")
    except (OSError, AttributeError, ValueError):
        return v
    m = re.search(r"^__version__\b[^=\n]*=\s*['\"]([^'\"]+)['\"]", src, re.M)
    full = m.group(1) if m else None
    return full if full and full.split("+")[0] == v else v


def stack_record(pins):
    """[(name, found, pinned), …] for every pinned_stack entry not installed as pinned (empty = the pinned stack)."""
    st = pins.get("pinned_stack") or {}
    found = {"python": sys.version.split()[0], "torch": torch_build()}
    found.update({n: dist_version(n) for n in STACK_DISTS if n != "torch"})
    return [(n, found.get(n), str(st[n])) for n in ("python",) + STACK_DISTS if n in st and found.get(n) != str(st[n])]


def check(pins=None, wheel_path=None):
    """(bad, detail): bad = refusal lines (empty = pinned); detail = what was found (``stack``: the not-as-pinned entries, a record)."""
    pins = pins or read_pins()
    want = str(pins["protenix_version"])
    wheel_path = wheel_path or os.path.join(HERE, os.path.basename(pins["wheel"]["file"]))
    detail = {"python": sys.version.split()[0], "stack": stack_record(pins), PACKAGE: None}
    try:
        dist = md.distribution(PACKAGE)
    except md.PackageNotFoundError:
        return [f"{PACKAGE} is not installed; want {PACKAGE}=={want} (pip install {pins['wheel']['file']})"], detail
    bad = []
    d = {"version": dist.version, "location": str(dist.locate_file("")), "pinned": False, "files_checked": 0, "files_differ": []}
    if dist.version != want:
        bad.append(f"{PACKAGE} {dist.version} installed; want {want}")
    if os.path.isfile(wheel_path):
        base = str(dist.locate_file(""))
        for rel, digest in wheel_record(wheel_path).items():
            d["files_checked"] += 1
            p = os.path.join(base, rel)
            if not os.path.isfile(p) or _sha(p) != digest:
                d["files_differ"].append(rel)
        if d["files_differ"]:
            bad.append(f"{PACKAGE} {dist.version} at {base}: {len(d['files_differ'])} file(s) differ from the pinned wheel ({d['files_differ'][:3]}...)")
        if d["files_checked"] == 0:
            bad.append(f"the pinned wheel {wheel_path} lists no .py files")
    else:
        bad.append(f"pinned wheel not found: {wheel_path} (stock/PINS.json wheel.file)")
    d["pinned"] = not bad
    detail[PACKAGE] = d
    return bad, detail


def stack_line(entries):
    """The one STACK line for not-as-pinned entries (None when the list is empty)."""
    if not entries:
        return None
    return "STACK not pinned: " + "; ".join(f"{n} {f or 'absent'} (pinned {w})" for n, f, w in entries) + " — proceeding: `check` / `pred` name the levers they refuse on this stack"


def main():
    quiet = "--quiet" in sys.argv[1:]
    pins = read_pins()
    bad, detail = check(pins)
    if not quiet:
        p = detail.get(PACKAGE) or {}
        if p.get("pinned"):
            print(f"{PACKAGE} {p['version']}: pinned ({p['files_checked']} files match the wheel) at {p['location']}")
        st = pins.get("pinned_stack") or {}
        print(("the pinned stack" if not detail["stack"] else "not the pinned stack") + f" (stock/PINS.json pinned_stack: {', '.join(f'{k} {v}' for k, v in st.items())}); python {detail['python']}")
    line = stack_line(detail["stack"])
    if line:                                                                  # the record, loud on stderr whatever --quiet says; the check passes
        print(line, file=sys.stderr)
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
