#!/usr/bin/env python3
"""Refuse unless the pinned upstream packages are installed at the pinned commits (stock/PINS.json "upstream").

usage: python -I stock/check_pins.py [--quiet] [--require-all]
       exit 0 = every pinned package installed at its pin; 3 = not (one line per package on stderr)

Reads only package metadata (no torch import): the PEP 610 direct_url.json that pip writes for a git install carries the commit
(vcs_info.commit_id); an install from one of the archives in stock/ carries that archive's own path in its "url" (archive_info
present and non-empty), matched by filename against the pin's archive. Any other install (PyPI, a different commit, an editable
checkout) is refused. `protpardelle` is pinned but only the ensemble32 variant needs it: when it is absent the line says so and
the exit code is still 0 unless --require-all. Standard library only, so run.sh, the configs and the package can all call it;
this file is the one place the pin check lives.
"""
import importlib.metadata as md
import json
import os
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))
OPTIONAL = ("protpardelle",)                 # needed by one variant only (PINS.json "upstream".protpardelle.note)


def check(pins, require_all=False):
    """Every pinned package against its installed distribution metadata. Returns ``(bad, detail)``: ``bad`` = one line per package
    that is not at its pin (empty = all pinned); ``detail`` = per package ``{version, commit, archive_name, pinned, source}``."""
    bad, detail = [], {}
    for name, pin in pins.items():
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            detail[name] = {"version": None, "commit": None, "archive_name": None, "pinned": False, "source": "not installed",
                            "optional": name in OPTIONAL}
            if name in OPTIONAL and not require_all:
                continue
            bad.append(f"{name}: not installed (want {pin['repo']} @ {pin['commit']})")
            continue
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        commit = (info.get("vcs_info") or {}).get("commit_id")
        archive_name = os.path.basename(urllib.parse.urlparse(info.get("url") or "").path) if info.get("archive_info") else None
        want_archive = os.path.basename(pin["archive"])
        pinned = bool(commit == pin["commit"] or (archive_name and archive_name == want_archive))
        how = f"commit {commit}" if commit else (f"archive {archive_name}" if archive_name else "a non-git, non-archive source")
        detail[name] = {"version": dist.version, "commit": commit, "archive_name": archive_name, "pinned": pinned, "source": how,
                        "optional": name in OPTIONAL}
        if not pinned:
            bad.append(f"{name} {dist.version}: installed from {how}; want {pin['repo']} @ {pin['commit']} (or {pin['archive']})")
    return bad, detail


def main():
    pins = json.load(open(os.path.join(HERE, "PINS.json")))["upstream"]
    argv = sys.argv[1:]
    quiet = "--quiet" in argv
    bad, detail = check(pins, require_all="--require-all" in argv)
    if not quiet:
        for name, d in detail.items():
            if d["pinned"]:
                print(f"{name} {d['version']}: pinned ({d['source']})")
            elif d["optional"] and d["source"] == "not installed":
                print(f"{name}: not installed (needed by the ensemble32 variant only)")
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
