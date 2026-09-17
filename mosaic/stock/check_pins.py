#!/usr/bin/env python3
"""Refuse unless the pinned upstream packages are installed at the pinned commits (stock/PINS.json "upstream": mosaic, joltz, boltz).

usage: python -I stock/check_pins.py [--quiet]        exit 0 = all three installed at their pins; 3 = not (one line per package on stderr)

Reads only package metadata (no jax / torch import): the PEP 610 direct_url.json that pip writes for a git install carries the commit
(vcs_info.commit_id); an install from one of the archives in stock/ carries that archive's own path in the recorded url, checked against
the pin's archive name (the archives are this repo's own tracked files: named by the git commit, not by a digest restated here). Any other
install (PyPI, a different commit, an editable checkout) is refused. Standard library only, so run.sh, the configs and the package can
all call it; this file is the one place the pin check lives.
"""
import importlib.metadata as md
import json
import os
import sys
import urllib.parse

HERE = os.path.dirname(os.path.abspath(__file__))


def check(pins):
    """Every pinned package against its installed distribution metadata. Returns ``(bad, detail)``: ``bad`` = one line per package that is
    not at its pin (empty = all pinned); ``detail`` = per package ``{version, commit, archive_name, pinned, source}``."""
    bad, detail = [], {}
    for name, pin in pins.items():
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            bad.append(f"{name}: not installed (want {pin['repo']} @ {pin['commit']})")
            detail[name] = {"version": None, "commit": None, "archive_name": None, "pinned": False, "source": "not installed"}
            continue
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        commit = (info.get("vcs_info") or {}).get("commit_id")
        archive_name = os.path.basename(urllib.parse.urlsplit(info.get("url") or "").path) if info.get("archive_info") else None
        want_archive = os.path.basename(pin["archive"])
        pinned = bool(commit == pin["commit"] or archive_name == want_archive)
        how = f"commit {commit}" if commit else (f"archive {archive_name}" if archive_name else "a non-git, non-archive source")
        detail[name] = {"version": dist.version, "commit": commit, "archive_name": archive_name, "pinned": pinned, "source": how}
        if not pinned:
            bad.append(f"{name} {dist.version}: installed from {how}; want {pin['repo']} @ {pin['commit']} (or {pin['archive']})")
    return bad, detail


def main():
    pins = json.load(open(os.path.join(HERE, "PINS.json")))["upstream"]
    quiet = "--quiet" in sys.argv[1:]
    bad, detail = check(pins)
    if not quiet:
        for name, d in detail.items():
            if d["pinned"]:
                print(f"{name} {d['version']}: pinned ({d['source']})")
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
