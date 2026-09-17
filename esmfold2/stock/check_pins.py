#!/usr/bin/env python3
"""Refuse unless the pinned upstream forks are installed at the pinned commits (stock/PINS.json "upstream") and the pinned stack's
accelerated layer is installed (stock/PINS.json "image": the packages of that one layer at their pinned versions — flash-attn).

usage: python -I stock/check_pins.py [--quiet]        exit 0 = both forks at their pins and the layer's packages at theirs; 3 = not (one line per
                                                      fork / package on stderr)

Reads only package metadata (no torch import): the PEP 610 direct_url.json that pip writes for a git install carries the commit
(vcs_info.commit_id); an install from one of the archives in stock/ carries that archive's own local path (archive_info, a file:// url),
matched by filename against the pin's "archive" (the archive is a git-tracked file — it is the commit that added it, and its
name already embeds that commit's first 8 hex digits, so no digest of it is restated here). Any other install (PyPI, a different commit,
an editable checkout) is refused. The layer check reads the installed distribution versions of "image".python_packages: a box without
flash-attn or with another version is refused BY NAME — `image pins NOT MET: flash_attn: expected <v> got None — the pinned stack's
accelerated layer (stock/PINS.json "image") is not installed here`. Whether flash-attn is LIVE in the
interpreter (imports, and upstream's FLASH_ATTN_AVAILABLE is set) is the package's runtime reading (opt/esmfold2_opt/attn.py: the
atom_attn= word of every route, the ESMFOLD2_OPT_REQUIRE_FAST_ENV=1 refusal), not this metadata check. Standard library only, so run.sh,
the configs and the package can all call it; this file is the one place the pin check lives.
"""
import importlib.metadata as md
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def check(pins):
    """Every pinned fork against its installed distribution metadata. Returns ``(bad, detail)``: ``bad`` = one line per fork that is not at
    its pin (empty = all pinned); ``detail`` = per fork ``{version, commit, archive, pinned, source}``."""
    bad, detail = [], {}
    for name, pin in pins.items():
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            bad.append(f"{name}: not installed (want {pin['repo']} @ {pin['commit']})")
            detail[name] = {"version": None, "commit": None, "archive": None, "pinned": False, "source": "not installed"}
            continue
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        commit = (info.get("vcs_info") or {}).get("commit_id")
        url = info.get("url") or ""
        archive_name = os.path.basename(url) if info.get("archive_info") is not None and url.startswith("file://") else None
        want_archive = os.path.basename(pin["archive"])
        pinned = bool(commit == pin["commit"] or (archive_name and archive_name == want_archive))
        how = f"commit {commit}" if commit else (f"archive {archive_name}" if archive_name else "a non-git, non-archive source")
        detail[name] = {"version": dist.version, "commit": commit, "archive": archive_name, "pinned": pinned, "source": how}
        if not pinned:
            bad.append(f"{name} {dist.version}: installed from {how}; want {pin['repo']} @ {pin['commit']} (or {pin['archive']})")
    return bad, detail


def image_pins(image):
    """The pinned stack's accelerated layer (stock/PINS.json "image".python_packages: import name -> version) against the installed
    distribution metadata. Returns ``(bad, detail)``: ``bad`` = one line per package not at its pin; ``detail`` = per package
    ``{expected, got, pinned}``. An empty "image" block pins nothing."""
    bad, detail = [], {}
    image = image or {}
    for name, want in (image.get("python_packages") or {}).items():
        got = None
        for dist_name in (name, name.replace("_", "-")):
            try:
                got = md.version(dist_name)
                break
            except md.PackageNotFoundError:
                continue
        pinned = got == want
        detail[name] = {"expected": want, "got": got, "pinned": pinned}
        if not pinned:
            bad.append(f"{name}: expected {want} got {got} — the pinned stack's accelerated layer (stock/PINS.json \"image\") is not installed here")
    return bad, detail


def main():
    allpins = json.load(open(os.path.join(HERE, "PINS.json")))
    quiet = "--quiet" in sys.argv[1:]
    bad, detail = check(allpins["upstream"])
    ibad, idetail = image_pins(allpins.get("image"))
    if not quiet:
        for name, d in detail.items():
            if d["pinned"]:
                print(f"{name} {d['version']}: pinned ({d['source']})")
        for name, d in idetail.items():
            if d["pinned"]:
                print(f"{name} {d['got']}: pinned (stock/PINS.json \"image\" layer)")
    if bad or ibad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        if ibad:
            print("check_pins: image pins NOT MET: " + "; ".join(ibad), file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
