#!/usr/bin/env python3
"""Refuse unless the pinned upstream package is installed at its pin (stock/PINS.json "upstream").

usage: python -I stock/check_pins.py [--quiet] [--stack stock|kit] [--freeze stock|kit]
  exit 0 = enformer-pytorch installed at the pin: the pinned version, and every pinned package file present where it is installed
           (stock/PINS.json "upstream" -> "files": name -> size_bytes); 3 = not (one line per finding on stderr)
  --stack <id>   also compare the installed distributions with the named stack's freeze (stock/PINS.json "stacks" -> <id> -> "packages");
                 every difference is listed; exit 4 when any (the stack gate, after the package gate)
  --freeze <id>  print the named stack's freeze as requirement lines (name==version) and exit 0 — a view of PINS.json, the one place the
                 stack facts live

Reads only package metadata and the installed files (no torch import). enformer-pytorch is a PyPI package: pip writes no PEP 610
direct_url.json for a PyPI install, so the version alone would not pin the bytes — the installed files are checked present at the
pinned relative paths. An install from one of the archives in stock/ carries the archive's sha256 in direct_url.json
(archive_info.hashes), matched live against the pinned sdist/wheel in stock/ (the repo's own tracked bytes — no separate manifest) and
reported by name; an editable checkout is refused (its bytes can change under the process). Standard library only; run.sh install calls it, and it is the one place the install-time pin check lives (the package checks the
upstream version again at activation).
"""
import hashlib
import importlib.metadata as md
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PINS_PATH = os.path.join(HERE, "PINS.json")


def read_pins(path=PINS_PATH):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _local_archive_shas(pin):
    """{basename: sha256} of the pin's own sdist/wheel, hashed live from stock/ (the repo's own tracked bytes; no separate manifest)."""
    out = {}
    for key in ("sdist", "wheel"):
        rel = pin.get(key)
        if rel:
            p = os.path.join(HERE, os.path.basename(rel))
            if os.path.isfile(p):
                out[os.path.basename(rel)] = sha256_file(p)
    return out


def _direct_url(dist):
    try:
        txt = dist.read_text("direct_url.json")
    except Exception:  # noqa: BLE001
        txt = None
    if not txt:
        return None
    try:
        return json.loads(txt)
    except ValueError:
        return None


def check(pins):
    """(bad, detail): bad = the refusal lines (empty = pinned); detail[name] = what was found."""
    bad, detail = [], {}
    for name, pin in pins.items():
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            bad.append(f"{name}: not installed; want {pin['install']} ({pin['sdist']} or {pin['wheel']})")
            detail[name] = {"installed": False, "pinned": False}
            continue
        version_ok = dist.version == pin["version"]
        du = _direct_url(dist)
        how = "PyPI (no direct_url.json)"
        editable = False
        archive_sha = None
        if du:
            if (du.get("dir_info") or {}).get("editable"):
                editable = True
                how = f"editable checkout {du.get('url')}"
            elif du.get("archive_info"):
                archive_sha = (du["archive_info"].get("hashes") or {}).get("sha256") or (du["archive_info"].get("hash") or "").replace("sha256=", "") or None
                local = _local_archive_shas(pin)
                known = [n for n, s in local.items() if s == archive_sha]
                how = f"archive {known[0]}" if known else f"archive {du.get('url')} (not one of the pinned stock/ archives)"
            elif du.get("vcs_info"):
                how = f"vcs {du.get('url')} @ {du['vcs_info'].get('commit_id')}"
        # presence: every pinned package file, present where it is installed (the git commit names the bytes; no byte re-checking)
        missing = [rel for rel in (pin.get("files") or {}) if not os.path.isfile(dist.locate_file(rel))]
        files_ok = not missing and bool(pin.get("files"))
        pinned = version_ok and files_ok and not editable
        detail[name] = {"installed": True, "version": dist.version, "source": how, "archive_sha256": archive_sha, "editable": editable,
                        "files_checked": len(pin.get("files") or {}), "files_missing": missing, "pinned": pinned}
        if not version_ok:
            bad.append(f"{name} {dist.version}: want {pin['version']} ({pin['install']}; {pin['sdist']} or {pin['wheel']} in stock/)")
        if editable:
            bad.append(f"{name} {dist.version}: installed as an {how}; an editable install is refused — install {pin['install']} or one of the archives in stock/")
        if missing:
            bad.append(f"{name} {dist.version}: installed files missing vs the pin ({pin['sdist']}): " + ", ".join(missing))
    return bad, detail


def stack_check(stacks, stack_id):
    """Compare every distribution of the named stack's freeze with what is installed -> list of difference lines (empty = identical)."""
    if stack_id not in stacks:
        return [f"unknown stack '{stack_id}' (PINS.json stacks: {', '.join(k for k in stacks if isinstance(stacks[k], dict) and 'packages' in stacks[k])})"]
    want = stacks[stack_id]["packages"]
    installed = {}
    for d in md.distributions():
        n = (d.metadata["Name"] or "").strip()
        if n:
            installed.setdefault(_norm(n), d.version)
    diffs = []
    for name, ver in sorted(want.items(), key=lambda kv: kv[0].lower()):
        have = installed.get(_norm(name))
        if have is None:
            diffs.append(f"{name}: not installed; the {stack_id} stack has {ver}")
        elif have != ver:
            diffs.append(f"{name}: installed {have}; the {stack_id} stack has {ver}")
    return diffs


def _norm(name):
    return name.lower().replace("_", "-").replace(".", "-")


def freeze_lines(stacks, stack_id):
    return [f"{n}=={v}" for n, v in sorted(stacks[stack_id]["packages"].items(), key=lambda kv: kv[0].lower())]


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    quiet = "--quiet" in argv
    stack_id = argv[argv.index("--stack") + 1] if "--stack" in argv else None
    freeze_id = argv[argv.index("--freeze") + 1] if "--freeze" in argv else None
    pins = read_pins()
    if freeze_id:
        if freeze_id not in pins["stacks"]:
            print(f"check_pins: unknown stack '{freeze_id}'", file=sys.stderr)
            return 2
        print("\n".join(freeze_lines(pins["stacks"], freeze_id)))
        return 0
    bad, detail = check(pins["upstream"])
    if not quiet:
        for name, d in detail.items():
            if d.get("pinned"):
                print(f"{name} {d['version']}: pinned ({d['source']}; {d['files_checked']} files present)")
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        return 3
    if stack_id:
        diffs = stack_check(pins["stacks"], stack_id)
        if diffs:
            for line in diffs:
                print(f"check_pins: stack {stack_id}: {line}", file=sys.stderr)
            return 4
        if not quiet:
            print(f"stack {stack_id}: every distribution of the stack's freeze installed at its version ({len(pins['stacks'][stack_id]['packages'])} distributions)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
