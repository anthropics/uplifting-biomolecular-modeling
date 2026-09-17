#!/usr/bin/env python3
"""Refuse unless the pinned upstream is installed at the pin and its files are in a stock state (stock/PINS.json).

usage: python -I stock/check_pins.py [--quiet] [--expect pristine|any] [--json]
  exit 0 = the rc-foundry distribution is at the pin (its version string, and the commit or archive sha its install metadata
           carries), foundry/utils/torch.py is the upstream file (no deterministic-reference overlay), and the rfd3 file tree is in
           the expected state: --expect pristine (default) = the eight target files at their stock sha and the kit's two new files
           absent; --expect any = pristine or not (the patched interpreter; the package's registry classifies the kit state)
  exit 3 = not (one line per finding on stderr)

Reads only package metadata and file bytes (no torch, no rfd3 import). Three install sources are accepted, always together with the
pinned version string: the distribution's PEP 610 direct_url.json carrying the commit of a git install (vcs_info.commit_id), the
sha256 of an archive install (archive_info.hashes, checked against the tracked archive's own sha256), or a local directory (a `file:` url — a checkout
at the pin or the extracted archive, the kit's own install route — accepted on the version string alone, which names the commit:
0.2.1.dev13+g4010e3e2e); an install from any other source (an index, no direct_url.json) is refused. The file tree is located with the
path finder alone (nothing is executed). Standard library only, so run.sh, the configs and the package can all call it; this file is
the one place the pin check lives. Stack versions (python, torch, triton) are reported against PINS.json "pinned_stack" and never
refused (a run reports what its box has beside the pinned versions).
"""
import hashlib
import importlib.metadata as md
import importlib.machinery
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXIT_NOT_PINNED = 3


def sha256_of(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def site_dir(package="rfd3"):
    """The directory that holds the installed package (its site-packages), found without executing it and without consulting
    sys.meta_path (the path finder alone); None when not installed."""
    spec = importlib.machinery.PathFinder.find_spec(package)
    if spec is None or not spec.origin:
        return None
    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


def archive_sha256(pin):
    """The tracked archive's own sha256, hashed on the fly (the archive is the reference; nothing restates its digest)."""
    path = os.path.join(HERE, os.path.basename(pin["archive"]))
    return sha256_of(path) if os.path.isfile(path) else None


def distribution_state(pin):
    """The rc-foundry distribution against the pin: {installed, version, commit, archive_sha256, source, pinned}."""
    try:
        dist = md.distribution(pin["distribution"])
    except md.PackageNotFoundError:
        return {"installed": False, "version": None, "commit": None, "archive_sha256": None, "source": "not installed", "pinned": False}
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    commit = (info.get("vcs_info") or {}).get("commit_id")
    archive_sha = ((info.get("archive_info") or {}).get("hashes") or {}).get("sha256")
    want_archive = archive_sha256(pin)
    version_ok = dist.version == pin["version"]
    src_ok = bool(commit == pin["commit"] or (archive_sha and archive_sha == want_archive))
    how = (f"commit {commit}" if commit else f"archive sha256 {archive_sha[:12]}…" if archive_sha
           else "a local directory" if (info.get("url") or "").startswith("file:") else "a non-git, non-archive source")
    if not commit and not archive_sha and (info.get("url") or "").startswith("file:"):
        src_ok = version_ok            # a build from the extracted archive or the checkout: the version string is the evidence
    return {"installed": True, "version": dist.version, "commit": commit, "archive_sha256": archive_sha, "source": how,
            "pinned": version_ok and src_ok, "version_ok": version_ok}


KIT_STOCK_TABLE = os.path.join(HERE, "..", "opt", "forward", "xattempt_addon", "STOCK_SHA256_upstream_4010e3e.txt")


def kit_stock_table():
    """The kit's own upstream-at-pin reference (opt/forward/xattempt_addon/STOCK_SHA256_upstream_4010e3e.txt): {path: sha} for
    the eight rfd3/* files as they stood at the pinned commit — the one place that reference is recorded."""
    out = {}
    for line in open(KIT_STOCK_TABLE, encoding="utf-8"):
        if line.strip():
            sha, path = line.split()
            out[path] = sha
    return out


def stock_reference(pins):
    """Every file this check classifies against a stock hash: the kit's own upstream table (the eight rfd3/* paths) plus
    PINS.json "stock_files_sha256" (foundry/utils/torch.py — no other tracked home)."""
    ref = kit_stock_table()
    ref.update(pins["stock_files_sha256"])
    return ref


def tree_state(pins, ref, site):
    """Every file of `ref` (the stock reference) against the installed tree, plus the kit's new files.
    Returns {file: state} with state in stock | differs | absent | present (present = a kit new file that exists)."""
    out = {}
    for rel, want in ref.items():
        p = os.path.join(site, rel)
        out[rel] = "absent" if not os.path.isfile(p) else ("stock" if sha256_of(p) == want else "differs")
    for rel in pins["kit_new_files_absent_in_stock"]:
        out[rel] = "present" if os.path.isfile(os.path.join(site, rel)) else "absent"
    return out


def stack_versions():
    vers = {"python": ".".join(str(x) for x in sys.version_info[:3])}
    for name in ("torch", "triton"):
        try:
            vers[name] = md.version(name)
        except md.PackageNotFoundError:
            vers[name] = None
    return vers


def check(pins, expect="pristine"):
    """Returns (bad, detail): bad = the findings that refuse (empty = pinned), detail = everything measured."""
    pin = pins["upstream"]["foundry"]
    bad = []
    dist = distribution_state(pin)
    if not dist["pinned"]:
        bad.append(f"{pin['distribution']} {dist['version']}: installed from {dist['source']}; want version {pin['version']} from {pin['repo']} @ {pin['commit']} (or {pin['archive']})")
    site = site_dir("rfd3")
    files = {}
    ref = stock_reference(pins)
    if site is None:
        bad.append("rfd3 is not importable in this interpreter")
    else:
        files = tree_state(pins, ref, site)
        torch_py = files.get("foundry/utils/torch.py")
        if torch_py != "stock":
            bad.append(f"foundry/utils/torch.py is {torch_py}: the deterministic-reference overlay or another change is installed (want sha {ref['foundry/utils/torch.py'][:12]}…)")
        rfd3_files = {k: v for k, v in files.items() if k.startswith("rfd3/")}
        pristine = all(v in ("stock", "absent") for v in rfd3_files.values()) and all(
            files[k] == "stock" for k in ref if k.startswith("rfd3/"))
        if expect == "pristine" and not pristine:
            bad.append("rfd3 tree is not pristine: " + ", ".join(f"{k}={v}" for k, v in sorted(rfd3_files.items()) if v not in ("stock",) and not (v == "absent" and k in pins["kit_new_files_absent_in_stock"])))
    detail = {"distribution": dist, "site": site, "files": files, "pristine": site is not None and not [k for k, v in files.items() if k.startswith("rfd3/") and v not in ("stock",) and not (v == "absent" and k in pins["kit_new_files_absent_in_stock"])],
              "stack": stack_versions(), "pinned_stack": {k: pins["pinned_stack"].get(k) for k in ("python", "torch", "triton")}}
    return bad, detail


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quiet, as_json, expect = "--quiet" in argv, "--json" in argv, "pristine"
    if "--expect" in argv:
        expect = argv[argv.index("--expect") + 1]
    if expect not in ("pristine", "any"):
        print("check_pins: --expect pristine|any", file=sys.stderr)
        return 2
    pins = json.load(open(os.path.join(HERE, "PINS.json"), encoding="utf-8"))
    bad, detail = check(pins, expect)
    if as_json:
        print(json.dumps({"bad": bad, **detail}, indent=1, default=str))
    elif not quiet:
        d = detail["distribution"]
        if d["installed"]:
            print(f"rc-foundry {d['version']}: {'pinned' if d['pinned'] else 'NOT pinned'} ({d['source']}); rfd3 tree {'pristine' if detail['pristine'] else 'not pristine'} at {detail['site']}")
        s, r = detail["stack"], detail["pinned_stack"]
        print("stack: " + " ".join(f"{k}={s.get(k)}{'' if str(s.get(k)) == str(r.get(k)) or (k == 'torch' and r.get(k, '').startswith(str(s.get(k)))) else ' (record ' + str(r.get(k)) + ')'}" for k in ("python", "torch", "triton")))
    for b in bad:
        print(f"check_pins: {b}", file=sys.stderr)
    return EXIT_NOT_PINNED if bad else 0


if __name__ == "__main__":
    sys.exit(main())
