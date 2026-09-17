#!/usr/bin/env python3
"""The environment against the stock pin (stock/PINS.json): the stock files' bytes at the stock directory and, on request, the weights'
bytes are REFUSED off their pins (they are what "stock" means); the interpreter and the three pinned packages are REPORTED against the
pinned stack — a version off its pin is noted, never refused (the levers were tested on the pinned stack; a package absent is refused:
nothing runs without it).

usage: python -I stock/check_pins.py [--stock-dir DIR] [--weights-dir DIR [--size SIZE ...] [--hash-weights]]
                                     [--skip {interpreter,packages,stock-files,weights} ...] [--quiet]
exit 0 = the stock files (and weights) checked are at their pins (version notes, if any, on stderr); 3 = not (one line per failure on
stderr); 2 = a directory to check is absent

What is checked, and where the pin comes from:
- interpreter: sys.version_info major.minor vs PINS "pins.python" (the granularity the kits record) — noted when it differs.
- packages: importlib.metadata versions of torch / transformers / tokenizers vs PINS "pins" (torch's version carries its +cu tag in
  the wheel's own metadata), plus the sha256 of the transformers members named in PINS "transformers_wheel.members_sha256" (the
  sampler and the loader live in the wheel, not in the checkout) — a difference is noted; a package not installed is refused.
  Package metadata only: no torch import.
- stock files: every file of PINS "files" under progen2/ present at --stock-dir (default $PROGEN2_STOCK_DIR, else the tree's own copy
  stock/src/progen2) — the directory the CLIs run from — and every file of PINS "stock_files.files" at its pinned digest there (the
  bytes of salesforce/progen at the pinned commit): a file missing or off its digest is not the stock the modes are defined against.
- weights (only with --weights-dir): per size, <dir>/<progen2-size>/ (or <dir>/<size>/) holds pytorch_model.bin whose sha256 must
  equal PINS "weights.<size>.files.pytorch_model.bin.sha256": read from a SHA256SUMS beside it when one lists the file, from the bytes
  with --hash-weights or when no SHA256SUMS lists it; config.json likewise when PINS carries its sha256 (downloaded weights: hashed).
Standard library only, so run.sh, the configs, the package and the tests can all call it; this file is the one place the pin check lives.
"""
import argparse
import hashlib
import importlib.metadata as md
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGES = ("torch", "transformers", "tokenizers")
CHECKS = ("interpreter", "packages", "stock-files", "weights")


def sha256_file(path, bufsize=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(bufsize), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pins(path=None):
    import json

    with open(path or os.path.join(HERE, "PINS.json")) as f:
        return json.load(f)


def check_interpreter(pins):
    """Returns (bad, noted, detail): the interpreter off its pin is noted, never refused."""
    want = pins["pins"]["python"]
    got = ".".join(map(str, sys.version_info[:2]))
    full = ".".join(map(str, sys.version_info[:3]))
    ok = got == want
    noted = [] if ok else [f"python {full} differs from the pin {want}.x (pinned stack {pins['pinned_stack']['python']})"]
    return [], noted, {"version": full, "pinned": ok, "want": want, "pinned_stack": pins["pinned_stack"]["python"]}


def check_packages(pins):
    """Returns (bad, noted, detail): a version or a hashed member off its pin is noted; a package not installed is refused."""
    bad, noted, detail = [], [], {}
    for name in PACKAGES:
        want = pins["pins"][name]
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            bad.append(f"{name}: not installed (want {want})")
            detail[name] = {"version": None, "pinned": False, "want": want}
            continue
        ok = dist.version == want
        detail[name] = {"version": dist.version, "pinned": ok, "want": want}
        if not ok:
            noted.append(f"{name} {dist.version} differs from the pin {want}")
        if name == "transformers" and ok:
            members = {}
            for rel, want_sha in pins["transformers_wheel"]["members_sha256"].items():
                p = str(dist.locate_file(rel))
                got = sha256_file(p) if os.path.exists(p) else None
                members[rel] = got
                if got != want_sha:
                    noted.append(f"transformers member {rel} differs from the pinned wheel's: sha256 {got} != pin {want_sha} ({p})")
                    detail[name]["pinned"] = False
            detail[name]["members_sha256"] = members
    return bad, noted, detail


def stock_dir_default():
    return os.environ.get("PROGEN2_STOCK_DIR") or os.path.join(HERE, "src", "progen2")


def check_stock_files(pins, stock_dir):
    """Every file of the pinned progen2/ subdir present at ``stock_dir`` (the directory the CLIs run from), and every file of
    ``stock_files.files`` at its pinned digest there: the bytes of the pinned commit are what the modes are defined against, wherever
    the stock dir points. Returns (bad, detail); ``detail["sha256"]`` = the digests read."""
    bad, detail = [], {"dir": stock_dir, "files": {}, "sha256": {}}
    if not os.path.isdir(stock_dir):
        return [f"stock dir absent: {stock_dir}"], detail
    for rel in pins["files"]:
        if not rel.startswith("progen2/"):
            continue
        sub = rel[len("progen2/"):]
        p = os.path.join(stock_dir, sub)
        present = os.path.exists(p)
        detail["files"][sub] = present
        if not present:
            bad.append(f"stock file missing: {p}")
    for rel, want in pins["stock_files"]["files"].items():
        sub = rel[len("progen2/"):]
        p = os.path.join(stock_dir, sub)
        if not os.path.exists(p):
            if detail["files"].get(sub, True):                                 # not already named missing above
                bad.append(f"stock file missing: {p}")
            detail["files"][sub] = False
            continue
        got = sha256_file(p)
        detail["sha256"][sub] = got
        if got != want:
            bad.append(f"stock file differs from the pinned commit: {sub}: {got[:8]} != {want[:8]} ({p})")
    detail["pinned"] = not bad
    return bad, detail


def size_dir(weights_dir, size):
    """<weights_dir>/<progen2-size> or <weights_dir>/<size without the prefix> — whichever exists."""
    for name in (size, size[len("progen2-"):].lower()):
        p = os.path.join(weights_dir, name)
        if os.path.isdir(p):
            return p
    return None


def listed_sha(sums_path, filename):
    if not os.path.exists(sums_path):
        return None
    with open(sums_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2 and parts[1].lstrip("*").lstrip("./") == filename:
                return parts[0]
    return None


def check_weights(pins, weights_dir, sizes=None, hash_bytes=False):
    """Per size: pytorch_model.bin (and config.json when pinned) at their pins — the SHA256SUMS line beside the file, else the bytes."""
    bad, detail = [], {"dir": weights_dir, "sizes": {}}
    if not os.path.isdir(weights_dir):
        return [f"weights dir absent: {weights_dir}"], detail
    for size in sizes or pins["sizes"]["names"]:
        pin = pins["weights"].get(size)
        if pin is None:
            bad.append(f"unknown size {size!r}; known {pins['sizes']['names']}")
            continue
        d = size_dir(weights_dir, size)
        rec = {"dir": d, "files": {}}
        detail["sizes"][size] = rec
        if d is None:
            bad.append(f"{size}: no checkpoint dir under {weights_dir}")
            continue
        for fname, fpin in pin["files"].items():
            want = fpin.get("sha256")
            if not want:
                continue
            p = os.path.join(d, fname)
            if not os.path.exists(p):
                bad.append(f"{size}: {p} missing")
                rec["files"][fname] = None
                continue
            got = None if hash_bytes else listed_sha(os.path.join(d, "SHA256SUMS"), fname)
            how = "SHA256SUMS line"
            if got is None:
                got, how = sha256_file(p), "bytes"
            rec["files"][fname] = {"sha256": got, "from": how}
            if got != want:
                bad.append(f"{size}: {fname} sha256 {got} ({how}) != pin {want}")
    detail["pinned"] = not bad
    return bad, detail


def check_all(pins, stock_dir=None, weights_dir=None, sizes=None, hash_weights=False, skip=()):
    """Run the selected checks; returns ``(bad, detail)``: ``bad`` = one line per pin refused (empty = nothing refused);
    ``detail["noted"]`` = one line per interpreter / package difference (reported, never refused)."""
    bad, detail = [], {"noted": []}
    if "interpreter" not in skip:
        b, n, detail["interpreter"] = check_interpreter(pins)
        bad += b; detail["noted"] += n
    if "packages" not in skip:
        b, n, detail["packages"] = check_packages(pins)
        bad += b; detail["noted"] += n
    if "stock-files" not in skip:
        b, detail["stock_files"] = check_stock_files(pins, stock_dir or stock_dir_default())
        bad += b
    if weights_dir is not None and "weights" not in skip:
        b, detail["weights"] = check_weights(pins, weights_dir, sizes, hash_weights)
        bad += b
    return bad, detail


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stock-dir", default=None, help="the progen2/ directory the CLIs run from (default: $PROGEN2_STOCK_DIR, else stock/src/progen2)")
    ap.add_argument("--weights-dir", default=None, help="root holding <progen2-size>/pytorch_model.bin per size (checked only when given)")
    ap.add_argument("--size", action="append", default=None, help="restrict the weights check to this size (repeatable)")
    ap.add_argument("--hash-weights", action="store_true", help="hash the weight bytes even when a SHA256SUMS beside them lists the file")
    ap.add_argument("--skip", action="append", choices=CHECKS, default=[], help="skip a check (repeatable)")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args(argv)
    pins = load_pins()
    stock_dir = args.stock_dir or stock_dir_default()
    if "stock-files" not in args.skip and not os.path.isdir(stock_dir):
        print(f"check_pins: stock dir absent: {stock_dir}", file=sys.stderr)
        return 2
    if args.weights_dir is not None and "weights" not in args.skip and not os.path.isdir(args.weights_dir):
        print(f"check_pins: weights dir absent: {args.weights_dir}", file=sys.stderr)
        return 2
    bad, detail = check_all(pins, stock_dir, args.weights_dir, args.size, args.hash_weights, args.skip)
    if not args.quiet:
        d = detail.get("interpreter")
        if d and d["pinned"]:
            print(f"python {d['version']}: pinned ({d['want']}.x)")
        for name, p in (detail.get("packages") or {}).items():
            if p["pinned"]:
                extra = f" ({len(p['members_sha256'])} members hashed)" if p.get("members_sha256") else ""
                print(f"{name} {p['version']}: pinned{extra}")
        d = detail.get("stock_files")
        if d and d.get("pinned"):
            print(f"stock files at {d['dir']}: pinned ({len(d['files'])} files)")
        d = detail.get("weights")
        if d and d.get("pinned"):
            how = ", ".join(f"{s}: {r['files']['pytorch_model.bin']['from']}" for s, r in d["sizes"].items())
            print(f"weights at {d['dir']}: pinned ({how})")
    for n in detail["noted"]:
        print(f"check_pins: noted: {n} — the levers were tested on the pinned stack", file=sys.stderr)
    for b in bad:
        print(f"check_pins: {b}", file=sys.stderr)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
