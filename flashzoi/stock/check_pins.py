#!/usr/bin/env python3
"""Refuse unless this environment carries the stock pin (stock/PINS.json).

usage: python -I stock/check_pins.py [--package-only] [--weights] [--hf-home <dir>] [--quiet]
       exit 0 = every checked pin holds; 3 = not (one line per mismatch on stderr, named); 2 = usage

What is checked, from package metadata and file bytes only (no torch import):
  package  the installed `borzoi-pytorch` distribution is the pinned version and every file the pinned wheel's RECORD lists
           (stock/<wheel>, present next to this file) is installed with the same sha256 — a PyPI
           install, the wheel in stock/ and an unpacked copy of it all pass; another version, an edited file or a checkout
           of another commit does not. The `borzoi_pytorch` that `import` resolves must be that distribution's copy.
  stack    python major.minor, torch, triton, flash-attn, transformers at the pinned versions (a torch local tag such
           as +cu124 is ignored). Skipped by --package-only.
  weights  (--weights) the four replicate snapshots in the hub cache as huggingface_hub resolves it ($HF_HUB_CACHE, else
           $HF_HOME/hub or --hf-home/hub): models--johahi--flashzoi-replicate-<k>/snapshots/<revision>/: model.safetensors bytes + sha256 and
           config.json sha256 per PINS.json. Four files of the pinned size are hashed: seconds each.
Standard library only, so run.sh, the configs and the package can all call it; this file is the one place the pin check lives.
"""
import base64
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
STACK_DISTS = {"torch": ("torch",), "triton": ("triton",), "flash-attn": ("flash-attn", "flash_attn"), "transformers": ("transformers",)}


def sha256_file(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def wheel_record(wheel_path):
    """{relative file path: sha256 hex} for the package files the wheel's RECORD lists (dist-info entries excluded)."""
    want = {}
    with zipfile.ZipFile(wheel_path) as z:
        record = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
        for line in z.read(record).decode().splitlines():
            path, digest, _size = line.rsplit(",", 2)
            if not digest or ".dist-info/" in path:
                continue
            algo, b64 = digest.split("=", 1)
            assert algo == "sha256", (path, algo)
            want[path] = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).hex()
    return want


def check_package(pin):
    bad, detail = [], {"name": pin["name"], "version": None, "files": 0, "pinned": False}
    wheel_path = os.path.join(HERE, pin["wheel"])
    if not os.path.isfile(wheel_path):
        return [f"stock/{pin['wheel']}: missing (the pinned wheel must sit next to this file)"], detail
    try:
        dist = md.distribution(pin["name"])
    except md.PackageNotFoundError:
        return [f"{pin['name']}: not installed; want {pin['version']} from {pin['wheel']}"], detail
    detail["version"] = dist.version
    if dist.version != pin["version"]:
        bad.append(f"{pin['name']} {dist.version}: want {pin['version']} ({pin['wheel']})")
    want = wheel_record(wheel_path)
    mismatched, missing = [], []
    for rel, sha in sorted(want.items()):
        path = str(dist.locate_file(rel))
        if not os.path.isfile(path):
            missing.append(rel)
        elif sha256_file(path) != sha:
            mismatched.append(rel)
    detail["files"] = len(want) - len(missing) - len(mismatched)
    if missing:
        bad.append(f"{pin['name']}: files the wheel's RECORD lists are absent from the install: {', '.join(missing)}")
    if mismatched:
        bad.append(f"{pin['name']}: files differ from the wheel's bytes: {', '.join(mismatched)}")
    top = os.path.basename(next(iter(want)).split("/")[0])
    spec = importlib.util.find_spec(top)
    installed_dir = os.path.realpath(str(dist.locate_file(top)))
    imported_dir = os.path.realpath(os.path.dirname(spec.origin)) if spec and spec.origin else None
    detail["import_dir"] = imported_dir
    if imported_dir != installed_dir:
        bad.append(f"{top}: `import {top}` resolves to {imported_dir}, not the pinned distribution's {installed_dir} (a shadowing copy on sys.path)")
    detail["pinned"] = not bad
    return bad, detail


def check_stack(pin):
    bad, detail = [], {}
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    detail["python"] = sys.version.split()[0]
    if py != pin["python"]:
        bad.append(f"python {detail['python']}: want {pin['python']}")
    for key, names in STACK_DISTS.items():
        have = None
        for n in names:
            try:
                have = md.version(n)
                break
            except md.PackageNotFoundError:
                continue
        detail[key] = have
        if have is None:
            bad.append(f"{key}: not installed; want {pin[key]}")
        elif have.split("+", 1)[0] != pin[key]:
            bad.append(f"{key} {have}: want {pin[key]}")
    return bad, detail


def hub_cache_dir(hf_home=None):
    """The hub cache directory as huggingface_hub resolves it: HF_HUB_CACHE when set, else <HF_HOME or --hf-home>/hub, else None."""
    if os.environ.get("HF_HUB_CACHE"):
        return os.environ["HF_HUB_CACHE"]
    home = hf_home or os.environ.get("HF_HOME")
    return os.path.join(home, "hub") if home else None


def check_weights(pins, hf_home):
    bad, detail = [], {}
    cache = hub_cache_dir(hf_home)
    if not cache:
        return ["weights: neither HF_HUB_CACHE nor HF_HOME is set and no --hf-home given"], detail
    detail["hub_cache"] = cache
    for repo, pin in pins["weights"].items():
        snap = os.path.join(cache, "models--" + repo.replace("/", "--"), "snapshots", pin["revision"])
        d = detail[repo] = {"snapshot": snap, "pinned": False}
        st, cfg = os.path.join(snap, "model.safetensors"), os.path.join(snap, "config.json")
        if not (os.path.isfile(st) and os.path.isfile(cfg)):
            bad.append(f"{repo}: no model.safetensors + config.json under {snap}")
            continue
        size = os.path.getsize(st)
        if size != pin["bytes"]:
            bad.append(f"{repo}: model.safetensors is {size} bytes; want {pin['bytes']}")
            continue
        st_sha, cfg_sha = sha256_file(st), sha256_file(cfg)
        if st_sha != pin["model.safetensors_sha256"]:
            bad.append(f"{repo}: model.safetensors sha256 {st_sha[:12]}…; want {pin['model.safetensors_sha256'][:12]}…")
        if cfg_sha != pins["config_json_sha256"]:
            bad.append(f"{repo}: config.json sha256 {cfg_sha[:12]}…; want {pins['config_json_sha256'][:12]}…")
        d["pinned"] = st_sha == pin["model.safetensors_sha256"] and cfg_sha == pins["config_json_sha256"]
    return bad, detail


def check(pins, package_only=False, weights=False, hf_home=None):
    """(bad, detail): every mismatch as one named line; detail per section for the package's report."""
    bad, detail = check_package(pins["package"])
    detail = {"package": detail}
    if not package_only:
        b, d = check_stack(pins["stack"])
        bad += b
        detail["stack"] = d
    if weights:
        b, d = check_weights(pins, hf_home)
        bad += b
        detail["weights"] = d
    return bad, detail


def main(argv):
    args = argv[1:]
    hf_home = None
    if "--hf-home" in args:
        i = args.index("--hf-home")
        if i + 1 >= len(args):
            print("usage: check_pins.py [--package-only] [--weights] [--hf-home <dir>] [--quiet]", file=sys.stderr)
            return 2
        hf_home = args[i + 1]
        del args[i:i + 2]
    unknown = [a for a in args if a not in ("--package-only", "--weights", "--quiet")]
    if unknown:
        print(f"check_pins: unknown option(s) {unknown}", file=sys.stderr)
        return 2
    pins = json.load(open(os.path.join(HERE, "PINS.json")))
    bad, detail = check(pins, package_only="--package-only" in args, weights="--weights" in args, hf_home=hf_home)
    if "--quiet" not in args:
        p = detail["package"]
        if p["pinned"]:
            print(f"{p['name']} {p['version']}: pinned ({p['files']} files == the wheel's RECORD)")
        if detail.get("stack") and not any(b.split(":")[0].split(" ")[0] in ("python", *STACK_DISTS) for b in bad):
            s = detail["stack"]
            print("stack: pinned (" + ", ".join(f"{k} {v}" for k, v in s.items()) + ")")
        for repo, d in (detail.get("weights") or {}).items():
            if isinstance(d, dict) and d["pinned"]:
                print(f"{repo}: pinned ({d['snapshot']})")
    for b in bad:
        print(f"check_pins: {b}", file=sys.stderr)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
