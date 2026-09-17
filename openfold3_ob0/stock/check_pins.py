#!/usr/bin/env python3
"""Refuse unless the installed `openfold3` is the pinned wheel, file for file (stock/PINS.json "wheel", "upstream").

usage: python -I stock/check_pins.py [--quiet] [--wheel-vs-source] [--weights PATH] [--json PATH]
       exit 0 = the installed distribution is openfold3 0.5.0 and every file its RECORD lists under openfold3/ has the sha256 the pinned
                wheel's RECORD carries (so an edited install — e.g. a hand-changed model_config.py — is refused by name);
       exit 3 = not (one line per finding on stderr).
       --wheel-vs-source additionally checks every packaged openfold3/ file (the pinned wheel's own RECORD) against stock/src (the
                tagged source): the wheel == tag byte check.
       --weights PATH hashes a checkpoint and refuses unless it is the pinned weights (stock/PINS.json "weights"; the package's
                routes run the same comparison themselves: `python -m openfold3_ob0_opt pred|check|warm`).

Reads only package metadata and file bytes (no torch, no openfold3 import). Standard library only, so run.sh, the configs and the
package can all call it; this file is the one place the pin check lives. PINS.json names the files and carries one digest per input
that lives outside git (the checkpoint "weights"."sha256", the Chemical Component Dictionary "ccd"."sha256") beside the vendored wheel's
and the freeze's. The printed line is the stock box's proof:
`[openfold3_ob0-opt] pins: openfold3 <version> installed==wheel <n>/<n> files (model_config.py <sha8>); wheel <sha8> == tag <tag> <commit8>`.
"""
import argparse
import base64
import hashlib
import importlib.metadata as md
import json
import os
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "[openfold3_ob0-opt]"
MODEL_CONFIG = "openfold3/projects/of3_all_atom/config/model_config.py"      # the file the stock configuration lives in; printed by name


def read_pins(path=None):
    with open(path or os.path.join(HERE, "PINS.json"), encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path):                     # the stock tree's own hasher on purpose: this checker runs in the STOCK venv (run.sh's stock arm, PINS checks) where neither the kit nor opt_core is installed
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def record_hashes(record_text):
    """RECORD lines -> {path: sha256 hex} for the openfold3/ package files (the dist-info and script entries are install-specific)."""
    out = {}
    for line in record_text.splitlines():
        if not line or not line.startswith("openfold3/"):
            continue
        name, digest, _size = line.rsplit(",", 2)
        if not digest.startswith("sha256="):
            continue
        b64 = digest[len("sha256="):]
        out[name] = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).hex()
    return out


def wheel_record(pins):
    path = os.path.join(HERE, os.path.basename(pins["wheel"]["file"]))
    with zipfile.ZipFile(path) as z:
        name = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
        return path, record_hashes(z.read(name).decode())


def check_installed(pins):
    """The installed distribution against the pinned wheel's RECORD. Returns (bad, detail)."""
    bad, detail = [], {"version": None, "files": 0, "ok": 0, "mismatch": [], "missing": []}
    try:
        dist = md.distribution("openfold3")
    except md.PackageNotFoundError:
        return [f"openfold3 is not installed (pinned {pins['openfold3_version']})"], detail
    detail["version"] = dist.version
    if dist.version != pins["openfold3_version"]:
        bad.append(f"openfold3 {dist.version} installed, pinned {pins['openfold3_version']}")
    _, want = wheel_record(pins)
    detail["files"] = len(want)
    root = dist.locate_file("")
    for name, digest in sorted(want.items()):
        p = os.path.join(str(root), name)
        if not os.path.isfile(p):
            detail["missing"].append(name)
            continue
        got = sha256_file(p)
        if got == digest:
            detail["ok"] += 1
        else:
            detail["mismatch"].append((name, got[:8], digest[:8]))
    detail["model_config_sha256"] = sha256_file(os.path.join(str(root), MODEL_CONFIG)) if os.path.isfile(os.path.join(str(root), MODEL_CONFIG)) else None
    detail["model_config_pinned_sha256"] = want.get(MODEL_CONFIG)
    if detail["missing"]:
        bad.append(f"{len(detail['missing'])} wheel files missing from the install: {detail['missing'][:3]}")
    if detail["mismatch"]:
        bad.append(f"{len(detail['mismatch'])} installed files differ from the pinned wheel: " + "; ".join(f"{n} {g}!={w}" for n, g, w in detail["mismatch"][:3]))
    return bad, detail


def check_wheel_vs_source(pins):
    """Every packaged openfold3/ file (the wheel's own RECORD) against stock/src. Returns (bad, detail)."""
    bad, detail = [], {"wheel_sha256": None, "files": 0, "identical": 0, "differ": []}
    path, want = wheel_record(pins)
    detail["wheel_sha256"] = sha256_file(path)
    src = os.path.join(HERE, "src")
    detail["files"] = len(want)
    for name, digest in sorted(want.items()):
        p = os.path.join(src, name)
        if os.path.isfile(p) and sha256_file(p) == digest:
            detail["identical"] += 1
        else:
            detail["differ"].append(name)
    if detail["differ"]:
        bad.append(f"{len(detail['differ'])} wheel files not byte-identical to stock/src (tag {pins['upstream']['tag']}): {detail['differ'][:3]}")
    return bad, detail


def check_weights(pins, path):
    """A checkpoint against the pinned weights: the file and digest stock/PINS.json "weights" names."""
    name, want = pins["weights"]["file"], pins["weights"]["sha256"]
    if not os.path.isfile(path):
        return [f"weights {path}: no such file"], {"path": path, "exists": False, "pinned_file": name, "pinned_sha256": want}
    got = sha256_file(path)
    bad = [] if got == want else [f"weights {path} sha256 {got[:8]} != {name} {want[:8]}: NOT the pinned weights"]
    return bad, {"path": path, "exists": True, "sha256": got, "pinned_file": name, "pinned_sha256": want, "is_pinned": got == want}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--wheel-vs-source", action="store_true", help="also check every packaged openfold3/ file against stock/src (the tagged source)")
    ap.add_argument("--weights", default=None, help="also hash this checkpoint against the pinned weights (stock/PINS.json)")
    ap.add_argument("--json", default=None, help="write the detail dict here")
    a = ap.parse_args(argv)
    pins = read_pins()
    bad, detail = check_installed(pins)
    out = {"installed": detail}
    line = (f"{PREFIX} pins: openfold3 {detail.get('version')} installed==wheel {detail['ok']}/{detail['files']} files "
            f"(model_config.py {(detail.get('model_config_sha256') or '-')[:8]})")
    if a.wheel_vs_source:
        bad2, d2 = check_wheel_vs_source(pins)
        bad += bad2
        out["wheel_vs_source"] = d2
        line += f"; wheel {d2['wheel_sha256'][:8]} == tag {pins['upstream']['tag']} {pins['upstream']['commit'][:8]}: {d2['identical']}/{d2['files']} files"
    if a.weights:
        bad4, d4 = check_weights(pins, a.weights)
        bad += bad4
        out["weights"] = d4
        line += f"; weights {(d4.get('sha256') or '-')[:8]} {'==' if d4.get('is_pinned') else '!='} {d4['pinned_file']} {d4['pinned_sha256'][:8]}"
    if a.json:
        with open(a.json, "w", encoding="utf-8") as fh:
            json.dump(out, fh, indent=1)
            fh.write("\n")
    if not a.quiet:
        print(line, file=sys.stderr, flush=True)
    for b in bad:
        print(f"{PREFIX} NOT STOCK: {b}", file=sys.stderr, flush=True)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
