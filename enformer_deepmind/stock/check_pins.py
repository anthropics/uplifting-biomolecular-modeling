#!/usr/bin/env python3
"""Check an install against stock/PINS.json.

usage: python -I stock/check_pins.py [--quiet] [--stack] [--weights DIR]
  --stack        compare the installed distributions tensorflow / tensorflow-hub / numpy with "pins"; every difference is listed; exit 4 when any
  --weights DIR  compare the SavedModel files under DIR (a TFHUB_CACHE_DIR entry or an extracted model directory) with "weights" -> files;
                 exit 3 when a file is missing or differs (one line per finding)
Standard library only; nothing is imported from the stack.
"""
import hashlib
import importlib.metadata as md
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PINS = json.load(open(os.path.join(HERE, "PINS.json"), encoding="utf-8"))
TAG = "[enformer-deepmind-opt]"


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv):
    quiet = "--quiet" in argv
    rc = 0
    if "--weights" in argv:
        d = argv[argv.index("--weights") + 1]
        w = PINS["weights"]["tfhub:deepmind/enformer/1"]
        if os.path.isdir(os.path.join(d, w["tfhub_cache_key"])):           # a TFHUB_CACHE_DIR: the model's entry inside it
            d = os.path.join(d, w["tfhub_cache_key"])
        for rel, meta in w["files"].items():
            p = os.path.join(d, rel)
            if not os.path.isfile(p):
                print(f"{TAG} WEIGHTS REFUSED: {p} missing", file=sys.stderr); rc = 3; continue
            digest = sha256_file(p)
            if digest != meta["sha256"]:
                print(f"{TAG} WEIGHTS REFUSED: {p} sha256 {digest} != the pinned {meta['sha256']}", file=sys.stderr); rc = 3
            elif not quiet:
                print(f"{TAG} WEIGHTS OK: {rel} ({meta['size_bytes']} bytes)")
        if rc:
            return rc
    if "--stack" in argv:
        diffs = []
        for dist, want in PINS.get("pins", {}).items():
            try:
                have = md.version(dist)
            except md.PackageNotFoundError:
                have = None
            if have != want:
                diffs.append(f"{dist}: installed {have}, pinned {want}")
        for d in diffs:
            print(f"{TAG} STACK DIFFERS: {d}", file=sys.stderr)
        if diffs:
            return 4
        if not quiet:
            print(f"{TAG} STACK OK: " + ", ".join(f"{k}=={v}" for k, v in PINS.get("pins", {}).items()))
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
