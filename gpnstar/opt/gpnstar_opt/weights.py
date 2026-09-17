"""`run.sh install --weights DIR [--model KEY|REPO]` — stage the checkpoint snapshot stock GPN-Star reads from the Hugging Face cache into
DIR (an ``HF_HOME``) with the hub library's own downloader, then check every staged file against the tree's pins (``stock/PINS.json``
``weights.<key>.files``: sha256 and size per file). DIR is then the ``HF_HOME`` every route reads; nothing is fetched at run time.

The transfer is ``huggingface_hub.snapshot_download(repo, revision=<pinned>, allow_patterns=None: the whole snapshot, so every pinned file is present)`` into
``DIR/hub`` — the library's own cache layout (``models--songlab--<name>/snapshots/<rev>/…``), which is what ``from_pretrained(<repo id>)``
resolves offline under ``HF_HOME=DIR`` once ``refs/main`` names the commit — the step writes it (a download by commit leaves no ref). A
snapshot that is already complete is kept and only checked. A file whose digest is not the pin's is
named and the step fails (exit 1) — the file is left in place for inspection, never deleted. A checkpoint whose digests the pins do not carry
yet is staged and its digests are PRINTED (`DIGEST` lines) with exit 1 and the sentence saying the pins lack them: an unchecked snapshot is
never reported as checked. For this process DIR is bound as ``HF_HOME`` / ``HF_HUB_CACHE`` and the offline switches are lifted.

    python -m gpnstar_opt.weights DIR                       # the primary model
    python -m gpnstar_opt.weights DIR --model ce11-25m      # another pinned model, by key or repository id
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

from . import registry
from .stack import sha256_of

PREFIX = "[gpnstar-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
OFFLINE_SWITCHES = ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE")


def say(msg: str) -> None:
    sys.stderr.write(f"{PREFIX} {msg}\n")
    sys.stderr.flush()


def stage(hf_home: str, key: str, download=None) -> str:
    """Download (or find) the pinned snapshot of `key` under hf_home/hub; returns the snapshot directory."""
    ck = registry.checkpoint(key)
    os.makedirs(os.path.join(hf_home, "hub"), exist_ok=True)
    for k in OFFLINE_SWITCHES:
        os.environ.pop(k, None)
    os.environ["HF_HOME"] = hf_home
    os.environ["HF_HUB_CACHE"] = os.path.join(hf_home, "hub")
    if download is None:
        from huggingface_hub import snapshot_download as download
    say(f"staging {ck['repo']} @ {ck['rev'][:12]} into {hf_home}/hub (the whole snapshot)")
    path = download(ck["repo"], revision=ck["rev"], allow_patterns=registry.ALLOW_PATTERNS, cache_dir=os.path.join(hf_home, "hub"))
    write_ref(hf_home, ck)
    return str(path)


def write_ref(hf_home: str, ck: dict, name: str = "main") -> str:
    """Write `refs/<name>` = the pinned commit in the repository's cache directory: a download by commit writes no ref, and without one the
    bare repository id does not resolve offline (`from_pretrained("songlab/...")` under HF_HUB_OFFLINE=1 reads refs/main). Returns the path."""
    repo_dir = os.path.join(hf_home, "hub", f"models--{ck['repo'].replace('/', '--')}")
    os.makedirs(os.path.join(repo_dir, "refs"), exist_ok=True)
    p = os.path.join(repo_dir, "refs", name)
    with open(p, "w", encoding="utf-8") as fh:
        fh.write(ck["rev"])
    say(f"refs/{name} -> {ck['rev']} ({p})")
    return p


def listing(snap: str) -> dict:
    out = {}
    for root, _dirs, files in os.walk(snap):
        for f in sorted(files):
            p = os.path.join(root, f)
            rel = os.path.relpath(p, snap).replace(os.sep, "/")
            out[rel] = {"sha256": sha256_of(p), "bytes": os.path.getsize(p)}
    return dict(sorted(out.items()))


def check(snap: str, key: str) -> int:
    ck = registry.checkpoint(key)
    have = listing(snap)
    for rel, ent in have.items():
        say(f"DIGEST model={key} file={rel} sha256={ent['sha256']} bytes={ent['bytes']}")
    if not ck["files"]:
        say(f"FAIL model={key}: stock/PINS.json weights.{key}.files carries no digests for revision {ck['rev'][:12]} — the snapshot is staged but UNCHECKED "
            f"(record the DIGEST lines above in the pins, then re-run)")
        return EXIT_FAIL
    bad = []
    for rel, ent in ck["files"].items():
        h = have.get(rel)
        if h is None:
            bad.append(f"{rel}: absent")
        elif ent.get("sha256") and h["sha256"] != ent["sha256"]:
            bad.append(f"{rel}: sha256 {h['sha256'][:12]} != pinned {ent['sha256'][:12]}")
        elif ent.get("bytes") is not None and int(ent["bytes"]) != h["bytes"]:
            bad.append(f"{rel}: {h['bytes']} bytes != pinned {ent['bytes']}")
    if bad:
        say(f"FAIL model={key} snapshot={snap}: " + "; ".join(bad) + " (the files are left in place)")
        return EXIT_FAIL
    say(f"OK model={key} files={len(ck['files'])} snapshot={snap} — set HF_HOME={os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(snap))))} for the runs")
    return EXIT_OK


def main(argv=None, download=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gpnstar_opt.weights", description="stage the pinned checkpoint snapshot into DIR (an HF_HOME) and check it against stock/PINS.json")
    ap.add_argument("dir", help="the HF_HOME to stage into")
    ap.add_argument("--model", default=None, help=f"model key or repository id (default: {registry.PRIMARY})")
    a = ap.parse_args(argv)
    key = registry.resolve(a.model)
    if key is None:
        say(f"--model {a.model!r} is not one of the pinned checkpoints ({' | '.join(registry.keys())})")
        return EXIT_USAGE
    hf_home = os.path.abspath(a.dir)
    try:
        snap = stage(hf_home, key, download=download)
    except Exception as e:  # noqa: BLE001 — the library's own failure, relayed by name
        say(f"FAIL model={key}: the download did not complete: {type(e).__name__}: {e}")
        return EXIT_FAIL
    return check(snap, key)


if __name__ == "__main__":
    sys.exit(main())
