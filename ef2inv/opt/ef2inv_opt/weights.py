"""Fetch the six pinned weight snapshots into a Hugging Face cache root and check them — `run.sh install --weights DIR` (README.md §Setup).

DIR is used as HF_HOME: the snapshots land at DIR/hub/models--biohub--<name>/snapshots/<commit>/… exactly as huggingface_hub lays a cache
out, so `HF_HOME=DIR` at run time is the whole configuration (configs/*.env keep the run offline). The fetch is huggingface_hub's own
``snapshot_download(repo_id, revision=<the pinned snapshot commit>, cache_dir=DIR/hub)`` — the library the stock script's ``from_pretrained``
calls resolve and download through — once per repository of stock/PINS.json ``weights``, online for this step whatever HF_HUB_OFFLINE says
(HF_TOKEN is honoured if set; the six repositories are public). A repository whose pinned snapshot already holds every pinned file is named
and not fetched. The stock script loads each repository at revision ``main`` (``from_pretrained(repo_id)``, offline), so a repository whose
``refs/main`` is absent or names another commit gets it pointed at the pinned snapshot, said on a line. Then every file is checked against
stock/PINS.json — sha256 per file, the check stock/check_pins.py ``weights_gate`` makes at ``run.sh check``: all six snapshots at their pins
→ ``WEIGHTS OK`` and exit 0; a file that differs or is missing → named, exit 1, nothing deleted (the file stays for inspection).

  python -m ef2inv_opt.weights DIR
"""
from __future__ import annotations

import importlib.util
import os
import sys

from .cli import _model_opt, _pins


def check_pins_module(model_opt=None):
    """stock/check_pins.py of this tree, imported by path (its weights_gate, repo_dir and snapshot_dir are the one reading of the cache layout)."""
    path = os.path.join(model_opt or _model_opt(), "stock", "check_pins.py")
    spec = importlib.util.spec_from_file_location("ef2inv_stock_check_pins", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def upstream_fetch(hub_dir):
    """The downloader the stock script's loads go through: huggingface_hub.snapshot_download into ``hub_dir`` (= DIR/hub), online for this
    call. Returns fetch(repo_id, commit) -> the snapshot directory written."""
    if os.environ.get("HF_HUB_OFFLINE", "").strip() not in ("", "0"):
        print(f"[ef2inv-opt install] HF_HUB_OFFLINE={os.environ['HF_HUB_OFFLINE']} in this shell — the fetch runs online for this step")
    os.environ["HF_HUB_OFFLINE"] = "0"
    from huggingface_hub import snapshot_download   # after the switch: the library reads it at import

    def fetch(repo_id, commit):
        return snapshot_download(repo_id=repo_id, revision=commit, cache_dir=hub_dir)
    return fetch


def complete(snapshot, rec):
    """True when the snapshot directory holds every pinned file of the repository at its pinned size (the digest check comes after)."""
    return all(os.path.isfile(os.path.join(snapshot, rel)) and os.path.getsize(os.path.join(snapshot, rel)) == fr["size_bytes"] for rel, fr in rec["files"].items())


def point_refs_main(cp, hf_home, repo, commit):
    """refs/main of the repository's cache dir → the pinned commit, when it is absent or names another revision (said on a line)."""
    ref = os.path.join(cp.repo_dir(hf_home, repo), "refs", "main")
    have = open(ref, encoding="utf-8").read().strip() if os.path.isfile(ref) else None
    if have == commit:
        return
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w", encoding="utf-8") as fh:
        fh.write(commit)
    print(f"  {repo}: refs/main {'was ' + have[:12] + ', now' if have else 'set to'} {commit[:12]} — the revision from_pretrained('{repo}') loads offline")


def fetch(dir_, fetcher=None, cp=None, table=None):
    """Fetch what is missing under DIR with huggingface_hub, point refs/main at the pins, then check every file against stock/PINS.json. Returns the exit code."""
    dir_ = os.path.abspath(dir_); hub = os.path.join(dir_, "hub")
    cp = cp or check_pins_module(); table = table or _pins(_model_opt())["weights"]
    os.makedirs(hub, exist_ok=True); os.environ["HF_HOME"] = dir_
    n_files = sum(len(rec["files"]) for rec in table.values()); gib = sum(fr["size_bytes"] for rec in table.values() for fr in rec["files"].values()) / 2 ** 30
    print(f"[ef2inv-opt install] weights → {dir_} (HF_HOME): {len(table)} repositories, {n_files} files, {gib:.1f} GiB at their pins (stock/PINS.json weights)")
    for repo, rec in table.items():
        snap = cp.snapshot_dir(dir_, repo, rec)
        if complete(snap, rec):
            print(f"  present: {repo}@{rec['snapshot_commit'][:12]} ({len(rec['files'])} files) — kept and checked, no fetch")
            continue
        size = sum(fr["size_bytes"] for fr in rec["files"].values()) / 2 ** 30
        print(f"  fetching {repo}@{rec['snapshot_commit'][:12]} ({len(rec['files'])} files, {size:.2f} GiB) with huggingface_hub.snapshot_download", flush=True)
        fetcher = fetcher or upstream_fetch(hub)
        try:
            fetcher(repo, rec["snapshot_commit"])
        except Exception as e:   # named, and the check below still runs on what is there
            print(f"  FAILED fetching {repo}@{rec['snapshot_commit'][:12]}: {type(e).__name__}: {e}", file=sys.stderr)
    for repo, rec in table.items():
        if os.path.isdir(cp.snapshot_dir(dir_, repo, rec)):
            point_refs_main(cp, dir_, repo, rec["snapshot_commit"])
    fails, _status, census, lines = cp.weights_gate(table, dir_, hash_files=True)
    for line in lines:
        print("  " + line)
    off = {repo: c for repo, c in census.items() if c["word"] != "pinned"}
    if fails or off:
        for repo, c in off.items():
            print(f"[ef2inv-opt install] REFUSED {repo}: {c['word']}" + (f" — {'; '.join(c['differs'])}" if c["differs"] else "") + (f" — missing {', '.join(c['missing'])}" if c["missing"] else ""), file=sys.stderr)
        print(f"[ef2inv-opt install] WEIGHTS FAILED: {len(off)}/{len(table)} repositories are not at their pins under {dir_} (files left in place)", file=sys.stderr)
        return 1
    print(f"[ef2inv-opt install] WEIGHTS OK: {len(table)}/{len(table)} snapshots, {n_files} files at their pins (sha256 = stock/PINS.json) under {dir_} — export HF_HOME={dir_}")
    return 0


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m ef2inv_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr); return 2
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
