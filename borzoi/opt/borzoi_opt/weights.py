"""`run.sh install --weights DIR`: the trained weights the documented command takes as its ``<model_file>`` argument, fetched into DIR and
held to ``stock/PINS.json`` ``weights`` (sha256).

    python -m borzoi_opt.weights DIR      # exit 0 = every wired file is under DIR with the pinned digest; 1 = a fetch failed or a digest differs

One file is wired at this pin: ``f0/model0_best.h5`` (the published human fold-0 replicate) → ``DIR/f0/model0_best.h5``. Upstream ships no
per-file downloader (its ``download_models.sh`` fetches all four replicates plus gene annotation into the checkout with wget), so the file
comes from the URL the pin table names (``weights.<file>.url``: the address upstream's README publishes), over HTTPS, into a temporary name
beside the target and renamed into place when the transfer completes. A file already present is kept and checked, never refetched; a file
whose digest differs from the pin is REFUSED by name and left where it is for inspection (delete it to fetch again). The genome, targets and
params the command also reads are inputs, not weights: README.md 'Install' says where each comes from.
"""
from __future__ import annotations

import http.client
import os
import shutil
import sys
import urllib.request
from typing import Callable, Dict, Optional

from . import kit

TAG = "[borzoi-opt install]"


def wired(pins: Optional[dict] = None) -> Dict[str, dict]:
    """{relative path under DIR: {"url", "sha256", "bytes"}} — the ``weights`` entries of stock/PINS.json that carry a url and a digest."""
    table = (pins if pins is not None else kit.pins())["weights"]
    return {rel: dict(e) for rel, e in table.items() if isinstance(e, dict) and e.get("url") and e.get("sha256")}


def download(url: str, dest: str, *, timeout: float = 60.0) -> None:
    """HTTPS GET of ``url`` into ``dest``: written to ``dest + '.part'`` and renamed into place only when complete."""
    part = dest + ".part"
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    with urllib.request.urlopen(url, timeout=timeout) as resp, open(part, "wb") as out:
        shutil.copyfileobj(resp, out, length=1 << 22)
    os.replace(part, dest)


def fetch(directory: str, *, files: Optional[Dict[str, dict]] = None, fetch_one: Callable[[str, str], None] = download, out=print) -> int:
    """Populate ``directory`` with every wired file and check each against its pin; returns the process exit code (0 ok, 1 refused/failed)."""
    files = wired() if files is None else files
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    bad = []
    for rel, e in sorted(files.items()):
        dest = os.path.join(directory, rel)
        if os.path.isfile(dest):
            out(f"{TAG} present:  {rel}")
        else:
            out(f"{TAG} fetching: {rel} <- {e['url']} ({e.get('size_bytes', '?')} B)")
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            try:
                fetch_one(e["url"], dest)
            except (OSError, http.client.HTTPException) as x:   # a failed transfer (URLError and HTTPError are OSErrors) is named, every file is still reported, the exit code carries it
                bad.append(rel); out(f"{TAG} FAILED:   {rel}: {type(x).__name__}: {x}"); continue
        digest = kit.sha256_file(dest)
        if digest != e["sha256"]:
            bad.append(rel)
            out(f"{TAG} REFUSED:  {rel}: sha256 {digest} is not the pinned {e['sha256']} (stock/PINS.json weights) — left in place at {dest}; delete it to fetch again")
        else:
            out(f"{TAG} ok:       {rel} sha256 {digest[:16]}… = the pin")
    if bad:
        out(f"{TAG} WEIGHTS NOT READY: {len(bad)} of {len(files)} file(s) failed or differ from the pin: {', '.join(bad)}")
        return 1
    out(f"{TAG} WEIGHTS OK: {len(files)}/{len(files)} file(s) under {directory} have the pinned digest — "
        + ", ".join(os.path.join(directory, rel) for rel in sorted(files)) + " is the <model_file> argument of `run.sh sad`")
    return 0


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m borzoi_opt.weights DIR", file=sys.stderr)
        return 2
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
