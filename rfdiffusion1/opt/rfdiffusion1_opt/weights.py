"""`run.sh install --weights DIR` — fetch the pinned checkpoint files into DIR and check each one against ``stock/PINS.json`` ``weights``
(sha256). DIR is then the ``WEIGHTS`` directory every route reads (README.md 'Setup'); nothing is fetched at run time.

The files are the entries of PINS ``weights`` that carry a digest (``Complex_base_ckpt.pt``, the checkpoint every mode loads, and
``Base_ckpt.pt``, upstream's default for runs without hotspots). Upstream's own downloader, ``scripts/download_models.sh`` of the pinned
checkout, is a ``wget`` of seven checkpoints with no digest check and no notion of a file already present, so it is not called: each pinned
file that is absent from DIR is transferred from the https URL PINS records for it (``url_kit_installer``; upstream's README lists the same
path over http) into a ``<name>.part`` file beside it, renamed into place when complete; a file already in DIR is kept and only checked. No
URL is restated here. Every file is then hashed with the pin check's own routine (``stock/check_pins.py`` ``sha256_file``); a file whose
size or digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never deleted; so is the
``.part`` file of an interrupted transfer (remove it by hand). ``python -m rfdiffusion1_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
import urllib.request
from typing import Callable, Dict, List, Optional

from .report import TAG

PREFIX = "[" + TAG + " install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
URL_KEY = "url_kit_installer"                    # the https URL of each PINS weights entry
PART_SUFFIX = ".part"                            # the transfer's temporary file beside the target


def pinned_files(pins: Optional[dict] = None) -> List[Dict[str, object]]:
    """The PINS ``weights`` entries that carry a digest, in file order: [{"name", "sha256", "size_bytes", "url"}]."""
    if pins is None:
        from . import stack
        pins = stack.pins()
    table = pins.get("weights") or {}
    return [{"name": name, "sha256": e["sha256"], "size_bytes": e.get("size_bytes"), "url": e.get(URL_KEY)}
            for name, e in table.items() if isinstance(e, dict) and e.get("sha256")]


def transfer(url: str, dest: str, chunk: int = 1 << 20) -> None:
    """``url`` → ``dest``: streamed into ``dest`` + PART_SUFFIX, renamed onto ``dest`` when complete."""
    part = dest + PART_SUFFIX
    with urllib.request.urlopen(url, timeout=60) as r, open(part, "wb") as fh:
        for block in iter(lambda: r.read(chunk), b""):
            fh.write(block)
    os.replace(part, dest)


def fetch(weights_dir: str, files: Optional[List[Dict[str, object]]] = None, fetcher: Optional[Callable[[str, str], None]] = None,
          hasher: Optional[Callable[[str], str]] = None, out=None) -> int:
    """Fetch what is absent, check everything; EXIT_OK when every pinned file in ``weights_dir`` has the pin's size and sha256.
    ``files`` / ``fetcher`` / ``hasher`` default to PINS ``weights``, ``transfer`` and ``stock/check_pins.py`` ``sha256_file``."""
    out = out or sys.stderr
    weights_dir = os.path.abspath(weights_dir)
    os.makedirs(weights_dir, exist_ok=True)
    if files is None or hasher is None:
        from . import stack
        files = pinned_files(stack.pins()) if files is None else files
        hasher = stack._check_pins_module().sha256_file if hasher is None else hasher
    fetcher = fetcher or transfer
    if not files:
        print(f"{PREFIX} REFUSED: stock/PINS.json weights lists no file with a digest — nothing to fetch or check", file=out); return EXIT_FAIL
    for f in files:
        path = os.path.join(weights_dir, str(f["name"]))
        if os.path.isfile(path):
            print(f"{PREFIX} {f['name']}: present", file=out); continue
        if not f.get("url"):
            print(f"{PREFIX} FAILED {f['name']}: absent from {weights_dir} and stock/PINS.json records no URL for it ({URL_KEY})", file=out); return EXIT_FAIL
        print(f"{PREFIX} {f['name']}: fetching{' (%d bytes)' % f['size_bytes'] if f.get('size_bytes') else ''} from {f['url']}", file=out); out.flush()
        try:
            fetcher(str(f["url"]), path)
        except Exception as e:                                       # noqa: BLE001 — the transfer's error, named; a partial .part file is left in place
            print(f"{PREFIX} FAILED fetching {f['name']}: {type(e).__name__}: {e}"
                  f"{' — the partial file ' + path + PART_SUFFIX + ' is left in place' if os.path.exists(path + PART_SUFFIX) else ''}", file=out); return EXIT_FAIL
    bad = []
    for f in files:
        path = os.path.join(weights_dir, str(f["name"]))
        if not os.path.isfile(path):
            bad.append(f"{f['name']} (absent after the transfer)"); continue
        size = os.path.getsize(path)
        if f.get("size_bytes") and size != f["size_bytes"]:
            bad.append(f"{f['name']} ({size} bytes ≠ pin {f['size_bytes']})"); continue
        digest = hasher(path)
        if digest != f["sha256"]:
            bad.append(f"{f['name']} (sha256 {digest[:16]}… ≠ pin {str(f['sha256'])[:16]}…)")
    if bad:
        print(f"{PREFIX} REFUSED: {len(bad)} of {len(files)} files under {weights_dir} do not match stock/PINS.json weights: {'; '.join(bad)} — "
              f"left in place; remove them and re-run", file=out); return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {len(files)}/{len(files)} files under {weights_dir} have the pinned size and digest (stock/PINS.json weights: "
          f"{', '.join(str(f['name']) for f in files)}) — export WEIGHTS={weights_dir}", file=out)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    from ._core_gate import gate
    from .report import TAG as _TAG
    gate(__file__, tag=_TAG)                                          # the core pin gate: statement one of the entry, before any opt_core import (exit 3 by name)
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) != 1 or argv[0].startswith("-"):
        print(f"usage: python -m rfdiffusion1_opt.weights DIR   (run.sh install --weights DIR): fetch the pinned checkpoints into DIR and check them against stock/PINS.json", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
