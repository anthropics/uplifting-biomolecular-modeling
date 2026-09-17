"""`run.sh install --weights DIR [--model M]` — fetch the ProGen2 checkpoints into DIR the way upstream's README does and check every file against
``stock/PINS.json`` (sha256). DIR is then the ``PROGEN2_WEIGHTS`` root every route reads (STOCK.md 'Weights'); nothing is fetched at run time.

Upstream ships no downloader code: ``stock/src/progen2/README.md`` fetches one archive per size with ``wget -P checkpoints/${model}
https://storage.googleapis.com/sfr-progen-research/checkpoints/${model}.tar.gz`` and unpacks it in place with ``tar -xvf``. Those URLs are the ones
``stock/PINS.json`` records per size (``weights.<progen2-size>.url``, with the archive's own sha256 and its two members); this module is that
wget + tar in the standard library: the archive streams to ``DIR/<progen2-size>/<progen2-size>.tar.gz.part``, is renamed when the transfer completes,
is refused by name when its digest is not the pin's (kept for inspection), else its two members ``pytorch_model.bin`` and ``config.json`` are unpacked
beside it and the archive is removed. A size whose two files are already in DIR is kept and only checked. Sizes: ``--model`` (the stock's checkpoint
name, `progen2-small` … `progen2-xlarge`) names one; without it, all seven (29.5 GB unpacked; the largest archive, xlarge, needs 25 GB free while it unpacks).

The check is the kit's one weights checker, ``stock/check_pins.py --weights-dir DIR --hash-weights --size …`` (the bytes hashed, in a clean
interpreter — what ``run.sh check`` and the routes' activation consult): a file whose digest is not the pin's is named and the step fails (exit 1);
the file is left in place for inspection, never deleted, and so is the ``.part`` file of an interrupted transfer (a re-run starts that transfer
afresh). The pins are ``stock/PINS.json`` ``weights.<size>.files`` (``pytorch_model.bin`` of every size; ``config.json`` and the archive where the file carries their
digest). Once a size's files check, ``SHA256SUMS`` (the pinned digests, ``sha256sum`` format) is written beside them, the layout later checks read
instead of re-hashing. ``python -m progen2_opt.weights DIR [--model M]``. Exit: 0 every file at its pin · 1 a transfer or a digest failed · 2 usage.
"""
from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import tarfile
import time
from typing import Callable, List, Optional, Sequence, Tuple

from . import modes, stack

PREFIX = "[progen2-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
SUMS_NAME = "SHA256SUMS"                      # beside a size's files: `<sha256>  <file>` per line (stock/check_pins.py listed_sha reads it)
CHUNK = 1 << 24


class Refused(Exception):
    """A transfer or an archive that is not the pin's — named, the step fails, nothing is deleted."""


def sizes_for(variant: Optional[str]) -> List[str]:
    """The upstream checkpoint names to fetch: the variant's one, or all seven in the stock order."""
    return [modes.UPSTREAM_NAME[variant]] if variant else [modes.UPSTREAM_NAME[v] for v in modes.VARIANTS]


def download(url: str, dest: str, out=None) -> str:
    """Stream ``url`` to ``dest`` (through ``dest.part``, renamed when complete); returns the sha256 of the bytes written."""
    import urllib.request
    out = out or sys.stdout
    part = dest + ".part"
    h = hashlib.sha256(); n = 0; t0 = time.time(); mark = 0
    with urllib.request.urlopen(url) as r, open(part, "wb") as fh:
        while True:
            b = r.read(CHUNK)
            if not b:
                break
            fh.write(b); h.update(b); n += len(b)
            if n - mark >= (1 << 30):
                mark = n; print(f"{PREFIX}   … {n / 1e9:.1f} GB in {time.time() - t0:.0f}s", file=out, flush=True)
    os.replace(part, dest)
    return h.hexdigest()


def unpack(archive: str, size_dir: str, names: Sequence[str]) -> List[str]:
    """Unpack the members of ``archive`` whose base name is in ``names`` into ``size_dir`` (each through a `.part` file); returns the names written."""
    written = []
    with tarfile.open(archive, "r:*") as tf:
        for m in tf:
            base = os.path.basename(m.name)
            if not m.isfile() or base not in names:
                continue
            dst = os.path.join(size_dir, base)
            src = tf.extractfile(m)
            with open(dst + ".part", "wb") as fh:
                while True:
                    b = src.read(CHUNK)
                    if not b:
                        break
                    fh.write(b)
            os.replace(dst + ".part", dst)
            written.append(base)
    return written


def pins_check(weights_dir: str, sizes: Sequence[str]) -> Tuple[int, str]:
    """``stock/check_pins.py`` on the weights alone, the bytes hashed: (rc, its output). The one weights checker of the tree, in a clean interpreter."""
    script = os.path.join(stack.tree_home(), "stock", "check_pins.py")
    cmd = [stack.python(), "-I", script, "--skip", "interpreter", "--skip", "packages", "--skip", "stock-files",
           "--weights-dir", weights_dir, "--hash-weights"]
    for s in sizes:
        cmd += ["--size", s]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=stack.PROCESS_TIMEOUT_S)
    return r.returncode, (r.stdout + r.stderr)


def write_sums(size_dir: str, files: dict) -> None:
    """``SHA256SUMS`` beside a size's checked files: the pinned digests in `sha256sum` format."""
    with open(os.path.join(size_dir, SUMS_NAME), "w", encoding="utf-8") as fh:
        for name, pin in files.items():
            if pin.get("sha256"):
                fh.write(f"{pin['sha256']}  {name}\n")


def fetch(weights_dir: str, sizes: Sequence[str], pins: Optional[dict] = None, out=None,
          transfer: Optional[Callable[[str, str], str]] = None, checker: Optional[Callable[[str, Sequence[str]], Tuple[int, str]]] = None) -> int:
    """Fetch what is absent, then check every file's sha256 against the pin. ``transfer(url, dest) -> sha256`` (the download) and
    ``checker(dir, sizes) -> (rc, text)`` (stock/check_pins.py) are injectable for the package tests."""
    out = out or sys.stdout
    pins = pins or stack.pins()
    transfer = transfer or (lambda url, dest: download(url, dest, out))
    checker = checker or pins_check
    weights_dir = os.path.abspath(weights_dir)
    os.makedirs(weights_dir, exist_ok=True)
    n_files = n_pinned = 0; failed = []
    print(f"{PREFIX} weights root {weights_dir}: sizes {', '.join(sizes)}", file=out, flush=True)
    for size in sizes:
        pin = pins["weights"][size]; files = pin["files"]; n_files += len(files); n_pinned += sum(1 for f in files.values() if f.get("sha256"))
        size_dir = os.path.join(weights_dir, size)
        present = [f for f in files if os.path.isfile(os.path.join(size_dir, f))]
        if len(present) == len(files):
            print(f"{PREFIX} {size}: {', '.join(files)} present — kept, checked below", file=out, flush=True)
            continue
        os.makedirs(size_dir, exist_ok=True)
        url = pin["url"]; archive = os.path.join(size_dir, os.path.basename(url)); tar_pin = pin.get("tarball") or {}
        try:
            gb = f" ({tar_pin['size_bytes'] / 1e9:.1f} GB)" if tar_pin.get("size_bytes") else ""
            print(f"{PREFIX} {size}: fetching {url}{gb} -> {size_dir}/", file=out, flush=True)
            t0 = time.time()
            try:
                got = transfer(url, archive)
            except Refused:
                raise
            except Exception as e:                                            # noqa: BLE001 — the transfer's own words, named
                raise Refused(f"{size}: transfer of {url} failed: {type(e).__name__}: {e} (a partial {os.path.basename(archive)}.part may remain in {size_dir})")
            if tar_pin.get("sha256") and got != tar_pin["sha256"]:
                raise Refused(f"{size}: {archive} sha256 {got} is not the pin's {tar_pin['sha256']} — kept for inspection, nothing unpacked")
            written = unpack(archive, size_dir, list(files))
            missing = [f for f in files if f not in written]
            if missing:
                raise Refused(f"{size}: {archive} holds no member named {', '.join(missing)} — kept for inspection")
            os.remove(archive)
            print(f"{PREFIX} {size}: fetched and unpacked in {time.time() - t0:.0f}s ({', '.join(written)}); archive removed", file=out, flush=True)
        except Refused as e:
            failed.append(str(e)); print(f"{PREFIX} FAILED {e}", file=out, flush=True)
    if failed:
        print(f"{PREFIX} REFUSED: {len(failed)} of {len(sizes)} sizes not fetched (above); the files already in {weights_dir} are left as they are", file=out, flush=True)
        return EXIT_FAIL
    rc, text = checker(weights_dir, sizes)
    for line in text.splitlines():
        print(f"{PREFIX} check_pins: {line}", file=out, flush=True)
    if rc != 0:
        print(f"{PREFIX} REFUSED: stock/check_pins.py exit {rc} — a file above is not at its pin (left in place for inspection)", file=out, flush=True)
        return EXIT_FAIL
    for size in sizes:
        write_sums(os.path.join(weights_dir, size), pins["weights"][size]["files"])
    print(f"{PREFIX} WEIGHTS OK: {n_pinned}/{n_pinned} pinned files at their pins, {n_files} files present under {weights_dir} "
          f"({len(sizes)} size{'s' if len(sizes) != 1 else ''}; {SUMS_NAME} written beside each) — export PROGEN2_WEIGHTS={weights_dir}", file=out, flush=True)
    return EXIT_OK


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m progen2_opt.weights", description=__doc__.splitlines()[0], allow_abbrev=False)
    p.add_argument("weights_dir", metavar="DIR", help="the weights root to fill: <DIR>/<progen2-size>/{pytorch_model.bin, config.json}")
    p.add_argument("--model", default=None, help=f"one size, by the stock's checkpoint name ({'|'.join(modes.MODEL_NAMES)}); default: all seven")
    a = p.parse_args(argv)
    variant = None
    if a.model is not None:
        try:
            variant = modes.variant_of_model(a.model)
        except ValueError as e:
            print(f"{PREFIX} usage: {e}", file=sys.stderr)
            return EXIT_USAGE
    return fetch(a.weights_dir, sizes_for(variant))


if __name__ == "__main__":
    sys.exit(main())
