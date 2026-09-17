"""`run.sh install --weights DIR` — fetch the Boltz-2 checkpoint and its CCD molecules into DIR with boltz's own downloader and check the
checkpoint against ``stock/PINS.json`` ``weights`` (sha256). DIR is then the ``MOSAIC_CACHE_DIR`` every route reads
(``<DIR>/boltz/boltz2_conf.ckpt`` + ``<DIR>/boltz/mols/``); nothing is fetched at run time — design / check / warm refuse by name
when the checkpoint is absent (`stack.data_path_gate`).

Upstream owns the transfer and the kit's fetch tool owns the call: ``tools/fetch_public_inputs.py ensure_weights(DIR)`` (the same call
``warm`` stages with) runs ``boltz.main.download_boltz2(<DIR>/boltz)`` when the checkpoint or the molecules directory is absent — the public
Boltz hosts, no URL restated here — and hashes ``boltz2_conf.ckpt``; a checkpoint already in DIR is kept and only checked. The digest is
compared with the pin's; a file whose digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection,
never deleted. The CCD molecules arrive as one archive, ``<DIR>/boltz/mols.tar``, which the same downloader unpacks to ``mols/``: the
archive is the pinned unit (``weights.ccd.sha256``) and is hashed and compared the same way — off the pin, the step names it and fails; an
unpacked ``mols/`` from an earlier install with no archive beside it is accepted and said in one line (the archive is checked whenever it
is fetched); the count under ``mols/`` is printed. The ProteinMPNN weights ship inside the pinned `mosaic`
package and are not fetched. ``python -m mosaic_opt.weights DIR``.
"""
from __future__ import annotations

import hashlib
import importlib
import os
import re
import sys
from typing import Callable, Optional

PREFIX = "[mosaic-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
FETCH_TOOL = "fetch_public_inputs"                      # the kit's tools/fetch_public_inputs.py, importable once the recipe accessor has put tools/ on sys.path
PINNED_FILE = "boltz2_conf.ckpt"                        # the checkpoint stock/PINS.json "weights" pins by digest
CCD_ARCHIVE = "mols.tar"                                 # the CCD molecules as boltz's downloader fetches them, beside the checkpoint, unpacked to mols/; pinned by "weights".ccd.sha256


def sha256_file(path: str, block: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(block), b""):
            h.update(chunk)
    return h.hexdigest()


def ccd_pin(pin: dict) -> Optional[str]:
    """The pinned sha256 of the CCD archive ("weights".ccd.sha256), or None when the pin table given carries none (the package tests' own tables)."""
    want = (pin.get("ccd") or {}).get("sha256")
    return want if isinstance(want, str) and re.fullmatch(r"[0-9a-f]{64}", want) else None


def fetch_tool():
    """The kit's fetch tool as a module (``recipe.module()`` places the kit's tools/ directory on sys.path, as the driver has it)."""
    from . import recipe
    recipe.module()
    return importlib.import_module(FETCH_TOOL)


def fetch(weights_dir: str, pin: dict, ensure: Callable[[str], dict], log: Callable[[str], None] = print) -> dict:
    """Stage the weights into ``weights_dir`` through ``ensure`` (the fetch tool's ``ensure_weights``) and judge the checkpoint against
    ``pin`` (stock/PINS.json "weights"), then the CCD archive against its pin when the table carries one (``ccd_pin``):
    ``{"path", "present_before", "fetched_now", "sha256", "want", "match", "n_mols", "mols_dir", "ccd_archive", "ccd_sha256", "ccd_want", "ccd_match"}``."""
    from . import stack
    want = pin[PINNED_FILE]["sha256"]
    path = os.path.join(weights_dir, stack.WEIGHTS_RELPATH)
    present = os.path.isfile(path)
    log(f"{PREFIX} {stack.WEIGHTS_RELPATH}: " + ("present — kept and checked" if present else
        f"fetching into {os.path.dirname(path)} with boltz's own downloader (boltz.main.download_boltz2: the checkpoint and the CCD molecules, about 5 GB)"))
    from pathlib import Path
    rec = ensure(Path(weights_dir))
    got = rec.get("boltz2_conf_ckpt_sha256")
    ccd_want = ccd_pin(pin); archive = os.path.join(os.path.dirname(path), CCD_ARCHIVE)
    ccd_got = sha256_file(archive) if ccd_want and os.path.isfile(archive) else None
    if ccd_want:
        log(f"{PREFIX} boltz/{CCD_ARCHIVE}: " + (f"sha256 {ccd_got}" if ccd_got else "absent — the CCD molecules cannot be compared with the pin"))
    return {"path": path, "present_before": present, "fetched_now": bool(rec.get("fetched_now")), "sha256": got, "want": want,
            "match": got == want, "n_mols": int(rec.get("n_mols") or 0), "mols_dir": os.path.join(os.path.dirname(path), "mols"),
            "ccd_archive": archive, "ccd_sha256": ccd_got, "ccd_want": ccd_want, "ccd_match": ccd_want is None or ccd_got == ccd_want}


def main(argv=None, ensure: Optional[Callable] = None, pin: Optional[dict] = None) -> int:
    argv = sys.argv[1:] if argv is None else list(argv)
    if len(argv) != 1 or not argv[0] or argv[0].startswith("-"):
        print(f"usage: python -m mosaic_opt.weights DIR   (the MOSAIC_CACHE_DIR to fill: <DIR>/boltz/{PINNED_FILE} + <DIR>/boltz/mols/)", file=sys.stderr)
        return EXIT_USAGE
    from . import stack
    weights_dir = os.path.abspath(argv[0])
    os.makedirs(weights_dir, exist_ok=True)
    os.environ[stack.LIBRARY_CACHE_ENV] = weights_dir                      # the library's own cache variable names the same directory for anything imported below
    pin = stack.pins()["weights"] if pin is None else pin
    if ensure is None:
        ensure = fetch_tool().ensure_weights
    try:
        res = fetch(weights_dir, pin, ensure)
    except Exception as e:                                              # the downloader's own failure (network, disk): its words, exit 1; whatever landed stays in place
        print(f"{PREFIX} FAILED: the fetch did not complete ({type(e).__name__}: {e}); files already under {weights_dir} are left in place", file=sys.stderr)
        return EXIT_FAIL
    if not res["match"]:
        print(f"{PREFIX} REFUSED: {res['path']} sha256 {res['sha256']} is not the pin's {res['want']} (stock/PINS.json \"weights\"); "
              f"the file is left in place for inspection — remove it and run the step again to fetch afresh", file=sys.stderr)
        return EXIT_FAIL
    if not res["ccd_match"]:
        if res["ccd_sha256"] is None and res["n_mols"] == 0:
            print(f"{PREFIX} REFUSED: {res['ccd_archive']} is absent and {res['mols_dir']} holds no CCD molecules, so nothing can be compared with the pin "
                  f"(stock/PINS.json \"weights\".ccd: the archive boltz's downloader fetches and unpacks); remove {res['mols_dir']} and run the step again to fetch afresh", file=sys.stderr)
            return EXIT_FAIL
        if res["ccd_sha256"] is not None:
            print(f"{PREFIX} REFUSED: {res['ccd_archive']} sha256 {res['ccd_sha256']} is not the pin's {res['ccd_want']} (stock/PINS.json \"weights\".ccd); "
                  f"the archive and {res['mols_dir']} are left in place for inspection — remove both and run the step again to fetch afresh", file=sys.stderr)
            return EXIT_FAIL
    fetched = ', fetched now' if res['fetched_now'] else ''
    if res["ccd_want"] and res["ccd_sha256"] is None:                    # an earlier install's unpacked molecules without the archive beside them: accepted, said in one line
        print(f"{PREFIX} WEIGHTS OK: 1/1 pinned file matches stock/PINS.json ({stack.WEIGHTS_RELPATH}{fetched}); CCD molecules: {res['n_mols']} files under {res['mols_dir']} "
              f"present from an earlier install without the archive boltz/{CCD_ARCHIVE} beside them — the archive's digest is checked whenever the archive is fetched; "
              f"remove {res['mols_dir']} to re-fetch and verify; MOSAIC_CACHE_DIR={weights_dir}", flush=True)
        return EXIT_OK
    if res["ccd_want"]:
        print(f"{PREFIX} WEIGHTS OK: 2/2 pinned files match stock/PINS.json ({stack.WEIGHTS_RELPATH}, boltz/{CCD_ARCHIVE}{fetched}); "
              f"CCD molecules: {res['n_mols']} files under {res['mols_dir']} (the archive they come in is the pinned unit); MOSAIC_CACHE_DIR={weights_dir}", flush=True)
    else:
        print(f"{PREFIX} WEIGHTS OK: 1/1 pinned file matches stock/PINS.json ({stack.WEIGHTS_RELPATH}{fetched}); "
              f"CCD molecules: {res['n_mols']} files under {res['mols_dir']} (no digest is pinned for them); MOSAIC_CACHE_DIR={weights_dir}", flush=True)
    return EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
