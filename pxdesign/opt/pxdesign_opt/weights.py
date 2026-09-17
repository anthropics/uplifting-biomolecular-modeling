"""`run.sh install --weights DIR` — fetch the seven files a PXDesign run reads into DIR with upstream's own downloader and check each one against
``stock/PINS.json`` (sha256): the checkpoint ``pxdesign_v0.1.0.pt`` and the three Protenix checkpoints upstream requires beside it into
``DIR/checkpoint`` (``weights.checkpoint``, ``weights.required_in_dir``), the CCD cache into ``DIR/ccd_cache`` (``ccd_cache.files``). Those two
directories are then the ``PXDESIGN_CKPT_DIR`` and ``PROTENIX_DATA_ROOT_DIR`` every route reads (README.md 'Setup'); nothing is fetched at run
time.

Upstream owns the transfer: ``pxdesign.utils.infer.download_inference_cache(configs)`` writes each file that is absent (straight to its final
path, with its own URL table ``pxdesign.utils.infer.URL``) and skips a file already present, so a file in DIR is kept and only checked. The
four checkpoints are the exception this step transfers itself, first (``prefetch_checkpoints``): upstream ``torch.load``s a checkpoint it has
just downloaded, before any digest could be compared, so each absent checkpoint comes from upstream's own URL for its name into
``<file>.part``, its sha256 is compared with the pin, and only a file with the pinned digest is renamed into place — upstream's routine then
finds it present and neither downloads nor loads it; a transfer with any other digest stays at ``.part``, named, never loaded, and the
step fails. The ``configs`` upstream's routine takes are upstream's data-cache paths under
``PROTENIX_DATA_ROOT_DIR`` (``pxdesign.configs.configs_data.data_configs``, read when that module is imported — the variable is set to
``DIR/ccd_cache`` first, in this process), the checkpoint directory and the model name of the pin; no URL and no file name is restated here.
A file whose digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never deleted; an
interrupted transfer leaves a short file at its final path, which the digest check refuses the same way (remove it and re-run).
``python -m pxdesign_opt.weights DIR``.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
import urllib.request
from typing import Callable, Dict, List, Optional, Tuple

PREFIX = "[pxdesign-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
CKPT_SUBDIR, CCD_SUBDIR = "checkpoint", "ccd_cache"          # DIR/checkpoint = PXDESIGN_CKPT_DIR, DIR/ccd_cache = PROTENIX_DATA_ROOT_DIR (README.md 'Install')


def pinned_files(pins: dict) -> List[dict]:
    """The seven files of the pin as [{local, sha256, bytes}] — ``local`` relative to DIR: the checkpoint and its three companions under
    checkpoint/, the CCD cache under ccd_cache/ (stock/PINS.json weights.checkpoint, weights.required_in_dir, ccd_cache)."""
    w = pins["weights"]
    out = [{"local": os.path.join(CKPT_SUBDIR, w["checkpoint"]["file"]), "sha256": w["checkpoint"]["sha256"], "bytes": w["checkpoint"].get("bytes", "?")}]
    req = w.get("required_in_dir") or {}
    for name, digest in req.get("files", {}).items():
        out.append({"local": os.path.join(CKPT_SUBDIR, name), "sha256": digest, "bytes": (req.get("bytes") or {}).get(name, "?")})
    for name, digest in pins["ccd_cache"]["files"].items():
        out.append({"local": os.path.join(CCD_SUBDIR, name), "sha256": digest, "bytes": "?"})
    return out


def upstream_fetcher(root: str, model_name: str) -> Callable[[], object]:
    """The upstream call that fetches every absent file of the set into ``root``/checkpoint and ``root``/ccd_cache. Bound after
    ``PROTENIX_DATA_ROOT_DIR`` is set: ``pxdesign`` and its ``configs_data`` read the variable at import."""
    ckpt_dir, ccd_dir = os.path.join(root, CKPT_SUBDIR), os.path.join(root, CCD_SUBDIR)
    os.environ["PROTENIX_DATA_ROOT_DIR"] = ccd_dir
    from ml_collections.config_dict import ConfigDict                  # noqa: E402 — upstream's own configs type
    from pxdesign.configs.configs_data import data_configs             # noqa: E402 — after the variable is set, by design
    from pxdesign.utils import infer as upstream                       # noqa: E402 — imports torch and protenix; reads nothing from the two directories
    bound = os.path.realpath(os.path.dirname(data_configs["ccd_components_file"]))
    if bound != os.path.realpath(ccd_dir):
        raise RuntimeError(f"pxdesign.configs.configs_data resolves its CCD cache to {bound}, not {ccd_dir}: the module was imported before "
                           f"PROTENIX_DATA_ROOT_DIR was set in this process — run this step in a fresh interpreter (python -m pxdesign_opt.weights DIR)")
    configs = ConfigDict({"data": dict(data_configs), "load_checkpoint_dir": ckpt_dir, "model_name": model_name})
    return lambda: upstream.download_inference_cache(configs)


def digest(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def check(root: str, files: List[dict]) -> Tuple[List[dict], float]:
    """[{local, sha256 (found), pinned}] for every file of the set under ``root`` ('absent' for a missing file), and the seconds spent hashing."""
    t0 = time.time(); rows = []
    for f in files:
        p = os.path.join(root, f["local"])
        found = digest(p) if os.path.isfile(p) else "absent"
        rows.append({"local": f["local"], "sha256": found, "pinned": found == f["sha256"]})
    return rows, round(time.time() - t0, 1)


def prefetch_checkpoints(root: str, files: List[dict], retrieve: Optional[Callable[[str, str], object]] = None,
                         url_for: Optional[Callable[[str], str]] = None, out=None) -> Optional[str]:
    """Every absent file of the set under checkpoint/ — the files upstream would ``torch.load`` right after downloading them — transferred by this
    step before upstream's routine runs: upstream's URL for the file's model name (``pxdesign.utils.infer.URL``, the table its downloader reads),
    ``urllib.request.urlretrieve`` (upstream's call) into ``<file>.part``, that file's sha256 compared with the pin, then the rename to the
    checkpoint's name — so nothing reads a checkpoint before its digest is the pin's. Returns None with every absent checkpoint in place and
    pinned, else the line naming the first refusal (its ``.part`` file left where it is, the rest not transferred). ``retrieve`` / ``url_for``
    are injectable for the package tests."""
    out = out or sys.stdout
    for f in files:
        final = os.path.join(root, f["local"])
        if os.path.dirname(f["local"]) != CKPT_SUBDIR or os.path.isfile(final):
            continue
        if url_for is None:
            from pxdesign.utils.infer import URL                           # upstream's URL table, keyed by model name; no URL is restated here
            url_for = URL.__getitem__
        name = os.path.basename(f["local"])
        url = url_for(name[:-3] if name.endswith(".pt") else name)
        part = final + ".part"
        print(f"{PREFIX} {f['local']}: transferring {url} to {f['local']}.part — compared with the pin before it takes the checkpoint's name", file=out, flush=True)
        (retrieve or urllib.request.urlretrieve)(url, part)
        found = digest(part)
        if found != f["sha256"]:
            return (f"{f['local']}.part sha256 {found} is not the pin {f['sha256']} (stock/PINS.json weights): the transfer from {url} is left at {part} "
                    f"for inspection and was never loaded — remove it and re-run this step")
        os.replace(part, final)
        print(f"{PREFIX} {f['local']}: sha256 {found[:16]}… = the pin; in place before upstream's routine runs", file=out, flush=True)
    return None


def fetch(root: str, files: Optional[List[dict]] = None, fetcher: Optional[Callable[[], object]] = None, out=None,
          prefetch: Optional[Callable[..., Optional[str]]] = None) -> int:
    """Fetch what is absent — absent checkpoints first, by ``prefetch`` (``prefetch_checkpoints``: each compared with its pin before it is in
    place), the rest by ``fetcher`` (upstream's downloader, ``upstream_fetcher``) — then check every file's sha256 against the pin. ``files``
    (the pin's table, ``pinned_files``), ``fetcher`` and ``prefetch`` are injectable for the package tests (an injected ``fetcher`` writes
    every file unless a ``prefetch`` is injected with it)."""
    out = out or sys.stdout
    root = os.path.abspath(root)
    if files is None or fetcher is None:
        from .options import read_pins
        pins = read_pins()
        files = files if files is not None else pinned_files(pins)
    for d in (CKPT_SUBDIR, CCD_SUBDIR):
        os.makedirs(os.path.join(root, d), exist_ok=True)
    absent = []
    for f in files:
        state = "present" if os.path.isfile(os.path.join(root, f["local"])) else "fetching"
        print(f"{PREFIX} {f['local']}: {state}" + (f" ({f.get('bytes', '?')} bytes, upstream's downloader)" if state == "fetching" else ""), file=out, flush=True)
        if state == "fetching":
            absent.append(f["local"])
    if absent:
        try:
            if fetcher is None:
                fetcher = upstream_fetcher(root, pins["weights"]["model_name"])
                prefetch = prefetch_checkpoints if prefetch is None else prefetch
            refused = prefetch(root, files, out=out) if prefetch is not None else None
            if refused:
                print(f"{PREFIX} REFUSED: {refused}", file=out, flush=True)
                return EXIT_FAIL
            fetcher()
        except Exception as e:  # noqa: BLE001 — upstream not importable here, or upstream's own error (network, disk): relayed by name; nothing is deleted
            print(f"{PREFIX} FAILED fetching {', '.join(absent)}: {type(e).__name__}: {e} — any partly written file is left in place", file=out, flush=True)
            return EXIT_FAIL
        still = [a for a in absent if not os.path.isfile(os.path.join(root, a))]
        if still:
            print(f"{PREFIX} FAILED: {', '.join(still)} not under {root} after upstream's fetch (upstream pxdesign has no download entry for a file stock/PINS.json names)", file=out, flush=True)
            return EXIT_FAIL
    rows, seconds = check(root, files)
    bad = [r for r in rows if not r["pinned"]]
    for r in bad:
        print(f"{PREFIX} {r['local']}: sha256 {r['sha256'][:16]}… is not the pin's", file=out, flush=True)
    if bad:
        print(f"{PREFIX} REFUSED: {len(bad)} of {len(rows)} files under {root} do not have the pinned digest (stock/PINS.json weights / ccd_cache): "
              f"{', '.join(r['local'] for r in bad)} — left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {len(rows)}/{len(files)} files under {root} have the pinned digest ({seconds} s hashing) — "
          f"export PXDESIGN_CKPT_DIR={os.path.join(root, CKPT_SUBDIR)} PROTENIX_DATA_ROOT_DIR={os.path.join(root, CCD_SUBDIR)}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m pxdesign_opt.weights DIR   (run.sh install --weights DIR; fills DIR/checkpoint and DIR/ccd_cache)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
