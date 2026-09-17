"""`run.sh install --weights DIR` — fetch the eight stock weight files into DIR with upstream's own downloader and check each one against
``stock/PINS.json`` ``weights.files`` (sha256). DIR is then the ``CHAI_DOWNLOADS_DIR`` every route reads (README.md 'Install'); nothing is
fetched at run time.

Upstream owns the transfer: ``chai_lab.utils.paths`` reads ``CHAI_DOWNLOADS_DIR`` when it is imported and ``download_if_not_exists`` writes a
file only when it is absent (a temporary file beside it, renamed into place when complete), so a file already in DIR is kept and only checked. The six ``models_v2``
components come through ``paths.chai1_component``, the conformer cache through ``paths.cached_conformers``, the ESM2-3B file through the same
routine with the URL constant of ``chai_lab.data.dataset.embeddings.esm``; no URL is restated here. A file whose digest is not the pin's is
named and the step fails (exit 1) — the file is left in place for inspection, never deleted; so is the temporary file of an interrupted
transfer (a ``*.download_tmp_*`` file beside it; remove it by hand). The comparison is ``stack.weights_match`` — the one the routes' activation
line reports — run afresh, so its digest memo is warm for the first ``check`` / ``warm``. ``python -m chai1_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional

PREFIX = "[chai1-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def upstream_route(downloads_dir: str) -> Callable[[str], Optional[Callable[[], object]]]:
    """``route(local)`` → the upstream call that fetches PINS' ``local`` file into ``downloads_dir`` (None for a file upstream has no entry
    point for). Bound after ``CHAI_DOWNLOADS_DIR`` is set: ``chai_lab.utils.paths`` reads the variable at import."""
    os.environ["CHAI_DOWNLOADS_DIR"] = downloads_dir
    from chai_lab.utils import paths                                   # noqa: E402 — after the variable is set, by design
    root = os.path.realpath(str(paths.downloads_path))
    if root != os.path.realpath(downloads_dir):
        raise RuntimeError(f"chai_lab.utils.paths resolves its downloads root to {root}, not {downloads_dir}: the module was imported before "
                           f"CHAI_DOWNLOADS_DIR was set in this process — run this step in a fresh interpreter (python -m chai1_opt.weights DIR)")

    def esm():
        from chai_lab.data.dataset.embeddings.esm import ESM_URL       # upstream's URL constant (the module imports torch; no model is loaded)
        return paths.download_if_not_exists(ESM_URL, paths.downloads_path.joinpath("esm", os.path.basename(ESM_URL)))

    def route(local: str):
        if local == "conformers_v1.apkl":
            return paths.cached_conformers.get_path                    # upstream's Downloadable for the conformer cache
        if local.startswith("models_v2/") and local.endswith(".pt"):
            return lambda: paths.chai1_component(local.split("/", 1)[1])   # upstream's per-component entry point
        if local.startswith("esm/"):
            return esm
        return None
    return route


def fetch(downloads_dir: str, files: Optional[List[dict]] = None, route: Optional[Callable[[str], Optional[Callable[[], object]]]] = None,
          matcher: Optional[Callable[[str], dict]] = None, out=None) -> int:
    """Fetch what is absent, then check every file's sha256 against the pin. ``files`` (PINS weights.files), ``route`` (upstream's downloader)
    and ``matcher`` (``stack.weights_match``, afresh) are injectable for the package tests."""
    out = out or sys.stdout
    downloads_dir = os.path.abspath(downloads_dir)
    os.makedirs(downloads_dir, exist_ok=True)
    if files is None:
        from . import stack
        files = stack.pins()["weights"]["files"]
    if matcher is None:
        from . import stack
        matcher = lambda d: stack.weights_match(d, afresh=True)          # noqa: E731 — the routes' own comparator, hashed afresh (memo rewritten)
    try:
        if route is None:
            route = upstream_route(downloads_dir)
    except Exception as e:  # noqa: BLE001 — upstream not importable here, or its downloads root already bound elsewhere: named, nothing fetched
        print(f"{PREFIX} FAILED: {type(e).__name__}: {e}", file=out, flush=True)
        return EXIT_FAIL
    for w in files:
        local = w["local"]; target = os.path.join(downloads_dir, local)
        state = "present" if os.path.isfile(target) else "fetching"
        print(f"{PREFIX} {local}: {state}" + (f" ({w.get('bytes', '?')} bytes, upstream's downloader)" if state == "fetching" else ""), file=out, flush=True)
        if state == "present":
            continue
        call = route(local)
        if call is None:
            print(f"{PREFIX} FAILED: upstream chai_lab has no download entry point for {local} (stock/PINS.json weights.files names a file this step cannot fetch)", file=out, flush=True)
            return EXIT_FAIL
        try:
            call()
        except Exception as e:  # noqa: BLE001 — upstream's own error (network, disk, an interrupted transfer), relayed by name; nothing is deleted
            print(f"{PREFIX} FAILED fetching {local}: {type(e).__name__}: {e} — any *.download_tmp_* file beside it is left in place", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.isfile(target):
            print(f"{PREFIX} FAILED: {local} is not at {target} after upstream's fetch", file=out, flush=True)
            return EXIT_FAIL
    wid = matcher(downloads_dir)
    for u in wid.get("unknown", []):
        print(f"{PREFIX} {u['local']}: sha256 {u['sha256'][:16]}… is not the pin's", file=out, flush=True)
    if wid.get("status") != "pinned":
        bad = [u["local"] for u in wid.get("unknown", [])]
        print(f"{PREFIX} REFUSED: {len(bad)} of {wid.get('files', len(files))} files under {downloads_dir} do not have the pinned digest (stock/PINS.json weights.files): "
              f"{', '.join(bad)} — left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {wid.get('files', len(files))}/{len(files)} files under {downloads_dir} have the pinned digest ({wid.get('seconds', '?')} s hashing) — export CHAI_DOWNLOADS_DIR={downloads_dir}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print(f"usage: python -m chai1_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
