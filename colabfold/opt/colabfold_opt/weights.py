"""`run.sh install --weights DIR` — fetch the AlphaFold2-Multimer v3 parameters into DIR with upstream's own downloader and check the five
files against ``stock/PINS.json`` ``weights.files`` (sha256). DIR is then the ``COLABFOLD_OPT_DATA_DIR`` every route reads (README.md Setup);
nothing is fetched at run time.

Upstream owns the transfer: ``colabfold.download.download_alphafold_params(<model type>, DIR)`` (the model type is PINS ``weights.model_type``,
``alphafold2_multimer_v3``) streams its one archive into ``DIR/params/`` and then writes the empty marker file PINS ``weights.marker`` names;
with the marker present it returns at once and fetches nothing — its own rule — so a populated DIR is only checked. The unit is the archive:
there is no per-file fetch, files of the same names under ``DIR/params/`` are rewritten by upstream's extraction, and the archive also carries
upstream's other AlphaFold2 parameter files, which land beside the five and are left alone. No URL is restated here. A file whose digest is not
the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never deleted; a pinned file missing while the marker
is present is named too (remove the marker and re-run: upstream then fetches afresh). The comparison is ``stack.weights_check`` — the one the
routes' weights lines report — run afresh (``refresh=True``), so the digest memo is warm for the first ``check`` / ``warm``.
``python -m colabfold_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional

PREFIX = "[colabfold-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def upstream_route(data_dir: str) -> Callable[[str], object]:
    """``route(model_type)`` → upstream's download of that model type's parameters into ``data_dir`` (``colabfold.download``: the archive into
    ``<data_dir>/params``, then the marker). Importing it loads requests / tqdm / appdirs only — no jax, no model code."""
    from pathlib import Path
    from colabfold.download import download_alphafold_params            # noqa: PLC0415 — upstream's own downloader, imported when a fetch is due
    return lambda model_type: download_alphafold_params(model_type, Path(data_dir))


def fetch(data_dir: str, pins: Optional[dict] = None, route: Optional[Callable[[str], object]] = None,
          checker: Optional[Callable[[str], tuple]] = None, out=None) -> int:
    """Fetch through upstream when its marker is absent, then check every pinned file's sha256. ``pins`` (stock/PINS.json), ``route`` (upstream's
    downloader) and ``checker`` (``stack.weights_check``, afresh: ``(reasons, report)``) are injectable for the package tests."""
    out = out or sys.stdout
    data_dir = os.path.abspath(data_dir)
    os.makedirs(data_dir, exist_ok=True)
    if pins is None or checker is None:
        from . import _package                                              # noqa: PLC0415 — the package's guarded import: the pinned shared core, or NOT ACTIVE by name and exit 3
        _modes, stack = _package()
        pins = stack.pins() if pins is None else pins
        checker = (lambda d: stack.weights_check(d, refresh=True)) if checker is None else checker   # noqa: E731 — the routes' own comparator, hashed afresh (memo rewritten)
    w = pins["weights"]; files = list(w["files"]); marker_rel = w["marker"]["file"]; model_type = w["model_type"]
    marker = os.path.join(data_dir, marker_rel)
    for rel in files:
        print(f"{PREFIX} {rel}: {'present' if os.path.isfile(os.path.join(data_dir, rel)) else 'absent'}", file=out, flush=True)
    if os.path.isfile(marker):
        print(f"{PREFIX} {marker_rel}: present — upstream's marker of a finished download: nothing is fetched, the files are checked", file=out, flush=True)
    else:
        total = sum(int(spec.get("bytes", 0)) for spec in w["files"].values())
        print(f"{PREFIX} {marker_rel}: absent — fetching the {model_type} parameters into {os.path.join(data_dir, 'params')} with upstream's downloader "
              f"(colabfold.download: upstream's one archive — the five pinned files, {total / 1e9:.1f} GB, inside it beside upstream's other AlphaFold2 parameter files; files of the same names are rewritten)", file=out, flush=True)
        try:
            if route is None:
                route = upstream_route(data_dir)
            route(model_type)
        except Exception as e:  # noqa: BLE001 — upstream not importable here, or its own error (network, disk, an interrupted transfer), relayed by name; nothing is deleted
            print(f"{PREFIX} FAILED fetching the {model_type} parameters: {type(e).__name__}: {e} — whatever upstream wrote under {os.path.join(data_dir, 'params')} is left in place", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.isfile(marker):
            print(f"{PREFIX} FAILED: upstream's download returned but its marker {marker} is absent (the transfer did not finish)", file=out, flush=True)
            return EXIT_FAIL
    reasons, rep = checker(data_dir)
    bad = []
    for rel in files:
        d = (rep.get("files") or {}).get(rel) or {}
        state = d.get("state", "missing")
        if state == "pinned":
            print(f"{PREFIX} {rel}: sha256 {d['sha256'][:16]}… (pinned)", file=out, flush=True)
        elif state == "missing":
            print(f"{PREFIX} {rel}: MISSING (the marker says the download finished; remove {marker} and re-run this step to fetch afresh)", file=out, flush=True); bad.append(rel)
        else:
            print(f"{PREFIX} {rel}: sha256 {str(d.get('sha256', '?'))[:16]}… is not the pin's ({d.get('bytes', '?')} bytes)", file=out, flush=True); bad.append(rel)
    if reasons or bad or rep.get("pinned") is not True:
        for r in reasons:
            print(f"{PREFIX} {r}", file=out, flush=True)
        print(f"{PREFIX} REFUSED: {len(bad)} of {len(files)} parameter files under {data_dir} are missing or do not have the pinned digest (stock/PINS.json weights.files): "
              f"{', '.join(bad) or '-'} — left in place; remove them (and the marker, so upstream fetches afresh) and re-run this step", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {len(files)}/{len(files)} parameter files under {data_dir} have the pinned digest and the marker is present — export {w['env']}={data_dir}", file=out, flush=True)
    return EXIT_OK


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m colabfold_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
