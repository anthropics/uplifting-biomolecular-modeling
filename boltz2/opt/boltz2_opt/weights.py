"""`run.sh install --weights DIR` — populate DIR, the Boltz-2 cache directory every route reads as ``BOLTZ_CACHE``, with upstream's own downloader,
then check it: every entry the routes require present (``stack.CACHE_FILES``: boltz2_conf.ckpt, boltz2_aff.ckpt, ccd.pkl, mols/, mols.tar — the
cache gate that otherwise refuses a run by name), the structure checkpoint's sha256 equal to the pin, and every other entry that carries a
digest in ``stock/PINS.json`` ``weights.files`` (boltz2_aff.ckpt, ccd.pkl, mols.tar — ``other_pins``) equal to its own; mols/ is mols.tar
unpacked and is checked for presence. Nothing is fetched at run time.

Upstream owns the transfer: ``boltz.main.download_boltz2(cache)`` writes mols.tar (and unpacks mols/ from it), boltz2_conf.ckpt and boltz2_aff.ckpt
into the cache, each only when absent, from upstream's own URL lists with their fallbacks; ccd.pkl comes from upstream's ``boltz.main.CCD_URL``
through the same ``urllib.request.urlretrieve`` statement upstream uses for it. No URL is restated here. A file already in DIR is kept and only
checked. A checkpoint whose digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never deleted;
so is the partial file of an interrupted transfer (upstream writes straight to the final name: remove that file by hand, then re-run this step).
The digest comparison is ``stack.weights_status`` — the one the routes' WEIGHTS line reports — hashed afresh, so its digest memo
(``weights_digests.json``) is warm for the first ``check`` / ``warm``. ``python -m boltz2_opt.weights DIR``.
"""
from __future__ import annotations

import os
import re
import sys
import urllib.request
from pathlib import Path
from typing import Callable, List, Optional

PREFIX = "[boltz2-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
CCD_FILE = "ccd.pkl"                                                   # the one cache entry download_boltz2 does not write (upstream's CCD_URL names it)


def upstream_route(cache_dir: str) -> Callable[[str], Optional[Callable[[], object]]]:
    """``route(name)`` → the upstream call that writes the cache entry ``name`` into ``cache_dir`` (None for an entry upstream has no source for).
    ``boltz.main`` is imported here, once: its module body imports torch and the model code, nothing is downloaded at import."""
    from boltz import main as upstream                                # noqa: E402 — the pinned upstream CLI module: download_boltz2, CCD_URL
    cache = Path(cache_dir)

    def download_boltz2():
        return upstream.download_boltz2(cache)                         # mols.tar (+ mols/ unpacked), boltz2_conf.ckpt, boltz2_aff.ckpt — each only when absent

    def ccd():
        return urllib.request.urlretrieve(upstream.CCD_URL, str(cache / CCD_FILE))   # upstream's statement for this file (main.py download_boltz1), its URL constant

    written_by_download_boltz2 = {"mols.tar", "mols", "boltz2_conf.ckpt", "boltz2_aff.ckpt"}

    def route(name: str):
        if name in written_by_download_boltz2:
            return download_boltz2
        if name == CCD_FILE:
            return ccd
        return None
    return route


def routes_gate(cache_dir: str) -> List[str]:
    """The routes' own cache gate on ``cache_dir`` (``stack.cache_check`` with BOLTZ_CACHE naming it): [] = every required entry is present."""
    from . import stack
    os.environ["BOLTZ_CACHE"] = cache_dir
    return stack.cache_check()


def other_pins(pins: dict, pinned_file: str) -> dict:
    """{cache entry: sha256} for every ``weights.files`` entry besides ``pinned_file`` that carries a 64-hex digest in stock/PINS.json
    (boltz2_aff.ckpt, ccd.pkl, mols.tar) — the entries this step hashes itself after the fetch."""
    files = (pins.get("weights") or {}).get("files") or {}
    return {name: e.get("sha256") for name, e in files.items()
            if name != pinned_file and isinstance(e, dict) and re.fullmatch(r"[0-9a-f]{64}", str(e.get("sha256") or ""))}


def fetch(cache_dir: str, entries: Optional[List[str]] = None, route: Optional[Callable[[str], Optional[Callable[[], object]]]] = None,
          matcher: Optional[Callable[[str], dict]] = None, gate: Optional[Callable[[str], List[str]]] = None, pinned_file: Optional[str] = None,
          out=None, others: Optional[dict] = None) -> int:
    """Fetch what is absent, then check: every entry present (the routes' gate) and the pinned checkpoint's sha256 equal to the pin. ``entries``
    (stack.CACHE_FILES), ``route`` (upstream's downloader), ``matcher`` (``stack.weights_status``, afresh), ``gate`` (``stack.cache_check``) and
    ``pinned_file`` (stack.WEIGHTS_FILE) are injectable for the package tests; so is ``others`` ({entry: sha256} of the further pinned entries,
    ``other_pins`` of stock/PINS.json on the real route, none when the rest is injected without it)."""
    out = out or sys.stdout
    cache_dir = os.path.abspath(cache_dir)
    os.makedirs(cache_dir, exist_ok=True)
    if entries is None or matcher is None or gate is None or pinned_file is None:
        from . import stack
        entries = list(stack.CACHE_FILES) if entries is None else entries
        matcher = (lambda d: stack.weights_status(d, refresh=True)) if matcher is None else matcher   # noqa: E731 — the routes' own comparator, hashed afresh (memo rewritten)
        gate = routes_gate if gate is None else gate
        pinned_file = stack.WEIGHTS_FILE if pinned_file is None else pinned_file
        others = other_pins(stack.load_pins(), pinned_file) if others is None else others
    others = {} if others is None else others
    try:
        if route is None:
            route = upstream_route(cache_dir)
    except Exception as e:  # noqa: BLE001 — upstream not importable in this environment: named, nothing fetched
        print(f"{PREFIX} FAILED: upstream boltz could not be imported for its downloader: {type(e).__name__}: {e}", file=out, flush=True)
        return EXIT_FAIL
    for name in entries:
        target = os.path.join(cache_dir, name)
        if os.path.exists(target):
            print(f"{PREFIX} {name}: present", file=out, flush=True)
            continue
        print(f"{PREFIX} {name}: absent — fetching with upstream's downloader (it writes every entry the cache lacks; its own lines follow)", file=out, flush=True)
        call = route(name)
        if call is None:
            print(f"{PREFIX} FAILED: upstream boltz has no download source for {name} (a cache entry the routes require that this step cannot fetch)", file=out, flush=True)
            return EXIT_FAIL
        try:
            call()
        except Exception as e:  # noqa: BLE001 — upstream's own error (network, disk, an interrupted transfer), relayed by name; nothing is deleted
            print(f"{PREFIX} FAILED fetching {name}: {type(e).__name__}: {e} — a partial file, if any, is left in place at {target}", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.exists(target):
            print(f"{PREFIX} FAILED: {name} is not at {target} after upstream's fetch", file=out, flush=True)
            return EXIT_FAIL
    missing = gate(cache_dir)
    if missing:
        print(f"{PREFIX} REFUSED: {'; '.join(missing)}", file=out, flush=True)
        return EXIT_FAIL
    w = matcher(cache_dir)
    digest = str(w.get("sha256") or "absent")
    if w.get("status") != "pinned":
        print(f"{PREFIX} REFUSED: {pinned_file} under {cache_dir} has sha256 {digest[:16]}…, not the pin's {str(w.get('pinned_sha256'))[:16]}… "
              f"(stock/PINS.json weights.files) — left in place; remove it and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    from .digest_memo import sha256_file                                   # the kit's one file hasher (the WEIGHTS line's)
    off = []
    for name, want in others.items():
        got = sha256_file(os.path.join(cache_dir, name))
        if got != want:
            off.append(name)
            print(f"{PREFIX} {name}: sha256 {got} is not the pin {want} (stock/PINS.json weights.files) — left in place", file=out, flush=True)
        else:
            print(f"{PREFIX} {name}: sha256 {got[:16]}… = the pin", file=out, flush=True)
    if off:
        print(f"{PREFIX} REFUSED: {len(off)} of {len(others)} further pinned entries under {cache_dir} do not have the pinned digest: {', '.join(off)} "
              f"— left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    note = f"; {w['memo_note']}" if w.get("memo_note") else ""
    more = f"; {len(others)} further entries at their pins ({', '.join(others)})" if others else ""
    print(f"{PREFIX} WEIGHTS OK: {len(entries)}/{len(entries)} entries under {cache_dir} ({', '.join(entries)}); {pinned_file} sha256 {digest[:16]}… "
          f"= the pin{more} (stock/PINS.json weights.files; mols/ is mols.tar unpacked and is checked for presence){note} — export BOLTZ_CACHE={cache_dir}",
          file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m boltz2_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
