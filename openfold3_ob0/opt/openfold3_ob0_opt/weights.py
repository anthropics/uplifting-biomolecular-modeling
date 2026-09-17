"""`run.sh install [--weights DIR] [--ccd FILE]` — the two fetched inputs of this kit, each placed by upstream's own setup code (or, for
the CCD, from a file fetched beforehand) and checked against ``stock/PINS.json``; nothing is fetched at run time.

``--ccd``: upstream's full Chemical Component Dictionary. ``openfold3.setup_openfold.setup_biotite_ccd(ccd_path=biotite.setup_ccd.OUTPUT_CCD,
force_download=False)`` — the call ``setup_openfold`` makes — writes ``s3://openfold3-data/components.bcif`` over the CCD subset biotite ships
inside its own package directory; the file is then hashed against PINS ``ccd`` (sha256 + bytes). A file already at the pin is only hashed:
no network, no write (a read-only image keeps working). An environment that skipped this step runs stock on biotite's subset silently, which
is why the install step owns it. ``--ccd FILE`` (a machine without network access): FILE, fetched beforehand from the pin's source, is hashed
against the same pin FIRST — any other digest is refused by name and nothing is written — then copied to biotite's CCD path in place of
upstream's fetch (``CCD placed from --ccd FILE sha256=<digest> (no fetch)``); upstream's placement call is bound but never made.

``DIR``: the OpenBind-0 checkpoint. ``openfold3.entry_points.parameters.download_model_parameters`` fetches the registry entry whose file
name is the pin's (``skip_confirmation=True``: no prompt) into DIR; a file already in DIR is kept and only checked. The comparison is
``stack.weights_gate`` — the one the routes' WEIGHTS line reports — run afresh, so its digest memo is warm for the first ``check`` / ``warm``.
DIR/<file> is then what ``OPENFOLD3_OB0_CKPT`` (or ``--ckpt``) names.

A digest off the pin is named and the step fails (exit 1); files are left in place for inspection, never deleted — so is the partial file
of an interrupted transfer (remove it by hand and re-run). No URL, bucket or file name is restated here: upstream's constants and the pin
are the only sources. ``python -m openfold3_ob0_opt.weights --ccd [FILE] | DIR``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Optional, Tuple

PREFIX = "[openfold3_ob0-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


# ------------------------------------------------------------------------------------------------------------------- ccd ----
def upstream_ccd() -> Tuple[str, Callable[[], object]]:
    """(the CCD path inside the installed biotite, upstream's placement call bound to it) — ``setup_openfold``'s own step."""
    import biotite.setup_ccd
    from openfold3.setup_openfold import setup_biotite_ccd
    path = biotite.setup_ccd.OUTPUT_CCD
    return str(path), (lambda: setup_biotite_ccd(ccd_path=path, force_download=False))


def place_file(src: str, dst: str) -> None:
    """Copy ``src`` to ``dst`` the way upstream's fetch lands its file (``download_s3_file``: the parent directory made, the bytes written whole):
    a sibling temporary then an atomic rename, so an interrupted copy never leaves a truncated dictionary at biotite's path."""
    import shutil
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    tmp = dst + ".part"
    shutil.copyfile(src, tmp)
    os.replace(tmp, dst)


def ccd(home: Optional[str] = None, located=None, hasher=None, out=None, source: Optional[str] = None) -> int:
    """Place (when absent or off the pin) and check upstream's CCD; EXIT_OK when the file at biotite's CCD path has the pinned digest and size.
    ``source`` (``--ccd FILE``): a dictionary fetched beforehand — hashed against the pin first (refused by name on any other digest, nothing
    written), then copied to biotite's CCD path by ``place_file``; upstream's fetch never runs."""
    out = out or sys.stdout
    from . import env as _env, manifest as _manifest
    home = home or _env.tree_home()
    pin = _env.pins(home).get("ccd") or {}
    if not pin.get("sha256") or not pin.get("bytes"):
        print(f"{PREFIX} FAILED: stock/PINS.json carries no ccd sha256 / bytes to check against", file=out); return EXIT_FAIL
    hasher = hasher or (lambda p: _manifest.sha256_file(p, strict=True, home=home))
    try:
        path, place = (located or upstream_ccd)()
    except Exception as e:  # noqa: BLE001 — biotite / openfold3 not importable: the stack is not installed as README.md 'Setup' says
        print(f"{PREFIX} FAILED: upstream's CCD step could not be bound ({type(e).__name__}: {e}) — install the pinned stack first (README.md 'Install')", file=out); return EXIT_FAIL
    pinned = f"sha256 {pin['sha256'][:12]}…, {pin['bytes']} bytes; stock/PINS.json ccd"

    def at_pin(p: str) -> Tuple[bool, str, Optional[str]]:
        if not os.path.isfile(p):
            return False, "absent", None
        size = os.path.getsize(p); digest = hasher(p)
        return (digest == pin["sha256"] and size == pin["bytes"]), f"sha256 {digest[:12]}…, {size} bytes", digest

    src = None
    if source is not None:                                              # --ccd FILE: the file is judged BEFORE anything is written
        src = os.path.abspath(source)
        ok_src, seen_src, digest_src = at_pin(src)
        if digest_src is None:
            print(f"{PREFIX} REFUSED: --ccd {src} is not a file — nothing placed", file=out); return EXIT_FAIL
        if not ok_src:
            print(f"{PREFIX} REFUSED: --ccd {src} is {seen_src}, not the pinned Chemical Component Dictionary ({pinned}) — nothing placed", file=out); return EXIT_FAIL
    ok, seen, _ = at_pin(path)
    if ok:
        how = f"present, nothing fetched{f' (--ccd {src} matches it: nothing written)' if src else ''}"
        print(f"{PREFIX} CCD OK: {path} is the pinned Chemical Component Dictionary ({seen}; stock/PINS.json ccd) — {how}", file=out); return EXIT_OK
    if src:
        print(f"{PREFIX} CCD: {path} is not the pinned dictionary ({seen}); placing --ccd {src} there (upstream's fetch is not run) …", file=out); out.flush()
        try:
            place_file(src, path)
        except Exception as e:  # noqa: BLE001 — a read-only package directory or a full disk, relayed verbatim
            print(f"{PREFIX} FAILED: placing --ccd {src} at {path} raised {type(e).__name__}: {e}", file=out); return EXIT_FAIL
        print(f"{PREFIX} CCD placed from --ccd {src} sha256={digest_src} (no fetch)", file=out)
        placed_by = f"placed from --ccd {src}"
    else:
        print(f"{PREFIX} CCD: {path} is not the pinned dictionary ({seen}); running upstream's placement (openfold3.setup_openfold.setup_biotite_ccd) …", file=out); out.flush()
        try:
            place()
        except Exception as e:  # noqa: BLE001 — upstream's own error (network, a read-only package directory), relayed verbatim
            local = isinstance(e, OSError) and not isinstance(e, (ConnectionError, TimeoutError))   # a file-system error needs no network advice
            hint = ((" — the dictionary could not be written at biotite's path; with the file fetched beforehand the same step is" if local else
                     " — on a machine without network access, fetch the pinned dictionary beforehand and pass it:") + " run.sh install --ccd FILE (README.md 'Install')")
            print(f"{PREFIX} FAILED: the CCD placement raised {type(e).__name__}: {e}{hint}", file=out); return EXIT_FAIL
        placed_by = "placed by upstream's setup"
    ok, seen, _ = at_pin(path)
    if ok:
        print(f"{PREFIX} CCD OK: {path} is the pinned Chemical Component Dictionary ({seen}; stock/PINS.json ccd) — {placed_by}", file=out); return EXIT_OK
    why = (f"after the copy from --ccd {src} is {seen}, not the pin ({pinned}) — the copy did not land whole" if src else
           f"after upstream's placement is {seen}, not the pin ({pinned}) — upstream serves a dictionary other than the pinned one")
    print(f"{PREFIX} REFUSED: {path} {why}; the file is left in place", file=out)
    return EXIT_FAIL


# --------------------------------------------------------------------------------------------------------------- weights ----
def upstream_route(local: str) -> Optional[Callable[[str], object]]:
    """``fetch_into(dir)`` — upstream's parameter download for the registry entry whose file name is ``local`` (None when upstream's
    registry has no such file: nothing here invents a URL)."""
    from openfold3.entry_points import parameters as P
    name = next((n for n, e in P.OPENFOLD_MODEL_CHECKPOINT_REGISTRY.items() if e.file_name == local), None)
    if name is None:
        return None
    return lambda d: P.download_model_parameters(d, name, force_download=False, skip_confirmation=True)


def fetch(weights_dir: str, home: Optional[str] = None, route=None, gate=None, out=None) -> int:
    """Fetch-if-absent then check the pinned checkpoint in ``weights_dir``; EXIT_OK when it matches the pin, EXIT_FAIL otherwise."""
    out = out or sys.stdout
    from . import env as _env
    home = home or _env.tree_home()
    pin = _env.pins(home).get("weights") or {}
    local = pin.get("file")
    if not local or not pin.get("sha256"):
        print(f"{PREFIX} FAILED: stock/PINS.json carries no weights file / sha256 to fetch and check against", file=out); return EXIT_FAIL
    weights_dir = os.path.abspath(weights_dir)
    os.makedirs(weights_dir, exist_ok=True)
    path = os.path.join(weights_dir, local)
    if gate is None:
        from . import stack
        gate = lambda p: stack.weights_gate(p, home, hash_it=True, refresh=True)  # noqa: E731 — the routes' own comparison, hashed afresh
    if os.path.isfile(path):
        print(f"{PREFIX} present  {local}", file=out)
    else:
        try:
            call = (route or upstream_route)(local)
        except Exception as e:  # noqa: BLE001 — openfold3 not importable: the stack is not installed as README.md 'Setup' says
            print(f"{PREFIX} FAILED: upstream's parameter download could not be bound ({type(e).__name__}: {e}) — install the pinned stack first (README.md 'Install')", file=out); return EXIT_FAIL
        if call is None:
            print(f"{PREFIX} FAILED: {local}: upstream's checkpoint registry (openfold3.entry_points.parameters) has no entry with this file name — place the file in {weights_dir} by hand and re-run", file=out); return EXIT_FAIL
        print(f"{PREFIX} fetching {local} into {weights_dir} with upstream's downloader (openfold3.entry_points.parameters.download_model_parameters) …", file=out); out.flush()
        try:
            call(weights_dir)
        except Exception as e:  # noqa: BLE001 — upstream's own error, relayed verbatim
            print(f"{PREFIX} FAILED: {local}: the download raised {type(e).__name__}: {e}", file=out); return EXIT_FAIL
        if not os.path.isfile(path):
            print(f"{PREFIX} FAILED: {local}: upstream's downloader returned but {path} is absent", file=out); return EXIT_FAIL
    info = gate(path)
    if info.get("is_pinned"):
        print(f"{PREFIX} WEIGHTS OK: 1/1 file in {weights_dir} matches stock/PINS.json ({local} sha256 {info['sha256'][:12]}…, {info.get('bytes')} bytes) — export OPENFOLD3_OB0_CKPT={path}", file=out)
        return EXIT_OK
    print(f"{PREFIX} REFUSED: {path} sha256 {str(info.get('sha256'))[:12]}… ({info.get('bytes')} bytes) is not the pinned {info.get('pinned_file')} "
          f"(sha256 {str(info.get('pinned_sha256'))[:12]}…, stock/PINS.json weights) — the file is left in place; remove it and re-run to fetch afresh", file=out)
    return EXIT_FAIL


USAGE = ("usage: python -m openfold3_ob0_opt.weights --ccd [FILE] | DIR   (run.sh install [--weights DIR] [--ccd FILE]: upstream's full CCD into biotite — fetched by "
         "upstream's setup call, or copied from FILE fetched beforehand — or the pinned checkpoint into DIR)")


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv == ["--ccd"]:
        return ccd()
    if argv[:1] == ["--ccd"] and len(argv) == 2 and argv[1] and not argv[1].startswith("-"):
        return ccd(source=argv[1])
    if len(argv) == 1 and argv[0].startswith("--ccd=") and argv[0][len("--ccd="):]:
        return ccd(source=argv[0][len("--ccd="):])
    if len(argv) != 1 or argv[0].startswith("-"):
        print(USAGE, file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
