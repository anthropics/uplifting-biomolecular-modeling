"""`run.sh install --weights DIR` — fetch upstream's RFdiffusion3 checkpoint into DIR with upstream's own installer and check it against
``stock/PINS.json`` ``checkpoint`` (bytes, sha256). ``DIR/<filename>`` (``rfd3_latest.ckpt``) is then the file ``RFD3_CKPT`` names for every
route (README.md 'Setup'); nothing is fetched at run time.

Upstream owns the transfer: ``foundry_cli.download_checkpoints.install_model(<name>, DIR)`` — the code behind ``foundry install <name>
--checkpoint-dir DIR`` — writes its registry entry's URL (``foundry.inference_engines.checkpoint_registry.REGISTERED_CHECKPOINTS[<name>]``)
to ``DIR/<its filename>`` and leaves a file that is already there alone; no URL is restated here: the pin names the registry entry
(``checkpoint.registry_name``) and records what it resolves to, and an entry that no longer says what the pin recorded is named and nothing is
fetched. The digest is ``registry.checkpoint_sha256`` — the one ``design`` writes into ``opt_manifest.json`` — so its cache is warm for the first
run on the file. Lines go through the package's one printer (``_emit``: stderr, each at a line start). A file whose size or digest is not the pin's is named and the step fails (exit 1); the file is left in place for inspection,
never deleted (upstream's installer writes in place, so an interrupted transfer leaves a short file that a re-run would keep: remove it by hand
and re-run). ``python -m rfdiffusion3_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Tuple

from ._emit import emit

PREFIX = "[rfdiffusion3-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def upstream_installer(pin: dict) -> Callable[[str], None]:
    """``fetch_into(DIR)``: upstream's installer bound to the pinned registry entry. Raises when upstream's registry entry does not say what
    the pin recorded of it (name, file name, URL) — the kit then fetches nothing."""
    from foundry.inference_engines.checkpoint_registry import REGISTERED_CHECKPOINTS   # upstream's registry (no model code is imported)
    from foundry_cli.download_checkpoints import install_model                        # upstream's `foundry install` step for one entry
    name = pin["registry_name"]
    entry = REGISTERED_CHECKPOINTS.get(name)
    if entry is None:
        raise RuntimeError(f"upstream's checkpoint registry has no entry {name!r} (stock/PINS.json checkpoint.registry_name)")
    if entry.filename != pin["filename"] or entry.url != pin["url"]:
        raise RuntimeError(f"upstream's registry entry {name!r} is {entry.filename} from {entry.url}; stock/PINS.json checkpoint records "
                           f"{pin['filename']} from {pin['url']} — not the pinned upstream")

    def fetch_into(directory: str) -> None:
        from pathlib import Path
        install_model(name, Path(directory))                                          # skips a file that exists; raises (typer.Exit) on a failed transfer
    return fetch_into


def fetch(directory: str, pin: Optional[dict] = None, installer: Optional[Callable[[str], None]] = None,
          digest: Optional[Callable[[str], Tuple[str, str]]] = None, out=None) -> int:
    """Fetch the checkpoint when it is absent, then check its size and sha256 against the pin. ``pin`` (PINS ``checkpoint``), ``installer``
    (upstream's) and ``digest`` (``registry.checkpoint_sha256``) are injectable for the package tests; ``out`` is the stream (default: stderr)."""
    directory = os.path.abspath(directory)
    os.makedirs(directory, exist_ok=True)
    if pin is None:
        from . import registry
        pin = registry.pins()["checkpoint"]
    if digest is None:
        from . import registry
        digest = registry.checkpoint_sha256
    target = os.path.join(directory, pin["filename"])
    state = "present" if os.path.isfile(target) else "fetching"
    emit(f"{PREFIX} {pin['filename']}: {state}" + (f" ({pin.get('bytes', '?')} bytes, upstream's installer)" if state == "fetching" else ""), stream=out)
    if state == "fetching":
        try:
            if installer is None:
                installer = upstream_installer(pin)
            installer(directory)
        except BaseException as e:  # noqa: BLE001 — upstream not importable here, its registry off the pin, or its own transfer error (typer.Exit included): named; nothing is deleted
            if isinstance(e, KeyboardInterrupt):
                raise
            emit(f"{PREFIX} FAILED fetching {pin['filename']}: {type(e).__name__}: {e} — a partial file, if any, is left at {target}", stream=out)
            return EXIT_FAIL
        if not os.path.isfile(target):
            emit(f"{PREFIX} FAILED: {pin['filename']} is not at {target} after upstream's installer ran", stream=out)
            return EXIT_FAIL
    size = os.path.getsize(target)
    if pin.get("bytes") is not None and size != pin["bytes"]:
        emit(f"{PREFIX} REFUSED: {target} is {size} bytes; stock/PINS.json checkpoint says {pin['bytes']} — left in place; remove it and re-run this step to fetch afresh", stream=out)
        return EXIT_FAIL
    sha, source = digest(target)
    if sha != pin["sha256"]:
        emit(f"{PREFIX} REFUSED: {target} has sha256 {sha[:16]}…, not the pin's {pin['sha256'][:16]}… (stock/PINS.json checkpoint) — left in place; remove it and re-run this step to fetch afresh", stream=out)
        return EXIT_FAIL
    emit(f"{PREFIX} WEIGHTS OK: {pin['filename']} under {directory} has the pinned digest ({size} bytes; digest {source}) — export RFD3_CKPT={target}", stream=out)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        emit("usage: python -m rfdiffusion3_opt.weights DIR   (run.sh install --weights DIR)")
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
