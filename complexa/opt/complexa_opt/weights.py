"""`run.sh install --weights DIR` — fetch the two stock checkpoints (``complexa.ckpt``, ``complexa_ae.ckpt``) into DIR with upstream's own
downloader and check both against ``stock/PINS.json`` ``weights.files`` (byte count, then sha256). DIR is then the ``CKPT_PATH`` every
route reads (README.md 'Install'); nothing is fetched at run time.

Upstream owns the transfer: ``env/download_startup.sh --complexa`` of the pinned checkout — the script upstream's ``complexa download``
runs; its ``download_complexa_weights`` fetches each file with ``wget`` into ``<project root>/ckpts/`` and keeps a file that is already
there and non-empty. The project root is the parent of the script's own directory, so this step runs a byte copy of the script from
``<scratch>/env/`` with ``<scratch>/ckpts`` a symbolic link to DIR: upstream's lines fetch straight into DIR, the checkout is never written
to, and no URL is restated here. The script needs ``wget`` on PATH and prints its own progress; its exit status is relayed but is not the
verdict (it moves past a failed transfer by design) — the verdict is the pin check that follows, ``stack.weights_gate`` with the digests
recomputed (``stock/check_pins.py`` ``check_weights``: the comparison the routes' WEIGHTS line cites). A file that is absent or off its pin
is named and the step fails (exit 1); files are left in place for inspection, never deleted. ``python -m complexa_opt.weights DIR``.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from typing import Callable, List, Optional

from . import TAG

PREFIX = f"[{TAG} install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
UPSTREAM_SCRIPT = os.path.join("env", "download_startup.sh")     # relative to upstream's checkout; `complexa download` runs this file
UPSTREAM_FLAG = "--complexa"                                      # its option for the protein-target pair (complexa.ckpt + complexa_ae.ckpt)
UPSTREAM_CKPT_DIR = "ckpts"                                       # where the script puts them, relative to its project root


def upstream_script() -> str:
    """The pinned checkout's ``env/download_startup.sh``: the checkout is ``$LOCAL_CODE_PATH``, else the editable install's directory
    (``stock/check_pins.py`` ``checkout_dir`` — the one rule). RuntimeError naming what is missing."""
    from . import stack
    cp = stack.check_pins()
    _version, pkg_dir, direct_url = cp.installed_upstream()
    checkout = cp.checkout_dir(direct_url, pkg_dir)
    if not checkout:
        raise RuntimeError(f"upstream's checkout cannot be located ({cp.ENV_UPSTREAM} is unset and {cp.DIST} is not installed editable): "
                           f"export {cp.ENV_UPSTREAM}=<the Proteina-Complexa checkout at the pin> (README.md 'Install')")
    script = os.path.join(checkout, UPSTREAM_SCRIPT)
    if not os.path.isfile(script):
        raise RuntimeError(f"{script} is absent: {checkout} is not upstream's checkout at the pin (stock/PINS.json upstream)")
    return script


def upstream_route(downloads_dir: str, script: Optional[str] = None) -> Callable[[], int]:
    """``route()`` runs upstream's downloader so that its ``ckpts/`` IS ``downloads_dir``: a scratch project root holding a byte copy of the
    script under ``env/`` and ``ckpts`` → ``downloads_dir`` (a symbolic link); returns the script's exit status. The scratch root is removed
    afterwards (the link and the copy only — never anything under ``downloads_dir``)."""
    script = script or upstream_script()

    def route() -> int:
        root = tempfile.mkdtemp(prefix="complexa_weights_")
        env_dir = os.path.join(root, os.path.dirname(UPSTREAM_SCRIPT)); copy = os.path.join(root, UPSTREAM_SCRIPT); link = os.path.join(root, UPSTREAM_CKPT_DIR)
        try:
            os.makedirs(env_dir)
            shutil.copy2(script, copy)
            os.symlink(downloads_dir, link)
            return subprocess.run(["bash", copy, UPSTREAM_FLAG], cwd=root).returncode
        finally:
            if os.path.islink(link): os.unlink(link)
            if os.path.isfile(copy): os.remove(copy)
            for d in (env_dir, root):
                if os.path.isdir(d) and not os.listdir(d): os.rmdir(d)
    return route


def fetch(downloads_dir: str, files: Optional[List[dict]] = None, route: Optional[Callable[[], int]] = None,
          gate: Optional[Callable[[str], dict]] = None, out=None) -> int:
    """Fetch what is absent through upstream's downloader, then check every file against the pin. ``files`` (PINS weights.files),
    ``route`` (upstream's downloader bound to the directory) and ``gate`` (``stack.weights_gate`` with sha256) are injectable for the
    package tests."""
    out = out or sys.stdout
    downloads_dir = os.path.abspath(downloads_dir)
    os.makedirs(downloads_dir, exist_ok=True)
    if files is None:
        from . import stack
        files = stack.pins()["weights"]["files"]
    if gate is None:
        from . import stack
        gate = lambda d: stack.weights_gate({stack.ENV_WEIGHTS: d}, with_sha=True)   # noqa: E731 — the routes' own comparator, digests recomputed
    absent = [w["file"] for w in files if not os.path.isfile(os.path.join(downloads_dir, w["file"]))]
    for w in files:
        state = "fetching" if w["file"] in absent else "present"
        print(f"{PREFIX} {w['file']}: {state}" + (f" ({w.get('bytes', '?')} bytes, upstream's downloader)" if state == "fetching" else ""), file=out, flush=True)
    if absent:
        try:
            if route is None:
                route = upstream_route(downloads_dir)
            rc = route()
        except Exception as e:  # noqa: BLE001 — upstream's script missing, no bash, the scratch root not creatable: named, nothing deleted
            print(f"{PREFIX} FAILED fetching {', '.join(absent)}: {type(e).__name__}: {e}", file=out, flush=True)
            return EXIT_FAIL
        print(f"{PREFIX} upstream's {UPSTREAM_SCRIPT} {UPSTREAM_FLAG} exited {rc}; the pin check below is the verdict", file=out, flush=True)
    verdict = gate(downloads_dir)
    print(f"{PREFIX} {verdict['line']}", file=out, flush=True)
    if not verdict["pinned"]:
        print(f"{PREFIX} REFUSED: {len(verdict['bad'])} finding(s) under {downloads_dir} against stock/PINS.json weights.files: {'; '.join(verdict['bad'])} "
              f"— files left in place; remove a bad file and re-run this step to fetch it afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {len(files)}/{len(files)} files under {downloads_dir} have the pinned digest — export CKPT_PATH={downloads_dir}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m complexa_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
