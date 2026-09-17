"""`caliby_opt.weights` fetches weight files into DIR with upstream's own downloader and checks each against its pinned sha256.

DIR becomes ``MODEL_PARAMS_DIR``, the directory every route reads; nothing else is fetched at run time (``HF_HUB_OFFLINE=1``).

Upstream owns the transfer: ``resolve_ckpt_path(name)`` fetches ``caliby/<name>.ckpt`` when absent under ``$MODEL_PARAMS_DIR``
(the call ``caliby.load_model`` makes), and ``ensure_dir(path)`` fetches one repository directory when absent (the calls
``caliby.generate_ensembles`` makes for ``protpardelle-1c/`` and ``proteinmpnn/`` on the ensemble32 route).

This step runs one ``resolve_ckpt_path`` per pinned checkpoint, one ``ensure_dir`` for ``protpardelle-1c`` and one for
``proteinmpnn`` (none of its files is pinned). A file or directory already in DIR is kept and only checked, per upstream's
rule of fetching only when absent. ``MODEL_PARAMS_DIR`` is bound to DIR and ``HF_HUB_OFFLINE`` lifted for this process only.

Upstream's calls name no revision, so the repository's current files come and the pin decides: a file whose digest does not
match the pin is named and the step fails (exit 1); the file is left in place, never deleted.

The digest is computed by the kit's ``stack.sha256_file``. Run directly as ``python -m caliby_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, Dict, List, Optional, Tuple

PREFIX = "[caliby-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
CKPT_DIR, PROTPARDELLE_DIR = "caliby/", "protpardelle-1c/"          # PINS weights.files layout under MODEL_PARAMS_DIR (weights.layout_note)
COMPANION_DIRS = ("proteinmpnn",)                                     # directories generate_ensembles ensures beside protpardelle-1c/ that hold no pinned file

Route = Callable[[str, dict], Optional[Callable[[], object]]]


def upstream_route(weights_dir: str, out=None) -> Tuple[Route, List[Tuple[str, Callable[[], object]]]]:
    """``(route, companions)``: ``route(rel, info)`` → the upstream call that fetches PINS' ``rel`` file into ``weights_dir`` (None for a
    file upstream has no entry point for); ``companions`` = the ``(name/, call)`` directory fetches the routes read besides the pinned files.
    Binds ``MODEL_PARAMS_DIR`` to ``weights_dir`` and lifts ``HF_HUB_OFFLINE`` before upstream's downloader is imported."""
    out = out or sys.stdout
    os.environ["MODEL_PARAMS_DIR"] = weights_dir
    if os.environ.pop("HF_HUB_OFFLINE", None) not in (None, "", "0"):
        print(f"{PREFIX} HF_HUB_OFFLINE lifted for this step (the one step that fetches; every route keeps it at 1)", file=out, flush=True)
    hub = sys.modules.get("huggingface_hub.constants")
    if hub is not None and getattr(hub, "HF_HUB_OFFLINE", False):
        raise RuntimeError("huggingface_hub was imported with HF_HUB_OFFLINE=1 before this step bound the environment — run it in a fresh "
                           "interpreter (python -m caliby_opt.weights DIR)")
    from caliby import weights as up                                   # noqa: E402 — upstream's downloader, after the environment is bound, by design

    def route(rel: str, info: dict):
        name = info.get("ckpt")
        if rel.startswith(CKPT_DIR) and name:
            if up.MODEL_REGISTRY.get(name) != rel:
                return None                                            # the pin's registry name does not resolve to this file upstream: no entry point
            return lambda: up.resolve_ckpt_path(name)                  # upstream's per-checkpoint fetch (load_model's own call)
        if rel.startswith(PROTPARDELLE_DIR):
            return lambda: up.ensure_dir(os.path.join(weights_dir, PROTPARDELLE_DIR.rstrip("/")))   # upstream's directory fetch (generate_ensembles' own call)
        return None
    companions = [(d + "/", (lambda d=d: up.ensure_dir(os.path.join(weights_dir, d)))) for d in COMPANION_DIRS]
    return route, companions


def fetch(weights_dir: str, files: Optional[Dict[str, dict]] = None, route: Optional[Route] = None,
          companions: Optional[List[Tuple[str, Callable[[], object]]]] = None, hasher: Optional[Callable[[str], Optional[str]]] = None, out=None) -> int:
    """Fetch what is absent, then check every pinned file's sha256. ``files`` (PINS weights.files), ``route`` / ``companions`` (upstream's
    downloader) and ``hasher`` (``stack.sha256_file``) are injectable for the package tests."""
    out = out or sys.stdout
    weights_dir = os.path.abspath(weights_dir)
    os.makedirs(weights_dir, exist_ok=True)
    if files is None or hasher is None:
        from . import stack
        files = stack.pins()["weights"]["files"] if files is None else files
        hasher = stack.sha256_file if hasher is None else hasher
    try:
        if route is None:
            route, ups = upstream_route(weights_dir, out)
            companions = ups if companions is None else companions
    except Exception as e:  # noqa: BLE001 — upstream not importable here, or the hub client already bound offline: named, nothing fetched
        print(f"{PREFIX} FAILED: {type(e).__name__}: {e}", file=out, flush=True)
        return EXIT_FAIL
    companions = companions or []
    for rel, info in files.items():
        target = os.path.join(weights_dir, rel)
        state = "present" if os.path.isfile(target) else "fetching"
        print(f"{PREFIX} {rel}: {state}" + (f" ({info.get('size_bytes', '?')} bytes, upstream's downloader)" if state == "fetching" else ""), file=out, flush=True)
        if state == "present":
            continue
        call = route(rel, info)
        if call is None:
            print(f"{PREFIX} FAILED: upstream caliby has no download entry point for {rel} (stock/PINS.json weights.files names a file this step cannot fetch)", file=out, flush=True)
            return EXIT_FAIL
        try:
            call()
        except Exception as e:  # noqa: BLE001 — upstream's own error (network, disk, an interrupted transfer), relayed by name; nothing is deleted
            print(f"{PREFIX} FAILED fetching {rel}: {type(e).__name__}: {e} — anything it left under {weights_dir} is left in place", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.isfile(target):
            print(f"{PREFIX} FAILED: {rel} is not at {target} after upstream's fetch", file=out, flush=True)
            return EXIT_FAIL
    for name, call in companions:
        d = os.path.join(weights_dir, name)
        state = "present" if os.path.isdir(d) else "fetching"
        print(f"{PREFIX} {name}: {state} (a directory the ensemble32 route reads, no pinned file in it" + ("; upstream's downloader)" if state == "fetching" else ")"), file=out, flush=True)
        if state == "present":
            continue
        try:
            call()
        except Exception as e:  # noqa: BLE001
            print(f"{PREFIX} FAILED fetching {name}: {type(e).__name__}: {e} — anything it left under {weights_dir} is left in place", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.isdir(d):
            print(f"{PREFIX} FAILED: {name} is not at {d} after upstream's fetch", file=out, flush=True)
            return EXIT_FAIL
    bad = []
    for rel, info in files.items():
        digest = hasher(os.path.join(weights_dir, rel)) or "absent"
        if digest != info["sha256"]:
            bad.append(rel)
            print(f"{PREFIX} {rel}: sha256 {digest[:16]}… is not the pin's ({info['sha256'][:16]}…)", file=out, flush=True)
    if bad:
        print(f"{PREFIX} REFUSED: {len(bad)} of {len(files)} files under {weights_dir} do not have the pinned digest (stock/PINS.json weights.files): "
              f"{', '.join(bad)} — left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {len(files)}/{len(files)} files under {weights_dir} have the pinned digest — export MODEL_PARAMS_DIR={weights_dir}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m caliby_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
