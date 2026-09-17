"""`run.sh install --weights DIR` — fetch the weights root into DIR with upstream's own downloader and check every file against ``stock/PINS.json``
(sha256): the checkpoint ``checkpoint/opendde.pt`` ("checkpoint") and the four ``common/`` files ("common" ``managed_assets``: the two CCD
caches every prediction reads, the two PDB date tables ``--use_template true`` reads). DIR is then the ``OPENDDE_ROOT_DIR`` every route reads
(README.md 'Setup'); nothing is fetched at run time.

Upstream owns the transfer: ``opendde.utils.download`` fetches one "managed asset" by name into a path (a staging file beside it, bounded
retries, upstream's own size + sha256 validation, renamed into place when complete) from the URL table of ``opendde.config.dependency_url``
(which honours upstream's ``OPENDDE_DEPENDENCY_URL`` / ``OPENDDE_COMMON_URL`` overrides); the file a PINS entry names is matched to
upstream's asset by file name, the rule upstream applies itself — no URL is restated here. A file already in DIR is kept and only checked. A
file whose digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never deleted; so is the
staging file of an interrupted transfer (a ``.<name>.*.part`` file beside it; remove it by hand). The checkpoint's comparison is
``manifest.weights`` — the one the routes' WEIGHTS line reports — hashed afresh, so its digest memo is warm for the first ``check`` / ``warm``;
the ``common/`` files are hashed with the same helper (``digest_memo.sha256_file``). ``python -m opendde_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional

PREFIX = "[opendde-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
CHECKPOINT = "checkpoint/opendde.pt"                                   # upstream's layout under the root (frozen.REQUIRED_FILES[0]; stock/PINS.json "checkpoint".file)


def pinned_files(tree: str) -> List[dict]:
    """[{local, sha256, bytes}] for every file of the weights root, from stock/PINS.json: "checkpoint", then "common".managed_assets in
    the order "common".files lists them."""
    from . import stack
    p = stack.pins(tree)
    ck = p["checkpoint"]; common = p["common"]; assets = common["managed_assets"]
    out = [{"local": ck["file"], "sha256": ck["sha256"], "bytes": int(ck["bytes"])}]
    for name in common["files"]:
        out.append({"local": common["dir"].rstrip("/") + "/" + name, "sha256": assets[name]["sha256"], "bytes": int(assets[name]["bytes"])})
    return out


def upstream_route() -> Callable[[str, str], Optional[Callable[[], object]]]:
    """``route(local, target)`` → the upstream call that fetches PINS' ``local`` file to ``target`` (None for a file upstream has no asset for).
    Upstream's asset table is keyed by name (the model name for the checkpoint, a config key per ``common/`` file) and every entry's URL ends
    in the file's own name — the match is by that name."""
    from urllib.parse import urlsplit
    from opendde.config import dependency_url as du                    # upstream's URL table + managed-asset identities (reads its URL overrides at import)
    from opendde.utils import download as dl                           # upstream's downloader (imports torch; no model is loaded)
    by_name = {os.path.basename(urlsplit(u).path): key for key, u in du.URL.items() if key in du.MANAGED_ASSETS}
    fetch_one = dl._ensure_managed_asset                               # noqa: SLF001 — the per-asset step upstream's download_inference_cache composes: fetch when absent, validate size + sha256, atomic rename

    def route(local: str, target: str):
        key = by_name.get(os.path.basename(local))
        return (lambda: fetch_one(key, target)) if key else None
    return route


def check(root: str, files: List[dict], tree: str) -> dict:
    """{status pinned|unknown, files, unknown [{local, sha256}], seconds}: every file's sha256 against its pin — the checkpoint through
    ``manifest.weights`` (afresh: the memo entry is rewritten), the common/ files with ``digest_memo.sha256_file``; an absent file reads `absent`."""
    import time
    from . import digest_memo, manifest
    t0 = time.time(); unknown = []
    for f in files:
        p = os.path.join(root, f["local"])
        if f["local"] == CHECKPOINT:
            w = manifest.weights(root, tree, refresh=True)
            digest = w.get("checkpoint_sha256") or "absent"
        else:
            digest = digest_memo.sha256_file(p) if os.path.isfile(p) else "absent"
        if digest != f["sha256"]:
            unknown.append({"local": f["local"], "sha256": digest})
    return {"status": "unknown" if unknown else "pinned", "files": len(files), "unknown": unknown, "seconds": round(time.time() - t0, 1)}


def fetch(root: str, files: Optional[List[dict]] = None, route: Optional[Callable[[str, str], Optional[Callable[[], object]]]] = None,
          matcher: Optional[Callable[[str], dict]] = None, out=None) -> int:
    """Fetch what is absent, then check every file's sha256 against the pin. ``files`` (PINS entries), ``route`` (upstream's downloader) and
    ``matcher`` (``check`` bound to the pins) are injectable for the package tests."""
    out = out or sys.stdout
    root = os.path.abspath(root)
    os.makedirs(root, exist_ok=True)
    tree = None
    if files is None or matcher is None:
        from . import stack
        tree = stack.tree_root()
    if files is None:
        files = pinned_files(tree)
    if matcher is None:
        matcher = lambda d: check(d, files, tree)                      # noqa: E731 — the routes' own comparator for the checkpoint, hashed afresh (memo rewritten)
    try:
        if route is None:
            route = upstream_route()
    except Exception as e:  # noqa: BLE001 — upstream not importable here (or its downloader moved): named, nothing fetched
        print(f"{PREFIX} FAILED: {type(e).__name__}: {e}", file=out, flush=True)
        return EXIT_FAIL
    for w in files:
        local = w["local"]; target = os.path.join(root, local)
        state = "present" if os.path.isfile(target) else "fetching"
        print(f"{PREFIX} {local}: {state}" + (f" ({w.get('bytes', '?')} bytes, upstream's downloader)" if state == "fetching" else ""), file=out, flush=True)
        if state == "present":
            continue
        call = route(local, target)
        if call is None:
            print(f"{PREFIX} FAILED: upstream opendde has no download entry point for {local} (stock/PINS.json names a file this step cannot fetch)", file=out, flush=True)
            return EXIT_FAIL
        try:
            os.makedirs(os.path.dirname(target), exist_ok=True)
            call()
        except Exception as e:  # noqa: BLE001 — upstream's own error (network, disk, its validation of the transfer), relayed by name; nothing is deleted here
            print(f"{PREFIX} FAILED fetching {local}: {type(e).__name__}: {e} — any .{os.path.basename(local)}.*.part file beside it is left in place", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.isfile(target):
            print(f"{PREFIX} FAILED: {local} is not at {target} after upstream's fetch", file=out, flush=True)
            return EXIT_FAIL
    wid = matcher(root)
    for u in wid.get("unknown", []):
        print(f"{PREFIX} {u['local']}: sha256 {u['sha256'][:16]}… is not the pin's", file=out, flush=True)
    if wid.get("status") != "pinned":
        bad = [u["local"] for u in wid.get("unknown", [])]
        print(f"{PREFIX} REFUSED: {len(bad)} of {wid.get('files', len(files))} files under {root} do not have the pinned digest (stock/PINS.json \"checkpoint\" / \"common\"): "
              f"{', '.join(bad)} — left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    print(f"{PREFIX} WEIGHTS OK: {wid.get('files', len(files))}/{len(files)} files under {root} have the pinned digest ({wid.get('seconds', '?')} s hashing) — export OPENDDE_ROOT_DIR={root}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m opendde_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
