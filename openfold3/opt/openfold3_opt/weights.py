"""`run.sh install --weights DIR` — fetch the pinned checkpoint into DIR and check it against ``stock/PINS.json`` ``weights`` (file name,
source URL and sha256). ``DIR/of3-p2-155k.pt`` is then the ``OPENFOLD3_CKPT`` every route reads (README.md 'Install'); nothing is fetched at
run time.

The source is the pin's own URL (``stock/PINS.json`` ``weights.url``, a public S3 object) and upstream owns the transfer: an
``https://<bucket>.s3.amazonaws.com/<key>`` URL goes through ``openfold3.core.utils.s3.download_s3_file(bucket, key, path)`` (upstream's
unsigned S3 client, with its progress bar); any other URL is a plain HTTPS download. A file already in DIR is kept and only checked. Upstream's
own parameter registry (``openfold3.entry_points.parameters``, ``setup_openfold``) serves ``of3-p2-155k.pt`` from another bucket as a
re-serialised copy of the same tensors (no ``version_tensor`` entry, other bytes), which the pin refuses by design — hence the pin's URL, not the
registry, is the route. The comparison is ``stack.weights_gate`` — the one every route's WEIGHTS line reports — run afresh, so its digest memo
is warm for the first ``check`` / ``pred``. A file whose digest is not the pin's is named and the step fails (exit 1) — the file is left in
place for inspection, never deleted. ``python -m openfold3_opt.weights DIR``.
"""
from __future__ import annotations

import os
import sys
from typing import Callable, List, Optional, Tuple

PREFIX = "[openfold3-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2


def s3_bucket_key(url: str) -> Optional[Tuple[str, str]]:
    """(bucket, key) of a virtual-hosted S3 URL ``https://<bucket>.s3[.<region>].amazonaws.com/<key>``; None for any other URL."""
    from urllib.parse import urlparse
    u = urlparse(url); host = (u.hostname or "").lower()
    if u.scheme != "https" or not host.endswith(".amazonaws.com") or ".s3" not in host or not u.path.lstrip("/"):
        return None
    return host.split(".s3", 1)[0], u.path.lstrip("/")


def upstream_route(download_dir: str, s3_download=None, https_download=None) -> Callable[[dict], Optional[Callable[[], object]]]:
    """``route(entry)`` → the call that fetches the pin-table ``entry`` ({local, url, sha256}) into ``download_dir``: upstream's S3 client for an
    S3 URL, a plain HTTPS download otherwise; None for an entry without a URL. The two transports are injectable for the package tests."""
    from pathlib import Path
    if s3_download is None:
        from openfold3.core.utils.s3 import download_s3_file as s3_download     # upstream's unsigned public-S3 client (boto3), progress bar included

    if https_download is None:
        def https_download(url: str, path: str) -> None:
            import shutil, urllib.request
            with urllib.request.urlopen(url) as r, open(path + ".part", "wb") as f:   # written beside the target, renamed whole
                shutil.copyfileobj(r, f, length=1 << 20)
            os.replace(path + ".part", path)

    def route(entry: dict):
        url = entry.get("url"); target = os.path.join(download_dir, entry["local"])
        if not url:
            return None
        bk = s3_bucket_key(url)
        if bk is not None:
            return lambda: s3_download(bk[0], bk[1], Path(target))
        return lambda: https_download(url, target)
    return route


def pinned_matcher(home: Optional[str] = None) -> Callable[[str, List[dict]], dict]:
    """``matcher(dir, files)`` → {status pinned|unknown, files, unknown [{local, sha256}], seconds}: every file of the pin table hashed by the routes'
    own comparator (stack.weights_gate, afresh — its digest memo entry rewritten)."""
    import time
    from . import stack

    def matcher(d: str, files: List[dict]) -> dict:
        t0 = time.time(); unknown = []
        for f in files:
            rec = stack.weights_gate(os.path.join(d, f["local"]), home, hash_it=True, refresh=True)
            if not rec["is_pinned"]:
                unknown.append({"local": f["local"], "sha256": rec["sha256"] or "absent"})
        return {"status": "unknown" if unknown else "pinned", "files": len(files), "unknown": unknown, "seconds": round(time.time() - t0, 1)}
    return matcher


def pinned_files(home: Optional[str] = None) -> List[dict]:
    """The pin table: stock/PINS.json ``weights`` names one file and carries its digest (stack.pinned_weights)."""
    from . import env as _env
    w = _env.pins(home or _env.tree_home())["weights"]                       # stack.pinned_weights reads the same entry (file, sha256); the url rides along here
    return [{"local": w["file"], "sha256": w["sha256"], "url": w.get("url")}]


def fetch(download_dir: str, files: Optional[List[dict]] = None, route: Optional[Callable[[str], Optional[Callable[[], object]]]] = None,
          matcher: Optional[Callable[[str, List[dict]], dict]] = None, out=None) -> int:
    """Fetch what is absent, then check every file's sha256 against the pin. ``files`` (the pin table), ``route`` (the transfer) and
    ``matcher`` (``stack.weights_gate`` per file, afresh) are injectable for the package tests."""
    out = out or sys.stdout
    download_dir = os.path.abspath(download_dir)
    os.makedirs(download_dir, exist_ok=True)
    if files is None:
        files = pinned_files()
    if matcher is None:
        matcher = pinned_matcher()
    try:
        if route is None:
            route = upstream_route(download_dir)
    except Exception as e:  # noqa: BLE001 — upstream not importable here: named, nothing fetched
        print(f"{PREFIX} FAILED: {type(e).__name__}: {e}", file=out, flush=True)
        return EXIT_FAIL
    for w in files:
        local = w["local"]; target = os.path.join(download_dir, local)
        state = "present" if os.path.isfile(target) else "fetching"
        print(f"{PREFIX} {local}: {state}" + (f" {w.get('url')} (upstream's S3 client: openfold3.core.utils.s3)" if state == "fetching" else ""), file=out, flush=True)
        if state == "present":
            continue
        call = route(w)
        if call is None:
            print(f"{PREFIX} FAILED: stock/PINS.json weights carries no url for {local} — this step cannot fetch it", file=out, flush=True)
            return EXIT_FAIL
        try:
            call()
        except Exception as e:  # noqa: BLE001 — upstream's own error (network, disk, an interrupted transfer), relayed by name; nothing is deleted
            print(f"{PREFIX} FAILED fetching {local}: {type(e).__name__}: {e} — any partial file upstream's transfer left beside it stays in place", file=out, flush=True)
            return EXIT_FAIL
        if not os.path.isfile(target):
            print(f"{PREFIX} FAILED: {local} is not at {target} after upstream's fetch", file=out, flush=True)
            return EXIT_FAIL
    wid = matcher(download_dir, files)
    for u in wid.get("unknown", []):
        print(f"{PREFIX} {u['local']}: sha256 {u['sha256'][:16]}… is not the pin's", file=out, flush=True)
    if wid.get("status") != "pinned":
        bad = [u["local"] for u in wid.get("unknown", [])]
        print(f"{PREFIX} REFUSED: {len(bad)} of {wid.get('files', len(files))} files under {download_dir} do not have the pinned digest (stock/PINS.json weights): "
              f"{', '.join(bad)} — left in place; remove them and re-run this step to fetch afresh", file=out, flush=True)
        return EXIT_FAIL
    ckpt = os.path.join(download_dir, files[0]["local"]) if files else download_dir
    print(f"{PREFIX} WEIGHTS OK: {wid.get('files', len(files))}/{len(files)} files under {download_dir} have the pinned digest ({wid.get('seconds', '?')} s hashing) — export OPENFOLD3_CKPT={ckpt}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m openfold3_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
