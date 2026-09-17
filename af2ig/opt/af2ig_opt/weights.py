"""`run.sh install --weights DIR` — fetch the one AlphaFold-2 parameter file this kit reads into ``DIR/params/`` and check it against
``stock/PINS.json`` ``weights`` (sha256). DIR is then the ``AF2_PARAMS`` every route reads (README.md 'Setup'); nothing is fetched at run time.

Upstream has no downloader code: dl_binder_design's README ('Download AlphaFold2 Model Weights') documents fetching DeepMind's parameter archive
and unpacking it into the params directory (``wget <archive>``, ``tar --extract``). This module does that with the archive ``stock/PINS.json``
``weights.source`` names (url, sha256, bytes — the URL upstream documents; no URL is stated here): the archive is read once over HTTPS, its sha256
taken as it streams, and the member this kit reads (``weights.file``, ``params/params_model_1_ptm.npz``) is written under DIR — to a
``*.download_tmp`` file beside its final name, renamed into place when the member is complete; the archive's other members (the remaining
AlphaFold-2 models) are not kept. A file already in DIR is kept and only checked. The comparison is the routes' own — ``stack.weights_gate``
(``stock/check_pins.py`` ``weights_verdict`` through the package's digest memo), refreshed, so the memo is warm for the first ``check`` /
``pred``. A file whose digest is not the pin's is named and the step fails (exit 1) — the file is left in place for inspection, never deleted.
``python -m af2ig_opt.weights DIR``.
"""
from __future__ import annotations

import hashlib
import os
import shutil
import sys
import tarfile
from typing import Callable, List, Optional

PREFIX = "[af2ig-opt install]"
EXIT_OK, EXIT_FAIL, EXIT_USAGE = 0, 1, 2
TMP_SUFFIX = ".download_tmp"
CHUNK = 8 << 20
PROGRESS_EVERY = 512 << 20


class _HashingReader:
    """A read-only stream over the transfer: every byte read goes through sha256 and a byte count; a progress line every PROGRESS_EVERY bytes."""

    def __init__(self, raw, total, out):
        self.raw, self.total, self.out = raw, total, out
        self.sha256, self.bytes, self._next = hashlib.sha256(), 0, PROGRESS_EVERY

    def read(self, size=-1):
        b = self.raw.read(size)
        if b:
            self.sha256.update(b); self.bytes += len(b)
            if self.bytes >= self._next:
                print(f"{PREFIX}   {self.bytes / 1e9:.1f} GB of {self.total / 1e9:.1f} GB read" if self.total else f"{PREFIX}   {self.bytes / 1e9:.1f} GB read", file=self.out, flush=True)
                self._next += PROGRESS_EVERY
        return b

    def drain(self):
        while self.read(CHUNK):
            pass


def _urlopen(url):
    import urllib.request
    return urllib.request.urlopen(url, timeout=120)                      # noqa: S310 — the https URL of stock/PINS.json weights.source


def fetch_member(source: dict, member: str, target: str, opener: Callable = _urlopen, out=None) -> dict:
    """Stream the archive ``source`` ({url, sha256, bytes}) and write its member whose base name is ``member``'s to ``target`` (a
    ``*.download_tmp`` beside it first, renamed when complete). Returns {archive_sha256, archive_bytes, archive_pinned, member_bytes}.
    Raises on a transfer error, an archive without the member, or a short member; the temporary file is then left as it is."""
    out = out or sys.stdout
    os.makedirs(os.path.dirname(target), exist_ok=True)
    tmp = target + TMP_SUFFIX
    if os.path.exists(tmp):
        print(f"{PREFIX}   {tmp}: an earlier, interrupted transfer — overwritten", file=out, flush=True)
    resp = opener(source["url"])
    try:
        total = getattr(resp, "length", None)
        if total is None:
            try: total = int(resp.headers.get("Content-Length"))            # type: ignore[union-attr]
            except Exception: total = source.get("bytes") or 0          # noqa: BLE001
        reader = _HashingReader(resp, int(total or source.get("bytes") or 0), out)
        want = os.path.basename(member); got = None
        with tarfile.open(fileobj=reader, mode="r|*") as tf:
            for ti in tf:
                if not ti.isfile() or os.path.basename(ti.name) != want:
                    continue
                src = tf.extractfile(ti)
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(src, fh, CHUNK)
                got = ti.size
                if os.path.getsize(tmp) != ti.size:
                    raise OSError(f"{tmp}: {os.path.getsize(tmp)} bytes written, the archive member has {ti.size}")
        reader.drain()                                                    # the archive's trailing blocks: its digest covers every byte
    finally:
        close = getattr(resp, "close", None)
        if close: close()
    if got is None:
        raise FileNotFoundError(f"the archive at {source['url']} holds no member named {want} ({reader.bytes} bytes read)")
    os.replace(tmp, target)
    digest = reader.sha256.hexdigest()
    return {"archive_sha256": digest, "archive_bytes": reader.bytes, "archive_pinned": digest == source.get("sha256"), "member_bytes": got}


def _routes_gate(params_dir: str) -> dict:
    """``stack.weights_gate`` for AF2_PARAMS=params_dir, refreshed: the verdict the routes print, its digest memo rewritten now."""
    from . import stack
    environ = dict(os.environ); environ[stack.ENV_AF2_PARAMS] = params_dir
    return stack.weights_gate(environ=environ, refresh=True)


def fetch(params_dir: str, pins: Optional[dict] = None, opener: Optional[Callable] = None, matcher: Optional[Callable[[str], dict]] = None, out=None) -> int:
    """Fetch the parameter file when absent, then check its sha256 against the pin. ``pins`` (stock/PINS.json), ``opener`` (url -> a readable
    stream of the archive's bytes) and ``matcher`` (``stack.weights_gate``, refreshed) are injectable for the package tests."""
    out = out or sys.stdout
    params_dir = os.path.abspath(params_dir)
    os.makedirs(params_dir, exist_ok=True)
    if pins is None:
        from . import stack
        pins = stack.pins()
    W = pins["weights"]; rel = W["file"]; target = os.path.join(params_dir, rel)
    matcher = matcher or _routes_gate
    if os.path.isfile(target):
        print(f"{PREFIX} {rel}: present", file=out, flush=True)
    else:
        src = W.get("source") or {}
        if not src.get("url"):
            print(f"{PREFIX} FAILED: stock/PINS.json weights.source names no archive URL for {rel} — this step cannot fetch it", file=out, flush=True)
            return EXIT_FAIL
        print(f"{PREFIX} {rel}: fetching — the AlphaFold-2 parameter archive {src['url']} ({src.get('bytes', '?')} bytes, {src.get('members', '?')} members) "
              f"is streamed once and this one member ({W.get('bytes', '?')} bytes) is kept", file=out, flush=True)
        try:
            r = fetch_member(src, rel, target, opener=opener or _urlopen, out=out)
        except Exception as e:  # noqa: BLE001 — the transfer's own error (network, disk, an archive without the member), relayed by name; nothing is deleted
            print(f"{PREFIX} FAILED fetching {rel}: {type(e).__name__}: {e} — any {os.path.basename(rel)}{TMP_SUFFIX} beside it is left in place", file=out, flush=True)
            return EXIT_FAIL
        line = f"{PREFIX}   archive read: {r['archive_bytes']} bytes, sha256 {r['archive_sha256'][:16]}…"
        print(line + (" (the pinned archive)" if r["archive_pinned"] else f" — NOT the digest stock/PINS.json weights.source pins ({str(src.get('sha256'))[:16]}…); the member's own digest decides below"), file=out, flush=True)
    v = matcher(params_dir)
    if not v.get("pinned"):
        have = v.get("sha256")
        print(f"{PREFIX} REFUSED: {rel} under {params_dir} " + (f"has sha256 {str(have)[:16]}…, not the pinned digest {str(v.get('pin') or W['sha256'])[:16]}… (stock/PINS.json weights) — left in place; remove it and re-run this step to fetch afresh" if have else f"is not there after the step ({'; '.join(v.get('bad') or []) or 'absent'})"), file=out, flush=True)
        return EXIT_FAIL
    note = f"; {v['memo_note']}" if v.get("memo_note") else ""
    print(f"{PREFIX} WEIGHTS OK: 1/1 files under {params_dir} have the pinned digest ({rel} sha256 {str(v.get('sha256'))[:16]}…{note}) — export AF2_PARAMS={params_dir}", file=out, flush=True)
    return EXIT_OK


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 1 or argv[0].startswith("-"):
        print("usage: python -m af2ig_opt.weights DIR   (run.sh install --weights DIR)", file=sys.stderr)
        return EXIT_USAGE
    return fetch(argv[0])


if __name__ == "__main__":
    sys.exit(main())
