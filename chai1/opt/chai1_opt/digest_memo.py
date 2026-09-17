"""On-disk memo of weights sha256 digests.

A match is decided by digest only. The memo maps a file's
(realpath, st_size, st_mtime_ns, st_ino) to the sha256 computed from the
file's full contents; those stat fields select a memo entry, they never
decide the match. An entry is written only after a full sha256 completes,
atomically (temporary file + os.replace). `check` passes refresh=True: the
file is hashed afresh and its entry rewritten.
"""
import hashlib
import json
import os
import time

MEMO_NAME = "weights_digests.json"
CHUNK = 8 << 20


def sha256_file(path, chunk=CHUNK):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def stat_key(path):
    real = os.path.realpath(path)
    st = os.stat(real)
    return json.dumps([real, st.st_size, st.st_mtime_ns, st.st_ino])


def _load(memo_path):
    try:
        with open(memo_path) as fh:
            table = json.load(fh)
    except (OSError, ValueError):
        return {}
    return table if isinstance(table, dict) else {}


def _store(memo_path, table):
    os.makedirs(os.path.dirname(memo_path) or ".", exist_ok=True)
    tmp = "%s.tmp.%d" % (memo_path, os.getpid())
    with open(tmp, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    os.replace(tmp, memo_path)


def digest(path, memo_dir, refresh=False, hasher=sha256_file):
    """Return (sha256_hex, cached_utc).

    cached_utc is None when the digest was computed by this call (refresh=True,
    or no entry for the file's stat key); otherwise the UTC time at which the
    memoised digest was computed. A hasher exception propagates and nothing
    is written.
    """
    key = stat_key(path)
    memo_path = os.path.join(memo_dir, MEMO_NAME)
    if not refresh:
        entry = _load(memo_path).get(key)
        if isinstance(entry, dict) and isinstance(entry.get("sha256"), str) and len(entry["sha256"]) == 64:
            return entry["sha256"], entry.get("utc") or "unknown-time"
    hexdigest = hasher(os.path.realpath(path))
    if not (isinstance(hexdigest, str) and len(hexdigest) == 64):
        raise ValueError("hasher returned %r, not a sha256 hex digest" % (hexdigest,))
    utc = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    table = _load(memo_path)
    table[key] = {"sha256": hexdigest, "utc": utc}
    _store(memo_path, table)
    return hexdigest, None


def word(status, cached_utc):
    """status ('pinned' | 'unknown' | the kit's own word) + ' (cached digest <utc>)' when served from the memo."""
    return status if cached_utc is None else "%s (cached digest %s)" % (status, cached_utc)
