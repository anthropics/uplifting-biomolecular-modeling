"""Run facts the printed lines use: the kit cache directory, the pinned checkpoint's identity and its digest memo
(``weights_status`` feeds the WEIGHTS line).
"""
from __future__ import annotations

import os

from . import _core  # noqa: F401
from . import digest_memo

CHECKPOINT_NAME = "protenix-v2"                                       # the pinned checkpoint (the stock CLI's --model_name of this engine)
CHECKPOINT_RELPATH = os.path.join("checkpoint", f"{CHECKPOINT_NAME}.pt")
# the frozen inputs the stock CLI downloads when absent, and the switch that refuses instead: protenix_opt/_frozen.py (the leaf module the
# start-up hook and the CLI share)


from ._autoload import ENV_CACHE_DIR as CACHE_DIR_ENV                      # PROTENIX_OPT_CACHE_DIR, the kit's cache root (configs/h100.env: $MODEL_OPT_JIT_ROOT/weights when that root is set, beside the JIT caches); a declared name


def cache_dir() -> str:
    """The kit's cache root: $PROTENIX_OPT_CACHE_DIR (configs/h100.env exports it beside TRITON_CACHE_DIR / TORCH_EXTENSIONS_DIR), else
    ~/.cache/protenix_opt. The weights digest memo lives at <cache root>/weights_digests.json (digest_memo.MEMO_NAME)."""
    return os.environ.get(CACHE_DIR_ENV) or os.path.join(os.path.expanduser("~"), ".cache", "protenix_opt")


def checkpoint_info(root_dir: str | None = None, hash_it: bool = True, refresh: bool = False, memo_dir: str | None = None) -> dict:
    """``{"path", "present", "bytes", "sha256", "digest_cached_utc"}`` for $PROTENIX_ROOT_DIR/checkpoint/protenix-v2.pt. The sha256 comes
    through the on-disk digest memo (``digest_memo.digest``: keyed by realpath, size, mtime and inode — they select the memo entry, never
    decide a match; written only after a full hash; ``refresh`` hashes afresh and rewrites the entry — `check` always does);
    ``digest_cached_utc`` is the memo entry's time on a hit, None when hashed now. A hashing error is reported in ``error`` rather than raised."""
    root = root_dir if root_dir is not None else os.environ.get("PROTENIX_ROOT_DIR")
    if not root:
        return {"path": None, "present": False, "reason": "PROTENIX_ROOT_DIR is not set"}
    path = os.path.join(root, CHECKPOINT_RELPATH)
    if not os.path.isfile(path):
        return {"path": path, "present": False, "reason": "file not found"}
    info: dict = {"path": path, "present": True, "bytes": os.path.getsize(path)}
    if hash_it:
        mdir = memo_dir or cache_dir()
        try:
            info["sha256"], info["digest_cached_utc"] = digest_memo.digest(path, mdir, refresh=refresh)
        except OSError as e:                                  # the checkpoint unreadable, or the memo directory unwritable (read-only mount, not a directory, no permission)
            try:
                info["sha256"], info["digest_cached_utc"] = digest_memo.sha256_file(path), None   # hashed afresh without the memo; the memo's failure is NAMED, never silent, never a refusal
                info["memo_note"] = f"digest memo unwritable at {mdir} ({type(e).__name__}: {e.strerror or e}); hashed afresh"
            except OSError as e2:
                info["error"] = f"sha256 failed: {e2!r}"
    return info


def weights_status(pins_sha256: str | None, root_dir: str | None = None, refresh: bool = False, memo_dir: str | None = None) -> dict:
    """The checkpoint under $PROTENIX_ROOT_DIR against the pinned checkpoint (stock/PINS.json ``checkpoint.sha256``):
    ``checkpoint_info()`` plus ``digest`` — "pinned" | "unknown" | "absent" — and the one ``line`` the CLI prints (WEIGHTS …).
    An unknown checkpoint is NAMED and the run proceeds in every mode; nothing here refuses (`exact`'s equality-to-stock statements
    are made for the pinned checkpoint). Absence is the frozen-weights gate's concern (_frozen), not this note's."""
    info = checkpoint_info(root_dir, refresh=refresh, memo_dir=memo_dir)
    sha = info.get("sha256"); cached = info.get("digest_cached_utc")
    if not info.get("present"):
        info["digest"] = "absent"
        info["line"] = f"WEIGHTS absent ({info.get('reason')}: {info.get('path') or CHECKPOINT_RELPATH}); proceeding — the stock CLI resolves --model_name {CHECKPOINT_NAME} itself"
    elif sha is not None and pins_sha256 is not None and sha == pins_sha256:
        info["digest"] = "pinned"
        info["line"] = f"WEIGHTS {digest_memo.word('pinned', cached)} sha256={sha[:16]}… ({CHECKPOINT_RELPATH} = stock/PINS.json checkpoint)"
    else:
        info["digest"] = "unknown"
        info["line"] = (f"WEIGHTS {digest_memo.word('unknown', cached)} sha256={(sha or '?')[:16]}… (not the pinned checkpoint {(pins_sha256 or '?')[:16]}…: "
                        f"`exact`'s equality-to-stock statements are made for the pinned checkpoint); proceeding")
    if info.get("memo_note"):
        info["line"] += f" [{info['memo_note']}]"
    return info
