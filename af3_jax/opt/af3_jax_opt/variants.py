"""Variants: the public checkpoint the tree runs, as a converted parameter directory. ``p2`` = the OF3-p2 checkpoint; its name, URL, sha256 and
byte count live only in ``stock/PINS.json`` ``variants`` (this module's sole source), converted by the fork's own ``convert_of3_weights.py``
into ``<AF3_JAX_PARAMS_ROOT>/<variant>/``. It owns two model flags (``--of3_weights``, ``--model_dir``); a caller's own ``--model_dir``
always replaces them. Weights are sha256'd at every activation and compared against the hash recorded in ``stock/PINS.json``: a match is
pinned, any other digest is NOT PINNED but still RUNS — only a missing file is refused. One variant per process."""
from __future__ import annotations

import os
from typing import Optional

VARIANTS = ("p2",)
PARAMS_FILE = "of3_ported_weights.bin.zst"


def pins() -> dict:
    from .stack import pins as _pins
    return _pins()


def info(variant: str) -> dict:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; variants: {' | '.join(VARIANTS)}")
    return pins()["variants"][variant]


def params_dir(variant: str, root: Optional[str] = None, model_dir: Optional[str] = None) -> Optional[str]:
    """The weights directory: the caller's ``model_dir`` when given (always accepted), else ``<AF3_JAX_PARAMS_ROOT>/<variant>``."""
    if model_dir:
        return os.path.abspath(model_dir)
    from .stack import params_root
    root = root or params_root()
    if not root:
        return None
    return os.path.join(root, variant)


MODEL_DIR_FLAG = "--model_dir"                                     # the one weights flag a caller may pass (``--of3_weights`` stays the variant's)


def params_root_unset_reason(variant: str) -> str:
    """The refusal when neither the parameters root nor a caller's --model_dir names the weights (check_params, model_flags): the variable by name."""
    from .stack import ENV_PARAMS_ROOT
    return f"{ENV_PARAMS_ROOT} is not set (the converted parameters root, README Variables: <root>/{variant}/{PARAMS_FILE}) and no {MODEL_DIR_FLAG} given"


def model_flags(variant: str, root: Optional[str] = None, model_dir: Optional[str] = None) -> list:
    d = params_dir(variant, root, model_dir)
    if not d:
        raise ValueError(params_root_unset_reason(variant))
    return ["--of3_weights", f"{MODEL_DIR_FLAG}={d}"]


def split_model_dir(user_args: list) -> tuple:
    """(the caller's ``--model_dir`` value or None, the remaining args) — both absl spellings (``--model_dir=X``, ``--model_dir X``); the last one wins."""
    value, rest, i = None, [], 0
    args = list(user_args or [])
    while i < len(args):
        a = args[i]
        if a == MODEL_DIR_FLAG and i + 1 < len(args):
            value = args[i + 1]; i += 2; continue
        if a.startswith(MODEL_DIR_FLAG + "="):
            value = a.split("=", 1)[1]; i += 1; continue
        rest.append(a); i += 1
    return value, rest


_DIGESTS: dict = {}                        # this process's memo {digest_memo.stat_key: (sha256, cached_utc)} in front of the on-disk memo
_hasher = None                             # None = digest_memo.sha256_file (the full read); a test may set a callable(realpath) -> hexdigest


def memo_dir() -> str:
    """The on-disk weights digest memo lives under the cache root: <AF3_JAX_CACHE_ROOT>/weights_digests.json (digest_memo.MEMO_NAME)."""
    from .stack import cache_root
    return cache_root()


def _digest(path: str, refresh: bool = False):
    """(sha256, cached_utc) of the parameters file through the kit's digest memo (digest_memo.py): ``refresh=True`` (the `check` verb, the
    install step) hashes the 1.4 GB file afresh and rewrites its memo entry; otherwise an entry for the file's (realpath, size, mtime_ns, inode)
    serves the digest with the UTC time it was computed (cached_utc; None when computed by this call). Those stat fields select a memo
    entry, they never decide which weights these are: the digest alone does. One wrapper process digests per job at any --n_gpu."""
    from . import digest_memo
    key = digest_memo.stat_key(path)
    if not refresh and key in _DIGESTS:
        return _DIGESTS[key]
    hasher = _hasher or digest_memo.sha256_file
    try:
        sha, cached = digest_memo.digest(path, memo_dir(), refresh=refresh, hasher=hasher); unwritable = None
    except OSError:                                                     # the memo dir cannot be used (read-only mount, or no such cache dir at all — never a
        sha, cached = hasher(os.path.realpath(path)), None             # refusal): the digest above is this process's own read, computed afresh and not
        unwritable = os.path.join(memo_dir(), digest_memo.MEMO_NAME)   # remembered. Only the path is kept — never the exception repr: a log scanner that
        # greps stderr for generic error markers (Error/Exception:) would match an embedded `f"{type(e).__name__}: {e}"`;
        # the memo is an optimisation, its absence is not an error (weights_line below).
    _DIGESTS[key] = (sha, cached, unwritable)
    return sha, cached, unwritable


NOT_PINNED_WORDS = "NOT PINNED — the kit's speed and output statements hold for the pinned weights only"


def check_params(variant: str, root: Optional[str] = None, digest: bool = True, model_dir: Optional[str] = None, refresh: bool = False) -> dict:
    """The weights of this activation: ``dir`` (the caller's ``model_dir`` or the variant's), ``present``, and with digest=True (the default:
    every activation) ``sha256`` + ``pinned`` (True when the weights are stock/PINS.json ``variants.<v>.converted``). Only a MISSING file carries a
    ``reason`` (refused by name); any other digest runs — ``weights_line`` says which it is."""
    d = params_dir(variant, root, model_dir)
    res = {"variant": variant, "dir": d, "source": "model_dir" if model_dir else "variant", "present": False, "reason": None, "sha256": None,
           "pinned": None, "matches_pin": None}                             # pinned = the weights verdict (matches_pin: the same fact, the install step's word)
    if not d:
        res["reason"] = params_root_unset_reason(variant)
        return res
    w = os.path.join(d, PARAMS_FILE)
    if not os.path.isfile(w):
        res["reason"] = (f"no {PARAMS_FILE} under {d} (run `run.sh install --variant {variant}`: fetch + convert)" if not model_dir else
                         f"no {PARAMS_FILE} under {MODEL_DIR_FLAG} {d} (the directory must hold the converted parameters file)")
        return res
    res["present"] = True
    res["file"] = w
    res["bytes"] = os.path.getsize(w)
    exp = info(variant).get("converted") or {}
    res["pinned_sha256"] = exp.get("sha256")
    if digest:
        res["sha256"], res["digest_cached_utc"], res["digest_memo_unwritable"] = _digest(w, refresh=refresh)   # refresh: the `check` verb hashes afresh; digest_cached_utc None = computed by this process now; digest_memo_unwritable = the memo path when it could not be used (the digest still afresh)
        res["pinned"] = res["matches_pin"] = res["sha256"] == exp.get("sha256")      # the weights verdict: by digest only
    return res


def weights_line(pc: dict) -> Optional[str]:
    """`[af3-jax-opt] WEIGHTS weights=<variant> sha256=<12> (pinned)` for the pinned digest, `[af3-jax-opt] WEIGHTS weights sha256=<12>
    NOT PINNED — the kit's speed and output statements hold for the pinned weights only` for any other (the run proceeds); None when no file was digested."""
    from .report import PREFIX
    if not pc or not pc.get("present") or not pc.get("sha256"):
        return None
    from . import digest_memo
    short = pc["sha256"][:12]; cached = pc.get("digest_cached_utc")   # a digest served from the memo names when it was computed: `… (cached digest <utc>)`
    if pc.get("pinned"):                                              # the pinned weights
        line = f"{PREFIX} WEIGHTS weights={pc['variant']} sha256={short} {digest_memo.word('(pinned)', cached)}"
    else:
        line = f"{PREFIX} WEIGHTS weights sha256={short} {digest_memo.word(NOT_PINNED_WORDS, cached)}"
    if pc.get("digest_memo_unwritable"):                                # the cache dir could not be used (read-only mount, or none at all): the digest above
        # is this process's own read, never a refusal. Worded to read unambiguously as informational, not a failure: no "unwritable"/
        # "not writable", and no exception class name or OS error text embedded, so that a log scanner grepping stderr for generic error
        # markers — the (Error|Exception): shape — does not flag this informational line on a read-only mount.
        line += (f"\n{PREFIX} WEIGHTS memo skipped: {pc['digest_memo_unwritable']} is not available for caching this process; "
                 f"every file above was digested in full, just not memoised")
    return line
