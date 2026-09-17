"""The deterministic recipe of this model: an autotune class, not a switch. No switch-based determinism exists on the JAX path — outputs repeat bit for bit
only inside one autotune class, built by one warm-up run and read by every arm under the same GPU model, jax build and defaults; ``exact`` and
the stock route under ``detrecipe`` share that ``$CACHE``, ``fast`` builds its own, ``off`` under ``stock`` is not part of the recipe. This
module names the class in the logs (the ``$CACHE`` digest, the tokamax table, the cache key).
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from typing import List, Optional, Sequence

def _walk(root: str, skip_dir: Optional[str] = None) -> List[str]:
    out = []
    for d, dirs, fs in os.walk(root):
        if skip_dir and os.path.abspath(d) == os.path.abspath(skip_dir):
            dirs[:] = []
            continue
        for f in fs:
            out.append(os.path.relpath(os.path.join(d, f), root))
    return sorted(out)


def cache_digest(cache_dir: Optional[str]) -> Optional[dict]:
    """sha256 over the sorted (relative path, size, sha256) of every file in $CACHE (executables/ excluded); None if absent."""
    if not cache_dir or not os.path.isdir(cache_dir):
        return None
    from .stack import executables_dir, sha256_file
    files = _walk(cache_dir, executables_dir(cache_dir))
    h = hashlib.sha256()
    for rel in files:
        p = os.path.join(cache_dir, rel)
        h.update(f"{rel}\0{os.path.getsize(p)}\0{sha256_file(p)}\n".encode())
    return {"dir": cache_dir, "files": len(files), "sha256": h.hexdigest()}


def executables_report(cache_dir: Optional[str]) -> List[dict]:
    if not cache_dir:
        return []
    from .stack import executables, sha256_file
    return [{"file": os.path.basename(p), "bytes": os.path.getsize(p), "sha256": sha256_file(p)} for p in executables(cache_dir)]


XLA_CACHES_ENV = "JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES"   # jax >= 0.4.36 (config jax_persistent_cache_enable_xla_caches, default 'xla_gpu_per_fusion_autotune_cache_dir'): which XLA caches ride the
                                                            # compilation-cache directory. Under the default the class carries XLA's per-fusion autotune results — the recipe's carrier: `off` cannot reuse
                                                            # `exact`'s executables (another program) but loads the kernel choices recorded there. `none` empties the carrier here.
AUTOTUNE_SUBDIR = os.path.join("jax", "xla_gpu_per_fusion_autotune_cache_dir")   # <class>/jax/xla_gpu_per_fusion_autotune_cache_dir/<key>.textproto (jax/_src/compiler.py get_compile_options)
AUTOTUNE_WORD = "xla_gpu_per_fusion_autotune_cache_dir"


def xla_caches_state(environ: Optional[dict] = None) -> dict:
    """``{'word': 'default' | <the variable's value> | 'blank', 'carried': bool}`` — whether a model process started in ``environ`` lets its
    persistent cache carry XLA's per-fusion autotune results (jax: unset -> the default names the autotune dir; 'all' or a list naming it -> carried;
    'none', blank or a list without it -> not carried)."""
    env = os.environ if environ is None else environ
    if XLA_CACHES_ENV not in env:
        return {"word": "default", "carried": True}
    v = str(env[XLA_CACHES_ENV]).strip()
    return {"word": v or "blank", "carried": v == "all" or AUTOTUNE_WORD in v}


def autotune_entries(cache_dir: Optional[str]) -> Optional[int]:
    """Files under <class>/jax/xla_gpu_per_fusion_autotune_cache_dir (XLA's per-fusion autotune results the class carries), None when absent."""
    d = os.path.join(cache_dir, AUTOTUNE_SUBDIR) if cache_dir else None
    if not d or not os.path.isdir(d):
        return None
    return sum(1 for f in os.listdir(d) if os.path.isfile(os.path.join(d, f)))


def detclass_line(cache_dir: Optional[str], environ: Optional[dict] = None) -> str:
    """`[af3-jax-opt] DETCLASS carrier=xla_autotune_results dir=<class>/jax/xla_gpu_per_fusion_autotune_cache_dir entries=<n|absent> xla_caches=<default|value|blank>
    state=carried|not_carried [note=...]` — the deterministic recipe's carrier inside the class: `exact` == `off` byte for byte rests on both processes
    loading these entries; a JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES that does not name the autotune dir empties it (named, never refused)."""
    from .report import line
    st = xla_caches_state(environ)
    n = autotune_entries(cache_dir)
    note = None if st["carried"] else f"exact==off_not_expected({XLA_CACHES_ENV}={st['word']}:the_class_cannot_carry_XLA_autotune_results;unset_it)"
    return line("DETCLASS", carrier="xla_autotune_results", dir=os.path.join(cache_dir, AUTOTUNE_SUBDIR) if cache_dir else None,
                entries=n if n is not None else "absent", xla_caches=st["word"], state="carried" if st["carried"] else "not_carried", note=note)


TOKAMAX_TABLE = "tokamax_autotune.json"        # <--cache_dir>/tokamax_autotune.json (run_alphafold.py _autotune_cache_path): written by the model process when tokamax.autotune() succeeds, loaded at ModelRunner init by every later process of the class — the stock script's own mechanism, every route
TOKAMAX_POLICY_FLAG = "tokamax_autotuning_cache_miss_fallback"   # tokamax/_src/config.py: the rule for a kernel shape with no autotuning-table entry — heuristics (tokamax's default: a fixed function of the shape) | autotune (measured in-process: box-dependent) | error
TOKAMAX_POLICY_DEFAULT = "heuristics"
TOKAMAX_RX = {"loaded": re.compile(r"^Loading tokamax autotune cache from (?P<path>\S+)"),          # run_alphafold.py _load_autotune_cache
              "saved": re.compile(r"^Tokamax autotune cache saved to (?P<path>\S+)"),               # run_alphafold.py run_inference (tokamax.autotune succeeded; the table now exists)
              "unavailable": re.compile(r"^Tokamax autotune unavailable \((?P<exc>\w+)")}         # the kit script (patches/01): the attempt raised — cause named once; the stock script swallows it silently


def tokamax_table_path(cache_dir: Optional[str]) -> Optional[str]:
    return os.path.join(cache_dir, TOKAMAX_TABLE) if cache_dir else None


def tokamax_table(cache_dir: Optional[str]) -> dict:
    """The class's tokamax autotuning table: {path, present, bytes, sha256, entries} — entries = len(data) of an AutotuningResult dump, None when unreadable."""
    from .stack import sha256_file
    path = tokamax_table_path(cache_dir)
    out = {"path": path, "present": bool(path and os.path.isfile(path)), "bytes": None, "sha256": None, "entries": None}
    if out["present"]:
        out["bytes"], out["sha256"] = os.path.getsize(path), sha256_file(path)
        try:
            with open(path, encoding="utf-8") as f:
                doc = json.load(f)
            out["entries"] = len(doc.get("data") or []) if isinstance(doc, dict) else None
        except (ValueError, OSError):
            out["entries"] = None
    return out


def tokamax_policy(argv: Sequence[str]) -> str:
    """The cache-miss rule the model process runs with: the last --tokamax_autotuning_cache_miss_fallback on its command line (an absl flag tokamax defines),
    else tokamax's default. The package composes no such flag; a caller's pass-through flag is what this names."""
    val = TOKAMAX_POLICY_DEFAULT
    for i, t in enumerate(argv):
        if t.startswith(f"--{TOKAMAX_POLICY_FLAG}="):
            val = t.split("=", 1)[1]
        elif t == f"--{TOKAMAX_POLICY_FLAG}" and i + 1 < len(argv):
            val = argv[i + 1]
    return val


def is_tokamax_line(ln: str) -> bool:
    return any(rx.match(ln.strip()) for rx in TOKAMAX_RX.values())


def tokamax_account(before: dict, after: dict, lines: Sequence[str], argv: Sequence[str]) -> dict:
    """One pass's tokamax record: the table before/after the model process (state absent | written | loaded | rewritten), what the transcript said
    about the autotune attempt (loaded | saved | unavailable:<Exception> | none — the stock script's failed attempt is silent), and the cache-miss policy."""
    if after.get("present"):
        state = ("loaded" if before.get("sha256") == after.get("sha256") else "rewritten") if before.get("present") else "written"
    else:
        state = "absent"
    autotune = "none"
    for ln in lines:
        for name, rx in TOKAMAX_RX.items():
            m = rx.match(ln.strip())
            if m:
                autotune = f"unavailable:{m.group('exc')}" if name == "unavailable" else name
    return {"path": after.get("path") or before.get("path"), "state": state, "entries": after.get("entries"), "sha256": after.get("sha256"),
            "sha256_before": before.get("sha256"), "bytes": after.get("bytes"), "policy": tokamax_policy(argv), "autotune": autotune}


def tokamax_line(acc: dict) -> str:
    """`[af3-jax-opt] TOKAMAX table=<path|none> state=absent|written|loaded|rewritten entries=<n|na> sha256=<16 hex|none> policy=<heuristics|autotune|error> autotune=loaded|saved|unavailable:<Exc>|none`"""
    from .report import line
    return line("TOKAMAX", table=acc.get("path") or "none", state=acc["state"], entries=acc["entries"] if acc.get("entries") is not None else "na",
                sha256=(acc.get("sha256") or "none")[:16], policy=acc["policy"], autotune=acc["autotune"])


CACHEKEY_RX = re.compile(r"^\[af3-jax-opt\] CACHEKEY accelerator=(?P<word>\S+) jax=(?P<jax>\S+)")   # inprocess/portable_cache_key.py install(): the model process's key rebinding
CACHE_KEY_DEFAULT = "topology"        # no CACHEKEY line in the transcript: jax's own accelerator_config component (the box's device-topology fingerprint)


def is_cachekey_line(ln: str) -> bool:
    return bool(CACHEKEY_RX.match(ln.strip()))


def cache_key_word(lines: Sequence[str]) -> str:
    """The persistent-cache key's accelerator component this pass's model process used, read from its transcript: ``device_kind`` (the
    tree's launchers installed inprocess/portable_cache_key.py) or ``topology`` (no CACHEKEY line, or the rebinding named itself off under another jax: jax's own, box-specific)."""
    for ln in lines:
        m = CACHEKEY_RX.match(ln.strip())
        if m:
            return m.group("word")
    return CACHE_KEY_DEFAULT


def cache_line(cache_dir: Optional[str], when: str, key: Optional[str] = None) -> str:
    """`[af3-jax-opt] CACHE when=before|after dir=<class> files=<n> sha256=<16 hex> executables=<n> [cache_key=<word>]` — the class census;
    ``cache_key`` (the after line of a pass: cache_key_word) names the key's accelerator component the process compiled and looked up under."""
    from .report import line
    d = cache_digest(cache_dir)
    extra = {"cache_key": key} if key else {}
    if not d:
        return line("CACHE", when=when, dir=cache_dir, state="absent", **extra)
    return line("CACHE", when=when, dir=cache_dir, files=d["files"], sha256=d["sha256"][:16], executables=len(executables_report(cache_dir)), **extra)
