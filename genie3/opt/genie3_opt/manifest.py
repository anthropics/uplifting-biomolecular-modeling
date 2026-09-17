"""``opt_manifest.json`` beside the outputs: what ran, on what, with which levers, and what came out — the activation report and
the run record in one file (schema ``genie3_opt.manifest/1``).

Fields: ``core`` (the shared core the activation was gated against: at minimum its version — opt_core.manifest.core_block, whose full
shape is the core's own to define), ``activation`` (the report `enable()` returned: mode, tier, attach, levers_planned, gpu, stack, the pins
report's summary, tools, opt_core), ``line`` (the kit line as run), ``precision`` (fp32 | tf32) and ``trimul`` (the line's TriangleMultiplication
provider: fpf = lever L7 | stock), ``request`` (the input file's path and the package's composed keys: problems, n_sample, seed, batch_size), ``shard`` (``K/M`` under upstream's --num-shards M --shard-id K: the pass wrote the shard's share of every problem's designs; else null),
``stock_flags`` (upstream's generate flags passed to the stock child), ``recipe`` (det.describe), ``weights`` (directory or the
request's own files, the checkpoint and the model config with their sha256 — memoised, digest_memo — each ``pinned`` or not; ``weights_record``,
the one census every route reads), ``driver_pass`` (command, rc, wall, log, timings JSON and its figures, the evidence found / missing /
forbidden, ``ready_s`` and ``first_design``) or ``stock`` (command, rc, the proof, the KERNELS census, ``ready_s``), ``env_dropped``, ``outputs``
(per problem: the design files this pass wrote, by name), ``status`` (ok | incomplete — the PDB count differs from the request, exit 1 | failed —
the driver / stock process failed, a forbidden driver-log line, a numerics readback off the line, a broken lever; exit 1 | refused — by name, exit 3:
before anything ran (a deployment fact, a batch the memory model refuses) or after the pass when a planned lever could not run, ``refused`` says
which), ``levers_missing`` (planned levers that left no evidence: they could not run here, and a mode is all of its levers — the mode refused),
``levers_broken`` (planned levers that ran and whose own record contradicts their contract, a `judged` predicate: the pass is failed),
``forbidden_lines`` (the count),
``levers_declined`` (planned levers that declined THIS request by name — the pass is ok, their calls took the module's own forward; reasons under
``driver_pass.evidence.declined``), ``incomplete`` (``<found>/<expected>``), timestamps.
"""
from __future__ import annotations

import json
import os
import time

from . import __version__, _core, det, digest_memo
from .modes import Resolution
from .stack import WEIGHT_FILES, pins, registry_order

SCHEMA = "genie3_opt.manifest/1"
NAME = "opt_manifest.json"


def _utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def memo_dir() -> str:
    """The weights-digest memo's directory: ``$XDG_CACHE_HOME/genie3_opt`` (else ``~/.cache/genie3_opt``; digest_memo.MEMO_NAME inside). An unwritable
    location is not an error: the digest is computed afresh and nothing is stored (weights_record)."""
    base = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return os.path.join(base, "genie3_opt")


def _digest(path: str, refresh: bool):
    """(sha256, cached_utc | None) through the memo; a memo that cannot be read or written falls back to hashing afresh."""
    try:
        return digest_memo.digest(path, memo_dir(), refresh=refresh)
    except OSError:
        return digest_memo.sha256_file(os.path.realpath(path)), None


NOT_PINNED_NOTE = "NOT PINNED — not the digest pinned in stock/PINS.json weights; the pass runs on these files, labelled"


def weights_record(weights_dir: str | None, paths=None, refresh: bool = False) -> dict:
    """The weights census — the ONE reader of the weight files every route uses (the activation gate on the kit and the stock route,
    the run record): the two files the pass reads — ``paths`` (checkpoint, config) when the request names its own, else the two under
    ``weights_dir`` in the stock layout, else upstream's default under the checkout when neither is set (stock/PINS.json weights, keyed by file
    name) — with size and sha256 when present (digest_memo: memoised on disk by the file's stat key, ``refresh`` hashes afresh — `check`), each
    compared with the pin — ``pinned`` (the pinned digest) or not (any other digest: the pass RUNS, labelled by ``lines``); ``missing`` names the
    absent files (the gate refuses those by name); ``pinned`` (top level) = every file present and at the pinned digest."""
    W = pins().get("weights") or {}
    out = {"dir": weights_dir, "paths": list(paths) if paths else None, "files": {}, "missing": [], "pinned": True, "lines": [], "memo": os.path.join(memo_dir(), digest_memo.MEMO_NAME)}
    for i, rel in enumerate(WEIGHT_FILES):
        p = paths[i] if paths else os.path.join(weights_dir or "", *rel.split("/"))
        name = os.path.basename(rel)
        want = (W.get(name) or {}).get("sha256")
        if p and os.path.isfile(p):
            got, cached = _digest(p, refresh)
            ok = bool(want) and got == want
            out["files"][rel] = {"path": os.path.abspath(p), "size_bytes": os.path.getsize(p), "sha256": got, "pin_sha256": want, "pinned": ok, "cached_utc": cached}
            out["lines"].append(f"weights={name} sha256={got[:12]} " + digest_memo.word("(pinned)" if ok else NOT_PINNED_NOTE, cached))
            out["pinned"] = out["pinned"] and ok
        else:
            out["files"][rel] = {"missing": True, "path": p}
            out["missing"].append(p or rel)
            out["pinned"] = False
    return out


def core_block() -> dict:
    """The imported shared core's version block (opt_core.manifest.core_block) — the core every activation was gated against (every entry's
    statement one: _core.core_gate). Passed through verbatim: at least ``{"version"}``; the exact shape is opt_core's own to define."""
    _core.ensure_importable()
    from opt_core.manifest import core_block as _cb
    return _cb()


def pins_summary(activation: dict) -> dict:
    """The pins REPORT as the run record carries it (reported, never gated): the differences named, the checkout's root / HEAD / counts, the stack verdict."""
    pd = activation.get("pins", {}) or {}
    co = (pd.get("detail") or {}).get("checkout") or {}
    return {"bad": pd.get("bad"), "checkout": {k: co.get(k) for k in ("root", "pinned", "files_checked", "git_head") if k in co} | {"modified": len(co.get("git_modified") or []), "differing": len(co.get("differing") or co.get("files_differing") or [])},
            "stack": (pd.get("detail") or {}).get("stack")}


def start(activation: dict, res: Resolution, request: dict, out_dir: str, tag: str, *, seed, det_level: int = det.DEFAULT_LEVEL) -> dict:
    act = {k: v for k, v in activation.items() if k not in ("pins", "weights_files", "weights_lines")}
    act["pins"] = pins_summary(activation)
    return {"schema": SCHEMA, "package_version": __version__, "core": core_block(), "started_utc": _utc(), "out_dir": out_dir, "tag": tag,
            "mode": res.mode, "tier": res.tier, "attach": res.attach, "line": res.line, "levers_planned": list(res.levers), "flags": list(res.flags),
            "activation": act, "request": request, "recipe": det.describe(seed, det_level),
            "weights": weights_record(activation.get("weights"), paths=activation.get("weights_paths")), "status": "running"}


def finish(man: dict, out_dir: str) -> str:
    man["finished_utc"] = _utc()
    if "driver_pass" in man:
        ev = man["driver_pass"].get("evidence") or {}
        man["levers_evidenced"] = registry_order(ev.get("applied", []))
        man["levers_missing"] = sorted(ev.get("missing", []))
        man["forbidden_lines"] = len(ev.get("forbidden", []))
    path = os.path.join(out_dir, NAME)
    os.makedirs(out_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(man, fh, indent=1, default=str)
    return path
