"""The exact-class row ``triattn_exact``: this face's glue around the carried package ``opt_core/kernels/triattn_exact/``.

The carried package is a CUDA triangle-attention forward whose output equals the cuEquivariance op BIT FOR BIT on every (library version,
ops build, card, dtype, head_dim, shape class) cell its own table ``CELLS.json`` marks ``proven``; anything else it refuses BY NAME with a
typed ``Refused`` raised before any launch (never a silent fallback).  Its first call per (process, device, library version, route) runs a
self-check against the live library on a few small shapes.  On a card a cell certifies prebuilt binaries for, the member is loaded
through the CUDA driver (no compiler in the image); a cell certified on source compiles the member with ``nvcc`` at first use (cached on disk).

This module (standard library at import; torch only inside functions):

* ``load()``       routes ``import triattn_exact`` to the core copy (``opt_core.kernels.route``), points the package at its own cell table
                   and sources, returns its face module;
* ``install()``    the resolve-time preparation the provider runs once per CUDA process, OUTSIDE any served call or captured region: import,
                   build, and -- when a proven cell covers this process -- one small call through the face (which runs the self-check);
* ``serve()``      one call: the face, or on a typed refusal the kit's stock callable for THAT call, counted by reason word;
* ``library_gate`` the static (library version, ops build) check ``admits`` uses, read through ``importlib.metadata`` (no library import);
* ``counts()`` / ``evidence_line()``  what a kit prints next to its LEVER line: ``served n/m calls (refused: {reason: k})`` plus
                   ``cells=external:<name>`` / ``selfcheck=SKIPPED`` whenever the environment moved the package off this tree's evidence;
* capture rule     until the member has launched once in the process (the resolve-time probe does that where a proven cell exists) a
                   call made while a CUDA graph is being captured is refused by name (``capture_first_use``) and the stock op serves it.

Environment (the package's own words, passed through unchanged): ``TRIATTN_EXACT_CELLS`` (cell table path; default: the carried table),
``TRIATTN_EXACT_CACHE`` (build cache; default here: ``<JIT root>/<stack key>/triattn_exact`` when the process runs under opt_core's JIT-cache
root, else the package default), ``TRIATTN_EXACT_SKIP_SELFCHECK``, ``TRIATTN_EXACT_ARCHS``, ``CUEQ_TRIATTN_FALLBACK_THRESHOLD`` (the
library's; part of the cell key).
"""
from __future__ import annotations

import importlib
import json
import logging
import os
import re
import threading
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

ROW = "triattn_exact"
NAME = "triattn_exact"                                   # the routed top-level import name == the carried package directory under kernels/
HERE = os.path.dirname(os.path.abspath(__file__))
PKG_DIR = os.path.normpath(os.path.join(HERE, "..", NAME))
CELLS_FILE = os.path.join(PKG_DIR, "CELLS.json")
UPSTREAM_FILE = os.path.join(PKG_DIR, "UPSTREAM.json")
SOURCE_ATTRS = ("_SRC", "_SRC_V1", "_SRC_V2", "_SRC_V3", "_SRC_V4")   # the cuda_mma member's source-path constants (repointed under the package)
SOURCE_FP_LOGICAL = "csrc/cuda_mma/triattn_v3.cu"                 # the carried .cu's logical name in the package's route-source fingerprint (layout independent)
OPS_DISTS = ("cuequivariance-ops-torch-cu13", "cuequivariance-ops-torch-cu12")

log = logging.getLogger("opt_core.kernels.triattn_exact")
_lock = threading.Lock()
_STATE: Dict[str, Any] = {"face": None, "installed": None, "primed": False}   # primed: the member launched once in this process (a capture may replay it)
_COUNTS: Dict[str, Any] = {"served": 0, "refused": {}}
_LOGGED: set = set()


# ------------------------------------------------------------------------------------------------------------------ static records
def upstream() -> dict:
    """The carried package's record: {"checkpoint", "files": {relpath: sha256}, "routes", "cells"} (consumed by the tests and by describe)."""
    with open(UPSTREAM_FILE, encoding="utf-8") as fh:
        return json.load(fh)


def cells() -> List[dict]:
    """The package's cell rows ({id, status, route, when}); ``proven`` rows are the only ones the face serves."""
    path = os.environ.get("TRIATTN_EXACT_CELLS", CELLS_FILE)
    with open(path, encoding="utf-8") as fh:
        return list(json.load(fh).get("cells", []))


def proven_cells() -> List[dict]:
    return [c for c in cells() if c.get("status") == "proven"]


def _canon_version(v: str) -> str:
    m = re.match(r"^(\d+)\.(\d+)(?:\.(\d+))?", str(v))
    return f"{int(m.group(1))}.{int(m.group(2))}.{int(m.group(3) or 0)}" if m else str(v)


def ops_distribution() -> Optional[Tuple[str, str]]:
    """(version, ops build) of the installed cuequivariance ops distribution, read through importlib.metadata (nothing imported); None when no
    distribution is visible (a CPU interpreter answering table queries)."""
    from importlib import metadata
    for dist in OPS_DISTS:
        try:
            return _canon_version(metadata.version(dist)), dist.rsplit("-", 1)[1]
        except metadata.PackageNotFoundError:
            continue
    return None


def library_gate(R: dict) -> Tuple[bool, str]:
    """The row's static library check for ``admits``: the installed ops distribution must be a (version, build) the row record lists.
    Offline (no distribution visible) the row admits structurally and says so."""
    found = ops_distribution()
    if found is None:
        return True, "library: unverified offline"
    version, build = found
    if version not in list(R.get("lib_versions", [])):
        return False, f"lib_version:{version}"
    if build not in list(R.get("ops_builds", [])):
        return False, f"ops_build:{build}"
    return True, f"library {version} {build}"


# ------------------------------------------------------------------------------------------------------------------ import / build
def _default_cache_dir() -> Optional[str]:
    """<JIT root>/<stack key>/triattn_exact when the process runs under opt_core's JIT-cache tree (TRITON_CACHE_DIR = <root>/<key>/triton is
    how every kit lays it out), else None (the package's own default applies)."""
    tc = os.environ.get("TRITON_CACHE_DIR", "")
    if tc and os.path.basename(os.path.normpath(tc)) == "triton":
        return os.path.join(os.path.dirname(os.path.normpath(tc)), NAME)
    return None


def load():
    """Route ``import triattn_exact`` to the core copy, point the package at its own cell table and sources, and return its face module.
    Idempotent; raises what the import raises (the provider turns that into a refusal by name)."""
    with _lock:
        if _STATE["face"] is not None:
            return _STATE["face"]
        os.environ.setdefault("TRIATTN_EXACT_CELLS", CELLS_FILE)                 # the package's layout helpers (_paths) read these two: the carried cell
        os.environ.setdefault("TRIATTN_EXACT_CSRC", os.path.join(PKG_DIR, "csrc"))   # table and the kernel sources live INSIDE the carried directory here
        d = _default_cache_dir()
        if d and "TRIATTN_EXACT_CACHE" not in os.environ:
            os.environ["TRIATTN_EXACT_CACHE"] = d
        from opt_core import kernels as K
        K.route(NAME)
        face = importlib.import_module(NAME + ".face")
        member = importlib.import_module(NAME + ".cuda_mma")
        paths = importlib.import_module(NAME + "._paths")             # fail closed if the layout the package resolves is not the carried one
        if os.path.realpath(paths.cells_path()) != os.path.realpath(os.environ["TRIATTN_EXACT_CELLS"]) or not os.path.isdir(paths.csrc_dir("cuda_mma")):
            raise RuntimeError("layout:_paths")
        for attr in SOURCE_ATTRS:                                  # the member's source-path constants must resolve under the carried csrc/
            if not hasattr(member, attr) or os.path.dirname(getattr(member, attr)) != paths.csrc_dir("cuda_mma"):
                raise RuntimeError(f"layout:{attr}")
        _STATE["face"] = face
        return face


def refused_class():
    """The package's typed refusal (``triattn_exact.Refused``)."""
    load()
    return importlib.import_module(NAME).Refused


def _probe_call(face, device):
    """One small call the face will SERVE in this process (a proven cell covers the live library version / ops build / card) as
    (q, k, v, bias, mask, route), or (None, reason of the last refusal).  The cell match is the face's own (describe + select_route),
    so this never second-guesses the table; select_route also binds the certified binaries a prebuilt cell names."""
    import torch
    R = refused_class()
    dev = torch.device("cuda", torch.cuda.current_device()) if device is None else torch.device(device)
    last = "no proven cell in the table"
    for c in proven_cells():
        w = c.get("when", {})
        dts = w.get("dtype", "bfloat16"); dts = dts[0] if isinstance(dts, list) else dts
        dt = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}.get(str(dts), torch.bfloat16)
        D = w.get("head_dim", 32); D = int(D[0] if isinstance(D, list) else D)
        H = w.get("H", "any"); H = 4 if H == "any" else int((4 if 4 in H else H[0]) if isinstance(H, list) else H)
        S = max(int(w.get("S_min", 1)), 128)
        if w.get("S_max") is not None:
            S = min(S, int(w["S_max"]))
        N = S if w.get("N_equals_S") else 4
        bd = w.get("bias_dtype", ["float32"]); bd = bd if isinstance(bd, list) else [bd]
        bdt = torch.float32 if "float32" in bd else dt
        g = torch.Generator(device=dev).manual_seed(S)
        q, k, v = (torch.randn((1, N, H, S, D), device=dev, dtype=dt, generator=g) for _ in range(3))
        bias = torch.randn((1, 1, H, S, S), device=dev, dtype=bdt, generator=g)
        mask = torch.rand((1, N, 1, 1, S), device=dev, generator=g) > 0.2
        try:
            route, _cell = face.select_route(face.describe(q, k, v, bias, mask, None), dev)
        except R as e:
            last = e.reason
            continue
        return (q, k, v, bias, mask, route), None
    return None, last


def install(device=None) -> dict:
    """Resolve-time preparation (once per CUDA process, outside any served call / captured region): import the package and, when a proven
    cell covers this process, serve one small call through the face -- which loads the certified prebuilt binary for this card (no compiler
    involved) or, for a cell certified on source, compiles the member with nvcc (cached on disk), and runs the package's own self-check
    against the live library.  Returns the report; raises on an import failure or when the covering cell's route cannot be prepared here
    (the caller refuses the row by name for the process)."""
    face = load()
    R = refused_class()
    t0 = time.perf_counter()
    rep: Dict[str, Any] = {"proven_cells": len(proven_cells()), "cache_dir": os.environ.get("TRIATTN_EXACT_CACHE", "")}
    probe, why = _probe_call(face, device)
    if probe is None:
        rep["route"] = None
        rep["probe"] = f"nothing to serve in this process ({reason_word(why)}): every call refuses by name (the stock op serves)"
    else:
        import torch
        q, k, v, bias, mask, route = probe
        rep["route"] = route
        try:
            face.triangle_attention(q, k, v, bias, mask=mask)
            torch.cuda.synchronize()
        except R as e:                                              # the covering cell's route cannot run here (no compiler for a source cell, ...)
            raise RuntimeError(f"prepare_refused:{route}:{e.reason}") from None
        _STATE["primed"] = True
        rep["probe"] = f"served S={q.shape[3]} H={q.shape[2]} via {route} (self-check passed)"
        if route.startswith("_prebuilt."):                        # which driver binding loaded the certified binary (ctypes | cuda-python)
            try:
                rep["backend"] = type(importlib.import_module(NAME + "._prebuilt.driver").backend()).__name__.strip("_").replace("Backend", "").lower()
            except (ImportError, AttributeError, OSError, RuntimeError) as e:   # informative only
                rep["backend"] = f"unknown({type(e).__name__})"
    rep["prepare_s"] = round(time.perf_counter() - t0, 2)
    _STATE["installed"] = rep
    log.info("triattn_exact installed: %s", rep)
    return rep


# ------------------------------------------------------------------------------------------------------------------ serving
_REASON_WORDS = (
    (re.compile(r"^no proven cell"), "no_proven_cell"),
    (re.compile(r"CUEQ_TRIATTN_FALLBACK_THRESHOLD"), "small_s"),
    (re.compile(r"requires grad"), "autograd"),
    (re.compile(r"self-check"), "selfcheck_failed"),
    (re.compile(r"^route '"), "route_refuses"),
    (re.compile(r"kv_lengths"), "kv_lengths"),
    (re.compile(r"aligned|stride|contiguous"), "layout"),
    (re.compile(r"^unknown route"), "unknown_route"),
    (re.compile(r"nvcc|certifies prebuilt"), "cannot_prepare_here"),
    (re.compile(r"differs from the source certified|fingerprint"), "source_differs"),
)


def reason_word(reason: str) -> str:
    """A short stable word for a refusal reason (the census / evidence-line key)."""
    s = str(reason)
    for pat, word in _REASON_WORDS:
        if pat.search(s):
            return word
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")[:40] or "refused"


def _count_refused(word: str, text: str) -> None:
    r = _COUNTS["refused"]
    r[word] = r.get(word, 0) + 1
    if word not in _LOGGED:
        _LOGGED.add(word)
        log.info("triattn_exact refuses by name (%s): %s -- the stock op serves these calls", word, text[:200])


def serve(q, k, v, bias, mask, scale, stock: Callable):
    """One call: the face's output (bit-identical to the library op by the package's contract), or -- on a typed refusal, raised before any
    launch -- the kit's ``stock`` callable for this call.  Counted either way; nothing else is caught."""
    face = load()
    R = refused_class()
    if not _STATE["primed"] and _capturing(q):                    # first build / module load / self-check may not run while a graph records: by name, this call
        _count_refused("capture_first_use", "the member has not launched in this process yet and a CUDA graph is being captured")
        return stock(q, k, v, bias, mask=mask, scale=scale)
    try:
        out = face.triangle_attention(q, k, v, bias, mask=mask, scale=scale)
    except R as e:
        _count_refused(reason_word(e.reason), str(e))
        return stock(q, k, v, bias, mask=mask, scale=scale)
    _STATE["primed"] = True
    _COUNTS["served"] += 1
    return out


def _capturing(t) -> bool:
    """True while the current stream of ``t``'s device records a CUDA graph (metadata query; no sync)."""
    try:
        import torch
        return bool(t.is_cuda and torch.cuda.is_current_stream_capturing())
    except (ImportError, AttributeError, RuntimeError):
        return False


def cells_source() -> str:
    """``carried`` when the face reads this tree's cell table, else ``external:<file name>`` (TRIATTN_EXACT_CELLS names another file: what it
    serves is then not this tree's evidence, and every evidence line says so)."""
    path = os.environ.get("TRIATTN_EXACT_CELLS", CELLS_FILE)
    try:
        same = os.path.samefile(path, CELLS_FILE)
    except OSError:
        same = os.path.abspath(path) == os.path.abspath(CELLS_FILE)
    return "carried" if same else "external:" + os.path.basename(path)


def selfcheck_state() -> str:
    """``on`` unless TRIATTN_EXACT_SKIP_SELFCHECK=1 switched the package's self-check off (then ``SKIPPED``, shown on every evidence line)."""
    return "SKIPPED" if os.environ.get("TRIATTN_EXACT_SKIP_SELFCHECK") == "1" else "on"


def counts() -> dict:
    """{"served": n, "refused": {word: n}, "calls": n, "cells": carried | external:<name>, "selfcheck": on | SKIPPED, "installed": report | None}."""
    refused = dict(_COUNTS["refused"])
    return {"served": int(_COUNTS["served"]), "refused": refused, "calls": int(_COUNTS["served"]) + sum(refused.values()),
            "cells": cells_source(), "selfcheck": selfcheck_state(),
            "installed": dict(_STATE["installed"]) if _STATE["installed"] else None}


def reset_counts() -> None:
    _COUNTS["served"] = 0
    _COUNTS["refused"] = {}


def primed() -> bool:
    """True once the member has launched in this process (resolve-time probe or a served call): a CUDA-graph capture may then include it."""
    return bool(_STATE["primed"])


def evidence_line() -> str:
    """``triattn_exact: served n/m calls (refused: {word: k, ...})`` -- the line a kit prints beside its LEVER line."""
    c = counts()
    inner = ", ".join(f"{w}: {n}" for w, n in sorted(c["refused"].items()))
    tail = ("" if c["cells"] == "carried" else f" cells={c['cells']}") + ("" if c["selfcheck"] == "on" else f" selfcheck={c['selfcheck']}")
    return f"triattn_exact: served {c['served']}/{c['calls']} calls (refused: {{{inner}}}){tail}"


__all__ = ["ROW", "NAME", "PKG_DIR", "upstream", "cells", "proven_cells", "ops_distribution", "library_gate", "load", "install", "serve",
           "refused_class", "reason_word", "counts", "reset_counts", "evidence_line", "cells_source", "selfcheck_state", "primed"]
