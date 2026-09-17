"""fastln_prebuilt.py — load a PREBUILT stream-correct Protenix fast_layer_norm extension (.so) instead of rebuilding it in every process.

Why (infopt_graphs/protenix/fastln_stream.py): protenix's fast_layernorm CUDA extension launches every kernel on the legacy default stream
(no stream argument at its launch sites), so under CUDA-graph capture the launch is not recorded and a replayed graph reads LayerNorm
outputs nobody wrote (NaN).  The fix rebuilds the IDENTICAL sources with `, 0, at::cuda::getCurrentCUDAStream()` at every launch site
(numerics-free, kernel-level bitwise) — but the rebuild costs an nvcc build.  This module ships the compiled .so for the pinned stack
(lib/fastln_prebuilt/: the .so, the patched sources, manifest.json) and, at install time in a fresh process:
  1. holds the .so to the SHA256SUMS line beside it (opt_core.gates.binary_refusal: re-hashed before anything maps it; unlisted,
     altered or no SHA256SUMS = refused by name on stderr) and checks the manifest against the live environment: torch + CUDA versions,
     sha256 of the stock kernel sources in site-packages (the .so must have been built from exactly these sources), sha256 of the .so itself;
  2. loads the .so as a Python extension module and checks it exposes every public symbol of the original extension;
  3. re-runs the equality check in THIS process (check_equal: 21 shape/dtype cases, torch.equal on output, mean, invvar, affine and
     non-affine) plus the side-stream and graph-replay tests (check_side_stream);
  4. swaps the module object exactly as install_stream_correct_fastln does (LN.fast_layer_norm_cuda_v2 = mod; _infopt_stream_patched).
If any check fails it returns installed=False with the reason and the caller falls back to the source rebuild (fastln_stream), which
re-checks bit equality itself.
"""
from __future__ import annotations
from opt_core.oom import is_oom          # rerouting handlers re-raise an out-of-memory first

import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import time
from typing import Any, Dict

import torch


def _sha(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for ch in iter(lambda: f.read(1 << 20), b""):
            h.update(ch)
    return h.hexdigest()


def _sums_refusal(so: str):
    """``None`` when the shipped .so is the ``SHA256SUMS`` line beside it and re-hashes to it (``opt_core.gates.binary_refusal``: one verdict
    per path per process; unlisted or altered bytes are refused by name with one stderr line), else the reason.  The prebuilt directory is
    the kit's, outside the core package, where the core holds a binary only when a SHA256SUMS sits beside it — so an absent SHA256SUMS is
    refused here, in the same words.  A refused .so is never mapped: the caller steps to the source rebuild exactly as for an absent one."""
    from opt_core import gates as G
    p = os.path.abspath(so)
    if not os.path.isfile(os.path.join(os.path.dirname(p), G.BINARY_SUMS)):
        reason = f"no {G.BINARY_SUMS} in its directory"
        print(f"[opt_core] {G.BINARY_SUMS}: refused {p}: {reason}", file=sys.stderr, flush=True)
        return reason
    return G.binary_refusal(p)


def _kernel_source_shas() -> Dict[str, str]:
    import protenix.model.layer_norm as LNPKG
    kdir = os.path.join(os.path.dirname(LNPKG.__file__), "kernel")
    return {fn: _sha(os.path.join(kdir, fn)) for fn in sorted(os.listdir(kdir)) if os.path.isfile(os.path.join(kdir, fn))}


def build_prebuilt_package(out_dir: str, build_dir: str | None = None) -> Dict[str, Any]:
    """Run ONCE on the pinned stack (GPU process): source rebuild via fastln_stream.build_stream_correct_fastln, bitwise + side-stream
    checks against the original, then copy the .so + patched sources + manifest.json into out_dir."""
    import shutil
    import protenix.model.layer_norm.layer_norm as LN
    from infopt_graphs.protenix import fastln_stream as FS

    build_dir = build_dir or FS.default_build_dir()
    t0 = time.time()
    rep = FS.build_stream_correct_fastln(build_dir)
    mod = rep.pop("module")
    so = mod.__file__
    orig = LN.fast_layer_norm_cuda_v2
    bw = FS.check_equal(orig, mod)
    ss = FS.check_side_stream(mod)
    os.makedirs(out_dir, exist_ok=True)
    shutil.copy(so, os.path.join(out_dir, os.path.basename(so)))
    srcdir = os.path.join(out_dir, "patched_src"); os.makedirs(srcdir, exist_ok=True)
    for fn in rep["patched_source_sha256"]:
        shutil.copy(os.path.join(build_dir, fn), os.path.join(srcdir, fn))
    man = {"so": os.path.basename(so), "so_sha256": _sha(so), "module_name": mod.__name__, "torch": torch.__version__, "cuda": torch.version.cuda,
           "python": sys.version.split()[0], "gpu": torch.cuda.get_device_name(0),
           "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "build_s": rep["build_s"], "total_s": round(time.time() - t0, 1),
           "source_sha256": rep["source_sha256"], "patched_source_sha256": rep["patched_source_sha256"], "patched_launches": rep["patched_launches"],
           "bitwise_at_build": bw["all_bitwise_equal"], "n_cases": bw["n_cases"], "side_stream_at_build": ss,
           "public_symbols_orig": sorted(s for s in dir(orig) if not s.startswith("_")), "public_symbols_new": sorted(s for s in dir(mod) if not s.startswith("_"))}
    json.dump(man, open(os.path.join(out_dir, "manifest.json"), "w"), indent=1)
    with open(os.path.join(out_dir, "SHA256SUMS"), "w", encoding="utf-8") as fh:      # the line install_prebuilt_fastln holds the .so to (sha256sum format)
        fh.write(f"{man['so_sha256']}  {man['so']}\n")
    return man


def install_prebuilt_fastln(pre_dir: str, require_bitwise: bool = True) -> Dict[str, Any]:
    t0 = time.time()
    rep: Dict[str, Any] = {"installed": False, "prebuilt_dir": pre_dir}
    import protenix.model.layer_norm.layer_norm as LN
    from infopt_graphs.protenix import fastln_stream as FS
    if getattr(LN, "_infopt_stream_patched", False):
        # A SECOND caller in the same process (another component installed the patch first) must
        # receive the check fields of the FIRST install (bitwise / side_stream / shas), not a bare "already installed" — callers gate graph
        # capture on bitwise.all_bitwise_equal being present and True.  If the patch was applied by the SOURCE-REBUILD path (infopt_graphs FASTLN_REPORT), use that.
        cached = getattr(LN, "_infopt_fastln_report", None)
        if not cached:
            try:
                from infopt_graphs.protenix import FASTLN_REPORT as _FR
                cached = dict(_FR) if _FR.get("bitwise") else None
            except Exception:
                cached = None
        if cached:
            rep.update({k: v for k, v in cached.items() if k not in ("prebuilt_dir",)})
            rep.update(installed=True, reason="already installed (equality report of the first install in this process)"); return rep
        # patched by someone who left no report: re-check the LIVE module in this process so the caller gets real fields (cheap)
        try:
            live = LN.fast_layer_norm_cuda_v2; orig = getattr(LN, "_infopt_fastln_orig", None)
            rep["bitwise"] = FS.check_equal(orig, live) if orig is not None else {"all_bitwise_equal": None, "n_cases": 0, "note": "original module not retained by the first installer"}
            rep["side_stream"] = FS.check_side_stream(live)
            rep.update(installed=True, reason="already installed (live module re-checked in this process)"); return rep
        except Exception as e:  # noqa
            if is_oom(e): raise                                   # an out-of-memory is the caller's to see: never rerouted (opt_core.oom)
            rep.update(installed=True, reason=f"already installed (re-check raised: {e!r})"); return rep
    mp = os.path.join(pre_dir, "manifest.json")
    if not os.path.exists(mp):
        rep["reason"] = f"no manifest at {mp}"; return rep
    man = json.load(open(mp))
    rep["manifest"] = {k: man.get(k) for k in ("so", "so_sha256", "torch", "cuda", "image", "gpu", "built_utc", "build_s", "patched_launches")}
    so = os.path.join(pre_dir, man["so"])
    checks: Dict[str, bool] = {"so_exists": os.path.exists(so)}
    if checks["so_exists"]:
        checks["so_sha256sums"] = _sums_refusal(so) is None          # the SHA256SUMS line beside the .so, before any loader maps it
        checks["so_sha256"] = _sha(so) == man["so_sha256"]
    checks["torch_version"] = torch.__version__ == man["torch"]
    checks["cuda_version"] = torch.version.cuda == man["cuda"]
    checks["kernel_sources_sha256"] = _kernel_source_shas() == man["source_sha256"]
    rep["checks"] = checks
    if not all(checks.values()):
        rep["reason"] = "prebuilt checks failed: " + json.dumps(checks); return rep
    name = man.get("module_name", "fast_layer_norm_cuda_v2_stream")
    try:
        loader = importlib.machinery.ExtensionFileLoader(name, so)
        spec = importlib.util.spec_from_file_location(name, so, loader=loader)
        mod = importlib.util.module_from_spec(spec)
        loader.exec_module(mod)
    except Exception as e:  # noqa
        rep["reason"] = f"load failed: {e!r}"; return rep
    orig = LN.fast_layer_norm_cuda_v2
    missing = [s for s in dir(orig) if not s.startswith("_") and not hasattr(mod, s)]
    checks["symbols"] = not missing
    if missing:
        rep["reason"] = f"missing symbols: {missing}"; return rep
    rep["bitwise"] = FS.check_equal(orig, mod)
    rep["side_stream"] = FS.check_side_stream(mod)
    rep["source_sha256"] = man["source_sha256"]; rep["patched_source_sha256"] = man["patched_source_sha256"]; rep["patched_launches"] = man["patched_launches"]
    rep["build_s"] = 0.0; rep["load_s"] = round(time.time() - t0, 2)
    ok = rep["bitwise"]["all_bitwise_equal"] and rep["side_stream"]["side_stream_equal"] and rep["side_stream"]["graph_replay_equal"] and rep["side_stream"]["graph_replay_finite"]
    if require_bitwise and not ok:
        rep["reason"] = f"equality check failed in this process: bitwise={rep['bitwise']['all_bitwise_equal']} side_stream={rep['side_stream']}"; return rep
    LN.fast_layer_norm_cuda_v2 = mod
    LN._infopt_stream_patched = True
    LN._infopt_fastln_orig = orig
    rep["installed"] = True; rep["reason"] = "prebuilt .so loaded and checked (bitwise + side-stream + graph replay)"
    LN._infopt_fastln_report = dict(rep)          # cached for later callers in this process (see the "already installed" branch)
    try:                                          # and mirror into infopt_graphs' FASTLN_REPORT so a later caller sees the same fields whichever path installed first
        from infopt_graphs.protenix import FASTLN_REPORT as _FR
        if not _FR.get("bitwise"):
            _FR.update({k: v for k, v in rep.items() if k in ("installed", "build_s", "patched_launches", "source_sha256", "patched_source_sha256", "side_stream", "bitwise", "reason")})
    except Exception:
        pass
    return rep
