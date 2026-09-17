"""infopt_graphs.protenix.fastln_stream — make Protenix's fast_layernorm CUDA extension stream-correct (capture-safe).

Protenix's fast LayerNorm extension (protenix/model/layer_norm/kernel/layer_norm_cuda_kernel.cu) writes every forward launch as
    LayerNormForwardV2<T, V><<<grid, block>>>(...)
i.e. WITHOUT a stream argument -> the kernel always runs on the LEGACY DEFAULT STREAM, whatever torch's current stream is.
In stock eager inference torch's current stream IS the legacy default stream, so the stock path is correct.  Under CUDA-graph
capture (which must run on a non-default stream) the launch is not recorded — the extension never checks the launch error —
so a replayed graph reads LayerNorm outputs that were never written (=> NaN coordinates).  The
same launches also race with any multi-stream use of the model.

Fix (numerics-free): rebuild the SAME sources with `, 0, at::cuda::getCurrentCUDAStream()` added to every `<<<grid, block>>>`
launch (forward and backward), with the SAME compiler flags Protenix uses (protenix/model/layer_norm/torch_ext_compile.py),
into a job-local build directory, and swap the module object that `protenix.model.layer_norm.layer_norm` calls.  The kernel
code, grid/block geometry, dtypes and epsilon are untouched; only the stream the kernel is enqueued on changes.
`check_equal()` compares the patched extension with the original on a set of shapes/dtypes (torch.equal on output, mean,
invvar) — kernel-level bit equality.

Usage:  from infopt_graphs.protenix.fastln_stream import install_stream_correct_fastln
        rep = install_stream_correct_fastln()   # builds (once per machine and stack; cached under build_dir), checks, swaps
The build lands in `default_build_dir()` (<TORCH_EXTENSIONS_DIR or torch's extension cache>/fastln_stream) and is reused by every later
process: the patched sources are rewritten only when their text changes, so ninja finds nothing to do and the module loads in seconds.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import time
from typing import Any, Dict, List

import torch

_LAUNCH_RE = re.compile(r"<<<\s*([^<>]*?)\s*>>>\s*\(")


def _patch_launch_sites(src: str) -> tuple[str, int]:
    """Add `, 0, at::cuda::getCurrentCUDAStream()` to every launch that has only <grid, block> (2 args)."""
    n = 0

    def repl(m):
        nonlocal n
        args = m.group(1)
        # count top-level commas (dim3(a, b) contains commas inside parentheses)
        depth = 0
        top_commas = 0
        for ch in args:
            if ch == "(":
                depth += 1
            elif ch == ")":
                depth -= 1
            elif ch == "," and depth == 0:
                top_commas += 1
        if top_commas == 1:  # <<<grid, block>>>
            n += 1
            return f"<<<{args}, 0, at::cuda::getCurrentCUDAStream()>>>("
        return m.group(0)

    out = _LAUNCH_RE.sub(repl, src)
    return out, n


def build_stream_correct_fastln(build_dir: str, verbose: bool = False) -> Dict[str, Any]:
    import protenix.model.layer_norm as LNPKG
    from protenix.model.layer_norm.torch_ext_compile import compile as ptx_compile

    kdir = os.path.join(os.path.dirname(LNPKG.__file__), "kernel")
    os.makedirs(build_dir, exist_ok=True)
    rep: Dict[str, Any] = {"kernel_dir": kdir, "build_dir": build_dir, "source_sha256": {}, "patched_launches": {}}
    srcs: List[str] = []
    for fn in sorted(os.listdir(kdir)):
        p = os.path.join(kdir, fn)
        if not os.path.isfile(p):
            continue
        raw = open(p, "rb").read()
        rep["source_sha256"][fn] = hashlib.sha256(raw).hexdigest()
        text = raw.decode()
        if fn.endswith(".cu"):
            if "ATen/cuda/CUDAContext.h" not in text:
                text = "#include <ATen/cuda/CUDAContext.h>\n" + text
            text, n = _patch_launch_sites(text)
            rep["patched_launches"][fn] = n
        dst = os.path.join(build_dir, fn)
        if not (os.path.isfile(dst) and open(dst).read() == text):     # unchanged sources keep their mtime: the build tool sees nothing to rebuild and the cached .so loads
            open(dst, "w").write(text)
        if fn.endswith((".cu", ".cpp")):
            srcs.append(dst)
    rep["patched_source_sha256"] = {fn: hashlib.sha256(open(os.path.join(build_dir, fn), "rb").read()).hexdigest() for fn in rep["source_sha256"]}
    t0 = time.time()
    mod = ptx_compile(name="fast_layer_norm_cuda_v2_stream", sources=srcs, extra_include_paths=[build_dir], build_directory=build_dir)
    rep["build_s"] = round(time.time() - t0, 1)
    rep["module"] = mod
    return rep


def default_build_dir() -> str:
    """Where the stream-correct build lives when the caller names no directory: `<TORCH_EXTENSIONS_DIR>/fastln_stream` when that variable is
    set (the kit configs export it next to the Triton cache), else torch's own extension cache for this interpreter and CUDA version
    (`~/.cache/torch_extensions/py<ver>_cu<ver>/fastln_stream`). One directory per machine and stack: built once, loaded by later processes."""
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if root:
        return os.path.join(root, "fastln_stream")
    try:
        from torch.utils.cpp_extension import _get_build_directory      # torch's cache root for JIT extensions (creates the directory)
        return _get_build_directory("fastln_stream", False)
    except Exception:
        return os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions", "fastln_stream")


def check_equal(orig, patched, device="cuda") -> Dict[str, Any]:
    """The original extension vs the stream-correct rebuild on the current (legacy) stream: outputs must be bitwise equal."""
    torch.manual_seed(0)
    cases = []
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for rows, cols in ((1302, 128), (167, 384), (167 * 167, 128), (4096, 768), (33, 449), (7, 16), (1, 384)):
            x = torch.randn(rows, cols, device=device, dtype=dtype) * 3
            w = torch.randn(cols, device=device, dtype=dtype)
            b = torch.randn(cols, device=device, dtype=dtype)
            eps = 1e-5
            o1 = orig.forward_with_both_affine(x, (cols,), w, b, eps)
            o2 = patched.forward_with_both_affine(x, (cols,), w, b, eps)
            torch.cuda.synchronize()
            eq = [bool(torch.equal(a, c)) for a, c in zip(o1, o2)]
            o3 = orig.forward_none_affine(x, (cols,), eps)
            o4 = patched.forward_none_affine(x, (cols,), eps)
            torch.cuda.synchronize()
            eq2 = [bool(torch.equal(a, c)) for a, c in zip(o3, o4)]
            cases.append({"dtype": str(dtype), "rows": rows, "cols": cols, "both_affine_equal": eq, "none_affine_equal": eq2,
                          "finite": bool(torch.isfinite(o2[0]).all())})
    ok = all(all(c["both_affine_equal"]) and all(c["none_affine_equal"]) and c["finite"] for c in cases)
    return {"all_bitwise_equal": ok, "n_cases": len(cases), "cases": cases}


def check_side_stream(patched, device="cuda") -> Dict[str, Any]:
    """On a side stream, the patched extension must give the same result as on the default stream (the original cannot:
    it launches on the legacy stream regardless)."""
    x = torch.randn(1302, 128, device=device) * 3
    w = torch.randn(128, device=device); b = torch.randn(128, device=device)
    ref = patched.forward_with_both_affine(x, (128,), w, b, 1e-5)[0].clone()
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        y = x * 1.0  # produce the input on the side stream, then LN must follow it in stream order
        out = patched.forward_with_both_affine(y, (128,), w, b, 1e-5)[0]
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    # capture test: LN inside a CUDA graph
    g = torch.cuda.CUDAGraph()
    xs = x.clone()
    with torch.cuda.graph(g):
        og = patched.forward_with_both_affine(xs, (128,), w, b, 1e-5)[0]
    g.replay(); torch.cuda.synchronize()
    return {"side_stream_equal": bool(torch.equal(out, ref)), "graph_replay_equal": bool(torch.equal(og, ref)),
            "graph_replay_finite": bool(torch.isfinite(og).all())}


def install_stream_correct_fastln(build_dir: str = None, require_bitwise: bool = True) -> Dict[str, Any]:
    """Build (cached), check and swap in the stream-correct extension.  Returns the report (no module objects)."""
    import protenix.model.layer_norm.layer_norm as LN
    if getattr(LN, "_infopt_stream_patched", False):
        cached = getattr(LN, "_infopt_fastln_report", None)      # return the first installer's equality report (prebuilt or rebuild)
        if isinstance(cached, dict) and cached.get("installed"):
            rep = dict(cached); rep["reason"] = "already installed (cached equality report of the first install: " + str(cached.get("reason")) + ")"; rep["cached_report"] = True
            return rep
        return {"installed": True, "reason": "already installed (NO cached equality report)", "cached_report": False}
    orig = LN.fast_layer_norm_cuda_v2
    build_dir = build_dir or default_build_dir()
    rep = build_stream_correct_fastln(build_dir)
    mod = rep.pop("module")
    rep["bitwise"] = check_equal(orig, mod)
    rep["side_stream"] = check_side_stream(mod)
    if require_bitwise and not rep["bitwise"]["all_bitwise_equal"]:
        raise RuntimeError(f"stream-correct fast LayerNorm is NOT bitwise equal to the original: {rep['bitwise']}")
    LN.fast_layer_norm_cuda_v2 = mod
    LN._infopt_stream_patched = True
    LN._infopt_fastln_orig = orig
    rep["installed"] = True; rep["cached_report"] = False
    if "reason" not in rep: rep["reason"] = "source rebuild loaded and checked (bitwise + side-stream)"
    LN._infopt_fastln_report = dict(rep)        # cached for any later installer in this process
    return rep


def uninstall_stream_correct_fastln():
    import protenix.model.layer_norm.layer_norm as LN
    if getattr(LN, "_infopt_stream_patched", False):
        LN.fast_layer_norm_cuda_v2 = LN._infopt_fastln_orig
        LN._infopt_stream_patched = False
        LN._infopt_fastln_report = None
