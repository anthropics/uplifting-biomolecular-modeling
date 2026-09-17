#!/usr/bin/env python3
"""Build exactln's shipped CUBIN bundles: every (form, C, affine) instantiation the provider's cells serve, ONE NVRTC program per arch (all
name expressions in one module), written to prebuilt/exactln_fwd-sm<cc>.cubin + prebuilt/index.json (source sha256, compile options, NVRTC /
torch versions, the lowered name of every expression).  NVRTC cross-compiles: the archs need not be present (an H100 box builds sm_80 too).
Runs where exactln compiles today (cuda.bindings or the ctypes bindings).  A serving process then loads the module and compiles nothing.

    python -m opt_core.kernels.ln.exactln.build_prebuilt [--archs 90,80] [--widths 64,128,256,384,768] [--out <dir>]
"""
import argparse, hashlib, json, os, sys, time


FORMS = {"fp32": ("float32", "float32", "float32"), "bf16": ("bfloat16", "bfloat16", "bfloat16"),
         "bf16w": ("bfloat16", "float32", "float32"), "bf16o": ("bfloat16", "float32", "bfloat16")}     # (tin, tpar, tout) per provider form word
AFFINE = (3, 0)                                                                                         # weight+bias | neither (the provider admits both-or-neither)


def served_keys(widths):
    from opt_core.kernels.ln import exactln as E
    keys = []
    for form, (tin, tpar, tout) in FORMS.items():
        for C in widths:
            for a in AFFINE:
                keys.append((form, tin, tpar, tout, int(C), int(a), E._rows_per_warp(int(C))))
    return keys


def build(archs, widths, out_dir):
    from opt_core.kernels.ln import exactln as E
    nvrtc = E._bindings("nvrtc")
    src = E._source()
    index = {"schema": "exactln_prebuilt/v1", "source": "exactln_fwd.cu", "source_sha256": hashlib.sha256(src).hexdigest(), "opts": [o.decode() for o in E._opts(None)],
             "nvrtc": E._nvrtc_version(), "bindings": E._STATE["bindings"], "built_with_torch": __import__("torch").__version__, "python": sys.version.split()[0],
             "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "forms": {k: list(v) for k, v in FORMS.items()}, "affine": list(AFFINE), "widths": list(widths), "bundles": {}}
    keys = served_keys(widths)
    os.makedirs(out_dir, exist_ok=True)
    for cc in archs:
        opts = E._opts(cc)
        t0 = time.time()
        prog = E._check(nvrtc.nvrtcCreateProgram(src, b"exactln_fwd.cu", 0, [], []), "nvrtcCreateProgram")[0]
        try:
            exprs = []
            for (_form, tin, tpar, tout, C, a, rpw) in keys:
                expr = E._name_expr(tin, tpar, tout, C, a, rpw)
                if expr not in exprs:
                    exprs.append(expr)
                    E._check(nvrtc.nvrtcAddNameExpression(prog, expr.encode()), "nvrtcAddNameExpression")
            err = nvrtc.nvrtcCompileProgram(prog, len(opts), opts)[0]
            if int(err) != 0:
                n = E._check(nvrtc.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize")[0]
                log = b" " * n; nvrtc.nvrtcGetProgramLog(prog, log)
                raise RuntimeError("NVRTC compile (sm_%s) failed:\n%s" % (cc, log.decode(errors="replace")[:4000]))
            kernels = {}
            for expr in exprs:
                low = E._check(nvrtc.nvrtcGetLoweredName(prog, expr.encode()), "nvrtcGetLoweredName")[0]
                kernels[expr] = (bytes(low) if not isinstance(low, bytes) else low).decode()
            n = E._check(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize")[0]
            cubin = b" " * n
            E._check((nvrtc.nvrtcGetCUBIN(prog, cubin)[0],), "nvrtcGetCUBIN")
            cubin = bytes(cubin)
        finally:
            nvrtc.nvrtcDestroyProgram(prog)
        fname = "exactln_fwd-sm%s.cubin" % cc
        open(os.path.join(out_dir, fname), "wb").write(cubin)
        index["bundles"]["sm%s" % cc] = {"file": fname, "sha256": hashlib.sha256(cubin).hexdigest(), "bytes": len(cubin), "kernels": kernels, "n_kernels": len(kernels),
                                          "compile_s": round(time.time() - t0, 1)}
        print("sm_%s: %d kernels, %d bytes, %.1f s" % (cc, len(kernels), len(cubin), time.time() - t0), flush=True)
    json.dump(index, open(os.path.join(out_dir, "index.json"), "w"), indent=1, sort_keys=True); open(os.path.join(out_dir, "index.json"), "a").write("\n")
    return index


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--archs", default="90,80"); ap.add_argument("--widths", default="64,128,256,384,768")
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "prebuilt"))
    a = ap.parse_args()
    idx = build([x.strip() for x in a.archs.split(",")], [int(x) for x in a.widths.split(",")], a.out)
    print(json.dumps({k: v for k, v in idx.items() if k != "bundles"}), flush=True)
    print(json.dumps({k: {kk: vv for kk, vv in v.items() if kk != "kernels"} for k, v in idx["bundles"].items()}), flush=True)
