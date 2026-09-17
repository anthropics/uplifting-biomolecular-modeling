"""esmfold2_trimul v6.1 — Python face of the ESMFold2 TriangleMultiplication kernel package: generation v6 + the k3v6 RELEASE-ORDERING FIX
(src/PATCHES/k3v6_release_ordering_fix.diff, ../README.md §2): the two consumer-side mbarrier releases (x^T/z stage, residual slot) are issued
only after every ldmatrix instruction that read the slot has returned its data (+ fence.proxy.async), which removes the WAR race against the
producer's next TMA fill that made sibling builds of v6 nondeterministic.  Arithmetic is byte-for-byte v6's (the sealed v6 vectors replay BITWISE).

Op (one TriMul call on one batch element, c_z = 256, latent channels D = 256, bf16 in/out, fp32 math):
    zn  = LN_in(z)                                   z [N, N, 256]
    a|b = (zn Wp^T) * sigmoid(zn Wg_in^T) [* mask]   Wp, Wg_in [512, 256]: rows [0,256) -> a, [256,512) -> b
    x_ij = sum_k a_ik * b_jk   (outgoing)   |   x_ij = sum_k a_ki * b_kj   (incoming)          per channel d
    out = z + sigmoid(zn Wg_out^T) * (LN_out(x) Wz^T)                                        Wz, Wg_out [256, 256]
Route structure (FlashPairformer fpf_trimul_v4 lineage, credited in src/ef2_trimul_v5.py):  K1 (LN_in + gated dual projection + mask -> channel-
major bf16 planes ab[2D, Np, Np], Np = ceil16(N)) -> cuBLAS strided-batched bmm (bf16, fp32 accumulate) -> K3 (LN_out + out-projection + output
gate recomputed from z + residual).  Numerics class: TOLERANCE (fast tier).  Lever words (src/ef2_trimul_v5.py LEVERS): incnt, formtab, stagger =
engineering (GEMM form / plane orientation / tile order; values unchanged up to cuBLAS's summation order), sigmoid = tanh.approx gate sigmoid,
k3cute = K3 as the CuTe kernel of src/ef2_trimul_v6.py, lnfold = LN affine folded into K3's projection weights (W' = bf16(W*gamma), c = W beta fp32).

Kernels of this package (`kernel=`):
    "v5"      K1 `_k1x` + bmm + K3 `_k3x` (Triton, source + the kit's fpf_trimul_v4 cell rows in src/ef2_w4_fpf_trimul_v4_cells.json), levers
              DEFAULT_V5 = incnt, formtab, sigmoid, lnfold(no-op without k3cute) — the v5 interim configuration
    "v6"      the same route with K3 = `k3v6` (CUDA C++ / CuTe, sm_90a, 384 threads, ~221 KB smem; src/ef2_trimul_v6.py::K3_SRC == src/k3v6.cu),
              levers DEFAULT_V6 = incnt, formtab, sigmoid, k3cute, lnfold — this line's configuration as the engine runs it.  The K3 cubin is loaded
              from bin/k3v6.1_fastsig<0|1>_lnfold1.sm_90a.cubin (prebuilt; no compiler or headers needed) or built at first use with NVRTC
              (nvidia-cutlass + nvidia-cuda-cccl header wheels).  LNFOLD=0 builds are not shipped (the configuration of record folds the affine;
              with the fix an LNFOLD=0 build passed 0/60000+ but it is not part of this package's sealed set): refused unless ALLOW_UNSEALED_LNFOLD0.
    "k3cute" / "k3v5"   K3 alone (the CuTe kernel / the v5 Triton kernel) on given planes x [D, Np, Np]: k3_forward(...)

Tensor contract:
    z      [N, N, 256]  torch.bfloat16 CUDA contiguous (one batch element), 16 <= N;  mask [N, N] float/bool or None;  direction "outgoing"|"incoming"
    weights (pack_weights): norm_in_w/b [256], p_in_w [512, 256], g_in_w [512, 256], norm_out_w/b [256], p_out_w (Wz) [256, 256], g_out_w [256, 256]
            — the eight tensors ESMFold2's fused TriMul call receives (`TriangleMultiplication._engine`: norm_start, split_kernel_weights() of
            proj_bundle, norm_mix, proj_emit, proj_gate); any float dtype (rounded to bf16 = the values the stock bf16 path consumes)
    returns out [N, N, 256] bf16 = z + TriMul(z)

    face.pack_weights(...) -> w                     face.forward(z, direction, mask, w, kernel="v6", out=None, levers=None) -> out
    face.k3_forward(x, z, w, kernel="k3cute", out=None, fastsig=True, lnfold=True) -> out      x [256, Np, Np] bf16 planes (the bmm output)
    face.reference_fp32(z, direction, mask, <the eight weights>, eps=1e-5)                     the module's math in fp32 (bf16-rounded weights)
    face.load(prebuilt=None) -> (v5 module, v6 module)

Provenance, cells, numerics: ../README.md, ../CELLS.json.  Nothing is imported from any kit (../src/SOURCES.json records each copy's origin).
"""
import hashlib
import importlib.util
import json
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.dirname(HERE)
SRC = os.path.join(PKG, "src")
BIN = os.path.join(PKG, "bin")
KERNELS = ("v5", "v6", "k3cute", "k3v5")
ALLOW_UNSEALED_LNFOLD0 = False            # k3v6 built with LNFOLD=0 is not part of the sealed set (no cubin, no vectors); set True to build it with NVRTC for investigation
ALLOW_WITHDRAWN_LNFOLD0 = False           # v6 name kept for scripts written against v6 (either flag enables the build)
DEFAULT_V5 = dict(incnt=True, formtab=True, sigmoid=True, stagger=False, k3cute=False, lnfold=True)
DEFAULT_V6 = dict(incnt=True, formtab=True, sigmoid=True, stagger=False, k3cute=True, lnfold=True)
_MODS = {}
_INFO = dict(prebuilt={}, foreign_modules={}, cell=None)


def _sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _import(name, filename):
    """src/<filename> as module <name>.  ef2_trimul_v5 imports ef2_trimul_v6 BY NAME, so both are registered under their own names; a module of
    that name already imported from elsewhere (a kit's driver dir) is used as is and NAMED in describe()['foreign_modules']."""
    if name in _MODS:
        return _MODS[name]
    path = os.path.join(SRC, filename)
    if name in sys.modules and os.path.abspath(getattr(sys.modules[name], "__file__", "") or "") != os.path.abspath(path):
        _INFO["foreign_modules"][name] = getattr(sys.modules[name], "__file__", "?")
        _MODS[name] = sys.modules[name]
        return _MODS[name]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    _MODS[name] = mod
    return mod


def load(prebuilt=None):
    """-> (ef2_trimul_v5, ef2_trimul_v6).  prebuilt=None: every bin/ cubin whose manifest matches src/ is injected into v6's kernel table (no NVRTC);
    prebuilt=True: the four variants must all load (raises by name); prebuilt=False: nothing injected (v6 compiles with NVRTC at first use)."""
    v6 = _import("ef2_trimul_v6", "ef2_trimul_v6.py")
    v5 = _import("ef2_trimul_v5", "ef2_trimul_v5.py")
    if prebuilt is not False and not _INFO["prebuilt"]:
        _load_prebuilt(v6, required=(prebuilt is True))
    return v5, v6


def prebuilt_manifest():
    p = os.path.join(BIN, "manifest.json")
    return json.load(open(p)) if os.path.exists(p) else None


def _load_prebuilt(v6, required):
    man = prebuilt_manifest()
    problems = []
    if man is None:
        problems.append("bin/manifest.json absent")
    elif not torch.cuda.is_available():
        problems.append("no CUDA device")
    else:
        cc = torch.cuda.get_device_capability()
        src_sha = hashlib.sha256(v6.K3_SRC.encode()).hexdigest()
        for ent in man.get("kernels", []):
            if ent.get("kernel") != "k3v6":
                continue
            key = (int(ent["fastsig"]), int(ent["lnfold"]))
            cub = os.path.join(BIN, ent["cubin"])
            if f"sm_{cc[0]}{cc[1]}" != ent["arch"].rstrip("a"):
                problems.append(f"{ent['cubin']}: device is sm_{cc[0]}{cc[1]}, cubin is {ent['arch']}"); continue
            if not os.path.exists(cub) or _sha256(cub) != ent["cubin_sha256"]:
                problems.append(f"{ent['cubin']}: absent or sha256 differs from the manifest"); continue
            if ent["source_sha256"]["K3_SRC"] != src_sha:
                problems.append(f"{ent['cubin']}: built from a K3_SRC whose sha256 differs from src/ef2_trimul_v6.py's"); continue
            if ent.get("cfg") and dict(ent["cfg"]) != dict(v6._CFG):
                problems.append(f"{ent['cubin']}: built with cfg {ent['cfg']} != module _CFG {v6._CFG}"); continue
            if key in v6._STATE["kernels"]:
                continue
            v6._check_device()
            torch.cuda.init(); torch.empty(1, device="cuda")          # a current CUDA context before cuModuleLoadData (the kit always loads after the model is resident)
            kern = v6._Kernel(open(cub, "rb").read(), "k3v6", v6._smem_bytes())
            v6._STATE["kernels"][key] = kern
            v6._STATE["compiles"].append(dict(fastsig=key[0], lnfold=key[1], smem=kern.smem, origin=f"prebuilt:{ent['cubin']}"))
            _INFO["prebuilt"][f"fastsig{key[0]}_lnfold{key[1]}"] = ent["cubin"]
    if required and (problems or len(_INFO["prebuilt"]) < 2):
        raise RuntimeError(f"esmfold2_trimul: prebuilt k3v6 cubins not usable: {problems or 'fewer than the 2 shipped builds (fastsig 0|1, lnfold 1) in the manifest'}")
    _INFO["prebuilt_problems"] = problems
    return not problems


def cell(device=None):
    """(cfg = {k1, k3}, key) from src/ef2_w4_fpf_trimul_v4_cells.json for this device's 'cc|triton' (lookup: exact -> 'cc|*' -> '*|*'), as ef2_w4 resolves it."""
    if _INFO["cell"] is None:
        import triton
        cells = json.load(open(os.path.join(SRC, "ef2_w4_fpf_trimul_v4_cells.json")))
        cc = "%d.%d" % torch.cuda.get_device_capability(device)
        tmm = ".".join(triton.__version__.split(".")[:2])
        for key in (f"{cc}|{tmm}", f"{cc}|*", "*|*"):
            c = cells.get(key)
            if c:
                _INFO["cell"] = (dict(k1=dict(c["k1"]), k3=dict(c["k3"])), key)
                break
        if _INFO["cell"] is None:
            raise RuntimeError(f"esmfold2_trimul: no cell row for {cc}|{tmm}")
    return _INFO["cell"]


def pack_weights(norm_in_w, norm_in_b, p_in_w, g_in_w, norm_out_w, norm_out_b, p_out_w, g_out_w):
    """the fpf_trimul_v4 / v5 weight pack (== ef2_w4._t9_weights): LN affine as the bf16-rounded values held fp32; K1 reads W^T [256, 512] bf16,
    K3 reads Wz / Wg_out native [256, 256] bf16.  v6's folded weights (lnfold) are added lazily by ef2_trimul_v6.fold_weights on first use."""
    for name, t, shape in (("p_in_w", p_in_w, (512, 256)), ("g_in_w", g_in_w, (512, 256)), ("p_out_w", p_out_w, (256, 256)), ("g_out_w", g_out_w, (256, 256)),
                           ("norm_in_w", norm_in_w, (256,)), ("norm_out_w", norm_out_w, (256,))):
        if tuple(t.shape) != shape:
            raise ValueError(f"esmfold2_trimul: {name} {tuple(t.shape)} (expected {shape})")
    bf = torch.bfloat16
    def f32(t): return t.detach().to(bf).float().contiguous()
    def b16(t): return t.detach().to(bf)
    return dict(ln_in_w=f32(norm_in_w), ln_in_b=f32(norm_in_b), ln_out_w=f32(norm_out_w), ln_out_b=f32(norm_out_b),
                wgT_in=b16(g_in_w).t().contiguous(), wpT_in=b16(p_in_w).t().contiguous(), wz=b16(p_out_w).contiguous(), wg_out=b16(g_out_w).contiguous(),
                C=int(g_out_w.shape[0]), CH=int(p_out_w.shape[1]))


def forward(z, direction, mask, w, kernel="v6", out=None, levers=None, eps=1e-5, prebuilt=None):
    """z [N, N, 256] bf16 -> z + TriMul(z).  kernel 'v5' | 'v6' selects the lever set (DEFAULT_V5 / DEFAULT_V6) unless `levers` is given."""
    v5, v6 = load(prebuilt=prebuilt)
    if direction not in ("outgoing", "incoming"):
        raise ValueError(f"esmfold2_trimul: direction {direction!r}")
    lv = dict(dict(v5=DEFAULT_V5, v6=DEFAULT_V6)[kernel] if kernel in ("v5", "v6") else DEFAULT_V6)
    if levers:
        lv.update(levers)
    if kernel not in ("v5", "v6"):
        raise ValueError(f"esmfold2_trimul: forward serves kernel 'v5' or 'v6' (K3 alone: k3_forward); got {kernel!r}")
    _check_lnfold(lv.get("lnfold", True), k3cute=bool(lv.get("k3cute")))
    cfg, _ = cell(z.device)
    return v5.trimul_v5_forward(z, direction == "outgoing", mask, w, cfg, eps=eps, residual=True, stock_round=False, out=out, levers=lv)


def planes(z, direction, mask, w, levers=None, eps=1e-5):
    """K1 + bmm of the route (for K3-alone use): -> (x [D, Np, Np] bf16, N, Np).  Same code path trimul_v5_forward takes up to K3."""
    v5, _ = load()
    lv = dict(DEFAULT_V6); lv.update(levers or {})
    cfg, _ = cell(z.device)
    k1, k3 = v5._resolve_cfg(cfg)
    z3 = z.contiguous(); N = z3.shape[0]; D = int(w["wz"].shape[1]); Np = v5._ceil_to(N, 16)
    m2 = None
    if mask is not None:
        m2 = mask.to(torch.float32) if mask.dtype == torch.bool else mask
        m2 = m2.reshape(N, N).contiguous()
    outgoing = direction == "outgoing"
    form = v5._form(Np, outgoing, lv, z3.device); trans = (form == "NT") != bool(outgoing)
    ab = torch.empty((2 * D, Np, Np), dtype=torch.bfloat16, device=z3.device); x = torch.empty((D, Np, Np), dtype=torch.bfloat16, device=z3.device)
    v5.launch_k1(z3, w, m2, ab, N, Np, k1, eps, trans, lv)
    v5.launch_bmm(ab, x, form)
    return x, N, Np


def _check_lnfold(lnfold, k3cute=True):
    if k3cute and not lnfold and not (ALLOW_UNSEALED_LNFOLD0 or ALLOW_WITHDRAWN_LNFOLD0):
        raise RuntimeError("esmfold2_trimul v6.1: the k3v6 LNFOLD=0 build is not part of this package's sealed set (no cubin, no vectors); "
                           "use lnfold=True (the configuration of record) or the Triton K3 (kernel='k3v5' / 'v5'), or set face.ALLOW_UNSEALED_LNFOLD0 = True to build it with NVRTC")


def k3_forward(x, z, w, kernel="k3cute", out=None, fastsig=True, lnfold=True, eps=1e-5, prebuilt=None):
    """K3 alone: x [256, Np, Np] bf16 channel-major planes (the bmm output), z [N, N, 256] bf16 -> out = z + gate(z) * (LN_out(x^T) Wz^T)."""
    _check_lnfold(lnfold, k3cute=(kernel == "k3cute"))
    v5, v6 = load(prebuilt=prebuilt)
    z3 = z.contiguous(); N = z3.shape[0]; Np = x.shape[1]
    if out is None:
        out = torch.empty_like(z3)
    if kernel == "k3cute":
        return v6.launch_k3(x, z3, w, out, N, Np, eps, fastsig=bool(fastsig), lnfold=bool(lnfold), residual=True, stock_round=False)
    if kernel == "k3v5":
        cfg, _ = cell(z.device)
        lv = dict(DEFAULT_V5); lv["sigmoid"] = bool(fastsig); lv["k3cute"] = False
        v5.launch_k3(x, z3, w, out, N, Np, dict(cfg["k3"]), True, False, eps, lv)
        return out
    raise ValueError(f"esmfold2_trimul: k3_forward serves 'k3cute' | 'k3v5', got {kernel!r}")


def reference_fp32(z, direction, mask, norm_in_w, norm_in_b, p_in_w, g_in_w, norm_out_w, norm_out_b, p_out_w, g_out_w, eps=1e-5):
    """the module's math in fp32 from the bf16 input with bf16-rounded weights (the values every bf16 path consumes)."""
    f = lambda t: t.detach().to(torch.bfloat16).float()
    zf = z.float(); N = zf.shape[0]
    zn = torch.nn.functional.layer_norm(zf, (256,), f(norm_in_w), f(norm_in_b), eps)
    ab = (zn @ f(p_in_w).t()) * torch.sigmoid(zn @ f(g_in_w).t())
    if mask is not None:
        ab = ab * mask.reshape(N, N, 1).float()
    a, b = ab[..., :256], ab[..., 256:]
    if direction == "outgoing":
        x = torch.einsum("ikd,jkd->ijd", a, b)
    else:
        x = torch.einsum("kid,kjd->ijd", a, b)
    return zf + torch.sigmoid(zn @ f(g_out_w).t()) * (torch.nn.functional.layer_norm(x, (256,), f(norm_out_w), f(norm_out_b), eps) @ f(p_out_w).t())


def describe():
    d = dict(package="esmfold2_trimul", generation="v6.1", version=open(os.path.join(PKG, "VERSION")).read().strip() if os.path.exists(os.path.join(PKG, "VERSION")) else "?",
             kernels=KERNELS, src=SRC, prebuilt=dict(_INFO["prebuilt"]), prebuilt_problems=_INFO.get("prebuilt_problems"), foreign_modules=dict(_INFO["foreign_modules"]))
    if "ef2_trimul_v6" in _MODS:
        d["v6"] = _MODS["ef2_trimul_v6"].describe()
    if "ef2_trimul_v5" in _MODS:
        d["v5"] = _MODS["ef2_trimul_v5"].describe()
    try:
        d["cell"] = cell()
    except Exception as e:
        d["cell"] = f"unresolved: {e}"
    return d


if __name__ == "__main__":
    sys.path.insert(0, os.path.join(PKG, "tests"))
    import test_vectors
    sys.exit(test_vectors.main())
