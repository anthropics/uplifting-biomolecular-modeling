"""kernels.ln row ``fast_layernorm_ext`` -- the trunks' fused LayerNorm CUDA extension (fp32 statistics per row, Welford, output in the input's
dtype; gamma / beta in their own dtype) with every kernel launch on torch's CURRENT stream, loaded from a PREBUILT binary of the running
torch ABI key (no compiler at run time) carried under ``kernels/ln/rows/fast_layernorm_ext/`` (the producing kit's verbatim move: patched
sources, prebuilt binaries and their manifests; this module is the face's loader over that directory and imports nothing from it but the
binary).  Outputs are bitwise those of the trunks' own (legacy-stream) build of the extension -- same device
code --, so for an engine whose stock LayerNorm IS that extension this row is exact-class; against ATen it is tolerance-class (fp32
statistics, another summation order).  Unlike the legacy build it is CUDA-graph capture-safe.  See rows/fast_layernorm_ext/ (NOTICE / PROVENANCE of the move).

    from opt_core.kernels.ln import ext_loader as S
    S.available()                 -> (True, key) | (False, "no_prebuilt:<key>" | "import:<why>")
    mod = S.load()                -> the extension module (forward_* / backward_* entry points of the upstream extension), digest-checked
    y = S.layer_norm(x, (C,), weight, bias, eps)      # == mod.forward_*_affine(...)[0]; x CUDA, contiguous rows (copied contiguous otherwise)

Standard library at import; torch inside the calls.
"""
import hashlib
import importlib.machinery
import importlib.util
import json
import os
import sys
import sysconfig

PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ROWS_DIR = os.path.join(PKG_DIR, "rows", "fast_layernorm_ext")            # the carried bytes (absent until the move lands: every call refuses not_landed by name)
ROW = "fast_layernorm_ext"
EXPECTED_SYMBOLS = ("forward_none_affine", "forward_with_bias_affine", "forward_with_weight_affine", "forward_with_both_affine",
                    "backward_none_affine", "backward_with_bias_affine", "backward_with_weight_affine", "backward_with_both_affine")
UPSTREAM_SOURCE_SHA256 = {"layer_norm_cuda_kernel.cu": "ad75251f6fabbf59bbc83637a9600204c62a20c6ac563a742694bbfc8c0c180a",
                          "layer_norm_cuda.cpp": "b03d995042fede126b1e36a8e9c9e73433a598f523ad914789d7aeaeb8ac842d",
                          "compat.h": "24da65d5e0a58c6483abd60e7a6d6fca7e4abc9303f08c902975c77bdeacbc08",
                          "type_shim.h": "824a0dfdb5bfb1316187b3a3f7c10c9b41496b97259ddf84c882152ccec66a7a"}
PATCHED_CU_SHA256 = "fc1540b4410728ffd8c0b761fd4c3494290c5a6256e40d84a928f608c6cbbc36"
_STATE = {"modules": {}, "load_s": {}, "refused": {}}


class Unavailable(RuntimeError):
    """The row cannot serve in this process; ``.kind`` is the word (not_landed | no_prebuilt:<key> | so_digest:<key> | symbols:<missing> | load:<error>)."""
    def __init__(self, kind):
        RuntimeError.__init__(self, "fast_layernorm_ext unavailable: %s" % kind)
        self.kind = kind


def landed():
    """True when rows/fast_layernorm_ext/ carries at least one prebuilt manifest."""
    return bool(_manifests())


def _manifests():
    """{key: manifest path} for every manifest.json under ROWS_DIR that names a binary (key = torch<version>-cp<xy> from the manifest's torch /
    python fields, else the directory name)."""
    out = {}
    if not os.path.isdir(ROWS_DIR):
        return out
    for root, _dirs, files in os.walk(ROWS_DIR):
        if "manifest.json" not in files:
            continue
        p = os.path.join(root, "manifest.json")
        try:
            with open(p) as fh:
                m = json.load(fh)
        except (OSError, ValueError):
            continue
        so = m.get("so") or m.get("file") or (m.get("name", "") + ".so" if m.get("name") else None)
        if not so or not os.path.isfile(os.path.join(root, so)):
            sos = [f for f in files if f.endswith(".so")]
            if len(sos) != 1:
                continue
            so = sos[0]
        tv = str(m.get("torch") or "")
        py = str(m.get("python") or "")
        if tv:
            key = "torch%s-cp%s" % (tv, "".join(py.split(".")[:2]) if py else "%d%d" % sys.version_info[:2])
        else:
            key = os.path.basename(root)
        out[key] = (p, os.path.join(root, so), m)
    return out


def stack_key(torch_version=None):
    """The prebuilt key of a torch build: ``torch<M.m.p+cuXYZ>-cp<xy>`` (the running interpreter / torch when not given)."""
    if torch_version is None:
        import torch
        torch_version = torch.__version__
    tag = sysconfig.get_config_var("SOABI") or ""
    py = "cp%d%d" % sys.version_info[:2]
    if tag.startswith("cpython-"):
        py = "cp" + tag.split("-")[1]
    return "torch%s-%s" % (str(torch_version), py)


def keys():
    """The prebuilt keys carried under rows/fast_layernorm_ext/ (torch<version>-cp<xy>)."""
    return sorted(_manifests())


def key_for_stack(stack_word):
    """The carried key a cell-table stack word (e.g. 'H100:torch2.13.0+cu130/3.7.1/cueq0.11.1') or a torch version stands for, else None."""
    s = str(stack_word)
    for k in keys():
        tv = k.split("-")[0][len("torch"):]
        if ("torch" + tv) in s.replace(":", ":torch") or tv in s:
            return k
    return None


def manifest(key=None):
    key = key or stack_key()
    ms = _manifests()
    if not ms:
        raise Unavailable("not_landed")
    if key not in ms:
        raise Unavailable("no_prebuilt:%s" % key)
    p, so, m = ms[key]
    m = dict(m)
    m["_key"] = key
    m["_so"] = so
    m["_module_name"] = m.get("module_name") or m.get("name") or os.path.splitext(os.path.basename(so))[0]
    return m


def available(key=None):
    """(True, key) when a prebuilt of the key exists and its digest matches the manifest; else (False, '<kind>').  Does not load."""
    try:
        m = manifest(key)
        _check_digest(m)
        return True, m["_key"]
    except Unavailable as e:
        return False, e.kind


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_digest(m):
    so = m["_so"]
    if not os.path.isfile(so):
        raise Unavailable("no_prebuilt:%s(%s missing)" % (m["_key"], os.path.basename(so)))
    from opt_core.gates import binary_refusal  # noqa: PLC0415
    why = binary_refusal(so)
    if why:
        raise Unavailable("so_refused:%s(%s)" % (m["_key"], why))
    want = m.get("so_sha256")
    if want and _sha256(so) != want:
        raise Unavailable("so_digest:%s" % m["_key"])


def load(key=None):
    """The extension module of ``key`` (default: the running stack's), loaded once per process after the digest and symbol checks."""
    key = key or stack_key()
    mod = _STATE["modules"].get(key)
    if mod is not None:
        return mod
    if key in _STATE["refused"]:
        raise Unavailable(_STATE["refused"][key])
    import time
    t0 = time.time()
    try:
        m = manifest(key)
        _check_digest(m)
        import torch  # noqa: F401  (libtorch symbols must be loaded before the extension)
        name = m["_module_name"]
        try:
            loader = importlib.machinery.ExtensionFileLoader(name, m["_so"])
            spec = importlib.util.spec_from_file_location(name, m["_so"], loader=loader)
            mod = importlib.util.module_from_spec(spec)
            loader.exec_module(mod)
        except Exception as e:  # noqa: BLE001  (an ABI / arch mismatch names itself here)
            from opt_core.oom import is_oom
            if is_oom(e): raise
            raise Unavailable("load:%s:%s" % (key, ("%s: %s" % (type(e).__name__, e))[:160].replace(" ", "_")))
        missing = [s for s in EXPECTED_SYMBOLS if not hasattr(mod, s)]
        if missing:
            raise Unavailable("symbols:%s" % ",".join(missing))
    except Unavailable as e:
        _STATE["refused"][key] = e.kind
        raise
    _STATE["modules"][key] = mod
    _STATE["load_s"][key] = round(time.time() - t0, 3)
    return mod


def layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5, key=None):
    """y = LayerNorm(x) over ``normalized_shape`` by the extension's forward entry point matching the affine operands given (parameters cast
    to x's dtype as the trunks' module does; statistics fp32; output in x's dtype).  Returns the output tensor (mean / invvar dropped)."""
    mod = load(key)
    ns = tuple(int(s) for s in normalized_shape)
    xc = x if x.is_contiguous() else x.contiguous()
    if weight is not None and weight.dtype != xc.dtype:                      # the trunks' module convention: parameters in the activation's dtype
        weight = weight.to(xc.dtype)                                          # (bf16 x under autocast with fp32 gamma / beta -> bf16 operands, fp32 statistics, bf16 out)
    if bias is not None and bias.dtype != xc.dtype:
        bias = bias.to(xc.dtype)
    if weight is not None and bias is not None:
        out = mod.forward_with_both_affine(xc, ns, weight.contiguous(), bias.contiguous(), float(eps))
    elif weight is not None:
        out = mod.forward_with_weight_affine(xc, ns, weight.contiguous(), float(eps))
    elif bias is not None:
        out = mod.forward_with_bias_affine(xc, ns, bias.contiguous(), float(eps))
    else:
        out = mod.forward_none_affine(xc, ns, float(eps))
    return out[0]


def describe():
    out = {"landed": landed(), "rows_dir": os.path.relpath(ROWS_DIR, os.path.dirname(PKG_DIR)), "keys": keys(), "running_key": None, "loaded": sorted(_STATE["modules"]),
           "load_s": dict(_STATE["load_s"]), "refused": dict(_STATE["refused"]), "patched_cu_sha256": PATCHED_CU_SHA256, "upstream_source_sha256": dict(UPSTREAM_SOURCE_SHA256)}
    try:
        out["running_key"] = stack_key()
    except (ImportError, RuntimeError, AttributeError):                      # no torch importable here: the key stays unknown (describe never raises)
        pass
    for k in out["keys"]:
        try:
            m = manifest(k)
            out.setdefault("prebuilt", {})[k] = {"so": os.path.basename(m["_so"]), "so_sha256": (m.get("so_sha256") or "")[:16], "module": m["_module_name"],
                                                "torch": m.get("torch"), "cuda": m.get("cuda"), "archs": m.get("archs")}
        except Unavailable as e:
            out.setdefault("prebuilt", {})[k] = {"error": e.kind}
    return out
