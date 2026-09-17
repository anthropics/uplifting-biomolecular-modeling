"""Fused residual and residual + LayerNorm + qkv CUDA kernels composed into the ``fused`` lever (bf16 activations, packed
[tokens, hidden] varlen rows, Transformer Engine LayerNormLinear / LayerNormMLP, flash-attn varlen, Triton rotary), delivered as
entries for the fused patch's `_patch._OPS` seam contract:

  ops(model) -> dict — the entries enabled in TESTED:
    'residual'(x, y, sf) -> bf16 [T, D]      = the block's `x + y / sf` (aten div-by-scalar + add) in ONE launch: t = bf16_rn(float(y) * inv_b),
                                               r = bf16_rn(float(x) + float(t)), inv_b = fp32(1) / fp32(sf) (ATen's opmath form); 16-byte vectors.
                                               inv_b is computed once per sf value and cached (the key is the Python float sf, so the same sf on
                                               different modules shares one constant and a different sf gets its own): the same fp32 value the
                                               per-call two-tensor expression produces — bitwise by construction, without two CPU tensors per call.
    'residual_ln_qkv'(x, y, sf, mod) -> (x_new, qkv) = the residual above (x_new, the same bytes as 'residual') + TE's ln_fwd_general
                                               numerics mirrored (mode 2) on it + te.Linear(mod.weight) on the LN output = the module's own qkv.
  Not fused here, and why: the rotary (the Triton rotary is already one kernel per q/k on the packed layout), the SwiGLU (TE gated_act is
  one kernel inside LayerNormMLP), the ffn-side residual+LN (LayerNormMLP cannot be split), the final norm (1 launch/forward), the q|k LN
  (the qk_rotary item's).
  The extension is built ONCE from kernels.cu by build.py's recipe (EXTENSION.json `build`: torch's JIT extension builder driving the
  CUDA toolkit's nvcc) into a per-user cache keyed by the ABI key {torch, cuda, sm, python} + a sha over the source and recipe — up front
  by the install step (`python -m esmc_opt.kits.residual_ln.build`, which `run.sh install` runs), else at the first apply on a machine;
  every later process imports that cached binary (one info line names it). Without nvcc / ninja nothing can be built: the kernels cannot
  engage and the refusal names what is missing.
"""
import hashlib
import importlib.util
import json
import os
import time

KIT = "residual_ln"
KIT_DIR = os.path.dirname(os.path.abspath(__file__))
TESTED = {"residual": True, "residual_ln_qkv": True}      # which seam entries ops() returns
CACHE_HOME = os.path.join("~", ".cache", "esmc_sdkfused")   # the per-user cache of in-place builds: <CACHE_HOME>/<key_dir>__<build.cache_tag()[:12]>/ (expanduser at use)
SERVED_SM = ("80", "90")                                     # the compute capabilities the kit serves (A100 / H100 class): what the install step builds for when no device is visible
BUILT_FILE = "built.json"                                   # the record of a finished in-place build beside its build/ dir (present = the cache holds a loadable binary)
_state = {"ext": None, "ext_load": None, "lin": {}, "inv_b": {}}


class KitRefused(RuntimeError):
    pass


def runtime_key() -> dict:
    from . import build as _build
    return _build.runtime_key()


def cache_dir(key: dict, tag: str) -> str:
    """<CACHE_HOME>/<key_dir>__<tag[:12]>, tag = build.cache_tag() (source bytes + recipe) — one dir per (ABI key, source, recipe); a changed
    kernels.cu or flag list never meets a stale binary."""
    from . import build as _build
    return os.path.join(os.path.expanduser(CACHE_HOME), f"{_build.key_dir(key)}__{tag[:12]}")


def build_line(key: dict, wall_s: float, cdir: str) -> str:
    """The ONE info line of a process that built the extension (the install step, or the first apply on a machine)."""
    from . import build as _build
    return f"[{KIT}] extension {_build.key_dir(key)}: built from kernels.cu in {wall_s:.1f}s (cached at {cdir})"


def cache_line(key: dict, wall_s: float, cdir: str) -> str:
    """The ONE info line of a process that imported the cached build."""
    from . import build as _build
    return f"[{KIT}] extension {_build.key_dir(key)}: loaded the cached build at {cdir} in {wall_s:.2f}s"


def _import_so(path: str):
    """The extension module from the cached binary on disk: an ExtensionFileLoader import by path. The module name is the binary's own
    stem — torch's builder names the module `<extension>` or, when one process builds it a second time (the install step's second
    capability), `<extension>_v<n>`; the binary's init symbol follows that name, so the import must too."""
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path); ext = importlib.util.module_from_spec(spec); spec.loader.exec_module(ext)
    return ext


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _cache_refusal(cdir: str, *files: str):
    """Why the cached build in `cdir` must not be imported (reason + fix), or None. The sha in built.json sits beside the binary, so whoever
    can write the directory can rewrite both: the owner / mode rule is the check. `cdir`, the binary's directory and — unless `cdir` is closed
    to group and other — the files must not be writable by group or other, and all must belong to this uid or to root; uid 0 accepts any owner
    (a container's root reading a bind-mounted host directory)."""
    uid, closed = os.geteuid(), os.stat(cdir).st_mode & 0o077 == 0
    for p in (cdir,) + tuple(sorted({os.path.dirname(f) for f in files} - {cdir})) + files:
        st = os.stat(p)
        if st.st_mode & 0o022 and not (closed and p != cdir):
            return f"{p} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {p}"
        if uid != 0 and st.st_uid not in (uid, 0):
            return f"{p} belongs to uid {st.st_uid}, not to this process (uid {uid}) or root; fix: remove {cdir} (it is rebuilt), or chown it"
    return None


def _cached_build(cdir: str):
    """(so_path, record) of a finished in-place build in `cdir`, else None: built.json present and the binary it names on disk at its
    recorded size and sha. A build another account could have written (`_cache_refusal`) is REFUSED on stderr — what, why, the fix — and is
    then no cached build: `load_ext` builds again, in a private directory."""
    meta = os.path.join(cdir, BUILT_FILE)
    if not os.path.isfile(meta):
        return None
    try:
        with open(meta) as f:
            rec = json.load(f)
        so = os.path.join(cdir, rec["so_rel"])
        why = _cache_refusal(cdir, meta, so) if os.path.isfile(so) else None
        if why is not None:
            import sys
            print(f"[{KIT}] REFUSED cached build {cdir}: {why} — not imported; built again for this process in a private directory", file=sys.stderr, flush=True)
            _state["refused"] = cdir
            return None
        if os.path.isfile(so) and os.path.getsize(so) == int(rec["size"]) and _sha256_file(so) == rec["so_sha256"]:
            return so, rec
    except (OSError, ValueError, KeyError, TypeError):
        pass
    return None


def _writable_cache_dir(cdir: str) -> str:
    """`cdir` created (parents too) when the user's cache home is writable; else a fresh temporary dir (the build then serves this process
    and its line says where it went)."""
    try:
        if _state.get("refused") == cdir:                  # a refused cache dir is not built into either
            raise OSError("refused")
        os.makedirs(cdir, mode=0o700, exist_ok=True)       # closed to group and other: what _cache_refusal accepts whatever the builder's file modes
        probe = os.path.join(cdir, f".w{os.getpid()}")
        with open(probe, "w") as f:
            f.write("")
        os.remove(probe)
        return cdir
    except OSError:
        import tempfile
        return tempfile.mkdtemp(prefix="esmc_sdkfused_")


def _build_in_place(key: dict, cdir: str):
    """Build kernels.cu for `key` into <cdir>/build by build.py's recipe, record built.json, return (module, record). KitRefused names
    a missing toolchain (nothing attempted) or a failed build (the builder's output kept in <cdir>/build_error.txt)."""
    from . import build as _build
    tc = _build.toolchain()
    if tc["missing"]:
        raise KitRefused(f"the extension for {_build.key_dir(key)} is not built and cannot be built here — missing {', '.join(tc['missing'])}; "
                         f"the fused kernels cannot engage (install the CUDA toolkit's nvcc and ninja, then `run.sh install` builds it once)")
    cdir = _writable_cache_dir(cdir)
    try:
        rec = _build.build(os.path.join(cdir, "build"), key["sm"], kit_dir=KIT_DIR, verbose=False)
    except Exception as e:  # noqa: BLE001 — the builder's failure is named with its log, never swallowed
        log = os.path.join(cdir, "build_error.txt")
        try:
            with open(log, "w") as f:
                f.write(f"{type(e).__name__}: {e}\n")
        except OSError:
            log = "(unwritable)"
        raise KitRefused(f"the build of kernels.cu for {_build.key_dir(key)} failed — {type(e).__name__}: "
                         f"{str(e).strip().splitlines()[0][:240] if str(e).strip() else ''} (full output: {log})") from e
    built = {"key": key, "key_dir": _build.key_dir(key), "so_rel": os.path.relpath(rec["so_path"], cdir), "size": rec["size"], "so_sha256": rec["so_sha256"],
             "cu_sha256": _build.source_sha256(KIT_DIR), "arch": rec["arch"], "nvcc": rec["nvcc"], "recipe": rec["recipe"], "wall_s": rec["wall_s"],
             "build_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    tmp = os.path.join(cdir, f".{BUILT_FILE}.{os.getpid()}")
    with open(tmp, "w") as f:
        json.dump(built, f, indent=1)
    os.replace(tmp, os.path.join(cdir, BUILT_FILE))
    return rec["module"], built, cdir


def load_ext():
    """The extension for THIS runtime, once per process: (1) the cached build for the runtime's ABI key (made by the install step or an
    earlier process) — imported, ONE info line; (2) else kernels.cu built now by build.py's recipe — ONE info line naming the key, the
    seconds and the cache dir; a missing toolchain or a failed build is a KitRefused naming it (the lever's refusal: the package's exit
    rule names the escapes)."""
    if _state["ext"] is not None:
        return _state["ext"]
    from . import build as _build
    t0 = time.time()
    key = runtime_key()
    cdir = cache_dir(key, _build.cache_tag(KIT_DIR))
    hit = _cached_build(cdir)
    if hit is not None:
        so, rec = hit
        ext = _import_so(so)
        wall = round(time.time() - t0, 4)
        print(cache_line(key, wall, cdir), flush=True)
        _state["ext"] = ext; _state["ext_load"] = {"source": "cache", "path": so, "key": key, "key_dir": _build.key_dir(key), "wall_s": wall, "cache_dir": cdir, "built": rec}
        return ext
    ext, rec, cdir = _build_in_place(key, cdir)
    wall = round(time.time() - t0, 4)
    print(build_line(key, rec["wall_s"], cdir), flush=True)
    _state["ext"] = ext; _state["ext_load"] = {"source": "built", "path": os.path.join(cdir, rec["so_rel"]), "key": key, "key_dir": _build.key_dir(key), "wall_s": wall, "cache_dir": cdir, "built": rec}
    return ext


def build_cache(sms=None) -> list:
    """The install step's build: the extension built into the per-user cache (the dir load_ext() reads) for each compute capability in
    `sms` — default: device 0's; with no CUDA device visible (an image builder), every capability the kit serves (SERVED_SM). A
    capability whose cache already holds a build for this source and recipe is left alone. One line per capability; returns the records."""
    from . import build as _build
    key0 = runtime_key()
    sms = [str(x) for x in (sms or ([key0["sm"]] if key0["sm"] != "?" else SERVED_SM))]
    out = []
    for sm in sms:
        key = dict(key0, sm=sm)
        cdir = cache_dir(key, _build.cache_tag(KIT_DIR))
        hit = _cached_build(cdir)
        if hit is not None:
            print(f"[{KIT}] extension {_build.key_dir(key)}: already built at {cdir}", flush=True)
            out.append({"key": key, "cache_dir": cdir, "built": hit[1], "source": "cache"})
            continue
        _, rec, cdir = _build_in_place(key, cdir)
        print(build_line(key, rec["wall_s"], cdir), flush=True)
        out.append({"key": key, "cache_dir": cdir, "built": rec, "source": "built"})
    return out


def ext_load() -> dict | None:
    return _state["ext_load"]


def inv_b_r1_form(sf: float) -> float:
    """The per-call form (the reference the cached constant must equal): ATen's div-by-scalar inv_b = opmath_t(1.0) /
    scalar_value<opmath_t>(sf) with opmath_t = float for bf16 tensors."""
    import torch
    return float(torch.tensor(1.0, dtype=torch.float32) / torch.tensor(float(sf), dtype=torch.float32))


def inv_b_of(sf: float) -> float:
    """r1_1: the SAME constant, computed once per sf value (cache keyed by the Python float sf) — no tensor work per call."""
    key = float(sf)
    v = _state["inv_b"].get(key)
    if v is None:
        v = inv_b_r1_form(key)
        _state["inv_b"][key] = v
    return v


def residual(x, y, sf):
    """_OPS['residual'] — the block's x + y / sf on bf16 [T, D] (packed), one launch, bitwise the stock pair."""
    return load_ext().residual_bf16(x, y, inv_b_of(sf))


def _te_linear_for(mod):
    """A te.Linear sharing mod.weight (no bias), built once per module: the same cublasLt GEMM as LayerNormLinear's internal one
    (GEMM equality row: te.Linear on the module's ln_out == the module's out, 0 differing on every site, boxes 1-3)."""
    key = id(mod)
    lin = _state["lin"].get(key)
    if lin is None:
        import transformer_engine.pytorch as te
        lin = te.Linear(mod.in_features, mod.out_features, bias=False, params_dtype=mod.weight.dtype, device=mod.weight.device)
        lin.weight = mod.weight
        _state["lin"][key] = lin
    return lin


def residual_ln_qkv(x, y, sf, mod):
    """_OPS['residual_ln_qkv'] — (x_new, qkv): x_new = the residual (bitwise 'residual'), qkv = te.Linear(mod.weight)(Z) with Z = TE's
    ln_fwd_general<8192,1,4,16> numerics mirrored on x_new (mode 2). Present in ops() only when TESTED."""
    ext = load_ext()
    x_new, z = ext.residual_ln_te(x, y, inv_b_of(sf), mod.layer_norm_weight, mod.layer_norm_bias, float(mod.eps), 2)
    return x_new, _te_linear_for(mod)(z)


LN_SERVES_HIDDEN = (513, 8192)                             # residual_ln_te mirrors TE's general registrations lower_bound(cols) picks for 512 < N <= 8192: <1024|2048,4,1,16> and <8192,1,4,16>


def _hidden_of(model):
    cfg = getattr(model, "config", None)
    h = getattr(cfg, "hidden_size", None)
    if h is None and model is not None:
        try:
            h = int(model.esmc.embed.weight.shape[-1])
        except Exception:  # noqa: BLE001
            h = None
    return h


def ops(model=None) -> dict:
    """The seam entries, TESTED ONLY (an untested kernel never appears): ``residual`` at any hidden size (elementwise); ``residual_ln_qkv``
    only where the LN kernel mirrors the TE config of the model's width (LN_SERVES_HIDDEN: 512 < N <= 8192) — outside it TE runs another registration and the seam
    is left to the stock LayerNormLinear (named on the fused kit's line)."""
    load_ext()
    d = {}
    if TESTED["residual"]:
        d["residual"] = residual
    h = _hidden_of(model)
    if TESTED["residual_ln_qkv"] and (h is None or LN_SERVES_HIDDEN[0] <= int(h) <= LN_SERVES_HIDDEN[1]):
        d["residual_ln_qkv"] = residual_ln_qkv
    return d
