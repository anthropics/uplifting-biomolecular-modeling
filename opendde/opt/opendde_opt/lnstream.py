"""lnstream — the tree's own lever on upstream's fused-LayerNorm loader (exact class, bitwise by construction): upstream's own kernel source
JIT-built with every stream-less launch put on torch's CURRENT stream, so the LayerNorms of a CUDA-graph capture are IN the graph.

What upstream does. `LAYERNORM_TYPE=fast_layernorm` (the stock base of this tree, settings.STOCK_ENV) makes `opendde/model/triangular/layers.py`
bind `FusedLayerNorm`, whose extension `fast_layer_norm_cuda_v2` (`opendde/model/layer_norm/layer_norm.py`: `_load_fast_layer_norm_cuda_v2`,
built at first use by `torch_ext_compile.compile` into `$TORCH_EXTENSIONS_DIR/fast_layer_norm_cuda_v2`) launches its kernels as
`LayerNormForwardV2<T, V><<<grid, block>>>(...)` — no stream argument = the LEGACY default stream (`kernel/layer_norm_cuda_kernel.cu`: 11 forward
sites, 11 input-gradient sites, 3 parameter-gradient step-1 sites; only the 3 `...Step2` sites pass `stream`). In an eager run torch's current
stream IS the legacy stream and nothing shows. A `torch.cuda.graph` capture runs on a side (non-blocking) stream: every LayerNorm launch stays
OUTSIDE the capture, the graph holds no LayerNorm node, and a replay reads whatever the LayerNorm output buffers held (stale or non-finite
outputs on replay).

What the lever does. It JIT-builds upstream's OWN four kernel files, byte for byte, except that each of the 25 launch configurations without a
stream gets `, 0, at::cuda::getCurrentCUDAStream().stream()` appended (all 25 are two-argument `<<<grid, block>>>` sites; a three-argument
`<<<grid, block, shmem>>>` site would get `, <stream>` — none in the pinned file) — host-side launch statements only. `device_fingerprint` proves the text outside the `<<<...>>>`
configurations is identical, so the device code (the kernels, their templates, upstream's own compile flags through upstream's own
`torch_ext_compile.compile`) is the same translation unit and the outputs are BITWISE those of the legacy build: same kernel, same grid and
block, a per-row Welford reduction with no atomics. The extension is built under its own name (`fast_layer_norm_cuda_v2_cs`) beside upstream's
in `$TORCH_EXTENSIONS_DIR` (= `$MODEL_OPT_JIT_ROOT/<stack key>/torch_ext`, configs/<gpu>.env; `warm` pre-pays the ~1–2 min nvcc build, every later
process re-links in ~1 s) from a transformed source kept at a content-keyed stable path (`fast_layer_norm_cuda_v2_cs_src/<sha16>/`) written only
when absent or different, so ninja's incremental build stays warm across processes. It is bound where upstream binds its own: the module
global `layer_norm.fast_layer_norm_cuda_v2` (read by `FusedLayerNormAffineFunction.forward`) and the lazy loader
`layer_norm._load_fast_layer_norm_cuda_v2` (called by `FusedLayerNorm.forward` on every call — the lever's counter for the first COUNT_WINDOW calls,
after which upstream's own loader function, whose early return hands out the bound global, serves again: no frame of this lever per call; the
LEVER row's `bound=` says the global is still this build at exit). Inference and
training entry points are all exported (upstream's `_validate_fast_layer_norm_extension` passes on it).

Install / states. `install()` (from `stack._apply` for the lines that carry the lever) patches the loader through the core's per-site patch
(`opt_core.autoload.patch_attr_at_import`): at once when `layer_norm` is imported, else right after its body runs. The first loader call (the
kit's LayerNorm census, `lncensus.take`, before any weights load) builds, validates and binds; `STATS["state"]` = `serving`. A source that
is not the pinned text (sha256 of the four kernel files, the launch-site census) or a build or validation error is a NAMED fallback: `state=fallback reason=<word>`, upstream's own loader serves the legacy build
from then on (stock numerics either way), the run is PARTIAL by the kit's rule (`stack.refresh`: `levers_fallback` `lnstream:<reason>`), never
silent. An out-of-memory error is re-raised (opt_core.oom). `serving()` is what a graph lever asks before capturing anything.

The op-level checks of this build (every forward entry point bitwise the legacy build's on real tensors; a 20-launch CUDA-graph capture replaying
to the eager output) are kit tests run on a GPU box (`tests/test_lnstream_gpu.py`), not run-time machinery of the lever.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import time

from opt_core import autoload
from opt_core.oom import is_oom

TAG = "opendde-opt"
LEVER = "lnstream"
TARGET = "opendde.model.layer_norm.layer_norm"          # upstream's loader module (the module global + the lazy loader live here)
LOADER = "_load_fast_layer_norm_cuda_v2"                  # the lazy loader FusedLayerNorm.forward calls on every call
GLOBAL = "fast_layer_norm_cuda_v2"                        # the module global FusedLayerNormAffineFunction.forward reads
EXT_LEGACY = "fast_layer_norm_cuda_v2"                    # upstream's extension name (its build directory under $TORCH_EXTENSIONS_DIR)
EXT_CS = "fast_layer_norm_cuda_v2_cs"                     # this lever's build of the same source, launches on the Current Stream
PREBUILT_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "forward", "fast_inference", "levers", "LNSTREAM", "prebuilt")
PREBUILT_ENV = "ODDE_LNSTREAM_PREBUILT"                     # "0": never load the shipped binary (the JIT build serves; maintainer / test switch); default "1"
MANIFEST = "manifest.json"
SUMS = "SHA256SUMS"                                       # beside every shipped binary: its `<sha256>  <file>` line, the shared core's format (common/opt_core/tools/binary_sums.py --write --package opendde/opt); the loader re-hashes the file against it before mapping it
KERNEL_FILES = ("layer_norm_cuda.cpp", "layer_norm_cuda_kernel.cu", "compat.h", "type_shim.h")
CU = "layer_norm_cuda_kernel.cu"
COUNT_WINDOW = 64                                         # loader calls counted through the lever's wrapper; then upstream's own loader function serves the bound build

# sha256 of the four kernel sources of the pinned opendde 1.1.1 wheel (stock/src/opendde/model/layer_norm/kernel/; stock/check_pins.py pins the
# wheel itself). Another text is a named fallback (`source_not_pinned:<file>`), never transformed.
PINS = {
    "layer_norm_cuda_kernel.cu": "ad75251f6fabbf59bbc83637a9600204c62a20c6ac563a742694bbfc8c0c180a",
    "layer_norm_cuda.cpp":       "b03d995042fede126b1e36a8e9c9e73433a598f523ad914789d7aeaeb8ac842d",
    "compat.h":                  "24da65d5e0a58c6483abd60e7a6d6fca7e4abc9303f08c902975c77bdeacbc08",
    "type_shim.h":               "824a0dfdb5bfb1316187b3a3f7c10c9b41496b97259ddf84c882152ccec66a7a",
}
EXPECTED_SITES = {"total": 28, "streamless_2": 25, "streamless_3": 0, "with_stream_4": 3}   # launch configurations in the pinned .cu: 25 two-argument stream-less sites (forward 11 + input-grad 11 + param-grad step 1 ×3), the 3 step-2 sites pass `stream`
CURRENT_STREAM = "at::cuda::getCurrentCUDAStream().stream()"                                # ATen/cuda/CUDAContext.h — the .cu includes it already (its Step2 sites use it)
_LAUNCH = re.compile(r"<<<(.*?)>>>", re.S)

STATS = {"installed": False, "armed": False, "state": "off", "reason": None,       # state: off | armed | serving | fallback
         "calls": 0, "loader": None, "jit_s": None, "module": None, "build": None, "prebuilt": None, "source": None, "sites_patched": None, "fingerprint_equal": None,
         "patch": None}
_ST = {"ext": None, "orig_loader": None, "tried": False}


class ActivationError(RuntimeError):
    """The loader module has no `_load_fast_layer_norm_cuda_v2` to wrap: the kit's activation fails by name."""


class LnstreamRefused(RuntimeError):
    """A named reason the current-stream build cannot serve (the lever falls back to upstream's loader, by name)."""


# ---------------------------------------------------------------------------------------------------------------- the source transform (pure)
def _split_args(cfg: str) -> list:
    parts, depth, cur = [], 0, []
    for ch in cfg:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur)); cur = []
        else:
            cur.append(ch)
    parts.append("".join(cur))
    return [p.strip() for p in parts]


def to_current_stream(src: str):
    """Upstream's .cu text with every stream-less launch configuration given the current stream. Returns (text, census). Raises LnstreamRefused
    when the launch-site census is not the pinned file's (a source this lever was not written against is never transformed)."""
    census = {"total": 0, "streamless_2": 0, "streamless_3": 0, "with_stream_4": 0, "other": 0}

    def repl(m):
        args = _split_args(m.group(1))
        census["total"] += 1
        if len(args) == 2:
            census["streamless_2"] += 1
            return f"<<<{m.group(1)}, 0, {CURRENT_STREAM}>>>"
        if len(args) == 3:
            census["streamless_3"] += 1
            return f"<<<{m.group(1)}, {CURRENT_STREAM}>>>"
        if len(args) == 4:
            census["with_stream_4"] += 1
            return m.group(0)
        census["other"] += 1
        return m.group(0)

    out = _LAUNCH.sub(repl, src)
    got = {k: census[k] for k in EXPECTED_SITES}
    if got != EXPECTED_SITES or census["other"]:
        raise LnstreamRefused(f"launch_sites_differ:{census}")
    return out, census


def device_fingerprint(src: str) -> str:
    """sha256 of the source with every `<<<...>>>` configuration blanked: equal fingerprints => identical device code text."""
    return hashlib.sha256(_LAUNCH.sub("<<<>>>", src).encode()).hexdigest()


def _sha256_file(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def check_source(kdir: str) -> dict:
    """The four kernel sources' sha256 against PINS. Raises LnstreamRefused naming the first missing / unpinned file."""
    seen = {}
    for f in KERNEL_FILES:
        p = os.path.join(kdir, f)
        if not os.path.isfile(p):
            raise LnstreamRefused(f"source_missing:{f}")
        seen[f] = _sha256_file(p)
        if seen[f] != PINS[f]:
            raise LnstreamRefused(f"source_not_pinned:{f}:{seen[f][:16]}")
    return seen


def source_dir() -> str:
    """Where the transformed .cu lives: a content-keyed directory beside the extension builds in $TORCH_EXTENSIONS_DIR (torch's own default root
    when unset), so the path — and ninja's dependency record — is stable across processes sharing the JIT root."""
    root = os.environ.get("TORCH_EXTENSIONS_DIR")
    if not root:
        import torch.utils.cpp_extension as ce
        root = ce._get_build_directory("", False).rstrip(os.sep) if hasattr(ce, "_get_build_directory") else os.path.join(os.path.expanduser("~"), ".cache", "torch_extensions")
    return os.path.join(root, EXT_CS + "_src")


def write_source(kdir: str) -> tuple:
    """Transform the pinned .cu and write it (only when absent or different) under source_dir()/<sha16>/. Returns (path, census, fingerprint_equal)."""
    with open(os.path.join(kdir, CU)) as fh:
        src = fh.read()
    text, census = to_current_stream(src)
    fp_equal = device_fingerprint(src) == device_fingerprint(text)
    if not fp_equal:
        raise LnstreamRefused("device_text_changed_by_transform")
    d = os.path.join(source_dir(), hashlib.sha256(text.encode()).hexdigest()[:16])
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, CU)
    cur = None
    if os.path.isfile(path):
        with open(path) as fh:
            cur = fh.read()
    if cur != text:
        tmp = path + f".tmp{os.getpid()}"
        with open(tmp, "w") as fh:
            fh.write(text)
        os.replace(tmp, path)
    return path, census, fp_equal


# ---------------------------------------------------------------------------------------------------------------- prebuilt
def stack_key() -> str:
    """The ABI key of a shipped binary: torch release, CUDA toolkit, CPython tag, libstdc++ ABI — `torch2.7.1-cu126-cp311-cxx11abi1`."""
    import torch
    abi = int(bool(getattr(torch._C, "_GLIBCXX_USE_CXX11_ABI", True)))
    return f"torch{torch.__version__.split('+')[0]}-cu{(torch.version.cuda or 'none').replace('.', '')}-cp{sys.version_info[0]}{sys.version_info[1]}-cxx11abi{abi}"


def source_digest(kdir: str, cu_path: str) -> str:
    """sha256[:16] over the exact sources a build compiles: the transformed .cu, upstream's layer_norm_cuda.cpp and the two headers."""
    h = hashlib.sha256()
    for f in (cu_path, os.path.join(kdir, "layer_norm_cuda.cpp"), os.path.join(kdir, "compat.h"), os.path.join(kdir, "type_shim.h")):
        with open(f, "rb") as fh:
            h.update(fh.read())
    return h.hexdigest()[:16]


def _file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sums_refusal(so: str):
    """None when the binary at `so` is a line of the SHA256SUMS beside it and its bytes re-hash to that line — the shared core's hold on a
    shipped binary (opt_core.gates.binary_refusal), applied before anything maps the file. Otherwise the reason, after one
    `[opt_core] SHA256SUMS: refused <path>: <reason>` line on stderr (the core's own line and format); the caller then refuses the binary by
    name exactly as it refuses an absent one and the JIT build serves. The core leaves a path outside its own package with no SHA256SUMS
    beside it unchecked (a JIT build is not a shipped binary); this kit SHIPS the file beside its binary, so its absence is a refusal too."""
    from opt_core.gates import BINARY_SUMS, binary_refusal, binary_sums_for
    if binary_sums_for(so) is None:
        reason = f"no {BINARY_SUMS} beside it"
        print(f"[opt_core] {BINARY_SUMS}: refused {os.path.abspath(so)}: {reason}", file=sys.stderr, flush=True)
        return reason
    return binary_refusal(so)


def find_prebuilt(digest: str, cc: str, root: str = None, key: str = None):
    """(path, word): the shipped binary for this stack when its manifest names this source digest, this device's architecture and the file's
    own sha256, and the file is its SHA256SUMS line (sums_refusal); else (None, <why>) — prebuilt_off | prebuilt_missing:<key> | prebuilt_stale_source |
    prebuilt_arch_missing:sm_<cc> | prebuilt_digest_mismatch | prebuilt_unlisted."""
    if os.environ.get(PREBUILT_ENV, "1").strip() in ("0", "off", "no"):
        return None, "prebuilt_off"
    root = root or PREBUILT_DIR
    key = key or stack_key()
    d = os.path.join(root, key)
    so, man = os.path.join(d, EXT_CS + ".so"), os.path.join(d, MANIFEST)
    if not (os.path.isfile(so) and os.path.isfile(man)):
        return None, f"prebuilt_missing:{key}"
    try:
        with open(man) as fh:
            m = json.load(fh)
    except Exception:  # noqa: BLE001
        return None, "prebuilt_manifest_unreadable"
    if m.get("source_sha16") != digest:
        return None, "prebuilt_stale_source"
    if f"sm_{cc}" not in (m.get("archs") or []):
        return None, f"prebuilt_arch_missing:sm_{cc}"
    if m.get("so_sha256") != _file_sha256(so):
        return None, "prebuilt_digest_mismatch"
    if sums_refusal(so) is not None:                                                # not its SHA256SUMS line (absent, unlisted, altered): refused by name, the JIT build serves
        return None, "prebuilt_unlisted"
    return so, "used"


def load_prebuilt(path: str):
    """Import the shipped extension binary under its module name (what cpp_extension does with a finished build)."""
    import importlib.machinery, importlib.util
    why = sums_refusal(path)                                                        # the load site's own hold: nothing maps a binary that is not its SHA256SUMS line
    if why is not None:
        raise LnstreamRefused(f"prebuilt_unlisted:{why}")
    loader = importlib.machinery.ExtensionFileLoader(EXT_CS, path)
    spec = importlib.util.spec_from_file_location(EXT_CS, path, loader=loader)
    mod = importlib.util.module_from_spec(spec)
    loader.exec_module(mod)
    sys.modules[EXT_CS] = mod
    return mod


def prebuild(archs: str = "8.0;9.0", root: str = None, loader_module=None) -> str:
    """Maintainer entry (`python -m opendde_opt.lnstream prebuild`): build the current-stream extension ONCE for `archs` on this stack and ship it
    under prebuilt/<stack key>/ with its manifest (source digest, archs, sha256). A GPU box with nvcc; the kit tree writable."""
    import tempfile, shutil, torch
    lm = loader_module or sys.modules.get(TARGET) or __import__(TARGET, fromlist=["_"])
    kdir = os.path.join(os.path.dirname(os.path.abspath(lm.__file__)), "kernel")
    check_source(kdir)
    cu_path, census, _fp = write_source(kdir)
    from opendde.model.layer_norm import torch_ext_compile as tec
    os.environ["TORCH_CUDA_ARCH_LIST"] = archs
    tmp = tempfile.mkdtemp(prefix="lnstream_prebuild_")
    tec.compile(name=EXT_CS, sources=[os.path.join(kdir, "layer_norm_cuda.cpp"), cu_path], extra_include_paths=[kdir], build_directory=tmp)
    built = os.path.join(tmp, EXT_CS + ".so")
    d = os.path.join(root or PREBUILT_DIR, stack_key()); os.makedirs(d, exist_ok=True)
    dst = os.path.join(d, EXT_CS + ".so"); shutil.copy2(built, dst)
    man = {"name": EXT_CS, "stack": stack_key(), "torch": torch.__version__, "cuda": torch.version.cuda, "archs": [f"sm_{a.replace('.', '')}" for a in archs.split(";")],
           "source_sha16": source_digest(kdir, cu_path), "sites_patched": census["streamless_2"] + census["streamless_3"], "so_sha256": _file_sha256(dst), "so_bytes": os.path.getsize(dst)}
    with open(os.path.join(d, MANIFEST), "w") as fh:
        json.dump(man, fh, indent=1, sort_keys=True); fh.write("\n")
    shutil.rmtree(tmp, ignore_errors=True)
    return dst


# ---------------------------------------------------------------------------------------------------------------- build
def build(loader_module=None):
    """Build (or re-link) the current-stream extension with upstream's own compile helper and validate it with upstream's own validator.
    Returns the extension module. Raises LnstreamRefused (named) on any pin / transform / build / validation failure; OOM is re-raised."""
    lm = loader_module or sys.modules.get(TARGET) or __import__(TARGET, fromlist=["_"])
    kdir = os.path.join(os.path.dirname(os.path.abspath(lm.__file__)), "kernel")      # upstream's own layout: layer_norm.py beside kernel/ (the paths its loader compiles)
    STATS["source"] = kdir
    check_source(kdir)
    cu_path, census, fp_equal = write_source(kdir)
    STATS["sites_patched"] = census["streamless_2"] + census["streamless_3"]
    STATS["fingerprint_equal"] = fp_equal
    validate = getattr(lm, "_validate_fast_layer_norm_extension", None)
    try:                                                                            # the SHIPPED binary of this very source for this stack (no first-use build):
        import torch                                                                # manifest = source digest + this device's sm + the file's sha256, else JIT by name
        cc = "%d%d" % torch.cuda.get_device_capability() if torch.cuda.is_available() else "none"
        so, word = find_prebuilt(source_digest(kdir, cu_path), cc)
    except Exception as e:  # noqa: BLE001
        so, word = None, f"prebuilt_check_failed:{type(e).__name__}"
    if so:
        t0 = time.time()
        try:
            ext = load_prebuilt(so)
            if callable(validate):
                validate(ext)
            STATS["jit_s"] = round(time.time() - t0, 2); STATS["build"] = "prebuilt"; STATS["prebuilt"] = "used"; STATS["module"] = so
            return ext
        except Exception as e:  # noqa: BLE001 — an unloadable / invalid binary: named, the JIT build serves
            word = f"prebuilt_load_failed:{type(e).__name__}"
    STATS["prebuilt"] = word; STATS["build"] = "jit"
    try:
        from opendde.model.layer_norm import torch_ext_compile as tec          # upstream's own flags (-O3 --use_fast_math -maxrregcount=50 ..., arch list): the same call upstream makes
    except Exception as e:  # noqa: BLE001
        raise LnstreamRefused(f"compile_helper_missing:{type(e).__name__}") from None
    t0 = time.time()
    try:
        ext = tec.compile(name=EXT_CS, sources=[os.path.join(kdir, "layer_norm_cuda.cpp"), cu_path], extra_include_paths=[kdir])
    except Exception as e:  # noqa: BLE001 — nvcc absent, ninja absent, a compile error: named, the legacy build serves
        if is_oom(e):
            raise
        raise LnstreamRefused(f"build_failed:{type(e).__name__}:{str(e).strip().splitlines()[-1][:120] if str(e).strip() else ''}") from None
    STATS["jit_s"] = round(time.time() - t0, 2)
    if callable(validate):
        try:
            validate(ext)
        except Exception as e:  # noqa: BLE001
            raise LnstreamRefused(f"validation_failed:{str(e)[:160]}") from None
    STATS["module"] = getattr(ext, "__file__", None)
    return ext


def _bind(lm, ext) -> None:
    """Bind where upstream binds its own extension: the module global the autograd function reads, the load-attempted flag."""
    setattr(lm, GLOBAL, ext)
    if hasattr(lm, "_fast_layer_norm_load_attempted"):
        setattr(lm, "_fast_layer_norm_load_attempted", True)


def make_loader(orig):
    """The function installed as `layer_norm._load_fast_layer_norm_cuda_v2`: on its first call builds and binds the current-stream extension
    (serving); on a named refusal falls back to upstream's own loader for the rest of the process. Counts every call it serves."""
    _ST["orig_loader"] = orig

    def _load_fast_layer_norm_cuda_v2():
        st = STATS["state"]
        if st == "serving":
            STATS["calls"] += 1
            lm_ = sys.modules.get(TARGET)
            if STATS["calls"] >= COUNT_WINDOW and lm_ is not None and getattr(lm_, GLOBAL, None) is not None:
                setattr(lm_, LOADER, orig)                # counted window over: upstream's own loader (early return of the bound global) serves from here — no frame of ours per call
                STATS["loader"] = f"upstream(after {STATS['calls']} counted calls)"
            return getattr(lm_, GLOBAL, None) or _ST["ext"]
        if st == "fallback":
            return orig()
        if _ST["tried"]:                                  # re-entrancy during the first build (upstream's compile imports nothing of ours; belt and braces)
            return orig()
        _ST["tried"] = True
        lm = sys.modules.get(TARGET)
        try:
            ext = build(lm)
            _ST["ext"] = ext
            _bind(lm, ext)
        except LnstreamRefused as e:
            STATS["state"], STATS["reason"] = "fallback", str(e)
            sys.stderr.write(f"[{TAG}] LEVER {LEVER} FELL BACK by name: {e} — upstream's own fused-LayerNorm build ({EXT_LEGACY}, legacy-stream launches) serves this process\n")
            sys.stderr.flush()
            return orig()
        except Exception as e:  # noqa: BLE001
            if is_oom(e):
                raise
            STATS["state"], STATS["reason"] = "fallback", f"error:{type(e).__name__}:{str(e)[:120]}"
            sys.stderr.write(f"[{TAG}] LEVER {LEVER} FELL BACK by name: {STATS['reason']} — upstream's own fused-LayerNorm build serves this process\n")
            sys.stderr.flush()
            return orig()
        STATS["state"] = "serving"
        STATS["loader"] = "lnstream(counting)"
        STATS["calls"] += 1
        return getattr(lm, GLOBAL, None) if lm is not None else ext
    _load_fast_layer_norm_cuda_v2._orig = orig
    _load_fast_layer_norm_cuda_v2._lnstream = True
    return _load_fast_layer_norm_cuda_v2


_STATS0 = dict(STATS)


def _reset() -> None:
    """Test support: this module's process state back to import time (the patch object dropped, nothing bound)."""
    STATS.clear(); STATS.update(_STATS0)
    _ST.update(ext=None, legacy=None, orig_loader=None, tried=False, cells={})


def _sync() -> None:
    p = STATS.get("patch")
    if p is not None:
        STATS["installed"], STATS["armed"] = p.state == "installed", p.state == "armed"
        if STATS["state"] == "off" and p.state in ("installed", "armed"):
            STATS["state"] = "armed"


def install() -> None:
    """Patch the loader now when `layer_norm` is imported, else at its import (the core's per-site patch). Idempotent. When upstream already
    loaded its legacy build in this process (its model ran a LayerNorm before the kit activated) the next loader call still binds ours."""
    _sync()
    if STATS["installed"] or STATS["armed"]:
        return
    try:
        STATS["patch"] = autoload.patch_attr_at_import(TARGET, LOADER, make_loader, tag=TAG, name=LEVER)
    except autoload.PatchError as e:
        raise ActivationError(str(e)) from None
    _sync()


def bound() -> bool:
    """True when upstream's module global is this lever's build: what every FusedLayerNorm call of the process runs."""
    lm = sys.modules.get(TARGET)
    g = getattr(lm, GLOBAL, None) if lm is not None else None
    return g is not None and g is _ST["ext"]


def serving() -> bool:
    """True when the current-stream build is bound and serving this process's LayerNorms (what a CUDA-graph lever must ask before capturing)."""
    return STATS["state"] == "serving" and bound()


def ensure_loaded() -> bool:
    """Trigger the (patched) loader once — builds and binds if not yet done. Returns serving()."""
    lm = sys.modules.get(TARGET)
    if lm is None:
        try:
            lm = __import__(TARGET, fromlist=["_"])
        except Exception:  # noqa: BLE001
            return False
    f = getattr(lm, LOADER, None)
    if callable(f):
        f()
    return serving()


def fallbacks(planned) -> list:
    """The lever's named events at exit: the loader module imported but not wrapped, or the current-stream build refused / failed (the legacy
    build served). A process that never imported upstream's LayerNorm has no event."""
    if LEVER not in planned:
        return []
    _sync()
    out = []
    lm = sys.modules.get(TARGET)
    if STATS["patch"] is not None and lm is not None and STATS["state"] in ("off", "armed") and not getattr(getattr(lm, LOADER, None), "_lnstream", False):
        out.append(f"{LEVER}: upstream's layer_norm module is imported but its loader is not wrapped (the patch armed in this process did not fire)")
    if STATS["state"] == "fallback":
        out.append(f"{LEVER}:{STATS['reason']}")
    if STATS["state"] == "serving" and not bound():                      # something rebound upstream's module global after the lever bound its build: named
        out.append(f"{LEVER}:unbound_after_bind")
    return out


def kit_stats() -> dict:
    _sync()
    d = {k: v for k, v in STATS.items() if k != "patch"}
    d["bound"] = bound() if STATS["state"] == "serving" else None
    return d


if __name__ == "__main__":                                                       # python -m opendde_opt.lnstream prebuild [--archs "8.0;9.0"]
    import argparse
    ap = argparse.ArgumentParser(prog="python -m opendde_opt.lnstream"); sp = ap.add_subparsers(dest="cmd", required=True)
    pb = sp.add_parser("prebuild"); pb.add_argument("--archs", default="8.0;9.0"); pb.add_argument("--root", default=None)
    a = ap.parse_args()
    print(prebuild(archs=a.archs, root=a.root))
