"""enformer_deepmind_opt.ops — the kit's TensorFlow GPU ops (edm_ops.so, built by build.sh from the sources beside this file).

    from enformer_deepmind_opt.ops import load, check
    edm = load()                               # the op module: edm.edm_scale_shift_gelu(x, scale, shift), edm.edm_softmax_pool2(x, logits)
    info = check()                             # {so_path, sha256, ops, tensorflow, archs, ptx, build}; raises by name when something does not hold

`load()` loads edm_ops.so into the running TensorFlow once (tf.load_op_library) and returns the module, after re-hashing the file against its
line of the SHA256SUMS beside it (`sums_refusal`): a library that is not the shipped one by digest is refused by name exactly as an absent
one is. `check()` confirms the library is the one BUILD.json describes (object and source digests), that the running TensorFlow is the
version it was built against, and that every op and its GPU kernel registered. The ops (BUILD.json `ops`; README.md documents each one's
arithmetic) are float32, GPU only, computed with TensorFlow 2.17.1's own float32 arithmetic. `python .../ops/__init__.py` prints check()
as JSON.
"""
import hashlib
import json
import os
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
SO_PATH = os.path.join(HERE, "edm_ops.so")
BUILD_JSON = os.path.join(HERE, "BUILD.json")
SUMS = "SHA256SUMS"                        # beside edm_ops.so: its `<sha256>  edm_ops.so` line (sha256sum format); build.sh refreshes it with the library
SUMS_PATH = os.path.join(HERE, SUMS)
TAG = "[enformer-deepmind-opt] ops:"

_lib = None
_lock = threading.Lock()
_PACKAGE_DIR = os.path.dirname(HERE)
_VERDICTS = {}                             # library path -> None (listed, digest equal) | the refusal reason; hashed once per process
HELD = []                                  # [(path relative to the package, sha256[:16])] of every library that passed, in load order

__all__ = ["load", "check", "build_info", "serves", "sums_refusal", "SO_PATH", "BUILD_JSON", "SUMS_PATH"]


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sums_line(sums_path, name):
    """The digest the SHA256SUMS at ``sums_path`` lists for the file ``name`` (`<sha256>  <name>` lines; `#` comments), else None."""
    want = None
    with open(sums_path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.split(None, 1) if ln.strip() and not ln.startswith("#") else []
            if len(parts) == 2 and parts[1].strip() == name:
                want = parts[0].lower()
    return want


def sums_refusal(path, data=None):
    """``None`` when the compiled library at ``path`` is a line of the SHA256SUMS in its own directory and its bytes re-hash to that line
    (``data``: the bytes the caller already read, hashed in place of the file). Otherwise the reason, one clause — no SHA256SUMS, not
    listed, unreadable, or the digest differing — after one ``[enformer-deepmind-opt] SHA256SUMS: refused <path>: <reason>`` line on
    stderr; load() then refuses the library by name exactly as it refuses an absent one. One verdict per path per process."""
    p = os.path.abspath(path)
    if p in _VERDICTS:
        return _VERDICTS[p]
    inside = p.startswith(_PACKAGE_DIR + os.sep)
    rel = os.path.relpath(p, _PACKAGE_DIR).replace(os.sep, "/") if inside else p
    sums_path = os.path.join(os.path.dirname(p), SUMS)
    srel = os.path.relpath(sums_path, _PACKAGE_DIR).replace(os.sep, "/") if inside else sums_path
    reason = digest = None
    if not os.path.isfile(sums_path):
        reason = f"no {SUMS} in its directory lists it"
    else:
        try:
            want = _sums_line(sums_path, os.path.basename(p))
        except (OSError, ValueError):
            want = None
        if want is None:
            reason = f"not listed in {srel}"
        else:
            try:
                digest = hashlib.sha256(data).hexdigest() if data is not None else _sha256(p)
            except OSError as e:
                reason = f"unreadable ({type(e).__name__})"
            else:
                if digest != want:
                    reason = f"sha256 {digest[:16]} != {srel} {want[:16]}"
    _VERDICTS[p] = reason
    if reason is None:
        HELD.append((rel, digest[:16]))
    else:
        import sys  # noqa: PLC0415
        print(f"[enformer-deepmind-opt] {SUMS}: refused {rel}: {reason}", file=sys.stderr, flush=True)
    return reason


def build_info():
    """BUILD.json as a dict (FileNotFoundError naming build.sh when absent)."""
    if not os.path.isfile(BUILD_JSON):
        raise FileNotFoundError(f"{TAG} {BUILD_JSON} is missing — build the ops: bash {os.path.join(HERE, 'build.sh')}")
    with open(BUILD_JSON) as f:
        return json.load(f)


def _preload_cuda_runtime():
    """Make libcudart.so.12 resident before edm_ops.so (which links it) is opened: from the nvidia-cuda-runtime-cu12 pip package TensorFlow
    itself uses when present, else by soname from the loader's search path. A failure here surfaces as tf.load_op_library's own error."""
    import ctypes
    import importlib.util
    mode = getattr(os, "RTLD_GLOBAL", 0) | getattr(os, "RTLD_NOW", 2)
    try:
        spec = importlib.util.find_spec("nvidia.cuda_runtime")
    except (ImportError, ValueError):
        spec = None
    for d in (list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []):
        cand = os.path.join(d, "lib", "libcudart.so.12")
        if os.path.isfile(cand):
            ctypes.CDLL(cand, mode=mode)
            return cand
    try:
        ctypes.CDLL("libcudart.so.12", mode=mode)
        return "libcudart.so.12"
    except OSError:
        return None


def load():
    """The loaded op module (tf.load_op_library on edm_ops.so, once per process). FileNotFoundError when the .so is absent, or is not the
    shipped library by its SHA256SUMS line (sums_refusal; nothing is loaded); RuntimeError when the running TensorFlow is not the version
    BUILD.json says the library was built against."""
    global _lib
    if _lib is not None:
        return _lib
    with _lock:
        if _lib is not None:
            return _lib
        import tensorflow as tf
        if not os.path.isfile(SO_PATH):
            raise FileNotFoundError(f"{TAG} {SO_PATH} is missing — build the ops: bash {os.path.join(HERE, 'build.sh')}")
        why = sums_refusal(SO_PATH)
        if why is not None:                                            # refused by name before TensorFlow maps it, as an absent library is
            raise FileNotFoundError(f"{TAG} {SO_PATH} is not the shipped library ({why}) — restore it, or rebuild the ops: "
                                    f"bash {os.path.join(HERE, 'build.sh')}")
        want = build_info().get("tensorflow")
        if want and tf.__version__ != want:
            raise RuntimeError(f"{TAG} edm_ops.so was built against tensorflow {want}; the running tensorflow is {tf.__version__} — "
                               f"rebuild the ops for it: bash {os.path.join(HERE, 'build.sh')}")
        _preload_cuda_runtime()
        _lib = tf.load_op_library(SO_PATH)
        return _lib


def check():
    """Confirm the ops as built: edm_ops.so present with the sha256 BUILD.json records, the sources beside it unchanged since the build, the
    running TensorFlow = the one built against, and every op of BUILD.json registered with a GPU kernel. Returns
    {so_path, sha256, ops, tensorflow, archs, ptx, build}; raises FileNotFoundError / RuntimeError naming what does not hold."""
    build = build_info()
    if not os.path.isfile(SO_PATH):
        raise FileNotFoundError(f"{TAG} {SO_PATH} is missing — build the ops: bash {os.path.join(HERE, 'build.sh')}")
    digest = _sha256(SO_PATH)
    want = (build.get("object") or {}).get("sha256")
    if digest != want:
        raise RuntimeError(f"{TAG} edm_ops.so sha256 {digest} differs from BUILD.json's {want} — the library is not the recorded build "
                           f"(restore the file, or rebuild: bash {os.path.join(HERE, 'build.sh')})")
    for name, recorded in (build.get("sources") or {}).items():
        path = os.path.join(HERE, name)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{TAG} source {name} recorded in BUILD.json is missing from {HERE}")
        if _sha256(path) != recorded:
            raise RuntimeError(f"{TAG} {name} differs from the source edm_ops.so was built from (BUILD.json) — rebuild: bash {os.path.join(HERE, 'build.sh')}")
    lib = load()
    import tensorflow as tf
    from tensorflow.python.framework import kernels as _kernels
    from tensorflow.python.framework import op_def_registry as _op_defs
    names = list(build.get("ops") or [])
    for name in names:
        if _op_defs.get(name) is None:
            raise RuntimeError(f"{TAG} op {name} did not register from {SO_PATH}")
        wrapper = "".join("_" + ch.lower() if ch.isupper() else ch for ch in name).lstrip("_")
        if not hasattr(lib, wrapper):
            raise RuntimeError(f"{TAG} op {name} registered but the loaded module has no wrapper {wrapper}()")
        devices = {k.device_type for k in _kernels.get_registered_kernels_for_op(name).kernel}
        if "GPU" not in devices:
            raise RuntimeError(f"{TAG} op {name} registered without its GPU kernel (kernels: {sorted(devices) or 'none'})")
    return {
        "so_path": SO_PATH,
        "sha256": digest,
        "ops": names,
        "tensorflow": build.get("tensorflow"),
        "archs": [list(a) for a in build.get("archs") or []],
        "ptx": [list(p) for p in build.get("ptx") or []],
        "build": build.get("build") or "+".join("sm_%d%d" % tuple(a) for a in build.get("archs") or []),
    }


if __name__ == "__main__":
    print(json.dumps(check(), indent=1))


def serves(sm):
    """The build serving a device of compute capability ``sm`` = (major, minor): its own native code, the same-major native code of a lower
    minor (binary compatible), or the library's PTX compiled by the driver for a newer device. RuntimeError when nothing in the library can run
    there. Runs check() first."""
    info = check()
    sm = (int(sm[0]), int(sm[1]))
    archs = [tuple(a) for a in info["archs"]]; ptx = [tuple(p) for p in info["ptx"]]
    if sm in archs:
        return f"sm_{sm[0]}{sm[1]}"
    same_major = [a for a in archs if a[0] == sm[0] and a[1] <= sm[1]]
    if same_major:
        a = max(same_major)
        return f"sm_{a[0]}{a[1]} (sm_{sm[0]}{sm[1]} runs the sm_{a[0]}{a[1]} build)"
    newer = [p for p in ptx if sm >= p]
    if newer:
        p = max(newer)
        return f"compute_{p[0]}{p[1]} PTX (compiled for sm_{sm[0]}{sm[1]} by the driver at load)"
    raise RuntimeError(f"{TAG} edm_ops.so has no build that runs on compute capability {sm[0]}.{sm[1]} (native {info['build']}, PTX {ptx})")
