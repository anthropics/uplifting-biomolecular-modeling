# PROVENANCE: carried from the esmfold2 kit, opt/forward/fast_inference/driver/ef2_nvjit.py at commit cab5aaaa56b6 (file sha256 29b69358101a),
# byte-for-byte except this header, three constants (PREFIX, SHIPPED_ROOT -> k/ef2_t16/, PREBUILD_MODULES -> ef2_t16_transition) and the SHA256SUMS
# check of the shipped cubin (shipped_cubin). The run-time
# route of this kit is the SHIPPED-cubin route only (shipped_cubin / Kernel / TensorMap / require_device); the NVRTC build route (compile_cubin,
# build_all) stays in the text unchanged and is exercised by the esmfold2 kit's build script, not by this kit (no header wheels in this image).
"""ef2_nvjit — the kit's ONE helper for its CUDA C++ / CuTe kernels (ef2_transition_cute: lever t16): load the
SHIPPED cubin through the CUDA driver API (cuda-bindings, part of the pinned stack) and launch it on torch's current stream (optionally as
thread-block clusters).  ONE named error class: every failure raises ``NvjitUnavailable`` naming the reason — the lever modules let it through,
ef2_server.configure fails, and the package refuses the mode by name before the first fold (never a silent fallback, never a compile at run time).

RELEASE = SHIPPED CUBINS ONLY.  ``driver/prebuilt/<arch>/<stem>.cubin`` + ``manifest.json`` (written by ``driver/prebuilt/build_prebuilt.py`` — the
ONLY compile route: NVRTC with the nvidia-cutlass / nvidia-cuda-cccl header wheels, run where those are installed).  At run time a kernel is served
when ALL hold, else NvjitUnavailable by name (the lever cannot run -> the mode refuses at install, never mid-fold):
  * the manifest holds an entry for the kernel whose ``source_key`` (sha256 of the source text + compile options, install-location independent, no
    toolchain needed) equals this tree's — a tree whose kernel source moved is not served by an old binary;
  * the file's sha256 equals the manifest's ``cubin_sha256``;
  * the manifest records the build its spec demands (ptxas statistics present; no C75xx advisory but C7517; spill-free — stack_bytes ==
    spill_stores == spill_loads == 0 — where the kernel overlaps an epilogue with in-flight wgmma groups, as ef2_transition_cute does);
  * after cuModuleLoadData the DEVICE reports CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES == 0 and CU_FUNC_ATTRIBUTE_NUM_REGS == the manifest's record;
  * the module's install canary on the loaded function passes (ef2_transition_cute.canary: back-to-back launches bytewise identical AND within
    tolerance of the fp32 statement).
A locally compiled cubin is not the audited binary: there is no run-time JIT.  Each load prints ONE ``[esmfold2-opt] nvrtc: …`` line on stderr.

    cub = shipped_cubin(src, "name.cu", macros={...}, tag="lever …")            # -> Cubin (data, sha256, manifest entry) or NvjitUnavailable by name
    k = Kernel(cub, "kernel_name", smem_bytes=...)                             # .regs / .lmem / .smem / .cubin_sha256 / .source for the LEVER line; refuses lmem != 0, regs != manifest
    k((grid_x,1,1), (384,1,1), tensor_map, tensor, 3, 1.5, ..., stream=None)   # args: TensorMap | torch.Tensor (device ptr) | int (i32) | float (f32) | ctypes scalar | None (null u64)
    python ef2_nvjit.py cubins                                                  # run.sh install (CPU): shipped cubins present, sha256 == manifest, source keys == this tree's sources
    python prebuilt/build_prebuilt.py build                                     # the build route (header wheels; no GPU); `check --write` on an H100 records device regs / load checks
"""
import ctypes
import hashlib
import os
import sys
import sysconfig
import tempfile
import time

__all__ = ["NvjitUnavailable", "include_dirs", "nvrtc_version", "header_versions", "shipped_dir", "shipped_manifest", "source_key", "shipped_cubin",
           "compile_cubin", "Cubin", "Kernel", "TensorMap", "require_device", "toolchain", "build_all", "check_shipped", "kernel_specs", "ptxas_stats", "require_spill_free",
           "elf_sections", "elf_sections_equal", "PREBUILD_MODULES", "records"]

PREFIX = "[opt_core] t16 loader:"
ARCH = "sm_90a"                                                        # the one architecture the kit's CuTe kernels are written for (wgmma / TMA / setmaxnreg: the `a` feature set of 9.0)
PREBUILD_MODULES = ("ef2_t16_transition",)
HERE = os.path.dirname(os.path.abspath(__file__))
SHIPPED_ROOT = os.path.join(HERE, "ef2_t16")                           # k/ef2_t16/<arch>/{<stem>.cubin, manifest.json} + PROVENANCE.md: the carried cubin (byte-for-byte the esmfold2 kit's driver/prebuilt/<arch>/)
MANIFEST = "manifest.json"
_STATE = dict(incs=None, nvrtc_version=None, header_versions=None)
_RECORDS = []                                                          # one dict per compile_cubin() outcome in this process (describe() material for the LEVER lines)


class NvjitUnavailable(RuntimeError):
    """An NVRTC-built kernel of the kit cannot be compiled, loaded or launched on this stack / device (the message names the reason:
    a missing header wheel, the compiler log, the CUDA driver error, the device class)."""


# ------------------------------------------------------------------------------------------------------------------------------ toolchain
def _nvidia_cu13_include():
    """<site-packages>/nvidia/cu13/include of the CUDA 13 wheel set (nvidia-cuda-cccl / nvidia-cuda-runtime / nvidia-cuda-nvrtc install under the `nvidia`
    namespace package; its __path__ names every directory it spans)."""
    cands = []
    try:
        import nvidia
        cands += [os.path.join(d, "cu13", "include") for d in list(getattr(nvidia, "__path__", []))]
    except ImportError:
        pass
    cands += [os.path.join(sysconfig.get_paths()[k], "nvidia", "cu13", "include") for k in ("platlib", "purelib")]
    for d in cands:
        if os.path.isdir(os.path.join(d, "cccl")):
            return d
    return cands[0] if cands else os.path.join(sysconfig.get_paths()["platlib"], "nvidia", "cu13", "include")


def include_dirs():
    """[cutlass include, cccl include, cuda include]; raises NvjitUnavailable naming the missing header wheel."""
    if _STATE["incs"] is not None:
        return _STATE["incs"]
    try:
        import cutlass_library
    except ImportError:
        raise NvjitUnavailable("ef2_nvjit: the nvidia-cutlass wheel (cutlass_library) is not installed: CuTe headers unavailable (environment/requirements.lock pins it)") from None
    d_cu13 = _nvidia_cu13_include()
    d_cutlass = os.path.join(os.path.dirname(cutlass_library.__file__), "source", "include")
    d_cccl = os.path.join(d_cu13, "cccl")
    d_cuda = d_cu13
    for d, probe, wheel in ((d_cutlass, "cute/tensor.hpp", "nvidia-cutlass"), (d_cccl, "cuda/std/cstdint", "nvidia-cuda-cccl"), (d_cuda, "cuda_bf16.h", "nvidia-cuda-runtime")):
        if not os.path.exists(os.path.join(d, probe)):
            raise NvjitUnavailable(f"ef2_nvjit: header {probe} not found under {d} (wheel {wheel}; environment/requirements.lock pins it)")
    _STATE["incs"] = [d_cutlass, d_cccl, d_cuda]
    return _STATE["incs"]


def header_versions():
    """{wheel: version} of the header wheels whose CONTENT enters a cubin (part of the cache key, so a header move re-keys every cubin)."""
    if _STATE["header_versions"] is None:
        from importlib import metadata
        out = {}
        for dist in ("nvidia-cutlass", "nvidia-cuda-cccl", "nvidia-cuda-runtime", "nvidia-cuda-nvrtc"):
            try:
                out[dist] = metadata.version(dist)
            except metadata.PackageNotFoundError:
                out[dist] = "absent"
        _STATE["header_versions"] = out
    return dict(_STATE["header_versions"])


def _drv():
    try:
        from cuda.bindings import driver as cu
    except ImportError as e:
        raise NvjitUnavailable(f"ef2_nvjit: cuda.bindings (the cuda-bindings wheel) is not importable: {e}") from None
    return cu


def _nvrtc():
    try:
        from cuda.bindings import nvrtc
    except ImportError as e:
        raise NvjitUnavailable(f"ef2_nvjit: cuda.bindings.nvrtc (the cuda-bindings / nvidia-cuda-nvrtc wheels) is not importable: {e}") from None
    return nvrtc


def _chk(res, what="call"):
    """cuda-python calls return (err, *vals); a failing status raises NvjitUnavailable naming it."""
    if isinstance(res, tuple):
        err, vals = res[0], res[1:]
    else:
        err, vals = res, ()
    mod = type(err).__module__ or ""
    name = type(err).__name__
    if name == "CUresult":
        cu = _drv()
        if err != cu.CUresult.CUDA_SUCCESS:
            _, ename = cu.cuGetErrorName(err)
            raise NvjitUnavailable(f"ef2_nvjit: CUDA driver error {ename.decode() if isinstance(ename, bytes) else ename} in {what}")
    elif name == "nvrtcResult":
        nvrtc = _nvrtc()
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            raise NvjitUnavailable(f"ef2_nvjit: NVRTC error {err} in {what}")
    elif "cuda" in mod:                                                # an unexpected status type from the bindings: never swallowed
        raise NvjitUnavailable(f"ef2_nvjit: unexpected status {err!r} in {what}")
    return vals[0] if len(vals) == 1 else vals


def nvrtc_version():
    if _STATE["nvrtc_version"] is None:
        _STATE["nvrtc_version"] = tuple(int(v) for v in _chk(_nvrtc().nvrtcVersion(), "nvrtcVersion"))
    return _STATE["nvrtc_version"]


def require_device(smem_bytes=0, who="ef2_nvjit"):
    """Raise NvjitUnavailable by name unless the current CUDA device is compute capability 9.0 (the kernels are sm_90a code) with at least
    ``smem_bytes`` of opt-in dynamic shared memory per block. Returns torch's device properties."""
    import torch
    if not torch.cuda.is_available():
        raise NvjitUnavailable(f"{who}: no CUDA device (the kernel is {ARCH} code)")
    dev = torch.cuda.get_device_properties(torch.cuda.current_device())
    if (dev.major, dev.minor) != (9, 0):
        raise NvjitUnavailable(f"{who}: {dev.name} is sm_{dev.major}{dev.minor}; the kernel is {ARCH} (compute capability 9.0: H100 / H200) only")
    optin = int(getattr(dev, "shared_memory_per_block_optin", 232448))
    if smem_bytes and smem_bytes > optin:
        raise NvjitUnavailable(f"{who}: needs {smem_bytes} B of dynamic shared memory per block, {dev.name} allows {optin} B")
    return dev


# ------------------------------------------------------------------------------------------------------------------------------ cubin cache
def shipped_dir(arch=ARCH, root=None):
    """The kit's shipped-cubin directory for ``arch``: driver/prebuilt/<arch>."""
    return os.path.join(root or SHIPPED_ROOT, arch)


def shipped_manifest(arch=ARCH, root=None):
    """driver/prebuilt/<arch>/manifest.json as a dict ({} when absent or unreadable): {"arch", "cubins": {<stem>: {file, cubin_sha256, bytes, source_key, …}}, …}."""
    p = os.path.join(shipped_dir(arch, root), MANIFEST)
    try:
        with open(p, encoding="utf-8") as f:
            import json
            m = json.load(f)
        return m if isinstance(m, dict) else {}
    except (OSError, ValueError):
        return {}


def toolchain():
    """(True, None) when the NVRTC build route is available in this interpreter (header wheels + NVRTC importable), else (False, <why>)."""
    try:
        include_dirs(); nvrtc_version()
        return True, None
    except NvjitUnavailable as e:
        return False, str(e)


PTXAS_VERBOSE = "--ptxas-options=-v"                                   # the STATISTICS TWIN's extra option: ptxas' per-entry `Used N registers` / `bytes stack frame, spill stores,
                                                                       # spill loads` in the log. The option is recorded in the cubin's .note.nv.tkinfo (and nowhere else), so the
                                                                       # shipping cubin is built WITHOUT it and its statistics come from a twin build that must be identical in every
                                                                       # other ELF section (elf_sections_equal) — the shipped object stays the audited object.
TKINFO_NOTE = ".note.nv.tkinfo"


def elf_sections(data):
    """{name: (type, sha256-of-content)} of an ELF64 little-endian object (a cubin) — a stdlib reader of the section header table."""
    import struct
    if data[:4] != b"\x7fELF" or data[4] != 2 or data[5] != 1:
        raise NvjitUnavailable("ef2_nvjit: not an ELF64 little-endian object")
    e_shoff, = struct.unpack_from("<Q", data, 0x28); e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", data, 0x3A)
    hdrs = [struct.unpack_from("<IIQQQQIIQQ", data, e_shoff + i * e_shentsize) for i in range(e_shnum)]
    st = hdrs[e_shstrndx]; strtab = data[st[4]:st[4] + st[5]]
    out = {}
    for (name_off, typ, flags, addr, off, size, link, info, align, entsize) in hdrs:
        name = strtab[name_off:strtab.index(b"\0", name_off)].decode()
        body = b"" if typ == 8 else data[off:off + size]                # SHT_NOBITS has no file content
        out[name] = (typ, hashlib.sha256(body).hexdigest())
    return out


def elf_sections_equal(a, b, ignore=(TKINFO_NOTE,)):
    """-> (equal, differing section names) comparing two cubins section by section (content sha256), ignoring the toolkit-info note."""
    sa, sb = elf_sections(a), elf_sections(b)
    names = sorted((set(sa) | set(sb)) - set(ignore))
    diff = [n for n in names if sa.get(n) != sb.get(n)]
    return (not diff), diff


def ptxas_stats(log):
    """Parse ptxas' verbose lines of a build log -> dict(regs, stack_bytes, spill_stores, spill_loads, c7512 = the `wgmma serialized` advisory is
    present, lines). Numbers absent from the log read None (a build without the verbose option)."""
    import re
    st = dict(regs=None, stack_bytes=None, spill_stores=None, spill_loads=None, c7512=("C7512" in (log or "")), lines=[], diagnostics=[])
    for ln in (log or "").splitlines():
        if "ptxas" in ln or "bytes stack frame" in ln:
            st["lines"].append(ln.strip())
        m = re.search(r"\((C7\d\d\d)\)\s*(.*)$", ln)                    # ptxas scheduling advisories: (C7517) warpgroup.wait injected … — recorded as the build's FINGERPRINT;
        if m:                                                          # any other C75xx (C7510 / C7512: wgmma serialized …) fails the build (require_spill_free)
            st["diagnostics"].append(dict(code=m.group(1), text=m.group(2).strip()[:240]))
        m = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores, (\d+) bytes spill loads", ln)
        if m:
            v = [int(t) for t in m.groups()]
            st["stack_bytes"] = max(st["stack_bytes"] or 0, v[0]); st["spill_stores"] = max(st["spill_stores"] or 0, v[1]); st["spill_loads"] = max(st["spill_loads"] or 0, v[2])
        m = re.search(r"Used (\d+) registers", ln)
        if m:
            st["regs"] = max(st["regs"] or 0, int(m.group(1)))
    return st


def require_build_policy(st, tag, spill_free=True):
    """The build-log gate of a kernel spec: ptxas statistics must be PRESENT (recorded in the manifest either way) and no C75xx advisory other
    than C7517 may appear; with ``spill_free`` (kernels that overlap an epilogue with in-flight wgmma groups: ef2_transition_cute) additionally
    stack == spill stores == spill loads == 0 (require_spill_free). Raises NvjitUnavailable by name; returns st."""
    if spill_free:
        return require_spill_free(st, tag)
    missing = [k for k in ("stack_bytes", "spill_stores", "spill_loads") if st.get(k) is None]
    if missing:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: the build log carries no ptxas statistics ({', '.join(missing)}) — refusing by name")
    bad = [f"{d['code']}({d['text'][:60]})" for d in st.get("diagnostics", []) if d["code"] != "C7517"]
    if bad:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: ptxas advisories {', '.join(bad)} in the build log (only C7517 is accepted) — refusing by name; nothing written")
    return st


def require_spill_free(st, tag):
    """Spill-free is an ENFORCED BUILD PROPERTY of the shipped object (so that the audited binary is the binary that runs): a build whose ptxas log
    shows a stack frame, spill stores / loads, or a C75xx advisory other than C7517 is REFUSED by name, nothing written. (The rule's origin: a
    parked sibling CuTe kernel of this line released a shared-memory slot with an mbarrier arrive that had no scoreboard dependency on the slot's
    pending ldmatrix reads — the producer's TMA refill tore the read; register spills only widened that window. The shipped transition kernel
    was audited at SASS level for that class: no LDS / LDSM feeds a release; fences, BAR.SYNC and the wgmma waits precede every release arrive;
    0 B stack, 0 B spill.) Raises NvjitUnavailable; returns st."""
    missing = [k for k in ("stack_bytes", "spill_stores", "spill_loads") if st.get(k) is None]
    if missing:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: the build log carries no ptxas statistics ({', '.join(missing)}; built without {PTXAS_VERBOSE}?) — a kernel of this kit "
                               "must show it is spill-free; refusing by name")
    bad = [f"{k}={st[k]}" for k in ("stack_bytes", "spill_stores", "spill_loads") if st[k] != 0]
    bad += [f"{d['code']}({d['text'][:60]})" for d in st.get("diagnostics", []) if d["code"] != "C7517"]   # C7517 (warpgroup.wait injected: conservative) is recorded only; every other advisory refuses
    if bad:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: this build is not spill-free ({', '.join(bad)}): the kernel's spec demands a build without local-memory spills "
                               "— refusing by name; nothing written")
    return st


def compile_options(arch=ARCH, macros=None, opts=()):
    """The NVRTC options of a build WITHOUT the include paths (those are install locations; the header wheels' versions key their content)."""
    options = [f"--gpu-architecture={arch}", "-std=c++17", "-default-device"]
    if str(arch).endswith("a"):
        options += ["-DCUTE_ARCH_MMA_SM90A_ENABLED", "-DCUTLASS_ARCH_MMA_SM90A_ENABLED=1"]
    options += [f"-D{k}={v}" for k, v in sorted((macros or {}).items())]
    options += [str(o) for o in opts]
    return options


def source_key(src, options):
    """sha256 over the kernel source text and its compile options (less include paths): what a cubin was built FROM — no toolchain needed to compute it."""
    return hashlib.sha256((src + "\n" + "\n".join(options)).encode()).hexdigest()


def cubin_key(src, options, ver=None, headers=None):
    """The JIT-cache key: source_key + the NVRTC and header-wheel versions of THIS interpreter (a toolchain move re-keys every cached cubin)."""
    ver = nvrtc_version() if ver is None else ver
    headers = header_versions() if headers is None else headers
    text = source_key(src, options) + "\n" + f"nvrtc={ver}" + "\n" + ",".join(f"{k}={v}" for k, v in sorted(headers.items()))
    return hashlib.sha256(text.encode()).hexdigest()[:24]


def source_key_text(options):
    return ""


def stem_of(name):
    return os.path.splitext(os.path.basename(name))[0]


def cubin_filename(name, key):
    return f"{stem_of(name)}-{key}.cubin"


class Cubin:
    """One compile_cubin() outcome: the cubin bytes and where they came from."""
    __slots__ = ("data", "name", "tag", "key", "sha256", "source", "path", "secs", "log", "arch", "nvrtc", "options", "manifest", "skey", "ptxas", "spill_free")

    def __init__(self, data, name, tag, key, source, path, secs, log, arch, nvrtc, options):
        self.data, self.name, self.tag, self.key = bytes(data), name, tag, key
        self.sha256 = hashlib.sha256(self.data).hexdigest()
        self.source, self.path, self.secs, self.log, self.arch, self.nvrtc, self.options = source, path, float(secs), log, arch, tuple(nvrtc), list(options)
        self.manifest = {}                                             # the shipped manifest entry when source == "shipped"
        self.skey = source_key_text(options)                           # placeholder until the creator sets the real source key (it knows the source text)
        self.ptxas = {}                                                # ptxas_stats of the build log (a compiled cubin) or the manifest's record (a shipped one)
        self.spill_free = True                                         # the spec's policy (compile_cubin sets it; a shipped cubin carries it in its manifest entry)

    def record(self):
        return dict(tag=self.tag, name=self.name, file=(os.path.basename(self.path) if self.path else cubin_filename(self.name, self.key)), sha256=self.sha256, source=self.source,
                    path=self.path, secs=round(self.secs, 1), arch=self.arch, nvrtc=f"{self.nvrtc[0]}.{self.nvrtc[1]}", bytes=len(self.data),
                    source_key=self.skey)


def records():
    """Every cubin outcome of this process (dicts), oldest first."""
    return [dict(r) for r in _RECORDS]


def _say(text):
    print(f"{PREFIX} {text}", file=sys.stderr, flush=True)


def _nvrtc_compile(src, name, options):
    """NVRTC source -> (cubin bytes, log). Raises NvjitUnavailable with the log's error excerpt."""
    nvrtc = _nvrtc()
    prog = _chk(nvrtc.nvrtcCreateProgram(src.encode(), name.encode(), 0, [], []), "nvrtcCreateProgram")
    try:
        res = nvrtc.nvrtcCompileProgram(prog, len(options), [o.encode() for o in options])
        err = res[0] if isinstance(res, tuple) else res
        log_size = _chk(nvrtc.nvrtcGetProgramLogSize(prog), "nvrtcGetProgramLogSize"); log = b" " * log_size
        _chk(nvrtc.nvrtcGetProgramLog(prog, log), "nvrtcGetProgramLog")
        log = log.decode(errors="replace").strip("\x00").strip()
        if err != nvrtc.nvrtcResult.NVRTC_SUCCESS:
            lines = log.split("\n")
            hits = [k for k, l in enumerate(lines) if " error" in l or l.startswith("error")]
            excerpt = "\n".join("\n".join(lines[max(0, k - 2):k + 4]) for k in hits[:6]) or log[-3000:]
            raise NvjitUnavailable(f"ef2_nvjit: NVRTC compilation of {name} failed ({err}):\n{excerpt}")
        size = _chk(nvrtc.nvrtcGetCUBINSize(prog), "nvrtcGetCUBINSize"); cubin = b" " * size
        _chk(nvrtc.nvrtcGetCUBIN(prog, cubin), "nvrtcGetCUBIN")
        return bytes(cubin), log
    finally:
        nvrtc.nvrtcDestroyProgram(prog)


def _write_atomic(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + f".{os.getpid()}.tmp"
    with open(tmp, "wb") as f:
        f.write(data)
    os.replace(tmp, path)


def shipped_cubin(src, name="kernel.cu", arch=ARCH, macros=None, opts=(), verbose=True, tag=None):
    """The run-time route -> the kit's shipped Cubin for this source + options. Raises NvjitUnavailable BY NAME when the manifest has no entry for
    the kernel, its source_key is another (the tree's source moved: rebuild with driver/prebuilt/build_prebuilt.py), the file is absent, its
    sha256 differs from the manifest's, or the manifest does not record a spill-free build. Never compiles."""
    tag = tag or name
    options = compile_options(arch, macros, opts)
    man = shipped_manifest(arch)
    ent = (man.get("cubins") or {}).get(stem_of(name))
    where = shipped_dir(arch)
    if not ent:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: no shipped cubin for {stem_of(name)} in {where} (manifest {'absent' if not man else 'has ' + ','.join(sorted(man.get('cubins') or {})) or 'no entries'}); "
                               "the kit runs shipped cubins only (driver/prebuilt/build_prebuilt.py builds them) — refusing by name")
    skey = source_key(src, options)
    if ent.get("source_key") != skey:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: the shipped cubin {ent.get('file')} was built from source key {str(ent.get('source_key'))[:12]}, this tree's {stem_of(name)} source + "
                               f"options hash to {skey[:12]} (the kernel source or its options moved: rebuild with driver/prebuilt/build_prebuilt.py) — refusing by name")
    path = os.path.join(where, ent.get("file") or "")
    if not os.path.isfile(path):
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: the manifest of {where} names {ent.get('file')} but the file is absent (an incomplete kit tree) — refusing by name")
    with open(path, "rb") as f:
        data = f.read()
    from opt_core.gates import binary_refusal  # noqa: PLC0415
    why = binary_refusal(path, data)
    if why:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: shipped cubin {path} refused: {why} — refusing by name")
    sha = hashlib.sha256(data).hexdigest()
    if sha != ent.get("cubin_sha256"):
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: shipped cubin {path} sha256 {sha[:12]} != manifest {str(ent.get('cubin_sha256'))[:12]} (a corrupt or hand-edited file; "
                               "driver/prebuilt/build_prebuilt.py rebuilds it) — refusing by name")
    rec = {k: ent.get(k) for k in ("stack_bytes", "spill_stores", "spill_loads")}
    strict = bool(ent.get("spill_free_required", True))
    if any(v is None for v in rec.values()) or (strict and any(int(v) != 0 for v in rec.values())) or any(d.get("code") != "C7517" for d in (ent.get("diagnostics") or [])):
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: the manifest of {where} does not record an acceptable build for {ent.get('file')} (stack_bytes/spill_stores/spill_loads = {rec}, "
                               f"spill-free required = {strict}, advisories {[d.get('code') for d in (ent.get('diagnostics') or [])]}) — refusing by name")
    nv = tuple(int(v) for v in str(ent.get("nvrtc") or "0.0").split(".")[:2]) if ent.get("nvrtc") else (0, 0)
    cub = Cubin(data, name, tag, skey[:24], "shipped", path, 0.0, "", arch, nv, options)
    cub.manifest = dict(ent); cub.skey = skey; cub.ptxas = dict(rec, regs=ent.get("regs_ptxas"), c7512=bool(ent.get("c7512")))
    _RECORDS.append(cub.record())
    if verbose:
        _say(f"{tag}: cubin {os.path.basename(path)} (sha256 {sha[:12]}, {len(data)} B) shipped with the kit ({where}; built with nvrtc {ent.get('nvrtc')} from source key "
             f"{str(ent.get('source_key'))[:12]}; ptxas {ent.get('regs_ptxas')} regs, stack {rec['stack_bytes']} B, spill {rec['spill_stores']}/{rec['spill_loads']} B; no compile)")
    return cub


def compile_cubin(src, name="kernel.cu", arch=ARCH, macros=None, opts=(), verbose=True, tag=None, dest_dir=None, spill_free=True):
    """The BUILD route (driver/prebuilt/build_prebuilt.py; never the run time) -> Cubin compiled with NVRTC into ``dest_dir``. Raises NvjitUnavailable
    by name when the toolchain is unavailable, the compiler fails (its log), or the build is not spill-free (ptxas stack / spill / C7512:
    nothing is written)."""
    tag = tag or name
    options = compile_options(arch, macros, opts)
    dopts = " ".join(o for o in options if o.startswith("-D") and "ARCH_MMA" not in o) or "no -D"
    ok, why_tool = toolchain()
    if not ok:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: the NVRTC build route is unavailable in this interpreter ({why_tool})")
    ver = nvrtc_version(); incs = include_dirs()
    key = cubin_key(src, options, ver, header_versions())
    fn = cubin_filename(name, key)
    if verbose:
        _say(f"{tag}: compiling {name} with NVRTC {ver[0]}.{ver[1]} for {arch} ({dopts}) ...")
    t0 = time.time()
    data, log = _nvrtc_compile(src, name, options + [f"-I{d}" for d in incs])                     # the SHIPPING object
    twin, twin_log = _nvrtc_compile(src, name, options + [PTXAS_VERBOSE] + [f"-I{d}" for d in incs])   # its statistics twin (ptxas -v in the log; the option lands in .note.nv.tkinfo only)
    secs = time.time() - t0
    same, diff = elf_sections_equal(data, twin)
    if not same:
        raise NvjitUnavailable(f"ef2_nvjit: {tag}: the statistics twin differs from the shipping build outside {TKINFO_NOTE} (sections {diff}): its ptxas numbers would not "
                               "describe the shipped object — refusing by name; nothing written")
    st = require_build_policy(ptxas_stats(twin_log), tag, spill_free)  # a build outside its spec's policy (spills where none are allowed / a C75xx advisory but C7517) is refused by name: nothing written
    st["twin_sha256"] = hashlib.sha256(twin).hexdigest(); st["twin_note"] = f"ptxas statistics read from a twin build with {PTXAS_VERBOSE}; every ELF section but {TKINFO_NOTE} identical to the shipped cubin"
    log = (log or "") + "\n---- statistics twin (" + PTXAS_VERBOSE + ") ----\n" + (twin_log or "")
    dest = dest_dir or tempfile.mkdtemp(prefix="ef2_nvjit_")
    path = os.path.join(dest, fn)
    _write_atomic(path, data)
    with open(path[:-6] + ".log", "w") as f:                            # the build log (ptxas statistics included) beside the cubin
        f.write(log or "")
    cub = Cubin(data, name, tag, key, "compiled", path, secs, log, arch, ver, options); cub.skey = source_key(src, options); cub.ptxas = st; cub.spill_free = bool(spill_free)
    _RECORDS.append(cub.record())
    if verbose:
        _say(f"{tag}: compiled in {secs:.1f}s -> {path} (sha256 {cub.sha256[:12]}, {len(data)} B; ptxas: {st['regs']} regs, stack {st['stack_bytes']} B, "
             f"spill {st['spill_stores']}/{st['spill_loads']} B){(' log: ' + log[-800:]) if log else ''}")
    return cub


# ------------------------------------------------------------------------------------------------------------------------------ launch
class TensorMap:
    """CUtensorMap over a 2-D..5-D view of a tensor: dims innermost-first, strides_bytes for dims 1.., box innermost-first, swizzle bytes in {0,32,64,128}."""

    def __init__(self, tensor, dims, strides_bytes, box, swizzle=128, dtype="bf16", l2=128, keep=True):
        cu = _drv()
        dt = {"bf16": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16, "f32": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
              "f16": cu.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_FLOAT16}[dtype]
        sw = {0: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_NONE, 32: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_32B,
              64: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_64B, 128: cu.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B}[swizzle]
        l2p = {0: cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_NONE, 64: cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
               128: cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_128B, 256: cu.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B}[l2]
        rank = len(dims)
        if not (2 <= rank <= 5 and len(strides_bytes) == rank - 1 and len(box) == rank):
            raise NvjitUnavailable(f"ef2_nvjit.TensorMap: rank {rank} dims/strides/box {dims}/{strides_bytes}/{box} inconsistent")
        ptr = tensor.data_ptr() if hasattr(tensor, "data_ptr") else int(tensor)
        if ptr % 16:
            raise NvjitUnavailable("ef2_nvjit.TensorMap: the TMA base address must be 16-byte aligned")
        self.tm = _chk(cu.cuTensorMapEncodeTiled(dt, rank, ptr, [cu.cuuint64_t(d) for d in dims], [cu.cuuint64_t(s) for s in strides_bytes],
                                              [cu.cuuint32_t(b) for b in box], [cu.cuuint32_t(1)] * rank,
                                              cu.CUtensorMapInterleave.CU_TENSOR_MAP_INTERLEAVE_NONE, sw, l2p,
                                              cu.CUtensorMapFloatOOBfill.CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE), "cuTensorMapEncodeTiled")
        buf = (ctypes.c_uint64 * 16)()
        ctypes.memmove(buf, int(self.tm.getPtr()), 128)
        self.cbuf = buf
        self.keep = tensor if keep else None          # keep=False: the descriptor records only the address (activation buffers must not be pinned)


class Kernel:
    """A loaded cubin function: ``Kernel(cubin, "name", smem_bytes=…)(grid, block, *args, stream=None)``. Attributes for the LEVER line:
    ``regs`` (registers per thread), ``lmem`` (local = spill bytes per thread), ``smem`` (dynamic shared bytes requested), ``cubin_sha256``, ``source``."""

    def __init__(self, cubin, func_name, smem_bytes=0, cluster=None):
        import torch
        cu = _drv()
        torch.cuda.init()
        torch.empty(1, device="cuda")                                    # a current context
        data = cubin.data if isinstance(cubin, Cubin) else bytes(cubin)
        self.module = _chk(cu.cuModuleLoadData(data), f"cuModuleLoadData({func_name})")
        self.func = _chk(cu.cuModuleGetFunction(self.module, func_name.encode()), f"cuModuleGetFunction({func_name})")
        if smem_bytes > 48 * 1024:
            _chk(cu.cuFuncSetAttribute(self.func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, smem_bytes), "cuFuncSetAttribute(smem)")
        self.smem = smem_bytes
        self.cluster = tuple(cluster) if cluster else None
        if self.cluster and (self.cluster[0] * self.cluster[1] * self.cluster[2]) > 8:
            _chk(cu.cuFuncSetAttribute(self.func, cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NON_PORTABLE_CLUSTER_SIZE_ALLOWED, 1), "cuFuncSetAttribute(cluster)")
        self.regs = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_NUM_REGS, self.func), "cuFuncGetAttribute(regs)")
        self.lmem = _chk(cu.cuFuncGetAttribute(cu.CUfunction_attribute.CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, self.func), "cuFuncGetAttribute(lmem)")
        man = (cubin.manifest or {}) if isinstance(cubin, Cubin) else {}
        strict = bool(man.get("spill_free_required", True)) if man else bool(getattr(cubin, "spill_free", True))
        if strict and int(self.lmem) != 0:                             # the device's own word on the loaded function: a kernel that must be spill-free and reports local memory refuses by name
            raise NvjitUnavailable(f"ef2_nvjit: {func_name}: the loaded kernel uses {int(self.lmem)} B of local memory per thread (register spill / stack) and its policy is spill-free "
                                   "— refusing by name")
        for key, got in (("regs", int(self.regs)), ("spill_bytes", int(self.lmem))):   # the manifest's device record (build_prebuilt.py check on an H100): the same binary reports the same numbers
            want = man.get(key)
            if want is not None and int(want) != got:
                raise NvjitUnavailable(f"ef2_nvjit: {func_name}: the loaded kernel reports {key}={got}, the shipped manifest records {int(want)} — not the audited binary / driver pairing; "
                                       "refusing by name")
        self.name = func_name
        self.cubin_sha256 = cubin.sha256 if isinstance(cubin, Cubin) else hashlib.sha256(data).hexdigest()
        self.source = cubin.source if isinstance(cubin, Cubin) else "bytes"
        self.compile_secs = cubin.secs if isinstance(cubin, Cubin) else 0.0
        self.log = cubin.log if isinstance(cubin, Cubin) else ""

    def evidence(self):
        """Blank-free words for the LEVER line: cubin sha prefix, where it came from, regs / spill / smem, arch."""
        return dict(cubin=self.cubin_sha256[:12], cubin_source=self.source, regs=int(self.regs), spill_bytes=int(self.lmem), smem=int(self.smem), arch=ARCH)

    def max_active_clusters(self, grid_x, block_x):
        cu = _drv()
        cfg = cu.CUlaunchConfig()
        cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid_x, 1, 1
        cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block_x, 1, 1
        cfg.sharedMemBytes = self.smem
        attr = cu.CUlaunchAttribute(); attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
        attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = self.cluster or (1, 1, 1)
        cfg.attrs = [attr]; cfg.numAttrs = 1
        return _chk(cu.cuOccupancyMaxActiveClusters(self.func, cfg), "cuOccupancyMaxActiveClusters")

    def __call__(self, grid, block, *args, smem=None, stream=None):
        import torch
        cu = _drv()
        holders, ptrs = [], []
        for a in args:
            if isinstance(a, torch.Tensor):
                h = ctypes.c_uint64(a.data_ptr())
            elif isinstance(a, TensorMap):
                h = a.cbuf
            elif isinstance(a, bool):
                h = ctypes.c_int32(int(a))
            elif isinstance(a, int):
                h = ctypes.c_int32(a)
            elif isinstance(a, float):
                h = ctypes.c_float(a)
            elif isinstance(a, (ctypes.c_int32, ctypes.c_int64, ctypes.c_uint64, ctypes.c_float, ctypes.c_double, ctypes.c_uint32)):
                h = a
            elif a is None:
                h = ctypes.c_uint64(0)
            else:
                raise TypeError(f"ef2_nvjit.Kernel: unsupported argument type {type(a)}")
            holders.append(h); ptrs.append(ctypes.addressof(h))
        arr = (ctypes.c_void_p * len(ptrs))(*ptrs)
        st = stream if stream is not None else torch.cuda.current_stream().cuda_stream
        sm = self.smem if smem is None else smem
        if self.cluster:
            cfg = cu.CUlaunchConfig()
            cfg.gridDimX, cfg.gridDimY, cfg.gridDimZ = grid
            cfg.blockDimX, cfg.blockDimY, cfg.blockDimZ = block
            cfg.sharedMemBytes = sm
            cfg.hStream = cu.CUstream(st)
            attr = cu.CUlaunchAttribute(); attr.id = cu.CUlaunchAttributeID.CU_LAUNCH_ATTRIBUTE_CLUSTER_DIMENSION
            attr.value.clusterDim.x, attr.value.clusterDim.y, attr.value.clusterDim.z = self.cluster
            cfg.attrs = [attr]; cfg.numAttrs = 1
            _chk(cu.cuLaunchKernelEx(cfg, self.func, arr, 0), f"cuLaunchKernelEx({self.name})")
        else:
            _chk(cu.cuLaunchKernel(self.func, grid[0], grid[1], grid[2], block[0], block[1], block[2], sm, st, arr, 0), f"cuLaunchKernel({self.name})")
        return holders


# ------------------------------------------------------------------------------------------------------------------------------ the shipped set: specs, the install-time check (run.sh install), build (prebuilt/build_prebuilt.py)
def kernel_specs(modules=PREBUILD_MODULES):
    """[spec, …] of every kernel build the fast / big set's NVRTC levers name (``<module>.nvrtc_sources()``): dict(tag, src, name, macros, opts, module, entry, smem)."""
    import importlib
    if HERE not in sys.path:
        sys.path.insert(0, HERE)
    out = []
    for mod_name in modules:
        mod = importlib.import_module(mod_name)
        for spec in mod.nvrtc_sources():
            out.append(dict(spec, module=mod_name))
    return out


def check_shipped(arch=ARCH, modules=PREBUILD_MODULES):
    """CPU-only check of the shipped set against THIS tree: for every kernel spec, the manifest holds an entry of the same source_key whose file
    is present with the manifest's sha256. Returns [record, …]; raises NvjitUnavailable naming the first defect (absent manifest / entry for
    another source key / missing file / sha256 mismatch)."""
    man = shipped_manifest(arch)
    if not man.get("cubins"):
        raise NvjitUnavailable(f"ef2_nvjit: no shipped-cubin manifest at {os.path.join(shipped_dir(arch), MANIFEST)} (driver/prebuilt/build_prebuilt.py writes it)")
    out = []
    for spec in kernel_specs(modules):
        cub = shipped_cubin(spec["src"], spec["name"], arch=arch, macros=spec.get("macros"), opts=tuple(spec.get("opts") or ()), verbose=False, tag=spec.get("tag") or spec["name"])
        out.append(dict(cub.record(), module=spec["module"], regs=cub.manifest.get("regs"), regs_ptxas=cub.manifest.get("regs_ptxas"), stack_bytes=cub.manifest.get("stack_bytes"),
                        spill_stores=cub.manifest.get("spill_stores"), spill_loads=cub.manifest.get("spill_loads"), loadcheck=cub.manifest.get("loadcheck")))
    return out


def build_all(dest, arch=ARCH, modules=PREBUILD_MODULES, verbose=True):
    """Compile every kernel spec with NVRTC into ``dest`` (fresh: neither the shipped set nor the JIT cache is consulted) -> [(spec, Cubin), …].
    Needs the header wheels + NVRTC (NvjitUnavailable by name otherwise); no GPU."""
    out = []
    for spec in kernel_specs(modules):
        cub = compile_cubin(spec["src"], spec["name"], arch=arch, macros=spec.get("macros"), opts=tuple(spec.get("opts") or ()), verbose=verbose,
                            tag=spec.get("tag") or spec["name"], dest_dir=dest, spill_free=bool(spec.get("spill_free", True)))
        out.append((spec, cub))
    return out


def _main(argv):
    import argparse
    ap = argparse.ArgumentParser(prog="ef2_nvjit.py", description="the kit's CUDA-kernel helper: check the shipped sm_90a cubins against this tree (run.sh install)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("cubins", help="CPU-only: shipped cubins present, sha256 == manifest, source keys == this tree's kernel sources")
    sub.add_parser("list", help="print the shipped directory's manifest entries")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        man = shipped_manifest()
        for stem, ent in sorted((man.get("cubins") or {}).items()):
            print(f"{PREFIX} shipped {shipped_dir()}/{ent.get('file')} sha256={str(ent.get('cubin_sha256'))[:12]} bytes={ent.get('bytes')} source_key={str(ent.get('source_key'))[:12]} "
                  f"nvrtc={ent.get('nvrtc')} regs={ent.get('regs')} spill_bytes={ent.get('spill_bytes')} smem={ent.get('smem')}", file=sys.stderr)
        return 0
    try:
        recs = check_shipped()
    except NvjitUnavailable as e:
        print(f"{PREFIX} shipped cubins REFUSED: {e}", file=sys.stderr)
        return 1
    print(f"{PREFIX} shipped set {shipped_dir()} matches this tree: " + ", ".join(f"{r['file']} sha256={r['sha256'][:12]} ({r['bytes']} B, source_key={r['source_key'][:12]}, "
                                                                          f"ptxas regs={r['regs_ptxas']} stack={r['stack_bytes']} spill={r['spill_stores']}/{r['spill_loads']}, device regs={r['regs']})" for r in recs), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(_main(sys.argv[1:]))
