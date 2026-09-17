"""kernels.trimul.esm_v61 -- the ESM-family engine's sealed triangle-multiplication package (Triton K1 planes with tensor-descriptor loads |
cuBLAS bmm | K3 = one warp-specialized CUDA C++/CuTe kernel shipped as a PREBUILT sm_90a cubin and launched through the CUDA driver) carried under
``pkg/<ACTIVE_PKG>/`` and served through ONE face, whatever the package version.

    from opt_core.kernels.trimul import esm_v61 as E
    E.ACTIVE_PKG                         # "v6.1": the payload directory this face serves (a version swap edits this constant and drops pkg/<new>/)
    rep = E.install()                    # digests -> driver binding -> cubin load + load check -> the package's own vectors (byte gate); dict of facts
    wp = E.pack(w10)                     # the package's weight pack from the ten canonical tensors (plain torch; == row esm_v5_fwd's pack + folded LN)
    out = E.forward(z, outgoing, mask, wp, residual=True)      # z + update (the engine's form); residual=False -> the update alone (cofolding form)

Contract (the provider's sealed-package interface, as kernels.triattn.triattn_native):

* payload = ``pkg/<ver>/`` = the producer's package tree byte for byte except: ``README.md`` and ``src/SOURCES.json`` not carried (documentation and
  provenance records; kernels/META/trimul.json names them), four text lines restated in release-tree vocabulary (``python/face.py`` line 21 -- a
  docstring; ``tests/vectors/vectors.json`` line 1306 -- a free-text note; ``bin/manifest.json`` line 15 -- the free-text build note;
  ``src/ef2_trimul_v6.py`` lines 520-523 -- the compile-cache directory helper no longer imports the engine's package: this copy never compiles), and ``tests/vectors/vectors.pt`` (the sealed tensors the package's test
  reads from its first lookup path) placed under ``tests/vectors/``.  ``DIGESTS`` pins every carried file; a byte that differs is the refusal
  ``digest:<file>`` before anything loads.  The kernel source of the cubins (``K3_SRC``) is untouched: its sha256 is the one the manifest records.
* the K3 object is keyed by ARCHITECTURE (an sm_90a cubin loaded with ``cuModuleLoadData``), not by the framework / interpreter ABI: one payload serves
  every image whose driver loads it.  Interpreter-side needs: torch, a triton with tensor descriptors (the K1 planes; 3.6+), and a driver binding --
  ``cudrv``: the ``cuda.bindings`` wheel when the image has it, else ctypes over ``libcuda.so.1``; the package's one driver accessor (``_drv``) is
  pointed at ``cudrv.modules`` at install, nothing in ``sys.modules`` is added or replaced.  Nothing here imports a model package of any image.
* never compiles in a serving process: the manifest-checked cubins are injected into the package's kernel table before any call; a cubin that fails
  its checks is the refusal ``prebuilt:<reason>``, not a build (the package's NVRTC path stays unreachable: ``cudrv``'s NVRTC half refuses by name).
* load check = the producer's ptxas fingerprint read back off the loaded functions (``LOADCHECK``: 168 registers, 0 bytes local); byte gate = the
  package's own ``tests/test_vectors.py`` over its sealed vectors (inputs sha256-checked, every output BITWISE == the recorded bytes for the sm90
  device class, else TOLERANCE within its stated ulp bound; any FAIL is the refusal ``gate:<n>``).
* residual: ``True`` is the package's epilogue (z + update).  ``False`` (the cofolding trunks' form) is served by the SAME cubin with the epilogue's
  residual tensor map laid over a zero tile (every residual box the kernel loads is then either the tile or out of bounds of the map, which the
  tensor-map unit zero-fills): out = update + 0.0 exactly, no extra pass over z.  ``ZERO_TILE_RESIDUAL = False`` turns that into the refusal
  ``needs_residual``.
* binding: the modules this face serves are ALWAYS the carried files (``carried_module()``; ``carried_binding()`` parks any module an engine
  registered under the package's names while ours load, gate and launch, and restores it after): a foreign copy contributes nothing -- it is
  admitted in the process only when its sealed bytes equal ours (``foreign_check``), refused by name otherwise.
* refusals are ``Unavailable`` (``.kind`` = the word): ``digest:<file>`` | ``driver:<reason>`` | ``device:<reason>`` | ``cc:<cc>!=9.0(sm_90a cubin)``
  | ``triton:<v>(<reason>)`` | ``import:<module>(<reason>)`` | ``prebuilt:<reason>`` | ``loadcheck:<what>`` | ``gate:<summary>`` |
  ``foreign_module_digest_*:<name>`` | ``load_failed:<exception class>`` (anything else that breaks while binding / loading / gating) |
  ``unsupported:<the package's own reason>`` | ``dtype:`` / ``shape:`` / ``c_z=`` / ``n<`` envelope words | ``needs_residual``.
"""
import contextlib
import hashlib
import importlib.util
import json
import os
import sys
import threading

ACTIVE_PKG = "v6.1"
ARCH = "sm_90a"                    # the cubins' architecture: capability 9.0 devices only (other capabilities: rows esm_v5_fwd / v4)
KERNEL = "k3v6"                    # the function name inside both cubins
C = 256                            # c_z == c_hidden == 256, the engine's width (128: row esm_v5_fwd; 384 / 64: the stock rows)
N_MIN = 16
TRITON_MIN = (3, 6)                # the K1 planes are written on tl.make_tensor_descriptor
LOADCHECK = {"regs": 168, "local_bytes": 0}      # the producer's ptxas fingerprint of both shipped cubins (0 spill; a build reporting otherwise is not this schedule)
ZERO_TILE_RESIDUAL = True          # residual=False through the zero-tile residual map (see the module docstring)
FACE_MODULE = "opt_core_kernels_trimul_esm_v61_face"
GATE_MODULE = "opt_core_kernels_trimul_esm_v61_gate"
NOT_CARRIED = ("README.md", "src/SOURCES.json")
RESTATED = {"python/face.py": "21", "tests/vectors/vectors.json": "1306", "src/ef2_trimul_v6.py": "520-523", "bin/manifest.json": "15"}
DIGESTS = {                        # sha256 of every carried file of pkg/v6.1/ as carried (the restated files: of the restated bytes)
    "CELLS.json": "33a56ace48e5a502d4311e7a0bad7c638f5200c04b3c8e360e61f9061ee7cb7e",
    "SHA256SUMS": "d1a0d844844728fd7050ad59ca028a0d458e634c59733f72fe7d450a7dd54c43",
    "VERSION": "aca65051611b86150b1acec60d4e959818d21332e20328a35f986329cc144f41",
    "bin/k3v6.1_fastsig0_lnfold1.sm_90a.cubin": "ef4b5b2f3ef0d24c7550ffe7c294d5f31537c2070499b0431263ec2c6e2881ce",
    "bin/k3v6.1_fastsig1_lnfold1.sm_90a.cubin": "7fb328f9bff9ea39f9d32f3c81844b118f7c0cde7baa304ee471443abdc332af",
    "bin/manifest.json": "59742cdf114a90ce1f225c2e9a205f23fecc86ed82f21fc72a0126431e1a8998",
    "python/face.py": "8d2378c015ca2b311d67e674508ad8f4a89a669cd97ab3f66e3d848699dbec2e",
    "src/PATCHES/k3v6_release_ordering_fix.diff": "10f4ed4d4e4a4f10012d389deb6377ff50f6db186597391406e2fd9cba6cae79",
    "src/ef2_trimul_v5.py": "a96d57bf7a294828835ef1e694b975f28076f4d15ce6ac0803637abaea5d296a",
    "src/ef2_trimul_v6.py": "943a98bc7b58c705b911aafc7ebf70b9c4cb1da8f2bdd8444970deabf568dd2b",
    "src/ef2_w4_fpf_trimul_v4_cells.json": "ea6f9639180ea896aaeff125d0b6de111a10e0c043e045ae924a2c63fd611bd9",
    "src/k3v6.cu": "92156718392b7ec09d58ab000c6ec9d9ce80685981680a4a32c6eb97a26e423c",
    "tests/test_vectors.py": "c267d5573a347525a59b94bbc652762e15467425f06ccb2ae7313b1335807047",
    "tests/vectors/vectors.json": "8d57a79354a0bc17b6adba0811768ec0ec387c94221b823e2fbe314e6f416245",
    "tests/vectors/vectors.pt": "f74c061cea469d128727c687e7eec14d422971cd6c351e03de1ab2ac938e1c7c",
}
K3_SRC_SHA256 = "558a71ac1cc047bfe3301d9d603ed56b44478b990dd4d6d585a86533b516ca1f"     # bin/manifest.json source_sha256.K3_SRC of both cubins
HERE = os.path.dirname(os.path.abspath(__file__))
_LOCK = threading.Lock()
_BIND_LOCK = threading.RLock()          # around every sys.modules park / restore of the package names
_STATE = {}

__all__ = ["ACTIVE_PKG", "foreign_check", "foreign_digests", "foreign_modules", "carried_binding", "carried_module", "PKG_MODULE_NAMES", "ARCH", "C", "N_MIN", "DIGESTS", "Unavailable", "pkg_dir", "verify_digests", "manifest", "face", "install", "report",
           "pack", "check", "forward", "describe"]


class Unavailable(RuntimeError):
    def __init__(self, kind, detail=""):
        RuntimeError.__init__(self, "esm_v61@%s: %s%s" % (ACTIVE_PKG, kind, (" (%s)" % detail) if detail else ""))
        self.kind = kind
        self.reason = kind


def pkg_dir(pkg=""):
    return os.path.join(HERE, "pkg", pkg or ACTIVE_PKG)


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_digests(pkg=""):
    """Every DIGESTS file present with its recorded sha256 and no NOT_CARRIED file present; raises Unavailable('digest:<rel>') by name."""
    root = pkg_dir(pkg)
    if not os.path.isdir(root):
        raise Unavailable("digest:pkg/%s(absent)" % (pkg or ACTIVE_PKG))
    for rel in sorted(DIGESTS):
        p = os.path.join(root, rel)
        if not os.path.isfile(p):
            raise Unavailable("digest:%s(absent)" % rel)
        if _sha256(p) != DIGESTS[rel]:
            raise Unavailable("digest:%s" % rel)
        if rel.endswith(".cubin"):
            from opt_core.gates import binary_refusal  # noqa: PLC0415
            why = binary_refusal(p)
            if why:
                raise Unavailable("digest:%s(%s)" % (rel, why))
    for rel in NOT_CARRIED:
        if os.path.exists(os.path.join(root, rel)):
            raise Unavailable("digest:%s(not part of the carried payload)" % rel)
    return dict(DIGESTS)


def manifest(pkg=""):
    """bin/manifest.json of the payload (the producer's build record of both cubins)."""
    with open(os.path.join(pkg_dir(pkg), "bin", "manifest.json"), encoding="utf-8") as f:
        return json.load(f)


def cubins(pkg=""):
    """{(fastsig, lnfold): {cubin, cubin_sha256, cubin_bytes, arch, regs, lmem}} from the manifest."""
    out = {}
    for ent in manifest(pkg).get("kernels", []):
        if ent.get("kernel") == KERNEL:
            out[(int(ent["fastsig"]), int(ent["lnfold"]))] = dict(cubin=ent["cubin"], cubin_sha256=ent["cubin_sha256"], cubin_bytes=ent["cubin_bytes"],
                                                                    arch=ent["arch"], regs=ent.get("regs"), lmem=ent.get("lmem"))
    return out


def _load_by_path(name, path):
    m = sys.modules.get(name)
    if m is not None and os.path.abspath(getattr(m, "__file__", "") or "") == os.path.abspath(path):
        return m
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


def face(pkg=""):
    """The package's own python/face.py loaded by path (imports torch).  Its load() registers the carried ef2_trimul_v5 / ef2_trimul_v6 modules under
    those names and would use a module of either name already imported from elsewhere instead -- install() never lets it: the carried files are
    bound inside ``carried_binding()`` (foreign entries parked), and a foreign copy whose sealed bytes differ from ours is refused by name first."""
    return _load_by_path(FACE_MODULE + ("_" + pkg.replace(".", "_") if pkg else ""), os.path.join(pkg_dir(pkg), "python", "face.py"))


PKG_MODULE_NAMES = ("ef2_trimul_v6", "ef2_trimul_v5")


def _is_carried(m):
    """True when module ``m`` was loaded from this package's payload directory."""
    f = os.path.abspath(getattr(m, "__file__", "") or "")
    return bool(f) and f.startswith(os.path.abspath(pkg_dir()) + os.sep)


def foreign_modules():
    """{name: module} for modules registered in sys.modules under the package's names but loaded from elsewhere (an engine's own copy)."""
    return {n: m for n in PKG_MODULE_NAMES for m in (sys.modules.get(n),) if m is not None and not _is_carried(m)}


@contextlib.contextmanager
def carried_binding(F=None):
    """For the duration: ``sys.modules[<package names>]`` are the CARRIED modules (the face's registry ``F._MODS`` when loaded) and never a foreign
    copy -- foreign entries are parked and restored on exit, so the engine keeps importing its own module by name afterwards.  Everything of this
    package that resolves those names (the face's loader, the v5 host module's ``import ef2_trimul_v6``, the byte gate's fresh face) runs inside."""
    F = F if F is not None else face()
    parked, placed = {}, {}
    with _BIND_LOCK:
        for name in PKG_MODULE_NAMES:
            m = sys.modules.get(name)
            if m is not None and not _is_carried(m):
                parked[name] = sys.modules.pop(name)
            ours = (getattr(F, "_MODS", None) or {}).get(name)
            if ours is not None and _is_carried(ours) and sys.modules.get(name) is not ours:
                placed[name] = ours
                sys.modules[name] = ours
        try:
            yield parked
        finally:
            for name, ours in placed.items():
                if sys.modules.get(name) is ours:
                    del sys.modules[name]
            for name, m in parked.items():
                sys.modules[name] = m


_CACHE_DIRS: dict = {}


def _private_cache_dir(path: str, tag: str, own_only: bool = False) -> str:
    """``path`` (created 0700 when absent) when nothing another account could have written would be loaded from it; else — one stderr line:
    what, why, the fix — a fresh directory private to this process, where its build products are compiled again. Refused: a directory (or,
    when group or other can enter it, a file in it) that is writable by group or other, or whose owner is neither this uid nor root — uid 0
    accepts any owner (a container's root reading a bind-mounted host directory). ``own_only`` (the per-user default under the shared
    temporary directory): this uid alone, and never a symbolic link. No digest kept beside a file would add to this: whoever can write the
    directory can rewrite the digest, so the owner / mode rule is the check."""
    import stat
    if path in _CACHE_DIRS:
        return _CACHE_DIRS[path]
    os.makedirs(path, mode=0o700, exist_ok=True)
    uid, why = os.geteuid(), None
    names = [""] + (sorted(os.listdir(path)) if os.stat(path).st_mode & 0o011 else [])
    for name in names:
        p = os.path.join(path, name) if name else path
        st = os.lstat(p) if own_only else os.stat(p)
        if own_only and not name and (os.path.islink(p) or st.st_uid != uid):
            why = f"{p} is a symbolic link or belongs to uid {st.st_uid}, not to this process (uid {uid}); fix: remove it, or name a directory of your own"
        elif st.st_mode & 0o022 and not stat.S_ISLNK(st.st_mode):     # a link's own mode says nothing (os.stat above already followed it unless own_only)
            why = f"{p} is writable by group or other (mode {st.st_mode & 0o7777:04o}); fix: chmod go-w {p}"
        elif uid != 0 and st.st_uid not in (uid, 0):
            why = f"{p} belongs to uid {st.st_uid}, not to this process (uid {uid}) or root; fix: name a cache directory of your own, or chown {p}"
        if why is not None:
            break
    out = path
    if why is not None:
        import tempfile
        out = tempfile.mkdtemp(prefix=tag + "-")
        print(f"[opt_core] {tag}: REFUSED cache directory {path}: {why} — nothing in it is loaded; this process compiles again, privately, in {out}",
              file=sys.stderr, flush=True)
    _CACHE_DIRS[path] = out
    return out


def _jit_dir_private(carried_jit_dir):
    """The carried module's compile-cache directory (a cubin found there is loaded as found) held to :func:`_private_cache_dir`; its default
    under the shared temporary directory becomes per user (``-uid<uid>``). A directory that cannot be created is returned as it is: nothing
    is read from it, and the carried code keeps its cubin in memory."""
    import tempfile
    d = carried_jit_dir()
    shared = os.path.join(tempfile.gettempdir(), "esmfold2_opt_nvrtc")
    own = d == shared
    try:
        return _private_cache_dir("%s-uid%d" % (shared, os.getuid()) if own else d, "esm_v61_nvrtc", own_only=own)
    except OSError:
        return d


def carried_module(name="ef2_trimul_v6"):
    """The CARRIED module of that name as bound by install() (the loaded face's registry) -- never a sys.modules lookup of the package name, and it
    loads nothing itself: None before install (or in a process where the face never loaded)."""
    F = sys.modules.get(FACE_MODULE)                                     # the face as install() loaded it by path; absent = nothing bound yet
    m = (getattr(F, "_MODS", None) or {}).get(name) if F is not None else None
    return m if (m is not None and _is_carried(m)) else None


def _triton_word():
    try:
        import triton
        import triton.language as tl
    except ImportError as e:
        raise Unavailable("import:triton(%s)" % str(e).split("\n")[0][:60].replace(" ", "_"))
    v = str(getattr(triton, "__version__", "0.0"))
    mm = tuple(int(x) for x in (v.split(".") + ["0"])[:2] if x.isdigit())
    if len(mm) < 2 or mm < TRITON_MIN or not hasattr(tl, "make_tensor_descriptor") or not hasattr(triton, "set_allocator"):
        raise Unavailable("triton:%s<3.6(tl.make_tensor_descriptor)" % v)
    return v


def foreign_digests(m):
    """(k3_src_sha256 | None, {cubin file: sha256} | {}) of a module named like the package's, from what it exposes: its ``K3_SRC`` text and the
    ``bin/manifest.json`` beside its ``src/`` directory."""
    src = getattr(m, "K3_SRC", None)
    k3 = hashlib.sha256(src.encode()).hexdigest() if isinstance(src, str) else None
    cub = {}
    f = getattr(m, "__file__", None)
    if f:
        man = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(f))), "bin", "manifest.json")
        try:
            with open(man, encoding="utf-8") as fh:
                j = json.load(fh)
            for ent in (j.get("kernels") or j.get("cubins") or []):
                if isinstance(ent, dict) and ent.get("cubin") and ent.get("cubin_sha256"):
                    cub[str(ent["cubin"])] = str(ent["cubin_sha256"])
        except (OSError, ValueError, AttributeError):
            pass
    return k3, cub


def foreign_check(name, m):
    """A module registered under one of the package's names but loaded from elsewhere is admitted only when its sealed bytes are ours: the K3
    source digest must equal K3_SRC_SHA256 and every cubin digest its manifest lists must be one of ours.  Raises Unavailable BY NAME otherwise
    (``foreign_module_digest_unknown:<name>`` when it exposes no K3 source, ``foreign_module_digest_mismatch:<name>:kit_<sha8>!=core_<sha8>``)."""
    if name != "ef2_trimul_v6":
        return None                                                      # the v5 host module carries no kernel bytes of its own; v6 decides
    k3, cub = foreign_digests(m)
    if k3 is None:
        raise Unavailable("foreign_module_digest_unknown:%s" % name, getattr(m, "__file__", "?"))
    if k3 != K3_SRC_SHA256:
        raise Unavailable("foreign_module_digest_mismatch:%s:kit_%s!=core_%s" % (name, k3[:8], K3_SRC_SHA256[:8]), getattr(m, "__file__", "?"))
    ours = {str(e.get("cubin_sha256")) for e in (manifest().get("kernels") or manifest().get("cubins") or []) if isinstance(e, dict) and e.get("cubin_sha256")}
    bad = sorted(v for v in cub.values() if ours and v not in ours)
    if bad:
        raise Unavailable("foreign_module_digest_mismatch:%s:cubin_kit_%s!=core_%s" % (name, bad[0][:8], sorted(ours)[0][:8]), getattr(m, "__file__", "?"))
    return k3


def install(binding=None, gate=True, device=None):
    """Verify and load, once per process (later calls return the first report): payload digests -> driver binding (``binding``: None = the wheel
    else ctypes; 'ctypes' / 'wheel' force one) -> device is capability 9.0 -> triton has tensor descriptors -> the package face loads both cubins
    after its manifest checks (cubin sha256, kernel-source sha256, launch configuration, architecture) -> load check (registers / local bytes of the
    loaded functions == LOADCHECK) -> byte gate (``gate=False`` skips it: for a caller that gates elsewhere).  Raises Unavailable(kind) by name."""
    with _LOCK:
        rep = _STATE.get("report")
        if rep is not None:
            return rep
        err = _STATE.get("error")
        if err is not None:
            raise Unavailable(err[0], err[1])
        try:
            rep = _install(binding, gate, device)
        except Unavailable as e:
            _STATE["error"] = (e.kind, str(e))
            raise
        except Exception as e:                                           # noqa: BLE001 -- whatever breaks while binding / loading / gating is a REFUSAL BY NAME
            from ....oom import is_oom
            if is_oom(e):
                raise                                                    # device memory exhaustion is the caller's to reroute, never disguised as a refusal
            err = Unavailable("load_failed:%s" % type(e).__name__, str(e).split("\n")[0][:200])   # (a word steps aside to the next row), never an
            _STATE["error"] = (err.kind, str(err))                                              # uncaught raise through the provider
            raise err from e
        _STATE["report"] = rep
        return rep


def _install(binding, gate, device):
    verify_digests()
    from . import cudrv
    try:
        word = cudrv.ensure(binding)
    except cudrv.Unresolvable as e:
        raise Unavailable("driver:%s" % str(e).replace(" ", "_")[:100])
    try:
        import torch
    except ImportError as e:
        raise Unavailable("import:torch(%s)" % str(e).split("\n")[0][:60].replace(" ", "_"))
    if not torch.cuda.is_available():
        raise Unavailable("device:none(cuda only)")
    dev = torch.device(device) if device is not None else torch.device("cuda", torch.cuda.current_device())
    cc = torch.cuda.get_device_capability(dev)
    if tuple(cc) != (9, 0):
        raise Unavailable("cc:%d.%d!=9.0(sm_90a cubin)" % (cc[0], cc[1]))
    tv = _triton_word()
    F = face()
    foreign = foreign_modules()                                          # an engine's own copy registered under the package's names is admitted in the
    for name, m in foreign.items():                                      # PROCESS only when its sealed bytes equal ours (foreign_module_digest_* by name
        foreign_check(name, m)                                           # otherwise); even then nothing of it is used: the carried modules are what binds
    try:
        with carried_binding(F):                                         # foreign entries parked: the face's loader and the v5 host module's own
            v6m = F._import("ef2_trimul_v6", "ef2_trimul_v6.py")        # `import ef2_trimul_v6` resolve to the CARRIED files, registered in F._MODS
            v6m._drv = cudrv.modules                                     # its single driver accessor -> the ensured binding (wheel or ctypes); the file is untouched
            v6m._jit_dir = lambda _carried=v6m._jit_dir: _jit_dir_private(_carried)   # its compile-cache directory -> held to _private_cache_dir; the file is untouched
            with torch.cuda.device(dev):
                v5, v6 = F.load(prebuilt=True)
        for name in PKG_MODULE_NAMES:                                    # whatever the process had registered, the face bound OUR files
            ours = (getattr(F, "_MODS", None) or {}).get(name)
            if ours is None or not _is_carried(ours):
                raise Unavailable("foreign_module_bound:%s" % name, getattr(ours, "__file__", "?"))
        if v6 is not carried_module("ef2_trimul_v6") or not hasattr(v6, "_Kernel") or not hasattr(v6, "_STATE"):
            raise Unavailable("foreign_module_bound:ef2_trimul_v6", getattr(v6, "__file__", "?"))
    except ImportError as e:
        raise Unavailable("import:%s(%s)" % (getattr(e, "name", None) or "module", str(e).split("\n")[0][:60].replace(" ", "_")))
    except RuntimeError as e:                                            # the face's own words (manifest / sha / cfg / arch problems) or V6Unavailable
        raise Unavailable("prebuilt:%s" % str(e).split("\n")[0][:160].replace(" ", "_"))
    if hashlib.sha256(v6.K3_SRC.encode()).hexdigest() != K3_SRC_SHA256:
        raise Unavailable("digest:K3_SRC")
    want = cubins()
    have = getattr(v6, "_STATE", {}).get("kernels", {})
    kern_facts = {}
    for key, ent in sorted(want.items()):
        kern = have.get(key)
        if kern is None:
            raise Unavailable("prebuilt:%s(not loaded)" % ent["cubin"])
        attrs = cudrv.func_attrs(kern.func)
        if attrs.get("regs") != LOADCHECK["regs"] or attrs.get("local_bytes") != LOADCHECK["local_bytes"]:
            raise Unavailable("loadcheck:%s(regs=%s,local=%s;want %d,%d)" % (ent["cubin"], attrs.get("regs"), attrs.get("local_bytes"), LOADCHECK["regs"], LOADCHECK["local_bytes"]))
        kern_facts["fastsig%d_lnfold%d" % key] = dict(cubin=ent["cubin"], sha256=ent["cubin_sha256"], bytes=ent["cubin_bytes"], regs=attrs["regs"],
                                                     local_bytes=attrs["local_bytes"], binary_version=attrs.get("binary_version"), smem_bytes=int(kern.smem))
    gate_word = "skipped"
    gate_lines = []
    if gate:
        rc, gate_lines = run_gate(verbose=False)
        fails = [l for l in gate_lines if l.startswith("FAIL")]
        if rc != 0 or fails:
            raise Unavailable("gate:%d_fail(rc=%s)" % (len(fails), rc), "; ".join(fails[:3]))
        gate_word = "pass(%d bitwise, %d tolerance, %d lines)" % (sum(l.startswith("BITWISE") for l in gate_lines), sum(l.startswith("TOLERANCE") for l in gate_lines), len(gate_lines))
    try:
        cell = F.cell(dev)[1]
    except Exception as e:                                               # noqa: BLE001 -- a fact for the report, not a gate
        cell = "unresolved(%s)" % str(e)[:60]
    props = torch.cuda.get_device_properties(dev)
    return dict(pkg=ACTIVE_PKG, arch=ARCH, binding=word, driver=cudrv.driver_version(), torch=str(torch.__version__), triton=tv,
                device=str(props.name), cc="%d.%d" % tuple(cc), kernels=kern_facts, gate=gate_word, cell=cell,
                foreign_modules=dict(F.describe().get("foreign_modules") or {}), zero_tile_residual=bool(ZERO_TILE_RESIDUAL))


def run_gate(verbose=False):
    """The package's tests/test_vectors.py main() over its sealed vectors -> (rc, verdict lines).  rc 0 = every case BITWISE or TOLERANCE."""
    G = _load_by_path(GATE_MODULE, os.path.join(pkg_dir(), "tests", "test_vectors.py"))
    import io
    buf = io.StringIO()
    with carried_binding(), contextlib.redirect_stdout(buf):            # the test builds a fresh face whose loader resolves the package names through
        rc = G.main(verbose=True)                                        # sys.modules: bound to the carried modules install() loaded, never a foreign copy
    lines = [l.strip() for l in buf.getvalue().split("\n") if l.strip()]
    if verbose:
        sys.stdout.write("\n".join(lines) + "\n")
    verdicts = [l for l in lines if l.split(" ", 1)[0] in ("BITWISE", "TOLERANCE", "FAIL", "N/A", "differs")]
    return rc, verdicts


def report():
    """The install report of this process (None before install / after a refusal)."""
    return _STATE.get("report")


def pack(weights):
    """The package's weight pack from the ten canonical tensors (``ln_in_w ln_in_b w_ag w_ap w_bg w_bp ln_out_w ln_out_b w_o w_og``): [value | gate]
    input projections stacked a-then-b as the engine's module holds them, every tensor cast to bf16 first, LN affine held fp32 (== row esm_v5_fwd's
    pack for the shared keys); the K3's folded weights (W * gamma in bf16, W @ beta in fp32) are added by the package at the first call."""
    import torch
    w = weights
    F = face()
    try:
        return F.pack_weights(w["ln_in_w"], w["ln_in_b"], torch.cat([w["w_ap"], w["w_bp"]], 0), torch.cat([w["w_ag"], w["w_bg"]], 0),
                              w["ln_out_w"], w["ln_out_b"], w["w_o"], w["w_og"])
    except ValueError as e:
        raise Unavailable("shape:%s" % str(e).split("\n")[0][:100].replace(" ", "_"))


def check(z, wp):
    """Raise Unavailable when (z, pack) is outside this row's envelope (before any launch)."""
    import torch
    if not z.is_cuda:
        raise Unavailable("device:%s(cuda only)" % z.device.type)
    if z.dtype != torch.bfloat16:
        raise Unavailable("dtype:%s!=bf16" % str(z.dtype).replace("torch.", ""))
    if z.dim() not in (3, 4) or z.shape[-2] != z.shape[-3]:
        raise Unavailable("shape:%s(z must be [N,N,C] or [B,N,N,C])" % (tuple(z.shape),))
    Cz, N = int(z.shape[-1]), int(z.shape[-2])
    if Cz != C or int(wp["C"]) != C or int(wp["CH"]) != C:
        raise Unavailable("c_z=%d,c_hidden=%d!=%d" % (Cz, int(wp["CH"]), C))
    if N < N_MIN:
        raise Unavailable("n<%d" % N_MIN)


def forward(z, outgoing, mask, wp, *, residual=True, out=None, eps=1e-5, fastsig=True):
    """Serve the op: ``z`` [N,N,256] or [B,N,N,256] bf16 CUDA (pre-LayerNorm pair), ``mask`` [N,N] / [B,N,N] (0/1, float or bool) or None, ``wp`` =
    ``pack(weights)``; returns z + update (``residual=True``, the package's epilogue) or the update alone (``False``: zero-tile residual map) in bf16,
    same shape as z.  A batch is one launch set per element (the engine's own policy).  ``fastsig`` picks the cubin (tanh.approx gate = the engine's
    lever of this line; False = the ex2/rcp form)."""
    import torch
    install()
    check(z, wp)
    if not residual and not ZERO_TILE_RESIDUAL:
        raise Unavailable("needs_residual")
    F = face()
    zs = z if z.dim() == 4 else z[None]
    zs = zs if zs.is_contiguous() else zs.contiguous()
    B, N = int(zs.shape[0]), int(zs.shape[1])
    ms = None
    if mask is not None:
        ms = mask if mask.dim() == 3 else mask[None]
        if ms.dtype == torch.bool:
            ms = ms.to(torch.float32)
        if tuple(ms.shape[-2:]) != (N, N):
            raise Unavailable("mask:%s(expected [..,%d,%d])" % (tuple(mask.shape), N, N))
    o4 = out if out is not None else torch.empty_like(zs)
    o4 = o4 if o4.dim() == 4 else o4[None]
    direction = "outgoing" if outgoing else "incoming"
    levers = dict(F.DEFAULT_V6)
    levers["sigmoid"] = bool(fastsig)
    v6 = carried_module("ef2_trimul_v6")                                 # the CARRIED K3 host module install() bound (never whatever sys.modules holds)
    if v6 is None:
        raise Unavailable("load_failed:carried_module_unbound(ef2_trimul_v6)")
    V6U = getattr(v6, "V6Unavailable", RuntimeError)
    try:
        with carried_binding(F):                                         # the face's per-call name lookups resolve to the carried modules too
            for b in range(B):
                mb = None if ms is None else ms[min(b, ms.shape[0] - 1)].contiguous()
                if residual:
                    F.forward(zs[b], direction, mb, wp, kernel="v6", out=o4[b], levers=levers, eps=eps, prebuilt=True)
                else:
                    x, n, n_pad = F.planes(zs[b], direction, mb, wp, levers=levers, eps=eps)
                    _launch_k3_update(v6, x, zs[b], wp, o4[b], n, n_pad, eps, bool(fastsig))
    except V6U as e:
        raise Unavailable("unsupported:%s" % str(e).split("\n")[0][:160].replace(" ", "_"))
    except (AttributeError, ImportError, KeyError) as e:                 # a binding-class failure (a module lacking what the face names) is a refusal by name
        raise Unavailable("load_failed:%s" % type(e).__name__, str(e).split("\n")[0][:160])
    return o4 if z.dim() == 4 else o4[0]


_ZERO = {}


def _zero_tile(device, rows, cols):
    import torch
    key = (str(device), rows, cols)
    t = _ZERO.get(key)
    if t is None:
        t = _ZERO[key] = torch.zeros((rows, cols), dtype=torch.bfloat16, device=device)
    return t


def _launch_k3_update(v6, x, z3, w, out, N, Np, eps, fastsig):
    """The package's ``launch_k3`` with ONE tensor map changed: the residual map covers a [64 x BNO] zero tile instead of z, so every residual box the
    kernel loads ([BNO channels, 64 rows, 1 column] at (c0, i0, j)) is the tile itself (c0 = i0 = j = 0) or lies outside the map and is zero-filled
    by the tensor-map unit: the epilogue adds +0.0 exactly and writes the update alone.  Everything else (checks, folded weights, maps, grid, argument
    order) is the package's launch, line for line."""
    import ctypes
    import torch
    Cz = int(z3.shape[-1])
    CH = int(x.shape[0])
    if Cz != 256 or CH != 256 or int(x.shape[1]) != Np or int(x.shape[2]) != Np or Np % 8:
        raise v6.V6Unavailable("ef2_trimul_v6: unsupported shapes x %s z %s Np %d" % (tuple(x.shape), tuple(z3.shape), Np))
    if not (x.is_contiguous() and z3.is_contiguous() and out.is_contiguous()):
        raise v6.V6Unavailable("ef2_trimul_v6: x, z, out must be contiguous")
    if x.dtype != torch.bfloat16 or z3.dtype != torch.bfloat16 or out.dtype != torch.bfloat16:
        raise v6.V6Unavailable("ef2_trimul_v6: x/z/out must be bf16")
    kern = v6.build(fastsig, True)                                           # the injected prebuilt (no compile: the key is loaded)
    v6.fold_weights(w)
    wo, wg = w["v6_wo"], w["v6_wg"]
    BT, BNO = v6._CFG["BT"], v6._CFG["BNO"]
    T = v6._TensorMap
    zt = _zero_tile(z3.device, 64, BNO)
    tm_x = T(x, dims=[Np, Np, CH], strides_bytes=[Np * 2, Np * Np * 2], box=[64, 1, CH], swizzle=128)
    tm_z = T(z3, dims=[Cz, N, N], strides_bytes=[Cz * 2, N * Cz * 2], box=[64, BT, 1], swizzle=128)
    tm_wo = T(wo, dims=[CH, Cz], strides_bytes=[CH * 2], box=[64, BNO], swizzle=128)
    tm_wg = T(wg, dims=[Cz, Cz], strides_bytes=[Cz * 2], box=[64, BNO], swizzle=128)
    tm_res = T(zt, dims=[BNO, 64, 1], strides_bytes=[BNO * 2, 64 * BNO * 2], box=[BNO, 64, 1], swizzle=64)
    tm_out = T(out, dims=[Cz, N, N], strides_bytes=[Cz * 2, N * Cz * 2], box=[BNO, 16, 1], swizzle=64)
    nJ = (N + BT - 1) // BT
    n_tiles = N * nJ
    sms = torch.cuda.get_device_properties(z3.device).multi_processor_count
    grid = (min(sms, n_tiles), 1, 1)
    stream = torch.cuda.current_stream(z3.device).cuda_stream
    kern(grid, (384, 1, 1), [tm_x, tm_z, tm_wo, tm_wg, tm_res, tm_out, w["ln_out_w"], w["ln_out_b"], w["ln_in_w"], w["ln_in_b"], w["v6_bp"], w["v6_bg"],
                             int(N), int(Np), int(nJ), int(n_tiles), float(eps), ctypes.c_uint64(0)], stream)
    return out


def describe(device=None):
    """Plain data for a LEVER / census line: package version, binding, kernels loaded, gate verdict, cell (None fields before install)."""
    rep = _STATE.get("report")
    err = _STATE.get("error")
    d = dict(row="esm_v61", pkg=ACTIVE_PKG, arch=ARCH, c=C, installed=rep is not None, error=None if err is None else err[0],
             zero_tile_residual=bool(ZERO_TILE_RESIDUAL))
    if rep is not None:
        d.update(binding=rep["binding"], kernels=sorted(rep["kernels"]), gate=rep["gate"], cell=rep["cell"], device=rep["device"])
    return d
