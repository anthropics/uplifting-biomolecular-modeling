"""kernels.triattn.triattn_native -- the sealed triangle-attention package (router + Gluon / Triton kernels + three sm_90a CUDA extensions + from
generation 11 one sm_80 CUDA extension) carried byte-for-byte under ``pkg/<version>/`` and served through ONE face, whatever the package version.

    from opt_core.kernels.triattn import triattn_native as C
    C.ACTIVE_PKG                     # "v11": the payload directory this face serves (a version swap edits this constant and drops pkg/<new>/)
    C.stack_key()                    # "torch2.13.0+cu130-cpython-312-x86_64-linux-gnu": the package's own ABI key for this interpreter
    C.stacks_built()                 # {key: [extension, ...]} read live off pkg/<ACTIVE_PKG>/triattn_pkg/prebuilt/
    rep = C.install()                # verify every prebuilt of this stack against its manifest, load, run the byte gate; raises Unavailable by name
    out = C.triangle_attention(q, k, v, bias, mask, scale)      # the package's routed forward; raises Unavailable by name, never substitutes

Contract (the provider's 'sealed-package interface'):
* payload = ``pkg/<ver>/`` = the producer's package tree unmodified except the lines named in kernels/META/triattn.json ``not_byte_identical``
  (release-tree vocabulary) and the documentation files named in ``not_carried``; nothing under it is imported at opt_core import time.
* ABI key = ``torch<torch.__version__>-<SOABI>`` (the package's ``stack_tag()``); one directory ``triattn_pkg/prebuilt/<key>/`` per key holding
  ``<ext>.so`` + ``<ext>.json`` per CUDA extension.  The ``.json`` record carries the producer's build facts plus the digests this face verifies
  before anything is loaded: ``so_sha256`` (the extension bytes), ``source_sha256`` (every file under that extension's csrc), ``loadcheck``
  ({test-vector id: sha256/16 of the expected output bytes}); a record without digests (the producer's own builds) is verified by the byte gate alone.
* never JIT inside a serving process (``TRIATTN_PKG_PREBUILT=always``): a key without a prebuilt directory is the refusal ``no_prebuilt:<pkg>@<key>``,
  a route whose extension is absent ``no_prebuilt:<pkg>@<key>:<ext>``; a digest mismatch ``digest:<ext>:<field>``; a byte-gate mismatch
  ``loadcheck_failed:<case>``; the package's own typed refusal ``unsupported:<its reason>``; a host process that already owns one of the generic
  top-level module names the package's router claims (``namespace_collision:<name>``).  Every refusal is ``Unavailable`` (its ``.kind`` = the word).
* the byte gate = the package's own test vectors (``testvectors/manifest.json`` + ``<case>.expected.pt``): inputs regenerated from (seed, shape) by the
  package's ``cases.py`` and checked against the manifest's input digests, outputs compared bitwise to the expected bytes.
"""
import hashlib
import warnings
import re
import importlib.util
import json
import os
import sys
import sysconfig
import threading
from typing import Dict, List, Optional

HONOURED = {"v11": {"v10": ((9, 0),)}}      # active payload -> {earlier generation: the device classes on which its words, routes and OUTPUT BYTES are carried unchanged}


class PkgVersion(str):
    """The active payload's version word.  It compares EQUAL to itself and to every generation the active payload honours (``HONOURED``):
    generation 11 carries generation 10's routes and output bytes unchanged, so a kit that pinned ``triattn_native.ACTIVE_PKG == "v10"`` when it
    certified keeps binding the package after the lift instead of dropping to its next preference (a slower row) -- the served bytes are the
    ones it certified.  ``str(ACTIVE_PKG)`` / formatting / hashing / json are the plain word ("v11"); ``honours(ver, cc)`` states the device
    classes exactly (the word-level gate ``triattn_native@v10`` uses it)."""
    __slots__ = ()

    def __eq__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return str.__eq__(self, other) or str(other) in HONOURED.get(str.__str__(self), {})

    def __ne__(self, other):
        if not isinstance(other, str):
            return NotImplemented
        return not self.__eq__(other)

    __hash__ = str.__hash__


ACTIVE_PKG = PkgVersion("v11")
HERE = os.path.dirname(os.path.abspath(__file__))
GENERIC_NAMES = ("kernels", "triattn", "pins", "candidate_tri", "triattn_cuda", "triattn_mw", "triattn_m1", "triattn_sm80")   # top-level names the payload's router claims
LOADCHECK_CASES = {"triattn_sm90_ext": ("s384_endstrided",), "triattn_m1_ext": ("s512_prefixvar", "s1024_prefix"), "triattn_mw_ext_g3x4": ("s3584_prefix",),
                   "triattn_sm80_ext": ("a80_s512_prefixvar", "a80_s1024_endstrided", "a80_d64_s512_none")}
# extension -> router name, payload directory, code architecture, the device cc whose routes reach it, first package generation that ships it
EXTENSIONS = {"triattn_sm90_ext":    {"route": "cuda",    "dir": "cuda",    "arch": "sm_90a", "cc": (9, 0), "since": "v10"},
              "triattn_mw_ext_g3x4": {"route": "cuda_c",  "dir": "cuda_c",  "arch": "sm_90a", "cc": (9, 0), "since": "v10"},
              "triattn_m1_ext":      {"route": "cuda_b",  "dir": "cuda_b",  "arch": "sm_90a", "cc": (9, 0), "since": "v10"},
              "triattn_sm80_ext":    {"route": "cuda_80", "dir": "cuda_80", "arch": "sm_80",  "cc": (8, 0), "since": "v11"}}
SERVED_CC = {"v10": ((9, 0), (8, 0)), "v11": ((9, 0), (8, 0))}   # device classes a generation serves (v10 on 8.0: its plain Triton member; v11 on 8.0: the sm_80 extension); other devices are refused by name in the provider face
# version words an active generation honours, per device class: generation 11 = generation 10 on cc 9.0 BY OUTPUT (the router's table name for name, the
# `cuda/` sources byte-identical, `cuda_b/` + `cuda_c/` = generation 10 + the dead-row early exit whose live AND dead rows are byte-identical, the 17
# cc-9.0 test vectors identical) -> a kit's `triattn_native@v10` is served by v11 on 9.0; on 8.0 it is refused by name (v10 served 8.0 with its plain
# Triton member, v11 serves the sm_80 extension: a different kernel, so the version word does not carry)
_LOCK = threading.Lock()
_STATE: Dict[str, object] = {}


class Unavailable(RuntimeError):
    def __init__(self, kind: str, detail: str = ""):
        super().__init__(f"triattn_native@{ACTIVE_PKG}: {kind}" + (f" ({detail})" if detail else ""))
        self.kind = kind


def pkg_dir(pkg: str = "") -> str:
    return os.path.join(HERE, "pkg", pkg or ACTIVE_PKG)


def payload_dir(pkg: str = "") -> str:
    return os.path.join(pkg_dir(pkg), "triattn_pkg")


def stack_key() -> str:
    import torch
    return "torch%s-%s" % (torch.__version__, sysconfig.get_config_var("SOABI"))


_ARCH_SUFFIX_RE = re.compile(r"-sm_?\d+a?$")


def provider_key_to_pkg_key(key: str) -> str:
    """The provider's keys carry an arch suffix ('-sm90' on cc 9.0, '-sm80' on cc 8.0, any '-sm<NN>[a]'); the package's keys do not."""
    return _ARCH_SUFFIX_RE.sub("", key) if key else key


def _read_json(path: str):
    with open(path, encoding="utf-8") as fh:                      # closed deterministically (no ResourceWarning under -W error / dev mode)
        return json.load(fh)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


_DIR_CACHE: Dict[str, Dict[str, List[str]]] = {}      # pkg -> {key: [extension, ...]}: the prebuilt tree is immutable in a serving process (read once)
_GLUON: Dict[str, Optional[bool]] = {}


def refresh() -> None:
    """Forget the cached prebuilt listing (after build_prebuilt.py added a key in THIS process)."""
    _DIR_CACHE.clear(); _GLUON.clear()


def stacks_built(pkg: str = "") -> Dict[str, List[str]]:
    hit = _DIR_CACHE.get(pkg or ACTIVE_PKG)
    if hit is not None:
        return hit
    root = os.path.join(payload_dir(pkg), "prebuilt")
    out = {}
    if os.path.isdir(root):
        for key in sorted(os.listdir(root)):
            d = os.path.join(root, key)
            if os.path.isdir(d):
                out[key] = sorted(f[:-3] for f in os.listdir(d) if f.endswith(".so"))
    _DIR_CACHE[pkg or ACTIVE_PKG] = out
    return out


def record(key: str, ext: str, pkg: str = "") -> Optional[dict]:
    key = provider_key_to_pkg_key(key)
    p = os.path.join(payload_dir(pkg), "prebuilt", key, ext + ".json")
    return _read_json(p) if os.path.isfile(p) else None


def extensions() -> Dict[str, dict]:
    """extension name -> {route, dir, arch, cc, since} for the extensions the ACTIVE generation ships (EXTENSIONS restates the payload's table; nothing imported)."""
    return {e: dict(d) for e, d in EXTENSIONS.items() if _pkg_rank(ACTIVE_PKG) >= _pkg_rank(d["since"])}


def source_sha256(ext: str, pkg: str = "") -> Dict[str, str]:
    d = os.path.join(payload_dir(pkg), extensions()[ext]["dir"], "csrc")
    out = {}
    for r, _, fs in os.walk(d):
        for f in sorted(fs):
            p = os.path.join(r, f)
            out[os.path.relpath(p, d)] = _sha256_file(p)
    return dict(sorted(out.items()))


SM90 = (9, 0)
SM80 = (8, 0)


def extensions_for_cc(cc) -> List[str]:
    """The extensions whose routes a device of this cc reaches in the active generation."""
    c = _cc(cc)
    return sorted(e for e, d in EXTENSIONS.items() if tuple(d["cc"]) == c and _pkg_rank(ACTIVE_PKG) >= _pkg_rank(d["since"]))


def honours(ver: str, cc=None):
    """(ok, why): whether the active payload serves the version word triattn_native@<ver> on device class `cc` ((major, minor), 'M.m' or None = any honoured)."""
    if str.__eq__(ACTIVE_PKG, str(ver)):                      # the identical word (PkgVersion equality also admits honoured generations; the cc gate below decides those)
        return True, "active"
    on = HONOURED.get(ACTIVE_PKG, {}).get(ver)
    if on is None:
        return False, "%s!=active:%s" % (ver, ACTIVE_PKG)
    if cc is not None and _cc(cc) not in on:
        return False, "%s!=active:%s on cc %d.%d (%s served this device with another kernel)" % ((ver, ACTIVE_PKG) + _cc(cc) + (ver,))
    return True, "served by %s: generation %s's routes and outputs carried unchanged on cc %s" % (ACTIVE_PKG, ver.lstrip("v"), "/".join("%d.%d" % t for t in on))


def _pkg_rank(pkg: str) -> int:
    try:
        return int(str(pkg).lstrip("v").split(".")[0])
    except ValueError:
        return 0


def _cc(cc) -> tuple:
    if cc is None:
        return SM90
    if isinstance(cc, str):
        a, _, b = cc.partition(".")
        return (int(a), int(b or 0))
    return (int(cc[0]), int(cc[1]))


def route_kind(dtype: str, head_dim: int, n: int, strided: bool = False, cc=None) -> str:
    """'ext:<name>' for a CUDA extension route, 'gluon' for the persistent Gluon kernel (bf16 D32 S<512 contiguous; fp16 or D16 up to 640),
    'triton' for the plain Triton kernel (D64 / D128, fp16 or D16 above 640; and EVERY call off cc 9.0: the package's any-GPU member) -- the
    router's table restated without importing it.  Generation >= 11 on cc 8.0: bf16 D16 / D32 / D64 (any S, either layout) -> 'ext:triattn_sm80_ext'."""
    ext = route_extension(dtype, head_dim, n, strided, cc)
    if _cc(cc) != SM90:
        return ("ext:" + ext) if ext is not None else "triton"
    if ext is not None:
        return "ext:" + ext
    if dtype == "bf16" and head_dim == 32:
        return "gluon"
    if head_dim in (64, 128):
        return "triton"
    return "gluon" if n <= 640 else "triton"


def route_extension(dtype: str, head_dim: int, n: int, strided: bool = False, cc=None) -> Optional[str]:
    """The CUDA extension the payload's router takes for (dtype, head_dim, S=n, layout), or None for a Gluon / Triton route -- the router's table
    restated for admittance checks without importing it (cc 9.0, bf16 D32 square: S<512 contiguous k13 | S<512 strided cuda | 512..3072 and >4096
    cuda_b | 3072<S<=4096 cuda_c; fp16 / D16 / D64 / D128: Gluon / Triton routes; cc 8.0 from generation 11: bf16 D16 / D32 / D64 -> cuda_80 = triattn_sm80_ext
    at any S and either layout, fp16 / D128 the plain Triton member; generation 10 off cc 9.0: the plain Triton member for everything, no extension)."""
    if _cc(cc) == SM80 and "triattn_sm80_ext" in extensions_for_cc(SM80):
        return "triattn_sm80_ext" if (dtype == "bf16" and head_dim in (16, 32, 64)) else None
    if _cc(cc) != SM90 or dtype != "bf16" or head_dim != 32:
        return None
    if n < 512:
        return "triattn_sm90_ext" if strided else None
    if 3072 < n <= 4096:
        return "triattn_mw_ext_g3x4"
    return "triattn_m1_ext"


def admits(dtype: str, head_dim: int, n: int, *, key: Optional[str] = None, strided: bool = False, triton_gluon: Optional[bool] = None, cc=None):
    """(ok, why) without importing the payload: dtype / head_dim domain, then the prebuilt this stack needs for the route.  ``cc``: the device's
    compute capability (default 9.0); off 9.0 every call takes the package's plain Triton member (no extension, no prebuilt) -- it needs a
    triton whose Gluon dialect imports (the member's module imports it at load), so ``triton_gluon=False`` refuses it by name there too."""
    if _cc(cc) not in SERVED_CC.get(ACTIVE_PKG, (SM90,)):
        return False, "cc %d.%d: no member of %s for this device" % (_cc(cc) + (ACTIVE_PKG,))
    if dtype not in ("bf16", "fp16"):
        return False, "dtype:" + dtype
    if head_dim not in (16, 32, 64, 128):
        return False, "head_dim:D%d" % head_dim
    ext = route_extension(dtype, head_dim, n, strided, cc)
    if ext is None:
        kind = route_kind(dtype, head_dim, n, strided, cc)
        if triton_gluon is False and (kind == "gluon" or _cc(cc) != SM90):
            return False, "needs_triton_gluon"
        return True, kind + " route" + ("" if _cc(cc) == SM90 else " (cc %d.%d: the package's any-GPU Triton member)" % _cc(cc))
    key = provider_key_to_pkg_key(key) if key else None
    if key is None:
        return True, "route " + ext + " (stack not named)"
    built = stacks_built().get(key)
    if built is None:
        return False, "no_prebuilt:%s@%s" % (ACTIVE_PKG, key)
    if ext not in built:
        return False, "no_prebuilt:%s@%s:%s" % (ACTIVE_PKG, key, ext)
    return True, "prebuilt " + ext


def _module():
    """The payload imported ONCE under a private dotted name (its relative imports work; its router later puts its own directories on sys.path)."""
    M = _STATE.get("module")
    if M is not None:
        return M
    with _LOCK:
        M = _STATE.get("module")
        if M is not None:
            return M
        root = payload_dir()
        for name in GENERIC_NAMES:                                   # a host module of the same top-level name that is not the payload's own
            mod = sys.modules.get(name)
            if mod is not None and not str(getattr(mod, "__file__", "") or "").startswith(pkg_dir()):
                raise Unavailable("namespace_collision:" + name, getattr(mod, "__file__", "?"))
        os.environ["TRIATTN_PKG_PREBUILT"] = "always"                # never a JIT build inside a serving process
        name = __name__ + "._payload_" + ACTIVE_PKG
        spec = importlib.util.spec_from_file_location(name, os.path.join(root, "__init__.py"), submodule_search_locations=[root])
        M = importlib.util.module_from_spec(spec)
        sys.modules[name] = M
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", ResourceWarning)      # the sealed payload's own record reads (json.load(open(...))) are left as shipped; their warning is filtered here, not patched there
                spec.loader.exec_module(M)
        except Exception as e:                                       # torch / triton absent, or the payload's own import-time refusal
            sys.modules.pop(name, None)
            raise Unavailable("import:" + type(e).__name__, str(e)[:200])
        _STATE["module"] = M
        return M


def verify_manifests(key: Optional[str] = None) -> Dict[str, dict]:
    """Digest checks of every extension built for `key` (default: this interpreter's): so bytes and csrc bytes against the record where the record
    carries digests.  Raises Unavailable('digest:<ext>:<field>')."""
    key = provider_key_to_pkg_key(key) if key else stack_key()          # the load path takes the package's own key; a provider-spelt key ('-sm80' / '-sm90') normalizes the same way as admits()
    built = stacks_built().get(key)
    if built is None:
        raise Unavailable("no_prebuilt:%s@%s" % (ACTIVE_PKG, key))
    rep = {}
    for ext in built:
        rec = record(key, ext) or {}
        so = os.path.join(payload_dir(), "prebuilt", key, ext + ".so")
        from opt_core.gates import binary_refusal  # noqa: PLC0415
        why = binary_refusal(so)
        if why:
            raise Unavailable("digest:%s:SHA256SUMS" % ext, why)
        sha = _sha256_file(so)
        if "so_sha256" in rec and rec["so_sha256"] != sha:
            raise Unavailable("digest:%s:so_sha256" % ext, "%s != %s" % (sha[:16], rec["so_sha256"][:16]))
        if "source_sha256" in rec and rec["source_sha256"] != source_sha256(ext):
            raise Unavailable("digest:%s:source_sha256" % ext)
        rep[ext] = {"so_sha256": sha, "digests_in_record": sorted(k for k in ("so_sha256", "source_sha256", "loadcheck") if k in rec), "nvcc": rec.get("nvcc"), "cutlass_tag": rec.get("cutlass_tag")}
    return rep


def loadcheck(cases=None, key: Optional[str] = None) -> Dict[str, dict]:
    """The package's byte gate on this device for the extensions of `key`: regenerate each case's inputs (digest-checked), run the routed forward,
    compare output bytes to testvectors/<case>.expected.pt.  Raises Unavailable('loadcheck_failed:<case>') on any difference."""
    import torch
    M = _module()
    key = provider_key_to_pkg_key(key) if key else stack_key()
    dev_cc = _device_cc() if torch.cuda.is_available() else None
    built = [e for e in stacks_built().get(key, []) if dev_cc is None or tuple(EXTENSIONS.get(e, {}).get("cc", dev_cc)) == dev_cc]   # byte-gate the extensions THIS device executes
    tv = os.path.join(pkg_dir(), "testvectors")
    man = _read_json(os.path.join(tv, "manifest.json"))
    by_id = {c["id"]: c for c in man["cases"]}
    C = _STATE.get("cases")
    if C is None:
        cname = __name__ + "._cases_" + ACTIVE_PKG
        spec = importlib.util.spec_from_file_location(cname, os.path.join(tv, "cases.py"))
        C = importlib.util.module_from_spec(spec); sys.modules[cname] = C          # dataclasses resolve the defining module through sys.modules
        spec.loader.exec_module(C); _STATE["cases"] = C
    wanted = cases if cases is not None else [c for ext in built for c in LOADCHECK_CASES.get(ext, ())]
    rep = {}
    for cid in wanted:
        case = C.BY_ID[cid]; rec = by_id[cid]
        t = C.make_inputs(case, device="cuda")
        dig = {nme: C.tensor_digest(x) for nme, x in t.items() if hasattr(x, "dtype")}
        bad = [nme for nme, want in rec.get("inputs_sha256", {}).items() if want is not None and nme in dig and dig[nme] != want]
        if bad:
            raise Unavailable("loadcheck_failed:%s" % cid, "input digests differ on this stack: %s" % ",".join(bad))
        out = M.triangle_attention(t["q"], t["k"], t["v"], t["bias"], mask=t.get("mask"))
        torch.cuda.synchronize()
        ep = os.path.join(tv, cid + ".expected.pt")
        if not os.path.isfile(ep):                                   # a case whose expected vector the payload does not record: the gate cannot pass, by name
            raise Unavailable("loadcheck_failed:%s" % cid, "expected vector not recorded in the payload")
        exp = torch.load(ep, map_location="cpu")
        rows = rec.get("expected", {}).get("rows", "all")
        got_t = out.detach().contiguous().cpu() if rows == "all" else out.detach()[:, :int(rows)].contiguous().cpu()
        same = got_t.shape == exp.shape and got_t.dtype == exp.dtype and bool(torch.equal(got_t, exp))
        sha16 = C.tensor_digest(got_t)[:16]
        rep[cid] = {"route": M.route(t["q"], t["k"], t["v"], t["bias"], t.get("mask")), "bitwise": same, "sha256_16": sha16}
        if not same:
            raise Unavailable("loadcheck_failed:%s" % cid, "output bytes differ from the package's expected vector")
    return rep


def _device_cc() -> tuple:
    import torch
    return tuple(torch.cuda.get_device_capability()) if torch.cuda.is_available() else (0, 0)


def selfcheck_triton_member() -> Dict[str, dict]:
    """Off cc 9.0 the package serves every call with its plain Triton member, for which it ships no expected vectors (those are the sm_90
    extensions'): the load gate there is a determinism + finiteness self-check of the member on this device (two calls, identical bytes)."""
    import torch
    M = _module()
    g = torch.Generator(device="cuda").manual_seed(7)
    S, H, D = 384, 4, 32
    q, k, v = (torch.randn((1, S, H, S, D), device="cuda", dtype=torch.bfloat16, generator=g) for _ in range(3))
    bias = torch.randn((1, 1, H, S, S), device="cuda", dtype=torch.bfloat16, generator=g).float()
    lens = torch.randint(S // 2, S + 1, (1, S), device="cuda", generator=g)
    mask = (torch.arange(S, device="cuda")[None, None, :] < lens[:, :, None])[:, :, None, None, :].contiguous()
    a = M.triangle_attention(q, k, v, bias, mask=mask); b = M.triangle_attention(q, k, v, bias, mask=mask); torch.cuda.synchronize()
    same = bool(torch.equal(a, b)); fin = bool(torch.isfinite(a).all().item())
    rep = {"selfcheck_s384_mask": {"route": M.route(q, k, v, bias, mask), "bitwise": same and fin, "deterministic": same, "finite": fin}}
    if not (same and fin):
        raise Unavailable("loadcheck_failed:selfcheck_s384_mask", "the Triton member is not deterministic / finite on this device")
    return rep


def _verdict_dir() -> Optional[str]:
    """Where a passed byte gate is remembered across processes: <MODEL_OPT_JIT_ROOT>[/<MODEL_OPT_STACK_KEY>]/triattn (the kit's JIT root, beside its
    Triton / extension caches), else beside TRITON_CACHE_DIR, else none (the gate then runs once per process, ~0.5 s of GPU work)."""
    root = os.environ.get("MODEL_OPT_JIT_ROOT")
    if root:
        base = os.path.join(root, os.environ["MODEL_OPT_STACK_KEY"]) if os.environ.get("MODEL_OPT_STACK_KEY") else root
    elif os.environ.get("TRITON_CACHE_DIR"):
        base = os.path.dirname(os.path.abspath(os.environ["TRITON_CACHE_DIR"]))
    else:
        return None
    d = os.path.join(base, "triattn")
    try:
        os.makedirs(d, exist_ok=True)
        return d
    except OSError:
        return None


def _verdict_key(key: str, cc: tuple, manifests: Dict[str, dict]) -> str:
    """sha256 over (payload version, ABI key, device cc + name, driver, every served extension's so digest, the test-vector manifest digest):
    a process whose tuple already passed the byte gate on this machine may skip re-running it (the digests are still verified every process)."""
    import hashlib, torch
    tv = os.path.join(pkg_dir(), "testvectors", "manifest.json")
    parts = [ACTIVE_PKG, key, "%d.%d" % cc, torch.cuda.get_device_name() if torch.cuda.is_available() else "-", str(getattr(torch.version, "cuda", "")),
             str(torch.cuda.driver_version()) if hasattr(torch.cuda, "driver_version") else "-", _sha256_file(tv) if os.path.isfile(tv) else "-"]
    parts += ["%s=%s" % (e, m.get("so_sha256", "")) for e, m in sorted(manifests.items())]
    return hashlib.sha256("\0".join(parts).encode()).hexdigest()


def _gated_by_vectors(cc: tuple, key: str) -> bool:
    """Whether install() byte-gates this (device cc, stack key) with the payload's test vectors: always on cc 9.0; on cc 8.0 when the sm_80
    extension is built for the key (otherwise the plain Triton member's self-check is the gate, as generation 10)."""
    return tuple(cc) == SM90 or any(tuple(EXTENSIONS.get(e, {}).get("cc", ())) == tuple(cc) for e in stacks_built().get(key, []))


def install(check: bool = True) -> dict:
    """Verify + load + byte-gate the extensions of this interpreter's stack (cc 9.0), or self-check the Triton member (other devices); cached per
    process, and the PASSED verdict cached on disk (``_verdict_dir()``, keyed by ``_verdict_key``) so a later process on the same machine / stack /
    payload verifies digests only.  Returns the report; raises Unavailable by name.  Kits call ``warm()`` at activation so none of this lands
    inside the first served call."""
    import time as _time
    rep = _STATE.get("install")
    if rep is not None:
        return rep
    t0 = _time.perf_counter()
    key = stack_key()
    cc = _device_cc()
    rep = {"pkg": ACTIVE_PKG, "stack_key": key, "cc": "%d.%d" % cc, "manifests": verify_manifests(key)}
    M = _module()
    vdir = _verdict_dir(); vkey = _verdict_key(key, cc, rep["manifests"]) if check else None
    vpath = os.path.join(vdir, "gate-%s.json" % vkey[:24]) if (vdir and vkey) else None
    if not check:
        rep["loadcheck"] = "skipped"
    elif vpath and os.path.isfile(vpath):
        try:
            rep["loadcheck"] = _read_json(vpath); rep["verdict_cache"] = "hit"
        except (OSError, ValueError):
            rep["loadcheck"] = loadcheck(key=key) if _gated_by_vectors(cc, key) else selfcheck_triton_member(); rep["verdict_cache"] = "unreadable"
    elif _gated_by_vectors(cc, key):                                     # cc 9.0 (the sm_90a three), or cc 8.0 with the sm_80 extension built for this key: the package's vectors of this cc
        rep["loadcheck"] = loadcheck(key=key); rep["verdict_cache"] = "miss" if vpath else "none"
    else:
        try:
            rep["loadcheck"] = selfcheck_triton_member()                # the sm_90 extensions' vectors do not apply off 9.0
        except ImportError as e:                                         # the member's module imports the Gluon dialect at load
            raise Unavailable("needs_triton_gluon:" + str(e)[:120])
        except NotImplementedError as e:
            raise Unavailable("unsupported:" + str(e)[:160])
        rep["verdict_cache"] = "miss" if vpath else "none"
    if vpath and rep.get("verdict_cache") == "miss" and isinstance(rep.get("loadcheck"), dict):
        try:                                                             # remember the passed gate for later processes (atomic write)
            tmp = vpath + ".%d.tmp" % os.getpid()
            with open(tmp, "w", encoding="utf-8") as fh:
                json.dump(rep["loadcheck"], fh)
            os.replace(tmp, vpath)
        except OSError:
            rep["verdict_cache"] = "unwritable"
    rep["install_s"] = round(_time.perf_counter() - t0, 3)
    rep["prebuilt_status"] = {k: (v if not isinstance(v, dict) else {kk: str(vv)[:60] for kk, vv in v.items()}) for k, v in M.prebuilt_status().get("resolved", {}).items()}
    _STATE["install"] = rep
    return rep


def warm(shapes=(), dtype=None) -> dict:
    """Activation-time hook: import the payload, verify digests, run (or read the cached verdict of) the byte gate, and optionally serve one call
    per (H, S, D) in ``shapes`` so the Triton member's JIT and the extensions' first-launch costs land here, not in the first model call.
    Returns install()'s report plus per-shape first-call seconds."""
    import time as _time, torch
    rep = dict(install(check=True)); firsts = {}
    dt = dtype or torch.bfloat16
    for (H, S, D) in shapes:
        g = torch.Generator(device="cuda").manual_seed(S)
        q, k, v = (torch.randn((1, min(S, 64), H, S, D), device="cuda", dtype=dt, generator=g) for _ in range(3))
        bias = torch.randn((1, 1, H, S, S), device="cuda", dtype=dt, generator=g).float()
        torch.cuda.synchronize(); t0 = _time.perf_counter()
        triangle_attention(q, k, v, bias, None); torch.cuda.synchronize()
        firsts["H%d_S%d_D%d" % (H, S, D)] = round(_time.perf_counter() - t0, 3)
    rep["first_call_s"] = firsts
    return rep


def route(q, k, v, bias, mask=None) -> str:
    M = _module()
    try:
        return M.route(q, k, v, bias, mask)
    except NotImplementedError as e:
        raise Unavailable("unsupported:" + str(e)[:120])


def triangle_attention(q, k, v, bias, mask=None, scale=None):
    M = _module()
    if "install" not in _STATE:
        install(check=True)
    try:
        return M.triangle_attention(q, k, v, bias, mask=mask, scale=scale)
    except NotImplementedError as e:                                 # the package's typed refusal (Unsupported is a NotImplementedError)
        raise Unavailable("unsupported:" + str(e)[:160])
    except ImportError as e:                                         # a Gluon / Triton route on a stack whose triton has no experimental.gluon (torch 2.7.x): by name
        raise Unavailable("needs_triton_gluon:" + str(e)[:120])


def triton_gluon_available() -> Optional[bool]:
    """True / False: this interpreter's triton has / lacks triton.experimental.gluon (the package's k13 route); None: no triton.  Probed once."""
    if "v" not in _GLUON:
        _GLUON["v"] = _triton_gluon_probe()
    return _GLUON["v"]


def _triton_gluon_probe() -> Optional[bool]:
    """Whether this interpreter's triton carries the Gluon dialect the package's k13 / k12 routes need (torch >= 2.10 stacks); no import of triton itself."""
    try:
        if importlib.util.find_spec("triton") is None:
            return None                                              # no triton at all (a CPU-only interpreter): unknown, not a refusal
        return importlib.util.find_spec("triton.experimental") is not None and importlib.util.find_spec("triton.experimental.gluon") is not None
    except (ImportError, ValueError):
        return False
