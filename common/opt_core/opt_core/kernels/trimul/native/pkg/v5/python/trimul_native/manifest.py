"""trimul_native.manifest -- the build record of the arch-keyed cubins (``build/manifest.json``) and its verification.

Schema (one JSON object; ``units`` keyed "<unit>/<arch>")::

    {"schema": 1, "package": "trimul_native",
     "units": {"probe/sm_90a": {"unit": "probe", "arch": "sm_90a", "cubin": "sm_90a/probe.cubin", "cubin_sha256": "<hex>", "cubin_bytes": n,
                                "sources": {"csrc/dev_probe/probe.cu": "<sha256>", ...},      # every file the unit compiles (the .cu + headers it includes)
                                "nvcc": "<nvcc --version release line>", "flags": [...], "built_utc": "...",
                                "elf": {"abi_version": n, "flags": "0x..", "toolkit_version": n},   # cubin ELF header facts (what a driver checks at load)
                                "kernels": {"<entry>": {"regs": n, "smem": n, "stack": n, "spill_stores": n, "spill_loads": n, "cmem0": n,
                                                        "max_dynamic_smem": n (optional: opted in at load)}},
                                "ptxas_log": "sm_90a/probe.ptxas.txt"}}}

``verify_unit`` recomputes the cubin digest (and the source digests when the sources are carried beside the build, as in the source tree and
in a sealed package) and raises ``ManifestError(kind)``: ``absent`` | ``no_entry:<unit>/<arch>`` | ``cubin_sha256:<unit>/<arch>`` |
``source_absent:<file>`` | ``source_sha256:<file>``.
"""
import hashlib
import json
import os

__all__ = ["ManifestError", "load", "save", "entry", "verify_unit", "sha256_file", "sha256_bytes", "elf_facts"]

SCHEMA = 1


class ManifestError(RuntimeError):
    def __init__(self, kind, detail=""):
        RuntimeError.__init__(self, kind + ((" (" + detail + ")") if detail else ""))
        self.kind = kind


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def path(build_dir):
    return os.path.join(build_dir, "manifest.json")


def load(build_dir):
    p = path(build_dir)
    if not os.path.isfile(p):
        raise ManifestError("absent", p)
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def save(build_dir, man):
    man["schema"] = SCHEMA
    man.setdefault("package", "trimul_native")
    p = path(build_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(man, f, indent=1, sort_keys=True)
        f.write("\n")
    os.replace(tmp, p)


def entry(build_dir, unit, arch):
    man = load(build_dir)
    e = (man.get("units") or {}).get("%s/%s" % (unit, arch))
    if e is None:
        raise ManifestError("no_entry:%s/%s" % (unit, arch))
    return e


def root_of(build_dir):
    """The tree root the manifest's source paths are relative to: the parent of build/."""
    return os.path.abspath(os.path.join(build_dir, ".."))


def verify_unit(build_dir, unit, arch, check_sources=True, source_root=None):
    """``source_root``: the tree the manifest's source paths resolve against (default: the parent of ``build_dir``, the sealed layout)."""
    e = entry(build_dir, unit, arch)
    cub = os.path.join(build_dir, e.get("cubin") or os.path.join(arch, unit + ".cubin"))
    if not os.path.isfile(cub):
        raise ManifestError("no_entry:%s/%s" % (unit, arch), "%s absent" % cub)
    got = sha256_file(cub)
    if got != e.get("cubin_sha256"):
        raise ManifestError("cubin_sha256:%s/%s" % (unit, arch), "%s.. != %s.." % (got[:12], str(e.get("cubin_sha256"))[:12]))
    if check_sources:
        root = os.path.abspath(source_root) if source_root else root_of(build_dir)
        for rel, want in sorted((e.get("sources") or {}).items()):
            p = os.path.join(root, rel)
            if not os.path.isfile(p):
                raise ManifestError("source_absent:%s" % rel)
            if sha256_file(p) != want:
                raise ManifestError("source_sha256:%s" % rel)
    return e


def elf_facts(cubin_bytes):
    """Header facts of a cubin ELF that decide whether a driver loads it: EI_ABIVERSION (the cubin ELF layout generation), e_flags (carries
    the sm number), and the ``.note.nv.cuinfo`` note newer toolkits write (its last word is the producing toolkit's CUDA version x10, e.g. 130).
    A driver older than the CUDA line recorded here may refuse the image (``CUDA_ERROR_UNSUPPORTED_PTX_VERSION`` / ``INVALID_IMAGE``)."""
    import struct
    b = cubin_bytes
    if len(b) < 64 or b[:4] != b"\x7fELF" or b[4] != 2:
        return {"elf": False}
    abi = b[8]
    e_version = struct.unpack_from("<I", b, 20)[0]
    e_shoff = struct.unpack_from("<Q", b, 40)[0]
    e_flags = struct.unpack_from("<I", b, 48)[0]
    e_shentsize, e_shnum, e_shstrndx = struct.unpack_from("<HHH", b, 58)
    out = {"elf": True, "osabi": b[7], "abi_version": abi, "version": e_version, "flags": "0x%08x" % e_flags,
           "sm": ((e_flags >> 8) & 0xff) if abi >= 8 else (e_flags & 0xff)}
    try:                                                                    # section walk for the NVIDIA notes (best effort; absent on old toolkits)
        def shdr(i):
            o = e_shoff + i * e_shentsize
            name, typ = struct.unpack_from("<II", b, o)
            off, size = struct.unpack_from("<QQ", b, o + 24)
            return name, typ, off, size
        _, _, stroff, strsize = shdr(e_shstrndx)
        strtab = b[stroff:stroff + strsize]
        for i in range(e_shnum):
            name, typ, off, size = shdr(i)
            nm = strtab[name:strtab.find(b"\x00", name)].decode("ascii", "replace")
            if nm in (".note.nv.cuinfo", ".note.nv.tkinfo") and size >= 12:
                namesz, descsz, ntype = struct.unpack_from("<III", b, off)
                desc = b[off + 12 + ((namesz + 3) & ~3): off + 12 + ((namesz + 3) & ~3) + descsz]
                if nm == ".note.nv.cuinfo":
                    out["cuinfo"] = desc.hex()
                    if len(desc) >= 8:
                        out["cuinfo_sm"] = struct.unpack_from("<H", desc, 2)[0]
                        out["toolkit_version"] = struct.unpack_from("<I", desc, 4)[0]
                else:
                    txt = [t.decode("ascii", "replace") for t in desc[24:].split(b"\x00") if t]
                    out["tkinfo"] = txt[:4]
    except Exception:
        pass
    return out
