"""engines.enformer.kits.class_pins — the reader of the tree's class kit dirs: the kit's extension objects are cross-built for sm_90 (H100, the
class). A device off the class runs THE SAME KIT SOURCES with a build that runs on it: a class kit dir under ``opt/forward/classes/<slug>/``
(CLASS_PINS.json + the rebuilt objects; found by the device's capability — its own, or the same-major build of a lower minor, since a cubin runs on
every later minor of its major), or, on a device newer than the class, the kit's own objects through the PTX they carry (``ptx_serves``); where
neither exists the kit cannot run there and says so."""
from __future__ import annotations

import hashlib
import json
import os

FILE = "CLASS_PINS.json"
CLASSES_DIR = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "..", "classes"))   # opt/forward/classes/<slug>/: the tree's own frozen class kit dirs


def tree_class_dir(sm) -> str | None:
    """The tree's frozen class kit dir whose build runs on a device of capability ``sm`` (opt/forward/classes/<slug>/): the dir whose
    CLASS_PINS.json `sm` IS the device's, else the same-major dir of the highest minor below it (a cubin runs on every later minor of its
    major: an sm_80 build serves 8.6 / 8.9), else None."""
    if sm is None or not os.path.isdir(CLASSES_DIR):
        return None
    sm = tuple(sm)
    found = {}
    for slug in sorted(os.listdir(CLASSES_DIR)):
        try:
            with open(os.path.join(CLASSES_DIR, slug, FILE)) as fh:
                csm = tuple(int(x) for x in (json.load(fh).get("sm") or ()))
        except (OSError, ValueError, TypeError):
            continue
        if len(csm) == 2:
            found.setdefault(csm, os.path.join(CLASSES_DIR, slug))
    if sm in found:
        return found[sm]
    compat = sorted(c for c in found if c[0] == sm[0] and c[1] < sm[1])
    return found[compat[-1]] if compat else None


def runs_on(build_sm, sm) -> bool:
    """Whether a cubin built for ``build_sm`` runs on a device of capability ``sm``: the same capability, or the same major at a later minor."""
    b, d = tuple(build_sm), tuple(sm)
    return b == d or (b[0] == d[0] and b[1] < d[1])


def ptx_serves(kit_pins: dict, sm) -> bool:
    """Whether the kit's own objects serve a device of capability ``sm`` off the class through the PTX they carry (``kit_pins["ptx"]``, the
    virtual architecture): the driver compiles it for any device of a later capability."""
    ptx = kit_pins.get("ptx")
    return bool(ptx) and sm is not None and tuple(sm) > tuple(ptx)


def sha256_file(p: str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def device_sm():
    try:
        import torch
        return tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    except Exception:                                                  # no torch / no device (the package's CPU tests run this way): the caller decides
        return None


def read_dir(d: str, kit_name: str) -> dict:
    """Read a class kit dir of the tree: its CLASS_PINS.json for the class table (slug, sm, binaries, sources), every named binary present,
    and the dir declared for this kit."""
    cp_path = os.path.join(d, FILE)
    if not os.path.exists(cp_path):
        raise RuntimeError(f"class pins: {d} carries no {FILE}")
    cp = json.load(open(cp_path))
    slug = cp.get("slug") or cp.get("class")
    sm = cp.get("sm")
    if not slug or not isinstance(sm, (list, tuple)) or len(sm) != 2:
        raise RuntimeError(f"class pins: {cp_path} carries no slug / sm")
    bins = cp.get("binaries") or {}
    if not isinstance(bins, dict) or not bins or any(not isinstance(v, dict) or not v.get("sha256") for v in bins.values()):
        raise RuntimeError(f"class pins: {cp_path} names no binaries with sha256")
    for name in bins:
        if not os.path.exists(os.path.join(d, name)):
            raise RuntimeError(f"class pins: binary {name} missing from {d}")
    kit_declared = cp.get("kit")
    if kit_declared and kit_declared != kit_name:
        raise RuntimeError(f"class pins {slug}: the dir is kit {kit_declared!r}, not {kit_name!r}")
    return {"slug": slug, "sm": tuple(int(x) for x in sm), "dir": d, "class_pins_sha256": sha256_file(cp_path),
            "binaries": {n: {"path": os.path.join(d, n), "sha256": b["sha256"], "sass_targets": b.get("sass_targets")} for n, b in bins.items()},
            "build_image": cp.get("build_image"), "driver": cp.get("driver"), "device": cp.get("device") or cp.get("device_parts")}


def active(kit_pins: dict, kit_name: str) -> dict | None:
    """The class pins in force on this device (the class kit dir whose build runs on it; ``serves`` set when that is a same-major build of a
    lower minor), or None when the kit's own objects serve the device (the class itself, or a newer device through their PTX). Raises by name
    when no build in the tree can run on the device."""
    sm = device_sm()
    if sm is None or sm == tuple(kit_pins["sm"]):
        return None
    d = tree_class_dir(sm)
    if not d:
        if ptx_serves(kit_pins, sm):                                 # a device newer than the class: the kit's own objects through their PTX (the caller names the architecture on its line)
            return None
        raise RuntimeError(f"kit {kit_name}: no build in this tree runs on a device of capability sm_{sm[0]}{sm[1]} — the kit's objects are "
                           f"sm_{kit_pins['sm'][0]}{kit_pins['sm'][1]} cubins" + (f" with compute_{kit_pins['ptx'][0]}{kit_pins['ptx'][1]} PTX (devices of a later capability)" if kit_pins.get("ptx") else "")
                           + f", the class kit dirs under {CLASSES_DIR} serve their own major from their minor up; cannot run here")
    cp = read_dir(d, kit_name)
    if not runs_on(cp["sm"], sm):
        raise RuntimeError(f"class pins {cp['slug']}: the dir's build sm {cp['sm']} does not run on the device's {sm} — cannot run here")
    if cp["sm"] != sm:                                               # a same-major build serving a later minor: named by the caller on its line
        cp["serves"] = f"sm_{sm[0]}{sm[1]} served by the sm_{cp['sm'][0]}{cp['sm'][1]} class build (same-major binary compatibility)"
    return cp


def resolve_binary(cp: dict | None, filename: str) -> dict | None:
    """The class build of ``filename`` ({path, sha256}) or None on the class; a binary the class dir lacks refuses by name."""
    if cp is None:
        return None
    b = cp["binaries"].get(filename) or cp["binaries"].get(os.path.basename(filename))
    if b is None:
        raise RuntimeError(f"class pins {cp['slug']}: no binary {filename} in the class dir's CLASS_PINS.json — cannot run here")
    return {"path": b["path"], "sha256": b["sha256"]}


def stamp(cp: dict | None) -> dict:
    if cp is None:
        return {"class_pins": None}
    return {"class_pins": {k: v for k, v in cp.items() if k != "binaries"} | {"binaries": {n: b["sha256"] for n, b in cp["binaries"].items()}}}
