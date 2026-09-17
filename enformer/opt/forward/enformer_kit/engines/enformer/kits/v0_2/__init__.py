"""engines.enformer.kits.v0_2 — the Enformer exact kit: the module levers of enformer_fastkit.py (poscache, fused BN-GELU / AttentionPool
kernels, exact rel-position attention ``enformer_xattn``) plus the wrapper's CUDA-graph replay of the trunk per batch shape with the stock heads
(kits.v0.KitV0) — byte-identical to the stock ``model(x)`` at equal batch shape on H100 (the objects' class, sm_90) and A100 (the class build under
opt/forward/classes/a100_80gb, sm_80). The two extension objects are cross-built per BUILD.json and vendored; ``attach`` loads the build that runs
on the device (class_pins: the kit's own objects on the class or through their PTX on a newer device, a class kit dir off it) and refuses by
name where no build can run. Each object is held to its line of the SHA256SUMS beside it before anything loads it (``sums_refusal``, called
by ``check_files``): an object that is not the shipped one by digest is refused by name exactly as a missing one is."""
from __future__ import annotations

import hashlib
import importlib
import os
import sys

from engines.enformer.kits import class_pins
from engines.enformer.kits.v0 import KitV0, sha256_file, method_snapshot

HERE = os.path.dirname(os.path.abspath(__file__))
PREFIX = "[enformer-opt]"                                              # the package's stderr line prefix (enformer_opt._runtime.PREFIX)
SUMS = "SHA256SUMS"                                                    # beside each extension object: its `<sha256>  <file name>` line (sha256sum format)
FORWARD_DIR = os.path.dirname(class_pins.CLASSES_DIR)                  # opt/forward: bin/ (the kit's own objects) and classes/<slug>/ (class builds) lie under it
_VERDICTS = {}                                                         # object path -> None (listed, digest equal) | the refusal reason; hashed once per process
HELD = []                                                              # [(path relative to opt/forward, sha256[:16])] of every object that passed, in check order
KIT_PINS = {
    "version": "v0.2",
    "module": "enformer_fastkit.py",
    "xattn_module": "enformer_xattn.py",
    "fused_so": "enformer_fastkit_fused.so",
    "xattn_so": "enformer_xattn_ext.so",
    "build": "nvcc 13.0.88 + torch 2.13.0+cu130 headers, load_inline -O3 (BUILD.json)",
    "runtime_key": {"torch": "2.13.0+cu130", "cuda": "13.0", "cudnn": 92000},
    "sm": (9, 0),
    "ptx": (9, 0),                                                     # the vendored objects carry compute_90 PTX beside the sm_90 cubin: a newer device runs them through the driver's JIT (class_pins.ptx_serves)
    "levers": ("poscache", "fused", "graph", "xattn"),
    "graph_scope": "trunk (model(x, return_only_embeddings=True)) + eager heads (kits.v0.KitV0)",
}
BINARIES = ("fused_so", "xattn_so")


def _sums_line(sums_path: str, name: str):
    """The digest the SHA256SUMS at ``sums_path`` lists for the file ``name`` (`<sha256>  <name>` lines; `#` comments), else None."""
    want = None
    with open(sums_path, encoding="utf-8") as f:
        for ln in f:
            parts = ln.split(None, 1) if ln.strip() and not ln.startswith("#") else []
            if len(parts) == 2 and parts[1].strip() == name:
                want = parts[0].lower()
    return want


def sums_refusal(path: str, data: bytes | None = None):
    """``None`` when the extension object at ``path`` is a line of the SHA256SUMS in its own directory and its bytes re-hash to that line
    (``data``: the bytes the caller already read, hashed in place of the file). Otherwise the reason, one clause — no SHA256SUMS, not
    listed, unreadable, or the digest differing — after one ``[enformer-opt] SHA256SUMS: refused <path>: <reason>`` line on stderr;
    check_files then refuses the object by name exactly as it refuses a missing one, before anything loads it. One verdict per path per
    process."""
    p = os.path.abspath(path)
    if p in _VERDICTS:
        return _VERDICTS[p]
    inside = p.startswith(FORWARD_DIR + os.sep)
    rel = os.path.relpath(p, FORWARD_DIR).replace(os.sep, "/") if inside else p
    sums_path = os.path.join(os.path.dirname(p), SUMS)
    srel = os.path.relpath(sums_path, FORWARD_DIR).replace(os.sep, "/") if inside else sums_path
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
                digest = hashlib.sha256(data).hexdigest() if data is not None else sha256_file(p)
            except OSError as e:
                reason = f"unreadable ({type(e).__name__})"
            else:
                if digest != want:
                    reason = f"sha256 {digest[:16]} != {srel} {want[:16]}"
    _VERDICTS[p] = reason
    if reason is None:
        HELD.append((rel, digest[:16]))
    else:
        print(f"{PREFIX} {SUMS}: refused {rel}: {reason}", file=sys.stderr, flush=True)
    return reason


def check_files(so_paths: dict) -> dict:
    """Every vendored source present, then the two extension objects that will load on this device: the class build when a class kit dir serves
    the device (class_pins.active), else ``so_paths`` (the kit's own objects), each present and held to its SHA256SUMS line (sums_refusal).
    Returns the stamp fields; each ``<key>_sha256`` is the hash of what loads. Raises by name when a file is missing, when an object is not
    the shipped one by digest (FileNotFoundError, as for a missing object; nothing is loaded), or when no build runs here."""
    out = {}
    for key in ("module", "xattn_module"):
        path = os.path.join(HERE, KIT_PINS[key])
        if not os.path.exists(path):
            raise FileNotFoundError(f"kit v0.2: {path} missing")
        out[f"{key}_sha256"] = sha256_file(path)
    cp = class_pins.active(KIT_PINS, "v0_2")
    out.update(class_pins.stamp(cp))
    for key in BINARIES:
        rb = class_pins.resolve_binary(cp, KIT_PINS[key])
        path = rb["path"] if rb else (so_paths or {}).get(key)
        if not path:
            raise RuntimeError(f"kit v0.2: no object named for {key} ({KIT_PINS[key]})")
        if not os.path.exists(path):
            raise FileNotFoundError(f"kit v0.2: {path} missing")
        why = sums_refusal(path)
        if why is not None:                                            # not the shipped object by its SHA256SUMS line: refused by name before anything loads it, as a missing one is
            raise FileNotFoundError(f"kit v0.2: {path} is not the shipped object ({why}) — restore it, or rebuild it and its {SUMS} line (BUILD.json)")
        out[f"{key}_sha256"] = sha256_file(path)
        out[f"{key}_path"] = path
    return out


def attach(model, so_paths: dict, batch_shapes=(), device: str = "cuda"):
    """The lever set on ``model`` (eval mode, on ``device``) -> (KitV0, stamp). The extension objects are loaded, the module patches applied and one
    trunk CUDA graph captured now for each of ``batch_shapes``; any other batch shape is captured at its first forward (KitV0.lazy_capture)."""
    import torch
    stamp = check_files(so_paths)
    sm = tuple(torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else None
    if sm != KIT_PINS["sm"] and not stamp.get("class_pins") and not class_pins.ptx_serves(KIT_PINS, sm):
        raise RuntimeError(f"kit v0.2: device sm {sm} != the objects' {KIT_PINS['sm']} and no class kit dir or PTX serves it — cannot run here")
    xa = importlib.import_module("engines.enformer.kits.v0_2.enformer_xattn")
    sys.modules.setdefault("enformer_xattn", xa)                       # the module does `import enformer_xattn` (a top-level name): the vendored file
    xa.SO_PATH = stamp["xattn_so_path"]
    mod = importlib.import_module("engines.enformer.kits.v0_2.enformer_fastkit")
    mod.FUSED_SO = stamp["fused_so_path"]
    missing = [f for f in mod.FUSED_FUNCTIONS if not hasattr(mod.fused_ext(), f)]
    if missing:                                                        # an object built from older sources: refused by name (the levers need every entry point)
        raise RuntimeError(f"kit v0.2: {stamp['fused_so_path']} lacks {missing} — rebuild it from enformer_fastkit.py's sources (BUILD.json)")
    key = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version()}
    if key != KIT_PINS["runtime_key"]:                                 # a torch / CUDA / cuDNN runtime other than the pinned one: recorded (the package names it on its line), never a refusal — an object that cannot load on it raises by name when loaded
        stamp["runtime_key_note"] = f"runtime key {key} != the kit's key {KIT_PINS['runtime_key']}"
    levers = KIT_PINS["levers"]
    pre = method_snapshot(model)
    fk = mod.FastKit(model, levers=tuple(l for l in levers if l != "graph"), device=device)
    kit = KitV0(fk, model, tuple(int(b) for b in batch_shapes), mod.SEQ_LEN, device, pre_snapshot=pre, levers=levers)
    kit.lazy_capture = True
    stamp.update(kit="v0.2", version=KIT_PINS["version"], levers=list(levers), batch_shapes=sorted(kit.graphs),
                 build=KIT_PINS["build"], torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                 gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, module_stamp=kit.stamp())
    return kit, stamp
