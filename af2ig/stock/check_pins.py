#!/usr/bin/env python3
"""check_pins.py — is this environment, checkout and weights directory the pinned stack? (stock/PINS.json against what is installed.)

    python stock/check_pins.py                     # one line per named pin; exit 3 when any pin is not met
    python stock/check_pins.py --quiet             # exit code only
    python stock/check_pins.py --gpu               # also require the CUDA-only pins (the jax CUDA plugin and the NVIDIA runtime wheels)
    python stock/check_pins.py --checkout <dir>    # the patched dl_binder_design checkout (AF2IG_DIR's parent): every relpath of PINS.json "checkout" present, the patched relpaths equal byte for byte to the kit's own copies
    python stock/check_pins.py --weights <dir>     # AF2_PARAMS: params/params_model_1_ptm.npz against PINS.json "weights"

Versions are compared after normalisation: a four-component version written as ``a.b.c (4th version field: d)`` reads as ``a.b.c.d``,
and trailing ``.0`` components are dropped on both sides. Stdlib only, no imports of the pinned packages (metadata only), so
run.sh, the config and the package can all call it; this file is the one place the pin, checkout and weights checks live.
"""
import filecmp
import hashlib
import importlib.metadata as md
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
FREEZE_4TH = re.compile(r"^(\S+) \(4th version field: (\d+)\)$")


def normalize(v):
    """'12.9.2 (4th version field: 10)' -> '12.9.2.10'; '1.11.0' -> '1.11'; local tags kept as written."""
    if v is None:
        return None
    v = str(v).strip()
    m = FREEZE_4TH.match(v)
    if m:
        v = f"{m.group(1)}.{m.group(2)}"
    parts = v.split("+")[0].split(".")
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts) + ("+" + v.split("+", 1)[1] if "+" in v else "")


def read_pins(path=None):
    with open(path or os.path.join(HERE, "PINS.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def installed_version(name, installed=None):
    if installed is not None:
        for k, v in installed.items():
            if k.lower().replace("_", "-") == name.lower().replace("_", "-"):
                return v
        return None
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def check(pins, installed=None, gpu=False):
    """Every named pin against the installed distributions. Returns ``(bad, detail)``: ``bad`` = one line per pin not met (empty = the
    pinned stack); ``detail`` = per pin ``{want, have, pinned, required}``. ``installed`` (name -> version) replaces the metadata
    lookup (tests)."""
    bad, detail = [], {}
    gpu_only = set(pins.get("pins_gpu_only") or [])
    for name, want in pins["pins"].items():
        have = installed_version(name, installed)
        required = gpu or name not in gpu_only
        ok = have is not None and normalize(have) == normalize(want)
        detail[name] = {"want": want, "have": have, "pinned": ok, "required": required}
        if not ok and required:
            bad.append(f"{name}: {'not installed' if have is None else 'installed ' + have}; want {want}")
    return bad, detail


def sha256(path):                                   # stdlib on purpose, not opt_core.gates.sha256_file: this script runs from run.sh / configs without the package
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 22), b""):
            h.update(c)
    return h.hexdigest()


def check_checkout(pins, checkout_dir, kit_dir=None):
    """The patched checkout (the directory holding af2_initial_guess/) against PINS.json "checkout": every relpath of "relpaths" present
    (the vendored archive is the git commit's; no re-hash of its bytes), no extra file beyond that set, and every
    relpath of "patched_relpaths" equal byte for byte to the kit's own patches/patched_files/ copy (a live comparison against a
    tracked file, never a stored digest). `kit_dir`, when given, is the kit's resolved location (an overridden AF2IG_OPT_KIT honored);
    it defaults to PINS.json "kit" "dir" under this script's own tree. Returns (bad, n_checked)."""
    C = pins["checkout"]
    root = os.path.join(checkout_dir, C["subtree"])
    if not os.path.isdir(root):
        return [f"{root}: not a directory"], 0
    found = set()
    for d, dirs, fs in os.walk(root):
        dirs[:] = [x for x in dirs if x != "__pycache__"]
        for f in fs:
            if f.endswith(".pyc"):
                continue
            p = os.path.join(d, f)
            if os.path.islink(p):
                continue
            found.add(os.path.relpath(p, checkout_dir))
    expected = set(C["relpaths"])
    bad = [f"{rel}: missing" for rel in sorted(expected - found)]
    bad += [f"{rel}: not a file of the pinned checkout" for rel in sorted(found - expected)]
    patched_dir = os.path.join(kit_dir or os.path.join(TREE, pins["kit"]["dir"]), "patches", "patched_files")
    for rel in C["patched_relpaths"]:
        if rel not in found:
            continue                                        # already named above (missing)
        want = os.path.join(patched_dir, rel)
        if os.path.isfile(want) and not filecmp.cmp(os.path.join(checkout_dir, rel), want, shallow=False):
            bad.append(f"{rel}: does not match the kit's patched copy")
    return bad, len(expected)


_WEIGHTS_DIGESTS = {}   # (realpath, size, mtime_ns) -> sha256: the parameter file is 373 MB, its digest costs ~1 s cold; computed once per process per file state
WEIGHTS_NOT_PINNED = "not pinned (pinned: {pin12}) — proceeding"      # a parameter file of other bytes than stock/PINS.json weights.sha256: named, and the run proceeds


def weights_verdict(pins, params_dir, digest=None):
    """AF2_PARAMS against PINS.json "weights". The user's parameter file is ALWAYS accepted when present:
    {file, name, present, sha256, bytes, pinned, pin, reason, cached_utc} — decided BY DIGEST: pinned = the sha256 of the file's full contents equals
    the pin (its size is compared too and can only agree with an equal digest); a present file of another digest is not pinned (named on its line;
    the run proceeds); a missing file is the one refusal (reason). `digest`, when given, is the caller's digest source
    `digest(path) -> (sha256_hex, cached_utc)` (the package's on-disk memo; cached_utc names when a memoised digest was computed, None when hashed now);
    without it the file is hashed here, afresh (this script's own `--weights` check)."""
    W = pins["weights"]
    p = os.path.join(params_dir, W["file"])
    name = os.path.basename(W["file"])
    if not os.path.isfile(p):
        return {"file": p, "name": name, "present": False, "sha256": None, "bytes": None, "pinned": False, "pin": W["sha256"], "reason": f"{p}: not found"}
    st = os.stat(p)
    cached_utc = None
    if digest is not None:
        have, cached_utc = digest(p)
    else:
        key = (os.path.realpath(p), st.st_size, st.st_mtime_ns)         # in-process only: one hash per file per process; the key selects, the digest decides
        if key not in _WEIGHTS_DIGESTS:
            _WEIGHTS_DIGESTS[key] = sha256(p)
        have = _WEIGHTS_DIGESTS[key]
    return {"file": p, "name": name, "present": True, "sha256": have, "bytes": st.st_size, "pinned": have == W["sha256"] and st.st_size == W["bytes"], "pin": W["sha256"], "reason": None, "cached_utc": cached_utc}


def weights_words(v):
    """The one line: `weights=<name> sha256=<12> (pinned)` or `weights sha256=<12> not pinned (pinned: <12>) — proceeding` (None when the file is missing)."""
    if not v.get("present"):
        return None
    s12 = str(v["sha256"])[:12]
    return f"weights={v['name']} sha256={s12} (pinned)" if v.get("pinned") else f"weights sha256={s12} {WEIGHTS_NOT_PINNED.format(pin12=str(v.get('pin'))[:12])}"


def check_weights(pins, params_dir):
    """(bad, path): bad only when the parameter file is missing — a present file is accepted whatever its bytes (weights_verdict names it pinned or not)."""
    v = weights_verdict(pins, params_dir)
    return ([v["reason"]] if v["reason"] else []), v["file"]


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    pins = read_pins()
    quiet, gpu = "--quiet" in argv, "--gpu" in argv
    bad, detail = [], {}
    if "--checkout" in argv:
        b, n = check_checkout(pins, argv[argv.index("--checkout") + 1])
        if not quiet:
            within_n = [x for x in b if not x.endswith(": not a file of the pinned checkout")]     # "missing" and "does not match" are of the n expected; "not a file" entries are extras beyond it
            print(f"checkout: {n - len(within_n)}/{n} pinned files present" + (f"; {len(b)} problems" if b else ""))
        bad += [f"checkout {x}" for x in b]
    elif "--weights" in argv:
        v = weights_verdict(pins, argv[argv.index("--weights") + 1])
        if not quiet:
            print(f"weights: {v['file']} " + (weights_words(v) or "not found"))
        bad += [f"weights {v['reason']}"] if v["reason"] else []
    else:
        bad, detail = check(pins, gpu=gpu)
        if not quiet:
            for name, d in detail.items():
                state = "pinned" if d["pinned"] else ("absent (cpu-only host)" if d["have"] is None and not d["required"] else "NOT PINNED")
                print(f"{name} {d['have']}: {state} (want {d['want']})")
    for b in bad:
        print(f"check_pins: {b}", file=sys.stderr)
    return 3 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
