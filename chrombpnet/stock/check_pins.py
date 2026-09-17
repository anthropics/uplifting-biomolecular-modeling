#!/usr/bin/env python3
"""Name this environment against the pinned chrombpnet stack (stock/PINS.json); refuse only when stock itself is not the pinned stock.

    python -I stock/check_pins.py [--stack s1|s2] [--quiet] [--no-cuda-libs]

exit 0  the installed `chrombpnet` package is the pinned one (version, entry module, checkout commit); every other pin either matches
        (`pinned`) or is NAMED as drift (`DRIFT … want …`, one line per pin, and `check_pins[drift]: …` on stderr) — drift in the
        Python version, a companion package, a CUDA library fingerprint or the /opt/torch stack is stated, never a refusal: the kit's
        levers engage on their mechanism and its byte-identity to stock is established on the pinned stack
exit 2  usage
exit 3  stock is not the pinned stock: the `chrombpnet` package is not installed, its version is not the pin, its entry module is
        missing, or the checkout's git HEAD is not the pinned commit — that changes what "stock" means, so every verb stops here

Reads only package metadata and files (no tensorflow, no torch import); standard library only, so run.sh, the configs and the
package can all call it. This file is the one place the pin check lives. --stack defaults to s1 when /opt/torch exists, else s2.
"""
import hashlib
import importlib.metadata as md
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
EXIT_USAGE, EXIT_STOCK = 2, 3
STOCK_PIN = "chrombpnet"   # the one pin whose mismatch refuses: the stock package itself


def read_pins(path=None):
    with open(path or os.path.join(HERE, "PINS.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def normalize(v):
    """'2.8.0' == '2.8.0'; local tags ('+cu124') kept; trailing '.0' components dropped on both sides."""
    if v is None:
        return None
    v = str(v).strip()
    local = ("+" + v.split("+", 1)[1]) if "+" in v else ""
    parts = v.split("+")[0].split(".")
    while len(parts) > 1 and parts[-1] == "0":
        parts.pop()
    return ".".join(parts) + local


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


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_pins(pins, installed=None):
    """Every named pin against the installed distributions. Returns ``(bad, detail)``: ``bad`` = one line per pin not met;
    ``detail`` = per pin ``{want, have, pinned}``. ``installed`` (name -> version) replaces the metadata lookup (tests)."""
    bad, detail = [], {}
    for name, want in pins["pins"].items():
        have = installed_version(name, installed)
        ok = have is not None and normalize(have) == normalize(want)
        detail[name] = {"want": want, "have": have, "pinned": ok}
        if not ok:
            bad.append(f"{name}: {'not installed' if have is None else 'installed ' + have}; want {want}")
    return bad, detail


def check_python(pins):
    have = "%d.%d.%d" % sys.version_info[:3]
    want = pins["python"]
    return ([] if have == want else [f"python: running {have}; want {want}"]), {"want": want, "have": have}


def installed_package_dir():
    """The directory of the installed chrombpnet package, from its distribution metadata (no import)."""
    try:
        dist = md.distribution("chrombpnet")
    except md.PackageNotFoundError:
        return None
    raw = dist.read_text("direct_url.json")
    info = json.loads(raw) if raw else {}
    url = info.get("url") or ""
    if url.startswith("file://"):
        root = url[len("file://"):]
        if os.path.isdir(os.path.join(root, "chrombpnet")):
            return root
    for f in dist.files or []:
        p = str(f)
        if p.startswith("chrombpnet/") and p.endswith("__init__.py"):
            return os.path.dirname(os.path.dirname(str(dist.locate_file(f))))
    return None


def check_source(pins):
    """The installed package's entry module is present; the checkout's git HEAD against the pinned commit when the package lives
    in a git checkout (the dockerfile's own install: `pip install -e chrombpnet` at that commit). Returns ``(bad, detail)``."""
    bad, detail = [], {"root": None, "entry_present": False, "git_head": None}
    root = installed_package_dir()
    detail["root"] = root
    if root is None:
        return [f"chrombpnet: not installed (want {pins['upstream']['chrombpnet']['repo']} @ {pins['upstream']['chrombpnet']['commit']})"], detail
    entry = os.path.join(root, "chrombpnet", "CHROMBPNET.py")
    detail["entry_present"] = os.path.isfile(entry)
    if not detail["entry_present"]:
        bad.append(f"chrombpnet: {entry} is missing")
    if os.path.isdir(os.path.join(root, ".git")):
        try:
            head = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
        except Exception:
            head = None
        detail["git_head"] = head
        if head and head != pins["upstream"]["chrombpnet"]["commit"]:
            bad.append(f"chrombpnet: checkout HEAD {head} != pinned commit {pins['upstream']['chrombpnet']['commit']}")
    return bad, detail


def ldconfig_paths():
    try:
        out = subprocess.run(["ldconfig", "-p"], capture_output=True, text=True, timeout=30).stdout
    except Exception:
        return {}
    paths = {}
    for line in out.splitlines():
        m = re.match(r"\s*(\S+)\s+\(.*\)\s+=>\s+(\S+)", line)
        if m:
            paths.setdefault(m.group(1), m.group(2))
    return paths


def check_cuda_libs(pins, paths=None):
    """The three CUDA libraries the pinned stack was read with: ldconfig path -> sha256[:16] == the pin."""
    bad, detail = [], {}
    paths = ldconfig_paths() if paths is None else paths
    for lib, want in pins["stack"]["cuda_libs_sha256_16"].items():
        p = paths.get(lib)
        have = sha256_file(p)[:16] if p and os.path.isfile(p) else None
        detail[lib] = {"path": p, "want": want, "have": have, "pinned": have == want}
        if have != want:
            bad.append(f"{lib}: {'not found by ldconfig' if have is None else 'sha256_16 ' + have + ' at ' + p}; want {want}")
    return bad, detail


def _version_from_file(path, pattern=r"""__version__\s*=\s*['"]([^'"]+)['"]"""):
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            m = re.search(pattern, fh.read())
        return m.group(1) if m else None
    except OSError:
        return None


def check_torch_root(pins):
    """--stack s1: /opt/torch present with the pinned torch and triton (read from their version files; nothing imported)."""
    bad, detail = [], {}
    s1 = pins["pins_s1_only"]
    root = s1["root"]
    if not os.path.isdir(root):
        return [f"{root}: absent (s1 requires torch {s1['torch']} / triton {s1['triton']} there)"], {"root": root, "present": False}
    torch_v = _version_from_file(os.path.join(root, "torch", "version.py"))
    triton_v = _version_from_file(os.path.join(root, "triton", "__init__.py"))
    if triton_v is None:
        for d in os.listdir(root):
            if d.lower().startswith("triton-") and d.endswith(".dist-info"):
                triton_v = d[len("triton-"):-len(".dist-info")]
    detail = {"root": root, "present": True, "torch": torch_v, "triton": triton_v}
    if normalize(torch_v) != normalize(s1["torch"]):
        bad.append(f"torch at {root}: {torch_v}; want {s1['torch']}")
    if normalize(triton_v) != normalize(s1["triton"]):
        bad.append(f"triton at {root}: {triton_v}; want {s1['triton']}")
    return bad, detail


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    quiet = "--quiet" in argv
    stack = None
    for i, a in enumerate(argv):
        if a == "--stack":
            stack = argv[i + 1] if i + 1 < len(argv) else None
        elif a.startswith("--stack="):
            stack = a.split("=", 1)[1]
    for a in argv:
        if a not in ("--quiet", "--no-cuda-libs", "--stack") and not a.startswith("--stack=") and a not in ("s1", "s2"):
            print(__doc__, file=sys.stderr)
            return EXIT_USAGE
    if stack not in (None, "s1", "s2"):
        print(f"check_pins: --stack must be s1 or s2, not {stack!r}", file=sys.stderr)
        return EXIT_USAGE
    pins = read_pins()
    if stack is None:
        stack = "s1" if os.path.isdir(pins["pins_s1_only"]["root"]) else "s2"

    bad_py, d_py = check_python(pins)
    bad_pins, d_pins = check_pins(pins)
    bad_src, d_src = check_source(pins)
    bad_libs, d_libs = ([], {}) if "--no-cuda-libs" in argv else check_cuda_libs(pins)
    bad_torch, d_torch = check_torch_root(pins) if stack == "s1" else ([], {"root": pins["pins_s1_only"]["root"], "required": False})
    bad_stock = [b for b in bad_pins if b.startswith(STOCK_PIN + ":")] + bad_src          # stock itself: the only refusal
    drift = bad_py + [b for b in bad_pins if not b.startswith(STOCK_PIN + ":")] + bad_libs + bad_torch

    if not quiet:
        print(f"python {d_py['have']}: {'pinned' if not bad_py else 'DRIFT'} (want {d_py['want']})")
        for name, d in d_pins.items():
            word = "pinned" if d["pinned"] else ("NOT PINNED" if name == STOCK_PIN else "DRIFT")
            print(f"{name} {d['have']}: {word} (want {d['want']})")
        print(f"chrombpnet source: entry module {'present' if d_src['entry_present'] else 'MISSING'}; root {d_src['root']}; git HEAD {d_src['git_head']}")
        for lib, d in d_libs.items():
            print(f"{lib} {d['have']}: {'pinned' if d['pinned'] else 'DRIFT'} (want {d['want']}; {d['path']})")
        print(f"stack {stack}: torch root {d_torch.get('root')} " + (f"torch {d_torch.get('torch')} triton {d_torch.get('triton')}" if stack == "s1" else "not required"))
    for b in drift:
        print(f"check_pins[drift]: {b}", file=sys.stderr)
    if drift and not quiet:
        print(f"check_pins: {len(drift)} pin(s) differ from the stack this kit was verified on (named above) — not a refusal: the levers engage "
              f"on their mechanism; byte-identity to stock is established on the pinned stack (environment/)")
    for b in bad_stock:
        print(f"check_pins[stock]: {b}", file=sys.stderr)
    return EXIT_STOCK if bad_stock else 0

if __name__ == "__main__":
    sys.exit(main())
