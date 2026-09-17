#!/usr/bin/env python3
"""Refuse unless the pinned upstream is installed: chai_lab 0.6.1 with the files of the pinned wheel (stock/PINS.json "upstream").

usage: python -I stock/check_pins.py [--quiet] [--strict-stack]      exit 0 = pinned; 3 = not (one line per refusal on stderr)

Reads only package metadata and file bytes (no torch, no chai_lab import): the installed distribution's version must be the pin and
every ``chai_lab/*.py`` file the wheel's RECORD lists must be on disk with the RECORD's sha256 — so a PyPI install, an install of the
wheel in stock/ and a git install at the tag all pass (same bytes), and an editable checkout at another commit, a patched
site-packages or another version is refused.

The torch / CUDA stack is a RECORD, not a gate (``stack_verdict``): the tree's pinned stacks are the builds listed in PINS "stacks"
(torch 2.13.0+cu130 / CUDA 13.0 — outside the range the 0.6.1 wheel declares, PINS "declared_ranges"; chai_lab 0.6.1 is installed on it
regardless and runs unmodified). A torch that IS one of those builds — same version and same CUDA build — is the pinned stack and nothing
is printed; any other torch prints ONE line to stderr, ``STACK not pinned: torch <version>+<cuda> (pinned: torch 2.13.0+cu130)``, and the
check passes — the run proceeds with that record (the package prints it again as a NOTE line at activation and carries it in its
activation report: stack_pinned / stack_line). Strict switch: ``--strict-stack`` on this command line or ``CHAI1_OPT_STRICT_STACK=1`` in
the environment makes any other stack a refusal by name (exit 3) — for a caller that requires the pinned stack rather than a record of
another one. Standard library only, so run.sh, the configs and the package can all call it; this file is the one place the pin check and
the stack verdict live.
"""
import base64
import hashlib
import importlib.metadata as md
import json
import os
import re
import sys
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
ENV_STRICT_STACK = "CHAI1_OPT_STRICT_STACK"            # =1: a torch/CUDA stack other than the pinned one is a refusal (exit 3); unset or 0: it is a named record and the run proceeds
STRICT_FLAG = "--strict-stack"                          # the same switch on this command line (run.sh --strict-stack exports the variable)


def strict_stack(argv=(), environ=None):
    """The strict switch: ``--strict-stack`` among ``argv`` or ``CHAI1_OPT_STRICT_STACK=1`` in ``environ`` (default os.environ). Any other
    value of the variable than 1 / 0 / empty is a ValueError naming it (never read as permissive in silence)."""
    environ = os.environ if environ is None else environ
    v = str(environ.get(ENV_STRICT_STACK, "") or "").strip()
    if v not in ("", "0", "1"):
        raise ValueError(f"{ENV_STRICT_STACK}={v!r} is not 1 (strict: a torch/CUDA stack other than the pinned one is refused) or 0 / unset (it is recorded and the run proceeds)")
    return STRICT_FLAG in list(argv) or v == "1"


def torch_build():
    """torch's build as its installed distribution states it, WITHOUT importing torch: ``(version, cuda)``. ``version`` is the distribution's
    version, completed with the local tag of ``torch/version.py`` (``__version__``) when the metadata carries none; ``cuda`` is
    ``torch/version.py``'s ``cuda`` (None for a CPU build or when unreadable). ``(None, None)`` when torch is not installed."""
    try:
        dist = md.distribution("torch")
    except md.PackageNotFoundError:
        return None, None
    version, cuda = dist.version, None
    try:
        src = dist.locate_file("torch/version.py").read_text(encoding="utf-8")
    except (OSError, AttributeError, ValueError):
        return version, cuda
    m = re.search(r"^__version__\b[^=\n]*=\s*['\"]([^'\"]+)['\"]", src, re.M)
    full = m.group(1) if m else None
    m = re.search(r"^cuda\b[^=\n]*=\s*['\"]([^'\"]*)['\"]", src, re.M)
    cuda = (m.group(1) or None) if m else None
    if full and "+" not in version and full.split("+")[0] == version:
        version = full                                                        # the metadata's version sans local tag, the module's with it: one build
    return version, cuda


def cuda_of_tag(tag):
    """The CUDA version a torch local tag names (``cu130`` -> ``13.0``, ``cu128`` -> ``12.8``), or None (``cpu``, ``rocm6.1``, no tag)."""
    m = re.fullmatch(r"cu(\d+)(\d)", str(tag or ""))
    return f"{m.group(1)}.{m.group(2)}" if m else None


def cuda_build(version, cuda):
    """The CUDA version of an installed torch build: ``torch/version.py``'s ``cuda`` when readable, else what the version's local tag names;
    None when neither says (a CPU build, or a build whose CUDA cannot be read)."""
    tag = version.split("+", 1)[1] if version and "+" in version else None
    return (str(cuda) if cuda else None) or cuda_of_tag(tag)


def torch_label(version, cuda):
    """``torch <version>+<cuda tag>`` for the STACK line: a version carrying its local tag as is (2.13.0+cu130, 2.13.0+cpu); one without it
    completed from ``cuda`` (2.13.0 + 13.0 -> 2.13.0+cu130) or, with no CUDA readable either, ``torch 2.13.0 (CUDA build unknown)``;
    ``torch not installed`` when there is none."""
    if not version:
        return "torch not installed"
    if "+" in version:
        return f"torch {version}"
    if not cuda:
        return f"torch {version} (CUDA build unknown)"
    return f"torch {version}+cu{str(cuda).replace('.', '')}"


def build_matches(entry, version, cuda):
    """The installed torch build (``version`` with or without its local tag, ``cuda`` from torch/version.py) IS the build a PINS "stacks"
    entry names (``torch`` 2.13.0+cu130, ``cuda`` 13.0): the same torch version AND the same CUDA build — the local tags equal when the
    installed version carries one, and the installed CUDA version (torch/version.py, else the tag's) equals the entry's. A build whose CUDA
    cannot be read matches nothing (it is named on the STACK line, never taken for the pinned one)."""
    et = str(entry.get("torch") or "")
    ev, etag = et.split("+", 1)[0], (et.split("+", 1)[1] if "+" in et else None)
    iv, itag = version.split("+", 1)[0], (version.split("+", 1)[1] if "+" in version else None)
    if not ev or iv != ev:
        return False
    if itag is not None and etag is not None and itag != etag:
        return False
    icuda, ecuda = cuda_build(version, cuda), (str(entry["cuda"]) if entry.get("cuda") else cuda_of_tag(etag))
    return icuda is not None and ecuda is not None and icuda == ecuda


def stack_verdict(pins=None):
    """The torch / CUDA stack verdict — a record, never a gate here: ``{"version", "cuda", "declared_range", "in_range", "pinned_stack",
    "pinned", "pinned_builds", "line"}``. ``pinned`` = the installed torch is the build of a stack listed in PINS "stacks" — the same
    torch version AND the same CUDA build (``build_matches``: 2.13.0+cu128, 2.13.0+cpu or a 2.13.0 whose CUDA build cannot be read are NOT the
    pinned 2.13.0+cu130); ``line`` = None when pinned, else the one contract line ``STACK not pinned: torch <version>+<cuda> (pinned:
    torch 2.13.0+cu130)``."""
    pins = pins or read_pins()
    up = pins["upstream"]
    spec = next((r for r in up.get("declared_ranges", []) if r.startswith("torch")), None)
    stacks = {k: v for k, v in (pins.get("stacks") or {}).items() if isinstance(v, dict)}     # the tree's pinned stacks: name -> its pins
    builds = [str(v["torch"]) for v in stacks.values() if v.get("torch")]
    tv, cuda = torch_build()
    pinned = next((k for k, v in stacks.items() if build_matches(v, tv, cuda)), None) if tv else None
    t = {"version": tv, "cuda": cuda, "cuda_build": (cuda_build(tv, cuda) if tv else None), "declared_range": spec,
         "in_range": (torch_in_range(tv, spec) if (tv and spec) else None), "pinned_stack": pinned, "pinned": pinned is not None, "pinned_builds": builds}
    t["line"] = None if t["pinned"] else f"STACK not pinned: {torch_label(tv, cuda)} (pinned: {', '.join('torch ' + b for b in builds) or 'no stack listed'})"
    return t


def read_pins(path=None):
    with open(path or os.path.join(HERE, "PINS.json"), "r", encoding="utf-8") as fh:
        return json.load(fh)


def wheel_record(wheel_path):
    """{relative path: sha256 hex} for the package files of the wheel (RECORD carries urlsafe-b64 sha256)."""
    out = {}
    with zipfile.ZipFile(wheel_path) as z:
        rec = next(n for n in z.namelist() if n.endswith(".dist-info/RECORD"))
        for line in z.read(rec).decode().splitlines():
            parts = line.split(",")
            if len(parts) < 2 or not parts[1].startswith("sha256=") or not parts[0].startswith("chai_lab/"):
                continue
            b64 = parts[1][len("sha256="):]
            out[parts[0]] = base64.urlsafe_b64decode(b64 + "=" * (-len(b64) % 4)).hex()
    return out


def _sha(path):
    h = hashlib.sha256()                                     # its own digest loop by design: this checker runs on the stock interpreter, where opt_core is absent
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def torch_in_range(version, spec):
    """``version`` (e.g. 2.13.0+cu130) against a spec like ``torch<2.7,>=2.3.1`` (numeric tuples; the local tag is ignored)."""
    def tup(v):
        return tuple(int(x) for x in re.findall(r"\d+", v.split("+")[0])[:3])
    v = tup(version)
    for clause in spec.split(",")[0:]:
        m = re.match(r"^\s*torch\s*|\s*", clause)
        c = clause[m.end():] if clause.strip().startswith("torch") else clause.strip()
        m = re.match(r"(<=|>=|==|<|>|~=)\s*([\d.]+)", c)
        if not m:
            continue
        op, bound = m.group(1), tup(m.group(2))
        vb = v[:len(bound)] if op in ("<", "<=", ">", ">=") else v
        ok = {"<": vb < bound, "<=": vb <= bound, ">": vb > bound, ">=": vb >= bound, "==": vb == bound, "~=": vb >= bound}[op]
        if not ok:
            return False
    return True


def check(pins=None, wheel_path=None, strict=False):
    """(bad, detail): bad = refusal lines (empty = pinned); detail = what was found. The torch / CUDA stack verdict (``stack_verdict``) is
    ``detail["torch"]``; a stack other than the pinned one is a refusal line only when ``strict``."""
    pins = pins or read_pins()
    up = pins["upstream"]
    wheel_path = wheel_path or os.path.join(HERE, os.path.basename(up["wheel"]["file"]))
    bad, detail = [], {"python": sys.version.split()[0]}
    detail["torch"] = stack_verdict(pins)                                    # the stack record: pinned | the STACK not pinned line (a record whatever the upstream check finds)
    detail["torch"]["strict"] = bool(strict)
    if strict and not detail["torch"]["pinned"]:
        bad.append(f"{detail['torch']['line']} — refused under {STRICT_FLAG} / {ENV_STRICT_STACK}=1 (a torch/CUDA stack other than the pinned one is a refusal in strict form; "
                   f"without the switch it is recorded and the run proceeds)")
    try:
        dist = md.distribution(up["package"])
    except md.PackageNotFoundError:
        detail["chai_lab"] = None
        return [f"{up['package']} is not installed; want {up['package']}=={up['version']} ({up['install']})"] + bad, detail
    bad_up = []                                                              # the upstream pin refusals (hard, whatever the strict switch says)
    d = {"version": dist.version, "location": str(dist.locate_file("")), "pinned": False, "files_checked": 0, "files_differ": []}
    if dist.version != up["version"]:
        bad_up.append(f"{up['package']} {dist.version} installed; want {up['version']}")
    if os.path.isfile(wheel_path):
        rec = wheel_record(wheel_path)
        base = dist.locate_file("")
        differ = []
        for rel, want in rec.items():
            if not rel.endswith(".py"):
                continue
            p = os.path.join(str(base), rel)
            d["files_checked"] += 1
            if not os.path.isfile(p) or _sha(p) != want:
                differ.append(rel)
        d["files_differ"] = differ
        if differ:
            bad_up.append(f"{up['package']} {dist.version} at {base}: {len(differ)} file(s) differ from the pinned wheel ({differ[:3]}...)")
        if d["files_checked"] == 0:
            bad_up.append(f"the pinned wheel {wheel_path} lists no chai_lab .py files")
    else:
        bad_up.append(f"pinned wheel not found: {wheel_path} (stock/PINS.json upstream.wheel.file)")
    d["pinned"] = not bad_up
    detail["chai_lab"] = d
    return bad_up + bad, detail


def main():
    quiet = "--quiet" in sys.argv[1:]
    pins = read_pins()
    try:
        strict = strict_stack(sys.argv[1:])
    except ValueError as e:                                                  # a malformed strict switch is named, never read as permissive
        print(f"check_pins: {e}", file=sys.stderr)
        sys.exit(3)
    bad, detail = check(pins, strict=strict)
    t = detail.get("torch") or {}
    if not quiet:
        c = detail.get("chai_lab") or {}
        if c.get("pinned"):
            print(f"chai_lab {c['version']}: pinned ({c['files_checked']} files match the wheel) at {c['location']}")
        ps = f"; the pinned stack {t['pinned_stack']} (modes {','.join((pins.get('stacks') or {}).get(t['pinned_stack'], {}).get('modes', []))})" if t.get("pinned_stack") else "; not a pinned stack (stock/PINS.json stacks)"
        print(f"torch {t.get('version')}: declared range {t.get('declared_range')} -> {'in range' if t.get('in_range') else t.get('in_range')}{ps}; python {detail.get('python')}")
    if t.get("line") and not strict:                                         # the record, loud on stderr whatever --quiet says; the run proceeds
        print(f"{t['line']} — proceeding ({STRICT_FLAG} / {ENV_STRICT_STACK}=1 refuses instead)", file=sys.stderr)
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
