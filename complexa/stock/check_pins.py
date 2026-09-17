#!/usr/bin/env python3
"""Refuse unless the pinned upstream is what this interpreter would run: ``proteinfoundation`` 1.1.0 installed from a checkout of
NVIDIA-BioNeMo/Proteina-Complexa at the pinned commit, clean under src/ and configs/ when the checkout is a git repository, the
pinned generation-stage config files present in that checkout, and, when asked, the two pinned checkpoint files
(stock/PINS.json "upstream", "configs", "weights").

usage: python -I stock/check_pins.py [--quiet] [--json] [--no-torch] [--weights DIR] [--no-sha]
       exit 0 = pinned; 3 = not (one line per finding on stderr)

check_upstream(): package metadata, version, and ``git rev-parse`` / ``git status`` of the checkout (a checkout not at the pinned commit,
or dirty — or whose cleanliness could not be determined, e.g. ``git status`` itself failed — under src/ or configs/, is refused;
generate.py — the module `complexa generate` execs — must be present) — no torch import and no proteinfoundation import (importing it
imports torch, jax and atomworks).
check_configs(): the config files the stock route composes (the pipeline file, binder_generate, model_sampling, base_gen_data) are present in
the checkout ``$LOCAL_CODE_PATH`` (else the editable install's directory) — their bytes are check_upstream's job (the checkout's commit
and cleanliness), not a hash here.
stack_report(): the running python / torch / CUDA / cuDNN / lightning / hydra / numpy against "pinned_stack" — a LABEL, never a gate.
check_weights(): the pinned checkpoints in a directory (byte counts, then sha256 unless with_sha=False) — a gate only when asked for.
Standard library only, so run.sh, the configs and the package all call this one file.
"""
import hashlib
import importlib.metadata as md
import importlib.util
import json
import os
import platform
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TREE = os.path.dirname(HERE)
PINS_PATH = os.path.join(HERE, "PINS.json")
DIST = "proteinfoundation"
PACKAGE = "proteinfoundation"
TAG = "complexa-opt"
ENV_UPSTREAM = "LOCAL_CODE_PATH"
EXIT_NOT_PINNED = 3


def load_pins(path=PINS_PATH):
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for b in iter(lambda: fh.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def installed_upstream():
    """(distribution version, package directory, PEP 610 direct_url dict) of the installed proteinfoundation, or (None, None, {})."""
    try:
        dist = md.distribution(DIST)
    except md.PackageNotFoundError:
        return None, None, {}
    pkg_dir = None
    spec = importlib.util.find_spec(PACKAGE)
    if spec is not None and spec.submodule_search_locations:
        pkg_dir = os.path.abspath(list(spec.submodule_search_locations)[0])
    raw = dist.read_text("direct_url.json")
    try:
        direct_url = json.loads(raw) if raw else {}
    except ValueError:
        direct_url = {}
    return dist.version, pkg_dir, direct_url


def checkout_dir(direct_url, pkg_dir):
    """The pinned checkout's location: $LOCAL_CODE_PATH when set, else the editable install's directory (direct_url file://…), else two levels above the package (src layout)."""
    env = os.environ.get(ENV_UPSTREAM)
    if env:
        return os.path.abspath(env)
    url = (direct_url or {}).get("url") or ""
    if url.startswith("file://"):
        return os.path.abspath(url[len("file://"):])
    if pkg_dir:
        return os.path.dirname(os.path.dirname(pkg_dir))
    return None


def git_head(path):
    """(commit or None, note, dirty file count or None): ``git rev-parse HEAD`` and the porcelain status of ``path`` under src/ and
    configs/; (None, note, 0) when it is not a git checkout or git is absent — version + presence are the gate then. ``dirty`` is
    None when the commit IS known but the status check itself failed (a ``git status`` error is not evidence of a clean tree — the
    caller must refuse, not assume clean)."""
    if not path or not os.path.isdir(os.path.join(path, ".git")):
        return None, "no .git in the checkout (version + presence are the gate)", 0
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"git unavailable ({type(e).__name__})", 0
    if head.returncode != 0:
        return None, f"git rev-parse failed: {head.stderr.strip()[-200:]}", 0
    commit = head.stdout.strip()
    try:
        stat = subprocess.run(["git", "-C", path, "status", "--porcelain", "--", "src", "configs"], capture_output=True, text=True, timeout=120)
    except (OSError, subprocess.SubprocessError) as e:
        return commit, f"git status unavailable ({type(e).__name__}): cleanliness unknown", None
    if stat.returncode != 0:
        return commit, f"git status failed: {stat.stderr.strip()[-200:]} (cleanliness unknown)", None
    dirty = [ln for ln in stat.stdout.splitlines() if ln.strip()]
    return commit, (f"{len(dirty)} modified/untracked under src/ or configs/" if dirty else "clean under src/ and configs/"), len(dirty)


def check_upstream(pins):
    """The proteinfoundation pin: version, checkout commit, and (when the checkout is a git repository) cleanliness under src/ and
    configs/ — a checkout at the pinned commit with local edits there is not the pin either — plus generate.py's presence (the module
    `complexa generate` execs, ``python -m proteinfoundation.generate``). Returns (bad lines, detail); empty bad = pinned."""
    want = pins["upstream"][DIST]
    version, pkg_dir, direct_url = installed_upstream()
    detail = {"version": version, "pinned_version": want["version"], "package_dir": pkg_dir,
              "editable": bool((direct_url.get("dir_info") or {}).get("editable")),
              "checkout": None, "commit": None, "git_note": None, "dirty": None, "generate_py_present": None}
    bad = []
    if version is None:
        bad.append(f"{DIST}: not installed (want {want['install']} of the checkout at {want['commit'][:12]})")
        detail["pinned"] = False
        return bad, detail
    if version != want["version"]:
        bad.append(f"{DIST} {version}: not the pin {want['version']}")
    if pkg_dir is None or not os.path.isdir(pkg_dir):
        bad.append(f"{DIST}: no importable {PACKAGE}/ package directory in this interpreter")
        detail["pinned"] = False
        return bad, detail
    detail["generate_py_present"] = os.path.isfile(os.path.join(pkg_dir, "generate.py"))
    if not detail["generate_py_present"]:
        bad.append(f"{pkg_dir}/generate.py: absent (the module `complexa generate` execs)")
    co = checkout_dir(direct_url, pkg_dir)
    detail["checkout"] = co
    commit, note, dirty = git_head(co)
    detail["commit"], detail["git_note"], detail["dirty"] = commit, note, dirty
    if commit is not None:
        if commit != want["commit"]:
            bad.append(f"checkout {co} is at {commit[:12]}, not the pin {want['commit'][:12]}")
        elif dirty is None:
            bad.append(f"checkout {co} is at the pin {commit[:12]} but {note}: refusing rather than assuming clean")
        elif dirty:
            bad.append(f"checkout {co} is at the pin {commit[:12]} but {note}: not the pin's bytes")
    detail["pinned"] = not bad
    return bad, detail


def check_configs(pins, checkout=None):
    """The pinned generation-stage config files are present in the checkout (their bytes are check_upstream's job: the checkout's
    commit and cleanliness already fix them). Returns (bad, detail)."""
    rec = pins["configs"]
    if checkout is None:
        _, pkg_dir, direct_url = installed_upstream()
        checkout = checkout_dir(direct_url, pkg_dir)
    detail = {"checkout": checkout, "files": {}}
    bad = []
    if not checkout or not os.path.isdir(checkout):
        return [f"configs: no checkout to read ({ENV_UPSTREAM} unset and no editable install)"], detail
    for rel in rec["files"]:
        p = os.path.join(checkout, rel)
        present = os.path.isfile(p)
        detail["files"][rel] = "present" if present else "absent"
        if not present:
            bad.append(f"{rel}: absent from {checkout}")
    detail["pinned"] = not bad
    return bad, detail


def _dist_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def stack_report(pins, import_torch=True):
    """The running stack against stock/PINS.json "pinned_stack". A report, never a gate."""
    sor = pins.get("pinned_stack") or {}
    record = dict(sor.get("pins") or {})
    running = {"python": platform.python_version(), "torch": _dist_version("torch"), "cuda": None, "nvidia-cudnn-cu12": _dist_version("nvidia-cudnn-cu12"),
               "cudnn_runtime": None, "numpy": _dist_version("numpy"), "lightning": _dist_version("lightning"), "hydra-core": _dist_version("hydra-core"),
               "triton": _dist_version("triton"), "jax": _dist_version("jax"), "atomworks": _dist_version("atomworks"), DIST: _dist_version(DIST), "torch_imported": False}
    if import_torch:
        try:
            import torch  # the one place torch is imported; never at module import
            running["torch"] = torch.__version__
            running["cuda"] = torch.version.cuda
            running["cudnn_runtime"] = torch.backends.cudnn.version()
            running["torch_imported"] = True
        except Exception as e:
            running["torch_import_error"] = f"{type(e).__name__}: {e}"
    keys = [k for k in ("python", "torch", "cuda", "nvidia-cudnn-cu12", "cudnn_runtime", "numpy", "lightning", "hydra-core", "triton", "jax", "atomworks", DIST) if k in record]
    matches = {k: running.get(k) == record.get(k) for k in keys}
    unread = [k for k in keys if running.get(k) is None]
    return {"running": running, "record": {k: record[k] for k in keys}, "matches": matches, "unread": unread,
            "matches_record": all(matches.values())}


def check_weights(pins, path, with_sha=True):
    """The pinned checkpoints in directory ``path``: byte count first, sha256 only when the counts match and ``with_sha``. Returns (bad, detail)."""
    detail, bad = {"dir": path, "files": {}}, []
    if not path or not os.path.isdir(path):
        bad.append(f"weights: {path!r} is not a directory holding {', '.join(w['file'] for w in pins['weights']['files'])} (stock/PINS.json \"weights\")")
        return bad, detail
    for rec in pins["weights"]["files"]:
        p = os.path.join(path, rec["file"])
        d = {"path": p}
        detail["files"][rec["file"]] = d
        if not os.path.isfile(p):
            bad.append(f"{rec['file']}: absent from {path}")
            continue
        size = os.path.getsize(p)
        d["bytes_ok"] = size == rec["bytes"]
        if not d["bytes_ok"]:
            bad.append(f"{rec['file']}: {size} bytes, the pin is {rec['bytes']}")
            continue
        if with_sha:
            d["sha256_ok"] = sha256_file(p) == rec["sha256"]
            if not d["sha256_ok"]:
                bad.append(f"{rec['file']}: bytes do not hash to the pin {rec['sha256'][:12]}")
        else:
            d["sha256_ok"] = None
    return bad, detail


def weights_line(pins, path, pinned=True, with_sha=True):
    """``[complexa-opt] WEIGHTS dir=<path> complexa.ckpt=<12 hex> complexa_ae.ckpt=<12 hex> (pinned[, bytes only])`` — or ``UNPINNED`` — the parent's weights line."""
    parts = " ".join(f"{w['file']}={w['sha256'][:12]}" for w in pins["weights"]["files"])
    tail = ("(pinned)" if with_sha else "(pinned, bytes only: sha256 not recomputed)") if pinned else "UNPINNED (not the bytes of stock/PINS.json)"
    return f"[{TAG}] WEIGHTS dir={path} {parts} {tail}"


def _fmt_stack(rep):
    r, rec = rep["running"], rep["record"]
    return (f"stack: python {r['python']} torch {r['torch']} cuda {r['cuda']} cudnn {r.get('nvidia-cudnn-cu12') or r.get('cudnn_runtime')} numpy {r['numpy']}"
            f" lightning {r['lightning']} hydra {r['hydra-core']} jax {r['jax']} atomworks {r['atomworks']} {DIST} {r[DIST]}; pinned stack python {rec.get('python')} torch {rec.get('torch')}"
            f" cuda {rec.get('cuda')}: {'match' if rep['matches_record'] else 'differs (a label, not a gate)'}"
            + (f"; not read: {', '.join(rep['unread'])}" if rep["unread"] else ""))


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    quiet, want_json, no_torch, no_sha = "--quiet" in args, "--json" in args, "--no-torch" in args, "--no-sha" in args
    weights = None
    if "--weights" in args:
        i = args.index("--weights")
        if i + 1 >= len(args) or args[i + 1].startswith("--"):
            print("check_pins: --weights needs a directory", file=sys.stderr)
            sys.exit(2)
        weights = args[i + 1]
    pins = load_pins()
    bad, detail = check_upstream(pins)
    cbad, cdetail = check_configs(pins, detail.get("checkout"))
    bad += cbad
    rep = stack_report(pins, import_torch=not no_torch)
    out = {DIST: detail, "configs": cdetail, "stack": rep}
    if weights is not None:
        wbad, wdetail = check_weights(pins, weights, with_sha=not no_sha)
        bad += wbad
        out["weights"] = {"dir": weights, "detail": wdetail, "pinned": not wbad}
    if want_json:
        print(json.dumps(dict(out, bad=bad, pinned=not bad), indent=1))
    elif not quiet:
        if detail.get("pinned"):
            print(f"{DIST} {detail['version']}: pinned (checkout {detail['checkout']} at {(detail['commit'] or 'no-git')[:12]}, {detail['git_note']}; "
                  f"generate.py present; {len(cdetail['files'])} pinned config files present)")
        print(_fmt_stack(rep))
        if weights is not None and out["weights"]["pinned"]:
            print(f"weights: {', '.join(w['file'] for w in pins['weights']['files'])} at {weights}: pinned")
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(EXIT_NOT_PINNED)


if __name__ == "__main__":
    main()
