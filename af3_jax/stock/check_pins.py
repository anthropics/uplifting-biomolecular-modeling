#!/usr/bin/env python3
"""check_pins.py [--py PYTHON] [--repo DIR] [--digest] [--json] [--quiet] — check an environment against stock/PINS.json.

Checks: (1) the archive under stock/ and the lock (PINS freeze.file: environment/requirements.lock, the pinned stack) are present (the files
are the checked-in bytes; git is the record, not a stored sum); (2) the fork's interpreter (--py, default $AF3_JAX_PY or PINS image.python) exists and
answers, and its python version and every package of PINS `check_packages` are READ against the lock; (3) the pinned XLA variables in the current
environment are READ against PINS image.env; (4) the install carries the nine stock files the add-ons pin or patch at run time (--repo, default
$AF3_JAX_REPO or PINS image.repo_dir; the package's site directory from the interpreter): present or missing — every one must be present (an image
without stock/patches applied is not distinguished from one that is: this check is presence-only, not a content comparison); (5) with
--digest, the converted parameters under $AF3_JAX_PARAMS_ROOT/<variant>/ against PINS `variants.<v>.converted.sha256` (an external,
downloaded artifact — 1.4 GB read each).
The rule: a stack that DIFFERS from the pin — another python, jax, jaxlib, tokamax, triton … release, another XLA_FLAGS, a pool variable unset — is
DRIFT: named, one `[af3-jax-opt] PINS drift …` line per item on stderr, exit 0 (the modes run on it; what they print names the stack they ran
on). A difference that changes nothing the model computes is a NOTE (torch, the weights converter's package, at the lock's release with any local
tag; a python patch level; a memory-pool variable at another value). Only what no mode can run without is NOT MET, exit 3: the kit's own files
absent (1), no interpreter or one that cannot answer (2), a stock install that is not the pinned stock's (4).
The report is printed as text or JSON (--quiet: the NOTE / drift lines only, and the text report when not met). Needs nothing beyond the standard
library.
"""
import argparse
from urllib.parse import unquote
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


# stdlib on purpose: the stock proof runs without the kit or the shared core installed (that absence is what it proves) — no opt_core import here
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pins():
    with open(os.path.join(HERE, "PINS.json"), encoding="utf-8") as f:
        return json.load(f)


def lock_path(pins):
    """The pinned stack's one list: PINS freeze.file, relative to the kit tree (environment/requirements.lock)."""
    return os.path.join(os.path.dirname(HERE), pins["freeze"]["file"])


def freeze_versions(pins):
    out = {}
    with open(lock_path(pins), encoding="utf-8") as f:
        for ln in f:
            ln = ln.strip()
            if ln.startswith(("#", "-")) or not ln: continue                  # the header; no option lines are expected, none are packages
            if " @ " in ln:                                                  # `name @ <url>/<name>-<version>-<tags>.whl#sha256=…`: a wheel pinned by URL (torch's CPU build) — the version is the file name's
                name, url = (x.strip() for x in ln.split(" @ ", 1))
                out[name.lower().replace("_", "-")] = unquote(url.split("#", 1)[0].rsplit("/", 1)[-1]).split("-")[1]
            elif "==" in ln:
                name, ver = ln.split("==", 1)
                out[name.lower().replace("_", "-")] = ver
    return out


def check_files(pins):
    """The lock is present (presence only: its bytes are git's record, not a stored sum); the upstream source archive under stock/ is present or
    absent BY NAME — a tree without it lays the pinned source out by cloning upstream.repo at upstream.commit (stock/unpack_src.sh, an install
    step), so its absence is inventory, not a refusal."""
    res = {"ok": True, "files": {}}
    arc = pins["upstream"]["archive"]["file"]
    res["archive"] = "present" if os.path.isfile(os.path.join(HERE, arc)) else "absent"
    res["files"][arc] = "ok" if res["archive"] == "present" else "absent (install clones upstream.repo at upstream.commit: stock/unpack_src.sh)"
    ok = os.path.isfile(lock_path(pins))
    res["files"][pins["freeze"]["file"]] = "ok" if ok else "MISSING"
    res["ok"] &= ok
    return res


def interpreter_report(py, pins):
    if not py or not os.path.isfile(py):
        return {"ok": False, "python": py, "reason": "interpreter not found"}
    code = ("import json, sys, importlib.metadata as m\n"
            "names = json.loads(sys.argv[1])\n"
            "out = {}\n"
            "for n in names:\n"
            "    try: out[n] = m.version(n)\n"
            "    except m.PackageNotFoundError: out[n] = None\n"
            "print(json.dumps({'python': sys.version.split()[0], 'packages': out}))")
    try:
        p = subprocess.run([py, "-I", "-c", code, json.dumps(pins["check_packages"])], capture_output=True, text=True, timeout=120)
        got = json.loads(p.stdout.strip().splitlines()[-1])
    except Exception as e:  # noqa: BLE001 - any failure is the report
        return {"ok": False, "python": py, "reason": f"{type(e).__name__}: {e}"}
    want = freeze_versions(pins)
    res = {"ok": True, "python": py, "python_version": got["python"], "packages": {}, "notes": [], "drift": []}
    exp_py = pins["image"]["python_version"]
    if got["python"] != exp_py:
        res["python_version_expected"] = exp_py
        if got["python"].split(".")[:2] != exp_py.split(".")[:2]:            # another MAJOR.MINOR (3.11.x / 3.13.x against a 3.12 pin): drift, named — the modes run on it
            res["drift"].append(f"python={got['python']} (pinned {exp_py})")
        else:                                                                # a patch-level difference (3.12.3 against the pinned 3.12.1): a NOTE
            res["python_note"] = f"python {got['python']} differs from the pinned {exp_py} in patch level only — proceeding"
    for n in pins["check_packages"]:
        exp = want.get(n.lower().replace("_", "-"))
        entry = package_entry(n, exp, got["packages"].get(n))
        if entry.get("note"): res["notes"].append(entry["note"])
        if entry.get("drift"): res["drift"].append(entry["drift"])
        res["packages"][n] = entry
    res["met"] = not res["drift"]
    return res


LOCAL_TAG_FREE = ("torch",)                                                    # packages held to the lock's RELEASE, not its build: torch serves the weights converter only
                                                                                # (`run.sh install`; no model process imports it), so PyTorch's CUDA builds of the pinned release
                                                                                # (2.7.1+cu126, …) stand in for the lock's CPU build (2.7.1+cpu) — named on a NOTE line, never silent


def package_entry(name, expected, actual):
    """One check_packages reading: {expected, actual, met[, note | drift]}. ``met``: equal to the lock, or a LOCAL_TAG_FREE package that differs
    from the lock's in its local tag only (the text after '+') — accepted with one NOTE, the same release, another build. Anything else (another
    release, the package absent) is ``drift``: its words for the one PINS drift line — named, never a refusal."""
    entry = {"expected": expected, "actual": actual, "met": actual == expected}
    if not entry["met"] and name.lower() in LOCAL_TAG_FREE and expected and actual and actual.split("+", 1)[0] == expected.split("+", 1)[0]:
        entry["met"] = True
        entry["note"] = f"{name}={actual} (the lock's is {expected}: the same release, another build — {name} serves the weights converter only; outputs unchanged)"
    elif not entry["met"]:
        entry["drift"] = f"{name}={actual or 'absent'} (pinned {expected})"
    return entry


MEMORY_POOL_VARS = ("XLA_PYTHON_CLIENT_PREALLOCATE", "XLA_CLIENT_MEM_FRACTION")   # JAX's device-memory pool: whether it is preallocated and how large it may grow —
                                                                                # allocator behaviour only, no effect on what the model computes (upstream's own
                                                                                # docs/performance.md changes both for large inputs)


def env_report(pins):
    """Per XLA variable of PINS image.env: expected (the pinned image's value), actual, met. The two memory-pool variables (MEMORY_POOL_VARS)
    at another value are a NOTE (a smaller or on-demand pool changes how much device memory the process holds, not its outputs); ``XLA_FLAGS``
    at another value (it selects code paths of the compiled program) or any pinned variable unset is DRIFT — named on one PINS drift line,
    never a refusal: the modes run with the environment as it stands."""
    res = {"ok": True, "vars": {}, "notes": [], "drift": []}
    for k, v in pins["image"]["env"].items():
        actual = os.environ.get(k)
        entry = {"expected": v, "actual": actual, "met": actual == v}
        if k in MEMORY_POOL_VARS and actual is not None and actual != v:
            entry["met"] = True
            entry["note"] = f"{k}={actual} (the pinned image's value is {v}; a memory-pool setting: outputs unchanged)"
            res["notes"].append(entry["note"])
        elif not entry["met"]:
            entry["drift"] = f"{k}={'unset' if actual is None else repr(actual)} (pinned {v!r})"
            res["drift"].append(entry["drift"])
        res["vars"][k] = entry
    res["met"] = not res["drift"]
    return res


def install_report(py, repo, pins):
    site = None
    if py and os.path.isfile(py):
        try:
            p = subprocess.run([py, "-I", "-c", "import alphafold3, os; print(os.path.dirname(alphafold3.__file__))"], capture_output=True, text=True, timeout=120)
            site = p.stdout.strip().splitlines()[-1] if p.returncode == 0 and p.stdout.strip() else None
        except Exception:  # noqa: BLE001
            site = None
    res = {"repo": repo, "site": site, "files": {}}
    for rel in pins["stock_files"]:
        if rel.startswith("src/alphafold3/"):
            path = os.path.join(site, rel[len("src/alphafold3/"):]) if site else None
        else:
            path = os.path.join(repo, rel) if repo else None
        res["files"][rel] = "present" if path and os.path.isfile(path) else "missing"
    states = set(res["files"].values())
    res["install"] = states.pop() if len(states) == 1 else "mixed"
    return res


def digest_report(pins):
    root = os.environ.get("AF3_JAX_PARAMS_ROOT")
    res = {"root": root, "variants": {}}
    if not root:
        return res
    for v, spec in pins["variants"].items():
        p = os.path.join(root, v, spec["converted"]["file"])
        if not os.path.isfile(p):
            res["variants"][v] = {"present": False}
            continue
        s = sha256(p)
        res["variants"][v] = {"present": True, "sha256": s, "expected": spec["converted"]["sha256"], "ok": s == spec["converted"]["sha256"]}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--py", default=None); ap.add_argument("--repo", default=None)
    ap.add_argument("--digest", action="store_true"); ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print the NOTE / drift lines only, and the text report when the pins are not met")
    a = ap.parse_args(argv)
    pins = load_pins()
    py = a.py or os.environ.get("AF3_JAX_PY") or pins["image"]["python"]
    repo = a.repo or os.environ.get("AF3_JAX_REPO") or pins["image"]["repo_dir"]
    rep = {"files": check_files(pins), "interpreter": interpreter_report(py, pins), "env": env_report(pins), "install": install_report(py, repo, pins)}
    if a.digest:
        rep["digest"] = digest_report(pins)
    ok = rep["files"]["ok"] and rep["interpreter"]["ok"] and rep["install"]["install"] == "present"   # what no mode can run without: the kit's files, an interpreter that answers, the pinned stock installed
    drift = rep["interpreter"].get("drift", []) + rep["env"].get("drift", [])                             # a stack other than the pin: named, exit 0
    rep["ok"], rep["drift"], rep["met"] = ok, drift, bool(ok and not drift)
    if a.json:
        print(json.dumps(rep, indent=1, sort_keys=True))
    else:
        if rep["interpreter"].get("python_note"):                        # the accepted patch-level python difference: ONE NOTE line, on stderr, even under --quiet with everything met
            print(f"[af3-jax-opt] NOTE {rep['interpreter']['python_note']}", file=sys.stderr)
        for note in rep["interpreter"].get("notes", []) + rep["env"].get("notes", []):   # an accepted package build (LOCAL_TAG_FREE) or memory-pool value other than the
            print(f"[af3-jax-opt] NOTE {note}", file=sys.stderr)             # image's is never silent — one NOTE line each, on stderr, even under --quiet and even when the pins are otherwise fully met
        for d in drift:                                                   # a stack other than the pin: ONE drift line per item, on stderr, even under --quiet — named, exit 0; the modes
            print(f"[af3-jax-opt] PINS drift {d} — named, not refused: the modes run on this stack and their lines name it", file=sys.stderr)   # run on it (README: The pinned stack)
        if not (a.quiet and ok):
            print(f"[af3-jax-opt] PINS {'met' if rep['met'] else ('drift' if ok else 'NOT MET')} files={'ok' if rep['files']['ok'] else 'MISSING'} "
                  f"interpreter={('ok' if rep['interpreter'].get('met', True) else 'drift') if rep['interpreter']['ok'] else rep['interpreter'].get('reason') or 'unusable'} "
                  f"env={'ok' if rep['env']['met'] else 'drift'} install={rep['install']['install']}")
            if rep["files"]["archive"] == "absent":                           # inventory, not a condition: the source came (or comes) from upstream.repo at the pinned commit instead
                u = pins["upstream"]
                print(f"[af3-jax-opt] PINS inventory upstream_archive=absent file=stock/{u['archive']['file']} source_route=clone repo={u['repo']} commit={u['commit'][:12]} (stock/unpack_src.sh)")
            for n, d in rep["interpreter"].get("packages", {}).items():
                if not d["met"]:
                    print(f"  {n}: pinned {d['expected']} got {d['actual']}")
            for k, d in rep["env"]["vars"].items():
                if not d["met"]:
                    print(f"  {k}: pinned {d['expected']!r} got {d['actual']!r}")
            for rel, state in rep["install"]["files"].items():
                if state != "present":
                    print(f"  {rel}: {state} (stock/PINS.json stock_files; the pinned stock is stock/src = the archive + stock/patches)")
            if a.digest:
                for v, d in rep["digest"]["variants"].items():
                    print(f"  params {v}: {'ok' if d.get('ok') else ('absent' if not d.get('present') else 'differs ' + d['sha256'][:16])}")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
