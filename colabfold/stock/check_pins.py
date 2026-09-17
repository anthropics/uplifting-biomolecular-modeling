#!/usr/bin/env python3
"""check_pins.py [--data DIR] [--digest] [--json] [--quiet] [--checks LIST] — check this interpreter's environment against stock/PINS.json.

Checks: (1) the installed distributions: `colabfold` and `alphafold-colabfold` at the pinned versions and every package of PINS
`check_packages` at the lock's version (environment/requirements.lock, PINS `freeze.file`: the pinned stack); (2) the installed copies of the three pinned files
(PINS `stock_files`: alphafold/model/modules.py — the Attention class the kit patches at run time —, colabfold/batch.py and
colabfold/input.py) byte-identical to this tree's own stock/src/ copies; (3) the pinned image's environment (PINS image.env) in the
current environment — reported, not judged (a caller may unset JAX_COMPILATION_CACHE_DIR or change the memory fraction; the stack is the
same); (4) the parameters under --data (default $COLABFOLD_OPT_DATA_DIR): the marker file and the five files PRESENT is the gate; their
sizes and, with --digest, their sha256 (1.9 GB read) against the pin are a state printed per file — `weights=<name> sha256=<12> (pinned)`
or `… NOT PINNED — the kit's measurements apply to the pinned weights only` — never a refusal (another checkpoint under the stock file
names passes). Exit 0 when (1)-(2) hold and, when a data directory is known, the parameters are present; 3 otherwise; the report is
printed as text or JSON (--quiet: text only when not met). `--checks` selects a subset of the four checks by name (packages, files, env,
weights; default: all four) — `run.sh install` runs `--checks packages,files`, the software pins, before any parameters exist; a check
left out reads `skipped` on the line. Runs on the interpreter that carries colabfold (`python -I stock/check_pins.py`);
imports nothing of colabfold, jax or alphafold — the files are located through the distributions' RECORD.
"""
import argparse
import hashlib
import importlib.metadata as md
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CHECKS = ("packages", "files", "env", "weights")          # the four checks; --checks selects a subset (run.sh install: packages,files)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pins():
    with open(os.path.join(HERE, "PINS.json"), encoding="utf-8") as f:
        return json.load(f)


def _norm(name):
    return name.lower().replace("_", "-")


def freeze_versions(pins):
    """name → version from the stack's one package list, PINS `freeze.file` (environment/requirements.lock, relative to the kit root; `#` lines are its header)."""
    out = {}
    with open(os.path.normpath(os.path.join(HERE, os.pardir, pins["freeze"]["file"])), encoding="utf-8") as f:
        for ln in f:
            if "==" in ln and not ln.lstrip().startswith("#"):
                name, ver = ln.strip().split("==", 1)
                out[_norm(name)] = ver
    return out


def dist_version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def dist_file(dist_name, rel):
    """The installed path of a file of a distribution (its RECORD), or None; nothing is imported."""
    try:
        d = md.distribution(dist_name)
    except md.PackageNotFoundError:
        return None
    p = d.locate_file(rel)
    return str(p) if os.path.isfile(p) else None


def packages_report(pins):
    want = freeze_versions(pins)
    res = {"ok": True, "python_version": sys.version.split()[0], "pinned_python_version": pins["image"]["python_version"], "packages": {}}
    for name in pins["check_packages"]:
        exp = want.get(_norm(name))
        got = dist_version(name)
        res["packages"][name] = {"expected": exp, "actual": got, "ok": got == exp}
        res["ok"] &= got == exp
    return res


def files_report(pins):
    """The installed copies of the pinned files (PINS stock_files) against this tree's own stock/src/ copies, byte for byte —
    no stored digest: stock/src/<rel> is tracked in this repo at its commit; the check just reads it."""
    res = {"ok": True, "files": {}}
    for rel in pins["stock_files"]:
        dist = "colabfold" if rel.startswith("colabfold/") else "alphafold-colabfold"
        path = dist_file(dist, rel)
        if not path:
            res["files"][rel] = {"path": None, "state": "missing"}; res["ok"] = False
            continue
        with open(path, "rb") as f1, open(os.path.join(HERE, "src", rel), "rb") as f2:
            same = f1.read() == f2.read()
        res["files"][rel] = {"path": path, "state": "ok" if same else "MISMATCH"}
        res["ok"] &= same
    return res


def env_report(pins):
    res = {"vars": {}}
    for k, v in pins["image"]["env"].items():
        res["vars"][k] = {"pinned": v, "actual": os.environ.get(k), "same": os.environ.get(k) == v}
    res["differ"] = sorted(k for k, d in res["vars"].items() if not d["same"])
    return res


def weights_report(pins, data_dir, digest=False, digest_fn=None):
    """The parameters under `data_dir` against stock/PINS.json "weights". PRESENCE is the gate: `ok` is False when the marker or a
    parameter file is missing (colabfold would fetch from the network), None when no data directory is named. The PIN STATE is
    recorded, never a refusal: each present file is `pinned` when its size and (with `digest`) its sha256 are the pinned ones,
    `not_pinned` otherwise (another checkpoint under the stock file name: it runs; the kit's measurements apply to the
    pinned weights only); without `digest` a file of the pinned size is `size_ok` (digest not computed). `digest_fn(path) -> hex` replaces
    the plain sha256 (the package passes a cached one)."""
    w = pins["weights"]
    res = {"data_dir": data_dir, "ok": True, "files": {}, "pinned": None}
    if not data_dir:
        res["ok"] = None
        res["reason"] = f"no data directory (--data or ${w['env']}): the parameters were not checked"
        return res
    marker = os.path.join(data_dir, w["marker"]["file"])
    res["marker"] = {"path": marker, "present": os.path.isfile(marker)}
    res["ok"] &= res["marker"]["present"]
    fn = digest_fn or sha256
    for rel, spec in w["files"].items():
        p = os.path.join(data_dir, rel)
        if not os.path.isfile(p):
            res["files"][rel] = {"state": "missing"}; res["ok"] = False
            continue
        size_ok = os.path.getsize(p) == spec["bytes"]
        entry = {"path": p, "bytes": os.path.getsize(p), "size_ok": size_ok, "pinned_sha256": spec["sha256"]}
        if digest:
            got = fn(p)
            entry.update(sha256=got, digest_ok=got == spec["sha256"])
            entry["state"] = "pinned" if (size_ok and got == spec["sha256"]) else "not_pinned"
        else:
            entry["state"] = "size_ok" if size_ok else "not_pinned"
        res["files"][rel] = entry
    states = [d["state"] for d in res["files"].values()]
    if res["ok"] and states:
        res["pinned"] = True if all(s == "pinned" for s in states) else (False if any(s == "not_pinned" for s in states) else None)
    return res


def weights_lines(wrep, quiet=False, prefix="[colabfold-opt]"):
    """One line per present parameter file: `weights=<name> sha256=<12> (pinned)` for the pinned bytes, `weights=<name> sha256=<12>
    NOT PINNED — the kit's measurements apply to the pinned weights only` for any other checkpoint (it runs), `weights=<name>
    bytes=<n> NOT PINNED (size differs from the pin; digest not computed) …` when only sizes were compared; `quiet` keeps the
    NOT PINNED lines and drops the pinned ones."""
    out = []
    for rel, d in (wrep or {}).get("files", {}).items():
        name = os.path.basename(rel)
        if d["state"] == "pinned" and not quiet:
            out.append(f"{prefix} weights={name} sha256={d['sha256'][:12]} (pinned)")
        elif d["state"] == "not_pinned" and "sha256" in d:
            out.append(f"{prefix} weights={name} sha256={d['sha256'][:12]} NOT PINNED — the kit's measurements apply to the pinned weights only")
        elif d["state"] == "not_pinned":
            out.append(f"{prefix} weights={name} bytes={d['bytes']} NOT PINNED (size differs from the pin; digest not computed) — the kit's measurements apply to the pinned weights only")
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=None, help="the parameters root (default $COLABFOLD_OPT_DATA_DIR)")
    ap.add_argument("--digest", action="store_true", help="sha256 of the five parameter files")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--quiet", action="store_true", help="print the text report only when the pins are not met")
    ap.add_argument("--checks", default=",".join(CHECKS), help="comma-separated subset of packages,files,env,weights (default: all four); run.sh install passes packages,files")
    a = ap.parse_args(argv)
    selected = [c.strip() for c in a.checks.split(",") if c.strip()]
    unknown = [c for c in selected if c not in CHECKS]
    if unknown or not selected:
        print(f"check_pins.py: --checks takes a comma-separated subset of {','.join(CHECKS)} (got {a.checks!r})", file=sys.stderr)
        return 2
    pins = load_pins()
    data_dir = a.data or os.environ.get(pins["weights"]["env"])
    rep = {"packages": packages_report(pins) if "packages" in selected else {"ok": True, "skipped": True, "packages": {}},
           "files": files_report(pins) if "files" in selected else {"ok": True, "skipped": True, "files": {}},
           "env": env_report(pins) if "env" in selected else {"skipped": True, "vars": {}, "differ": []},
           "weights": weights_report(pins, data_dir, a.digest) if "weights" in selected else {"ok": None, "skipped": True, "data_dir": data_dir, "files": {}, "reason": "not selected (--checks)"}}
    ok = rep["packages"]["ok"] and rep["files"]["ok"] and rep["weights"]["ok"] is not False     # weights: presence only — another checkpoint is reported NOT PINNED and passes
    rep["ok"] = ok; rep["checks"] = selected
    if a.json:
        print(json.dumps(rep, indent=1, sort_keys=True))
    elif not (a.quiet and ok):
        word = lambda name, bad: "skipped" if rep[name].get("skipped") else ("ok" if rep[name]["ok"] else bad)   # noqa: E731
        wstate = "skipped" if rep["weights"].get("skipped") else {True: "ok", False: "NOT MET", None: "unchecked"}[rep["weights"]["ok"]]
        print(f"[colabfold-opt] PINS {'ok' if ok else 'NOT MET'} "
              f"packages={word('packages', 'MISMATCH')} files={word('files', 'MISMATCH')} "
              f"weights={wstate} env_differs={'skipped' if rep['env'].get('skipped') else (','.join(rep['env']['differ']) or 'none')}")
        for n, d in rep["packages"]["packages"].items():
            if not d["ok"]:
                print(f"  {n}: expected {d['expected']} got {d['actual']}")
        for rel, d in rep["files"]["files"].items():
            if d["state"] != "ok":
                print(f"  {rel}: {d['state']} ({d.get('path')})")
        if rep["weights"]["ok"] is False:
            if not rep["weights"].get("marker", {}).get("present", True):
                print(f"  marker missing: {rep['weights']['marker']['path']} (colabfold would fetch the parameters from the network)")
            for rel, d in rep["weights"]["files"].items():
                if d["state"] == "missing":
                    print(f"  {rel}: missing")
    if not a.json:
        for line in weights_lines(rep["weights"], quiet=a.quiet):        # the pin state is reported, never a refusal: NOT PINNED lines print even under --quiet
            print(line)
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
