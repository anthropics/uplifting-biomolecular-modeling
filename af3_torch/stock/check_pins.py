#!/usr/bin/env python3
"""check_pins.py [--torch-py PY] [--jax-py PY] [--digest] [--json] [--quiet] — check an environment against stock/PINS.json.

Checks: (1) the two interpreters (--torch-py / --jax-py, default $AF3_TORCH_PY / $AF3_TORCH_JAX_PY; no other default — an unset one is
named and fails): each exists and reports every package of PINS `check_packages` at
the pinned version (torch + triton on the torch venv, jax + jaxlib on the JAX venv); (2) with --digest, the checkpoint under
$AF3_TORCH_PARAMS_DIR (the pinned file name, else the directory's one *.bin.zst | *.bin — the package's own rule) digested against PINS
`variants.p2.converted`: pinned (the pinned bytes) or UNPINNED (any other digest — it runs; the kit's tests and timings cover the
pinned checkpoint only; ONE `weights sha256=<12> UNPINNED — …` line) both pass, a missing checkpoint fails; `variants.p2.conventions`
(an optional layout record beside the checkpoint) is reported, not judged. The upstream archive (PINS `upstream.archive`, read by the
stock route only) is inventory, not a verdict: absent, the text report names it on one `PINS inventory …` line with the install step that
fetches it (stock/fetch_upstream.py). Exit 0 when (1) holds (and (2) when asked), 3 otherwise; the report is printed as text or JSON
(--quiet: text only when not met). Needs nothing beyond the standard library.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_pins():
    with open(os.path.join(HERE, "PINS.json"), encoding="utf-8") as f:
        return json.load(f)


def interpreter_report(py, packages, var):
    rep = {"python": py, "ok": True, "packages": {}}
    if not py:
        rep.update(ok=False, reason=f"{var} not set")
        return rep
    if not (os.path.isfile(py) and os.access(py, os.X_OK)):
        rep.update(ok=False, reason="not an executable interpreter")
        return rep
    code = ("import importlib.metadata as m, json, sys\n"
            "def v(n):\n"
            "    try: return m.version(n)\n"
            "    except m.PackageNotFoundError: return None\n"
            "print(json.dumps({'python': sys.version.split()[0], **{n: v(n) for n in sys.argv[1:]}}))")
    try:
        out = subprocess.run([py, "-c", code, *packages], capture_output=True, text=True, timeout=120)
        got = json.loads(out.stdout.strip().splitlines()[-1]) if out.returncode == 0 and out.stdout.strip() else {}
        if out.returncode != 0:
            rep["reason"] = f"query rc={out.returncode}: {out.stderr.strip()[-300:]}"
    except (OSError, ValueError, subprocess.TimeoutExpired) as e:
        got = {}
        rep["reason"] = f"could not query: {e}"
    rep["python_version"] = got.get("python")
    for n, want in packages.items():
        actual = got.get(n)
        rep["packages"][n] = {"expected": want, "actual": actual, "ok": actual == want}
        rep["ok"] &= actual == want
    return rep


WEIGHT_SUFFIXES = (".bin.zst", ".bin")      # a parameters file; the package resolves its checkpoint through this module (af3_torch_opt.stack.pins_tool) — one rule


def resolve_checkpoint(root, pinned):
    """(path, verdict-if-none): the pinned file name when present, else the directory's first *.bin.zst | *.bin in name order (xfold's
    loader takes a directory's first parameters file the same way); none -> 'missing'."""
    if not root or not os.path.isdir(root):
        return None, "missing"
    cands = sorted(f for f in os.listdir(root) if f.endswith(WEIGHT_SUFFIXES))
    if pinned in cands:
        return os.path.join(root, pinned), None
    if cands:
        return os.path.join(root, cands[0]), None
    return None, "missing"


def digest_report(pins):
    """--digest: the checkpoint under $AF3_TORCH_PARAMS_DIR resolved as the package resolves it and digested against `variants`: verdict
    pinned (the pinned bytes) | unpinned (any other digest: it RUNS — the kit's tests and timings cover the pinned checkpoint only;
    not a failure of this check) | missing (a failure). The optional conventions record beside the checkpoint is reported, not judged."""
    root = os.environ.get("AF3_TORCH_PARAMS_DIR")
    rep = {"params_dir": root, "variants": {}, "ok": True, "verdict": None, "sha256": None, "file": None}
    for v, spec in pins["variants"].items():
        path, none_verdict = resolve_checkpoint(root, spec["converted"]["file"])
        got = sha256(path) if path else None
        nbytes = os.path.getsize(path) if path else None
        pinned = got is not None and got == spec["converted"]["sha256"] and nbytes == spec["converted"].get("bytes", nbytes)
        verdict = none_verdict or ("pinned" if pinned else "unpinned")
        rep["variants"][v] = {"file": path, "present": bool(path), "sha256": got, "bytes": nbytes, "expected": spec["converted"]["sha256"],
                              "expected_bytes": spec["converted"].get("bytes"), "pinned": pinned, "verdict": verdict, "ok": verdict in ("pinned", "unpinned")}
        conv = spec.get("conventions")
        if conv:
            cpath = os.path.join(root, conv["file"]) if root else None
            cpresent = bool(cpath) and os.path.isfile(cpath)
            cgot = sha256(cpath) if cpresent else None
            rep["variants"][v]["conventions"] = {"file": conv["file"], "present": cpresent, "sha256": cgot, "expected": conv["sha256"], "ok": cgot == conv["sha256"]}
        rep["ok"] &= rep["variants"][v]["ok"]
        rep["verdict"], rep["sha256"], rep["file"] = verdict, got, path
    return rep


def digest_word(d):
    """`pinned:<12>` | `UNPINNED:<12>` (both pass) | `MISSING` | `AMBIGUOUS` (fail)."""
    v = d["verdict"]
    return f"{'pinned' if v == 'pinned' else 'UNPINNED'}:{d['sha256'][:12]}" if v in ("pinned", "unpinned") else v.upper()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--torch-py", default=None); ap.add_argument("--jax-py", default=None)
    ap.add_argument("--digest", action="store_true"); ap.add_argument("--json", action="store_true"); ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    pins = load_pins()
    torch_py = a.torch_py or os.environ.get("AF3_TORCH_PY")
    jax_py = a.jax_py or os.environ.get("AF3_TORCH_JAX_PY")
    rep = {"interpreters": {"torch_python": interpreter_report(torch_py, pins["check_packages"]["torch_python"], "AF3_TORCH_PY"),
                            "jax_python": interpreter_report(jax_py, pins["check_packages"]["jax_python"], "AF3_TORCH_JAX_PY")}}
    ok = all(r["ok"] for r in rep["interpreters"].values())
    if a.digest:
        rep["digest"] = digest_report(pins); ok = ok and rep["digest"]["ok"]
    rep["ok"] = ok
    if a.digest and rep["digest"]["verdict"] == "unpinned":        # ONE line, printed under --quiet too (stderr under --json, the report stays parseable)
        print(f"[af3-torch-opt] weights sha256={rep['digest']['sha256'][:12]} UNPINNED — the kit's tests and timings cover the pinned checkpoint only",
              file=sys.stderr if a.json else sys.stdout, flush=True)
    if a.json:
        print(json.dumps(rep, indent=1, sort_keys=True))
    elif not (a.quiet and ok):
        I = rep["interpreters"]
        print(f"[af3-torch-opt] PINS {'ok' if ok else 'NOT MET'} "
              f"torch_py={'ok' if I['torch_python']['ok'] else I['torch_python'].get('reason') or 'MISMATCH'} "
              f"jax_py={'ok' if I['jax_python']['ok'] else I['jax_python'].get('reason') or 'MISMATCH'}"
              + (f" digest={digest_word(rep['digest'])}" if a.digest else ""))
        for which, r in I.items():
            for n, d in r.get("packages", {}).items():
                if not d["ok"]:
                    print(f"  {which} {n}: expected {d['expected']} got {d['actual']}")
        if a.digest:
            for v, d in rep["digest"]["variants"].items():
                print(f"  params {v}: {'ok' if d['ok'] else ('absent' if not d['present'] else 'MISMATCH ' + d['sha256'][:16])}")
        u = pins["upstream"]                      # inventory, not a verdict: the stock route's input, which `run.sh install` fetches when the tree arrived without it
        if not os.path.isfile(os.path.join(HERE, u["archive"]["file"])):
            print(f"[af3-torch-opt] PINS inventory upstream_archive=absent file=stock/{u['archive']['file']} (read by `run.sh stock` only; `run.sh install` fetches it: "
                  f"stock/fetch_upstream.py, {u['repo']}/archive/{u['commit']}.tar.gz)")
    return 0 if ok else 3


if __name__ == "__main__":
    sys.exit(main())
