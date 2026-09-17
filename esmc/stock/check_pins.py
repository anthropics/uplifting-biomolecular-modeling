#!/usr/bin/env python3
"""The pin check of this tree (stock/PINS.json). REFUSED (exit 3), because it changes what "stock" means or nothing can run: an upstream
package (esm, the transformers fork) absent or not at its pinned COMMIT; torch that does not import; with --weights, a checkpoint file off
its pinned sha256. NAMED (printed, exit 0), because it is environment drift the modes run through and state on their own lines: the stack
off its pins (stock/PINS.json "stacks": `accel` = flash_attn + transformer_engine at their versions — another version, absent, or an
accelerator module that does not import), xformers present. A lever that cannot run on what is installed steps aside by name at apply;
the KERNELS line of every run says what the built model engages.

usage: python -I stock/check_pins.py [--quiet] [--weights <dir> [--variant V]]   exit 0 = the upstream commits pinned (drift, if any, named); exit 3 = each refusal named

The pins are read from package metadata: the PEP 610 direct_url.json that pip writes for a git install carries the commit
(vcs_info.commit_id); an install from the archive in stock/ carries that archive's file:// URL (archive_info present), matched to the
pinned archive by name. Any other install of the upstream packages (PyPI, a different commit, an editable checkout) is refused. The stack
is then PROVEN BY IMPORT (`import_probe`): every module upstream binds from the accelerators the stack pins (esm/models/esmc/kernels.py:27-61
— `flash_attn`, its CUDA interface, `bert_padding`, the Triton rotary; `transformer_engine`, `transformer_engine.pytorch`) is imported in a
fresh `python -I` child after torch; a module that does not import, or whose package version differs from the pin, is named with its
import error — metadata says installed, the import says usable. This file itself is standard library only, so run.sh, the configs and
the package can all call it; it is the one place the pin check lives.
"""
import importlib.metadata as md
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def _version(name):
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def check(pins):
    """Every pinned upstream package against its installed distribution metadata. Returns ``(bad, detail)``: ``bad`` = one line per
    package not at its pin (empty = all pinned); ``detail`` = per package ``{version, commit, archive_url, pinned, source}``."""
    bad, detail = [], {}
    for name, pin in pins.items():
        try:
            dist = md.distribution(name)
        except md.PackageNotFoundError:
            bad.append(f"{name}: not installed (want {pin['repo']} @ {pin['commit']})")
            detail[name] = {"version": None, "commit": None, "archive_url": None, "pinned": False, "source": "not installed"}
            continue
        raw = dist.read_text("direct_url.json")
        info = json.loads(raw) if raw else {}
        commit = (info.get("vcs_info") or {}).get("commit_id")
        archive_url = info.get("url") if info.get("archive_info") is not None else None
        from_our_archive = bool(archive_url and os.path.basename(archive_url) == os.path.basename(pin["archive"]))
        pinned = bool(commit == pin["commit"] or from_our_archive)
        how = f"commit {commit}" if commit else (f"archive {os.path.basename(archive_url)}" if archive_url else "a non-git, non-archive source")
        detail[name] = {"version": dist.version, "commit": commit, "archive_url": archive_url, "pinned": pinned, "source": how}
        if not pinned:
            bad.append(f"{name} {dist.version}: installed from {how}; want {pin['repo']} @ {pin['commit']} (or {pin['archive']})")
    return bad, detail


def stack(pins_all):
    """The installed stack against stock/PINS.json "stacks": accel (flash_attn + transformer_engine at their pins). Returns (name, detail,
    drift): ``name`` = "accel" at the pins, else None; ``drift`` = one line per difference (a version off its pin or absent, xformers
    present) — NAMED by the caller, never a refusal."""
    fa, te, xf = _version("flash_attn"), _version("transformer_engine"), _version("xformers")
    acc = pins_all["stacks"]["accel"]
    detail = {"flash_attn": fa, "transformer_engine": te, "xformers": xf}
    drift = []
    if xf is not None:
        drift.append(f"xformers {xf} is installed (the pinned stack has it absent: upstream binds it when importable)")
    if fa == acc["flash_attn"] and te == acc["transformer_engine"]:
        return "accel", detail, drift
    drift.append(f"not the pinned stack: flash_attn={fa} transformer_engine={te} (accel wants flash_attn {acc['flash_attn']} + transformer_engine {acc['transformer_engine']}: upstream's `accel` extra, environment/requirements.lock)")
    return None, detail, drift


# The modules esm/models/esmc/kernels.py binds from each accelerator a stack may pin (kernels.py:27-33 transformer_engine; :43-61 flash_attn,
# whose flash_attn_interface loads the CUDA extension, bert_padding the varlen helpers, ops.triton.rotary the RoPE kernel).
ACCEL_IMPORTS = {
    "flash_attn": ("flash_attn", "flash_attn.flash_attn_interface", "flash_attn.bert_padding", "flash_attn.ops.triton.rotary"),
    "transformer_engine": ("transformer_engine", "transformer_engine.pytorch"),
}
_PROBE = r"""
import importlib, json, sys
mods = json.loads(sys.argv[1])
out = {}
try:
    import torch
    out["torch"] = {"ok": True, "version": torch.__version__, "cuda": torch.version.cuda}
except Exception as e:
    out["torch"] = {"ok": False, "error": "%s: %s" % (type(e).__name__, e)}
for m in mods:
    try:
        mod = importlib.import_module(m)
        out[m] = {"ok": True, "version": getattr(mod, "__version__", None), "file": getattr(mod, "__file__", None)}
    except Exception as e:
        out[m] = {"ok": False, "error": ("%s: %s" % (type(e).__name__, e))[:400]}
print("PROBE " + json.dumps(out))
"""


def import_probe(name, pins_all, python=None, timeout=900, runner=None):
    """Prove the named stack by import: every module of ACCEL_IMPORTS for the accelerators the stack pins, imported after torch in a fresh
    `python -I` child (``runner(cmd) -> (returncode, stdout, stderr)`` replaces the child in tests). Returns (bad, drift, detail): ``bad`` =
    torch does not import, or the probe did not report (nothing can run: refused); ``drift`` = one line per accelerator module that fails to
    import (its error named) or whose package `__version__` differs from the pin (a local `+tag` aside) — named, never refused."""
    st = pins_all["stacks"][name]
    want = {a: st.get(a) for a in ACCEL_IMPORTS}
    mods = [m for a, ms in ACCEL_IMPORTS.items() if want[a] for m in ms]
    if not mods:
        return [], [], {"stack": name, "modules": {}, "note": "the stack pins no accelerator: nothing to import"}
    cmd = [python or sys.executable, "-I", "-c", _PROBE, json.dumps(mods)]
    if runner is None:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        rc, out, err = r.returncode, r.stdout, r.stderr
    else:
        rc, out, err = runner(cmd)
    line = next((l for l in out.splitlines() if l.startswith("PROBE ")), None)
    if line is None:
        return [f"import probe of stack {name} did not report (rc={rc}): {(err or out).strip()[-300:]}"], [], {"stack": name, "modules": {}, "rc": rc}
    res = json.loads(line[len("PROBE "):])
    bad, drift = [], []
    if not (res.get("torch") or {}).get("ok"):
        bad.append(f"torch does not import: {(res.get('torch') or {}).get('error')}")
    for m in mods:
        d = res.get(m) or {"ok": False, "error": "not reported"}
        if not d.get("ok"):
            drift.append(f"{m}: import fails on this stack ({d.get('error')}) — stack {name} pins it usable; a lever that needs it steps aside by name")
    for accel, ver in want.items():
        if not ver:
            continue
        got = (res.get(accel) or {}).get("version")
        if got is not None and str(got).split("+")[0] != str(ver).split("+")[0]:
            drift.append(f"{accel}: imports as version {got}; stack {name} pins {ver}")
    return bad, drift, {"stack": name, "modules": res, "rc": rc}


def weights(pins_all, root, variant=None):
    """Every pinned weights file of the variant (or of every variant) under <root>/hf (the HF cache layout: stock/PINS.json weights_layout) at its
    sha256. Returns (bad, n_checked)."""
    import hashlib
    bad, n = [], 0
    for v, V in pins_all["variants"].items():
        if variant and v != variant:
            continue
        repo = V["hf_repo"]; W = pins_all["weights"][repo]
        snap = os.path.join(root, "hf", "hub", "models--" + repo.replace("/", "--"), "snapshots", W["snapshot_commit"])
        for f in W["weights_files"]:
            want = W["files"][f]["sha256"]; p = os.path.join(snap, f)
            if not os.path.exists(p):
                bad.append(f"{repo}/{f}: missing at {p}"); continue
            h = hashlib.sha256()
            with open(p, "rb") as fh:
                for c in iter(lambda: fh.read(1 << 24), b""):
                    h.update(c)
            n += 1
            if h.hexdigest() != want:
                bad.append(f"{repo}/{f}: sha256 {h.hexdigest()[:16]}… != pinned {want[:16]}…")
    return bad, n


def main():
    pins_all = json.load(open(os.path.join(HERE, "PINS.json")))
    argv = sys.argv[1:]
    quiet = "--quiet" in argv
    bad, detail = check(pins_all["upstream"])
    name, sdetail, drift = stack(pins_all)
    ibad, idrift, idetail = import_probe("accel", pins_all)                            # the pinned stack's modules, whatever is installed: torch failing is a refusal, an accelerator failing is drift
    bad += ibad
    drift += idrift
    if "--weights" in argv:
        root = argv[argv.index("--weights") + 1]
        variant = argv[argv.index("--variant") + 1] if "--variant" in argv else None
        wbad, n = weights(pins_all, root, variant)
        bad += wbad
        if not quiet:
            print(f"weights {root}: {n} files at their sha256" if not wbad else f"weights {root}: {len(wbad)} finding(s)")
    if not quiet:
        for n, d in detail.items():
            if d["pinned"]:
                print(f"{n} {d['version']}: pinned ({d['source']})")
        mods = (idetail or {}).get("modules") or {}
        imported = ",".join(m for m, d in mods.items() if m != "torch" and d.get("ok")) or "none"
        print(f"stack {name or 'drift'}: flash_attn={sdetail['flash_attn']} transformer_engine={sdetail['transformer_engine']} xformers={sdetail['xformers']} imported={imported}")
    for d in drift:                                                                     # environment drift: NAMED (stderr), the exit unaffected — the modes run and state it on their lines
        print(f"check_pins: drift (named, not refused): {d}", file=sys.stderr)
    if bad:
        for b in bad:
            print(f"check_pins: {b}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
