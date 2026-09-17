#!/usr/bin/env python3
"""Check an environment against the E1 stock pin (stock/PINS.json). Standard library only; imports no model code.

    python stock/check_pins.py [--variant 150m|300m|600m|all] [--checks package,stack,weights,kernel,env] [--with-torch] [--stock] [--json report.json]

Checks (each a named line; any drift -> exit 3, the JSON report carries every reading):
  package   E1 is installed at the pinned commit: PEP 610 direct_url.json (vcs commit; an archive/dir install route is not the pinned
            route), the distribution version, and every vendored src/E1/... module present in the installed package directory
  stack     the pinned versions of torch, triton, transformers, tokenizers, kernels, flash-attn and the Python minor version
            (--with-torch also IMPORTS, in a subprocess, torch — torch.__version__ / torch.version.cuda — and the two accelerator
            modules the model imports, flash_attn (its flash_attn_varlen_func, its compiled extension flash_attn_2_cuda loaded) and
            kernels (get_kernel): a distribution at the pinned version whose module does not import is upstream's silent fallback
            route, a DRIFT here — stack.flash_attn_import / stack.kernels_import)
  weights   model.safetensors of the requested variant(s) under the Hugging Face cache at the pinned snapshot: bytes and sha256
  kernel    the hub RMSNorm kernel snapshot at the pinned revision (the kernels package's cache or the hub cache): layer_norm.py sha256
  env       the kit variables (informational; a drift under --stock) and the stock's own route variables at their defaults
--checks names the checks to run (default: all five); `run.sh install` runs `--checks package,stack` — the software pins — before any
weights are staged, and `run.sh install --weights DIR` then checks the staged files.
Exit 0: ok. Exit 3: at least one DRIFT line. Exit 2: usage.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import importlib.metadata as md
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
KIT_PREFIXES = ("E1_OPT", "E1_KIT", "E1_VARIANT", "MODEL_OPT")
STOCK_ROUTE_DEFAULTS = {"USE_FLASH_ATTN": "1", "FAST_BLOCK_MASK": "1"}
CHECKS = ("package", "stack", "weights", "kernel", "env")      # the named checks, in the order they run; --checks selects a subset
STACK_DISTS = {"torch": ("torch",), "triton": ("triton",), "transformers": ("transformers",), "tokenizers": ("tokenizers",),
               "kernels": ("kernels",), "flash_attn": ("flash_attn", "flash-attn")}


def sha256_of(path: str, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()


def dist_version(names) -> tuple[str | None, object]:
    for n in names:
        try:
            d = md.distribution(n)
            return d.version, d
        except md.PackageNotFoundError:
            continue
    return None, None


def expand(p: str | None) -> str | None:
    return os.path.abspath(os.path.expanduser(os.path.expandvars(p))) if p else None


def hub_roots() -> list[str]:
    """Every root the weights / kernel snapshot may live under (first hit wins): the explicit caches, then the defaults."""
    roots = [expand(os.environ.get("HF_HUB_CACHE")),
             expand(os.path.join(os.environ["HF_HOME"], "hub")) if os.environ.get("HF_HOME") else None,
             expand("~/.cache/huggingface/hub")]
    return [r for r in roots if r]


def kernel_roots() -> list[str]:
    roots = [expand(os.environ.get("KERNELS_CACHE")), expand(os.environ.get("HF_KERNELS_CACHE"))] + hub_roots()
    seen, out = set(), []
    for r in roots:
        if r and r not in seen:
            seen.add(r); out.append(r)
    return out


class Report:
    def __init__(self):
        self.checks: dict = {}
        self.drift: list[str] = []

    def ok(self, check: str, **kw):
        self.checks.setdefault(check, {}).update(kw, ok=True)
        print(f"check_pins: ok {check}: " + ", ".join(f"{k}={v}" for k, v in kw.items() if k != "detail"), flush=True)

    def bad(self, check: str, why: str, **kw):
        self.checks.setdefault(check, {}).update(kw, ok=False, why=why)
        self.drift.append(f"{check}: {why}")
        print(f"check_pins: DRIFT {check}: {why}", flush=True)


def check_package(rep: Report, pins: dict) -> None:
    up = pins["upstream"]["E1"]
    ver, d = dist_version(("E1", "e1"))
    if d is None:
        rep.bad("package", "E1 is not installed (no distribution metadata)"); return
    if ver != up["version"]:
        rep.bad("package.version", f"E1 {ver} installed, pin {up['version']}")
    else:
        rep.ok("package.version", version=ver)
    # PEP 610
    route = None
    try:
        raw = d.read_text("direct_url.json")
    except Exception:
        raw = None
    if raw:
        du = json.loads(raw)
        if "vcs_info" in du:
            cid = du["vcs_info"].get("commit_id")
            route = f"vcs {du.get('url')}@{cid}"
            if cid == up["commit"]:
                rep.ok("package.commit", commit=cid, route="vcs")
            else:
                rep.bad("package.commit", f"installed from {du.get('url')} at {cid}, pin {up['commit']}")
        elif "archive_info" in du:
            hsh = str(du["archive_info"].get("hash", ""))
            route = f"archive {du.get('url')} {hsh}"
            rep.bad("package.commit", f"installed from an archive ({hsh or 'no hash'}): the kit's pins gate reads the commit from a git install record only — install {up['install']}")
        elif "dir_info" in du:
            route = f"dir {du.get('url')} editable={du['dir_info'].get('editable')}"
            rep.checks.setdefault("package.commit", {})["route"] = route
    if route is None:
        rep.checks.setdefault("package.commit", {})["route"] = "no direct_url.json (a wheel or an editable install without PEP 610 data)"
    # the installed package directory carries every module vendored at stock/src/E1 in this repo (the pin is the commit
    # checked above; this is a presence check, not a byte comparison — git already tracks the vendored copies' bytes)
    pkg_dir = None
    for f in d.files or []:
        s = str(f)
        if s.replace(os.sep, "/").endswith("E1/modeling.py"):
            pkg_dir = os.path.dirname(str(d.locate_file(f)))
            break
    if pkg_dir is None:
        rep.bad("package.source", "the E1 package directory could not be located from the distribution's RECORD"); return
    vendor_dir = os.path.join(HERE, "src", "E1")
    vendored = [os.path.relpath(os.path.join(root, fn), vendor_dir).replace(os.sep, "/")
                for root, _, files in os.walk(vendor_dir) for fn in files if fn.endswith(".py")]
    missing, checked = [], 0
    for rel in sorted(vendored):
        checked += 1
        if not os.path.exists(os.path.join(pkg_dir, rel)):
            missing.append(rel)
    if missing:
        rep.bad("package.source", f"{len(missing)} of {checked} vendored modules are missing from the installed package: " + "; ".join(missing[:6]), package_dir=pkg_dir)
    else:
        rep.ok("package.source", files=checked, package_dir=pkg_dir)


def check_stack(rep: Report, pins: dict, with_torch: bool) -> None:
    want = pins["pinned_stack"]
    py = f"{sys.version_info.major}.{sys.version_info.minor}"
    if py == want["python"]:
        rep.ok("stack.python", python=sys.version.split()[0])
    else:
        rep.bad("stack.python", f"python {sys.version.split()[0]}, pin {want['python']}.x")
    for key, names in STACK_DISTS.items():
        ver, _ = dist_version(names)
        if ver is None:
            rep.bad(f"stack.{key}", f"{names[0]} is not installed, pin {want[key]}")
        elif ver != want[key]:
            rep.bad(f"stack.{key}", f"{names[0]} {ver} installed, pin {want[key]}")
        else:
            rep.ok(f"stack.{key}", version=ver)
    if with_torch:
        code = ("import json, sys, torch\n"
                "r = {'v': torch.__version__, 'cuda': torch.version.cuda}\n"
                "try:\n"
                "    import flash_attn\n"
                "    r['flash_attn'] = {'ok': True, 'version': str(flash_attn.__version__), 'varlen': getattr(flash_attn, 'flash_attn_varlen_func', None) is not None, 'ext': 'flash_attn_2_cuda' in sys.modules}\n"
                "except Exception as e:\n"
                "    r['flash_attn'] = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:200]}'}\n"
                "try:\n"
                "    import kernels\n"
                "    r['kernels'] = {'ok': hasattr(kernels, 'get_kernel'), 'version': str(getattr(kernels, '__version__', None)), 'error': None if hasattr(kernels, 'get_kernel') else 'no get_kernel'}\n"
                "except Exception as e:\n"
                "    r['kernels'] = {'ok': False, 'error': f'{type(e).__name__}: {str(e)[:200]}'}\n"
                "print(json.dumps(r))\n")
        try:
            out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)
            got = json.loads(out.stdout.strip().splitlines()[-1])
        except Exception as e:  # noqa: BLE001
            rep.bad("stack.torch_build", f"torch could not be imported in a subprocess ({type(e).__name__}: {e})"); return
        if got["v"] == want["torch_version_str"] and got["cuda"] == want["cuda"]:
            rep.ok("stack.torch_build", torch=got["v"], cuda=got["cuda"])
        else:
            rep.bad("stack.torch_build", f"torch {got['v']} cuda {got['cuda']}, pin {want['torch_version_str']} cuda {want['cuda']}")
        fa = got.get("flash_attn") or {}
        if not fa.get("ok"):
            rep.bad("stack.flash_attn_import", f"flash_attn does not import ({fa.get('error')}): the model would take upstream's varlen-flex fallback (src/E1/model/attention.py L299-311)")
        elif fa.get("version") != want["flash_attn"] or not fa.get("varlen"):
            rep.bad("stack.flash_attn_import", f"flash_attn imports as {fa.get('version')} varlen_func={fa.get('varlen')}, pin {want['flash_attn']}")
        else:
            rep.ok("stack.flash_attn_import", version=fa["version"], flash_attn_2_cuda=fa.get("ext"))
        kr = got.get("kernels") or {}
        if not kr.get("ok"):
            rep.bad("stack.kernels_import", f"kernels does not import ({kr.get('error')}): src/E1/modeling.py L22-26 would fall back to torch rms_norm")
        else:
            rep.ok("stack.kernels_import", version=kr.get("version"))


def check_weights(rep: Report, pins: dict, variants: list[str]) -> None:
    roots = hub_roots()
    for size in variants:
        w = pins["weights"][size]
        rel = os.path.join("models--" + w["repo"].replace("/", "--"), "snapshots", w["rev"], w["file"])
        hits = [os.path.join(r, rel) for r in roots if os.path.exists(os.path.join(r, rel))]
        if not hits:
            rep.bad(f"weights.{size}", f"{rel} not found under {roots}"); continue
        p = hits[0]
        nbytes = os.path.getsize(p)
        if nbytes != w["bytes"]:
            rep.bad(f"weights.{size}", f"{p}: {nbytes} bytes, pin {w['bytes']}"); continue
        digest = sha256_of(p)
        if digest != w["sha256"]:
            rep.bad(f"weights.{size}", f"{p}: sha256 {digest[:16]}..., pin {w['sha256'][:16]}..."); continue
        rep.ok(f"weights.{size}", path=p, bytes=nbytes, sha256=digest[:16] + "...")


def check_kernel(rep: Report, pins: dict) -> None:
    k = pins["hub_kernel"]
    snap_rel = os.path.join("models--" + k["repo"].replace("/", "--"), "snapshots", k["rev"])
    roots = kernel_roots()
    found = []
    for r in roots:
        snap = os.path.join(r, snap_rel)
        if os.path.isdir(snap):
            pref = os.path.join(snap, k["file"])
            cands = [pref] if os.path.exists(pref) else sorted(glob.glob(os.path.join(snap, "**", "layer_norm.py"), recursive=True))
            for c in cands:
                found.append((c, sha256_of(c), os.path.getsize(c)))
    if not found:
        rep.bad("kernel", f"snapshot {snap_rel} not found under {roots}"); return
    good = [f for f in found if f[1] == k["layer_norm_py_sha256"]]
    if good:
        rep.ok("kernel", path=good[0][0], bytes=good[0][2], sha256=good[0][1][:16] + "...")
    else:
        rep.bad("kernel", f"layer_norm.py at the pinned snapshot has sha256 {found[0][1][:16]}... ({found[0][0]}), pin {k['layer_norm_py_sha256'][:16]}...")


def check_env(rep: Report, stock: bool) -> None:
    present = sorted(k for k in os.environ if k.startswith(KIT_PREFIXES))
    if present and stock:
        rep.bad("env.kit_variables", f"set in this process: {present}")
    else:
        rep.ok("env.kit_variables", present=present or "none")
    off = {k: os.environ[k] for k, dflt in STOCK_ROUTE_DEFAULTS.items() if os.environ.get(k, dflt) != dflt}
    if off:
        rep.bad("env.stock_route", f"the stock's own route variables are off their defaults: {off}")
    else:
        rep.ok("env.stock_route", USE_FLASH_ATTN=os.environ.get("USE_FLASH_ATTN", "unset"), FAST_BLOCK_MASK=os.environ.get("FAST_BLOCK_MASK", "unset"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--variant", default="all", choices=["150m", "300m", "600m", "all"])
    ap.add_argument("--with-torch", action="store_true", help="also read torch.__version__ / torch.version.cuda in a subprocess")
    ap.add_argument("--stock", action="store_true", help="a kit variable in the environment is a drift (the stock arm's process)")
    ap.add_argument("--json", default=None, help="write the report here (default: stdout after the lines)")
    ap.add_argument("--checks", default=",".join(CHECKS), help=f"comma-separated subset of {','.join(CHECKS)} (default: all)")
    a = ap.parse_args(argv)
    chosen = [c.strip() for c in a.checks.split(",") if c.strip()]
    unknown = [c for c in chosen if c not in CHECKS]
    if unknown or not chosen:
        ap.error(f"--checks: {', '.join(unknown) or 'nothing'} is not a check ({','.join(CHECKS)})")
    pins = json.load(open(os.path.join(HERE, "PINS.json")))
    rep = Report()
    if "package" in chosen: check_package(rep, pins)
    if "stack" in chosen: check_stack(rep, pins, a.with_torch)
    if "weights" in chosen: check_weights(rep, pins, ["150m", "300m", "600m"] if a.variant == "all" else [a.variant])
    if "kernel" in chosen: check_kernel(rep, pins)
    if "env" in chosen: check_env(rep, a.stock)
    report = {"model": pins["model"], "commit": pins["upstream"]["E1"]["commit"], "ok": not rep.drift, "drift": rep.drift, "checks": rep.checks, "checks_run": chosen,
              "python": sys.version.split()[0], "executable": sys.executable,
              "env": {k: os.environ.get(k) for k in ("HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "KERNELS_CACHE", "PYTHONHASHSEED", "CUDA_VISIBLE_DEVICES")}}
    text = json.dumps(report, indent=1, default=str)
    if a.json:
        with open(a.json, "w") as f:
            f.write(text + "\n")
        print(f"check_pins: report {a.json}")
    else:
        print(text)
    if rep.drift:
        print(f"check_pins: DRIFT ({len(rep.drift)}): " + " | ".join(rep.drift))
        return 3
    print("check_pins: ok (every pin matches)" if chosen == list(CHECKS) else f"check_pins: ok ({','.join(chosen)}: every pin checked matches)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
