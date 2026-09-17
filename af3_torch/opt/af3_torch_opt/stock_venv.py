"""The stock interpreter: one venv that runs xfold's own CLI (``run_alphafold.py``: the fork's data pipeline, featuriser and writers AND torch
inference in one process) on the GPU. Neither kit venv is one — the torch venv (``AF3_TORCH_PY``'s) has no ``alphafold3``/``absl``, the
JAX venv's (``AF3_TORCH_JAX_PY``'s) torch is CPU-only — so the stock venv is COMPOSED on the box from the two venvs' own
bytes, no download: a copy of the JAX venv with every distribution of the torch venv overlaid (the torch venv's versions win: torch, triton,
their CUDA libraries and dependencies; the two venvs share CPython 3.12 and numpy 2.4.1). A preflight in the composed venv asserts the imports
the CLI needs; ``<dest>/stock_venv.json`` is the record the stock verb reads back (strategy, compatibility modules, preflight).
``--strategy uv`` is the named fallback (network): the torch venv copied and the fork's package installed into it from its checkout.
The fork's package layout is not the one xfold's CLI was written against (``alphafold3.model.components.base_model`` and
``alphafold3.model.diffusion.model`` do not exist in it; the same objects live in ``alphafold3.model.model``): the composer writes the
two names as re-export modules into the composed venv's copy of the package (``COMPAT``, listed in the record) — the CLI's bytes are not
touched, the fork's are not touched, the objects are the fork's own.

    python -m af3_torch_opt.stock_venv --dest <dir> [--jax-venv <venv> --torch-venv <venv> --jax-repo <checkout>]
    (defaults: the venvs of AF3_TORCH_JAX_PY / AF3_TORCH_PY and AF3_TORCH_JAX_REPO; a missing one is refused by name)
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from typing import Dict, List, Optional

from . import stack

PREFLIGHT = ("import json, sys; import alphafold3.cpp, alphafold3.model.post_processing, alphafold3.data.featurisation, alphafold3.common.folding_input, "
             "alphafold3.model.components.base_model, alphafold3.model.diffusion.model, alphafold3.model.features, alphafold3.data.pipeline, "
             "jax, absl.app, absl.flags, numpy, torch, triton; "
             "print(json.dumps({'python': sys.version.split()[0], 'torch': torch.__version__, 'triton': triton.__version__, 'jax': jax.__version__, "
             "'numpy': numpy.__version__, 'cuda_available': torch.cuda.is_available(), 'cuda': torch.version.cuda}))")


COMPAT = {   # re-export modules written into the composed venv's alphafold3 package: the names xfold's run_alphafold.py imports, the fork's objects
    "alphafold3/model/components/base_model.py": "from typing import Any, Mapping\nfrom alphafold3.model.model import InferenceResult\nfrom alphafold3.model import model as _model\n"
                                                "ModelResult = getattr(_model, 'ModelResult', Mapping[str, Any])\n",
    "alphafold3/model/diffusion/__init__.py": "",
    "alphafold3/model/diffusion/model.py": "from alphafold3.model.model import Model as Diffuser\n",
}


def write_compat(dest: str) -> List[str]:
    site = site_packages(dest)
    written = []
    for rel, body in COMPAT.items():
        p = os.path.join(site, rel)
        if os.path.exists(p):
            raise SystemExit(f"stock_venv: {p} exists in the fork's package; the compat module would shadow it")
        os.makedirs(os.path.dirname(p), exist_ok=True)
        open(p, "w", encoding="utf-8").write(body); written.append(rel)
    return written


def site_packages(venv: str) -> str:
    cands = sorted(d for d in __import__("glob").glob(os.path.join(venv, "lib", "python3.*", "site-packages")))
    if len(cands) != 1:
        raise SystemExit(f"stock_venv: expected one site-packages under {venv}, found {cands}")
    return cands[0]


def distributions(site: str) -> Dict[str, dict]:
    """{distribution name (normalized): {dist_info, files}} from every *.dist-info/RECORD under a site-packages dir."""
    out = {}
    for d in sorted(os.listdir(site)):
        if not d.endswith(".dist-info"):
            continue
        name = d[: -len(".dist-info")].split("-")[0].lower().replace("_", "-")
        rec = os.path.join(site, d, "RECORD")
        files = []
        if os.path.isfile(rec):
            for ln in open(rec, encoding="utf-8", errors="replace"):
                p = ln.split(",")[0].strip()
                if p and not p.startswith(".."):
                    files.append(p)
        out[name] = {"dist_info": d, "files": files}
    return out


def _remove(site: str, files: List[str]) -> int:
    """Remove a distribution's files (from its RECORD) under site-packages; the top-level dirs left empty go too."""
    n = 0
    tops = set()
    for rel in files:
        p = os.path.join(site, rel)
        tops.add(rel.split("/")[0])
        if os.path.isfile(p) or os.path.islink(p):
            os.remove(p); n += 1
    for t in sorted(tops):
        p = os.path.join(site, t)
        if os.path.isdir(p):
            for d, dirs, fs in os.walk(p, topdown=False):
                if not os.listdir(d):
                    os.rmdir(d)
    return n


def overlay(jax_venv: str, torch_venv: str, dest: str) -> dict:
    t0 = time.time()
    if os.path.exists(dest):
        raise SystemExit(f"stock_venv: {dest} exists; the composed venv is built once per box into a fresh directory")
    shutil.copytree(jax_venv, dest, symlinks=True)
    site_d, site_t = site_packages(dest), site_packages(torch_venv)
    base, top = distributions(site_d), distributions(site_t)
    replaced, added = [], []
    for name, info in top.items():
        if name in base:
            _remove(site_d, base[name]["files"]); shutil.rmtree(os.path.join(site_d, base[name]["dist_info"]), ignore_errors=True)
            replaced.append({"dist": name, "from": base[name]["dist_info"], "to": info["dist_info"]})
        else:
            added.append(info["dist_info"])
        seen = set()
        for rel in info["files"]:
            top_ = rel.split("/")[0]
            if top_ in seen:
                continue
            seen.add(top_)
            src, dst = os.path.join(site_t, top_), os.path.join(site_d, top_)
            if os.path.isdir(src):
                if os.path.isdir(dst):
                    shutil.rmtree(dst)
                shutil.copytree(src, dst, symlinks=True)
            elif os.path.isfile(src):
                shutil.copy2(src, dst)
    return {"strategy": "overlay", "jax_venv": jax_venv, "torch_venv": torch_venv, "dest": dest, "replaced": replaced, "added": added, "compose_s": round(time.time() - t0, 1)}


def uv_install(torch_venv: str, jax_repo: str, dest: str) -> dict:
    t0 = time.time()
    shutil.copytree(torch_venv, dest, symlinks=True)
    py = os.path.join(dest, "bin", "python")
    cmd = ["uv", "pip", "install", "--python", py, jax_repo, "absl-py"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    return {"strategy": "uv", "torch_venv": torch_venv, "jax_repo": jax_repo, "dest": dest, "cmd": cmd, "rc": r.returncode, "tail": (r.stdout + r.stderr)[-2000:], "compose_s": round(time.time() - t0, 1)}


def preflight(dest: str) -> dict:
    py = os.path.join(dest, "bin", "python")
    r = subprocess.run([py, "-c", PREFLIGHT], capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    rep = {"rc": r.returncode, "stderr_tail": r.stderr[-1500:]}
    if r.returncode == 0:
        rep.update(json.loads(r.stdout.strip().splitlines()[-1]))
    return rep


def venv_of(python: Optional[str]) -> Optional[str]:
    """The venv root of an interpreter path (<venv>/bin/python -> <venv>), None for None."""
    return os.path.dirname(os.path.dirname(python)) if python else None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--dest", required=True)
    ap.add_argument("--jax-venv", default=venv_of(stack.jax_python()), help="the JAX venv to copy (default: AF3_TORCH_JAX_PY's venv)")
    ap.add_argument("--torch-venv", default=venv_of(stack.torch_python()), help="the torch venv overlaid on it (default: AF3_TORCH_PY's venv)")
    ap.add_argument("--jax-repo", default=stack.jax_repo(), help="the fork checkout (--strategy uv; default: AF3_TORCH_JAX_REPO)"); ap.add_argument("--strategy", choices=("overlay", "uv"), default="overlay")
    ap.add_argument("--expect-torch", default=None, help="the torch version the composed venv must import (the torch venv's, the kit's pinned stack)")
    a = ap.parse_args(argv)
    need = (("--jax-venv", "AF3_TORCH_JAX_PY", a.jax_venv), ("--torch-venv", "AF3_TORCH_PY", a.torch_venv)) if a.strategy == "overlay" else (("--torch-venv", "AF3_TORCH_PY", a.torch_venv), ("--jax-repo", "AF3_TORCH_JAX_REPO", a.jax_repo))
    missing = [f"{flag} (or {var})" for flag, var, val in need if not val]
    if missing:
        ap.error("nothing names " + ", ".join(missing) + ": the venvs to compose come by flag or from the deployment variables (README Variables)")
    rec = overlay(a.jax_venv, a.torch_venv, a.dest) if a.strategy == "overlay" else uv_install(a.torch_venv, a.jax_repo, a.dest)
    rec["compat_modules"] = write_compat(a.dest)
    rec["preflight"] = preflight(a.dest)
    ok = rec["preflight"]["rc"] == 0 and (a.expect_torch is None or rec["preflight"].get("torch") == a.expect_torch)
    rec["ok"] = ok
    json.dump(rec, open(os.path.join(a.dest, "stock_venv.json"), "w"), indent=1)
    print(json.dumps({k: rec[k] for k in ("strategy", "dest", "compose_s", "compat_modules", "ok")} | {"preflight": {k: v for k, v in rec["preflight"].items() if k != "stderr_tail"}}))
    if not ok:
        print(rec["preflight"]["stderr_tail"], file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
