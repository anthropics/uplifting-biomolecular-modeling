"""build — the ONE recipe that turns kernels.cu into the extension binary, for one target compute capability: torch's own JIT
extension builder (torch.utils.cpp_extension.load: ninja driving the CUDA toolkit's nvcc and the host compiler) with the flags
EXTENSION.json `build` names and TORCH_CUDA_ARCH_LIST set to that one capability (sm90 -> "9.0", sm80 -> "8.0"). Every binary of the
kit comes out of this function — at install (`python -m esmc_opt.kits.residual_ln.build`, run by `run.sh install`) or at the first
apply on a machine (residual_ln.load_ext); same source, same flags, same builder.

  python -m esmc_opt.kits.residual_ln.build              build into the per-user cache for device 0's capability (no device: sm80 + sm90)
  python -m esmc_opt.kits.residual_ln.build --sm 80,90   … for the listed capabilities
  python -m esmc_opt.kits.residual_ln.build --out DIR    build into DIR and print the record (the cache is left alone)

torch is imported inside the functions only: the module imports on a CPU host without it (the kit tests read the recipe there).
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PIN_FILE = "EXTENSION.json"
ARCH_ENV = "TORCH_CUDA_ARCH_LIST"


def pin(kit_dir: str = HERE) -> dict:
    """EXTENSION.json whole: {extension, source, build: {tool, extra_cuda_cflags, extra_cflags, extra_ldflags}}."""
    with open(os.path.join(kit_dir, PIN_FILE)) as f:
        return json.load(f)


def recipe(kit_dir: str = HERE) -> dict:
    """The build recipe read from EXTENSION.json: the extension name, the source file and the builder's flag lists (nothing else)."""
    p = pin(kit_dir)
    b = dict(p["build"])
    return {"extension": p["extension"], "source": p["source"], "tool": b["tool"],
            "extra_cuda_cflags": list(b.get("extra_cuda_cflags") or []), "extra_cflags": list(b.get("extra_cflags") or []),
            "extra_ldflags": list(b.get("extra_ldflags") or [])}


def source_sha256(kit_dir: str = HERE) -> str:
    with open(os.path.join(kit_dir, pin(kit_dir)["source"]), "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def cache_tag(kit_dir: str = HERE) -> str:
    """sha256 over the source bytes AND the recipe (canonical JSON): the identity of what an in-place build produces for a given ABI key —
    a changed kernels.cu or a changed flag list never meets a stale cached binary."""
    with open(os.path.join(kit_dir, pin(kit_dir)["source"]), "rb") as f:
        h = hashlib.sha256(f.read())
    h.update(json.dumps(recipe(kit_dir), sort_keys=True).encode())
    return h.hexdigest()


def key_dir(key: dict) -> str:
    """The one spelling of an ABI key as a name: torch<torch>__cuda<cuda>__sm<sm>__py<python> (the cache dir's name)."""
    return f"torch{key['torch']}__cuda{key['cuda']}__sm{key['sm']}__py{key['python']}"


def arch_of(sm: str) -> str:
    """The TORCH_CUDA_ARCH_LIST spelling of one capability: sm '90' -> '9.0', '80' -> '8.0', '120' -> '12.0'."""
    sm = str(sm)
    if not sm.isdigit() or len(sm) < 2:
        raise ValueError(f"sm {sm!r}: the compute capability digits, e.g. '90'")
    return f"{sm[:-1]}.{sm[-1]}"


def runtime_key() -> dict:
    """The ABI key of THIS process: {torch, cuda, sm, python} ('sm' = the digits of device 0's capability; '?' without a CUDA device)."""
    import torch
    sm = "".join(str(x) for x in torch.cuda.get_device_capability(0)) if torch.cuda.is_available() else "?"
    return {"torch": torch.__version__, "cuda": str(torch.version.cuda)[:4], "sm": sm, "python": f"{sys.version_info.major}.{sys.version_info.minor}"}


def _nvcc_version(nvcc: str) -> str | None:
    try:
        out = subprocess.run([nvcc, "--version"], capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    lines = [l.strip() for l in out.stdout.splitlines() if l.strip()]
    return lines[-1] if lines else None      # "Build cuda_13.0.r13.0/compiler.36424714_0" — the form the build record's `nvcc` field keeps


def stack_cuda_include() -> list:
    """The CUDA library headers of THIS stack for the compile (cublas / cusparse / cusolver / cudart, which ATen's CUDA headers include):
    the `nvidia` wheels torch itself runs against ship them under <site-packages>/nvidia/cu<major>/include — named beside CUDA_HOME's
    include dir so a toolkit install of nvcc alone (no library dev packages) compiles kernels.cu. [] when the wheels carry no headers."""
    import sysconfig
    out = []
    for base in {sysconfig.get_paths().get("purelib"), sysconfig.get_paths().get("platlib")}:
        if not base: continue
        root = os.path.join(base, "nvidia")
        if not os.path.isdir(root): continue
        for d in sorted(os.listdir(root)):
            inc = os.path.join(root, d, "include")
            if d.startswith("cu") and os.path.isfile(os.path.join(inc, "cublas_v2.h")) and inc not in out:
                out.append(inc)
    return out


def toolchain() -> dict:
    """What the build needs on this image, each found or not: the CUDA toolkit's nvcc (torch's CUDA_HOME/bin/nvcc), ninja (torch's own
    probe), a host C++ compiler, the Python headers. `missing` names the absent ones; empty = the build can run."""
    import sysconfig
    import torch.utils.cpp_extension as ce
    cuda_home = ce.CUDA_HOME
    nvcc = os.path.join(cuda_home, "bin", "nvcc") if cuda_home else None
    nvcc = nvcc if nvcc and os.path.isfile(nvcc) else None
    nvcc_version = _nvcc_version(nvcc) if nvcc else None
    try:
        ninja = bool(ce.is_ninja_available())
    except Exception:  # noqa: BLE001
        ninja = False
    cxx = shutil.which("c++") or shutil.which("g++")
    inc = sysconfig.get_paths().get("include") or ""
    python_h = os.path.isfile(os.path.join(inc, "Python.h"))
    missing = []
    if not nvcc or not nvcc_version:
        missing.append(f"nvcc (the CUDA toolkit compiler; CUDA_HOME={cuda_home})")
    if not ninja:
        missing.append("ninja")
    if not cxx:
        missing.append("a host C++ compiler (c++/g++)")
    if not python_h:
        missing.append(f"the Python headers ({inc}/Python.h)")
    return {"cuda_home": cuda_home, "nvcc": nvcc, "nvcc_version": nvcc_version, "ninja": ninja, "cxx": cxx, "python_h": python_h, "missing": missing}


def build(build_dir: str, sm: str, kit_dir: str = HERE, verbose: bool = False) -> dict:
    """Build kernels.cu for capability `sm` into `build_dir` by the recipe and import the result. Returns {module, so_path, so_sha256,
    size, wall_s, arch, nvcc, recipe}. TORCH_CUDA_ARCH_LIST is set to the one capability for the duration of the call and restored.
    Raises what the builder raises (RuntimeError carrying the compiler output on a failed compile)."""
    import torch.utils.cpp_extension as ce
    r = recipe(kit_dir)
    if r["tool"] != "torch.utils.cpp_extension.load":
        raise ValueError(f"{PIN_FILE} build.tool {r['tool']!r}: this recipe drives torch.utils.cpp_extension.load only")
    arch = arch_of(sm)
    os.makedirs(build_dir, exist_ok=True)
    prev = os.environ.get(ARCH_ENV)
    os.environ[ARCH_ENV] = arch
    t0 = time.time()
    try:
        mod = ce.load(name=r["extension"], sources=[os.path.join(kit_dir, r["source"])], extra_cflags=r["extra_cflags"],
                      extra_cuda_cflags=r["extra_cuda_cflags"], extra_ldflags=r["extra_ldflags"], extra_include_paths=stack_cuda_include(),
                      build_directory=build_dir, verbose=verbose, is_python_module=True)
    finally:
        if prev is None:
            os.environ.pop(ARCH_ENV, None)
        else:
            os.environ[ARCH_ENV] = prev
    wall = round(time.time() - t0, 2)
    so_path = os.path.join(build_dir, r["extension"] + ".so")
    if not os.path.isfile(so_path):     # the builder names its product <name><LIB_EXT>; a differing suffix is found by prefix
        cands = sorted(f for f in os.listdir(build_dir) if f.startswith(r["extension"]) and f.endswith((".so", ".pyd")))
        if not cands:
            raise RuntimeError(f"the build returned but no {r['extension']}.so is in {build_dir}")
        so_path = os.path.join(build_dir, cands[0])
    with open(so_path, "rb") as f:
        so_sha = hashlib.sha256(f.read()).hexdigest()
    tc = toolchain()
    return {"module": mod, "so_path": so_path, "so_sha256": so_sha, "size": os.path.getsize(so_path), "wall_s": wall, "arch": f"{ARCH_ENV}={arch}",
            "nvcc": tc.get("nvcc_version"), "recipe": r, "cuda_include": stack_cuda_include()}


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="python -m esmc_opt.kits.residual_ln.build", description="build the fused residual+LayerNorm extension from kernels.cu by the EXTENSION.json recipe")
    ap.add_argument("--sm", default=None, help="the target compute capabilities, comma-separated digits (default: device 0's; with no device: 80,90)")
    ap.add_argument("--out", default=None, help="build into this directory and print the record instead of filling the per-user cache (one --sm only)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    sms = [x.strip() for x in a.sm.split(",") if x.strip()] if a.sm else None
    if a.out:
        sm = (sms or [runtime_key()["sm"]])[0]
        if sm == "?":
            print("build: no CUDA device and no --sm: name the target capability", file=sys.stderr)
            return 2
        rec = build(os.path.abspath(a.out), sm, verbose=not a.quiet)
        rec.pop("module", None)
        print(json.dumps(rec, indent=1))
        return 0
    from esmc_opt.kits import residual_ln
    try:
        residual_ln.build_cache(sms)
    except residual_ln.KitRefused as e:
        print(f"build: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
