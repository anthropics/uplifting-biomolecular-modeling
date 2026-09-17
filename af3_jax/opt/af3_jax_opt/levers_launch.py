#!/usr/bin/env python
"""levers_launch.py <levers dir> <script.py> [flags...] — run <script.py> as __main__ with the Pallas add-on's levers installed first. The
tree's composition point for the exact mode: it installs the levers (``af3_pallas_levers.apply_levers()``, selected by the add-on's
``AF3P_*`` switches the mode table sets) and then runs the given script. It prints the sha256 of the lever module and script first.
Standard library only: run by the fork's interpreter, never imports af3_jax_opt.
"""
import hashlib
import os
import runpy
import sys

PREFIX = "[af3-jax-opt]"


# stdlib on purpose: this launcher runs in the fork's interpreter, where neither the wrapper package nor the core is INSTALLED — the core's
# directory is on PYTHONPATH for the add-on's carried-kernel import only (stack.model_process_env); the sha lines print before that import
def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def install(levers_dir, script):
    """Install the add-on's levers from <levers dir> (``apply_levers()``, selected by the AF3P_* variables), print the LEVERS line, refuse the
    pair-attention chunk lever by name; returns the add-on's lever dict. Shared by main() (mode exact) and big_launch.py (mode big: the
    fallback kernels at the sites the disengaged fast levers vacate) — one installation, one LEVERS line."""
    levers_dir, script = os.path.abspath(levers_dir), os.path.abspath(script)
    module = os.path.join(levers_dir, "af3_pallas_levers.py")
    for p in (module, script):
        if not os.path.isfile(p):
            sys.exit(f"{PREFIX} LEVERS FAILED: no such file {p}")
    sys.path.insert(0, os.path.dirname(script))
    sys.path.insert(0, levers_dir)
    import af3_pallas_levers  # noqa: E402 - the add-on's module, found through levers_dir
    levers = af3_pallas_levers.apply_levers()
    active = sorted(k for k in levers if not k.startswith("_"))
    print(f"{PREFIX} LEVERS active={'+'.join(active) or 'none'} af3_pallas_levers.py={sha256(module)} script={os.path.basename(script)} "
          f"sha256={sha256(script)}", flush=True)
    return levers


def main(argv):
    if len(argv) < 3 or not argv[2].endswith(".py"):
        sys.exit(f"usage: {os.path.basename(argv[0])} <levers dir> <script.py> [flags...]")
    script = os.path.abspath(argv[2])
    install(argv[1], script)
    from fpf_launch import install_cache_key, install_template_guard     # the sibling launcher's hooks (this directory is the script's: sys.path[0] at start)
    install_cache_key()                                                   # before the script compiles anything (the levers above import, they do not compile)
    install_template_guard()
    sys.argv = [script] + list(argv[3:])
    runpy.run_path(script, run_name="__main__")


if __name__ == "__main__":
    main(sys.argv)
