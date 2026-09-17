#!/usr/bin/env python
"""The stock route's launcher: runs xfold's ``run_alphafold.py`` (the pinned archive's pristine bytes) as ``__main__`` on the composed stock
venv, after the named adaptations between the fork's package (alphafold3-open 3.1.4, the image's) and the package the CLI was written
against — every one a re-pointing of a name to the fork's own object, none a change of what the CLI computes; each is listed in
``ADAPTATIONS`` and on the wrapper's ``STOCK-CLI`` line (``launcher_compat=``), and this file's sha256 travels on that line too.

    <stock venv python> stock_launch.py <run_alphafold.py> <the CLI's own flags…>

This script imports nothing of the af3_torch_opt package (the stock venv has neither the package nor the core).
"""
import runpy
import sys

ADAPTATIONS = (
    "of3.OF3=True",            # xfold.of3.OF3: the OpenFold3 parameter layout (pristine xfold refuses the checkpoint: xfold/params.py:747-748)
    "loaders-as-lists",        # alphafold3.common.folding_input.load_fold_inputs_from_dir/path return iterators; the CLI takes len() (run_alphafold.py:596)
    "cached_ccd-as-Ccd",       # alphafold3.constants.chemical_components.cached_ccd(user_ccd=) is Ccd(user_ccd=) in the fork (its own CLI, run_alphafold.py:721)
)


def adapt():
    from xfold import of3
    of3.OF3 = True
    from alphafold3.common import folding_input
    for name in ("load_fold_inputs_from_dir", "load_fold_inputs_from_path"):
        setattr(folding_input, name, (lambda f: (lambda *a, **k: list(f(*a, **k))))(getattr(folding_input, name)))
    from alphafold3.constants import chemical_components
    if not hasattr(chemical_components, "cached_ccd"):
        chemical_components.cached_ccd = lambda user_ccd=None: chemical_components.Ccd(user_ccd=user_ccd)


def main():
    if len(sys.argv) < 2:
        sys.exit("stock_launch.py: usage: stock_launch.py <run_alphafold.py> [flags…]")
    adapt()
    sys.argv = sys.argv[1:]
    runpy.run_path(sys.argv[0], run_name="__main__")


if __name__ == "__main__":
    main()
