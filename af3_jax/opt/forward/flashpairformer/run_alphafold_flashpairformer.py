#!/usr/bin/env python
"""Launcher: run the stock run_alphafold.py with the add-on active, without editing it.
   AF3_FLASHPAIRFORMER=both python run_alphafold_flashpairformer.py /path/to/run_alphafold.py --json_path=... (all stock flags)
Equivalent to adding `import af3_flashpairformer` at the top of run_alphafold.py."""
import os, runpy, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import af3_flashpairformer  # noqa: F401  (installs according to AF3_FLASHPAIRFORMER, default 'both')
if len(sys.argv) < 2 or not sys.argv[1].endswith(".py"):
    sys.exit("usage: run_alphafold_flashpairformer.py /path/to/run_alphafold.py [run_alphafold flags...]")
script = sys.argv[1]; sys.argv = [script] + sys.argv[2:]
runpy.run_path(script, run_name="__main__")
