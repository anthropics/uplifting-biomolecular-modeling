"""ef2inv_opt — explicit interface to the ESMFold2 binder-inversion (hallucination) optimizations of the EF2 inversion-acceleration kit.

The upstream is a script, not an API: the ESM cookbook's ``binder_design.py`` (stock/PINS.json "upstream") — a 150-step gradient-guided
optimization of a soft binder sequence through ESMFold2-Experimental-Fast (+ -Cutoff2025) distograms with an ESMC-6B pseudo-perplexity
term, scored by four hero critics (which every kit mode folds on the stock arm's two switches). The kit switches its lever sets with ONE environment variable (``EF2_FAST_KIT``, modes.py) and is
installed on that file's imported module from outside (fastkit.py). This package is a launcher: ``design`` runs one trajectory per process
through ``launch.py`` -> ``stock_design.py`` (the one caller of the loop, the same bytes for every arm), proves the process environment
(envproof.py), reads the kit's own counters and lines back as evidence and turns every quiet path of the carried kit into a named
refusal (evidence.py). No ``.pth`` autoload ships: nothing attaches to an ``import`` of the upstream packages.

    ef2inv-opt design --mode off|exact|fast|big --target-name NAME [--target-sequence SEQ] --binder-len 80 --seed 0 --out DIR
    ef2inv-opt check | warm            (cli.py)
"""
__version__ = "0.6.3"
from .modes import DEFAULT_MODE, MODES, resolve  # noqa: F401
