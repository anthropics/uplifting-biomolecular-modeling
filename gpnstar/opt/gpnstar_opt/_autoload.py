"""The environment route: ``GPNSTAR_OPT=exact`` in front of the unchanged stock command. The package's .pth (site-packages, written by
``pip install -e``; a guarded ``import gpnstar_opt._autoload``: opt/_build_backend.py pth_text) imports this module at interpreter start.

With GPNSTAR_OPT=exact the core's meta-path finder (opt_core.autoload) waits for the first import of upstream's inference module
``gpn.star.inference`` (what ``gpn star vep|logits|embedding`` imports at dispatch) and, right after that module's own body has executed,
calls ``gpnstar_opt.enable("exact", strict=True)``: the gates run and the three wrapper constructors are armed, so the levers apply to every
model they build once it is on the GPU. When the kit cannot engage the package prints its NOT ACTIVE line and the process exits 3 — stock
never runs silently under a set switch. Until the trigger nothing else is imported; with GPNSTAR_OPT unset or ``off`` nothing is installed at
all (a stock process loads nothing of the core through this file); any other value is refused at the trigger
(``[gpnstar-opt] NOT ACTIVE: unknown GPNSTAR_OPT='<value>' (expected exact|off)``, exit 3).
"""
import os
import sys

ENV, MODE, EXIT_NOT_ACTIVE = "GPNSTAR_OPT", "exact", 3          # import-free copies (_names.ENV, _names.MODE, _names.EXIT_NOT_ACTIVE): this file runs in every interpreter
FINDER = None
_word = (os.environ.get(ENV) or "").strip().lower()
if _word not in ("", "off"):
    try:
        from ._core_gate import gate as _gate
        _gate(__file__, "gpnstar-opt")                    # an absent / stale core: one NOT ACTIVE line, exit 3 (CoreGateRefused is a SystemExit)
        from opt_core.autoload import AutoloadSpec, install
    except ImportError:
        sys.stderr.write("[gpnstar-opt] NOT ACTIVE: opt_core not importable under GPNSTAR_OPT=%s (install the core beside this kit: run.sh install)\n" % _word)
        sys.stderr.flush()
        os._exit(EXIT_NOT_ACTIVE)
    FINDER = install(AutoloadSpec(env=ENV, package="gpnstar_opt", tag="gpnstar-opt", triggers=("gpn.star.inference",),
                                  modes=(MODE, "off"), exit_not_active=EXIT_NOT_ACTIVE))
