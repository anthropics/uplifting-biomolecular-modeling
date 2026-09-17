"""``python -m colabdesign_opt ...`` / ``colabdesign-opt ...`` (and ``run.sh``, which execs the former) -> :func:`colabdesign_opt.cli.main` behind two
guards, in this order, before anything of the command line resolves: (1) the core pin — ``_core_gate.gate`` (the tree's kit_template copy, standard
library only) locates ``opt_core`` without importing it and compares the kit's ``[tool.opt_core]`` pin with the located core's ``__version__``: an absent,
older, newer or edited core is ``[colabdesign-opt] NOT ACTIVE: reason=core_missing:opt_core …`` / ``reason=core_mismatch: …`` /
``reason=core_pin_unreadable: …``, exit 3; (2) the import of the command line itself, which imports the core's sub-modules: a pinned core that lacks
one is named (``[colabdesign-opt] NOT ACTIVE: core_missing:<module> …``, exit 3) — never a traceback, never a stock run in its place."""
import sys

from . import TAG

EXIT_INACTIVE = 3                                   # == report.EXIT_NOT_ACTIVE == _core_gate.EXIT_NOT_ACTIVE (restated import-free; tests/test_cli.py holds them equal)


def main(argv=None):
    from ._core_gate import gate
    gate(__file__, tag=TAG)                          # (1) NOT ACTIVE: reason=core_missing:opt_core | core_mismatch: … | core_pin_unreadable: … -> SystemExit(3)
    try:
        from .cli import main as cli_main           # (2) the pinned core is importable: a sub-module it lacks is named
    except ImportError as e:
        missing = getattr(e, "name", None) or str(e)
        sys.stderr.write(f"[{TAG}] NOT ACTIVE: core_missing:{missing} ({e}); install the tree's core at the kit's pin: pip install -e <tree>/common/opt_core -e <tree>/colabdesign/opt\n")
        sys.stderr.flush()
        return EXIT_INACTIVE
    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main())
